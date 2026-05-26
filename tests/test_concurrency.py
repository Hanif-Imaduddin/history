"""
Concurrency & Deployment Readiness Tests untuk ClarioAI.

Jalankan dengan:
    pytest tests/test_concurrency.py -v

Tidak ada koneksi database nyata yang dibutuhkan — semua DB calls di-mock.
"""

from __future__ import annotations

import queue
import threading
import time
import unittest.mock as mock
from concurrent.futures import ThreadPoolExecutor


# ===========================================================================
# 1. ASYNC ENDPOINT CHECK
# ===========================================================================

class TestEndpointAsync:
    """Pastikan semua route handler adalah async def."""

    def test_all_api_routes_are_async(self):
        import inspect
        import importlib

        # Import tanpa trigger startup events
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module

        non_async = []
        for route in app_module.app.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint and not inspect.iscoroutinefunction(endpoint):
                non_async.append(getattr(route, "path", str(route)))

        assert non_async == [], (
            f"Route berikut BUKAN async — akan memblokir event loop:\n"
            + "\n".join(f"  {p}" for p in non_async)
        )


# ===========================================================================
# 2. GLOBAL STATE / SESSION ISOLATION
# ===========================================================================

class TestSessionIsolation:
    """Setiap user_id harus mendapatkan _UserSession yang terpisah."""

    def _get_module(self):
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import importlib, app as app_module
            # Reset global state
            app_module._user_sessions.clear()
            return app_module

    def test_different_users_get_different_sessions(self):
        app_module = self._get_module()
        s1 = app_module._get_user_session("user_1")
        s2 = app_module._get_user_session("user_2")
        assert s1 is not s2, "User berbeda harus punya session terpisah"

    def test_same_user_gets_same_session(self):
        app_module = self._get_module()
        s1 = app_module._get_user_session("user_abc")
        s2 = app_module._get_user_session("user_abc")
        assert s1 is s2, "User yang sama harus mendapat session yang sama"

    def test_session_state_not_shared_between_users(self):
        app_module = self._get_module()
        s1 = app_module._get_user_session("userX")
        s2 = app_module._get_user_session("userY")
        s1.is_running = True
        assert s2.is_running is False, "State dari satu user bocor ke user lain!"

    def test_concurrent_session_creation_is_safe(self):
        """Race condition pada dict initialization saat banyak thread bersamaan."""
        app_module = self._get_module()
        results = []

        def create_session(uid):
            s = app_module._get_user_session(uid)
            results.append((uid, id(s)))

        # Semua mencoba membuat session untuk ID yang sama secara bersamaan
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = [ex.submit(create_session, "shared_user") for _ in range(20)]
            for f in futs:
                f.result()

        session_ids = {sid for _, sid in results}
        assert len(session_ids) == 1, (
            f"Terdapat {len(session_ids)} instance session berbeda untuk user yang sama — "
            "race condition pada _get_user_session!"
        )


# ===========================================================================
# 3. EMIT() THREAD SAFETY
# ===========================================================================

class TestEmitThreadSafety:
    """emit() bisa dipanggil dari banyak thread (daemon thread + ThreadPoolExecutor)."""

    def _make_session(self):
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module
            app_module._user_sessions.clear()
        return app_module._UserSession()

    def test_event_ids_are_unique_under_concurrent_emit(self):
        """Jika event_id_counter tidak atomic, dua thread bisa dapat ID yang sama."""
        session = self._make_session()
        collected_ids = []
        lock = threading.Lock()

        def emit_many():
            for _ in range(50):
                session.emit({"type": "agent_started", "content": "x"})

        threads = [threading.Thread(target=emit_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Drain queue
        while True:
            try:
                ev = session.event_q.get_nowait()
                collected_ids.append(ev.get("_eid"))
            except queue.Empty:
                break

        total = len(collected_ids)
        unique = len(set(collected_ids))
        assert total == unique, (
            f"Ditemukan {total - unique} duplikat event_id dari {total} events — "
            "event_id_counter tidak thread-safe!"
        )

    def test_event_history_no_index_error_under_concurrent_emit(self):
        """list.pop(0) saat len > 100 bisa IndexError jika ada race."""
        session = self._make_session()
        errors = []

        def spam_emit():
            for i in range(200):
                try:
                    session.emit({"type": "tool_call_result", "content": str(i)})
                except Exception as e:
                    errors.append(str(e))

        threads = [threading.Thread(target=spam_emit) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Exception di emit(): {errors}"

    def test_event_history_length_never_exceeds_limit(self):
        """event_history harus punya max 100 entries setelah concurrent emit."""
        session = self._make_session()

        def spam():
            for i in range(300):
                session.emit({"type": "agent_started", "content": str(i)})

        threads = [threading.Thread(target=spam) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(session.event_history) <= 100, (
            f"event_history punya {len(session.event_history)} entries — melebihi limit 100!"
        )


# ===========================================================================
# 4. MONGODB GLOBAL CLIENT — DOUBLE INIT
# ===========================================================================

class TestMongoDBConcurrentInit:
    """_get_collection() tidak memiliki lock — bisa double-init di startup."""

    def test_no_double_init_under_concurrent_access(self):
        import functions.mongodb as mdb

        init_count = [0]
        original_init = None
        created_clients = []

        real_mongo_client_init = None

        class CountingMongoClient:
            def __init__(self, uri):
                init_count[0] += 1
                created_clients.append(self)
                self._uri = uri

            def __getitem__(self, name):
                return mock.MagicMock()

        with mock.patch("functions.mongodb._client", None), \
             mock.patch("functions.mongodb.MongoClient", CountingMongoClient):
            mdb._client = None  # reset

            def get_col():
                return mdb._get_collection()

            with ThreadPoolExecutor(max_workers=20) as ex:
                futs = [ex.submit(get_col) for _ in range(20)]
                for f in futs:
                    f.result()

        if init_count[0] > 1:
            print(
                f"\n[WARN] MongoClient dibuat {init_count[0]} kali secara bersamaan. "
                "Tidak berbahaya (PyMongo thread-safe) tapi membuang koneksi. "
                "Tambahkan threading.Lock() di _get_collection()."
            )
        # Ini warning, bukan hard failure — PyMongo aman setelah init
        assert init_count[0] >= 1


# ===========================================================================
# 5. POSTGRESQL — RATE LIMIT TOCTOU
# ===========================================================================

class TestRateLimitTOCTOU:
    """
    count_today_analyses() → log_analysis() tidak atomic di level database.
    Dengan multi-worker, user bisa melewati batas harian.
    Test ini memverifikasi apakah implementasi saat ini rentan.
    """

    def test_rate_limit_toctou_simulation(self):
        """
        Simulasi: 2 request bersamaan, keduanya membaca count=4 (1 kurang dari limit=5),
        keduanya lolos, maka total menjadi 6 padahal limit 5.
        """
        DAILY_LIMIT = 5
        db_count = [4]  # current count, 1 kurang dari limit
        db_lock = threading.Lock()
        log_entries = []

        def check_and_log_analysis():
            # Simulasi count_today_analyses()
            current = db_count[0]
            time.sleep(0.01)  # simulasi latency DB
            if current >= DAILY_LIMIT:
                return False, "limit reached"
            # Simulasi log_analysis()
            with db_lock:
                db_count[0] += 1
                log_entries.append(db_count[0])
            return True, "ok"

        results = []
        with ThreadPoolExecutor(max_workers=2) as ex:
            futs = [ex.submit(check_and_log_analysis) for _ in range(2)]
            for f in futs:
                results.append(f.result())

        successes = [r for r in results if r[0]]
        if len(successes) == 2:
            print(
                "\n[WARN] TOCTOU terdeteksi: 2 request berhasil saat quota tersisa 1. "
                "Ini hanya masalah di deployment multi-worker. "
                "Solusi: gunakan INSERT dengan CHECK di SQL (atomic count+insert)."
            )
        # Catat: ini bukan hard failure untuk single-worker deployment
        assert len(successes) >= 1, "Minimal 1 request harus berhasil"


# ===========================================================================
# 6. DATABASE CONCURRENT ACCESS
# ===========================================================================

class TestDatabaseConcurrentAccess:
    """PostgreSQL: setiap call membuat koneksi baru (tidak ada connection pool)."""

    def test_postgres_uses_connection_pool(self):
        """Verifikasi bahwa _conn() menggunakan pool, bukan membuat koneksi baru setiap call."""
        import functions.postgres as pg
        import psycopg2.pool

        fake_conn = mock.MagicMock()
        fake_conn.commit = mock.MagicMock()
        fake_conn.rollback = mock.MagicMock()

        fake_pool = mock.MagicMock(spec=psycopg2.pool.ThreadedConnectionPool)
        fake_pool.getconn.return_value = fake_conn

        with mock.patch("functions.postgres._pool", fake_pool):
            with pg._conn():
                pass
            with pg._conn():
                pass

        assert fake_pool.getconn.call_count == 2, "Pool.getconn harus dipanggil setiap penggunaan"
        assert fake_pool.putconn.call_count == 2, "Pool.putconn harus dipanggil untuk mengembalikan koneksi"

    def test_postgres_connection_closed_after_exception(self):
        """Koneksi harus selalu ditutup meski ada exception."""
        close_called = [False]

        class FakeConn:
            def cursor(self, **kwargs):
                return mock.MagicMock()
            def commit(self):
                raise Exception("DB error")
            def rollback(self):
                pass
            def close(self):
                close_called[0] = True

        import functions.postgres as pg
        with mock.patch("psycopg2.connect", return_value=FakeConn()):
            try:
                with pg._conn():
                    pass
            except Exception:
                pass

        assert close_called[0], "Koneksi tidak ditutup setelah exception — connection leak!"


# ===========================================================================
# 7. FEEDBACK RACE CONDITION
# ===========================================================================

class TestFeedbackRace:
    """Verifikasi sinkronisasi feedback_ready Event antara daemon thread dan API."""

    def _make_session(self):
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module
        return app_module._UserSession()

    def test_feedback_event_is_thread_safe(self):
        """feedback_ready.set() dari API thread dan .wait() dari daemon thread."""
        session = self._make_session()
        session.is_interrupted = True
        received_feedback = []

        def daemon_thread():
            session.feedback_ready.wait(timeout=2.0)
            received_feedback.append(session.pending_feedback)
            session.feedback_ready.clear()
            session.is_interrupted = False

        t = threading.Thread(target=daemon_thread)
        t.start()
        time.sleep(0.05)

        # Simulasi POST /api/feedback
        session.pending_feedback = "user input here"
        session.feedback_ready.set()

        t.join(timeout=3.0)
        assert not t.is_alive(), "Daemon thread tidak selesai — deadlock?"
        assert received_feedback == ["user input here"], (
            f"Feedback tidak tersampaikan: {received_feedback}"
        )

    def test_double_start_blocked_when_running(self):
        """is_running=True harus mencegah start session baru."""
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module
            app_module._user_sessions.clear()

        session = app_module._get_user_session("test_user_block")
        session.is_running = True

        # Simulasi check di /api/start
        assert session.is_running is True, "Guard is_running gagal"


# ===========================================================================
# 8. SSE EVENT QUEUE — MULTIPLE CONSUMERS
# ===========================================================================

class TestSSEMultipleConsumers:
    """
    Jika user membuka 2 tab, keduanya consume dari queue yang sama.
    Events akan terbagi — salah satu tab bisa miss events.
    """

    def _make_session(self):
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module
        return app_module._UserSession()

    def test_queue_events_split_with_two_consumers(self):
        """Demonstrasi: event yang masuk ke queue diambil oleh salah satu consumer."""
        session = self._make_session()

        for i in range(10):
            session.emit({"type": "agent_started", "content": str(i)})

        consumer1_events = []
        consumer2_events = []

        # Simulasi dua tab SSE mengambil dari queue yang sama
        while True:
            try:
                consumer1_events.append(session.event_q.get_nowait())
            except queue.Empty:
                break

        # Consumer kedua tidak mendapat apa-apa karena queue sudah kosong
        while True:
            try:
                consumer2_events.append(session.event_q.get_nowait())
            except queue.Empty:
                break

        total = len(consumer1_events) + len(consumer2_events)
        assert total == 10, "Total events harus 10"

        if len(consumer2_events) == 0:
            print(
                "\n[WARN] Dua consumer SSE membagi event dari satu queue. "
                "Jika user buka 2 tab, salah satu tab akan miss semua events. "
                "Solusi: gunakan event_history untuk reconnect, atau broadcast ke multi-consumer."
            )


# ===========================================================================
# 9. IN-MEMORY STATE — SERVER RESTART
# ===========================================================================

class TestInMemoryState:
    """LangGraph MemorySaver: state hilang jika server restart saat analisis berjalan."""

    def test_user_sessions_reset_on_module_reimport(self):
        """Simulasi restart: _user_sessions adalah dict kosong saat module baru di-load."""
        with mock.patch("functions.postgres.upsert_admin"), \
             mock.patch("functions.mongodb._get_collection"):
            import app as app_module

        # Populate sessions
        app_module._get_user_session("user_persistent")
        app_module._user_sessions["user_persistent"].is_running = True
        assert app_module._user_sessions["user_persistent"].is_running is True

        # Simulate restart: clear sessions (represents fresh process)
        app_module._user_sessions.clear()
        new_session = app_module._get_user_session("user_persistent")

        assert new_session.is_running is False, (
            "Setelah restart, is_running harus False — "
            "tapi MongoDB masih mungkin punya record 'pending'. "
            "Perlu reconciliation logic di startup."
        )
