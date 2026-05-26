"""PostgreSQL database operations for user accounts and logging."""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/clario_ai")

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_pool() -> psycopg2.pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = psycopg2.pool.ThreadedConnectionPool(2, 10, DATABASE_URL)
    return _pool


@contextmanager
def _conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ── User queries ───────────────────────────────────────────────────────────────

def get_user_by_username(username: str) -> Optional[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, password_hash, role, is_active FROM users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, role, is_active, created_at FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def create_user(username: str, password_hash: str, role: str = "user") -> int:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) RETURNING id",
                (username, password_hash, role),
            )
            return cur.fetchone()[0]


def upsert_admin(username: str, password_hash: str) -> int:
    """Create or update the admin user from .env. Returns admin user_id."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, role)
                VALUES (%s, %s, 'admin')
                ON CONFLICT (username) DO UPDATE
                  SET password_hash = EXCLUDED.password_hash,
                      role          = 'admin',
                      is_active     = TRUE
                RETURNING id
                """,
                (username, password_hash),
            )
            return cur.fetchone()[0]


def list_all_users() -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT u.id, u.username, u.role, u.is_active, u.created_at,
                       COUNT(CASE WHEN al.created_at >= CURRENT_DATE THEN 1 END) AS today_analyses
                FROM users u
                LEFT JOIN analysis_logs al ON al.user_id = u.id
                GROUP BY u.id
                ORDER BY u.created_at DESC
                """
            )
            rows = cur.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if isinstance(d.get("created_at"), datetime):
                    d["created_at"] = d["created_at"].isoformat()
                result.append(d)
            return result


def delete_user(user_id: int) -> bool:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
            return cur.rowcount > 0


# ── Device fingerprint ─────────────────────────────────────────────────────────

def check_fingerprint(fingerprint_hash: str) -> Optional[int]:
    """Returns user_id if fingerprint is already registered, else None."""
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM device_fingerprints WHERE fingerprint_hash = %s",
                (fingerprint_hash,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def add_fingerprint(user_id: int, fingerprint_hash: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO device_fingerprints (user_id, fingerprint_hash) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (user_id, fingerprint_hash),
            )


# ── Analysis rate limiting ─────────────────────────────────────────────────────

def count_today_analyses(user_id: int) -> int:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM analysis_logs WHERE user_id = %s AND created_at >= CURRENT_DATE",
                (user_id,),
            )
            return cur.fetchone()[0]


def log_analysis(user_id: int, username: str, session_id: str, sector: str) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO analysis_logs (user_id, username, session_id, sector) VALUES (%s, %s, %s, %s)",
                (user_id, username, session_id, sector),
            )


# ── Auth event logging ─────────────────────────────────────────────────────────

def log_auth_event(
    user_id: Optional[int],
    username: str,
    event_type: str,
    details: str = "",
    ip_address: str = "",
) -> None:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth_logs (user_id, username, event_type, details, ip_address) VALUES (%s, %s, %s, %s, %s)",
                (user_id, username, event_type, details, ip_address),
            )


# ── Admin log queries ──────────────────────────────────────────────────────────

def get_analysis_logs(limit: int = 200) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, session_id, sector, created_at FROM analysis_logs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if isinstance(d.get("created_at"), datetime):
                    d["created_at"] = d["created_at"].isoformat()
                result.append(d)
            return result


def get_auth_logs(limit: int = 200) -> list[dict]:
    with _conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, event_type, details, ip_address, created_at FROM auth_logs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            result = []
            for row in rows:
                d = dict(row)
                if isinstance(d.get("created_at"), datetime):
                    d["created_at"] = d["created_at"].isoformat()
                result.append(d)
            return result
