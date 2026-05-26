"""FastAPI backend for ClarioAI UI."""
from __future__ import annotations

import json
import logging
import os
import queue as _queue
import threading
from typing import Any, Optional

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Cookie
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from functions.auth import (
    DAILY_ANALYSIS_LIMIT,
    create_token,
    get_current_user,
    hash_password,
    require_admin,
    _decode_token,
)
from functions.mongodb import BussinessConstraints, create_new_state, get_session_detail, list_sessions, save_state
from functions.postgres import (
    count_today_analyses,
    log_analysis,
    upsert_admin,
)
from graphs.ebp_graph import graph
from langgraph.types import Command
from jose import JWTError

logger = logging.getLogger("clario")


@asynccontextmanager
async def lifespan(app: FastAPI):
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin")
    try:
        admin_id = upsert_admin(admin_username, hash_password(admin_password))
        from functions.mongodb import _get_collection
        col = _get_collection()
        col.update_many({"user_id": "default_user"}, {"$set": {"user_id": str(admin_id)}})
        logger.info("Startup: admin user synced (id=%s), old sessions migrated.", admin_id)
    except Exception as exc:
        logger.warning("Startup DB init failed: %s", exc)
    yield


app = FastAPI(title="ClarioAI", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Include routers ────────────────────────────────────────────────────────────
from routers.auth_router import router as auth_router
from routers.admin_router import router as admin_router

app.include_router(auth_router)
app.include_router(admin_router)


# ── Per-user session state ─────────────────────────────────────────────────────

class _UserSession:
    """Holds all mutable in-flight state for one logged-in user."""

    def __init__(self) -> None:
        self.state_id: Optional[str] = None
        self.is_running: bool = False
        self.is_interrupted: bool = False
        self.username: Optional[str] = None
        self.event_q: _queue.Queue = _queue.Queue()
        self.event_history: list[dict] = []
        self.event_id_counter: int = 0
        self.feedback_ready: threading.Event = threading.Event()
        self.pending_feedback: Optional[str] = None
        self._emit_lock: threading.Lock = threading.Lock()

    def emit(self, event: dict) -> None:
        with self._emit_lock:
            self.event_id_counter += 1
            event["_eid"] = self.event_id_counter
            self.event_q.put(event)
            if event.get("type") not in ("heartbeat", "connected"):
                self.event_history.append(event)
                if len(self.event_history) > 100:
                    self.event_history.pop(0)

    def reset_events(self) -> None:
        self.event_id_counter = 0
        self.event_history.clear()
        while not self.event_q.empty():
            try:
                self.event_q.get_nowait()
            except _queue.Empty:
                break


_user_sessions: dict[str, _UserSession] = {}
_user_sessions_lock = threading.Lock()


def _get_user_session(user_id: str) -> _UserSession:
    with _user_sessions_lock:
        if user_id not in _user_sessions:
            _user_sessions[user_id] = _UserSession()
        return _user_sessions[user_id]


# ── Pydantic models ────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    sector_and_domain: str
    audience: str
    initial_prompt: str
    max_iterations: int = 3


class FeedbackRequest(BaseModel):
    feedback: str


# ── Internal helpers ───────────────────────────────────────────────────────────

_AGENT_LABELS = {
    "market_scout": "Market Scout",
    "strategic_architect": "Strategic Architect",
    "financial_analyst": "Financial Analyst",
    "ethics_agent": "Ethics Guardian",
    "lead_orchestrator": "Lead Orchestrator",
    "final_summary": "Final Summary",
}


def _extract_messages_full(updates: dict) -> list[dict]:
    msgs = updates.get("messages", [])
    if not isinstance(msgs, list):
        msgs = [msgs]
    result = []
    for m in msgs:
        if isinstance(m, dict):
            msg_type = m.get("type", "message")
            content = m.get("content", "")
            tool_name = m.get("name", "")
        else:
            msg_type = getattr(m, "type", type(m).__name__)
            content = getattr(m, "content", "")
            tool_name = getattr(m, "name", "") or ""

        if isinstance(content, list):
            content = " ".join(str(c) for c in content if c)

        content_str = str(content).strip() if content else ""
        if content_str:
            entry: dict = {"type": str(msg_type), "content": content_str}
            if tool_name:
                entry["tool_name"] = str(tool_name)
            result.append(entry)
    return result


def _parse_chunk(chunk: dict, session: _UserSession) -> None:
    for node_name, updates in chunk.items():
        if not isinstance(updates, dict):
            continue

        label = _AGENT_LABELS.get(node_name, node_name)
        messages = _extract_messages_full(updates)

        if node_name == "lead_orchestrator":
            status = updates.get("approval_status", "pending")
            feedback = updates.get("orchestrator_feedback") or ""
            iteration = updates.get("iteration", 0)
            session.emit({
                "type": "orchestrator_evaluation",
                "label": label,
                "status": status,
                "feedback": feedback,
                "iteration": iteration,
                "messages": [{"type": "ai", "content": feedback}] if feedback else [],
            })

        elif node_name == "final_summary":
            final_md = updates.get("final_result", "")
            session.emit({"type": "final_result", "content": final_md})

        else:
            session.emit({
                "type": "agent_complete",
                "agent": node_name,
                "label": label,
                "messages": messages,
            })


def _get_interrupt_data(thread_config: dict) -> dict:
    try:
        snap = graph.get_state(thread_config)
        for task in snap.tasks:
            for intr in task.interrupts:
                val = intr.value if hasattr(intr, "value") else {}
                if isinstance(val, dict):
                    return val
    except Exception:
        pass
    return {}


def _persist_final(state: dict, state_id: str) -> None:
    if not state_id:
        return
    try:
        from functions.mongodb import load_state
        existing = load_state(state_id)
        if existing is None:
            return
        for key in ("approval_status", "orchestrator_feedback", "iteration", "final_result", "user_feedback"):
            if key in state:
                existing[key] = state[key]  # type: ignore[literal-required]
        save_state(existing)
    except Exception:
        pass


def _graph_runner(initial_state: dict, thread_config: dict, user_id: str) -> None:
    from functions.agent_utils import set_emit_callback, clear_emit_callback

    session = _get_user_session(user_id)

    # Wire agent_utils intermediate events (agent_started, tool_call_start, etc.)
    # to this user's session emit so they reach the correct SSE stream.
    set_emit_callback(session.emit)

    try:
        state_or_command: Any = initial_state

        while True:
            for chunk in graph.stream(state_or_command, config=thread_config, stream_mode="updates"):
                _parse_chunk(chunk, session)

            snap = graph.get_state(thread_config)
            if not snap.next:
                final_state = snap.values
                _persist_final(final_state, session.state_id or "")
                break

            intr_data = _get_interrupt_data(thread_config)
            session.emit({
                "type": "feedback_required",
                "orchestrator_feedback": intr_data.get("orchestrator_feedback", ""),
                "synthesis": intr_data.get("synthesis", ""),
                "iteration": intr_data.get("iteration", 0),
            })
            session.is_interrupted = True

            if not session.feedback_ready.wait(timeout=1800):
                session.emit({"type": "info", "message": "Feedback timeout — continuing without user input."})
            session.feedback_ready.clear()

            user_fb = session.pending_feedback or ""
            session.pending_feedback = None
            session.is_interrupted = False
            session.is_running = True

            state_or_command = Command(resume=user_fb)

        session.emit({"type": "done"})

    except Exception as exc:
        session.emit({"type": "error", "message": str(exc)})
    finally:
        session.is_running = False
        clear_emit_callback()


# ── Page routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def root(access_token: Optional[str] = Cookie(default=None)):
    if not access_token:
        return RedirectResponse(url="/login")
    try:
        _decode_token(access_token)
    except JWTError:
        return RedirectResponse(url="/login")
    return FileResponse("static/index.html")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.get("/admin")
async def admin_page(access_token: Optional[str] = Cookie(default=None)):
    if not access_token:
        return RedirectResponse(url="/login")
    try:
        payload = _decode_token(access_token)
        if payload.get("role") != "admin":
            return RedirectResponse(url="/")
    except JWTError:
        return RedirectResponse(url="/login")
    return FileResponse("static/admin.html")


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status(current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["user_id"])
    session = _get_user_session(user_id)

    state_id = session.state_id
    state_info = None
    interrupt_info = None
    if state_id:
        thread_config = {"configurable": {"thread_id": state_id}}
        try:
            snap = graph.get_state(thread_config)
            if snap and snap.values:
                v = snap.values
                state_info = {
                    "state_id": state_id,
                    "approval_status": v.get("approval_status"),
                    "iteration": v.get("iteration", 0),
                    "max_iterations": v.get("max_iterations", 3),
                    "final_result": v.get("final_result"),
                }
            if session.is_interrupted:
                intr_data = _get_interrupt_data(thread_config)
                interrupt_info = {
                    "orchestrator_feedback": intr_data.get("orchestrator_feedback", ""),
                    "synthesis": intr_data.get("synthesis", ""),
                    "iteration": intr_data.get("iteration", 0),
                }
        except Exception:
            pass
    return {
        "is_running": session.is_running,
        "is_interrupted": session.is_interrupted,
        "state": state_info,
        "interrupt_info": interrupt_info,
    }


@app.get("/api/sessions")
async def get_sessions(current_user: dict = Depends(get_current_user)):
    try:
        user_id_str = str(current_user["user_id"])
        return list_sessions(user_id=user_id_str)
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/sessions/{state_id}")
async def get_session(state_id: str, current_user: dict = Depends(get_current_user)):
    detail = get_session_detail(state_id)
    if detail is None:
        raise HTTPException(404, "Session not found.")
    return detail


@app.post("/api/start")
async def start_session(req: StartRequest, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["user_id"])
    session = _get_user_session(user_id)

    if session.is_running:
        raise HTTPException(400, "A session is already running.")

    # Enforce daily limit for non-admin users
    if current_user["role"] != "admin":
        used = await run_in_threadpool(count_today_analyses, current_user["user_id"])
        if used >= DAILY_ANALYSIS_LIMIT:
            raise HTTPException(
                429,
                f"Batas analisis harian ({DAILY_ANALYSIS_LIMIT}x) sudah tercapai. Coba lagi besok.",
            )

    constraints = BussinessConstraints(
        sector_and_domain=req.sector_and_domain,
        audience=req.audience,
        initial_prompt=req.initial_prompt,
    )
    initial_state = create_new_state(
        constraints=constraints,
        user_id=user_id,
        max_iterations=req.max_iterations,
    )
    save_state(initial_state)

    state_id = initial_state["state_id"]
    session.state_id = state_id
    session.is_running = True
    session.is_interrupted = False
    session.username = current_user["username"]
    session.feedback_ready.clear()
    session.reset_events()

    # Log this analysis
    try:
        await run_in_threadpool(
            log_analysis,
            current_user["user_id"],
            current_user["username"],
            state_id,
            req.sector_and_domain,
        )
    except Exception:
        pass

    session.emit({"type": "session_started", "state_id": state_id})

    thread_config = {"configurable": {"thread_id": state_id}}
    t = threading.Thread(
        target=_graph_runner,
        args=(initial_state, thread_config, user_id),
        daemon=True,
    )
    t.start()

    return {"state_id": state_id}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    user_id = str(current_user["user_id"])
    session = _get_user_session(user_id)

    if not session.is_interrupted:
        raise HTTPException(400, "No feedback is currently awaited.")
    session.pending_feedback = req.feedback
    session.is_running = True
    session.feedback_ready.set()
    return {"status": "ok"}


@app.get("/api/events")
async def stream_events(
    last_event_id: int = 0,
    current_user: dict = Depends(get_current_user),
):
    import asyncio

    user_id = str(current_user["user_id"])
    session = _get_user_session(user_id)

    async def generator():
        yield "data: {\"type\": \"connected\"}\n\n"

        if last_event_id > 0:
            for ev in [e for e in list(session.event_history) if e.get("_eid", 0) > last_event_id]:
                eid = ev["_eid"]
                ev_clean = {k: v for k, v in ev.items() if k != "_eid"}
                yield f"id: {eid}\ndata: {json.dumps(ev_clean, ensure_ascii=False)}\n\n"

        loop = asyncio.get_running_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, session.event_q.get, True, 25.0)
                eid = event.get("_eid", 0)
                ev_clean = {k: v for k, v in event.items() if k != "_eid"}
                yield f"id: {eid}\ndata: {json.dumps(ev_clean, ensure_ascii=False)}\n\n"
            except _queue.Empty:
                yield "data: {\"type\": \"heartbeat\"}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
