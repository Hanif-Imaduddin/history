"""FastAPI backend for ClarioAI UI."""
from __future__ import annotations

import json
import queue as _queue
import threading
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from functions.mongodb import BussinessConstraints, create_new_state, get_session_detail, list_sessions, save_state
from graphs.ebp_graph import graph
from langgraph.types import Command

app = FastAPI(title="ClarioAI")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Single-user session state ──────────────────────────────────────────────────

_session: dict[str, Any] = {
    "state_id": None,
    "is_running": False,
    "is_interrupted": False,
}

_event_q: _queue.Queue = _queue.Queue()
_feedback_ready = threading.Event()
_pending_feedback: Optional[str] = None

_event_id_counter: int = 0
_event_history: list[dict] = []   # replay buffer for SSE reconnects


# ── Pydantic models ────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    sector_and_domain: str
    audience: str
    initial_prompt: str
    max_iterations: int = 3


class FeedbackRequest(BaseModel):
    feedback: str


# ── Internal helpers ───────────────────────────────────────────────────────────

def _emit(event: dict) -> None:
    global _event_id_counter
    _event_id_counter += 1
    event["_eid"] = _event_id_counter
    _event_q.put(event)
    if event.get("type") not in ("heartbeat", "connected"):
        _event_history.append(event)
        if len(_event_history) > 100:
            _event_history.pop(0)


_AGENT_LABELS = {
    "market_scout": "Market Scout",
    "strategic_architect": "Strategic Architect",
    "financial_analyst": "Financial Analyst",
    "ethics_agent": "Ethics Guardian",
    "lead_orchestrator": "Lead Orchestrator",
    "final_summary": "Final Summary",
}


def _extract_messages_preview(updates: dict) -> list[str]:
    msgs = updates.get("messages", [])
    if not isinstance(msgs, list):
        msgs = [msgs]
    previews = []
    for m in msgs:
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        if content:
            previews.append(str(content)[:600])
    return previews


def _parse_chunk(chunk: dict) -> None:
    """Translate a LangGraph stream chunk into SSE events."""
    for node_name, updates in chunk.items():
        if not isinstance(updates, dict):
            continue

        label = _AGENT_LABELS.get(node_name, node_name)
        previews = _extract_messages_preview(updates)

        if node_name == "lead_orchestrator":
            status = updates.get("approval_status", "pending")
            feedback = updates.get("orchestrator_feedback") or ""
            iteration = updates.get("iteration", 0)
            _emit({
                "type": "orchestrator_evaluation",
                "label": label,
                "status": status,
                "feedback": feedback,
                "iteration": iteration,
                "messages": previews,
            })

        elif node_name == "final_summary":
            final_md = updates.get("final_result", "")
            _emit({"type": "final_result", "content": final_md})

        else:
            _emit({
                "type": "agent_complete",
                "agent": node_name,
                "label": label,
                "messages": previews,
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


def _graph_runner(initial_state: dict, thread_config: dict) -> None:
    """Runs in a background thread. Drives the graph and emits SSE events."""
    global _pending_feedback, _session

    try:
        state_or_command: Any = initial_state

        while True:
            # Run graph until it finishes or hits an interrupt
            for chunk in graph.stream(state_or_command, config=thread_config, stream_mode="updates"):
                _parse_chunk(chunk)

            snap = graph.get_state(thread_config)
            if not snap.next:
                # Graph completed normally
                final_state = snap.values
                _persist_final(final_state)
                break

            # Graph interrupted — ask user for feedback
            intr_data = _get_interrupt_data(thread_config)
            _emit({
                "type": "feedback_required",
                "orchestrator_feedback": intr_data.get("orchestrator_feedback", ""),
                "synthesis": intr_data.get("synthesis", ""),
                "iteration": intr_data.get("iteration", 0),
            })
            _session["is_interrupted"] = True

            # Wait up to 30 minutes; resume with empty feedback on timeout
            if not _feedback_ready.wait(timeout=1800):
                _emit({"type": "info", "message": "Feedback timeout — continuing without user input."})
            _feedback_ready.clear()

            user_fb = _pending_feedback or ""
            _pending_feedback = None
            _session["is_interrupted"] = False
            _session["is_running"] = True

            state_or_command = Command(resume=user_fb)

        _emit({"type": "done"})

    except Exception as exc:
        _emit({"type": "error", "message": str(exc)})
    finally:
        _session["is_running"] = False


def _persist_final(state: dict) -> None:
    state_id = _session.get("state_id")
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


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/status")
async def get_status():
    state_id = _session["state_id"]
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
            if _session["is_interrupted"]:
                intr_data = _get_interrupt_data(thread_config)
                interrupt_info = {
                    "orchestrator_feedback": intr_data.get("orchestrator_feedback", ""),
                    "synthesis": intr_data.get("synthesis", ""),
                    "iteration": intr_data.get("iteration", 0),
                }
        except Exception:
            pass
    return {
        "is_running": _session["is_running"],
        "is_interrupted": _session["is_interrupted"],
        "state": state_info,
        "interrupt_info": interrupt_info,
    }


@app.get("/api/sessions")
async def get_sessions():
    try:
        return list_sessions(user_id="default_user")
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/sessions/{state_id}")
async def get_session(state_id: str):
    detail = get_session_detail(state_id)
    if detail is None:
        raise HTTPException(404, "Session not found.")
    return detail


@app.post("/api/start")
async def start_session(req: StartRequest):
    if _session["is_running"]:
        raise HTTPException(400, "A session is already running.")

    constraints = BussinessConstraints(
        sector_and_domain=req.sector_and_domain,
        audience=req.audience,
        initial_prompt=req.initial_prompt,
    )
    initial_state = create_new_state(
        constraints=constraints,
        user_id="default_user",
        max_iterations=req.max_iterations,
    )
    save_state(initial_state)

    state_id = initial_state["state_id"]
    _session["state_id"] = state_id
    _session["is_running"] = True
    _session["is_interrupted"] = False
    _feedback_ready.clear()

    global _event_id_counter, _event_history
    _event_id_counter = 0
    _event_history.clear()

    # Drain stale events from a previous run
    while not _event_q.empty():
        try:
            _event_q.get_nowait()
        except _queue.Empty:
            break

    _emit({"type": "session_started", "state_id": state_id})

    thread_config = {"configurable": {"thread_id": state_id}}
    t = threading.Thread(
        target=_graph_runner,
        args=(initial_state, thread_config),
        daemon=True,
    )
    t.start()

    return {"state_id": state_id}


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    global _pending_feedback
    if not _session["is_interrupted"]:
        raise HTTPException(400, "No feedback is currently awaited.")
    _pending_feedback = req.feedback
    _session["is_running"] = True
    _feedback_ready.set()
    return {"status": "ok"}


@app.get("/api/events")
async def stream_events(last_event_id: int = 0):
    """SSE endpoint — streams graph events to the browser.

    last_event_id: the last event ID the client received (used for replay on reconnect).
    """
    import asyncio

    async def generator():
        yield "data: {\"type\": \"connected\"}\n\n"

        # Replay missed events when client reconnects mid-session
        if last_event_id > 0:
            for ev in [e for e in list(_event_history) if e.get("_eid", 0) > last_event_id]:
                eid = ev["_eid"]
                ev_clean = {k: v for k, v in ev.items() if k != "_eid"}
                yield f"id: {eid}\ndata: {json.dumps(ev_clean, ensure_ascii=False)}\n\n"

        loop = asyncio.get_event_loop()
        while True:
            try:
                event = await loop.run_in_executor(None, _event_q.get, True, 25.0)
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
