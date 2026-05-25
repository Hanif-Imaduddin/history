from __future__ import annotations

import logging
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pymongo import MongoClient

from states.schema import (
    BussinessConstraints,
    EBPState,
    EthicsAnalysisReport,
    FinancialAnalysisReport,
    MarketScoutReport,
    StrategicReport,
)

import os
from dotenv import load_dotenv

load_dotenv()

# Suppress noisy MongoDB driver heartbeat logs
logging.getLogger("pymongo").setLevel(logging.WARNING)

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

_client: Optional[MongoClient] = None


def _get_collection():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME][COLLECTION_NAME]


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    result = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            result.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            tool_calls = []
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    })
            result.append({"type": "ai", "content": msg.content, "tool_calls": tool_calls})
        elif isinstance(msg, SystemMessage):
            result.append({"type": "system", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            result.append({
                "type": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
            })
        else:
            result.append({"type": "unknown", "content": str(msg.content)})
    return result


def _deserialize_messages(data: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for m in data:
        t = m.get("type", "")
        content = m.get("content", "")
        if t == "human":
            messages.append(HumanMessage(content=content))
        elif t == "ai":
            tool_calls = m.get("tool_calls", [])
            messages.append(AIMessage(content=content, tool_calls=tool_calls))
        elif t == "system":
            messages.append(SystemMessage(content=content))
        elif t == "tool":
            messages.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id", "")))
    return messages


def _serialize_dataclass(obj) -> Optional[dict]:
    return asdict(obj) if obj is not None else None


def _dataclass_from_dict(cls, data: Optional[dict]):
    return cls(**data) if data is not None else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_state(state: EBPState) -> str:
    """Persist EBPState to MongoDB. Returns state_id."""
    col = _get_collection()
    doc = {
        "state_id": state["state_id"],
        "user_id": state["user_id"],
        "bussiness_constraints": _serialize_dataclass(state.get("bussiness_constraints")),
        "market_scout_report": _serialize_dataclass(state.get("market_scout_report")),
        "strategic_report": _serialize_dataclass(state.get("strategic_report")),
        "financial_analysis_report": _serialize_dataclass(state.get("financial_analysis_report")),
        "ethics_analysis_report": _serialize_dataclass(state.get("ethics_analysis_report")),
        "approval_status": state.get("approval_status", "pending"),
        "orchestrator_feedback": state.get("orchestrator_feedback"),
        "messages": _serialize_messages(state.get("messages", [])),
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 3),
        "user_feedback": state.get("user_feedback"),
        "final_result": state.get("final_result"),
    }
    col.update_one(
        {"state_id": state["state_id"]},
        {"$set": doc, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return state["state_id"]


def load_state(state_id: str) -> Optional[EBPState]:
    """Load EBPState from MongoDB by state_id. Returns None if not found."""
    col = _get_collection()
    doc = col.find_one({"state_id": state_id})
    if doc is None:
        return None

    return EBPState(
        state_id=doc["state_id"],
        user_id=doc["user_id"],
        bussiness_constraints=_dataclass_from_dict(BussinessConstraints, doc.get("bussiness_constraints")),
        market_scout_report=_dataclass_from_dict(MarketScoutReport, doc.get("market_scout_report")),
        strategic_report=_dataclass_from_dict(StrategicReport, doc.get("strategic_report")),
        financial_analysis_report=_dataclass_from_dict(FinancialAnalysisReport, doc.get("financial_analysis_report")),
        ethics_analysis_report=_dataclass_from_dict(EthicsAnalysisReport, doc.get("ethics_analysis_report")),
        approval_status=doc.get("approval_status", "pending"),
        orchestrator_feedback=doc.get("orchestrator_feedback"),
        messages=_deserialize_messages(doc.get("messages", [])),
        iteration=doc.get("iteration", 0),
        max_iterations=doc.get("max_iterations", 3),
        user_feedback=doc.get("user_feedback"),
        final_result=doc.get("final_result"),
    )


def list_sessions(user_id: str = "default_user") -> list[dict]:
    """Return summary of all sessions for a user, newest first."""
    col = _get_collection()
    cursor = col.find(
        {"user_id": user_id},
        {
            "_id": 0,
            "state_id": 1,
            "bussiness_constraints": 1,
            "approval_status": 1,
            "final_result": 1,
            "created_at": 1,
            "iteration": 1,
            "max_iterations": 1,
        },
    ).sort("created_at", -1).limit(50)
    sessions = []
    for doc in cursor:
        bc = doc.get("bussiness_constraints") or {}
        created = doc.get("created_at")
        sessions.append({
            "state_id": doc["state_id"],
            "sector": bc.get("sector_and_domain", ""),
            "audience": bc.get("audience", ""),
            "prompt": bc.get("initial_prompt", ""),
            "approval_status": doc.get("approval_status", "pending"),
            "has_report": bool(doc.get("final_result")),
            "created_at": created.isoformat() if isinstance(created, datetime) else None,
            "iteration": doc.get("iteration", 0),
            "max_iterations": doc.get("max_iterations", 3),
        })
    return sessions


def get_session_detail(state_id: str) -> Optional[dict]:
    """Return full detail of one session (including final_result)."""
    col = _get_collection()
    doc = col.find_one(
        {"state_id": state_id},
        {
            "_id": 0,
            "state_id": 1,
            "bussiness_constraints": 1,
            "approval_status": 1,
            "final_result": 1,
            "orchestrator_feedback": 1,
            "created_at": 1,
            "iteration": 1,
            "max_iterations": 1,
        },
    )
    if doc is None:
        return None
    bc = doc.get("bussiness_constraints") or {}
    created = doc.get("created_at")
    return {
        "state_id": doc["state_id"],
        "sector": bc.get("sector_and_domain", ""),
        "audience": bc.get("audience", ""),
        "prompt": bc.get("initial_prompt", ""),
        "approval_status": doc.get("approval_status", "pending"),
        "final_result": doc.get("final_result"),
        "orchestrator_feedback": doc.get("orchestrator_feedback"),
        "created_at": created.isoformat() if isinstance(created, datetime) else None,
        "iteration": doc.get("iteration", 0),
        "max_iterations": doc.get("max_iterations", 3),
    }


def create_new_state(
    constraints: BussinessConstraints,
    user_id: str = "default_user",
    max_iterations: int = 3,
) -> EBPState:
    """Create a fresh EBPState with a new UUID."""
    return EBPState(
        state_id=str(uuid.uuid4()),
        user_id=user_id,
        bussiness_constraints=constraints,
        market_scout_report=None,
        strategic_report=None,
        financial_analysis_report=None,
        ethics_analysis_report=None,
        approval_status="pending",
        orchestrator_feedback=None,
        messages=[],
        iteration=0,
        max_iterations=max_iterations,
        user_feedback=None,
        final_result=None,
    )
