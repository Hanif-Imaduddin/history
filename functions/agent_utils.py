"""Berisi fungsi helper yang digunakan untuk memakai agent dengan mudah"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

_MAX_TOKENS_BEFORE_COMPACT = 131_072  # setengah dari context window model (262,144)

_AGENT_LABELS = {
    "market_scout": "Market Scout",
    "strategic_architect": "Strategic Architect",
    "financial_analyst": "Financial Analyst",
    "ethics_agent": "Ethics Guardian",
    "lead_orchestrator": "Lead Orchestrator",
}

# ── Thread-local emit callback ─────────────────────────────────────────────────
_local = threading.local()


def set_emit_callback(fn) -> None:
    _local.emit = fn


def clear_emit_callback() -> None:
    _local.emit = None


def _emit_event(event: dict) -> None:
    fn = getattr(_local, "emit", None)
    if fn:
        try:
            fn(event)
        except Exception:
            pass


# ── Search result parser ───────────────────────────────────────────────────────

def _parse_search_output(result_str: str) -> dict:
    """Parse internet_search string output into {query, summaries} dict."""
    if not isinstance(result_str, str):
        return {"query": "", "summaries": []}

    query = ""
    m = re.search(r"Search results for '(.+?)':", result_str)
    if m:
        query = m.group(1)

    summaries: list[dict] = []
    if "-- Page Summaries --" in result_str:
        section = result_str.split("-- Page Summaries --", 1)[1]
        current_url: str | None = None
        current_lines: list[str] = []
        for line in section.split("\n"):
            stripped = line.strip()
            # Lines like "[https://example.com]"
            if stripped.startswith("[") and stripped.endswith("]") and "://" in stripped:
                if current_url and current_lines:
                    text = " ".join(current_lines).strip()
                    if text:
                        summaries.append({"url": current_url, "summary": text[:500]})
                current_url = stripped[1:-1]
                current_lines = []
            elif current_url and stripped:
                current_lines.append(stripped)
        if current_url and current_lines:
            text = " ".join(current_lines).strip()
            if text:
                summaries.append({"url": current_url, "summary": text[:500]})

    return {"query": query, "summaries": summaries}


# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_clario_logging() -> None:
    logger = logging.getLogger("clario")
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d  %(levelname)-5s  [%(name)s]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False


_setup_clario_logging()


def _format_tool_args(args: dict) -> str:
    if "query" in args:
        q = str(args["query"])
        return f'query="{q[:80]}{"..." if len(q) > 80 else ""}"'
    preview = str(args)
    return preview[:100] + ("..." if len(preview) > 100 else "")


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum(len(str(m.content)) for m in messages) // 4


def _compact_history(
    original_messages: list[BaseMessage],
    accumulated: list[BaseMessage],
    agent_name: str = "agent",
) -> list[BaseMessage]:
    """Ringkas accumulated ToolMessages menggunakan LLM ringan."""
    from functions.llm import get_compact_llm

    logger = logging.getLogger(f"clario.{agent_name}")
    tool_texts = [m.content for m in accumulated if isinstance(m, ToolMessage)]
    if not tool_texts:
        return original_messages

    combined = "\n\n---\n\n".join(tool_texts)
    logger.debug("[COMPACT] Context melebihi batas — meringkas history dengan LLM ringan...")
    t0 = time.perf_counter()
    compact_llm = get_compact_llm()
    summary = compact_llm.invoke([
        SystemMessage(content=(
            "Kamu adalah asisten ringkas. Buat ringkasan komprehensif dari hasil "
            "penelitian internet berikut. Pertahankan semua fakta penting, angka, "
            "statistik, regulasi, dan data relevan."
        )),
        HumanMessage(content=f"Ringkas temuan penelitian berikut:\n\n{combined[:500_000]}"),
    ])
    logger.debug(f"[COMPACT] Selesai dalam {time.perf_counter() - t0:.2f}s")
    return original_messages + [
        HumanMessage(content=f"=== RINGKASAN HASIL PENELITIAN SEBELUMNYA ===\n{summary.content}")
    ]


def _execute_tool_call(
    tc: dict,
    tool_map: dict,
    search_count_ref: list,
    search_lock: threading.Lock,
    max_search_calls: int,
    agent_name: str,
    label: str,
    emit_fn,
    logger,
) -> tuple[str, str]:
    """Eksekusi satu tool call; thread-safe untuk parallel execution.

    Returns (call_id, result_str).
    emit_fn harus di-capture dari thread utama karena _local adalah thread-local.
    """
    tool_name = tc["name"]
    tool_args = tc["args"]
    call_id   = tc["id"]
    query     = tool_args.get("query", "") if tool_name == "internet_search" else ""

    if emit_fn:
        try:
            emit_fn({"type": "tool_call_start", "agent": agent_name, "label": label,
                     "tool_name": tool_name, "query": query, "call_id": call_id})
        except Exception:
            pass

    t_tool = time.perf_counter()

    if tool_name == "internet_search":
        with search_lock:
            if search_count_ref[0] >= max_search_calls:
                result = (
                    f"[Search limit] Batas {max_search_calls} pencarian sudah tercapai. "
                    "Gunakan informasi yang sudah dikumpulkan untuk menyusun laporan."
                )
                logger.debug(f"[TOOL] ✗ {tool_name} diblokir — batas {max_search_calls} tercapai")
                if emit_fn:
                    try:
                        emit_fn({"type": "tool_call_result", "agent": agent_name, "label": label,
                                 "tool_name": tool_name, "query": query, "call_id": call_id,
                                 "blocked": True, "search_data": None})
                    except Exception:
                        pass
                return call_id, result
            search_count_ref[0] += 1

    try:
        result = tool_map[tool_name].invoke(tool_args)
        logger.debug(f"[TOOL] ✓ {tool_name} selesai dalam {time.perf_counter() - t_tool:.2f}s")
    except Exception as exc:
        result = f"[Tool error] {exc}"
        logger.debug(f"[TOOL] ✗ {tool_name} gagal dalam {time.perf_counter() - t_tool:.2f}s — {exc}")

    search_data = _parse_search_output(str(result)) if tool_name == "internet_search" else None
    if emit_fn:
        try:
            emit_fn({"type": "tool_call_result", "agent": agent_name, "label": label,
                     "tool_name": tool_name, "query": query, "call_id": call_id,
                     "blocked": False, "search_data": search_data})
        except Exception:
            pass
    return call_id, str(result)


def run_react_loop(
    llm_with_tools,
    messages: list[BaseMessage],
    tools: list[BaseTool],
    max_tool_rounds: int = 4,
    agent_name: str = "agent",
    max_search_calls: int = 3,
) -> tuple[list[BaseMessage], AIMessage]:
    """Execute a ReAct (Reason + Act) loop.

    Keeps calling the LLM until it stops issuing tool calls or
    `max_tool_rounds` is exhausted.  Returns (all_new_messages, final_ai_msg).
    Jika context melebihi setengah context window, history diringkas otomatis
    menggunakan LLM ringan sebelum invocation berikutnya.

    Emits SSE events via thread-local callback (set_emit_callback) for:
      agent_started, tool_call_start, tool_call_result
    """
    logger = logging.getLogger(f"clario.{agent_name}")
    tool_map = {t.name: t for t in tools}
    new_messages: list[BaseMessage] = []
    search_call_count = 0
    label = _AGENT_LABELS.get(agent_name, agent_name)

    current = list(messages)
    final_response: AIMessage = AIMessage(content="")

    _emit_event({"type": "agent_started", "agent": agent_name, "label": label})

    for round_num in range(max_tool_rounds + 1):
        if _estimate_tokens(current) > _MAX_TOKENS_BEFORE_COMPACT:
            current = _compact_history(messages, new_messages, agent_name=agent_name)

        logger.debug(f"[LLM] Round {round_num + 1} — memanggil LLM...")
        t_llm = time.perf_counter()
        response: AIMessage = llm_with_tools.invoke(current)
        n_calls = len(response.tool_calls or [])
        logger.debug(
            f"[LLM] Round {round_num + 1} selesai dalam {time.perf_counter() - t_llm:.2f}s"
            f" | {n_calls} tool call(s)"
        )
        new_messages.append(response)
        current.append(response)
        final_response = response

        if not getattr(response, "tool_calls", None):
            break

        # Capture emit callback sebelum spawn worker threads
        # (_local adalah thread-local, tidak tersedia di worker threads)
        emit_fn = getattr(_local, "emit", None)

        search_count_ref = [search_call_count]
        search_lock = threading.Lock()
        n_workers = len(response.tool_calls)

        logger.debug(f"[TOOL] Menjalankan {n_workers} tool call(s) secara paralel...")
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _execute_tool_call,
                    tc, tool_map, search_count_ref, search_lock,
                    max_search_calls, agent_name, label, emit_fn, logger,
                ): tc["id"]
                for tc in response.tool_calls
            }
            results_by_id: dict[str, str] = {}
            for future in as_completed(futures):
                call_id = futures[future]
                try:
                    _, result = future.result()
                except Exception as exc:
                    result = f"[Tool error] {exc}"
                results_by_id[call_id] = result

        search_call_count = search_count_ref[0]

        # Tambah ToolMessages dalam URUTAN ASLI (wajib untuk LangGraph)
        for tc in response.tool_calls:
            result = results_by_id.get(tc["id"], "[Tool error] Result not found")
            tool_msg = ToolMessage(content=result, tool_call_id=tc["id"])
            new_messages.append(tool_msg)
            current.append(tool_msg)

    return new_messages, final_response


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output.

    Tries markdown code-fenced JSON first, then bare JSON, then returns
    {"raw": text} as a last resort.
    """
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw": text}


def format_constraints(bc) -> str:
    if bc is None:
        return "No constraints provided."
    return (
        f"Sector/Domain: {bc.sector_and_domain}\n"
        f"Target Audience: {bc.audience}\n"
        f"Business Idea: {bc.initial_prompt}"
    )
