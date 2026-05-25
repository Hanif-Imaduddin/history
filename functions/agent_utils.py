"""Berisi fungsi helper yang digunakan untuk memakai agent dengan mudah"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

_MAX_TOKENS_BEFORE_COMPACT = 131_072  # setengah dari context window model (262,144)


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
    """
    logger = logging.getLogger(f"clario.{agent_name}")
    tool_map = {t.name: t for t in tools}
    new_messages: list[BaseMessage] = []
    search_call_count = 0

    current = list(messages)
    final_response: AIMessage = AIMessage(content="")

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

        for tc in response.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            arg_preview = _format_tool_args(tool_args)
            logger.debug(f"[TOOL] → {tool_name}({arg_preview})")
            t_tool = time.perf_counter()
            if tool_name == "internet_search":
                if search_call_count >= max_search_calls:
                    result = (
                        f"[Search limit] internet_search sudah dipanggil {max_search_calls} kali "
                        "dalam round ini. Gunakan informasi yang sudah dikumpulkan."
                    )
                    logger.debug(f"[TOOL] ✗ {tool_name} diblokir — batas {max_search_calls} panggilan tercapai")
                    tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
                    new_messages.append(tool_msg)
                    current.append(tool_msg)
                    continue
                search_call_count += 1
            try:
                result = tool_map[tool_name].invoke(tool_args)
                logger.debug(f"[TOOL] ✓ {tool_name} selesai dalam {time.perf_counter() - t_tool:.2f}s")
            except Exception as exc:
                result = f"[Tool error] {exc}"
                logger.debug(f"[TOOL] ✗ {tool_name} gagal dalam {time.perf_counter() - t_tool:.2f}s — {exc}")

            tool_msg = ToolMessage(content=str(result), tool_call_id=tc["id"])
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
