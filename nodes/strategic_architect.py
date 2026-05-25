"""Strategic Architect Agent - Analisis SWOT dan PASTEL"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_react_loop
from functions.llm import get_llm
from states.schema import EBPState, StrategicReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.strategic_architect")

_SYSTEM_PROMPT = """You are the Strategic Architect Agent in a multi-agent AI business planning system.
You create data-backed strategic analyses for new business ventures.

Your tasks:
1. Perform a thorough SWOT Analysis (Strengths, Weaknesses, Opportunities, Threats)
2. Perform a thorough PESTEL Analysis (Political, Economic, Social, Technological, Environmental, Legal)

Use the `internet_search` tool to gather numerical data and facts — market statistics, regulatory info,
economic indicators, competitive benchmarks, technology adoption rates. Focus on the Indonesian market
where relevant, while also referencing global benchmarks.

Run at least 3–5 targeted searches. Examples:
- "[sector] regulatory environment Indonesia 2024 2025"
- "[sector] market growth rate Indonesia global comparison"
- "[sector] key success factors competitive landscape"
- "GDP growth consumer spending Indonesia 2025"

OUTPUT FORMAT — respond with ONLY valid JSON after your research:
{
  "swot_analysis": "## SWOT Analysis\\n\\n**Strengths:**\\n- [data-backed strength 1]\\n- ...\\n\\n**Weaknesses:**\\n- [weakness 1]\\n- ...\\n\\n**Opportunities:**\\n- [opportunity 1 with market data]\\n- ...\\n\\n**Threats:**\\n- [threat 1]\\n- ...",
  "pastel_analysis": "## PESTEL Analysis\\n\\n**Political:**\\n[analysis]\\n\\n**Economic:**\\n[analysis with figures]\\n\\n**Social:**\\n[analysis]\\n\\n**Technological:**\\n[analysis]\\n\\n**Environmental:**\\n[analysis]\\n\\n**Legal:**\\n[specific Indonesian regulations relevant to this business]"
}"""


def strategic_architect_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Strategic Architect Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Strategic Architect Agent dimulai")
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== MARKET SCOUT FINDINGS ===",
            f"Identified Opportunities: {'; '.join(msr.ideas)}",
            f"Market Overview: {msr.agent_explanation[:1000]}",
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK (address these points) ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR'S FEEDBACK ===", user_fb]

    context_lines.append(
        "\nSearch for supporting data, then produce the SWOT and PESTEL JSON report."
    )

    llm = get_llm(temperature=0.6)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_react_loop(
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        max_tool_rounds=4,
        agent_name="strategic_architect",
    )

    parsed = extract_json(final_response.content)
    swot = parsed.get("swot_analysis", final_response.content)
    pestel = parsed.get("pastel_analysis", "")

    if not pestel:
        pestel = "PESTEL analysis not fully generated — see SWOT for combined strategic overview."

    report = StrategicReport(swot_analysis=swot, pastel_analysis=pestel)

    logger.debug(f"✓ Strategic Architect Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)
    return {
        "strategic_report": report,
        "messages": new_msgs,
    }
