"""Market Scout Agent — mengidentifikasi peluang bisnis dan tren pasar secara real-time."""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, MarketScoutReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.market_scout")

_SYSTEM_PROMPT = """You are the Market Scout Agent in a multi-agent AI business planning system.
Your mission is to perform real-time market research and identify viable business opportunities
for the given business constraints.

Search results will be provided for you in bulk. Once you have them, synthesise all findings
into a comprehensive MarketScoutReport. If critical data is still missing after reviewing the
results, you may call `internet_search` for one or two additional targeted queries.

OUTPUT FORMAT — respond with ONLY valid JSON when ready, no other text:
{
  "ideas": [
    "Specific business opportunity 1 with supporting evidence",
    "Specific business opportunity 2 with supporting evidence",
    "Specific business opportunity 3 with supporting evidence"
  ],
  "agent_explanation": "Comprehensive narrative explaining the market landscape, key findings, data sources, trends, and why these opportunities are viable. Include market size estimates, growth rates, and competitive dynamics."
}"""

_SEARCH_TOPICS = [
    "market size and growth trends in the target sector",
    "key competitors and their positioning",
    "consumer demand signals and unmet needs",
    "emerging technologies or regulations affecting the sector",
    "success stories or case studies of similar businesses in Indonesia",
]


def market_scout_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Market Scout Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Market Scout Agent dimulai")
    bc = state.get("bussiness_constraints")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== YOUR MISSION ===",
        "Research the market and identify the best business opportunities.",
        "\n=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]
    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK (address these points) ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR'S FEEDBACK ===", user_fb]

    context_lines.append(
        "\nStart searching now. After gathering enough data, output the JSON report."
    )
    user_message = "\n".join(context_lines)

    llm = get_llm(temperature=0.7)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        tools=[internet_search],
        planning_topics=_SEARCH_TOPICS,
        max_followup_rounds=2,
        agent_name="market_scout",
        max_search_calls=7,
    )

    parsed = extract_json(final_response.content)
    raw_ideas = parsed.get("ideas", [])
    explanation = parsed.get("agent_explanation", final_response.content)

    # Normalise ideas to list of strings
    if isinstance(raw_ideas, list):
        ideas = [str(i) for i in raw_ideas if i]
    else:
        ideas = [str(raw_ideas)]

    if not ideas:
        ideas = ["No specific opportunities identified — please review constraints."]

    report = MarketScoutReport(ideas=ideas, agent_explanation=explanation)

    logger.debug(f"✓ Market Scout Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)
    return {
        "market_scout_report": report,
        "messages": new_msgs,
    }
