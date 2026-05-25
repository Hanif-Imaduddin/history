"""Ethics Agent - Mengevaluasi kepatuhan terhadap peraturan hukum dan validasi etika."""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, EthicsAnalysisReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.ethics_agent")

_SYSTEM_PROMPT = """You are the Ethics Guardian Agent in a multi-agent AI business planning system.
You ensure that every business plan is legally compliant and ethically sound, especially under
Indonesian law and relevant international standards.

Your review covers:
1. **Legal Compliance (Indonesia)** — business licensing (NIB/OSS), sector-specific permits,
   OJK/BI regulations (if FinTech), BPOM (if health/food), Kominfo (if digital services),
   UU ITE, UU PDP (data privacy law), etc.
2. **Ethical Considerations** — fair employment, consumer protection, transparency, no deception
3. **Environmental Impact** — UU Lingkungan Hidup compliance, sustainability practices
4. **Data Privacy** — UU No. 27/2022 (Personal Data Protection) requirements
5. **Tax Obligations** — NPWP, PKP status, applicable tax rates
6. **Intellectual Property** — trademark, patent, copyright considerations
7. **Consumer Protection** — UU Perlindungan Konsumen No. 8/1999

Search results will be provided for you in bulk. Once you have them, synthesise all findings
into the compliance report. If critical data is still missing, you may call `internet_search`
for one or two additional targeted queries.

SCORING:
- Assign an Ethics & Compliance Score from 1–10 (10 = fully compliant, no ethical concerns)
- Score ≥ 7 is acceptable; below 7 requires mandatory remediation steps

OUTPUT FORMAT — respond with ONLY valid JSON after your research:
{
  "analysis_result": "Full ethics and compliance analysis in well-structured markdown covering all 7 areas above. Include: compliance status per area, identified risks, specific mitigation steps, and the overall Ethics & Compliance Score (X/10) with justification."
}"""


def ethics_agent_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Ethics Guardian Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Ethics Guardian Agent dimulai")
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== PROPOSED BUSINESS IDEAS ===",
            "; ".join(msr.ideas[:3]),
        ]

    if sr:
        context_lines += [
            "\n=== STRATEGIC OVERVIEW (Legal section from PESTEL) ===",
            sr.pastel_analysis[:600],
        ]

    if far:
        context_lines += [
            "\n=== FINANCIAL STRUCTURE (summary) ===",
            far.analysis_result[:400],
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK (address these points) ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR'S FEEDBACK ===", user_fb]

    context_lines.append(
        "\nReview the search results provided and produce the ethics/compliance JSON report."
    )

    _search_topics = [
        "business licensing NIB OSS requirements Indonesia for the target sector",
        "data privacy UU PDP Kominfo digital services regulations Indonesia 2024 2025",
        "consumer protection UU Perlindungan Konsumen sector-specific compliance Indonesia",
        "recent regulatory changes enforcement penalties in the sector Indonesia",
    ]

    llm = get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        planning_topics=_search_topics,
        max_followup_rounds=2,
        agent_name="ethics_agent",
        max_search_calls=6,
    )

    parsed = extract_json(final_response.content)
    analysis = parsed.get("analysis_result", final_response.content)

    report = EthicsAnalysisReport(analysis_result=analysis)

    logger.debug(f"✓ Ethics Guardian Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)
    return {
        "ethics_analysis_report": report,
        "messages": new_msgs,
    }
