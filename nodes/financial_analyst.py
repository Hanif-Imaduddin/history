"""Financial Analyst Agent — projections, risk assessment, dan Monte Carlo summary."""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_react_loop
from functions.llm import get_llm
from states.schema import EBPState, FinancialAnalysisReport
from tools.internet_search import internet_search

_SYSTEM_PROMPT = """You are the Financial Analyst Agent in a multi-agent AI business planning system.
You produce rigorous financial analyses for early-stage business ventures.

Your deliverables:
1. **Initial Investment Estimate** — startup costs broken down by category
2. **Revenue Projections** — Year 1, Year 2, Year 3 with assumptions stated
3. **Cost Structure** — fixed vs. variable costs, CAC, LTV
4. **Break-even Analysis** — when the business becomes cash-flow positive
5. **Cash Flow Summary** — quarterly for Year 1
6. **Risk Assessment** — top 3–5 financial risks with probability and impact
7. **Monte Carlo Summary** — describe optimistic / base / pessimistic scenarios
   with probability-weighted outcomes (simulate conceptually; use ranges)
8. **Key Financial Metrics** — ROI, payback period, CAGR, LTV:CAC ratio

Use the `internet_search` tool to gather real data:
- Average salaries and operational costs in Indonesia for this sector
- Market pricing benchmarks for similar products/services
- Typical CAC and conversion rates in the sector
- Funding landscape (seed round sizes, valuations)

Run at least 3–5 searches.

OUTPUT FORMAT — respond with ONLY valid JSON after your research:
{
  "analysis_result": "Full financial analysis in well-structured markdown, covering all 8 deliverables above with real data cited where possible."
}"""


def financial_analyst_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Financial Analyst Agent."""
    bc = state.get("bussiness_constraints")
    sr = state.get("strategic_report")
    msr = state.get("market_scout_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== MARKET OPPORTUNITIES IDENTIFIED ===",
            "; ".join(msr.ideas[:3]),
        ]

    if sr:
        context_lines += [
            "\n=== STRATEGIC CONTEXT (SWOT excerpt) ===",
            sr.swot_analysis[:800],
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK (address these points) ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR'S FEEDBACK ===", user_fb]

    context_lines.append(
        "\nSearch for cost benchmarks and pricing data, then produce the financial analysis JSON."
    )

    llm = get_llm(temperature=0.4)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_react_loop(
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        max_tool_rounds=6,
    )

    parsed = extract_json(final_response.content)
    analysis = parsed.get("analysis_result", final_response.content)

    report = FinancialAnalysisReport(analysis_result=analysis)

    return {
        "financial_analysis_report": report,
        "messages": new_msgs,
    }
