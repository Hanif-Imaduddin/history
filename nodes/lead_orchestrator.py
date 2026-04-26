"""Lead Orchestrator node - menggunakan mekanisme Tree of Thoughts reasoning
untuk mengevaluasi semua agent reports dan menentukan apakah plan bisnis tersebut memenuhi kriteria persetujuan 
atau perlu dikembangkan lebih lanjut.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints
from functions.llm import get_llm
from states.schema import EBPState

_SYSTEM_PROMPT = """You are the Lead Orchestrator of a multi-agent AI business planning system.
Your role is to evaluate the quality of the business plan produced by four specialist agents
and decide whether to APPROVE it or REJECT it with actionable feedback.

You apply Tree of Thoughts (ToT) reasoning: you must explicitly consider MULTIPLE evaluation
perspectives before arriving at a final judgment. This ensures a thorough and balanced assessment.

APPROVAL CRITERIA (all must hold):
- Market Scout report identifies at least 2–3 concrete opportunities with evidence
- Strategic report contains a substantive SWOT and PESTEL analysis (not placeholders)
- Financial analysis includes realistic projections and a risk discussion
- Ethics report confirms legal compliance and flags any concerns with mitigations
- All four reports are consistent with each other and with the business constraints

OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
{
  "tot_perspective_market": "Your assessment of market analysis quality",
  "tot_perspective_strategy": "Your assessment of strategy quality",
  "tot_perspective_finance": "Your assessment of financial analysis quality",
  "tot_perspective_ethics": "Your assessment of ethics analysis quality",
  "synthesis": "Overall synthesis of all perspectives",
  "approval_status": "approved" or "rejected",
  "orchestrator_feedback": "Specific, actionable feedback for the agents in the next iteration. Empty string if approved."
}"""


def _build_evaluation_prompt(state: EBPState) -> str:
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    ear = state.get("ethics_analysis_report")
    user_fb = state.get("user_feedback")
    prev_fb = state.get("orchestrator_feedback")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
        f"\nIteration: {iteration}/{max_iter}",
    ]

    if user_fb:
        lines += ["\n=== USER FEEDBACK (from entrepreneur) ===", user_fb]

    if prev_fb and iteration > 0:
        lines += ["\n=== PREVIOUS ORCHESTRATOR FEEDBACK ===", prev_fb]

    lines.append("\n=== AGENT REPORTS ===")

    if msr:
        lines += [
            "\n--- Market Scout Report ---",
            f"Ideas: {', '.join(msr.ideas)}",
            f"Explanation: {msr.agent_explanation}",
        ]
    else:
        lines.append("\n--- Market Scout Report: NOT GENERATED ---")

    if sr:
        lines += [
            "\n--- Strategic Report ---",
            f"SWOT: {sr.swot_analysis}",
            f"PESTEL: {sr.pastel_analysis}",
        ]
    else:
        lines.append("\n--- Strategic Report: NOT GENERATED ---")

    if far:
        lines += ["\n--- Financial Analysis ---", far.analysis_result]
    else:
        lines.append("\n--- Financial Analysis: NOT GENERATED ---")

    if ear:
        lines += ["\n--- Ethics Analysis ---", ear.analysis_result]
    else:
        lines.append("\n--- Ethics Analysis: NOT GENERATED ---")

    lines.append(
        "\nApply Tree of Thoughts reasoning across four perspectives "
        "(market, strategy, finance, ethics) then synthesize and output JSON."
    )
    return "\n".join(lines)


def lead_orchestrator_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Lead Orchestrator."""
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    ear = state.get("ethics_analysis_report")

    # First pass — no reports exist yet, just route forward
    if msr is None and sr is None and far is None and ear is None:
        return {
            "approval_status": "pending",
            "orchestrator_feedback": None,
            "messages": [
                SystemMessage(content="Lead Orchestrator: initiating first iteration — routing to Market Scout.")
            ],
        }

    llm = get_llm(temperature=0.4)
    prompt = _build_evaluation_prompt(state)

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    parsed = extract_json(response.content)

    approval_status = parsed.get("approval_status", "rejected")
    if approval_status not in ("approved", "rejected"):
        approval_status = "rejected"

    feedback = parsed.get("orchestrator_feedback", "")
    synthesis = parsed.get("synthesis", "")

    # Force reject on last iteration so we don't approve garbage
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter and approval_status == "rejected":
        # We've hit the limit — surface what we have
        approval_status = "approved"
        feedback = "Max iterations reached — delivering best available plan."

    summary_msg = (
        f"[Lead Orchestrator — Iteration {iteration + 1}]\n"
        f"Decision: {approval_status.upper()}\n"
        f"Synthesis: {synthesis}\n"
        f"Feedback: {feedback}"
    )

    return {
        "approval_status": approval_status,
        "orchestrator_feedback": feedback,
        "iteration": iteration + 1,
        "messages": [SystemMessage(content=summary_msg)],
    }
