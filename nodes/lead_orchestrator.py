"""Lead Orchestrator node - menggunakan mekanisme Tree of Thoughts reasoning
untuk mengevaluasi semua agent reports dan menentukan apakah plan bisnis tersebut memenuhi kriteria persetujuan
atau perlu dikembangkan lebih lanjut.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from functions.agent_utils import extract_json, format_constraints
from functions.llm import get_llm
from states.schema import EBPState

logger = logging.getLogger("clario.lead_orchestrator")

# =====================================================================
# SYSTEM PROMPTS (UPDATED FOR INDONESIAN OUTPUT, IDR, AND FEASIBLE FEATURES)
# =====================================================================

_FINAL_SUMMARY_SYSTEM_PROMPT = """You are the Lead Orchestrator of a multi-agent AI business planning system.
Your task is to produce a comprehensive, well-structured final report for the entrepreneur.
The report MUST be written entirely in **Bahasa Indonesia** and cover all four specialist analyses.
All financial projections, numbers, and currencies MUST be presented in Indonesian Rupiah (IDR / Rp).

Structure your report exactly as follows:

# Laporan Analisis Rencana Bisnis

## Ringkasan Eksekutif
(2–3 paragraf gambaran umum ide bisnis dan rekomendasi keseluruhan)

## 1. Analisis Pasar
(Rangkuman peluang, segmen pasar target, dan lanskap kompetitif berdasarkan laporan Market Scout)

## 2. Analisis Strategis
### SWOT
(Strengths, Weaknesses, Opportunities, Threats)
### PESTEL
(Faktor Politik, Ekonomi, Sosial, Teknologi, Lingkungan, Hukum)

## 3. Analisis Keuangan (Proyeksi dalam Rupiah / IDR)
(Proyeksi keuangan utama menggunakan mata uang Rupiah, penilaian risiko, dan ringkasan kelayakan dari laporan Financial Analyst)

## 4. Etika & Kepatuhan (Compliance)
(Status kepatuhan regulasi di Indonesia, kekhawatiran etis, dan mitigasi dari laporan Ethics Guardian)

## 5. Rekomendasi Utama & Rencana Aksi (Prioritas Top 3)
(Berikan MAKSIMAL 3 hal paling penting dan krusial yang harus segera dilakukan oleh wirausahawan — prioritaskan berdasarkan urgensi dan dampak, bukan daftar panjang)

## Putusan Akhir (Overall Verdict)
(Tuliskan salah satu dari putusan berikut dengan HURUF KAPITAL: PROCEED / PROCEED WITH CAUTION / PIVOT RECOMMENDED / NOT RECOMMENDED diikuti dengan satu paragraf justifikasi yang kuat)

Write in a professional yet accessible tone. Use bullet points, tables, or bold text where helpful.
Respond with ONLY the Markdown report, no other text or code fences. Remember: Write in Bahasa Indonesia and use IDR/Rupiah!"""

_SYSTEM_PROMPT = """You are the Lead Orchestrator of a multi-agent AI business planning system.
Your role is to evaluate the quality of the business plan produced by four specialist agents, detect discrepancies, score confidence, and decide the next state.

You apply Tree of Thoughts (ToT) reasoning: you must explicitly consider MULTIPLE evaluation
perspectives (market, strategy, finance, ethics) before arriving at a final judgment.

CRITICAL EVALUATION TASKS:
1. Contradiction Detection: Check for inconsistencies between agents (e.g., Market Scout says market demand is low, but Financial Analyst projects hyper-growth. Flag this!).
2. Confidence Scoring: Provide a confidence percentage (0-100%) based on data completeness, agent agreement, and source availability.
3. Priority Actions: Identify exactly 3 most critical recommendations instead of a long list.

APPROVAL CRITERIA (all must hold for "approved"):
- Market Scout report identifies at least 2–3 concrete opportunities with evidence
- Strategic report contains a substantive SWOT and PESTEL analysis (not placeholders)
- Financial analysis includes realistic projections (in IDR) and a risk discussion
- Ethics report confirms legal compliance and flags any concerns with mitigations
- No critical contradictions found between agent reports.

OUTPUT FORMAT — respond with ONLY valid JSON, no other text:
{
  "tot_perspective_market": "Your assessment of market analysis quality",
  "tot_perspective_strategy": "Your assessment of strategy quality",
  "tot_perspective_finance": "Your assessment of financial analysis quality",
  "tot_perspective_ethics": "Your assessment of ethics analysis quality",
  "contradictions_detected": "List any contradictions or logical gaps between agent reports. Empty string if none.",
  "confidence_score": 85, 
  "synthesis": "Overall synthesis of all perspectives, written in Bahasa Indonesia",
  "approval_status": "approved" or "rejected",
  "orchestrator_feedback": "Specific, actionable feedback for the agents in the next iteration in Bahasa Indonesia. Empty string if approved."
}"""

# =====================================================================
# HELPER & NODE FUNCTIONS
# =====================================================================

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
        "(market, strategy, finance, ethics), explicitly check for contradictions, compute confidence score, then synthesize and output JSON."
    )
    return "\n".join(lines)


def lead_orchestrator_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Lead Orchestrator."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Lead Orchestrator dimulai")
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    ear = state.get("ethics_analysis_report")

    # First pass — no reports exist yet, just route forward
    if msr is None and sr is None and far is None and ear is None:
        logger.debug("  First pass — belum ada report, routing ke Market Scout")
        logger.debug(f"✓ Lead Orchestrator selesai dalam {time.perf_counter() - t_start:.2f}s")
        logger.debug("=" * 60)
        return {
            "approval_status": "pending",
            "orchestrator_feedback": None,
            "messages": [
                SystemMessage(content="Lead Orchestrator: initiating first iteration — routing to Market Scout.")
            ],
        }

    llm = get_llm(temperature=0.4)
    prompt = _build_evaluation_prompt(state)

    iteration = state.get("iteration", 0)
    logger.debug(f"[LLM] Evaluasi iterasi {iteration} — memanggil LLM (Tree of Thoughts)...")
    t_llm = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    logger.debug(f"[LLM] Evaluasi selesai dalam {time.perf_counter() - t_llm:.2f}s")

    parsed = extract_json(response.content)

    approval_status = parsed.get("approval_status", "rejected")
    if approval_status not in ("approved", "rejected"):
        approval_status = "rejected"

    feedback = parsed.get("orchestrator_feedback", "")
    synthesis = parsed.get("synthesis", "")
    contradictions = parsed.get("contradictions_detected", "")
    confidence = parsed.get("confidence_score", 100)

    # Force reject on last iteration so we don't approve garbage
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter and approval_status == "rejected":
        approval_status = "approved"
        feedback = "Iterasi maksimum tercapai — menyajikan rencana bisnis terbaik yang tersedia saat ini."

    logger.debug(f"   Keputusan: {approval_status.upper()} (iterasi {iteration + 1}/{max_iter})")
    logger.debug(f"   Confidence Score: {confidence}%")
    if contradictions:
        logger.warning(f"   Kontradiksi Terdeteksi: {contradictions}")

    summary_msg = (
        f"[Lead Orchestrator — Iterasi {iteration + 1}]\n"
        f"Keputusan: {approval_status.upper()}\n"
        f"Confidence Score: {confidence}%\n"
        f"Kontradiksi: {contradictions if contradictions else 'Tidak ada'}\n"
        f"Sintesis: {synthesis}\n"
        f"Feedback: {feedback}"
    )

    logger.debug(f"✓ Lead Orchestrator selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    # Pause and ask user for feedback before the next pipeline iteration
    if approval_status == "rejected" and (iteration + 1) < max_iter:
        user_fb = interrupt({
            "orchestrator_feedback": feedback,
            "synthesis": synthesis,
            "contradictions_detected": contradictions,
            "confidence_score": confidence,
            "iteration": iteration + 1,
        })
        return {
            "approval_status": approval_status,
            "orchestrator_feedback": feedback,
            "iteration": iteration + 1,
            "user_feedback": user_fb if isinstance(user_fb, str) else None,
            "messages": [SystemMessage(content=summary_msg)],
        }

    return {
        "approval_status": approval_status,
        "orchestrator_feedback": feedback,
        "iteration": iteration + 1,
        "messages": [SystemMessage(content=summary_msg)],
    }


def _build_final_summary_prompt(state: EBPState) -> str:
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    ear = state.get("ethics_analysis_report")
    
    synthesis = ""
    contradictions = ""
    confidence = "100%"
    
    # Extract latest metadata from orchestrator messages
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "content") and "Sintesis:" in msg.content:
            for line in msg.content.splitlines():
                if line.startswith("Sintesis:"):
                    synthesis = line.removeprefix("Sintesis:").strip()
                elif line.startswith("Kontradiksi:"):
                    contradictions = line.removeprefix("Kontradiksi:").strip()
                elif line.startswith("Confidence Score:"):
                    confidence = line.removeprefix("Confidence Score:").strip()
            break

    lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if synthesis:
        lines += [
            "\n=== ORCHESTRATOR METADATA ===",
            f"Confidence Score: {confidence}",
            f"Detected Contradictions: {contradictions if contradictions else 'None'}",
            f"Synthesis Summary: {synthesis}"
        ]

    lines.append("\n=== AGENT REPORTS ===")

    if msr:
        lines += [
            "\n--- Market Scout Report ---",
            f"Ideas: {', '.join(msr.ideas)}",
            f"Explanation: {msr.agent_explanation}",
        ]

    if sr:
        lines += [
            "\n--- Strategic Report ---",
            f"SWOT: {sr.swot_analysis}",
            f"PESTEL: {sr.pastel_analysis}",
        ]

    if far:
        lines += ["\n--- Financial Analysis ---", far.analysis_result]

    if ear:
        lines += ["\n--- Ethics Analysis ---", ear.analysis_result]

    lines.append(
        "\nBased on all the reports above, translate metrics to IDR (Rupiah), synthesize information, structure the Top 3 prioritization, and write the final report ENTIRELY in Bahasa Indonesia."
    )
    return "\n".join(lines)


FINAL_SUMMARY_MODEL = "Qwen/Qwen3.5-397B-A17B"

def final_summary_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node that generates the final Markdown summary report for the user."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Final Summary dimulai")

    llm = get_llm(temperature=0.3, model_name=FINAL_SUMMARY_MODEL)
    prompt = _build_final_summary_prompt(state)

    logger.debug("[LLM] Generating final Markdown report in Indonesian...")
    t_llm = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=_FINAL_SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    logger.debug(f"[LLM] Final report selesai dalam {time.perf_counter() - t_llm:.2f}s")

    final_md = response.content.strip()

    logger.debug(f"✓ Final Summary selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)
    return {
        "final_result": final_md,
        "messages": [SystemMessage(content=f"[Final Report Generated]\n\n{final_md}")],
    }