"""
Lead Orchestrator (Refactored) — Verdict Engine

Perubahan dari versi lama:
- TIDAK generate essay atau synthesis panjang
- Membaca ValidationReport dari validation layer (rule-based checks sudah dilakukan)
- Hanya: compute final verdict, top 3 recommendations, orchestrator_feedback concise
- final_summary_node dipisah dan generate report dari structured state
- Messages hanya simpan concise metadata, bukan full synthesis
- Tidak ada Tree of Thoughts panjang — verdict cepat dari structured inputs
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from functions.agent_utils import extract_json
from functions.llm import get_llm
from states.schema import EBPState

logger = logging.getLogger("clario.lead_orchestrator")


# ─────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────

_ORCHESTRATOR_PROMPT = """You are the Lead Orchestrator of a multi-agent business planning system.
You receive STRUCTURED data from 4 specialist agents + a validation report.
Your job: issue a final verdict and top 3 recommendations. Be concise and decisive.

VERDICT OPTIONS:
- "PROCEED": semua sinyal positif, risiko terkendali
- "PROCEED_WITH_CAUTION": ada risiko signifikan tapi bisnis viable, perlu mitigation
- "PIVOT_RECOMMENDED": business model perlu diubah fundamental sebelum proceed
- "NOT_RECOMMENDED": terlalu banyak critical issues, risiko terlalu tinggi

VERDICT RULES:
- NOT_RECOMMENDED jika: financial verdict TIDAK_LAYAK + legal_risk HIGH + demand_score < 4
- PIVOT_RECOMMENDED jika: validation has_critical=True, atau break-even > 24 bulan
- PROCEED_WITH_CAUTION jika: ada 1-2 warning flags tapi financial viable
- PROCEED jika: semua agent confidence >= 7, tidak ada critical flags, financial LAYAK

OUTPUT FORMAT — ONLY valid JSON:
{
  "final_verdict": "PROCEED_WITH_CAUTION",
  "final_recommendations": [
    "Rekomendasi prioritas 1 yang paling krusial dan actionable",
    "Rekomendasi 2",
    "Rekomendasi 3"
  ],
  "approval_status": "approved",
  "orchestrator_feedback": "Feedback concise untuk agent jika rejected (kosong jika approved)",
  "verdict_rationale": "1-2 kalimat justifikasi verdict"
}"""


_FINAL_SUMMARY_PROMPT = """You are the Report Generator for a business planning system.
Generate a clear, professional final report in Bahasa Indonesia from structured business analysis data.

RULES:
- Tulis dalam Bahasa Indonesia
- Semua angka keuangan dalam Rupiah (Rp)
- Pakai format Markdown
- Ringkas tapi komprehensif — bukan akademis
- Fokus pada: executive summary, angka kunci, risiko utama, rekomendasi top 3, verdict
- JANGAN tulis ulang semua data — sintesis, bukan transcription
- Maksimal ~600 kata

STRUKTUR LAPORAN:
# Laporan Analisis Bisnis — [Nama Ide Bisnis]

## Ringkasan Eksekutif
(2 paragraf: gambaran bisnis + verdict overall)

## Angka Kunci
| Metrik | Nilai |
(tabel: revenue/bulan, profit/bulan, BEP, startup cost, break-even)

## Analisis Pasar
(3-4 bullet: demand score, competition, pain points, peluang)

## Risiko Utama
(3-5 bullet: financial risks + legal risks yang paling kritikal)

## Rekomendasi Top 3
(numbered list: 3 aksi paling penting dan urgen)

## Putusan Akhir
**[PROCEED / PROCEED WITH CAUTION / PIVOT RECOMMENDED / NOT RECOMMENDED]**
(1 paragraf justifikasi)"""


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def _build_orchestrator_prompt(state: EBPState) -> str:
    """Build concise structured summary untuk orchestrator — bukan dump semua data."""
    market = state.get("market_report")
    financial = state.get("financial_report")
    ethics = state.get("ethics_report")
    strategy = state.get("strategy_report")
    validation = state.get("validation_report")
    bc = state.get("business_constraints")

    lines = [
        "=== STRUCTURED AGENT OUTPUTS ===",
        f"Bisnis: {bc.business_idea if bc else '-'} | Sektor: {bc.sector if bc else '-'}",
        "",
    ]

    # Market summary
    if market:
        lines += [
            "MARKET:",
            f"  demand_score={market.demand_score}/10 | competition={market.competition_level} | opportunity={market.opportunity_score}/10",
            f"  confidence={market.confidence_score}/10",
        ]

    # Financial summary — angka kunci saja
    if financial:
        lines += [
            "FINANCIAL:",
            f"  monthly_revenue=Rp {financial.estimated_monthly_revenue_rp:,}",
            f"  monthly_net_profit=Rp {financial.estimated_monthly_net_profit_rp:,}",
            f"  startup_cost=Rp {financial.estimated_startup_cost_rp:,}",
            f"  break_even={financial.estimated_break_even_months} bulan",
            f"  cash_runway={financial.cash_runway_months} bulan",
            f"  verdict={financial.financial_feasibility_verdict}",
            f"  confidence={financial.confidence_score}/10",
        ]

    # Ethics summary
    if ethics:
        lines += [
            "ETHICS:",
            f"  legal_risk={ethics.legal_risk_level} | compliance_score={ethics.compliance_score}/10",
            f"  key_risks: {'; '.join(ethics.critical_risk_alerts[:2])}",
        ]

    # Strategy summary
    if strategy:
        lines += [
            "STRATEGY:",
            f"  top_priorities: {'; '.join(strategy.top_problem_priorities[:2])}",
            f"  confidence={strategy.confidence_score}/10",
        ]

    # Validation report — paling penting untuk orchestrator
    if validation:
        lines += [
            "",
            "=== VALIDATION LAYER RESULTS ===",
            f"  is_valid={validation.is_valid}",
            f"  overall_confidence={validation.overall_confidence}/10",
        ]
        if validation.contradictions:
            lines.append("  CONTRADICTIONS:")
            for c in validation.contradictions:
                lines.append(f"    [{c.severity.upper()}] {c.description}")
        if validation.missing_assumptions:
            lines.append(f"  MISSING: {'; '.join(validation.missing_assumptions)}")
        if validation.unrealistic_claims:
            lines.append(f"  UNREALISTIC: {'; '.join(validation.unrealistic_claims)}")
        if validation.auto_flags:
            lines.append(f"  FLAGS: {'; '.join(validation.auto_flags)}")

    lines.append("\nBerikan verdict final dan top 3 recommendations. Return ONLY JSON.")
    return "\n".join(lines)


def _build_final_summary_prompt(state: EBPState) -> str:
    """Build prompt untuk final report generator dari structured state."""
    market = state.get("market_report")
    financial = state.get("financial_report")
    ethics = state.get("ethics_report")
    strategy = state.get("strategy_report")
    validation = state.get("validation_report")
    bc = state.get("business_constraints")

    lines = [
        "=== DATA BISNIS UNTUK LAPORAN AKHIR ===",
        f"Ide Bisnis: {bc.business_idea if bc else '-'}",
        f"Sektor: {bc.sector if bc else '-'}",
        f"Lokasi: {bc.location if bc else '-'}",
        f"Budget: {bc.budget_range if bc else '-'}",
        "",
    ]

    if market:
        lines += [
            "PASAR:",
            f"  Demand Score: {market.demand_score}/10",
            f"  Competition: {market.competition_level}",
            f"  Pain Points: {', '.join(market.customer_pain_points)}",
            f"  Peluang: {', '.join(market.validated_opportunities)}",
            f"  TAM: {market.tam_estimate} | SAM: {market.sam_estimate} | SOM: {market.som_estimate}",
        ]

    if financial:
        lines += [
            "",
            "KEUANGAN:",
            f"  Revenue/bulan: Rp {financial.estimated_monthly_revenue_rp:,}",
            f"  COGS/bulan: Rp {financial.estimated_monthly_cogs_rp:,}",
            f"  Fixed Costs/bulan: Rp {financial.estimated_monthly_fixed_costs_rp:,}",
            f"  Net Profit/bulan: Rp {financial.estimated_monthly_net_profit_rp:,}",
            f"  BEP Revenue: Rp {financial.bep_revenue_rp:,} ({financial.bep_units_per_month} unit/bulan)",
            f"  Startup Cost: Rp {financial.estimated_startup_cost_rp:,}",
            f"  Break-Even: {financial.estimated_break_even_months} bulan",
            f"  Cash Runway: {financial.cash_runway_months} bulan",
            f"  Verdict Finansial: {financial.financial_feasibility_verdict}",
            f"  Key Risks: {', '.join(financial.key_financial_risks[:3])}",
            f"  Asumsi: {', '.join(financial.assumptions[:3])}",
        ]
        if financial.sensitivity_scenarios:
            lines.append("  Stress Test:")
            for s in financial.sensitivity_scenarios:
                lines.append(f"    - {s.label}: Rp {s.net_profit_rp:,}")

    if strategy:
        lines += [
            "",
            "STRATEGI:",
            f"  Strengths: {', '.join(strategy.strengths)}",
            f"  Weaknesses: {', '.join(strategy.weaknesses)}",
            f"  Opportunities: {', '.join(strategy.opportunities)}",
            f"  Threats: {', '.join(strategy.threats)}",
            f"  PESTEL: {', '.join(strategy.pestel_highlights[:4])}",
            f"  Execution Order: {strategy.strategic_order_of_operations}",
        ]
        m1 = strategy.roadmap.get("month_1")
        m2 = strategy.roadmap.get("month_2")
        m3 = strategy.roadmap.get("month_3")
        if m1:
            lines.append(f"  Bulan 1: {m1.objective} | KPIs: {', '.join(m1.kpis)}")
        if m2:
            lines.append(f"  Bulan 2: {m2.objective} | KPIs: {', '.join(m2.kpis)}")
        if m3:
            lines.append(f"  Bulan 3: {m3.objective} | KPIs: {', '.join(m3.kpis)}")

    if ethics:
        lines += [
            "",
            "HUKUM & KEPATUHAN:",
            f"  Izin Wajib: {', '.join(ethics.mandatory_permits)}",
            f"  Izin Sektoral: {', '.join(ethics.sector_specific_permits)}",
            f"  Legal Risk Level: {ethics.legal_risk_level}",
            f"  Kompleksitas Birokrasi: {ethics.bureaucracy_complexity} ({ethics.bureaucracy_estimated_duration})",
            f"  Risk Alerts: {', '.join(ethics.critical_risk_alerts[:2])}",
        ]

    if validation:
        lines += [
            "",
            "VALIDATION:",
            f"  Overall Confidence: {validation.overall_confidence}/10",
            f"  Auto Flags: {', '.join(validation.auto_flags)}",
        ]

    final_verdict = state.get("final_verdict", "PROCEED_WITH_CAUTION")
    final_recs = state.get("final_recommendations", [])
    lines += [
        "",
        f"VERDICT FINAL: {final_verdict}",
        f"REKOMENDASI TOP 3: {', '.join(final_recs)}",
    ]

    lines.append("\nGenerate laporan akhir dalam Bahasa Indonesia sesuai format Markdown yang diminta.")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR NODE
# ─────────────────────────────────────────────

def lead_orchestrator_node(state: EBPState) -> dict[str, Any]:
    """
    LangGraph node — Lead Orchestrator.
    
    Hanya: compute verdict + top 3 recs + approval status.
    Bukan essay generator.
    """
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Lead Orchestrator dimulai")

    market = state.get("market_report")
    financial = state.get("financial_report")
    ethics = state.get("ethics_report")
    strategy = state.get("strategy_report")

    # First pass — tidak ada report → routing ke market scout
    if all(r is None for r in [market, financial, ethics, strategy]):
        logger.debug("  First pass → routing ke Market Scout")
        return {
            "approval_status": "pending",
            "orchestrator_feedback": None,
            "messages": [SystemMessage(content="[Orchestrator] First pass — routing to Market Scout.")],
        }

    llm = get_llm(temperature=0.2)
    prompt = _build_orchestrator_prompt(state)
    iteration = state.get("iteration", 0)

    logger.debug(f"  Evaluating iteration {iteration}...")
    t_llm = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=_ORCHESTRATOR_PROMPT),
        HumanMessage(content=prompt),
    ])
    logger.debug(f"  LLM selesai dalam {time.perf_counter() - t_llm:.2f}s")

    parsed = extract_json(response.content)

    # ── Extract fields ──
    approval_status = parsed.get("approval_status", "rejected")
    if approval_status not in ("approved", "rejected"):
        approval_status = "rejected"

    raw_verdict = parsed.get("final_verdict", "PROCEED_WITH_CAUTION")
    valid_verdicts = {"PROCEED", "PROCEED_WITH_CAUTION", "PIVOT_RECOMMENDED", "NOT_RECOMMENDED"}
    final_verdict = raw_verdict if raw_verdict in valid_verdicts else "PROCEED_WITH_CAUTION"

    final_recs = parsed.get("final_recommendations", [])[:3]  # Max 3
    feedback = parsed.get("orchestrator_feedback", "")
    rationale = parsed.get("verdict_rationale", "")

    # ── Force approve on max iteration ──
    max_iter = state.get("max_iterations", 3)
    if iteration >= max_iter and approval_status == "rejected":
        approval_status = "approved"
        feedback = "Iterasi maksimum tercapai."

    # ── Concise summary message ──
    summary_msg = (
        f"[Orchestrator iter={iteration + 1}] "
        f"verdict={final_verdict} | status={approval_status} | "
        f"rationale={rationale}"
    )

    logger.debug(f"   verdict={final_verdict} | status={approval_status}")
    logger.debug(f"✓ Lead Orchestrator selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    # ── Interrupt untuk user feedback jika rejected ──
    if approval_status == "rejected" and (iteration + 1) < max_iter:
        user_fb = interrupt({
            "orchestrator_feedback": feedback,
            "final_verdict": final_verdict,
            "final_recommendations": final_recs,
            "iteration": iteration + 1,
        })
        return {
            "approval_status": approval_status,
            "orchestrator_feedback": feedback,
            "final_verdict": final_verdict,
            "final_recommendations": final_recs,
            "iteration": iteration + 1,
            "user_feedback": user_fb if isinstance(user_fb, str) else None,
            "messages": [SystemMessage(content=summary_msg)],
        }

    return {
        "approval_status": approval_status,
        "orchestrator_feedback": feedback,
        "final_verdict": final_verdict,
        "final_recommendations": final_recs,
        "iteration": iteration + 1,
        "messages": [SystemMessage(content=summary_msg)],
    }


# ─────────────────────────────────────────────
# FINAL SUMMARY NODE
# ─────────────────────────────────────────────

def final_summary_node(state: EBPState) -> dict[str, Any]:
    """
    LangGraph node — Final Report Generator.
    
    Dipisah dari orchestrator.
    Membaca dari structured state → generate human-readable Markdown.
    Bisa menggunakan model lebih murah/cepat.
    """
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Final Summary dimulai")

    llm = get_llm(temperature=0.3)
    prompt = _build_final_summary_prompt(state)

    logger.debug("  Generating final Markdown report...")
    t_llm = time.perf_counter()
    response = llm.invoke([
        SystemMessage(content=_FINAL_SUMMARY_PROMPT),
        HumanMessage(content=prompt),
    ])
    logger.debug(f"  LLM selesai dalam {time.perf_counter() - t_llm:.2f}s")

    final_md = response.content.strip()

    logger.debug(f"✓ Final Summary selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "final_result": final_md,
        # Tidak simpan full report di messages — simpan di final_result
        "messages": [SystemMessage(content="[Final Report Generated]")],
    }
