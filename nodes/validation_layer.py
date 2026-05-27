"""
Validation Layer — Rule-Based Contradiction Detection

Dijalankan SETELAH semua 4 agent selesai, SEBELUM Lead Orchestrator.
Tidak butuh LLM — murni rule-based logic.

Checks:
1. Financial vs Market contradictions
2. Budget vs Cost contradictions  
3. Risk level consistency
4. Missing critical assumptions
5. Unrealistic number detection
6. Cross-agent confidence gap

Output: ValidationReport yang dibaca orchestrator untuk scoring final.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from states.schema import (
    ContradictionFlag,
    EBPState,
    ValidationReport,
)

logger = logging.getLogger("clario.validation_layer")


# ─────────────────────────────────────────────
# RULE ENGINE
# ─────────────────────────────────────────────

def _check_financial_vs_market(
    financial,
    market,
    flags: list,
    unrealistic: list,
):
    """Rule: Jika market competition=high tapi profit margin sangat tinggi, flag."""
    if not financial or not market:
        return

    monthly_revenue = financial.estimated_monthly_revenue_rp
    monthly_cost = financial.estimated_monthly_cogs_rp + financial.estimated_monthly_fixed_costs_rp
    if monthly_cost > 0:
        gross_margin = (monthly_revenue - monthly_cost) / monthly_cost * 100
    else:
        gross_margin = 0

    # Rule 1: Competition tinggi tapi margin > 80%
    if market.competition_level == "high" and gross_margin > 80:
        flags.append(ContradictionFlag(
            severity="critical",
            description=(
                f"Market competition=HIGH namun projected gross margin={gross_margin:.0f}%. "
                "Di pasar kompetitif, margin setinggi ini tidak realistis tanpa differensiasi kuat."
            ),
            source_agents=["market_scout", "financial_analyst"],
        ))

    # Rule 2: Demand rendah tapi proyeksi revenue sangat agresif
    if market.demand_score <= 3 and monthly_revenue > 50_000_000:
        flags.append(ContradictionFlag(
            severity="critical",
            description=(
                f"Market demand_score={market.demand_score}/10 (sangat rendah) "
                f"namun projected monthly revenue=Rp {monthly_revenue:,}. "
                "Proyeksi revenue tidak konsisten dengan sinyal permintaan pasar."
            ),
            source_agents=["market_scout", "financial_analyst"],
        ))

    # Rule 3: BEP tidak tercapai dalam runway
    if financial.cash_runway_months > 0 and financial.estimated_break_even_months > 0:
        if financial.estimated_break_even_months > financial.cash_runway_months:
            flags.append(ContradictionFlag(
                severity="critical",
                description=(
                    f"BEP diperkirakan bulan ke-{financial.estimated_break_even_months}, "
                    f"namun cash runway hanya {financial.cash_runway_months} bulan. "
                    "Bisnis akan kehabisan modal sebelum mencapai titik impas."
                ),
                source_agents=["financial_analyst"],
            ))


def _check_budget_vs_startup_cost(
    financial,
    constraints,
    flags: list,
    unrealistic: list,
):
    """Rule: Startup cost tidak boleh jauh melebihi budget range user."""
    if not financial or not constraints:
        return

    # Parse budget range — ambil angka maksimum dari string seperti "Rp 10.000.000 – Rp 50.000.000"
    import re
    nums = re.findall(r"[\d\.]+", constraints.budget_range.replace(".", "").replace(",", ""))
    budget_max = int(nums[-1]) if nums else 0

    startup_cost = financial.estimated_startup_cost_rp

    # Rule: Startup cost > 150% budget max → flag
    if budget_max > 0 and startup_cost > budget_max * 1.5:
        flags.append(ContradictionFlag(
            severity="critical",
            description=(
                f"Estimated startup cost=Rp {startup_cost:,} melebihi 150% dari "
                f"budget maksimum user (Rp {budget_max:,}). "
                "Bisnis tidak terjangkau dengan modal yang tersedia."
            ),
            source_agents=["financial_analyst"],
        ))

    # Rule: Startup cost > budget max (tapi < 150%) → warning
    elif budget_max > 0 and startup_cost > budget_max:
        flags.append(ContradictionFlag(
            severity="warning",
            description=(
                f"Startup cost Rp {startup_cost:,} sedikit melebihi budget maksimum "
                f"Rp {budget_max:,}. Perlu pertimbangan sumber pendanaan tambahan."
            ),
            source_agents=["financial_analyst"],
        ))


def _check_risk_consistency(
    financial,
    ethics,
    market,
    constraints,
    flags: list,
):
    """Rule: Risk tolerance user vs actual risk level dari semua agent."""
    if not constraints:
        return

    risk_tolerance = constraints.risk_tolerance  # "low", "medium", "high"

    risk_signals = []

    if market and market.competition_level == "high":
        risk_signals.append("market:high_competition")

    if financial and financial.financial_feasibility_verdict == "TIDAK_LAYAK":
        risk_signals.append("financial:not_feasible")

    if ethics and ethics.legal_risk_level == "high":
        risk_signals.append("ethics:high_legal_risk")

    # Rule: User ingin low risk tapi banyak high-risk signals
    if risk_tolerance == "low" and len(risk_signals) >= 2:
        flags.append(ContradictionFlag(
            severity="warning",
            description=(
                f"User risk_tolerance=LOW namun terdeteksi {len(risk_signals)} sinyal risiko tinggi: "
                f"{', '.join(risk_signals)}. Bisnis ini mungkin tidak sesuai profil risiko user."
            ),
            source_agents=["market_scout", "financial_analyst", "ethics_agent"],
        ))


def _check_unrealistic_numbers(financial, flags: list, unrealistic: list):
    """Rule: Deteksi angka yang secara logika tidak masuk akal."""
    if not financial:
        return

    revenue = financial.estimated_monthly_revenue_rp
    net_profit = financial.estimated_monthly_net_profit_rp

    # Rule: Net profit > revenue (impossible)
    if net_profit > revenue:
        unrealistic.append(
            f"Net profit (Rp {net_profit:,}) > monthly revenue (Rp {revenue:,}). "
            "Ini secara matematika tidak mungkin — periksa kalkulasi agent."
        )

    # Rule: Revenue 0 atau negatif
    if revenue <= 0:
        unrealistic.append(
            "Projected monthly revenue <= 0. Agent mungkin gagal mengkalkulasi proyeksi."
        )

    # Rule: Break even < 0 bulan
    if financial.estimated_break_even_months < 0:
        unrealistic.append(
            "Break-even months negatif — indikasi error kalkulasi pada Financial Analyst."
        )


def _check_missing_assumptions(financial, market, flags: list, missing: list):
    """Rule: Cek apakah assumptions sudah eksplisit dideklarasikan."""
    if financial and len(financial.assumptions) == 0:
        missing.append(
            "Financial Analyst tidak mendeklarasikan asumsi kalkulasi. "
            "Angka tidak dapat diverifikasi."
        )

    if market and len(market.customer_pain_points) == 0:
        missing.append(
            "Market Scout tidak mengidentifikasi customer pain points. "
            "Validasi peluang pasar tidak lengkap."
        )

    if market and not market.market_sizing_methodology:
        missing.append(
            "Market Scout tidak menjelaskan metodologi TAM-SAM-SOM. "
            "Angka market size tidak dapat diverifikasi."
        )


def _compute_overall_confidence(
    market,
    financial,
    ethics,
    strategy,
    contradictions: list,
    missing: list,
    unrealistic: list,
) -> int:
    """
    Hybrid confidence scoring — bukan full LLM imagination.
    
    Faktor:
    - Base score dari rata-rata confidence agent
    - Penalty per critical contradiction (-15)
    - Penalty per warning (-5)
    - Penalty per missing assumption (-5)
    - Penalty per unrealistic claim (-10)
    """
    scores = []
    if market:
        scores.append(market.confidence_score)
    if financial:
        scores.append(financial.confidence_score)
    if ethics:
        scores.append(ethics.confidence_score)
    if strategy:
        scores.append(strategy.confidence_score)

    base = int(sum(scores) / len(scores)) if scores else 5

    # Penalties
    penalty = 0
    for c in contradictions:
        if c.severity == "critical":
            penalty += 15
        elif c.severity == "warning":
            penalty += 5

    penalty += len(missing) * 5
    penalty += len(unrealistic) * 10

    return max(1, min(10, base - penalty // 10))


# ─────────────────────────────────────────────
# MAIN NODE
# ─────────────────────────────────────────────

def validation_layer_node(state: EBPState) -> dict[str, Any]:
    """
    LangGraph node — Rule-based validation layer.
    
    Tidak memanggil LLM. Murni logic checks.
    Dijalankan setelah semua agent selesai.
    """
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Validation Layer dimulai")

    market = state.get("market_report")
    financial = state.get("financial_report")
    ethics = state.get("ethics_report")
    strategy = state.get("strategy_report")
    constraints = state.get("business_constraints")

    contradiction_flags: list[ContradictionFlag] = []
    missing_assumptions: list[str] = []
    unrealistic_claims: list[str] = []
    auto_flags: list[str] = []

    # ── Jalankan semua rules ──
    _check_financial_vs_market(financial, market, contradiction_flags, unrealistic_claims)
    _check_budget_vs_startup_cost(financial, constraints, contradiction_flags, unrealistic_claims)
    _check_risk_consistency(financial, ethics, market, constraints, contradiction_flags)
    _check_unrealistic_numbers(financial, contradiction_flags, unrealistic_claims)
    _check_missing_assumptions(financial, market, contradiction_flags, missing_assumptions)

    # ── Auto-flags informatif ──
    if market and market.demand_score >= 8 and market.competition_level == "low":
        auto_flags.append("🟢 Blue ocean signal: demand tinggi + kompetisi rendah.")

    if financial and financial.financial_feasibility_verdict == "TIDAK_LAYAK":
        auto_flags.append("🔴 Financial verdict TIDAK LAYAK — orchestrator wajib mempertimbangkan pivot.")

    if ethics and ethics.legal_risk_level == "high":
        auto_flags.append("🔴 Legal risk HIGH — wajib konsultasi hukum sebelum launch.")

    # ── Compute confidence ──
    overall_confidence = _compute_overall_confidence(
        market, financial, ethics, strategy,
        contradiction_flags, missing_assumptions, unrealistic_claims,
    )

    # ── Determine is_valid ──
    has_critical = any(c.severity == "critical" for c in contradiction_flags)
    is_valid = not has_critical

    report = ValidationReport(
        contradictions=contradiction_flags,
        missing_assumptions=missing_assumptions,
        unrealistic_claims=unrealistic_claims,
        auto_flags=auto_flags,
        overall_confidence=overall_confidence,
        is_valid=is_valid,
    )

    critical_count = sum(1 for c in contradiction_flags if c.severity == "critical")
    warning_count = sum(1 for c in contradiction_flags if c.severity == "warning")

    logger.debug(
        f"   Validation selesai — "
        f"Critical: {critical_count} | Warning: {warning_count} | "
        f"Missing: {len(missing_assumptions)} | Confidence: {overall_confidence}/10"
    )
    logger.debug(f"✓ Validation Layer selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "validation_report": report,
        "confidence_score": overall_confidence,
    }
