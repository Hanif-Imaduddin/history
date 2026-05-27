"""
Financial Analyst Agent (Refactored) — Structured Evaluator

Perubahan dari versi lama:
- Output ke FinancialAnalysisReport typed dataclass, bukan text blob
- Semua angka sebagai integer (IDR), bukan string
- SensitivityScenario sebagai typed objects
- Assumptions eksplisit dideklarasikan
- financial_feasibility_verdict sebagai Literal enum
- Max search dikurangi (benchmark data cukup 1-2 query)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import (
    EBPState,
    FinancialAnalysisReport,
    SensitivityScenario,
)
from tools.internet_search import internet_search

logger = logging.getLogger("clario.financial_analyst")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Financial Analyst Agent in a multi-agent business planning system.
Your job is to calculate financial projections and return STRUCTURED numerical data — not essays.

CRITICAL: Output ONLY valid JSON. All monetary values MUST be integers in Indonesian Rupiah (IDR). No "Rp" prefix in number fields — just the integer.

CALCULATION REQUIREMENTS:
1. Monthly cashflow = revenue - COGS - fixed_costs
2. BEP units = fixed_costs / (price_per_unit - variable_cost_per_unit)
3. BEP revenue = BEP_units × price_per_unit
4. Cash runway = startup_cost / |monthly_loss| (only if projected loss)
5. Break even months = startup_cost / monthly_net_profit (if profit > 0)
6. Sensitivity: recalculate net_profit for each scenario

SENSITIVITY SCENARIOS (always include these 3):
- Scenario A: sales volume turun 20% → new net profit
- Scenario B: COGS naik 15% → new net profit
- Scenario C: marketing/ads cost naik 50% → new net profit

VERDICT RULES:
- "LAYAK": net profit > 0, break_even < 12 months, startup cost within budget
- "LAYAK_DENGAN_CATATAN": net profit > 0 but break_even 12-24 months, OR startup cost slightly over budget
- "TIDAK_LAYAK": net profit <= 0, OR break_even > 24 months, OR startup cost > 2x budget

IMPORTANT: Be explicit about every assumption. If you don't know the exact price, STATE the assumption clearly.

OUTPUT FORMAT — ONLY valid JSON (all monetary = integer IDR):
{
  "estimated_monthly_revenue_rp": 15000000,
  "estimated_monthly_cogs_rp": 6000000,
  "estimated_monthly_fixed_costs_rp": 4000000,
  "estimated_monthly_net_profit_rp": 5000000,
  "bep_units_per_month": 150,
  "bep_revenue_rp": 9000000,
  "cash_runway_months": 6,
  "estimated_startup_cost_rp": 25000000,
  "estimated_break_even_months": 5,
  "sensitivity_scenarios": [
    {
      "label": "Volume penjualan turun 20%",
      "net_profit_rp": 2000000
    },
    {
      "label": "COGS naik 15%",
      "net_profit_rp": 1500000
    },
    {
      "label": "Marketing cost naik 50%",
      "net_profit_rp": 3000000
    }
  ],
  "industry_benchmark_notes": "2-3 kalimat perbandingan margin vs standar industri",
  "financial_feasibility_verdict": "LAYAK_DENGAN_CATATAN",
  "key_financial_risks": [
    "Risiko keuangan spesifik 1",
    "Risiko 2",
    "Risiko 3"
  ],
  "assumptions": [
    "Diasumsikan harga jual Rp X per unit",
    "Diasumsikan traffic 50 pelanggan/hari pada bulan pertama",
    "Diasumsikan sewa lokasi Rp Y/bulan berdasarkan benchmark area"
  ],
  "affordability_score": 7,
  "confidence_score": 7
}"""

_SEARCH_TOPICS = [
    "benchmark biaya operasional sewa dan gaji karyawan sektor target indonesia",
    "standar gross margin dan customer acquisition cost industri target indonesia",
]


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

def financial_analyst_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node — Financial Analyst Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Financial Analyst Agent dimulai")

    bc = state.get("business_constraints")
    market = state.get("market_report")
    strategy = state.get("strategy_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if market:
        context_lines += [
            "\n=== MARKET CONTEXT ===",
            f"Demand Score: {market.demand_score}/10",
            f"Competition Level: {market.competition_level}",
            f"TAM: {market.tam_estimate} | SAM: {market.sam_estimate} | SOM: {market.som_estimate}",
            f"Market Sizing Methodology: {market.market_sizing_methodology}",
        ]

    if strategy:
        context_lines += [
            "\n=== STRATEGY CONTEXT ===",
            f"Revenue Streams: {strategy.revenue_streams}",
            f"Cost Structure: {strategy.cost_structure}",
            f"Recommended Sales Channel: {strategy.recommended_sales_channel}",
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== FEEDBACK ENTREPRENEUR ===", user_fb]

    context_lines.append(
        "\nHitung proyeksi keuangan. Return ONLY structured JSON. Semua angka uang = integer IDR."
    )

    # Temperature 0.1 — minimasi halusinasi matematika
    llm = get_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        planning_topics=_SEARCH_TOPICS,
        max_followup_rounds=1,
        agent_name="financial_analyst",
        max_search_calls=2,     # Cukup untuk benchmark data
    )

    parsed = extract_json(final_response.content)

    # ── Parse sensitivity scenarios ──
    raw_scenarios = parsed.get("sensitivity_scenarios", [])
    scenarios = []
    for s in raw_scenarios:
        if isinstance(s, dict):
            scenarios.append(SensitivityScenario(
                label=s.get("label", ""),
                net_profit_rp=int(s.get("net_profit_rp", 0)),
            ))

    # ── Validate & normalize verdict ──
    raw_verdict = parsed.get("financial_feasibility_verdict", "LAYAK_DENGAN_CATATAN")
    valid_verdicts = {"LAYAK", "LAYAK_DENGAN_CATATAN", "TIDAK_LAYAK"}
    verdict = raw_verdict if raw_verdict in valid_verdicts else "LAYAK_DENGAN_CATATAN"

    report = FinancialAnalysisReport(
        estimated_monthly_revenue_rp=int(parsed.get("estimated_monthly_revenue_rp", 0)),
        estimated_monthly_cogs_rp=int(parsed.get("estimated_monthly_cogs_rp", 0)),
        estimated_monthly_fixed_costs_rp=int(parsed.get("estimated_monthly_fixed_costs_rp", 0)),
        estimated_monthly_net_profit_rp=int(parsed.get("estimated_monthly_net_profit_rp", 0)),
        bep_units_per_month=int(parsed.get("bep_units_per_month", 0)),
        bep_revenue_rp=int(parsed.get("bep_revenue_rp", 0)),
        cash_runway_months=int(parsed.get("cash_runway_months", 0)),
        estimated_startup_cost_rp=int(parsed.get("estimated_startup_cost_rp", 0)),
        estimated_break_even_months=int(parsed.get("estimated_break_even_months", 0)),
        sensitivity_scenarios=scenarios,
        industry_benchmark_notes=parsed.get("industry_benchmark_notes", ""),
        financial_feasibility_verdict=verdict,
        key_financial_risks=parsed.get("key_financial_risks", []),
        assumptions=parsed.get("assumptions", []),
        affordability_score=int(parsed.get("affordability_score", 5)),
        confidence_score=int(parsed.get("confidence_score", 5)),
    )

    logger.debug(
        f"   verdict={report.financial_feasibility_verdict} | "
        f"net_profit=Rp {report.estimated_monthly_net_profit_rp:,} | "
        f"BEP={report.estimated_break_even_months} bulan | "
        f"confidence={report.confidence_score}"
    )
    logger.debug(f"✓ Financial Analyst selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "financial_report": report,
        "messages": new_msgs,
    }
