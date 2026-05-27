"""Financial Analyst Agent (Enterprise Version) — Menghitung proyeksi kas dinamis,
analisis titik impas (BEP), runway, simulasi sensitivitas worst-case, dan evaluasi benchmark industri.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, FinancialAnalysisReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.financial_analyst_enterprise")

_ENTERPRISE_FINANCIAL_PROMPT = """You are the Senior Financial Analyst Agent in an enterprise business planning system.
Your mission is to act as a dynamic financial calculator and decision system that processes quantitative constraints into actionable viability reports.

You must parse the raw financial inputs (capital/modal, price/harga produk, traffic estimations, COGS, rent, and salary) and explicitly calculate the core operational metrics below in Bahasa Indonesia, using Indonesian Rupiah (Rp).

CRITICAL FINANCIAL TASKS:
1. Dynamic Financial Calculator: Compute the monthly cashflow projection, Break-Even Point (BEP in units and Rp value), and the capital cash runway in months.
2. Explicit Sensitivity Analysis: Stress-test the business model under worst-case parameters and display the re-computed net profit results for:
   - Base Case (Kondisi normal sesuai asumsi awal)
   - Skenario A: Volume penjualan / sales turun 20%
   - Skenario B: Biaya bahan baku / COGS naik 15%
   - Skenario C: Biaya pemasaran / ads cost melonjak naik
3. Industry Benchmark Evaluation: Cross-reference the calculated metrics against stored standard industry thresholds:
   - F&B: Standard Gross Margin 60-70%
   - E-commerce / Digital Service: Standard Conversion Rate 1-3%, benchmark Customer Acquisition Cost (CAC)
   - Retail / Traditional Trading: Standard Gross Margin 40-50%, Startup burn rate guidelines
4. Feasibility Verdict: Issue a clear financial judgment (LAYAK / LAYAK DENGAN CATATAN / TIDAK LAYAK).

DO NOT output multi-year Monte Carlo simulations, complex cohort models, or raw uncalculated numbers.

OUTPUT FORMAT — respond with ONLY valid JSON when ready, no other text or code fences:
{
  "monthly_cashflow_projection": {
    "estimated_revenue": "Proyeksi pendapatan bulanan normal dalam Rp",
    "total_cogs": "Total pengeluaran COGS bulanan normal dalam Rp",
    "fixed_operating_expenses": "Total biaya operasional tetap (gaji, sewa) dalam Rp",
    "net_operating_profit": "Keuntungan bersih bulanan normal dalam Rp"
  },
  "core_decision_metrics": {
    "bep_monthly_volume": "Target volume penjualan bulanan untuk mencapai BEP dalam unit",
    "bep_monthly_value": "Nilai target omzet bulanan untuk mencapai BEP dalam Rp",
    "cash_runway_months": "Durasi kekuatan modal bertahan dalam hitungan bulan jika operasional awal merugi"
  },
  "sensitivity_stress_test": {
    "base_case_net_profit": "Keuntungan bersih kondisi normal dalam Rp",
    "sales_drop_20_net_profit": "Keuntungan bersih jika volume penjualan turun 20% dalam Rp",
    "cogs_rise_15_net_profit": "Keuntungan bersih jika harga bahan baku COGS naik 15% dalam Rp",
    "ads_cost_surge_impact": "Dampak dan sisa keuntungan jika biaya pemasaran beriklan naik signifikan"
  },
  "industry_benchmark_alignment": "Analisis perbandingan margin keuntungan dan efisiensi biaya proyeksi terhadap rata-rata industri riil",
  "financial_feasibility_verdict": "LAYAK / LAYAK DENGAN CATATAN / TIDAK LAYAK",
  "markdown_financial_report": "Laporan analisis keuangan lengkap terstruktur menggunakan format Markdown Bahasa Indonesia untuk dibaca pengguna"
}"""

_ENTERPRISE_SEARCH_TOPICS = [
    "benchmark biaya operasional standar komponen gaji dan sewa ruko komersial di indonesia",
    "rata rata gross profit margin industri target serta biaya customer acquisition cost indonesia"
]


def financial_analyst_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Enterprise Financial Analyst Agent."""
    t_start = time.perf_counter()
    logger.debug("============================================================")
    logger.debug("-> Enterprise Financial Analyst Agent Dimulai")
    bc = state.get("bussiness_constraints")
    sr = state.get("strategic_report")
    msr = state.get("market_scout_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== QUANTITATIVE INPUT CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== OPPORTUNITY DATA CONTEXT ===",
            f"Peluang Terpilih: {', '.join(msr.ideas[:2])}",
            f"Analisis Pasar: {msr.agent_explanation[:600]}"
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR FEEDBACK ===", user_fb]

    context_lines.append(
        "\nProses kalkulasi matematika finansial di atas. Keluarkan analisis kelayakan dalam format JSON."
    )

    # Temperature 0.2 untuk meminimalisir kesalahan perhitungan matematika pada model
    llm = get_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_ENTERPRISE_FINANCIAL_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        planning_topics=_ENTERPRISE_SEARCH_TOPICS,
        max_followup_rounds=1,
        agent_name="enterprise_financial_analyst",
        max_search_calls=3,
    )

    parsed = extract_json(final_response.content)
    
    monthly_projections = parsed.get("monthly_cashflow_projection", {})
    metrics = parsed.get("core_decision_metrics", {})
    sensitivity = parsed.get("sensitivity_stress_test", {})
    benchmark = parsed.get("industry_benchmark_alignment", "")
    verdict = parsed.get("financial_feasibility_verdict", "LAYAK DENGAN CATATAN")
    main_markdown = parsed.get("markdown_financial_report", final_response.content)

    # Menyusun output terstruktur ke dalam parameter tunggal state tanpa merusak arsitektur data lama
    formatted_analysis = (
        f"{main_markdown}\n\n"
        f"### KESIMPULAN METRIK KEPUTUSAN STRATEGIS ENTERPRISE\n"
        f"- Putusan Kelayakan Finansial: **{verdict}**\n"
        f"- Target BEP Bulanan (Volume): {metrics.get('bep_monthly_volume', '-')} unit\n"
        f"- Target BEP Bulanan (Nilai): {metrics.get('bep_monthly_value', '-')}\n"
        f"- Estimasi Cash Runway: {metrics.get('cash_runway_months', '-')} bulan\n\n"
        f"### RE-KOMPUTASI ANALISIS SENSITIVITAS OPERASIONAL\n"
        f"- Pendapatan Bersih (Kondisi Normal): {sensitivity.get('base_case_net_profit', '-')}\n"
        f"- Pendapatan Bersih (Volume Penjualan Turun 20%): {sensitivity.get('sales_drop_20_net_profit', '-')}\n"
        f"- Pendapatan Bersih (Harga Bahan Baku Naik 15%): {sensitivity.get('cogs_rise_15_net_profit', '-')}\n"
        f"- Evaluasi Risiko Kenaikan Ads Cost: {sensitivity.get('ads_cost_surge_impact', '-')}\n\n"
        f"### EVALUASI KESELARASAN TOLOK UKUR INDUSTRI\n{benchmark}"
    )

    report = FinancialAnalysisReport(analysis_result=formatted_analysis)

    logger.debug(f"Enterprise Analysis Completed - Verdict: {verdict} | BEP: {metrics.get('bep_monthly_value', '-')}")
    logger.debug(f"Financial Analyst Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("============================================================")
    return {
        "financial_analysis_report": report,
        "messages": new_msgs,
    }