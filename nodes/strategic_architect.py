"""
Strategic Architect Agent (Refactored) — Structured Evaluator

Perubahan dari versi lama:
- SWOT → List[str] max 3 poin per section, bukan essay
- PESTEL → pestel_highlights: List[str] max 6 poin, bukan markdown blob
- Roadmap → MonthlyMilestone typed, bukan narasi
- Lean Canvas → individual fields, bukan satu string panjang
- Output ke StrategicReport typed dataclass
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
    MonthlyMilestone,
    StrategicReport,
)
from tools.internet_search import internet_search

logger = logging.getLogger("clario.strategic_architect")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Strategic Architect Agent in a multi-agent business planning system.
Your job is to build a data-backed strategic execution model and return STRUCTURED data — not essays.

CRITICAL: Output ONLY valid JSON. No prose, no markdown fences, no preamble.
CRITICAL LANGUAGE RULE: All values, bullet points, highlights, and descriptions inside the JSON MUST be written in professional Bahasa Indonesia. Penggunaan bahasa Inggris hanya untuk sebutan seperti SWOT dan PESTEL.

RULES:
- SWOT: MAXIMUM 3 bullet points per section. Be specific, not generic.
- PESTEL: MAXIMUM 6 highlights total, only the most relevant for this specific business in Indonesia.
- Roadmap: Each month must have 1 clear objective + 2-3 QUANTITATIVE KPIs (with numbers).
- Lean Canvas: Each field = 1-2 sentences max.
- top_problem_priorities: max 3 items.
- strategic_order_of_operations: max 3 sentences — what must happen BEFORE scaling.

BAD example (generic, useless):
  strengths: ["Produk berkualitas tinggi", "Tim yang berpengalaman"]

GOOD example (specific, grounded):
  strengths: ["Harga 30% lebih murah dari Gojek Food karena no middleware fee", "Lokasi di area residensial dengan 0 kompetitor sejenis dalam radius 1km"]

OUTPUT FORMAT — ONLY valid JSON:
{
  "strengths": ["Kekuatan spesifik 1", "Kekuatan spesifik 2", "Kekuatan spesifik 3"],
  "weaknesses": ["Kelemahan 1", "Kelemahan 2", "Kelemahan 3"],
  "opportunities": ["Peluang pasar 1 berbasis data", "Peluang 2", "Peluang 3"],
  "threats": ["Ancaman 1 konkret", "Ancaman 2", "Ancaman 3"],
  "pestel_highlights": [
    "Political: [Analisis faktor kebijakan/politik Indonesia yang berdampak]",
    "Economic: [Analisis daya beli, inflasi, atau tren ekonomi makro lokal]",
    "Social: [Analisis faktor sosial yang berdampak]",
    "Technology: [Analisis perkembangan teknologi yang relevan]",
    "Environmental: [Analisis faktor lingkungan yang berdampak]",
    "Legal: [Regulasi spesifik Indonesia yang relevan]"
  ],
  "top_problem_priorities": [
    "Masalah kritis 1 yang harus diselesaikan pertama",
    "Masalah kritis 2",
    "Masalah kritis 3"
  ],
  "strategic_order_of_operations": "Kalimat singkat: apa yang harus dilakukan SEBELUM scaling/marketing",
  "roadmap": {
    "month_1": {
      "objective": "Objektif bulan 1 yang terukur",
      "kpis": ["KPI kuantitatif 1 (e.g. 50 transaksi/minggu)", "KPI kuantitatif 2"]
    },
    "month_2": {
      "objective": "Objektif bulan 2",
      "kpis": ["KPI 1", "KPI 2"]
    },
    "month_3": {
      "objective": "Objektif bulan 3",
      "kpis": ["KPI 1", "KPI 2"]
    }
  },
  "value_proposition": "1-2 kalimat UVP yang membedakan dari kompetitor",
  "customer_segments": "Deskripsi spesifik segmen target (demografi + psikografi)",
  "channels": "Kanal distribusi utama",
  "cost_structure": "Komponen biaya terbesar",
  "revenue_streams": "Mekanisme monetisasi",
  "recommended_sales_channel": "Channel penjualan utama yang paling relevan",
  "confidence_score": 7
}"""

_SEARCH_TOPICS = [
    "regulasi perizinan dan kebijakan terbaru sektor target indonesia 2025",
    "tren adopsi digital dan faktor keberhasilan bisnis sektor target indonesia",
]


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

def strategic_architect_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node — Strategic Architect Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Strategic Architect Agent dimulai")

    bc = state.get("business_constraints")
    market = state.get("market_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if market:
        context_lines += [
            "\n=== MARKET INTELLIGENCE (dari Market Scout) ===",
            f"Demand Score: {market.demand_score}/10",
            f"Competition Level: {market.competition_level}",
            f"Opportunity Score: {market.opportunity_score}/10",
            f"Validated Opportunities: {'; '.join(market.validated_opportunities)}",
            f"Customer Pain Points: {'; '.join(market.customer_pain_points)}",
            f"Market Trend: {market.market_trend_summary}",
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== FEEDBACK ENTREPRENEUR ===", user_fb]

    context_lines.append("\nBangun arsitektur strategi. Return ONLY structured JSON sesuai format.")

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
        planning_topics=_SEARCH_TOPICS,
        max_followup_rounds=1,      # Dikurangi dari 2
        agent_name="strategic_architect",
        max_search_calls=3,         # Dikurangi dari 5
    )

    parsed = extract_json(final_response.content)

    # ── Parse roadmap ──
    raw_roadmap = parsed.get("roadmap", {})
    roadmap: dict[str, MonthlyMilestone] = {}
    for month_key in ("month_1", "month_2", "month_3"):
        month_data = raw_roadmap.get(month_key, {})
        roadmap[month_key] = MonthlyMilestone(
            objective=month_data.get("objective", ""),
            kpis=month_data.get("kpis", []),
        )

    report = StrategicReport(
        strengths=parsed.get("strengths", []),
        weaknesses=parsed.get("weaknesses", []),
        opportunities=parsed.get("opportunities", []),
        threats=parsed.get("threats", []),
        pestel_highlights=parsed.get("pestel_highlights", []),
        top_problem_priorities=parsed.get("top_problem_priorities", []),
        strategic_order_of_operations=parsed.get("strategic_order_of_operations", ""),
        roadmap=roadmap,
        value_proposition=parsed.get("value_proposition", ""),
        customer_segments=parsed.get("customer_segments", ""),
        channels=parsed.get("channels", ""),
        cost_structure=parsed.get("cost_structure", ""),
        revenue_streams=parsed.get("revenue_streams", ""),
        recommended_sales_channel=parsed.get("recommended_sales_channel", ""),
        confidence_score=int(parsed.get("confidence_score", 5)),
    )

    logger.debug(f"   confidence_score={report.confidence_score}")
    logger.debug(f"✓ Strategic Architect selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "strategy_report": report,
        "messages": new_msgs,
    }
