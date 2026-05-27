"""
Market Scout Agent (Refactored) — Structured Evaluator

Perubahan dari versi lama:
- Output ke MarketScoutReport typed dataclass, bukan text blob
- TAM-SAM-SOM tetap ada tapi dengan metodologi eksplisit
- demand_score, competition_level, opportunity_score sebagai angka terukur
- Competitor insights structured (bukan narasi panjang)
- customer_pain_points sebagai List[str], bukan paragraf
- Max search calls dikurangi (efisiensi latency)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import (
    CompetitorInsight,
    EBPState,
    MarketScoutReport,
)
from tools.internet_search import internet_search

logger = logging.getLogger("clario.market_scout")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Market Scout Agent in a multi-agent business planning system.
Your job is to evaluate market viability and return STRUCTURED data — not essays.

CRITICAL: Output ONLY valid JSON. No prose, no markdown fences, no preamble.

Your analysis covers:
1. Demand & competition scoring (integer 1-10, not words)
2. Top 3-5 customer pain points (from real competitor reviews)
3. Max 3 validated opportunities (specific, not generic)
4. TAM-SAM-SOM with explicit methodology (population × spending estimate)
5. Max 3 competitor insights with concrete weaknesses

SCORING GUIDE:
- demand_score 1-3: Niche/declining market
- demand_score 4-6: Moderate, stable demand  
- demand_score 7-10: High/growing demand
- competition_level: "low" (<3 established players), "medium" (3-10), "high" (>10 or 1-2 dominant)
- opportunity_score: composite of demand_score - competition penalty + pain point severity

OUTPUT FORMAT — ONLY valid JSON:
{
  "demand_score": 7,
  "competition_level": "medium",
  "opportunity_score": 6,
  "market_trend_summary": "Maksimal 2 kalimat ringkas tren pasar",
  "customer_pain_points": [
    "Pain point spesifik 1 berdasarkan keluhan pelanggan nyata",
    "Pain point spesifik 2",
    "Pain point spesifik 3"
  ],
  "competitors": [
    {
      "name": "Nama Kompetitor A",
      "strengths": ["Kekuatan 1", "Kekuatan 2"],
      "weaknesses": ["Kelemahan nyata 1 dari ulasan pelanggan", "Kelemahan 2"],
      "market_position": "Pemimpin pasar lokal / challenger / niche"
    }
  ],
  "tam_estimate": "Rp X Triliun/tahun",
  "sam_estimate": "Rp X Miliar/tahun",
  "som_estimate": "Rp X Miliar/tahun (target 12 bulan pertama)",
  "market_sizing_methodology": "Populasi target: X juta orang × rata-rata spending Rp Y/bulan = TAM",
  "validated_opportunities": [
    "Peluang 1: spesifik dengan bukti dari pain point atau gap kompetitor",
    "Peluang 2: spesifik",
    "Peluang 3: spesifik"
  ],
  "source_links": ["https://...", "https://..."],
  "confidence_score": 7
}"""

_SEARCH_TOPICS = [
    "tren pasar dan permintaan konsumen sektor target indonesia terbaru",
    "kompetitor utama dan keluhan pelanggan ulasan negatif sektor target indonesia",
    "data populasi dan spending konsumen target demografi indonesia",
]


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

def market_scout_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node — Market Scout Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Market Scout Agent dimulai")

    bc = state.get("business_constraints")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]
    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK (iteration sebelumnya) ===", feedback]
    if user_fb:
        context_lines += ["\n=== FEEDBACK ENTREPRENEUR ===", user_fb]
    context_lines.append(
        "\nEvaluasi kelayakan pasar. Return ONLY structured JSON sesuai format."
    )

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
        max_followup_rounds=1,      # Dikurangi dari 2 → latency lebih cepat
        agent_name="market_scout",
        max_search_calls=4,         # Dikurangi dari 6
    )

    parsed = extract_json(final_response.content)

    # ── Parse competitors ──
    raw_competitors = parsed.get("competitors", [])
    competitors = []
    for c in raw_competitors:
        if isinstance(c, dict):
            competitors.append(CompetitorInsight(
                name=c.get("name", "Unknown"),
                strengths=c.get("strengths", []),
                weaknesses=c.get("weaknesses", []),
                market_position=c.get("market_position"),
            ))

    report = MarketScoutReport(
        demand_score=int(parsed.get("demand_score", 5)),
        competition_level=parsed.get("competition_level", "medium"),
        opportunity_score=int(parsed.get("opportunity_score", 5)),
        market_trend_summary=parsed.get("market_trend_summary", ""),
        customer_pain_points=parsed.get("customer_pain_points", []),
        competitors=competitors,
        tam_estimate=parsed.get("tam_estimate", "-"),
        sam_estimate=parsed.get("sam_estimate", "-"),
        som_estimate=parsed.get("som_estimate", "-"),
        market_sizing_methodology=parsed.get("market_sizing_methodology", ""),
        validated_opportunities=parsed.get("validated_opportunities", []),
        source_links=parsed.get("source_links", []),
        confidence_score=int(parsed.get("confidence_score", 5)),
    )

    logger.debug(
        f"   demand_score={report.demand_score} | "
        f"competition={report.competition_level} | "
        f"opportunity={report.opportunity_score} | "
        f"confidence={report.confidence_score}"
    )
    logger.debug(f"✓ Market Scout selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "market_report": report,
        "messages": new_msgs,
    }
