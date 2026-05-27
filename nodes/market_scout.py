"""Market Scout Agent (Enterprise Model) — Memvalidasi kelayakan pasar menggunakan
multi-source retrieval, analisis sentimen komplain kompetitor, dan pemodelan TAM-SAM-SOM berbasis data riil.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, MarketScoutReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.market_scout_enterprise")

_ENTERPRISE_SYSTEM_PROMPT = """You are the Senior Market Scout Agent in an enterprise business planning system.
Your mission is to rigorously validate whether a target market is viable to enter by transforming bulk search data into highly actionable market intelligence.

Your analysis must be synthesized entirely in Bahasa Indonesia and formatted into a structured JSON response.

CRITICAL REQUIREMENTS:
1. Multi-Source Integration: Synthesize ground truths from Google Search, Google Trends signals, industry news, and competitor reviews.
2. Competitor Sentiment Extraction: Drill down into public reviews (Google Maps, marketplace, public forums) to extract the most common customer complaints (e.g., slow delivery, poor packaging, high prices) to identify clear, unmet market opportunities.
3. Grounded TAM-SAM-SOM Sederhana: Calculate market sizes using a logical methodology based on retrieved factual data (Target Population Estimate x Estimated Average Annual Spending). Do NOT hallucinate baseline population numbers.

OUTPUT FORMAT — respond with ONLY valid JSON when ready, no other text or code fences:
{
  "ideas": [
    "Peluang bisnis spesifik 1 dengan bukti validasi pasar dan solusi terhadap kelemahan kompetitor",
    "Peluang bisnis spesifik 2 dengan bukti validasi pasar dan solusi terhadap kelemahan kompetitor"
  ],
  "market_trend_summary": "Rangkuman tren pasar makro, momentum pertumbuhan, dan sinyal Google Trends",
  "competitor_snapshot": "Pemetaan lanskap kompetitor utama dan posisi operasional mereka di Indonesia",
  "common_customer_complaints": [
    "Keluhan pelanggan utama 1 terhadap kompetitor yang ada di pasar",
    "Keluhan pelanggan utama 2 terhadap kompetitor yang ada di pasar",
    "Keluhan pelanggan utama 3 terhadap kompetitor yang ada di pasar"
  ],
  "tam_sam_som_analysis": {
    "methodology": "Penjelasan sumber data populasi target dan estimasi rata-rata spending belanja per tahun",
    "tam": "Total Addressable Market dalam mata uang Rupiah (Rp)",
    "sam": "Serviceable Addressable Market dalam mata uang Rupiah (Rp)",
    "som": "Serviceable Obtainable Market yang rasional dijangkau dalam waktu dekat dalam Rupiah (Rp)"
  },
  "agent_explanation": "Analisis naratif komprehensif dalam Bahasa Indonesia yang mengintegrasikan seluruh temuan di atas untuk membuktikan kelayakan komersial pasar.",
  "sources": [
    "Cantumkan URL spesifik dari berita, Google Trends, atau basis data populasi yang Anda temukan saat melakukan search",
    "Contoh: https://nasional.kontan.co.id/news/...",
    "Contoh: Google Trends Keyword X (Indonesia, Past 12 Months)"
  ]
}"""

_ENTERPRISE_SEARCH_TOPICS = [
    "tren industri perkembangan pasar dan sinyal google trends terbaru sektor target di indonesia",
    "kompetitor utama bisnis serupa di indonesia beserta ulasan negatif review pelanggan",
    "keluhan konsumen customer complaints paling umum di marketplace dan google maps industri target",
    "data demografi populasi target serta rata rata pengeluaran spending bulanan di indonesia untuk sektor ini"
]


def market_scout_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Enterprise Market Scout Agent."""
    t_start = time.perf_counter()
    logger.debug("============================================================")
    logger.debug("-> Enterprise Market Scout Agent Dimulai")
    bc = state.get("bussiness_constraints")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== ENTERPRISE MISSION ===",
        "Validasi kelayakan masuk pasar, petakan kegagalan layanan kompetitor, dan kalkulasi TAM-SAM-SOM.",
        "\n=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]
    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR FEEDBACK ===", user_fb]

    context_lines.append(
        "\nJalankan riset multi-source terencana. Setelah data terkumpul, formulasikan laporan dalam JSON Bahasa Indonesia."
    )
    user_message = "\n".join(context_lines)

    # Temperature 0.4 untuk keseimbangan antara analisis taktis dan akurasi ekstraksi data angka
    llm = get_llm(temperature=0.4)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_ENTERPRISE_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        tools=[internet_search],
        planning_topics=_ENTERPRISE_SEARCH_TOPICS,
        max_followup_rounds=2,
        agent_name="enterprise_market_scout",
        max_search_calls=6,
    )

    # 1. Ekstrak JSON hasil respon akhir
    parsed = extract_json(final_response.content)
    raw_ideas = parsed.get("ideas", [])
    trend = parsed.get("market_trend_summary", "")
    competitors = parsed.get("competitor_snapshot", "")
    complaints = parsed.get("common_customer_complaints", [])
    sizing = parsed.get("tam_sam_som_analysis", {})
    explanation = parsed.get("agent_explanation", final_response.content)
    
    # 2. Ambil Sources yang ditulis oleh LLM
    extracted_sources = parsed.get("sources", [])

    # 3. BACKUP PLAN: Jika LLM tidak menuliskan source di JSON, 
    # kita scan secara manual dari hasil pencarian asli di `new_msgs`
    if not extracted_sources:
        for msg in new_msgs:
            # Cari pesan dari ToolMessage yang berisi raw data dari internet_search
            if msg.type == "tool" and hasattr(msg, "content"):
                # Sederhananya, cari string berupa tautan/URL di dalam konten pencarian
                import re
                urls = re.findall(r'(https?://[^\s\"\'\>]+)', msg.content)
                for url in urls:
                    # Bersihkan URL dan batasi agar unik (misal ambil domain/link utama saja)
                    clean_url = url.split(')')[0].split(']')[0] # bersihkan markdown artifact
                    if clean_url not in extracted_sources and len(extracted_sources) < 5:
                        extracted_sources.append(clean_url)

    # Format penjelasan teks
    formatted_explanation = (
        f"{explanation}\n\n"
        f"### 1. RANGKUMAN TREN PASAR & GOOGLE TRENDS\n{trend}\n\n"
        f"### 2. SNAPSHOT KOMPETITOR & SENTIMEN NEGATIF\n"
        f"- Lanskap Kompetisi: {competitors}\n"
        f"- Komplain Utama Konsumen: {', '.join(complaints)}\n\n"
        f"### 3. ESTIMASI UKURAN PASAR (TAM-SAM-SOM)\n"
        f"- Metodologi Sizing: {sizing.get('methodology', '-')}\n"
        f"- TAM: {sizing.get('tam', '-')}\n"
        f"- SAM: {sizing.get('sam', '-')}\n"
        f"- SOM: {sizing.get('som', '-')}"
    )

    if isinstance(raw_ideas, list):
        ideas = [str(i) for i in raw_ideas if i]
    else:
        ideas = [str(raw_ideas)]

    if not ideas:
        ideas = ["Gagal mengidentifikasi peluang spesifik yang tervalidasi dari hasil pencarian."]

    # 4. Masukkan array sources ke dalam objek laporan akhir
    report = MarketScoutReport(
        ideas=ideas, 
        agent_explanation=formatted_explanation,
        sources=extracted_sources  # <--- Sumber tersimpan di sini!
    )

    logger.debug(f"Market Scout Agent selesai dengan {len(extracted_sources)} sitasi valid.")
    logger.debug("============================================================")
    return {
        "market_scout_report": report,
        "messages": new_msgs,
    }