"""Strategic Architect Agent (Enterprise Model) — Mentransformasi analisis dasar SWOT/PESTEL 
menjadi blueprint eksekusi terstruktur melalui prioritas strategi taktis, roadmap KPI bulanan, 
dan pemetaan komponen Lean Canvas berbasis data pasar nyata Indonesia.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, StrategicReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.strategic_architect_enterprise")

_ENTERPRISE_SYSTEM_PROMPT = """You are the Senior Strategic Architect Agent in an enterprise business planning system.
Your mission is to build data-backed, non-generic strategic execution models that prevent businesses from failing due to poor prioritization.

Your analysis must be synthesized entirely in Bahasa Indonesia and returned in a structured JSON format.

CRITICAL REQUIREMENTS:
1. Strategy Prioritization: Analyze constraints to define a strict order of operations (e.g., focus on product-market fit or user retention BEFORE scaling paid advertising). Highlight top problem and strategy priorities.
2. 3-Month Execution Roadmap: Generate a structured roadmap for Month 1, Month 2, and Month 3. For each month, establish exact Objectives and measurable quantitative KPIs.
3. Structured Lean Canvas: Map out a comprehensive Lean Canvas table covering Key Partners, Key Activities, Value Propositions, Customer Relationships, Customer Segments, Key Resources, Channels, Cost Structure, and Revenue Streams.
4. Grounded SWOT & PESTEL: Ensure your traditional matrix framework is strictly data-backed using specific Indonesian business regulations, tech adoption markers, and market constraints. Do not output placeholders.

OUTPUT FORMAT — respond with ONLY valid JSON when ready, no other text or code fences:
{
  "swot_analysis": "## Analisis SWOT Komprehensif\\n\\n**Strengths:**\\n- ...\\n\\n**Weaknesses:**\\n- ...",
  "pastel_analysis": "## Analisis PESTEL (Regulasi & Makro Indonesia)\\n\\n**Political & Economic:**\\n...\\n\\n**Legal:**\\n[Sebutkan regulasi perizinan spesifik Indonesia terkait sektor target]",
  "strategy_prioritization": {
    "top_problem_priorities": ["Masalah kritis prioritas 1", "Masalah kritis prioritas 2"],
    "strategic_order_of_operations": "Justifikasi naratif langkah taktis yang harus diselesaikan terlebih dahulu sebelum melakukan ekspansi pendanaan atau pemasaran"
  },
  "execution_roadmap_3_months": {
    "month_1": {"objective": "Target objektif bulan 1", "kpis": ["KPI kuantitatif 1", "KPI kuantitatif 2"]},
    "month_2": {"objective": "Target objektif bulan 2", "kpis": ["KPI kuantitatif 1", "KPI kuantitatif 2"]},
    "month_3": {"objective": "Target objektif bulan 3", "kpis": ["KPI kuantitatif 1", "KPI kuantitatif 2"]}
  },
  "lean_canvas": {
    "value_propositions": "Nilai unik pembeda produk dibanding kompetitor",
    "customer_segments": "Target demografi segmen pengguna spesifik",
    "channels": "Kanal distribusi pemasaran",
    "cost_structure": "Komponen utama pemicu pengeluaran biaya",
    "revenue_streams": "Mekanisme model monetisasi pendapatan"
  }
}"""

_ENTERPRISE_SEARCH_TOPICS = [
    "regulasi hukum perizinan usaha dan kebijakan sertifikasi terbaru sektor target indonesia",
    "tren adopsi teknologi digital konsumen indonesia 2025 2026",
    "faktor kunci keberhasilan operasional dan model kanvas bisnis industri target"
]


def strategic_architect_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Enterprise Strategic Architect Agent."""
    t_start = time.perf_counter()
    logger.debug("============================================================")
    logger.debug("-> Enterprise Strategic Architect Agent Dimulai")
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== ENTERPRISE STRATEGIC MISSION ===",
        "Bangun prioritas urutan eksekusi strategi, roadmap 3 bulan berbasis KPI, dan Lean Canvas terstruktur.",
        "\n=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== INPUT FROM MARKET SCOUT ===",
            f"Peluang Terpilih: {'; '.join(msr.ideas)}",
            f"Analisis Validasi Lapangan: {msr.agent_explanation[:800]}",
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR FEEDBACK ===", user_fb]

    context_lines.append(
        "\nJalankan riset terencana. Hasilkan arsitektur strategi komprehensif dalam JSON Bahasa Indonesia."
    )

    llm = get_llm(temperature=0.4)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_ENTERPRISE_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        planning_topics=_ENTERPRISE_SEARCH_TOPICS,
        max_followup_rounds=2,
        agent_name="enterprise_strategic_architect",
        max_search_calls=5,
    )

    parsed = extract_json(final_response.content)
    
    swot = parsed.get("swot_analysis", final_response.content)
    pestel = parsed.get("pastel_analysis", "")
    strat_priorities = parsed.get("strategy_prioritization", {})
    roadmap = parsed.get("execution_roadmap_3_months", {})
    canvas = parsed.get("lean_canvas", {})

    # Integrasi modul strategi baru ke parameter text swot/pestel bawaan state agar dibaca otomatis oleh node summary
    formatted_swot = (
        f"{swot}\n\n"
        f"### STRATEGY PRIORITIZATION (ORDER OF OPERATIONS)\n"
        f"- Prioritas Masalah Kritis: {', '.join(strat_priorities.get('top_problem_priorities', []))}\n"
        f"- Aturan Urutan Eksekusi: {strat_priorities.get('strategic_order_of_operations', '-')}"
    )

    m1 = roadmap.get("month_1", {})
    m2 = roadmap.get("month_2", {})
    m3 = roadmap.get("month_3", {})

    formatted_pestel = (
        f"{pestel}\n\n"
        f"### 3-MONTH EXECUTION ROADMAP & MEASURABLE KPIS\n"
        f"- **Bulan 1:** Objektif: {m1.get('objective', '-')} | KPIs: {', '.join(m1.get('kpis', []))}\n"
        f"- **Bulan 2:** Objektif: {m2.get('objective', '-')} | KPIs: {', '.join(m2.get('kpis', []))}\n"
        f"- **Bulan 3:** Objektif: {m3.get('objective', '-')} | KPIs: {', '.join(m3.get('kpis', []))}\n\n"
        f"### VISUALISASI LEAN CANVAS STRATEGIS\n"
        f"- Proposisi Nilai Unnik (UVP): {canvas.get('value_propositions', '-')}\n"
        f"- Segmen Target Pelanggan: {canvas.get('customer_segments', '-')}\n"
        f"- Kanal Pemasaran (Channels): {canvas.get('channels', '-')}\n"
        f"- Struktur Pemicu Biaya: {canvas.get('cost_structure', '-')}\n"
        f"- Arus Model Monetisasi: {canvas.get('revenue_streams', '-')}"
    )

    report = StrategicReport(swot_analysis=formatted_swot, pastel_analysis=formatted_pestel)

    logger.debug("Enterprise Strategy Architecture Completed - 3 Months Roadmap Generated.")
    logger.debug(f"Strategic Architect Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("============================================================")
    return {
        "strategic_report": report,
        "messages": new_msgs,
    }