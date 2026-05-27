"""Ethics & Compliance Agent (Enterprise Version) — Mengevaluasi kepatuhan hukum komersial Indonesia, 
menentukan instrumen perizinan berbasis aturan (rules-based licensing), menyusun checklist regulasi, 
serta mendeteksi risiko pelanggaran hukum operasional.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from functions.agent_utils import extract_json, format_constraints, run_planned_search_loop
from functions.llm import get_llm
from states.schema import EBPState, EthicsAnalysisReport
from tools.internet_search import internet_search

logger = logging.getLogger("clario.ethics_agent_enterprise")

_ENTERPRISE_ETHICS_PROMPT = """You are the Senior Ethics & Compliance Agent in an enterprise business planning system.
Your mission is to act as a rules-based legal validation engine that identifies mandatory licensing, calculates regulatory risks, and builds operational compliance checklists under Indonesian law.

Your analysis must be compiled entirely in Bahasa Indonesia and returned in a clear, structured JSON format.

CRITICAL REQUIREMENTS:
1. Rules-Based Licensing System: Apply strict logical routing based on business sectors (e.g., IF F&B/Kuliner AND packaged, THEN require NIB via OSS, Sertifikasi Halal BPJPH, PIRT/BPOM, and NPWP Badan/Perorangan. IF Digital/SaaS, THEN require PSE Kominfo and UU PDP Compliance).
2. Comprehensive Compliance Checklist: Generate a structured checklist covering 3 crucial pillars: Izin Operasional, Kewajiban Pajak (NPWP/PPh), and Legalitas Hukum Usaha.
3. Actionable Risk Alerts: Identify high-impact legal traps (e.g., selling non-certified food products to Muslim demographics under mandatory Halal laws, or processing customer data without clear privacy policies).
4. Estimated Bureaucracy Effort: Provide a realistic evaluation of the time, cost, and complexity required to process these permits in Indonesia.

OUTPUT FORMAT — respond with ONLY valid JSON when ready, no other text or code fences:
{
  "licensing_requirements": {
    "mandatory_permits": ["Daftar izin wajib 1 (misal: NIB)", "Daftar izin wajib 2"],
    "sector_specific_permits": ["Izin spesifik sektor (misal: PIRT/Sertifikasi Halal/PSE Kominfo)"]
  },
  "compliance_checklist": {
    "perizinan_dasar": ["Langkah checklist izin 1", "Langkah checklist izin 2"],
    "perpajakan_dan_pajak": ["Langkah checklist pajak 1", "Langkah checklist pajak 2"],
    "legalitas_dan_kontrak": ["Langkah checklist legalitas kontrak karyawan/mitra"]
  },
  "critical_risk_alerts": [
    "Peringatan risiko hukum kritis 1 disertai dampak penalti",
    "Peringatan risiko hukum kritis 2 disertai dampak penalti"
  ],
  "bureaucracy_effort_estimation": {
    "estimated_time_frames": "Estimasi durasi waktu total pengurusan perizinan",
    "complexity_level": "RENDAH / SEDANG / TINGGI disertai alasan singkat"
  },
  "compliance_score": "Skala nilai angka 1-10 beserta parameter justifikasi kepatuhan awal",
  "markdown_compliance_report": "Laporan analisis kepatuhan hukum komprehensif terstruktur menggunakan format Markdown Bahasa Indonesia"
}"""

_ENTERPRISE_SEARCH_TOPICS = [
    "persyaratan perizinan usaha oss nib sertifikasi halal bpom pse kominfo indonesia terbaru",
    "sanksi hukum pelanggaran regulasi sektor bisnis target indonesia perlindungan konsumen uu pdp"
]


def ethics_agent_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node for the Enterprise Ethics & Compliance Agent."""
    t_start = time.perf_counter()
    logger.debug("============================================================")
    logger.debug("-> Enterprise Ethics & Compliance Agent Dimulai")
    bc = state.get("bussiness_constraints")
    msr = state.get("market_scout_report")
    sr = state.get("strategic_report")
    far = state.get("financial_analysis_report")
    feedback = state.get("orchestrator_feedback")
    user_fb = state.get("user_feedback")

    context_lines = [
        "=== ENTERPRISE LEGAL & ETHICS MISSION ===",
        "Tentukan perizinan kondisional berbasis aturan, susun checklist kepatuhan, dan petakan risiko regulasi Indonesia.",
        "\n=== BUSINESS CONSTRAINTS ===",
        format_constraints(bc),
    ]

    if msr:
        context_lines += [
            "\n=== PROPOSED CORE OPPORTUNITY ===",
            f"Ide Bisnis Terpilih: {'; '.join(msr.ideas[:2])}"
        ]

    if sr:
        context_lines += [
            "\n=== STRATEGIC CONTEXT (PESTEL Legal Segment) ===",
            sr.pastel_analysis[:500]
        ]

    if far:
        context_lines += [
            "\n=== QUANTITATIVE FINANCIAL STRUCTURE ===",
            far.analysis_result[:400]
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== ENTREPRENEUR FEEDBACK ===", user_fb]

    context_lines.append(
        "\nValidasi seluruh batasan operasional di atas terhadap hukum positif Indonesia. Keluarkan hasil dalam bentuk JSON."
    )

    llm = get_llm(temperature=0.2)
    llm_with_tools = llm.bind_tools([internet_search])

    new_msgs, final_response = run_planned_search_loop(
        llm=llm,
        llm_with_tools=llm_with_tools,
        messages=[
            SystemMessage(content=_ENTERPRISE_ETHICS_PROMPT),
            HumanMessage(content="\n".join(context_lines)),
        ],
        tools=[internet_search],
        planning_topics=_ENTERPRISE_SEARCH_TOPICS,
        max_followup_rounds=2,
        agent_name="enterprise_ethics_guardian",
        max_search_calls=4,
    )

    parsed = extract_json(final_response.content)
    
    licensing = parsed.get("licensing_requirements", {})
    checklist = parsed.get("compliance_checklist", {})
    risks = parsed.get("critical_risk_alerts", [])
    bureaucracy = parsed.get("bureaucracy_effort_estimation", {})
    score = parsed.get("compliance_score", "7/10")
    main_markdown = parsed.get("markdown_compliance_report", final_response.content)

    # Menggabungkan data terstruktur baru ke parameter tunggal teks analitik state model lama
    formatted_analysis = (
        f"{main_markdown}\n\n"
        f"### KESIMPULAN SISTEM VALIDASI REGULASI ENTERPRISE\n"
        f"- Skor Kepatuhan & Etika: **{score}**\n"
        f"- Estimasi Kompleksitas Birokrasi: {bureaucracy.get('complexity_level', '-')}\n"
        f"- Estimasi Durasi Pengurusan: {bureaucracy.get('estimated_time_frames', '-')}\n\n"
        f"### DAFTAR PERIZINAN WAJIB (RULES-BASED MODEL)\n"
        f"- Izin Dasar: {', '.join(licensing.get('mandatory_permits', []))}\n"
        f"- Izin Spesifik Sektor: {', '.join(licensing.get('sector_specific_permits', []))}\n\n"
        f"### SIGNAL PERINGATAN RISIKO HUKUM OPERASIONAL\n- " + "\n- ".join(risks)
    )

    report = EthicsAnalysisReport(analysis_result=formatted_analysis)

    logger.debug(f"Enterprise Compliance Scoring Completed - Score: {score}")
    logger.debug(f"Ethics Guardian Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("============================================================")
    return {
        "ethics_analysis_report": report,
        "messages": new_msgs,
    }