"""
Ethics & Compliance Agent (Refactored) — Structured Evaluator

Perubahan dari versi lama:
- Output ke EthicsAnalysisReport typed dataclass, bukan text blob
- mandatory_permits, sector_specific_permits sebagai List[str]
- Compliance checklist per-kategori sebagai List[str]
- legal_risk_level sebagai Literal enum
- bureaucracy_complexity sebagai Literal enum
- Tidak ada markdown blob di dalam output JSON
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
    EthicsAnalysisReport,
)
from tools.internet_search import internet_search

logger = logging.getLogger("clario.ethics_agent")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """You are the Ethics & Compliance Agent in a multi-agent business planning system.
Your job is to evaluate Indonesian regulatory compliance and return STRUCTURED data — not essays.

CRITICAL: Output ONLY valid JSON. No prose, no markdown fences, no preamble.

RULES-BASED LICENSING LOGIC (apply strictly):
- F&B / Kuliner: → NIB (OSS), PIRT (produksi rumahan) atau BPOM (skala pabrik), Sertifikasi Halal BPJPH (jika target Muslim), NPWP
- Digital / SaaS / E-commerce: → NIB, PSE Kominfo (jika layanan digital), UU PDP compliance (jika handle data pengguna), NPWP
- Retail / Toko: → NIB, SIUP (jika diperlukan), NPWP, izin lokasi dari Pemda
- Jasa / Konsultasi: → NIB, NPWP, kontrak kerja sesuai UU Ketenagakerjaan
- Manufaktur: → NIB, izin lingkungan AMDAL/UKL-UPL, SNI (jika produk wajib), NPWP

legal_risk_level guidelines:
- "low": semua izin mudah diurus, tidak ada regulasi ketat sektor
- "medium": ada 1-2 izin sektoral yang butuh waktu/biaya signifikan, atau regulasi ambigu
- "high": regulasi ketat, potensi pelanggaran besar, atau sektor sensitif (makanan tanpa BPOM, data tanpa UU PDP)

IMPORTANT: critical_risk_alerts HARUS menyebutkan dampak penalti spesifik (denda/pidana) jika melanggar.

OUTPUT FORMAT — ONLY valid JSON:
{
  "mandatory_permits": [
    "NIB (Nomor Induk Berusaha) via OSS",
    "NPWP Badan/Perorangan"
  ],
  "sector_specific_permits": [
    "Sertifikasi Halal BPJPH (wajib untuk produk makanan minuman targetkan Muslim)",
    "PIRT dari Dinas Kesehatan (untuk produksi rumahan skala kecil)"
  ],
  "compliance_checklist_perizinan": [
    "Daftar di OSS (oss.go.id) untuk mendapatkan NIB",
    "Ajukan izin lokasi ke Pemda setempat"
  ],
  "compliance_checklist_pajak": [
    "Daftarkan NPWP perorangan/badan di KPP terdekat",
    "Laporkan PPh 21 jika memiliki karyawan"
  ],
  "compliance_checklist_legalitas": [
    "Buat perjanjian kerja sesuai UU No. 13/2003 tentang Ketenagakerjaan",
    "Buat terms & conditions dan kebijakan privasi jika ada platform digital"
  ],
  "critical_risk_alerts": [
    "Menjual produk makanan tanpa izin BPOM/PIRT: denda hingga Rp 4 miliar atau pidana 5 tahun (UU No. 18/2012 tentang Pangan)",
    "Operasional platform digital tanpa pendaftaran PSE Kominfo: dapat diblokir dan denda administratif (PP No. 71/2019)"
  ],
  "legal_risk_level": "medium",
  "bureaucracy_estimated_duration": "3–6 minggu untuk NIB + PIRT (proses paralel)",
  "bureaucracy_complexity": "SEDANG",
  "compliance_score": 7,
  "confidence_score": 8
}"""

_SEARCH_TOPICS = [
    "persyaratan perizinan izin usaha oss nib sektor target indonesia 2025",
    "sanksi hukum pelanggaran regulasi sektor target uu perlindungan konsumen indonesia",
]


# ─────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────

def ethics_agent_node(state: EBPState) -> dict[str, Any]:
    """LangGraph node — Ethics & Compliance Agent."""
    t_start = time.perf_counter()
    logger.debug("=" * 60)
    logger.debug("→ Ethics & Compliance Agent dimulai")

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
            f"Sektor: {bc.sector if bc else '-'}",
            f"Peluang bisnis: {'; '.join(market.validated_opportunities[:2])}",
        ]

    if strategy:
        context_lines += [
            "\n=== STRATEGY CONTEXT (PESTEL — legal segment) ===",
            f"PESTEL Highlights: {'; '.join(strategy.pestel_highlights)}",
            f"Revenue Streams: {strategy.revenue_streams}",
        ]

    if feedback:
        context_lines += ["\n=== ORCHESTRATOR FEEDBACK ===", feedback]
    if user_fb:
        context_lines += ["\n=== FEEDBACK ENTREPRENEUR ===", user_fb]

    context_lines.append(
        "\nEvaluasi kepatuhan hukum Indonesia. Terapkan rules-based licensing logic. "
        "Return ONLY structured JSON sesuai format."
    )

    llm = get_llm(temperature=0.1)     # Rendah — compliance harus akurat
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
        agent_name="ethics_agent",
        max_search_calls=3,
    )

    parsed = extract_json(final_response.content)

    # ── Validate enums ──
    raw_risk = parsed.get("legal_risk_level", "medium")
    legal_risk = raw_risk if raw_risk in ("low", "medium", "high") else "medium"

    raw_complexity = parsed.get("bureaucracy_complexity", "SEDANG")
    complexity = raw_complexity if raw_complexity in ("RENDAH", "SEDANG", "TINGGI") else "SEDANG"

    report = EthicsAnalysisReport(
        mandatory_permits=parsed.get("mandatory_permits", []),
        sector_specific_permits=parsed.get("sector_specific_permits", []),
        compliance_checklist_perizinan=parsed.get("compliance_checklist_perizinan", []),
        compliance_checklist_pajak=parsed.get("compliance_checklist_pajak", []),
        compliance_checklist_legalitas=parsed.get("compliance_checklist_legalitas", []),
        critical_risk_alerts=parsed.get("critical_risk_alerts", []),
        legal_risk_level=legal_risk,
        bureaucracy_estimated_duration=parsed.get("bureaucracy_estimated_duration", "-"),
        bureaucracy_complexity=complexity,
        compliance_score=int(parsed.get("compliance_score", 5)),
        confidence_score=int(parsed.get("confidence_score", 5)),
    )

    logger.debug(
        f"   legal_risk={report.legal_risk_level} | "
        f"compliance_score={report.compliance_score} | "
        f"complexity={report.bureaucracy_complexity}"
    )
    logger.debug(f"✓ Ethics Agent selesai dalam {time.perf_counter() - t_start:.2f}s")
    logger.debug("=" * 60)

    return {
        "ethics_report": report,
        "messages": new_msgs,
    }
