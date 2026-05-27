"""
ClarioAI — EBP State Schema (Refactored)

Perubahan utama dari versi lama:
- Semua report sekarang pakai typed dataclasses, bukan text blob
- Setiap agent output ke fields terukur (int, List[str], Literal)
- Ditambahkan ValidationReport untuk contradiction detection
- Messages hanya simpan concise summaries, bukan full reports
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# INPUT CONSTRAINTS
# ─────────────────────────────────────────────

@dataclass
class BusinessConstraints:
    """Batasan bisnis yang diberikan pengguna — input awal pipeline."""
    sector: str
    target_audience: str
    business_idea: str
    location: str
    budget_range: str                    # e.g. "Rp 10.000.000 – Rp 50.000.000"
    business_model: Literal["online", "offline", "hybrid"]
    experience_level: Literal["beginner", "intermediate", "experienced"]
    risk_tolerance: Literal["low", "medium", "high"]


# ─────────────────────────────────────────────
# MARKET SCOUT REPORT
# ─────────────────────────────────────────────

@dataclass
class CompetitorInsight:
    name: str
    strengths: List[str]
    weaknesses: List[str]          # Sumber keluhan pelanggan yang nyata
    market_position: Optional[str] = None


@dataclass
class MarketScoutReport:
    """Output structured dari Market Scout Agent."""
    # Scoring
    demand_score: int               # 1–10, bukan angka imajinasi
    competition_level: Literal["low", "medium", "high"]
    opportunity_score: int          # 1–10, seberapa terbuka celah pasar

    # Pasar
    market_trend_summary: str       # Maks ~2 kalimat
    customer_pain_points: List[str] # Maks 5 pain points konkret
    competitors: List[CompetitorInsight]

    # TAM-SAM-SOM — simplified, berbasis data retrieval, bukan imajinasi LLM
    tam_estimate: str               # e.g. "Rp 4,5 Triliun/tahun"
    sam_estimate: str
    som_estimate: str
    market_sizing_methodology: str  # Sumber & metode kalkulasi

    # Peluang
    validated_opportunities: List[str]  # Maks 3 peluang konkret
    source_links: List[str]

    confidence_score: int           # 1–10


# ─────────────────────────────────────────────
# STRATEGIC REPORT
# ─────────────────────────────────────────────

@dataclass
class MonthlyMilestone:
    objective: str
    kpis: List[str]   # KPI kuantitatif, bukan narasi


@dataclass
class StrategicReport:
    """Output structured dari Strategic Architect Agent."""
    # SWOT — max 3 poin per section
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]

    # PESTEL — ringkas, poin-poin
    pestel_highlights: List[str]    # Maks 6 poin paling relevan

    # Prioritas & Eksekusi
    top_problem_priorities: List[str]   # Maks 3
    strategic_order_of_operations: str  # Narasi singkat: apa dulu sebelum apa

    # Roadmap 3 bulan
    roadmap: Dict[str, MonthlyMilestone]  # key: "month_1", "month_2", "month_3"

    # Lean Canvas — ringkas
    value_proposition: str
    customer_segments: str
    channels: str
    cost_structure: str
    revenue_streams: str

    recommended_sales_channel: str
    confidence_score: int


# ─────────────────────────────────────────────
# FINANCIAL ANALYSIS REPORT
# ─────────────────────────────────────────────

@dataclass
class SensitivityScenario:
    label: str           # e.g. "Volume turun 20%"
    net_profit_rp: int   # Dalam IDR


@dataclass
class FinancialAnalysisReport:
    """Output structured dari Financial Analyst Agent."""
    # Proyeksi bulanan (IDR)
    estimated_monthly_revenue_rp: int
    estimated_monthly_cogs_rp: int
    estimated_monthly_fixed_costs_rp: int
    estimated_monthly_net_profit_rp: int

    # BEP
    bep_units_per_month: int
    bep_revenue_rp: int
    cash_runway_months: int          # Berapa bulan modal bertahan jika rugi

    # Startup
    estimated_startup_cost_rp: int
    estimated_break_even_months: int

    # Stress test
    sensitivity_scenarios: List[SensitivityScenario]  # Maks 3

    # Benchmark & verdict
    industry_benchmark_notes: str    # Singkat, 2–3 kalimat
    financial_feasibility_verdict: Literal["LAYAK", "LAYAK_DENGAN_CATATAN", "TIDAK_LAYAK"]

    # Risk & assumptions
    key_financial_risks: List[str]   # Maks 5
    assumptions: List[str]           # Eksplisit — apa yang diasumsikan

    affordability_score: int         # 1–10
    confidence_score: int            # 1–10


# ─────────────────────────────────────────────
# ETHICS & COMPLIANCE REPORT
# ─────────────────────────────────────────────

@dataclass
class EthicsAnalysisReport:
    """Output structured dari Ethics & Compliance Agent."""
    # Perizinan
    mandatory_permits: List[str]          # Izin wajib (NIB, NPWP, dll)
    sector_specific_permits: List[str]    # Izin spesifik sektor

    # Checklist compliance
    compliance_checklist_perizinan: List[str]
    compliance_checklist_pajak: List[str]
    compliance_checklist_legalitas: List[str]

    # Risiko
    critical_risk_alerts: List[str]       # Peringatan hukum kritis + dampak penalti
    legal_risk_level: Literal["low", "medium", "high"]

    # Effort birokrasi
    bureaucracy_estimated_duration: str   # e.g. "2–4 minggu"
    bureaucracy_complexity: Literal["RENDAH", "SEDANG", "TINGGI"]

    compliance_score: int                 # 1–10
    confidence_score: int                 # 1–10


# ─────────────────────────────────────────────
# VALIDATION REPORT (NEW — Rule-based layer)
# ─────────────────────────────────────────────

@dataclass
class ContradictionFlag:
    severity: Literal["critical", "warning", "info"]
    description: str              # Apa kontradiksinya
    source_agents: List[str]      # Agent mana yang berkontradiksi


@dataclass
class ValidationReport:
    """Output dari Validation Layer — rule-based contradiction detection."""
    contradictions: List[ContradictionFlag]
    missing_assumptions: List[str]   # Asumsi yang belum dideklarasikan agent
    unrealistic_claims: List[str]    # Klaim yang secara logika tidak masuk akal
    auto_flags: List[str]            # Flag otomatis dari rule engine

    overall_confidence: int          # 1–10, dihitung dari hybrid scoring
    is_valid: bool                   # False jika ada critical contradiction


# ─────────────────────────────────────────────
# MAIN GRAPH STATE
# ─────────────────────────────────────────────

ApprovalStatus = Literal["pending", "approved", "rejected", "feasible", "risky", "not_recommended"]


class EBPState(TypedDict):
    state_id: str

    # Input
    business_constraints: BusinessConstraints

    # Agent outputs — semua typed, bukan text blob
    market_report: Optional[MarketScoutReport]
    strategy_report: Optional[StrategicReport]
    financial_report: Optional[FinancialAnalysisReport]
    ethics_report: Optional[EthicsAnalysisReport]

    # Validation layer output
    validation_report: Optional[ValidationReport]

    # Orchestrator
    approval_status: ApprovalStatus
    orchestrator_feedback: Optional[str]   # Concise, bukan essay
    confidence_score: int                  # Computed dari validation + agent scores
    final_verdict: Optional[Literal["PROCEED", "PROCEED_WITH_CAUTION", "PIVOT_RECOMMENDED", "NOT_RECOMMENDED"]]
    final_recommendations: List[str]       # Maks 3

    # Iteration control
    iteration: int
    max_iterations: int
    user_feedback: Optional[str]

    # Final output
    final_result: Optional[str]            # Markdown report untuk user

    # Messages — HANYA concise summaries, bukan full reports
    messages: Annotated[List[BaseMessage], add_messages]
