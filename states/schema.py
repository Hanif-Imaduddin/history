from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


@dataclass
class BussinessConstraints:
    """Batasan bisnis yang diberikan oleh pengguna"""
    sector_and_domain: str
    audience: str
    initial_prompt: str


@dataclass
class MarketScoutReport:
    """Hasil dari market scout report yang dihasilkan oleh agent"""
    ideas: List[str]
    agent_explanation: str


@dataclass
class StrategicReport:
    """Hasil dari strategic report yang dihasilkan oleh agent"""
    swot_analysis: str
    pastel_analysis: str


@dataclass
class FinancialAnalysisReport:
    """Hasil dari financial analysis report yang dihasilkan oleh agent"""
    analysis_result: str


@dataclass
class EthicsAnalysisReport:
    """Hasil dari ethics analysis report yang dihasilkan oleh agent"""
    analysis_result: str


ApprovalStates = Literal['pending', 'approved', 'rejected']


class EBPState(TypedDict):
    state_id: str
    user_id: str
    bussiness_constraints: Optional[BussinessConstraints]
    market_scout_report: Optional[MarketScoutReport]
    strategic_report: Optional[StrategicReport]
    financial_analysis_report: Optional[FinancialAnalysisReport]
    ethics_analysis_report: Optional[EthicsAnalysisReport]
    approval_status: ApprovalStates
    orchestrator_feedback: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    iteration: int
    max_iterations: int
    user_feedback: Optional[str]
