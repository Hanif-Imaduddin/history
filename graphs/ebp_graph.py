"""Pendefinisian LangGraph graph dengan menguhungkan semua node agent dalam EBP workflow."""
from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from nodes.ethics_agent import ethics_agent_node
from nodes.financial_analyst import financial_analyst_node
from nodes.lead_orchestrator import final_summary_node, lead_orchestrator_node
from nodes.market_scout import market_scout_node
from nodes.strategic_architect import strategic_architect_node
from states.schema import EBPState

checkpointer = MemorySaver()


def _route_from_orchestrator(state: EBPState) -> Literal["market_scout", "final_summary"]:
    """Decide whether to run another pipeline iteration or generate the final report."""
    status = state.get("approval_status", "pending")
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)

    if status == "approved" or iteration >= max_iter:
        return "final_summary"
    return "market_scout"


def build_graph() -> StateGraph:
    """Build and compile the EBP multi-agent graph."""
    workflow = StateGraph(EBPState)

    # Register nodes
    workflow.add_node("lead_orchestrator", lead_orchestrator_node)
    workflow.add_node("market_scout", market_scout_node)
    workflow.add_node("strategic_architect", strategic_architect_node)
    workflow.add_node("financial_analyst", financial_analyst_node)
    workflow.add_node("ethics_agent", ethics_agent_node)
    workflow.add_node("final_summary", final_summary_node)

    # Entry point
    workflow.set_entry_point("lead_orchestrator")

    # Orchestrator decides: continue pipeline or generate final report
    workflow.add_conditional_edges(
        "lead_orchestrator",
        _route_from_orchestrator,
        {"market_scout": "market_scout", "final_summary": "final_summary"},
    )

    # Final summary always leads to END
    workflow.add_edge("final_summary", END)

    # Sequential specialist pipeline
    workflow.add_edge("market_scout", "strategic_architect")
    workflow.add_edge("strategic_architect", "financial_analyst")
    workflow.add_edge("financial_analyst", "ethics_agent")

    # After ethics, return to orchestrator for evaluation
    workflow.add_edge("ethics_agent", "lead_orchestrator")

    return workflow.compile(checkpointer=checkpointer)


# Module-level compiled graph (lazy import safe)
graph = build_graph()
