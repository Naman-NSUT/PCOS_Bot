"""
src/graphs/analysis_graph.py
LangGraph StateGraph — Full PCOS report analysis pipeline.

Graph topology (linear):
  parse_report → flag_indicators → run_crag → generate_narrative → END

The run_crag node calls the compiled CRAG graph as a sub-graph to retrieve
relevant clinical context before generating the narrative.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from src.state.analysis_state import AnalysisState
from src.nodes.parse_report       import parse_report_node
from src.nodes.flag_indicators    import flag_indicators_node
from src.nodes.generate_narrative import generate_narrative_node
from src.graphs.crag_graph        import run_crag


# ── CRAG wrapper node (bridges the two graphs) ────────────────────────────

def run_crag_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Execute the CRAG sub-graph using the crag_query built by flag_indicators_node.
    Stores context, sources, and rewritten_query into AnalysisState.
    """
    query = state.get("crag_query", "PCOS diagnostic criteria and hormone levels")
    result = run_crag(query)
    return {
        "crag_context":          result.get("context", ""),
        "crag_sources":          result.get("sources", []),
        "crag_rewritten_query":  result.get("rewritten_query"),
    }


# ── Graph builder ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_analysis_graph():
    """
    Build and compile the analysis LangGraph.

    Usage:
        graph = build_analysis_graph()
        result = graph.invoke({
            "report_text": "LH: 12.4 mIU/mL, Testosterone: 85 ng/dL …",
            "patient_context": "27F, irregular periods",
        })
        print(result["narrative"])
    """
    workflow = StateGraph(AnalysisState)

    workflow.add_node("parse",    parse_report_node)
    workflow.add_node("flag",     flag_indicators_node)
    workflow.add_node("retrieve", run_crag_node)
    workflow.add_node("generate", generate_narrative_node)

    workflow.set_entry_point("parse")

    workflow.add_edge("parse",    "flag")
    workflow.add_edge("flag",     "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_analysis(report_text: str, patient_context: Optional[str] = None) -> dict:
    """
    Convenience function — run the full analysis pipeline.

    Returns the final AnalysisState dict with keys:
      parsed_values, diagnostic_flags, crag_context, crag_sources,
      crag_rewritten_query, narrative, disclaimer.
    """
    graph = build_analysis_graph()
    initial_state: AnalysisState = {
        "report_text":          report_text,
        "patient_context":      patient_context,
        "parsed_values":        {},
        "diagnostic_flags":     {},
        "crag_query":           "",
        "crag_context":         "",
        "crag_sources":         [],
        "crag_rewritten_query": None,
        "narrative":            "",
        "disclaimer":           "",
    }
    return graph.invoke(initial_state)
