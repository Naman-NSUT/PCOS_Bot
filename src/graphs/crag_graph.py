"""
src/graphs/crag_graph.py
LangGraph StateGraph — 5-step Corrective RAG pipeline.

Graph topology:
  retrieve → grade → (conditional) → rewrite ─┐
                  ↓                           │
                refine → assemble → END      ←┘
                                 (loops back to retrieve on rewrite)

Conditional edge logic:
  If needs-rewrite AND retrieval_passes < MAX  → "rewrite"
  Otherwise                                    → "refine"
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from langgraph.graph import StateGraph, END

from config.settings import MAX_RETRIEVAL_PASSES
from src.state.crag_state import CRAGState
from src.nodes.retrieve import retrieve_node
from src.nodes.grade   import grade_node
from src.nodes.rewrite import rewrite_node
from src.nodes.refine  import refine_node
from src.nodes.assemble import assemble_node


# ── Conditional edge ──────────────────────────────────────────────────────

def _should_rewrite(state: CRAGState) -> Literal["rewrite", "refine"]:
    """
    Decide whether to rewrite the query (and do a second retrieval pass)
    or proceed directly to the refine step.
    """
    graded   = state.get("graded_chunks", [])
    passes   = state.get("retrieval_passes", 1)
    grades   = [g["grade"] for g in graded]

    all_irrelevant = all(g == "IRRELEVANT" for g in grades) if grades else False
    has_ambiguous  = any(g == "AMBIGUOUS"  for g in grades)

    if (all_irrelevant or has_ambiguous) and passes < MAX_RETRIEVAL_PASSES:
        return "rewrite"
    return "refine"


# ── Graph builder ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_crag_graph():
    """
    Build and compile the CRAG LangGraph.

    Returns a compiled Runnable that accepts CRAGState-compatible input dicts
    and returns the final CRAGState.

    Usage:
        graph = build_crag_graph()
        result = graph.invoke({"query": "PCOS hormone levels", "active_query": "PCOS hormone levels"})
        print(result["context"])
    """
    workflow = StateGraph(CRAGState)

    # ── Nodes ────────────────────────────────────────────────────────────
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade",    grade_node)
    workflow.add_node("rewrite",  rewrite_node)
    workflow.add_node("refine",   refine_node)
    workflow.add_node("assemble", assemble_node)

    # ── Edges ────────────────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        _should_rewrite,
        {"rewrite": "rewrite", "refine": "refine"},
    )

    # Rewrite loops back to retrieve for a second pass
    workflow.add_edge("rewrite",  "retrieve")
    workflow.add_edge("refine",   "assemble")
    workflow.add_edge("assemble", END)

    return workflow.compile()


def run_crag(query: str) -> dict:
    """
    Convenience function — run the CRAG graph for a single query.

    Returns the final state dict with keys: context, sources, grade_summary,
    rewritten_query, retrieval_passes.
    """
    graph = build_crag_graph()
    initial_state: CRAGState = {
        "query":            query,
        "active_query":     query,
        "chunks":           [],
        "retrieval_passes": 0,
        "graded_chunks":    [],
        "rewritten_query":  None,
        "refined_chunks":   [],
        "context":          "",
        "sources":          [],
        "grade_summary":    {},
    }
    return graph.invoke(initial_state)
