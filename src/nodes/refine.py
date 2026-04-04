"""
src/nodes/refine.py
CRAG Node 4 — Filter graded chunks by relevance and trim AMBIGUOUS ones.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.state.crag_state import CRAGState, RefinedChunkDict

logger = logging.getLogger(__name__)


def _keyword_filter(text: str, query: str) -> str:
    """Keep only sentences that share keywords with the query."""
    keywords = {w.lower() for w in query.split() if len(w) > 3}
    kept = []
    for sentence in text.replace("\n", " ").split(". "):
        s = sentence.strip()
        if s and any(kw in s.lower() for kw in keywords):
            kept.append(s)
    return ". ".join(kept)


def refine_node(state: CRAGState) -> Dict[str, Any]:
    """
    Refine strategy:
      RELEVANT   → pass through unchanged
      AMBIGUOUS  → sentence-level keyword filter; discard if nothing survives
      IRRELEVANT → discard entirely
    """
    query  = state.get("active_query") or state["query"]
    graded = state.get("graded_chunks", [])
    refined: List[RefinedChunkDict] = []

    for g in graded:
        if g["grade"] == "IRRELEVANT":
            continue
        if g["grade"] == "RELEVANT":
            refined.append(
                RefinedChunkDict(page_content=g["page_content"], metadata=g["metadata"])
            )
        else:  # AMBIGUOUS
            filtered = _keyword_filter(g["page_content"], query)
            if filtered:
                refined.append(
                    RefinedChunkDict(page_content=filtered, metadata=g["metadata"])
                )

    logger.info("[refine] %d/%d chunks kept after refinement", len(refined), len(graded))
    return {"refined_chunks": refined}
