"""
src/state/crag_state.py
LangGraph state schema for the CRAG retrieval pipeline.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document

# operator.add is used for list fields so LangGraph merges them across nodes
import operator


class GradedChunkDict(TypedDict):
    """Serialisable representation of a graded chunk (avoids storing Document objects in state)."""
    page_content: str
    metadata: Dict[str, Any]
    grade: str    # RELEVANT | AMBIGUOUS | IRRELEVANT
    reason: str


class RefinedChunkDict(TypedDict):
    page_content: str     # may differ from original if AMBIGUOUS was filtered
    metadata: Dict[str, Any]


class CRAGState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                          # original user query — never mutated
    active_query: str                   # current query (rewritten after pass 1 if needed)

    # ── Retrieval ──────────────────────────────────────────────────────────
    chunks: List[Dict[str, Any]]        # raw retrieved chunks (page_content + metadata)
    retrieval_passes: int               # how many retrieval passes have happened

    # ── Grading ───────────────────────────────────────────────────────────
    graded_chunks: List[GradedChunkDict]

    # ── Rewrite ───────────────────────────────────────────────────────────
    rewritten_query: Optional[str]

    # ── Refine ────────────────────────────────────────────────────────────
    refined_chunks: List[RefinedChunkDict]

    # ── Assembly (final outputs) ───────────────────────────────────────────
    context: str
    sources: List[Dict[str, Any]]
    grade_summary: Dict[str, int]
