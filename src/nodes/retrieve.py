"""
src/nodes/retrieve.py
CRAG Node 1 — Retrieve top-K chunks from ChromaDB.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from config.settings import RETRIEVAL_K
from src.ingestion.vector_store import get_retriever
from src.state.crag_state import CRAGState

logger = logging.getLogger(__name__)


def retrieve_node(state: CRAGState) -> Dict[str, Any]:
    """
    Retrieves top-K semantically similar chunks for the current active_query.
    On the first pass, active_query == query.  On subsequent passes it is the
    rewritten query produced by the rewrite node.
    """
    query  = state.get("active_query") or state["query"]
    passes = state.get("retrieval_passes", 0)

    retriever = get_retriever(k=RETRIEVAL_K)
    docs = retriever.invoke(query)

    chunks = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    logger.info("[retrieve] pass=%d  query='%s'  chunks=%d", passes + 1, query[:60], len(chunks))

    return {
        "chunks": chunks,
        "retrieval_passes": passes + 1,
        "active_query": query,  # carry forward (may already be the rewritten one)
    }
