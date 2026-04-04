"""
src/nodes/assemble.py
CRAG Node 5 — Deduplicate, rank by knowledge tier, and build final context.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.state.crag_state import CRAGState

logger = logging.getLogger(__name__)

_TIER_RANK = {"guideline": 0, "research": 1, "patient_education": 2, "handcrafted": 3}


def assemble_node(state: CRAGState) -> Dict[str, Any]:
    """
    Build the final context string from refined chunks:
      1. Deduplicate by (source, start_index)
      2. Sort by tier rank — guideline docs go first
      3. Format with inline citations [Source: file | tier]
    """
    refined = state.get("refined_chunks", [])

    if not refined:
        logger.warning("[assemble] no refined chunks — empty context.")
        return {"context": "", "sources": []}

    # Dedup
    seen: set = set()
    unique = []
    for chunk in refined:
        key = (
            chunk["metadata"].get("source", ""),
            chunk["metadata"].get("start_index", chunk["page_content"][:40]),
        )
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    # Sort by tier
    unique.sort(key=lambda c: _TIER_RANK.get(c["metadata"].get("tier", "handcrafted"), 99))

    sections: List[str] = []
    sources: List[Dict[str, Any]] = []

    for chunk in unique:
        src   = chunk["metadata"].get("source", "unknown")
        tier  = chunk["metadata"].get("tier",   "unknown")
        page  = chunk["metadata"].get("page",   None)

        citation = f"[Source: {src}"
        if page is not None:
            citation += f", p.{int(page) + 1}"
        citation += f" | {tier}]"

        sections.append(f"{citation}\n{chunk['page_content']}")
        sources.append({"source": src, "tier": tier, "page": page})

    context = "\n\n---\n\n".join(sections)
    logger.info("[assemble] %d chunks → context length=%d chars", len(unique), len(context))
    return {"context": context, "sources": sources}
