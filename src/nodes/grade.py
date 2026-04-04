"""
src/nodes/grade.py
CRAG Node 2 — Grade each retrieved chunk with Gemini Flash.
Labels: RELEVANT | AMBIGUOUS | IRRELEVANT
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from functools import lru_cache
from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GRADER_MODEL
from src.state.crag_state import CRAGState, GradedChunkDict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    """Lazy Gemini client — created on first call, not at import time."""
    return genai.Client(api_key=GEMINI_API_KEY)

GRADE_LABELS = {"RELEVANT", "AMBIGUOUS", "IRRELEVANT"}

_SYSTEM = """\
You are a strict relevance grader for a PCOS (Polycystic Ovary Syndrome) clinical \
knowledge base.  Assess how well a retrieved chunk answers the user query.

Use EXACTLY one label:
  RELEVANT   — chunk directly and specifically addresses the query
  AMBIGUOUS  — chunk is related to PCOS but does not squarely answer the query
  IRRELEVANT — chunk provides no useful information for the query

Respond with ONLY valid JSON, no markdown:
{"grade": "<LABEL>", "reason": "<one sentence>"}
"""


def grade_node(state: CRAGState) -> Dict[str, Any]:
    """Grade each chunk from the latest retrieval pass."""
    query  = state.get("active_query") or state["query"]
    chunks = state.get("chunks", [])

    graded: List[GradedChunkDict] = []

    for chunk in chunks:
        prompt = (
            f"Query: {query}\n\n"
            f"Chunk (source: {chunk['metadata'].get('source', '?')}):\n"
            f"{chunk['page_content'][:600]}"
        )
        try:
            resp = _get_client().models.generate_content(
                model=GRADER_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=_SYSTEM),
            )
            raw = resp.text.strip().lstrip("```json").rstrip("```").strip()
            data = json.loads(raw)
            grade  = data.get("grade", "IRRELEVANT").upper()
            reason = data.get("reason", "")
            if grade not in GRADE_LABELS:
                grade = "IRRELEVANT"
        except Exception as exc:
            logger.warning("[grade] error for '%s': %s", chunk["metadata"].get("source"), exc)
            grade, reason = "AMBIGUOUS", "Grading error."

        graded.append(
            GradedChunkDict(
                page_content=chunk["page_content"],
                metadata=chunk["metadata"],
                grade=grade,
                reason=reason,
            )
        )

    summary = {label: sum(1 for g in graded if g["grade"] == label) for label in GRADE_LABELS}
    logger.info("[grade] summary=%s", summary)

    return {"graded_chunks": graded, "grade_summary": summary}
