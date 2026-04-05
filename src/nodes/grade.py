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
knowledge base. Assess how well retrieved chunks address the user query.

For each chunk, use EXACTLY one label:
  RELEVANT   — chunk directly and specifically addresses the query
  AMBIGUOUS  — chunk is related to PCOS but does not squarely answer the query
  IRRELEVANT — chunk provides no useful information for the query

Respond with ONLY a JSON array of objects, no markdown, no other text:
[
  {"grade": "<LABEL>", "reason": "<one sentence>"},
  ...
]
"""


def grade_node(state: CRAGState) -> Dict[str, Any]:
    """Grade all chunks in a single batch call to Gemini Flash."""
    query  = state.get("active_query") or state["query"]
    chunks = state.get("chunks", [])

    if not chunks:
        return {"graded_chunks": [], "grade_summary": {}}

    # 1. Build batch prompt
    batch_prompt = f"Query: {query}\n\n"
    for i, chunk in enumerate(chunks):
        batch_prompt += f"--- Chunk {i} (source: {chunk['metadata'].get('source', '?')}) ---\n"
        batch_prompt += f"{chunk['page_content'][:800]}\n\n"

    graded: List[GradedChunkDict] = []
    
    # 2. Call Gemini for the whole batch
    try:
        resp = _get_client().models.generate_content(
            model=GRADER_MODEL,
            contents=batch_prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM),
        )
        raw = resp.text.strip().lstrip("```json").rstrip("```").strip()
        data = json.loads(raw)
        
        if not isinstance(data, list):
            raise ValueError("Expected JSON array from batch grader")

        for i, chunk in enumerate(chunks):
            # Try to match the index, default to AMBIGUOUS if missing
            item = data[i] if i < len(data) else {"grade": "AMBIGUOUS", "reason": "Missing in batch."}
            grade  = str(item.get("grade", "IRRELEVANT")).upper()
            reason = str(item.get("reason", ""))
            
            if grade not in GRADE_LABELS:
                grade = "IRRELEVANT"
                
            graded.append(
                GradedChunkDict(
                    page_content=chunk["page_content"],
                    metadata=chunk["metadata"],
                    grade=grade,
                    reason=reason,
                )
            )
    except Exception as exc:
        logger.error("[grade] Batch grading failed: %s", exc)
        # Fallback to AMBIGUOUS for all chunks if the entire batch call fails
        for chunk in chunks:
            graded.append(
                GradedChunkDict(
                    page_content=chunk["page_content"],
                    metadata=chunk["metadata"],
                    grade="AMBIGUOUS",
                    reason="Batch grading error fallback.",
                )
            )

    summary = {label: sum(1 for g in graded if g["grade"] == label) for label in GRADE_LABELS}
    logger.info("[grade] summary=%s", summary)

    return {"graded_chunks": graded, "grade_summary": summary}
