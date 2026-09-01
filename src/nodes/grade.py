"""
src/nodes/grade.py
CRAG Node 2 — Grade each retrieved chunk with the grader model.
Labels: RELEVANT | AMBIGUOUS | IRRELEVANT
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import GRADER_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from src.state.crag_state import CRAGState, GradedChunkDict

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    """Lazy chat client — created on first call."""
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=GRADER_MODEL,
        temperature=0.1,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

GRADE_LABELS = {"RELEVANT", "AMBIGUOUS", "IRRELEVANT"}

# NOTE: OpenAI-style JSON mode requires the top-level value to be an OBJECT —
# asking for a bare JSON array makes the model wrap it under an arbitrary key.
# So we pin the wrapper key explicitly and read `grades` back out.
_SYSTEM = """\
You are a strict relevance grader for a PCOS (Polycystic Ovary Syndrome) clinical \
knowledge base. Assess how well retrieved chunks address the user query.

For each chunk, use EXACTLY one label:
  RELEVANT   — chunk directly and specifically addresses the query
  AMBIGUOUS  — chunk is related to PCOS but does not squarely answer the query
  IRRELEVANT — chunk provides no useful information for the query

Respond with ONLY a JSON object, no markdown, no other text. The object has a
single key "grades" whose value is an array with EXACTLY one entry per chunk.
Each entry MUST carry the "index" of the chunk it grades, matching the
"Chunk <n>" header it was given. Never omit a chunk and never merge two.
{
  "grades": [
    {"index": 0, "grade": "<LABEL>", "reason": "<one sentence>"},
    {"index": 1, "grade": "<LABEL>", "reason": "<one sentence>"}
  ]
}
"""


def _strip_code_fence(text: str) -> str:
    """
    Remove a surrounding ```json … ``` fence if the model added one.

    Note: str.lstrip("```json") would strip any leading run of the characters
    ` j s o n — mangling a payload that legitimately starts with one of them.
    """
    t = text.strip()
    if t.startswith("```"):
        newline = t.find("\n")
        t = t[newline + 1:] if newline != -1 else t[3:]
    if t.endswith("```"):
        t = t[:-3]
    return t.strip()


def _extract_grades(raw: str) -> List[Dict[str, Any]]:
    """
    Parse the grader payload into a list of {grade, reason} dicts.

    Tolerates the three shapes models actually emit: the requested
    {"grades": [...]} object, a bare [...] array, and {"<other_key>": [...]}
    when the model renames the wrapper.
    """
    data = json.loads(raw)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("grades"), list):
            return data["grades"]
        # Model renamed the wrapper key — take the first list-of-objects value.
        for value in data.values():
            if isinstance(value, list):
                return value
        # Single ungrouped object, e.g. {"grade": "...", "reason": "..."}
        if "grade" in data:
            return [data]
    raise ValueError(f"Unrecognised grader payload shape: {type(data).__name__}")


def _fallback_grades(chunks: List[Dict[str, Any]], reason: str) -> List[GradedChunkDict]:
    """One AMBIGUOUS entry per chunk — used when the batch call is unusable."""
    return [
        GradedChunkDict(
            page_content=c["page_content"], metadata=c["metadata"],
            grade="AMBIGUOUS", reason=reason,
        )
        for c in chunks
    ]


def _pair_grades(
    chunks: List[Dict[str, Any]], data: List[Any]
) -> List[GradedChunkDict]:
    """
    Zip parsed grades onto chunks, always returning EXACTLY len(chunks) entries.

    The model controls the length and element types of `data`, so neither is
    trusted: a short list, a long list, or a non-dict element (a bare
    "RELEVANT" string is a shape JSON mode readily emits) must degrade that
    one entry rather than raise — a raise mid-loop would otherwise abandon
    already-graded chunks and fall through to the whole-batch fallback.

    Grades are matched by their self-declared "index" where present. Models
    routinely return fewer grades than chunks, and under positional matching a
    single skipped chunk shifts every later grade onto the wrong chunk —
    silently mislabelling content that is then fed to the generator. Explicit
    indices make a dropped grade affect only its own chunk.
    """
    if len(data) != len(chunks):
        logger.warning(
            "[grade] grader returned %d grades for %d chunks — "
            "unmatched chunks default to AMBIGUOUS",
            len(data), len(chunks),
        )

    # Map by declared index when the model provides one; fall back to position.
    by_index: Dict[int, Any] = {}
    positional: List[Any] = []
    for pos, item in enumerate(data):
        idx = item.get("index") if isinstance(item, dict) else None
        if isinstance(idx, bool) or not isinstance(idx, int):
            idx = None
        if idx is not None and 0 <= idx < len(chunks) and idx not in by_index:
            by_index[idx] = item
        else:
            positional.append(item)

    graded: List[GradedChunkDict] = []
    for i, chunk in enumerate(chunks):
        item = by_index.get(i)
        if item is None:
            # No index-matched grade: consume the next unindexed entry, but only
            # when indices were absent entirely (pure positional payload).
            item = positional.pop(0) if positional and not by_index else None

        if isinstance(item, dict):
            grade  = str(item.get("grade", "IRRELEVANT")).upper()
            reason = str(item.get("reason", ""))
        elif isinstance(item, str):
            # Model emitted a bare label instead of an object.
            grade, reason = item.upper(), ""
        else:
            grade, reason = "AMBIGUOUS", "Missing or malformed grade in batch."

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
    return graded


def grade_node(state: CRAGState) -> Dict[str, Any]:
    """Grade all chunks in a single batch call to the grader model."""
    query  = state.get("active_query") or state["query"]
    chunks = state.get("chunks", [])

    if not chunks:
        return {"graded_chunks": [], "grade_summary": {}}

    # 1. Build batch prompt
    batch_prompt = f"Query: {query}\n\n"
    for i, chunk in enumerate(chunks):
        batch_prompt += f"--- Chunk {i} (source: {chunk['metadata'].get('source', '?')}) ---\n"
        batch_prompt += f"{chunk['page_content'][:800]}\n\n"

    try:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=batch_prompt)
        ]
        resp = _get_client().invoke(messages)
        raw = _strip_code_fence(resp.content.strip())
        data = _extract_grades(raw)
        graded = _pair_grades(chunks, data)
    except Exception as exc:
        logger.error("[grade] Batch grading failed: %s", exc)
        # Whole-batch fallback. Rebuilt from scratch — never appended to a
        # partially-filled list, or graded_chunks would end up longer than
        # chunks and duplicate entries would reach the generator.
        graded = _fallback_grades(chunks, "Batch grading error fallback.")

    summary = {label: sum(1 for g in graded if g["grade"] == label) for label in GRADE_LABELS}
    logger.info("[grade] summary=%s", summary)

    return {"graded_chunks": graded, "grade_summary": summary}
