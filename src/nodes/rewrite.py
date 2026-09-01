"""
src/nodes/rewrite.py
CRAG Node 3 — Rewrite the query when grading signals poor retrieval.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import GRADER_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL
from src.state.crag_state import CRAGState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=GRADER_MODEL,
        temperature=0.1,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

_SYSTEM = """\
You are a medical search query optimiser for a PCOS clinical knowledge base.
Given the original query and feedback from a relevance grader, produce a \
more specific query likely to match clinical documentation on PCOS \
(symptoms, diagnosis, hormone levels, treatment, lifestyle).
Respond with ONLY the improved query — plain text, no explanation.
"""


def rewrite_node(state: CRAGState) -> Dict[str, Any]:
    """
    Generates a better query using grading feedback.
    Sets active_query to the rewritten version; stores it in rewritten_query.
    """
    original  = state["query"]
    graded    = state.get("graded_chunks", [])

    reasons = "; ".join(
        f"[{g['grade']}] {g['reason']}"
        for g in graded
        if g["grade"] != "RELEVANT"
    )[:400]

    prompt = (
        f"Original query: {original}\n\n"
        f"Grader feedback (why results did not match):\n{reasons}\n\n"
        "Write an improved PCOS knowledge base search query:"
    )

    try:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=prompt)
        ]
        resp = _get_client().invoke(messages)
        rewritten = resp.content.strip()
    except Exception as exc:
        logger.warning("[rewrite] failed: %s — keeping original.", exc)
        rewritten = original

    logger.info("[rewrite] '%s' → '%s'", original[:60], rewritten[:60])
    return {"active_query": rewritten, "rewritten_query": rewritten}
