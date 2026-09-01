"""
src/nodes/generate_narrative.py
Analysis Node 3 — the generator model writes the evidence-grounded narrative.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import (
    GENERATOR_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SAFETY_DISCLAIMER,
)
from src.state.analysis_state import AnalysisState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=GENERATOR_MODEL,
        temperature=0.3,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

_SYSTEM = """\
You are a knowledgeable PCOS health information assistant.  Generate a clear, \
empathetic, evidence-grounded report summarising a patient's lab results in the \
context of PCOS.

Structure your response with these exact headings:
## Lab Results Summary
## PCOS Indicator Findings
## What These Results Mean
## Recommended Next Steps

Rules:
- NEVER state or imply a diagnosis of PCOS.
- Always refer the patient to their healthcare provider.
- Use plain, empathetic language.
- Ground every claim in the retrieved context when available.
- Do NOT add a disclaimer — it will be appended separately.
"""


def generate_narrative_node(state: AnalysisState) -> Dict[str, Any]:
    """Call the generator model with parsed values, flags, and CRAG context to produce the narrative."""
    ctx = state.get("crag_context") or "No clinical context retrieved — use general PCOS knowledge."

    user_prompt = ""
    if state.get("patient_context"):
        user_prompt += f"PATIENT CONTEXT:\n{state['patient_context']}\n\n"

    user_prompt += f"""\
LAB VALUES (extracted):
{json.dumps(state.get("parsed_values", {}), indent=2)}

PCOS INDICATOR FLAGS (rule-based):
{json.dumps(state.get("diagnostic_flags", {}), indent=2)}

RETRIEVED CLINICAL CONTEXT:
{ctx}
"""

    try:
        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_prompt)
        ]
        resp = _get_client().invoke(messages)
        narrative = resp.content.strip()
    except Exception as exc:
        logger.error("[generate] LLM call failed: %s", exc)
        narrative = (
            "Narrative generation failed due to an API error. "
            "Please review the parsed values and flags directly."
        )

    return {"narrative": narrative, "disclaimer": SAFETY_DISCLAIMER}
