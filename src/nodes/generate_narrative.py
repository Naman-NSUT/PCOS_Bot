"""
src/nodes/generate_narrative.py
Analysis Node 3 — Gemini Pro generates the evidence-grounded narrative.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from functools import lru_cache
from google import genai
from google.genai import types

from config.settings import GEMINI_API_KEY, GENERATOR_MODEL, SAFETY_DISCLAIMER
from src.state.analysis_state import AnalysisState

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)

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
    """Call Gemini Pro with parsed values, flags, and CRAG context to produce the narrative."""
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
        resp = _get_client().models.generate_content(
            model=GENERATOR_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=_SYSTEM),
        )
        narrative = resp.text.strip()
    except Exception as exc:
        logger.error("[generate] Gemini call failed: %s", exc)
        narrative = (
            "Narrative generation failed due to an API error. "
            "Please review the parsed values and flags directly."
        )

    return {"narrative": narrative, "disclaimer": SAFETY_DISCLAIMER}
