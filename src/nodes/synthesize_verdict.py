"""
src/nodes/synthesize_verdict.py
Report Node 4 — combine what the LABS say with what the CONVERSATION said.

Why combining is not merely additive presentation:

The 2023 guideline follows Rotterdam, which requires TWO of three features —
hyperandrogenism, ovulatory dysfunction, and polycystic ovarian morphology.
Bloodwork and conversation cover different ones, and neither covers all three:

  hyperandrogenism        labs give the BIOCHEMICAL form (testosterone, DHEAS)
                          conversation gives the CLINICAL form (hirsutism, acne)
  ovulatory dysfunction   conversation only — no blood test establishes it;
                          LH/FSH is a weak proxy at best
  polycystic morphology   ULTRASOUND only. AMH is a proxy, not a substitute.

So a report read in isolation cannot reach two criteria, and a conversation in
isolation usually cannot either. Reading them together is the point.

The concordance below is computed deterministically from the rule engine's flags
and the regex-derived symptom list. The model narrates that structure; it never
decides it.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config.settings import (
    GENERATOR_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, SAFETY_DISCLAIMER,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_client():
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=GENERATOR_MODEL, temperature=0.3,
        api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL,
    )


# Symptom tags that speak to each criterion, from the conversation side.
_CLINICAL_SUPPORT: Dict[str, List[str]] = {
    "hyperandrogenism":     ["hirsutism", "acne", "hair_loss", "acanthosis"],
    "ovulatory_dysfunction": ["irregular_periods", "fertility_concerns"],
    "metabolic":            ["weight_gain", "insulin_related", "acanthosis", "fatigue"],
}

_CRITERION_LABEL = {
    "hyperandrogenism":      "Excess androgen activity",
    "ovulatory_dysfunction": "Irregular or absent ovulation",
    "polycystic_morphology": "Polycystic ovarian morphology",
    "metabolic":             "Metabolic / insulin picture",
}


def build_concordance(
    diagnostic_flags: Dict[str, Any],
    symptom_list: List[str],
    report_notes: List[str],
) -> Dict[str, Any]:
    """
    Deterministically cross-reference labs against conversation.

    No model involved. Each criterion records what the LABS support, what the
    CONVERSATION supports, and whether the two agree — including the cases that
    matter most clinically: a criterion only one side can see.
    """
    flags = diagnostic_flags or {}
    symptoms = set(symptom_list or [])
    notes_blob = " ".join(report_notes or []).lower()

    def clinical(criterion: str) -> List[str]:
        return sorted(symptoms & set(_CLINICAL_SUPPORT.get(criterion, [])))

    concordance: Dict[str, Any] = {}

    # 1. Hyperandrogenism — both sides can speak to this.
    concordance["hyperandrogenism"] = {
        "lab_support": bool(flags.get("hyperandrogenism")),
        "lab_evidence": flags.get("hyperandrogenism_evidence", []),
        "clinical_support": clinical("hyperandrogenism"),
    }

    # 2. Ovulatory dysfunction — conversation is the primary source. The lab side
    #    is deliberately marked as a proxy so it is never presented as equivalent.
    lh_fsh = flags.get("polycystic_morphology_evidence", [])
    concordance["ovulatory_dysfunction"] = {
        "lab_support": False,
        "lab_proxy": [e for e in lh_fsh if "LH/FSH" in str(e)],
        "clinical_support": clinical("ovulatory_dysfunction"),
    }

    # 3. Polycystic morphology — ultrasound only. AMH is a proxy; a printed
    #    ultrasound remark on the report is the real thing.
    concordance["polycystic_morphology"] = {
        "lab_support": False,
        "lab_proxy": [e for e in lh_fsh if "AMH" in str(e)],
        "imaging_mentioned": any(
            k in notes_blob for k in ("ultrasound", "follicle", "ovarian volume", "sonograph")
        ),
        "clinical_support": [],
    }

    # 4. Metabolic — not a Rotterdam criterion, but the main long-term risk.
    concordance["metabolic"] = {
        "lab_support": bool(flags.get("insulin_resistance")),
        "lab_evidence": flags.get("insulin_resistance_evidence", []),
        "risk_score": flags.get("metabolic_risk_score", 0),
        "clinical_support": clinical("metabolic"),
    }

    # How many Rotterdam criteria have ANY support from either side. Reported as
    # "features present", never as a diagnosis — Rotterdam also requires other
    # causes to be excluded, which this system cannot do.
    rotterdam = ["hyperandrogenism", "ovulatory_dysfunction", "polycystic_morphology"]
    supported = [
        c for c in rotterdam
        if concordance[c].get("lab_support")
        or concordance[c].get("clinical_support")
        or concordance[c].get("imaging_mentioned")
    ]

    return {
        "criteria": concordance,
        "features_with_support": supported,
        "feature_count": len(supported),
        "corroborated": [
            c for c in rotterdam
            if concordance[c].get("lab_support") and concordance[c].get("clinical_support")
        ],
        "conversation_only": [
            c for c in rotterdam
            if concordance[c].get("clinical_support") and not concordance[c].get("lab_support")
        ],
        "labs_only": [
            c for c in rotterdam
            if concordance[c].get("lab_support") and not concordance[c].get("clinical_support")
        ],
        "not_assessable": [
            c for c in rotterdam
            if not concordance[c].get("lab_support")
            and not concordance[c].get("clinical_support")
            and not concordance[c].get("imaging_mentioned")
        ],
    }


_SYSTEM = """\
You are Maya, explaining a lab report to the person whose report it is.

You are given a DETERMINISTIC cross-reference between their bloodwork and what
they told you in conversation. It was computed by rules, not by you. Explain it;
do not recompute it, and never contradict it.

Structure your reply with these headings exactly:
## What your results show
## How this lines up with what you've told me
## What this does and doesn't tell us
## What to ask your doctor

Rules:
- NEVER state or imply that they have PCOS. Diagnosis requires a clinician, an
  ultrasound, and exclusion of other causes — say so plainly.
- Where labs and conversation agree, say so; that agreement is the useful part.
- Where only one source speaks to something, name the gap explicitly. In
  particular: no blood test establishes irregular ovulation, and AMH is not a
  substitute for an ultrasound.
- Use plain language. Give the number and what it means, not just the number.
- Be warm and direct. Do not catastrophise a single out-of-range value.
- Do NOT add a disclaimer; one is appended separately.
"""


def synthesize_verdict(
    parsed_values: Dict[str, Any],
    diagnostic_flags: Dict[str, Any],
    concordance: Dict[str, Any],
    symptom_list: List[str],
    crag_context: str = "",
    memory_block: str = "",
    report_notes: List[str] | None = None,
) -> Dict[str, Any]:
    """Narrate the deterministic concordance. Returns {verdict, disclaimer}."""
    parts = [
        f"LAB VALUES (parsed and classified by rules):\n{json.dumps(parsed_values, indent=2)}",
        f"INDICATOR FLAGS (rule engine):\n{json.dumps(diagnostic_flags, indent=2)}",
        f"CROSS-REFERENCE — labs vs conversation (computed, authoritative):\n"
        f"{json.dumps(concordance, indent=2)}",
        f"SYMPTOMS THEY DESCRIBED IN CONVERSATION: "
        f"{', '.join(s.replace('_', ' ') for s in symptom_list) if symptom_list else 'none recorded'}",
    ]
    if report_notes:
        parts.append("NON-NUMERIC REMARKS PRINTED ON THE REPORT:\n- " + "\n- ".join(report_notes))
    if memory_block:
        parts.append(memory_block)
    if crag_context:
        parts.append(f"RETRIEVED CLINICAL CONTEXT (ground factual claims in this):\n{crag_context}")

    try:
        resp = _get_client().invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content="\n\n".join(parts)),
        ])
        verdict = resp.content.strip()
    except Exception as exc:
        logger.error("[verdict] generation failed: %s", exc)
        verdict = (
            "I could read your report, but something went wrong while writing it up. "
            "The values and flags are still available above — please try again in a moment."
        )

    return {"verdict": verdict, "disclaimer": SAFETY_DISCLAIMER}
