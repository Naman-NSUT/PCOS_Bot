"""
src/agents/report_agent.py — Report analyser subagent.

Delegated to by the conversation agent when someone sends photographs of a lab
report. It produces a verdict that draws on BOTH the bloodwork and what the
person has already said, which is the only way to reach the criteria a clinician
actually assesses — no blood test establishes irregular ovulation, and the
conversation cannot measure testosterone.

Pipeline, and note where the model is and is not allowed to act:

  photos
    -> transcribe        VISION. Transcription only, never interpretation.
    -> parse_report      DETERMINISTIC. Regex + 2023-guideline reference ranges,
                         plus derived LH/FSH and HOMA-IR. Reused unchanged.
    -> flag_indicators   DETERMINISTIC. Rule-based indicator flags. Unchanged.
    -> concordance       DETERMINISTIC. Cross-references flags against the
                         conversation's symptom list.
    -> CRAG              Retrieval, grounded in the tiered knowledge base.
    -> synthesize        MODEL. Narrates the concordance it was handed.

The two model steps sit at the ends — reading pixels and writing prose. Every
clinical judgement in between is code you can unit-test.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from config.settings import MEMORY_ENABLED
from src.core.session import Session
from src.memory import store as memory_store
from src.memory.models import Asserter, Predicate
from src.nodes.flag_indicators import flag_indicators_node
from src.nodes.parse_report import parse_report_node
from src.nodes.synthesize_verdict import build_concordance, synthesize_verdict
from src.nodes.transcribe_report import to_report_text, transcribe_images

logger = logging.getLogger(__name__)


def analyse_report(
    images: Sequence[bytes],
    session: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Analyse report photographs in the context of an ongoing consultation.

    `session` supplies the conversation half: its symptom_list and long-term
    memory snapshot. Passing None still works and yields a labs-only reading,
    with the gaps named honestly.
    """
    symptom_list: List[str] = list(session.symptom_list) if session else []

    # ── 1. Vision: pixels -> printed values ──────────────────────────────
    transcription = transcribe_images(images)
    if transcription.get("error") and not transcription["analytes"]:
        return _unreadable(transcription["error"])
    if not transcription["analytes"]:
        return _unreadable("no lab values were legible in the image(s)")

    report_text = to_report_text(transcription["analytes"])
    # Log the SHAPE, never the values. This line previously wrote transcribed
    # lab results ("Total Testosterone: 85 ng/dL, ...") straight into the
    # application log at INFO, putting clinical data in server logs.
    logger.info(
        "[report] transcribed %d analyte(s), %d note(s)",
        len(transcription["analytes"]), len(transcription["notes"]),
    )

    # ── 2-3. Deterministic parse + flags (the existing, tested engine) ───
    parsed = parse_report_node({"report_text": report_text})["parsed_values"]
    if not parsed:
        return _unreadable(
            "none of the values matched biomarkers this tool knows how to read"
        )

    flag_out = flag_indicators_node({"parsed_values": parsed})
    flags = flag_out.get("diagnostic_flags", {})
    crag_query = flag_out.get("crag_query", "PCOS diagnostic criteria and hormone levels")

    # ── 4. Cross-reference labs against the conversation ────────────────
    concordance = build_concordance(flags, symptom_list, transcription["notes"])

    # ── 5. Retrieval ─────────────────────────────────────────────────────
    crag_context, sources = _retrieve(crag_query, symptom_list)

    # ── 6. Narrate ───────────────────────────────────────────────────────
    memory_block = ""
    if session is not None and session.memory is not None:
        try:
            from src.memory.hydration import render_memory_block
            memory_block = render_memory_block(session.memory)
        except Exception as exc:
            logger.error("[report] memory block render failed: %s", exc)

    out = synthesize_verdict(
        parsed_values=parsed,
        diagnostic_flags=flags,
        concordance=concordance,
        symptom_list=symptom_list,
        crag_context=crag_context,
        memory_block=memory_block,
        report_notes=transcription["notes"],
    )

    _remember(session, parsed, flags)

    return {
        "ok": True,
        "verdict": out["verdict"],
        "disclaimer": out["disclaimer"],
        "parsed_values": parsed,
        "diagnostic_flags": flags,
        "concordance": concordance,
        "transcription_notes": transcription["notes"],
        "unreadable": transcription["unreadable"],
        "sources": sources,
    }


def _retrieve(query: str, symptom_list: List[str]) -> tuple[str, list]:
    """CRAG retrieval, widened with the conversation's symptoms. Never fatal."""
    if symptom_list:
        query = f"{query} | reported symptoms: {', '.join(symptom_list)}"
    try:
        from src.graphs.crag_graph import run_crag
        result = run_crag(query)
        return result.get("context", ""), result.get("sources", [])
    except Exception as exc:
        logger.error("[report] CRAG failed, continuing without context: %s", exc)
        return "", []


def _remember(session: Optional[Session], parsed: Dict[str, Any],
              flags: Dict[str, Any]) -> None:
    """
    Record that a report was analysed and which markers were out of range.

    Deliberately narrow: abnormal biomarker NAMES only, never values. It is
    enough for Maya to recall "your testosterone came back high last time" and
    ask what the doctor said — storing the numbers would put a durable clinical
    dataset in a store built for conversational continuity.

    Nothing here touches session.symptom_list: lab findings are not self-reported
    symptoms, and symptom_count gates the PCOS disclosure rules.
    """
    if session is None or not session.user_id or not MEMORY_ENABLED:
        return
    abnormal = [
        name for name, v in parsed.items()
        if isinstance(v, dict) and v.get("status") in ("high", "low", "critical_high", "critical_low")
    ]
    try:
        memory_store.record_facts(
            session.user_id, Predicate.SUGGESTED_TEST, abnormal[:8],
            session.session_id, asserter=Asserter.SYSTEM,
        )
        logger.info("[report] remembered %d abnormal marker(s)", len(abnormal))
    except Exception as exc:
        logger.error("[report] could not record report facts: %s", exc)


def _unreadable(reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "verdict": (
            "I couldn't read that clearly enough to be useful. Could you try a "
            "photo taken straight on, in good light, with the whole results table "
            "in frame? If it's a multi-page report, send each page."
        ),
        "reason": reason,
        "parsed_values": {}, "diagnostic_flags": {}, "concordance": {},
        "transcription_notes": [], "unreadable": [], "sources": [],
        "disclaimer": "",
    }
