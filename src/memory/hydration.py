"""
src/memory/hydration.py — Turn stored facts into a session-start snapshot.

Design rule that governs this whole module: retrieved memory is ADVISORY PROSE.
It never writes back into session.symptom_list, because symptom_count gates the
PCOS disclosure rules and the lab-test rules in llm_client.py. Letting a
remembered symptom raise that count would silently move a clinical threshold on
the strength of something the user has not said in this conversation.

So the snapshot carries what memory knows, the live Session carries what THIS
conversation has established, and the prompt keeps them in separate blocks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

from src.memory import store
from src.memory.models import Predicate

logger = logging.getLogger(__name__)

# Counters that must NEVER be hydrated. Each one is a per-session pacing control,
# and carrying it across sessions fires an instant closure on the returning
# user's first message:
#   total_exchanges >= 20      -> "natural" closure
#   topic_repeat_count >= 4    -> "natural" closure
#   symptom_count >= 8         -> "urgent" escalation, then an endless
#                                 welcome-back/escalate loop
NEVER_HYDRATE = frozenset({
    "total_exchanges", "phase_exchange_count", "topic_repeat_count",
    "symptom_list", "symptom_count", "covered_guidance_topics",
    "doctor_nudge_sent", "asked_questions", "history",
})


class MemorySnapshot(BaseModel):
    """Everything remembered about a user, loaded once at session start."""
    user_id: str
    is_returning: bool = False
    consultation_count: int = 0
    last_seen: Optional[datetime] = None

    preferred_name: Optional[str] = None
    age_range: Optional[str] = None
    gender: Optional[str] = None

    active_symptoms:   List[str] = Field(default_factory=list)
    resolved_symptoms: List[str] = Field(default_factory=list)
    dominant_concerns: List[str] = Field(default_factory=list)
    tests_suggested:   List[str] = Field(default_factory=list)
    guidance_covered:  List[str] = Field(default_factory=list)
    emotions_seen:     List[str] = Field(default_factory=list)

    pcos_previously_disclosed: bool = False
    doctor_visit_mentioned:    bool = False

    def is_empty(self) -> bool:
        return not self.is_returning


def _readable(tag: str) -> str:
    return tag.replace("_", " ")


def _days_since(when: Optional[datetime]) -> Optional[int]:
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0, (datetime.now(timezone.utc) - when).days)


def load_snapshot(user_id: str) -> MemorySnapshot:
    """Read the user's whole memory. One indexed query set, no LLM, no network."""
    try:
        user = store.get_user(user_id)          # read-only: never creates a row
        if user is None:
            return MemorySnapshot(user_id=user_id)
        facts = store.get_facts(user_id, include_resolved=True)
    except Exception as exc:
        # Memory is advisory: a store failure degrades personalisation, it must
        # never break the consultation.
        logger.error("[memory] snapshot load failed for %s: %s", user_id[:8], exc)
        return MemorySnapshot(user_id=user_id)

    live = [f for f in facts if not f["is_resolved"]]
    dead = [f for f in facts if f["is_resolved"]]

    def objs(rows, pred):
        return [r["object"] for r in rows if r["predicate"] == pred]

    active = objs(live, Predicate.REPORTED_SYMPTOM)
    snapshot = MemorySnapshot(
        user_id=user_id,
        # "Returning" means we have actually seen them before, not merely that a
        # row exists — get_or_create_user creates one on first sight.
        is_returning=bool(facts) or user["consultation_count"] > 0,
        consultation_count=user["consultation_count"],
        last_seen=user["last_seen_at"],
        preferred_name=user["preferred_name"],
        age_range=user["age_range"],
        gender=user["gender"],
        active_symptoms=active,
        resolved_symptoms=objs(dead, Predicate.REPORTED_SYMPTOM),
        dominant_concerns=[
            r["object"] for r in live
            if r["predicate"] == Predicate.REPORTED_SYMPTOM and r["occurrences"] > 1
        ][:4],
        tests_suggested=objs(live, Predicate.SUGGESTED_TEST)[:8],
        guidance_covered=objs(live, Predicate.RECEIVED_GUIDANCE),
        emotions_seen=objs(live, Predicate.EXPRESSED_EMOTION)[:3],
        pcos_previously_disclosed=bool(objs(live, Predicate.DISCLOSED_PCOS)),
        doctor_visit_mentioned=bool(objs(live, Predicate.MENTIONED_DOCTOR)),
    )
    return snapshot


def render_memory_block(snap: MemorySnapshot) -> str:
    """
    Render the snapshot as a labelled prompt block.

    Explicitly framed as PRIOR-session context so the model cannot mistake it for
    things said in this conversation — which would let it claim the user just
    told it something they told it last month.
    """
    if snap.is_empty():
        return ""

    lines: List[str] = [
        "LONG-TERM MEMORY (previous consultations — NOT said in this conversation):",
    ]

    days = _days_since(snap.last_seen)
    when = "recently" if days is None else (
        "earlier today" if days == 0 else
        "yesterday" if days == 1 else f"about {days} days ago"
    )
    lines.append(
        f"- You have spoken with this person {snap.consultation_count} time(s) before; last {when}."
    )

    if snap.preferred_name:
        lines.append(f"- They go by: {snap.preferred_name}")
    if snap.active_symptoms:
        lines.append(
            "- Symptoms they raised previously: "
            + ", ".join(_readable(s) for s in snap.active_symptoms)
        )
    if snap.dominant_concerns:
        lines.append(
            "- They keep returning to: "
            + ", ".join(_readable(s) for s in snap.dominant_concerns)
        )
    if snap.resolved_symptoms:
        lines.append(
            "- Previously reported but since RESOLVED (do not treat as current): "
            + ", ".join(_readable(s) for s in snap.resolved_symptoms)
        )
    if snap.guidance_covered:
        lines.append(
            "- Guidance already given (do not repeat verbatim; build on it): "
            + ", ".join(_readable(t) for t in snap.guidance_covered)
        )
    if snap.tests_suggested:
        lines.append(
            "- Lab tests already suggested (ask whether they got them rather than re-listing): "
            + ", ".join(snap.tests_suggested)
        )
    if snap.emotions_seen:
        lines.append(
            "- How they have tended to present emotionally: "
            + ", ".join(snap.emotions_seen)
        )
    if snap.pcos_previously_disclosed:
        lines.append("- PCOS has ALREADY been raised with them in a previous consultation.")
    if snap.doctor_visit_mentioned:
        lines.append("- They have mentioned seeing a doctor before.")

    lines.append(
        "HOW TO USE THIS: for continuity and warmth only. Do NOT assume any of it is "
        "still true — ask. Symptom state for THIS consultation comes only from what "
        "they tell you now."
    )
    return "\n".join(lines)


def welcome_back_greeting(snap: MemorySnapshot) -> str:
    """Opening line for a returning user, replacing the ask-for-your-name greeting."""
    name = snap.preferred_name or "there"
    if snap.active_symptoms:
        topic = ", ".join(_readable(s) for s in snap.active_symptoms[:2])
        return (
            f"Welcome back, {name}. Last time we talked about {topic}. "
            "How have things been since then?"
        )
    return f"Welcome back, {name}. How have you been since we last spoke?"
