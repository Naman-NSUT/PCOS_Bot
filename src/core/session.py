"""
src/core/session.py — Session state management for Maya consultation flow.

Every conversation is a multi-phase clinical consultation. This module owns
the session dataclass, in-memory store, phase-transition logic, and all
state mutation helpers.

Phases:
  1  First Contact   — learn name, build safety
  2  Intake          — collect age, gender, reason for visit
  3  Symptom Explore — focused clinical conversation
  4  Guidance        — personalised advice and ongoing support
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Data Models ───────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    name: Optional[str] = None
    age_range: Optional[str] = None
    gender: Optional[str] = None
    general_feeling: Optional[str] = None
    reason_for_visit: Optional[str] = None
    lifestyle_notes: List[str] = Field(default_factory=list)


class Turn(BaseModel):
    role: str          # "user" or "assistant"
    content: str


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    # Durable cross-session identity. None means an anonymous, unremembered
    # consultation — the app must work exactly as before in that case.
    user_id: Optional[str] = None
    is_returning_user: bool = False

    # Read-only view of long-term memory, loaded once at session start.
    # Deliberately NOT merged into the fields below: symptom_count gates the PCOS
    # disclosure and lab-test rules, so a remembered symptom must never raise it.
    memory: Optional[Any] = None
    pcos_disclosed_previously: bool = False

    phase: int = 1
    phase_exchange_count: int = 0
    total_exchanges: int = 0

    user_profile: UserProfile = Field(default_factory=UserProfile)

    symptom_list: List[str] = Field(default_factory=list)
    symptom_count: int = 0
    emotional_state: Optional[str] = None
    last_topic: Optional[str] = None
    pending_question: Optional[str] = None
    pcos_mentioned: bool = False
    asked_questions: List[str] = Field(default_factory=list)

    # ── Closure / escalation fields ───────────────────────────────────
    closure_triggered: bool = False
    closure_mode: Optional[str] = None   # "natural", "urgent", "user_initiated"
    is_closed: bool = False
    topic_repeat_count: int = 0
    covered_guidance_topics: List[str] = Field(default_factory=list)
    doctor_nudge_sent: bool = False
    previous_session_summary: Optional[str] = None  # set on restart
    previous_session_name: Optional[str] = None      # set on restart

    history: List[Turn] = Field(default_factory=list)


# ── In-Memory Store ──────────────────────────────────────────────────────

_store: Dict[str, Session] = {}


def create_session(user_id: Optional[str] = None) -> Session:
    """Create a new session and persist it in the store."""
    session = Session(user_id=user_id)
    _store[session.session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    """Retrieve a session by ID.  Returns None if not found."""
    return _store.get(session_id)


def get_or_create_session(session_id: Optional[str] = None) -> Session:
    """Get existing session or create a new one."""
    if session_id and session_id in _store:
        return _store[session_id]
    return create_session()


def save_session(session: Session) -> None:
    """Persist updated session back to the store."""
    _store[session.session_id] = session


# ── Turn Helpers ─────────────────────────────────────────────────────────

def add_turn(session: Session, role: str, content: str) -> None:
    """Append a turn and bump exchange counters."""
    session.history.append(Turn(role=role, content=content))
    if role == "user":
        session.phase_exchange_count += 1
        session.total_exchanges += 1


# ── Symptom Helpers ──────────────────────────────────────────────────────

def update_symptoms(session: Session, new_tags: List[str]) -> None:
    """Merge new symptom tags (deduplicated) and refresh the count."""
    for tag in new_tags:
        if tag not in session.symptom_list:
            session.symptom_list.append(tag)
    session.symptom_count = len(session.symptom_list)


def resolve_symptom(session: Session, tag: str) -> bool:
    """
    Drop a symptom from the live session because the user says it no longer
    applies. Returns True if it was present.

    Its long-term counterpart is store.resolve_fact(), which sets invalid_at so
    the symptom stops being current without erasing that it once was.
    """
    if tag in session.symptom_list:
        session.symptom_list.remove(tag)
        session.symptom_count = len(session.symptom_list)
        return True
    return False


def pcos_threshold_reached(session: Session) -> bool:
    """Returns True when ≥ 3 unique symptom tags have been recorded."""
    return session.symptom_count >= 3


# ── Phase Transition ─────────────────────────────────────────────────────

def check_phase_transition(session: Session) -> bool:
    """
    Evaluate whether the session should advance to the next phase.
    Returns True if a transition occurred.

    Transition rules:
      1 → 2 : name collected
      2 → 3 : name + age_range + gender all present
      3 → 4 : phase_exchange_count >= 4 AND symptom_count >= 1
      4 → 4 : no further transitions
    """
    old_phase = session.phase

    if session.phase == 1:
        if session.user_profile.name is not None:
            session.phase = 2
            session.phase_exchange_count = 0

    elif session.phase == 2:
        if (session.user_profile.name is not None
                and session.user_profile.age_range is not None
                and session.user_profile.gender is not None):
            session.phase = 3
            session.phase_exchange_count = 0

    elif session.phase == 3:
        if session.phase_exchange_count >= 4 and session.symptom_count >= 1:
            session.phase = 4
            session.phase_exchange_count = 0

    # Phase 4 stays at 4 forever.
    return session.phase != old_phase


def reset_store() -> None:
    """Clear all sessions (useful for testing)."""
    _store.clear()


def purge_user_sessions(user_id: str) -> int:
    """
    Drop every in-memory session belonging to a user.

    Erasure that only deletes database rows is incomplete: a live session still
    holds the person's name, symptoms and full transcript in RAM and would keep
    serving them until the process restarts.
    """
    doomed = [sid for sid, s in _store.items() if s.user_id == user_id]
    for sid in doomed:
        del _store[sid]
    return len(doomed)


# ── Closure Helpers ──────────────────────────────────────────────────────

def mark_closed(session: Session, mode: str) -> None:
    """
    Mark a session as closed with the given closure mode.
    mode: "natural" | "urgent" | "user_initiated"
    """
    session.closure_triggered = True
    session.closure_mode = mode
    session.is_closed = True


def update_topic_repeat(session: Session, current_topic: Optional[str]) -> None:
    """
    Track consecutive turns on the same topic.
    Resets to 1 whenever the topic changes.
    """
    if current_topic and current_topic == session.last_topic:
        session.topic_repeat_count += 1
    else:
        session.topic_repeat_count = 1


def add_covered_guidance(session: Session, topic: str) -> None:
    """Record that guidance has been given on a specific topic."""
    if topic not in session.covered_guidance_topics:
        session.covered_guidance_topics.append(topic)
