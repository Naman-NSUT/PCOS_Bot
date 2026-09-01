"""
src/core/session.py — Session state management for Maya consultation flow.

Every conversation is a multi-phase clinical consultation. This module owns
the session dataclass, in-memory store, phase-transition logic, and all
state mutation helpers.

Phases — shaped like a real consultation, not an interview:

  1  OPENING     name + what brought them in. One or two turns, not three.
  2  HISTORY     focused, bundled questions. CAPPED, so it cannot meander.
  3  ASSESSMENT  Maya states her impression. Exactly one turn, and the only
                 turn that deliberately ends WITHOUT a question.
  4  PLAN        investigations to ask for, lifestyle measures, red flags.
  5  ONGOING     pre-therapy, post-therapy, prevention, follow-up.

The cap on HISTORY and the existence of ASSESSMENT are the whole design. A
doctor gathers what they need, then tells you what they think. The previous
model had no assessment turn at all, so the conversation could only ever be
question after question.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Phases ────────────────────────────────────────────────────────────────

PHASE_OPENING    = 1
PHASE_HISTORY    = 2
PHASE_ASSESSMENT = 3
PHASE_PLAN       = 4
PHASE_ONGOING    = 5

PHASE_NAMES = {
    PHASE_OPENING:    "opening",
    PHASE_HISTORY:    "history",
    PHASE_ASSESSMENT: "assessment",
    PHASE_PLAN:       "plan",
    PHASE_ONGOING:    "ongoing",
}

# History ends here whatever happens. A clinician does not keep taking history
# until the patient runs out of things to say; they gather enough and move on.
MAX_HISTORY_EXCHANGES = 6
# ...and can end as early as this once there is enough to form an impression.
MIN_HISTORY_EXCHANGES = 3
MIN_SYMPTOMS_FOR_ASSESSMENT = 2


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

    # Therapy context — drives pre- and post-therapy support in PHASE_ONGOING.
    treatments: List[str] = Field(default_factory=list)
    treatment_stage: Optional[str] = None   # considering | ongoing | adverse | stopped

    # Set once the assessment has been delivered, so it is never repeated and
    # the conversation can move on to the plan.
    assessment_given: bool = False
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
    Advance the consultation. Returns True if the phase changed.

      1 -> 2  a name, and any hint of why they came
      2 -> 3  enough history to form an impression, OR the history cap is hit
      3 -> 4  always — the assessment is exactly one turn
      4 -> 5  the plan has been laid out

    Note what is NOT here: no rule requires age and gender before moving on.
    Demanding them cost two turns of interrogation before anyone talked about
    health, and a returning user has them already. Ask when they matter.
    """
    old_phase = session.phase

    if session.phase == PHASE_OPENING:
        if session.user_profile.name is not None and session.phase_exchange_count >= 1:
            session.phase = PHASE_HISTORY
            session.phase_exchange_count = 0

    elif session.phase == PHASE_HISTORY:
        enough = (session.phase_exchange_count >= MIN_HISTORY_EXCHANGES
                  and session.symptom_count >= MIN_SYMPTOMS_FOR_ASSESSMENT)
        capped = session.phase_exchange_count >= MAX_HISTORY_EXCHANGES
        if enough or capped:
            session.phase = PHASE_ASSESSMENT
            session.phase_exchange_count = 0

    elif session.phase == PHASE_ASSESSMENT:
        # One turn only. It is delivered on entry, so once the user has replied
        # to it we are done and the plan follows.
        if session.assessment_given:
            session.phase = PHASE_PLAN
            session.phase_exchange_count = 0

    elif session.phase == PHASE_PLAN:
        if session.phase_exchange_count >= 3:
            session.phase = PHASE_ONGOING
            session.phase_exchange_count = 0

    return session.phase != old_phase


def update_treatments(session: Session, tags: List[str], stage: Optional[str]) -> None:
    """Merge newly mentioned therapies and update where they are with them."""
    for tag in tags:
        if tag not in session.treatments:
            session.treatments.append(tag)
    if stage:
        session.treatment_stage = stage


def jump_to_ongoing(session: Session) -> None:
    """
    Skip straight to ongoing support.

    Someone who opens with "I just started metformin, what should I expect?"
    does not need a history and an assessment first — they have a specific
    question and want it answered.
    """
    session.phase = PHASE_ONGOING
    session.phase_exchange_count = 0
    session.assessment_given = True


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
