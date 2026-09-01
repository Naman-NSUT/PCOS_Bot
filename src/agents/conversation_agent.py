"""
src/agents/conversation_agent.py — Phase-driven consultation orchestrator.

This is the "brain" of Maya.  Called once per user message, it:
  1. Loads session state
  2. Handles closed-session restart
  3. Runs intent detection (name, age, gender, symptoms, emotion)
  4. Checks closure / escalation conditions
  5. Checks phase transitions
  6. Decides whether to call CRAG
  7. Builds the dynamic system prompt
  8. Generates the response via the configured LLM
  9. Updates session state
  10. Returns the response + metadata

No direct LLM logic lives here — it delegates to llm_client and
intent_detector.  No RAG/CRAG code is imported at module level; the run_crag
function is imported lazily only when needed.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from config.settings import MEMORY_ENABLED
from src.memory import store as memory_store
from src.memory.hydration import (
    MemorySnapshot,
    load_snapshot,
    welcome_back_greeting,
)
from src.memory.models import Asserter, Predicate
from src.core.session import (
    Session,
    add_covered_guidance,
    add_turn,
    check_phase_transition,
    create_session,
    get_or_create_session,
    get_session,
    mark_closed,
    pcos_threshold_reached,
    resolve_symptom,
    save_session,
    update_symptoms,
    update_topic_repeat,
)
from src.core.intent_detector import (
    detect_emotion,
    detect_resolved_symptoms,
    detect_symptoms,
    extract_age,
    extract_gender,
    extract_name,
)
from src.core.llm_client import (
    build_system_prompt,
    build_user_prompt,
    generate_response,
)

logger = logging.getLogger(__name__)


# ── Greeting (bot speaks first) ──────────────────────────────────────────

_GREETING = (
    "Hi there, I'm Maya — a health assistant here to listen and help however I can. "
    "What's your name?"
)


def get_greeting(user_id: Optional[str] = None) -> dict:
    """
    Start a new consultation and return Maya's opening line.
    No user message needed — the bot speaks first.

    For a known user the whole long-term memory is loaded HERE, before any turn
    is committed, because the greeting itself differs: a returning person must
    not be asked their name again.
    """
    session = create_session(user_id=user_id)
    greeting = _GREETING

    snapshot = _hydrate(session)
    if snapshot is not None and snapshot.is_returning:
        greeting = welcome_back_greeting(snapshot)
        session.pending_question = "How have things been since we last spoke?"
    else:
        session.pending_question = "What is your name?"

    add_turn(session, "assistant", greeting)
    save_session(session)

    return {
        "answer": greeting,
        "sources": [],
        "session_id": session.session_id,
        "phase": session.phase,
    }


def _hydrate(session: Session) -> Optional[MemorySnapshot]:
    """
    Load long-term memory into a session.

    What is hydrated: stable profile (name/age/gender), whether PCOS was already
    disclosed, and the advisory snapshot.

    What is deliberately NOT hydrated, and why — every one of these was verified
    to break a returning user on their first message:
      symptom_list / symptom_count  -> symptom_count >= 8 returns "urgent" from
                                       _check_closure_conditions, closing the
                                       session on "hi" and then looping forever
      total_exchanges               -> >= 20 returns "natural" closure
      topic_repeat_count            -> >= 4 returns "natural" closure
      covered_guidance_topics       -> grows across sessions while symptom_list
                                       does not, so the phase-4 completion check
                                       len(covered) >= len(symptoms) becomes
                                       permanently true and every future session
                                       self-terminates after 5 exchanges
      doctor_nudge_sent             -> either nudges on turn 1 or never again
      reason_for_visit              -> is per-visit; carrying it means this
                                       month's consultation is framed by last
                                       month's reason, forever
    """
    if not session.user_id or not MEMORY_ENABLED:
        return None

    try:
        snapshot = load_snapshot(session.user_id)
    except Exception as exc:
        logger.error("[agent] memory hydration failed: %s", exc)
        return None

    session.memory = snapshot
    if not snapshot.is_returning:
        return snapshot

    session.is_returning_user = True
    session.pcos_disclosed_previously = snapshot.pcos_previously_disclosed

    profile = session.user_profile
    profile.name = snapshot.preferred_name
    profile.age_range = snapshot.age_range
    profile.gender = snapshot.gender
    # reason_for_visit stays None on purpose — see the docstring.

    # Skip intake for someone we already know. Phase 1 and 2 exist only to
    # collect name/age/gender, and check_phase_transition would clear them
    # immediately anyway; jumping avoids re-asking questions we have answers to.
    if profile.name:
        session.phase = 3
        session.phase_exchange_count = 0

    logger.info(
        "[agent] hydrated returning user %s — %d prior consultation(s)",
        session.user_id[:8], snapshot.consultation_count,
    )
    return snapshot


# ═══════════════════════════════════════════════════════════════════════════
# CLOSURE / ESCALATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

_CLOSING_INTENTS = [
    "thank you", "thanks", "thank u", "thx",
    "that's helpful", "thats helpful",
    "okay got it", "ok got it", "okay i got it",
    "i think i understand", "i understand now",
    "thanks so much", "thank you so much",
    "that's all", "thats all", "that is all",
    "i'm good", "im good", "i am good",
    "nothing else", "no more questions",
    "bye", "goodbye", "good bye", "see you",
    "take care",
]

_URGENT_PHYSICAL = [
    "severe pain", "extreme pain", "unbearable pain",
    "chest pain", "difficulty breathing", "can't breathe",
    "cant breathe", "hard to breathe",
    "passing out", "fainted", "fainting", "blacking out",
    "heavy bleeding", "hemorrhage", "bleeding heavily",
]

_URGENT_PERIOD_PATTERNS = [
    re.compile(
        r"(?:haven.t|have not|no|missed|without).{0,30}"
        r"period.{0,20}(?:\d+\s*months?|three\s*months|[3-9]\s*months|half\s*a?\s*year|year)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:\d+|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*months?\s*"
        r"(?:without|no|missed|since).{0,20}period",
        re.IGNORECASE,
    ),
]

_URGENT_EMOTIONAL_CRISIS = [
    "suicid", "kill myself", "end my life", "want to die",
    "don't want to live", "dont want to live",
    "self harm", "self-harm", "cutting myself",
    "hurting myself",
]


def _detect_closing_intent(text: str) -> bool:
    """Check if the user is signaling they're done."""
    lower = text.lower().strip().rstrip("!.")
    for phrase in _CLOSING_INTENTS:
        if phrase in lower:
            return True
    return False


def _detect_urgent_escalation(text: str) -> Optional[str]:
    """
    Check for urgent escalation triggers.
    Returns the escalation category or None.
    Categories: "severe_physical", "prolonged_amenorrhea", "emotional_crisis"
    """
    lower = text.lower()

    # Emotional crisis — check first, highest priority
    for kw in _URGENT_EMOTIONAL_CRISIS:
        if kw in lower:
            return "emotional_crisis"

    # Severe physical symptoms
    for kw in _URGENT_PHYSICAL:
        if kw in lower:
            return "severe_physical"

    # Prolonged missed periods
    for pattern in _URGENT_PERIOD_PATTERNS:
        if pattern.search(text):
            return "prolonged_amenorrhea"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# CLOSURE CONDITION CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def _check_closure_conditions(session: Session, user_message: str) -> Optional[str]:
    """
    Evaluate all closure conditions. Returns the closure mode if triggered,
    or None if the conversation should continue.

    Conditions (checked in priority order):
      5 → urgent escalation              → "urgent"
      4 → user signals done              → "user_initiated"
      1 → total_exchanges >= 20          → "natural"
      2 → topic_repeat_count >= 4        → "natural"
      3 → Phase 4, 5+ exchanges, topics  → "natural"
    """
    # Condition 5: urgent escalation
    escalation = _detect_urgent_escalation(user_message)
    if escalation:
        session.emotional_state = escalation  # store for prompt
        return "urgent"

    # Only check closure in Phase 3+ (don't close during intake)
    if session.phase < 3:
        return None

    # Condition 4: user signals done
    if _detect_closing_intent(user_message):
        return "user_initiated"

    # Condition 1: exchange limit
    if session.total_exchanges >= 20:
        return "natural"

    # Condition 2: topic saturation
    if session.topic_repeat_count >= 4:
        return "natural"

    # Condition 3: guidance complete
    if (session.phase == 4
            and session.phase_exchange_count >= 5
            and len(session.covered_guidance_topics) >= len(session.symptom_list)
            and len(session.covered_guidance_topics) >= 1):
        return "natural"

    # Condition 5b: high symptom count without doctor visit
    if session.symptom_count >= 8:
        return "urgent"

    return None


# ═══════════════════════════════════════════════════════════════════════════
# CLOSURE RESPONSE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _generate_closure_response(session: Session, mode: str) -> str:
    """
    Build a closing response using session data.
    Uses the LLM with a closure-specific system prompt.
    """
    name = session.user_profile.name or "there"
    symptoms_str = ", ".join(
        s.replace("_", " ") for s in session.symptom_list[:5]
    ) if session.symptom_list else "what you've been experiencing"
    topics_str = ", ".join(
        t.replace("_", " ") for t in session.covered_guidance_topics[:4]
    ) if session.covered_guidance_topics else "your symptoms and wellbeing"

    if mode == "urgent":
        return _build_urgent_response(session)

    # Natural or user-initiated closure
    test_list = _build_closing_test_list(session)
    test_section = ""
    if test_list:
        test_section = (
            f"\n4. WHAT TO TELL THE DOCTOR AND TESTS TO ASK ABOUT (2-3 sentences): "
            f"Give 2-3 specific things to mention based on their symptoms: {symptoms_str}. "
            f"Include these specific tests to ask about, woven into the advice naturally:\n"
            f"{test_list}\n"
        )
    else:
        test_section = (
            f"\n4. WHAT TO TELL THE DOCTOR (1-2 sentences): Give 2-3 specific things to mention "
            f"based on their symptoms: {symptoms_str}. Suggest a relevant test to ask about.\n"
        )

    closure_prompt = (
        "You are Maya. You are closing this consultation warmly and completely.\n"
        "Write a closing response with these exact sections, in this order:\n\n"
        f"1. SUMMARY (2-3 sentences): Warmly summarise what was discussed. "
        f"Use the name '{name}'. Reference these symptoms: {symptoms_str}. "
        f"Reference these topics covered: {topics_str}.\n\n"
        "2. KEY TAKEAWAY (1 sentence): The single most important thing from today.\n\n"
        "3. DOCTOR REFERRAL (2 sentences): A warm, empowering recommendation to see a "
        "healthcare professional. Frame it as empowerment, not alarm. End with: "
        "'You now know exactly what to ask for.'\n"
        f"{test_section}\n"
        "5. CLOSING OFFER (1 sentence): One warm line offering to help in the future.\n\n"
        "RULES:\n"
        "- Do NOT end with a question. This is the only response that doesn't ask a question.\n"
        "- Do NOT use markdown formatting like bold or italic.\n"
        "- For the test recommendations, use a short structured list — this is the one place where a brief list is acceptable.\n"
        "- Frame tests as 'worth asking your doctor about', never as prescriptions.\n"
        "- Write warmly and personally.\n"
        "- Do NOT say 'As an AI'.\n"
    )

    return generate_response(closure_prompt, f"Close the conversation for {name}.")


def _build_urgent_response(session: Session) -> str:
    """Build an urgent escalation response based on the escalation category."""
    name = session.user_profile.name or "there"
    category = session.emotional_state  # stored during detection

    if category == "emotional_crisis":
        return (
            f"{name}, it sounds like you're going through a really hard time right now. "
            "Your mental health matters just as much as your physical health. "
            "Please reach out to someone you trust today — a friend, a family member, "
            "or a mental health professional. You do not have to carry this alone."
        )

    if category == "prolonged_amenorrhea":
        return (
            f"{name}, going without a period for that long is something your body is "
            "signalling needs attention. Please make an appointment with a gynaecologist "
            "or your GP this week if you can — they'll be able to run the right tests "
            "and figure out what's going on."
        )

    # severe_physical or default
    symptoms_str = ", ".join(
        s.replace("_", " ") for s in session.symptom_list[:3]
    ) if session.symptom_list else "what you're describing"
    return (
        f"{name}, what you're describing sounds like something that needs to be seen by "
        "a doctor soon — not because I want to alarm you, but because this is something "
        "that should be assessed properly and quickly. Please reach out to your doctor "
        "or visit urgent care as soon as you can."
    )


# ═══════════════════════════════════════════════════════════════════════════
# SESSION RESTART
# ═══════════════════════════════════════════════════════════════════════════

def _tests_named_in(answer: str, symptom_list: List[str]) -> List[str]:
    """Which of the candidate lab tests were actually named in this reply."""
    if not answer:
        return []
    lower = answer.lower()
    return [
        test for test in _get_relevant_tests(symptom_list, limit=8)
        if test.split(" (")[0].lower() in lower
    ]


def _persist_memory(session: Session, closing: bool = False,
                    answer: str = "") -> None:
    """
    Write this turn's structured facts to long-term memory.

    Zero LLM calls: every value written here was already extracted
    deterministically by the regex intent detector or looked up in a table. The
    fact extraction hosted memory vendors bill for is work this app has already
    done for free.

    Failures are swallowed by design — memory is advisory, and a store problem
    must never break a live consultation.
    """
    if not session.user_id or not MEMORY_ENABLED:
        return

    try:
        # Open the consultation record lazily: at greeting time there may be no
        # user row yet, and a visitor who never says anything should not create
        # one just by loading the page.
        memory_store.start_consultation(session.user_id, session.session_id)

        named_tests = _tests_named_in(answer, session.symptom_list)
        p = session.user_profile
        memory_store.update_profile(
            session.user_id, name=p.name, age_range=p.age_range, gender=p.gender,
        )

        cid = session.session_id
        memory_store.record_facts(
            session.user_id, Predicate.REPORTED_SYMPTOM, session.symptom_list, cid)
        # System-authored: Maya gave the guidance, the user did not report it.
        memory_store.record_facts(
            session.user_id, Predicate.RECEIVED_GUIDANCE,
            session.covered_guidance_topics, cid, asserter=Asserter.SYSTEM)
        if session.emotional_state:
            memory_store.record_fact(
                session.user_id, Predicate.EXPRESSED_EMOTION,
                session.emotional_state, cid)

        # DISCLOSED_PCOS records that PCOS was genuinely raised WITH the user.
        # It is not enough for the word to appear in a reply: under the
        # never-mention rule the model still says things like "I can't tell you
        # whether this is PCOS", and recording that as a disclosure makes the
        # NEXT session take the permissive "already disclosed" branch of the
        # gate at symptom_count 0 — memory manufacturing evidence out of a
        # refusal. Only count it once the gate actually permitted disclosure.
        if session.pcos_disclosed_previously or (
            session.pcos_mentioned and session.symptom_count >= 3
        ):
            memory_store.record_fact(
                session.user_id, Predicate.DISCLOSED_PCOS, "pcos", cid)

        # SUGGESTED_TEST likewise records what was actually SAID, not what the
        # symptom profile would have justified saying. Writing the ranked list
        # on symptom_count alone made the next session claim five specific tests
        # had been recommended when none were ever named.
        if named_tests:
            memory_store.record_facts(
                session.user_id, Predicate.SUGGESTED_TEST, named_tests, cid,
                asserter=Asserter.SYSTEM)

        if closing:
            memory_store.finish_consultation(
                cid,
                phase_reached=session.phase,
                exchange_count=session.total_exchanges,
                closure_mode=session.closure_mode,
                summary=", ".join(session.symptom_list) or None,
            )
    except Exception as exc:
        logger.error("[agent] memory persist failed: %s", exc)


def _handle_closed_session(old_session: Session,
                           user_id: Optional[str] = None) -> dict:
    """
    When a user messages into a closed session, create a new one
    and generate a warm welcome-back greeting referencing the previous session.
    """
    # Only reference the closed session's content when the caller owns it.
    caller_owns = (user_id or None) == (old_session.user_id or None)
    name = (old_session.user_profile.name if caller_owns else None) or "there"
    symptoms_str = ", ".join(
        s.replace("_", " ") for s in old_session.symptom_list[:3]
    ) if (caller_owns and old_session.symptom_list) else "how you were feeling"

    new_session = create_session(user_id=user_id or old_session.user_id)
    new_session.previous_session_name = name
    new_session.previous_session_summary = symptoms_str

    # Long-term memory takes precedence when we have it. The profile is copied
    # off the closed session ONLY when the same user owns both — otherwise this
    # path hands one person's name and symptoms to another, with no LLM in the
    # loop for any prompt rule to prevent it.
    _hydrate(new_session)
    same_owner = (old_session.user_id or None) == (new_session.user_id or None)
    if not new_session.user_profile.name and same_owner:
        new_session.user_profile = old_session.user_profile.model_copy()
    new_session.phase = 3  # jump to symptom exploration
    new_session.phase_exchange_count = 0

    restart_msg = (
        f"Welcome back, {name}. Last time we spoke about {symptoms_str}. "
        "Would you like to pick up where we left off or start fresh?"
    )
    add_turn(new_session, "assistant", restart_msg)
    new_session.pending_question = "Pick up where we left off or start fresh?"
    save_session(new_session)

    return {
        "answer": restart_msg,
        "sources": [],
        "session_id": new_session.session_id,
        "phase": new_session.phase,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROACTIVE DOCTOR NUDGE
# ═══════════════════════════════════════════════════════════════════════════

_DOCTOR_NUDGE = (
    " I also want to say — at some point it would be really worth "
    "taking what you've been sharing with me to an actual doctor. "
    "Everything we discuss here is helpful context, but a proper "
    "examination and some blood tests will give you real answers."
)


def _should_nudge_doctor(session: Session) -> bool:
    """
    Check if we should insert a proactive doctor referral nudge.

    `total_exchanges` is per-session and never hydrated, so this cannot fire on
    turn 1 of a returning session. The extra memory check stops us delivering the
    same 40-word "go see a doctor" paragraph in every consultation to someone who
    has already told us they saw one.
    """
    if session.total_exchanges < 12 or session.doctor_nudge_sent or session.phase < 3:
        return False
    snap = session.memory
    if snap is not None and getattr(snap, "doctor_visit_mentioned", False):
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════
# LAB TEST SUGGESTION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

# Symptom tag → relevant lab tests (ordered by priority)
_SYMPTOM_TO_TESTS: dict[str, List[str]] = {
    "irregular_periods": ["LH", "FSH", "Progesterone (mid-luteal)", "AMH", "Prolactin"],
    "hirsutism":         ["Total Testosterone", "Free Testosterone", "SHBG", "DHEA-S", "17-OHP"],
    "acne":              ["Total Testosterone", "Free Testosterone", "SHBG"],
    "weight_gain":       ["Fasting Glucose", "Fasting Insulin", "HOMA-IR", "HbA1c", "Lipid Panel", "TSH"],
    "fatigue":           ["TSH", "Free T3/T4", "Vitamin D", "CBC", "Fasting Insulin"],
    "fertility_concerns":["AMH", "LH", "FSH", "Estradiol", "Progesterone", "Prolactin", "Pelvic Ultrasound"],
    "mood_disturbance":  ["TSH", "Vitamin D", "Fasting Insulin"],
    "hair_loss":         ["Total Testosterone", "Free Testosterone", "SHBG", "TSH", "Prolactin", "Ferritin/CBC"],
    "bloating":          ["Fasting Glucose", "Fasting Insulin", "Liver Enzymes (AST/ALT)"],
    "acanthosis":        ["Fasting Insulin", "HOMA-IR", "Fasting Glucose", "HbA1c", "Lipid Panel"],
    "insulin_related":   ["Fasting Glucose", "Fasting Insulin", "HOMA-IR", "HbA1c", "OGTT", "Lipid Panel"],
    "sleep_disturbance": ["TSH", "Vitamin D", "Fasting Insulin"],
    "headaches":         ["TSH", "Prolactin"],
    "pelvic_pain":       ["Pelvic Ultrasound", "Progesterone", "hCG"],
}

# Phrases that indicate the user is asking about tests
_TEST_REQUEST_KEYWORDS = [
    "what test", "which test", "blood test", "blood work",
    "lab test", "lab work", "get tested", "should i test",
    "should i get", "tests should i", "what to ask for",
    "what should i ask", "investigations", "what tests",
    "run some tests", "doctor wants to", "what to expect",
]


def _detect_test_request(text: str) -> bool:
    """Check if the user is asking about lab tests."""
    lower = text.lower()
    return any(kw in lower for kw in _TEST_REQUEST_KEYWORDS)


def _get_relevant_tests(symptom_list: List[str], limit: int = 5) -> List[str]:
    """
    Get deduplicated, prioritized list of tests based on symptoms.
    Tests that appear for multiple symptoms get higher priority.
    """
    test_scores: dict[str, int] = {}
    for tag in symptom_list:
        tests = _SYMPTOM_TO_TESTS.get(tag, [])
        for i, test in enumerate(tests):
            # Score: earlier in list = higher priority, each symptom adds +1
            score = len(tests) - i
            test_scores[test] = test_scores.get(test, 0) + score

    # Sort by score descending, take top N
    ranked = sorted(test_scores.items(), key=lambda x: x[1], reverse=True)
    return [test for test, _ in ranked[:limit]]


def _build_closing_test_list(session: Session) -> str:
    """
    Build a short test recommendation list for the closing response.
    Picks top 3 tests with one-line symptom-based reasons.
    """
    tests = _get_relevant_tests(session.symptom_list, limit=3)
    if not tests:
        return ""

    # Build simple reason based on which symptom maps to each test
    lines = []
    for test in tests:
        # Find which of the user's symptoms this test is relevant to
        reasons = []
        for tag in session.symptom_list:
            if test in _SYMPTOM_TO_TESTS.get(tag, []):
                reasons.append(tag.replace("_", " "))
        reason_str = " and ".join(reasons[:2]) if reasons else "your symptoms"
        lines.append(f"{test} — because of the {reason_str} you described")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TURN PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def process_turn(session_id: str, user_message: str,
                 user_id: Optional[str] = None) -> dict:
    """
    Process a single user message within an existing session.

    Turn pipeline:
      1.  Load session (handle closed sessions with restart)
      2.  Record user turn
      3.  Detect emotion → update emotional_state
      4.  Check urgent escalation + closure conditions
      5.  Phase 1-2: extract name / age / gender
      6.  All phases: detect symptoms → update symptom list
      7.  Update topic repetition tracking
      8.  Check PCOS threshold
      9.  Check phase transition
      10. Decide CRAG retrieval
      11. Build system + user prompts
      12. Generate LLM response (or closure/escalation response)
      13. Apply doctor nudge if applicable
      14. Record assistant turn
      15. Update pending_question / last_topic / covered_guidance
      16. Persist session
      17. Return response payload
    """
    # 1. Load session — handle restart if closed
    session = get_session(session_id)

    # OWNERSHIP CHECK. session_id was previously the only credential: anyone who
    # knew it could read a stranger's whole consultation (name, symptoms, verbatim
    # messages) and, worse, have their own turns written into that stranger's
    # permanent record — because _persist_memory keys writes off session.user_id.
    # A session id is not a secret: it is returned in every response body, logged,
    # and stored as consultation_id on every row. Treat a mismatch as an unknown
    # session rather than serving it.
    if session is not None and session.user_id and session.user_id != user_id:
        logger.warning(
            "[agent] rejected session %s — owner mismatch", session_id[:8]
        )
        session = None

    if session is None:
        # Unknown session id: the process restarted, or the client sent a stale
        # id. Previously this minted a blank Phase-1 session, so the bot abruptly
        # re-asked for a name mid-conversation and long-term memory appeared to
        # evaporate on every deploy. Rebuild from durable identity instead.
        session = create_session(user_id=user_id)
        _hydrate(session)
    elif session.is_closed:
        return _handle_closed_session(session, user_id=user_id)

    # Late-binding: a client that only learns its user_id mid-conversation still
    # gets its memory attached rather than silently losing the session's writes.
    if user_id and not session.user_id:
        session.user_id = user_id
        _hydrate(session)

    # 2. Record user turn
    add_turn(session, "user", user_message)

    # 3. Detect emotion
    emotion = detect_emotion(user_message)
    if emotion:
        session.emotional_state = emotion

    # 4. Check closure / escalation conditions
    closure_mode = _check_closure_conditions(session, user_message)
    if closure_mode:
        answer = _generate_closure_response(session, closure_mode)
        mark_closed(session, closure_mode)
        add_turn(session, "assistant", answer)
        save_session(session)
        _persist_memory(session, closing=True, answer=answer)
        return {
            "answer": answer,
            "sources": [],
            "session_id": session.session_id,
            "phase": session.phase,
        }

    # 5. Phase-specific extraction
    _run_phase_extraction(session, user_message)

    # 6. Detect symptoms (all phases)
    new_tags = detect_symptoms(user_message)

    # 6b. Detect symptoms the user says have STOPPED, and retire them.
    #
    # This is the point of tracking two timelines. Without it, "my periods are
    # regular again" would be stored next to "irregular periods" and both would
    # stay live forever, letting a resolved symptom resurface months later and
    # move the symptom_count thresholds that gate PCOS disclosure.
    resolved_tags = detect_resolved_symptoms(user_message)
    for tag in resolved_tags:
        resolve_symptom(session, tag)          # drop from the live session
        if tag in new_tags:
            # The same message both named and retired it, e.g. "my irregular
            # periods are regular again" — the mention is history, not a report.
            new_tags.remove(tag)
        if session.user_id and MEMORY_ENABLED:
            try:
                memory_store.resolve_fact(session.user_id, Predicate.REPORTED_SYMPTOM, tag)
                logger.info("[agent] retired resolved symptom: %s", tag)
                # Refresh the cached snapshot. The memory block is rendered every
                # turn from a snapshot taken at session start, so without this a
                # symptom the user just told us had cleared would keep appearing
                # as active in the prompt for the rest of the conversation.
                session.memory = load_snapshot(session.user_id)
            except Exception as exc:
                logger.error("[agent] could not retire %s: %s", tag, exc)

    if new_tags:
        update_symptoms(session, new_tags)

    # 7. Update topic repetition tracking
    current_topic = (
        new_tags[-1] if new_tags
        else (session.symptom_list[-1] if session.symptom_list else None)
    )
    update_topic_repeat(session, current_topic)

    # 8. PCOS threshold is checked via session.symptom_count in the prompt builder

    # 9. Check phase transition
    transitioned = check_phase_transition(session)
    if transitioned:
        logger.info(
            "[agent] Session %s transitioned to Phase %d",
            session.session_id[:8],
            session.phase,
        )

    # 10. Decide CRAG retrieval
    crag_context = ""
    sources: list = []
    if _should_use_crag(session, user_message):
        crag_context, sources = _run_crag(session, user_message)

    # 11. Build prompts
    system_prompt = build_system_prompt(session)
    user_prompt = build_user_prompt(session, user_message, crag_context)

    # 12. Generate response
    answer = generate_response(system_prompt, user_prompt)

    # 13. Check and update pcos_mentioned
    if not session.pcos_mentioned:
        if _mentions_pcos(answer):
            session.pcos_mentioned = True

    # 14. Apply doctor nudge if applicable
    if _should_nudge_doctor(session):
        answer += _DOCTOR_NUDGE
        session.doctor_nudge_sent = True

    # 15. Record assistant turn
    add_turn(session, "assistant", answer)

    # 16. Update tracking fields
    _update_tracking(session, user_message, answer)

    # 17. Persist — in-process first, then long-term.
    # Write through on EVERY turn, not only at closure: most real consultations
    # end by abandonment, which fires no closure signal at all, so a
    # persist-on-close design would silently lose them.
    save_session(session)
    _persist_memory(session, answer=answer)

    return {
        "answer": answer,
        "sources": sources,
        "session_id": session.session_id,
        "phase": session.phase,
    }


def record_report_turn(session_id: Optional[str], images, 
                       user_id: Optional[str] = None) -> dict:
    """
    Handle report photographs as a turn in the consultation.

    Delegates the analysis to the report subagent, then records the exchange in
    the session so the rest of the conversation can refer back to it — "you
    mentioned your testosterone came back high" only works if the verdict is part
    of the transcript.

    Ownership is checked exactly as it is for a text turn: a session_id is not a
    credential, and uploading into someone else's consultation must not be
    possible.
    """
    from src.agents.report_agent import analyse_report

    session = get_session(session_id) if session_id else None
    if session is not None and session.user_id and session.user_id != user_id:
        logger.warning("[agent] rejected report upload — session owner mismatch")
        session = None
    if session is not None and session.is_closed:
        session = None

    if session is None:
        session = create_session(user_id=user_id)
        _hydrate(session)

    add_turn(session, "user", "[sent photographs of a lab report]")

    result = analyse_report(images, session=session)

    answer = result["verdict"]
    if result.get("ok") and result.get("disclaimer"):
        answer = f"{answer}\n\n{result['disclaimer']}"

    add_turn(session, "assistant", answer)

    # A report is substantive clinical content: move past intake if we are still
    # there, so the follow-up conversation is at the right depth.
    if result.get("ok") and session.phase < 3:
        session.phase = 3
        session.phase_exchange_count = 0

    session.last_topic = "lab report"
    session.pending_question = "Does any of that raise a question for you?"

    save_session(session)
    _persist_memory(session, answer=answer)

    return {
        "answer": answer,
        "sources": result.get("sources", []),
        "session_id": session.session_id,
        "phase": session.phase,
        "ok": bool(result.get("ok")),
        "parsed_values": result.get("parsed_values", {}),
        "diagnostic_flags": result.get("diagnostic_flags", {}),
        "concordance": result.get("concordance", {}),
        "unreadable": result.get("unreadable", []),
    }


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _run_phase_extraction(session: Session, text: str) -> None:
    """
    Extract profile fields relevant to the current phase.

    Returning users are treated differently on purpose. The original guards
    (`phase <= 2` plus `is None`) meant a hydrated user could never correct
    anything: "actually call me Pri, and this time I'm here about fertility"
    left the name as "Priya" and kept feeding last month's reason to the model
    forever. A person we already know must still be able to update their own
    details, so corrections are accepted in any phase for them.
    """
    profile = session.user_profile
    returning = session.is_returning_user

    # Name — collected in intake, correctable later by a known user.
    #
    # A user we already know is only RENAMED on an explicit request ("call me
    # Pri"). The loose patterns are appropriate only right after the bot asks
    # "what's your name?"; used as a standing trigger they rewrite the stored
    # name from ordinary speech — "hi, im back" was observed renaming a user to
    # "Back".
    if profile.name is None or returning:
        name = extract_name(text, explicit_only=(profile.name is not None))
        if name and name != profile.name:
            if profile.name is None:
                logger.info("[agent] captured name (len=%d)", len(name))
            else:
                logger.info("[agent] updated name on explicit request")
            profile.name = name

    if profile.age_range is None or returning:
        age = extract_age(text)
        if age and age != profile.age_range:
            profile.age_range = age
            logger.info("[agent] captured age range")

    if profile.gender is None or returning:
        gender = extract_gender(text)
        if gender and gender != profile.gender:
            profile.gender = gender
            logger.info("[agent] captured gender")

    # Reason for visit is PER-VISIT and is never hydrated, so a returning user
    # gets a fresh one captured in phase 3 as well as phase 2.
    if profile.reason_for_visit is None and session.phase in (2, 3):
        if len(text.strip()) > 15 and (
            returning
            or profile.age_range is not None
            or profile.gender is not None
        ):
            profile.reason_for_visit = text.strip()[:200]
            logger.info("[agent] Captured reason for visit")


def _should_use_crag(session: Session, user_message: str = "") -> bool:
    """
    Determine whether CRAG retrieval is needed for this turn.
    Phase 1-2: never.  Phase 3: sometimes.  Phase 4: always.
    Also triggers for explicit test requests in Phase 3+.
    """
    if session.phase <= 2:
        return False
    if session.phase == 4:
        return True
    # Phase 3: trigger for symptom context OR explicit test questions
    if _detect_test_request(user_message):
        return True
    return session.symptom_count >= 1


def _run_crag(session: Session, user_message: str) -> tuple[str, list]:
    """
    Build a CRAG query from session context and run retrieval.
    Returns (context_string, sources_list).
    """
    from src.graphs.crag_graph import run_crag

    # Build a focused query from symptoms + user message + last topic
    query_parts = []
    if session.symptom_list:
        query_parts.append("PCOS symptoms: " + ", ".join(session.symptom_list))
    if session.last_topic:
        query_parts.append(f"Topic: {session.last_topic}")
    query_parts.append(user_message)
    query = " ".join(query_parts)

    try:
        result = run_crag(query)
        return result.get("context", ""), result.get("sources", [])
    except Exception as exc:
        logger.error("[agent] CRAG failed: %s", exc)
        return "", []


def _build_test_suggestion_context(session: Session) -> str:
    """
    Build a lab test context string to inject into the system prompt
    when test suggestions are appropriate.
    """
    if session.phase < 3 or session.symptom_count < 3:
        return ""

    tests = _get_relevant_tests(session.symptom_list, limit=5)
    if not tests:
        return ""

    lines = []
    for test in tests:
        reasons = []
        for tag in session.symptom_list:
            if test in _SYMPTOM_TO_TESTS.get(tag, []):
                reasons.append(tag.replace("_", " "))
        reason_str = ", ".join(reasons[:2]) if reasons else "their symptoms"
        lines.append(f"- {test}: relevant because of {reason_str}")

    return (
        "\nRELEVANT LAB TESTS FOR THIS USER (based on their symptom profile):\n"
        + "\n".join(lines)
        + "\nUse these to inform test suggestions when appropriate. "
        "Never dump the full list — suggest 1-2 tests per turn, tied to the symptom being discussed."
    )


def _mentions_pcos(text: str) -> bool:
    """Check if the response text mentions PCOS."""
    return bool(re.search(r"\bpcos\b|polycystic\s+ovar", text, re.IGNORECASE))


def _update_tracking(session: Session, user_message: str, answer: str) -> None:
    """Update last_topic, pending_question, and covered_guidance from the response."""
    # Extract the question from the end of the answer as pending_question
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    for sent in reversed(sentences):
        if sent.strip().endswith("?"):
            session.pending_question = sent.strip()
            if sent.strip() not in session.asked_questions:
                session.asked_questions.append(sent.strip())
            break

    # Update last_topic from symptom tags or user message keywords
    if session.symptom_list:
        session.last_topic = session.symptom_list[-1].replace("_", " ")
    elif len(user_message) > 10:
        session.last_topic = user_message[:60].strip()

    # Track covered guidance topics in Phase 4
    if session.phase == 4 and session.symptom_list:
        # If the response references a symptom area, mark it as covered
        lower_answer = answer.lower()
        for tag in session.symptom_list:
            readable = tag.replace("_", " ")
            if readable in lower_answer or tag in lower_answer:
                add_covered_guidance(session, tag)
