"""
tests/test_hydration_conflicts.py

Regression tests for the eight ways cross-session memory broke a returning user.

Each was reproduced against the real code before the fix: hydrating the whole
prior Session state made the bot escalate to urgent care, close the session, or
freeze the user's profile on their very first message. These tests hydrate a
heavy history and assert the consultation proceeds normally.

Zero LLM calls — generate_response is mocked throughout.
"""
import pytest
from unittest.mock import patch

from src.memory import store
from src.memory.models import Predicate
from src.agents import conversation_agent as agent
from src.core import llm_client
from src.core.session import get_session, reset_store


HEAVY_SYMPTOMS = [
    "irregular_periods", "acne", "weight_gain", "fatigue",
    "hair_loss", "bloating", "mood_disturbance", "sleep_disturbance",
]


@pytest.fixture(autouse=True)
def _fresh():
    store.reset_store_for_tests("sqlite://")
    reset_store()
    yield


@pytest.fixture
def veteran_user():
    """A user with a long, realistic history — 8 symptoms across 6 consultations."""
    uid = "veteran"
    store.get_or_create_user(uid)
    store.update_profile(uid, name="Priya", age_range="27", gender="female")
    for tag in HEAVY_SYMPTOMS:
        store.record_fact(uid, Predicate.REPORTED_SYMPTOM, tag)
        store.record_fact(uid, Predicate.REPORTED_SYMPTOM, tag)   # reaffirmed
    for topic in ("acne", "fatigue", "weight_gain", "sleep_disturbance"):
        store.record_fact(uid, Predicate.RECEIVED_GUIDANCE, topic)
    store.record_fact(uid, Predicate.DISCLOSED_PCOS, "pcos")
    for i in range(6):
        store.start_consultation(uid, f"c{i}")
    return uid


@pytest.fixture(autouse=True)
def reply():
    """
    Mock BOTH the generator and CRAG retrieval.

    _run_crag must be mocked too: any message that trips symptom detection at
    phase 3 fires a real retrieval, which embeds and calls the grader model — a
    live, billed network call from the test suite.
    """
    with patch.object(agent, "generate_response", return_value="Okay. How are you?") as m, \
         patch.object(agent, "_run_crag", return_value=("", [])):
        yield m


def _first_turn(uid, message, reply_mock):
    greet = agent.get_greeting(user_id=uid)
    return agent.process_turn(greet["session_id"], message, user_id=uid), greet


# ── 1. urgent-escalation loop ────────────────────────────────────────────

class TestConflict1_UrgentEscalationLoop:
    def test_eight_remembered_symptoms_do_not_escalate_on_hello(self, veteran_user, reply):
        """
        Before: symptom_count >= 8 hydrated at phase 3 returned "urgent" from
        _check_closure_conditions, so the user said "hi" and was told to go to
        urgent care and shown the door — then every later message re-hydrated and
        re-closed, an infinite welcome-back/escalate loop.
        """
        out, _ = _first_turn(veteran_user, "hi maya, im back", reply)
        session = get_session(out["session_id"])
        assert not session.is_closed
        assert "urgent care" not in out["answer"].lower()

    def test_live_symptom_count_starts_at_zero(self, veteran_user, reply):
        greet = agent.get_greeting(user_id=veteran_user)
        session = get_session(greet["session_id"])
        assert session.symptom_count == 0
        assert session.symptom_list == []

    def test_genuine_crisis_still_escalates(self, veteran_user, reply):
        """The fix must not blunt real escalation."""
        out, _ = _first_turn(veteran_user, "i want to kill myself", reply)
        assert get_session(out["session_id"]).is_closed
        assert "not have to carry this alone" in out["answer"]


# ── 2 & 3. pacing counters ───────────────────────────────────────────────

class TestConflict2and3_PacingCounters:
    def test_prior_exchanges_do_not_close_session(self, veteran_user, reply):
        out, _ = _first_turn(veteran_user, "hello again", reply)
        assert not get_session(out["session_id"]).is_closed

    def test_counters_reset_per_session(self, veteran_user, reply):
        out, _ = _first_turn(veteran_user, "hello again", reply)
        s = get_session(out["session_id"])
        assert s.total_exchanges == 1
        assert s.topic_repeat_count <= 1


# ── 4. guidance-topic accumulation ───────────────────────────────────────

class TestConflict4_GuidanceAccumulation:
    def test_covered_topics_not_hydrated(self, veteran_user, reply):
        greet = agent.get_greeting(user_id=veteran_user)
        assert get_session(greet["session_id"]).covered_guidance_topics == []

    def test_phase4_does_not_instantly_self_terminate(self, veteran_user, reply):
        """
        Before: covered_guidance_topics grew across sessions while symptom_list
        did not, so len(covered) >= len(symptoms) was permanently true and every
        phase-4 session auto-closed after 5 exchanges.
        """
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        s.phase = 4
        s.phase_exchange_count = 6
        assert agent._check_closure_conditions(s, "tell me more") is None


# ── 5. PCOS disclosure gate ──────────────────────────────────────────────

class TestConflict5_DisclosureGate:
    def test_previously_disclosed_still_gets_a_rule(self, veteran_user):
        """
        Before: pcos_disclosed_previously=True with 3-4 symptoms fell through
        every elif branch, leaving pcos_rules empty — the model got NO guidance
        on whether it may say "PCOS". A clinical-safety hole, not a UX wart.
        """
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        s.symptom_list = ["acne", "fatigue", "weight_gain"]
        s.symptom_count = 3
        prompt = llm_client.build_system_prompt(s)
        assert "PCOS MENTION RULE" in prompt

    def test_returning_user_not_re_given_first_time_speech(self, veteran_user):
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        s.symptom_list = ["acne", "fatigue", "weight_gain"]
        s.symptom_count = 3
        prompt = llm_client.build_system_prompt(s)
        assert "You MAY introduce PCOS softly" not in prompt
        assert "already raised with this person" in prompt

    def test_gate_is_covered_at_every_symptom_count(self, veteran_user):
        """No count may produce an empty rule block for a returning user."""
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        for n in range(0, 9):
            s.symptom_list = [f"s{i}" for i in range(n)]
            s.symptom_count = n
            assert "PCOS MENTION RULE" in llm_client.build_system_prompt(s), f"hole at {n}"

    def test_new_user_gate_unchanged(self):
        from src.core.session import Session
        s = Session()
        s.symptom_count = 0
        assert "must NEVER mention PCOS" in llm_client.build_system_prompt(s)


# ── 6. frozen profile ────────────────────────────────────────────────────

class TestConflict6_FrozenProfile:
    def test_returning_user_can_correct_their_name(self, veteran_user, reply):
        """Before: name stayed "Priya" forever; phase<=2 guard blocked updates."""
        out, _ = _first_turn(veteran_user, "actually call me Pri now", reply)
        assert get_session(out["session_id"]).user_profile.name == "Pri"

    def test_reason_for_visit_is_not_carried_over(self, veteran_user, reply):
        greet = agent.get_greeting(user_id=veteran_user)
        assert get_session(greet["session_id"]).user_profile.reason_for_visit is None

    def test_fresh_reason_captured_this_visit(self, veteran_user, reply):
        out, _ = _first_turn(
            veteran_user, "this time I am here about fertility and trying to conceive", reply)
        reason = get_session(out["session_id"]).user_profile.reason_for_visit
        assert reason is not None and "fertility" in reason


# ── 7. doctor nudge & CRAG misfire ───────────────────────────────────────

class TestConflict7_NudgeAndCrag:
    def test_nudge_does_not_fire_on_first_turn(self, veteran_user, reply):
        out, _ = _first_turn(veteran_user, "hi", reply)
        assert "worth taking what you've been sharing" not in out["answer"]

    def test_nudge_suppressed_when_doctor_already_seen(self, veteran_user, reply):
        store.record_fact(veteran_user, Predicate.MENTIONED_DOCTOR, "visit")
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        s.total_exchanges = 15
        s.phase = 3
        assert agent._should_nudge_doctor(s) is False

    def test_no_crag_retrieval_on_a_bare_greeting(self, veteran_user, reply):
        greet = agent.get_greeting(user_id=veteran_user)
        s = get_session(greet["session_id"])
        assert agent._should_use_crag(s, "hi") is False


# ── 8. phase entry ───────────────────────────────────────────────────────

class TestConflict8_PhaseEntry:
    def test_returning_user_is_not_asked_their_name(self, veteran_user):
        greet = agent.get_greeting(user_id=veteran_user)
        assert "What's your name" not in greet["answer"]
        assert "Welcome back" in greet["answer"]

    def test_returning_user_skips_intake(self, veteran_user):
        greet = agent.get_greeting(user_id=veteran_user)
        assert greet["phase"] == 3

    def test_first_time_user_unchanged(self):
        greet = agent.get_greeting(user_id="brand-new")
        assert "What's your name" in greet["answer"]
        assert greet["phase"] == 1

    def test_anonymous_user_behaves_exactly_as_before(self):
        """Memory must be strictly additive — no user_id means no behaviour change."""
        greet = agent.get_greeting(user_id=None)
        s = get_session(greet["session_id"])
        assert greet["phase"] == 1
        assert s.user_id is None
        assert s.memory is None
        assert "What's your name" in greet["answer"]


# ── write-through ────────────────────────────────────────────────────────

class TestPersistence:
    def test_symptoms_persist_without_closure(self, reply):
        """
        Most consultations end by abandonment, which fires no closure signal, so
        a persist-on-close-only design would lose them entirely.
        """
        greet = agent.get_greeting(user_id="u-new")
        agent.process_turn(greet["session_id"], "I have acne and bad bloating", user_id="u-new")
        remembered = {f["object"] for f in store.get_facts("u-new")}
        assert "acne" in remembered and "bloating" in remembered

    def test_memory_survives_process_restart(self, reply):
        greet = agent.get_greeting(user_id="u-new")
        agent.process_turn(greet["session_id"], "I have acne", user_id="u-new")
        reset_store()                      # simulate a redeploy: in-memory store gone
        greet2 = agent.get_greeting(user_id="u-new")
        assert "Welcome back" in greet2["answer"]

    def test_unknown_session_id_rehydrates_instead_of_blanking(self, reply):
        """Before: an unknown id minted a blank Phase-1 session mid-conversation."""
        greet = agent.get_greeting(user_id="u-new")
        agent.process_turn(greet["session_id"], "I have acne", user_id="u-new")
        reset_store()
        out = agent.process_turn("stale-session-id", "so what should I do?", user_id="u-new")
        s = get_session(out["session_id"])
        assert s.is_returning_user is True
        assert s.user_profile is not None

    def test_persist_failure_does_not_break_the_turn(self, reply, monkeypatch):
        greet = agent.get_greeting(user_id="u-new")
        monkeypatch.setattr(agent.memory_store, "update_profile",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db gone")))
        out = agent.process_turn(greet["session_id"], "I have acne", user_id="u-new")
        assert out["answer"]


class TestNameStability:
    """
    Letting a returning user correct their name must not let ordinary speech
    rename them. "hi, im back" was observed overwriting a stored "Priya" with
    "Back" in a live end-to-end run.
    """

    @pytest.mark.parametrize("message", [
        "hi, im back", "I'm exhausted", "I am struggling",
        "I'm fine thanks", "it's been a while", "I am worried about this",
    ])
    def test_ordinary_speech_does_not_rename(self, veteran_user, message, reply):
        out, _ = _first_turn(veteran_user, message, reply)
        assert get_session(out["session_id"]).user_profile.name == "Priya"

    @pytest.mark.parametrize("message,expected", [
        ("actually call me Pri now", "Pri"),
        ("my name is Priyanka actually", "Priyanka"),
        ("I go by Pri", "Pri"),
    ])
    def test_explicit_rename_still_works(self, veteran_user, message, expected, reply):
        out, _ = _first_turn(veteran_user, message, reply)
        assert get_session(out["session_id"]).user_profile.name == expected

    def test_first_time_user_loose_capture_unchanged(self, reply):
        """Right after the bot asks, loose patterns must still work."""
        greet = agent.get_greeting(user_id="fresh")
        out = agent.process_turn(greet["session_id"], "I'm Ananya", user_id="fresh")
        assert get_session(out["session_id"]).user_profile.name == "Ananya"


class TestSymptomRetirement:
    """
    A symptom the user says has stopped must leave the live session AND be
    retired in long-term memory — the capability an append-only store lacks.
    """

    def test_resolution_retires_the_stored_fact(self, veteran_user, reply):
        out, _ = _first_turn(veteran_user, "my periods are regular again now", reply)
        live = {f["object"] for f in store.get_facts(veteran_user, Predicate.REPORTED_SYMPTOM)}
        assert "irregular_periods" not in live

    def test_retired_fact_is_still_history(self, veteran_user, reply):
        _first_turn(veteran_user, "my periods are regular again now", reply)
        history = store.get_facts(veteran_user, Predicate.REPORTED_SYMPTOM, include_resolved=True)
        entry = next(f for f in history if f["object"] == "irregular_periods")
        assert entry["is_resolved"] is True

    def test_resolution_shows_in_next_session_memory_block(self, veteran_user, reply):
        _first_turn(veteran_user, "my periods are regular again now", reply)
        from src.memory.hydration import load_snapshot, render_memory_block
        block = render_memory_block(load_snapshot(veteran_user))
        assert "RESOLVED" in block and "irregular periods" in block

    def test_mere_improvement_does_not_retire(self, veteran_user, reply):
        _first_turn(veteran_user, "my acne is a bit better this month", reply)
        live = {f["object"] for f in store.get_facts(veteran_user, Predicate.REPORTED_SYMPTOM)}
        assert "acne" in live

    def test_same_message_naming_and_retiring_does_not_re_add(self, veteran_user, reply):
        out, _ = _first_turn(veteran_user, "my irregular periods are regular again", reply)
        s = get_session(out["session_id"])
        assert "irregular_periods" not in s.symptom_list


class TestSnapshotFreshness:
    def test_resolved_symptom_leaves_the_block_immediately(self, veteran_user, reply):
        """
        The memory block renders every turn from a snapshot taken at session
        start. Without refreshing on resolve, a symptom the user just said had
        cleared kept showing as active for the rest of the conversation.
        """
        from src.memory.hydration import render_memory_block
        greet = agent.get_greeting(user_id=veteran_user)
        sid = greet["session_id"]
        assert "irregular periods" in render_memory_block(get_session(sid).memory)

        agent.process_turn(sid, "my periods are regular again now", user_id=veteran_user)
        block = render_memory_block(get_session(sid).memory)
        assert "RESOLVED" in block
        active_line = next(
            (l for l in block.splitlines() if l.startswith("- Symptoms they raised")), "")
        assert "irregular periods" not in active_line
