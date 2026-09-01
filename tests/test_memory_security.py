"""
tests/test_memory_security.py

Regression tests for defects found by adversarial review of the memory layer.
Each was reproduced against the real code before being fixed. The two criticals
share a root cause: session_id was acting as an implicit bearer credential.

Zero network calls — the LLM and CRAG boundaries are mocked.
"""
import pytest
from unittest.mock import patch

from src.memory import store
from src.memory.models import Predicate
from src.agents import conversation_agent as agent
from src.core import llm_client
from src.core.identity import mint_user_token, verify_user_token
from src.core.session import get_session, purge_user_sessions, reset_store


@pytest.fixture(autouse=True)
def _fresh():
    store.reset_store_for_tests("sqlite://")
    reset_store()
    with patch.object(agent, "generate_response", return_value="Okay. And how is that?"), \
         patch.object(agent, "_run_crag", return_value=("", [])):
        yield


def _victim():
    """Build a real consultation for a victim and return (user_id, session_id)."""
    uid = "victim-user"
    greet = agent.get_greeting(user_id=uid)
    sid = greet["session_id"]
    agent.process_turn(sid, "My name is Priya", user_id=uid)
    agent.process_turn(sid, "I'm 29, female", user_id=uid)
    agent.process_turn(sid, "I have irregular periods, acne and hair loss", user_id=uid)
    return uid, sid


class TestSessionHijack:
    """
    CRITICAL. POST /chat accepted any session_id without checking it belonged to
    the caller, so knowing an id was enough to read a stranger's consultation and
    to write your own turns into their permanent record.
    """

    def test_attacker_cannot_read_victims_session(self):
        _, victim_sid = _victim()
        out = agent.process_turn(victim_sid, "what do you know about me?", user_id="attacker")
        assert out["session_id"] != victim_sid
        assert get_session(out["session_id"]).user_profile.name != "Priya"

    def test_attacker_cannot_write_into_victims_record(self):
        victim_uid, victim_sid = _victim()
        before = {f["object"] for f in store.get_facts(victim_uid, Predicate.REPORTED_SYMPTOM)}
        agent.process_turn(victim_sid, "actually I have severe acanthosis now", user_id="attacker")
        after = {f["object"] for f in store.get_facts(victim_uid, Predicate.REPORTED_SYMPTOM)}
        assert after == before

    def test_anonymous_caller_cannot_read_owned_session(self):
        _, victim_sid = _victim()
        out = agent.process_turn(victim_sid, "what do you know about me?", user_id=None)
        assert get_session(out["session_id"]).user_profile.name != "Priya"

    def test_owner_still_reaches_their_own_session(self):
        victim_uid, victim_sid = _victim()
        out = agent.process_turn(victim_sid, "and I am tired too", user_id=victim_uid)
        assert out["session_id"] == victim_sid


class TestClosedSessionLeak:
    """
    CRITICAL. _handle_closed_session returned a literal 'Welcome back, {name}.
    Last time we spoke about {symptoms}' built from the CLOSED session — with no
    LLM in the path, so no prompt rule could prevent the disclosure — and copied
    that profile into the caller's durable record.
    """

    def test_stranger_gets_no_name_or_symptoms(self):
        victim_uid, victim_sid = _victim()
        agent.process_turn(victim_sid, "thanks, that's all", user_id=victim_uid)
        assert get_session(victim_sid).is_closed

        out = agent.process_turn(victim_sid, "hi", user_id="attacker")
        assert "Priya" not in out["answer"]
        assert "irregular periods" not in out["answer"]

    def test_stranger_profile_not_copied(self):
        victim_uid, victim_sid = _victim()
        agent.process_turn(victim_sid, "thanks, that's all", user_id=victim_uid)
        out = agent.process_turn(victim_sid, "hi", user_id="attacker")
        assert get_session(out["session_id"]).user_profile.name != "Priya"

    def test_owner_still_gets_their_welcome_back(self):
        victim_uid, victim_sid = _victim()
        agent.process_turn(victim_sid, "thanks, that's all", user_id=victim_uid)
        out = agent.process_turn(victim_sid, "hi", user_id=victim_uid)
        assert "Priya" in out["answer"]


class TestIdentityRobustness:
    @pytest.mark.parametrize("token", ["café.abc123", "useré.deadbeef", "🙂.sig"])
    def test_non_ascii_token_degrades_not_crashes(self, token):
        """compare_digest raises TypeError on non-ASCII str — must not 500."""
        assert verify_user_token(token) is None

    def test_valid_token_still_verifies(self):
        uid, token = mint_user_token()
        assert verify_user_token(token) == uid


class TestErasureIsComplete:
    def test_live_session_is_dropped_too(self):
        """Deleting rows while a live session still holds the transcript in RAM
        is not erasure."""
        victim_uid, victim_sid = _victim()
        assert get_session(victim_sid) is not None
        store.forget_user(victim_uid)
        assert purge_user_sessions(victim_uid) >= 1
        assert get_session(victim_sid) is None


class TestNoFabricatedMemory:
    """Memory must record what happened, not what could have been said."""

    def test_tests_not_recorded_unless_named(self):
        uid = "u1"
        greet = agent.get_greeting(user_id=uid)
        agent.process_turn(greet["session_id"],
                           "I have acne, irregular periods and weight gain", user_id=uid)
        assert store.get_facts(uid, Predicate.SUGGESTED_TEST) == []

    def test_tests_recorded_when_actually_named(self):
        uid = "u2"
        greet = agent.get_greeting(user_id=uid)
        with patch.object(agent, "generate_response",
                          return_value="It would be worth asking your doctor about "
                                       "Total Testosterone. Does that sound manageable?"):
            agent.process_turn(greet["session_id"],
                               "I have acne, hirsutism and hair loss", user_id=uid)
        named = {f["object"] for f in store.get_facts(uid, Predicate.SUGGESTED_TEST)}
        assert "Total Testosterone" in named

    def test_refusal_does_not_count_as_pcos_disclosure(self):
        """
        Under the never-mention rule the model still says things like "I can't
        tell you whether this is PCOS". Recording that as a disclosure made the
        NEXT session take the permissive branch of the gate at symptom_count 0.
        """
        uid = "u3"
        greet = agent.get_greeting(user_id=uid)
        with patch.object(agent, "generate_response",
                          return_value="I can't tell you whether this is PCOS. "
                                       "How long has that been going on?"):
            agent.process_turn(greet["session_id"], "I feel tired", user_id=uid)
        assert store.get_facts(uid, Predicate.DISCLOSED_PCOS) == []

    def test_next_session_gate_stays_restrictive(self):
        uid = "u4"
        greet = agent.get_greeting(user_id=uid)
        with patch.object(agent, "generate_response",
                          return_value="I cannot say if this is PCOS. How are you sleeping?"):
            agent.process_turn(greet["session_id"], "I feel tired", user_id=uid)
        greet2 = agent.get_greeting(user_id=uid)
        s2 = get_session(greet2["session_id"])
        assert s2.pcos_disclosed_previously is False
        assert "must NEVER mention PCOS" in llm_client.build_system_prompt(s2)


class TestReadsDoNotWrite:
    def test_hydrating_unknown_user_creates_no_row(self):
        from src.memory.hydration import load_snapshot
        load_snapshot("never-seen")
        assert store.get_user("never-seen") is None

    def test_greeting_alone_creates_no_row(self):
        agent.get_greeting(user_id="window-shopper")
        assert store.get_user("window-shopper") is None

    def test_first_real_turn_does_create_a_row(self):
        greet = agent.get_greeting(user_id="real-user")
        agent.process_turn(greet["session_id"], "My name is Sam", user_id="real-user")
        assert store.get_user("real-user") is not None


class TestWorseningIsNotResolution:
    """A symptom getting worse must never retire it."""

    @pytest.mark.parametrize("message", [
        "my acne has gone crazy this month",
        "my hair loss has gone from bad to worse",
        "my periods have gone haywire",
    ])
    def test_worsening_keeps_the_symptom(self, message):
        uid = "u5"
        greet = agent.get_greeting(user_id=uid)
        agent.process_turn(greet["session_id"], "I have acne, hair loss and irregular periods",
                           user_id=uid)
        before = {f["object"] for f in store.get_facts(uid, Predicate.REPORTED_SYMPTOM)}
        agent.process_turn(greet["session_id"], message, user_id=uid)
        after = {f["object"] for f in store.get_facts(uid, Predicate.REPORTED_SYMPTOM)}
        assert after == before, f"{message!r} retired a symptom"

    def test_genuine_resolution_still_retires(self):
        uid = "u6"
        greet = agent.get_greeting(user_id=uid)
        agent.process_turn(greet["session_id"], "I have irregular periods", user_id=uid)
        agent.process_turn(greet["session_id"], "my periods are regular again", user_id=uid)
        live = {f["object"] for f in store.get_facts(uid, Predicate.REPORTED_SYMPTOM)}
        assert "irregular_periods" not in live
