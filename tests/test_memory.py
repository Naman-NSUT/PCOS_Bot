"""
tests/test_memory.py — Long-term memory: store, identity, hydration.

Every test runs against an in-memory SQLite database and touches no network.
The heart of this file is TestHydrationConflicts, which pins the eight ways a
naive "restore the whole user state" implementation broke a returning user on
their very first message.
"""
import pytest

from src.memory import store
from src.memory.models import Predicate
from src.memory.hydration import (
    NEVER_HYDRATE, load_snapshot, render_memory_block, welcome_back_greeting,
)
from src.core.identity import mint_user_token, sign_user_id, verify_user_token
from src.core.session import Session, create_session, reset_store, resolve_symptom, update_symptoms


@pytest.fixture(autouse=True)
def _fresh_db():
    store.reset_store_for_tests("sqlite://")
    reset_store()
    yield


# ── Identity ─────────────────────────────────────────────────────────────

class TestIdentity:
    def test_round_trip(self):
        uid, token = mint_user_token()
        assert verify_user_token(token) == uid

    def test_ids_are_unique(self):
        assert mint_user_token()[0] != mint_user_token()[0]

    def test_tampered_id_is_rejected(self):
        """The whole point: you must not be able to read someone else's record."""
        uid, token = mint_user_token()
        forged = "deadbeef" + token[len(uid):]
        assert verify_user_token(forged) is None

    def test_bad_signature_rejected(self):
        uid, _ = mint_user_token()
        assert verify_user_token(f"{uid}.notarealsignature") is None

    def test_unsigned_id_rejected(self):
        assert verify_user_token("just-a-plain-id") is None

    @pytest.mark.parametrize("bad", [None, "", ".", "a.", ".b"])
    def test_malformed_tokens_rejected(self, bad):
        assert verify_user_token(bad) is None

    def test_resigning_is_stable(self):
        uid, token = mint_user_token()
        assert sign_user_id(uid) == token


# ── Store ────────────────────────────────────────────────────────────────

class TestStore:
    def test_new_user_flagged_new_then_not(self):
        assert store.get_or_create_user("u1")["is_new"] is True
        assert store.get_or_create_user("u1")["is_new"] is False

    def test_facts_are_idempotent(self):
        for _ in range(3):
            store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        facts = store.get_facts("u1")
        assert len(facts) == 1
        assert facts[0]["occurrences"] == 3

    def test_profile_update_does_not_erase(self):
        store.update_profile("u1", name="Priya", age_range="27", gender="female")
        store.update_profile("u1", name="Pri")          # later session, no age given
        u = store.get_or_create_user("u1")
        assert u["preferred_name"] == "Pri"
        assert u["age_range"] == "27"                    # not wiped

    def test_unknown_predicate_rejected(self):
        with pytest.raises(ValueError):
            store.record_fact("u1", "NOT_A_PREDICATE", "x")

    def test_facts_are_per_user(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert store.get_facts("u2") == []


class TestSymptomResolution:
    """The capability an append-only memory cannot provide."""

    def test_resolved_symptom_stops_being_current(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        assert store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        assert store.get_facts("u1") == []                       # not current

    def test_resolved_symptom_is_still_history(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        history = store.get_facts("u1", include_resolved=True)
        assert len(history) == 1
        assert history[0]["is_resolved"] is True
        assert history[0]["invalid_at"] is not None

    def test_resolving_absent_fact_is_false(self):
        assert store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "nope") is False

    def test_symptom_can_come_back_as_a_new_episode(self):
        """
        A recurrence must open a NEW episode, not reopen the old row.

        Reopening in place cleared invalid_at and moved valid_at forward, which
        erased the earlier interval — "had it Jan-Jun, then again from
        September" collapsed into one row claiming it began in September. That
        is precisely the history the bi-temporal schema exists to keep.
        """
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")

        live = store.get_facts("u1")
        assert len(live) == 1                       # exactly one CURRENT episode
        assert live[0]["is_resolved"] is False
        assert live[0]["episode"] == 2

        allf = store.get_facts("u1", include_resolved=True)
        assert len(allf) == 2                       # both episodes survive

    def test_prior_episode_interval_is_preserved(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")

        history = store.episode_history("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        assert [e["episode"] for e in history] == [1, 2]
        assert history[0]["is_resolved"] is True
        assert history[0]["invalid_at"] is not None      # closed interval intact
        assert history[1]["is_resolved"] is False
        assert history[1]["invalid_at"] is None

    def test_three_episodes_all_retained(self):
        for _ in range(3):
            store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
            store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert len(store.episode_history("u1", Predicate.REPORTED_SYMPTOM, "acne")) == 3

    def test_reaffirming_within_an_episode_does_not_add_a_row(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        allf = store.get_facts("u1", include_resolved=True)
        assert len(allf) == 1
        assert allf[0]["occurrences"] == 2
        assert allf[0]["episode"] == 1

    def test_session_level_resolution(self):
        s = Session()
        update_symptoms(s, ["acne", "fatigue"])
        assert resolve_symptom(s, "acne") is True
        assert s.symptom_list == ["fatigue"]
        assert s.symptom_count == 1
        assert resolve_symptom(s, "acne") is False


class TestGraphTraversal:
    def test_co_occurring_symptoms(self):
        for tag in ("acne", "hirsutism", "fatigue"):
            store.record_fact("u1", Predicate.REPORTED_SYMPTOM, tag)
        neighbours = store.co_occurring_symptoms("u1", "acne")
        assert set(neighbours) == {"hirsutism", "fatigue"}

    def test_resolved_symptoms_are_not_neighbours(self):
        for tag in ("acne", "hirsutism"):
            store.record_fact("u1", Predicate.REPORTED_SYMPTOM, tag)
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "hirsutism")
        assert store.co_occurring_symptoms("u1", "acne") == []

    def test_unknown_anchor_returns_empty(self):
        assert store.co_occurring_symptoms("u1", "acne") == []


class TestErasure:
    def test_forget_user_removes_everything(self):
        store.get_or_create_user("u1")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        store.start_consultation("u1", "c1")
        assert store.forget_user("u1") is True
        assert store.get_facts("u1") == []
        assert store.get_or_create_user("u1")["is_new"] is True

    def test_forget_unknown_user_is_false(self):
        assert store.forget_user("nobody") is False


# ── Hydration ────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_brand_new_user_is_not_returning(self):
        assert load_snapshot("u1").is_returning is False

    def test_known_user_is_returning(self):
        store.get_or_create_user("u1")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert load_snapshot("u1").is_returning is True

    def test_snapshot_separates_active_from_resolved(self):
        store.get_or_create_user("u1")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        snap = load_snapshot("u1")
        assert snap.active_symptoms == ["acne"]
        assert snap.resolved_symptoms == ["irregular_periods"]

    def test_empty_snapshot_renders_nothing(self):
        assert render_memory_block(load_snapshot("u1")) == ""

    def test_block_marks_content_as_prior_session(self):
        store.get_or_create_user("u1")
        store.update_profile("u1", name="Priya")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        block = render_memory_block(load_snapshot("u1"))
        assert "NOT said in this conversation" in block
        assert "Do NOT assume any of it is still true" in block

    def test_resolved_symptoms_flagged_in_block(self):
        store.get_or_create_user("u1")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        store.resolve_fact("u1", Predicate.REPORTED_SYMPTOM, "irregular_periods")
        block = render_memory_block(load_snapshot("u1"))
        assert "RESOLVED" in block

    def test_welcome_back_uses_name(self):
        store.get_or_create_user("u1")
        store.update_profile("u1", name="Priya")
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert "Priya" in welcome_back_greeting(load_snapshot("u1"))

    def test_store_failure_degrades_to_empty(self, monkeypatch):
        """Memory is advisory: a broken store must not break the consultation."""
        monkeypatch.setattr(store, "get_facts",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
        snap = load_snapshot("u1")
        assert snap.is_returning is False
        assert render_memory_block(snap) == ""


class TestNeverHydrateContract:
    """These field names are load-bearing — see TestHydrationConflicts."""

    @pytest.mark.parametrize("field", [
        "total_exchanges", "phase_exchange_count", "topic_repeat_count",
        "symptom_list", "symptom_count", "covered_guidance_topics",
        "doctor_nudge_sent",
    ])
    def test_pacing_field_is_blocklisted(self, field):
        assert field in NEVER_HYDRATE

    def test_snapshot_exposes_no_blocklisted_field(self):
        snap = load_snapshot("u1")
        leaked = NEVER_HYDRATE & set(snap.model_dump().keys())
        assert not leaked, f"snapshot leaks pacing state: {leaked}"


class TestProvenance:
    """
    Who asserted a fact is currently implicit in the predicate name. Making it
    explicit is what lets a predicate ever come from either side, and is a
    prerequisite for reconciling with any clinical record.
    """

    def test_user_reported_symptom_defaults_to_user(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert store.get_facts("u1")[0]["asserter"] == "user"

    def test_system_authored_fact_is_marked(self):
        from src.memory.models import Asserter
        store.record_fact("u1", Predicate.SUGGESTED_TEST, "TSH", asserter=Asserter.SYSTEM)
        assert store.get_facts("u1")[0]["asserter"] == "system"

    def test_verification_defaults_to_reported(self):
        store.record_fact("u1", Predicate.REPORTED_SYMPTOM, "acne")
        assert store.get_facts("u1")[0]["verification"] == "reported"
