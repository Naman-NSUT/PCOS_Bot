"""
tests/test_session.py — Unit tests for session state management.
Zero API calls — purely local logic.
"""
import pytest

from src.core.session import (
    Session,
    UserProfile,
    add_turn,
    check_phase_transition,
    create_session,
    get_or_create_session,
    get_session,
    pcos_threshold_reached,
    reset_store,
    save_session,
    update_symptoms,
)


@pytest.fixture(autouse=True)
def _clean_store():
    """Reset the in-memory session store before each test."""
    reset_store()
    yield
    reset_store()


# ── Session creation ─────────────────────────────────────────────────────

class TestSessionCreation:
    def test_creates_with_defaults(self):
        s = create_session()
        assert s.phase == 1
        assert s.total_exchanges == 0
        assert s.phase_exchange_count == 0
        assert s.symptom_list == []
        assert s.symptom_count == 0
        assert s.pcos_mentioned is False
        assert s.user_profile.name is None
        assert s.history == []

    def test_session_id_is_unique(self):
        s1 = create_session()
        s2 = create_session()
        assert s1.session_id != s2.session_id

    def test_get_session_returns_stored(self):
        s = create_session()
        retrieved = get_session(s.session_id)
        assert retrieved is not None
        assert retrieved.session_id == s.session_id

    def test_get_session_returns_none_for_missing(self):
        assert get_session("nonexistent") is None

    def test_get_or_create_with_none_creates_new(self):
        s = get_or_create_session(None)
        assert s.phase == 1

    def test_get_or_create_with_existing_returns_same(self):
        s = create_session()
        s2 = get_or_create_session(s.session_id)
        assert s2.session_id == s.session_id


# ── Turn management ──────────────────────────────────────────────────────

class TestTurns:
    def test_add_user_turn_increments_counts(self):
        s = create_session()
        add_turn(s, "user", "hello")
        assert s.total_exchanges == 1
        assert s.phase_exchange_count == 1
        assert len(s.history) == 1

    def test_add_assistant_turn_does_not_increment_exchange(self):
        s = create_session()
        add_turn(s, "assistant", "hi")
        assert s.total_exchanges == 0
        assert s.phase_exchange_count == 0
        assert len(s.history) == 1

    def test_multiple_turns(self):
        s = create_session()
        add_turn(s, "user", "hello")
        add_turn(s, "assistant", "hi")
        add_turn(s, "user", "how are you")
        assert s.total_exchanges == 2
        assert len(s.history) == 3


# ── Phase transitions ───────────────────────────────────────────────────

class TestPhaseTransitions:
    def test_phase1_to_2_on_name(self):
        s = create_session()
        assert s.phase == 1
        s.user_profile.name = "Priya"
        changed = check_phase_transition(s)
        assert changed is True
        assert s.phase == 2
        assert s.phase_exchange_count == 0  # reset

    def test_phase1_stays_without_name(self):
        s = create_session()
        changed = check_phase_transition(s)
        assert changed is False
        assert s.phase == 1

    def test_phase2_to_3_on_full_profile(self):
        s = create_session()
        s.user_profile.name = "Priya"
        check_phase_transition(s)  # → phase 2
        s.user_profile.age_range = "27"
        s.user_profile.gender = "female"
        changed = check_phase_transition(s)
        assert changed is True
        assert s.phase == 3

    def test_phase2_stays_without_gender(self):
        s = create_session()
        s.user_profile.name = "Priya"
        check_phase_transition(s)  # → phase 2
        s.user_profile.age_range = "27"
        # gender is still None
        changed = check_phase_transition(s)
        assert changed is False
        assert s.phase == 2

    def test_phase3_to_4_on_exchanges_and_symptoms(self):
        s = create_session()
        # Fast-forward to phase 3
        s.user_profile.name = "Priya"
        s.user_profile.age_range = "27"
        s.user_profile.gender = "female"
        check_phase_transition(s)  # 1→2
        check_phase_transition(s)  # 2→3
        assert s.phase == 3

        # Add exchanges + symptoms
        for _ in range(4):
            add_turn(s, "user", "test")
        update_symptoms(s, ["irregular_periods"])

        changed = check_phase_transition(s)
        assert changed is True
        assert s.phase == 4

    def test_phase3_stays_without_enough_exchanges(self):
        s = create_session()
        s.user_profile.name = "Priya"
        s.user_profile.age_range = "27"
        s.user_profile.gender = "female"
        check_phase_transition(s)
        check_phase_transition(s)
        assert s.phase == 3

        add_turn(s, "user", "test")  # only 1 exchange
        update_symptoms(s, ["fatigue"])
        changed = check_phase_transition(s)
        assert changed is False
        assert s.phase == 3

    def test_phase4_stays_at_4(self):
        s = create_session()
        s.phase = 4
        changed = check_phase_transition(s)
        assert changed is False
        assert s.phase == 4


# ── Symptom helpers ──────────────────────────────────────────────────────

class TestSymptoms:
    def test_update_symptoms_adds_unique(self):
        s = create_session()
        update_symptoms(s, ["fatigue", "acne"])
        assert s.symptom_count == 2
        assert set(s.symptom_list) == {"fatigue", "acne"}

    def test_update_symptoms_deduplicates(self):
        s = create_session()
        update_symptoms(s, ["fatigue", "acne"])
        update_symptoms(s, ["fatigue", "bloating"])
        assert s.symptom_count == 3
        assert "fatigue" in s.symptom_list
        assert s.symptom_list.count("fatigue") == 1

    def test_pcos_threshold(self):
        s = create_session()
        assert pcos_threshold_reached(s) is False
        update_symptoms(s, ["fatigue", "acne", "irregular_periods"])
        assert pcos_threshold_reached(s) is True
