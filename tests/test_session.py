"""
tests/test_session.py — Unit tests for session state management.
Zero API calls — purely local logic.
"""
import pytest

from src.core.session import (
    MAX_HISTORY_EXCHANGES,
    MIN_HISTORY_EXCHANGES,
    PHASE_ASSESSMENT,
    PHASE_HISTORY,
    PHASE_ONGOING,
    PHASE_OPENING,
    PHASE_PLAN,
    Session,
    jump_to_ongoing,
    update_treatments,
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
    """
    The consultation is shaped like a real one: opening -> history ->
    ASSESSMENT -> plan -> ongoing.

    Two properties matter most and are asserted here. History is CAPPED, so the
    bot cannot interrogate indefinitely. And the assessment is exactly one turn,
    so the impression is delivered once and the conversation moves on.
    """

    def test_opening_advances_once_a_name_is_given(self):
        s = Session()
        s.user_profile.name = "Priya"
        s.phase_exchange_count = 1
        assert check_phase_transition(s) is True
        assert s.phase == PHASE_HISTORY

    def test_opening_holds_without_a_name(self):
        s = Session()
        s.phase_exchange_count = 3
        assert check_phase_transition(s) is False
        assert s.phase == PHASE_OPENING

    def test_age_and_gender_are_no_longer_gatekeepers(self):
        """
        The old model demanded age AND gender before any health talk, costing
        two turns of interrogation up front. A clinician asks when it matters.
        """
        s = Session()
        s.user_profile.name = "Priya"
        s.phase_exchange_count = 1
        check_phase_transition(s)
        assert s.phase == PHASE_HISTORY
        assert s.user_profile.age_range is None
        assert s.user_profile.gender is None

    def test_history_advances_once_there_is_enough_to_assess(self):
        s = Session(phase=PHASE_HISTORY)
        s.phase_exchange_count = MIN_HISTORY_EXCHANGES
        update_symptoms(s, ["irregular_periods", "acne"])
        assert check_phase_transition(s) is True
        assert s.phase == PHASE_ASSESSMENT

    def test_history_holds_while_the_picture_is_thin(self):
        s = Session(phase=PHASE_HISTORY)
        s.phase_exchange_count = MIN_HISTORY_EXCHANGES
        update_symptoms(s, ["acne"])              # only one symptom
        assert check_phase_transition(s) is False
        assert s.phase == PHASE_HISTORY

    def test_history_is_capped_even_with_nothing_to_go_on(self):
        """The anti-interrogation guarantee: history cannot run forever."""
        s = Session(phase=PHASE_HISTORY)
        s.phase_exchange_count = MAX_HISTORY_EXCHANGES
        assert s.symptom_count == 0
        assert check_phase_transition(s) is True
        assert s.phase == PHASE_ASSESSMENT

    def test_assessment_is_exactly_one_turn(self):
        s = Session(phase=PHASE_ASSESSMENT)
        assert check_phase_transition(s) is False   # not delivered yet
        s.assessment_given = True
        assert check_phase_transition(s) is True
        assert s.phase == PHASE_PLAN

    def test_plan_hands_over_to_ongoing(self):
        s = Session(phase=PHASE_PLAN)
        s.phase_exchange_count = 3
        assert check_phase_transition(s) is True
        assert s.phase == PHASE_ONGOING

    def test_ongoing_is_terminal(self):
        s = Session(phase=PHASE_ONGOING)
        s.phase_exchange_count = 50
        assert check_phase_transition(s) is False
        assert s.phase == PHASE_ONGOING

    def test_exchange_counter_resets_on_every_transition(self):
        s = Session()
        s.user_profile.name = "Priya"
        s.phase_exchange_count = 4
        check_phase_transition(s)
        assert s.phase_exchange_count == 0

    def test_a_full_consultation_reaches_ongoing(self):
        """End to end: no path loops back into endless history."""
        s = Session()
        s.user_profile.name = "Priya"
        s.phase_exchange_count = 1
        check_phase_transition(s)
        update_symptoms(s, ["irregular_periods", "hirsutism"])
        for _ in range(MIN_HISTORY_EXCHANGES):
            s.phase_exchange_count += 1
        check_phase_transition(s)
        assert s.phase == PHASE_ASSESSMENT
        s.assessment_given = True
        check_phase_transition(s)
        assert s.phase == PHASE_PLAN
        s.phase_exchange_count = 3
        check_phase_transition(s)
        assert s.phase == PHASE_ONGOING


class TestTreatmentTracking:
    def test_treatments_accumulate_without_duplicates(self):
        s = Session()
        update_treatments(s, ["metformin"], "considering")
        update_treatments(s, ["metformin", "inositol"], "ongoing")
        assert s.treatments == ["metformin", "inositol"]
        assert s.treatment_stage == "ongoing"

    def test_stage_is_not_cleared_by_a_later_bare_mention(self):
        s = Session()
        update_treatments(s, ["metformin"], "adverse")
        update_treatments(s, ["metformin"], None)
        assert s.treatment_stage == "adverse"

    def test_jump_to_ongoing_skips_assessment(self):
        """A specific treatment question deserves an answer, not a history."""
        s = Session()
        jump_to_ongoing(s)
        assert s.phase == PHASE_ONGOING
        assert s.assessment_given is True
        assert check_phase_transition(s) is False


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
