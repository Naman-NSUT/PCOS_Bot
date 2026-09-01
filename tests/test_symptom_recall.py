"""
tests/test_symptom_recall.py — symptom detection against realistic phrasing.

Why this file exists. The deterministic extractor was measured against
hand-labelled phrasings written the way patients actually talk, and recalled
13%: it missed "periods all over the place" (plural), "chin hair" (word order),
and "gained 12kg" (no literal "weight gain").

That is not a cosmetic problem. symptom_count gates the PCOS disclosure rules
and the lab-test rules in llm_client.py, decides what reaches long-term memory,
and now decides when the consultation moves from history to assessment. Low
recall silently changes clinical behaviour.

Precision matters more than recall in the other direction: tagging a symptom the
person explicitly DENIED is worse than missing one, because it inflates the same
gate. Both are asserted here.
"""
import pytest

from src.core.intent_detector import detect_symptoms


REALISTIC = [
    ("my periods are all over the place",          "irregular_periods"),
    ("periods all over the place for 2 years",     "irregular_periods"),
    ("my cycle is really unpredictable",           "irregular_periods"),
    ("I go months between periods",                "irregular_periods"),
    ("I haven't had a period since March",         "irregular_periods"),
    ("my periods come whenever they feel like it", "irregular_periods"),
    ("lots of chin hair",                          "hirsutism"),
    ("hair on my chin and jawline",                "hirsutism"),
    ("I've started growing a beard basically",     "hirsutism"),
    ("dark hairs on my upper lip",                 "hirsutism"),
    ("I have to shave my face",                    "hirsutism"),
    ("gained 12kg this year",                      "weight_gain"),
    ("I've put on a lot of weight",                "weight_gain"),
    ("the scale keeps going up",                   "weight_gain"),
    ("nothing I do shifts the weight",             "weight_gain"),
    ("always exhausted",                           "fatigue"),
    ("I'm shattered all the time",                 "fatigue"),
    ("zero energy",                                "fatigue"),
    ("my skin has been terrible",                  "acne"),
    ("breaking out along my jaw",                  "acne"),
    ("my hair is coming out in handfuls",          "hair_loss"),
    ("my parting is getting wider",                "hair_loss"),
    ("we've been trying for a baby for 18 months", "fertility_concerns"),
    ("I can't seem to get pregnant",               "fertility_concerns"),
    ("dark velvety patches on my neck",            "acanthosis"),
    ("I crave sugar constantly",                   "insulin_related"),
    ("I feel really low a lot of the time",        "mood_disturbance"),
    ("I'm bloated all the time",                   "bloating"),
]


class TestRecall:
    @pytest.mark.parametrize("text,tag", REALISTIC)
    def test_realistic_phrasing_is_detected(self, text, tag):
        assert tag in detect_symptoms(text), f"missed {tag} in {text!r}"

    def test_overall_recall_stays_high(self):
        hits = sum(1 for txt, tag in REALISTIC if tag in detect_symptoms(txt))
        assert hits / len(REALISTIC) >= 0.90, f"recall regressed to {hits}/{len(REALISTIC)}"


class TestPrecision:
    """A denied symptom must never be tagged — it inflates the same gate."""

    @pytest.mark.parametrize("text", [
        "hi, how are you",
        "my name is Priya",
        "I'm 27 and female",
        "thanks, that's really helpful",
        "what tests should I ask for?",
        "I'm worried about my appointment",
        "can you explain what that means",
    ])
    def test_ordinary_conversation_tags_nothing(self, text):
        assert detect_symptoms(text) == []

    @pytest.mark.parametrize("text,denied", [
        ("my periods are completely regular",   "irregular_periods"),
        ("my periods are regular and on time",  "irregular_periods"),
        ("no weight change at all",             "weight_gain"),
        ("I sleep really well actually",        "sleep_disturbance"),
        ("my skin has been great lately",       "acne"),
        ("I have a lot of hair on my head, it's thick", "hair_loss"),
    ])
    def test_explicit_denial_is_not_tagged(self, text, denied):
        assert denied not in detect_symptoms(text), f"{text!r} wrongly tagged {denied}"

    def test_irregular_is_not_swallowed_by_the_regular_guard(self):
        """The negation guard must not cancel the word it is a substring of."""
        assert "irregular_periods" in detect_symptoms("my periods are irregular")
        assert "irregular_periods" in detect_symptoms("I have irregular periods")


class TestConsultationFlow:
    """Recall now gates the phase machine, so a normal history must reach assessment."""

    def test_a_realistic_history_reaches_the_threshold(self):
        from src.core.session import Session, check_phase_transition, update_symptoms, PHASE_HISTORY, PHASE_ASSESSMENT
        s = Session(phase=PHASE_HISTORY)
        for msg in ["Periods all over the place for 2 years",
                    "Every 2-3 months, heavy. Lots of chin hair too",
                    "I'm 27, gained 12kg, always exhausted"]:
            update_symptoms(s, detect_symptoms(msg))
            s.phase_exchange_count += 1
        assert s.symptom_count >= 2, f"only found {s.symptom_list}"
        check_phase_transition(s)
        assert s.phase == PHASE_ASSESSMENT
