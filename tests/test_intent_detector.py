"""
tests/test_intent_detector.py — Unit tests for intent detection functions.
Zero API calls — purely regex/keyword logic.
"""
import pytest

from src.core.intent_detector import (
    detect_emotion,
    detect_symptoms,
    extract_age,
    extract_gender,
    extract_name,
)


# ── extract_name ─────────────────────────────────────────────────────────

class TestExtractName:
    def test_my_name_is(self):
        assert extract_name("my name is Priya") == "Priya"

    def test_im_format(self):
        assert extract_name("I'm Ananya") == "Ananya"

    def test_i_am_format(self):
        assert extract_name("I am Rohit") == "Rohit"

    def test_call_me(self):
        assert extract_name("call me Neha") == "Neha"

    def test_its_format(self):
        assert extract_name("it's Maya") == "Maya"

    def test_bare_name(self):
        assert extract_name("Priya") == "Priya"

    def test_ignores_common_words(self):
        assert extract_name("Fine") is None
        assert extract_name("Hello") is None
        assert extract_name("Yes") is None

    def test_no_name_found(self):
        assert extract_name("I don't know") is None

    def test_short_word_rejected(self):
        assert extract_name("I") is None


# ── extract_age ──────────────────────────────────────────────────────────

class TestExtractAge:
    def test_im_27(self):
        assert extract_age("I'm 27") == "27"

    def test_years_old(self):
        assert extract_age("I am 30 years old") == "30"

    def test_27f(self):
        assert extract_age("27F") == "27"

    def test_bare_number(self):
        assert extract_age("25") == "25"

    def test_late_twenties(self):
        result = extract_age("I'm in my late 20s")
        assert result is not None
        assert "20s" in result

    def test_out_of_range_rejected(self):
        assert extract_age("5") is None
        assert extract_age("95") is None

    def test_no_age(self):
        assert extract_age("hello there") is None


# ── extract_gender ───────────────────────────────────────────────────────

class TestExtractGender:
    def test_female(self):
        assert extract_gender("I'm female") == "female"

    def test_woman(self):
        assert extract_gender("I'm a woman") == "female"

    def test_male(self):
        assert extract_gender("male") == "male"

    def test_nonbinary(self):
        assert extract_gender("I'm non-binary") == "non-binary"

    def test_27f_shorthand(self):
        assert extract_gender("27F") == "female"

    def test_skip(self):
        assert extract_gender("prefer not to say") == "not specified"

    def test_no_gender(self):
        assert extract_gender("hello there") is None


# ── detect_symptoms ──────────────────────────────────────────────────────

class TestDetectSymptoms:
    def test_irregular_periods(self):
        tags = detect_symptoms("my periods are really irregular")
        assert "irregular_periods" in tags

    def test_acne(self):
        tags = detect_symptoms("I've been getting bad acne")
        assert "acne" in tags

    def test_weight_gain(self):
        tags = detect_symptoms("I can't lose weight no matter what I try")
        assert "weight_gain" in tags

    def test_fatigue(self):
        tags = detect_symptoms("I'm so tired all the time")
        assert "fatigue" in tags

    def test_hirsutism(self):
        tags = detect_symptoms("I have unwanted facial hair")
        assert "hirsutism" in tags

    def test_mood(self):
        tags = detect_symptoms("I've been feeling very anxious lately")
        assert "mood_disturbance" in tags

    def test_multiple_symptoms(self):
        text = "I have acne, I'm always tired, and my periods are irregular"
        tags = detect_symptoms(text)
        assert "acne" in tags
        assert "fatigue" in tags
        assert "irregular_periods" in tags
        assert len(tags) == 3

    def test_no_symptoms(self):
        tags = detect_symptoms("I had a good day today")
        assert tags == []

    def test_hair_loss(self):
        tags = detect_symptoms("my hair is falling out so much")
        assert "hair_loss" in tags

    def test_sleep(self):
        tags = detect_symptoms("I have insomnia and can't sleep")
        assert "sleep_disturbance" in tags

    def test_fertility(self):
        tags = detect_symptoms("we've been trying to conceive for a year")
        assert "fertility_concerns" in tags

    def test_insulin(self):
        tags = detect_symptoms("my doctor said something about insulin resistance")
        assert "insulin_related" in tags

    def test_bloating(self):
        tags = detect_symptoms("I feel so bloated all the time")
        assert "bloating" in tags

    def test_pelvic_pain(self):
        tags = detect_symptoms("I have bad cramps every month")
        assert "pelvic_pain" in tags


# ── detect_emotion ───────────────────────────────────────────────────────

class TestDetectEmotion:
    def test_frustrated(self):
        assert detect_emotion("I'm so frustrated with all of this") == "frustrated"

    def test_scared(self):
        assert detect_emotion("I'm really scared about what's wrong") == "scared"

    def test_overwhelmed(self):
        assert detect_emotion("everything feels overwhelming right now") == "overwhelmed"

    def test_sad(self):
        assert detect_emotion("I've been crying all day") == "sad"

    def test_hopeless(self):
        assert detect_emotion("I feel like I've given up") == "hopeless"

    def test_confused(self):
        assert detect_emotion("I don't understand what's happening") == "confused"

    def test_worried(self):
        assert detect_emotion("I'm really worried about this") == "worried"

    def test_no_emotion(self):
        assert detect_emotion("my period was two days late") is None

    def test_ashamed(self):
        assert detect_emotion("I feel so embarrassed about my skin") == "ashamed"
