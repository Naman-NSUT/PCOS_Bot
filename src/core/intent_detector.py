"""
src/core/intent_detector.py — Pure-function intent extraction.

Zero LLM calls.  Everything here is regex / keyword based for speed and
determinism.  Functions are stateless — they take text in, return structured
data out.
"""
from __future__ import annotations

import re
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

# EXPLICIT patterns state an intent to be called something. Safe to act on at
# any point in a conversation.
_NAME_PATTERNS_EXPLICIT = [
    re.compile(r"\bmy\s+name\s+is\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bcall\s+me\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bi\s+go\s+by\s+([A-Z][a-z]+)", re.IGNORECASE),
]

# LOOSE patterns only mean a name when we have just ASKED for one. Acting on
# them unconditionally renames people from ordinary speech: "hi, im back"
# parses as the name "Back", and "I'm fine" as "Fine".
_NAME_PATTERNS_LOOSE = [
    re.compile(r"\bi[''']?m\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bi\s+am\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bit[''']?s\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"\bthis\s+is\s+([A-Z][a-z]+)", re.IGNORECASE),
    re.compile(r"^(?:hey[,.]?\s+)?([A-Z][a-z]+)\s+here\b", re.IGNORECASE),
]

_NAME_PATTERNS = _NAME_PATTERNS_EXPLICIT + _NAME_PATTERNS_LOOSE

# Words that look like names but aren't
_NAME_BLOCKLIST = {
    "fine", "good", "great", "okay", "ok", "bad", "not", "yes", "no",
    "hi", "hey", "hello", "sure", "well", "really", "actually", "just",
    "tired", "sad", "happy", "stressed", "doing", "feeling", "much",
    "thanks", "thank", "please", "sorry", "scared", "worried", "confused",
    "the", "what", "how", "who", "why", "when", "there", "here",
    # Observed false positives from ordinary conversational openers.
    "back", "home", "late", "early", "better", "worse", "still", "again",
    "afraid", "unsure", "curious", "hoping", "trying", "struggling",
    "wondering", "asking", "looking", "new", "old", "ready", "done",
    "free", "busy", "glad", "nervous", "exhausted", "overwhelmed",
}


def extract_name(text: str, explicit_only: bool = False) -> Optional[str]:
    """
    Try to extract a personal name from the user message.

    explicit_only=True restricts matching to phrases that state an intent to be
    called something ("my name is", "call me", "I go by"). Use it whenever the
    bot has NOT just asked for a name — otherwise ordinary speech renames the
    user: "hi, im back" was observed setting the stored name to "Back".
    """
    patterns = _NAME_PATTERNS_EXPLICIT if explicit_only else _NAME_PATTERNS
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in _NAME_BLOCKLIST and len(candidate) >= 2:
                return candidate.capitalize()

    if explicit_only:
        return None

    # Fallback: if the entire message is a single word that starts uppercase.
    # Split on ANY whitespace — checking only for " " let a multi-line message
    # through, so "Priya\nI have acne" was read as the name.
    stripped = text.strip().rstrip(".!?,")
    if stripped and len(stripped.split()) == 1 and stripped[0].isupper():
        if stripped.lower() not in _NAME_BLOCKLIST and len(stripped) >= 2:
            return stripped.capitalize()

    return None


# ═══════════════════════════════════════════════════════════════════════════
# AGE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_AGE_PATTERNS = [
    # "I'm 27" / "I am 27" / "I'm 27 years old" / "age 27"
    re.compile(r"\b(?:i[''']?m|i\s+am|age[d]?)\s*(\d{1,2})\b", re.IGNORECASE),
    # "27 years old" / "27 yo" / "27F" / "27M"
    re.compile(r"\b(\d{1,2})\s*(?:years?\s*old|yo|yrs?|[fFmM])\b"),
    # bare number 13–65
    re.compile(r"^(\d{1,2})$"),
    # "late 20s" / "early 30s" / "mid 20s"
    re.compile(r"\b((?:early|mid|late)\s+\d{2}s)\b", re.IGNORECASE),
    # "20s" / "thirties"
    re.compile(r"\b(\d{2}s|teens?|twenties|thirties|forties|fifties)\b", re.IGNORECASE),
]


def extract_age(text: str) -> Optional[str]:
    """Try to extract an age or age range from the user message."""
    for pattern in _AGE_PATTERNS:
        m = pattern.search(text.strip())
        if m:
            val = m.group(1).strip()
            # Validate bare numbers are plausible ages
            if val.isdigit():
                n = int(val)
                if 10 <= n <= 80:
                    return val
            else:
                return val
    return None


# ═══════════════════════════════════════════════════════════════════════════
# GENDER EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

_GENDER_MAP = {
    "female": "female",
    "woman": "female",
    "girl": "female",
    "she": "female",
    "her": "female",
    "lady": "female",
    "male": "male",
    "man": "male",
    "boy": "male",
    "he": "male",
    "him": "male",
    "guy": "male",
    "non-binary": "non-binary",
    "nonbinary": "non-binary",
    "nb": "non-binary",
    "enby": "non-binary",
    "genderqueer": "non-binary",
    "trans": "transgender",
    "transgender": "transgender",
    "prefer not": "not specified",
    "rather not": "not specified",
    "skip": "not specified",
}

# Pattern to catch "27F" or "27M" style
_SHORTHAND_GENDER = re.compile(r"\b\d{1,2}\s*([FfMm])\b")


def extract_gender(text: str) -> Optional[str]:
    """Try to extract gender identity from the user message (inclusive)."""
    lower = text.lower()

    # Check shorthand first (27F, 27M)
    m = _SHORTHAND_GENDER.search(text)
    if m:
        ch = m.group(1).upper()
        return "female" if ch == "F" else "male"

    for keyword, label in _GENDER_MAP.items():
        # Use word-boundary matching to avoid false positives (e.g. "her" in "there")
        if re.search(r'\b' + re.escape(keyword) + r'\b', lower):
            return label

    return None


# ═══════════════════════════════════════════════════════════════════════════
# SYMPTOM DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Maps keyword fragments → internal tag
_SYMPTOM_KEYWORDS: dict[str, str] = {
    # irregular_periods
    "irregular period":   "irregular_periods",
    "missed period":      "irregular_periods",
    "late period":        "irregular_periods",
    "no period":          "irregular_periods",
    "period all over":    "irregular_periods",
    "cycle is off":       "irregular_periods",
    "cycle has been":     "irregular_periods",
    "irregular cycle":    "irregular_periods",
    "periods are":        "irregular_periods",
    "skipping period":    "irregular_periods",
    "haven't had my period": "irregular_periods",
    "menstrual":          "irregular_periods",
    "amenorrhea":         "irregular_periods",
    "oligomenorrhea":     "irregular_periods",

    # hirsutism
    "facial hair":        "hirsutism",
    "extra hair":         "hirsutism",
    "hair on my face":    "hirsutism",
    "hair on my chin":    "hirsutism",
    "hair on my chest":   "hirsutism",
    "hair on my back":    "hirsutism",
    "hirsutism":          "hirsutism",
    "unwanted hair":      "hirsutism",
    "body hair":          "hirsutism",
    "excess hair":        "hirsutism",

    # acne
    "acne":               "acne",
    "breakout":           "acne",
    "pimple":             "acne",
    "oily skin":          "acne",
    "cystic acne":        "acne",

    # weight_gain
    "weight gain":        "weight_gain",
    "gaining weight":     "weight_gain",
    "can't lose weight":  "weight_gain",
    "cant lose weight":   "weight_gain",
    "hard to lose":       "weight_gain",
    "put on weight":      "weight_gain",
    "getting heavier":    "weight_gain",
    "belly fat":          "weight_gain",

    # fatigue
    "fatigue":            "fatigue",
    "tired":              "fatigue",
    "exhausted":          "fatigue",
    "no energy":          "fatigue",
    "low energy":         "fatigue",
    "drained":            "fatigue",
    "wiped out":          "fatigue",

    # fertility_concerns
    "trying to conceive": "fertility_concerns",
    "fertility":          "fertility_concerns",
    "getting pregnant":   "fertility_concerns",
    "can't get pregnant":  "fertility_concerns",
    "cant get pregnant":  "fertility_concerns",
    "ttc":                "fertility_concerns",
    "ivf":                "fertility_concerns",
    "ovulat":             "fertility_concerns",

    # mood_disturbance
    "mood swing":         "mood_disturbance",
    "mood change":        "mood_disturbance",
    "anxiety":            "mood_disturbance",
    "anxious":            "mood_disturbance",
    "depress":            "mood_disturbance",
    "irritable":          "mood_disturbance",
    "irritability":       "mood_disturbance",
    "emotional":          "mood_disturbance",
    "crying":             "mood_disturbance",

    # hair_loss
    "hair loss":          "hair_loss",
    "hair falling":       "hair_loss",
    "hair is falling":    "hair_loss",
    "hair thinning":      "hair_loss",
    "thinning hair":      "hair_loss",
    "losing hair":        "hair_loss",
    "bald":               "hair_loss",
    "shedding hair":      "hair_loss",
    "falling out":        "hair_loss",

    # bloating
    "bloat":              "bloating",
    "bloating":           "bloating",
    "digestive":          "bloating",
    "stomach issue":      "bloating",
    "belly swelling":     "bloating",

    # acanthosis
    "dark patch":         "acanthosis",
    "skin darkening":     "acanthosis",
    "dark skin":          "acanthosis",
    "acanthosis":         "acanthosis",
    "dark neck":          "acanthosis",
    "dark armpit":        "acanthosis",

    # insulin_related
    "insulin":            "insulin_related",
    "blood sugar":        "insulin_related",
    "sugar craving":      "insulin_related",
    "glucose":            "insulin_related",
    "pre-diabetic":       "insulin_related",
    "prediabetic":        "insulin_related",
    "metformin":          "insulin_related",

    # sleep_disturbance
    "can't sleep":        "sleep_disturbance",
    "cant sleep":         "sleep_disturbance",
    "insomnia":           "sleep_disturbance",
    "sleep problem":      "sleep_disturbance",
    "sleep issue":        "sleep_disturbance",
    "not sleeping":       "sleep_disturbance",
    "waking up tired":    "sleep_disturbance",
    "sleep apnea":        "sleep_disturbance",
    "unrefreshed":        "sleep_disturbance",

    # headaches
    "headache":           "headaches",
    "migraine":           "headaches",
    "head pain":          "headaches",

    # pelvic_pain
    "pelvic pain":        "pelvic_pain",
    "cramp":              "pelvic_pain",
    "lower abdominal":    "pelvic_pain",
    "ovary pain":         "pelvic_pain",
    "ovarian pain":       "pelvic_pain",
}


def detect_symptoms(text: str) -> List[str]:
    """
    Scan user text for symptom keywords and return a deduplicated list
    of internal symptom tags.
    """
    lower = text.lower()
    found: List[str] = []
    for keyword, tag in _SYMPTOM_KEYWORDS.items():
        if keyword in lower and tag not in found:
            found.append(tag)
    return found


# ═══════════════════════════════════════════════════════════════════════════
# EMOTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

_EMOTION_KEYWORDS: dict[str, str] = {
    "frustrated":   "frustrated",
    "frustrating":  "frustrated",
    "annoyed":      "frustrated",
    "annoying":     "frustrated",
    "scared":       "scared",
    "afraid":       "scared",
    "terrified":    "scared",
    "fear":         "scared",
    "overwhelm":    "overwhelmed",
    "too much":     "overwhelmed",
    "can't handle": "overwhelmed",
    "cant handle":  "overwhelmed",
    "relieved":     "relieved",
    "relief":       "relieved",
    "hopeful":      "hopeful",
    "hope":         "hopeful",
    "ashamed":      "ashamed",
    "shame":        "ashamed",
    "embarrass":    "ashamed",
    "gross":        "ashamed",
    "ugly":         "ashamed",
    "exhausted":    "exhausted",
    "burnt out":    "exhausted",
    "burned out":   "exhausted",
    "angry":        "angry",
    "furious":      "angry",
    "mad":          "angry",
    "pissed":       "angry",
    "sad":          "sad",
    "crying":       "sad",
    "cry":          "sad",
    "lonely":       "sad",
    "heartbroken":  "sad",
    "confused":     "confused",
    "don't understand": "confused",
    "dont understand":  "confused",
    "lost":         "confused",
    "numb":         "numb",
    "empty":        "numb",
    "nothing":      "numb",
    "worried":      "worried",
    "worry":        "worried",
    "anxious":      "worried",
    "nervous":      "worried",
    "hopeless":     "hopeless",
    "given up":     "hopeless",
    "no point":     "hopeless",
    "discouraged":  "hopeless",
}


def detect_emotion(text: str) -> Optional[str]:
    """
    Scan user text for emotional tone keywords.
    Returns the first (dominant) emotion detected, or None.
    """
    lower = text.lower()
    for keyword, emotion in _EMOTION_KEYWORDS.items():
        if keyword in lower:
            return emotion
    return None


# ═══════════════════════════════════════════════════════════════════════════
# RESOLUTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Phrases that assert a symptom has STOPPED, not merely improved. The bar is
# deliberately high: "my acne is a bit better" must NOT retire the symptom, only
# an unambiguous statement that it is over should.
_RESOLUTION_CUES = [
    "regular again", "back to normal", "back to regular",
    "cleared up", "cleared completely", "completely cleared",
    "gone away", "gone completely", "gone entirely",
    "no longer", "not anymore", "no more",
    "stopped completely", "completely stopped",
    "fully resolved", "resolved now", "sorted itself out",
    "don't have", "dont have", "doesn't happen", "doesnt happen",
    "haven't had any", "havent had any",
]

# Negations that flip a cue back to "still present": "my periods are NOT regular
# again" must not be read as a resolution.
_RESOLUTION_NEGATORS = ["not ", "n't ", "never ", "hardly ", "barely ", "still "]

# Words that mean the symptom got WORSE. Their presence in the clause vetoes any
# resolution cue: "my acne has gone crazy" and "my hair loss has gone from bad to
# worse" both contain a cue but assert the opposite, and mis-reading them retires
# a live symptom AND permanently stamps invalid_at on it in long-term memory.
_WORSENING_MARKERS = [
    "worse", "worsen", "crazy", "haywire", "wild", "mad",
    "through the roof", "out of control", "flaring", "flare",
    "terrible", "awful", "horrible", "worst", "bad to worse",
    "more frequent", "heavier", "increased", "increasing", "spiking",
]


def detect_resolved_symptoms(text: str) -> List[str]:
    """
    Detect symptoms the user says have STOPPED, returning their internal tags.

    Works sentence by sentence so a resolution cue is only attributed to a
    symptom named in the SAME sentence — otherwise "my acne is awful but my
    periods are regular again" would retire both.

    This is what lets stored memory retire a fact instead of accumulating
    contradictions: an append-only memory keeps "irregular periods" alive
    alongside "periods are regular again" and lets ranking pick a winner.
    """
    resolved: List[str] = []

    # Split on sentence ends AND contrastive conjunctions. Without "but", the
    # clause "my acne is awful but my periods are regular again" attributes the
    # resolution cue to acne as well.
    _CLAUSE_SPLIT = r"(?<=[.!?])\s+|,\s+(?=(?:my|the|i)\b)|\b(?:and|but|though|however|although|while)\b"
    for sentence in re.split(_CLAUSE_SPLIT, text.lower()):
        sentence = sentence.strip()
        if not sentence:
            continue

        cue = next((c for c in _RESOLUTION_CUES if c in sentence), None)
        if cue is None:
            continue

        # A clause that also reports worsening is not a resolution, whatever cue
        # it happens to contain.
        if any(w in sentence for w in _WORSENING_MARKERS):
            continue

        # Reject a negated cue: look at the words immediately before it.
        head = sentence[: sentence.find(cue)]
        if any(neg in head[-14:] for neg in _RESOLUTION_NEGATORS):
            continue

        for keyword, tag in _SYMPTOM_KEYWORDS.items():
            if keyword in sentence and tag not in resolved:
                resolved.append(tag)

    return resolved
