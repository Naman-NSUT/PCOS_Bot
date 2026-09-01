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


# Literal keywords catch the obvious phrasings; these patterns catch how people
# ACTUALLY talk. Measured against hand-labelled realistic phrasings, the keyword
# list alone recalled 13% — it missed "periods all over the place" (plural),
# "chin hair" (wrong word order), and "gained 12kg" (no literal "weight gain").
#
# That matters far beyond context quality: symptom_count gates the PCOS
# disclosure rules, the lab-test rules, what reaches long-term memory, and when
# the consultation moves to assessment. Low recall silently changes clinical
# behaviour, so it is worth the extra vocabulary here.
_SYMPTOM_PATTERNS: list[tuple[str, str]] = [
    # ── irregular_periods ────────────────────────────────────────────────
    (r"period['\u2019]?s?\b.{0,25}\b(irregular|all over|unpredictable|erratic|"
     r"inconsistent|random|sporad)", "irregular_periods"),
    (r"\b(irregular|missed|late|skipped?|infrequent|absent)\s+(period|cycle)", "irregular_periods"),
    (r"\b(month|week)s?\s+(between|apart|without|since)\b.{0,20}period", "irregular_periods"),
    (r"period.{0,20}\b(month|week)s?\s+(apart|between)", "irregular_periods"),
    (r"(haven|have|has)\s*n[o']?t\s+had\s+(a\s+)?period", "irregular_periods"),
    (r"\bcycle['\u2019]?s?\b.{0,25}\b(irregular|off|unpredictable|erratic|all over|"
     r"weird|strange|messed)", "irregular_periods"),
    (r"period['\u2019]?s?\s+come\b.{0,25}(whenever|random|as they)", "irregular_periods"),
    (r"\b\d+\s*(-|to|or)\s*\d+\s*months?\s*(apart|between)?", "irregular_periods"),
    (r"\b(amenorrh|oligomenorrh)", "irregular_periods"),

    # ── hirsutism ────────────────────────────────────────────────────────
    (r"\b(chin|facial|face|jaw|jawline|upper\s*lip|chest|back|neck|tummy|stomach)\s+hairs?",
     "hirsutism"),
    (r"hairs?\s+(on|around)\s+(my\s+)?(chin|face|jaw|lip|chest|back|neck|nipple)", "hirsutism"),
    (r"\b(dark|coarse|thick|unwanted|excess|extra|stray|black)\s+hairs?\b", "hirsutism"),
    (r"growing\s+a\s+(beard|moustache|mustache)", "hirsutism"),
    (r"\b(shave|shaving|wax|waxing|pluck|plucking|thread|threading|epilat|laser)\b"
     r".{0,20}(face|chin|lip|jaw)", "hirsutism"),
    (r"\bhirsut", "hirsutism"),

    # ── weight_gain ──────────────────────────────────────────────────────
    (r"\b(gained|gaining|put\s+on|piled\s+on|packed\s+on)\b.{0,20}"
     r"(weight|kg|kilo|pound|lb|stone)", "weight_gain"),
    (r"\bweight\b.{0,25}\b(gain|up|crept|climbing|creeping|increas|piling)", "weight_gain"),
    (r"\b(can|could|cannot)\s*n[o']?t\b.{0,15}\b(lose|shift|drop|budge)\b.{0,12}weight",
     "weight_gain"),
    (r"\b(struggl|hard|difficult|impossible|unable)\w*\b.{0,20}\b(lose|losing|shift)\b"
     r".{0,12}weight", "weight_gain"),
    (r"\bscale\b.{0,25}\b(up|climbing|going up|creeping)", "weight_gain"),
    (r"nothing\b.{0,25}\b(shift|budge|move)\w*\b.{0,12}weight", "weight_gain"),
    (r"\b(heavier|belly fat|bigger around)", "weight_gain"),

    # ── fatigue ──────────────────────────────────────────────────────────
    (r"\b(exhaust|shattered|knackered|drained|wiped\s*out|worn\s*out|running on empty)",
     "fatigue"),
    (r"\b(no|zero|low|little|hardly any)\s+energy", "fatigue"),
    (r"\b(tired|fatigue)", "fatigue"),

    # ── acne ─────────────────────────────────────────────────────────────
    (r"\b(acne|pimple|spots?\b|breakout|breaking\s+out|blemish|cystic|zits?)", "acne"),
    (r"\bskin\b.{0,25}\b(bad|terrible|awful|worse|horrible|flaring|angry)", "acne"),
    (r"\b(oily|greasy)\s+skin", "acne"),

    # ── hair_loss ────────────────────────────────────────────────────────
    (r"\bhair\b.{0,30}\b(falling|coming\s+out|thinning|shedding|loss|receding|"
     r"handful|clump)", "hair_loss"),
    (r"\b(losing|shedding|thinning)\b.{0,15}hair", "hair_loss"),
    (r"\bpart(ing|ed)?\b.{0,20}(wider|widening|bigger)", "hair_loss"),
    (r"\bbald|scalp\s+show", "hair_loss"),

    # ── fertility_concerns ───────────────────────────────────────────────
    (r"trying\b.{0,20}\b(for a baby|to conceive|for a child|to get pregnant|to fall pregnant)",
     "fertility_concerns"),
    (r"\b(can'?t|cannot|could\s*n'?o?t|couldn'?t|unable)\b.{0,25}\b(get|getting|fall|falling|become|conceiv)\w*\s*(pregnant)?",
     "fertility_concerns"),
    (r"\b(struggl|trouble|difficult)\w*\b.{0,25}(conceiv|pregnan)", "fertility_concerns"),
    (r"\b(fertility|infertil|ttc|ivf|iui|ovulat|conceiv)", "fertility_concerns"),

    # ── mood_disturbance ─────────────────────────────────────────────────
    (r"\bmood\s+(swing|change|dip)", "mood_disturbance"),
    (r"\b(anxiet|anxious|depress|irritab|tearful|on edge)", "mood_disturbance"),
    (r"feel\w*\b.{0,15}\b(low|down|flat|numb|awful|rubbish|miserable)\b", "mood_disturbance"),
    (r"\b(crying|cry a lot|in tears)", "mood_disturbance"),

    # ── acanthosis ───────────────────────────────────────────────────────
    (r"\bdark\b.{0,20}\b(patch|patches|skin|neck|armpit|underarm|fold|groin)", "acanthosis"),
    (r"\bvelvet|acanthosis", "acanthosis"),

    # ── insulin_related ──────────────────────────────────────────────────
    (r"\b(insulin|blood\s*sugar|glucose|pre.?diabet|metformin|hba1c)", "insulin_related"),
    (r"\bcrav\w*\b.{0,15}(sugar|carb|sweet|chocolate|bread)", "insulin_related"),
    (r"sugar\s+(crash|spike|low)", "insulin_related"),

    # ── sleep_disturbance ────────────────────────────────────────────────
    (r"\b(can'?t|cannot|couldn'?t)\s+sleep", "sleep_disturbance"),
    (r"\b(insomnia|not sleeping|sleep\s+(problem|issue|trouble)|sleep\s*apnoea|"
     r"sleep\s*apnea|snor)", "sleep_disturbance"),
    (r"wake?\w*\s+up\b.{0,20}(tired|exhausted|unrefreshed)", "sleep_disturbance"),

    # ── bloating ─────────────────────────────────────────────────────────
    (r"\b(bloat|distend|puffy\s+(tummy|stomach|belly))", "bloating"),
    (r"\b(stomach|tummy|gut|digest)\w*\b.{0,20}\b(issue|problem|trouble|off)", "bloating"),

    # ── headaches ────────────────────────────────────────────────────────
    (r"\b(headache|migraine|head\s+pain)", "headaches"),

    # ── pelvic_pain ──────────────────────────────────────────────────────
    (r"\b(pelvic\s+pain|cramp|ovary\s+pain|ovarian\s+pain|lower\s+abdominal)", "pelvic_pain"),
]

_COMPILED_SYMPTOM_PATTERNS = [
    (re.compile(rx, re.IGNORECASE), tag) for rx, tag in _SYMPTOM_PATTERNS
]


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

    for pattern, tag in _COMPILED_SYMPTOM_PATTERNS:
        if tag not in found and pattern.search(text):
            found.append(tag)

    # Drop anything the person explicitly said they do NOT have. Tagging a
    # denied symptom is worse than missing one: it inflates symptom_count, which
    # gates the PCOS disclosure rules and the lab-test rules.
    return [tag for tag in found if not _is_negated(tag, lower)]


# Phrasings that assert a symptom is ABSENT or NORMAL, per tag.
_NEGATION_PATTERNS: dict[str, str] = {
    "irregular_periods": r"period\w*\b.{0,25}\b(regular|normal|fine|like clockwork|on time)"
                         r"|regular\s+period",
    "weight_gain":       r"\bno\b.{0,15}weight\s*(change|gain)|weight\b.{0,15}\b(stable|steady|same)",
    "sleep_disturbance": r"sleep\w*\b.{0,20}\b(well|fine|great|ok|no problem)",
    "acne":              r"skin\b.{0,20}\b(great|fine|clear|good|ok)",
    "hair_loss":         r"hair\b.{0,25}\b(thick|fine|healthy|normal|no.{0,10}loss)",
    "fatigue":           r"\b(plenty of|good|lots of)\s+energy|not\s+tired",
}
_COMPILED_NEGATIONS = {
    tag: __import__("re").compile(rx, __import__("re").IGNORECASE)
    for tag, rx in _NEGATION_PATTERNS.items()
}


def _is_negated(tag: str, lower_text: str) -> bool:
    """True when the text asserts this symptom is absent or normal."""
    rx = _COMPILED_NEGATIONS.get(tag)
    if rx is None:
        return False
    # "irregular" contains "regular", so require the negation match to not be
    # part of the word "irregular".
    for m in rx.finditer(lower_text):
        span = lower_text[max(0, m.start() - 2):m.end()]
        if "irregular" in lower_text[max(0, m.start() - 12):m.end()]:
            continue
        return True
    return False


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


# ═══════════════════════════════════════════════════════════════════════════
# TREATMENT DETECTION  (drives pre- and post-therapy support)
# ═══════════════════════════════════════════════════════════════════════════

# Named therapy → internal tag. Detecting WHICH therapy matters because the
# questions worth asking before starting, and the things worth watching after,
# are completely different between them.
_TREATMENT_KEYWORDS: dict[str, str] = {
    "metformin": "metformin", "glucophage": "metformin",
    "inositol": "inositol", "myo-inositol": "inositol", "myoinositol": "inositol",
    "spironolactone": "spironolactone", "spiro": "spironolactone", "aldactone": "spironolactone",
    "birth control": "combined_pill", "the pill": "combined_pill", "ocp": "combined_pill",
    "oral contraceptive": "combined_pill", "combined pill": "combined_pill",
    "yasmin": "combined_pill", "diane": "combined_pill", "marvelon": "combined_pill",
    "letrozole": "ovulation_induction", "femara": "ovulation_induction",
    "clomid": "ovulation_induction", "clomiphene": "ovulation_induction",
    "progesterone": "progestin", "progestin": "progestin", "medroxyprogesterone": "progestin",
    "ozempic": "glp1", "semaglutide": "glp1", "wegovy": "glp1",
    "mounjaro": "glp1", "tirzepatide": "glp1",
    "ivf": "fertility_treatment", "iui": "fertility_treatment",
}

# Where they are relative to that therapy. Ordered by specificity — "stopped"
# and "side effects" must win over the vaguer "on it", because they change what
# is useful to say completely.
_TREATMENT_STAGE_CUES: list[tuple[str, list[str]]] = [
    ("stopped", [
        "stopped taking", "came off", "went off", "quit taking", "gave up on",
        "stopped it", "had to stop", "discontinued", "stopped the", "stopped my",
        "no longer taking", "no longer on", "off the pill", "not taking it",
    ]),
    ("adverse", [
        "side effect", "side-effect", "making me sick", "makes me sick",
        "nauseous", "not agreeing with", "reacting badly", "can't tolerate",
        "cant tolerate", "upset stomach", "feeling worse since",
    ]),
    ("considering", [
        "thinking about", "considering", "should i take", "should i start",
        "offered me", "suggested i take", "recommended i start", "want to try",
        "is it worth", "prescribed me", "about to start", "starting next",
        "going to start", "doctor wants me on", "put me on",
    ]),
    ("ongoing", [
        "i take", "i'm taking", "im taking", "i am taking", "been taking",
        "i'm on", "im on", "i am on", "been on", "started taking", "started on",
        "currently taking", "months on", "weeks on",
    ]),
]


def detect_treatments(text: str) -> List[str]:
    """Named therapies mentioned in the message, as internal tags."""
    lower = text.lower()
    found: List[str] = []
    for keyword, tag in _TREATMENT_KEYWORDS.items():
        if keyword in lower and tag not in found:
            found.append(tag)
    return found


def detect_treatment_stage(text: str) -> Optional[str]:
    """
    Where the person is relative to a therapy:
      "considering" -> hasn't started; wants to know if they should
      "ongoing"     -> currently on it
      "adverse"     -> on it and it is not going well
      "stopped"     -> no longer on it

    Returns None when no therapy stage is expressed. Checked in priority order:
    "I stopped metformin, the side effects were awful" is a STOPPED situation,
    not an ongoing one.
    """
    lower = text.lower()
    for stage, cues in _TREATMENT_STAGE_CUES:
        if any(cue in lower for cue in cues):
            return stage
    return None
