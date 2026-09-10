"""
src/nodes/parse_report.py
Analysis Node 1 — Regex-based lab report parser.
Extracts biomarker values and classifies them against reference ranges.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from src.state.analysis_state import AnalysisState

logger = logging.getLogger(__name__)

_NUMBER = r"(\d+(?:\.\d+)?)"

# (canonical_name, [aliases], [units])
_BIOMARKERS = [
    ("LH",                ["LH", "Luteinizing Hormone", "Luteinising Hormone"],
                          ["mIU/mL", "IU/L", "mIU/L"]),
    ("FSH",               ["FSH", "Follicle.Stimulating Hormone", "Follicle Stimulating Hormone"],
                          ["mIU/mL", "IU/L", "mIU/L"]),
    ("AMH",               ["AMH", "Anti.Mullerian Hormone", "Anti Mullerian Hormone"],
                          ["ng/mL", "pmol/L"]),
    ("Total Testosterone",["Total Testosterone", "Testosterone Total", "Testosterone"],
                          ["ng/dL", "nmol/L", "ng/mL"]),
    ("Free Testosterone", ["Free Testosterone", "Free T"],
                          ["pg/mL", "pmol/L"]),
    ("DHEAS",             ["DHEAS", "DHEA.S", "Dehydroepiandrosterone Sulfate"],
                          ["µg/dL", "ug/dL"]),
    ("Prolactin",         ["Prolactin"],
                          ["ng/mL", "mIU/L"]),
    ("TSH",               ["TSH", "Thyroid Stimulating Hormone"],
                          ["µIU/mL", "mIU/L", "uIU/mL"]),
    ("Fasting Glucose",   ["Fasting Glucose", "Fasting Blood Sugar", "FBS", "Glucose"],
                          ["mg/dL", "mmol/L"]),
    ("Fasting Insulin",   ["Fasting Insulin", "Insulin"],
                          ["µIU/mL", "uIU/mL", "mIU/L"]),
    ("HOMA-IR",           ["HOMA.IR", "HOMA IR"], [""]),
    ("Total Cholesterol", ["Total Cholesterol", "Cholesterol Total", "Cholesterol"],
                          ["mg/dL", "mmol/L"]),
    ("HDL",               ["HDL", "HDL Cholesterol"],  ["mg/dL", "mmol/L"]),
    ("LDL",               ["LDL", "LDL Cholesterol"],  ["mg/dL", "mmol/L"]),
    ("Triglycerides",     ["Triglycerides", "TG"],      ["mg/dL", "mmol/L"]),
    ("BMI",               ["BMI", "Body Mass Index"],  ["kg/m2", "kg/m²", ""]),
    ("Waist Circumference",["Waist Circumference", "Waist"], ["cm", "inches"]),
]

# Reference ranges (2023 PCOS guideline)
_REFS: Dict[str, tuple] = {
    "LH":                 (1.0,   18.0),
    "FSH":                (3.0,   10.0),
    "LH/FSH Ratio":       (None,   2.0),
    "AMH":                (1.0,    3.5),
    "Total Testosterone": (6.0,   82.0),
    "Free Testosterone":  (0.3,    1.9),
    "DHEAS":              (35.0, 430.0),
    "Prolactin":          (3.0,   25.0),
    "TSH":                (0.4,    4.0),
    "Fasting Glucose":    (70.0,  99.0),
    "Fasting Insulin":    (2.0,   20.0),
    "HOMA-IR":            (None,   2.5),
    "Total Cholesterol":  (None, 200.0),
    "HDL":                (50.0,  None),
    "LDL":                (None, 130.0),
    "Triglycerides":      (None, 150.0),
    "BMI":                (18.5,  24.9),
    "Waist Circumference":(None,  88.0),
}


# The unit each reference range in _REFS is expressed in. A value in ANY other
# unit cannot be compared against it.
#
# This was the most dangerous gap in the pipeline: _BIOMARKERS whitelists SI
# units (nmol/L, pmol/L, mIU/L, mmol/L) while _REFS holds conventional-unit
# ranges, so an entirely normal SI-unit panel — the standard in the UK, EU and
# much of India — scored as multiple CRITICAL results. Testosterone 1.8 nmol/L
# (normal) was read as critical_low against a 6-82 ng/dL range; prolactin
# 340 mIU/L (normal) as critical_high against 3-25 ng/mL.
_REF_UNITS: Dict[str, str] = {
    "LH": "miu/ml", "FSH": "miu/ml", "AMH": "ng/ml",
    "Total Testosterone": "ng/dl", "Free Testosterone": "pg/ml",
    "DHEAS": "ug/dl", "Prolactin": "ng/ml", "TSH": "uiu/ml",
    "Fasting Glucose": "mg/dl", "Fasting Insulin": "uiu/ml",
    "Total Cholesterol": "mg/dl", "HDL": "mg/dl", "LDL": "mg/dl",
    "Triglycerides": "mg/dl", "Waist Circumference": "cm",
    # Dimensionless — any or no unit is fine.
    "HOMA-IR": "", "BMI": "", "LH/FSH Ratio": "",
}

# Units that mean the same thing as the reference unit.
_UNIT_SYNONYMS = {
    "miu/ml": {"miu/ml", "miu/l", "iu/l", "miu/ml"},
    "uiu/ml": {"uiu/ml", "µiu/ml", "miu/l", "uiu/l"},
    "ng/ml":  {"ng/ml", "ug/l", "µg/l"},
    "ng/dl":  {"ng/dl"},
    "pg/ml":  {"pg/ml", "ng/l"},
    "ug/dl":  {"ug/dl", "µg/dl"},
    "mg/dl":  {"mg/dl"},
    "cm":     {"cm"},
}


def _unit_matches(name: str, unit: str) -> bool:
    """
    True when `unit` can be compared against this biomarker's reference range.

    An UNKNOWN unit (empty string) is accepted — many reports omit it and the
    existing regex often fails to capture one — but a unit that is present and
    demonstrably different is not.
    """
    expected = _REF_UNITS.get(name)
    if expected is None or expected == "":
        return True
    u = (unit or "").strip().lower().replace("μ", "µ")
    if not u:
        return True                       # not stated; fall back to the range
    if u == expected:
        return True
    return u in _UNIT_SYNONYMS.get(expected, {expected})


def _classify(name: str, value: float) -> str:
    ref = _REFS.get(name)
    if not ref:
        return "normal"
    low, high = ref
    if low is not None and value < low:
        return "critical_low" if value < low * 0.7 else "low"
    if high is not None and value > high:
        return "critical_high" if value > high * 1.5 else "high"
    return "normal"


def _build_pattern(aliases, units) -> re.Pattern:
    alias_re = "|".join(re.escape(a).replace(r"\ ", r"[\s\-]?") for a in aliases)
    unit_re  = "|".join(re.escape(u) for u in units if u)
    unit_grp = rf"(?:\s*(?:{unit_re}))?" if unit_re else ""
    return re.compile(
        rf"(?i)(?:{alias_re})"      # name
        rf"(?![0-9])"               # ...not immediately followed by a digit
        rf"(?:\s*\([^)]+\))?"       # optional parenthesised abbreviation
        rf"\s*[:\-=]?\s*"           # separator
        rf"{_NUMBER}"               # value
        rf"{unit_grp}",             # unit
    )
    # The (?![0-9]) guard matters more than it looks. The Free Testosterone
    # alias "Free T" otherwise matches "Free T3: 3.1" — consuming the "3" of
    # "T3" as the start of the value — so a thyroid FT3 result was parsed as a
    # free testosterone level and raised biochemical hyperandrogenism, the
    # strongest Rotterdam lab criterion, from a normal thyroid panel.


_COMPILED = [(name, units, _build_pattern(aliases, units)) for name, aliases, units in _BIOMARKERS]


def parse_report_node(state: AnalysisState) -> Dict[str, Any]:
    """Extract biomarker values from report_text and classify each one."""
    text = state["report_text"]
    parsed: Dict[str, Any] = {}

    for name, units, pattern in _COMPILED:
        if name in parsed:
            continue
        m = pattern.search(text)
        if m:
            value = float(m.group(1))
            raw   = m.group(0)
            unit  = next((u for u in units if u and u.lower() in raw.lower()), "")
            if _unit_matches(name, unit):
                status = _classify(name, value)
            else:
                # Comparing this number to the range would be meaningless, so it
                # is recorded but NOT classified — and never flagged.
                status = "unit_mismatch"
                logger.warning(
                    "[parse] %s reported in %r but the reference range is in %r "
                    "— not classified", name, unit, _REF_UNITS.get(name),
                )
            parsed[name] = {"value": value, "unit": unit, "status": status}

    # Derived: LH/FSH
    if "LH" in parsed and "FSH" in parsed and parsed["FSH"]["value"] > 0:
        ratio = round(parsed["LH"]["value"] / parsed["FSH"]["value"], 2)
        parsed["LH/FSH Ratio"] = {"value": ratio, "unit": "ratio",
                                   "status": _classify("LH/FSH Ratio", ratio)}

    # Derived: HOMA-IR
    if "HOMA-IR" not in parsed and "Fasting Glucose" in parsed and "Fasting Insulin" in parsed:
        g = parsed["Fasting Glucose"]["value"]
        i = parsed["Fasting Insulin"]["value"]
        homa = round((g / 18.0 * i) / 22.5, 2)
        parsed["HOMA-IR"] = {"value": homa, "unit": "ratio", "status": _classify("HOMA-IR", homa)}

    logger.info("[parse] extracted %d biomarkers", len(parsed))
    return {"parsed_values": parsed}
