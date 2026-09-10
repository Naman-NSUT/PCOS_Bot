"""
src/nodes/flag_indicators.py
Analysis Node 2 — Rule-based PCOS indicator engine.
Maps parsed biomarkers → DiagnosticFlags dict and builds the CRAG query.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.state.analysis_state import AnalysisState

logger = logging.getLogger(__name__)


def flag_indicators_node(state: AnalysisState) -> Dict[str, Any]:
    """
    Run rule-based PCOS indicator rules and also build the CRAG retrieval query
    from detected flags so retrieval is symptom-driven.
    """
    bm: Dict[str, Any] = state.get("parsed_values", {})

    def val(name: str):
        if name not in bm or bm[name].get("status") == "unit_mismatch":
            return None
        return bm[name]["value"]

    def status(name: str) -> str:
        """
        A value whose unit does not match its reference range is reported as
        "unit_mismatch" and must never raise a flag — comparing it to the range
        is meaningless, so it is treated as if the marker were absent.
        """
        return bm[name]["status"] if name in bm else "normal"

    def usable(name: str) -> bool:
        return name in bm and bm[name].get("status") != "unit_mismatch"

    flags: Dict[str, Any] = {
        "hyperandrogenism":             False,
        "hyperandrogenism_evidence":    [],
        "insulin_resistance":           False,
        "insulin_resistance_evidence":  [],
        "polycystic_morphology_proxy":  False,
        "polycystic_morphology_evidence": [],
        "thyroid_flag":                 False,
        "thyroid_evidence":             [],
        "prolactin_flag":               False,
        "prolactin_evidence":           [],
        "metabolic_risk_score":         0,
        "metabolic_risk_factors":       [],
        "critical_flags":               [],
    }

    # ── 1. Hyperandrogenism ──────────────────────────────────────────────
    for biomarker, label in [
        ("Total Testosterone", "Total Testosterone"),
        ("Free Testosterone",  "Free Testosterone"),
        ("DHEAS",              "DHEAS (adrenal androgen)"),
    ]:
        if status(biomarker) in ("high", "critical_high"):
            flags["hyperandrogenism"] = True
            v = val(biomarker)
            flags["hyperandrogenism_evidence"].append(
                f"{label} elevated: {v} {bm[biomarker]['unit']}"
            )

    # ── 2. Insulin Resistance ─────────────────────────────────────────────
    homa = val("HOMA-IR")
    if homa and homa > 2.5:
        flags["insulin_resistance"] = True
        flags["insulin_resistance_evidence"].append(f"HOMA-IR = {homa} (>2.5)")
    for bname, desc in [("Fasting Insulin", "Fasting insulin"), ("Fasting Glucose", "Fasting glucose")]:
        if status(bname) in ("high", "critical_high"):
            flags["insulin_resistance"] = True
            flags["insulin_resistance_evidence"].append(
                f"{desc}: {val(bname)} {bm[bname]['unit']}"
            )

    # ── 3. Polycystic Ovarian Morphology Proxy ────────────────────────────
    amh = val("AMH")
    if amh and amh > 3.5:
        flags["polycystic_morphology_proxy"] = True
        flags["polycystic_morphology_evidence"].append(f"AMH = {amh} ng/mL (>3.5)")
    lh_fsh = val("LH/FSH Ratio")
    if lh_fsh and lh_fsh >= 2.0:
        flags["polycystic_morphology_proxy"] = True
        flags["polycystic_morphology_evidence"].append(f"LH/FSH ratio = {lh_fsh} (≥2.0)")

    # ── 4. Thyroid ─────────────────────────────────────────────────────────
    if status("TSH") not in ("normal",):
        flags["thyroid_flag"] = True
        flags["thyroid_evidence"].append(
            f"TSH = {val('TSH')} µIU/mL ({status('TSH')}) — rule out thyroid dysfunction"
        )

    # ── 5. Prolactin ───────────────────────────────────────────────────────
    if status("Prolactin") in ("high", "critical_high"):
        flags["prolactin_flag"] = True
        flags["prolactin_evidence"].append(
            f"Prolactin elevated: {val('Prolactin')} ng/mL — rule out hyperprolactinemia"
        )

    # ── 6. Metabolic Risk (0–4) ────────────────────────────────────────────
    metabolic_checks = [
        ("Total Cholesterol", ("high", "critical_high"), "Elevated total cholesterol"),
        ("LDL",               ("high", "critical_high"), "Elevated LDL"),
        ("Triglycerides",     ("high", "critical_high"), "Elevated triglycerides"),
        ("HDL",               ("low",  "critical_low"),  "Low HDL"),
        ("Waist Circumference",("high","critical_high"), "Elevated waist circumference"),
        ("BMI",               ("high", "critical_high"), "Elevated BMI"),
    ]
    for name, bad, desc in metabolic_checks:
        if status(name) in bad:
            flags["metabolic_risk_score"] = min(flags["metabolic_risk_score"] + 1, 4)
            flags["metabolic_risk_factors"].append(f"{desc}: {val(name)} {bm.get(name, {}).get('unit','')}")

    # ── 7. Critical flags ──────────────────────────────────────────────────
    for name, info in bm.items():
        if info["status"] in ("critical_high", "critical_low"):
            flags["critical_flags"].append(
                f"{name} CRITICALLY {'HIGH' if 'high' in info['status'] else 'LOW'}: "
                f"{info['value']} {info['unit']}"
            )

    # ── Build CRAG query ───────────────────────────────────────────────────
    parts = ["PCOS diagnostic criteria and hormone levels"]
    if flags["hyperandrogenism"]:
        parts.append("elevated testosterone hyperandrogenism PCOS")
    if flags["insulin_resistance"]:
        parts.append("insulin resistance HOMA-IR PCOS management")
    if flags["polycystic_morphology_proxy"]:
        parts.append("AMH LH FSH ratio polycystic ovarian morphology")
    if flags["thyroid_flag"]:
        parts.append("thyroid dysfunction PCOS differential diagnosis")
    if flags["metabolic_risk_score"] >= 2:
        parts.append("metabolic syndrome cardiovascular risk PCOS")

    crag_query = " | ".join(parts)
    logger.info("[flag] flags set: HA=%s IR=%s PCO=%s metab=%d",
                flags["hyperandrogenism"], flags["insulin_resistance"],
                flags["polycystic_morphology_proxy"], flags["metabolic_risk_score"])

    return {"diagnostic_flags": flags, "crag_query": crag_query}
