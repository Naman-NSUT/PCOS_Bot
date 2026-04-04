"""
src/state/analysis_state.py
LangGraph state schema for the full report analysis pipeline.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AnalysisState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    report_text: str
    patient_context: Optional[str]

    # ── Parse node output ──────────────────────────────────────────────────
    parsed_values: Dict[str, Any]   # { biomarker: {value, unit, status} }

    # ── Flag node output ───────────────────────────────────────────────────
    diagnostic_flags: Dict[str, Any]

    # ── CRAG node output ───────────────────────────────────────────────────
    crag_query: str
    crag_context: str
    crag_sources: List[Dict[str, Any]]
    crag_rewritten_query: Optional[str]

    # ── Generate node output ───────────────────────────────────────────────
    narrative: str
    disclaimer: str
