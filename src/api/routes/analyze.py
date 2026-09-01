"""
src/api/routes/analyze.py — POST /analyze-report
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.graphs.analysis_graph import run_analysis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze-report", tags=["Report Analysis"])


class AnalyzeRequest(BaseModel):
    report_text: str = Field(
        ..., min_length=10,
        description="Raw text from a lab report.",
        examples=["LH: 12.4 mIU/mL, FSH: 5.1 mIU/mL, Testosterone: 85 ng/dL, AMH: 7.2 ng/mL"],
    )
    patient_context: Optional[str] = Field(
        None,
        description="Optional patient context (age, symptoms, etc.).",
        examples=["27-year-old female, irregular periods for 2 years."],
    )


class AnalyzeResponse(BaseModel):
    parsed_values:       Dict[str, Any]
    diagnostic_flags:    Dict[str, Any]
    narrative:           str
    sources:             List[Dict[str, Any]]
    rewritten_query:     Optional[str]
    disclaimer:          str


@router.post("", response_model=AnalyzeResponse,
             summary="Analyse a PCOS lab report",
             description=(
                 "Parses biomarker values, applies rule-based PCOS indicator flags, "
                 "retrieves relevant clinical guidelines via the CRAG LangGraph, and "
                 "generates an evidence-grounded narrative with an LLM. Not a diagnostic tool."
             ))
def analyze_report_endpoint(request: AnalyzeRequest) -> AnalyzeResponse:
    # Plain `def`, not `async def`: run_analysis is fully synchronous and would
    # block the event loop for every other request if run directly on it.
    try:
        state = run_analysis(
            report_text=request.report_text,
            patient_context=request.patient_context,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return AnalyzeResponse(
        parsed_values=state.get("parsed_values", {}),
        diagnostic_flags=state.get("diagnostic_flags", {}),
        narrative=state.get("narrative", ""),
        sources=state.get("crag_sources", []),
        rewritten_query=state.get("crag_rewritten_query"),
        disclaimer=state.get("disclaimer", ""),
    )
