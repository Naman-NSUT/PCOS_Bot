"""
tests/test_nodes.py
Unit tests for node functions (parse_report, flag_indicators, refine, assemble).
Zero Gemini API calls — purely local logic.
"""
import pytest
from langchain_core.documents import Document

from src.nodes.parse_report    import parse_report_node
from src.nodes.flag_indicators import flag_indicators_node
from src.nodes.refine          import refine_node
from src.nodes.assemble        import assemble_node
from src.state.crag_state      import GradedChunkDict, RefinedChunkDict


# ── Helpers ──────────────────────────────────────────────────────────────────

def _analysis_state(**kwargs):
    defaults = {
        "report_text": "", "patient_context": None,
        "parsed_values": {}, "diagnostic_flags": {},
        "crag_query": "", "crag_context": "", "crag_sources": [],
        "crag_rewritten_query": None, "narrative": "", "disclaimer": "",
    }
    defaults.update(kwargs)
    return defaults


def _crag_state(**kwargs):
    defaults = {
        "query": "pcos", "active_query": "pcos", "chunks": [],
        "retrieval_passes": 1, "graded_chunks": [], "rewritten_query": None,
        "refined_chunks": [], "context": "", "sources": [], "grade_summary": {},
    }
    defaults.update(kwargs)
    return defaults


def _gc(content, grade, source="x.pdf", tier="guideline") -> GradedChunkDict:
    return GradedChunkDict(page_content=content,
                           metadata={"source": source, "tier": tier, "start_index": 0},
                           grade=grade, reason="test")


def _rc(content, source="x.pdf", tier="guideline") -> RefinedChunkDict:
    return RefinedChunkDict(page_content=content,
                            metadata={"source": source, "tier": tier, "start_index": 0, "page": 0})


# ── parse_report_node ─────────────────────────────────────────────────────

SAMPLE = "LH: 12.4 mIU/mL\nFSH: 5.1 mIU/mL\nTotal Testosterone: 85 ng/dL\nAMH: 7.2 ng/mL\nFasting Glucose: 102 mg/dL\nFasting Insulin: 22 µIU/mL"

class TestParseReportNode:
    def test_extracts_lh(self):
        out = parse_report_node(_analysis_state(report_text=SAMPLE))
        assert "LH" in out["parsed_values"]
        assert out["parsed_values"]["LH"]["value"] == 12.4

    def test_computes_lh_fsh_ratio(self):
        out = parse_report_node(_analysis_state(report_text=SAMPLE))
        assert "LH/FSH Ratio" in out["parsed_values"]
        assert out["parsed_values"]["LH/FSH Ratio"]["value"] == round(12.4 / 5.1, 2)

    def test_computes_homa_ir(self):
        out = parse_report_node(_analysis_state(report_text=SAMPLE))
        assert "HOMA-IR" in out["parsed_values"]
        assert out["parsed_values"]["HOMA-IR"]["value"] > 0

    def test_testosterone_classified_high(self):
        out = parse_report_node(_analysis_state(report_text=SAMPLE))
        assert out["parsed_values"]["Total Testosterone"]["status"] == "high"

    def test_empty_report(self):
        out = parse_report_node(_analysis_state(report_text="No lab values here."))
        assert out["parsed_values"] == {}

    def test_parenthesised_amh(self):
        out = parse_report_node(_analysis_state(report_text="Anti Mullerian Hormone (AMH): 4.5 ng/mL"))
        assert "AMH" in out["parsed_values"]
        assert out["parsed_values"]["AMH"]["value"] == 4.5


# ── flag_indicators_node ──────────────────────────────────────────────────

class TestFlagIndicatorsNode:
    def test_elevated_testosterone_hyperandrogenism(self):
        parsed = {"Total Testosterone": {"value": 90.0, "unit": "ng/dL", "status": "high"}}
        out = flag_indicators_node(_analysis_state(parsed_values=parsed))
        assert out["diagnostic_flags"]["hyperandrogenism"] is True

    def test_high_homa_ir_insulin_resistance(self):
        parsed = {"HOMA-IR": {"value": 3.5, "unit": "ratio", "status": "high"}}
        out = flag_indicators_node(_analysis_state(parsed_values=parsed))
        assert out["diagnostic_flags"]["insulin_resistance"] is True

    def test_high_amh_polycystic_proxy(self):
        parsed = {"AMH": {"value": 5.0, "unit": "ng/mL", "status": "high"}}
        out = flag_indicators_node(_analysis_state(parsed_values=parsed))
        assert out["diagnostic_flags"]["polycystic_morphology_proxy"] is True

    def test_abnormal_tsh_thyroid_flag(self):
        parsed = {"TSH": {"value": 7.0, "unit": "µIU/mL", "status": "high"}}
        out = flag_indicators_node(_analysis_state(parsed_values=parsed))
        assert out["diagnostic_flags"]["thyroid_flag"] is True

    def test_crag_query_built(self):
        parsed = {"Total Testosterone": {"value": 90.0, "unit": "ng/dL", "status": "high"}}
        out = flag_indicators_node(_analysis_state(parsed_values=parsed))
        assert "crag_query" in out
        assert len(out["crag_query"]) > 0

    def test_empty_parsed_no_flags(self):
        out = flag_indicators_node(_analysis_state(parsed_values={}))
        flags = out["diagnostic_flags"]
        assert flags["hyperandrogenism"] is False
        assert flags["metabolic_risk_score"] == 0


# ── refine_node ───────────────────────────────────────────────────────────

class TestRefineNode:
    def test_relevant_passes_through(self):
        state = _crag_state(
            active_query="PCOS testosterone",
            graded_chunks=[_gc("PCOS causes elevated testosterone levels.", "RELEVANT")],
        )
        out = refine_node(state)
        assert len(out["refined_chunks"]) == 1

    def test_irrelevant_discarded(self):
        state = _crag_state(
            active_query="PCOS",
            graded_chunks=[_gc("The liver metabolises glucose.", "IRRELEVANT")],
        )
        out = refine_node(state)
        assert len(out["refined_chunks"]) == 0

    def test_ambiguous_filtered_by_keyword(self):
        state = _crag_state(
            active_query="PCOS testosterone androgen",
            graded_chunks=[_gc(
                "PCOS causes excess testosterone. The appendix is vestigial.",
                "AMBIGUOUS",
            )],
        )
        out = refine_node(state)
        assert len(out["refined_chunks"]) >= 1
        assert "PCOS" in out["refined_chunks"][0]["page_content"]


# ── assemble_node ─────────────────────────────────────────────────────────

class TestAssembleNode:
    def test_empty_refined_returns_empty(self):
        out = assemble_node(_crag_state(refined_chunks=[]))
        assert out["context"] == ""
        assert out["sources"] == []

    def test_guideline_ranked_before_research(self):
        state = _crag_state(refined_chunks=[
            _rc("Research content", source="r.pdf", tier="research"),
            _rc("Guideline content", source="g.pdf", tier="guideline"),
        ])
        out = assemble_node(state)
        assert out["context"].index("[Source: g.pdf") < out["context"].index("[Source: r.pdf")

    def test_sources_populated(self):
        state = _crag_state(refined_chunks=[_rc("content", source="a.pdf", tier="guideline")])
        out = assemble_node(state)
        assert len(out["sources"]) == 1
        assert out["sources"][0]["source"] == "a.pdf"

    def test_deduplication(self):
        chunk = _rc("Duplicate content", source="d.pdf")
        state = _crag_state(refined_chunks=[chunk, chunk])
        out = assemble_node(state)
        assert len(out["sources"]) == 1
