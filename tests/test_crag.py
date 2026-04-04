"""
tests/test_crag.py
Unit tests for the CRAG pipeline steps.
All Gemini API calls are mocked — zero external dependencies.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.crag.grader import GradedChunk
from src.crag.refiner import refine_chunks
from src.crag.assembler import assemble, TIER_RANK
from src.crag.query_rewriter import needs_rewrite


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_doc(content: str, source: str = "test.pdf", tier: str = "guideline", page: int = 0) -> Document:
    return Document(
        page_content=content,
        metadata={"source": source, "tier": tier, "page": page, "start_index": 0},
    )


def _graded(doc: Document, grade: str) -> GradedChunk:
    return GradedChunk(document=doc, grade=grade, reason="test")


# ── Grader tests ──────────────────────────────────────────────────────────

class TestNeedsRewrite:
    def test_all_irrelevant_triggers_rewrite(self):
        chunks = [
            _graded(_make_doc("irrelevant"), "IRRELEVANT"),
            _graded(_make_doc("also irrelevant"), "IRRELEVANT"),
        ]
        assert needs_rewrite(chunks) is True

    def test_any_ambiguous_triggers_rewrite(self):
        chunks = [
            _graded(_make_doc("relevant PCOS content"), "RELEVANT"),
            _graded(_make_doc("vaguely related"), "AMBIGUOUS"),
        ]
        assert needs_rewrite(chunks) is True

    def test_all_relevant_no_rewrite(self):
        chunks = [
            _graded(_make_doc("PCOS guideline A"), "RELEVANT"),
            _graded(_make_doc("PCOS guideline B"), "RELEVANT"),
        ]
        assert needs_rewrite(chunks) is False


# ── Refiner tests ────────────────────────────────────────────────────────

class TestRefiner:
    def test_relevant_chunks_pass_through(self):
        doc = _make_doc("PCOS is a hormonal disorder affecting fertility.")
        graded = [_graded(doc, "RELEVANT")]
        refined = refine_chunks(graded, "PCOS hormones")
        assert len(refined) == 1
        assert refined[0][1] == doc.page_content

    def test_irrelevant_chunks_discarded(self):
        doc = _make_doc("Unrelated medical text about the liver.")
        graded = [_graded(doc, "IRRELEVANT")]
        refined = refine_chunks(graded, "PCOS")
        assert len(refined) == 0

    def test_ambiguous_keeps_relevant_sentences(self):
        content = "PCOS causes irregular cycles. The appendix is a vestigial organ."
        doc = _make_doc(content)
        graded = [_graded(doc, "AMBIGUOUS")]
        refined = refine_chunks(graded, "PCOS irregular cycles")
        assert len(refined) >= 1
        assert "PCOS" in refined[0][1]

    def test_ambiguous_strips_off_topic_sentences(self):
        content = "The appendix is a vestigial organ. Unrelated medical fact."
        doc = _make_doc(content)
        graded = [_graded(doc, "AMBIGUOUS")]
        # Query has no keywords overlapping with content
        refined = refine_chunks(graded, "PCOS testosterone ovary")
        # Should be discarded (no sentences kept)
        assert len(refined) == 0


# ── Assembler tests ──────────────────────────────────────────────────────

class TestAssembler:
    def test_empty_input_returns_empty(self):
        result = assemble([])
        assert result.context == ""
        assert result.sources == []

    def test_guideline_ranked_before_patient_education(self):
        guideline_doc = _make_doc("Guideline content", source="guide.pdf", tier="guideline")
        edu_doc       = _make_doc("Education content", source="edu.txt",  tier="patient_education")
        refined = [(edu_doc, edu_doc.page_content), (guideline_doc, guideline_doc.page_content)]
        result = assemble(refined)
        # Guideline should appear first in context
        assert result.context.index("[Source: guide.pdf") < result.context.index("[Source: edu.txt")

    def test_sources_populated(self):
        doc = _make_doc("Some content", source="s.pdf", tier="research")
        result = assemble([(doc, doc.page_content)])
        assert len(result.sources) == 1
        assert result.sources[0]["source"] == "s.pdf"
        assert result.sources[0]["tier"]   == "research"

    def test_duplicate_chunks_deduplicated(self):
        doc = _make_doc("Duplicate content", source="dup.pdf")
        refined = [(doc, doc.page_content), (doc, doc.page_content)]
        result = assemble(refined)
        # Should appear only once in sources
        assert len(result.sources) == 1

    def test_citation_format(self):
        doc = _make_doc("Content", source="g.pdf", tier="guideline", page=2)
        result = assemble([(doc, doc.page_content)])
        assert "[Source: g.pdf" in result.context
        assert "p.3" in result.context  # page 2 → p.3 (0-indexed)
        assert "guideline" in result.context
