"""
tests/test_crag_graph.py
Tests for the CRAG LangGraph graph logic (conditional edges, state flow).
ChromaDB and Gemini calls are fully mocked.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.graphs.crag_graph import _should_rewrite
from src.state.crag_state import GradedChunkDict


def _gc(grade: str) -> GradedChunkDict:
    return GradedChunkDict(page_content="text", metadata={}, grade=grade, reason="test")


class TestShouldRewrite:
    def test_all_irrelevant_under_max_passes_rewrites(self):
        state = {
            "graded_chunks": [_gc("IRRELEVANT"), _gc("IRRELEVANT")],
            "retrieval_passes": 1,
        }
        assert _should_rewrite(state) == "rewrite"

    def test_any_ambiguous_under_max_passes_rewrites(self):
        state = {
            "graded_chunks": [_gc("RELEVANT"), _gc("AMBIGUOUS")],
            "retrieval_passes": 1,
        }
        assert _should_rewrite(state) == "rewrite"

    def test_all_relevant_goes_to_refine(self):
        state = {
            "graded_chunks": [_gc("RELEVANT"), _gc("RELEVANT")],
            "retrieval_passes": 1,
        }
        assert _should_rewrite(state) == "refine"

    def test_max_passes_reached_goes_to_refine(self):
        """Even with bad grades, if we've used all passes we go to refine."""
        state = {
            "graded_chunks": [_gc("IRRELEVANT")],
            "retrieval_passes": 2,  # MAX_RETRIEVAL_PASSES = 2
        }
        assert _should_rewrite(state) == "refine"

    def test_empty_chunks_goes_to_refine(self):
        state = {"graded_chunks": [], "retrieval_passes": 1}
        assert _should_rewrite(state) == "refine"


class TestCRAGGraphIntegration:
    """Smoke test: graph compiles and flows from a mocked retrieval."""

    @patch("src.nodes.retrieve.get_retriever")
    @patch("src.nodes.grade._get_client")
    def test_graph_runs_with_mocks(self, mock_get_client, mock_get_retriever):
        from langchain_core.documents import Document
        from src.graphs.crag_graph import run_crag

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="PCOS is characterised by elevated LH.", metadata={
                "source": "guide.pdf", "tier": "guideline", "page": 0, "start_index": 0
            })
        ]
        mock_get_retriever.return_value = mock_retriever

        # Mock lazy Gemini client → RELEVANT grade
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '[{"grade": "RELEVANT", "reason": "Directly addresses PCOS"}]'
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = run_crag("PCOS hormone levels")
        assert result["context"] != ""
        assert len(result["sources"]) >= 1
        assert result["retrieval_passes"] == 1
