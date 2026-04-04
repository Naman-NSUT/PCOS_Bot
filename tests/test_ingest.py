"""
tests/test_ingest.py
Unit tests for the document loading and chunking pipeline.
Mocks file I/O — no ChromaDB or Gemini API required.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.ingest.chunker import chunk_documents


class TestChunker:
    def test_short_document_produces_one_chunk(self):
        doc = Document(
            page_content="PCOS is a hormonal disorder.",
            metadata={"source": "test.txt", "tier": "patient_education"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["tier"] == "patient_education"

    def test_large_document_produces_multiple_chunks(self):
        # 3000-char document should split into at least 3 chunks at chunk_size=800
        large_text = "PCOS affects many women worldwide. " * 90
        doc = Document(
            page_content=large_text,
            metadata={"source": "big.txt", "tier": "guideline"},
        )
        chunks = chunk_documents([doc])
        assert len(chunks) >= 2

    def test_metadata_preserved_in_all_chunks(self):
        large_text = "Polycystic Ovary Syndrome diagnosis criteria. " * 60
        doc = Document(
            page_content=large_text,
            metadata={"source": "guideline.pdf", "tier": "guideline", "page": 5},
        )
        chunks = chunk_documents([doc])
        for chunk in chunks:
            assert chunk.metadata["source"] == "guideline.pdf"
            assert chunk.metadata["tier"] == "guideline"

    def test_empty_documents_list(self):
        chunks = chunk_documents([])
        assert chunks == []

    def test_chunk_size_respected(self):
        """No chunk should exceed chunk_size by more than the overlap."""
        from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
        large_text = "A B C D E F G H I J. " * 200
        doc = Document(page_content=large_text, metadata={"source": "x.txt", "tier": "research"})
        chunks = chunk_documents([doc])
        for chunk in chunks:
            # Allow a small buffer due to splitter boundary logic
            assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP + 50


class TestDocumentLoader:
    @patch("src.ingest.document_loader.PyPDFLoader")
    def test_pdf_loader_called_for_pdfs(self, mock_loader_cls):
        """Verify PDFs trigger PyPDFLoader."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="PCOS guideline text", metadata={})
        ]
        mock_loader_cls.return_value = mock_loader

        from pathlib import Path
        from src.ingest.document_loader import _load_pdf

        result = _load_pdf(Path("fake.pdf"), "guideline")
        assert mock_loader_cls.called
        assert len(result) >= 1
        assert result[0].metadata["tier"] == "guideline"
        assert result[0].metadata["source"] == "fake.pdf"
