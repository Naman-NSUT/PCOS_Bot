"""
tests/test_ingestion.py
Tests for the ingestion module (chunker, loader mock).
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


class TestChunker:
    def test_short_document_one_chunk(self):
        doc = Document(page_content="PCOS is a hormonal disorder.",
                       metadata={"source": "t.txt", "tier": "patient_education"})
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "t.txt"

    def test_large_document_multiple_chunks(self):
        text = "PCOS affects many women. " * 90
        doc = Document(page_content=text, metadata={"source": "big.pdf", "tier": "guideline"})
        chunks = chunk_documents([doc])
        assert len(chunks) >= 2

    def test_metadata_flows_through(self):
        doc = Document(page_content="x " * 150,
                       metadata={"source": "g.pdf", "tier": "research", "page": 3})
        chunks = chunk_documents([doc])
        for c in chunks:
            assert c.metadata["source"] == "g.pdf"
            assert c.metadata["tier"] == "research"

    def test_chunk_size_respected(self):
        doc = Document(page_content="word " * 400, metadata={"source": "x.txt", "tier": "handcrafted"})
        chunks = chunk_documents([doc])
        for c in chunks:
            assert len(c.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP + 50

    def test_empty_list(self):
        assert chunk_documents([]) == []
