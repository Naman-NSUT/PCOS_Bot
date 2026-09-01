"""
tests/test_ingestion.py
Tests for the ingestion module (chunker).

The chunker uses SemanticChunker, which embeds text to find split points.
Embeddings are mocked here so the suite stays offline and costs nothing —
never let these tests reach a real embedding endpoint.
"""
import pytest
from unittest.mock import patch
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


class _FakeEmbeddings:
    """
    Deterministic offline embedder.

    Vectors are derived from topic keywords so that sentences about the same
    topic sit close together and sentences about different topics sit far
    apart — enough structure for SemanticChunker to behave meaningfully
    without any network call.
    """

    _TOPICS = ("hormone", "insulin", "ultrasound")

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @classmethod
    def _vec(cls, text):
        low = text.lower()
        vec = [0.05] * len(cls._TOPICS)
        for i, topic in enumerate(cls._TOPICS):
            if topic in low:
                vec[i] = 1.0
        return vec


@pytest.fixture(autouse=True)
def _mock_embeddings():
    """Ensure no chunker test ever hits a live embedding API."""
    with patch("src.ingestion.chunker.get_embeddings", return_value=_FakeEmbeddings()):
        yield


class TestChunker:
    def test_short_document_one_chunk(self):
        doc = Document(page_content="PCOS is a hormonal disorder.",
                       metadata={"source": "t.txt", "tier": "patient_education"})
        chunks = chunk_documents([doc])
        assert len(chunks) >= 1
        assert chunks[0].metadata["source"] == "t.txt"

    def test_metadata_flows_through(self):
        doc = Document(page_content="Hormone levels vary. " * 40,
                       metadata={"source": "g.pdf", "tier": "research", "page": 3})
        chunks = chunk_documents([doc])
        assert chunks
        for c in chunks:
            assert c.metadata["source"] == "g.pdf"
            assert c.metadata["tier"] == "research"
            assert c.metadata["page"] == 3

    def test_content_is_preserved(self):
        """Chunking must not silently drop text."""
        sentences = [
            "Hormone testing is the first step.",
            "Insulin resistance is common in PCOS.",
            "Ultrasound shows follicle counts.",
        ]
        doc = Document(page_content=" ".join(sentences),
                       metadata={"source": "x.txt", "tier": "handcrafted"})
        chunks = chunk_documents([doc])
        rejoined = " ".join(c.page_content for c in chunks)
        for s in sentences:
            assert s in rejoined

    def test_multiple_documents_all_represented(self):
        docs = [
            Document(page_content="Hormone panel details.",
                     metadata={"source": "a.txt", "tier": "guideline"}),
            Document(page_content="Insulin resistance details.",
                     metadata={"source": "b.txt", "tier": "research"}),
        ]
        chunks = chunk_documents(docs)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"a.txt", "b.txt"}

    def test_empty_list(self):
        assert chunk_documents([]) == []
