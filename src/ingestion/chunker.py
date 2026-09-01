"""
src/ingestion/chunker.py — Split documents into semantic chunks.
"""
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from src.ingestion.embedder import get_embeddings
from typing import List


def chunk_documents(documents: List[Document]) -> List[Document]:
    embeddings = get_embeddings()
    splitter = SemanticChunker(embeddings)
    return splitter.split_documents(documents)
