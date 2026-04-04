"""
src/ingestion/vector_store.py — ChromaDB persistent store with tier-aware retrieval.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, RETRIEVAL_K
from src.ingestion.embedder import get_embeddings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    store = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embeddings(),
    )
    logger.info("ChromaDB ready — collection='%s'", CHROMA_COLLECTION_NAME)
    return store


def get_retriever(k: int = RETRIEVAL_K, tier_filter: Optional[str] = None) -> VectorStoreRetriever:
    """Build a retriever, optionally filtered by knowledge tier."""
    search_kwargs: dict = {"k": k}
    if tier_filter:
        search_kwargs["filter"] = {"tier": tier_filter}
    return get_vector_store().as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def chunk_count() -> int:
    return get_vector_store()._collection.count()
