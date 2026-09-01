"""
src/ingestion/embedder.py — Singleton OpenAI-compatible embedder.
"""
from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config.settings import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
