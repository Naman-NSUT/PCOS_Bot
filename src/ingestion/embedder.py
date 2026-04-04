"""
src/ingestion/embedder.py — Singleton Google text-embedding-004 wrapper.
"""
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import GEMINI_API_KEY, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GEMINI_API_KEY)
