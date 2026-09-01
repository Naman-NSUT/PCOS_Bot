"""
src/api/main.py — FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import API_HOST, API_PORT, LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_TRACING
from src.api.routes.analyze import router as analyze_router
from src.api.routes.chat import router as chat_router

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PCOS Bot API…")

    # ── LangSmith tracing ───────────────────────────────────────────────
    if LANGSMITH_TRACING and LANGSMITH_API_KEY:
        import os
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY",    LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_PROJECT",    LANGSMITH_PROJECT)
        logger.info("LangSmith tracing ENABLED → project='%s'", LANGSMITH_PROJECT)
    else:
        logger.info("LangSmith tracing DISABLED (set LANGCHAIN_TRACING_V2=true to enable)")

    # ── Long-term memory store ──────────────────────────────────────────
    try:
        from config.settings import MEMORY_ENABLED, _EPHEMERAL_SECRET
        if MEMORY_ENABLED:
            from src.memory.store import init_store
            init_store()
            if _EPHEMERAL_SECRET:
                logger.warning(
                    "APP_SECRET_KEY not set — user tokens are signed with a random "
                    "per-boot key, so every user looks new after a restart. Set "
                    "APP_SECRET_KEY in .env to make memory durable."
                )
        else:
            logger.info("Long-term memory DISABLED (MEMORY_ENABLED=false)")
    except Exception as exc:
        logger.error("Memory store init failed — continuing without memory: %s", exc)

    try:
        from src.ingestion.vector_store import chunk_count
        n = chunk_count()
        if n == 0:
            logger.warning("ChromaDB empty — run: python -m src.ingestion.run_ingest")
        else:
            logger.info("ChromaDB ready — %d chunks", n)

        # Pre-compile both LangGraph graphs (avoids cold-start on first request)
        from src.graphs.crag_graph     import build_crag_graph
        from src.graphs.analysis_graph import build_analysis_graph
        build_crag_graph()
        build_analysis_graph()
        logger.info("LangGraph graphs compiled.")
    except Exception as exc:
        logger.error("Startup error: %s", exc)
    yield
    logger.info("PCOS Bot API shutting down.")


app = FastAPI(
    title="PCOS Bot — Maya Health Consultation",
    description=(
        "Phase-driven PCOS health consultation powered by CRAG + LangGraph. "
        "Maya conducts a real clinical-style conversation: intake → symptom exploration → "
        "personalised guidance, all grounded in evidence-based PCOS guidelines."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(analyze_router)
app.include_router(chat_router)


@app.get("/", tags=["Health"])
async def root():
    return {"service": "PCOS Bot Diagnostic API", "version": "1.0.0",
            "endpoints": {"analyze_report": "POST /analyze-report", "docs": "/docs"}}


@app.get("/health", tags=["Health"])
async def health():
    from src.ingestion.vector_store import chunk_count
    try:
        n = chunk_count()
        status = "ok" if n > 0 else "empty — run ingestion"
    except Exception:
        n, status = 0, "error"
    return {"status": "ok", "knowledge_base_chunks": n, "kb_status": status}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)
