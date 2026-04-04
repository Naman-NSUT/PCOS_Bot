"""
src/api/main.py — FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import API_HOST, API_PORT
from src.api.routes.analyze import router as analyze_router

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting PCOS Bot API…")
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
    title="PCOS Bot — Diagnostic Report Analyser",
    description=(
        "CRAG + LangGraph pipeline that parses lab reports, applies rule-based PCOS "
        "indicator flags, retrieves clinical guidelines, and generates an evidence-grounded "
        "narrative via Gemini 1.5 Pro. Not a diagnostic tool."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(analyze_router)


@app.get("/", tags=["Health"])
async def root():
    return {"service": "PCOS Bot Diagnostic API", "version": "0.2.0",
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
