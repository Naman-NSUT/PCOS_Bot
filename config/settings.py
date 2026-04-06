"""
config/settings.py — Central project configuration.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"

KNOWLEDGE_TIERS = {
    "guideline":         KNOWLEDGE_BASE_DIR / "guidelines",
    "research":          KNOWLEDGE_BASE_DIR / "research_papers",
    "patient_education": KNOWLEDGE_BASE_DIR / "patient_education",
    "handcrafted":       KNOWLEDGE_BASE_DIR / "handcrafted_knowledge",
}

CHROMA_PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = "pcos_knowledge"

# ── Gemini ─────────────────────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL  = "models/gemini-embedding-001"
GRADER_MODEL     = "gemini-2.5-flash"    # per-chunk grading — fast
GENERATOR_MODEL  = "gemini-2.5-pro"      # final narrative — high quality

# ── Ingestion ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ── CRAG ───────────────────────────────────────────────────────────────────
RETRIEVAL_K          = 8
MAX_RETRIEVAL_PASSES = 2   # original + one rewrite pass

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8001

# ── LangSmith ──────────────────────────────────────────────────────────────
LANGSMITH_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "pcos-bot")
LANGSMITH_TRACING      = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

SAFETY_DISCLAIMER = (
    "⚠️ DISCLAIMER: This analysis is for informational purposes only and is NOT a "
    "medical diagnosis. PCOS can only be diagnosed by a qualified healthcare professional. "
    "Please consult your doctor or gynaecologist/endocrinologist for personalised advice."
)
