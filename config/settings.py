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
    "lab_guidance":      KNOWLEDGE_BASE_DIR / "lab_guidance",
}

CHROMA_PERSIST_DIR     = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))
CHROMA_COLLECTION_NAME = "pcos_knowledge"

# ── LLM Models (OpenAI-compatible API) ─────────────────────────────────────
# Any OpenAI-compatible endpoint works: OpenAI itself, OpenRouter, Together,
# a local vLLM/LM Studio server, etc.  Point OPENAI_BASE_URL at the provider
# and set OPENAI_API_KEY to that provider's key.
#
#   OpenAI     → https://api.openai.com/v1      models like "gpt-4o-mini"
#   OpenRouter → https://openrouter.ai/api/v1   models like "openai/gpt-4o-mini"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # 1536-dim
GRADER_MODEL    = os.getenv("GRADER_MODEL",    "openai/gpt-4o-mini")   # per-chunk grading
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-4o-mini")   # narrative + conversation
VISION_MODEL    = os.getenv("VISION_MODEL",    "openai/gpt-4o-mini")   # report photo OCR only

# Report photos are token-expensive (~25k input tokens at full resolution),
# so images are downscaled before sending. See src/nodes/transcribe_report.py.
MAX_REPORT_IMAGES = int(os.getenv("MAX_REPORT_IMAGES", "6"))

# Embedding dimensionality of EMBEDDING_MODEL.  Changing the embedding model
# changes this, and the Chroma collection must be re-ingested from scratch —
# querying a collection with differently-sized vectors fails outright.
EMBEDDING_DIM = 1536

# ── Ingestion ──────────────────────────────────────────────────────────────
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# ── CRAG ───────────────────────────────────────────────────────────────────
RETRIEVAL_K          = 8
MAX_RETRIEVAL_PASSES = 2   # original + one rewrite pass

# ── Memory (cross-session personalisation) ─────────────────────────────────
# SQLite by default: zero running cost, zero added latency, and no health data
# leaves the machine. Swap for a postgresql:// URL when you outgrow it.
MEMORY_DB_URL = os.getenv("MEMORY_DB_URL", f"sqlite:///{BASE_DIR / 'memory.db'}")
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

# Signs the user identity token. MUST be set in production — a predictable secret
# lets anyone forge another person's user id and read their health record.
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "")
if not APP_SECRET_KEY:
    import secrets as _secrets
    APP_SECRET_KEY = _secrets.token_hex(32)
    _EPHEMERAL_SECRET = True   # regenerated each boot -> tokens die on restart
else:
    _EPHEMERAL_SECRET = False

# ── API ────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8003

# ── LangSmith ──────────────────────────────────────────────────────────────
LANGSMITH_API_KEY      = os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "pcos-bot")
LANGSMITH_TRACING      = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

SAFETY_DISCLAIMER = (
    "⚠️ DISCLAIMER: This analysis is for informational purposes only and is NOT a "
    "medical diagnosis. PCOS can only be diagnosed by a qualified healthcare professional. "
    "Please consult your doctor or gynaecologist/endocrinologist for personalised advice."
)
