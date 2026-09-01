# PCOS Bot — CRAG + LangGraph Diagnostic Pipeline

AI-powered PCOS lab report analyser using **Corrective RAG (CRAG)** as a **LangGraph StateGraph**, powered by any **OpenAI-compatible API**. No ML model — pure rule-based diagnostics grounded in the 2023 PCOS guideline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env   # add your OPENAI_API_KEY

# 3. Ingest knowledge base
python3 -m src.ingestion.run_ingest

# 4. Start the API
uvicorn src.api.main:app --reload --port 8003
```

### Provider configuration

Any OpenAI-compatible endpoint works — set two variables in `.env`:

| Provider | `OPENAI_BASE_URL` | Model naming |
|----------|-------------------|--------------|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter (default) | `https://openrouter.ai/api/v1` | `openai/gpt-4o-mini` |
| Local vLLM / LM Studio | `http://localhost:8000/v1` | server-specific |

Models are overridable via `GRADER_MODEL`, `GENERATOR_MODEL`, and `EMBEDDING_MODEL`.

> ⚠️ **Changing `EMBEDDING_MODEL` changes the vector dimensionality.** ChromaDB
> cannot query a collection whose vectors are a different size, so you must
> re-ingest from scratch: `python3 -m src.ingestion.run_ingest --force`

Analyse a report:
```bash
curl -X POST http://localhost:8003/analyze-report \
  -H "Content-Type: application/json" \
  -d '{"report_text": "LH: 12.4 mIU/mL, FSH: 5.1 mIU/mL, Testosterone: 85 ng/dL, AMH: 7.2 ng/mL", "patient_context": "27F, irregular periods"}'
```

Interactive docs: **http://localhost:8003/docs**

---

## Project Structure

```
PCOS_Bot/
├── config/
│   └── settings.py                  # All config, loaded from .env
│
├── knowledge_base/
│   ├── guidelines/                  # 3 PDFs — 2023 PCOS guideline
│   ├── research_papers/             # 9 research paper PDFs
│   ├── patient_education/           # 5 patient education TXTs
│   ├── lab_guidance/                # Lab test awareness notes
│   └── handcrafted_knowledge/       # Add custom documents here
│
├── src/
│   ├── state/
│   │   ├── crag_state.py            # TypedDict: CRAGState
│   │   └── analysis_state.py        # TypedDict: AnalysisState
│   │
│   ├── nodes/                       # One function per graph node
│   │   ├── retrieve.py              # CRAG Step 1 — ChromaDB retrieval
│   │   ├── grade.py                 # CRAG Step 2 — LLM batch grading (JSON mode)
│   │   ├── rewrite.py               # CRAG Step 3 — LLM query rewrite
│   │   ├── refine.py                # CRAG Step 4 — chunk filtering
│   │   ├── assemble.py              # CRAG Step 5 — dedup + rank + cite
│   │   ├── parse_report.py          # Analysis Node 1 — regex parser + reference ranges
│   │   ├── flag_indicators.py       # Analysis Node 2 — rule-based PCOS flags
│   │   └── generate_narrative.py    # Analysis Node 3 — LLM narrative
│   │
│   ├── graphs/
│   │   ├── crag_graph.py            # LangGraph StateGraph (CRAG, 5 nodes)
│   │   └── analysis_graph.py        # LangGraph StateGraph (Analysis, 4 nodes)
│   │
│   ├── ingestion/
│   │   ├── loader.py                # PDF + TXT loader with tier tagging
│   │   ├── chunker.py               # SemanticChunker (embedding-based splits)
│   │   ├── embedder.py              # OpenAI-compatible embedder singleton
│   │   ├── vector_store.py          # ChromaDB persistent store
│   │   └── run_ingest.py            # CLI: python3 -m src.ingestion.run_ingest
│   │
│   ├── memory/
│   │   ├── models.py                # users / facts (bi-temporal) / consultations
│   │   ├── store.py                 # persistence API — zero LLM calls
│   │   └── hydration.py             # MemorySnapshot + prompt block + NEVER_HYDRATE
│   │
│   ├── core/
│   │   ├── session.py               # Session model + phase state machine
│   │   ├── identity.py              # HMAC-signed durable user tokens
│   │   ├── intent_detector.py       # Regex/keyword extraction (no LLM)
│   │   └── llm_client.py            # Prompt builder + chat client
│   │
│   ├── agents/
│   │   └── conversation_agent.py    # Phase-driven consultation orchestrator
│   │
│   └── api/
│       ├── main.py                  # FastAPI app (lifespan warm-up)
│       ├── routes/analyze.py        # POST /analyze-report
│       └── routes/chat.py           # POST /chat, /chat/new, DELETE /chat/memory
│
├── tests/                           # 222 tests, fully mocked
│   ├── test_nodes.py                # parse, flag, refine, assemble
│   ├── test_crag_graph.py           # graph logic + integration
│   ├── test_grade_node.py           # grader payload robustness
│   ├── test_ingestion.py            # chunker (embeddings mocked)
│   ├── test_session.py              # phase state machine
│   ├── test_intent_detector.py      # regex extraction
│   ├── test_memory.py               # store, identity, hydration
│   ├── test_hydration_conflicts.py  # returning-user regressions
│   └── test_memory_security.py      # hijack, leak, fabrication regressions
│
├── conftest.py
├── requirements.txt
└── .env.example
```

---

## Long-term Memory

Cross-session personalisation with **no third-party memory vendor**: SQLite plus a
bi-temporal fact table. Zero added LLM calls, zero added network latency, and no
health data leaves the machine.

The reason it needs no vendor: the app already extracts its facts deterministically.
Symptom tags come from regex intent detection, guidance topics and lab tests from
lookup tables. The LLM fact-extraction that hosted memory services bill for is work
this codebase has already done for free.

```
users           user_id, preferred_name, age_range, gender, consultation_count
facts           (user) --[predicate]--> (object), bi-temporal
consultations   per-visit record: phase reached, closure mode, exchange count
```

`facts` is the graph layer — an edge list rooted at the user, carrying two
independent timelines:

| Column | Meaning |
|--------|---------|
| `valid_at` / `invalid_at` | when the fact was true **in the world** |
| `recorded_at` | when the system **learned** it |

That separation is what lets a symptom stop being true. "My periods are regular
again" sets `invalid_at` on the `irregular_periods` edge: it stops being a current
symptom, and the history stays queryable. An append-only memory would keep both
statements alive and let retrieval ranking pick a winner.

### Identity

The server mints an HMAC-signed `user_token` (stdlib `hmac`, no new dependency).
Clients send it back to be recognised. A client-supplied id is never trusted — that
would turn an unguessable token into a forgeable permanent key to a health record.
Session ownership is checked on every turn: knowing a `session_id` is not enough.

```bash
# REQUIRED in production, else tokens are signed with a random per-boot key
python3 -c 'import secrets; print(secrets.token_hex(32))'   # -> APP_SECRET_KEY
```

### Design rule

Retrieved memory is **advisory prose only**. It never writes into
`session.symptom_list`, because `symptom_count` gates the PCOS disclosure rules and
the lab-test rules. Symptom state for the current consultation comes only from what
the user says now; memory informs tone, continuity, and what not to repeat.

Per-session pacing counters are never restored — hydrating them fired an instant
escalation or closure on a returning user's first message. See
`src/memory/hydration.py::NEVER_HYDRATE` and `tests/test_hydration_conflicts.py`.

### Erasure

```bash
curl -X DELETE http://localhost:8003/chat/memory \
  -H 'Content-Type: application/json' -d '{"user_token": "..."}'
```

Deletes the user, every fact and consultation (cascade), and any live in-memory
session. Owning the store is what makes this three lines rather than a vendor ticket.

---

## LangGraph Architecture

### CRAG Graph

```
retrieve → grade ──┬──(AMBIGUOUS/all IRRELEVANT + passes < 2)──▶ rewrite ─┐
                   │                                                         │
                   └──(RELEVANT / max passes)──▶ refine → assemble → END ◀─┘
```

- **State**: `CRAGState` TypedDict (query, active_query, chunks, graded_chunks, refined_chunks, context, sources…)
- **Conditional edge**: `_should_rewrite()` — loops back to `retrieve` for at most 2 passes total

### Analysis Graph

```
parse_report → flag_indicators → run_crag (sub-graph) → generate_narrative → END
```

- **State**: `AnalysisState` TypedDict
- `run_crag_node` calls the compiled CRAG graph as a sub-graph

---

## Diagnostic Engine (No ML)

| Flag | Trigger |
|------|---------|
| Hyperandrogenism | Testosterone > 82, Free T > 1.9, DHEAS > 430 |
| Insulin Resistance | HOMA-IR > 2.5, Fasting Insulin > 20, Glucose > 99 |
| Polycystic Morphology Proxy | AMH > 3.5, LH/FSH ≥ 2.0 |
| Thyroid | TSH out of 0.4–4.0 range |
| Prolactin | Prolactin > 25 |
| Metabolic Risk | Score 0–4 (cholesterol, TG, HDL, waist, BMI) |
| Critical | Any value > 1.5× upper or < 0.7× lower limit |

> ⚠️ **Indicators only — not a diagnosis. Always refer to a qualified healthcare provider.**

---

## Running Tests

```bash
python3 -m pytest tests/ -v   # 222 tests, no API key required (all LLM/embedding calls mocked)
```

## Re-ingest after adding documents

```bash
python3 -m src.ingestion.run_ingest --force
```
