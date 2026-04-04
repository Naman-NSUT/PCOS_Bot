# PCOS Bot — CRAG + LangGraph Diagnostic Pipeline

AI-powered PCOS lab report analyser using **Corrective RAG (CRAG)** as a **LangGraph StateGraph**, powered by **Gemini API**. No ML model — pure rule-based diagnostics grounded in the 2023 PCOS guideline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set Gemini API key
cp .env.example .env   # add your GEMINI_API_KEY

# 3. Ingest knowledge base
python3 -m src.ingestion.run_ingest

# 4. Start the API
uvicorn src.api.main:app --reload --port 8000
```

Analyse a report:
```bash
curl -X POST http://localhost:8000/analyze-report \
  -H "Content-Type: application/json" \
  -d '{"report_text": "LH: 12.4 mIU/mL, FSH: 5.1 mIU/mL, Testosterone: 85 ng/dL, AMH: 7.2 ng/mL", "patient_context": "27F, irregular periods"}'
```

Interactive docs: **http://localhost:8000/docs**

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
│   └── handcrafted_knowledge/       # Add custom documents here
│
├── src/
│   ├── state/
│   │   ├── crag_state.py            # TypedDict: CRAGState
│   │   └── analysis_state.py        # TypedDict: AnalysisState
│   │
│   ├── nodes/                       # One function per graph node
│   │   ├── retrieve.py              # CRAG Step 1 — ChromaDB retrieval
│   │   ├── grade.py                 # CRAG Step 2 — Gemini Flash grading
│   │   ├── rewrite.py               # CRAG Step 3 — Gemini query rewrite
│   │   ├── refine.py                # CRAG Step 4 — chunk filtering
│   │   ├── assemble.py              # CRAG Step 5 — dedup + rank + cite
│   │   ├── parse_report.py          # Analysis Node 1 — regex parser + reference ranges
│   │   ├── flag_indicators.py       # Analysis Node 2 — rule-based PCOS flags
│   │   └── generate_narrative.py    # Analysis Node 3 — Gemini Pro narrative
│   │
│   ├── graphs/
│   │   ├── crag_graph.py            # LangGraph StateGraph (CRAG, 5 nodes)
│   │   └── analysis_graph.py        # LangGraph StateGraph (Analysis, 4 nodes)
│   │
│   ├── ingestion/
│   │   ├── loader.py                # PDF + TXT loader with tier tagging
│   │   ├── chunker.py               # RecursiveCharacterTextSplitter
│   │   ├── embedder.py              # text-embedding-004 singleton
│   │   ├── vector_store.py          # ChromaDB persistent store
│   │   └── run_ingest.py            # CLI: python3 -m src.ingestion.run_ingest
│   │
│   └── api/
│       ├── main.py                  # FastAPI app (lifespan warm-up)
│       └── routes/analyze.py        # POST /analyze-report
│
├── tests/
│   ├── test_nodes.py                # 19 tests for parse, flag, refine, assemble
│   ├── test_crag_graph.py           # 6 tests for graph logic + integration
│   └── test_ingestion.py            # 5 tests for chunker
│
├── conftest.py
├── requirements.txt
└── .env.example
```

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
python3 -m pytest tests/ -v   # 30 tests, no API key required
```

## Re-ingest after adding documents

```bash
python3 -m src.ingestion.run_ingest --force
```
