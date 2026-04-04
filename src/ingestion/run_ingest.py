"""
src/ingestion/run_ingest.py
CLI: python -m src.ingestion.run_ingest [--force]
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PCOS knowledge base into ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if chunks exist.")
    args = parser.parse_args()

    from src.ingestion.vector_store import get_vector_store, chunk_count
    from src.ingestion.loader import load_all_documents
    from src.ingestion.chunker import chunk_documents

    count = chunk_count()
    if count > 0 and not args.force:
        print(f"\n  Already ingested ({count} chunks). Use --force to re-ingest.\n")
        return

    print("Loading documents…")
    docs = load_all_documents()
    print(f"  {len(docs)} document(s) loaded.")

    print("Chunking…")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks produced.")

    print("Embedding and storing…")
    store = get_vector_store()
    store.add_documents(chunks)

    by_tier: dict = defaultdict(int)
    for c in chunks:
        by_tier[c.metadata.get("tier", "unknown")] += 1

    print(f"\n── Ingestion Complete ─────────────────────────")
    print(f"  Total chunks: {len(chunks)}")
    for tier, n in by_tier.items():
        print(f"    {tier:<22} {n}")
    print("───────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
