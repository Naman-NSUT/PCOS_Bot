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

    # NOTE: the old collection is deliberately NOT deleted yet.
    #
    # Chunking embeds the whole corpus, so it is the first thing that touches
    # the provider — a bad key, model id, or base URL surfaces there. Deleting
    # first meant any such failure destroyed the knowledge base with nothing to
    # replace it, leaving the app with an empty index. Build the new chunks
    # first; only drop the old collection once we have something to write.

    print("Loading documents…")
    docs = load_all_documents()
    print(f"  {len(docs)} document(s) loaded.")

    print("Chunking…")
    chunks = chunk_documents(docs)
    print(f"  {len(chunks)} chunks produced.")

    # Drop blank chunks. Semantic splitting can emit whitespace-only fragments,
    # which carry no retrievable signal — and embedding APIs reject empty input
    # (OpenRouter returns 400 "expected string to have >=1 characters"), which
    # would otherwise abort the whole run.
    kept = [c for c in chunks if c.page_content and c.page_content.strip()]
    dropped = len(chunks) - len(kept)
    if dropped:
        print(f"  {dropped} blank chunk(s) discarded.")
    chunks = kept

    if not chunks:
        print("\n  Nothing to ingest — no non-empty chunks.\n")
        return

    # Chunking succeeded, so the provider is reachable and the corpus is ready.
    # Only now is it safe to drop the previous embeddings.
    if count > 0:
        print("  --force used. Removing previous embeddings…")
        get_vector_store().delete_collection()
        # Clear the lru_cache so get_vector_store() builds a fresh collection.
        get_vector_store.cache_clear()

    print("Embedding and storing…")
    store = get_vector_store()

    # Write in batches so a failure partway through doesn't discard the work
    # already embedded (the chunking pass alone costs a full pass over the corpus).
    BATCH = 100
    stored = 0
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        store.add_documents(batch)
        stored += len(batch)
        print(f"  stored {stored}/{len(chunks)}", flush=True)

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
