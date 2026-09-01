"""
src/ingestion/loader.py — Load PDFs and TXTs from all knowledge tiers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from config.settings import KNOWLEDGE_TIERS

logger = logging.getLogger(__name__)


def load_all_documents() -> List[Document]:
    """Walk all four knowledge tiers and return tagged Document objects."""
    all_docs: List[Document] = []

    for tier, folder in KNOWLEDGE_TIERS.items():
        folder = Path(folder)
        if not folder.exists():
            logger.warning("Missing folder — skipping: %s", folder)
            continue

        for pdf in folder.glob("*.pdf"):
            try:
                docs = PyPDFLoader(str(pdf)).load()
                for d in docs:
                    d.metadata.update({"tier": tier, "source": pdf.name})
                all_docs.extend(docs)
                logger.info("[%s] PDF '%s' → %d page(s)", tier, pdf.name, len(docs))
            except Exception as exc:
                logger.error("PDF load failed '%s': %s", pdf, exc)

        # Plain-text tiers. Markdown counts: the lab_guidance tier is authored
        # as .md, and globbing only *.txt silently ingested nothing from it.
        for txt in sorted(folder.glob("*.txt")) + sorted(folder.glob("*.md")):
            if txt.name.startswith("."):
                continue
            try:
                docs = TextLoader(str(txt), encoding="utf-8").load()
                for d in docs:
                    d.metadata.update({"tier": tier, "source": txt.name})
                all_docs.extend(docs)
                logger.info("[%s] TEXT '%s' loaded", tier, txt.name)
            except Exception as exc:
                logger.error("TEXT load failed '%s': %s", txt, exc)

    logger.info("Total documents: %d", len(all_docs))
    return all_docs
