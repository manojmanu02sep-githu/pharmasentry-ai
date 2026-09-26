#!/usr/bin/env python3
"""Build the persistent BM25 and FAISS vector indexes from the synthetic
golden dataset, and save them to the paths configured in Settings.

Uses whichever embedding provider is configured (`EMBEDDING_PROVIDER`,
default "deterministic" — offline, no model download). Run this after any
change to the golden dataset, the embedding provider/model, or the
retrieval corpus construction logic, before relying on the persisted
indexes (e.g. before `scripts/run_retrieval_evaluation.py` against a
persisted index, or before the app serves duplicate search from disk
rather than an in-memory corpus).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import get_settings  # noqa: E402
from src.retrieval.bm25_index import PersistentBm25Index  # noqa: E402
from src.retrieval.corpus import build_corpus_from_golden_dataset  # noqa: E402
from src.retrieval.embeddings import get_embedding_provider  # noqa: E402
from src.retrieval.faiss_index import PersistentFaissIndex  # noqa: E402


def run() -> dict[str, object]:
    settings = get_settings()
    corpus = build_corpus_from_golden_dataset()

    bm25_index = PersistentBm25Index(corpus)
    bm25_index.save(settings.bm25_index_path)

    provider = get_embedding_provider(
        settings.embedding_provider.value, settings.embedding_model
    )
    vectors = [provider.embed(doc.text) for doc in corpus]
    faiss_index = PersistentFaissIndex.build(
        case_ids=[doc.case_id for doc in corpus], vectors=vectors, provider=provider
    )
    faiss_index.save(settings.vector_index_path)

    return {
        "documents_indexed": len(corpus),
        "embedding_provider": provider.name,
        "embedding_dim": provider.dim,
        "bm25_index_path": settings.bm25_index_path,
        "vector_index_path": settings.vector_index_path,
    }


def main() -> int:
    try:
        summary = run()
    except Exception as exc:  # noqa: BLE001 - top-level script error reporting
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
