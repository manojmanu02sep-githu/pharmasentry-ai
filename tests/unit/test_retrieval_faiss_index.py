"""Unit tests for src/retrieval/faiss_index.py.

Skipped automatically if faiss-cpu is not installed in the current
environment (it is an optional dependency; only the deterministic
embedding provider is required by default — see src/retrieval/embeddings.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("faiss")

from src.retrieval.embeddings import DeterministicEmbeddingProvider  # noqa: E402
from src.retrieval.faiss_index import (  # noqa: E402
    FaissIndexMismatchError,
    PersistentFaissIndex,
)

PROVIDER = DeterministicEmbeddingProvider()
CASE_IDS = ["case_a", "case_b", "case_c"]
TEXTS = [
    "severe abdominal pain and vomiting after DemoGlutide",
    "mild headache after DemoBasalin",
    "another severe abdominal pain case with DemoGlutide",
]


def _build() -> PersistentFaissIndex:
    vectors = [PROVIDER.embed(t) for t in TEXTS]
    return PersistentFaissIndex.build(case_ids=CASE_IDS, vectors=vectors, provider=PROVIDER)


def test_search_ranks_semantically_closest_case_highest() -> None:
    index = _build()
    query_vector = PROVIDER.embed("severe abdominal pain and vomiting")
    results = index.search(query_vector, top_k=3)
    assert results[0].case_id in {"case_a", "case_c"}


def test_search_on_empty_index_returns_empty() -> None:
    index = PersistentFaissIndex.build(case_ids=[], vectors=[], provider=PROVIDER)
    assert index.search(PROVIDER.embed("anything"), top_k=5) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = _build()
    index.save(tmp_path)

    loaded = PersistentFaissIndex.load(tmp_path, expected_provider=PROVIDER.name)
    assert loaded.case_ids == CASE_IDS
    assert loaded.dim == PROVIDER.dim

    query_vector = PROVIDER.embed("severe abdominal pain")
    original_results = index.search(query_vector, top_k=3)
    loaded_results = loaded.search(query_vector, top_k=3)
    assert [r.case_id for r in original_results] == [r.case_id for r in loaded_results]


def test_load_with_mismatched_provider_raises(tmp_path: Path) -> None:
    index = _build()
    index.save(tmp_path)

    with pytest.raises(FaissIndexMismatchError):
        PersistentFaissIndex.load(tmp_path, expected_provider="sentence_transformers")
