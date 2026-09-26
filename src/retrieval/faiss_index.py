"""Persistent FAISS vector index.

Uses `IndexFlatIP` (inner product): embeddings from every provider in
`src.retrieval.embeddings` are L2-normalized, so inner product is
equivalent to cosine similarity — consistent with the `_cosine` helper in
`src.tools.retrieval`. `faiss` is imported lazily inside methods (not at
module level) so this module can be imported and type-checked in
environments where `faiss-cpu` is not installed; only building, saving,
loading, or searching an actual index requires it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.retrieval.embeddings import EmbeddingProvider
from src.tools.retrieval import ScoredCandidate

_INDEX_FILENAME = "index.faiss"
_META_FILENAME = "meta.json"


class FaissIndexMismatchError(RuntimeError):
    """The loaded index's embedding provider or dimension does not match
    what the caller expected — never searched anyway rather than silently
    returning meaningless scores."""


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is not installed; install it (see requirements.txt) "
            "to build, save, load, or search a persistent vector index"
        ) from exc
    return faiss


class PersistentFaissIndex:
    def __init__(self, case_ids: list[str], index: Any, dim: int, provider_name: str) -> None:
        self.case_ids = case_ids
        self._index = index
        self.dim = dim
        self.provider_name = provider_name

    @classmethod
    def build(
        cls, case_ids: list[str], vectors: list[list[float]], provider: EmbeddingProvider
    ) -> PersistentFaissIndex:
        faiss = _import_faiss()
        import numpy as np

        if len(case_ids) != len(vectors):
            raise ValueError("case_ids and vectors must be the same length")
        index = faiss.IndexFlatIP(provider.dim)
        if vectors:
            index.add(np.array(vectors, dtype="float32"))
        return cls(
            case_ids=list(case_ids), index=index, dim=provider.dim, provider_name=provider.name
        )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[ScoredCandidate]:
        import numpy as np

        if not self.case_ids:
            return []
        query = np.array([query_vector], dtype="float32")
        scores, indices = self._index.search(query, min(top_k, len(self.case_ids)))
        results = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            results.append(
                ScoredCandidate(case_id=self.case_ids[idx], score=round(float(score), 6))
            )
        return results

    def save(self, index_dir: str | Path) -> None:
        faiss = _import_faiss()
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILENAME))
        meta = {"case_ids": self.case_ids, "dim": self.dim, "provider_name": self.provider_name}
        (path / _META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(
        cls, index_dir: str | Path, expected_provider: str | None = None
    ) -> PersistentFaissIndex:
        faiss = _import_faiss()
        path = Path(index_dir)
        meta = json.loads((path / _META_FILENAME).read_text(encoding="utf-8"))
        if expected_provider is not None and meta["provider_name"] != expected_provider:
            raise FaissIndexMismatchError(
                f"index was built with provider {meta['provider_name']!r}, "
                f"but {expected_provider!r} was requested"
            )
        index = faiss.read_index(str(path / _INDEX_FILENAME))
        return cls(
            case_ids=meta["case_ids"],
            index=index,
            dim=meta["dim"],
            provider_name=meta["provider_name"],
        )
