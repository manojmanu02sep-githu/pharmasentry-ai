"""Embedding provider abstraction: deterministic (offline, default) or
sentence-transformers (real semantic embeddings, optional dependency).

Mirrors the existing `llm_provider` mock/real pattern (`config/settings.py`)
so tests and CI stay fast and fully offline by default, while production
deployments can opt into real embeddings via `EMBEDDING_PROVIDER`.
sentence-transformers is imported lazily so this module — and everything
that imports it — works even when that optional dependency is not
installed; only actually calling `.embed()` on the sentence-transformers
provider requires it, and it raises a clear ImportError rather than
silently falling back (CLAUDE.md: never fabricate results).
"""

from __future__ import annotations

from typing import Any, Protocol

from src.tools.retrieval import deterministic_embedding


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> list[float]: ...


class DeterministicEmbeddingProvider:
    """Wraps src.tools.retrieval.deterministic_embedding — the same
    hashed-bag-of-words embedding already used by VectorSearchTool, kept
    as a single source of truth for the "test embedding" implementation."""

    name = "deterministic"
    dim = 128

    def embed(self, text: str) -> list[float]:
        return deterministic_embedding(text, dim=self.dim)


class SentenceTransformerEmbeddingProvider:
    """Real semantic embeddings via sentence-transformers."""

    name = "sentence_transformers"
    dim = 384

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed; install it "
                    "(see requirements.txt) or set EMBEDDING_PROVIDER=deterministic"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]


def get_embedding_provider(provider_name: str, model_name: str | None = None) -> EmbeddingProvider:
    if provider_name == "deterministic":
        return DeterministicEmbeddingProvider()
    if provider_name == "sentence_transformers":
        if not model_name:
            raise ValueError("model_name is required for the sentence_transformers provider")
        return SentenceTransformerEmbeddingProvider(model_name)
    raise ValueError(f"unknown embedding provider: {provider_name!r}")
