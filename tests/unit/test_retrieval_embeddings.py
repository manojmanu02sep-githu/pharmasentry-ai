"""Unit tests for src/retrieval/embeddings.py."""

from __future__ import annotations

import pytest

from src.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)


def test_get_embedding_provider_deterministic() -> None:
    provider = get_embedding_provider("deterministic")
    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert provider.name == "deterministic"


def test_get_embedding_provider_sentence_transformers_requires_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        get_embedding_provider("sentence_transformers")


def test_get_embedding_provider_sentence_transformers_returns_lazy_provider() -> None:
    provider = get_embedding_provider("sentence_transformers", model_name="some-model")
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)
    assert provider._model is None  # never loaded until .embed() is called


def test_get_embedding_provider_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown embedding provider"):
        get_embedding_provider("nonsense")


def test_deterministic_provider_embed_is_reproducible_and_normalized() -> None:
    provider = DeterministicEmbeddingProvider()
    v1 = provider.embed("severe abdominal pain")
    v2 = provider.embed("severe abdominal pain")
    assert v1 == v2
    norm = sum(x * x for x in v1) ** 0.5
    assert abs(norm - 1.0) < 1e-9
