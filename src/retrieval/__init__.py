"""Hybrid RAG: BM25 + vector similarity + metadata filtering, combined by a
documented deterministic weighted-RRF method. Never makes the final
duplicate or regulatory decision — see CLAUDE.md's Hybrid RAG section.
"""

from __future__ import annotations

from src.retrieval.bm25_index import PersistentBm25Index
from src.retrieval.corpus import build_corpus_from_golden_dataset
from src.retrieval.duplicate_search import DuplicateSearchService
from src.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
    get_embedding_provider,
)
from src.retrieval.faiss_index import FaissIndexMismatchError, PersistentFaissIndex
from src.retrieval.hybrid_ranker import (
    HybridRanker,
    WeightedRanking,
    weighted_reciprocal_rank_fusion,
)
from src.retrieval.retrieval_evaluation import RetrievalEvaluationResult, run_retrieval_evaluation

__all__ = [
    "DeterministicEmbeddingProvider",
    "DuplicateSearchService",
    "EmbeddingProvider",
    "FaissIndexMismatchError",
    "HybridRanker",
    "PersistentBm25Index",
    "PersistentFaissIndex",
    "RetrievalEvaluationResult",
    "SentenceTransformerEmbeddingProvider",
    "WeightedRanking",
    "build_corpus_from_golden_dataset",
    "get_embedding_provider",
    "run_retrieval_evaluation",
    "weighted_reciprocal_rank_fusion",
]
