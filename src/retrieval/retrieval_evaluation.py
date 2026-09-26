"""Retrieval evaluation: precision@5, recall@5, and MRR for BM25-only,
vector-only, and hybrid duplicate retrieval (CLAUDE.md Hybrid RAG
requirement).

Methodology: evaluated only over golden-dataset cases that actually HAVE a
duplicate to find (non-empty `expected_duplicate_matches`), against the
FULL 138-case dataset rather than only the held-out test split. This
differs from agent/LLM evaluation, which must hold out data to avoid
overfitting: BM25/vector/RRF have no learned parameters here, so nothing
can overfit by seeing more of the corpus — using the full dataset just
gives a less noisy, real number. Mirrors the duplicate-retrieval baseline
methodology in `scripts/run_golden_evaluation.py`, replacing its naive
case_id-sorted baseline with real BM25/vector/hybrid retrieval.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from src.evaluation.golden_loader import load_all_cases
from src.evaluation.metrics import (
    mean_duplicate_precision_at_k,
    mean_duplicate_recall_at_k,
    mean_reciprocal_rank,
)
from src.models.enums import AgentName
from src.retrieval.corpus import build_corpus_from_golden_dataset
from src.retrieval.embeddings import DeterministicEmbeddingProvider, EmbeddingProvider
from src.retrieval.hybrid_ranker import WeightedRanking, weighted_reciprocal_rank_fusion
from src.tools.base import ToolContext
from src.tools.retrieval import (
    Bm25SearchInput,
    Bm25SearchTool,
    CorpusDocument,
    VectorSearchInput,
    VectorSearchTool,
)

logger = logging.getLogger(__name__)

_EVAL_TOP_K = 5
_EVAL_FUSION_BREADTH = 10
_EVAL_CTX = ToolContext(
    case_id="eval",
    trace_id="eval",
    agent=AgentName.DUPLICATE_SEARCH,
    authorized_tools=frozenset({"bm25_search", "vector_search"}),
)


class RetrievalEvaluationResult(BaseModel):
    method: str
    queries_evaluated: int
    precision_at_5: float
    recall_at_5: float
    mrr: float


def _run_queries(
    corpus: list[CorpusDocument],
    bm25_weight: float,
    vector_weight: float,
    embedding_provider: EmbeddingProvider,
) -> dict[str, list[tuple[list[str], set[str]]]]:
    cases = load_all_cases()
    by_id = {doc.case_id: doc for doc in corpus}
    bm25_tool = Bm25SearchTool()
    vector_tool = VectorSearchTool(embed_fn=embedding_provider.embed)

    per_method: dict[str, list[tuple[list[str], set[str]]]] = {
        "bm25": [],
        "vector": [],
        "hybrid": [],
    }

    for case in cases:
        relevant = set(case.expected_duplicate_matches)
        if not relevant:
            continue
        query_doc = by_id.get(case.case_id)
        if query_doc is None:
            continue
        rest_corpus = [doc for doc in corpus if doc.case_id != case.case_id]

        bm25_output, _ = bm25_tool.run(
            Bm25SearchInput(
                query_text=query_doc.text, corpus=rest_corpus, top_k=_EVAL_FUSION_BREADTH
            ),
            _EVAL_CTX,
        )
        vector_output, _ = vector_tool.run(
            VectorSearchInput(
                query_text=query_doc.text, corpus=rest_corpus, top_k=_EVAL_FUSION_BREADTH
            ),
            _EVAL_CTX,
        )
        bm25_ranking = [r.case_id for r in bm25_output.results]
        vector_ranking = [r.case_id for r in vector_output.results]
        hybrid_ranking = [
            cid
            for cid, _ in weighted_reciprocal_rank_fusion(
                [
                    WeightedRanking(case_ids=bm25_ranking, weight=bm25_weight),
                    WeightedRanking(case_ids=vector_ranking, weight=vector_weight),
                ],
                top_k=_EVAL_FUSION_BREADTH,
            )
        ]

        per_method["bm25"].append((bm25_ranking, relevant))
        per_method["vector"].append((vector_ranking, relevant))
        per_method["hybrid"].append((hybrid_ranking, relevant))

    return per_method


def run_retrieval_evaluation(
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, RetrievalEvaluationResult]:
    provider = embedding_provider or DeterministicEmbeddingProvider()
    corpus = build_corpus_from_golden_dataset()
    logger.info(
        "retrieval_evaluation provider=%s corpus_size=%d", provider.name, len(corpus)
    )
    per_method = _run_queries(corpus, bm25_weight, vector_weight, provider)

    return {
        method: RetrievalEvaluationResult(
            method=method,
            queries_evaluated=len(queries),
            precision_at_5=mean_duplicate_precision_at_k(queries, k=_EVAL_TOP_K),
            recall_at_5=mean_duplicate_recall_at_k(queries, k=_EVAL_TOP_K),
            mrr=mean_reciprocal_rank(queries),
        )
        for method, queries in per_method.items()
    }
