"""Deterministic weighted Reciprocal Rank Fusion — the "documented
deterministic method" CLAUDE.md's Hybrid RAG section requires for
combining BM25 and vector rankings.

Generalizes `src.tools.retrieval.ReciprocalRankFusionTool`'s unweighted
RRF formula (`score(d) = sum(1/(k+rank+1))`) to
`score(d) = sum(weight_i / (k + rank_i + 1))`; with all weights == 1.0
this reduces to exactly that tool's formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.memory.procedural import read_retrieval_policy
from src.models.audit import MemoryEvent
from src.tools.base import ToolContext


@dataclass
class WeightedRanking:
    case_ids: list[str]
    weight: float


def weighted_reciprocal_rank_fusion(
    rankings: list[WeightedRanking], k: int = 60, top_k: int = 5
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, case_id in enumerate(ranking.case_ids):
            scores[case_id] = scores.get(case_id, 0.0) + ranking.weight / (k + rank + 1)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return [(cid, round(score, 6)) for cid, score in ranked[:top_k]]


class HybridRanker:
    def __init__(
        self, bm25_weight: float = 0.5, vector_weight: float = 0.5, top_k: int = 5, k: int = 60
    ) -> None:
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.top_k = top_k
        self.k = k

    @classmethod
    def from_config(cls, ctx: ToolContext) -> tuple[HybridRanker, MemoryEvent]:
        policy, event = read_retrieval_policy(ctx)
        ranker = cls(
            bm25_weight=float(policy.get("bm25_weight", 0.5)),
            vector_weight=float(policy.get("vector_weight", 0.5)),
            top_k=int(policy.get("top_k", 5)),
        )
        return ranker, event

    def combine(
        self, bm25_ranking: list[str], vector_ranking: list[str]
    ) -> list[tuple[str, float]]:
        return weighted_reciprocal_rank_fusion(
            [
                WeightedRanking(case_ids=bm25_ranking, weight=self.bm25_weight),
                WeightedRanking(case_ids=vector_ranking, weight=self.vector_weight),
            ],
            k=self.k,
            top_k=self.top_k,
        )
