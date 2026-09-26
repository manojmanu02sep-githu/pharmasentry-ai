"""Unit tests for src/retrieval/hybrid_ranker.py."""

from __future__ import annotations

from src.models.enums import AgentName
from src.retrieval.hybrid_ranker import (
    HybridRanker,
    WeightedRanking,
    weighted_reciprocal_rank_fusion,
)
from src.tools.base import ToolContext

CTX = ToolContext(
    case_id="case_a",
    trace_id="trace_001",
    agent=AgentName.DUPLICATE_SEARCH,
    authorized_tools=frozenset(),
)


def test_equal_weights_matches_unweighted_rrf() -> None:
    rankings = [
        WeightedRanking(case_ids=["a", "b", "c"], weight=1.0),
        WeightedRanking(case_ids=["c", "a", "b"], weight=1.0),
    ]
    fused = weighted_reciprocal_rank_fusion(rankings, k=60, top_k=3)
    top_ids = {cid for cid, _ in fused[:2]}
    assert top_ids == {"a", "c"}


def test_zero_weight_ranking_contributes_nothing_to_score() -> None:
    rankings = [
        WeightedRanking(case_ids=["a", "b"], weight=1.0),
        WeightedRanking(case_ids=["z"], weight=0.0),
    ]
    fused = weighted_reciprocal_rank_fusion(rankings, k=60, top_k=2)
    ids = [cid for cid, _ in fused]
    # z's contribution is weight 0.0, so it never outranks a or b for the top slots
    assert ids == ["a", "b"]


def test_hybrid_ranker_combine_uses_configured_weights() -> None:
    ranker = HybridRanker(bm25_weight=1.0, vector_weight=0.0, top_k=2)
    combined = ranker.combine(bm25_ranking=["a", "b"], vector_ranking=["z", "y"])
    ids = [cid for cid, _ in combined]
    assert ids == ["a", "b"]  # vector ranking has zero weight, contributes nothing


def test_hybrid_ranker_from_config_reads_config_yaml() -> None:
    ranker, event = HybridRanker.from_config(CTX)
    assert ranker.bm25_weight == 0.5
    assert ranker.vector_weight == 0.5
    assert ranker.top_k == 5
    assert event.key == "retrieval"
