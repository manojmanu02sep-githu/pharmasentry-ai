"""Unit tests for src/retrieval/retrieval_evaluation.py.

Runs real (non-fabricated) retrieval against the golden dataset — see that
module's docstring for the evaluation methodology.
"""

from __future__ import annotations

from src.retrieval.retrieval_evaluation import run_retrieval_evaluation


def test_run_retrieval_evaluation_returns_all_three_methods() -> None:
    results = run_retrieval_evaluation(bm25_weight=0.5, vector_weight=0.5)
    assert set(results) == {"bm25", "vector", "hybrid"}


def test_run_retrieval_evaluation_evaluates_only_cases_with_duplicates() -> None:
    results = run_retrieval_evaluation(bm25_weight=0.5, vector_weight=0.5)
    for result in results.values():
        assert result.queries_evaluated > 0


def test_run_retrieval_evaluation_metrics_are_valid_fractions() -> None:
    results = run_retrieval_evaluation(bm25_weight=0.5, vector_weight=0.5)
    for result in results.values():
        assert 0.0 <= result.precision_at_5 <= 1.0
        assert 0.0 <= result.recall_at_5 <= 1.0
        assert 0.0 <= result.mrr <= 1.0
