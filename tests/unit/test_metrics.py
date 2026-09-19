"""Unit tests for the baseline metrics framework (src/evaluation/metrics.py)."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    citation_coverage,
    classification_prf1,
    duplicate_precision_at_k,
    duplicate_recall_at_k,
    extraction_prf1,
    mean_duplicate_precision_at_k,
    mean_duplicate_recall_at_k,
    mean_reciprocal_rank,
    seriousness_sensitivity_specificity,
    unsupported_claim_rate,
)


def test_classification_prf1_perfect() -> None:
    result = classification_prf1([True, True, False, False], [True, True, False, False])
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_classification_prf1_partial() -> None:
    # 1 true positive, 1 false positive, 1 false negative, 1 true negative
    result = classification_prf1([True, False, True, False], [True, True, False, False])
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_classification_prf1_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        classification_prf1([True], [True, False])


def test_classification_prf1_all_negative_precision_default_zero() -> None:
    result = classification_prf1([False, False], [False, False])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_extraction_prf1_exact_match() -> None:
    expected = {"product.product_name": "DemoGluca", "product.dose": "10mg"}
    predicted = {"product.product_name": "DemoGluca", "product.dose": "10mg"}
    result = extraction_prf1([(expected, predicted)])
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_extraction_prf1_missing_and_wrong_value() -> None:
    expected = {"product.product_name": "DemoGluca", "product.dose": "10mg"}
    predicted = {"product.product_name": "DemoZanix"}  # wrong value; dose missing
    result = extraction_prf1([(expected, predicted)])
    # product_name: predicted but wrong -> FN + FP; dose: missing -> FN only
    assert result.true_positives == 0
    assert result.false_negatives == 2
    assert result.false_positives == 1


def test_extraction_prf1_hallucinated_field_is_false_positive() -> None:
    expected: dict[str, str | None] = {"product.dose": None}
    predicted = {"product.dose": "10mg"}
    result = extraction_prf1([(expected, predicted)])
    assert result.false_positives == 1
    assert result.true_positives == 0
    assert result.false_negatives == 0


def test_seriousness_sensitivity_specificity() -> None:
    y_true = [True, True, False, False, False]
    y_pred = [True, False, False, False, True]
    result = seriousness_sensitivity_specificity(y_true, y_pred)
    assert result.true_positives == 1
    assert result.false_negatives == 1
    assert result.true_negatives == 2
    assert result.false_positives == 1
    assert result.sensitivity == pytest.approx(0.5)
    assert result.specificity == pytest.approx(2 / 3)


def test_duplicate_precision_and_recall_at_k() -> None:
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"c", "z"}
    assert duplicate_precision_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 5)
    assert duplicate_recall_at_k(retrieved, relevant, k=5) == pytest.approx(1 / 2)


def test_duplicate_recall_at_k_no_relevant_items_is_trivially_perfect() -> None:
    assert duplicate_recall_at_k(["a", "b"], set(), k=5) == 1.0


def test_duplicate_precision_at_k_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        duplicate_precision_at_k(["a"], {"a"}, k=0)


def test_mean_duplicate_precision_and_recall_at_k() -> None:
    queries = [
        (["a", "b", "c"], {"a"}),
        (["x", "y", "z"], {"q"}),
    ]
    # query 1: 1 hit / k=5 -> 0.2 ; query 2: 0 hits -> 0.0
    assert mean_duplicate_precision_at_k(queries, k=5) == pytest.approx(0.1)
    # query 1: found its only relevant item -> recall 1.0 ; query 2: missed -> 0.0
    assert mean_duplicate_recall_at_k(queries, k=5) == pytest.approx(0.5)


def test_mean_reciprocal_rank() -> None:
    queries = [
        (["a", "b", "c"], {"b"}),  # rank 2 -> 0.5
        (["x", "y", "z"], {"q"}),  # not found -> 0.0
    ]
    assert mean_reciprocal_rank(queries) == pytest.approx(0.25)


def test_mean_reciprocal_rank_empty_queries() -> None:
    assert mean_reciprocal_rank([]) == 0.0


def test_citation_coverage_full_and_partial() -> None:
    assert citation_coverage(10, 10) == 1.0
    assert citation_coverage(4, 2) == 0.5
    assert citation_coverage(0, 0) == 1.0  # no claims: trivially fully covered


def test_citation_coverage_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        citation_coverage(2, 3)


def test_unsupported_claim_rate() -> None:
    assert unsupported_claim_rate(10, 0) == 0.0
    assert unsupported_claim_rate(10, 3) == pytest.approx(0.3)
    assert unsupported_claim_rate(0, 0) == 0.0
