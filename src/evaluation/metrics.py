"""Baseline metrics framework.

Pure, side-effect-free scoring functions. Every function here takes plain
predicted/expected values and returns a real, computed number — nothing is
invented. These are wired to actual agent/tool output starting in later
phases (retrieval in Phase 5, agents in Phase 6); for now they are exercised
against the golden dataset by scripts/run_golden_evaluation.py using a
trivial, clearly-labeled baseline predictor, and by unit tests with
hand-picked inputs.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.evaluation.schemas import FIELD_KEYS


class PrecisionRecallF1(BaseModel):
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def _prf1_from_counts(tp: int, fp: int, fn: int) -> PrecisionRecallF1:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return PrecisionRecallF1(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def classification_prf1(
    y_true: list[bool], y_pred: list[bool]
) -> PrecisionRecallF1:
    """Binary classification precision/recall/F1 (e.g. safety vs non-safety).

    ``True`` is the positive class in both lists. Lists must be the same
    length and index-aligned (same case order).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    return _prf1_from_counts(tp, fp, fn)


def extraction_prf1(
    expected_and_predicted: list[tuple[dict[str, str | None], dict[str, str | None]]],
) -> PrecisionRecallF1:
    """Field-level slot-filling precision/recall/F1 across many cases.

    For every (expected, predicted) field dict pair and every known field
    key: a predicted non-null value that exactly matches a non-null expected
    value is a true positive; a predicted non-null value with no matching
    expected value (missing key, null expected, or wrong value) is a false
    positive; a non-null expected value the prediction failed to produce or
    got wrong is a false negative. Fields that are correctly absent in both
    are not counted (true negatives are uninformative for extraction).
    """
    tp = fp = fn = 0
    for expected, predicted in expected_and_predicted:
        for key in FIELD_KEYS:
            expected_value = expected.get(key)
            predicted_value = predicted.get(key)
            if expected_value is not None:
                if predicted_value == expected_value:
                    tp += 1
                else:
                    fn += 1
                    if predicted_value is not None:
                        fp += 1
            elif predicted_value is not None:
                fp += 1
    return _prf1_from_counts(tp, fp, fn)


class SensitivitySpecificity(BaseModel):
    sensitivity: float  # recall on the positive ("serious") class
    specificity: float  # recall on the negative ("not serious") class
    true_positives: int
    false_negatives: int
    true_negatives: int
    false_positives: int


def seriousness_sensitivity_specificity(
    y_true: list[bool], y_pred: list[bool]
) -> SensitivitySpecificity:
    """Sensitivity/specificity for the seriousness triage classification.

    ``True`` means "at least one explicit seriousness indicator present".
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)

    return SensitivitySpecificity(
        sensitivity=_safe_div(tp, tp + fn),
        specificity=_safe_div(tn, tn + fp),
        true_positives=tp,
        false_negatives=fn,
        true_negatives=tn,
        false_positives=fp,
    )


def duplicate_precision_at_k(
    retrieved: list[str], relevant: set[str], k: int = 5
) -> float:
    """Fraction of the top-k retrieved candidates that are truly relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k


def duplicate_recall_at_k(
    retrieved: list[str], relevant: set[str], k: int = 5
) -> float:
    """Fraction of all relevant candidates found within the top-k results.

    If there are no relevant candidates at all, recall is trivially 1.0
    (there was nothing to miss).
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mean_duplicate_precision_at_k(
    queries: list[tuple[list[str], set[str]]], k: int = 5
) -> float:
    if not queries:
        return 0.0
    return sum(duplicate_precision_at_k(r, rel, k) for r, rel in queries) / len(queries)


def mean_duplicate_recall_at_k(
    queries: list[tuple[list[str], set[str]]], k: int = 5
) -> float:
    if not queries:
        return 0.0
    return sum(duplicate_recall_at_k(r, rel, k) for r, rel in queries) / len(queries)


def mean_reciprocal_rank(queries: list[tuple[list[str], set[str]]]) -> float:
    """Mean reciprocal rank of the first relevant hit, across queries.

    Included alongside precision@5/recall@5 because CLAUDE.md's Hybrid RAG
    section requires MRR for retrieval evaluation; wired to real retrieval
    output in Phase 5.
    """
    if not queries:
        return 0.0
    reciprocal_ranks = []
    for retrieved, relevant in queries:
        rank = next(
            (i + 1 for i, cid in enumerate(retrieved) if cid in relevant), None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def citation_coverage(total_claims: int, cited_claims: int) -> float:
    """Fraction of narrative claims that carry at least one citation.

    A narrative with zero claims is trivially fully covered (1.0).
    """
    if cited_claims > total_claims:
        raise ValueError("cited_claims cannot exceed total_claims")
    return _safe_div(cited_claims, total_claims, default=1.0)


def unsupported_claim_rate(total_claims: int, unsupported_claims: int) -> float:
    """Fraction of narrative claims that are not supported by any citation."""
    if unsupported_claims > total_claims:
        raise ValueError("unsupported_claims cannot exceed total_claims")
    return _safe_div(unsupported_claims, total_claims, default=0.0)
