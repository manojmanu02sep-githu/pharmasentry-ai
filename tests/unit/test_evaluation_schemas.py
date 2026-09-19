"""Unit tests for src/evaluation/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.evaluation.schemas import (
    DatasetSplit,
    ExpectedMinimumCriteria,
    GoldenCase,
    GoldenCaseCategory,
)


def _minimal_case(**overrides: object) -> GoldenCase:
    defaults: dict[str, object] = dict(
        case_id="test_case_001",
        category=GoldenCaseCategory.COMPLETE,
        is_safety_report=True,
        email_text="synthetic email",
        attachment_text="synthetic attachment",
        expected_extracted_fields={"product.product_name": "DemoGluca"},
        expected_minimum_criteria=ExpectedMinimumCriteria(
            has_identifiable_patient=True,
            has_identifiable_reporter=True,
            has_suspect_product=True,
            has_adverse_event=True,
        ),
        expected_duplicate_family="fam_test_case_001",
    )
    defaults.update(overrides)
    return GoldenCase(**defaults)  # type: ignore[arg-type]


def test_minimal_valid_case_constructs() -> None:
    case = _minimal_case()
    assert case.category == GoldenCaseCategory.COMPLETE
    assert case.expected_minimum_criteria.meets_minimum_criteria


def test_unknown_extracted_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(expected_extracted_fields={"not.a.real.field": "x"})


def test_unknown_missing_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(expected_missing_fields=["not.a.real.field"])


def test_unknown_conflicting_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(conflicting_fields=["not.a.real.field"])


def test_expected_minimum_criteria_property() -> None:
    incomplete = ExpectedMinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=False,
        has_suspect_product=True,
        has_adverse_event=True,
        missing_criteria=["reporter"],
    )
    assert not incomplete.meets_minimum_criteria


def test_dataset_split_leakage_free() -> None:
    clean = DatasetSplit(train=["a", "b"], test=["c", "d"])
    assert clean.leakage_free()

    leaking = DatasetSplit(train=["a", "b"], test=["b", "c"])
    assert not leaking.leakage_free()
