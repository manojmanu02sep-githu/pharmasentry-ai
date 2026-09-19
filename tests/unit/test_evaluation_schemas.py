"""Unit tests for src/evaluation/schemas.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.evaluation.schemas import (
    DatasetSplit,
    ExpectedMinimumCriteria,
    GoldenCase,
    GoldenCaseCategory,
    SafetyClassification,
)


def _minimal_case(**overrides: object) -> GoldenCase:
    defaults: dict[str, object] = dict(
        case_id="test_case_001",
        case_type=GoldenCaseCategory.COMPLETE,
        duplicate_family_id="fam_test_case_001",
        expected_safety_classification=SafetyClassification.SAFETY_REPORT,
        email_subject="synthetic subject",
        email_body="synthetic email",
        attachment_text="synthetic attachment",
        expected_extracted_fields={"product.product_name": "DemoGluca"},
        expected_minimum_criteria=ExpectedMinimumCriteria(
            has_identifiable_patient=True,
            has_identifiable_reporter=True,
            has_suspect_product=True,
            has_adverse_event=True,
        ),
    )
    defaults.update(overrides)
    return GoldenCase(**defaults)  # type: ignore[arg-type]


def test_minimal_valid_case_constructs() -> None:
    case = _minimal_case()
    assert case.case_type == GoldenCaseCategory.COMPLETE
    assert case.expected_minimum_criteria.meets_minimum_criteria
    assert case.is_safety_report is True
    assert case.synthetic_data is True


def test_email_text_property_combines_subject_and_body() -> None:
    case = _minimal_case(email_subject="Subject line", email_body="Body text")
    assert case.email_text == "Subject: Subject line\n\nBody text"


def test_unknown_extracted_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(expected_extracted_fields={"not.a.real.field": "x"})


def test_unknown_missing_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(expected_missing_fields=["not.a.real.field"])


def test_unknown_conflicting_field_key_rejected() -> None:
    with pytest.raises(ValidationError):
        _minimal_case(expected_conflicting_fields=["not.a.real.field"])


def test_expected_minimum_criteria_property() -> None:
    incomplete = ExpectedMinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=False,
        has_suspect_product=True,
        has_adverse_event=True,
        missing_criteria=["reporter"],
        status="incomplete",
    )
    assert not incomplete.meets_minimum_criteria


def test_expected_minimum_criteria_status_must_match_criteria() -> None:
    with pytest.raises(ValidationError):
        ExpectedMinimumCriteria(
            has_identifiable_patient=True,
            has_identifiable_reporter=False,
            has_suspect_product=True,
            has_adverse_event=True,
            status="complete",  # inconsistent: a criterion is unmet
        )
    with pytest.raises(ValidationError):
        ExpectedMinimumCriteria(
            has_identifiable_patient=True,
            has_identifiable_reporter=True,
            has_suspect_product=True,
            has_adverse_event=True,
            status="incomplete",  # inconsistent: all criteria met
        )


def test_dataset_split_leakage_free() -> None:
    clean = DatasetSplit(train=["a", "b"], validation=["v"], test=["c", "d"])
    assert clean.leakage_free()

    leaking = DatasetSplit(train=["a", "b"], validation=["v"], test=["b", "c"])
    assert not leaking.leakage_free()

    leaking_via_validation = DatasetSplit(train=["a"], validation=["a"], test=["c"])
    assert not leaking_via_validation.leakage_free()
