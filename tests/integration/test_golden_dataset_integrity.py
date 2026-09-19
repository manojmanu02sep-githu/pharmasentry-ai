"""Integration tests that load the actual generated golden dataset from disk.

These exercise the real files under evaluations/golden_dataset/ (produced by
scripts/generate_golden_dataset.py), not mocks. If the dataset has not been
generated yet, run: python scripts/generate_golden_dataset.py
"""

from __future__ import annotations

from collections import Counter

from src.evaluation.golden_loader import (
    cases_by_family,
    load_all_cases,
    load_manifest,
    load_split,
    validate_no_family_leakage,
)
from src.evaluation.schemas import GoldenCaseCategory

EXPECTED_CATEGORY_COUNTS = {
    GoldenCaseCategory.COMPLETE.value: 20,
    GoldenCaseCategory.INCOMPLETE.value: 20,
    GoldenCaseCategory.SERIOUS.value: 15,
    GoldenCaseCategory.NON_SERIOUS.value: 15,
    GoldenCaseCategory.EXACT_DUPLICATE.value: 10,
    GoldenCaseCategory.NEAR_DUPLICATE.value: 10,
    GoldenCaseCategory.CONFLICTING.value: 5,
    GoldenCaseCategory.NON_SAFETY.value: 5,
}


def test_dataset_has_at_least_100_cases_with_expected_distribution() -> None:
    cases = load_all_cases()
    assert len(cases) >= 100

    counts = Counter(c.category.value for c in cases)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        assert counts[category] == expected_count, (
            f"category {category!r}: expected {expected_count}, got {counts[category]}"
        )


def test_manifest_matches_actual_case_files() -> None:
    manifest = load_manifest()
    cases = load_all_cases()
    assert manifest.total_cases == len(cases)
    assert manifest.category_counts == dict(Counter(c.category.value for c in cases))


def test_all_case_ids_are_unique() -> None:
    cases = load_all_cases()
    case_ids = [c.case_id for c in cases]
    assert len(case_ids) == len(set(case_ids))


def test_split_covers_every_case_exactly_once() -> None:
    cases = load_all_cases()
    split = load_split()
    assert split.leakage_free()

    all_case_ids = {c.case_id for c in cases}
    split_ids = set(split.train) | set(split.test)
    assert split_ids == all_case_ids


def test_no_duplicate_family_straddles_train_and_test() -> None:
    assert validate_no_family_leakage() == []


def test_duplicate_families_have_exactly_two_members() -> None:
    cases = load_all_cases()
    duplicate_categories = {GoldenCaseCategory.EXACT_DUPLICATE, GoldenCaseCategory.NEAR_DUPLICATE}
    families = cases_by_family(cases)

    duplicate_family_ids = {
        c.expected_duplicate_family for c in cases if c.category in duplicate_categories
    }
    for family_id in duplicate_family_ids:
        assert len(families[family_id]) == 2, f"family {family_id} should have 2 members"


def test_singleton_categories_have_singleton_families() -> None:
    cases = load_all_cases()
    families = cases_by_family(cases)
    singleton_categories = {
        GoldenCaseCategory.COMPLETE,
        GoldenCaseCategory.INCOMPLETE,
        GoldenCaseCategory.SERIOUS,
        GoldenCaseCategory.NON_SERIOUS,
        GoldenCaseCategory.CONFLICTING,
        GoldenCaseCategory.NON_SAFETY,
    }
    for case in cases:
        if case.category in singleton_categories:
            assert len(families[case.expected_duplicate_family]) == 1


def test_non_safety_cases_have_no_attachment_and_are_not_safety_reports() -> None:
    cases = load_all_cases()
    non_safety = [c for c in cases if c.category == GoldenCaseCategory.NON_SAFETY]
    assert len(non_safety) == 5
    for case in non_safety:
        assert case.is_safety_report is False
        assert case.attachment_text == ""
        assert case.expected_narrative_facts == []


def test_conflicting_cases_flag_a_conflicting_field() -> None:
    cases = load_all_cases()
    conflicting = [c for c in cases if c.category == GoldenCaseCategory.CONFLICTING]
    assert len(conflicting) == 5
    for case in conflicting:
        assert case.conflicting_fields
        for field in case.conflicting_fields:
            assert case.expected_extracted_fields.get(field) is None


def test_serious_cases_all_have_seriousness_indicators() -> None:
    cases = load_all_cases()
    serious = [c for c in cases if c.category == GoldenCaseCategory.SERIOUS]
    assert len(serious) == 15
    for case in serious:
        assert case.expected_seriousness_indicators


def test_non_serious_cases_have_no_seriousness_indicators() -> None:
    cases = load_all_cases()
    non_serious = [c for c in cases if c.category == GoldenCaseCategory.NON_SERIOUS]
    assert len(non_serious) == 15
    for case in non_serious:
        assert case.expected_seriousness_indicators == []


def test_every_case_expected_product_name_appears_in_email_text() -> None:
    """Internal-consistency spot check: the labeled product must actually be
    mentioned in the generated email for every safety-report case."""
    cases = load_all_cases()
    for case in cases:
        if not case.is_safety_report:
            continue
        product = case.expected_extracted_fields.get("product.product_name")
        assert product is not None
        assert product in case.email_text
