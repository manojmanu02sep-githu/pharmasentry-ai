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
    GoldenCaseCategory.COMPLETE.value: 15,
    GoldenCaseCategory.INCOMPLETE.value: 15,
    GoldenCaseCategory.SERIOUS.value: 12,
    GoldenCaseCategory.NON_SERIOUS.value: 12,
    GoldenCaseCategory.NON_SAFETY.value: 5,
    GoldenCaseCategory.EXACT_DUPLICATE.value: 8,
    GoldenCaseCategory.NEAR_DUPLICATE.value: 8,
    GoldenCaseCategory.SIMILAR_NON_DUPLICATE.value: 6,
    GoldenCaseCategory.CONFLICTING.value: 5,
    GoldenCaseCategory.MISSING_SUSPECT_PRODUCT.value: 4,
    GoldenCaseCategory.MISSING_ADVERSE_EVENT.value: 4,
    GoldenCaseCategory.MISSING_REPORTER.value: 4,
    GoldenCaseCategory.MISSING_IDENTIFIABLE_PATIENT.value: 4,
    GoldenCaseCategory.POOR_OCR.value: 5,
    GoldenCaseCategory.MULTILINGUAL.value: 5,
    GoldenCaseCategory.PRODUCT_ALIAS.value: 5,
    GoldenCaseCategory.EVENT_SYNONYM.value: 5,
    GoldenCaseCategory.MALFORMED_DATE.value: 5,
    GoldenCaseCategory.PROMPT_INJECTION.value: 5,
    GoldenCaseCategory.TOOL_INJECTION.value: 3,
    GoldenCaseCategory.APPROVAL_BYPASS_ATTEMPT.value: 3,
}
assert sum(EXPECTED_CATEGORY_COUNTS.values()) >= 120

# Categories where the product name in expected_extracted_fields is
# deliberately absent, or deliberately NOT the literal string appearing in
# the source text (alias resolution) — excluded from the literal
# "product name appears verbatim in email" spot check below.
_PRODUCT_TEXT_EXEMPT = {
    GoldenCaseCategory.MISSING_SUSPECT_PRODUCT,
    GoldenCaseCategory.PRODUCT_ALIAS,
}


def test_dataset_has_at_least_120_cases_with_expected_distribution() -> None:
    cases = load_all_cases()
    assert len(cases) >= 120

    counts = Counter(c.case_type.value for c in cases)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        assert counts[category] == expected_count, (
            f"category {category!r}: expected {expected_count}, got {counts[category]}"
        )


def test_manifest_matches_actual_case_files() -> None:
    manifest = load_manifest()
    cases = load_all_cases()
    assert manifest.total_cases == len(cases)
    assert manifest.category_counts == dict(Counter(c.case_type.value for c in cases))


def test_all_case_ids_are_unique() -> None:
    cases = load_all_cases()
    case_ids = [c.case_id for c in cases]
    assert len(case_ids) == len(set(case_ids))


def test_split_covers_every_case_exactly_once() -> None:
    cases = load_all_cases()
    split = load_split()
    assert split.leakage_free()

    all_case_ids = {c.case_id for c in cases}
    split_ids = set(split.train) | set(split.validation) | set(split.test)
    assert split_ids == all_case_ids


def test_no_duplicate_family_straddles_splits() -> None:
    """Proves there is no duplicate-family leakage across train/validation/test."""
    assert validate_no_family_leakage() == []


def test_duplicate_families_have_exactly_two_members() -> None:
    cases = load_all_cases()
    duplicate_categories = {GoldenCaseCategory.EXACT_DUPLICATE, GoldenCaseCategory.NEAR_DUPLICATE}
    families = cases_by_family(cases)

    duplicate_family_ids = {
        c.duplicate_family_id for c in cases if c.case_type in duplicate_categories
    }
    for family_id in duplicate_family_ids:
        assert len(families[family_id]) == 2, f"family {family_id} should have 2 members"


def test_singleton_categories_have_singleton_families() -> None:
    cases = load_all_cases()
    families = cases_by_family(cases)
    duplicate_categories = {GoldenCaseCategory.EXACT_DUPLICATE, GoldenCaseCategory.NEAR_DUPLICATE}
    for case in cases:
        if case.case_type not in duplicate_categories:
            assert len(families[case.duplicate_family_id]) == 1, (
                f"{case.case_id} (type={case.case_type.value}) should be a singleton family"
            )


def test_expected_duplicate_matches_are_consistent_with_family() -> None:
    cases = load_all_cases()
    families = cases_by_family(cases)
    for case in cases:
        family_members = {c.case_id for c in families[case.duplicate_family_id]} - {case.case_id}
        assert set(case.expected_duplicate_matches) == family_members


def test_similar_non_duplicate_cases_have_no_duplicate_matches() -> None:
    cases = load_all_cases()
    similar = [c for c in cases if c.case_type == GoldenCaseCategory.SIMILAR_NON_DUPLICATE]
    assert len(similar) == 6
    for case in similar:
        assert case.expected_duplicate_matches == []


def test_non_safety_cases_have_no_attachment_and_are_not_safety_reports() -> None:
    cases = load_all_cases()
    non_safety = [c for c in cases if c.case_type == GoldenCaseCategory.NON_SAFETY]
    assert len(non_safety) == 5
    for case in non_safety:
        assert case.is_safety_report is False
        assert case.attachment_text == ""
        assert case.attachment_metadata.has_attachment is False
        assert case.expected_narrative_facts == []


def test_conflicting_cases_flag_a_conflicting_field() -> None:
    cases = load_all_cases()
    conflicting = [c for c in cases if c.case_type == GoldenCaseCategory.CONFLICTING]
    assert len(conflicting) == 5
    for case in conflicting:
        assert case.expected_conflicting_fields
        for field in case.expected_conflicting_fields:
            assert case.expected_extracted_fields.get(field) is None


def test_serious_cases_all_have_seriousness_indicators() -> None:
    cases = load_all_cases()
    serious = [c for c in cases if c.case_type == GoldenCaseCategory.SERIOUS]
    assert len(serious) == 12
    for case in serious:
        assert case.expected_seriousness_indicators


def test_non_serious_cases_have_no_seriousness_indicators() -> None:
    cases = load_all_cases()
    non_serious = [c for c in cases if c.case_type == GoldenCaseCategory.NON_SERIOUS]
    assert len(non_serious) == 12
    for case in non_serious:
        assert case.expected_seriousness_indicators == []


def test_every_case_expected_product_name_appears_in_email_text() -> None:
    """Internal-consistency spot check: the labeled product must actually be
    mentioned in the generated email for every safety-report case, except
    the categories that deliberately break this (product omitted, or named
    via an alias that must be resolved by a lookup tool)."""
    cases = load_all_cases()
    for case in cases:
        if not case.is_safety_report or case.case_type in _PRODUCT_TEXT_EXEMPT:
            continue
        product = case.expected_extracted_fields.get("product.product_name")
        assert product is not None
        assert product in case.email_text


def test_missing_suspect_product_cases_never_name_a_product() -> None:
    cases = load_all_cases()
    missing = [c for c in cases if c.case_type == GoldenCaseCategory.MISSING_SUSPECT_PRODUCT]
    assert len(missing) == 4
    for case in missing:
        assert case.expected_extracted_fields.get("product.product_name") is None
        assert not case.expected_minimum_criteria.meets_minimum_criteria


def test_product_alias_cases_resolve_to_canonical_name() -> None:
    cases = load_all_cases()
    aliased = [c for c in cases if c.case_type == GoldenCaseCategory.PRODUCT_ALIAS]
    assert len(aliased) == 5
    canonical_products = {"DemoGluca", "DemoCardolol", "DemoZanix"}
    for case in aliased:
        resolved = case.expected_extracted_fields.get("product.product_name")
        assert resolved in canonical_products
        assert case.notes  # explains which alias string must be resolved


def test_event_synonym_cases_resolve_to_canonical_term() -> None:
    cases = load_all_cases()
    synonyms = [c for c in cases if c.case_type == GoldenCaseCategory.EVENT_SYNONYM]
    assert len(synonyms) == 5
    for case in synonyms:
        assert case.expected_extracted_fields.get("event.event_description")


def test_poor_ocr_cases_flag_low_ocr_quality_and_keep_synthetic_marker() -> None:
    cases = load_all_cases()
    poor_ocr = [c for c in cases if c.case_type == GoldenCaseCategory.POOR_OCR]
    assert len(poor_ocr) == 5
    for case in poor_ocr:
        assert case.attachment_metadata.ocr_quality == "degraded"
        assert case.attachment_metadata.ocr_applied is True
        assert "SYNTHETIC" in case.attachment_text
        assert case.expected_human_review_required is True


def test_injection_and_bypass_cases_require_human_review_and_prohibit_bypass() -> None:
    cases = load_all_cases()
    injection_categories = {
        GoldenCaseCategory.PROMPT_INJECTION,
        GoldenCaseCategory.TOOL_INJECTION,
        GoldenCaseCategory.APPROVAL_BYPASS_ATTEMPT,
    }
    flagged = [c for c in cases if c.case_type in injection_categories]
    assert len(flagged) == 11
    for case in flagged:
        assert case.expected_human_review_required is True
        assert case.prohibited_conclusions


def test_malformed_date_cases_flag_ambiguous_dates_for_review() -> None:
    cases = load_all_cases()
    malformed = [c for c in cases if c.case_type == GoldenCaseCategory.MALFORMED_DATE]
    assert len(malformed) == 5
    # At least one malformed-date case must be genuinely unresolvable and
    # therefore require human review rather than a silently guessed date.
    assert any(c.expected_human_review_required for c in malformed)


def test_all_cases_are_marked_synthetic() -> None:
    for case in load_all_cases():
        assert case.synthetic_data is True
