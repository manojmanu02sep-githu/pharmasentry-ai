"""Unit tests for src/tools/domain.py.

Several of these are cross-checked directly against the golden dataset's
malformed_date / product_alias / event_synonym fixtures, so the tool and
the fixtures that exercise it can't silently drift apart.
"""

from __future__ import annotations

from src.evaluation.golden_loader import load_all_cases
from src.evaluation.schemas import GoldenCaseCategory
from src.models.enums import AgentName, TriageIndicator
from src.tools.base import ToolContext
from src.tools.domain import (
    CheckMinimumCaseCriteriaInput,
    CheckMinimumCaseCriteriaTool,
    DetectExplicitTriageIndicatorsInput,
    DetectExplicitTriageIndicatorsTool,
    LookupEventTermInput,
    LookupEventTermTool,
    LookupProductAliasInput,
    LookupProductAliasTool,
    NormalizeDateInput,
    NormalizeDateTool,
    ValidateMedicalFieldInput,
    ValidateMedicalFieldTool,
)

CTX = ToolContext(
    case_id="case_001",
    trace_id="trace_001",
    agent=AgentName.MEDICAL_EXTRACTION,
    authorized_tools=frozenset(
        {
            "lookup_product_alias", "lookup_event_term", "normalize_date",
            "validate_medical_field", "check_minimum_case_criteria",
            "detect_explicit_triage_indicators",
        }
    ),
)


def test_lookup_product_alias_resolves_known_alias() -> None:
    output, _ = LookupProductAliasTool().run(
        LookupProductAliasInput(text="I was given DemoInsulex Flex yesterday."), CTX
    )
    assert output.canonical_name == "DemoInsulex"
    assert output.alias_matched == "DemoInsulex Flex"


def test_lookup_product_alias_no_match_returns_none() -> None:
    output, _ = LookupProductAliasTool().run(
        LookupProductAliasInput(text="No known product mentioned here."), CTX
    )
    assert output.canonical_name is None


def test_lookup_product_alias_against_every_golden_product_alias_case() -> None:
    cases = [c for c in load_all_cases() if c.case_type == GoldenCaseCategory.PRODUCT_ALIAS]
    assert cases
    for case in cases:
        expected = case.expected_extracted_fields["product.product_name"]
        output, _ = LookupProductAliasTool().run(
            LookupProductAliasInput(text=case.email_body), CTX
        )
        assert output.canonical_name == expected, case.case_id


def test_lookup_event_term_resolves_known_synonym() -> None:
    output, _ = LookupEventTermTool().run(
        LookupEventTermInput(text="The patient was throwing up repeatedly."), CTX
    )
    assert output.canonical_term == "vomiting"


def test_lookup_event_term_against_every_golden_event_synonym_case() -> None:
    cases = [c for c in load_all_cases() if c.case_type == GoldenCaseCategory.EVENT_SYNONYM]
    assert cases
    for case in cases:
        expected = case.expected_extracted_fields["event.event_description"]
        output, _ = LookupEventTermTool().run(LookupEventTermInput(text=case.email_body), CTX)
        assert output.canonical_term == expected, case.case_id


def test_normalize_date_iso_passthrough() -> None:
    output, _ = NormalizeDateTool().run(NormalizeDateInput(raw_date_text="2026-03-04"), CTX)
    assert output.status == "ok"
    assert output.normalized_date == "2026-03-04"


def test_normalize_date_against_every_golden_malformed_date_case() -> None:
    cases = [c for c in load_all_cases() if c.case_type == GoldenCaseCategory.MALFORMED_DATE]
    assert cases
    for case in cases:
        expected = case.expected_extracted_fields["product.treatment_start_date"]
        raw = case.notes.split("written as ")[1].split("; expected")[0].strip("'")
        output, _ = NormalizeDateTool().run(NormalizeDateInput(raw_date_text=raw), CTX)
        assert output.normalized_date == expected, (case.case_id, raw, output)


def test_normalize_date_invalid_month_day_flagged() -> None:
    output, _ = NormalizeDateTool().run(NormalizeDateInput(raw_date_text="2026.13.40"), CTX)
    assert output.status == "invalid"
    assert output.normalized_date is None


def test_validate_medical_field_age_range() -> None:
    ok, _ = ValidateMedicalFieldTool().run(
        ValidateMedicalFieldInput(field_name="patient.age", value="62"), CTX
    )
    assert ok.valid is True

    bad, _ = ValidateMedicalFieldTool().run(
        ValidateMedicalFieldInput(field_name="patient.age", value="999"), CTX
    )
    assert bad.valid is False


def test_validate_medical_field_none_value_is_valid() -> None:
    output, _ = ValidateMedicalFieldTool().run(
        ValidateMedicalFieldInput(field_name="patient.age", value=None), CTX
    )
    assert output.valid is True


def test_check_minimum_case_criteria_all_present() -> None:
    result, _ = CheckMinimumCaseCriteriaTool().run(
        CheckMinimumCaseCriteriaInput(
            has_identifiable_patient=True, has_identifiable_reporter=True,
            has_suspect_product=True, has_adverse_event=True,
        ),
        CTX,
    )
    assert result.meets_minimum_criteria
    assert result.missing_criteria == []


def test_check_minimum_case_criteria_reports_each_missing_one() -> None:
    result, _ = CheckMinimumCaseCriteriaTool().run(
        CheckMinimumCaseCriteriaInput(
            has_identifiable_patient=True, has_identifiable_reporter=False,
            has_suspect_product=True, has_adverse_event=False,
        ),
        CTX,
    )
    assert not result.meets_minimum_criteria
    assert result.missing_criteria == ["reporter", "adverse_event"]


def test_detect_explicit_triage_indicators_hospitalization() -> None:
    output, _ = DetectExplicitTriageIndicatorsTool().run(
        DetectExplicitTriageIndicatorsInput(
            text="The patient was hospitalized after the reaction."
        ),
        CTX,
    )
    assert TriageIndicator.HOSPITALIZATION in output.indicators


def test_detect_explicit_triage_indicators_none_for_mild_case() -> None:
    output, _ = DetectExplicitTriageIndicatorsTool().run(
        DetectExplicitTriageIndicatorsInput(text="The patient reported mild nausea."),
        CTX,
    )
    assert output.indicators == []


def test_detect_explicit_triage_indicators_against_golden_serious_cases() -> None:
    cases = [c for c in load_all_cases() if c.case_type == GoldenCaseCategory.SERIOUS]
    assert cases
    for case in cases:
        output, _ = DetectExplicitTriageIndicatorsTool().run(
            DetectExplicitTriageIndicatorsInput(
                text=case.email_body + "\n" + case.attachment_text
            ),
            CTX,
        )
        assert output.indicators, f"{case.case_id} should have a detected indicator"


def test_detect_explicit_triage_indicators_against_golden_non_serious_cases() -> None:
    cases = [c for c in load_all_cases() if c.case_type == GoldenCaseCategory.NON_SERIOUS]
    assert cases
    for case in cases:
        output, _ = DetectExplicitTriageIndicatorsTool().run(
            DetectExplicitTriageIndicatorsInput(
                text=case.email_body + "\n" + case.attachment_text
            ),
            CTX,
        )
        assert output.indicators == [], f"{case.case_id} should have no detected indicator"
