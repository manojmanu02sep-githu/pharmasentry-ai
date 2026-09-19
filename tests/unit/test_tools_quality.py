"""Unit tests for src/tools/quality.py."""

from __future__ import annotations

from src.models.enums import AgentName
from src.tools.base import ToolContext
from src.tools.quality import (
    CompareReportWithFieldsInput,
    CompareReportWithFieldsTool,
    DetectConflictingValuesInput,
    DetectConflictingValuesTool,
    DetectUnsupportedClaimsInput,
    DetectUnsupportedClaimsTool,
    SentenceWithCitations,
    SuccessCriterionStatus,
    ValidateCitationsInput,
    ValidateCitationsTool,
    ValidateGoalCompletionInput,
    ValidateGoalCompletionTool,
)

CTX = ToolContext(
    case_id="case_001",
    trace_id="trace_001",
    agent=AgentName.EVALUATOR,
    authorized_tools=frozenset(
        {
            "validate_citations", "detect_unsupported_claims", "compare_report_with_fields",
            "detect_conflicting_values", "validate_goal_completion",
        }
    ),
)


def test_validate_citations_all_valid() -> None:
    sentences = [
        SentenceWithCitations(text="Patient took DemoInsulex.", citation_passage_ids=["p1"])
    ]
    output, _ = ValidateCitationsTool().run(
        ValidateCitationsInput(sentences=sentences, valid_passage_ids=["p1", "p2"]), CTX
    )
    assert output.invalid_citations == []
    assert output.precision == 1.0


def test_validate_citations_flags_nonexistent_passage() -> None:
    sentences = [SentenceWithCitations(text="Fabricated claim.", citation_passage_ids=["p99"])]
    output, _ = ValidateCitationsTool().run(
        ValidateCitationsInput(sentences=sentences, valid_passage_ids=["p1", "p2"]), CTX
    )
    assert len(output.invalid_citations) == 1
    assert output.precision == 0.0


def test_detect_unsupported_claims_finds_sentences_with_no_citation() -> None:
    sentences = [
        SentenceWithCitations(text="Cited fact.", citation_passage_ids=["p1"]),
        SentenceWithCitations(text="Uncited claim.", citation_passage_ids=[]),
    ]
    output, _ = DetectUnsupportedClaimsTool().run(
        DetectUnsupportedClaimsInput(sentences=sentences), CTX
    )
    assert output.unsupported_sentences == ["Uncited claim."]
    assert output.unsupported_claim_rate == 0.5


def test_compare_report_with_fields_finds_missing_field() -> None:
    output, _ = CompareReportWithFieldsTool().run(
        CompareReportWithFieldsInput(
            report_text="Patient received DemoInsulex and developed nausea.",
            expected_fields={
                "product.product_name": "DemoInsulex",
                "outcome.outcome_description": "fully recovered",
            },
        ),
        CTX,
    )
    assert output.fields_missing_from_report == ["outcome.outcome_description"]


def test_detect_conflicting_values_finds_disagreeing_sources() -> None:
    output, _ = DetectConflictingValuesTool().run(
        DetectConflictingValuesInput(
            values_by_source={
                "email": {"product.dose": "10 mg", "product.route": "subcutaneous injection"},
                "attachment": {"product.dose": "20 mg", "product.route": "subcutaneous injection"},
            }
        ),
        CTX,
    )
    assert output.conflicting_fields == ["product.dose"]
    assert set(output.distinct_values["product.dose"]) == {"10 mg", "20 mg"}


def test_detect_conflicting_values_no_conflict_when_sources_agree() -> None:
    output, _ = DetectConflictingValuesTool().run(
        DetectConflictingValuesInput(
            values_by_source={
                "email": {"product.dose": "10 mg"},
                "attachment": {"product.dose": "10 mg"},
            }
        ),
        CTX,
    )
    assert output.conflicting_fields == []


def test_validate_goal_completion_all_met() -> None:
    output, _ = ValidateGoalCompletionTool().run(
        ValidateGoalCompletionInput(
            success_criteria=[
                SuccessCriterionStatus(criterion_id="c1", met=True),
                SuccessCriterionStatus(criterion_id="c2", met=True),
            ]
        ),
        CTX,
    )
    assert output.all_met is True
    assert output.unmet_criteria == []


def test_validate_goal_completion_reports_unmet() -> None:
    output, _ = ValidateGoalCompletionTool().run(
        ValidateGoalCompletionInput(
            success_criteria=[
                SuccessCriterionStatus(criterion_id="c1", met=True),
                SuccessCriterionStatus(criterion_id="c2", met=False),
            ]
        ),
        CTX,
    )
    assert output.all_met is False
    assert output.unmet_criteria == ["c2"]


def test_validate_goal_completion_empty_criteria_is_not_met() -> None:
    output, _ = ValidateGoalCompletionTool().run(
        ValidateGoalCompletionInput(success_criteria=[]), CTX
    )
    assert output.all_met is False
