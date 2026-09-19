"""Quality tools: citation validation, unsupported-claim detection,
report/field cross-checking, conflict detection, and goal-completion
checking.

All deterministic — these are the checks the Evaluator Agent and Report
Generator Agent lean on so "evidence-grounded" is actually verified, not
asserted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolContext, limit_results


class SentenceWithCitations(BaseModel):
    text: str
    citation_passage_ids: list[str] = Field(default_factory=list)


# --- validate_citations --------------------------------------------------------


class ValidateCitationsInput(BaseModel):
    sentences: list[SentenceWithCitations]
    valid_passage_ids: list[str]


class InvalidCitation(BaseModel):
    sentence_text: str
    passage_id: str


class ValidateCitationsOutput(BaseModel):
    invalid_citations: list[InvalidCitation] = Field(default_factory=list)
    precision: float = 1.0  # fraction of citations that reference a real passage


class ValidateCitationsTool(BaseTool[ValidateCitationsInput, ValidateCitationsOutput]):
    name = "validate_citations"
    purpose = "Check every citation actually points at a passage_id that exists."
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: ValidateCitationsInput, ctx: ToolContext
    ) -> ValidateCitationsOutput:
        valid_ids = set(tool_input.valid_passage_ids)
        invalid: list[InvalidCitation] = []
        total = 0
        for sentence in tool_input.sentences:
            for pid in sentence.citation_passage_ids:
                total += 1
                if pid not in valid_ids:
                    invalid.append(InvalidCitation(sentence_text=sentence.text, passage_id=pid))
        precision = 1.0 if total == 0 else (total - len(invalid)) / total
        limited, _ = limit_results(invalid, 100)
        return ValidateCitationsOutput(invalid_citations=limited, precision=round(precision, 4))


# --- detect_unsupported_claims --------------------------------------------------------


class DetectUnsupportedClaimsInput(BaseModel):
    sentences: list[SentenceWithCitations]


class DetectUnsupportedClaimsOutput(BaseModel):
    unsupported_sentences: list[str] = Field(default_factory=list)
    unsupported_claim_rate: float = 0.0


class DetectUnsupportedClaimsTool(
    BaseTool[DetectUnsupportedClaimsInput, DetectUnsupportedClaimsOutput]
):
    name = "detect_unsupported_claims"
    purpose = "Flag narrative sentences that carry zero citations."
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: DetectUnsupportedClaimsInput, ctx: ToolContext
    ) -> DetectUnsupportedClaimsOutput:
        sentences = tool_input.sentences
        if not sentences:
            return DetectUnsupportedClaimsOutput()
        unsupported = [s.text for s in sentences if not s.citation_passage_ids]
        limited, _ = limit_results(unsupported, 100)
        return DetectUnsupportedClaimsOutput(
            unsupported_sentences=limited,
            unsupported_claim_rate=round(len(unsupported) / len(sentences), 4),
        )


# --- compare_report_with_fields --------------------------------------------------------


class CompareReportWithFieldsInput(BaseModel):
    report_text: str = Field(max_length=1_000_000)
    expected_fields: dict[str, str | None]


class CompareReportWithFieldsOutput(BaseModel):
    fields_missing_from_report: list[str] = Field(default_factory=list)


class CompareReportWithFieldsTool(
    BaseTool[CompareReportWithFieldsInput, CompareReportWithFieldsOutput]
):
    name = "compare_report_with_fields"
    purpose = (
        "Check that every known non-null field value is actually reflected "
        "somewhere in the report text (a substring presence check, not "
        "semantic — cheap and deterministic)."
    )
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: CompareReportWithFieldsInput, ctx: ToolContext
    ) -> CompareReportWithFieldsOutput:
        missing = [
            field
            for field, value in tool_input.expected_fields.items()
            if value is not None and value not in tool_input.report_text
        ]
        return CompareReportWithFieldsOutput(fields_missing_from_report=sorted(missing))


# --- detect_conflicting_values --------------------------------------------------------


class DetectConflictingValuesInput(BaseModel):
    values_by_source: dict[str, dict[str, str | None]]  # source_name -> field -> value


class DetectConflictingValuesOutput(BaseModel):
    conflicting_fields: list[str] = Field(default_factory=list)
    distinct_values: dict[str, list[str]] = Field(default_factory=dict)


class DetectConflictingValuesTool(
    BaseTool[DetectConflictingValuesInput, DetectConflictingValuesOutput]
):
    name = "detect_conflicting_values"
    purpose = "Find fields where two or more sources report different non-null values."
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: DetectConflictingValuesInput, ctx: ToolContext
    ) -> DetectConflictingValuesOutput:
        by_field: dict[str, set[str]] = {}
        for source_values in tool_input.values_by_source.values():
            for field, value in source_values.items():
                if value is not None:
                    by_field.setdefault(field, set()).add(value)

        conflicting = sorted(field for field, values in by_field.items() if len(values) > 1)
        distinct_values = {
            field: sorted(values) for field, values in by_field.items() if len(values) > 1
        }
        return DetectConflictingValuesOutput(
            conflicting_fields=conflicting, distinct_values=distinct_values
        )


# --- validate_goal_completion --------------------------------------------------------


class SuccessCriterionStatus(BaseModel):
    criterion_id: str
    met: bool


class ValidateGoalCompletionInput(BaseModel):
    success_criteria: list[SuccessCriterionStatus]


class ValidateGoalCompletionOutput(BaseModel):
    all_met: bool
    unmet_criteria: list[str] = Field(default_factory=list)


class ValidateGoalCompletionTool(
    BaseTool[ValidateGoalCompletionInput, ValidateGoalCompletionOutput]
):
    name = "validate_goal_completion"
    purpose = "Deterministically check whether every success criterion is met."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: ValidateGoalCompletionInput, ctx: ToolContext
    ) -> ValidateGoalCompletionOutput:
        unmet = [c.criterion_id for c in tool_input.success_criteria if not c.met]
        return ValidateGoalCompletionOutput(
            all_met=len(unmet) == 0 and len(tool_input.success_criteria) > 0,
            unmet_criteria=unmet,
        )
