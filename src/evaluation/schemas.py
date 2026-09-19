"""Typed schemas for the golden evaluation dataset.

A `GoldenCase` is ground truth for one synthetic case: the raw email and
attachment a case would start from, plus every "expected_*" label future
agents are scored against (Phase 7+). This is deliberately a separate,
flatter schema from `src.models` (the runtime models agents populate) —
golden labels are plain expected values, not `FieldValue` objects with
citations, because there is no run to cite yet. Where a runtime enum
already exists and fits (routing reasons), we reuse it rather than
duplicating it, per the "extend, don't duplicate" project rule.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.models.enums import RouteReason

# Flattened dotted paths mirroring src/models/extraction.py's structure.
# A GoldenCase's expected_extracted_fields dict may only use these keys.
FIELD_KEYS: tuple[str, ...] = (
    "patient.age",
    "patient.sex",
    "reporter.reporter_type",
    "reporter.name",
    "product.product_name",
    "product.dose",
    "product.route",
    "product.treatment_start_date",
    "product.treatment_end_date",
    "event.event_description",
    "event.event_onset_date",
    "treatment.action_taken",
    "outcome.outcome_description",
    "outcome.hospitalized",
)


class GoldenCaseCategory(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    SERIOUS = "serious"
    NON_SERIOUS = "non_serious"
    NON_SAFETY = "non_safety"
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SIMILAR_NON_DUPLICATE = "similar_non_duplicate"
    CONFLICTING = "conflicting"
    MISSING_SUSPECT_PRODUCT = "missing_suspect_product"
    MISSING_ADVERSE_EVENT = "missing_adverse_event"
    MISSING_REPORTER = "missing_reporter"
    MISSING_IDENTIFIABLE_PATIENT = "missing_identifiable_patient"
    POOR_OCR = "poor_ocr"
    MULTILINGUAL = "multilingual"
    PRODUCT_ALIAS = "product_alias"
    EVENT_SYNONYM = "event_synonym"
    MALFORMED_DATE = "malformed_date"
    PROMPT_INJECTION = "prompt_injection"
    TOOL_INJECTION = "tool_injection"
    APPROVAL_BYPASS_ATTEMPT = "approval_bypass_attempt"


class SafetyClassification(str, Enum):
    SAFETY_REPORT = "safety_report"
    NON_SAFETY = "non_safety"
    UNCERTAIN = "uncertain"


class AttachmentMetadata(BaseModel):
    """Metadata a real Document Agent would record about the attachment.

    ``has_attachment=False`` (with the rest at their defaults) represents a
    case with no attachment at all, e.g. non-safety inquiries.
    """

    has_attachment: bool = False
    filename: str = ""
    media_type: str = ""
    size_bytes: int = 0
    page_count: int = 0
    ocr_applied: bool = False
    ocr_quality: Literal["good", "degraded", "unreadable", "not_applicable"] = "not_applicable"


class ExpectedMinimumCriteria(BaseModel):
    has_identifiable_patient: bool
    has_identifiable_reporter: bool
    has_suspect_product: bool
    has_adverse_event: bool
    missing_criteria: list[str] = Field(default_factory=list)
    # Set explicitly by the generator rather than always derived, so a case
    # can be labeled "uncertain" (ambiguous evidence) instead of flatly
    # "incomplete" (evidence absent) — mirrors the Minimum Criteria Agent's
    # three-way return value in CLAUDE.md.
    status: Literal["complete", "incomplete", "uncertain"] = "complete"

    @property
    def meets_minimum_criteria(self) -> bool:
        return (
            self.has_identifiable_patient
            and self.has_identifiable_reporter
            and self.has_suspect_product
            and self.has_adverse_event
        )

    @model_validator(mode="after")
    def _status_consistent_with_criteria(self) -> ExpectedMinimumCriteria:
        if self.meets_minimum_criteria and self.status != "complete":
            raise ValueError("status must be 'complete' when all four criteria are met")
        if not self.meets_minimum_criteria and self.status == "complete":
            raise ValueError("status cannot be 'complete' when a criterion is unmet")
        return self


class GoldenCase(BaseModel):
    case_id: str
    case_type: GoldenCaseCategory
    duplicate_family_id: str
    expected_duplicate_matches: list[str] = Field(default_factory=list)

    expected_safety_classification: SafetyClassification

    email_subject: str
    email_body: str
    attachment_metadata: AttachmentMetadata = Field(default_factory=AttachmentMetadata)
    attachment_text: str = ""

    expected_extracted_fields: dict[str, str | None] = Field(default_factory=dict)
    expected_source_evidence: list[str] = Field(default_factory=list)
    expected_minimum_criteria: ExpectedMinimumCriteria
    expected_seriousness_indicators: list[str] = Field(default_factory=list)
    expected_missing_fields: list[str] = Field(default_factory=list)
    expected_conflicting_fields: list[str] = Field(default_factory=list)
    expected_routing_decision: RouteReason = RouteReason.NORMAL
    expected_narrative_facts: list[str] = Field(default_factory=list)
    prohibited_conclusions: list[str] = Field(default_factory=list)
    expected_human_review_required: bool = True

    synthetic_data: Literal[True] = True
    notes: str = ""

    @property
    def is_safety_report(self) -> bool:
        return self.expected_safety_classification == SafetyClassification.SAFETY_REPORT

    @property
    def email_text(self) -> str:
        """Full rendered email, for tools/tests that want one blob of text."""
        return f"Subject: {self.email_subject}\n\n{self.email_body}"

    @field_validator("expected_extracted_fields")
    @classmethod
    def _keys_are_known(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        unknown = set(v) - set(FIELD_KEYS)
        if unknown:
            raise ValueError(f"Unknown expected_extracted_fields keys: {sorted(unknown)}")
        return v

    @field_validator("expected_missing_fields", "expected_conflicting_fields")
    @classmethod
    def _list_keys_are_known(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(FIELD_KEYS)
        if unknown:
            raise ValueError(f"Unknown field keys: {sorted(unknown)}")
        return v


class DatasetSplit(BaseModel):
    """Three-way, duplicate-family-aware split. ``validation`` may be empty
    for datasets too small to justify a third slice, but never overlaps
    train/test when populated."""

    train: list[str]
    validation: list[str] = Field(default_factory=list)
    test: list[str]

    def leakage_free(self) -> bool:
        train_set, val_set, test_set = set(self.train), set(self.validation), set(self.test)
        return not (
            (train_set & val_set) or (train_set & test_set) or (val_set & test_set)
        )


class DatasetManifest(BaseModel):
    schema_version: str = "2.0"
    seed: int
    generated_at: datetime
    total_cases: int
    category_counts: dict[str, int]
    synthetic_notice: str = (
        "All records are fabricated for this educational prototype. No real "
        "patient, reporter, product, or company data is present."
    )
