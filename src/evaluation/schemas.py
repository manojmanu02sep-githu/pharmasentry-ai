"""Typed schemas for the golden evaluation dataset.

A `GoldenCase` is ground truth for one synthetic case: the raw email and
attachment text a case would start from, plus every "expected_*" label
future agents will be scored against (Phase 6+). This is deliberately a
separate, flatter schema from `src.models` (the runtime models agents
populate) — golden labels are plain expected values, not `FieldValue`
objects with citations, because there is no run to cite yet.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

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
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    CONFLICTING = "conflicting"
    NON_SAFETY = "non_safety"


class ExpectedMinimumCriteria(BaseModel):
    has_identifiable_patient: bool
    has_identifiable_reporter: bool
    has_suspect_product: bool
    has_adverse_event: bool
    missing_criteria: list[str] = Field(default_factory=list)

    @property
    def meets_minimum_criteria(self) -> bool:
        return (
            self.has_identifiable_patient
            and self.has_identifiable_reporter
            and self.has_suspect_product
            and self.has_adverse_event
        )


class GoldenCase(BaseModel):
    case_id: str
    category: GoldenCaseCategory
    is_safety_report: bool

    email_text: str
    attachment_text: str = ""

    expected_extracted_fields: dict[str, str | None] = Field(default_factory=dict)
    expected_minimum_criteria: ExpectedMinimumCriteria
    expected_seriousness_indicators: list[str] = Field(default_factory=list)
    expected_missing_fields: list[str] = Field(default_factory=list)
    expected_duplicate_family: str
    expected_narrative_facts: list[str] = Field(default_factory=list)

    # Extra metadata beyond the minimum requested fields — useful for
    # conflicting-source cases and for humans auditing the dataset.
    conflicting_fields: list[str] = Field(default_factory=list)
    notes: str = ""

    @field_validator("expected_extracted_fields")
    @classmethod
    def _keys_are_known(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        unknown = set(v) - set(FIELD_KEYS)
        if unknown:
            raise ValueError(f"Unknown expected_extracted_fields keys: {sorted(unknown)}")
        return v

    @field_validator("expected_missing_fields", "conflicting_fields")
    @classmethod
    def _list_keys_are_known(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(FIELD_KEYS)
        if unknown:
            raise ValueError(f"Unknown field keys: {sorted(unknown)}")
        return v


class DatasetSplit(BaseModel):
    train: list[str]
    test: list[str]

    def leakage_free(self) -> bool:
        return not (set(self.train) & set(self.test))


class DatasetManifest(BaseModel):
    schema_version: str = "1.0"
    seed: int
    generated_at: datetime
    total_cases: int
    category_counts: dict[str, int]
    synthetic_notice: str = (
        "All records are fabricated for this educational prototype. No real "
        "patient, reporter, product, or company data is present."
    )
