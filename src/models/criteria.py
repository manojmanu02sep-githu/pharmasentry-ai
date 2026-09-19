"""Minimum-case-criteria status: patient, reporter, suspect product, event."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MinimumCriteriaResult(BaseModel):
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
