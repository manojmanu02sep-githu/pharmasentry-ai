"""Structured extracted fields, grouped by the case's factual dimensions.

Every leaf is a FieldValue so a reviewer always sees value + source +
citation + confidence + conflict status (Case Workspace requirement).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.evidence import FieldValue


class PatientInfo(BaseModel):
    age: FieldValue | None = None
    sex: FieldValue | None = None
    relevant_history: list[FieldValue] = Field(default_factory=list)


class ReporterInfo(BaseModel):
    """Reporter identity/contact fields.

    NOTE: per Context Isolation and Delegation, this object (or fields within
    it) must never be included in the handoff passed to the Duplicate Agent.
    """

    reporter_type: FieldValue | None = None
    name: FieldValue | None = None
    contact: FieldValue | None = None


class ProductInfo(BaseModel):
    product_name: FieldValue | None = None
    dose: FieldValue | None = None
    route: FieldValue | None = None
    treatment_start_date: FieldValue | None = None
    treatment_end_date: FieldValue | None = None


class EventInfo(BaseModel):
    event_description: FieldValue | None = None
    event_onset_date: FieldValue | None = None


class TreatmentInfo(BaseModel):
    action_taken: FieldValue | None = None
    concomitant_medications: list[FieldValue] = Field(default_factory=list)


class OutcomeInfo(BaseModel):
    outcome_description: FieldValue | None = None
    hospitalized: FieldValue | None = None


class ExtractedFields(BaseModel):
    patient: PatientInfo = Field(default_factory=PatientInfo)
    reporter: ReporterInfo = Field(default_factory=ReporterInfo)
    product: ProductInfo = Field(default_factory=ProductInfo)
    event: EventInfo = Field(default_factory=EventInfo)
    treatment: TreatmentInfo = Field(default_factory=TreatmentInfo)
    outcome: OutcomeInfo = Field(default_factory=OutcomeInfo)
