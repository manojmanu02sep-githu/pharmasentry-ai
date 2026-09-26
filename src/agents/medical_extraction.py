"""Medical Extraction Agent: extracts explicit patient/reporter/product/
event/treatment/outcome fields with citations, using deterministic
pattern rules (the mock LLM has no structured-extraction response model,
per CLAUDE.md's "deterministic code for exact rules" guidance) plus the
domain lookup/normalization tools. Also runs the minimum-case-criteria
check and derives missing-information items from unset fields."""

from __future__ import annotations

import re
import time

from src.agents._common import call_tool, make_agent_event, make_field_value, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.extraction import (
    EventInfo,
    ExtractedFields,
    OutcomeInfo,
    PatientInfo,
    ProductInfo,
    ReporterInfo,
    TreatmentInfo,
)
from src.models.missing_info import MissingInformationItem
from src.models.reasoning import AgentDecision
from src.tools.domain import (
    CheckMinimumCaseCriteriaInput,
    LookupEventTermInput,
    LookupProductAliasInput,
)
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.MEDICAL_EXTRACTION

_AGE_RE = re.compile(r"\b(\d{1,3})\s*[- ]year[- ]old\b", re.IGNORECASE)
_SEX_RE = re.compile(r"\b(male|female)\b", re.IGNORECASE)
_REPORTER_RE = re.compile(r"\b([A-Z]\.?\s?[A-Za-z]+),\s*(NP|MD|RN|PharmD|DO)\b")
_CREDENTIAL_TO_TYPE = {
    "NP": "Nurse Practitioner",
    "MD": "Physician",
    "RN": "Registered Nurse",
    "PharmD": "Pharmacist",
    "DO": "Physician",
}
_EVENT_CANDIDATES = (
    "abdominal pain",
    "pancreatitis",
    "vomiting",
    "loss of consciousness",
    "severe headache",
    "facial swelling",
    "severe hypoglycemia",
    "injection site reaction",
)
_ACTION_CANDIDATES = (
    "hospitalized", "hospitalised", "admitted to the hospital", "admitted to hospital",
)
_OUTCOME_CANDIDATES = (
    "stable outcome", "recovered", "resolved", "ongoing", "fatal outcome", "died",
)
_HOSPITALIZATION_PHRASES = (
    "hospitalized", "hospitalised", "admitted to the hospital", "admitted to hospital",
)


def _first_match(candidates: tuple[str, ...], text_lower: str) -> str | None:
    for candidate in candidates:
        if candidate in text_lower:
            return candidate
    return None


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    passages = state.get("source_passages") or []
    combined_text = "\n".join(p.text for p in passages)
    text_lower = combined_text.lower()

    age_match = _AGE_RE.search(combined_text)
    age_value = age_match.group(1) if age_match else None
    sex_match = _SEX_RE.search(combined_text)
    sex_value = sex_match.group(1).lower() if sex_match else None

    reporter_match = _REPORTER_RE.search(combined_text)
    reporter_name = reporter_match.group(1) if reporter_match else None
    reporter_credential = reporter_match.group(2) if reporter_match else None
    reporter_type = _CREDENTIAL_TO_TYPE.get(reporter_credential) if reporter_credential else None

    product_output = call_tool(
        "lookup_product_alias", LookupProductAliasInput(text=combined_text), ctx, tool_events
    )
    product_name = product_output.canonical_name
    product_alias_matched = product_output.alias_matched

    event_phrase = _first_match(_EVENT_CANDIDATES, text_lower)
    event_output = None
    if event_phrase:
        event_output = call_tool(
            "lookup_event_term", LookupEventTermInput(text=event_phrase), ctx, tool_events
        )
    event_canonical_term = event_output.canonical_term if event_output is not None else None
    event_synonym_matched = event_output.synonym_matched if event_output is not None else None
    event_has_canonical = bool(event_canonical_term)
    event_value = event_canonical_term if event_has_canonical else event_phrase
    event_citation_text = event_synonym_matched if event_has_canonical else event_phrase

    action_phrase = _first_match(_ACTION_CANDIDATES, text_lower)
    outcome_phrase = _first_match(_OUTCOME_CANDIDATES, text_lower)
    hospitalization_phrase_matched = _first_match(_HOSPITALIZATION_PHRASES, text_lower)
    hospitalized_value = "true" if hospitalization_phrase_matched else None

    patient = PatientInfo(
        age=make_field_value("patient.age", age_value, passages, confidence=0.9),
        sex=make_field_value("patient.sex", sex_value, passages, confidence=0.9),
    )
    reporter = ReporterInfo(
        reporter_type=make_field_value(
            "reporter.reporter_type",
            reporter_type,
            passages,
            confidence=0.7,
            citation_text=reporter_credential,
        ),
        name=make_field_value("reporter.name", reporter_name, passages, confidence=0.7),
        contact=make_field_value("reporter.contact", None, passages, confidence=0.0),
    )
    product = ProductInfo(
        product_name=make_field_value(
            "product.product_name",
            product_name,
            passages,
            confidence=0.85,
            citation_text=product_alias_matched,
        ),
        dose=make_field_value("product.dose", None, passages, confidence=0.0),
        route=make_field_value("product.route", None, passages, confidence=0.0),
        treatment_start_date=make_field_value(
            "product.treatment_start_date", None, passages, confidence=0.0
        ),
        treatment_end_date=make_field_value(
            "product.treatment_end_date", None, passages, confidence=0.0
        ),
    )
    event = EventInfo(
        event_description=make_field_value(
            "event.event_description",
            event_value,
            passages,
            confidence=0.8,
            citation_text=event_citation_text,
        ),
        event_onset_date=make_field_value("event.event_onset_date", None, passages, confidence=0.0),
    )
    treatment = TreatmentInfo(
        action_taken=make_field_value(
            "treatment.action_taken", action_phrase, passages, confidence=0.75
        ),
    )
    outcome = OutcomeInfo(
        outcome_description=make_field_value(
            "outcome.outcome_description", outcome_phrase, passages, confidence=0.75
        ),
        hospitalized=make_field_value(
            "outcome.hospitalized",
            hospitalized_value,
            passages,
            confidence=0.9,
            citation_text=hospitalization_phrase_matched,
        ),
    )
    extracted_fields = ExtractedFields(
        patient=patient,
        reporter=reporter,
        product=product,
        event=event,
        treatment=treatment,
        outcome=outcome,
    )

    has_identifiable_patient = bool(age_value or sex_value)
    has_identifiable_reporter = bool(reporter_name or reporter_type)
    has_suspect_product = bool(product_name)
    has_adverse_event = bool(event_value)
    minimum_criteria = call_tool(
        "check_minimum_case_criteria",
        CheckMinimumCaseCriteriaInput(
            has_identifiable_patient=has_identifiable_patient,
            has_identifiable_reporter=has_identifiable_reporter,
            has_suspect_product=has_suspect_product,
            has_adverse_event=has_adverse_event,
        ),
        ctx,
        tool_events,
    )

    missing_information = []
    for field_value, question in (
        (product.dose, "What was the dose of the suspect product administered?"),
        (
            product.treatment_start_date,
            "What date did the patient begin treatment with the suspect product?",
        ),
        (event.event_onset_date, "What date did the adverse event begin?"),
    ):
        if field_value is None:
            continue
        if field_value.value is None:
            missing_information.append(
                MissingInformationItem(
                    field_name=field_value.field_name,
                    reason="missing",
                    follow_up_question=question,
                )
            )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="fields_extracted",
            status=ToolCallStatus.SUCCESS
            if minimum_criteria.meets_minimum_criteria
            else ToolCallStatus.ERROR,
            summary=(
                f"Extracted fields; "
                f"minimum_criteria_met={minimum_criteria.meets_minimum_criteria}; "
                f"{len(missing_information)} missing field(s)."
            ),
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="extracted",
        evidence=[
            f"product={product_name}",
            f"event={event_value}",
            f"missing_criteria={minimum_criteria.missing_criteria}",
        ],
        confidence=0.8,
        decision_summary=f"Extracted structured fields from {len(passages)} passage(s); "
        f"minimum_criteria_met={minimum_criteria.meets_minimum_criteria}; "
        f"{len(missing_information)} field(s) missing.",
        next_action=DecisionOutcome.PROCEED
        if minimum_criteria.meets_minimum_criteria
        else DecisionOutcome.ESCALATE_TO_HUMAN,
        requires_human_review=not minimum_criteria.meets_minimum_criteria,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="medical_extraction",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "extracted_fields": extracted_fields,
        "minimum_criteria": minimum_criteria,
        "missing_information": missing_information,
        "current_step": "medical_extraction",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
