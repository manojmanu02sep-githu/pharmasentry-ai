"""Report Generator: drafts a source-cited narrative (via the LLM
abstraction, using only validated field values as facts) and an editable
follow-up-question draft for any missing information. Validates its own
output against citation/unsupported-claim/field-coverage checks before
handing off, flagging for human review whenever those checks find gaps."""

from __future__ import annotations

import time
from uuid import uuid4

from src.agents._common import call_llm_or_none, call_tool, make_agent_event, make_tool_context
from src.llm.schemas import NarrativeDraftResult
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.evidence import Citation
from src.models.missing_info import FollowUpDraft
from src.models.reasoning import AgentDecision
from src.models.report import ReportSection, TriageReport
from src.tools.quality import (
    CompareReportWithFieldsInput,
    DetectUnsupportedClaimsInput,
    SentenceWithCitations,
    ValidateCitationsInput,
)
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.REPORT_GENERATOR


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    fields = state.get("extracted_fields")
    passages = state.get("source_passages") or []
    passage_ids = {p.passage_id for p in passages}

    facts: list[str] = []
    sentences: list[SentenceWithCitations] = []
    expected_fields: dict[str, str | None] = {}
    if fields is not None:
        for field_value in (
            fields.patient.age,
            fields.patient.sex,
            fields.product.product_name,
            fields.event.event_description,
            fields.treatment.action_taken,
            fields.outcome.outcome_description,
            fields.outcome.hospitalized,
        ):
            expected_fields[field_value.field_name] = field_value.value
            if field_value.value is None:
                continue
            fact = f"{field_value.field_name.split('.')[-1].replace('_', ' ')}: {field_value.value}"
            facts.append(fact)
            sentences.append(
                SentenceWithCitations(
                    text=fact, citation_passage_ids=[c.passage_id for c in field_value.citations]
                )
            )

    narrative_result, _ = call_llm_or_none(NarrativeDraftResult, context={"facts": facts})
    narrative_text = narrative_result.narrative_text if narrative_result else "; ".join(facts)

    citation_output = call_tool(
        "validate_citations",
        ValidateCitationsInput(sentences=sentences, valid_passage_ids=list(passage_ids)),
        ctx,
        tool_events,
    )
    unsupported_output = call_tool(
        "detect_unsupported_claims",
        DetectUnsupportedClaimsInput(sentences=sentences),
        ctx,
        tool_events,
    )
    coverage_output = call_tool(
        "compare_report_with_fields",
        CompareReportWithFieldsInput(report_text=narrative_text, expected_fields=expected_fields),
        ctx,
        tool_events,
    )

    sections = [
        ReportSection(
            text=s.text,
            citations=[
                Citation(passage_id=pid, quoted_text=s.text)
                for pid in s.citation_passage_ids
                if pid in passage_ids
            ],
        )
        for s in sentences
    ]
    triage_report = TriageReport(
        report_id=f"report_{uuid4().hex[:12]}",
        case_id=case_id,
        sections=sections,
        unsupported_claim_flags=unsupported_output.unsupported_sentences,
    )

    missing_information = state.get("missing_information") or []
    follow_up_draft = None
    if missing_information:
        questions_text = "\n".join(f"- {m.follow_up_question}" for m in missing_information)
        follow_up_draft = FollowUpDraft(
            draft_id=f"followup_{uuid4().hex[:12]}",
            case_id=case_id,
            subject="Follow-up questions regarding your recent safety report",
            body=f"Thank you for your report. To complete our preliminary review, could you "
            f"please provide the following additional information?\n\n{questions_text}",
            editable=True,
            questions=missing_information,
        )

    has_gaps = bool(
        citation_output.invalid_citations
        or unsupported_output.unsupported_sentences
        or coverage_output.fields_missing_from_report
    )
    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="narrative_drafted",
            status=ToolCallStatus.SUCCESS if not has_gaps else ToolCallStatus.ERROR,
            summary=(
                f"{len(sections)} sentence(s) drafted; "
                f"citation_precision={citation_output.precision:.2f}; "
                f"unsupported_claim_rate={unsupported_output.unsupported_claim_rate:.2f}."
            ),
        ),
        ctx,
        tool_events,
    )

    follow_up_summary = (
        "a follow-up question draft"
        if follow_up_draft
        else "no follow-up draft (no missing fields)"
    )
    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="draft_ready",
        evidence=[f"citation_precision={citation_output.precision:.2f}"],
        confidence=citation_output.precision,
        decision_summary=(
            f"Drafted narrative ({len(sections)} sentences) and {follow_up_summary}; "
            f"unsupported_claims={len(unsupported_output.unsupported_sentences)}."
        ),
        next_action=DecisionOutcome.ESCALATE_TO_HUMAN if has_gaps else DecisionOutcome.PROCEED,
        requires_human_review=True,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="report_generator",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    result: dict = {
        "triage_report": triage_report,
        "current_step": "report_generator",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
    if follow_up_draft is not None:
        result["follow_up_draft"] = follow_up_draft
    return result
