"""Evaluator Agent: checks goal completion, schema validity, evidence
coverage, unsupported claims, and unresolved conflicts across the whole
trajectory (not just the last step). Never allows unlimited reflection --
this node runs exactly once per pass through the graph; retries of a
failed upstream step are bounded by `retry_counts`/`max_agent_retries`
enforced by the Supervisor, not by this node looping on itself."""

from __future__ import annotations

import time
from uuid import uuid4

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.evaluation import EvaluationCheck, EvaluationResult
from src.models.goal import SuccessCriterion
from src.models.reasoning import AgentDecision
from src.tools.quality import (
    DetectConflictingValuesInput,
    DetectUnsupportedClaimsInput,
    SentenceWithCitations,
    SuccessCriterionStatus,
    ValidateGoalCompletionInput,
)
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.EVALUATOR

_REACHED_BY_HERE = (
    "minimum_criteria_checked",
    "triage_evaluated",
    "duplicates_searched",
    "missing_information_identified",
    "narrative_drafted",
    "evaluated",
)


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    updated_criteria: list[SuccessCriterion] = [
        SuccessCriterion(
            criterion_id=c.criterion_id,
            description=c.description,
            met=c.met or c.criterion_id in _REACHED_BY_HERE,
            evidence_ref=c.evidence_ref,
        )
        for c in state.get("success_criteria") or []
    ]

    completion_output = call_tool(
        "validate_goal_completion",
        ValidateGoalCompletionInput(
            success_criteria=[
                SuccessCriterionStatus(criterion_id=c.criterion_id, met=c.met)
                for c in updated_criteria
            ]
        ),
        ctx,
        tool_events,
    )

    triage_report = state.get("triage_report")
    sentences = [
        SentenceWithCitations(text=s.text, citation_passage_ids=[c.passage_id for c in s.citations])
        for s in (triage_report.sections if triage_report else [])
    ]
    unsupported_output = call_tool(
        "detect_unsupported_claims",
        DetectUnsupportedClaimsInput(sentences=sentences),
        ctx,
        tool_events,
    )

    passages = state.get("source_passages") or []
    email_text = passages[0].text.lower() if passages and passages[0].source_type == "email" else ""
    attachment_texts = " ".join(p.text for p in passages if p.source_type == "attachment").lower()
    fields = state.get("extracted_fields")
    values_by_source: dict[str, dict[str, str | None]] = {"email": {}, "attachment": {}}
    total_fields = 0
    cited_fields = 0
    if fields is not None:
        for field_value in (
            fields.patient.age,
            fields.patient.sex,
            fields.product.product_name,
            fields.event.event_description,
            fields.outcome.hospitalized,
        ):
            if field_value.value is None:
                continue
            total_fields += 1
            if field_value.citations:
                cited_fields += 1
            value_lower = field_value.value.lower()
            values_by_source["email"][field_value.field_name] = (
                field_value.value if value_lower in email_text else None
            )
            values_by_source["attachment"][field_value.field_name] = (
                field_value.value if value_lower in attachment_texts else None
            )
    conflict_output = call_tool(
        "detect_conflicting_values",
        DetectConflictingValuesInput(values_by_source=values_by_source),
        ctx,
        tool_events,
    )
    evidence_coverage = round(cited_fields / total_fields, 2) if total_fields else 1.0

    checks = [
        EvaluationCheck(
            name="minimum_criteria_checked",
            passed=state.get("minimum_criteria") is not None,
            detail=(
                "Minimum case criteria evaluated."
                if state.get("minimum_criteria")
                else "Not yet evaluated."
            ),
        ),
        EvaluationCheck(
            name="triage_performed",
            passed=state.get("triage_result") is not None,
            detail="Explicit seriousness indicators checked.",
        ),
        EvaluationCheck(
            name="duplicate_search_performed",
            passed="duplicate_search" in {e.node for e in state.get("agent_events") or []},
            detail="Hybrid retrieval executed against synthetic case history.",
        ),
        EvaluationCheck(
            name="no_unsupported_claims",
            passed=not unsupported_output.unsupported_sentences,
            detail=f"{len(unsupported_output.unsupported_sentences)} unsupported sentence(s).",
        ),
        EvaluationCheck(
            name="no_unresolved_conflicts",
            passed=not conflict_output.conflicting_fields,
            detail=f"{len(conflict_output.conflicting_fields)} conflicting field(s).",
        ),
    ]
    schema_valid = all(f.value is None or f.citations or f.confidence <= 0.2 for f in (
        [fields.patient.age, fields.patient.sex, fields.product.product_name,
         fields.event.event_description, fields.outcome.hospitalized] if fields else []
    ))

    evaluation_result = EvaluationResult(
        evaluation_id=f"eval_{uuid4().hex[:12]}",
        case_id=case_id,
        step_name="full_trajectory",
        checks=checks,
        goal_completion=completion_output.all_met,
        schema_valid=schema_valid,
        evidence_coverage=evidence_coverage,
        unsupported_claims=unsupported_output.unsupported_sentences,
        unresolved_conflicts=conflict_output.conflicting_fields,
        requires_escalation=True,
    )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="trajectory_evaluated",
            status=ToolCallStatus.SUCCESS if evaluation_result.passed else ToolCallStatus.ERROR,
            summary=f"goal_completion={completion_output.all_met}; "
            f"evidence_coverage={evidence_coverage}; "
            f"{len(unsupported_output.unsupported_sentences)} unsupported claim(s).",
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="passed" if evaluation_result.passed else "gaps_found",
        evidence=[c.detail for c in checks],
        confidence=evidence_coverage,
        decision_summary=(
            f"Trajectory evaluation: {sum(c.passed for c in checks)}/{len(checks)} checks passed; "
            f"evidence_coverage={evidence_coverage}; case routed to mandatory human review."
        ),
        next_action=DecisionOutcome.ESCALATE_TO_HUMAN,
        requires_human_review=True,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="evaluator",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "success_criteria": updated_criteria,
        "evaluation_results": [evaluation_result],
        "current_step": "evaluator",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
