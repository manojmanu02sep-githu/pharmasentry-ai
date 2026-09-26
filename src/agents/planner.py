"""Planner Agent: creates the case-specific execution plan (which pipeline
steps run, in order, and why any are skipped). Persists plan revisions
with reasons; a later node (e.g. Supervisor on retry) may call
`revise_plan` to append a new PlanRevision."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import uuid4

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, ToolCallStatus
from src.models.plan import ExecutionPlan, PlanRevision, PlanStep
from src.models.reasoning import AgentDecision
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.PLANNER

_BASE_STEPS: tuple[tuple[AgentName, str], ...] = (
    (AgentName.SUBJECT_READER, "Validate and parse the inbound email."),
    (AgentName.EMAIL_BODY_READER, "Extract and classify the email body for safety relevance."),
    (AgentName.ATTACHMENT_READER, "Extract attachment text (PDF/OCR) with page-level evidence."),
    (
        AgentName.MEDICAL_EXTRACTION,
        "Extract structured patient/reporter/product/event fields with citations.",
    ),
    (AgentName.TRIAGE, "Detect explicit seriousness indicators (human confirmation required)."),
    (AgentName.DUPLICATE_SEARCH, "Search synthetic case history for potential duplicates."),
    (AgentName.REPORT_GENERATOR, "Draft a source-cited narrative and follow-up questions."),
    (AgentName.EVALUATOR, "Evaluate evidence coverage, unsupported claims, and goal completion."),
    (AgentName.HUMAN_REVIEW, "Pause for authorized human review."),
)


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []

    attachments = state.get("attachments") or []
    steps: list[PlanStep] = []
    for agent, description in _BASE_STEPS:
        skip = False
        skip_reason = None
        if agent == AgentName.ATTACHMENT_READER and not attachments:
            skip = True
            skip_reason = "No attachments present on this case."
        steps.append(
            PlanStep(
                step_id=f"step_{agent.value}",
                agent=agent,
                description=description,
                skip=skip,
                skip_reason=skip_reason,
            )
        )

    plan = ExecutionPlan(
        plan_id=f"plan_{uuid4().hex[:12]}", case_id=case_id, steps=steps, version=1
    )
    revision = PlanRevision(
        plan=plan,
        reason="Initial plan created from case inputs.",
        revised_at=datetime.now(UTC),
    )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="plan_created",
            status=ToolCallStatus.SUCCESS,
            summary=(
                f"Plan v1 created with {len(steps)} steps "
                f"({sum(s.skip for s in steps)} skipped)."
            ),
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="plan_created",
        evidence=[f"{len(attachments)} attachment(s) on case."],
        confidence=1.0,
        decision_summary=f"Created execution plan v1 with {len(steps)} steps; attachment step "
        f"{'skipped (no attachments)' if not attachments else 'included'}.",
        next_action=DecisionOutcome.PROCEED,
        requires_human_review=False,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="planner",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "execution_plan": plan,
        "plan_history": [revision],
        "current_step": "planner",
        "agent_events": [agent_event],
        "tool_events": tool_events,
    }
