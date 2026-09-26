"""Subject Reader: first Intake-stage node. Confirms the inbound email is
well-formed (required fields present) and defensively validates the body
and each attachment as untrusted content -- signature/type checks guard
against malformed, unsupported, or disguised payloads before anything
downstream trusts this data. Persists each attachment's real
`validation_status` (rather than leaving the model default) so the
graph's "unsupported or unsafe file" routing can branch on it."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, FileValidationStatus, ToolCallStatus
from src.models.intake import Attachment
from src.models.reasoning import AgentDecision
from src.tools.intake import ValidateFileInput, ValidateFileSignatureInput
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.SUBJECT_READER


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []
    errors: list = []

    email = state["email"]
    missing_fields = [
        name for name in ("sender", "subject", "body") if not getattr(email, name, "").strip()
    ]

    body_bytes = email.body.encode("utf-8")
    validate_output = call_tool(
        "validate_file",
        ValidateFileInput(
            filename=f"{email.message_id}_body.txt",
            media_type="text/plain",
            size_bytes=len(body_bytes),
        ),
        ctx,
        tool_events,
    )
    signature_output = call_tool(
        "validate_file_signature",
        ValidateFileSignatureInput(
            filename=f"{email.message_id}_body.txt",
            content_base64=base64.b64encode(body_bytes).decode("ascii"),
            declared_media_type="text/plain",
        ),
        ctx,
        tool_events,
    )
    body_valid = (
        validate_output.status == FileValidationStatus.VALID
        and signature_output.matches_declared
    )

    updated_attachments: list[Attachment] = []
    invalid_attachment_names: list[str] = []
    for attachment in state.get("attachments") or []:
        try:
            raw_bytes = Path(attachment.storage_path).read_bytes()
        except OSError as exc:
            errors.append(
                {
                    "case_id": case_id,
                    "trace_id": trace_id,
                    "node": "subject_reader",
                    "error_type": "attachment_read_failed",
                    "message": f"Could not read {attachment.storage_path}: {exc}",
                    "recoverable": False,
                }
            )
            updated_attachments.append(
                attachment.model_copy(
                    update={"validation_status": FileValidationStatus.UNSUPPORTED_TYPE}
                )
            )
            invalid_attachment_names.append(attachment.filename)
            continue

        att_validate_output = call_tool(
            "validate_file",
            ValidateFileInput(
                filename=attachment.filename,
                media_type=attachment.media_type,
                size_bytes=len(raw_bytes),
            ),
            ctx,
            tool_events,
        )
        att_signature_output = call_tool(
            "validate_file_signature",
            ValidateFileSignatureInput(
                filename=attachment.filename,
                content_base64=base64.b64encode(raw_bytes).decode("ascii"),
                declared_media_type=attachment.media_type,
            ),
            ctx,
            tool_events,
        )
        attachment_valid = (
            att_validate_output.status == FileValidationStatus.VALID
            and att_signature_output.matches_declared
        )
        if att_validate_output.status != FileValidationStatus.VALID:
            status = att_validate_output.status
        elif att_signature_output.matches_declared:
            status = FileValidationStatus.VALID
        else:
            status = FileValidationStatus.UNSUPPORTED_TYPE
        updated_attachments.append(attachment.model_copy(update={"validation_status": status}))
        if not attachment_valid:
            invalid_attachment_names.append(attachment.filename)

    is_valid = not missing_fields and body_valid and not invalid_attachment_names
    next_action = DecisionOutcome.PROCEED if is_valid else DecisionOutcome.ESCALATE_TO_HUMAN

    summary_parts = []
    if missing_fields:
        summary_parts.append(f"missing required field(s): {', '.join(missing_fields)}")
    if validate_output.status != FileValidationStatus.VALID:
        summary_parts.append(f"body validation status={validate_output.status.value}")
    if not signature_output.matches_declared:
        summary_parts.append("declared media type did not match detected signature")
    if invalid_attachment_names:
        summary_parts.append(
            f"unsupported/unsafe attachment(s): {', '.join(invalid_attachment_names)}"
        )
    summary = (
        "; ".join(summary_parts)
        if summary_parts
        else "email well-formed and body/attachment signatures verified"
    )

    call_tool(
        "record_audit_event",
        RecordAuditEventInput(
            case_id=case_id,
            trace_id=trace_id,
            agent=NODE,
            action_name="subject_validated",
            status=ToolCallStatus.SUCCESS if is_valid else ToolCallStatus.ERROR,
            summary=summary[:500],
        ),
        ctx,
        tool_events,
    )

    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="valid" if is_valid else "invalid",
        evidence=[f"sender={email.sender!r}", f"subject={email.subject!r}"],
        confidence=1.0,
        decision_summary=f"Email intake validation: {summary}.",
        next_action=next_action,
        requires_human_review=not is_valid,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="subject_reader",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "attachments": updated_attachments,
        "current_step": "subject_reader",
        "agent_events": [agent_event],
        "tool_events": tool_events,
        "errors": errors,
    }
