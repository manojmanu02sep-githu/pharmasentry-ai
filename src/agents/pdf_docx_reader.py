"""Attachment Reader (pdf_docx_reader): extracts text from each attachment
with page-level evidence. Branches on media type -- PDF text extraction,
OCR for images, or direct read for plain-text attachments (the demo
case's discharge summary is a standalone .txt file, not MIME-embedded, so
it is read from `Attachment.storage_path` directly rather than via
`extract_attachments`, which only surfaces multipart-email parts).
Persists each attachment's real `ocr_quality` (rather than leaving the
model default) so the graph's "low OCR quality" routing can branch on it."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from src.agents._common import call_tool, make_agent_event, make_tool_context
from src.models.enums import AgentName, DecisionOutcome, OCRQuality, ToolCallStatus
from src.models.evidence import SourcePassage
from src.models.reasoning import AgentDecision
from src.tools.document import (
    AssessOcrQualityInput,
    ExtractPdfTextInput,
    NormalizeSourcePassagesInput,
    PerformOcrInput,
)
from src.tools.intake import EnforcePageLimitInput
from src.tools.workflow import RecordAuditEventInput

NODE = AgentName.ATTACHMENT_READER
_OCR_SEVERITY = {
    OCRQuality.NOT_APPLICABLE: 0,
    OCRQuality.GOOD: 1,
    OCRQuality.DEGRADED: 2,
    OCRQuality.UNREADABLE: 3,
}


def run(state: dict) -> dict:
    started = time.monotonic()
    case_id = state["case_id"]
    trace_id = state["trace_id"]
    ctx = make_tool_context(case_id, trace_id, NODE)
    tool_events: list = []
    errors: list = []
    passages: list[SourcePassage] = []
    updated_attachments: list = []
    worst_quality = OCRQuality.NOT_APPLICABLE
    processed = 0

    for attachment in state.get("attachments") or []:
        try:
            raw_bytes = Path(attachment.storage_path).read_bytes()
        except OSError as exc:
            errors.append(
                {
                    "case_id": case_id,
                    "trace_id": trace_id,
                    "node": "pdf_docx_reader",
                    "error_type": "attachment_read_failed",
                    "message": f"Could not read {attachment.storage_path}: {exc}",
                    "recoverable": False,
                }
            )
            updated_attachments.append(attachment)
            continue

        raw_text = ""
        ocr_applied = False
        if attachment.media_type == "application/pdf":
            pdf_output = call_tool(
                "extract_pdf_text",
                ExtractPdfTextInput(content_base64=base64.b64encode(raw_bytes).decode("ascii")),
                ctx,
                tool_events,
            )
            raw_text = "\n\n".join(page.text for page in pdf_output.pages)
        elif attachment.media_type.startswith("image/"):
            ocr_output = call_tool(
                "perform_ocr",
                PerformOcrInput(image_base64=base64.b64encode(raw_bytes).decode("ascii")),
                ctx,
                tool_events,
            )
            raw_text = ocr_output.text
            ocr_applied = True
            if ocr_output.status != "ok":
                errors.append(
                    {
                        "case_id": case_id,
                        "trace_id": trace_id,
                        "node": "pdf_docx_reader",
                        "error_type": "ocr_unavailable",
                        "message": (
                            f"OCR {ocr_output.status} for {attachment.filename}: "
                            f"{ocr_output.notes}"
                        ),
                        "recoverable": True,
                    }
                )
        else:
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        quality_output = call_tool(
            "assess_ocr_quality", AssessOcrQualityInput(text=raw_text), ctx, tool_events
        )
        if _OCR_SEVERITY[quality_output.quality] > _OCR_SEVERITY[worst_quality]:
            worst_quality = quality_output.quality
        updated_attachments.append(
            attachment.model_copy(update={"ocr_quality": quality_output.quality})
        )

        call_tool(
            "enforce_page_limit",
            EnforcePageLimitInput(page_count=attachment.page_count or 1),
            ctx,
            tool_events,
        )

        normalize_output = call_tool(
            "normalize_source_passages",
            NormalizeSourcePassagesInput(
                raw_text=raw_text,
                source_document_id=attachment.attachment_id,
                source_type="attachment",
            ),
            ctx,
            tool_events,
        )
        passages.extend(
            SourcePassage(
                passage_id=p.passage_id,
                source_document_id=attachment.attachment_id,
                source_type="attachment",
                page_number=None,
                text=p.text,
                ocr_applied=ocr_applied,
            )
            for p in normalize_output.passages
        )
        processed += 1

        call_tool(
            "record_audit_event",
            RecordAuditEventInput(
                case_id=case_id,
                trace_id=trace_id,
                agent=NODE,
                action_name="attachment_extracted",
                status=ToolCallStatus.SUCCESS,
                summary=f"{attachment.filename}: {len(normalize_output.passages)} passage(s), "
                f"quality={quality_output.quality.value}.",
            ),
            ctx,
            tool_events,
        )

    email_passages = [
        SourcePassage(
            passage_id=f"{state['email'].message_id}_body",
            source_document_id=state["email"].message_id,
            source_type="email",
            page_number=None,
            text=state["email"].body,
            ocr_applied=False,
        )
    ] if state.get("email") else []

    requires_review = worst_quality in (OCRQuality.DEGRADED, OCRQuality.UNREADABLE)
    decision = AgentDecision(
        agent=NODE,
        case_id=case_id,
        decision="extracted",
        evidence=[f"{processed} attachment(s) processed", f"ocr_quality={worst_quality.value}"],
        confidence=1.0 if not requires_review else 0.4,
        decision_summary=(
            f"Extracted {len(email_passages) + len(passages)} source passage(s) from "
            f"{processed} attachment(s) plus the email body; "
            f"overall OCR quality={worst_quality.value}."
        ),
        next_action=(
            DecisionOutcome.ESCALATE_TO_HUMAN if requires_review else DecisionOutcome.PROCEED
        ),
        requires_human_review=requires_review,
    )
    latency_ms = (time.monotonic() - started) * 1000
    agent_event = make_agent_event(
        decision,
        trace_id,
        node="pdf_docx_reader",
        model_version=state["model_version"],
        latency_ms=latency_ms,
    )

    return {
        "source_passages": email_passages + passages,
        "attachments": updated_attachments,
        "current_step": "pdf_docx_reader",
        "agent_events": [agent_event],
        "tool_events": tool_events,
        "errors": errors,
    }
