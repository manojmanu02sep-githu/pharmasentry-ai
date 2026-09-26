"""Builds the initial CaseState for CLAUDE.md's synthetic demo case
(fictional product DemoGluca; see data/synthetic_emails/demo_case_001.eml
and data/synthetic_attachments/demo_case_001_discharge_summary.txt).

Parsing the raw .eml into an `EmailMessage` happens here, before any
agent node runs, using the `parse_email` tool directly -- this is
pre-graph ingestion, not one of the 12 bounded agent steps, so it is
scoped to exactly the one tool it needs rather than borrowing an agent's
allowlist.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from src.graph.state import CaseState, create_initial_state
from src.models.enums import AgentName, FileValidationStatus, OCRQuality
from src.models.intake import Attachment, EmailMessage
from src.tools.base import ToolContext
from src.tools.intake import ParseEmailInput
from src.tools.registry import ALL_TOOLS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEMO_EMAIL_PATH = REPO_ROOT / "data" / "synthetic_emails" / "demo_case_001.eml"
DEMO_ATTACHMENT_PATH = (
    REPO_ROOT / "data" / "synthetic_attachments" / "demo_case_001_discharge_summary.txt"
)


def _parse_demo_email() -> EmailMessage:
    raw_text = DEMO_EMAIL_PATH.read_text(encoding="utf-8")
    ctx = ToolContext(
        case_id="pre-graph-intake",
        trace_id="pre-graph-intake",
        agent=AgentName.SUBJECT_READER,
        authorized_tools=frozenset({"parse_email"}),
    )
    output, _event = ALL_TOOLS["parse_email"].run(ParseEmailInput(raw_email_text=raw_text), ctx)
    return EmailMessage(
        message_id=output.message_id,
        sender=output.sender,
        subject=output.subject,
        body=output.body,
    )


def _demo_attachment() -> Attachment:
    size_bytes = DEMO_ATTACHMENT_PATH.stat().st_size
    return Attachment(
        attachment_id=f"att_{uuid4().hex[:12]}",
        filename=DEMO_ATTACHMENT_PATH.name,
        media_type="text/plain",
        size_bytes=size_bytes,
        page_count=1,
        validation_status=FileValidationStatus.VALID,
        ocr_quality=OCRQuality.NOT_APPLICABLE,
        storage_path=str(DEMO_ATTACHMENT_PATH),
    )


def build_demo_case_initial_state(case_id: str | None = None) -> CaseState:
    email = _parse_demo_email()
    attachment = _demo_attachment()
    return create_initial_state(email=email, attachments=[attachment], case_id=case_id)
