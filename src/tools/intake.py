"""Intake tools: parsing untrusted email/attachments and validating files.

Every tool here treats its input text as untrusted DATA — nothing in this
module ever executes, evaluates, or follows instructions found inside an
email or attachment body.
"""

from __future__ import annotations

import base64
import re
from email import message_from_string
from email.message import Message
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from src.models.enums import FileValidationStatus
from src.tools.base import BaseTool, ToolContext, limit_results

DEFAULT_ALLOWED_MEDIA_TYPES = (
    "text/plain",
    "application/pdf",
    "image/png",
    "image/jpeg",
)
DEFAULT_MAX_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
DEFAULT_MAX_PAGES = 100

# Magic-byte signatures for the file types this prototype accepts.
_SIGNATURES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
    "image/png": b"\x89PNG\r\n\x1a\n",
    "image/jpeg": b"\xff\xd8\xff",
}

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


# --- parse_email ------------------------------------------------------------


class ParseEmailInput(BaseModel):
    raw_email_text: str = Field(max_length=2_000_000)


class ParseEmailOutput(BaseModel):
    message_id: str
    sender: str
    subject: str
    body: str
    attachment_filenames: list[str] = Field(default_factory=list)
    truncated: bool = False


class ParseEmailTool(BaseTool[ParseEmailInput, ParseEmailOutput]):
    name = "parse_email"
    purpose = "Parse a raw email (headers + body, optionally multipart) into structured fields."
    timeout_seconds = 3.0
    max_retries = 1  # pure parsing, safe to retry

    def _execute(self, tool_input: ParseEmailInput, ctx: ToolContext) -> ParseEmailOutput:
        msg = message_from_string(tool_input.raw_email_text)
        body, attachment_filenames = _walk_message(msg)
        output = ParseEmailOutput(
            message_id=msg.get("Message-ID", "").strip("<>") or "unknown",
            sender=msg.get("From", "unknown"),
            subject=msg.get("Subject", ""),
            body=body,
        )
        limited, truncated = limit_results(attachment_filenames, 50)
        output.attachment_filenames = limited
        output.truncated = truncated
        return output


def _walk_message(msg: Message) -> tuple[str, list[str]]:
    if not msg.is_multipart():
        payload = msg.get_payload(decode=False)
        return (payload if isinstance(payload, str) else str(payload)), []

    body_parts: list[str] = []
    attachment_filenames: list[str] = []
    for part in msg.walk():
        if part.is_multipart():
            continue
        filename = part.get_filename()
        if filename:
            attachment_filenames.append(filename)
        elif part.get_content_type() == "text/plain":
            payload = part.get_payload(decode=False)
            body_parts.append(payload if isinstance(payload, str) else str(payload))
    return "\n".join(body_parts), attachment_filenames


# --- extract_attachments ----------------------------------------------------


class ExtractAttachmentsInput(BaseModel):
    raw_email_text: str = Field(max_length=2_000_000)


class AttachmentStub(BaseModel):
    filename: str
    media_type: str
    size_bytes: int
    text_content: str = ""  # only populated for text-like parts


class ExtractAttachmentsOutput(BaseModel):
    attachments: list[AttachmentStub] = Field(default_factory=list)
    truncated: bool = False


class ExtractAttachmentsTool(BaseTool[ExtractAttachmentsInput, ExtractAttachmentsOutput]):
    name = "extract_attachments"
    purpose = "Extract attachment metadata (and text content where plain-text) from a raw email."
    timeout_seconds = 5.0
    max_retries = 1

    def _execute(
        self, tool_input: ExtractAttachmentsInput, ctx: ToolContext
    ) -> ExtractAttachmentsOutput:
        msg = message_from_string(tool_input.raw_email_text)
        stubs: list[AttachmentStub] = []
        if msg.is_multipart():
            for part in msg.walk():
                filename = part.get_filename()
                if not filename:
                    continue
                raw_payload = part.get_payload(decode=True)
                payload = raw_payload if isinstance(raw_payload, bytes) else b""
                media_type = part.get_content_type()
                text_content = ""
                if media_type.startswith("text/"):
                    text_content = payload.decode("utf-8", errors="replace")
                stubs.append(
                    AttachmentStub(
                        filename=filename,
                        media_type=media_type,
                        size_bytes=len(payload),
                        text_content=text_content,
                    )
                )
        limited, truncated = limit_results(stubs, 20)
        return ExtractAttachmentsOutput(attachments=limited, truncated=truncated)


# --- validate_file -----------------------------------------------------------


class ValidateFileInput(BaseModel):
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)
    allowed_media_types: tuple[str, ...] = DEFAULT_ALLOWED_MEDIA_TYPES
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES


class ValidateFileOutput(BaseModel):
    status: FileValidationStatus
    reasons: list[str] = Field(default_factory=list)


class ValidateFileTool(BaseTool[ValidateFileInput, ValidateFileOutput]):
    name = "validate_file"
    purpose = "Check a file's declared type and size against configured policy."
    timeout_seconds = 2.0
    max_retries = 1

    def _execute(self, tool_input: ValidateFileInput, ctx: ToolContext) -> ValidateFileOutput:
        reasons: list[str] = []
        if tool_input.media_type not in tool_input.allowed_media_types:
            return ValidateFileOutput(
                status=FileValidationStatus.UNSUPPORTED_TYPE,
                reasons=[f"media_type {tool_input.media_type!r} is not in the allowlist"],
            )
        if tool_input.size_bytes > tool_input.max_size_bytes:
            return ValidateFileOutput(
                status=FileValidationStatus.SIZE_EXCEEDED,
                reasons=[
                    f"size_bytes={tool_input.size_bytes} exceeds "
                    f"max_size_bytes={tool_input.max_size_bytes}"
                ],
            )
        return ValidateFileOutput(status=FileValidationStatus.VALID, reasons=reasons)


# --- validate_file_signature -------------------------------------------------


class ValidateFileSignatureInput(BaseModel):
    filename: str
    # ~3MB decoded; the signature check only ever needs a small prefix.
    content_base64: str = Field(max_length=4_000_000)
    declared_media_type: str


class ValidateFileSignatureOutput(BaseModel):
    status: FileValidationStatus
    detected_media_type: str | None
    matches_declared: bool


class ValidateFileSignatureTool(
    BaseTool[ValidateFileSignatureInput, ValidateFileSignatureOutput]
):
    name = "validate_file_signature"
    purpose = (
        "Sniff a file's magic bytes and confirm they match its declared media "
        "type, catching a misleading file extension/declared type."
    )
    timeout_seconds = 2.0
    max_retries = 1

    def _execute(
        self, tool_input: ValidateFileSignatureInput, ctx: ToolContext
    ) -> ValidateFileSignatureOutput:
        try:
            content = base64.b64decode(tool_input.content_base64, validate=True)
        except Exception:
            return ValidateFileSignatureOutput(
                status=FileValidationStatus.REJECTED,
                detected_media_type=None,
                matches_declared=False,
            )

        detected = None
        for media_type, signature in _SIGNATURES.items():
            if content.startswith(signature):
                detected = media_type
                break
        if detected is None and tool_input.declared_media_type == "text/plain":
            # Plain text has no magic bytes; accept if it decodes as UTF-8/ASCII.
            try:
                content.decode("utf-8")
                detected = "text/plain"
            except UnicodeDecodeError:
                detected = None

        matches = detected == tool_input.declared_media_type
        status = FileValidationStatus.VALID if matches else FileValidationStatus.SIGNATURE_MISMATCH
        return ValidateFileSignatureOutput(
            status=status, detected_media_type=detected, matches_declared=matches
        )


# --- sanitize_filename -------------------------------------------------------


class SanitizeFilenameInput(BaseModel):
    filename: str = Field(max_length=1024)


class SanitizeFilenameOutput(BaseModel):
    sanitized_filename: str
    was_modified: bool


class SanitizeFilenameTool(BaseTool[SanitizeFilenameInput, SanitizeFilenameOutput]):
    name = "sanitize_filename"
    purpose = (
        "Strip path components, null/control characters, and unsafe characters "
        "from a filename before it ever touches the filesystem."
    )
    timeout_seconds = 1.0
    max_retries = 1

    def _execute(
        self, tool_input: SanitizeFilenameInput, ctx: ToolContext
    ) -> SanitizeFilenameOutput:
        original = tool_input.filename
        # Take only the final path component (defeats path traversal such as
        # "../../etc/passwd" or absolute paths).
        base = PurePosixPath(original.replace("\\", "/")).name
        base = base.lstrip(".")  # no leading dots (hidden files / ".." remnants)
        cleaned = _SAFE_FILENAME_RE.sub("_", base)
        cleaned = cleaned[:255] or "unnamed_file"
        return SanitizeFilenameOutput(
            sanitized_filename=cleaned, was_modified=cleaned != original
        )


# --- enforce_file_size -------------------------------------------------------


class EnforceFileSizeInput(BaseModel):
    size_bytes: int = Field(ge=0)
    max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES


class EnforceFileSizeOutput(BaseModel):
    within_limit: bool
    size_bytes: int
    max_size_bytes: int


class EnforceFileSizeTool(BaseTool[EnforceFileSizeInput, EnforceFileSizeOutput]):
    name = "enforce_file_size"
    purpose = "Deterministic size-limit check."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: EnforceFileSizeInput, ctx: ToolContext
    ) -> EnforceFileSizeOutput:
        return EnforceFileSizeOutput(
            within_limit=tool_input.size_bytes <= tool_input.max_size_bytes,
            size_bytes=tool_input.size_bytes,
            max_size_bytes=tool_input.max_size_bytes,
        )


# --- enforce_page_limit ------------------------------------------------------


class EnforcePageLimitInput(BaseModel):
    page_count: int = Field(ge=0)
    max_pages: int = DEFAULT_MAX_PAGES


class EnforcePageLimitOutput(BaseModel):
    within_limit: bool
    page_count: int
    max_pages: int


class EnforcePageLimitTool(BaseTool[EnforcePageLimitInput, EnforcePageLimitOutput]):
    name = "enforce_page_limit"
    purpose = "Deterministic page-count-limit check."
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: EnforcePageLimitInput, ctx: ToolContext
    ) -> EnforcePageLimitOutput:
        return EnforcePageLimitOutput(
            within_limit=tool_input.page_count <= tool_input.max_pages,
            page_count=tool_input.page_count,
            max_pages=tool_input.max_pages,
        )
