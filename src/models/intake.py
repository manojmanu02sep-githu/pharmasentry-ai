"""Raw intake models: the untrusted email and attachments a case starts from.

Content on these models is treated as untrusted data, never as instructions,
per the Absolute Boundaries in CLAUDE.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from src.models.enums import FileValidationStatus, OCRQuality


class EmailMessage(BaseModel):
    message_id: str
    sender: str
    subject: str
    body: str
    received_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Attachment(BaseModel):
    attachment_id: str
    filename: str
    sanitized_filename: str | None = None
    media_type: str
    size_bytes: int
    page_count: int | None = None
    validation_status: FileValidationStatus = FileValidationStatus.VALID
    ocr_quality: OCRQuality = OCRQuality.NOT_APPLICABLE
    storage_path: str
