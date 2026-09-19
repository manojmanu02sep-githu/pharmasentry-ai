"""Unit tests for src/tools/intake.py."""

from __future__ import annotations

import base64

from src.models.enums import AgentName, FileValidationStatus
from src.tools.base import ToolContext
from src.tools.intake import (
    EnforceFileSizeInput,
    EnforceFileSizeTool,
    EnforcePageLimitInput,
    EnforcePageLimitTool,
    ExtractAttachmentsInput,
    ExtractAttachmentsTool,
    ParseEmailInput,
    ParseEmailTool,
    SanitizeFilenameInput,
    SanitizeFilenameTool,
    ValidateFileInput,
    ValidateFileSignatureInput,
    ValidateFileSignatureTool,
    ValidateFileTool,
)

CTX = ToolContext(
    case_id="case_001",
    trace_id="trace_001",
    agent=AgentName.SUBJECT_READER,
    authorized_tools=frozenset(
        {
            "parse_email", "extract_attachments", "validate_file",
            "validate_file_signature", "sanitize_filename", "enforce_file_size",
            "enforce_page_limit",
        }
    ),
)

DEMO_EMAIL = """From: r.tanaka@fictionalclinic-demo.example
To: safety-intake@pharmasentry-demo.example
Subject: Possible adverse event report
Message-ID: <demo-case-001@fictionalclinic-demo.example>

Hello Safety Team,

This is a synthetic report body.

Regards,
R. Tanaka
"""

MULTIPART_EMAIL = (
    "From: r@example.com\n"
    "To: safety@example.com\n"
    "Subject: With attachment\n"
    "Message-ID: <multi-001@example.com>\n"
    'Content-Type: multipart/mixed; boundary="BOUNDARY"\n\n'
    "--BOUNDARY\n"
    "Content-Type: text/plain\n\n"
    "Body text here.\n"
    "--BOUNDARY\n"
    'Content-Type: text/plain; name="notes.txt"\n'
    'Content-Disposition: attachment; filename="notes.txt"\n\n'
    "Attachment content.\n"
    "--BOUNDARY--\n"
)


def test_parse_email_extracts_headers_and_body() -> None:
    output, _ = ParseEmailTool().run(ParseEmailInput(raw_email_text=DEMO_EMAIL), CTX)
    assert output.message_id == "demo-case-001@fictionalclinic-demo.example"
    assert output.sender == "r.tanaka@fictionalclinic-demo.example"
    assert "Possible adverse event report" in output.subject
    assert "synthetic report body" in output.body
    assert output.attachment_filenames == []


def test_parse_email_finds_attachment_filenames_in_multipart() -> None:
    output, _ = ParseEmailTool().run(ParseEmailInput(raw_email_text=MULTIPART_EMAIL), CTX)
    assert "notes.txt" in output.attachment_filenames


def test_extract_attachments_returns_text_content() -> None:
    output, _ = ExtractAttachmentsTool().run(
        ExtractAttachmentsInput(raw_email_text=MULTIPART_EMAIL), CTX
    )
    assert len(output.attachments) == 1
    assert output.attachments[0].filename == "notes.txt"
    assert "Attachment content" in output.attachments[0].text_content


def test_validate_file_accepts_known_type_within_size() -> None:
    output, _ = ValidateFileTool().run(
        ValidateFileInput(filename="report.pdf", media_type="application/pdf", size_bytes=1000),
        CTX,
    )
    assert output.status == FileValidationStatus.VALID


def test_validate_file_rejects_unsupported_type() -> None:
    output, _ = ValidateFileTool().run(
        ValidateFileInput(
            filename="script.exe", media_type="application/x-msdownload", size_bytes=100
        ),
        CTX,
    )
    assert output.status == FileValidationStatus.UNSUPPORTED_TYPE


def test_validate_file_rejects_oversized_file() -> None:
    output, _ = ValidateFileTool().run(
        ValidateFileInput(
            filename="big.pdf", media_type="application/pdf", size_bytes=999,
            max_size_bytes=500,
        ),
        CTX,
    )
    assert output.status == FileValidationStatus.SIZE_EXCEEDED


def test_validate_file_signature_matches_real_pdf_bytes() -> None:
    content = base64.b64encode(b"%PDF-1.4 fake but correctly-signed content").decode()
    output, _ = ValidateFileSignatureTool().run(
        ValidateFileSignatureInput(
            filename="report.pdf", content_base64=content, declared_media_type="application/pdf"
        ),
        CTX,
    )
    assert output.matches_declared is True
    assert output.status == FileValidationStatus.VALID


def test_validate_file_signature_catches_misleading_extension() -> None:
    """A .pdf filename whose actual bytes are a PNG — the classic
    misleading-file-extension attack."""
    png_bytes = b"\x89PNG\r\n\x1a\nrest-of-file"
    content = base64.b64encode(png_bytes).decode()
    output, _ = ValidateFileSignatureTool().run(
        ValidateFileSignatureInput(
            filename="report.pdf", content_base64=content, declared_media_type="application/pdf"
        ),
        CTX,
    )
    assert output.matches_declared is False
    assert output.detected_media_type == "image/png"
    assert output.status == FileValidationStatus.SIGNATURE_MISMATCH


def test_sanitize_filename_strips_path_traversal() -> None:
    output, _ = SanitizeFilenameTool().run(
        SanitizeFilenameInput(filename="../../etc/passwd"), CTX
    )
    assert ".." not in output.sanitized_filename
    assert "/" not in output.sanitized_filename
    assert output.was_modified is True


def test_sanitize_filename_leaves_safe_names_untouched() -> None:
    output, _ = SanitizeFilenameTool().run(
        SanitizeFilenameInput(filename="discharge_summary.pdf"), CTX
    )
    assert output.sanitized_filename == "discharge_summary.pdf"
    assert output.was_modified is False


def test_sanitize_filename_strips_null_and_control_characters() -> None:
    output, _ = SanitizeFilenameTool().run(
        SanitizeFilenameInput(filename="evil\x00name\n.pdf"), CTX
    )
    assert "\x00" not in output.sanitized_filename
    assert "\n" not in output.sanitized_filename


def test_enforce_file_size_within_and_over_limit() -> None:
    ok, _ = EnforceFileSizeTool().run(EnforceFileSizeInput(size_bytes=10, max_size_bytes=100), CTX)
    assert ok.within_limit is True

    over, _ = EnforceFileSizeTool().run(
        EnforceFileSizeInput(size_bytes=200, max_size_bytes=100), CTX
    )
    assert over.within_limit is False


def test_enforce_page_limit_within_and_over_limit() -> None:
    ok, _ = EnforcePageLimitTool().run(EnforcePageLimitInput(page_count=5, max_pages=10), CTX)
    assert ok.within_limit is True

    over, _ = EnforcePageLimitTool().run(EnforcePageLimitInput(page_count=150, max_pages=100), CTX)
    assert over.within_limit is False
