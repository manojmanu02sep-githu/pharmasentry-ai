"""Unit tests for src/tools/document.py."""

from __future__ import annotations

import base64

import pymupdf

from src.models.enums import AgentName, OCRQuality
from src.tools.base import ToolContext
from src.tools.document import (
    AssessOcrQualityInput,
    AssessOcrQualityTool,
    CreatePageCitationsInput,
    CreatePageCitationsTool,
    ExtractPdfTextInput,
    ExtractPdfTextTool,
    NormalizeSourcePassagesInput,
    NormalizeSourcePassagesTool,
    PageForCitation,
    PerformOcrInput,
    PerformOcrTool,
)

CTX = ToolContext(
    case_id="case_001",
    trace_id="trace_001",
    agent=AgentName.ATTACHMENT_READER,
    authorized_tools=frozenset(
        {
            "extract_pdf_text", "perform_ocr", "assess_ocr_quality",
            "create_page_citations", "normalize_source_passages",
        }
    ),
)


def _make_synthetic_pdf(pages_text: list[str]) -> bytes:
    """Build a tiny real PDF in-memory with PyMuPDF, so extract_pdf_text is
    tested against an actual PDF round-trip, not a mock."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_pdf_text_round_trips_a_real_synthetic_pdf() -> None:
    pdf_bytes = _make_synthetic_pdf(["Page one synthetic text.", "Page two synthetic text."])
    content_b64 = base64.b64encode(pdf_bytes).decode()

    output, _ = ExtractPdfTextTool().run(ExtractPdfTextInput(content_base64=content_b64), CTX)

    assert output.page_count == 2
    assert len(output.pages) == 2
    assert "Page one synthetic text" in output.pages[0].text
    assert "Page two synthetic text" in output.pages[1].text
    assert output.truncated is False


def test_extract_pdf_text_respects_max_pages() -> None:
    pdf_bytes = _make_synthetic_pdf(["one", "two", "three"])
    content_b64 = base64.b64encode(pdf_bytes).decode()

    output, _ = ExtractPdfTextTool().run(
        ExtractPdfTextInput(content_base64=content_b64, max_pages=2), CTX
    )
    assert output.page_count == 3
    assert len(output.pages) == 2
    assert output.truncated is True


def test_perform_ocr_reports_unavailable_without_tesseract_binary() -> None:
    """This sandbox has no tesseract binary and no package-manager access
    to install one — perform_ocr must say so honestly, never fabricate
    OCR text."""
    tiny_png = base64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
        b"\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    ).decode()
    output, _ = PerformOcrTool().run(PerformOcrInput(image_base64=tiny_png), CTX)
    assert output.status == "unavailable"
    assert output.text == ""


def test_assess_ocr_quality_good_for_clean_text() -> None:
    output, _ = AssessOcrQualityTool().run(
        AssessOcrQualityInput(text="This is perfectly clean extracted text with no issues."),
        CTX,
    )
    assert output.quality == OCRQuality.GOOD
    assert output.illegible_token_ratio == 0.0


def test_assess_ocr_quality_degraded_and_unreadable_thresholds() -> None:
    words = ["word"] * 20
    words[0] = "[illegible]"  # 1/20 = 5% -> degraded (>=3%, <15%)
    degraded, _ = AssessOcrQualityTool().run(
        AssessOcrQualityInput(text=" ".join(words)), CTX
    )
    assert degraded.quality == OCRQuality.DEGRADED

    words = ["[illegible]"] * 4 + ["word"] * 16  # 4/20 = 20% -> unreadable
    unreadable, _ = AssessOcrQualityTool().run(
        AssessOcrQualityInput(text=" ".join(words)), CTX
    )
    assert unreadable.quality == OCRQuality.UNREADABLE


def test_create_page_citations_finds_matching_pages() -> None:
    pages = [
        PageForCitation(page_number=1, text="Patient developed severe abdominal pain."),
        PageForCitation(page_number=2, text="Discharge diagnosis: acute pancreatitis."),
    ]
    output, _ = CreatePageCitationsTool().run(
        CreatePageCitationsInput(pages=pages, quoted_text="acute pancreatitis"), CTX
    )
    assert len(output.citations) == 1
    assert output.citations[0].page_number == 2


def test_normalize_source_passages_splits_by_paragraph() -> None:
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    output, _ = NormalizeSourcePassagesTool().run(
        NormalizeSourcePassagesInput(
            raw_text=text, source_document_id="doc1", source_type="email"
        ),
        CTX,
    )
    assert len(output.passages) == 3
    assert output.passages[0].passage_id == "doc1_p1"
    assert output.passages[1].text == "Second paragraph."
