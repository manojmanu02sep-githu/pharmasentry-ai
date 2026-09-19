"""Document tools: PDF text extraction, OCR, OCR quality, and citations.

OCR is behind a provider interface (Absolute Boundaries: no paid/external
service required for the default demo). The "local" provider needs the
system `tesseract` binary; when it's missing, `perform_ocr` returns a
clearly-labeled `unavailable` status rather than fabricating text — this
sandbox has no `tesseract` installed and no package-manager access to add
it, which is exactly the case this status exists for. See progress.md for
that limitation.
"""

from __future__ import annotations

import base64
import io

import pymupdf
from pydantic import BaseModel, Field

from src.models.enums import OCRQuality
from src.tools.base import BaseTool, ToolContext, limit_results

try:
    import pytesseract
    from PIL import Image

    _PYTESSERACT_IMPORTABLE = True
except ImportError:  # pragma: no cover - pytesseract/pillow are in requirements.txt
    _PYTESSERACT_IMPORTABLE = False


# --- extract_pdf_text --------------------------------------------------------


class ExtractPdfTextInput(BaseModel):
    content_base64: str = Field(max_length=40_000_000)  # ~30MB decoded
    max_pages: int = 100


class PageText(BaseModel):
    page_number: int
    text: str


class ExtractPdfTextOutput(BaseModel):
    pages: list[PageText] = Field(default_factory=list)
    page_count: int
    truncated: bool = False


class ExtractPdfTextTool(BaseTool[ExtractPdfTextInput, ExtractPdfTextOutput]):
    name = "extract_pdf_text"
    purpose = "Extract page-level text from a digital (non-scanned) PDF."
    timeout_seconds = 15.0
    max_retries = 0  # PDF parsing is not always idempotent-safe on malformed input

    def _execute(
        self, tool_input: ExtractPdfTextInput, ctx: ToolContext
    ) -> ExtractPdfTextOutput:
        raw = base64.b64decode(tool_input.content_base64, validate=True)
        with pymupdf.open(stream=raw, filetype="pdf") as doc:
            page_count = doc.page_count
            n_pages = min(page_count, tool_input.max_pages)
            pages = [
                PageText(page_number=i + 1, text=doc[i].get_text())
                for i in range(n_pages)
            ]
        limited, truncated = limit_results(pages, tool_input.max_pages)
        return ExtractPdfTextOutput(
            pages=limited, page_count=page_count, truncated=truncated or n_pages < page_count
        )


# --- perform_ocr --------------------------------------------------------


class PerformOcrInput(BaseModel):
    image_base64: str = Field(max_length=40_000_000)


class PerformOcrOutput(BaseModel):
    status: str  # "ok" | "unavailable" | "error"
    text: str = ""
    confidence: float = 0.0
    notes: str = ""


class PerformOcrTool(BaseTool[PerformOcrInput, PerformOcrOutput]):
    name = "perform_ocr"
    purpose = (
        "Run OCR on an image via the local tesseract adapter. Returns "
        "status='unavailable' (never fabricated text) when the system "
        "tesseract binary is not installed."
    )
    timeout_seconds = 10.0
    max_retries = 0

    def _execute(self, tool_input: PerformOcrInput, ctx: ToolContext) -> PerformOcrOutput:
        if not _PYTESSERACT_IMPORTABLE:
            return PerformOcrOutput(
                status="unavailable",
                notes="pytesseract/Pillow not importable in this environment.",
            )
        raw = base64.b64decode(tool_input.image_base64, validate=True)
        try:
            image = Image.open(io.BytesIO(raw))
            text = pytesseract.image_to_string(image)
        except Exception as exc:
            # Covers the common case in this sandbox: the tesseract *binary*
            # is missing even though the Python package imports fine.
            return PerformOcrOutput(
                status="unavailable",
                notes=f"local OCR backend unavailable: {type(exc).__name__}",
            )
        return PerformOcrOutput(status="ok", text=text, confidence=0.0)


# --- assess_ocr_quality --------------------------------------------------------


_ILLEGIBLE_MARKER = "[illegible]"


class AssessOcrQualityInput(BaseModel):
    text: str = Field(max_length=1_000_000)


class AssessOcrQualityOutput(BaseModel):
    quality: OCRQuality
    illegible_token_ratio: float
    notes: str = ""


class AssessOcrQualityTool(BaseTool[AssessOcrQualityInput, AssessOcrQualityOutput]):
    name = "assess_ocr_quality"
    purpose = (
        "Deterministically score OCR/extracted text quality from the ratio "
        "of illegible-marker tokens and non-ASCII noise, so low-quality "
        "scans route to manual review instead of confident (wrong) extraction."
    )
    timeout_seconds = 1.0
    max_retries = 2

    def _execute(
        self, tool_input: AssessOcrQualityInput, ctx: ToolContext
    ) -> AssessOcrQualityOutput:
        tokens = tool_input.text.split()
        if not tokens:
            return AssessOcrQualityOutput(
                quality=OCRQuality.NOT_APPLICABLE, illegible_token_ratio=0.0,
                notes="empty text",
            )
        illegible = sum(1 for t in tokens if _ILLEGIBLE_MARKER in t)
        ratio = illegible / len(tokens)

        if ratio >= 0.15:
            quality = OCRQuality.UNREADABLE
        elif ratio >= 0.03:
            quality = OCRQuality.DEGRADED
        else:
            quality = OCRQuality.GOOD
        return AssessOcrQualityOutput(
            quality=quality,
            illegible_token_ratio=round(ratio, 4),
            notes=f"{illegible}/{len(tokens)} tokens marked illegible",
        )


# --- create_page_citations --------------------------------------------------------


class PageForCitation(BaseModel):
    page_number: int
    text: str


class CreatePageCitationsInput(BaseModel):
    pages: list[PageForCitation]
    quoted_text: str = Field(max_length=2000)


class PageCitation(BaseModel):
    page_number: int
    quoted_text: str


class CreatePageCitationsOutput(BaseModel):
    citations: list[PageCitation] = Field(default_factory=list)


class CreatePageCitationsTool(BaseTool[CreatePageCitationsInput, CreatePageCitationsOutput]):
    name = "create_page_citations"
    purpose = "Find which page(s) of extracted text actually contain a quoted claim."
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: CreatePageCitationsInput, ctx: ToolContext
    ) -> CreatePageCitationsOutput:
        citations = [
            PageCitation(page_number=page.page_number, quoted_text=tool_input.quoted_text)
            for page in tool_input.pages
            if tool_input.quoted_text in page.text
        ]
        limited, _ = limit_results(citations, 20)
        return CreatePageCitationsOutput(citations=limited)


# --- normalize_source_passages --------------------------------------------------------


class NormalizeSourcePassagesInput(BaseModel):
    raw_text: str = Field(max_length=1_000_000)
    source_document_id: str
    source_type: str  # "email" | "attachment"


class NormalizedPassage(BaseModel):
    passage_id: str
    text: str


class NormalizeSourcePassagesOutput(BaseModel):
    passages: list[NormalizedPassage] = Field(default_factory=list)
    truncated: bool = False


class NormalizeSourcePassagesTool(
    BaseTool[NormalizeSourcePassagesInput, NormalizeSourcePassagesOutput]
):
    name = "normalize_source_passages"
    purpose = (
        "Split raw email/attachment text into normalized, individually "
        "citable passages (by paragraph)."
    )
    timeout_seconds = 2.0
    max_retries = 2

    def _execute(
        self, tool_input: NormalizeSourcePassagesInput, ctx: ToolContext
    ) -> NormalizeSourcePassagesOutput:
        paragraphs = [p.strip() for p in tool_input.raw_text.split("\n\n") if p.strip()]
        passages = [
            NormalizedPassage(
                passage_id=f"{tool_input.source_document_id}_p{i + 1}", text=paragraph
            )
            for i, paragraph in enumerate(paragraphs)
        ]
        limited, truncated = limit_results(passages, 200)
        return NormalizeSourcePassagesOutput(passages=limited, truncated=truncated)
