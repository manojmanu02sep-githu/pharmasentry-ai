"""Builds the retrieval corpus from the synthetic golden dataset only
(CLAUDE.md: "The retrieval corpus must contain synthetic cases only").

`_METADATA_FIELDS` deliberately excludes `reporter.name` — the Duplicate
Agent must never see reporter contact/identifying details (Context
Isolation and Delegation).
"""

from __future__ import annotations

from src.evaluation.golden_loader import load_all_cases
from src.evaluation.schemas import FIELD_KEYS, GoldenCase
from src.tools.retrieval import CorpusDocument

_METADATA_FIELDS: tuple[str, ...] = tuple(k for k in FIELD_KEYS if k != "reporter.name")


def _case_text(case: GoldenCase) -> str:
    return f"{case.email_text}\n\n{case.attachment_text}".strip()


def _case_fields(case: GoldenCase) -> dict[str, str | None]:
    return {k: case.expected_extracted_fields.get(k) for k in _METADATA_FIELDS}


def build_corpus_from_golden_dataset(
    cases: list[GoldenCase] | None = None,
) -> list[CorpusDocument]:
    resolved = cases if cases is not None else load_all_cases()
    return [
        CorpusDocument(case_id=c.case_id, text=_case_text(c), fields=_case_fields(c))
        for c in resolved
    ]
