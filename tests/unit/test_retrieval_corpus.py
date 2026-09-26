"""Unit tests for src/retrieval/corpus.py."""

from __future__ import annotations

from src.evaluation.golden_loader import load_all_cases
from src.retrieval.corpus import build_corpus_from_golden_dataset


def test_corpus_covers_every_golden_case() -> None:
    cases = load_all_cases()
    corpus = build_corpus_from_golden_dataset(list(cases))
    assert len(corpus) == len(cases)
    assert {doc.case_id for doc in corpus} == {c.case_id for c in cases}


def test_corpus_never_exposes_reporter_name() -> None:
    corpus = build_corpus_from_golden_dataset()
    for doc in corpus:
        assert "reporter.name" not in doc.fields


def test_corpus_text_combines_email_and_attachment() -> None:
    cases = load_all_cases()
    case = next(c for c in cases if c.attachment_text)
    corpus = build_corpus_from_golden_dataset([case])
    assert case.email_body in corpus[0].text
    assert case.attachment_text.strip() in corpus[0].text
