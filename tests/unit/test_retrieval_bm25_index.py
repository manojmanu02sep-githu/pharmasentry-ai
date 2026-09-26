"""Unit tests for src/retrieval/bm25_index.py."""

from __future__ import annotations

from pathlib import Path

from src.retrieval.bm25_index import PersistentBm25Index
from src.tools.retrieval import CorpusDocument

CORPUS = [
    CorpusDocument(case_id="case_a", text="severe abdominal pain and vomiting after DemoGlutide"),
    CorpusDocument(case_id="case_b", text="mild headache after DemoBasalin"),
    CorpusDocument(case_id="case_c", text="another severe abdominal pain case with DemoGlutide"),
]


def test_search_ranks_lexically_similar_case_highest() -> None:
    index = PersistentBm25Index(CORPUS)
    results = index.search("severe abdominal pain and vomiting", top_k=3)
    assert results[0].case_id in {"case_a", "case_c"}


def test_search_on_empty_corpus_returns_empty() -> None:
    index = PersistentBm25Index([])
    assert index.search("anything", top_k=5) == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    index = PersistentBm25Index(CORPUS)
    index.save(tmp_path)

    loaded = PersistentBm25Index.load(tmp_path)
    assert [doc.case_id for doc in loaded.corpus] == [doc.case_id for doc in CORPUS]

    original_results = index.search("severe abdominal pain", top_k=3)
    loaded_results = loaded.search("severe abdominal pain", top_k=3)
    assert [r.case_id for r in original_results] == [r.case_id for r in loaded_results]
    assert [r.score for r in original_results] == [r.score for r in loaded_results]
