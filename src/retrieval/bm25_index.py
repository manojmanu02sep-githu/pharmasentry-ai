"""Persistent BM25 index.

`rank_bm25.BM25Okapi` has no native serialization, and unpickling an index
loaded from disk would be an unnecessary risk for no real benefit, so the
source corpus is persisted as plain JSON and `BM25Okapi` is rebuilt in
memory at load time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.tools.retrieval import CorpusDocument, ScoredCandidate, tokenize

logger = logging.getLogger(__name__)

_CORPUS_FILENAME = "corpus.json"


class PersistentBm25Index:
    def __init__(self, corpus: list[CorpusDocument]) -> None:
        self.corpus = corpus
        tokenized = [tokenize(doc.text) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query_text: str, top_k: int = 5) -> list[ScoredCandidate]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query_text))
        ranked = sorted(zip(self.corpus, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            ScoredCandidate(case_id=doc.case_id, score=round(float(score), 6))
            for doc, score in ranked[:top_k]
        ]

    def save(self, index_dir: str | Path) -> None:
        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        data = [doc.model_dump() for doc in self.corpus]
        (path / _CORPUS_FILENAME).write_text(json.dumps(data), encoding="utf-8")
        logger.info("bm25_index_saved path=%s documents=%d", path, len(self.corpus))

    @classmethod
    def load(cls, index_dir: str | Path) -> PersistentBm25Index:
        path = Path(index_dir)
        data = json.loads((path / _CORPUS_FILENAME).read_text(encoding="utf-8"))
        index = cls([CorpusDocument(**item) for item in data])
        logger.info("bm25_index_loaded path=%s documents=%d", path, len(index.corpus))
        return index
