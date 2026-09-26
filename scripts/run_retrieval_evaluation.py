#!/usr/bin/env python3
"""Run real (non-baseline) retrieval evaluation: BM25-only, vector-only,
and hybrid (weighted RRF) duplicate retrieval, scored with
precision@5/recall@5/MRR against the golden dataset.

Unlike scripts/run_golden_evaluation.py's duplicate-retrieval section
(a naive case_id-sorted baseline, since there was no retrieval yet), this
script exercises the real BM25/vector/RRF implementation in
src/retrieval — see that package's retrieval_evaluation module docstring
for the full evaluation methodology.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.settings import get_settings  # noqa: E402
from src.retrieval.retrieval_evaluation import run_retrieval_evaluation  # noqa: E402

RESULTS_PATH = REPO_ROOT / "evaluations" / "results" / "phase5_retrieval_metrics.json"


def run() -> dict[str, object]:
    settings = get_settings()
    # bm25_weight/vector_weight mirror config/config.yaml's retrieval policy
    # (HybridRanker.from_config reads the same values for a live case run).
    results = run_retrieval_evaluation(bm25_weight=0.5, vector_weight=0.5)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "note": (
            "Real BM25/vector/hybrid retrieval executed against the full "
            "138-case synthetic golden dataset (queries limited to cases "
            "with a non-empty expected_duplicate_matches) — not a baseline."
        ),
        "embedding_provider": settings.embedding_provider.value,
        "results": {method: r.model_dump() for method, r in results.items()},
    }


def main() -> int:
    try:
        results = run()
    except Exception as exc:  # noqa: BLE001 - top-level script error reporting
        print(f"Retrieval evaluation failed: {exc}", file=sys.stderr)
        return 1

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nWrote results to {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
