#!/usr/bin/env python3
"""Run the baseline metrics framework against the golden dataset test split.

IMPORTANT: there are no agents yet (see progress.md — agents are Phase 6).
Every number this script prints is computed for real against a trivial,
clearly-labeled NAIVE baseline predictor, not an LLM or agent:
  - classification: always predicts "safety report" (majority class)
  - extraction: always predicts every field as unknown (empty)
  - seriousness: always predicts "not serious"
  - duplicate retrieval: returns other test-set case_ids in case_id sort
    order (no real search — BM25/vector retrieval is Phase 5)
  - citation coverage / unsupported claim rate: computed on the golden
    dataset's OWN expected_narrative_facts (which are grounded by
    construction), not on any generated narrative — this exercises the
    metric functions honestly, it is not a claim about narrative quality

This exists so `make eval` produces real, executed numbers today instead of
a placeholder, and so the metrics framework itself is proven to run
end-to-end before any agent depends on it. Re-run after Phase 6 lands real
agents to get meaningful (non-baseline) scores.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (  # noqa: E402
    citation_coverage,
    classification_prf1,
    duplicate_precision_at_k,
    duplicate_recall_at_k,
    extraction_prf1,
    load_manifest,
    load_test_cases,
    seriousness_sensitivity_specificity,
    unsupported_claim_rate,
)

RESULTS_PATH = REPO_ROOT / "evaluations" / "results" / "phase2_baseline_metrics.json"


def run() -> dict[str, object]:
    manifest = load_manifest()
    test_cases = load_test_cases()

    # --- Classification baseline: always predict "safety report" ---
    y_true_safety = [c.is_safety_report for c in test_cases]
    y_pred_safety = [True] * len(test_cases)
    classification = classification_prf1(y_true_safety, y_pred_safety)

    # --- Extraction baseline: always predict every field unknown ---
    extraction_pairs: list[tuple[dict[str, str | None], dict[str, str | None]]] = [
        (c.expected_extracted_fields, {}) for c in test_cases
    ]
    extraction = extraction_prf1(extraction_pairs)

    # --- Seriousness baseline: always predict "not serious" ---
    y_true_serious = [bool(c.expected_seriousness_indicators) for c in test_cases]
    y_pred_serious = [False] * len(test_cases)
    seriousness = seriousness_sensitivity_specificity(y_true_serious, y_pred_serious)

    # --- Duplicate retrieval baseline: naive fixed-order candidate list ---
    # Only meaningful for test cases that actually have a duplicate family
    # member in the corpus; other test cases have no relevant candidate.
    all_test_ids = sorted(c.case_id for c in test_cases)
    families: dict[str, list[str]] = {}
    for c in test_cases:
        families.setdefault(c.expected_duplicate_family, []).append(c.case_id)

    duplicate_queries = []
    for c in test_cases:
        relevant = {cid for cid in families[c.expected_duplicate_family] if cid != c.case_id}
        if not relevant:
            continue
        naive_candidates = [cid for cid in all_test_ids if cid != c.case_id]
        duplicate_queries.append((c.case_id, naive_candidates, relevant))

    dup_precisions = [
        duplicate_precision_at_k(cands, rel, k=5) for _, cands, rel in duplicate_queries
    ]
    dup_recalls = [
        duplicate_recall_at_k(cands, rel, k=5) for _, cands, rel in duplicate_queries
    ]
    duplicate_precision_at_5 = (
        sum(dup_precisions) / len(dup_precisions) if dup_precisions else None
    )
    duplicate_recall_at_5 = sum(dup_recalls) / len(dup_recalls) if dup_recalls else None

    # --- Citation coverage / unsupported claim rate on gold narrative facts ---
    safety_test_cases = [c for c in test_cases if c.is_safety_report]
    total_claims = sum(len(c.expected_narrative_facts) for c in safety_test_cases)
    cited_claims = total_claims  # gold facts are grounded by construction
    unsupported_claims = 0
    coverage = citation_coverage(total_claims, cited_claims)
    unsupported_rate = unsupported_claim_rate(total_claims, unsupported_claims)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "seed": manifest.seed,
            "total_cases": manifest.total_cases,
            "test_case_count": len(test_cases),
            "category_counts": manifest.category_counts,
        },
        "warning": (
            "These are NAIVE BASELINE metrics (no agent/LLM involved) — see "
            "the module docstring in scripts/run_golden_evaluation.py. They "
            "establish that the metrics framework executes correctly; they "
            "are not a measure of PharmaSentry AI's actual performance."
        ),
        "classification_precision_recall_f1": classification.model_dump(),
        "extraction_precision_recall_f1": extraction.model_dump(),
        "seriousness_sensitivity_specificity": seriousness.model_dump(),
        "duplicate_precision_at_5": duplicate_precision_at_5,
        "duplicate_recall_at_5": duplicate_recall_at_5,
        "duplicate_queries_evaluated": len(duplicate_queries),
        "citation_coverage": coverage,
        "unsupported_claim_rate": unsupported_rate,
        "citation_metrics_note": (
            "Computed on the golden dataset's own expected_narrative_facts "
            "(grounded by construction), not on a generated narrative — "
            "there is no Narrative Agent yet (Phase 6)."
        ),
    }


def main() -> int:
    try:
        results = run()
    except Exception as exc:  # noqa: BLE001 - top-level script error reporting
        print(f"Golden evaluation failed: {exc}", file=sys.stderr)
        return 1

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"\nWrote results to {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
