"""Integration test: the baseline evaluation script actually runs end-to-end."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_run_golden_evaluation_script_executes_and_writes_results() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_golden_evaluation.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    results_path = REPO_ROOT / "evaluations" / "results" / "phase2_baseline_metrics.json"
    assert results_path.exists()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert data["dataset"]["total_cases"] == 100
    assert "classification_precision_recall_f1" in data
    assert "extraction_precision_recall_f1" in data
    assert "seriousness_sensitivity_specificity" in data
    assert data["duplicate_queries_evaluated"] > 0
    assert 0.0 <= data["citation_coverage"] <= 1.0
    assert 0.0 <= data["unsupported_claim_rate"] <= 1.0
    assert "NAIVE BASELINE" in data["warning"]
