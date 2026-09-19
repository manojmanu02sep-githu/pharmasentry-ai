"""Integration test: the generator is deterministic and produces valid output.

Runs the real script via subprocess against the default seed (42), which is
the same seed used to produce the committed dataset — re-running is
idempotent and must reproduce byte-identical case files.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_CASE = REPO_ROOT / "evaluations" / "golden_dataset" / "cases" / "complete_001.json"


def _run_generator() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_golden_dataset.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_generator_runs_successfully() -> None:
    result = _run_generator()
    assert result.returncode == 0, result.stderr
    assert "Wrote 100 golden cases" in result.stdout


def test_generator_is_deterministic_for_the_default_seed() -> None:
    first = hashlib.sha256(SAMPLE_CASE.read_bytes()).hexdigest()

    result = _run_generator()
    assert result.returncode == 0, result.stderr

    second = hashlib.sha256(SAMPLE_CASE.read_bytes()).hexdigest()
    assert first == second
