"""Loaders for the golden evaluation dataset under evaluations/golden_dataset/.

Read-only: nothing here generates or mutates the dataset (see
scripts/generate_golden_dataset.py for that). Kept separate so agents and
evaluation code (Phase 5+) can depend on a small, stable loading surface.
"""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from src.evaluation.schemas import DatasetManifest, DatasetSplit, GoldenCase

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DATASET_DIR = REPO_ROOT / "evaluations" / "golden_dataset"
CASES_DIR = GOLDEN_DATASET_DIR / "cases"
MANIFEST_PATH = GOLDEN_DATASET_DIR / "manifest.json"
SPLIT_PATH = GOLDEN_DATASET_DIR / "split.json"


class GoldenDatasetError(RuntimeError):
    """Raised when the golden dataset is missing or internally inconsistent."""


def load_manifest() -> DatasetManifest:
    if not MANIFEST_PATH.exists():
        raise GoldenDatasetError(
            f"No manifest at {MANIFEST_PATH}. Run "
            "`python scripts/generate_golden_dataset.py` first."
        )
    return DatasetManifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))


def load_split() -> DatasetSplit:
    if not SPLIT_PATH.exists():
        raise GoldenDatasetError(
            f"No split file at {SPLIT_PATH}. Run "
            "`python scripts/generate_golden_dataset.py` first."
        )
    return DatasetSplit.model_validate_json(SPLIT_PATH.read_text(encoding="utf-8"))


def _case_path(case_id: str) -> Path:
    return CASES_DIR / f"{case_id}.json"


def load_case(case_id: str) -> GoldenCase:
    path = _case_path(case_id)
    if not path.exists():
        raise GoldenDatasetError(f"No golden case file for case_id={case_id!r} at {path}")
    return GoldenCase.model_validate_json(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_all_cases() -> tuple[GoldenCase, ...]:
    if not CASES_DIR.exists():
        raise GoldenDatasetError(
            f"No cases directory at {CASES_DIR}. Run "
            "`python scripts/generate_golden_dataset.py` first."
        )
    case_files = sorted(CASES_DIR.glob("*.json"))
    if not case_files:
        raise GoldenDatasetError(f"No case files found under {CASES_DIR}.")
    cases = [GoldenCase.model_validate_json(p.read_text(encoding="utf-8")) for p in case_files]
    return tuple(sorted(cases, key=lambda c: c.case_id))


def load_cases(case_ids: list[str]) -> list[GoldenCase]:
    return [load_case(cid) for cid in case_ids]


def load_train_cases() -> list[GoldenCase]:
    split = load_split()
    return load_cases(split.train)


def load_validation_cases() -> list[GoldenCase]:
    split = load_split()
    return load_cases(split.validation)


def load_test_cases() -> list[GoldenCase]:
    split = load_split()
    return load_cases(split.test)


def cases_by_family(cases: tuple[GoldenCase, ...] | None = None) -> dict[str, list[GoldenCase]]:
    families: dict[str, list[GoldenCase]] = defaultdict(list)
    for case in cases if cases is not None else load_all_cases():
        families[case.duplicate_family_id].append(case)
    return dict(families)


def validate_no_family_leakage(split: DatasetSplit | None = None) -> list[str]:
    """Return family_ids (if any) whose members straddle more than one of
    train/validation/test. An empty list means the split is leakage-free.
    """
    resolved_split = split or load_split()
    all_cases = {c.case_id: c for c in load_all_cases()}

    families_by_slice = []
    for ids in (resolved_split.train, resolved_split.validation, resolved_split.test):
        families_by_slice.append(
            {all_cases[cid].duplicate_family_id for cid in ids if cid in all_cases}
        )

    leaked: set[str] = set()
    for i in range(len(families_by_slice)):
        for j in range(i + 1, len(families_by_slice)):
            leaked |= families_by_slice[i] & families_by_slice[j]
    return sorted(leaked)


def dataset_as_json_dict() -> dict[str, object]:
    """Convenience export used by the observability/audit export (Phase 9)."""
    return {
        "manifest": json.loads(load_manifest().model_dump_json()),
        "split": json.loads(load_split().model_dump_json()),
        "cases": [json.loads(c.model_dump_json()) for c in load_all_cases()],
    }
