#!/usr/bin/env python3
"""Seed and validate all synthetic data this project ships with.

Real executable, not a placeholder: it (1) checks the Phase 1 demo case
files exist, (2) (re)generates the golden dataset via
generate_golden_dataset.py, and (3) validates the generated dataset against
the checks CLAUDE.md/the build instructions require: record counts, unique
case IDs, split isolation (no duplicate-family leakage), label completeness,
evidence validity, synthetic-data markers, and duplicate-family consistency.

Exits non-zero (and prints why) on any validation failure — it never
reports success it didn't check for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (  # noqa: E402
    GoldenCase,
    cases_by_family,
    load_all_cases,
    load_manifest,
    load_split,
    validate_no_family_leakage,
)
from src.evaluation.schemas import FIELD_KEYS  # noqa: E402


def _check_demo_case_files() -> list[str]:
    errors = []
    demo_email = REPO_ROOT / "data" / "synthetic_emails" / "demo_case_001.eml"
    demo_attachment = (
        REPO_ROOT / "data" / "synthetic_attachments" / "demo_case_001_discharge_summary.txt"
    )
    if not demo_email.exists():
        errors.append(f"missing demo email: {demo_email}")
    if not demo_attachment.exists():
        errors.append(f"missing demo attachment: {demo_attachment}")
    return errors


def _run_generator() -> list[str]:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_golden_dataset.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return [f"generate_golden_dataset.py failed: {result.stderr.strip()}"]
    return []


def _validate_record_counts(cases: list[GoldenCase]) -> list[str]:
    errors = []
    if len(cases) < 120:
        errors.append(f"expected >=120 golden cases, found {len(cases)}")
    return errors


def _validate_unique_ids(cases: list[GoldenCase]) -> list[str]:
    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        seen: set[str] = set()
        dupes = {cid for cid in ids if cid in seen or seen.add(cid)}  # type: ignore[func-returns-value]
        return [f"duplicate case_id values found: {sorted(dupes)}"]
    return []


def _validate_split_isolation() -> list[str]:
    split = load_split()
    errors = []
    if not split.leakage_free():
        errors.append("train/validation/test splits overlap at the case_id level")
    leaked_families = validate_no_family_leakage(split)
    if leaked_families:
        errors.append(f"duplicate families leak across splits: {leaked_families}")
    return errors


def _validate_label_completeness(cases: list[GoldenCase]) -> list[str]:
    errors = []
    for case in cases:
        if not case.email_subject or not case.email_body:
            errors.append(f"{case.case_id}: missing email_subject/email_body")
        if case.expected_minimum_criteria is None:
            errors.append(f"{case.case_id}: missing expected_minimum_criteria")
        for key in case.expected_missing_fields + case.expected_conflicting_fields:
            if key not in FIELD_KEYS:
                errors.append(f"{case.case_id}: unknown field key {key!r}")
    return errors


def _validate_evidence_validity(cases: list[GoldenCase]) -> list[str]:
    """Every expected_source_evidence snippet must actually appear in that
    case's own email or attachment text — evidence must be real, not
    invented."""
    errors = []
    for case in cases:
        haystack = case.email_subject + "\n" + case.email_body + "\n" + case.attachment_text
        for snippet in case.expected_source_evidence:
            if snippet not in haystack:
                errors.append(
                    f"{case.case_id}: evidence snippet {snippet!r} not found in "
                    "its own email/attachment text"
                )
    return errors


def _validate_synthetic_markers(cases: list[GoldenCase]) -> list[str]:
    errors = []
    for case in cases:
        if case.synthetic_data is not True:
            errors.append(f"{case.case_id}: synthetic_data flag is not True")
        if case.attachment_metadata.has_attachment and "SYNTHETIC" not in case.attachment_text:
            errors.append(f"{case.case_id}: attachment missing SYNTHETIC marker")
        if "SYNTHETIC" not in case.email_body and "FICTIONAL" not in case.email_body:
            errors.append(f"{case.case_id}: email body missing synthetic/fictional marker")
    return errors


def _validate_duplicate_family_consistency(cases: list[GoldenCase]) -> list[str]:
    errors = []
    families = cases_by_family(tuple(cases))
    for case in cases:
        for match_id in case.expected_duplicate_matches:
            if match_id not in {c.case_id for c in families.get(case.duplicate_family_id, [])}:
                errors.append(
                    f"{case.case_id}: expected_duplicate_matches references "
                    f"{match_id!r}, which is not in family {case.duplicate_family_id!r}"
                )
    return errors


def main() -> int:
    all_errors: list[str] = []

    all_errors += _check_demo_case_files()
    all_errors += _run_generator()
    if all_errors:
        for err in all_errors:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1

    cases = list(load_all_cases())
    manifest = load_manifest()

    checks: list[tuple[str, list[str]]] = [
        ("record counts", _validate_record_counts(cases)),
        ("unique case IDs", _validate_unique_ids(cases)),
        ("split isolation", _validate_split_isolation()),
        ("label completeness", _validate_label_completeness(cases)),
        ("evidence validity", _validate_evidence_validity(cases)),
        ("synthetic-data markers", _validate_synthetic_markers(cases)),
        ("duplicate-family consistency", _validate_duplicate_family_consistency(cases)),
    ]

    ok = True
    for name, errors in checks:
        if errors:
            ok = False
            print(f"FAIL [{name}]: {len(errors)} issue(s)", file=sys.stderr)
            for err in errors[:10]:
                print(f"  - {err}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more", file=sys.stderr)
        else:
            print(f"OK   [{name}]")

    print(
        f"\n{manifest.total_cases} golden cases across "
        f"{len(manifest.category_counts)} case types (seed={manifest.seed})."
    )
    if not ok:
        print("\nSeed/validation FAILED — see FAIL lines above.", file=sys.stderr)
        return 1

    print("\nAll synthetic data checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
