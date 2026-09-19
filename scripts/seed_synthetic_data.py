#!/usr/bin/env python3
"""Seed the local synthetic case corpus used for retrieval/duplicate testing.

Phase 1 scaffold: the demo case files under data/synthetic_emails and
data/synthetic_attachments already exist. The full 100+ record golden
dataset generator is implemented in Phase 2 (see progress.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    demo_email = REPO_ROOT / "data" / "synthetic_emails" / "demo_case_001.eml"
    demo_attachment = (
        REPO_ROOT
        / "data"
        / "synthetic_attachments"
        / "demo_case_001_discharge_summary.txt"
    )
    if not demo_email.exists() or not demo_attachment.exists():
        print("Demo case files are missing under data/.", file=sys.stderr)
        return 1

    print(f"Demo case present: {demo_email.relative_to(REPO_ROOT)}")
    print(f"Demo attachment present: {demo_attachment.relative_to(REPO_ROOT)}")
    print(
        "Full golden-dataset generator (100+ labeled synthetic records) is "
        "implemented in Phase 2 — see progress.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
