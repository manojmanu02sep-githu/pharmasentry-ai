"""Sanity checks on the synthetic demo case fixture files.

These are not the real Intake/Document Agent tools (Phase 3+) — just a
check that the fixture files exist, are well-formed, and are explicitly
marked synthetic, matching the CLAUDE.md demo case description.
"""

from __future__ import annotations

from email import message_from_string
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_demo_email_exists_and_parses() -> None:
    email_path = REPO_ROOT / "data" / "synthetic_emails" / "demo_case_001.eml"
    assert email_path.exists()

    raw = email_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in raw
    assert "DemoGluca" in raw

    msg = message_from_string(raw)
    assert msg["Subject"] is not None
    assert "DemoGluca" in msg["Subject"]
    assert msg.get_payload()  # non-empty body


def test_demo_attachment_exists_and_mentions_expected_facts() -> None:
    attachment_path = (
        REPO_ROOT
        / "data"
        / "synthetic_attachments"
        / "demo_case_001_discharge_summary.txt"
    )
    assert attachment_path.exists()

    text = attachment_path.read_text(encoding="utf-8")
    assert "SYNTHETIC" in text
    assert "acute pancreatitis" in text.lower() or "pancreatitis" in text.lower()
    assert "62-year-old male" in text


def test_synthetic_data_notice_present() -> None:
    notice_path = REPO_ROOT / "data" / "SYNTHETIC_DATA_NOTICE.md"
    assert notice_path.exists()
    assert "fabricated" in notice_path.read_text(encoding="utf-8").lower()
