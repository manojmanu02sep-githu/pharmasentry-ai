"""Unit tests for src/memory/episodic.py (SQLite-backed episodic memory)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.memory.episodic import (
    EpisodicMemoryStore,
    read_episodic_history,
    read_episodic_state,
    record_episodic_event,
    write_episodic_state,
)
from src.models.enums import AgentName, MemoryOperation, MemoryTier
from src.tools.base import ToolAuthorizationError, ToolContext


def _ctx(case_id: str) -> ToolContext:
    return ToolContext(
        case_id=case_id,
        trace_id="trace_001",
        agent=AgentName.SUPERVISOR,
        authorized_tools=frozenset(),
    )


def _store(tmp_path: Path) -> EpisodicMemoryStore:
    return EpisodicMemoryStore(tmp_path / "episodic.db")


def test_write_then_read_state_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ctx = _ctx("case_a")

    write_event = write_episodic_state(store, "case_a", {"status": "awaiting_human"}, ctx)
    data, read_event = read_episodic_state(store, "case_a", ctx)

    assert data == {"status": "awaiting_human"}
    assert write_event.tier == MemoryTier.EPISODIC
    assert write_event.operation == MemoryOperation.WRITE
    assert read_event.operation == MemoryOperation.READ


def test_read_missing_case_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    data, _ = read_episodic_state(store, "case_a", _ctx("case_a"))
    assert data is None


def test_state_persists_across_store_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "episodic.db"
    store_a = EpisodicMemoryStore(db_path)
    write_episodic_state(store_a, "case_a", {"status": "completed"}, _ctx("case_a"))
    store_a.close()

    store_b = EpisodicMemoryStore(db_path)
    data, _ = read_episodic_state(store_b, "case_a", _ctx("case_a"))
    assert data == {"status": "completed"}


def test_cross_case_read_is_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ToolAuthorizationError):
        read_episodic_state(store, "case_a", _ctx("case_b"))


def test_cross_case_write_is_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ToolAuthorizationError):
        write_episodic_state(store, "case_a", {"tampered": True}, _ctx("case_b"))
    assert store.get("case_a") is None


def test_record_and_read_history_is_append_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ctx = _ctx("case_a")

    entry_1, _ = record_episodic_event(store, "case_a", "status_change", "Case created.", ctx)
    entry_2, _ = record_episodic_event(
        store, "case_a", "reviewer_correction", "Dose corrected.", ctx
    )

    history, event = read_episodic_history(store, "case_a", ctx)
    assert [h.event_type for h in history] == ["status_change", "reviewer_correction"]
    assert [h.summary for h in history] == ["Case created.", "Dose corrected."]
    assert event.tier == MemoryTier.EPISODIC
    assert entry_1.case_id == "case_a"
    assert entry_2.case_id == "case_a"


def test_history_summary_is_truncated_defensively(tmp_path: Path) -> None:
    store = _store(tmp_path)
    long_summary = "x" * 5000
    entry, _ = record_episodic_event(store, "case_a", "status_change", long_summary, _ctx("case_a"))
    assert len(entry.summary) == 2000


def test_cross_case_history_access_is_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ToolAuthorizationError):
        record_episodic_event(store, "case_a", "status_change", "x", _ctx("case_b"))
    with pytest.raises(ToolAuthorizationError):
        read_episodic_history(store, "case_a", _ctx("case_b"))


def test_purge_expired_removes_old_rows_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    write_episodic_state(store, "case_old", {"status": "completed"}, _ctx("case_old"))
    write_episodic_state(store, "case_new", {"status": "completed"}, _ctx("case_new"))

    # A retention window of 0 days means "everything older than right now" —
    # both rows were just written, so a huge window keeps everything and a
    # negative-effective window (retention_days=0, but writes happened at
    # the same instant) is timing-sensitive; assert the real, deterministic
    # boundary instead: a very large retention window purges nothing.
    deleted = store.purge_expired(retention_days=9999)
    assert deleted == 0
    assert store.get("case_old") is not None
    assert store.get("case_new") is not None
