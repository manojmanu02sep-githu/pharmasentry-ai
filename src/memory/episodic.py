"""Episodic memory: durable, case-isolated history of approved run status,
errors, and reviewer corrections, across process restarts.

Backed by stdlib `sqlite3` rather than SQLAlchemy: `requirements.txt`
declares SQLAlchemy but it is not installed in this environment, and the
storage shape here (one JSON blob per case plus an append-only event log)
does not need an ORM. This is a documented deviation, not a silent one —
see progress.md's Phase 4 section.

`EpisodicMemoryStore` satisfies the same `CaseStateStore` protocol
(`src/tools/workflow.py`) as `InMemoryCaseStateStore`, so
`ReadCaseStateTool`/`WriteCaseCheckpointTool` can use either backend
without changing their own code — exactly the seam that module's docstring
calls out ("Phase 4 adds a SQLite-backed episodic-memory store behind the
same protocol").

Every accessor here enforces case isolation and returns a `MemoryEvent` so
callers can append it to `CaseState.memory_events`. History summaries are
caller-provided, concise, evidence-based strings — never chain-of-thought
(truncated defensively in case a caller passes something too long).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.models.audit import MemoryEvent
from src.models.enums import MemoryOperation, MemoryTier
from src.tools.base import ToolAuthorizationError, ToolContext

_MAX_SUMMARY_CHARS = 2000


class EpisodicHistoryEntry(BaseModel):
    case_id: str
    event_type: str
    summary: str = Field(max_length=_MAX_SUMMARY_CHARS)
    version: int
    created_at: datetime


def _enforce_case_isolation(target_case_id: str, ctx: ToolContext) -> None:
    if target_case_id != ctx.case_id:
        raise ToolAuthorizationError(
            f"cross-case episodic-memory access blocked: agent for case "
            f"{ctx.case_id!r} requested case {target_case_id!r}"
        )


class EpisodicMemoryStore:
    """SQLite-backed episodic memory. One connection per store instance.

    Two tables:
      - `episodic_state`: latest checkpoint per case (upsert, versioned).
      - `episodic_history`: append-only log of status/error/correction
        events per case, each with a short caller-provided summary.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_state (
                case_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                version INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodic_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodic_history_case_id "
            "ON episodic_history(case_id)"
        )
        self._conn.commit()

    # --- CaseStateStore protocol (get/put) --------------------------------

    def get(self, case_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM episodic_state WHERE case_id = ?", (case_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def put(self, case_id: str, data: dict[str, Any]) -> None:
        now = datetime.now(UTC).isoformat()
        current_version = self.version_of(case_id)
        next_version = current_version + 1
        self._conn.execute(
            """
            INSERT INTO episodic_state (case_id, data, version, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(case_id) DO UPDATE SET
                data = excluded.data,
                version = excluded.version,
                updated_at = excluded.updated_at
            """,
            (case_id, json.dumps(data), next_version, now),
        )
        self._conn.commit()

    def version_of(self, case_id: str) -> int:
        row = self._conn.execute(
            "SELECT version FROM episodic_state WHERE case_id = ?", (case_id,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    # --- history log --------------------------------------------------------

    def append_history(self, case_id: str, event_type: str, summary: str) -> EpisodicHistoryEntry:
        truncated = summary[:_MAX_SUMMARY_CHARS]
        now = datetime.now(UTC)
        version = self.version_of(case_id)
        self._conn.execute(
            """
            INSERT INTO episodic_history (case_id, event_type, summary, version, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (case_id, event_type, truncated, version, now.isoformat()),
        )
        self._conn.commit()
        return EpisodicHistoryEntry(
            case_id=case_id,
            event_type=event_type,
            summary=truncated,
            version=version,
            created_at=now,
        )

    def history_for(self, case_id: str) -> list[EpisodicHistoryEntry]:
        rows = self._conn.execute(
            """
            SELECT case_id, event_type, summary, version, created_at
            FROM episodic_history WHERE case_id = ? ORDER BY id ASC
            """,
            (case_id,),
        ).fetchall()
        return [
            EpisodicHistoryEntry(
                case_id=r[0], event_type=r[1], summary=r[2], version=r[3], created_at=r[4]
            )
            for r in rows
        ]

    # --- retention -----------------------------------------------------------

    def purge_expired(self, retention_days: int) -> int:
        """Delete state and history rows older than the retention window.

        Returns the total number of rows deleted (state + history), so
        callers can produce an honest audit/observability count rather
        than an assumed one.
        """
        cutoff = datetime.now(UTC).timestamp() - retention_days * 86400
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=UTC).isoformat()
        state_cursor = self._conn.execute(
            "DELETE FROM episodic_state WHERE updated_at < ?", (cutoff_iso,)
        )
        history_cursor = self._conn.execute(
            "DELETE FROM episodic_history WHERE created_at < ?", (cutoff_iso,)
        )
        self._conn.commit()
        return state_cursor.rowcount + history_cursor.rowcount

    def close(self) -> None:
        self._conn.close()


# --- case-isolated, logged accessor functions -------------------------------


def read_episodic_state(
    store: EpisodicMemoryStore, case_id: str, ctx: ToolContext
) -> tuple[dict[str, Any] | None, MemoryEvent]:
    _enforce_case_isolation(case_id, ctx)
    data = store.get(case_id)
    event = MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.EPISODIC,
        operation=MemoryOperation.READ,
        key=f"episodic_state:{case_id}",
        agent=ctx.agent,
    )
    return data, event


def write_episodic_state(
    store: EpisodicMemoryStore, case_id: str, data: dict[str, Any], ctx: ToolContext
) -> MemoryEvent:
    _enforce_case_isolation(case_id, ctx)
    store.put(case_id, data)
    return MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.EPISODIC,
        operation=MemoryOperation.WRITE,
        key=f"episodic_state:{case_id}",
        agent=ctx.agent,
    )


def record_episodic_event(
    store: EpisodicMemoryStore,
    case_id: str,
    event_type: str,
    summary: str,
    ctx: ToolContext,
) -> tuple[EpisodicHistoryEntry, MemoryEvent]:
    """Append a concise, evidence-based history entry (approved run status,
    error, or reviewer correction) — never raw chain-of-thought."""
    _enforce_case_isolation(case_id, ctx)
    entry = store.append_history(case_id, event_type, summary)
    event = MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.EPISODIC,
        operation=MemoryOperation.WRITE,
        key=f"episodic_history:{case_id}:{event_type}",
        agent=ctx.agent,
    )
    return entry, event


def read_episodic_history(
    store: EpisodicMemoryStore, case_id: str, ctx: ToolContext
) -> tuple[list[EpisodicHistoryEntry], MemoryEvent]:
    _enforce_case_isolation(case_id, ctx)
    entries = store.history_for(case_id)
    event = MemoryEvent(
        case_id=ctx.case_id,
        trace_id=ctx.trace_id,
        tier=MemoryTier.EPISODIC,
        operation=MemoryOperation.READ,
        key=f"episodic_history:{case_id}",
        agent=ctx.agent,
    )
    return entries, event
