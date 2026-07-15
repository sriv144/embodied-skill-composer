from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


class LabRegistry:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def upsert_scenario(
        self,
        scenario_id: str,
        *,
        seed: int | None,
        split: str,
        payload: dict[str, object],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scenarios (id, seed, split, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    seed = excluded.seed,
                    split = excluded.split,
                    payload_json = excluded.payload_json
                """,
                (scenario_id, seed, split, _json(payload), _now()),
            )

    def list_scenarios(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, seed, split, payload_json, created_at FROM scenarios ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "seed": row["seed"],
                "split": row["split"],
                "payload": json.loads(row["payload_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def get_scenario(self, scenario_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, seed, split, payload_json, created_at FROM scenarios WHERE id = ?",
                (scenario_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "seed": row["seed"],
            "split": row["split"],
            "payload": json.loads(row["payload_json"]),
            "created_at": row["created_at"],
        }

    def upsert_policy(self, policy_id: str, controller: str, manifest: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO policies (id, controller, manifest_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    controller = excluded.controller,
                    manifest_json = excluded.manifest_json
                """,
                (policy_id, controller, _json(manifest), _now()),
            )

    def list_policies(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, controller, manifest_json, created_at FROM policies ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": row["id"],
                "controller": row["controller"],
                "manifest": json.loads(row["manifest_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_run(
        self,
        kind: str,
        config: dict[str, object],
        *,
        status: str = "queued",
    ) -> str:
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{kind}-{uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, kind, status, config_json, created_at, progress, cancel_requested
                ) VALUES (?, ?, ?, ?, ?, 0, 0)
                """,
                (run_id, kind, status, _json(config), _now()),
            )
        self.append_event(run_id, {"event": "run_created", "status": status})
        return run_id

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        artifact_dir: str | None = None,
        error: str | None = None,
    ) -> None:
        assignments = []
        values: list[object] = []
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
            if status == "running":
                assignments.append("started_at = COALESCE(started_at, ?)")
                values.append(_now())
            if status in TERMINAL_RUN_STATUSES:
                assignments.append("ended_at = ?")
                values.append(_now())
        if progress is not None:
            assignments.append("progress = ?")
            values.append(min(max(progress, 0.0), 1.0))
        if artifact_dir is not None:
            assignments.append("artifact_dir = ?")
            values.append(artifact_dir)
        if error is not None:
            assignments.append("error = ?")
            values.append(error)
        if not assignments:
            return
        values.append(run_id)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET cancel_requested = 1, status = 'cancel_requested'
                WHERE id = ? AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (run_id,),
            )
        if cursor.rowcount:
            self.append_event(run_id, {"event": "cancel_requested"})
        return bool(cursor.rowcount)

    def cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run_row(row) if row else None

    def list_runs(self, *, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_run_row(row) for row in rows]

    def append_event(self, run_id: str, payload: dict[str, object]) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO run_events (run_id, sequence, created_at, payload_json) VALUES (?, ?, ?, ?)",
                (run_id, sequence, _now(), _json(payload)),
            )
        return int(sequence)

    def list_events(self, run_id: str, *, after: int = 0) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, created_at, payload_json FROM run_events
                WHERE run_id = ? AND sequence > ? ORDER BY sequence
                """,
                (run_id, after),
            ).fetchall()
        return [
            {
                "sequence": row["sequence"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY,
                    seed INTEGER,
                    split TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policies (
                    id TEXT PRIMARY KEY,
                    controller TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    progress REAL NOT NULL,
                    artifact_dir TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sequence)
                );
                """
            )


def _run_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "config": json.loads(row["config_json"]),
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "progress": row["progress"],
        "artifact_dir": row["artifact_dir"],
        "error": row["error"],
        "cancel_requested": bool(row["cancel_requested"]),
    }


def _json(value: dict[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()
