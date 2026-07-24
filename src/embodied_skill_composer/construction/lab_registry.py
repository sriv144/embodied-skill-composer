from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4


RunStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "interrupted",
    "resuming",
    "completed",
    "failed",
    "cancelled",
]
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
QUIESCENT_RUN_STATUSES: frozenset[str] = TERMINAL_RUN_STATUSES | {"interrupted"}
ACTIVE_RUN_STATUSES: tuple[str, ...] = ("running", "cancel_requested")
RESUMABLE_RUN_STATUSES: tuple[str, ...] = ("interrupted", "failed", "cancelled")
DEFAULT_STALE_WORKER_TIMEOUT = timedelta(seconds=60)
PROCESS_TERMINATION_TIMEOUT_SECONDS = 5.0

QuiescentRunStatus = Literal["interrupted", "completed", "failed", "cancelled"]
PolicyRegistration = tuple[str, str, dict[str, object]]


class LostRunClaimError(RuntimeError):
    """Raised when a worker attempts to mutate a run after losing its lease."""


class LabRegistry:
    """SQLite-backed source of truth for experiments, events, and queue ownership."""

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
                "SELECT id, seed, split, payload_json, created_at FROM scenarios "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "seed": row["seed"],
                "split": str(row["split"]),
                "payload": _object_dict(row["payload_json"]),
                "created_at": str(row["created_at"]),
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
            "id": str(row["id"]),
            "seed": row["seed"],
            "split": str(row["split"]),
            "payload": _object_dict(row["payload_json"]),
            "created_at": str(row["created_at"]),
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
                "SELECT id, controller, manifest_json, created_at FROM policies "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "controller": str(row["controller"]),
                "manifest": _object_dict(row["manifest_json"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def create_run(
        self,
        kind: str,
        config: dict[str, object],
        *,
        status: RunStatus = "queued",
        input_payload: dict[str, object] | None = None,
        config_digest: str | None = None,
        source_commit: str | None = None,
        run_id: str | None = None,
    ) -> str:
        run_id = run_id or (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{kind}-{uuid4().hex[:8]}"
        )
        event_log_path = self.path.parent / "events" / f"{run_id}.jsonl"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    id, kind, status, config_json, input_json, created_at, progress,
                    cancel_requested, event_log_path, config_digest, source_commit
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
                """,
                (
                    run_id,
                    kind,
                    status,
                    _json(config),
                    _json(input_payload or {}),
                    _now(),
                    str(event_log_path),
                    config_digest,
                    source_commit,
                ),
            )
        self.append_event(run_id, {"event": "run_created", "status": status})
        return run_id

    def update_run(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        progress: float | None = None,
        artifact_dir: str | None = None,
        error: str | None = None,
        latest_checkpoint: str | None = None,
        pid: int | None = None,
        process_identity: str | None = None,
        heartbeat: bool = False,
        claim_token: str | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[object] = []
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
            if status == "running":
                assignments.append("started_at = COALESCE(started_at, ?)")
                values.append(_now())
            if status == "interrupted":
                interrupted_at = _now()
                assignments.extend(
                    [
                        "interrupted_at = ?",
                        "ended_at = ?",
                        "pid = NULL",
                        "process_identity = NULL",
                        "claim_token = NULL",
                    ]
                )
                values.extend([interrupted_at, interrupted_at])
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
        if latest_checkpoint is not None:
            assignments.append("latest_checkpoint = ?")
            values.append(latest_checkpoint)
        if pid is not None:
            assignments.append("pid = ?")
            values.append(pid)
        if process_identity is not None:
            assignments.append("process_identity = ?")
            values.append(process_identity)
        if heartbeat:
            assignments.append("heartbeat_at = ?")
            values.append(_now())
        if not assignments:
            return
        where = "id = ?"
        values.append(run_id)
        if claim_token is not None:
            where += " AND claim_token = ? AND status IN ('running', 'cancel_requested')"
            values.append(claim_token)
        with self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE runs SET {', '.join(assignments)} WHERE {where}",
                values,
            )
            if cursor.rowcount != 1:
                if claim_token is not None:
                    raise LostRunClaimError(run_id)
                raise KeyError(run_id)

    def claim_next_training(self) -> dict[str, object] | None:
        """Atomically claim one queued/resuming training job if the GPU slot is free."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                "SELECT id FROM runs WHERE kind = 'training' "
                "AND status IN ('running', 'cancel_requested') LIMIT 1"
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                """
                SELECT id FROM runs
                WHERE kind = 'training' AND status IN ('queued', 'resuming')
                ORDER BY created_at, id LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            run_id = str(row["id"])
            token = uuid4().hex
            now = _now()
            connection.execute(
                """
                UPDATE runs SET status = 'running', started_at = COALESCE(started_at, ?),
                    ended_at = NULL, error = NULL, claim_token = ?, attempt = attempt + 1,
                    heartbeat_at = ?, pid = NULL, process_identity = NULL
                WHERE id = ?
                """,
                (now, token, now, run_id),
            )
            claimed = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        assert claimed is not None
        self.append_event(
            run_id,
            {"event": "run_claimed", "attempt": int(claimed["attempt"])},
        )
        return _run_row(claimed)

    def verify_claim(self, run_id: str, claim_token: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT claim_token, status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return bool(
            row
            and str(row["claim_token"]) == claim_token
            and str(row["status"]) in ACTIVE_RUN_STATUSES
        )

    def finalize_run(
        self,
        run_id: str,
        *,
        status: QuiescentRunStatus,
        event: Mapping[str, object],
        claim_token: str | None = None,
        progress: float | None = None,
        artifact_dir: str | None = None,
        error: str | None = None,
        policy: PolicyRegistration | None = None,
    ) -> int:
        """Atomically commit a run's terminal state, event, and optional policy."""

        if policy is not None and status != "completed":
            raise ValueError("a policy may only be registered for a completed run")
        created_at = _now()
        serialized_event = _json(dict(event))
        bounded_progress = (
            min(max(progress, 0.0), 1.0) if progress is not None else None
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT claim_token, status, event_log_path FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if owner is None:
                raise KeyError(run_id)
            owns_active_run = str(owner["status"]) in ACTIVE_RUN_STATUSES
            if claim_token is not None:
                owns_active_run = (
                    owns_active_run and str(owner["claim_token"]) == claim_token
                )
            if not owns_active_run:
                if claim_token is not None:
                    raise LostRunClaimError(run_id)
                raise KeyError(run_id)
            if policy is not None:
                policy_id, controller, manifest = policy
                connection.execute(
                    """
                    INSERT INTO policies (id, controller, manifest_json, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        controller = excluded.controller,
                        manifest_json = excluded.manifest_json
                    """,
                    (policy_id, controller, _json(manifest), created_at),
                )
            sequence = _insert_event(
                connection,
                run_id,
                created_at=created_at,
                serialized_payload=serialized_event,
            )
            interrupted_at = created_at if status == "interrupted" else None
            cursor = connection.execute(
                """
                UPDATE runs SET status = ?, progress = COALESCE(?, progress),
                    artifact_dir = COALESCE(?, artifact_dir), error = ?, ended_at = ?,
                    interrupted_at = COALESCE(?, interrupted_at), heartbeat_at = ?,
                    pid = NULL, process_identity = NULL, claim_token = NULL
                WHERE id = ? AND status IN ('running', 'cancel_requested')
                """,
                (
                    status,
                    bounded_progress,
                    artifact_dir,
                    error,
                    created_at,
                    interrupted_at,
                    created_at,
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                if claim_token is not None:
                    raise LostRunClaimError(run_id)
                raise KeyError(run_id)
            event_log_path = owner["event_log_path"]
        _write_event_log_record(
            event_log_path,
            sequence=sequence,
            created_at=created_at,
            payload=event,
        )
        return sequence

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None or str(row["status"]) in QUIESCENT_RUN_STATUSES:
                return False
            current = str(row["status"])
            target = "cancelled" if current in {"queued", "resuming"} else "cancel_requested"
            cursor = connection.execute(
                """
                UPDATE runs SET cancel_requested = 1, status = ?,
                    ended_at = CASE WHEN ? = 'cancelled' THEN ? ELSE ended_at END
                WHERE id = ?
                """,
                (target, target, _now(), run_id),
            )
        if cursor.rowcount:
            self.append_event(run_id, {"event": target})
        return bool(cursor.rowcount)

    def request_resume(self, run_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status, latest_checkpoint FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if (
                row is None
                or str(row["status"]) not in RESUMABLE_RUN_STATUSES
                or not row["latest_checkpoint"]
                or not Path(str(row["latest_checkpoint"])).is_file()
            ):
                return False
            connection.execute(
                """
                UPDATE runs SET status = 'resuming', cancel_requested = 0, error = NULL,
                    ended_at = NULL, pid = NULL, process_identity = NULL,
                    claim_token = NULL, interrupted_at = NULL
                WHERE id = ?
                """,
                (run_id,),
            )
        self.append_event(
            run_id,
            {"event": "resume_requested", "checkpoint": str(row["latest_checkpoint"])},
        )
        return True

    def cancel_requested(self, run_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def reconcile_stale_runs(
        self,
        *,
        stale_after: timedelta = DEFAULT_STALE_WORKER_TIMEOUT,
        process_alive: Callable[[int], bool] | None = None,
        process_identity: Callable[[int], str | None] | None = None,
        terminate_owned_process: Callable[[int, str], bool] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Fence expired leases and release them only after the old worker is gone."""

        process_alive = process_alive or _process_alive
        process_identity = process_identity or _process_identity
        terminate_owned_process = terminate_owned_process or _terminate_owned_process
        now = now or datetime.now(UTC)
        interrupted: list[str] = []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM runs WHERE kind = 'training' "
                "AND status IN ('running', 'cancel_requested')"
            ).fetchall()
        for candidate in rows:
            run_id = str(candidate["id"])
            stale = self._fence_stale_worker(
                run_id,
                stale_after=stale_after,
                now=now,
            )
            if stale is None:
                continue
            pid_value = stale.get("pid")
            pid = int(pid_value) if isinstance(pid_value, int) else None
            expected_identity_value = stale.get("process_identity")
            expected_identity = (
                str(expected_identity_value) if expected_identity_value else None
            )
            safe_to_release = pid is None or not process_alive(pid)
            reason = "stale_worker"
            if not safe_to_release and pid is not None:
                actual_identity = process_identity(pid)
                if expected_identity is None or actual_identity is None:
                    continue
                if actual_identity != expected_identity:
                    safe_to_release = True
                    reason = "stale_worker_pid_reused"
                else:
                    safe_to_release = terminate_owned_process(pid, expected_identity)
                    reason = "stale_worker_terminated"
            if not safe_to_release:
                continue
            reaper_token = stale.get("claim_token")
            if not isinstance(reaper_token, str):
                continue
            try:
                self.finalize_run(
                    run_id,
                    status="interrupted",
                    event={"event": "run_interrupted", "reason": reason},
                    claim_token=reaper_token,
                    error="worker heartbeat expired",
                )
            except LostRunClaimError:
                continue
            interrupted.append(run_id)
        return interrupted

    def _fence_stale_worker(
        self,
        run_id: str,
        *,
        stale_after: timedelta,
        now: datetime,
    ) -> dict[str, object] | None:
        """Replace a stale worker claim while retaining the active GPU-slot state."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if row is None or str(row["status"]) not in ACTIVE_RUN_STATUSES:
                return None
            heartbeat = _parse_timestamp(row["heartbeat_at"])
            if heartbeat is not None and now - heartbeat <= stale_after:
                return None
            existing_token = str(row["claim_token"] or "")
            reaper_token = (
                existing_token if existing_token.startswith("reaper:") else f"reaper:{uuid4().hex}"
            )
            if reaper_token != existing_token:
                connection.execute(
                    "UPDATE runs SET claim_token = ? WHERE id = ?",
                    (reaper_token, run_id),
                )
            payload = _run_row(row)
            payload["claim_token"] = reaper_token
            return payload

    def get_run(self, run_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return _run_row(row) if row else None

    def list_runs(self, *, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_run_row(row) for row in rows]

    def append_event(
        self,
        run_id: str,
        payload: Mapping[str, object],
        *,
        claim_token: str | None = None,
    ) -> int:
        created_at = _now()
        serialized = _json(dict(payload))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if claim_token is not None:
                owner = connection.execute(
                    "SELECT claim_token, status FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if (
                    owner is None
                    or str(owner["claim_token"]) != claim_token
                    or str(owner["status"]) not in ACTIVE_RUN_STATUSES
                ):
                    raise LostRunClaimError(run_id)
            sequence = _insert_event(
                connection,
                run_id,
                created_at=created_at,
                serialized_payload=serialized,
            )
            row = connection.execute(
                "SELECT event_log_path FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        _write_event_log_record(
            row["event_log_path"] if row else None,
            sequence=sequence,
            created_at=created_at,
            payload=payload,
        )
        return sequence

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
                "sequence": int(row["sequence"]),
                "created_at": str(row["created_at"]),
                "payload": _object_dict(row["payload_json"]),
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
                    input_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    progress REAL NOT NULL,
                    artifact_dir TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    pid INTEGER,
                    process_identity TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    heartbeat_at TEXT,
                    event_log_path TEXT,
                    latest_checkpoint TEXT,
                    config_digest TEXT,
                    source_commit TEXT,
                    claim_token TEXT,
                    interrupted_at TEXT
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
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            migrations = {
                "input_json": "TEXT NOT NULL DEFAULT '{}'",
                "pid": "INTEGER",
                "process_identity": "TEXT",
                "attempt": "INTEGER NOT NULL DEFAULT 0",
                "heartbeat_at": "TEXT",
                "event_log_path": "TEXT",
                "latest_checkpoint": "TEXT",
                "config_digest": "TEXT",
                "source_commit": "TEXT",
                "claim_token": "TEXT",
                "interrupted_at": "TEXT",
            }
            for column, declaration in migrations.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {declaration}")
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_kind_status_created
                    ON runs(kind, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_heartbeat ON runs(heartbeat_at);
                PRAGMA user_version = 2;
                """
            )


def _run_row(row: sqlite3.Row) -> dict[str, object]:
    return {
        "id": str(row["id"]),
        "kind": str(row["kind"]),
        "status": str(row["status"]),
        "config": _object_dict(row["config_json"]),
        "input": _object_dict(row["input_json"]),
        "created_at": str(row["created_at"]),
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "progress": float(row["progress"]),
        "artifact_dir": row["artifact_dir"],
        "error": row["error"],
        "cancel_requested": bool(row["cancel_requested"]),
        "pid": row["pid"],
        "process_identity": row["process_identity"],
        "attempt": int(row["attempt"]),
        "heartbeat_at": row["heartbeat_at"],
        "event_log_path": row["event_log_path"],
        "latest_checkpoint": row["latest_checkpoint"],
        "config_digest": row["config_digest"],
        "source_commit": row["source_commit"],
        "claim_token": row["claim_token"],
        "interrupted_at": row["interrupted_at"],
    }


def _insert_event(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    created_at: str,
    serialized_payload: str,
) -> int:
    sequence_row = connection.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM run_events WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if sequence_row is None:
        raise RuntimeError(f"could not allocate an event sequence for {run_id}")
    sequence = int(sequence_row[0])
    connection.execute(
        "INSERT INTO run_events (run_id, sequence, created_at, payload_json) "
        "VALUES (?, ?, ?, ?)",
        (run_id, sequence, created_at, serialized_payload),
    )
    return sequence


def _write_event_log_record(
    event_log_path: object,
    *,
    sequence: int,
    created_at: str,
    payload: Mapping[str, object],
) -> None:
    if not event_log_path:
        return
    path = Path(str(event_log_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "sequence": sequence,
        "created_at": created_at,
        "payload": dict(payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _object_dict(value: object) -> dict[str, object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], parsed)


def _json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _process_identity(pid: int) -> str | None:
    """Return a stable creation identity so a recycled PID is never terminated."""

    if pid <= 0:
        return None
    if os.name == "nt":
        return _windows_process_identity(pid)
    proc_record = _proc_process_record(pid)
    if proc_record is not None:
        _, started_at = proc_record
        try:
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            boot_id = "unknown-boot"
        return f"linux:{boot_id}:{started_at}"
    try:
        result = subprocess.run(
            ["ps", "-o", "lstart=", "-o", "command=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    identity = " ".join(result.stdout.split())
    return f"posix:{identity}" if result.returncode == 0 and identity else None


def _process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.GetExitCodeProcess.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.OpenProcess(process_query_limited_information, 0, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    proc_record = _proc_process_record(pid)
    if proc_record is not None and proc_record[0] in {"X", "Z"}:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _terminate_owned_process(
    pid: int,
    expected_identity: str,
    *,
    timeout_seconds: float = PROCESS_TERMINATION_TIMEOUT_SECONDS,
) -> bool:
    """Terminate only the process whose creation identity matches the lease."""

    actual_identity = _process_identity(pid)
    if actual_identity is None:
        return not _process_alive(pid)
    if actual_identity != expected_identity:
        return True
    if os.name == "nt":
        return _terminate_owned_windows_process(
            pid,
            expected_identity,
            timeout_seconds=timeout_seconds,
        )
    return _terminate_owned_posix_process(
        pid,
        expected_identity,
        timeout_seconds=timeout_seconds,
    )


def _proc_process_record(pid: int) -> tuple[str, str] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return None
    command_end = raw.rfind(")")
    if command_end < 0:
        return None
    fields = raw[command_end + 1 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], fields[19]


def _windows_process_identity(pid: int) -> str | None:
    import ctypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    process_query_limited_information = 0x1000
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(process_query_limited_information, 0, pid)
    if not handle:
        return None
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel_time = FileTime()
        user_time = FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return None
        creation_ticks = (int(creation.high) << 32) | int(creation.low)
        return f"windows:{creation_ticks}"
    finally:
        kernel32.CloseHandle(handle)


def _terminate_owned_windows_process(
    pid: int,
    expected_identity: str,
    *,
    timeout_seconds: float,
) -> bool:
    import ctypes

    class FileTime(ctypes.Structure):
        _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

    process_terminate = 0x0001
    process_query_limited_information = 0x1000
    synchronize = 0x00100000
    still_active = 259
    wait_object_0 = 0
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
        ctypes.POINTER(FileTime),
    ]
    kernel32.GetProcessTimes.restype = ctypes.c_int
    kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int
    kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.TerminateProcess.restype = ctypes.c_int
    kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.WaitForSingleObject.restype = ctypes.c_ulong
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int
    access = process_terminate | process_query_limited_information | synchronize
    handle = kernel32.OpenProcess(access, 0, pid)
    if not handle:
        return not _process_alive(pid)
    try:
        creation = FileTime()
        exit_time = FileTime()
        kernel_time = FileTime()
        user_time = FileTime()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            return False
        actual_identity = f"windows:{(int(creation.high) << 32) | int(creation.low)}"
        if actual_identity != expected_identity:
            return True
        exit_code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            if exit_code.value != still_active:
                return True
        if not kernel32.TerminateProcess(handle, 1):
            return False
        wait_result = kernel32.WaitForSingleObject(
            handle,
            max(1, int(timeout_seconds * 1000)),
        )
        if wait_result == wait_object_0:
            return True
        return bool(
            kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            and exit_code.value != still_active
        )
    finally:
        kernel32.CloseHandle(handle)


def _terminate_owned_posix_process(
    pid: int,
    expected_identity: str,
    *,
    timeout_seconds: float,
) -> bool:
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    force_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
    if callable(pidfd_open) and callable(pidfd_send_signal):
        try:
            pidfd = pidfd_open(pid, 0)
        except OSError:
            return not _process_alive(pid)
        try:
            if _process_identity(pid) != expected_identity:
                return True
            try:
                pidfd_send_signal(pidfd, signal.SIGTERM, None, 0)
            except ProcessLookupError:
                return True
            if _wait_for_owned_process_exit(
                pid,
                expected_identity,
                timeout_seconds=timeout_seconds / 2,
            ):
                return True
            try:
                pidfd_send_signal(pidfd, force_signal, None, 0)
            except ProcessLookupError:
                return True
            return _wait_for_owned_process_exit(
                pid,
                expected_identity,
                timeout_seconds=timeout_seconds / 2,
            )
        finally:
            os.close(pidfd)
    if _process_identity(pid) != expected_identity:
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    if _wait_for_owned_process_exit(
        pid,
        expected_identity,
        timeout_seconds=timeout_seconds / 2,
    ):
        return True
    if _process_identity(pid) != expected_identity:
        return True
    try:
        os.kill(pid, force_signal)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return _wait_for_owned_process_exit(
        pid,
        expected_identity,
        timeout_seconds=timeout_seconds / 2,
    )


def _wait_for_owned_process_exit(
    pid: int,
    expected_identity: str,
    *,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        actual_identity = _process_identity(pid)
        if actual_identity is not None and actual_identity != expected_identity:
            return True
        time.sleep(0.05)
    return not _process_alive(pid)
