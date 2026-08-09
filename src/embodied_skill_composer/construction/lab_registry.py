from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from embodied_skill_composer.construction.lab_events import (
    RunEventEnvelope,
    dump_run_event_payload,
    validate_run_event_payload,
)


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
DURABLE_SUBPROCESS_KINDS: tuple[str, ...] = ("training", "matrix_evaluation")
CANCELLABLE_RUN_STATUSES: frozenset[str] = frozenset(
    {"queued", "running", "resuming"}
)
MATRIX_EVALUATION_DEDUPLICATED_STATUSES: frozenset[str] = frozenset(
    {
        "queued",
        "running",
        "cancel_requested",
        "resuming",
        "completed",
    }
)
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

    def upsert_evaluation(
        self,
        evaluation_id: str,
        *,
        matrix_id: str | None,
        split: str,
        payload: dict[str, object],
        artifact_dir: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO evaluations (
                    id, matrix_id, split, payload_json, artifact_dir, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    matrix_id = excluded.matrix_id,
                    split = excluded.split,
                    payload_json = excluded.payload_json,
                    artifact_dir = excluded.artifact_dir
                """,
                (
                    evaluation_id,
                    matrix_id,
                    split,
                    _json(payload),
                    artifact_dir,
                    _now(),
                ),
            )

    def list_evaluations(
        self,
        *,
        matrix_id: str | None = None,
    ) -> list[dict[str, object]]:
        query = (
            "SELECT id, matrix_id, split, payload_json, artifact_dir, created_at "
            "FROM evaluations"
        )
        parameters: tuple[object, ...] = ()
        if matrix_id is not None:
            query += " WHERE matrix_id = ?"
            parameters = (matrix_id,)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "id": str(row["id"]),
                "matrix_id": row["matrix_id"],
                "split": str(row["split"]),
                "payload": _object_dict(row["payload_json"]),
                "artifact_dir": str(row["artifact_dir"]),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def create_experiment_matrix(
        self,
        matrix_id: str,
        *,
        protocol_digest: str,
        protocol: dict[str, object],
        execution_profile: str,
        design: dict[str, object],
        runs: list[
            tuple[
                str,
                dict[str, object],
                str | None,
                str | None,
            ]
        ],
    ) -> list[str]:
        """Atomically persist a frozen experiment matrix and all queued runs."""

        if not runs:
            raise ValueError("an experiment matrix must contain at least one run")
        created_at = _now()
        run_records: list[tuple[str, Path, dict[str, object]]] = []
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                """
                SELECT id FROM experiment_matrices
                WHERE protocol_digest = ? AND execution_profile = ?
                """,
                (protocol_digest, execution_profile),
            ).fetchone()
            if duplicate is not None:
                raise ValueError(
                    "experiment matrix already exists for this protocol digest "
                    f"and profile: {duplicate['id']}"
                )
            connection.execute(
                """
                INSERT INTO experiment_matrices (
                    id, protocol_digest, protocol_json, execution_profile,
                    expected_run_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    matrix_id,
                    protocol_digest,
                    _json(protocol),
                    execution_profile,
                    len(runs),
                    created_at,
                ),
            )
            for ordinal, (run_key, config, config_digest, source_commit) in enumerate(runs):
                run_id = f"{matrix_id}-{run_key}"
                event_log_path = self.path.parent / "events" / f"{run_id}.jsonl"
                connection.execute(
                    """
                    INSERT INTO runs (
                        id, kind, status, config_json, input_json, created_at, progress,
                        cancel_requested, event_log_path, config_digest, source_commit,
                        resume_provenance_json
                    ) VALUES (?, 'training', 'queued', ?, ?, ?, 0, 0, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        _json(config),
                        _json({"design": design}),
                        created_at,
                        str(event_log_path),
                        config_digest,
                        source_commit,
                        _json(_initial_resume_provenance_history(config)),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO matrix_runs (matrix_id, run_id, run_key, ordinal)
                    VALUES (?, ?, ?, ?)
                    """,
                    (matrix_id, run_id, run_key, ordinal),
                )
                payload: dict[str, object] = {
                    "event": "run_created",
                    "status": "queued",
                    "matrix_id": matrix_id,
                    "run_key": run_key,
                }
                payload = _validated_event_payload(payload)
                _insert_event(
                    connection,
                    run_id,
                    created_at=created_at,
                    serialized_payload=_json(payload),
                )
                run_records.append((run_id, event_log_path, payload))
        for _, event_log_path, payload in run_records:
            _write_event_log_record(
                event_log_path,
                sequence=1,
                created_at=created_at,
                payload=payload,
            )
        return [run_id for run_id, _, _ in run_records]

    def list_experiment_matrices(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiment_matrices ORDER BY created_at DESC"
            ).fetchall()
        return [self._matrix_row(row) for row in rows]

    def get_experiment_matrix(self, matrix_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiment_matrices WHERE id = ?",
                (matrix_id,),
            ).fetchone()
        return self._matrix_row(row) if row is not None else None

    def freeze_policy_selection(
        self,
        matrix_id: str,
        run_key: str,
        selection: dict[str, object],
    ) -> None:
        """Persist one immutable validation-only checkpoint selection."""

        serialized = _json(selection)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            matrix_run = connection.execute(
                """
                SELECT 1 FROM matrix_runs
                WHERE matrix_id = ? AND run_key = ?
                """,
                (matrix_id, run_key),
            ).fetchone()
            if matrix_run is None:
                raise KeyError(f"{matrix_id}:{run_key}")
            existing = connection.execute(
                """
                SELECT selection_json FROM policy_selections
                WHERE matrix_id = ? AND run_key = ?
                """,
                (matrix_id, run_key),
            ).fetchone()
            if existing is not None:
                if str(existing["selection_json"]) == serialized:
                    return
                raise ValueError(
                    f"selection is already frozen for {matrix_id}:{run_key}"
                )
            connection.execute(
                """
                INSERT INTO policy_selections (
                    matrix_id, run_key, selection_json, frozen_at
                ) VALUES (?, ?, ?, ?)
                """,
                (matrix_id, run_key, serialized, _now()),
            )

    def list_policy_selections(self, matrix_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_key, selection_json, frozen_at
                FROM policy_selections
                WHERE matrix_id = ?
                ORDER BY run_key
                """,
                (matrix_id,),
            ).fetchall()
        return [
            {
                "run_key": str(row["run_key"]),
                "selection": _object_dict(row["selection_json"]),
                "frozen_at": str(row["frozen_at"]),
            }
            for row in rows
        ]

    def _matrix_row(self, row: sqlite3.Row) -> dict[str, object]:
        matrix_id = str(row["id"])
        with self._connect() as connection:
            run_rows = connection.execute(
                """
                SELECT mr.run_key, mr.ordinal, r.*
                FROM matrix_runs mr
                JOIN runs r ON r.id = mr.run_id
                WHERE mr.matrix_id = ?
                ORDER BY mr.ordinal
                """,
                (matrix_id,),
            ).fetchall()
            selection_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM policy_selections WHERE matrix_id = ?",
                    (matrix_id,),
                ).fetchone()[0]
            )
        runs = [
            {
                **_run_row(run_row),
                "run_key": str(run_row["run_key"]),
                "ordinal": int(run_row["ordinal"]),
            }
            for run_row in run_rows
        ]
        counts: dict[str, int] = {}
        for run in runs:
            status = str(run["status"])
            counts[status] = counts.get(status, 0) + 1
        expected = int(row["expected_run_count"])
        if selection_count == expected:
            status = "selected"
        elif any(item in counts for item in ("failed", "interrupted", "cancelled")):
            status = "attention"
        elif counts.get("completed", 0) == expected:
            status = "validating"
        elif counts.get("running", 0) or counts.get("resuming", 0):
            status = "running"
        else:
            status = "queued"
        return {
            "id": matrix_id,
            "protocol_digest": str(row["protocol_digest"]),
            "protocol": _object_dict(row["protocol_json"]),
            "execution_profile": str(row["execution_profile"]),
            "expected_run_count": expected,
            "selection_count": selection_count,
            "status": status,
            "status_counts": counts,
            "created_at": str(row["created_at"]),
            "runs": runs,
        }

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
                    cancel_requested, event_log_path, config_digest, source_commit,
                    resume_provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
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
                    _json(_initial_resume_provenance_history(config)),
                ),
            )
        self.append_event(run_id, {"event": "run_created", "status": status})
        return run_id

    def create_matrix_evaluation_run(
        self,
        matrix_id: str,
        config: dict[str, object],
        *,
        input_payload: dict[str, object],
    ) -> str:
        """Atomically create the only active or completed evaluation for a matrix."""

        if config.get("matrix_id") != matrix_id:
            raise ValueError("matrix evaluation config does not match its matrix id")
        run_id = (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
            f"matrix_evaluation-{uuid4().hex[:8]}"
        )
        event_log_path = self.path.parent / "events" / f"{run_id}.jsonl"
        created_at = _now()
        event = _validated_event_payload(
            {"event": "run_created", "status": "queued"}
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT id, status, config_json FROM runs
                WHERE kind = 'matrix_evaluation'
                """
            ).fetchall()
            for row in rows:
                existing_config = _object_dict(row["config_json"])
                if (
                    existing_config.get("matrix_id") == matrix_id
                    and str(row["status"])
                    in MATRIX_EVALUATION_DEDUPLICATED_STATUSES
                ):
                    raise ValueError(
                        "matrix evaluation already has an active or completed run: "
                        f"{row['id']}"
                    )
            connection.execute(
                """
                INSERT INTO runs (
                    id, kind, status, config_json, input_json, created_at, progress,
                    cancel_requested, event_log_path
                ) VALUES (?, 'matrix_evaluation', 'queued', ?, ?, ?, 0, 0, ?)
                """,
                (
                    run_id,
                    _json(config),
                    _json(input_payload),
                    created_at,
                    str(event_log_path),
                ),
            )
            sequence = _insert_event(
                connection,
                run_id,
                created_at=created_at,
                serialized_payload=_json(event),
            )
        _write_event_log_record(
            event_log_path,
            sequence=sequence,
            created_at=created_at,
            payload=event,
        )
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

    def claim_next_durable_job(
        self,
        *,
        kinds: Sequence[str] = DURABLE_SUBPROCESS_KINDS,
    ) -> dict[str, object] | None:
        """Atomically claim one queued durable job while its queue slot is free."""

        requested_kinds = tuple(dict.fromkeys(kinds))
        if not requested_kinds or any(
            kind not in DURABLE_SUBPROCESS_KINDS for kind in requested_kinds
        ):
            raise ValueError(f"unsupported durable subprocess kinds: {requested_kinds}")
        placeholders = ", ".join("?" for _ in requested_kinds)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                f"SELECT id FROM runs WHERE kind IN ({placeholders}) "
                "AND status IN ('running', 'cancel_requested') LIMIT 1",
                requested_kinds,
            ).fetchone()
            if active is not None:
                return None
            row = connection.execute(
                f"""
                SELECT id FROM runs
                WHERE kind IN ({placeholders}) AND status IN ('queued', 'resuming')
                ORDER BY created_at, id LIMIT 1
                """,
                requested_kinds,
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

    def claim_next_training(self) -> dict[str, object] | None:
        """Backward-compatible training-only durable claim."""

        return self.claim_next_durable_job(kinds=("training",))

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

    def record_resume_provenance(
        self,
        run_id: str,
        provenance: Mapping[str, object],
        *,
        claim_token: str,
    ) -> list[dict[str, object]]:
        """Append claim-owned resume lineage without mutating the frozen config."""

        normalized = dict(provenance)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT status, claim_token, attempt, latest_checkpoint,
                    resume_provenance_json
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                row is None
                or str(row["claim_token"]) != claim_token
                or str(row["status"]) not in ACTIVE_RUN_STATUSES
            ):
                raise LostRunClaimError(run_id)
            expected = {
                "run_id": run_id,
                "attempt": int(row["attempt"]),
                "checkpoint": str(row["latest_checkpoint"]),
            }
            if normalized != expected or row["latest_checkpoint"] is None:
                raise ValueError(
                    "resume provenance must identify the claimed run, attempt, "
                    "and persisted latest checkpoint"
                )
            history = _resume_provenance_history(row["resume_provenance_json"])
            same_attempt = [
                item
                for item in history
                if item.get("attempt") == normalized["attempt"]
            ]
            if same_attempt and same_attempt != [normalized]:
                raise ValueError(
                    f"resume provenance is already recorded for attempt "
                    f"{normalized['attempt']}"
                )
            if normalized not in history:
                history.append(normalized)
                connection.execute(
                    """
                    UPDATE runs SET resume_provenance_json = ?
                    WHERE id = ? AND claim_token = ?
                        AND status IN ('running', 'cancel_requested')
                    """,
                    (_json(history), run_id, claim_token),
                )
        return history

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
            normalized_event = _validated_event_payload(event)
            serialized_event = _json(normalized_event)
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
            payload=normalized_event,
        )
        return sequence

    def request_cancel(self, run_id: str) -> bool:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None or not run_can_cancel(str(row["status"])):
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
                """
                SELECT kind, status, latest_checkpoint
                FROM runs WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if (
                row is None
                or not run_can_resume(
                    kind=str(row["kind"]),
                    status=str(row["status"]),
                    latest_checkpoint=row["latest_checkpoint"],
                )
            ):
                return False
            restart_from_beginning = row["latest_checkpoint"] is None
            connection.execute(
                """
                UPDATE runs SET status = 'resuming', cancel_requested = 0, error = NULL,
                    ended_at = NULL, pid = NULL, process_identity = NULL,
                    claim_token = NULL, interrupted_at = NULL,
                    progress = CASE WHEN ? THEN 0 ELSE progress END,
                    artifact_dir = CASE WHEN ? THEN NULL ELSE artifact_dir END,
                    started_at = CASE WHEN ? THEN NULL ELSE started_at END
                WHERE id = ?
                """,
                (
                    restart_from_beginning,
                    restart_from_beginning,
                    restart_from_beginning,
                    run_id,
                ),
            )
        event: dict[str, object] = {"event": "resume_requested"}
        if restart_from_beginning:
            event["restart_from_beginning"] = True
        else:
            event["checkpoint"] = str(row["latest_checkpoint"])
        self.append_event(run_id, event)
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
        kinds: Sequence[str] = DURABLE_SUBPROCESS_KINDS,
        stale_after: timedelta = DEFAULT_STALE_WORKER_TIMEOUT,
        process_alive: Callable[[int], bool] | None = None,
        process_identity: Callable[[int], str | None] | None = None,
        terminate_owned_process: Callable[[int, str], bool] | None = None,
        now: datetime | None = None,
    ) -> list[str]:
        """Fence expired leases and release them only after the old worker is gone."""

        requested_kinds = tuple(dict.fromkeys(kinds))
        if not requested_kinds or any(
            kind not in DURABLE_SUBPROCESS_KINDS for kind in requested_kinds
        ):
            raise ValueError(f"unsupported durable subprocess kinds: {requested_kinds}")
        process_alive = process_alive or _process_alive
        process_identity = process_identity or _process_identity
        terminate_owned_process = terminate_owned_process or _terminate_owned_process
        now = now or datetime.now(UTC)
        interrupted: list[str] = []
        placeholders = ", ".join("?" for _ in requested_kinds)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id FROM runs WHERE kind IN ({placeholders}) "
                "AND status IN ('running', 'cancel_requested')",
                requested_kinds,
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
            normalized_payload = _validated_event_payload(payload)
            serialized = _json(normalized_payload)
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
            payload=normalized_payload,
        )
        return sequence

    def list_events(self, run_id: str, *, after: int = 0) -> list[dict[str, object]]:
        return [
            cast(
                dict[str, object],
                item.model_dump(mode="json", exclude_unset=True),
            )
            for item in self.list_event_envelopes(run_id, after=after)
        ]

    def list_event_envelopes(
        self,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[RunEventEnvelope]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, created_at, payload_json FROM run_events
                WHERE run_id = ? AND sequence > ? ORDER BY sequence
                """,
                (run_id, after),
            ).fetchall()
        return [
            RunEventEnvelope.model_validate(
                {
                    "sequence": int(row["sequence"]),
                    "created_at": str(row["created_at"]),
                    "payload": _object_dict(row["payload_json"]),
                },
                strict=True,
            )
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
                    resume_provenance_json TEXT NOT NULL DEFAULT '[]',
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
                CREATE TABLE IF NOT EXISTS experiment_matrices (
                    id TEXT PRIMARY KEY,
                    protocol_digest TEXT NOT NULL,
                    protocol_json TEXT NOT NULL,
                    execution_profile TEXT NOT NULL,
                    expected_run_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(protocol_digest, execution_profile)
                );
                CREATE TABLE IF NOT EXISTS matrix_runs (
                    matrix_id TEXT NOT NULL REFERENCES experiment_matrices(id) ON DELETE CASCADE,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    run_key TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY (matrix_id, run_key),
                    UNIQUE(run_id)
                );
                CREATE TABLE IF NOT EXISTS policy_selections (
                    matrix_id TEXT NOT NULL REFERENCES experiment_matrices(id) ON DELETE CASCADE,
                    run_key TEXT NOT NULL,
                    selection_json TEXT NOT NULL,
                    frozen_at TEXT NOT NULL,
                    PRIMARY KEY (matrix_id, run_key),
                    FOREIGN KEY (matrix_id, run_key)
                        REFERENCES matrix_runs(matrix_id, run_key) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id TEXT PRIMARY KEY,
                    matrix_id TEXT REFERENCES experiment_matrices(id) ON DELETE SET NULL,
                    split TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL
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
                "resume_provenance_json": "TEXT NOT NULL DEFAULT '[]'",
                "claim_token": "TEXT",
                "interrupted_at": "TEXT",
            }
            added_resume_provenance = "resume_provenance_json" not in columns
            for column, declaration in migrations.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {declaration}")
            if added_resume_provenance:
                legacy_rows = connection.execute(
                    "SELECT id, config_json FROM runs"
                ).fetchall()
                for row in legacy_rows:
                    config = _object_dict(row["config_json"])
                    connection.execute(
                        "UPDATE runs SET resume_provenance_json = ? WHERE id = ?",
                        (
                            _json(_initial_resume_provenance_history(config)),
                            str(row["id"]),
                        ),
                    )
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_runs_kind_status_created
                    ON runs(kind, status, created_at);
                CREATE INDEX IF NOT EXISTS idx_runs_heartbeat ON runs(heartbeat_at);
                CREATE INDEX IF NOT EXISTS idx_matrix_runs_matrix_ordinal
                    ON matrix_runs(matrix_id, ordinal);
                CREATE INDEX IF NOT EXISTS idx_evaluations_matrix_created
                    ON evaluations(matrix_id, created_at);
                PRAGMA user_version = 5;
                """
            )


def _run_row(row: sqlite3.Row) -> dict[str, object]:
    status = str(row["status"])
    kind = str(row["kind"])
    latest_checkpoint = row["latest_checkpoint"]
    return {
        "id": str(row["id"]),
        "kind": kind,
        "status": status,
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
        "latest_checkpoint": latest_checkpoint,
        "config_digest": row["config_digest"],
        "source_commit": row["source_commit"],
        "resume_provenance_history": _resume_provenance_history(
            row["resume_provenance_json"]
        ),
        "claim_token": row["claim_token"],
        "interrupted_at": row["interrupted_at"],
        "can_cancel": run_can_cancel(status),
        "can_resume": run_can_resume(
            kind=kind,
            status=status,
            latest_checkpoint=latest_checkpoint,
        ),
    }


def run_can_cancel(status: str) -> bool:
    """Return the exact persisted-state predicate used by the cancel endpoint."""

    return status in CANCELLABLE_RUN_STATUSES


def run_can_resume(
    *,
    kind: str,
    status: str,
    latest_checkpoint: object,
) -> bool:
    """Return the exact durable-state predicate used by the resume endpoint."""

    if (
        kind not in DURABLE_SUBPROCESS_KINDS
        or status not in RESUMABLE_RUN_STATUSES
    ):
        return False
    if latest_checkpoint is None:
        return True
    checkpoint = str(latest_checkpoint)
    return bool(checkpoint) and Path(checkpoint).is_file()


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


def _validated_event_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    return dump_run_event_payload(validate_run_event_payload(payload))


def _initial_resume_provenance_history(
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    value = config.get("resume_provenance")
    if value in (None, {}):
        return []
    if not isinstance(value, dict):
        raise ValueError("resume_provenance must be a JSON object")
    return [cast(dict[str, object], value)]


def _resume_provenance_history(value: object) -> list[dict[str, object]]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list) or not all(
        isinstance(item, dict) for item in parsed
    ):
        raise ValueError("expected resume provenance history to be a list of objects")
    return [cast(dict[str, object], item) for item in parsed]


def _json(value: object) -> str:
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
