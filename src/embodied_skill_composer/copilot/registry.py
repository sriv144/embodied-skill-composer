from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from embodied_skill_composer.copilot.paths import DEFAULT_COPILOT_DIR, DEFAULT_REGISTRY_PATH


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def new_run_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{kind}-{uuid4().hex[:8]}"


@dataclass(frozen=True)
class RunRecord:
    id: str
    kind: str
    run_dir: Path


class CopilotRegistry:
    def __init__(self, db_path: Path = DEFAULT_REGISTRY_PATH, runs_root: Path | None = None) -> None:
        self.db_path = db_path
        self.runs_root = runs_root or db_path.parent / "runs"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def create_run(
        self,
        kind: str,
        prompt: str = "",
        command: list[str] | None = None,
        runtime_profile: str = "",
    ) -> RunRecord:
        run_id = new_run_id(kind)
        run_dir = self.runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        command_text = json.dumps(command or [], indent=2)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, kind, prompt, command, status, started_at, runtime_profile, report_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, kind, prompt, command_text, "running", utc_now(), runtime_profile, ""),
            )
        return RunRecord(id=run_id, kind=kind, run_dir=run_dir)

    def complete_run(
        self,
        run_id: str,
        status: str,
        exit_code: int,
        report_path: Path | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?, exit_code = ?, report_path = ?
                WHERE id = ?
                """,
                (status, utc_now(), exit_code, str(report_path or ""), run_id),
            )

    def add_artifact(self, run_id: str, artifact_type: str, path: Path, description: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO artifacts (run_id, artifact_type, path, description) VALUES (?, ?, ?, ?)",
                (run_id, artifact_type, str(path), description),
            )

    def add_metric(
        self,
        run_id: str,
        policy_name: str,
        success_rate: float,
        mean_return: float,
        mean_beams_installed: float,
        mean_step_count: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics (
                    run_id, policy_name, success_rate, mean_return,
                    mean_beams_installed, mean_step_count
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, policy_name, success_rate, mean_return, mean_beams_installed, mean_step_count),
            )

    def recent_runs(self, limit: int = 10) -> list[dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, kind, status, started_at, ended_at, exit_code, runtime_profile, report_path
                FROM runs
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    prompt TEXT NOT NULL DEFAULT '',
                    command TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    exit_code INTEGER,
                    runtime_profile TEXT NOT NULL DEFAULT '',
                    report_path TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    mean_return REAL NOT NULL,
                    mean_beams_installed REAL NOT NULL,
                    mean_step_count REAL NOT NULL DEFAULT 0,
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    artifact_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(run_id) REFERENCES runs(id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def default_registry() -> CopilotRegistry:
    return CopilotRegistry(DEFAULT_REGISTRY_PATH, DEFAULT_COPILOT_DIR / "runs")

