from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event

import pytest

from embodied_skill_composer.construction import lab_service as lab_service_module
from embodied_skill_composer.construction.lab_registry import (
    LabRegistry,
    LostRunClaimError,
    RunStatus,
    _process_alive,
    _process_identity,
    _terminate_owned_process,
)
from embodied_skill_composer.construction.lab_service import LabService


def test_process_identity_and_owned_termination_are_pid_reuse_safe() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        identity = _process_identity(process.pid)
        assert identity is not None
        assert _process_alive(process.pid) is True
        assert process.poll() is None
        assert _terminate_owned_process(process.pid, "different-process") is True
        assert process.poll() is None
        assert _terminate_owned_process(process.pid, identity) is True
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_worker_mutations_are_fenced_by_claim_token(tmp_path: Path) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_id = registry.create_run("training", {"seed": 7})
    claimed = registry.claim_next_training()
    assert claimed is not None
    token = str(claimed["claim_token"])

    with pytest.raises(LostRunClaimError):
        registry.update_run(run_id, heartbeat=True, claim_token="stale-token")
    with pytest.raises(LostRunClaimError):
        registry.append_event(
            run_id,
            {"event": "stale_progress"},
            claim_token="stale-token",
        )

    registry.update_run(run_id, heartbeat=True, claim_token=token)
    registry.append_event(run_id, {"event": "owned_progress"}, claim_token=token)
    assert registry.list_events(run_id)[-1]["payload"]["event"] == "owned_progress"


def test_atomic_finalization_fences_policy_event_and_terminal_state(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_id = registry.create_run("training", {"seed": 7})
    claimed = registry.claim_next_training()
    assert claimed is not None
    token = str(claimed["claim_token"])
    manifest: dict[str, object] = {
        "policy_id": "owned-policy",
        "controller": "mappo",
    }
    event_count = len(registry.list_events(run_id))

    with pytest.raises(LostRunClaimError):
        registry.finalize_run(
            run_id,
            status="completed",
            event={"event": "training_completed"},
            claim_token="stale-token",
            progress=1.0,
            artifact_dir=str(tmp_path / "artifacts"),
            policy=("owned-policy", "mappo", manifest),
        )

    active = registry.get_run(run_id)
    assert active is not None and active["status"] == "running"
    assert len(registry.list_events(run_id)) == event_count
    assert registry.list_policies() == []

    registry.finalize_run(
        run_id,
        status="completed",
        event={"event": "training_completed"},
        claim_token=token,
        progress=1.0,
        artifact_dir=str(tmp_path / "artifacts"),
        policy=("owned-policy", "mappo", manifest),
    )

    completed = registry.get_run(run_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["progress"] == 1.0
    assert completed["claim_token"] is None
    assert completed["pid"] is None
    assert completed["process_identity"] is None
    assert registry.list_policies()[0]["id"] == "owned-policy"
    assert len(registry.list_events(run_id)) == event_count + 1
    assert registry.list_events(run_id)[-1]["payload"]["event"] == "training_completed"


def test_registry_migrates_legacy_runs_schema_in_place(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
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
            INSERT INTO runs (
                id, kind, status, config_json, created_at, progress, cancel_requested
            ) VALUES (
                'legacy-run', 'training', 'queued', '{"seed":7}',
                '2026-01-01T00:00:00+00:00', 0.25, 0
            );
            """
        )

    registry = LabRegistry(path)
    migrated = registry.get_run("legacy-run")

    assert migrated is not None
    assert migrated["config"] == {"seed": 7}
    assert migrated["input"] == {}
    assert migrated["attempt"] == 0
    assert migrated["claim_token"] is None
    assert migrated["process_identity"] is None
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        user_version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert {
        "input_json",
        "pid",
        "process_identity",
        "attempt",
        "heartbeat_at",
        "event_log_path",
        "latest_checkpoint",
        "config_digest",
        "source_commit",
        "claim_token",
        "interrupted_at",
    } <= columns
    assert user_version == 2


def test_training_claim_is_atomic_fifo_and_single_slot(tmp_path: Path) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    first_id = registry.create_run("training", {"seed": 7}, run_id="z-first")
    second_id = registry.create_run("training", {"seed": 8}, run_id="a-second")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET created_at = ? WHERE id = ?",
            ("2026-01-01T00:00:00+00:00", first_id),
        )
        connection.execute(
            "UPDATE runs SET created_at = ? WHERE id = ?",
            ("2026-01-01T00:00:01+00:00", second_id),
        )

    contenders = (LabRegistry(path), LabRegistry(path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: item.claim_next_training(), contenders))

    claims = [result for result in results if result is not None]
    assert len(claims) == 1
    assert claims[0]["id"] == first_id
    token = str(claims[0]["claim_token"])
    assert token
    assert registry.verify_claim(first_id, token)
    assert not registry.verify_claim(first_id, "stale-token")
    assert registry.claim_next_training() is None

    registry.update_run(first_id, status="completed")
    second_claim = registry.claim_next_training()
    assert second_claim is not None
    assert second_claim["id"] == second_id
    assert second_claim["attempt"] == 1


def test_queued_cancellation_is_terminal_and_never_claimed(tmp_path: Path) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_id = registry.create_run("training", {"seed": 7})

    assert registry.request_cancel(run_id)
    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == "cancelled"
    assert run["cancel_requested"] is True
    assert run["ended_at"] is not None
    assert registry.request_cancel(run_id) is False
    assert registry.claim_next_training() is None
    assert registry.list_events(run_id)[-1]["payload"] == {"event": "cancelled"}


def test_stale_reconciliation_terminates_owned_worker_and_ignores_reused_pid(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    dead_id = registry.create_run("training", {}, status="running", run_id="dead")
    owned_id = registry.create_run("training", {}, status="running", run_id="owned")
    reused_id = registry.create_run("training", {}, status="running", run_id="reused")
    old_heartbeat = "2026-01-01T00:00:00+00:00"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET pid = 111, process_identity = ?, heartbeat_at = ? WHERE id = ?",
            ("dead-identity", old_heartbeat, dead_id),
        )
        connection.execute(
            "UPDATE runs SET pid = 222, process_identity = ?, heartbeat_at = ? WHERE id = ?",
            ("owned-identity", old_heartbeat, owned_id),
        )
        connection.execute(
            "UPDATE runs SET pid = 333, process_identity = ?, heartbeat_at = ? WHERE id = ?",
            ("old-identity", old_heartbeat, reused_id),
        )

    terminated: list[tuple[int, str]] = []

    def terminate_owned(pid: int, identity: str) -> bool:
        terminated.append((pid, identity))
        return True

    interrupted = registry.reconcile_stale_runs(
        stale_after=timedelta(seconds=30),
        process_alive=lambda pid: pid in {222, 333},
        process_identity=lambda pid: {
            222: "owned-identity",
            333: "new-identity",
        }.get(pid),
        terminate_owned_process=terminate_owned,
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert set(interrupted) == {dead_id, owned_id, reused_id}
    assert terminated == [(222, "owned-identity")]
    for run_id in interrupted:
        run = registry.get_run(run_id)
        assert run is not None and run["status"] == "interrupted"
        assert run["pid"] is None
        assert run["process_identity"] is None
        assert run["claim_token"] is None
        assert run["interrupted_at"] is not None
        assert run["ended_at"] is not None
        assert run["error"] == "worker heartbeat expired"
        assert registry.list_events(run_id)[-1]["payload"]["event"] == "run_interrupted"
    assert registry.list_events(reused_id)[-1]["payload"]["reason"] == (
        "stale_worker_pid_reused"
    )


def test_stale_reconciliation_keeps_slot_fenced_until_termination_is_verified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    run_id = registry.create_run("training", {"seed": 7})
    claimed = registry.claim_next_training()
    assert claimed is not None
    original_token = str(claimed["claim_token"])
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE runs SET pid = 444, process_identity = ?, heartbeat_at = ? WHERE id = ?",
            ("owned-identity", "2026-01-01T00:00:00+00:00", run_id),
        )

    interrupted = registry.reconcile_stale_runs(
        stale_after=timedelta(seconds=30),
        process_alive=lambda _pid: True,
        process_identity=lambda _pid: "owned-identity",
        terminate_owned_process=lambda _pid, _identity: False,
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert interrupted == []
    fenced = registry.get_run(run_id)
    assert fenced is not None and fenced["status"] == "running"
    assert isinstance(fenced["claim_token"], str)
    assert str(fenced["claim_token"]).startswith("reaper:")
    assert fenced["claim_token"] != original_token
    assert not registry.verify_claim(run_id, original_token)
    assert registry.claim_next_training() is None

    assert registry.reconcile_stale_runs(
        stale_after=timedelta(seconds=30),
        process_alive=lambda _pid: False,
        now=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
    ) == [run_id]
    released = registry.get_run(run_id)
    assert released is not None and released["status"] == "interrupted"


def test_events_are_persisted_to_database_and_jsonl_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    run_id = registry.create_run("training", {"seed": 7}, run_id="jsonl-run")
    assert registry.append_event(run_id, {"event": "progress", "transitions": 10}) == 2

    reopened = LabRegistry(path)
    assert reopened.append_event(run_id, {"event": "checkpoint", "fraction": 0.25}) == 3
    run = reopened.get_run(run_id)
    assert run is not None
    event_path = Path(str(run["event_log_path"]))
    records = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]

    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert [record["payload"]["event"] for record in records] == [
        "run_created",
        "progress",
        "checkpoint",
    ]
    assert [event["payload"] for event in reopened.list_events(run_id)] == [
        record["payload"] for record in records
    ]


@pytest.mark.parametrize("status", ["interrupted", "failed", "cancelled"])
def test_resume_requires_resumable_status_and_existing_checkpoint(
    tmp_path: Path,
    status: RunStatus,
) -> None:
    registry = LabRegistry(tmp_path / f"{status}.sqlite")
    checkpoint = tmp_path / f"{status}.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_id = registry.create_run("training", {}, status=status, run_id=status)
    registry.update_run(run_id, latest_checkpoint=str(checkpoint))

    assert registry.request_resume(run_id)
    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == "resuming"
    assert run["cancel_requested"] is False
    assert registry.request_resume(run_id) is False
    assert registry.list_events(run_id)[-1]["payload"]["checkpoint"] == str(checkpoint)


def test_resume_rejects_missing_checkpoint_and_completed_run(tmp_path: Path) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    missing_id = registry.create_run("training", {}, status="interrupted", run_id="missing")
    registry.update_run(missing_id, latest_checkpoint=str(tmp_path / "missing.pt"))
    completed_checkpoint = tmp_path / "completed.pt"
    completed_checkpoint.write_bytes(b"checkpoint")
    completed_id = registry.create_run("training", {}, status="completed", run_id="completed")
    registry.update_run(completed_id, latest_checkpoint=str(completed_checkpoint))

    assert registry.request_resume(missing_id) is False
    assert registry.request_resume(completed_id) is False
    missing = registry.get_run(missing_id)
    completed = registry.get_run(completed_id)
    assert missing is not None and missing["status"] == "interrupted"
    assert completed is not None and completed["status"] == "completed"


def test_dispatcher_builds_claim_bound_worker_command_without_running_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_id = registry.create_run("training", {"seed": 7}, run_id="worker-command")
    launched = Event()
    captured: dict[str, object] = {}

    class FinishedProcess:
        pid = 4242
        returncode = 0

        @staticmethod
        def poll() -> int:
            return 0

    def fake_popen(command: list[str], **kwargs: object) -> FinishedProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        launched.set()
        return FinishedProcess()

    monkeypatch.setattr(lab_service_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lab_service_module,
        "_process_identity",
        lambda _pid: "fixture-process-identity",
    )
    service = LabService(registry)
    try:
        assert launched.wait(timeout=2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            run = registry.get_run(run_id)
            if run is not None and run["status"] == "failed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("dispatcher did not reconcile the finished worker")
    finally:
        service.shutdown()

    run = registry.get_run(run_id)
    assert run is not None
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [
        sys.executable,
        "-m",
        "embodied_skill_composer.construction.lab_worker",
    ]
    assert command[command.index("--registry") + 1] == str(registry.path)
    assert command[command.index("--run-id") + 1] == run_id
    assert isinstance(command[command.index("--claim-token") + 1], str)
    assert command[command.index("--claim-token") + 1]
    assert run["attempt"] == 1
    assert run["pid"] is None
    assert run["process_identity"] is None
    assert run["claim_token"] is None
    assert run["status"] == "failed"
    assert "exited with code 0" in str(run["error"])
    events = registry.list_events(run_id)
    assert any(event["payload"]["event"] == "worker_started" for event in events)
    assert any(event["payload"]["event"] == "worker_exited" for event in events)


def test_dispatcher_reconciles_heartbeats_while_worker_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_id = registry.create_run("training", {"seed": 7}, run_id="periodic-reconcile")
    reconcile_calls = 0

    class RunningProcess:
        pid = 5252
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    process = RunningProcess()

    def fake_popen(_command: list[str], **_kwargs: object) -> RunningProcess:
        return process

    def fake_reconcile() -> list[str]:
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls >= 2:
            process.returncode = 1
        return []

    monkeypatch.setattr(lab_service_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        lab_service_module,
        "_process_identity",
        lambda _pid: "fixture-process-identity",
    )
    monkeypatch.setattr(registry, "reconcile_stale_runs", fake_reconcile)
    service = LabService(registry)
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            run = registry.get_run(run_id)
            if run is not None and run["status"] == "failed":
                break
            time.sleep(0.01)
        else:
            pytest.fail("dispatcher did not reconcile while waiting for its worker")
    finally:
        service.shutdown()

    assert reconcile_calls >= 2
