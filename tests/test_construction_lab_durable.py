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
from pydantic import ValidationError

from embodied_skill_composer.construction import lab_registry as lab_registry_module
from embodied_skill_composer.construction import lab_service as lab_service_module
from embodied_skill_composer.construction.experiment_protocol import (
    load_experiment_protocol,
)
from embodied_skill_composer.construction.lab_registry import (
    LabRegistry,
    LostRunClaimError,
    RunStatus,
    _process_alive,
    _process_identity,
    _terminate_owned_process,
)
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.runtime import load_house_design


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


def test_windows_process_lease_helpers_are_platform_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ctypes
    from types import SimpleNamespace

    class KernelCall:
        def __init__(self, implementation: object) -> None:
            self.implementation = implementation
            self.argtypes: object = None
            self.restype: object = None

        def __call__(self, *args: object) -> object:
            assert callable(self.implementation)
            return self.implementation(*args)

    class Kernel32:
        def __init__(self) -> None:
            self.closed_handles = 0
            self.terminated_handles = 0
            self.open_process_result = 77
            self.process_times_succeed = True
            self.exit_code_succeeds = True
            self.exit_code_value = 259
            self.terminate_succeeds = True
            self.OpenProcess = KernelCall(self.open_process)
            self.GetProcessTimes = KernelCall(self.get_process_times)
            self.GetExitCodeProcess = KernelCall(self.get_exit_code_process)
            self.TerminateProcess = KernelCall(self.terminate_process)
            self.WaitForSingleObject = KernelCall(lambda *_args: 0)
            self.CloseHandle = KernelCall(self.close_handle)

        def open_process(self, *_args: object) -> int:
            return self.open_process_result

        def get_process_times(
            self,
            _handle: object,
            creation: object,
            _exit_time: object,
            _kernel_time: object,
            _user_time: object,
        ) -> int:
            if not self.process_times_succeed:
                return 0
            creation_time = getattr(creation, "_obj")
            creation_time.low = 7
            creation_time.high = 1
            return 1

        def get_exit_code_process(
            self,
            _handle: object,
            exit_code: object,
        ) -> int:
            getattr(exit_code, "_obj").value = self.exit_code_value
            return int(self.exit_code_succeeds)

        def terminate_process(self, _handle: object, _exit_code: object) -> int:
            self.terminated_handles += 1
            return int(self.terminate_succeeds)

        def close_handle(self, _handle: object) -> int:
            self.closed_handles += 1
            return 1

    kernel32 = Kernel32()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: kernel32, raising=False)
    monkeypatch.setattr(lab_registry_module, "os", SimpleNamespace(name="nt"))

    expected_identity = f"windows:{(1 << 32) | 7}"
    assert lab_registry_module._windows_process_identity(42) == expected_identity
    assert lab_registry_module._process_identity(42) == expected_identity
    assert lab_registry_module._process_alive(42) is True
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is True
    )
    assert (
        lab_registry_module._terminate_owned_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is True
    )
    assert kernel32.terminated_handles == 2
    assert kernel32.closed_handles == 6

    kernel32.open_process_result = 0
    assert lab_registry_module._windows_process_identity(42) is None
    assert lab_registry_module._process_alive(42) is False
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is True
    )
    assert (
        lab_registry_module._terminate_owned_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is True
    )

    kernel32.open_process_result = 77
    kernel32.process_times_succeed = False
    assert lab_registry_module._windows_process_identity(42) is None
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is False
    )

    kernel32.process_times_succeed = True
    kernel32.exit_code_succeeds = False
    assert lab_registry_module._process_alive(42) is False

    kernel32.exit_code_succeeds = True
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            "windows:different-process",
            timeout_seconds=0.01,
        )
        is True
    )
    kernel32.exit_code_value = 0
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is True
    )
    kernel32.exit_code_value = 259
    kernel32.terminate_succeeds = False
    assert (
        lab_registry_module._terminate_owned_windows_process(
            42,
            expected_identity,
            timeout_seconds=0.01,
        )
        is False
    )


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
    registry.append_event(run_id, {"event": "evaluation_started"}, claim_token=token)
    assert registry.list_events(run_id)[-1]["payload"]["event"] == "evaluation_started"


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
            event={"event": "training_completed", "artifacts": {}},
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
        event={"event": "training_completed", "artifacts": {}},
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
                'legacy-run', 'training', 'queued',
                '{"seed":7,"resume_provenance":{"legacy":true}}',
                '2026-01-01T00:00:00+00:00', 0.25, 0
            );
            """
        )

    registry = LabRegistry(path)
    migrated = registry.get_run("legacy-run")

    assert migrated is not None
    assert migrated["config"] == {
        "seed": 7,
        "resume_provenance": {"legacy": True},
    }
    assert migrated["input"] == {}
    assert migrated["attempt"] == 0
    assert migrated["claim_token"] is None
    assert migrated["process_identity"] is None
    assert migrated["resume_provenance_history"] == [{"legacy": True}]
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
        "resume_provenance_json",
        "claim_token",
        "interrupted_at",
    } <= columns
    assert user_version == 5


def test_resume_provenance_is_claim_owned_persistent_and_append_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint")
    run_id = registry.create_run(
        "training",
        {"seed": 7, "resume_provenance": {}},
        config_digest="immutable-config-digest",
        run_id="resumable-run",
    )
    registry.update_run(run_id, latest_checkpoint=str(checkpoint))

    first_claim = registry.claim_next_training()
    assert first_claim is not None
    first_token = str(first_claim["claim_token"])
    first_provenance = {
        "run_id": run_id,
        "attempt": 1,
        "checkpoint": str(checkpoint),
    }
    assert registry.record_resume_provenance(
        run_id,
        first_provenance,
        claim_token=first_token,
    ) == [first_provenance]
    assert registry.record_resume_provenance(
        run_id,
        first_provenance,
        claim_token=first_token,
    ) == [first_provenance]
    with pytest.raises(LostRunClaimError):
        registry.record_resume_provenance(
            run_id,
            first_provenance,
            claim_token="stale-token",
        )
    registry.finalize_run(
        run_id,
        status="interrupted",
        event={"event": "run_interrupted", "reason": "stale_worker"},
        claim_token=first_token,
    )

    assert registry.request_resume(run_id)
    second_claim = registry.claim_next_training()
    assert second_claim is not None
    second_token = str(second_claim["claim_token"])
    second_provenance = {
        "run_id": run_id,
        "attempt": 2,
        "checkpoint": str(checkpoint),
    }
    assert registry.record_resume_provenance(
        run_id,
        second_provenance,
        claim_token=second_token,
    ) == [first_provenance, second_provenance]

    reopened = LabRegistry(path).get_run(run_id)
    assert reopened is not None
    assert reopened["config"] == {"seed": 7, "resume_provenance": {}}
    assert reopened["config_digest"] == "immutable-config-digest"
    assert reopened["resume_provenance_history"] == [
        first_provenance,
        second_provenance,
    ]


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
    queued = registry.get_run(run_id)
    assert queued is not None
    assert queued["can_cancel"] is True
    assert queued["can_resume"] is False

    assert registry.request_cancel(run_id)
    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == "cancelled"
    assert run["cancel_requested"] is True
    assert run["ended_at"] is not None
    assert run["can_cancel"] is False
    assert run["can_resume"] is True
    assert registry.request_cancel(run_id) is False
    assert registry.claim_next_training() is None
    assert registry.list_events(run_id)[-1]["payload"] == {"event": "cancelled"}


def test_running_cancel_request_is_idempotently_rejected_after_first_request(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "running-cancel.sqlite")
    run_id = registry.create_run(
        "training",
        {"seed": 7},
        status="running",
    )

    assert registry.request_cancel(run_id) is True
    requested = registry.get_run(run_id)
    assert requested is not None
    assert requested["status"] == "cancel_requested"
    assert requested["can_cancel"] is False
    assert registry.request_cancel(run_id) is False


def test_resume_capability_is_limited_to_durable_runs_with_compatible_state(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "resume-capability.sqlite")
    restartable = registry.create_run(
        "training",
        {},
        status="interrupted",
        run_id="restartable",
    )
    nondurable = registry.create_run(
        "evaluation",
        {},
        status="interrupted",
        run_id="nondurable",
    )
    missing = registry.create_run(
        "training",
        {},
        status="interrupted",
        run_id="missing-checkpoint",
    )
    registry.update_run(missing, latest_checkpoint=str(tmp_path / "missing.pt"))

    restartable_record = registry.get_run(restartable)
    nondurable_record = registry.get_run(nondurable)
    missing_record = registry.get_run(missing)
    assert restartable_record is not None
    assert nondurable_record is not None
    assert missing_record is not None
    assert restartable_record["can_resume"] is True
    assert nondurable_record["can_resume"] is False
    assert missing_record["can_resume"] is False
    assert registry.request_resume(nondurable) is False


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
    assert registry.append_event(run_id, {"event": "evaluation_started"}) == 2

    reopened = LabRegistry(path)
    assert reopened.append_event(
        run_id,
        {"event": "evaluation_completed", "artifacts": {"fraction": 0.25}},
    ) == 3
    run = reopened.get_run(run_id)
    assert run is not None
    event_path = Path(str(run["event_log_path"]))
    records = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]

    assert [record["sequence"] for record in records] == [1, 2, 3]
    assert [record["payload"]["event"] for record in records] == [
        "run_created",
        "evaluation_started",
        "evaluation_completed",
    ]
    assert [event["payload"] for event in reopened.list_events(run_id)] == [
        record["payload"] for record in records
    ]


def test_invalid_run_events_are_rejected_on_append_and_read(tmp_path: Path) -> None:
    path = tmp_path / "invalid-events.sqlite"
    registry = LabRegistry(path)
    run_id = registry.create_run("training", {"seed": 7}, run_id="typed-events")

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        registry.append_event(run_id, {"event": "invented_progress"})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        registry.append_event(
            run_id,
            {"event": "evaluation_started", "transitions": 10},
        )
    assert len(registry.list_events(run_id)) == 1

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO run_events (
                run_id, sequence, created_at, payload_json
            ) VALUES (?, 2, ?, ?)
            """,
            (
                run_id,
                datetime.now(UTC).isoformat(),
                json.dumps({"event": "legacy_unknown"}),
            ),
        )
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        registry.list_event_envelopes(run_id)


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


@pytest.mark.parametrize("status", ["interrupted", "failed", "cancelled"])
def test_resume_restarts_training_before_first_checkpoint(
    tmp_path: Path,
    status: RunStatus,
) -> None:
    registry = LabRegistry(tmp_path / f"restart-{status}.sqlite")
    run_id = registry.create_run("training", {}, status=status, run_id=status)
    registry.update_run(run_id, progress=0.08, artifact_dir=str(tmp_path / "partial"))

    assert registry.request_resume(run_id) is True

    restarted = registry.get_run(run_id)
    assert restarted is not None
    assert restarted["status"] == "resuming"
    assert restarted["progress"] == 0
    assert restarted["artifact_dir"] is None
    assert restarted["started_at"] is None
    assert registry.list_events(run_id)[-1]["payload"] == {
        "event": "resume_requested",
        "restart_from_beginning": True,
    }


@pytest.mark.parametrize("status", ["interrupted", "failed", "cancelled"])
def test_restart_requeues_only_zero_progress_training_without_checkpoint(
    tmp_path: Path,
    status: RunStatus,
) -> None:
    registry = LabRegistry(tmp_path / f"safe-restart-{status}.sqlite")
    run_id = registry.create_run("training", {}, run_id=status)
    claimed = registry.claim_next_training()
    assert claimed is not None
    claim_token = str(claimed["claim_token"])
    terminal_event: dict[str, object]
    if status == "failed":
        terminal_event = {"event": "failed", "error": "preflight failed"}
    elif status == "cancelled":
        terminal_event = {"event": "cancelled", "error": "preflight cancelled"}
    else:
        terminal_event = {"event": "run_interrupted", "reason": "stale_worker"}
    registry.finalize_run(
        run_id,
        status=status,
        event=terminal_event,
        claim_token=claim_token,
        progress=0,
        artifact_dir=str(tmp_path / "partial"),
        error="operational preflight failure",
    )

    assert registry.request_restart(run_id) is True

    restarted = registry.get_run(run_id)
    assert restarted is not None
    assert restarted["status"] == "resuming"
    assert restarted["attempt"] == 1
    assert restarted["progress"] == 0
    assert restarted["artifact_dir"] is None
    assert restarted["error"] is None
    assert restarted["cancel_requested"] is False
    assert restarted["started_at"] is None
    assert restarted["ended_at"] is None
    assert restarted["interrupted_at"] is None
    assert restarted["heartbeat_at"] is None
    assert restarted["pid"] is None
    assert restarted["process_identity"] is None
    assert restarted["claim_token"] is None
    assert registry.list_events(run_id)[-1]["payload"] == {
        "event": "restart_requested",
        "previous_status": status,
        "restart_from_beginning": True,
    }
    assert registry.request_restart(run_id) is False

    second_claim = registry.claim_next_training()
    assert second_claim is not None
    assert second_claim["id"] == run_id
    assert second_claim["attempt"] == 2


def test_restart_rejects_nontraining_progress_checkpoint_and_active_status(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "restart-rejections.sqlite")
    wrong_kind = registry.create_run("matrix_evaluation", {}, status="failed")
    progressed = registry.create_run("training", {}, status="failed")
    registry.update_run(progressed, progress=0.01)
    checkpointed = registry.create_run("training", {}, status="failed")
    registry.update_run(checkpointed, latest_checkpoint=str(tmp_path / "snapshot.pt"))
    completed = registry.create_run("training", {}, status="completed")

    for run_id in (wrong_kind, progressed, checkpointed, completed, "missing"):
        before = registry.list_events(run_id) if run_id != "missing" else []
        assert registry.request_restart(run_id) is False
        if run_id != "missing":
            after = registry.list_events(run_id)
            assert after == before


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("claim_token", "live-claim"),
        ("pid", 4242),
        ("process_identity", "live-process"),
    ],
)
def test_restart_rejects_any_inconsistent_live_ownership(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    path = tmp_path / f"restart-live-{column}.sqlite"
    registry = LabRegistry(path)
    run_id = registry.create_run("training", {}, status="failed")
    with sqlite3.connect(path) as connection:
        connection.execute(
            f"UPDATE runs SET {column} = ? WHERE id = ?",
            (value, run_id),
        )

    assert registry.request_restart(run_id) is False
    rejected = registry.get_run(run_id)
    assert rejected is not None
    assert rejected["status"] == "failed"
    assert registry.list_events(run_id)[-1]["payload"]["event"] == "run_created"


def test_restart_is_atomic_under_concurrent_requests(tmp_path: Path) -> None:
    path = tmp_path / "restart-concurrent.sqlite"
    registry = LabRegistry(path)
    run_id = registry.create_run("training", {}, status="failed")

    contenders = (LabRegistry(path), LabRegistry(path))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda item: item.request_restart(run_id), contenders))

    assert sorted(results) == [False, True]
    restart_events = [
        item
        for item in registry.list_events(run_id)
        if item["payload"]["event"] == "restart_requested"
    ]
    assert len(restart_events) == 1


def test_restart_rolls_back_state_when_event_insert_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "restart-rollback.sqlite")
    run_id = registry.create_run("training", {}, status="failed")

    def fail_event_insert(*args: object, **kwargs: object) -> int:
        raise RuntimeError("synthetic event failure")

    monkeypatch.setattr(lab_registry_module, "_insert_event", fail_event_insert)
    with pytest.raises(RuntimeError, match="synthetic event failure"):
        registry.request_restart(run_id)

    rolled_back = registry.get_run(run_id)
    assert rolled_back is not None
    assert rolled_back["status"] == "failed"
    assert registry.list_events(run_id)[-1]["payload"]["event"] == "run_created"


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


def test_experiment_matrix_is_atomic_unique_and_tracks_frozen_selections(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "matrix.sqlite")
    runs = [
        (
            f"mappo-full-s{seed}",
            {"algorithm": "mappo", "seed": seed},
            f"digest-{seed}",
            "source-sha",
        )
        for seed in (7, 8)
    ]

    run_ids = registry.create_experiment_matrix(
        "matrix-fixture",
        protocol_digest="protocol-digest",
        protocol={"experiment_id": "construction_intelligence_v1"},
        execution_profile="unit",
        design={"design_id": "cottage_v1"},
        runs=runs,
    )

    assert run_ids == [
        "matrix-fixture-mappo-full-s7",
        "matrix-fixture-mappo-full-s8",
    ]
    matrix = registry.get_experiment_matrix("matrix-fixture")
    assert matrix is not None
    assert matrix["status"] == "queued"
    assert matrix["expected_run_count"] == 2
    assert [item["run_key"] for item in matrix["runs"]] == [
        "mappo-full-s7",
        "mappo-full-s8",
    ]
    assert all(
        registry.list_events(run_id)[0]["payload"]["matrix_id"] == "matrix-fixture"
        for run_id in run_ids
    )
    with pytest.raises(ValueError, match="already exists"):
        registry.create_experiment_matrix(
            "duplicate",
            protocol_digest="protocol-digest",
            protocol={"experiment_id": "construction_intelligence_v1"},
            execution_profile="unit",
            design={"design_id": "cottage_v1"},
            runs=runs,
        )

    selection = {
        "checkpoint_sha256": "checkpoint-sha",
        "validation_seeds": [800, 801, 802, 803, 804],
    }
    registry.freeze_policy_selection(
        "matrix-fixture",
        "mappo-full-s7",
        selection,
    )
    registry.freeze_policy_selection(
        "matrix-fixture",
        "mappo-full-s7",
        selection,
    )
    with pytest.raises(ValueError, match="already frozen"):
        registry.freeze_policy_selection(
            "matrix-fixture",
            "mappo-full-s7",
            {**selection, "checkpoint_sha256": "different"},
        )
    with pytest.raises(KeyError):
        registry.freeze_policy_selection(
            "matrix-fixture",
            "unknown",
            selection,
        )
    frozen = registry.list_policy_selections("matrix-fixture")
    assert frozen[0]["run_key"] == "mappo-full-s7"
    assert frozen[0]["selection"] == selection

    registry.upsert_evaluation(
        "evaluation-fixture",
        matrix_id="matrix-fixture",
        split="test",
        payload={"episode_count": 24},
        artifact_dir=str(tmp_path / "evaluation"),
    )
    evaluations = registry.list_evaluations(matrix_id="matrix-fixture")
    assert evaluations[0]["id"] == "evaluation-fixture"
    assert evaluations[0]["payload"] == {"episode_count": 24}


def test_matrix_evaluation_launch_is_atomic_deduplicated_and_relaunchable(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "matrix-evaluation.sqlite")
    matrix_id = "matrix-fixture"
    config: dict[str, object] = {
        "matrix_id": matrix_id,
        "protocol_digest": "protocol-digest",
        "output_root": str(tmp_path / "evidence"),
        "device": "cpu",
        "selection_evidence_path": None,
    }
    input_payload = {"design": {"design_id": "cottage_v1"}}

    def create() -> str:
        try:
            return registry.create_matrix_evaluation_run(
                matrix_id,
                config,
                input_payload=input_payload,
            )
        except ValueError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as executor:
        launched = list(executor.map(lambda _index: create(), range(2)))
    run_ids = [item for item in launched if item != "duplicate"]
    assert len(run_ids) == 1
    assert launched.count("duplicate") == 1

    first_id = run_ids[0]
    claimed = registry.claim_next_durable_job(kinds=("matrix_evaluation",))
    assert claimed is not None and claimed["id"] == first_id
    claim_token = str(claimed["claim_token"])
    assert registry.request_cancel(first_id) is True
    cancelling = registry.get_run(first_id)
    assert cancelling is not None and cancelling["status"] == "cancel_requested"
    registry.finalize_run(
        first_id,
        status="cancelled",
        event={"event": "cancelled"},
        claim_token=claim_token,
    )

    replacement_id = registry.create_matrix_evaluation_run(
        matrix_id,
        config,
        input_payload=input_payload,
    )
    replacement = registry.claim_next_durable_job(kinds=("matrix_evaluation",))
    assert replacement is not None and replacement["id"] == replacement_id
    registry.finalize_run(
        replacement_id,
        status="completed",
        event={
            "event": "matrix_evaluation_completed",
            "matrix_id": matrix_id,
            "evaluation_id": "fixture-evaluation",
            "acceptance": {"passed": True},
            "ablations": [],
        },
        progress=1.0,
        claim_token=str(replacement["claim_token"]),
    )
    with pytest.raises(ValueError, match="active or completed"):
        registry.create_matrix_evaluation_run(
            matrix_id,
            config,
            input_payload=input_payload,
        )


def test_matrix_evaluation_stale_worker_reconciles_and_restarts_from_beginning(
    tmp_path: Path,
) -> None:
    registry = LabRegistry(tmp_path / "matrix-restart.sqlite")
    run_id = registry.create_matrix_evaluation_run(
        "matrix-fixture",
        {
            "matrix_id": "matrix-fixture",
            "protocol_digest": "protocol-digest",
            "output_root": str(tmp_path / "evidence"),
            "device": "cpu",
        },
        input_payload={"design": {"design_id": "cottage_v1"}},
    )
    claimed = registry.claim_next_durable_job(kinds=("matrix_evaluation",))
    assert claimed is not None and claimed["id"] == run_id

    interrupted = registry.reconcile_stale_runs(
        kinds=("matrix_evaluation",),
        stale_after=timedelta(seconds=1),
        process_alive=lambda _pid: False,
        now=datetime.now(UTC) + timedelta(seconds=2),
    )
    assert interrupted == [run_id]
    stale = registry.get_run(run_id)
    assert stale is not None and stale["status"] == "interrupted"
    assert registry.request_resume(run_id) is True
    resuming = registry.get_run(run_id)
    assert resuming is not None and resuming["status"] == "resuming"
    assert registry.list_events(run_id)[-1]["payload"] == {
        "event": "resume_requested",
        "restart_from_beginning": True,
    }
    reclaimed = registry.claim_next_durable_job(kinds=("matrix_evaluation",))
    assert reclaimed is not None and reclaimed["id"] == run_id
    assert reclaimed["attempt"] == 2


def test_matrix_evaluation_dispatches_through_claim_bound_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "matrix-dispatch.sqlite")
    launched = Event()
    captured: dict[str, object] = {}

    class FinishedProcess:
        pid = 6262
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
        lambda _pid: "fixture-matrix-process",
    )
    design = load_house_design(
        Path("configs/construction/cottage_v1.yaml").resolve()
    )
    protocol = load_experiment_protocol()
    service = LabService(registry)
    try:
        run_id = service.launch_matrix_evaluation(
            design,
            matrix_id="matrix-subprocess",
            protocol=protocol,
            output_root=tmp_path / "evaluation",
            selection_evidence_path=tmp_path / "matrix_selections.json",
        )
        assert launched.wait(timeout=2)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            run = registry.get_run(run_id)
            if run is not None and run["status"] == "interrupted":
                break
            time.sleep(0.01)
        else:
            pytest.fail("matrix dispatcher did not reconcile the finished worker")
    finally:
        service.shutdown()

    run = registry.get_run(run_id)
    assert run is not None
    assert run["kind"] == "matrix_evaluation"
    assert run["status"] == "interrupted"
    assert run["config"]["selection_evidence_path"] == str(
        (tmp_path / "matrix_selections.json").resolve()
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [
        sys.executable,
        "-m",
        "embodied_skill_composer.construction.lab_worker",
    ]
    assert command[command.index("--run-id") + 1] == run_id
    assert run["attempt"] == 1
    assert "matrix_evaluation worker exited with code 0" in str(run["error"])
