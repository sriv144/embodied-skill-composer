from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from threading import Event, current_thread, enumerate as enumerate_threads

import pytest

from embodied_skill_composer.construction import lab_worker as worker
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import TrainingConfig

ROOT = Path(__file__).resolve().parents[1]


def _claimed_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str = "training",
) -> tuple[LabRegistry, str]:
    registry = LabRegistry(tmp_path / "bootstrap.sqlite")
    design = load_house_design(ROOT / "configs/construction/cottage_v1.yaml")
    config = TrainingConfig.for_profile("unit")
    run_id = registry.create_run(
        kind, (config.model_dump(mode="json") if kind == "training"
               else {"matrix_id": "bootstrap-matrix"}),
        input_payload={"design": design.model_dump(mode="json")},
        run_id="bootstrap-job",
    )
    claimed = registry.claim_next_durable_job(kinds=(kind,))
    assert claimed is not None
    monkeypatch.setattr(sys, "argv", [
        "lab_worker", "--registry", str(registry.path),
        "--run-id", run_id, "--claim-token", str(claimed["claim_token"]),
    ])
    return registry, run_id


def test_bootstrap_module_does_not_import_heavy_training_dependencies() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import embodied_skill_composer.construction.lab_worker; "
            "assert 'torch' not in sys.modules; "
            "assert 'embodied_skill_composer.construction.training' not in sys.modules"
        )], cwd=ROOT, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["training", "matrix_evaluation"])
def test_claim_heartbeat_runs_during_slow_dependency_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    registry, run_id = _claimed_job(tmp_path, monkeypatch, kind)
    renewed = Event()
    update = LabRegistry.update_run

    def observe_update(self, *args, **kwargs):
        update(self, *args, **kwargs)
        if current_thread().name == f"lab-heartbeat-{run_id}":
            renewed.set()

    load = worker._load_training_config

    def slow_load(payload=None):
        assert renewed.wait(timeout=2), "heartbeat must precede dependency loading"
        run = registry.get_run(run_id)
        assert run is not None and run["pid"] == os.getpid()
        assert not any(
            event["payload"]["event"] in {"training_started", "matrix_evaluation_started"}
            for event in registry.list_events(run_id)
        )
        return load(payload) if kind == "training" else None

    def fail_after_start(*_args, **_kwargs):
        raise RuntimeError("fixture stopped after initialization")

    monkeypatch.setattr(worker, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(LabRegistry, "update_run", observe_update)
    loader = "_load_training_config" if kind == "training" else "_load_evaluation_dependencies"
    executor = "_execute_training" if kind == "training" else "_execute_matrix_evaluation"
    monkeypatch.setattr(worker, loader, slow_load)
    monkeypatch.setattr(worker, executor, fail_after_start)
    assert worker.main() == 2
    run = registry.get_run(run_id)
    assert run is not None and run["status"] == "failed"
    assert renewed.is_set()
    assert not any(t.name == f"lab-heartbeat-{run_id}" for t in enumerate_threads())


@pytest.mark.parametrize("action", ["fence", "cancel"])
def test_initialization_rechecks_claim_and_cancellation_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str,
) -> None:
    registry, run_id = _claimed_job(tmp_path, monkeypatch)
    load = worker._load_training_config

    def interrupted_load(payload):
        if action == "fence":
            registry.update_run(run_id, status="interrupted")
        else:
            assert registry.request_cancel(run_id)
        return load(payload)

    def must_not_execute(*_args, **_kwargs):
        pytest.fail("a fenced or cancelled initialization must not begin training")

    monkeypatch.setattr(worker, "_load_training_config", interrupted_load)
    monkeypatch.setattr(worker, "_execute_training", must_not_execute)
    assert worker.main() == (3 if action == "fence" else 2)
    run = registry.get_run(run_id)
    assert run is not None
    assert run["status"] == ("interrupted" if action == "fence" else "cancelled")


@pytest.mark.parametrize("kind", ["training", "matrix_evaluation"])
def test_dependency_import_failure_is_persisted_and_stops_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    registry, run_id = _claimed_job(tmp_path, monkeypatch, kind)

    def failed_import(_payload=None):
        raise ImportError("optional training dependency is unavailable")

    loader = "_load_training_config" if kind == "training" else "_load_evaluation_dependencies"
    monkeypatch.setattr(worker, loader, failed_import)
    assert worker.main() == 2
    run = registry.get_run(run_id)
    assert run is not None and run["status"] == "failed"
    assert "dependency is unavailable" in str(run["error"])
    assert not any(
        event["payload"]["event"] in {"training_started", "matrix_evaluation_started"}
        for event in registry.list_events(run_id)
    )
    assert not any(t.name == f"lab-heartbeat-{run_id}" for t in enumerate_threads())


@pytest.mark.parametrize("kind", ["training", "matrix_evaluation"])
def test_hung_initialization_stops_renewing_without_releasing_live_gpu_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    registry, run_id = _claimed_job(tmp_path, monkeypatch, kind)
    expired = Event()
    append = LabRegistry.append_event

    def observe_event(self, event_run_id, payload, **kwargs):
        sequence = append(self, event_run_id, payload, **kwargs)
        if payload.get("event") == "worker_launch_failed":
            expired.set()
        return sequence

    load = worker._load_training_config

    def hung_load(payload=None):
        assert expired.wait(timeout=2)
        return load(payload) if kind == "training" else None

    def must_not_execute(*_args, **_kwargs):
        pytest.fail("expired initialization must not begin training")

    monkeypatch.setattr(worker, "_HEARTBEAT_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(worker, "_INITIALIZATION_TIMEOUT_SECONDS", 0.0)
    monkeypatch.setattr(LabRegistry, "append_event", observe_event)
    loader = "_load_training_config" if kind == "training" else "_load_evaluation_dependencies"
    executor = "_execute_training" if kind == "training" else "_execute_matrix_evaluation"
    monkeypatch.setattr(worker, loader, hung_load)
    monkeypatch.setattr(worker, executor, must_not_execute)
    assert worker.main() == 3
    run = registry.get_run(run_id)
    assert run is not None and run["status"] == "running"
    assert registry.claim_next_durable_job(kinds=(kind,)) is None
    assert not any(t.name == f"lab-heartbeat-{run_id}" for t in enumerate_threads())


@pytest.mark.parametrize("kind", ["training", "matrix_evaluation"])
def test_initialization_deadline_is_checked_before_execution_without_heartbeat_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    registry, run_id = _claimed_job(tmp_path, monkeypatch, kind)
    loader = "_load_training_config" if kind == "training" else "_load_evaluation_dependencies"
    executor = "_execute_training" if kind == "training" else "_execute_matrix_evaluation"

    def must_not_execute(*_args, **_kwargs):
        pytest.fail("an expired startup must not depend on heartbeat scheduling")

    if kind == "matrix_evaluation":
        monkeypatch.setattr(worker, loader, lambda: None)
    monkeypatch.setattr(worker, executor, must_not_execute)
    monkeypatch.setattr(worker, "_HEARTBEAT_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(worker, "_INITIALIZATION_TIMEOUT_SECONDS", 0.0)
    assert worker.main() == 3
    run = registry.get_run(run_id)
    assert run is not None and run["status"] == "running"
    events = [event["payload"]["event"] for event in registry.list_events(run_id)]
    assert events.count("worker_launch_failed") == 1
    assert "training_started" not in events and "matrix_evaluation_started" not in events
    assert not any(t.name == f"lab-heartbeat-{run_id}" for t in enumerate_threads())
