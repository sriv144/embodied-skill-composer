from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from embodied_skill_composer.construction import api as api_module
from embodied_skill_composer.construction import evaluation as evaluation_module
from embodied_skill_composer.construction import lab_worker as lab_worker_module
from embodied_skill_composer.construction import policy as policy_module
from embodied_skill_composer.construction.api import create_app
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import TrainingConfig


class FakeArtifacts:
    def __init__(
        self,
        run_dir: Path,
        *,
        manifest_payload: object | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_manifest_path = self.run_dir / "policy_manifest.json"
        if manifest_payload is not None:
            self.policy_manifest_path.write_text(
                json.dumps(manifest_payload),
                encoding="utf-8",
            )

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        return {"run_dir": str(self.run_dir), "mode": mode}


class FakeSocket:
    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeInference:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        assert mode == "json"
        return self.payload


def test_api_read_routes_disruptions_static_assets_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "house.glb").write_bytes(b"fixture-glb")
    monkeypatch.setattr(api_module, "DEFAULT_ASSETS", assets)

    def fake_connection(address: tuple[str, int], timeout: float) -> FakeSocket:
        assert address == ("127.0.0.1", 23001)
        assert timeout == 0.4
        return FakeSocket()

    monkeypatch.setattr(api_module.socket, "create_connection", fake_connection)
    app = create_app(
        registry_path=tmp_path / "api.sqlite",
        training_runner=_unused_training_runner,
    )
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ready"
        assert client.get("/api/health", headers={"host": "evil.example"}).status_code == 400

        project = client.get("/api/project")
        assert project.status_code == 200
        assert project.json()["geometry_asset_url"] == "/artifacts/house.glb"
        assert client.get("/artifacts/house.glb").content == b"fixture-glb"
        assert client.get("/api/report").json()["markdown"].startswith(
            "# Construction Research Report"
        )

        trace = client.get("/api/traces/sequential")
        assert trace.status_code == 200
        assert "metrics" in trace.json()
        assert client.get("/api/traces/not-a-controller").status_code == 404

        recovered = client.post(
            "/api/traces/sequential/disrupt",
            json={"failure_type": "obstacle", "timestamp_s": 0},
        )
        assert recovered.status_code == 200
        assert recovered.json()["metrics"]["recovery_cost_s"] == 14
        assert (
            client.post(
                "/api/traces/not-a-controller/disrupt",
                json={"failure_type": "obstacle", "timestamp_s": 0},
            ).status_code
            == 404
        )
        invalid_disruption = client.post(
            "/api/traces/sequential/disrupt",
            json={"failure_type": "obstacle", "timestamp_s": 1_000_000},
        )
        assert invalid_disruption.status_code == 422
        assert "before construction completes" in invalid_disruption.json()["detail"]

        fixture_id = project.json()["design"]["design_id"]
        assert client.get(f"/api/lab/scenarios/{fixture_id}").status_code == 200
        assert client.get("/api/lab/scenarios/missing").status_code == 404
        assert client.get("/api/lab/scenarios").status_code == 200
        assert client.get("/api/lab/policies").json() == []
        assert client.get("/api/lab/runs/missing").status_code == 404
        assert client.get("/api/lab/runs/missing/events").status_code == 404
        assert client.get("/api/lab/runs?limit=0").status_code == 422

        coppelia = client.get(
            "/api/lab/coppelia/health?host=simulator&port=23001"
        )
        assert coppelia.json() == {
            "reachable": True,
            "host": "127.0.0.1",
            "port": 23001,
            "detail": "ZeroMQ remote API port is reachable",
            "controller": "dynamic_base_logical_payload",
        }


def test_api_rebuild_and_floorplan_parse_success_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(
        registry_path=tmp_path / "api.sqlite",
        training_runner=_unused_training_runner,
    )
    with TestClient(app) as client:
        design = client.get("/api/project").json()["design"]
        rebuilt = client.post("/api/design/rebuild", json={"design": design})
        assert rebuilt.status_code == 200
        assert rebuilt.json()["design"]["design_id"] == design["design_id"]
        reviewed = client.get("/api/lab/scenarios").json()
        assert any(item["split"] == "reviewed" for item in reviewed)

        original_compile = api_module.compile_house_design

        def reject_design(_design: object) -> None:
            raise ValueError("synthetic compiler rejection")

        monkeypatch.setattr(api_module, "compile_house_design", reject_design)
        rejected = client.post("/api/design/rebuild", json={"design": design})
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == "synthetic compiler rejection"
        monkeypatch.setattr(api_module, "compile_house_design", original_compile)

        monkeypatch.setattr(
            api_module,
            "infer_orthogonal_floor_plan",
            lambda payload, known_width_m: FakeInference(
                {
                    "byte_count": len(payload),
                    "known_width_m": known_width_m,
                }
            ),
        )
        parsed = client.post(
            "/api/intent/parse?known_width_m=12",
            files={"file": ("floor.png", b"pixels", "image/png")},
        )
        assert parsed.status_code == 200
        assert parsed.json() == {"byte_count": 6, "known_width_m": 12.0}

        def reject_image(_payload: bytes, *, known_width_m: float) -> None:
            del known_width_m
            raise RuntimeError("ambiguous floor plan")

        monkeypatch.setattr(api_module, "infer_orthogonal_floor_plan", reject_image)
        rejected_parse = client.post(
            "/api/intent/parse?known_width_m=12",
            files={"file": ("floor.png", b"pixels", "image/png")},
        )
        assert rejected_parse.status_code == 422
        assert rejected_parse.json()["detail"] == "ambiguous floor plan"
        assert (
            client.post(
                "/api/intent/parse?known_width_m=1",
                files={"file": ("floor.png", b"pixels", "image/png")},
            ).status_code
            == 422
        )


def test_api_cancel_resume_run_events_and_websockets(tmp_path: Path) -> None:
    app = create_app(
        registry_path=tmp_path / "api.sqlite",
        training_runner=_unused_training_runner,
    )
    registry: LabRegistry = app.state.lab_registry
    cancellable = registry.create_run("training", {"seed": 7}, run_id="cancel-me")
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"checkpoint")
    resumable = registry.create_run(
        "training",
        {"seed": 8},
        status="interrupted",
        run_id="resume-me",
    )
    registry.update_run(resumable, latest_checkpoint=str(checkpoint))

    with TestClient(app) as client:
        cancelled = client.post(f"/api/lab/runs/{cancellable}/cancel")
        assert cancelled.status_code == 202
        assert cancelled.json()["cancel_requested"] is True
        assert client.post(f"/api/lab/runs/{cancellable}/cancel").status_code == 409
        assert client.post("/api/lab/runs/missing/cancel").status_code == 404

        resumed = client.post(f"/api/lab/runs/{resumable}/resume")
        assert resumed.status_code == 202
        assert resumed.json()["status"] == "resuming"
        assert client.post(f"/api/lab/runs/{resumable}/resume").status_code == 409
        assert client.post("/api/lab/runs/missing/resume").status_code == 404

        assert client.get(f"/api/lab/runs/{cancellable}").json()["status"] == "cancelled"
        assert client.get("/api/lab/runs").status_code == 200
        events = client.get(f"/api/lab/runs/{cancellable}/events").json()
        assert events
        sequence = events[0]["sequence"]
        assert client.get(
            f"/api/lab/runs/{cancellable}/events?after={sequence}"
        ).status_code == 200

        with client.websocket_connect(
            f"/api/lab/runs/{cancellable}/events/ws"
        ) as websocket:
            websocket_events = []
            while True:
                try:
                    websocket_events.append(websocket.receive_json())
                except WebSocketDisconnect as exc:
                    assert exc.code == 1000
                    break
        assert websocket_events

        with client.websocket_connect(
            "/api/lab/runs/missing/events/ws"
        ) as websocket:
            with pytest.raises(WebSocketDisconnect) as exc_info:
                websocket.receive_json()
            assert exc_info.value.code == 1008

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                f"/api/lab/runs/{cancellable}/events/ws",
                headers={"origin": "https://evil.example"},
            ):
                pass
        assert exc_info.value.code == 1008


def test_api_evaluation_policy_validation_and_completion(tmp_path: Path) -> None:
    captured_configs: list[dict[str, object]] = []

    def evaluation_runner(_design: object, config: dict[str, object]) -> FakeArtifacts:
        captured_configs.append(config)
        return FakeArtifacts(tmp_path / "evaluation" / str(len(captured_configs)))

    app = create_app(
        registry_path=tmp_path / "api.sqlite",
        training_runner=_unused_training_runner,
        evaluation_runner=evaluation_runner,
    )
    registry: LabRegistry = app.state.lab_registry
    with TestClient(app) as client:
        baseline = client.post(
            "/api/lab/evaluations",
            json={
                "seeds": [900],
                "controllers": ["sequential", "cp_sat"],
                "include_failures": False,
            },
        )
        assert baseline.status_code == 202
        baseline_run = _wait_for_status(
            registry,
            baseline.json()["run_id"],
            {"completed"},
        )
        assert baseline_run["progress"] == 1.0

        missing_policy = client.post(
            "/api/lab/evaluations",
            json={"controllers": ["mappo"], "policy_ids": {"mappo": "missing"}},
        )
        assert missing_policy.status_code == 422
        assert missing_policy.json()["detail"] == "missing mappo policy"

        no_checkpoint = _policy_manifest("no-checkpoint", checkpoint_path=None)
        registry.upsert_policy("no-checkpoint", "mappo", no_checkpoint)
        checkpoint_missing = client.post(
            "/api/lab/evaluations",
            json={
                "controllers": ["mappo"],
                "policy_ids": {"mappo": "no-checkpoint"},
            },
        )
        assert checkpoint_missing.status_code == 422
        assert checkpoint_missing.json()["detail"] == "mappo policy has no checkpoint"

        checkpoint_path = tmp_path / "mappo.pt"
        checkpoint_path.write_bytes(b"policy")
        registry.upsert_policy(
            "mappo-policy",
            "mappo",
            _policy_manifest("mappo-policy", checkpoint_path=str(checkpoint_path)),
        )
        learned = client.post(
            "/api/lab/evaluations",
            json={
                "seeds": [901],
                "controllers": ["mappo"],
                "policy_ids": {"mappo": "mappo-policy"},
            },
        )
        assert learned.status_code == 202
        _wait_for_status(registry, learned.json()["run_id"], {"completed"})
        assert captured_configs[-1]["policy_checkpoints"] == {
            "mappo": str(checkpoint_path)
        }

        invalid_controller = client.post(
            "/api/lab/evaluations",
            json={"controllers": ["unknown"]},
        )
        assert invalid_controller.status_code == 422


def test_lab_service_default_evaluation_and_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    design = load_house_design(api_module.DEFAULT_DESIGN)
    captured: dict[str, object] = {}

    def load_checkpoint(path: Path) -> str:
        captured["checkpoint"] = path
        return "loaded-policy"

    def run_suite(
        received_design: object,
        *,
        seeds: list[int],
        controllers: list[str],
        policies: dict[str, object],
        include_failure_suite: bool,
    ) -> str:
        captured.update(
            {
                "design": received_design,
                "seeds": seeds,
                "controllers": controllers,
                "policies": policies,
                "include_failure_suite": include_failure_suite,
            }
        )
        return "suite"

    def write_artifacts(suite: object, output_root: Path) -> FakeArtifacts:
        assert suite == "suite"
        captured["output_root"] = output_root
        return FakeArtifacts(tmp_path / "default-evaluation")

    monkeypatch.setattr(policy_module, "load_policy_checkpoint", load_checkpoint)
    monkeypatch.setattr(evaluation_module, "run_evaluation_suite", run_suite)
    monkeypatch.setattr(
        evaluation_module,
        "write_evaluation_artifacts",
        write_artifacts,
    )
    service = LabService(registry, training_runner=_unused_training_runner)
    try:
        completed_id = service.launch_evaluation(
            design,
            seeds=[900, 901],
            controllers=["mappo"],
            policy_checkpoints={"mappo": "fixture.pt"},
            include_failures=True,
            output_root=tmp_path / "requested-output",
        )
        completed = _wait_for_status(registry, completed_id, {"completed"})
        assert completed["artifact_dir"] == str(tmp_path / "default-evaluation")
        assert captured["checkpoint"] == Path("fixture.pt")
        assert captured["policies"] == {"mappo": "loaded-policy"}
        assert captured["include_failure_suite"] is True

        def failing_evaluation(_design: object, _config: object) -> FakeArtifacts:
            raise RuntimeError("evaluation fixture failed")

        service.evaluation_runner = failing_evaluation
        failed_id = service.launch_evaluation(
            design,
            seeds=[902],
            controllers=["sequential"],
            policy_checkpoints={},
            include_failures=False,
            output_root=tmp_path / "failed-output",
        )
        failed = _wait_for_status(registry, failed_id, {"failed"})
        assert failed["error"] == "evaluation fixture failed"
        assert any(
            event["payload"]["event"] == "failed"
            for event in registry.list_events(failed_id)
        )
    finally:
        service.shutdown()


def test_research_launch_rejects_dirty_source_before_queueing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodied_skill_composer.construction.training as training_module

    registry = LabRegistry(tmp_path / "lab.sqlite")
    service = LabService(registry, training_runner=_unused_training_runner)
    config = TrainingConfig.for_profile("research", seed=7)
    monkeypatch.setattr(
        training_module,
        "source_fingerprint",
        lambda: {"commit": "fixture", "dirty": True, "tree_digest": "dirty-tree"},
    )
    try:
        with pytest.raises(ValueError, match="clean source worktree"):
            service.launch_training(load_house_design(api_module.DEFAULT_DESIGN), config)
        assert registry.list_runs() == []
    finally:
        service.shutdown()


def test_lab_service_inline_training_failure_classification(tmp_path: Path) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    design = load_house_design(api_module.DEFAULT_DESIGN)
    config = TrainingConfig.for_profile("unit", seed=41)

    def invalid_progress(
        _design: object,
        _config: object,
        *,
        progress_callback: Any,
        cancel_check: Any,
    ) -> FakeArtifacts:
        assert not cancel_check()
        progress_callback({"event": "bad", "transitions": "four"})
        raise AssertionError("invalid progress should abort before this line")

    service = LabService(registry, training_runner=invalid_progress)
    try:
        invalid_id = service.launch_training(design, config.model_copy(deep=True))
        invalid = _wait_for_status(registry, invalid_id, {"failed"})
        assert invalid["error"] == "progress transitions must be an integer"

        def malformed_manifest(
            _design: object,
            _config: object,
            *,
            progress_callback: Any,
            cancel_check: Any,
        ) -> FakeArtifacts:
            del progress_callback, cancel_check
            return FakeArtifacts(
                tmp_path / "malformed-manifest",
                manifest_payload=[],
            )

        service.training_runner = malformed_manifest
        malformed_id = service.launch_training(design, config.model_copy(deep=True))
        malformed = _wait_for_status(registry, malformed_id, {"failed"})
        assert "policy manifest must contain an object" in str(malformed["error"])

        def cancelled_training(
            _design: object,
            _config: object,
            *,
            progress_callback: Any,
            cancel_check: Any,
        ) -> FakeArtifacts:
            del progress_callback, cancel_check
            raise RuntimeError("cancelled by fixture")

        service.training_runner = cancelled_training
        cancelled_id = service.launch_training(design, config.model_copy(deep=True))
        cancelled = _wait_for_status(registry, cancelled_id, {"cancelled"})
        assert cancelled["error"] == "cancelled by fixture"
    finally:
        service.shutdown()


def test_lab_worker_success_resumes_and_registers_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    design = load_house_design(api_module.DEFAULT_DESIGN)
    config = TrainingConfig.for_profile("unit", seed=43)
    run_id = registry.create_run(
        "training",
        config.model_dump(mode="json"),
        input_payload={"design": design.model_dump(mode="json")},
        run_id="worker-success",
    )
    resume_checkpoint = tmp_path / "resume.pt"
    resume_checkpoint.write_bytes(b"resume")
    registry.update_run(run_id, latest_checkpoint=str(resume_checkpoint))
    claimed = registry.claim_next_training()
    assert claimed is not None
    claim_token = str(claimed["claim_token"])
    latest_checkpoint = tmp_path / "checkpoint-latest.pt"
    captured: dict[str, object] = {}

    def fake_training(
        received_design: object,
        received_config: TrainingConfig,
        *,
        progress_callback: Any,
        cancel_check: Any,
    ) -> FakeArtifacts:
        captured["design"] = received_design
        captured["config"] = received_config
        assert received_config.resume_checkpoint == resume_checkpoint
        assert received_config.resume_provenance == {
            "run_id": run_id,
            "attempt": 1,
            "checkpoint": str(resume_checkpoint),
        }
        assert not cancel_check()
        progress_callback(
            {
                "event": "checkpoint_saved",
                "transitions": received_config.transitions,
                "checkpoint_path": str(latest_checkpoint),
            }
        )
        return FakeArtifacts(
            tmp_path / "worker-artifacts",
            manifest_payload=_policy_manifest(
                "worker-policy",
                checkpoint_path=str(latest_checkpoint),
            ),
        )

    monkeypatch.setattr(lab_worker_module, "train_swarm_policy", fake_training)
    _set_worker_argv(monkeypatch, registry, run_id, claim_token)
    assert lab_worker_module.main() == 0

    completed = registry.get_run(run_id)
    assert completed is not None
    assert completed["status"] == "completed"
    assert completed["progress"] == 1.0
    assert completed["latest_checkpoint"] == str(latest_checkpoint)
    assert registry.list_policies()[0]["id"] == "worker-policy"
    assert captured["design"] == design
    assert any(
        event["payload"]["event"] == "training_completed"
        for event in registry.list_events(run_id)
    )


def test_lab_worker_lost_claim_cannot_register_policy_or_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    design = load_house_design(api_module.DEFAULT_DESIGN)
    config = TrainingConfig.for_profile("unit", seed=45)
    run_id = registry.create_run(
        "training",
        config.model_dump(mode="json"),
        input_payload={"design": design.model_dump(mode="json")},
        run_id="worker-lost-claim",
    )
    claimed = registry.claim_next_training()
    assert claimed is not None
    claim_token = str(claimed["claim_token"])

    def stale_training(*_args: object, **_kwargs: object) -> FakeArtifacts:
        registry.update_run(run_id, status="interrupted")
        return FakeArtifacts(
            tmp_path / "stale-worker-artifacts",
            manifest_payload=_policy_manifest(
                "stale-worker-policy",
                checkpoint_path=str(tmp_path / "stale-policy.pt"),
            ),
        )

    monkeypatch.setattr(lab_worker_module, "train_swarm_policy", stale_training)
    _set_worker_argv(monkeypatch, registry, run_id, claim_token)

    assert lab_worker_module.main() == 3
    interrupted = registry.get_run(run_id)
    assert interrupted is not None and interrupted["status"] == "interrupted"
    assert registry.list_policies() == []
    assert not any(
        event["payload"]["event"] == "training_completed"
        for event in registry.list_events(run_id)
    )


def test_lab_worker_failure_unknown_run_and_invalid_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = LabRegistry(tmp_path / "lab.sqlite")
    design = load_house_design(api_module.DEFAULT_DESIGN)
    config = TrainingConfig.for_profile("unit", seed=47)

    _set_worker_argv(monkeypatch, registry, "unknown", "token")
    with pytest.raises(SystemExit, match="unknown run: unknown"):
        lab_worker_module.main()

    run_id = registry.create_run(
        "training",
        config.model_dump(mode="json"),
        input_payload={"design": design.model_dump(mode="json")},
        run_id="worker-failure",
    )
    claimed = registry.claim_next_training()
    assert claimed is not None
    claim_token = str(claimed["claim_token"])

    def failing_training(*_args: object, **_kwargs: object) -> FakeArtifacts:
        raise RuntimeError("simulated worker failure")

    monkeypatch.setattr(lab_worker_module, "train_swarm_policy", failing_training)
    _set_worker_argv(monkeypatch, registry, run_id, claim_token)
    assert lab_worker_module.main() == 2
    failed = registry.get_run(run_id)
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error"] == "simulated worker failure"

    stale_id = registry.create_run(
        "training",
        config.model_dump(mode="json"),
        input_payload={"design": design.model_dump(mode="json")},
        run_id="worker-stale",
    )
    stale_claim = registry.claim_next_training()
    assert stale_claim is not None
    assert stale_claim["id"] == stale_id
    _set_worker_argv(monkeypatch, registry, stale_id, "wrong-token")
    with pytest.raises(SystemExit, match="claim token is stale or invalid"):
        lab_worker_module.main()


def _unused_training_runner(*_args: object, **_kwargs: object) -> FakeArtifacts:
    raise AssertionError("training must not run in this test")


def _policy_manifest(
    policy_id: str,
    *,
    checkpoint_path: str | None,
) -> dict[str, object]:
    return {
        "policy_id": policy_id,
        "controller": "mappo",
        "git_sha": "fixture-commit",
        "seed": 7,
        "training_seed": 7,
        "transition_count": 64,
        "checkpoint_path": checkpoint_path,
    }


def _wait_for_status(
    registry: LabRegistry,
    run_id: str,
    expected: set[str],
) -> dict[str, object]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        run = registry.get_run(run_id)
        if run is not None and run["status"] in expected:
            return run
        time.sleep(0.01)
    pytest.fail(f"run {run_id} did not reach {sorted(expected)}")


def _set_worker_argv(
    monkeypatch: pytest.MonkeyPatch,
    registry: LabRegistry,
    run_id: str,
    claim_token: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lab_worker",
            "--registry",
            str(registry.path),
            "--run-id",
            run_id,
            "--claim-token",
            claim_token,
        ],
    )
