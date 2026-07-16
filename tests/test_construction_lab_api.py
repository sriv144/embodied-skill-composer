from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from embodied_skill_composer.construction.api import create_app
from embodied_skill_composer.construction.lab_registry import LabRegistry


class FakeArtifacts:
    def __init__(self, run_dir: Path, controller: str = "mappo") -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.policy_manifest_path = run_dir / "policy_manifest.json"
        self.policy_manifest_path.write_text(
            json.dumps(
                {
                    "policy_id": "fixture-policy",
                    "controller": controller,
                    "checkpoint_path": str(run_dir / "policy.pt"),
                }
            ),
            encoding="utf-8",
        )

    def model_dump(self, mode: str = "python") -> dict[str, str]:
        del mode
        return {"run_dir": str(self.run_dir)}


def test_lab_registry_persists_runs_events_scenarios_and_policies(tmp_path: Path) -> None:
    path = tmp_path / "lab.sqlite"
    registry = LabRegistry(path)
    registry.upsert_scenario(
        "scenario-1",
        seed=900,
        split="test",
        payload={"module_count": 24},
    )
    registry.upsert_policy(
        "policy-1",
        "mappo",
        {"checkpoint_path": "policy.pt"},
    )
    run_id = registry.create_run("training", {"seed": 7})
    registry.update_run(run_id, status="running", progress=0.25)
    registry.append_event(run_id, {"event": "quarter"})

    reopened = LabRegistry(path)
    assert reopened.get_scenario("scenario-1")["payload"]["module_count"] == 24
    assert reopened.list_policies()[0]["id"] == "policy-1"
    assert reopened.get_run(run_id)["progress"] == 0.25
    assert [item["sequence"] for item in reopened.list_events(run_id)] == [1, 2]
    assert reopened.request_cancel(run_id) is True
    assert reopened.get_run(run_id)["status"] == "cancel_requested"


def test_lab_api_gates_training_and_persists_completed_policy(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"

    def fake_training_runner(
        _design,
        config,
        *,
        progress_callback,
        cancel_check,
    ):
        assert not cancel_check()
        progress_callback({"event": "ppo_update", "transitions": config.transitions})
        return FakeArtifacts(artifact_root)

    app = create_app(
        registry_path=tmp_path / "api-lab.sqlite",
        training_runner=fake_training_runner,
    )
    with TestClient(app) as client:
        denied = client.post(
            "/api/lab/training",
            json={"algorithm": "mappo", "profile": "unit", "confirmed": False},
        )
        assert denied.status_code == 409

        created = client.post(
            "/api/lab/training",
            json={
                "algorithm": "mappo",
                "profile": "unit",
                "transitions": 4,
                "confirmed": True,
                "device": "cpu",
            },
        )
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        for _ in range(100):
            run = client.get(f"/api/lab/runs/{run_id}").json()
            if run["status"] == "completed":
                break
            time.sleep(0.02)
        assert run["status"] == "completed"
        assert run["progress"] == 1.0
        assert client.get("/api/lab/policies").json()[0]["id"] == "fixture-policy"
        events = client.get(f"/api/lab/runs/{run_id}/events").json()
        assert any(item["payload"]["event"] == "training_completed" for item in events)


def test_lab_api_generates_seeded_scenario_and_reports_coppelia_health(tmp_path: Path) -> None:
    with TestClient(create_app(registry_path=tmp_path / "lab.sqlite")) as client:
        response = client.post("/api/lab/scenarios", json={"seed": 900})
        assert response.status_code == 201
        assert response.json()["split"] == "test"
        scenarios = client.get("/api/lab/scenarios").json()
        assert any(item["seed"] == 900 for item in scenarios)
        health = client.get("/api/lab/coppelia/health?port=1")
        assert health.status_code == 200
        assert health.json()["controller"] == "dynamic_base_logical_payload"
