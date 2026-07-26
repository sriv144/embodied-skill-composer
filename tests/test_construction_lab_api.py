from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient

from embodied_skill_composer.construction import experiment_execution
from embodied_skill_composer.construction.api import create_app
from embodied_skill_composer.construction.experiment_execution import (
    CandidateEvaluation,
    RunSelectionEvidence,
)
from embodied_skill_composer.construction.experiment_protocol import (
    CheckpointValidationResult,
    ExperimentProtocol,
    select_validation_checkpoint,
)
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
    registry.append_event(run_id, {"event": "evaluation_started"})

    reopened = LabRegistry(path)
    reopened_scenario = reopened.get_scenario("scenario-1")
    assert reopened_scenario is not None
    scenario_payload = reopened_scenario["payload"]
    assert isinstance(scenario_payload, dict)
    assert scenario_payload["module_count"] == 24
    assert reopened.list_policies()[0]["id"] == "policy-1"
    reopened_run = reopened.get_run(run_id)
    assert reopened_run is not None
    assert reopened_run["progress"] == 0.25
    assert [item["sequence"] for item in reopened.list_events(run_id)] == [1, 2]
    assert reopened.request_cancel(run_id) is True
    cancelled_run = reopened.get_run(run_id)
    assert cancelled_run is not None
    assert cancelled_run["status"] == "cancel_requested"


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
        progress_callback(
            {
                "event": "ppo_update",
                "transitions": config.transitions,
                "update": 1,
                "loss_objective": 0.1,
                "loss_critic": 0.2,
                "loss_entropy": 0.3,
                "mean_episode_return": 1.0,
                "rollout_terminal_fraction": 1.0,
            }
        )
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
        event_response = client.get(f"/api/lab/runs/{run_id}/events")
        assert event_response.status_code == 200
        events = event_response.json()
        assert all(set(item) == {"sequence", "created_at", "payload"} for item in events)
        assert all(
            isinstance(item["sequence"], int)
            and isinstance(item["created_at"], str)
            and isinstance(item["payload"]["event"], str)
            for item in events
        )
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


def test_lab_api_launches_frozen_matrix_and_rejects_duplicate_profile(
    tmp_path: Path,
) -> None:
    def fake_training_runner(
        _design,
        config,
        *,
        progress_callback,
        cancel_check,
    ):
        del progress_callback, cancel_check
        return FakeArtifacts(
            tmp_path
            / f"unused-training-{config.experiment_variant}-{config.seed}"
        )

    app = create_app(
        registry_path=tmp_path / "matrix.sqlite",
        training_runner=fake_training_runner,
    )
    with TestClient(app) as client:
        denied = client.post(
            "/api/lab/experiment-matrices",
            json={"profile": "unit", "confirmed": False},
        )
        assert denied.status_code == 409

        launched = client.post(
            "/api/lab/experiment-matrices",
            json={"profile": "unit", "confirmed": True},
        )
        assert launched.status_code == 202
        payload = launched.json()
        assert payload["run_count"] == 20
        assert len(payload["run_ids"]) == 20

        matrix = client.get(
            f"/api/lab/experiment-matrices/{payload['matrix_id']}"
        ).json()
        assert matrix["protocol_digest"] == payload["protocol_digest"]
        assert matrix["expected_run_count"] == 20
        assert len(matrix["runs"]) == 20
        configs = [item["config"] for item in matrix["runs"]]
        assert {(item["experiment_variant"], item["seed"]) for item in configs} == {
            (variant, seed)
            for variant in (
                "mappo_full",
                "ippo_full",
                "mappo_no_bc",
                "mappo_no_failure_curriculum",
            )
            for seed in range(7, 12)
        }
        assert all(item["profile"] == "unit" for item in configs)
        assert all(item["transitions"] == 64 for item in configs)
        assert all(
            isinstance(item["protocol_run_digest"], str)
            and len(item["protocol_run_digest"]) == 64
            for item in configs
        )
        assert all(
            item["checkpoint_fractions"] == [0.1, 0.25, 0.5, 0.75, 1.0]
            for item in configs
        )
        assert sum(item["algorithm"] == "mappo" for item in configs) == 15
        assert sum(item["algorithm"] == "ippo" for item in configs) == 5

        listed = client.get("/api/lab/experiment-matrices").json()
        assert [item["id"] for item in listed] == [payload["matrix_id"]]
        duplicate = client.post(
            "/api/lab/experiment-matrices",
            json={"profile": "unit", "confirmed": True},
        )
        assert duplicate.status_code == 409
        assert "protocol digest and profile" in duplicate.json()["detail"]


def test_lab_api_freezes_validation_selections_before_heldout_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_training_runner(
        _design,
        config,
        *,
        progress_callback,
        cancel_check,
    ):
        del progress_callback, cancel_check
        return FakeArtifacts(
            tmp_path
            / f"unused-training-{config.experiment_variant}-{config.seed}"
        )

    def fake_evaluation_runner(_design, config):
        return FakeArtifacts(Path(config["output_root"]) / "fixture-evaluation")

    app = create_app(
        registry_path=tmp_path / "selection.sqlite",
        training_runner=fake_training_runner,
        evaluation_runner=fake_evaluation_runner,
    )
    with TestClient(app) as client:
        launched = client.post(
            "/api/lab/experiment-matrices",
            json={"profile": "unit", "confirmed": True},
        ).json()
        matrix_id = launched["matrix_id"]
        for _ in range(200):
            matrix = client.get(
                f"/api/lab/experiment-matrices/{matrix_id}"
            ).json()
            if matrix["status_counts"].get("completed") == 20:
                break
            time.sleep(0.01)
        assert matrix["status_counts"]["completed"] == 20
        evaluated_run_keys: list[str] = []

        def canonical_candidate_evaluator(
            _design: object,
            run: dict[str, object],
            protocol: ExperimentProtocol,
            *,
            matrix_id: str,
            output_root: Path,
            device: str,
        ) -> RunSelectionEvidence:
            assert device == "cpu"
            run_key = cast(str, run["run_key"])
            config = cast(dict[str, object], run["config"])
            resume_history = cast(
                list[dict[str, object]],
                run["resume_provenance_history"],
            )
            candidates: list[CandidateEvaluation] = []
            completion_by_fraction = {
                0.1: 0.60,
                0.25: 0.75,
                0.5: 0.90,
                0.75: 0.99,
                1.0: 0.99,
            }
            makespan_by_fraction = {
                0.1: 140.0,
                0.25: 130.0,
                0.5: 110.0,
                0.75: 90.0,
                1.0: 95.0,
            }
            for fraction in protocol.checkpoint_fractions:
                percentage = int(round(fraction * 100))
                result = CheckpointValidationResult(
                    checkpoint_id=f"{run_key}-checkpoint-{percentage:03d}pct",
                    experiment_id=cast(str, config["experiment_id"]),
                    experiment_variant=cast(str, config["experiment_variant"]),
                    training_seed=cast(int, config["training_seed"]),
                    checkpoint_fraction=fraction,
                    transition_count=max(1, int(round(64 * fraction))),
                    split="validation",
                    scenario_seeds=list(protocol.selection.scenario_seeds),
                    mean_completion_rate=completion_by_fraction[fraction],
                    mean_makespan_s=makespan_by_fraction[fraction],
                    checkpoint_path=str(
                        Path(cast(str, run["artifact_dir"]))
                        / "checkpoints"
                        / f"policy_{percentage:03d}pct.pt"
                    ),
                    checkpoint_sha256=f"{percentage:064x}",
                    checkpoint_lineage=[],
                    configuration_digest=cast(str, run["config_digest"]),
                    source_commit=cast(str, run["source_commit"]),
                    resume_provenance=(
                        dict(resume_history[-1]) if resume_history else {}
                    ),
                )
                candidates.append(CandidateEvaluation(result=result, episodes=[]))
            selected = select_validation_checkpoint(
                [candidate.result for candidate in candidates],
                validation_seeds=protocol.selection.scenario_seeds,
                required_fractions=protocol.checkpoint_fractions,
            )
            evaluated_run_keys.append(run_key)
            return RunSelectionEvidence(
                matrix_id=matrix_id,
                run_key=run_key,
                candidates=candidates,
                selected=selected,
                evidence_path=(
                    output_root / run_key / "validation_selection.json"
                ),
            )

        monkeypatch.setattr(
            experiment_execution,
            "evaluate_run_checkpoint_candidates",
            canonical_candidate_evaluator,
        )

        first = matrix["runs"][0]
        run_key = first["run_key"]
        denied = client.put(
            f"/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
            json={"confirmed": False},
        )
        assert denied.status_code == 409

        client_authored = client.put(
            f"/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
            json={
                "confirmed": True,
                "checkpoint_fraction": 1.0,
                "mean_completion_rate": 1.0,
                "candidate_ranking": ["client-choice"],
            },
        )
        assert client_authored.status_code == 422
        assert evaluated_run_keys == []

        def corrupt_candidate_evaluator(*_args: object, **_kwargs: object) -> None:
            raise pickle.UnpicklingError("corrupt fractional checkpoint")

        monkeypatch.setattr(
            experiment_execution,
            "evaluate_run_checkpoint_candidates",
            corrupt_candidate_evaluator,
        )
        corrupt = client.put(
            f"/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
            json={"confirmed": True},
        )
        assert corrupt.status_code == 409
        assert "corrupt fractional checkpoint" in corrupt.json()["detail"]
        assert client.get(
            f"/api/lab/experiment-matrices/{matrix_id}/selections"
        ).json() == []
        monkeypatch.setattr(
            experiment_execution,
            "evaluate_run_checkpoint_candidates",
            canonical_candidate_evaluator,
        )

        frozen = client.put(
            f"/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
            json={"confirmed": True},
        )
        assert frozen.status_code == 201
        assert frozen.json()["selection"]["checkpoint_fraction"] == 0.75
        assert frozen.json()["selection"]["mean_completion_rate"] == 0.99
        evidence_path = Path(frozen.json()["evidence_path"])
        assert evidence_path == (
            tmp_path
            / "evidence"
            / "validation_selections"
            / matrix_id
            / run_key
            / "validation_selection.json"
        )
        assert evidence_path.is_file()
        assert frozen.json()["matrix_evidence_path"] is None
        idempotent = client.put(
            f"/api/lab/experiment-matrices/{matrix_id}/selections/{run_key}",
            json={"confirmed": True},
        )
        assert idempotent.status_code == 201
        assert idempotent.json()["selection"] == frozen.json()["selection"]

        blocked = client.post(
            f"/api/lab/experiment-matrices/{matrix_id}/heldout",
            json={"confirmed": True},
        )
        assert blocked.status_code == 409
        assert "all 20 frozen validation selections" in blocked.json()["detail"]
        assert "registered=1" in blocked.json()["detail"]

        for run in matrix["runs"][1:]:
            response = client.put(
                (
                    f"/api/lab/experiment-matrices/{matrix_id}/selections/"
                    f"{run['run_key']}"
                ),
                json={"confirmed": True},
            )
            assert response.status_code == 201
        matrix_evidence_path = Path(response.json()["matrix_evidence_path"])
        assert matrix_evidence_path == (
            tmp_path
            / "evidence"
            / "validation_selections"
            / matrix_id
            / "matrix_selections.json"
        )
        matrix_evidence_payload = json.loads(
            matrix_evidence_path.read_text(encoding="utf-8")
        )
        assert len(matrix_evidence_payload["selections"]) == 20
        selections = client.get(
            f"/api/lab/experiment-matrices/{matrix_id}/selections"
        ).json()
        assert len(selections) == 20
        assert {
            item["selection"]["checkpoint_fraction"] for item in selections
        } == {0.75}

        accepted = client.post(
            f"/api/lab/experiment-matrices/{matrix_id}/heldout",
            json={"confirmed": True},
        )
        assert accepted.status_code == 202
        assert accepted.json()["selection_count"] == 20
        assert accepted.json()["evaluation_run_count"] == 1
        assert Path(accepted.json()["selection_evidence_path"]) == (
            matrix_evidence_path
        )
        evaluation_run_id = accepted.json()["run_ids"][0]
        evaluation_run = client.get(
            f"/api/lab/runs/{evaluation_run_id}"
        ).json()
        assert Path(
            evaluation_run["config"]["selection_evidence_path"]
        ) == matrix_evidence_path
