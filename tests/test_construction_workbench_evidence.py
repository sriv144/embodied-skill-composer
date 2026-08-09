from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from embodied_skill_composer.construction.api import (
    _stream_run_events,
    _validated_local_origins,
    create_app,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5ProvenanceManifest,
)
from embodied_skill_composer.construction import workbench_evidence
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.workbench_evidence import (
    build_local_coppelia_summary,
)


def test_preview_origin_is_explicitly_allowed_and_remote_origins_are_rejected(
    tmp_path: Path,
) -> None:
    app = create_app(
        registry_path=tmp_path / "lab.sqlite",
        coppelia_evidence_root=tmp_path / "phase5",
    )
    registry = app.state.lab_registry
    run_id = registry.create_run(
        "training",
        {"seed": 7},
        status="completed",
        run_id="completed-run",
    )

    with TestClient(app) as client:
        preflight = client.options(
            "/api/project",
            headers={
                "origin": "http://127.0.0.1:4173",
                "access-control-request-method": "GET",
            },
        )
        assert preflight.status_code == 200
        assert (
            preflight.headers["access-control-allow-origin"]
            == "http://127.0.0.1:4173"
        )

        research = client.get("/api/lab/evidence/research-summary")
        assert research.status_code == 200
        assert research.json()["status"] == "absent"
        coppelia = client.get("/api/lab/evidence/coppelia")
        assert coppelia.status_code == 200
        assert coppelia.json()["ready"] is False

        with client.websocket_connect(
            f"/api/lab/runs/{run_id}/events/ws",
            headers={"origin": "http://127.0.0.1:4173"},
        ) as websocket:
            assert websocket.receive_json()["payload"]["event"] == "run_created"
            with pytest.raises(WebSocketDisconnect) as closed:
                websocket.receive_json()
            assert closed.value.code == 1000

        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                f"/api/lab/runs/{run_id}/events/ws",
                headers={"origin": "https://evil.example"},
            ):
                pass
        assert rejected.value.code == 1008

    assert _validated_local_origins(
        ["http://127.0.0.1:4173"]
    ) == frozenset({"http://127.0.0.1:4173"})
    for unsafe in (
        "https://evil.example:4173",
        "http://127.0.0.1",
        "http://user@127.0.0.1:4173",
        "http://127.0.0.1:4173/path",
    ):
        with pytest.raises(ValueError, match="explicit HTTP"):
            _validated_local_origins([unsafe])


def test_normal_browser_websocket_disconnect_is_not_an_error(
    tmp_path: Path,
) -> None:
    app = create_app(registry_path=tmp_path / "lab.sqlite")
    registry = app.state.lab_registry
    run_id = registry.create_run(
        "training",
        {"seed": 7},
        status="running",
        run_id="disconnect-run",
    )

    class DisconnectingSocket:
        async def send_json(self, _payload: object) -> None:
            raise WebSocketDisconnect(code=1001, reason="browser navigated away")

        async def close(self, *, code: int) -> None:
            del code

    asyncio.run(
        _stream_run_events(
            cast(WebSocket, DisconnectingSocket()),
            registry,
            run_id,
        )
    )


def test_artifact_api_returns_file_links_without_exposing_host_paths(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "private-host-path" / "run"
    artifact_root.mkdir(parents=True)
    (artifact_root / "report.md").write_text("# Evidence\n", encoding="utf-8")
    (artifact_root / "metrics.json").write_text("{}\n", encoding="utf-8")
    (artifact_root / "unsafe.html").write_text("<script />", encoding="utf-8")
    outside = tmp_path / "secret.json"
    outside.write_text('{"secret":true}\n', encoding="utf-8")

    app = create_app(registry_path=tmp_path / "lab.sqlite")
    registry = app.state.lab_registry
    run_id = registry.create_run(
        "training",
        {"seed": 7},
        status="completed",
        run_id="artifact-run",
    )
    registry.update_run(run_id, artifact_dir=str(artifact_root))
    registry.upsert_evaluation(
        "evaluation-1",
        matrix_id=None,
        split="test",
        payload={},
        artifact_dir=str(artifact_root),
    )

    with TestClient(app) as client:
        response = client.get(f"/api/lab/runs/{run_id}/artifacts")
        assert response.status_code == 200
        references = response.json()
        assert {item["path"] for item in references} == {
            "metrics.json",
            "report.md",
        }
        serialized = json.dumps(references)
        assert str(tmp_path) not in serialized
        assert all(
            item["href"].startswith(
                f"/api/lab/runs/{run_id}/artifacts/"
            )
            for item in references
        )
        report = client.get(
            f"/api/lab/runs/{run_id}/artifacts/report.md"
        )
        assert report.status_code == 200
        assert report.text.splitlines() == ["# Evidence"]
        assert report.headers["content-disposition"].startswith("attachment;")
        assert (
            client.get(
                f"/api/lab/runs/{run_id}/artifacts/../secret.json"
            ).status_code
            == 404
        )
        assert (
            client.get(
                f"/api/lab/runs/{run_id}/artifacts/unsafe.html"
            ).status_code
            == 404
        )
        evaluation_references = client.get(
            "/api/lab/evaluations/evaluation-1/artifacts"
        ).json()
        assert {item["path"] for item in evaluation_references} == {
            "metrics.json",
            "report.md",
        }
        assert (
            client.get(
                "/api/lab/evaluations/evaluation-1/artifacts/metrics.json"
            ).status_code
            == 200
        )


def test_coppelia_readiness_requires_matching_verified_nominal_and_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifests = {
        "nominal-run": _phase5_manifest("nominal", "nominal-run"),
        "recovery-run": _phase5_manifest(
            "unavailable_robot_recovery",
            "recovery-run",
        ),
    }

    def verify(run_dir: Path) -> Phase5ProvenanceManifest:
        return manifests[run_dir.name]

    monkeypatch.setattr(
        workbench_evidence,
        "verify_phase5_artifact_bundle",
        verify,
    )
    _write_passing_metrics(tmp_path / "nominal-run", "nominal")
    incomplete = build_local_coppelia_summary(tmp_path)
    assert incomplete["status"] == "absent"
    assert incomplete["ready"] is False
    assert incomplete["verified_scenarios"] == ["nominal"]

    _write_passing_metrics(
        tmp_path / "recovery-run",
        "unavailable_robot_recovery",
    )
    complete = build_local_coppelia_summary(tmp_path)
    assert complete["status"] == "validated"
    assert complete["ready"] is True
    assert complete["claim_allowed"] is False
    assert isinstance(complete["nominal"], dict)
    assert isinstance(complete["recovery"], dict)
    serialized = json.dumps(complete)
    assert str(tmp_path) not in serialized
    assert "/api/lab/coppelia/evidence/nominal-run/artifacts/" in serialized
    assert "/api/lab/coppelia/evidence/recovery-run/artifacts/" in serialized


def test_artifact_helpers_bound_filter_and_resolve_downloads(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    assert workbench_evidence.artifact_references(missing, "/downloads") == []

    root = tmp_path / "artifacts"
    nested = root / "validation" / "seed 800"
    nested.mkdir(parents=True)
    report = nested / "result #1.json"
    report.write_text('{"complete": true}\n', encoding="utf-8")
    checkpoint = root / "policy-v1.pt"
    checkpoint.write_bytes(b"checkpoint")
    (root / "active.html").write_text("<script />", encoding="utf-8")

    references = workbench_evidence.artifact_references(
        root,
        "/api/evidence/",
    )
    assert {item["path"] for item in references} == {
        "policy-v1.pt",
        "validation/seed 800/result #1.json",
    }
    nested_reference = next(
        item for item in references if item["path"].endswith(".json")
    )
    assert nested_reference == {
        "label": "validation / seed 800: Result #1",
        "href": (
            "/api/evidence/validation/seed%20800/result%20%231.json"
        ),
        "path": "validation/seed 800/result #1.json",
        "media_type": "application/json",
    }
    assert workbench_evidence.resolve_artifact(
        root,
        "validation/seed 800/result #1.json",
    ) == report.resolve()

    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = root / "linked.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pass
    else:
        assert "linked.json" not in {
            item["path"]
            for item in workbench_evidence.artifact_references(
                root,
                "/downloads",
            )
        }
        with pytest.raises(FileNotFoundError):
            workbench_evidence.resolve_artifact(root, "linked.json")

    for relative in (
        "../outside.json",
        "active.html",
        "not-present.json",
    ):
        with pytest.raises(FileNotFoundError):
            workbench_evidence.resolve_artifact(root, relative)
    with pytest.raises(FileNotFoundError):
        workbench_evidence.resolve_artifact(missing, "result.json")

    for index in range(205):
        (root / f"bounded-{index:03}.json").write_text("{}\n", encoding="utf-8")
    bounded = workbench_evidence.artifact_references(root, "/downloads")
    assert len(bounded) == 200
    assert all(item["href"].startswith("/downloads/") for item in bounded)


def test_research_summary_aggregates_complete_matrix_and_learning_curves(
    tmp_path: Path,
) -> None:
    curve_dir = tmp_path / "curve-run"
    curve_dir.mkdir()
    columns = (
        "transitions,mean_episode_return,rollout_terminal_fraction,"
        "loss_objective,loss_critic,loss_entropy"
    )
    rows = [columns]
    rows.extend(
        f"{index},{index / 10},0.95,1.0,2.0,0.1"
        for index in range(170)
    )
    rows.extend(
        (
            "bad,1.0,0.95,1.0,2.0,0.1",
            "171,not-a-number,inf,,bad,0.2",
        )
    )
    (curve_dir / "learning_curve.csv").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    evaluation_dir = tmp_path / "evaluation"
    evaluation_dir.mkdir()
    (evaluation_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    source_commit = "a" * 40
    runs: list[dict[str, object]] = [
        {
            "id": "curve-file",
            "run_key": "mappo_seed_7",
            "status": "completed",
            "config": {
                "algorithm": "mappo",
                "experiment_variant": "primary",
                "training_seed": 7,
            },
            "artifact_dir": str(curve_dir),
            "source_commit": source_commit,
        },
        {
            "id": "curve-events",
            "run_key": "ippo_seed_7",
            "status": "running",
            "config": {
                "algorithm": "ippo",
                "experiment_variant": "primary",
                "seed": 7,
            },
            "source_commit": source_commit,
        },
        {
            "id": "curve-no-config",
            "status": "cancel_requested",
            "config": "invalid",
            "source_commit": source_commit,
        },
        {
            "id": "",
            "status": "running",
            "config": {},
            "source_commit": source_commit,
        },
    ]
    runs.extend(
        {
            "id": f"completed-{index}",
            "status": "completed",
            "config": {},
            "source_commit": source_commit,
        }
        for index in range(16)
    )
    research_matrix = {
        "id": "research-v1",
        "execution_profile": "research",
        "protocol_digest": "protocol-digest",
        "selection_count": "20",
        "status_counts": {"completed": "20"},
        "runs": runs,
    }
    evaluation_payload: dict[str, object] = {
        "grid_validation": {"complete": True},
        "episodes": [{"seed": 900}, {"seed": 901}],
        "summaries": [{"controller": "mappo", "completion_mean": 0.96}],
        "per_training_seed": [{"training_seed": 7, "completion_mean": 0.95}],
        "per_scenario_seed": [{"scenario_seed": 900, "completion_mean": 1.0}],
    }
    completion = {
        "event": "matrix_evaluation_completed",
        "acceptance": {"all_primary_gates_pass": True},
        "ablations": [
            {"variant": "without_behavior_cloning", "supported": True},
            {"variant": "without_failure_curriculum", "supported": False},
        ],
    }
    registry = _EvidenceRegistry(
        matrices=[
            {"id": "smoke", "execution_profile": "smoke", "runs": []},
            research_matrix,
        ],
        evaluations=[
            {
                "id": "heldout-v1",
                "payload": evaluation_payload,
                "artifact_dir": str(evaluation_dir),
            }
        ],
        runs=[
            {"id": "unrelated", "kind": "training", "config": {}},
            {
                "id": "evaluation-run",
                "kind": "matrix_evaluation",
                "config": {"matrix_id": "research-v1"},
            },
        ],
        events={
            "evaluation-run": [
                {"event": "matrix_evaluation_started"},
                completion,
            ],
            "curve-events": [
                {"event": "run_started"},
                {
                    "event": "ppo_update",
                    "transitions": 100,
                    "mean_episode_return": 4.5,
                    "rollout_terminal_fraction": 0.9,
                },
                {"event": "ppo_update", "transitions": "invalid"},
            ],
            "curve-no-config": [
                {
                    "event": "ppo_update",
                    "transitions": 200,
                    "loss_critic": 1.25,
                }
            ],
        },
    )

    summary = workbench_evidence.build_local_research_summary(
        cast(LabRegistry, registry)
    )

    assert summary["status"] == "local_complete"
    assert summary["claim_allowed"] is False
    assert summary["matrix_id"] == "research-v1"
    assert summary["source_commit"] == source_commit
    assert summary["training_run_count"] == 20
    assert summary["selection_count"] == 20
    assert summary["heldout_episode_count"] == 2
    assert summary["acceptance"] == {"all_primary_gates_pass": True}
    assert len(cast(list[object], summary["ablations"])) == 2
    assert summary["confidence_intervals"] == evaluation_payload["summaries"]
    assert summary["per_training_seed"] == evaluation_payload["per_training_seed"]
    assert summary["per_scenario_seed"] == evaluation_payload["per_scenario_seed"]
    assert summary["artifact_references"] == [
        {
            "label": "Summary",
            "href": "/api/lab/evaluations/heldout-v1/artifacts/summary.json",
            "path": "summary.json",
            "media_type": "application/json",
        }
    ]

    curves = cast(list[dict[str, object]], summary["learning_curves"])
    assert len(curves) == 3
    file_curve = next(item for item in curves if item["run_key"] == "mappo_seed_7")
    assert file_curve["controller"] == "mappo"
    assert file_curve["training_seed"] == 7
    file_points = cast(list[dict[str, object]], file_curve["points"])
    assert len(file_points) == 160
    assert file_points[0]["transitions"] == 0
    assert file_points[-1]["transitions"] == 171
    assert "mean_episode_return" not in file_points[-1]

    event_curve = next(item for item in curves if item["run_key"] == "ippo_seed_7")
    assert event_curve["controller"] == "ippo"
    assert event_curve["training_seed"] == 7
    assert event_curve["points"] == [
        {
            "transitions": 100,
            "mean_episode_return": 4.5,
            "rollout_terminal_fraction": 0.9,
        }
    ]
    no_config_curve = next(
        item for item in curves if item["run_key"] == "curve-no-config"
    )
    assert no_config_curve["controller"] is None
    assert no_config_curve["training_seed"] is None


def test_research_summary_handles_absent_and_malformed_local_state(
    tmp_path: Path,
) -> None:
    absent = workbench_evidence.build_local_research_summary(
        cast(LabRegistry, _EvidenceRegistry())
    )
    assert absent["status"] == "absent"
    assert absent["learning_curves"] == []

    registry = _EvidenceRegistry(
        matrices=[
            {
                "id": "smoke-only",
                "execution_profile": "smoke",
                "protocol_digest": None,
                "selection_count": "not-an-integer",
                "status_counts": "invalid",
                "runs": [
                    {
                        "id": "completed",
                        "status": "completed",
                        "source_commit": "a" * 40,
                    },
                    {
                        "id": "failed",
                        "status": "failed",
                        "source_commit": "b" * 40,
                    },
                ],
            }
        ],
        runs=[
            {
                "id": "wrong-matrix",
                "kind": "matrix_evaluation",
                "config": {"matrix_id": "another"},
            },
            {
                "id": "bad-config",
                "kind": "matrix_evaluation",
                "config": "invalid",
            },
        ],
    )
    summary = workbench_evidence.build_local_research_summary(
        cast(LabRegistry, registry)
    )
    assert summary["status"] == "in_progress"
    assert summary["training_run_count"] == 2
    assert summary["selection_count"] == 0
    assert summary["source_commit"] is None
    assert summary["heldout_episode_count"] == 0
    assert summary["confidence_intervals"] == []
    assert summary["per_training_seed"] == []
    assert summary["per_scenario_seed"] == []
    assert summary["acceptance"] is None
    assert summary["artifact_references"] == []

    malformed_evaluation = _EvidenceRegistry(
        matrices=[
            {
                "id": "malformed-evaluation",
                "execution_profile": "research",
                "selection_count": None,
                "runs": "not-a-list",
            }
        ],
        evaluations=[
            {
                "id": "malformed",
                "payload": "not-an-object",
                "artifact_dir": str(tmp_path / "missing-artifacts"),
            }
        ],
    )
    malformed_summary = workbench_evidence.build_local_research_summary(
        cast(LabRegistry, malformed_evaluation)
    )
    assert malformed_summary["status"] == "in_progress"
    assert malformed_summary["training_run_count"] == 0
    assert malformed_summary["artifact_references"] == []


def test_learning_curve_loaders_and_summary_helpers_reject_bad_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert workbench_evidence._curve_points_from_file({}) == []
    assert workbench_evidence._curve_points_from_file(
        {"artifact_dir": str(tmp_path / "missing")}
    ) == []
    assert workbench_evidence._curve_points_from_events(
        cast(LabRegistry, _EvidenceRegistry()),
        "",
    ) == []

    assert workbench_evidence._curve_point({"transitions": -1}) is None
    point = workbench_evidence._curve_point(
        {
            "transitions": "12",
            "mean_episode_return": "3.5",
            "rollout_terminal_fraction": "",
            "loss_objective": None,
            "loss_critic": "not-a-number",
            "loss_entropy": float("inf"),
        }
    )
    assert point == {"transitions": 12, "mean_episode_return": 3.5}
    short_points = [{"transitions": index} for index in range(4)]
    assert workbench_evidence._downsample(short_points) is short_points

    curve_dir = tmp_path / "unreadable-curve"
    curve_dir.mkdir()
    curve_path = curve_dir / "learning_curve.csv"
    curve_path.write_text("transitions\n1\n", encoding="utf-8")
    original_open = Path.open

    def fail_curve_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        if path == curve_path:
            raise OSError("fixture denies reading")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_curve_open)
    assert workbench_evidence._curve_points_from_file(
        {"artifact_dir": str(curve_dir)}
    ) == []

    object_path = tmp_path / "object.json"
    object_path.write_text('{"value": 1}\n', encoding="utf-8")
    list_path = tmp_path / "list.json"
    list_path.write_text("[]\n", encoding="utf-8")
    assert workbench_evidence._read_object(object_path) == {"value": 1}
    with pytest.raises(ValueError, match="must contain a JSON object"):
        workbench_evidence._read_object(list_path)

    assert workbench_evidence._object_list([{"value": 1}]) == [{"value": 1}]
    assert workbench_evidence._object_list([{"value": 1}, "bad"]) == []
    assert workbench_evidence._object_list("bad") == []
    assert workbench_evidence._list_length([1, 2]) == 2
    assert workbench_evidence._list_length("not-a-list") == 0
    assert workbench_evidence._integer(3.9, 0) == 3
    assert workbench_evidence._integer("bad", 7) == 7
    assert workbench_evidence._matrix_source_commit(
        [{"source_commit": "abc"}, {"source_commit": "abc"}]
    ) == "abc"
    assert workbench_evidence._matrix_source_commit(
        [{"source_commit": "abc"}, {"source_commit": "def"}]
    ) is None
    assert workbench_evidence._matrix_source_commit(
        [{"source_commit": None}, {"source_commit": 123}]
    ) is None
    assert workbench_evidence._artifact_label("report-file.md") == "Report File"
    assert (
        workbench_evidence._artifact_label("one/two/metric_table.csv")
        == "one / two: Metric Table"
    )
    assert workbench_evidence._media_type(".png") == "image/png"


def test_coppelia_loader_skips_invalid_bundles_and_protects_run_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert build_local_coppelia_summary(tmp_path / "missing")["ready"] is False

    root = tmp_path / "phase5"
    root.mkdir()
    (root / "ordinary-file").write_text("ignored\n", encoding="utf-8")
    (root / "no-manifest").mkdir()
    manifests = {
        "invalid-verification": _phase5_manifest(
            "nominal",
            "invalid-verification",
        ),
        "missing-metrics": _phase5_manifest("nominal", "missing-metrics"),
        "non-object-metrics": _phase5_manifest(
            "nominal",
            "non-object-metrics",
        ),
        "failed-gate": _phase5_manifest("nominal", "failed-gate"),
        "good": _phase5_manifest("nominal", "good"),
    }
    for name in manifests:
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "non-object-metrics" / "metrics.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    failed_metrics = _passing_metrics_payload("nominal")
    cast(dict[str, object], failed_metrics["acceptance"])[
        "all_modules_installed"
    ] = False
    (root / "failed-gate" / "metrics.json").write_text(
        json.dumps(failed_metrics),
        encoding="utf-8",
    )

    def verify(run_dir: Path) -> Phase5ProvenanceManifest:
        if run_dir.name == "invalid-verification":
            raise ValueError("invalid bundle")
        return manifests[run_dir.name]

    monkeypatch.setattr(
        workbench_evidence,
        "verify_phase5_artifact_bundle",
        verify,
    )

    summary = build_local_coppelia_summary(root)
    assert summary["ready"] is False
    assert summary["verified_scenarios"] == []
    assert workbench_evidence.phase5_run_directory(root, "good") == (
        root / "good"
    ).resolve()

    for unsafe in ("", "..", ".", "../outside", "nested/run", r"nested\run"):
        with pytest.raises(FileNotFoundError):
            workbench_evidence.phase5_run_directory(root, unsafe)
    with pytest.raises(ValueError, match="invalid bundle"):
        workbench_evidence.phase5_run_directory(
            root,
            "invalid-verification",
        )


def test_live_bundle_gate_checks_every_required_claim_and_pair_key(
    tmp_path: Path,
) -> None:
    manifest = _phase5_manifest("nominal", "nominal")
    metrics = _passing_metrics_payload("nominal")
    assert workbench_evidence._passing_live_bundle(manifest, metrics)

    manifest_updates: list[dict[str, object]] = [
        {"evidence_kind": "fixture"},
        {"live_evidence": False},
        {"run_status": "failed"},
        {"live_gate_passed": False},
        {"approval_gate_confirmed": False},
    ]
    for update in manifest_updates:
        assert not workbench_evidence._passing_live_bundle(
            manifest.model_copy(update=update),
            metrics,
        )

    metric_updates: list[dict[str, object]] = [
        {"schema_version": "wrong"},
        {"scenario": "unavailable_robot_recovery"},
        {"evidence_kind": "fixture"},
        {"status": "failed"},
        {"live_gate_passed": False},
        {"acceptance": []},
        {"acceptance": {}},
        {"acceptance": {"all_modules_installed": False}},
    ]
    for update in metric_updates:
        candidate = {**metrics, **update}
        assert not workbench_evidence._passing_live_bundle(
            manifest,
            candidate,
        )

    recovery_other_seed = _phase5_manifest(
        "unavailable_robot_recovery",
        "recovery-other-seed",
        scenario_seed=901,
    )
    unmatched = [
        (tmp_path / "nominal", manifest, metrics),
        (
            tmp_path / "recovery-other-seed",
            recovery_other_seed,
            _passing_metrics_payload("unavailable_robot_recovery"),
        ),
    ]
    assert workbench_evidence._matching_coppelia_pair(unmatched) is None

    recovery = _phase5_manifest(
        "unavailable_robot_recovery",
        "recovery",
    )
    duplicate_nominal = _phase5_manifest("nominal", "nominal-duplicate")
    matching = [
        *unmatched,
        (
            tmp_path / "nominal-duplicate",
            duplicate_nominal,
            metrics,
        ),
        (
            tmp_path / "recovery",
            recovery,
            _passing_metrics_payload("unavailable_robot_recovery"),
        ),
    ]
    pair = workbench_evidence._matching_coppelia_pair(matching)
    assert pair is not None
    assert pair[0][0] == tmp_path / "nominal"
    assert pair[1][0] == tmp_path / "recovery"


class _Payload:
    def __init__(self, value: dict[str, object]) -> None:
        self._value = value

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self._value


class _Envelope:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = _Payload(payload)


class _EvidenceRegistry:
    def __init__(
        self,
        *,
        matrices: list[dict[str, object]] | None = None,
        evaluations: list[dict[str, object]] | None = None,
        runs: list[dict[str, object]] | None = None,
        events: dict[str, list[dict[str, object]]] | None = None,
    ) -> None:
        self._matrices = matrices or []
        self._evaluations = evaluations or []
        self._runs = runs or []
        self._events = events or {}

    def list_experiment_matrices(self) -> list[dict[str, object]]:
        return self._matrices

    def list_evaluations(
        self,
        *,
        matrix_id: str | None = None,
    ) -> list[dict[str, object]]:
        del matrix_id
        return self._evaluations

    def list_runs(self, *, limit: int = 100) -> list[dict[str, object]]:
        assert limit == 500
        return self._runs

    def list_event_envelopes(self, run_id: str) -> list[_Envelope]:
        return [_Envelope(payload) for payload in self._events.get(run_id, [])]


def _phase5_manifest(
    scenario: str,
    run_id: str,
    *,
    scenario_seed: int = 900,
) -> Phase5ProvenanceManifest:
    return Phase5ProvenanceManifest.model_validate(
        {
            "run_id": run_id,
            "generated_at": "2026-07-28T00:00:00Z",
            "evidence_kind": "live_coppelia",
            "live_evidence": True,
            "run_status": "completed",
            "live_gate_passed": True,
            "approval_gate_confirmed": True,
            "scenario": scenario,
            "scenario_id": "cottage-v1-seed-900",
            "scenario_seed": scenario_seed,
            "source_commit": "1" * 40,
            "source_dirty": False,
            "source_tree_digest": "2" * 64,
            "plan_digest": "3" * 64,
            "configuration_digest": "4" * 64,
            "simulator_version": "CoppeliaSim fixture",
            "payload_transport_model": "logical_carrier",
            "limitations": ["Payload transport is logical."],
            "artifacts": [],
        }
    )


def _write_passing_metrics(root: Path, scenario: str) -> None:
    root.mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(_passing_metrics_payload(scenario)),
        encoding="utf-8",
    )


def _passing_metrics_payload(scenario: str) -> dict[str, object]:
    return {
        "schema_version": "construction_intelligence.coppelia_evidence.v1",
        "scenario": scenario,
        "evidence_kind": "live_coppelia",
        "status": "completed",
        "live_gate_passed": True,
        "acceptance": {
            "all_modules_installed": True,
            "zero_post_start_pose_writes": True,
        },
    }
