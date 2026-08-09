from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from math import isfinite
from pathlib import Path
from typing import cast
from urllib.parse import quote

from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5ProvenanceManifest,
    verify_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.lab_registry import LabRegistry


RESEARCH_SUMMARY_SCHEMA = "construction-intelligence-research-summary-v1"
COPPELIA_SUMMARY_SCHEMA = "construction-intelligence-coppelia-public-v1"
_MAX_ARTIFACT_REFERENCES = 200
_MAX_CURVE_POINTS = 160
_DOWNLOADABLE_SUFFIXES = {
    ".csv",
    ".glb",
    ".json",
    ".jsonl",
    ".md",
    ".mp4",
    ".onnx",
    ".png",
    ".pt",
    ".ttt",
}


def artifact_references(root: Path, href_prefix: str) -> list[dict[str, str]]:
    """Return bounded, concrete download links without exposing the host path."""

    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        return []
    references: list[dict[str, str]] = []
    for candidate in sorted(resolved_root.rglob("*")):
        if not candidate.is_file() or candidate.suffix.lower() not in _DOWNLOADABLE_SUFFIXES:
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(resolved_root):
            continue
        relative = resolved.relative_to(resolved_root).as_posix()
        references.append(
            {
                "label": _artifact_label(relative),
                "href": f"{href_prefix.rstrip('/')}/{quote(relative, safe='/')}",
                "path": relative,
                "media_type": _media_type(resolved.suffix.lower()),
            }
        )
        if len(references) >= _MAX_ARTIFACT_REFERENCES:
            break
    return references


def resolve_artifact(root: Path, relative_path: str) -> Path:
    """Resolve a downloadable artifact and reject traversal, links, and active content."""

    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if (
        not resolved_root.is_dir()
        or not candidate.is_relative_to(resolved_root)
        or not candidate.is_file()
        or candidate.suffix.lower() not in _DOWNLOADABLE_SUFFIXES
    ):
        raise FileNotFoundError(relative_path)
    return candidate


def build_local_research_summary(registry: LabRegistry) -> dict[str, object]:
    """Build a refreshable local view without promoting it to a release claim."""

    matrices = registry.list_experiment_matrices()
    matrix = next(
        (
            item
            for item in matrices
            if item.get("execution_profile") == "research"
        ),
        matrices[0] if matrices else None,
    )
    if matrix is None:
        return {
            "schema_version": RESEARCH_SUMMARY_SCHEMA,
            "status": "absent",
            "claim_allowed": False,
            "reason": "No experiment matrix is registered in this local lab.",
            "confidence_intervals": [],
            "per_training_seed": [],
            "per_scenario_seed": [],
            "learning_curves": [],
            "acceptance": None,
            "ablations": [],
            "artifact_references": [],
        }

    matrix_id = str(matrix["id"])
    evaluations = registry.list_evaluations(matrix_id=matrix_id)
    evaluation = evaluations[0] if evaluations else None
    evaluation_payload = (
        cast(dict[str, object], evaluation["payload"])
        if evaluation is not None and isinstance(evaluation.get("payload"), dict)
        else {}
    )
    completion = _matrix_completion_event(registry, matrix_id)
    acceptance = completion.get("acceptance") if completion else None
    ablations = completion.get("ablations", []) if completion else []
    runs_value = matrix.get("runs")
    runs = (
        cast(list[dict[str, object]], runs_value)
        if isinstance(runs_value, list)
        and all(isinstance(item, dict) for item in runs_value)
        else []
    )
    status_counts = matrix.get("status_counts")
    completed_count = (
        _integer(cast(dict[str, object], status_counts).get("completed"), 0)
        if isinstance(status_counts, dict)
        else sum(item.get("status") == "completed" for item in runs)
    )
    selection_count = _integer(matrix.get("selection_count"), 0)
    grid = evaluation_payload.get("grid_validation")
    grid_complete = isinstance(grid, dict) and grid.get("complete") is True
    local_complete = (
        matrix.get("execution_profile") == "research"
        and len(runs) == 20
        and completed_count == 20
        and selection_count == 20
        and grid_complete
        and isinstance(acceptance, dict)
        and isinstance(ablations, list)
        and len(ablations) == 2
    )
    evaluation_id = str(evaluation["id"]) if evaluation is not None else None
    references = (
        artifact_references(
            Path(str(evaluation["artifact_dir"])),
            f"/api/lab/evaluations/{quote(evaluation_id or '', safe='')}/artifacts",
        )
        if evaluation is not None
        else []
    )
    return {
        "schema_version": RESEARCH_SUMMARY_SCHEMA,
        "status": "local_complete" if local_complete else "in_progress",
        "claim_allowed": False,
        "reason": (
            "Local evidence is complete, but only the hash-pinned release export "
            "may make a canonical public claim."
            if local_complete
            else "The local matrix, validation selections, or held-out evaluation is incomplete."
        ),
        "source_commit": _matrix_source_commit(runs),
        "protocol_digest": matrix.get("protocol_digest"),
        "matrix_id": matrix_id,
        "training_run_count": len(runs),
        "selection_count": selection_count,
        "heldout_episode_count": _list_length(evaluation_payload.get("episodes")),
        "confidence_intervals": _object_list(evaluation_payload.get("summaries")),
        "per_training_seed": _object_list(
            evaluation_payload.get("per_training_seed")
        ),
        "per_scenario_seed": _object_list(
            evaluation_payload.get("per_scenario_seed")
        ),
        "learning_curves": _learning_curves(registry, runs),
        "acceptance": acceptance,
        "ablations": ablations if isinstance(ablations, list) else [],
        "artifact_references": references,
        "fidelity_boundary": (
            "Local learned-policy results are event-simulator evidence. They do "
            "not prove dynamic CoppeliaSim execution or physical grasping."
        ),
    }


def build_local_coppelia_summary(
    root: Path,
) -> dict[str, object]:
    """Require independently verified nominal and recovery native bundles."""

    verified: list[tuple[Path, Phase5ProvenanceManifest, dict[str, object]]] = []
    if root.is_dir():
        for run_dir in sorted(root.iterdir(), reverse=True):
            if not run_dir.is_dir() or not (run_dir / "manifest.json").is_file():
                continue
            try:
                manifest = verify_phase5_artifact_bundle(run_dir)
                metrics = _read_object(run_dir / "metrics.json")
            except (OSError, ValueError):
                continue
            if _passing_live_bundle(manifest, metrics):
                verified.append((run_dir.resolve(), manifest, metrics))

    pair = _matching_coppelia_pair(verified)
    if pair is None:
        scenarios = sorted({item[1].scenario for item in verified})
        return {
            "schema_version": COPPELIA_SUMMARY_SCHEMA,
            "status": "absent",
            "ready": False,
            "claim_allowed": False,
            "payload_transport": "logical",
            "reason": (
                "Validated live nominal and unavailable-robot recovery manifests "
                "from the same source are both required."
            ),
            "verified_scenarios": scenarios,
            "nominal": None,
            "recovery": None,
            "limitation": (
                "Payload transport is logical. No arm, gripper, grasp-contact, "
                "or payload dynamics are claimed."
            ),
        }

    nominal, recovery = pair
    return {
        "schema_version": COPPELIA_SUMMARY_SCHEMA,
        "status": "validated",
        "ready": True,
        "claim_allowed": False,
        "source_commit": nominal[1].source_commit,
        "payload_transport": "logical",
        "nominal": _local_coppelia_run("nominal", nominal),
        "recovery": _local_coppelia_run("recovery", recovery),
        "limitation": (
            "Payload transport is logical. No arm, gripper, grasp-contact, "
            "or payload dynamics are claimed."
        ),
    }


def phase5_run_directory(root: Path, run_id: str) -> Path:
    """Resolve a direct child evidence directory without accepting host paths."""

    resolved_root = root.resolve()
    candidate = (resolved_root / run_id).resolve()
    if (
        not run_id
        or "/" in run_id
        or "\\" in run_id
        or not candidate.is_relative_to(resolved_root)
        or candidate.parent != resolved_root
    ):
        raise FileNotFoundError(run_id)
    verify_phase5_artifact_bundle(candidate)
    return candidate


def _matrix_completion_event(
    registry: LabRegistry,
    matrix_id: str,
) -> dict[str, object] | None:
    matching = [
        item
        for item in registry.list_runs(limit=500)
        if item.get("kind") == "matrix_evaluation"
        and isinstance(item.get("config"), dict)
        and cast(dict[str, object], item["config"]).get("matrix_id") == matrix_id
    ]
    for run in matching:
        events = registry.list_event_envelopes(str(run["id"]))
        for envelope in reversed(events):
            payload = envelope.payload.model_dump(mode="json")
            if payload.get("event") == "matrix_evaluation_completed":
                return payload
    return None


def _learning_curves(
    registry: LabRegistry,
    runs: list[dict[str, object]],
) -> list[dict[str, object]]:
    series: list[dict[str, object]] = []
    for run in runs:
        points = _curve_points_from_file(run)
        if not points and run.get("status") in {"running", "cancel_requested"}:
            points = _curve_points_from_events(registry, str(run.get("id", "")))
        if not points:
            continue
        config = (
            cast(dict[str, object], run["config"])
            if isinstance(run.get("config"), dict)
            else {}
        )
        series.append(
            {
                "run_key": str(run.get("run_key") or run.get("id") or ""),
                "controller": config.get("algorithm"),
                "experiment_variant": config.get("experiment_variant"),
                "training_seed": config.get("training_seed", config.get("seed")),
                "status": run.get("status"),
                "points": _downsample(points),
            }
        )
    return series


def _curve_points_from_file(run: dict[str, object]) -> list[dict[str, float | int]]:
    artifact_dir = run.get("artifact_dir")
    if not isinstance(artifact_dir, str) or not artifact_dir:
        return []
    path = Path(artifact_dir) / "learning_curve.csv"
    if not path.is_file():
        return []
    points: list[dict[str, float | int]] = []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                point = _curve_point(row)
                if point is not None:
                    points.append(point)
    except OSError:
        return []
    return points


def _curve_points_from_events(
    registry: LabRegistry,
    run_id: str,
) -> list[dict[str, float | int]]:
    if not run_id:
        return []
    points: list[dict[str, float | int]] = []
    for envelope in registry.list_event_envelopes(run_id):
        payload = envelope.payload.model_dump(mode="json")
        if payload.get("event") != "ppo_update":
            continue
        point = _curve_point(payload)
        if point is not None:
            points.append(point)
    return points


def _curve_point(
    row: dict[str, object],
) -> dict[str, float | int] | None:
    transitions = _integer(row.get("transitions"), -1)
    if transitions < 0:
        return None
    point: dict[str, float | int] = {"transitions": transitions}
    for key in (
        "mean_episode_return",
        "rollout_terminal_fraction",
        "loss_objective",
        "loss_critic",
        "loss_entropy",
    ):
        value = row.get(key)
        try:
            if value is not None and value != "":
                parsed = float(cast(str | int | float, value))
                if isfinite(parsed):
                    point[key] = parsed
        except (TypeError, ValueError):
            continue
    return point


def _downsample(
    points: list[dict[str, float | int]],
) -> list[dict[str, float | int]]:
    if len(points) <= _MAX_CURVE_POINTS:
        return points
    indices = {
        round(index * (len(points) - 1) / (_MAX_CURVE_POINTS - 1))
        for index in range(_MAX_CURVE_POINTS)
    }
    return [points[index] for index in sorted(indices)]


def _passing_live_bundle(
    manifest: Phase5ProvenanceManifest,
    metrics: dict[str, object],
) -> bool:
    acceptance = metrics.get("acceptance")
    return (
        manifest.evidence_kind == "live_coppelia"
        and manifest.live_evidence
        and manifest.run_status == "completed"
        and manifest.live_gate_passed
        and manifest.approval_gate_confirmed
        and metrics.get("schema_version")
        == "construction_intelligence.coppelia_evidence.v1"
        and metrics.get("scenario") == manifest.scenario
        and metrics.get("evidence_kind") == "live_coppelia"
        and metrics.get("status") == "completed"
        and metrics.get("live_gate_passed") is True
        and isinstance(acceptance, dict)
        and bool(acceptance)
        and all(value is True for value in acceptance.values())
    )


def _matching_coppelia_pair(
    verified: Iterable[
        tuple[Path, Phase5ProvenanceManifest, dict[str, object]]
    ],
) -> tuple[
    tuple[Path, Phase5ProvenanceManifest, dict[str, object]],
    tuple[Path, Phase5ProvenanceManifest, dict[str, object]],
] | None:
    grouped: dict[
        tuple[str, str, str, int],
        dict[
            str,
            tuple[Path, Phase5ProvenanceManifest, dict[str, object]],
        ],
    ] = {}
    for item in verified:
        manifest = item[1]
        key = (
            manifest.source_commit,
            manifest.source_tree_digest,
            manifest.plan_digest,
            manifest.scenario_seed,
        )
        grouped.setdefault(key, {}).setdefault(manifest.scenario, item)
    for key in sorted(grouped, reverse=True):
        candidates = grouped[key]
        nominal = candidates.get("nominal")
        recovery = candidates.get("unavailable_robot_recovery")
        if nominal is not None and recovery is not None:
            return nominal, recovery
    return None


def _local_coppelia_run(
    prefix: str,
    item: tuple[Path, Phase5ProvenanceManifest, dict[str, object]],
) -> dict[str, object]:
    run_dir, manifest, metrics = item
    href_prefix = (
        f"/api/lab/coppelia/evidence/{quote(run_dir.name, safe='')}/artifacts"
    )
    return {
        "manifest": manifest.model_dump(mode="json"),
        "metrics": metrics,
        "artifact_references": artifact_references(run_dir, href_prefix),
        "label": prefix,
    }


def _read_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _object_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return []
    return cast(list[dict[str, object]], value)


def _list_length(value: object) -> int:
    return len(value) if isinstance(value, list) else 0


def _integer(value: object, default: int) -> int:
    try:
        return int(cast(str | int | float, value))
    except (TypeError, ValueError):
        return default


def _matrix_source_commit(runs: list[dict[str, object]]) -> str | None:
    values = {
        str(item.get("source_commit"))
        for item in runs
        if isinstance(item.get("source_commit"), str) and item.get("source_commit")
    }
    return next(iter(values)) if len(values) == 1 else None


def _artifact_label(relative: str) -> str:
    path = Path(relative)
    parent = " / ".join(path.parts[-3:-1])
    stem = path.stem.replace("_", " ").replace("-", " ").strip().title()
    return f"{parent}: {stem}" if parent else stem


def _media_type(suffix: str) -> str:
    return {
        ".csv": "text/csv",
        ".glb": "model/gltf-binary",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".md": "text/markdown",
        ".mp4": "video/mp4",
        ".onnx": "application/octet-stream",
        ".png": "image/png",
        ".pt": "application/octet-stream",
        ".ttt": "application/octet-stream",
    }[suffix]
