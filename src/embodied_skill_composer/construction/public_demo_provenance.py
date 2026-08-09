from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Literal, cast

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)

from embodied_skill_composer.construction.evaluation import (
    ControllerEvaluation,
    ControllerName,
    EpisodeEvaluation,
    EvaluationSuite,
    MetricSummary,
    render_evaluation_report,
    summarize_by_scenario_seed,
    summarize_by_training_seed,
)
from embodied_skill_composer.construction.experiment_execution import (
    PrimaryAcceptanceAudit,
    audit_primary_acceptance,
)
from embodied_skill_composer.construction.experiment_protocol import (
    AblationDecision,
    CheckpointValidationResult,
    SelectedCheckpoint,
    load_experiment_protocol,
    protocol_digest,
    select_validation_checkpoint,
)
from embodied_skill_composer.construction.release_identity import (
    validate_release_identity,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5ProvenanceManifest,
    verify_phase5_artifact_bundle,
)


PROVENANCE_SCHEMA_VERSION = "construction-intelligence-public-demo-provenance-v1"
BUNDLE_SCHEMA_VERSION = "construction-intelligence-public-demo-input-v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
_PROVENANCE_NAME = "provenance.json"
_GENERATED_AT = "1970-01-01T00:00:00Z"
_RESEARCH_ROLES = {
    "matrix",
    "selections",
    "evaluation",
    "episodes",
    "report",
    "acceptance",
    "ablations",
    "reproducibility_audit",
    "release_completeness",
}
_RESEARCH_ARTIFACT_CONTRACTS = {
    "matrix": ("evidence/research/matrix.json", "application/json"),
    "selections": ("evidence/research/selections.json", "application/json"),
    "evaluation": ("evidence/research/evaluation.json", "application/json"),
    "episodes": ("evidence/research/episodes.csv", "text/csv"),
    "report": ("evidence/research/report.md", "text/markdown"),
    "acceptance": ("evidence/research/acceptance.json", "application/json"),
    "ablations": ("evidence/research/ablations.json", "application/json"),
    "reproducibility_audit": (
        "evidence/research/reproducibility_audit.json",
        "application/json",
    ),
    "release_completeness": (
        "evidence/research/release_completeness.json",
        "application/json",
    ),
}
_SIMULATOR_FILENAMES = {
    "manifest": ("manifest.json", "application/json"),
    "scenario": ("scenario.json", "application/json"),
    "planned_jobs": ("planned_jobs.json", "application/json"),
    "replay": ("planned_vs_measured_replay.json", "application/json"),
    "wheel_commands": ("wheel_commands.jsonl", "application/x-ndjson"),
    "telemetry": ("measured_telemetry.jsonl", "application/x-ndjson"),
    "trace": ("trace.json", "application/json"),
    "metrics": ("metrics.json", "application/json"),
    "report": ("report.md", "text/markdown"),
    "scene": ("construction_intelligence.ttt", "application/octet-stream"),
}
_SIMULATOR_ROLES = {
    f"{scenario}_{artifact}"
    for scenario in ("nominal", "recovery")
    for artifact in _SIMULATOR_FILENAMES
}
_SIMULATOR_ARTIFACT_CONTRACTS = {
    f"{scenario}_{artifact}": (
        f"evidence/coppelia/{scenario}/{file_name}",
        media_type,
    )
    for scenario in ("nominal", "recovery")
    for artifact, (file_name, media_type) in _SIMULATOR_FILENAMES.items()
}
_RELEASE_DETERMINISTIC_TARGETS = {
    "project": "project.json",
    "scenarios": "scenarios.json",
    "policies": "policies.json",
    "runs": "runs.json",
    "report": "report.md",
    "house": "house.glb",
    "robot": "construction_robot.glb",
    "trace_sequential": "traces/sequential.json",
    "trace_greedy": "traces/greedy.json",
    "trace_optimized": "traces/optimized.json",
    "trace_recovery": "traces/recovery.json",
}
_RELEASE_DETERMINISTIC_ROLES = set(_RELEASE_DETERMINISTIC_TARGETS)
_EPISODE_LIST_ADAPTER = TypeAdapter(list[EpisodeEvaluation])
_EVALUATION_METRICS = (
    "structure_completion_rate",
    "makespan_s",
    "total_travel_m",
    "total_energy_wh",
    "idle_robot_seconds",
    "mean_robot_utilization",
    "collision_count",
    "wasted_work_s",
    "invalid_bid_count",
)


class PublicDemoExportError(ValueError):
    """Raised when evidence cannot support the requested public-demo channel."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceIdentity(_FrozenModel):
    commit: str = Field(pattern=_GIT_COMMIT_PATTERN)
    dirty: bool
    tree_digest: str = Field(min_length=1)


class BundleArtifact(_FrozenModel):
    role: str = Field(min_length=1)
    path: str = Field(min_length=1)
    target: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    media_type: str | None = None

    @model_validator(mode="after")
    def validate_target(self) -> BundleArtifact:
        _safe_relative_path(self.target)
        if self.target == _PROVENANCE_NAME:
            raise ValueError(f"{_PROVENANCE_NAME} is reserved for generated provenance")
        return self


class EvidenceBundleManifest(_FrozenModel):
    schema_version: Literal["construction-intelligence-public-demo-input-v1"]
    kind: Literal["deterministic", "research", "simulator"]
    evidence_status: Literal["fixture", "canonical"]
    source: SourceIdentity
    created_at: str = Field(min_length=1)
    configuration_digests: list[str] = Field(default_factory=list)
    protocol_digest: str | None = None
    profile: Literal["unit", "smoke", "research"] | None = None
    matrix_id: str | None = None
    artifacts: list[BundleArtifact] = Field(min_length=1)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        if parsed.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> EvidenceBundleManifest:
        roles = [item.role for item in self.artifacts]
        targets = [item.target for item in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("bundle artifact roles must be unique")
        if len(targets) != len(set(targets)):
            raise ValueError("bundle artifact targets must be unique")
        for digest in self.configuration_digests:
            _require_sha256(digest, "configuration digest")
        if self.protocol_digest is not None:
            _require_sha256(self.protocol_digest, "protocol digest")
        if self.kind == "research" and (
            self.profile is None
            or self.matrix_id is None
            or self.protocol_digest is None
        ):
            raise ValueError(
                "research bundles require profile, matrix_id, and protocol_digest"
            )
        if self.kind != "research" and (
            self.profile is not None or self.matrix_id is not None
        ):
            raise ValueError("only research bundles may declare profile or matrix_id")
        return self


@dataclass(frozen=True)
class _LoadedBundle:
    path: Path
    sha256: str
    manifest: EvidenceBundleManifest
    files: dict[str, Path]


@dataclass(frozen=True)
class _ValidatedResearch:
    matrix: dict[str, object]
    selection_records: list[dict[str, object]]
    suite: EvaluationSuite
    acceptance: PrimaryAcceptanceAudit
    ablations: list[AblationDecision]
    reproducibility_audit: dict[str, object]
    release_completeness: dict[str, object]


@dataclass(frozen=True)
class _ValidatedSimulator:
    nominal_manifest: Phase5ProvenanceManifest
    recovery_manifest: Phase5ProvenanceManifest
    nominal_metrics: dict[str, object]
    recovery_metrics: dict[str, object]


def export_public_demo_bundle(
    output_dir: Path,
    *,
    deterministic_bundle: Path,
    source: SourceIdentity,
    channel: Literal["preview", "release"] = "preview",
    research_bundle: Path | None = None,
    simulator_bundle: Path | None = None,
    release_version: str | None = None,
    release_tag: str | None = None,
) -> dict[str, object]:
    """Build a deterministic public bundle from hash-pinned evidence descriptors."""

    _validate_release_metadata(channel, release_version, release_tag)
    deterministic = _load_bundle(deterministic_bundle, expected_kind="deterministic")
    research = (
        _load_bundle(research_bundle, expected_kind="research")
        if research_bundle is not None
        else None
    )
    simulator = (
        _load_bundle(simulator_bundle, expected_kind="simulator")
        if simulator_bundle is not None
        else None
    )
    validated_research = _validate_research(research) if research is not None else None
    validated_simulator = (
        _validate_simulator(simulator) if simulator is not None else None
    )
    _validate_export_claims(
        channel,
        source,
        deterministic,
        research,
        simulator,
        validated_research,
        validated_simulator,
    )

    destination = output_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        artifact_sources: dict[str, tuple[str, str]] = {}
        _copy_bundle(deterministic, staging, artifact_sources)
        if research is not None and validated_research is not None:
            _copy_bundle(research, staging, artifact_sources)
            _write_research_views(
                staging,
                research,
                validated_research,
                artifact_sources,
            )
        else:
            _write_json(
                staging / "experiment-matrices.json",
                [],
            )
            _write_json(staging / "evaluations.json", [])
            _write_json(
                staging / "research-summary.json",
                {
                    "schema_version": "construction-intelligence-research-summary-v1",
                    "status": "absent",
                    "claim_allowed": False,
                    "reason": (
                        "No canonical 20-run research bundle was supplied. "
                        "Fixture, smoke, and unit evidence are not release claims."
                    ),
                    "confidence_intervals": [],
                    "per_training_seed": [],
                    "per_scenario_seed": [],
                    "learning_curves": [],
                    "acceptance": None,
                    "ablations": [],
                    "artifact_references": [],
                },
            )
            for name in (
                "experiment-matrices.json",
                "evaluations.json",
                "research-summary.json",
            ):
                artifact_sources[name] = ("generated", "research_absence")
        if simulator is not None and validated_simulator is not None:
            _copy_bundle(simulator, staging, artifact_sources)
            _write_simulator_view(
                staging,
                simulator,
                validated_simulator,
                artifact_sources,
            )
        else:
            _write_json(
                staging / "coppelia-evidence.json",
                {
                    "schema_version": "construction-intelligence-coppelia-public-v1",
                    "status": "absent",
                    "ready": False,
                    "claim_allowed": False,
                    "payload_transport": "logical",
                    "reason": (
                        "No live nominal and unavailable-robot Coppelia evidence "
                        "bundle was supplied."
                    ),
                    "nominal": None,
                    "recovery": None,
                    "limitation": (
                        "Payload transport is logical. This evidence does not "
                        "claim arm, gripper, grasp-contact, or payload dynamics."
                    ),
                },
            )
            artifact_sources["coppelia-evidence.json"] = (
                "generated",
                "simulator_absence",
            )
        _write_release_status(
            staging,
            channel=channel,
            deterministic=deterministic,
            research=research,
            simulator=simulator,
            release_version=release_version,
            release_tag=release_tag,
        )
        artifact_sources["release-status.json"] = ("generated", "release_status")
        provenance = _build_provenance(
            staging,
            channel=channel,
            source=source,
            bundles=[deterministic, research, simulator],
            artifact_sources=artifact_sources,
            release_version=release_version,
            release_tag=release_tag,
        )
        _write_json(staging / _PROVENANCE_NAME, provenance)
        verify_public_demo_export(staging)
        _replace_directory(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return cast(
        dict[str, object],
        json.loads((destination / _PROVENANCE_NAME).read_text(encoding="utf-8")),
    )


def verify_public_demo_export(
    output_dir: Path,
    *,
    expected_channel: Literal["preview", "release"] | None = None,
    expected_source: SourceIdentity | None = None,
    expected_release_version: str | None = None,
    expected_release_tag: str | None = None,
) -> dict[str, object]:
    """Re-hash and semantically verify a generated public-demo bundle."""

    root = output_dir.resolve()
    provenance_path = root / _PROVENANCE_NAME
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDemoExportError(f"provenance manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise PublicDemoExportError("provenance manifest must be a JSON object")
    if payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise PublicDemoExportError("unsupported provenance schema")
    expected_integrity = payload.get("manifest_payload_sha256")
    integrity_payload = dict(payload)
    integrity_payload["manifest_payload_sha256"] = None
    if expected_integrity != _sha256_bytes(_json_bytes(integrity_payload)):
        raise PublicDemoExportError("provenance manifest integrity check failed")
    channel = payload.get("channel")
    if channel not in {"preview", "release"}:
        raise PublicDemoExportError("provenance channel must be preview or release")
    typed_channel = cast(Literal["preview", "release"], channel)
    try:
        source = SourceIdentity.model_validate(payload.get("source"))
    except ValueError as exc:
        raise PublicDemoExportError(
            f"provenance source identity is invalid: {exc}"
        ) from exc
    release_version = payload.get("release_version")
    release_tag = payload.get("release_tag")
    if release_version is not None and not isinstance(release_version, str):
        raise PublicDemoExportError("release_version must be a string or null")
    if release_tag is not None and not isinstance(release_tag, str):
        raise PublicDemoExportError("release_tag must be a string or null")
    try:
        _validate_release_metadata(
            typed_channel,
            release_version,
            release_tag,
        )
    except ValueError as exc:
        raise PublicDemoExportError(str(exc)) from exc
    if expected_channel is not None and typed_channel != expected_channel:
        raise PublicDemoExportError(
            f"provenance channel mismatch: expected {expected_channel}, got {channel}"
        )
    if expected_source is not None and source != expected_source:
        raise PublicDemoExportError("provenance source identity does not match expectation")
    if (expected_release_version is None) != (expected_release_tag is None):
        raise PublicDemoExportError(
            "expected release version and tag must be supplied together"
        )
    if (
        expected_release_version is not None
        and expected_release_tag is not None
    ):
        try:
            validate_release_identity(
                expected_release_version,
                expected_release_tag,
            )
        except ValueError as exc:
            raise PublicDemoExportError(str(exc)) from exc
        if (
            release_version != expected_release_version
            or release_tag != expected_release_tag
        ):
            raise PublicDemoExportError(
                "provenance release version or tag does not match expectation"
            )
    _validate_provenance_inputs(payload, channel=typed_channel)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not all(
        isinstance(item, dict) for item in artifacts
    ):
        raise PublicDemoExportError("provenance artifacts must be a list of objects")
    artifact_records = cast(list[dict[str, object]], artifacts)
    expected_path_list: list[str] = []
    for item in artifact_records:
        relative = item.get("path")
        byte_count = item.get("bytes")
        sha256 = item.get("sha256")
        source_kind = item.get("source_kind")
        role = item.get("role")
        if (
            not isinstance(relative, str)
            or not isinstance(byte_count, int)
            or isinstance(byte_count, bool)
            or byte_count < 0
            or not isinstance(sha256, str)
            or not _is_sha256(sha256)
            or not isinstance(source_kind, str)
            or not source_kind
            or not isinstance(role, str)
            or not role
        ):
            raise PublicDemoExportError(
                "provenance contains a malformed artifact record"
            )
        try:
            _safe_relative_path(relative)
        except ValueError as exc:
            raise PublicDemoExportError(str(exc)) from exc
        expected_path_list.append(relative)
    if len(expected_path_list) != len(set(expected_path_list)):
        raise PublicDemoExportError("provenance artifact paths must be unique")
    if payload.get("artifact_count") != len(artifact_records):
        raise PublicDemoExportError("provenance artifact_count is inconsistent")
    expected_paths = set(expected_path_list)
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != _PROVENANCE_NAME
    }
    if expected_paths != observed_paths:
        missing = sorted(expected_paths - observed_paths)
        extra = sorted(observed_paths - expected_paths)
        raise PublicDemoExportError(
            f"provenance coverage mismatch; missing={missing}, extra={extra}"
        )
    for item in artifact_records:
        relative = str(item["path"])
        path = root / _safe_relative_path(relative)
        actual_size = path.stat().st_size
        actual_hash = _sha256_file(path)
        if item.get("bytes") != actual_size or item.get("sha256") != actual_hash:
            raise PublicDemoExportError(f"artifact integrity check failed: {relative}")
    _validate_release_status(
        root,
        channel=typed_channel,
        release_version=release_version,
        release_tag=release_tag,
    )
    return payload


def verify_public_demo_regeneration_identity(
    reference_dir: Path,
    regenerated_dir: Path,
) -> dict[str, object]:
    """Require two independently verified exports to be byte-identical by manifest."""

    reference = verify_public_demo_export(reference_dir)
    regenerated = verify_public_demo_export(regenerated_dir)
    if reference != regenerated:
        raise PublicDemoExportError(
            "public-demo regeneration identity mismatch"
        )
    return regenerated


def _load_bundle(
    path: Path,
    *,
    expected_kind: Literal["deterministic", "research", "simulator"],
) -> _LoadedBundle:
    manifest_path = path.resolve()
    try:
        manifest = EvidenceBundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(
            f"invalid {expected_kind} bundle manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.kind != expected_kind:
        raise PublicDemoExportError(
            f"expected a {expected_kind} bundle, got {manifest.kind}"
        )
    _validate_public_artifact_contract(manifest)
    files: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        source_path = Path(artifact.path)
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_file():
            raise PublicDemoExportError(
                f"{expected_kind} artifact is missing: {artifact.path}"
            )
        if _sha256_file(source_path) != artifact.sha256:
            raise PublicDemoExportError(
                f"{expected_kind} artifact hash mismatch: {artifact.path}"
            )
        files[artifact.role] = source_path
    return _LoadedBundle(
        path=manifest_path,
        sha256=_sha256_file(manifest_path),
        manifest=manifest,
        files=files,
    )


def _validate_public_artifact_contract(
    manifest: EvidenceBundleManifest,
) -> None:
    contracts: dict[str, tuple[str, str]]
    if manifest.kind == "research":
        contracts = _RESEARCH_ARTIFACT_CONTRACTS
    elif manifest.kind == "simulator":
        contracts = _SIMULATOR_ARTIFACT_CONTRACTS
    else:
        return
    for artifact in manifest.artifacts:
        expected = contracts.get(artifact.role)
        if expected is None:
            raise PublicDemoExportError(
                f"{manifest.kind} artifact role is not public-safe: {artifact.role}"
            )
        expected_target, expected_media_type = expected
        if (
            artifact.target != expected_target
            or artifact.media_type != expected_media_type
        ):
            raise PublicDemoExportError(
                f"{manifest.kind} artifact does not match its passive public contract: "
                f"{artifact.role}"
            )


def _validate_research(bundle: _LoadedBundle) -> _ValidatedResearch:
    manifest = bundle.manifest
    roles = set(bundle.files)
    if roles != _RESEARCH_ROLES:
        raise PublicDemoExportError(
            "research bundle roles must be exactly "
            f"{sorted(_RESEARCH_ROLES)}; got {sorted(roles)}"
        )
    if manifest.evidence_status != "canonical" or manifest.profile != "research":
        raise PublicDemoExportError(
            "research evidence must be canonical and use the research profile"
        )
    try:
        protocol = load_experiment_protocol()
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(
            f"the frozen research protocol is unavailable or invalid: {exc}"
        ) from exc
    if manifest.protocol_digest != protocol_digest(protocol):
        raise PublicDemoExportError(
            "research evidence does not match the frozen v1 protocol digest"
        )
    matrix = _read_object(bundle.files["matrix"], "research matrix")
    runs_value = matrix.get("runs")
    if not isinstance(runs_value, list) or not all(
        isinstance(item, dict) for item in runs_value
    ):
        raise PublicDemoExportError("research matrix runs are malformed")
    runs = cast(list[dict[str, object]], runs_value)
    if (
        matrix.get("id") != manifest.matrix_id
        or matrix.get("protocol_digest") != manifest.protocol_digest
        or matrix.get("execution_profile") != "research"
        or matrix.get("expected_run_count") != 20
        or len(runs) != 20
        or matrix.get("selection_count") != 20
    ):
        raise PublicDemoExportError(
            "research matrix metadata does not describe a complete 20-run research matrix"
        )
    run_keys = [str(run.get("run_key", "")) for run in runs]
    if len(set(run_keys)) != 20 or any(not item for item in run_keys):
        raise PublicDemoExportError("research matrix run keys must contain 20 unique values")
    if any(str(run.get("status", "")) != "completed" for run in runs):
        raise PublicDemoExportError("every research matrix run must be completed")
    run_configs: dict[str, dict[str, object]] = {}
    for run in runs:
        config = run.get("config")
        if not isinstance(config, dict):
            raise PublicDemoExportError(f"research run config is malformed: {run}")
        run_key = str(run["run_key"])
        run_configs[run_key] = config
        if (
            config.get("profile") != "research"
            or config.get("source_commit") != manifest.source.commit
            or config.get("source_dirty") is not False
        ):
            raise PublicDemoExportError(
                f"research run provenance mismatch: {run_key}"
            )
    expected_run_identities = {
        (variant, seed)
        for variant in (
            "mappo_full",
            "ippo_full",
            "mappo_no_bc",
            "mappo_no_failure_curriculum",
        )
        for seed in range(7, 12)
    }
    observed_run_identities = {
        (
            str(config.get("experiment_variant", "")),
            (
                cast(int, config["training_seed"])
                if isinstance(config.get("training_seed"), int)
                else -1
            ),
        )
        for config in run_configs.values()
    }
    expected_algorithms = {
        "mappo_full": "mappo",
        "ippo_full": "ippo",
        "mappo_no_bc": "mappo",
        "mappo_no_failure_curriculum": "mappo",
    }
    if (
        observed_run_identities != expected_run_identities
        or any(
            config.get("algorithm")
            != expected_algorithms.get(str(config.get("experiment_variant", "")))
            or config.get("transitions") != 1_500_000
            or config.get("checkpoint_fractions")
            != [0.1, 0.25, 0.5, 0.75, 1.0]
            for config in run_configs.values()
        )
    ):
        raise PublicDemoExportError(
            "research matrix does not match the frozen variants, seeds, "
            "transition budget, and checkpoint grid"
        )
    observed_digests = sorted(
        {
            str(config.get("configuration_digest", ""))
            for config in run_configs.values()
        }
    )
    if (
        any(not _is_sha256(item) for item in observed_digests)
        or observed_digests != sorted(manifest.configuration_digests)
    ):
        raise PublicDemoExportError(
            "research configuration digests do not match the bundle declaration"
        )

    selections_payload = _read_json(bundle.files["selections"], "research selections")
    if not isinstance(selections_payload, dict):
        raise PublicDemoExportError(
            "research selections must be canonical matrix selection evidence"
        )
    selection_protocol_digest = selections_payload.get("protocol_digest")
    if selections_payload.get("matrix_id") != manifest.matrix_id:
        raise PublicDemoExportError("selection matrix id does not match")
    selections_value = selections_payload.get("selections")
    if not isinstance(selections_value, list) or not all(
        isinstance(item, dict) for item in selections_value
    ):
        raise PublicDemoExportError("research selections are malformed")
    if selection_protocol_digest != manifest.protocol_digest:
        raise PublicDemoExportError("selection protocol digest does not match")
    selection_records = [
        _public_selection_record(cast(dict[str, object], item), manifest.created_at)
        for item in selections_value
    ]
    selected_keys = {str(item["run_key"]) for item in selection_records}
    if len(selection_records) != 20 or selected_keys != set(run_keys):
        raise PublicDemoExportError(
            "research selections must cover all 20 matrix runs exactly once"
        )
    raw_selection_by_run = {
        str(item["run_key"]): item
        for item in cast(list[dict[str, object]], selections_value)
    }
    selected_by_identity: dict[tuple[str, int], tuple[str, SelectedCheckpoint]] = {}
    for record in selection_records:
        run_key = str(record["run_key"])
        selection_payload = cast(dict[str, object], record["selection"])
        config = run_configs[run_key]
        try:
            selection = SelectedCheckpoint.model_validate(selection_payload)
        except ValueError as exc:
            raise PublicDemoExportError(
                f"frozen selection is malformed for {run_key}: {exc}"
            ) from exc
        if (
            selection.split != protocol.selection.split
            or selection.scenario_seeds != protocol.selection.scenario_seeds
            or selection.required_checkpoint_fractions
            != [0.1, 0.25, 0.5, 0.75, 1.0]
            or selection.source_commit != manifest.source.commit
            or selection.configuration_digest
            != config.get("configuration_digest")
            or selection.experiment_id != protocol.experiment_id
            or selection.experiment_variant
            != config.get("experiment_variant")
            or selection.training_seed != config.get("training_seed")
        ):
            raise PublicDemoExportError(
                f"frozen selection provenance is invalid: {run_key}"
            )
        candidates = raw_selection_by_run[run_key].get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(candidate, dict) for candidate in candidates
        ):
            raise PublicDemoExportError(
                f"validation candidates are malformed: {run_key}"
            )
        fractions: list[float] = []
        candidate_results: list[CheckpointValidationResult] = []
        for candidate in cast(list[dict[str, object]], candidates):
            result_payload = candidate.get("result")
            episodes_payload = candidate.get("episodes")
            if not isinstance(result_payload, dict) or not isinstance(
                episodes_payload, list
            ):
                raise PublicDemoExportError(
                    f"validation candidate evidence is malformed: {run_key}"
                )
            try:
                result = CheckpointValidationResult.model_validate(result_payload)
                episodes = _EPISODE_LIST_ADAPTER.validate_python(episodes_payload)
            except ValueError as exc:
                raise PublicDemoExportError(
                    f"validation candidate evidence is malformed for {run_key}: {exc}"
                ) from exc
            fraction = result.checkpoint_fraction
            if isinstance(fraction, bool):
                raise PublicDemoExportError(
                    f"validation checkpoint fraction is malformed: {run_key}"
                )
            fractions.append(float(fraction))
            cells = {
                (
                    episode.seed,
                    episode.failure_enabled,
                    episode.split,
                )
                for episode in episodes
            }
            expected_cells = {
                (seed, failure, "validation")
                for seed in protocol.selection.scenario_seeds
                for failure in protocol.selection.failure_modes
            }
            if (
                len(episodes) != 10
                or cells != expected_cells
                or result.source_commit != manifest.source.commit
                or result.configuration_digest
                != config.get("configuration_digest")
            ):
                raise PublicDemoExportError(
                    f"validation candidate grid or provenance is invalid: {run_key}"
                )
            _validate_validation_candidate(
                run_key,
                result,
                episodes,
                algorithm=str(config.get("algorithm", "")),
                experiment_variant=str(config.get("experiment_variant", "")),
                training_seed=cast(int, config.get("training_seed")),
            )
            candidate_results.append(result)
        if sorted(fractions) != [0.1, 0.25, 0.5, 0.75, 1.0]:
            raise PublicDemoExportError(
                f"validation candidates do not cover five checkpoints: {run_key}"
            )
        try:
            recomputed_selection = select_validation_checkpoint(
                candidate_results,
                validation_seeds=protocol.selection.scenario_seeds,
                required_fractions=protocol.checkpoint_fractions,
            )
        except ValueError as exc:
            raise PublicDemoExportError(
                f"validation ranking cannot be reproduced for {run_key}: {exc}"
            ) from exc
        if recomputed_selection != selection:
            raise PublicDemoExportError(
                f"selected checkpoint does not match deterministic validation "
                f"ranking: {run_key}"
            )
        identity = (selection.experiment_variant, selection.training_seed)
        if identity in selected_by_identity:
            raise PublicDemoExportError(
                f"duplicate selected policy identity: {identity}"
            )
        selected_by_identity[identity] = (run_key, selection)

    try:
        suite = EvaluationSuite.model_validate_json(
            bundle.files["evaluation"].read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(f"held-out evaluation is invalid: {exc}") from exc
    grid = suite.grid_validation
    learned_count = sum(
        episode.controller in {"mappo", "ippo"} for episode in suite.episodes
    )
    baseline_count = len(suite.episodes) - learned_count
    expected_summary_groups = {
        (algorithm, variant, failure)
        for variant, algorithm in (
            ("mappo_full", "mappo"),
            ("ippo_full", "ippo"),
            ("mappo_no_bc", "mappo"),
            ("mappo_no_failure_curriculum", "mappo"),
        )
        for failure in (False, True)
    } | {
        (controller, None, failure)
        for controller in ("sequential", "greedy", "auction", "cp_sat")
        for failure in (False, True)
    }
    observed_summary_groups = {
        (
            summary.controller,
            summary.experiment_variant,
            summary.failure_enabled,
        )
        for summary in suite.summaries
    }
    required_metrics = set(_EVALUATION_METRICS)
    observed_episode_cells = {
        (
            episode.controller,
            episode.experiment_variant,
            episode.training_seed,
            episode.seed,
            episode.failure_enabled,
        )
        for episode in suite.episodes
    }
    expected_episode_cells = {
        (algorithm, variant, seed, scenario_seed, failure)
        for variant, algorithm in (
            ("mappo_full", "mappo"),
            ("ippo_full", "ippo"),
            ("mappo_no_bc", "mappo"),
            ("mappo_no_failure_curriculum", "mappo"),
        )
        for seed in range(7, 12)
        for scenario_seed in range(900, 905)
        for failure in (False, True)
    } | {
        (controller, None, None, scenario_seed, failure)
        for controller in ("sequential", "greedy", "auction", "cp_sat")
        for scenario_seed in range(900, 905)
        for failure in (False, True)
    }
    if (
        suite.seeds != protocol.evaluation.scenario_seeds
        or set(suite.controllers) != set(protocol.evaluation.controllers)
        or len(suite.controllers) != len(protocol.evaluation.controllers)
        or suite.expected_split != protocol.evaluation.split
        or len(suite.episodes) != 240
        or grid is None
        or grid.observed_episode_count != 240
        or grid.expected_episode_count != 240
        or grid.expected_split != protocol.evaluation.split
        or learned_count != 200
        or baseline_count != 40
        or observed_episode_cells != expected_episode_cells
        or observed_summary_groups != expected_summary_groups
        or any(
            set(summary.metrics) != required_metrics
            for summary in suite.summaries
        )
        or len(suite.per_training_seed) != 48
        or len(suite.per_scenario_seed) != 80
        or any(
            set(summary.metrics) != required_metrics
            for summary in suite.per_training_seed
        )
        or any(
            set(summary.metrics) != required_metrics
            for summary in suite.per_scenario_seed
        )
    ):
        raise PublicDemoExportError(
            "held-out evaluation is not the complete 200 learned + 40 baseline grid"
        )
    canonical_summaries = _recompute_controller_summaries(suite.episodes)
    canonical_per_training_seed = summarize_by_training_seed(suite.episodes)
    canonical_per_scenario_seed = summarize_by_scenario_seed(suite.episodes)
    if (
        suite.summaries != canonical_summaries
        or suite.per_training_seed != canonical_per_training_seed
        or suite.per_scenario_seed != canonical_per_scenario_seed
    ):
        raise PublicDemoExportError(
            "published evaluation summaries do not match the canonical held-out episodes"
        )
    for episode in suite.episodes:
        if episode.seed not in suite.seeds or episode.split != protocol.evaluation.split:
            raise PublicDemoExportError(
                "held-out episodes contain the wrong split or scenario seeds"
            )
        if episode.controller in {"mappo", "ippo"}:
            identity = (cast(str, episode.experiment_variant), cast(int, episode.training_seed))
            selected_entry = selected_by_identity.get(identity)
            if selected_entry is None:
                raise PublicDemoExportError(
                    f"held-out learned episode has no frozen selection: {identity}"
                )
            run_key, selected = selected_entry
            _validate_heldout_selected_episode(
                episode,
                selected,
                matrix_id=cast(str, manifest.matrix_id),
                run_key=run_key,
                expected_controller=str(run_configs[run_key].get("algorithm", "")),
            )
        else:
            _validate_baseline_episode(episode)
    expected_csv = _evaluation_csv_text(suite)
    try:
        with bundle.files["episodes"].open(
            encoding="utf-8",
            newline="",
        ) as handle:
            observed_csv = handle.read()
    except OSError as exc:
        raise PublicDemoExportError(
            f"held-out episode CSV is unreadable: {exc}"
        ) from exc
    if observed_csv != expected_csv:
        raise PublicDemoExportError(
            "held-out episode CSV does not exactly match evaluation.json episodes"
        )
    try:
        observed_report = bundle.files["report"].read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicDemoExportError(
            f"held-out report is unreadable: {exc}"
        ) from exc
    if observed_report != render_evaluation_report(suite):
        raise PublicDemoExportError(
            "held-out report does not match the canonical evaluation suite"
        )
    try:
        acceptance = PrimaryAcceptanceAudit.model_validate_json(
            bundle.files["acceptance"].read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(f"acceptance evidence is invalid: {exc}") from exc
    recomputed_acceptance = audit_primary_acceptance(suite, protocol)
    if acceptance != recomputed_acceptance:
        raise PublicDemoExportError(
            "declared acceptance does not match recomputed held-out metrics"
        )
    if not recomputed_acceptance.passed:
        raise PublicDemoExportError(
            "one or more recomputed primary research thresholds failed"
        )
    try:
        ablations_payload = json.loads(
            bundle.files["ablations"].read_text(encoding="utf-8")
        )
        ablations = TypeAdapter(list[AblationDecision]).validate_python(
            ablations_payload
        )
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(f"ablation evidence is invalid: {exc}") from exc
    if {item.hypothesis for item in ablations} != {
        "behavior_cloning",
        "failure_curriculum",
    }:
        raise PublicDemoExportError(
            "both pre-registered ablation interpretations are required"
        )
    reproducibility_audit = _read_object(
        bundle.files["reproducibility_audit"],
        "research reproducibility audit",
    )
    if (
        reproducibility_audit.get("matrix_id") != manifest.matrix_id
        or reproducibility_audit.get("complete") is not True
        or not isinstance(
            reproducibility_audit.get("checked_file_count"), int
        )
        or cast(int, reproducibility_audit["checked_file_count"]) <= 0
        or reproducibility_audit.get("missing_paths") != []
        or reproducibility_audit.get("invalid_artifacts") != []
    ):
        raise PublicDemoExportError(
            "research reproducibility audit is incomplete or invalid"
        )
    release_completeness = _read_object(
        bundle.files["release_completeness"],
        "research release completeness",
    )
    if (
        release_completeness.get("complete") is not True
        or release_completeness.get("blocking_reasons") != []
        or any(
            release_completeness.get(field) != []
            for field in (
                "missing_training_run_ids",
                "missing_selected_policy_run_ids",
                "missing_heldout_evaluation_run_ids",
                "missing_ablation_hypotheses",
            )
        )
    ):
        raise PublicDemoExportError(
            "research release-completeness decision has blockers"
        )
    return _ValidatedResearch(
        matrix=matrix,
        selection_records=selection_records,
        suite=suite,
        acceptance=acceptance,
        ablations=ablations,
        reproducibility_audit=reproducibility_audit,
        release_completeness=release_completeness,
    )


def _recompute_controller_summaries(
    episodes: list[EpisodeEvaluation],
) -> list[ControllerEvaluation]:
    groups: dict[
        tuple[ControllerName, str | None, str | None, bool],
        list[EpisodeEvaluation],
    ] = {}
    for episode in episodes:
        key = (
            episode.controller,
            episode.experiment_id,
            episode.experiment_variant,
            episode.failure_enabled,
        )
        groups.setdefault(key, []).append(episode)
    bootstrap_indices: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}
    summaries: list[ControllerEvaluation] = []
    for key in sorted(
        groups,
        key=lambda item: (
            item[3],
            item[0],
            item[1] or "",
            item[2] or "",
        ),
    ):
        controller, experiment_id, experiment_variant, failure_enabled = key
        subset = groups[key]
        summaries.append(
            ControllerEvaluation(
                controller=controller,
                failure_enabled=failure_enabled,
                episode_count=len(subset),
                metrics={
                    field: _recompute_hierarchical_metric(
                        subset,
                        field=field,
                        bootstrap_indices=bootstrap_indices,
                    )
                    for field in _EVALUATION_METRICS
                },
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
                training_seed_count=len(
                    {
                        episode.training_seed
                        for episode in subset
                        if episode.training_seed is not None
                    }
                ),
                scenario_seed_count=len({episode.seed for episode in subset}),
            )
        )
    return summaries


def _recompute_hierarchical_metric(
    episodes: list[EpisodeEvaluation],
    *,
    field: str,
    bootstrap_indices: dict[
        tuple[int, int],
        tuple[np.ndarray, np.ndarray],
    ],
) -> MetricSummary:
    values = np.array(
        [float(getattr(episode, field)) for episode in episodes],
        dtype=np.float64,
    )
    training_groups: dict[int | None, dict[int, list[float]]] = {}
    for episode in episodes:
        scenario_groups = training_groups.setdefault(episode.training_seed, {})
        scenario_groups.setdefault(episode.seed, []).append(
            float(getattr(episode, field))
        )
    training_seeds = sorted(
        training_groups,
        key=lambda value: -1 if value is None else value,
    )
    scenario_grids = {
        tuple(sorted(scenarios)) for scenarios in training_groups.values()
    }
    if len(scenario_grids) != 1:
        raise PublicDemoExportError(
            "published summaries use inconsistent hierarchical scenario grids"
        )
    scenario_seeds = list(next(iter(scenario_grids)))
    grid = np.array(
        [
            [
                float(np.mean(training_groups[training_seed][scenario_seed]))
                for scenario_seed in scenario_seeds
            ]
            for training_seed in training_seeds
        ],
        dtype=np.float64,
    )
    point_mean = float(np.mean(np.mean(grid, axis=1)))
    if len(episodes) == 1:
        low = high = point_mean
        std = 0.0
    else:
        dimensions = (len(training_seeds), len(scenario_seeds))
        indices = bootstrap_indices.get(dimensions)
        if indices is None:
            rng = np.random.default_rng(2027)
            training_indices = np.empty((2000, dimensions[0]), dtype=np.int64)
            scenario_indices = np.empty((2000, dimensions[1]), dtype=np.int64)
            for bootstrap_index in range(2000):
                training_indices[bootstrap_index] = rng.integers(
                    0,
                    dimensions[0],
                    size=dimensions[0],
                )
                scenario_indices[bootstrap_index] = rng.integers(
                    0,
                    dimensions[1],
                    size=dimensions[1],
                )
            indices = (training_indices, scenario_indices)
            bootstrap_indices[dimensions] = indices
        training_indices, scenario_indices = indices
        sampled = grid[
            training_indices[:, :, np.newaxis],
            scenario_indices[:, np.newaxis, :],
        ]
        bootstrap = sampled.mean(axis=(1, 2))
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        std = float(values.std(ddof=1))
    return MetricSummary(
        mean=point_mean,
        std=std,
        bootstrap_ci95_low=float(low),
        bootstrap_ci95_high=float(high),
        median=float(median(values.tolist())),
    )


def _validate_validation_candidate(
    run_key: str,
    result: CheckpointValidationResult,
    episodes: list[EpisodeEvaluation],
    *,
    algorithm: str,
    experiment_variant: str,
    training_seed: int,
) -> None:
    if (
        result.experiment_id != "construction_intelligence_v1"
        or result.experiment_variant != experiment_variant
        or result.training_seed != training_seed
        or result.transition_count
        != int(round(1_500_000 * result.checkpoint_fraction))
    ):
        raise PublicDemoExportError(
            f"validation checkpoint identity is inconsistent: {run_key}"
        )
    for episode in episodes:
        if (
            episode.controller != algorithm
            or episode.policy_id != result.checkpoint_id
            or episode.experiment_id != result.experiment_id
            or episode.experiment_variant != result.experiment_variant
            or episode.training_seed != result.training_seed
            or episode.transition_count != result.transition_count
            or episode.checkpoint_path != result.checkpoint_path
            or episode.checkpoint_sha256 != result.checkpoint_sha256
            or episode.checkpoint_lineage != result.checkpoint_lineage
            or episode.configuration_digest != result.configuration_digest
            or episode.source_commit != result.source_commit
            or episode.resume_provenance != result.resume_provenance
        ):
            raise PublicDemoExportError(
                f"validation episode is not tied to its checkpoint: {run_key}"
            )
    mean_completion = sum(
        episode.structure_completion_rate for episode in episodes
    ) / len(episodes)
    mean_makespan = sum(episode.makespan_s for episode in episodes) / len(episodes)
    if not math.isclose(
        result.mean_completion_rate,
        mean_completion,
        rel_tol=0,
        abs_tol=1e-12,
    ) or not math.isclose(
        result.mean_makespan_s,
        mean_makespan,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise PublicDemoExportError(
            f"validation checkpoint aggregates do not match its episodes: {run_key}"
        )


def _validate_heldout_selected_episode(
    episode: EpisodeEvaluation,
    selected: SelectedCheckpoint,
    *,
    matrix_id: str,
    run_key: str,
    expected_controller: str,
) -> None:
    if (
        episode.controller != expected_controller
        or episode.policy_id != f"{matrix_id}-{run_key}-selected"
        or episode.experiment_id != selected.experiment_id
        or episode.experiment_variant != selected.experiment_variant
        or episode.training_seed != selected.training_seed
        or episode.transition_count != selected.transition_count
        or episode.checkpoint_path != selected.checkpoint_path
        or episode.checkpoint_sha256 != selected.checkpoint_sha256
        or episode.checkpoint_lineage != selected.checkpoint_lineage
        or episode.configuration_digest != selected.configuration_digest
        or episode.source_commit != selected.source_commit
        or episode.resume_provenance != selected.resume_provenance
    ):
        raise PublicDemoExportError(
            "held-out learned episode does not match its frozen selected "
            f"checkpoint lineage: {run_key}"
        )


def _validate_baseline_episode(episode: EpisodeEvaluation) -> None:
    if (
        episode.policy_id is not None
        or episode.experiment_id is not None
        or episode.experiment_variant is not None
        or episode.training_seed is not None
        or episode.transition_count is not None
        or episode.checkpoint_path is not None
        or episode.checkpoint_sha256 is not None
        or episode.checkpoint_lineage
        or episode.configuration_digest is not None
        or episode.source_commit is not None
        or episode.resume_provenance
    ):
        raise PublicDemoExportError(
            "baseline held-out episodes must not claim learned-policy provenance"
        )


def _evaluation_csv_text(suite: EvaluationSuite) -> str:
    rows = [episode.model_dump(mode="json") for episode in suite.episodes]
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _validate_simulator(bundle: _LoadedBundle) -> _ValidatedSimulator:
    manifest = bundle.manifest
    roles = set(bundle.files)
    if roles != _SIMULATOR_ROLES:
        raise PublicDemoExportError(
            "simulator bundle roles must be exactly "
            f"{sorted(_SIMULATOR_ROLES)}; got {sorted(roles)}"
        )
    if manifest.evidence_status != "canonical":
        raise PublicDemoExportError("live simulator evidence must be canonical")
    nominal_manifest = _verify_native_phase5_bundle(
        bundle,
        descriptor_prefix="nominal",
        expected_scenario="nominal",
    )
    recovery_manifest = _verify_native_phase5_bundle(
        bundle,
        descriptor_prefix="recovery",
        expected_scenario="unavailable_robot_recovery",
    )
    configuration_digests = sorted(
        {
            nominal_manifest.configuration_digest,
            recovery_manifest.configuration_digest,
        }
    )
    if configuration_digests != sorted(manifest.configuration_digests):
        raise PublicDemoExportError(
            "Coppelia configuration digests do not match the bundle declaration"
        )
    nominal_metrics = _read_object(
        bundle.files["nominal_metrics"],
        "nominal Coppelia metrics",
    )
    recovery_metrics = _read_object(
        bundle.files["recovery_metrics"],
        "recovery Coppelia metrics",
    )
    try:
        _validate_phase5_metrics(nominal_metrics, scenario="nominal")
        _validate_phase5_metrics(
            recovery_metrics,
            scenario="unavailable_robot_recovery",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PublicDemoExportError(
            f"Coppelia evidence does not pass all declared gates: {exc}"
        ) from exc
    return _ValidatedSimulator(
        nominal_manifest=nominal_manifest,
        recovery_manifest=recovery_manifest,
        nominal_metrics=nominal_metrics,
        recovery_metrics=recovery_metrics,
    )


def _verify_native_phase5_bundle(
    bundle: _LoadedBundle,
    *,
    descriptor_prefix: Literal["nominal", "recovery"],
    expected_scenario: Literal["nominal", "unavailable_robot_recovery"],
) -> Phase5ProvenanceManifest:
    manifest_path = bundle.files[f"{descriptor_prefix}_manifest"]
    expected_files = {
        "manifest": "manifest.json",
        "scenario": "scenario.json",
        "planned_jobs": "planned_jobs.json",
        "replay": "planned_vs_measured_replay.json",
        "wheel_commands": "wheel_commands.jsonl",
        "telemetry": "measured_telemetry.jsonl",
        "trace": "trace.json",
        "metrics": "metrics.json",
        "report": "report.md",
        "scene": "construction_intelligence.ttt",
    }
    if any(
        bundle.files[f"{descriptor_prefix}_{role}"].parent != manifest_path.parent
        or bundle.files[f"{descriptor_prefix}_{role}"].name != file_name
        for role, file_name in expected_files.items()
    ):
        raise PublicDemoExportError(
            f"{descriptor_prefix} descriptor paths do not match a native Phase 5 bundle"
        )
    try:
        phase5 = verify_phase5_artifact_bundle(manifest_path.parent)
    except (OSError, ValueError) as exc:
        raise PublicDemoExportError(
            f"{descriptor_prefix} Phase 5 bundle verification failed: {exc}"
        ) from exc
    if (
        phase5.scenario != expected_scenario
        or phase5.evidence_kind != "live_coppelia"
        or not phase5.live_evidence
        or phase5.run_status != "completed"
        or not phase5.live_gate_passed
        or not phase5.approval_gate_confirmed
        or phase5.source_commit != bundle.manifest.source.commit
        or phase5.source_dirty
        or phase5.source_tree_digest != bundle.manifest.source.tree_digest
        or phase5.payload_transport_model != "logical_carrier"
    ):
        raise PublicDemoExportError(
            f"{descriptor_prefix} Phase 5 manifest does not attest a passing live run"
        )
    return phase5


def _validate_phase5_metrics(
    payload: dict[str, object],
    *,
    scenario: Literal["nominal", "unavailable_robot_recovery"],
) -> None:
    if (
        payload.get("schema_version")
        != "construction_intelligence.coppelia_evidence.v1"
        or payload.get("scenario") != scenario
        or payload.get("evidence_kind") != "live_coppelia"
        or payload.get("status") != "completed"
        or payload.get("live_gate_passed") is not True
    ):
        raise ValueError(f"{scenario} metrics are not passing live evidence")
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or not acceptance or any(
        value is not True for value in acceptance.values()
    ):
        raise ValueError(f"{scenario} acceptance gates did not all pass")
    expected_module_count = payload.get("expected_module_count")
    installed = payload.get("installed_module_ids")
    if (
        not isinstance(expected_module_count, int)
        or not isinstance(installed, list)
        or len(installed) != expected_module_count
        or payload.get("metrics") is None
    ):
        raise ValueError(f"{scenario} did not install every module")
    diagnostics = payload["metrics"]
    if not isinstance(diagnostics, dict):
        raise ValueError(f"{scenario} metrics diagnostics are malformed")
    if (
        diagnostics.get("post_start_robot_pose_writes") != 0
        or diagnostics.get("payload_transport") != "logical_carrier"
        or diagnostics.get("live_evidence") is not True
        or diagnostics.get("live_gate_passed") is not True
    ):
        raise ValueError(f"{scenario} violates the physical-simulator boundary")
    if scenario == "nominal":
        if (
            diagnostics.get("collision_stops") != 0
            or acceptance.get("zero_nominal_collision_stops") is not True
        ):
            raise ValueError("nominal run recorded collision stops")
        return
    recovery = payload.get("recovery")
    if (
        not isinstance(recovery, dict)
        or float(recovery.get("completion_fraction_at_disable", -1)) < 0.25
        or recovery.get("commands_after_stop") != 0
        or not recovery.get("reassigned_job_ids")
        or acceptance.get("robot_disabled_after_25_percent") is not True
        or acceptance.get("disabled_robot_not_commanded_after_stop") is not True
        or acceptance.get("remaining_work_reassigned") is not True
    ):
        raise ValueError("recovery run did not pass disable-and-reassign gates")


def _validate_export_claims(
    channel: Literal["preview", "release"],
    source: SourceIdentity,
    deterministic: _LoadedBundle,
    research: _LoadedBundle | None,
    simulator: _LoadedBundle | None,
    validated_research: _ValidatedResearch | None,
    validated_simulator: _ValidatedSimulator | None,
) -> None:
    if channel == "preview":
        return
    if not deterministic.manifest.configuration_digests:
        raise PublicDemoExportError(
            "release deterministic bundle must declare at least one "
            "configuration digest"
        )
    if (
        source.dirty
        or not _is_sha256(source.tree_digest)
        or deterministic.manifest.evidence_status != "canonical"
        or research is None
        or simulator is None
        or validated_research is None
        or validated_simulator is None
    ):
        raise PublicDemoExportError(
            "release export requires clean canonical deterministic, research, "
            "and live simulator evidence"
        )
    if set(deterministic.files) != _RELEASE_DETERMINISTIC_ROLES:
        raise PublicDemoExportError(
            "release deterministic bundle does not contain every required artifact"
        )
    deterministic_targets = {
        item.role: item.target for item in deterministic.manifest.artifacts
    }
    if deterministic_targets != _RELEASE_DETERMINISTIC_TARGETS:
        raise PublicDemoExportError(
            "release deterministic artifact targets do not match the public contract"
        )
    bundles = (deterministic, research, simulator)
    if any(
        bundle.manifest.source.dirty
        or not _is_sha256(bundle.manifest.source.tree_digest)
        for bundle in bundles
    ):
        raise PublicDemoExportError(
            "release inputs must each record a clean, hashable source identity"
        )


def _copy_bundle(
    bundle: _LoadedBundle,
    staging: Path,
    artifact_sources: dict[str, tuple[str, str]],
) -> None:
    for artifact in bundle.manifest.artifacts:
        target = _safe_relative_path(artifact.target).as_posix()
        if target in artifact_sources:
            raise PublicDemoExportError(f"bundle target collision: {target}")
        if bundle.manifest.kind == "research" and not target.startswith(
            "evidence/research/"
        ):
            raise PublicDemoExportError(
                f"research artifacts must target evidence/research/: {target}"
            )
        if bundle.manifest.kind == "simulator" and not target.startswith(
            "evidence/coppelia/"
        ):
            raise PublicDemoExportError(
                f"simulator artifacts must target evidence/coppelia/: {target}"
            )
        destination = staging / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundle.files[artifact.role], destination)
        artifact_sources[target] = (bundle.manifest.kind, artifact.role)


def _write_research_views(
    staging: Path,
    bundle: _LoadedBundle,
    evidence: _ValidatedResearch,
    artifact_sources: dict[str, tuple[str, str]],
) -> None:
    matrix_id = cast(str, bundle.manifest.matrix_id)
    suite_payload = evidence.suite.model_dump(mode="json")
    _write_json(staging / "experiment-matrices.json", [evidence.matrix])
    selections_path = (
        staging / "experiment-matrices" / matrix_id / "selections.json"
    )
    _write_json(selections_path, evidence.selection_records)
    _write_json(
        staging / "evaluations.json",
        [
            {
                "id": evidence.suite.evaluation_id,
                "matrix_id": matrix_id,
                "split": "test",
                "payload": suite_payload,
                "artifact_dir": "evidence/research",
                "created_at": bundle.manifest.created_at,
            }
        ],
    )
    summary = {
        "schema_version": "construction-intelligence-research-summary-v1",
        "status": "validated",
        "claim_allowed": True,
        "source_commit": bundle.manifest.source.commit,
        "protocol_digest": bundle.manifest.protocol_digest,
        "configuration_digests": sorted(bundle.manifest.configuration_digests),
        "matrix_id": matrix_id,
        "training_run_count": 20,
        "selection_count": 20,
        "heldout_episode_count": 240,
        "learned_episode_count": 200,
        "baseline_episode_count": 40,
        "confidence_intervals": suite_payload["summaries"],
        "per_training_seed": suite_payload["per_training_seed"],
        "per_scenario_seed": suite_payload["per_scenario_seed"],
        "learning_curves": _research_validation_curves(bundle),
        "acceptance": evidence.acceptance.model_dump(mode="json"),
        "ablations": [item.model_dump(mode="json") for item in evidence.ablations],
        "artifact_references": _artifact_references_for_roles(
            bundle,
            {
                "evaluation": "Held-out evaluation",
                "episodes": "Held-out episodes",
                "report": "Held-out report",
                "acceptance": "Acceptance audit",
                "ablations": "Ablation decisions",
                "reproducibility_audit": "Reproducibility audit",
                "release_completeness": "Release completeness",
            },
        ),
        "reproducibility_audit": evidence.reproducibility_audit,
        "release_completeness": evidence.release_completeness,
        "fidelity_boundary": (
            "Learned-policy results are event-simulator evidence and do not by "
            "themselves prove dynamic CoppeliaSim execution or physical grasping."
        ),
    }
    _write_json(staging / "research-summary.json", summary)
    existing_policies = _read_json(staging / "policies.json", "deterministic policies")
    if not isinstance(existing_policies, list):
        raise PublicDemoExportError("deterministic policies.json must contain a list")
    selected_policies = [
        {
            "id": selection["checkpoint_id"],
            "controller": (
                "ippo"
                if str(selection["experiment_variant"]).startswith("ippo")
                else "mappo"
            ),
            "manifest": selection,
            "created_at": bundle.manifest.created_at,
        }
        for record in evidence.selection_records
        for selection in [cast(dict[str, object], record["selection"])]
    ]
    policies = [*existing_policies, *selected_policies]
    _write_json(staging / "policies.json", policies)
    generated = {
        "experiment-matrices.json": "matrix_index",
        selections_path.relative_to(staging).as_posix(): "selection_index",
        "evaluations.json": "evaluation_index",
        "research-summary.json": "research_summary",
        "policies.json": "selected_policy_index",
    }
    for path, role in generated.items():
        artifact_sources[path] = ("generated", role)


def _write_simulator_view(
    staging: Path,
    bundle: _LoadedBundle,
    evidence: _ValidatedSimulator,
    artifact_sources: dict[str, tuple[str, str]],
) -> None:
    _write_json(
        staging / "coppelia-evidence.json",
        {
            "schema_version": "construction-intelligence-coppelia-public-v1",
            "status": "validated",
            "ready": True,
            "claim_allowed": True,
            "source_commit": bundle.manifest.source.commit,
            "payload_transport": "logical",
            "nominal": {
                "manifest": evidence.nominal_manifest.model_dump(mode="json"),
                "metrics": evidence.nominal_metrics,
                "artifact_references": _artifact_references_for_prefix(
                    bundle,
                    "nominal",
                ),
            },
            "recovery": {
                "manifest": evidence.recovery_manifest.model_dump(mode="json"),
                "metrics": evidence.recovery_metrics,
                "artifact_references": _artifact_references_for_prefix(
                    bundle,
                    "recovery",
                ),
            },
            "limitation": (
                "Payload transport is logical. This evidence does not claim arm, "
                "gripper, grasp-contact, or payload dynamics."
            ),
        },
    )
    artifact_sources["coppelia-evidence.json"] = (
        "generated",
        "simulator_summary",
    )


def _artifact_references_for_roles(
    bundle: _LoadedBundle,
    labels: dict[str, str],
) -> list[dict[str, str]]:
    artifacts = {item.role: item for item in bundle.manifest.artifacts}
    return [
        {
            "label": label,
            "href": artifacts[role].target,
            "path": artifacts[role].target,
            "media_type": (
                artifacts[role].media_type or "application/octet-stream"
            ),
        }
        for role, label in labels.items()
        if role in artifacts
    ]


def _artifact_references_for_prefix(
    bundle: _LoadedBundle,
    prefix: Literal["nominal", "recovery"],
) -> list[dict[str, str]]:
    labels = {
        "manifest": "Gate manifest",
        "metrics": "Acceptance metrics",
        "report": "Run report",
        "replay": "Planned versus measured replay",
        "telemetry": "Measured telemetry",
        "wheel_commands": "Wheel commands",
        "trace": "Execution trace",
        "scene": "Reusable Coppelia scene",
    }
    return _artifact_references_for_roles(
        bundle,
        {
            f"{prefix}_{role}": f"{prefix.title()} {label}"
            for role, label in labels.items()
        },
    )


def _research_validation_curves(
    bundle: _LoadedBundle,
) -> list[dict[str, object]]:
    """Expose hash-pinned checkpoint validation trajectories as public curves."""

    payload = _read_object(bundle.files["selections"], "research selections")
    selections = payload.get("selections")
    if not isinstance(selections, list):
        return []
    curves: list[dict[str, object]] = []
    for item in selections:
        if not isinstance(item, dict):
            continue
        selected = item.get("selected", item.get("selection"))
        candidates = item.get("candidates")
        if not isinstance(selected, dict) or not isinstance(candidates, list):
            continue
        variant = selected.get("experiment_variant")
        seed = selected.get("training_seed")
        if not isinstance(variant, str) or not isinstance(seed, int):
            continue
        points: list[dict[str, float | int]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            result = candidate.get("result")
            if not isinstance(result, dict):
                continue
            transitions = result.get("transition_count")
            completion = result.get("mean_completion_rate")
            if (
                isinstance(transitions, int)
                and isinstance(completion, (int, float))
            ):
                points.append(
                    {
                        "transitions": transitions,
                        "rollout_terminal_fraction": float(completion),
                    }
                )
        if not points:
            continue
        curves.append(
            {
                "run_key": str(item.get("run_key", "")),
                "controller": (
                    "ippo" if variant.startswith("ippo") else "mappo"
                ),
                "experiment_variant": variant,
                "training_seed": seed,
                "status": "completed",
                "curve_kind": "validation_checkpoint_completion",
                "points": sorted(points, key=lambda point: point["transitions"]),
            }
        )
    return curves


def _validate_release_metadata(
    channel: Literal["preview", "release"],
    release_version: str | None,
    release_tag: str | None,
) -> None:
    if channel == "preview":
        if release_version is not None or release_tag is not None:
            raise PublicDemoExportError(
                "preview exports must not declare a release version or tag"
            )
        return
    if release_version is None or release_tag is None:
        raise PublicDemoExportError(
            "release exports require both release_version and release_tag"
        )
    try:
        validate_release_identity(release_version, release_tag)
    except ValueError as exc:
        raise PublicDemoExportError(str(exc)) from exc


def _validate_provenance_inputs(
    payload: dict[str, object],
    *,
    channel: Literal["preview", "release"],
) -> None:
    if payload.get("generated_at") != _GENERATED_AT:
        raise PublicDemoExportError(
            "provenance generated_at must use the deterministic Unix epoch"
        )
    coverage = payload.get("coverage")
    if (
        not isinstance(coverage, dict)
        or coverage.get("complete") is not True
        or coverage.get("excluded_self") != _PROVENANCE_NAME
    ):
        raise PublicDemoExportError("provenance coverage declaration is invalid")
    inputs_value = payload.get("inputs")
    if not isinstance(inputs_value, list) or not all(
        isinstance(item, dict) for item in inputs_value
    ):
        raise PublicDemoExportError("provenance inputs must be a list of objects")
    inputs = cast(list[dict[str, object]], inputs_value)
    kinds: list[str] = []
    for item in inputs:
        kind = item.get("kind")
        manifest_name = item.get("manifest_name")
        manifest_sha256 = item.get("manifest_sha256")
        evidence_status = item.get("evidence_status")
        configuration_digests = item.get("configuration_digests")
        if (
            kind not in {"deterministic", "research", "simulator"}
            or not isinstance(manifest_name, str)
            or not manifest_name
            or PurePosixPath(manifest_name).name != manifest_name
            or not isinstance(manifest_sha256, str)
            or not _is_sha256(manifest_sha256)
            or evidence_status not in {"fixture", "canonical"}
            or not isinstance(configuration_digests, list)
            or not all(
                isinstance(digest, str) and _is_sha256(digest)
                for digest in configuration_digests
            )
        ):
            raise PublicDemoExportError(
                "provenance contains a malformed input descriptor"
            )
        try:
            input_source = SourceIdentity.model_validate(item.get("source"))
        except ValueError as exc:
            raise PublicDemoExportError(
                f"provenance input source identity is invalid: {exc}"
            ) from exc
        protocol_digest = item.get("protocol_digest")
        if (
            protocol_digest is not None
            and (
                not isinstance(protocol_digest, str)
                or not _is_sha256(protocol_digest)
            )
        ):
            raise PublicDemoExportError(
                "provenance input protocol digest is invalid"
            )
        kinds.append(kind)
        if channel == "release":
            if (
                evidence_status != "canonical"
                or input_source.dirty
                or not _is_sha256(input_source.tree_digest)
            ):
                raise PublicDemoExportError(
                    "release provenance inputs must be clean canonical evidence"
                )
            if kind == "deterministic" and not configuration_digests:
                raise PublicDemoExportError(
                    "release deterministic provenance must declare at least "
                    "one configuration digest"
                )
    if len(kinds) != len(set(kinds)):
        raise PublicDemoExportError("provenance input kinds must be unique")
    if channel == "release" and set(kinds) != {
        "deterministic",
        "research",
        "simulator",
    }:
        raise PublicDemoExportError(
            "release provenance must contain deterministic, research, and "
            "simulator inputs"
        )
    try:
        source = SourceIdentity.model_validate(payload.get("source"))
    except ValueError as exc:
        raise PublicDemoExportError(
            f"provenance source identity is invalid: {exc}"
        ) from exc
    if channel == "release" and (
        source.dirty or not _is_sha256(source.tree_digest)
    ):
        raise PublicDemoExportError(
            "release provenance must record a clean source identity"
        )


def _validate_release_status(
    root: Path,
    *,
    channel: Literal["preview", "release"],
    release_version: str | None,
    release_tag: str | None,
) -> None:
    status = _read_object(root / "release-status.json", "release status")
    if (
        status.get("schema_version")
        != "construction-intelligence-public-status-v1"
        or status.get("channel") != channel
        or status.get("release_ready") is not (channel == "release")
    ):
        raise PublicDemoExportError(
            "release status does not match the provenance channel"
        )
    status_version = status.get("release_version")
    status_tag = status.get("release_tag")
    if status_version is not None and not isinstance(status_version, str):
        raise PublicDemoExportError(
            "release status version must be a string or null"
        )
    if status_tag is not None and not isinstance(status_tag, str):
        raise PublicDemoExportError(
            "release status tag must be a string or null"
        )
    if status_version != release_version or status_tag != release_tag:
        raise PublicDemoExportError(
            "release status version or tag does not match provenance"
        )
    evidence = status.get("evidence")
    if channel == "release" and (
        status.get("claim_status") != "release_evidence"
        or status.get("limitations") != []
        or evidence
        != {
            "deterministic": "canonical",
            "research": "validated",
            "coppelia": "validated",
        }
    ):
        raise PublicDemoExportError(
            "release status does not attest complete canonical evidence"
        )


def _write_release_status(
    staging: Path,
    *,
    channel: Literal["preview", "release"],
    deterministic: _LoadedBundle,
    research: _LoadedBundle | None,
    simulator: _LoadedBundle | None,
    release_version: str | None,
    release_tag: str | None,
) -> None:
    research_present = research is not None
    simulator_present = simulator is not None
    claim_status = (
        "release_evidence"
        if channel == "release"
        else (
            "canonical_preview"
            if deterministic.manifest.evidence_status == "canonical"
            else "fixture_preview"
        )
    )
    limitations: list[str] = []
    if not research_present:
        limitations.append(
            "Canonical 20-run research evidence is absent; no learned-policy "
            "release claim is made."
        )
    if not simulator_present:
        limitations.append(
            "Live nominal and recovery Coppelia evidence is absent; deterministic "
            "traces do not substitute for simulator execution."
        )
    if deterministic.manifest.evidence_status == "fixture":
        limitations.append(
            "Deterministic content is a reviewed fixture intended for preview use."
        )
    _write_json(
        staging / "release-status.json",
        {
            "schema_version": "construction-intelligence-public-status-v1",
            "channel": channel,
            "release_version": release_version,
            "release_tag": release_tag,
            "claim_status": claim_status,
            "release_ready": channel == "release",
            "evidence": {
                "deterministic": deterministic.manifest.evidence_status,
                "research": "validated" if research_present else "absent",
                "coppelia": "validated" if simulator_present else "absent",
            },
            "limitations": limitations,
        },
    )


def _build_provenance(
    staging: Path,
    *,
    channel: Literal["preview", "release"],
    source: SourceIdentity,
    bundles: list[_LoadedBundle | None],
    artifact_sources: dict[str, tuple[str, str]],
    release_version: str | None,
    release_tag: str | None,
) -> dict[str, object]:
    observed_paths = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != _PROVENANCE_NAME
    )
    if set(observed_paths) != set(artifact_sources):
        missing_sources = sorted(set(observed_paths) - set(artifact_sources))
        missing_files = sorted(set(artifact_sources) - set(observed_paths))
        raise PublicDemoExportError(
            "internal provenance coverage mismatch; "
            f"unclassified={missing_sources}, missing={missing_files}"
        )
    artifacts = []
    for relative in observed_paths:
        path = staging / relative
        source_kind, role = artifact_sources[relative]
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "source_kind": source_kind,
                "role": role,
            }
        )
    inputs = [
        {
            "kind": bundle.manifest.kind,
            "manifest_name": bundle.path.name,
            "manifest_sha256": bundle.sha256,
            "evidence_status": bundle.manifest.evidence_status,
            "source": bundle.manifest.source.model_dump(mode="json"),
            "configuration_digests": sorted(
                bundle.manifest.configuration_digests
            ),
            "protocol_digest": bundle.manifest.protocol_digest,
            "profile": bundle.manifest.profile,
            "matrix_id": bundle.manifest.matrix_id,
        }
        for bundle in bundles
        if bundle is not None
    ]
    payload: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "channel": channel,
        "release_version": release_version,
        "release_tag": release_tag,
        "source": source.model_dump(mode="json"),
        "generated_at": _GENERATED_AT,
        "timestamp_strategy": (
            "No wall-clock time is embedded. generated_at is the fixed Unix epoch; "
            "evidence timestamps come only from hash-pinned input bundles."
        ),
        "ordering_strategy": (
            "JSON object keys, input descriptors, and artifact paths are "
            "lexicographically sorted."
        ),
        "inputs": sorted(inputs, key=lambda item: str(item["kind"])),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "coverage": {
            "complete": True,
            "excluded_self": _PROVENANCE_NAME,
            "reason": (
                "A manifest cannot contain the hash of its own final bytes without "
                "self-reference. Its canonical payload hash is recorded separately."
            ),
        },
        "manifest_payload_sha256": None,
    }
    payload["manifest_payload_sha256"] = _sha256_bytes(_json_bytes(payload))
    return payload


def _public_selection_record(
    item: dict[str, object],
    frozen_at: str,
) -> dict[str, object]:
    run_key = item.get("run_key")
    selection_value = item.get("selection", item.get("selected"))
    if not isinstance(run_key, str) or not isinstance(selection_value, dict):
        raise PublicDemoExportError("selection record is missing run_key or selection")
    return {
        "run_key": run_key,
        "selection": selection_value,
        "frozen_at": str(item.get("frozen_at") or frozen_at),
    }


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicDemoExportError(f"{label} is invalid JSON: {exc}") from exc


def _read_object(path: Path, label: str) -> dict[str, object]:
    payload = _read_json(path, label)
    if not isinstance(payload, dict):
        raise PublicDemoExportError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _require_sha256(value: str, label: str) -> None:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _safe_relative_path(value: str) -> PurePosixPath:
    if "\\" in value:
        raise ValueError("artifact targets must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"artifact target is not a safe relative path: {value}")
    return path


def _replace_directory(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists():
        backup = destination.with_name(
            f".{destination.name}.previous-{os.getpid()}"
        )
        if backup.exists():
            shutil.rmtree(backup)
        destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)
