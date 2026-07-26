from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, TypeAdapter

from embodied_skill_composer.construction.evaluation import EvaluationSuite
from embodied_skill_composer.construction.experiment_execution import (
    MatrixSelectionEvidence,
    PrimaryAcceptanceAudit,
)
from embodied_skill_composer.construction.experiment_protocol import (
    AblationDecision,
    ExperimentProtocol,
    expand_experiment_matrix,
    protocol_digest,
)
from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.policy import file_sha256
from embodied_skill_composer.construction.training import TrainingConfig


class ReproducibilityArtifactAudit(BaseModel):
    matrix_id: str
    complete: bool
    checked_file_count: int = Field(ge=0)
    file_hashes: dict[str, str]
    missing_paths: list[str]
    invalid_artifacts: list[str]


def audit_reproducibility_artifacts(
    registry: LabRegistry,
    matrix_id: str,
    protocol: ExperimentProtocol,
    selection_evidence: MatrixSelectionEvidence,
    suite: EvaluationSuite,
    *,
    heldout_run_dir: Path,
    acceptance_path: Path,
    ablation_path: Path,
) -> ReproducibilityArtifactAudit:
    """Verify the canonical research inputs and outputs instead of asserting completeness."""

    missing: list[str] = []
    invalid: list[str] = []
    hashes: dict[str, str] = {}

    def require_file(path: Path) -> bool:
        resolved = path.resolve()
        key = str(resolved)
        if not resolved.is_file():
            missing.append(key)
            return False
        try:
            hashes[key] = file_sha256(resolved)
        except OSError as exc:
            invalid.append(f"{key}: cannot hash file: {exc}")
            return False
        return True

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        invalid.append(f"matrix not found: {matrix_id}")
        return _artifact_audit(matrix_id, hashes, missing, invalid)
    runs_value = matrix.get("runs")
    if not isinstance(runs_value, list) or not all(
        isinstance(item, dict) for item in runs_value
    ):
        invalid.append("matrix runs are malformed")
        return _artifact_audit(matrix_id, hashes, missing, invalid)
    runs = cast(list[dict[str, object]], runs_value)
    expected_protocol_digest = protocol_digest(protocol)
    if matrix.get("protocol_digest") != expected_protocol_digest:
        invalid.append("matrix protocol digest does not match")
    if matrix.get("execution_profile") != "research":
        invalid.append("reproducibility release audit requires the research profile")
    expected_run_keys = {
        spec.run_id for spec in expand_experiment_matrix(protocol, "research")
    }
    observed_run_keys = {str(run.get("run_key", "")) for run in runs}
    if observed_run_keys != expected_run_keys:
        invalid.append("matrix run keys do not match the frozen research expansion")
    if selection_evidence.matrix_id != matrix_id:
        invalid.append("selection evidence matrix id does not match")
    if selection_evidence.protocol_digest != expected_protocol_digest:
        invalid.append("selection evidence protocol digest does not match")
    if {
        selection.run_key for selection in selection_evidence.selections
    } != expected_run_keys:
        invalid.append("selection evidence does not cover every research run")
    if require_file(selection_evidence.evidence_path):
        try:
            persisted_selection = MatrixSelectionEvidence.model_validate_json(
                selection_evidence.evidence_path.read_text(encoding="utf-8")
            )
            if persisted_selection != selection_evidence:
                invalid.append("persisted matrix selection evidence differs from memory")
        except (OSError, ValueError) as exc:
            invalid.append(f"matrix selection evidence is invalid: {exc}")

    selection_by_run = {
        selection.run_key: selection for selection in selection_evidence.selections
    }
    frozen_by_run = {
        str(item["run_key"]): cast(dict[str, object], item["selection"])
        for item in registry.list_policy_selections(matrix_id)
    }
    if set(frozen_by_run) != expected_run_keys:
        invalid.append("registry does not contain all 20 frozen selections")
    percentages = [
        int(round(fraction * 100))
        for fraction in protocol.checkpoint_fractions
    ]
    for run in runs:
        run_key = str(run.get("run_key", ""))
        if str(run.get("status", "")) != "completed":
            invalid.append(f"{run_key}: training run is not completed")
        artifact_dir_value = run.get("artifact_dir")
        if not isinstance(artifact_dir_value, str) or not artifact_dir_value:
            invalid.append(f"{run_key}: artifact directory is missing")
            continue
        artifact_dir = Path(artifact_dir_value)
        canonical_files = [
            artifact_dir / "training_config.json",
            artifact_dir / "learning_curve.csv",
            artifact_dir / "policy_manifest.json",
            artifact_dir / "policy.pt",
            artifact_dir / "actor.onnx",
        ]
        for percentage in percentages:
            canonical_files.extend(
                [
                    artifact_dir
                    / "checkpoints"
                    / f"checkpoint_{percentage:03d}pct.pt",
                    artifact_dir
                    / "checkpoints"
                    / f"policy_{percentage:03d}pct.pt",
                ]
            )
        for path in canonical_files:
            require_file(path)
        _validate_training_artifacts(
            run_key,
            run,
            artifact_dir,
            invalid,
        )
        selection = selection_by_run.get(run_key)
        if selection is None:
            continue
        observed_fractions = sorted(
            candidate.result.checkpoint_fraction
            for candidate in selection.candidates
        )
        if (
            len(selection.candidates) != len(protocol.checkpoint_fractions)
            or observed_fractions != sorted(protocol.checkpoint_fractions)
        ):
            invalid.append(
                f"{run_key}: validation evidence does not contain the exact "
                "five-checkpoint grid"
            )
        if any(
            len(candidate.episodes)
            != (
                len(protocol.selection.scenario_seeds)
                * len(protocol.selection.failure_modes)
            )
            for candidate in selection.candidates
        ):
            invalid.append(
                f"{run_key}: validation evidence has an incomplete episode grid"
            )
        if (
            selection.selected.model_dump(mode="json")
            != frozen_by_run.get(run_key)
        ):
            invalid.append(
                f"{run_key}: selection evidence differs from the frozen registry record"
            )
        require_file(selection.evidence_path)
        for candidate in selection.candidates:
            checkpoint_path = Path(candidate.result.checkpoint_path)
            if require_file(checkpoint_path):
                actual_sha = hashes[str(checkpoint_path.resolve())]
                if actual_sha != candidate.result.checkpoint_sha256:
                    invalid.append(
                        f"{run_key}: candidate checkpoint hash mismatch: "
                        f"{checkpoint_path}"
                    )

    evaluation_json = heldout_run_dir / "evaluation.json"
    episodes_csv = heldout_run_dir / "episodes.csv"
    report_path = heldout_run_dir / "report.md"
    for path in (
        evaluation_json,
        episodes_csv,
        report_path,
        acceptance_path,
        ablation_path,
    ):
        require_file(path)
    if evaluation_json.is_file():
        try:
            persisted_suite = EvaluationSuite.model_validate_json(
                evaluation_json.read_text(encoding="utf-8")
            )
            if persisted_suite != suite:
                invalid.append("persisted held-out evaluation differs from memory")
            if (
                persisted_suite.grid_validation is None
                or not persisted_suite.grid_validation.complete
                or persisted_suite.grid_validation.observed_episode_count != 240
                or len(persisted_suite.episodes) != 240
            ):
                invalid.append("held-out evaluation grid is not the complete 240 episodes")
        except (OSError, ValueError) as exc:
            invalid.append(f"held-out evaluation JSON is invalid: {exc}")
    if episodes_csv.is_file():
        try:
            with episodes_csv.open(encoding="utf-8", newline="") as handle:
                episode_rows = sum(1 for _ in csv.DictReader(handle))
            if episode_rows != 240:
                invalid.append(
                    f"held-out episode CSV has {episode_rows} rows instead of 240"
                )
        except OSError as exc:
            invalid.append(f"held-out episode CSV is unreadable: {exc}")
    _validate_json_artifact(acceptance_path, "acceptance", invalid)
    _validate_json_artifact(ablation_path, "ablation", invalid)
    if acceptance_path.is_file():
        try:
            acceptance = PrimaryAcceptanceAudit.model_validate_json(
                acceptance_path.read_text(encoding="utf-8")
            )
            expected_acceptance_names = {
                "mappo_no_failure_mean_completion",
                "ippo_no_failure_mean_completion",
                "mappo_median_makespan_cp_sat_ratio",
                "mappo_failure_mean_completion",
            }
            if (
                not acceptance.passed
                or {item.name for item in acceptance.results}
                != expected_acceptance_names
                or not all(item.passed for item in acceptance.results)
            ):
                invalid.append("primary acceptance thresholds did not pass")
        except (OSError, ValueError) as exc:
            invalid.append(f"acceptance artifact schema is invalid: {exc}")
    if ablation_path.is_file():
        try:
            payload = json.loads(ablation_path.read_text(encoding="utf-8"))
            decisions = TypeAdapter(list[AblationDecision]).validate_python(
                payload
            )
            if {item.hypothesis for item in decisions} != {
                "behavior_cloning",
                "failure_curriculum",
            }:
                invalid.append(
                    "ablation artifact does not contain both pre-registered decisions"
                )
        except (OSError, ValueError) as exc:
            invalid.append(f"ablation artifact schema is invalid: {exc}")
    return _artifact_audit(matrix_id, hashes, missing, invalid)


def _validate_training_artifacts(
    run_key: str,
    run: dict[str, object],
    artifact_dir: Path,
    invalid: list[str],
) -> None:
    config_path = artifact_dir / "training_config.json"
    manifest_path = artifact_dir / "policy_manifest.json"
    try:
        persisted_config = TrainingConfig.model_validate_json(
            config_path.read_text(encoding="utf-8")
        )
        run_config = TrainingConfig.model_validate(run.get("config"))
        if (
            persisted_config.configuration_digest
            != run_config.configuration_digest
            or persisted_config.source_commit != run_config.source_commit
            or persisted_config.experiment_variant
            != run_config.experiment_variant
            or persisted_config.training_seed != run_config.training_seed
        ):
            invalid.append(f"{run_key}: persisted training config provenance mismatch")
    except (OSError, ValueError) as exc:
        invalid.append(f"{run_key}: invalid training config: {exc}")
    curve_path = artifact_dir / "learning_curve.csv"
    if curve_path.is_file():
        try:
            with curve_path.open(encoding="utf-8", newline="") as handle:
                if not list(csv.DictReader(handle)):
                    invalid.append(f"{run_key}: learning curve has no updates")
        except OSError as exc:
            invalid.append(f"{run_key}: learning curve is unreadable: {exc}")
    try:
        manifest = PolicyManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if (
            manifest.configuration_digest
            != cast(dict[str, object], run.get("config", {})).get(
                "configuration_digest"
            )
            or manifest.experiment_variant
            != cast(dict[str, object], run.get("config", {})).get(
                "experiment_variant"
            )
        ):
            invalid.append(f"{run_key}: policy manifest provenance mismatch")
        if manifest.checkpoint_path is None:
            invalid.append(f"{run_key}: policy manifest checkpoint path is missing")
        else:
            checkpoint_path = Path(manifest.checkpoint_path)
            if checkpoint_path.resolve() != (artifact_dir / "policy.pt").resolve():
                invalid.append(f"{run_key}: final policy checkpoint path mismatch")
            elif checkpoint_path.is_file() and (
                file_sha256(checkpoint_path) != manifest.checkpoint_sha256
            ):
                invalid.append(f"{run_key}: final policy checkpoint hash mismatch")
    except (OSError, ValueError) as exc:
        invalid.append(f"{run_key}: invalid policy manifest: {exc}")


def _validate_json_artifact(
    path: Path,
    label: str,
    invalid: list[str],
) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        invalid.append(f"{label} artifact is invalid JSON: {exc}")
        return
    if not isinstance(payload, (dict, list)):
        invalid.append(f"{label} artifact must contain an object or array")


def _artifact_audit(
    matrix_id: str,
    hashes: dict[str, str],
    missing: list[str],
    invalid: list[str],
) -> ReproducibilityArtifactAudit:
    return ReproducibilityArtifactAudit(
        matrix_id=matrix_id,
        complete=not missing and not invalid,
        checked_file_count=len(hashes),
        file_hashes=dict(sorted(hashes.items())),
        missing_paths=sorted(set(missing)),
        invalid_artifacts=sorted(set(invalid)),
    )
