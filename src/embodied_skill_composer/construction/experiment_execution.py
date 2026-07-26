from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
from pydantic import BaseModel

from embodied_skill_composer.construction.evaluation import (
    ControllerEvaluation,
    ControllerName,
    EpisodeEvaluation,
    EvaluationGridEntry,
    EvaluationSuite,
    evaluate_controller_episode,
    summarize_by_scenario_seed,
    summarize_by_training_seed,
    summarize_evaluations,
    validate_evaluation_grid,
    write_evaluation_artifacts,
)
from embodied_skill_composer.construction.experiment_protocol import (
    AblationDecision,
    AblationEvidence,
    CheckpointValidationResult,
    ExperimentProfile,
    ExperimentProtocol,
    SelectedCheckpoint,
    decide_ablation_support,
    expand_experiment_matrix,
    protocol_digest,
    select_validation_checkpoint,
)
from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.policy import (
    file_sha256,
    load_policy_checkpoint,
    load_policy_checkpoint_metadata,
)
from embodied_skill_composer.construction.training import (
    TrainingConfig,
    configuration_digest,
    design_digest,
)


BASELINE_CONTROLLERS: tuple[ControllerName, ...] = (
    "sequential",
    "greedy",
    "auction",
    "cp_sat",
)

_MATRIX_RUNTIME_CONFIG_FIELDS = {
    "checkpoint_lineage",
    "configuration_digest",
    "environment_fingerprint",
    "output_root",
    "resume_checkpoint",
    "resume_provenance",
    "source_commit",
    "source_dirty",
    "source_tree_digest",
}


class CandidateEvaluation(BaseModel):
    result: CheckpointValidationResult
    episodes: list[EpisodeEvaluation]


class RunSelectionEvidence(BaseModel):
    matrix_id: str
    run_key: str
    candidates: list[CandidateEvaluation]
    selected: SelectedCheckpoint
    evidence_path: Path


class MatrixSelectionEvidence(BaseModel):
    matrix_id: str
    protocol_digest: str
    selections: list[RunSelectionEvidence]
    evidence_path: Path


class AcceptanceResult(BaseModel):
    name: str
    passed: bool
    observed: float
    threshold: float
    comparison: Literal["min", "max"]


class PrimaryAcceptanceAudit(BaseModel):
    passed: bool
    results: list[AcceptanceResult]


def evaluate_and_freeze_matrix_selections(
    registry: LabRegistry,
    matrix_id: str,
    design: HouseDesign,
    protocol: ExperimentProtocol,
    *,
    output_root: Path,
    device: str = "cpu",
) -> MatrixSelectionEvidence:
    """Evaluate every fractional checkpoint on validation seeds and freeze selections."""

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise KeyError(matrix_id)
    if str(matrix["protocol_digest"]) != _protocol_digest_from_matrix(protocol, matrix):
        raise ValueError("matrix protocol digest does not match the supplied protocol")
    runs = _matrix_runs(matrix)
    _require_canonical_matrix(matrix, runs, protocol)
    _require_matrix_design(runs, design)
    incomplete = [
        str(run["run_key"])
        for run in runs
        if str(run["status"]) != "completed"
    ]
    if incomplete:
        raise ValueError(
            "all training runs must complete before validation selection: "
            + ", ".join(incomplete)
        )

    matrix_root = output_root.resolve() / matrix_id
    matrix_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        _evaluate_and_freeze_run_selection(
            registry,
            matrix_id,
            design,
            run,
            protocol,
            matrix_root=matrix_root,
            device=device,
        )
    matrix_evidence = materialize_matrix_selection_evidence(
        registry,
        matrix_id,
        protocol,
        output_root=output_root,
    )
    if matrix_evidence is None:
        raise RuntimeError("all matrix selections were frozen but evidence is incomplete")
    return matrix_evidence


def evaluate_and_freeze_run_selection(
    registry: LabRegistry,
    matrix_id: str,
    run_key: str,
    design: HouseDesign,
    protocol: ExperimentProtocol,
    *,
    output_root: Path,
    device: str = "cpu",
) -> RunSelectionEvidence:
    """Evaluate one completed run and freeze only the server-computed selection."""

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise KeyError(matrix_id)
    if str(matrix["protocol_digest"]) != _protocol_digest_from_matrix(protocol, matrix):
        raise ValueError("matrix protocol digest does not match the supplied protocol")
    runs = _matrix_runs(matrix)
    _require_canonical_matrix(matrix, runs, protocol)
    _require_matrix_design(runs, design)
    run = next((item for item in runs if str(item["run_key"]) == run_key), None)
    if run is None:
        raise KeyError(f"{matrix_id}:{run_key}")
    if str(run["status"]) != "completed":
        raise ValueError(f"selection requires a completed training run: {run_key}")
    matrix_root = output_root.resolve() / matrix_id
    matrix_root.mkdir(parents=True, exist_ok=True)
    return _evaluate_and_freeze_run_selection(
        registry,
        matrix_id,
        design,
        run,
        protocol,
        matrix_root=matrix_root,
        device=device,
    )


def materialize_matrix_selection_evidence(
    registry: LabRegistry,
    matrix_id: str,
    protocol: ExperimentProtocol,
    *,
    output_root: Path,
) -> MatrixSelectionEvidence | None:
    """Write canonical matrix evidence after every frozen run has persisted evidence."""

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise KeyError(matrix_id)
    if str(matrix["protocol_digest"]) != _protocol_digest_from_matrix(protocol, matrix):
        raise ValueError("matrix protocol digest does not match the supplied protocol")
    runs = _matrix_runs(matrix)
    _require_canonical_matrix(matrix, runs, protocol)
    frozen_records = {
        str(item["run_key"]): _required_dict(item, "selection")
        for item in registry.list_policy_selections(matrix_id)
    }
    if len(frozen_records) != len(runs):
        return None
    matrix_root = output_root.resolve() / matrix_id
    selections: list[RunSelectionEvidence] = []
    for run in runs:
        run_key = _required_string(run, "run_key")
        evidence_path = matrix_root / run_key / "validation_selection.json"
        if not evidence_path.is_file():
            raise ValueError(
                f"frozen selection evidence is missing for {run_key}: {evidence_path}"
            )
        evidence = RunSelectionEvidence.model_validate_json(
            evidence_path.read_text(encoding="utf-8")
        )
        if (
            evidence.matrix_id != matrix_id
            or evidence.run_key != run_key
            or evidence.evidence_path.resolve() != evidence_path.resolve()
        ):
            raise ValueError(f"selection evidence identity mismatch for {run_key}")
        if (
            evidence.selected.model_dump(mode="json")
            != frozen_records.get(run_key)
        ):
            raise ValueError(
                f"selection evidence does not match the frozen record for {run_key}"
            )
        selections.append(evidence)
    evidence_path = matrix_root / "matrix_selections.json"
    matrix_evidence = MatrixSelectionEvidence(
        matrix_id=matrix_id,
        protocol_digest=str(matrix["protocol_digest"]),
        selections=selections,
        evidence_path=evidence_path,
    )
    evidence_path.write_text(
        matrix_evidence.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return matrix_evidence


def _evaluate_and_freeze_run_selection(
    registry: LabRegistry,
    matrix_id: str,
    design: HouseDesign,
    run: dict[str, object],
    protocol: ExperimentProtocol,
    *,
    matrix_root: Path,
    device: str,
) -> RunSelectionEvidence:
    evidence = evaluate_run_checkpoint_candidates(
        design,
        run,
        protocol,
        matrix_id=matrix_id,
        output_root=matrix_root,
        device=device,
    )
    run_key = _required_string(run, "run_key")
    if evidence.matrix_id != matrix_id or evidence.run_key != run_key:
        raise ValueError(f"selection evaluator returned the wrong identity for {run_key}")
    evidence_path = matrix_root / run_key / "validation_selection.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = evidence.model_copy(update={"evidence_path": evidence_path})
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    registry.freeze_policy_selection(
        matrix_id,
        run_key,
        evidence.selected.model_dump(mode="json"),
    )
    config = TrainingConfig.model_validate(_required_dict(run, "config"))
    selected_manifest = _selected_policy_manifest(
        matrix_id,
        run_key,
        evidence.selected,
        config,
    )
    registry.upsert_policy(
        selected_manifest.policy_id,
        selected_manifest.controller,
        selected_manifest.model_dump(mode="json"),
    )
    return evidence


def evaluate_run_checkpoint_candidates(
    design: HouseDesign,
    run: dict[str, object],
    protocol: ExperimentProtocol,
    *,
    matrix_id: str,
    output_root: Path,
    device: str = "cpu",
) -> RunSelectionEvidence:
    """Evaluate exactly five checkpoints without accepting held-out scenario seeds."""

    if any(seed >= 900 for seed in protocol.selection.scenario_seeds):
        raise ValueError("held-out scenario seeds cannot be used for checkpoint selection")
    run_key = _required_string(run, "run_key")
    artifact_dir = Path(_required_string(run, "artifact_dir"))
    config = TrainingConfig.model_validate(_required_dict(run, "config"))
    resume_history = _run_resume_provenance_history(run, config)
    if config.configuration_digest is None or config.source_commit is None:
        raise ValueError(f"run {run_key} is missing reproducibility fingerprints")
    candidates: list[CandidateEvaluation] = []
    for fraction in protocol.checkpoint_fractions:
        percentage = int(round(fraction * 100))
        checkpoint_path = artifact_dir / "checkpoints" / f"policy_{percentage:03d}pct.pt"
        if not checkpoint_path.is_file():
            raise ValueError(
                f"run {run_key} is missing the {percentage}% policy checkpoint: "
                f"{checkpoint_path}"
            )
        metadata = _policy_checkpoint_metadata(checkpoint_path)
        transition_count = _required_int(metadata, "transition_count")
        checkpoint_fraction = _required_float(metadata, "checkpoint_fraction")
        if not np.isclose(checkpoint_fraction, fraction, rtol=0, atol=1e-12):
            raise ValueError(
                f"checkpoint fraction mismatch for {checkpoint_path}: "
                f"{checkpoint_fraction} != {fraction}"
            )
        checkpoint_resume_provenance = _validate_checkpoint_provenance(
            metadata,
            config=config,
            design=design,
            checkpoint_path=checkpoint_path,
            resume_history=resume_history,
        )
        lineage = _string_list(metadata.get("checkpoint_lineage"), "checkpoint_lineage")
        checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint_id = f"{run_key}-checkpoint-{percentage:03d}pct"
        manifest = _candidate_policy_manifest(
            checkpoint_id=checkpoint_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha=checkpoint_sha,
            transition_count=transition_count,
            config=config,
            lineage=lineage,
            resume_provenance=checkpoint_resume_provenance,
        )
        bundle = load_policy_checkpoint(checkpoint_path, device=device)
        if bundle.algorithm != config.algorithm:
            raise ValueError(
                f"checkpoint algorithm mismatch for {checkpoint_path}: "
                f"{bundle.algorithm} != {config.algorithm}"
            )
        episodes = [
            evaluate_controller_episode(
                design,
                seed=seed,
                controller=config.algorithm,
                bundle=bundle,
                policy_manifest=manifest,
                failure_enabled=failure_enabled,
                device=device,
                expected_split=protocol.selection.split,
            )
            for failure_enabled in protocol.selection.failure_modes
            for seed in protocol.selection.scenario_seeds
        ]
        result = CheckpointValidationResult(
            checkpoint_id=checkpoint_id,
            experiment_id=config.experiment_id,
            experiment_variant=config.experiment_variant,
            training_seed=config.training_seed or config.seed,
            checkpoint_fraction=fraction,
            transition_count=transition_count,
            split="validation",
            scenario_seeds=list(protocol.selection.scenario_seeds),
            mean_completion_rate=float(
                np.mean([episode.structure_completion_rate for episode in episodes])
            ),
            mean_makespan_s=float(np.mean([episode.makespan_s for episode in episodes])),
            checkpoint_path=str(checkpoint_path.resolve()),
            checkpoint_sha256=checkpoint_sha,
            checkpoint_lineage=lineage,
            configuration_digest=config.configuration_digest,
            source_commit=config.source_commit,
            resume_provenance=checkpoint_resume_provenance,
        )
        candidates.append(CandidateEvaluation(result=result, episodes=episodes))
    selected = select_validation_checkpoint(
        [candidate.result for candidate in candidates],
        validation_seeds=protocol.selection.scenario_seeds,
        required_fractions=protocol.checkpoint_fractions,
    )
    run_root = output_root / run_key
    run_root.mkdir(parents=True, exist_ok=True)
    evidence_path = run_root / "validation_selection.json"
    evidence = RunSelectionEvidence(
        matrix_id=matrix_id,
        run_key=run_key,
        candidates=candidates,
        selected=selected,
        evidence_path=evidence_path,
    )
    evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
    return evidence


def evaluate_frozen_heldout_matrix(
    registry: LabRegistry,
    matrix_id: str,
    design: HouseDesign,
    protocol: ExperimentProtocol,
    *,
    output_root: Path,
    device: str = "cpu",
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[EvaluationSuite, PrimaryAcceptanceAudit]:
    """Evaluate frozen selected policies and baselines on the held-out split."""

    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise KeyError(matrix_id)
    if str(matrix["protocol_digest"]) != _protocol_digest_from_matrix(protocol, matrix):
        raise ValueError("matrix protocol digest does not match the supplied protocol")
    runs = _matrix_runs(matrix)
    _require_canonical_matrix(matrix, runs, protocol)
    _require_matrix_design(runs, design)
    selections = {
        str(item["run_key"]): SelectedCheckpoint.model_validate(item["selection"])
        for item in registry.list_policy_selections(matrix_id)
    }
    if len(runs) != 20 or set(selections) != {
        str(run["run_key"]) for run in runs
    }:
        raise ValueError("held-out evaluation requires all 20 frozen selections")
    if any(seed < 900 for seed in protocol.evaluation.scenario_seeds):
        raise ValueError("final evaluation requires only held-out test seeds")

    episodes: list[EpisodeEvaluation] = []
    policy_grid: list[EvaluationGridEntry] = []
    for controller in BASELINE_CONTROLLERS:
        _raise_if_cancelled(cancel_check)
        policy_grid.append(EvaluationGridEntry(controller=controller))
        for failure_enabled in protocol.evaluation.failure_modes:
            for seed in protocol.evaluation.scenario_seeds:
                _raise_if_cancelled(cancel_check)
                episodes.append(
                    evaluate_controller_episode(
                        design,
                        seed=seed,
                        controller=controller,
                        failure_enabled=failure_enabled,
                        device=device,
                        expected_split=protocol.evaluation.split,
                    )
                )
    for run in runs:
        _raise_if_cancelled(cancel_check)
        run_key = str(run["run_key"])
        config = TrainingConfig.model_validate(_required_dict(run, "config"))
        selection = selections[run_key]
        if (
            selection.experiment_id != config.experiment_id
            or selection.experiment_variant != config.experiment_variant
            or selection.training_seed != config.training_seed
        ):
            raise ValueError(f"selected checkpoint identity mismatch for {run_key}")
        if selection.scenario_seeds != protocol.selection.scenario_seeds:
            raise ValueError(
                f"selected checkpoint used an invalid validation grid for {run_key}"
            )
        if selection.configuration_digest != config.configuration_digest:
            raise ValueError(
                f"selected checkpoint configuration digest mismatch for {run_key}"
            )
        if selection.source_commit != config.source_commit:
            raise ValueError(
                f"selected checkpoint source commit mismatch for {run_key}"
            )
        checkpoint_path = Path(selection.checkpoint_path)
        actual_sha = file_sha256(checkpoint_path)
        if actual_sha != selection.checkpoint_sha256:
            raise ValueError(
                f"selected checkpoint hash mismatch for {run_key}: "
                f"{actual_sha} != {selection.checkpoint_sha256}"
            )
        metadata = _policy_checkpoint_metadata(checkpoint_path)
        _validate_checkpoint_provenance(
            metadata,
            config=config,
            design=design,
            checkpoint_path=checkpoint_path,
            resume_history=_run_resume_provenance_history(run, config),
            expected_resume_provenance=selection.resume_provenance,
        )
        if _required_int(metadata, "transition_count") != selection.transition_count:
            raise ValueError(
                f"selected checkpoint transition count mismatch for {run_key}"
            )
        if not np.isclose(
            _required_float(metadata, "checkpoint_fraction"),
            selection.checkpoint_fraction,
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"selected checkpoint fraction mismatch for {run_key}"
            )
        manifest = _selected_policy_manifest(
            matrix_id,
            run_key,
            selection,
            config,
        )
        bundle = load_policy_checkpoint(checkpoint_path, device=device)
        policy_grid.append(
            EvaluationGridEntry(
                controller=config.algorithm,
                experiment_id=config.experiment_id,
                experiment_variant=config.experiment_variant,
                training_seed=config.training_seed,
            )
        )
        for failure_enabled in protocol.evaluation.failure_modes:
            for seed in protocol.evaluation.scenario_seeds:
                _raise_if_cancelled(cancel_check)
                episodes.append(
                    evaluate_controller_episode(
                        design,
                        seed=seed,
                        controller=config.algorithm,
                        bundle=bundle,
                        policy_manifest=manifest,
                        failure_enabled=failure_enabled,
                        device=device,
                        expected_split=protocol.evaluation.split,
                    )
                )
    _raise_if_cancelled(cancel_check)
    grid_validation = validate_evaluation_grid(
        episodes,
        scenario_seeds=protocol.evaluation.scenario_seeds,
        policy_grid=policy_grid,
        failure_modes=protocol.evaluation.failure_modes,
        expected_split=protocol.evaluation.split,
    )
    evaluation_id = (
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{protocol.experiment_id}-heldout"
    )
    suite = EvaluationSuite(
        evaluation_id=evaluation_id,
        seeds=list(protocol.evaluation.scenario_seeds),
        controllers=list(protocol.evaluation.controllers),
        episodes=episodes,
        summaries=summarize_evaluations(episodes),
        expected_split=protocol.evaluation.split,
        grid_validation=grid_validation,
        per_training_seed=summarize_by_training_seed(episodes),
        per_scenario_seed=summarize_by_scenario_seed(episodes),
    )
    _raise_if_cancelled(cancel_check)
    artifacts = write_evaluation_artifacts(suite, output_root)
    _raise_if_cancelled(cancel_check)
    registry.upsert_evaluation(
        suite.evaluation_id,
        matrix_id=matrix_id,
        split=protocol.evaluation.split,
        payload=suite.model_dump(mode="json"),
        artifact_dir=str(artifacts.run_dir),
    )
    return suite, audit_primary_acceptance(suite, protocol)


def _raise_if_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise RuntimeError("matrix evaluation cancelled")


def audit_primary_acceptance(
    suite: EvaluationSuite,
    protocol: ExperimentProtocol,
) -> PrimaryAcceptanceAudit:
    lookup = {
        (item.controller, item.experiment_variant, item.failure_enabled): item
        for item in suite.summaries
    }
    mappo_nominal = _required_summary(lookup, "mappo", "mappo_full", False)
    ippo_nominal = _required_summary(lookup, "ippo", "ippo_full", False)
    mappo_failure = _required_summary(lookup, "mappo", "mappo_full", True)
    cp_sat_nominal = _required_summary(lookup, "cp_sat", None, False)
    makespan_ratio = (
        mappo_nominal.metrics["makespan_s"].median
        / max(cp_sat_nominal.metrics["makespan_s"].median, 1e-9)
    )
    thresholds = protocol.acceptance
    results = [
        AcceptanceResult(
            name="mappo_no_failure_mean_completion",
            passed=(
                mappo_nominal.metrics["structure_completion_rate"].mean
                >= thresholds.mappo_no_failure_mean_completion_min
            ),
            observed=mappo_nominal.metrics["structure_completion_rate"].mean,
            threshold=thresholds.mappo_no_failure_mean_completion_min,
            comparison="min",
        ),
        AcceptanceResult(
            name="ippo_no_failure_mean_completion",
            passed=(
                ippo_nominal.metrics["structure_completion_rate"].mean
                >= thresholds.ippo_no_failure_mean_completion_min
            ),
            observed=ippo_nominal.metrics["structure_completion_rate"].mean,
            threshold=thresholds.ippo_no_failure_mean_completion_min,
            comparison="min",
        ),
        AcceptanceResult(
            name="mappo_median_makespan_cp_sat_ratio",
            passed=(
                makespan_ratio
                <= thresholds.mappo_median_makespan_cp_sat_ratio_max
            ),
            observed=makespan_ratio,
            threshold=thresholds.mappo_median_makespan_cp_sat_ratio_max,
            comparison="max",
        ),
        AcceptanceResult(
            name="mappo_failure_mean_completion",
            passed=(
                mappo_failure.metrics["structure_completion_rate"].mean
                >= thresholds.mappo_failure_mean_completion_min
            ),
            observed=mappo_failure.metrics["structure_completion_rate"].mean,
            threshold=thresholds.mappo_failure_mean_completion_min,
            comparison="min",
        ),
    ]
    return PrimaryAcceptanceAudit(
        passed=all(result.passed for result in results),
        results=results,
    )


def audit_ablation_hypotheses(
    suite: EvaluationSuite,
    protocol: ExperimentProtocol,
    *,
    selection_evidence: MatrixSelectionEvidence | None = None,
) -> list[AblationDecision]:
    lookup = {
        (item.controller, item.experiment_variant, item.failure_enabled): item
        for item in suite.summaries
    }
    mappo_nominal = _required_summary(lookup, "mappo", "mappo_full", False)
    mappo_failure = _required_summary(lookup, "mappo", "mappo_full", True)
    no_bc_nominal = _required_summary(lookup, "mappo", "mappo_no_bc", False)
    no_curriculum_nominal = _required_summary(
        lookup,
        "mappo",
        "mappo_no_failure_curriculum",
        False,
    )
    no_curriculum_failure = _required_summary(
        lookup,
        "mappo",
        "mappo_no_failure_curriculum",
        True,
    )
    full_transitions_to_95 = None
    no_bc_transitions_to_95 = None
    if selection_evidence is not None:
        full_transitions_to_95 = validation_transitions_to_completion(
            selection_evidence,
            protocol,
            experiment_variant="mappo_full",
        )
        no_bc_transitions_to_95 = validation_transitions_to_completion(
            selection_evidence,
            protocol,
            experiment_variant="mappo_no_bc",
        )
    return [
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="behavior_cloning",
                full_final_completion=mappo_nominal.metrics[
                    "structure_completion_rate"
                ].mean,
                ablated_final_completion=no_bc_nominal.metrics[
                    "structure_completion_rate"
                ].mean,
                full_transitions_to_95=full_transitions_to_95,
                ablated_transitions_to_95=no_bc_transitions_to_95,
            ),
        ),
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="failure_curriculum",
                full_failure_completion=mappo_failure.metrics[
                    "structure_completion_rate"
                ].mean,
                ablated_failure_completion=no_curriculum_failure.metrics[
                    "structure_completion_rate"
                ].mean,
                full_no_failure_completion=mappo_nominal.metrics[
                    "structure_completion_rate"
                ].mean,
                ablated_no_failure_completion=no_curriculum_nominal.metrics[
                    "structure_completion_rate"
                ].mean,
            ),
        ),
    ]


def validation_transitions_to_completion(
    evidence: MatrixSelectionEvidence,
    protocol: ExperimentProtocol,
    *,
    experiment_variant: str,
    threshold: float = 0.95,
) -> int | None:
    """Return the first frozen checkpoint where mean nominal validation completion passes."""

    if not 0 < threshold <= 1:
        raise ValueError("completion threshold must be in (0, 1]")
    if evidence.protocol_digest != protocol_digest(protocol):
        raise ValueError("selection evidence protocol digest does not match the protocol")
    variant_runs = [
        run
        for run in evidence.selections
        if run.selected.experiment_variant == experiment_variant
    ]
    observed_seeds = {run.selected.training_seed for run in variant_runs}
    if observed_seeds != set(protocol.training_seeds):
        raise ValueError(
            f"selection evidence for {experiment_variant} must cover training seeds "
            f"{protocol.training_seeds}"
        )
    for run in variant_runs:
        if (
            run.matrix_id != evidence.matrix_id
            or run.selected.experiment_id != protocol.experiment_id
        ):
            raise ValueError("selection evidence identity does not match the protocol")
        for candidate in run.candidates:
            if (
                candidate.result.experiment_id != protocol.experiment_id
                or candidate.result.experiment_variant != experiment_variant
                or candidate.result.training_seed != run.selected.training_seed
                or candidate.result.split != protocol.selection.split
                or candidate.result.scenario_seeds
                != protocol.selection.scenario_seeds
            ):
                raise ValueError(
                    "candidate evidence identity or validation grid is inconsistent"
                )
            if any(
                episode.experiment_id != protocol.experiment_id
                or episode.experiment_variant != experiment_variant
                or episode.training_seed != run.selected.training_seed
                for episode in candidate.episodes
            ):
                raise ValueError(
                    "candidate episodes do not match their training-run identity"
                )
    for fraction in protocol.checkpoint_fractions:
        candidates = [
            candidate
            for run in variant_runs
            for candidate in run.candidates
            if np.isclose(
                candidate.result.checkpoint_fraction,
                fraction,
                rtol=0,
                atol=1e-12,
            )
        ]
        if len(candidates) != len(protocol.training_seeds):
            raise ValueError(
                f"selection evidence for {experiment_variant} must contain one "
                f"{fraction:.0%} checkpoint per training seed"
            )
        transition_counts = {
            candidate.result.transition_count for candidate in candidates
        }
        if len(transition_counts) != 1:
            raise ValueError(
                f"selection evidence for {experiment_variant} has inconsistent "
                f"transition counts at checkpoint fraction {fraction}"
            )
        nominal_episodes = [
            episode
            for candidate in candidates
            for episode in candidate.episodes
            if not episode.failure_enabled
        ]
        expected_episode_count = (
            len(protocol.training_seeds)
            * len(protocol.selection.scenario_seeds)
        )
        if len(nominal_episodes) != expected_episode_count:
            raise ValueError(
                f"selection evidence for {experiment_variant} must contain exactly "
                f"{expected_episode_count} nominal validation episodes at each checkpoint"
            )
        if any(
            episode.split != protocol.selection.split
            or episode.seed not in protocol.selection.scenario_seeds
            or episode.experiment_id != protocol.experiment_id
            or episode.experiment_variant != experiment_variant
            or episode.training_seed not in protocol.training_seeds
            for episode in nominal_episodes
        ):
            raise ValueError("transitions-to-completion evidence must use validation seeds")
        mean_completion = float(
            np.mean(
                [
                    episode.structure_completion_rate
                    for episode in nominal_episodes
                ]
            )
        )
        if mean_completion > threshold or abs(mean_completion - threshold) <= 1e-12:
            return next(iter(transition_counts))
    return None


def _candidate_policy_manifest(
    *,
    checkpoint_id: str,
    checkpoint_path: Path,
    checkpoint_sha: str,
    transition_count: int,
    config: TrainingConfig,
    lineage: list[str],
    resume_provenance: dict[str, object],
) -> PolicyManifest:
    assert config.configuration_digest is not None
    assert config.source_commit is not None
    return PolicyManifest(
        policy_id=checkpoint_id,
        controller=config.algorithm,
        git_sha=config.source_commit,
        seed=config.seed,
        experiment_id=config.experiment_id,
        experiment_variant=config.experiment_variant,
        training_seed=config.training_seed,
        transition_count=transition_count,
        checkpoint_path=str(checkpoint_path.resolve()),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_lineage=lineage,
        configuration_digest=config.configuration_digest,
        source_commit=config.source_commit,
        source_dirty=config.source_dirty,
        source_tree_digest=config.source_tree_digest,
        resume_provenance=resume_provenance,
        environment_fingerprint=dict(config.environment_fingerprint),
        config=config.model_dump(mode="json"),
    )


def _selected_policy_manifest(
    matrix_id: str,
    run_key: str,
    selection: SelectedCheckpoint,
    config: TrainingConfig,
) -> PolicyManifest:
    return PolicyManifest(
        policy_id=f"{matrix_id}-{run_key}-selected",
        controller=config.algorithm,
        git_sha=selection.source_commit,
        seed=config.seed,
        experiment_id=selection.experiment_id,
        experiment_variant=selection.experiment_variant,
        training_seed=selection.training_seed,
        transition_count=selection.transition_count,
        checkpoint_path=selection.checkpoint_path,
        checkpoint_sha256=selection.checkpoint_sha256,
        checkpoint_lineage=list(selection.checkpoint_lineage),
        configuration_digest=selection.configuration_digest,
        source_commit=selection.source_commit,
        source_dirty=config.source_dirty,
        source_tree_digest=config.source_tree_digest,
        resume_provenance=dict(selection.resume_provenance),
        environment_fingerprint=dict(config.environment_fingerprint),
        config=config.model_dump(mode="json"),
    )


def _policy_checkpoint_metadata(path: Path) -> dict[str, object]:
    return load_policy_checkpoint_metadata(path, device="cpu")


def _validate_checkpoint_provenance(
    metadata: dict[str, object],
    *,
    config: TrainingConfig,
    design: HouseDesign,
    checkpoint_path: Path,
    resume_history: list[dict[str, object]],
    expected_resume_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    expected: dict[str, object] = {
        "experiment_id": config.experiment_id,
        "experiment_variant": config.experiment_variant,
        "training_seed": config.training_seed,
        "configuration_digest": config.configuration_digest,
        "source_commit": config.source_commit,
        "source_dirty": config.source_dirty,
        "source_tree_digest": config.source_tree_digest,
        "design_digest": design_digest(design),
        "environment_fingerprint": config.environment_fingerprint,
    }
    mismatches = [
        f"{name}: checkpoint={metadata.get(name)!r}, expected={value!r}"
        for name, value in expected.items()
        if metadata.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint provenance mismatch for {checkpoint_path} ("
            + "; ".join(mismatches)
            + ")"
        )
    raw_resume_provenance = metadata.get("resume_provenance", {})
    if not isinstance(raw_resume_provenance, dict):
        raise ValueError(
            f"checkpoint resume provenance must be an object for {checkpoint_path}"
        )
    resume_provenance = dict(raw_resume_provenance)
    allowed_resume_provenance = [{}, *resume_history]
    if resume_provenance not in allowed_resume_provenance:
        raise ValueError(
            f"checkpoint resume provenance is not recorded for {checkpoint_path}"
        )
    if (
        expected_resume_provenance is not None
        and resume_provenance != expected_resume_provenance
    ):
        raise ValueError(
            f"checkpoint resume provenance does not match the frozen selection "
            f"for {checkpoint_path}"
        )
    return resume_provenance


def _run_resume_provenance_history(
    run: dict[str, object],
    config: TrainingConfig,
) -> list[dict[str, object]]:
    value = run.get("resume_provenance_history")
    if value is None:
        return [dict(config.resume_provenance)] if config.resume_provenance else []
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("run resume provenance history must be a list of objects")
    return [dict(item) for item in value]


def _require_matrix_design(
    runs: list[dict[str, object]],
    design: HouseDesign,
) -> None:
    if not runs:
        raise ValueError("matrix has no persisted runs")
    expected_digest = design_digest(design)
    persisted_digests = set()
    for run in runs:
        input_payload = _required_dict(run, "input")
        persisted_design = HouseDesign.model_validate(
            _required_dict(input_payload, "design")
        )
        persisted_digests.add(design_digest(persisted_design))
    if len(persisted_digests) != 1:
        raise ValueError("matrix runs do not share one persisted design")
    if persisted_digests != {expected_digest}:
        raise ValueError(
            "evaluation design does not match the design persisted with the matrix"
        )


def _require_canonical_matrix(
    matrix: dict[str, object],
    runs: list[dict[str, object]],
    protocol: ExperimentProtocol,
) -> None:
    profile_value = matrix.get("execution_profile")
    if profile_value not in {"unit", "smoke", "research"}:
        raise ValueError(
            f"matrix execution profile is invalid: {profile_value!r}"
        )
    profile = cast(ExperimentProfile, profile_value)
    persisted_protocol = ExperimentProtocol.model_validate(
        _required_dict(matrix, "protocol")
    )
    matrix_digest = _required_string(matrix, "protocol_digest")
    if (
        protocol_digest(persisted_protocol) != matrix_digest
        or persisted_protocol != protocol
    ):
        raise ValueError(
            "matrix protocol payload does not match its persisted digest"
        )

    expected_specs = {
        spec.run_id: spec
        for spec in expand_experiment_matrix(protocol, profile)
    }
    run_keys = [_required_string(run, "run_key") for run in runs]
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("matrix contains duplicate run keys")
    actual_keys = set(run_keys)
    expected_keys = set(expected_specs)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "matrix run grid does not match the frozen protocol "
            f"(missing={missing}, unexpected={unexpected})"
        )

    for run in runs:
        run_key = _required_string(run, "run_key")
        spec = expected_specs[run_key]
        actual_config = TrainingConfig.model_validate(
            _required_dict(run, "config")
        )
        expected_config = TrainingConfig.model_validate(
            spec.training_config_payload()
        )
        actual_frozen = actual_config.model_dump(
            mode="json",
            exclude=_MATRIX_RUNTIME_CONFIG_FIELDS,
        )
        expected_frozen = expected_config.model_dump(
            mode="json",
            exclude=_MATRIX_RUNTIME_CONFIG_FIELDS,
        )
        changed_fields = sorted(
            field
            for field in set(actual_frozen) | set(expected_frozen)
            if actual_frozen.get(field) != expected_frozen.get(field)
        )
        if changed_fields:
            raise ValueError(
                f"matrix run {run_key} has mutated frozen configuration fields: "
                + ", ".join(changed_fields)
            )

        persisted_config_digest = _required_string(run, "config_digest")
        if actual_config.configuration_digest != persisted_config_digest:
            raise ValueError(
                f"matrix run {run_key} configuration digest column does not "
                "match its config payload"
            )
        recomputed_digest = configuration_digest(actual_config)
        if recomputed_digest != persisted_config_digest:
            raise ValueError(
                f"matrix run {run_key} configuration digest is invalid: "
                f"{persisted_config_digest} != {recomputed_digest}"
            )
        persisted_source_commit = _required_string(run, "source_commit")
        if actual_config.source_commit != persisted_source_commit:
            raise ValueError(
                f"matrix run {run_key} source commit column does not match "
                "its config payload"
            )


def _matrix_runs(matrix: dict[str, object]) -> list[dict[str, object]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list) or not all(isinstance(item, dict) for item in runs):
        raise ValueError("matrix runs are malformed")
    return cast(list[dict[str, object]], runs)


def _required_dict(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return cast(dict[str, object], value)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list of strings")
    return cast(list[str], value)


def _required_summary(
    lookup: dict[
        tuple[ControllerName, str | None, bool],
        ControllerEvaluation,
    ],
    controller: ControllerName,
    variant: str | None,
    failure_enabled: bool,
) -> ControllerEvaluation:
    summary = lookup.get((controller, variant, failure_enabled))
    if summary is None:
        raise ValueError(
            "missing acceptance summary for "
            f"{controller}/{variant or 'baseline'}/failures={failure_enabled}"
        )
    return summary


def _protocol_digest_from_matrix(
    protocol: ExperimentProtocol,
    matrix: dict[str, object],
) -> str:
    from embodied_skill_composer.construction.experiment_protocol import (
        protocol_digest,
    )

    digest = protocol_digest(protocol)
    matrix_digest = str(matrix["protocol_digest"])
    return digest if digest == matrix_digest else ""


def write_acceptance_artifact(
    audit: PrimaryAcceptanceAudit,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path
