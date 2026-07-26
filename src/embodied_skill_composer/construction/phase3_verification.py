from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.evaluation import EvaluationSuite
from embodied_skill_composer.construction.experiment_execution import (
    MatrixSelectionEvidence,
    evaluate_and_freeze_matrix_selections,
)
from embodied_skill_composer.construction.experiment_protocol import (
    DEFAULT_EXPERIMENT_PROTOCOL_PATH,
    ExperimentProtocol,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.lab_registry import (
    QUIESCENT_RUN_STATUSES,
    LabRegistry,
)
from embodied_skill_composer.construction.lab_service import LabService
from embodied_skill_composer.construction.policy import file_sha256
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import (
    TrainingConfig,
    source_fingerprint,
)


WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_DESIGN_PATH = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE / "logs" / "construction_intelligence" / "phase3_e2e"
)
ProgressCallback = Callable[[dict[str, object]], None]


class VerifiedArtifact(BaseModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=1)


class InterruptionEvidence(BaseModel):
    run_id: str
    run_key: str
    pid: int = Field(gt=0)
    process_identity: str
    checkpoint_path: Path
    checkpoint_transitions: int = Field(gt=0)
    progress: float = Field(gt=0, le=1)
    completed_attempt: int = Field(ge=2)
    resumed_transitions: int = Field(gt=0)


class Phase3VerificationResult(BaseModel):
    schema_version: Literal["construction-intelligence-phase3-e2e-v1"] = (
        "construction-intelligence-phase3-e2e-v1"
    )
    matrix_id: str
    protocol_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_profile: Literal["unit"] = "unit"
    source_commit: str
    source_tree_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    registry_path: Path
    output_root: Path
    training_run_count: Literal[20] = 20
    interruption: InterruptionEvidence
    fractional_policy_export_count: Literal[100] = 100
    selection_count: Literal[20] = 20
    validation_candidate_count: Literal[100] = 100
    validation_episode_count: Literal[1000] = 1000
    validation_seeds: list[int]
    heldout_evaluation_run_id: str
    heldout_evaluation_id: str
    heldout_episode_count: Literal[240] = 240
    learned_episode_count: Literal[200] = 200
    baseline_episode_count: Literal[40] = 40
    heldout_seeds: list[int]
    primary_acceptance_passed: bool
    artifacts: dict[str, VerifiedArtifact]
    verification_path: Path


def run_phase3_end_to_end_verification(
    output_root: Path,
    *,
    timeout_seconds: float = 3600.0,
    poll_seconds: float = 0.05,
    design_path: Path = DEFAULT_DESIGN_PATH,
    protocol_path: Path = DEFAULT_EXPERIMENT_PROTOCOL_PATH,
    progress_callback: ProgressCallback | None = None,
) -> Phase3VerificationResult:
    """Run the canonical unit matrix through real durable workers and evidence paths."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    poll_seconds = max(poll_seconds, 0.01)
    root = output_root.resolve()
    _require_safe_empty_output_root(root)
    root.mkdir(parents=True, exist_ok=False)
    pinned_source = _require_clean_source()
    deadline = time.monotonic() + timeout_seconds
    protocol = load_experiment_protocol(protocol_path)
    design = load_house_design(design_path)
    digest = protocol_digest(protocol)
    matrix_id = (
        f"{protocol.experiment_id}-unit-e2e-{digest[:12]}-"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    )
    registry = LabRegistry(root / "lab.sqlite")
    training_root = root / "training"
    validation_root = root / "validation"
    heldout_root = root / "heldout"
    launch_runs: list[tuple[str, TrainingConfig]] = []
    for spec in expand_experiment_matrix(protocol, "unit"):
        config = TrainingConfig.model_validate(spec.training_config_payload())
        config.output_root = training_root
        launch_runs.append((spec.run_id, config))

    service = LabService(registry)
    pipeline_completed = False
    try:
        run_ids = service.launch_training_matrix(
            design,
            matrix_id=matrix_id,
            protocol_digest=digest,
            protocol=protocol.model_dump(mode="json"),
            execution_profile="unit",
            runs=launch_runs,
        )
        if len(run_ids) != 20 or len(set(run_ids)) != 20:
            raise AssertionError("canonical unit matrix did not enqueue exactly 20 runs")
        _emit(
            progress_callback,
            {
                "stage": "training",
                "matrix_id": matrix_id,
                "status": "queued",
                "run_count": len(run_ids),
            },
        )
        service, interruption = _interrupt_and_resume_one_run(
            registry,
            service,
            matrix_id,
            pinned_source=pinned_source,
            deadline=deadline,
            poll_seconds=poll_seconds,
            progress_callback=progress_callback,
        )
        _wait_for_training_matrix(
            registry,
            service,
            matrix_id,
            pinned_source=pinned_source,
            deadline=deadline,
            poll_seconds=poll_seconds,
            progress_callback=progress_callback,
        )
        interruption = _complete_interruption_evidence(
            registry,
            interruption,
        )
        _require_source_unchanged(pinned_source)

        selection_evidence = evaluate_and_freeze_matrix_selections(
            registry,
            matrix_id,
            design,
            protocol,
            output_root=validation_root,
            device="cpu",
        )
        selection_counts, fractional_exports = _verify_validation_evidence(
            selection_evidence,
            protocol,
        )
        _emit(
            progress_callback,
            {
                "stage": "validation",
                "matrix_id": matrix_id,
                "status": "completed",
                **selection_counts,
            },
        )
        _require_deadline(deadline, "held-out evaluation launch")
        _require_source_unchanged(pinned_source)

        evaluation_run_id = service.launch_matrix_evaluation(
            design,
            matrix_id=matrix_id,
            protocol=protocol,
            output_root=heldout_root,
            device="cpu",
            selection_evidence_path=selection_evidence.evidence_path,
        )
        evaluation_run = _wait_for_run(
            registry,
            evaluation_run_id,
            pinned_source=pinned_source,
            deadline=deadline,
            poll_seconds=poll_seconds,
            progress_callback=progress_callback,
        )
        suite, primary_acceptance_passed, heldout_artifacts = (
            _verify_heldout_evidence(
                registry,
                matrix_id,
                evaluation_run_id,
                evaluation_run,
                protocol,
            )
        )
        _require_source_unchanged(pinned_source)
        artifacts = {
            "matrix_selection": _artifact(selection_evidence.evidence_path),
            **{
                f"heldout_{name}": _artifact(path)
                for name, path in heldout_artifacts.items()
            },
        }
        if len(fractional_exports) != 100:
            raise AssertionError(
                f"expected 100 unique fractional exports, got {len(fractional_exports)}"
            )
        verification_path = root / "verification.json"
        result = Phase3VerificationResult(
            matrix_id=matrix_id,
            protocol_digest=digest,
            source_commit=str(pinned_source["commit"]),
            source_tree_digest=str(pinned_source["tree_digest"]),
            registry_path=registry.path,
            output_root=root,
            interruption=interruption,
            validation_seeds=list(protocol.selection.scenario_seeds),
            heldout_evaluation_run_id=evaluation_run_id,
            heldout_evaluation_id=suite.evaluation_id,
            heldout_seeds=list(protocol.evaluation.scenario_seeds),
            primary_acceptance_passed=primary_acceptance_passed,
            artifacts=artifacts,
            verification_path=verification_path,
        )
        verification_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        pipeline_completed = True
        _emit(
            progress_callback,
            {
                "stage": "complete",
                "matrix_id": matrix_id,
                "status": "verified",
                "verification_path": str(verification_path),
            },
        )
        return result
    finally:
        if not pipeline_completed:
            _cancel_incomplete_runs(registry, service, matrix_id)
        service.shutdown()


def _interrupt_and_resume_one_run(
    registry: LabRegistry,
    service: LabService,
    matrix_id: str,
    *,
    pinned_source: Mapping[str, object],
    deadline: float,
    poll_seconds: float,
    progress_callback: ProgressCallback | None,
) -> tuple[LabService, InterruptionEvidence]:
    active = _wait_for_active_training_worker(
        registry,
        matrix_id,
        pinned_source=pinned_source,
        deadline=deadline,
        poll_seconds=poll_seconds,
    )
    run_id = _string_field(active, "id")
    run_key = _string_field(active, "run_key")
    pid = _integer_field(active, "pid")
    process_identity = _string_field(active, "process_identity")

    # Stop only the dispatcher. Its owned worker intentionally remains alive so
    # restart reconciliation can fence and terminate that exact process.
    service.shutdown()
    checkpoint_state = _wait_for_nonzero_checkpoint(
        registry,
        run_id,
        deadline=deadline,
        poll_seconds=poll_seconds,
    )
    checkpoint = Path(_string_field(checkpoint_state, "latest_checkpoint"))
    checkpoint_transitions = _latest_checkpoint_transitions(registry, run_id)
    progress = _float_field(checkpoint_state, "progress")
    interrupted = registry.reconcile_stale_runs(
        kinds=("training",),
        stale_after=timedelta(0),
        now=datetime.now(UTC) + timedelta(seconds=1),
    )
    if interrupted != [run_id]:
        current = registry.get_run(run_id)
        raise AssertionError(
            "restart reconciliation did not interrupt only the owned worker "
            f"{run_id}: interrupted={interrupted!r}, current={current!r}"
        )
    reconciled = registry.get_run(run_id)
    if reconciled is None or str(reconciled["status"]) != "interrupted":
        raise AssertionError(f"run was not persisted as interrupted: {run_id}")
    if not service.resume(run_id):
        raise AssertionError(f"interrupted run was not accepted for resume: {run_id}")
    _emit(
        progress_callback,
        {
            "stage": "recovery",
            "matrix_id": matrix_id,
            "status": "resuming",
            "run_id": run_id,
            "run_key": run_key,
            "checkpoint": str(checkpoint),
            "checkpoint_transitions": checkpoint_transitions,
        },
    )
    restarted_service = LabService(registry)
    return restarted_service, InterruptionEvidence(
        run_id=run_id,
        run_key=run_key,
        pid=pid,
        process_identity=process_identity,
        checkpoint_path=checkpoint,
        checkpoint_transitions=checkpoint_transitions,
        progress=progress,
        completed_attempt=2,
        resumed_transitions=checkpoint_transitions,
    )


def _wait_for_active_training_worker(
    registry: LabRegistry,
    matrix_id: str,
    *,
    pinned_source: Mapping[str, object],
    deadline: float,
    poll_seconds: float,
) -> dict[str, object]:
    next_source_check = 0.0
    while True:
        _require_deadline(deadline, "durable worker start")
        matrix = _required_matrix(registry, matrix_id)
        for run in _matrix_runs(matrix):
            if (
                str(run["status"]) == "running"
                and isinstance(run.get("pid"), int)
                and run.get("process_identity")
            ):
                return run
        if time.monotonic() >= next_source_check:
            _require_source_unchanged(pinned_source)
            next_source_check = time.monotonic() + 2.0
        time.sleep(poll_seconds)


def _wait_for_nonzero_checkpoint(
    registry: LabRegistry,
    run_id: str,
    *,
    deadline: float,
    poll_seconds: float,
) -> dict[str, object]:
    while True:
        _require_deadline(deadline, "nonzero durable checkpoint")
        run = registry.get_run(run_id)
        if run is None:
            raise AssertionError(f"active run disappeared: {run_id}")
        status = str(run["status"])
        checkpoint_value = run.get("latest_checkpoint")
        progress = _float_field(run, "progress")
        checkpoint_transitions = _positive_checkpoint_transitions(
            registry,
            run_id,
        )
        if (
            status == "running"
            and checkpoint_value
            and Path(str(checkpoint_value)).is_file()
            and progress > 0
            and checkpoint_transitions is not None
        ):
            return run
        if status in QUIESCENT_RUN_STATUSES:
            raise AssertionError(
                "unit worker completed before a nonzero checkpoint could be interrupted: "
                f"{run_id} ({status})"
            )
        time.sleep(poll_seconds)


def _wait_for_training_matrix(
    registry: LabRegistry,
    service: LabService,
    matrix_id: str,
    *,
    pinned_source: Mapping[str, object],
    deadline: float,
    poll_seconds: float,
    progress_callback: ProgressCallback | None,
) -> None:
    previous_counts: dict[str, int] | None = None
    next_source_check = 0.0
    while True:
        _require_deadline(deadline, "canonical unit training matrix")
        matrix = _required_matrix(registry, matrix_id)
        runs = _matrix_runs(matrix)
        counts = _status_counts(matrix)
        if counts != previous_counts:
            _emit(
                progress_callback,
                {
                    "stage": "training",
                    "matrix_id": matrix_id,
                    "status": str(matrix["status"]),
                    "status_counts": counts,
                },
            )
            previous_counts = counts
        if len(runs) == 20 and all(str(run["status"]) == "completed" for run in runs):
            return
        for run in runs:
            status = str(run["status"])
            if status == "interrupted" and run.get("latest_checkpoint"):
                if _integer_field(run, "attempt") >= 3 or not service.resume(
                    _string_field(run, "id")
                ):
                    raise AssertionError(
                        f"interrupted run could not resume: {run['run_key']}"
                    )
            elif status in {"failed", "cancelled"}:
                raise AssertionError(
                    f"matrix run {run['run_key']} ended as {status}: {run['error']}"
                )
        if time.monotonic() >= next_source_check:
            _require_source_unchanged(pinned_source)
            next_source_check = time.monotonic() + 2.0
        time.sleep(poll_seconds)


def _wait_for_run(
    registry: LabRegistry,
    run_id: str,
    *,
    pinned_source: Mapping[str, object],
    deadline: float,
    poll_seconds: float,
    progress_callback: ProgressCallback | None,
) -> dict[str, object]:
    previous_status: str | None = None
    next_source_check = 0.0
    while True:
        _require_deadline(deadline, f"run {run_id}")
        run = registry.get_run(run_id)
        if run is None:
            raise AssertionError(f"run disappeared: {run_id}")
        status = str(run["status"])
        if status != previous_status:
            _emit(
                progress_callback,
                {
                    "stage": "heldout",
                    "run_id": run_id,
                    "status": status,
                },
            )
            previous_status = status
        if status == "completed":
            return run
        if status in {"failed", "cancelled", "interrupted"}:
            raise AssertionError(
                f"held-out evaluation ended as {status}: {run.get('error')}"
            )
        if time.monotonic() >= next_source_check:
            _require_source_unchanged(pinned_source)
            next_source_check = time.monotonic() + 2.0
        time.sleep(poll_seconds)


def _complete_interruption_evidence(
    registry: LabRegistry,
    evidence: InterruptionEvidence,
) -> InterruptionEvidence:
    run = registry.get_run(evidence.run_id)
    if run is None or str(run["status"]) != "completed":
        raise AssertionError("interrupted run did not complete after resume")
    events = _event_payloads(registry, evidence.run_id)
    event_names = [str(event.get("event")) for event in events]
    for required in (
        "run_interrupted",
        "resume_requested",
        "training_resumed",
        "training_completed",
    ):
        if required not in event_names:
            raise AssertionError(f"recovery event is missing: {required}")
    attempts = {
        _integer_field(event, "attempt")
        for event in events
        if event.get("event") == "training_started"
    }
    if not {1, 2}.issubset(attempts):
        raise AssertionError(f"expected subprocess training attempts 1 and 2, got {attempts}")
    resumed = next(
        event for event in events if event.get("event") == "training_resumed"
    )
    resumed_transitions = _integer_field(resumed, "transitions")
    if resumed_transitions < evidence.checkpoint_transitions:
        raise AssertionError(
            "resumed checkpoint lost transitions: "
            f"{resumed_transitions} < {evidence.checkpoint_transitions}"
        )
    provenance = run.get("resume_provenance_history")
    if not isinstance(provenance, list) or not any(
        isinstance(item, dict)
        and item.get("attempt") == 2
        and item.get("checkpoint") == str(evidence.checkpoint_path)
        for item in provenance
    ):
        raise AssertionError("attempt-2 resume provenance was not persisted")
    return evidence.model_copy(
        update={
            "completed_attempt": _integer_field(run, "attempt"),
            "resumed_transitions": resumed_transitions,
        }
    )


def _verify_validation_evidence(
    evidence: MatrixSelectionEvidence,
    protocol: ExperimentProtocol,
) -> tuple[dict[str, int], set[Path]]:
    if len(evidence.selections) != 20:
        raise AssertionError(
            f"expected 20 frozen selections, got {len(evidence.selections)}"
        )
    expected_fractions = list(protocol.checkpoint_fractions)
    expected_seeds = set(protocol.selection.scenario_seeds)
    exports: set[Path] = set()
    candidate_count = 0
    episode_count = 0
    for run_selection in evidence.selections:
        if len(run_selection.candidates) != 5:
            raise AssertionError(
                f"selection {run_selection.run_key} did not evaluate five checkpoints"
            )
        fractions = [
            candidate.result.checkpoint_fraction
            for candidate in run_selection.candidates
        ]
        if fractions != expected_fractions:
            raise AssertionError(
                f"selection {run_selection.run_key} used fractions {fractions!r}"
            )
        if run_selection.selected.required_checkpoint_fractions != expected_fractions:
            raise AssertionError(
                f"selection {run_selection.run_key} did not freeze the five-fraction rule"
            )
        for candidate in run_selection.candidates:
            candidate_count += 1
            checkpoint = Path(candidate.result.checkpoint_path).resolve()
            exports.add(checkpoint)
            if not checkpoint.is_file():
                raise AssertionError(f"fractional policy export is missing: {checkpoint}")
            if file_sha256(checkpoint) != candidate.result.checkpoint_sha256:
                raise AssertionError(f"fractional policy export hash changed: {checkpoint}")
            if candidate.result.split != "validation":
                raise AssertionError("checkpoint selection consumed a non-validation result")
            if set(candidate.result.scenario_seeds) != expected_seeds:
                raise AssertionError("checkpoint selection used the wrong validation seeds")
            if len(candidate.episodes) != 10:
                raise AssertionError("a validation candidate did not execute ten real episodes")
            cells = {
                (episode.seed, episode.failure_enabled)
                for episode in candidate.episodes
            }
            expected_cells = {
                (seed, failure)
                for seed in protocol.selection.scenario_seeds
                for failure in protocol.selection.failure_modes
            }
            if cells != expected_cells or any(
                episode.split != "validation" for episode in candidate.episodes
            ):
                raise AssertionError("validation candidate grid or split is incomplete")
            episode_count += len(candidate.episodes)
    if candidate_count != 100 or episode_count != 1000:
        raise AssertionError(
            "canonical validation evidence has the wrong size: "
            f"candidates={candidate_count}, episodes={episode_count}"
        )
    return {
        "selection_count": len(evidence.selections),
        "candidate_count": candidate_count,
        "episode_count": episode_count,
    }, exports


def _verify_heldout_evidence(
    registry: LabRegistry,
    matrix_id: str,
    evaluation_run_id: str,
    evaluation_run: dict[str, object],
    protocol: ExperimentProtocol,
) -> tuple[EvaluationSuite, bool, dict[str, Path]]:
    artifact_dir_value = evaluation_run.get("artifact_dir")
    if not isinstance(artifact_dir_value, str) or not artifact_dir_value:
        raise AssertionError("held-out evaluation did not persist an artifact directory")
    artifact_dir = Path(artifact_dir_value).resolve()
    artifact_paths = {
        "evaluation": artifact_dir / "evaluation.json",
        "episodes": artifact_dir / "episodes.csv",
        "report": artifact_dir / "report.md",
    }
    if any(not path.is_file() for path in artifact_paths.values()):
        raise AssertionError(f"held-out artifact set is incomplete: {artifact_paths!r}")
    suite = EvaluationSuite.model_validate_json(
        artifact_paths["evaluation"].read_text(encoding="utf-8")
    )
    if suite.expected_split != "test":
        raise AssertionError("held-out evaluation did not retain the test split")
    if suite.seeds != protocol.evaluation.scenario_seeds:
        raise AssertionError("held-out evaluation used the wrong scenario seeds")
    if len(suite.episodes) != 240:
        raise AssertionError(f"expected 240 held-out episodes, got {len(suite.episodes)}")
    if suite.grid_validation is None or (
        suite.grid_validation.expected_episode_count != 240
        or suite.grid_validation.observed_episode_count != 240
        or suite.grid_validation.expected_split != "test"
    ):
        raise AssertionError("held-out evaluation grid is not complete")
    learned = [
        episode
        for episode in suite.episodes
        if episode.experiment_id == protocol.experiment_id
    ]
    baselines = [
        episode for episode in suite.episodes if episode.experiment_id is None
    ]
    if len(learned) != 200 or len(baselines) != 40:
        raise AssertionError(
            f"held-out learned/baseline grid is {len(learned)}/{len(baselines)}"
        )
    if any(
        episode.split != "test"
        or episode.seed not in protocol.evaluation.scenario_seeds
        for episode in suite.episodes
    ):
        raise AssertionError("held-out episodes crossed the frozen test split")
    if not suite.per_training_seed or not suite.per_scenario_seed:
        raise AssertionError("held-out hierarchical aggregation tables are empty")
    report = artifact_paths["report"].read_text(encoding="utf-8")
    if "hierarchical bootstrap" not in report or "240/240 expected episodes" not in report:
        raise AssertionError("held-out report omitted the aggregation or grid evidence")
    if len(artifact_paths["episodes"].read_text(encoding="utf-8").splitlines()) != 241:
        raise AssertionError("held-out episode CSV does not contain 240 data rows")
    evaluations = registry.list_evaluations(matrix_id=matrix_id)
    if len(evaluations) != 1 or evaluations[0]["id"] != suite.evaluation_id:
        raise AssertionError("held-out evaluation was not registered exactly once")
    completion_events = [
        event
        for event in _event_payloads(registry, evaluation_run_id)
        if event.get("event") == "matrix_evaluation_completed"
    ]
    if len(completion_events) != 1:
        raise AssertionError("matrix evaluation completion event is missing or duplicated")
    completion = completion_events[0]
    if completion.get("evaluation_id") != suite.evaluation_id:
        raise AssertionError("matrix evaluation event points at a different evaluation")
    acceptance = completion.get("acceptance")
    if not isinstance(acceptance, dict) or not isinstance(
        acceptance.get("passed"), bool
    ):
        raise AssertionError("matrix evaluation completion omitted the acceptance audit")
    ablations = completion.get("ablations")
    if not isinstance(ablations, list) or len(ablations) != 2:
        raise AssertionError("matrix evaluation completion omitted the two ablation audits")
    return suite, bool(acceptance["passed"]), artifact_paths


def _latest_checkpoint_transitions(registry: LabRegistry, run_id: str) -> int:
    latest = _positive_checkpoint_transitions(registry, run_id)
    if latest is None:
        raise AssertionError("interrupted run has no positive durable checkpoint event")
    return latest


def _positive_checkpoint_transitions(
    registry: LabRegistry,
    run_id: str,
) -> int | None:
    transitions = [
        _integer_field(event, "transitions")
        for event in _event_payloads(registry, run_id)
        if event.get("event") in {"checkpoint_saved", "behavior_cloning_checkpoint"}
    ]
    positive = [value for value in transitions if value > 0]
    return max(positive) if positive else None


def _event_payloads(
    registry: LabRegistry,
    run_id: str,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for envelope in registry.list_events(run_id):
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise AssertionError(f"malformed persisted event envelope for {run_id}")
        payloads.append(cast(dict[str, object], payload))
    return payloads


def _cancel_incomplete_runs(
    registry: LabRegistry,
    service: LabService,
    matrix_id: str,
) -> None:
    matrix = registry.get_experiment_matrix(matrix_id)
    run_ids: list[str] = []
    if matrix is not None:
        for run in _matrix_runs(matrix):
            if str(run["status"]) not in QUIESCENT_RUN_STATUSES:
                run_id = _string_field(run, "id")
                run_ids.append(run_id)
                service.cancel(run_id)
    for run in registry.list_runs(limit=200):
        if (
            str(run["kind"]) == "matrix_evaluation"
            and str(run["status"]) not in QUIESCENT_RUN_STATUSES
        ):
            run_id = _string_field(run, "id")
            run_ids.append(run_id)
            service.cancel(run_id)
    stop_deadline = time.monotonic() + 10.0
    while time.monotonic() < stop_deadline:
        all_quiescent = True
        for run_id in set(run_ids):
            current = registry.get_run(run_id)
            if (
                current is not None
                and str(current["status"]) not in QUIESCENT_RUN_STATUSES
            ):
                all_quiescent = False
                break
        if all_quiescent:
            return
        time.sleep(0.05)


def _require_clean_source() -> dict[str, object]:
    source = source_fingerprint()
    commit = source.get("commit")
    tree_digest = source.get("tree_digest")
    if (
        source.get("dirty") is not False
        or not isinstance(commit, str)
        or not commit
        or commit == "unknown"
        or not isinstance(tree_digest, str)
        or len(tree_digest) != 64
    ):
        raise RuntimeError(
            "Phase-3 end-to-end verification requires a clean, committed source "
            f"worktree; observed fingerprint={source!r}"
        )
    return source


def _require_source_unchanged(pinned: Mapping[str, object]) -> None:
    current = source_fingerprint()
    if current != dict(pinned):
        raise RuntimeError(
            "source worktree changed during Phase-3 verification: "
            f"pinned={dict(pinned)!r}, current={current!r}"
        )


def _require_safe_empty_output_root(root: Path) -> None:
    if root.exists():
        raise FileExistsError(f"verification output root already exists: {root}")
    if root.is_relative_to(WORKSPACE):
        allowed_roots = (
            (WORKSPACE / "logs").resolve(),
            (WORKSPACE / "artifacts").resolve(),
        )
        if not any(root.is_relative_to(allowed) for allowed in allowed_roots):
            raise ValueError(
                "workspace-local verification output must be under ignored logs/ "
                f"or artifacts/: {root}"
            )


def _required_matrix(
    registry: LabRegistry,
    matrix_id: str,
) -> dict[str, object]:
    matrix = registry.get_experiment_matrix(matrix_id)
    if matrix is None:
        raise AssertionError(f"matrix disappeared: {matrix_id}")
    return matrix


def _matrix_runs(matrix: dict[str, object]) -> list[dict[str, object]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        raise AssertionError("matrix runs are malformed")
    return cast(list[dict[str, object]], runs)


def _status_counts(matrix: dict[str, object]) -> dict[str, int]:
    value = matrix.get("status_counts")
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(count, int)
        for key, count in value.items()
    ):
        raise AssertionError("matrix status counts are malformed")
    return cast(dict[str, int], value)


def _artifact(path: Path) -> VerifiedArtifact:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AssertionError(f"verified artifact is missing: {resolved}")
    return VerifiedArtifact(
        path=resolved,
        sha256=_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(
    callback: ProgressCallback | None,
    payload: dict[str, object],
) -> None:
    if callback is not None:
        callback(payload)


def _require_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Phase-3 end-to-end verification timed out during {stage}")


def _string_field(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{name} must be a non-empty string")
    return value


def _integer_field(payload: Mapping[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int):
        raise AssertionError(f"{name} must be an integer")
    return value


def _float_field(payload: Mapping[str, object], name: str) -> float:
    value = payload.get(name)
    if not isinstance(value, (int, float)):
        raise AssertionError(f"{name} must be numeric")
    return float(value)


def result_as_json(result: Phase3VerificationResult) -> str:
    return json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True)
