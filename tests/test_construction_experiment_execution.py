from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
import torch

from embodied_skill_composer.construction import experiment_execution
from embodied_skill_composer.construction.evaluation import (
    ControllerEvaluation,
    ControllerName,
    EpisodeEvaluation,
    EvaluationSuite,
    MetricSummary,
)
from embodied_skill_composer.construction.experiment_execution import (
    CandidateEvaluation,
    MatrixSelectionEvidence,
    RunSelectionEvidence,
    audit_ablation_hypotheses,
    audit_primary_acceptance,
    evaluate_and_freeze_matrix_selections,
    evaluate_frozen_heldout_matrix,
    evaluate_run_checkpoint_candidates,
    validation_transitions_to_completion,
)
from embodied_skill_composer.construction.experiment_protocol import (
    CheckpointValidationResult,
    ExperimentProtocol,
    SelectedCheckpoint,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.policy import file_sha256
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.research_evidence import (
    audit_reproducibility_artifacts,
)
from embodied_skill_composer.construction.training import (
    TrainingConfig,
    configuration_digest,
    design_digest,
)


DIGEST = "a" * 64
SOURCE_COMMIT = "phase-3-source"


@pytest.fixture(scope="module")
def protocol() -> ExperimentProtocol:
    return load_experiment_protocol()


@pytest.fixture(scope="module")
def cottage_design() -> HouseDesign:
    workspace = Path(__file__).resolve().parents[1]
    return load_house_design(workspace / "configs" / "construction" / "cottage_v1.yaml")


def test_evaluate_run_checkpoint_candidates_uses_all_five_validation_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    run = _candidate_run(tmp_path)
    artifact_dir = Path(cast(str, run["artifact_dir"]))
    checkpoint_paths = _write_policy_checkpoints(artifact_dir)
    loaded_paths: list[Path] = []
    evaluation_cells: list[tuple[int, bool, str | None]] = []
    completion_by_transition = {
        6: 0.70,
        16: 0.80,
        32: 0.90,
        48: 0.99,
        64: 0.95,
    }

    def fake_load_policy_checkpoint(path: Path, *, device: str) -> object:
        assert device == "cpu"
        loaded_paths.append(path)
        return type("PolicyBundleStub", (), {"algorithm": "mappo"})()

    def fake_evaluate_controller_episode(
        _design: HouseDesign,
        *,
        seed: int,
        controller: ControllerName,
        policy_manifest: PolicyManifest | None = None,
        failure_enabled: bool = False,
        expected_split: str | None = None,
        **_kwargs: object,
    ) -> EpisodeEvaluation:
        assert controller == "mappo"
        assert policy_manifest is not None
        evaluation_cells.append((seed, failure_enabled, expected_split))
        return _episode(
            controller=controller,
            seed=seed,
            split=expected_split or "validation",
            failure_enabled=failure_enabled,
            completion=completion_by_transition[policy_manifest.transition_count or 0],
            makespan=150.0 - float(policy_manifest.transition_count or 0),
            manifest=policy_manifest,
        )

    monkeypatch.setattr(
        experiment_execution,
        "load_policy_checkpoint",
        fake_load_policy_checkpoint,
    )
    monkeypatch.setattr(
        experiment_execution,
        "evaluate_controller_episode",
        fake_evaluate_controller_episode,
    )

    evidence = evaluate_run_checkpoint_candidates(
        cottage_design,
        run,
        protocol,
        matrix_id="matrix-fixture",
        output_root=tmp_path / "evidence",
        device="cpu",
    )

    assert loaded_paths == checkpoint_paths
    assert len(evidence.candidates) == 5
    assert all(len(candidate.episodes) == 10 for candidate in evidence.candidates)
    assert {
        (seed, failure_enabled)
        for seed, failure_enabled, split in evaluation_cells
        if split == "validation"
    } == {
        (seed, failure_enabled)
        for seed in protocol.selection.scenario_seeds
        for failure_enabled in protocol.selection.failure_modes
    }
    assert len(evaluation_cells) == 50
    assert evidence.selected.checkpoint_fraction == 0.75
    assert evidence.selected.transition_count == 48
    assert evidence.selected.scenario_seeds == [800, 801, 802, 803, 804]
    assert evidence.selected.resume_provenance == {
        "attempt": 2,
        "resumed": True,
    }
    assert evidence.selected.candidate_ranking[0].endswith("075pct")
    assert all(
        candidate.result.checkpoint_sha256
        == file_sha256(Path(candidate.result.checkpoint_path))
        for candidate in evidence.candidates
    )
    assert evidence.evidence_path.is_file()


def test_checkpoint_candidate_evaluation_rejects_missing_fraction(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    run = _candidate_run(tmp_path)
    artifact_dir = Path(cast(str, run["artifact_dir"]))
    _write_policy_checkpoints(artifact_dir, omit_fraction=0.1)

    with pytest.raises(ValueError, match=r"missing the 10% policy checkpoint"):
        evaluate_run_checkpoint_candidates(
            cottage_design,
            run,
            protocol,
            matrix_id="matrix-fixture",
            output_root=tmp_path / "evidence",
        )


def test_checkpoint_candidate_evaluation_rejects_metadata_fraction_mismatch(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    run = _candidate_run(tmp_path)
    artifact_dir = Path(cast(str, run["artifact_dir"]))
    _write_policy_checkpoints(
        artifact_dir,
        metadata_fraction_overrides={0.1: 0.11},
    )

    with pytest.raises(ValueError, match=r"checkpoint fraction mismatch"):
        evaluate_run_checkpoint_candidates(
            cottage_design,
            run,
            protocol,
            matrix_id="matrix-fixture",
            output_root=tmp_path / "evidence",
        )


def test_checkpoint_candidate_evaluation_rejects_unrecorded_resume_provenance(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    run = _candidate_run(tmp_path)
    run["resume_provenance_history"] = [{"attempt": 3, "resumed": True}]
    artifact_dir = Path(cast(str, run["artifact_dir"]))
    _write_policy_checkpoints(artifact_dir)

    with pytest.raises(ValueError, match=r"resume provenance is not recorded"):
        evaluate_run_checkpoint_candidates(
            cottage_design,
            run,
            protocol,
            matrix_id="matrix-fixture",
            output_root=tmp_path / "evidence",
        )


def test_matrix_selection_orchestration_freezes_twenty_immutable_selections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    registry = _completed_matrix(tmp_path, protocol, cottage_design)

    def selection_evaluator(
        _design: HouseDesign,
        run: dict[str, object],
        _protocol: ExperimentProtocol,
        *,
        matrix_id: str,
        output_root: Path,
        device: str = "cpu",
    ) -> RunSelectionEvidence:
        del _design, _protocol, device
        run_key = cast(str, run["run_key"])
        return RunSelectionEvidence(
            matrix_id=matrix_id,
            run_key=run_key,
            candidates=[],
            selected=_selection_for_run(run, output_root / run_key / "selected.pt"),
            evidence_path=output_root / run_key / "validation_selection.json",
        )

    monkeypatch.setattr(
        experiment_execution,
        "evaluate_run_checkpoint_candidates",
        selection_evaluator,
    )
    first = evaluate_and_freeze_matrix_selections(
        registry,
        "matrix-fixture",
        cottage_design,
        protocol,
        output_root=tmp_path / "selection-evidence",
    )
    second = evaluate_and_freeze_matrix_selections(
        registry,
        "matrix-fixture",
        cottage_design,
        protocol,
        output_root=tmp_path / "selection-evidence",
    )

    assert len(first.selections) == len(second.selections) == 20
    assert first.protocol_digest == protocol_digest(protocol)
    assert first.evidence_path.is_file()
    assert len(registry.list_policy_selections("matrix-fixture")) == 20
    matrix = registry.get_experiment_matrix("matrix-fixture")
    assert matrix is not None
    assert matrix["status"] == "selected"

    def changed_selection_evaluator(
        design: HouseDesign,
        run: dict[str, object],
        frozen_protocol: ExperimentProtocol,
        *,
        matrix_id: str,
        output_root: Path,
        device: str = "cpu",
    ) -> RunSelectionEvidence:
        evidence = selection_evaluator(
            design,
            run,
            frozen_protocol,
            matrix_id=matrix_id,
            output_root=output_root,
            device=device,
        )
        return evidence.model_copy(
            update={
                "selected": evidence.selected.model_copy(
                    update={
                        "checkpoint_id": f"{evidence.run_key}-replacement",
                        "checkpoint_sha256": "f" * 64,
                        "candidate_ranking": [
                            f"{evidence.run_key}-replacement",
                            *evidence.selected.candidate_ranking[1:],
                        ],
                    }
                )
            }
        )

    monkeypatch.setattr(
        experiment_execution,
        "evaluate_run_checkpoint_candidates",
        changed_selection_evaluator,
    )
    with pytest.raises(ValueError, match=r"selection is already frozen"):
        evaluate_and_freeze_matrix_selections(
            registry,
            "matrix-fixture",
            cottage_design,
            protocol,
            output_root=tmp_path / "selection-evidence",
        )


@pytest.mark.parametrize("stage", ["selection", "heldout"])
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", r"run grid.*missing="),
        ("extra", r"run grid.*unexpected="),
        ("config", r"mutated frozen configuration fields: learning_rate"),
        (
            "protocol_run_digest",
            r"mutated frozen configuration fields: protocol_run_digest",
        ),
        ("config_digest", r"configuration digest column"),
        ("runtime_fingerprint", r"configuration digest is invalid"),
        ("source_commit", r"source commit column"),
    ],
)
def test_matrix_execution_rejects_noncanonical_persisted_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
    stage: str,
    mutation: str,
    message: str,
) -> None:
    registry = _completed_matrix(tmp_path, protocol, cottage_design)
    persisted = registry.get_experiment_matrix("matrix-fixture")
    assert persisted is not None
    matrix = deepcopy(persisted)
    runs = cast(list[dict[str, object]], matrix["runs"])

    if mutation == "missing":
        runs.pop()
    elif mutation == "extra":
        unexpected = deepcopy(runs[0])
        unexpected["run_key"] = "unexpected-matrix-run"
        runs.append(unexpected)
    else:
        config = cast(dict[str, object], runs[0]["config"])
        if mutation == "config":
            config["learning_rate"] = 0.123
        elif mutation == "protocol_run_digest":
            config["protocol_run_digest"] = "f" * 64
        elif mutation == "config_digest":
            runs[0]["config_digest"] = "f" * 64
        elif mutation == "runtime_fingerprint":
            config["environment_fingerprint"] = {"fixture": "mutated"}
        elif mutation == "source_commit":
            runs[0]["source_commit"] = "other-source"
        else:  # pragma: no cover - guarded by the parameter table
            raise AssertionError(mutation)

    monkeypatch.setattr(
        registry,
        "get_experiment_matrix",
        lambda _matrix_id: matrix,
    )
    with pytest.raises(ValueError, match=message):
        if stage == "selection":
            evaluate_and_freeze_matrix_selections(
                registry,
                "matrix-fixture",
                cottage_design,
                protocol,
                output_root=tmp_path / "selection-evidence",
            )
        else:
            evaluate_frozen_heldout_matrix(
                registry,
                "matrix-fixture",
                cottage_design,
                protocol,
                output_root=tmp_path / "heldout",
            )


def test_heldout_evaluation_requires_all_twenty_frozen_selections(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    registry = _completed_matrix(tmp_path, protocol, cottage_design)
    matrix = registry.get_experiment_matrix("matrix-fixture")
    assert matrix is not None
    runs = cast(list[dict[str, object]], matrix["runs"])
    for run in runs[:-1]:
        selection = _selection_for_run(
            run,
            tmp_path / "selected" / f"{run['run_key']}.pt",
        )
        registry.freeze_policy_selection(
            "matrix-fixture",
            cast(str, run["run_key"]),
            selection.model_dump(mode="json"),
        )

    with pytest.raises(ValueError, match=r"requires all 20 frozen selections"):
        evaluate_frozen_heldout_matrix(
            registry,
            "matrix-fixture",
            cottage_design,
            protocol,
            output_root=tmp_path / "heldout",
        )


def test_heldout_evaluation_rejects_a_changed_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protocol: ExperimentProtocol,
    cottage_design: HouseDesign,
) -> None:
    registry = _completed_matrix(tmp_path, protocol, cottage_design)
    matrix = registry.get_experiment_matrix("matrix-fixture")
    assert matrix is not None
    runs = cast(list[dict[str, object]], matrix["runs"])
    for run in runs:
        checkpoint_path = tmp_path / "selected" / f"{run['run_key']}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(cast(str, run["run_key"]).encode("utf-8"))
        actual_sha = file_sha256(checkpoint_path)
        selection = _selection_for_run(
            run,
            checkpoint_path,
            checkpoint_sha=f"{actual_sha[:-1]}{'0' if actual_sha[-1] != '0' else '1'}",
        )
        registry.freeze_policy_selection(
            "matrix-fixture",
            cast(str, run["run_key"]),
            selection.model_dump(mode="json"),
        )

    baseline_calls = 0

    def fake_baseline_evaluation(
        _design: HouseDesign,
        *,
        seed: int,
        controller: ControllerName,
        failure_enabled: bool = False,
        expected_split: str | None = None,
        **_kwargs: object,
    ) -> EpisodeEvaluation:
        nonlocal baseline_calls
        baseline_calls += 1
        return _episode(
            controller=controller,
            seed=seed,
            split=expected_split or "test",
            failure_enabled=failure_enabled,
            completion=1.0,
            makespan=100.0,
        )

    def unexpected_policy_load(_path: Path, *, device: str) -> object:
        del device
        pytest.fail("a hash-mismatched selected checkpoint must never be loaded")

    monkeypatch.setattr(
        experiment_execution,
        "evaluate_controller_episode",
        fake_baseline_evaluation,
    )
    monkeypatch.setattr(
        experiment_execution,
        "load_policy_checkpoint",
        unexpected_policy_load,
    )

    with pytest.raises(ValueError, match=r"selected checkpoint hash mismatch"):
        evaluate_frozen_heldout_matrix(
            registry,
            "matrix-fixture",
            cottage_design,
            protocol,
            output_root=tmp_path / "heldout",
        )
    assert baseline_calls == 40


def test_primary_acceptance_boundaries_are_inclusive_and_structured(
    protocol: ExperimentProtocol,
) -> None:
    suite = _acceptance_suite(
        mappo_nominal_completion=0.95,
        ippo_nominal_completion=0.95,
        mappo_nominal_makespan=115.0,
        cp_sat_nominal_makespan=100.0,
        mappo_failure_completion=0.85,
    )

    audit = audit_primary_acceptance(suite, protocol)

    assert audit.passed
    assert [result.name for result in audit.results] == [
        "mappo_no_failure_mean_completion",
        "ippo_no_failure_mean_completion",
        "mappo_median_makespan_cp_sat_ratio",
        "mappo_failure_mean_completion",
    ]
    assert [result.comparison for result in audit.results] == [
        "min",
        "min",
        "max",
        "min",
    ]
    assert [result.observed for result in audit.results] == pytest.approx(
        [0.95, 0.95, 1.15, 0.85]
    )
    assert all(result.observed == pytest.approx(result.threshold) for result in audit.results)


@pytest.mark.parametrize(
    (
        "mappo_nominal_completion",
        "ippo_nominal_completion",
        "mappo_nominal_makespan",
        "mappo_failure_completion",
        "failed_name",
    ),
    [
        (0.9499, 0.95, 115.0, 0.85, "mappo_no_failure_mean_completion"),
        (0.95, 0.9499, 115.0, 0.85, "ippo_no_failure_mean_completion"),
        (0.95, 0.95, 115.01, 0.85, "mappo_median_makespan_cp_sat_ratio"),
        (0.95, 0.95, 115.0, 0.8499, "mappo_failure_mean_completion"),
    ],
)
def test_primary_acceptance_rejects_values_just_outside_boundaries(
    protocol: ExperimentProtocol,
    mappo_nominal_completion: float,
    ippo_nominal_completion: float,
    mappo_nominal_makespan: float,
    mappo_failure_completion: float,
    failed_name: str,
) -> None:
    suite = _acceptance_suite(
        mappo_nominal_completion=mappo_nominal_completion,
        ippo_nominal_completion=ippo_nominal_completion,
        mappo_nominal_makespan=mappo_nominal_makespan,
        cp_sat_nominal_makespan=100.0,
        mappo_failure_completion=mappo_failure_completion,
    )

    audit = audit_primary_acceptance(suite, protocol)

    assert not audit.passed
    failed = [result.name for result in audit.results if not result.passed]
    assert failed == [failed_name]


def test_behavior_cloning_audit_uses_validation_transitions_to_95(
    tmp_path: Path,
    protocol: ExperimentProtocol,
) -> None:
    evidence = _transition_curve_evidence(
        tmp_path,
        protocol,
        full_curve=[0.60, 0.80, 0.95, 0.97, 0.98],
        no_bc_curve=[0.55, 0.75, 0.90, 0.95, 0.97],
    )

    assert (
        validation_transitions_to_completion(
            evidence,
            protocol,
            experiment_variant="mappo_full",
        )
        == 32
    )
    assert (
        validation_transitions_to_completion(
            evidence,
            protocol,
            experiment_variant="mappo_no_bc",
        )
        == 48
    )
    decisions = audit_ablation_hypotheses(
        _ablation_suite(),
        protocol,
        selection_evidence=evidence,
    )
    behavior_cloning = next(
        item for item in decisions if item.hypothesis == "behavior_cloning"
    )

    assert behavior_cloning.completion_boundary_met is False
    assert behavior_cloning.transition_boundary_met is True
    assert behavior_cloning.transitions_to_95_reduction == pytest.approx(1 / 3)
    assert behavior_cloning.supported is True


def test_reproducibility_audit_never_asserts_a_missing_matrix_complete(
    tmp_path: Path,
    protocol: ExperimentProtocol,
) -> None:
    registry = LabRegistry(tmp_path / "missing.sqlite")
    selection_evidence = MatrixSelectionEvidence(
        matrix_id="missing-matrix",
        protocol_digest=protocol_digest(protocol),
        selections=[],
        evidence_path=tmp_path / "missing-selection.json",
    )

    audit = audit_reproducibility_artifacts(
        registry,
        "missing-matrix",
        protocol,
        selection_evidence,
        _acceptance_suite(
            mappo_nominal_completion=0.95,
            ippo_nominal_completion=0.95,
            mappo_nominal_makespan=115.0,
            cp_sat_nominal_makespan=100.0,
            mappo_failure_completion=0.85,
        ),
        heldout_run_dir=tmp_path / "heldout",
        acceptance_path=tmp_path / "acceptance.json",
        ablation_path=tmp_path / "ablations.json",
    )

    assert audit.complete is False
    assert audit.checked_file_count == 0
    assert audit.invalid_artifacts == ["matrix not found: missing-matrix"]


def _transition_curve_evidence(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    *,
    full_curve: list[float],
    no_bc_curve: list[float],
) -> MatrixSelectionEvidence:
    fractions = list(protocol.checkpoint_fractions)
    transition_counts = [6, 16, 32, 48, 64]
    selections: list[RunSelectionEvidence] = []
    for variant, curve in (
        ("mappo_full", full_curve),
        ("mappo_no_bc", no_bc_curve),
    ):
        assert len(curve) == len(fractions)
        for training_seed in protocol.training_seeds:
            run_key = f"{variant}-s{training_seed}"
            candidates: list[CandidateEvaluation] = []
            for fraction, transitions, completion in zip(
                fractions,
                transition_counts,
                curve,
                strict=True,
            ):
                checkpoint_id = (
                    f"{run_key}-checkpoint-{int(round(fraction * 100)):03d}pct"
                )
                result = CheckpointValidationResult(
                    checkpoint_id=checkpoint_id,
                    experiment_id=protocol.experiment_id,
                    experiment_variant=variant,
                    training_seed=training_seed,
                    checkpoint_fraction=fraction,
                    transition_count=transitions,
                    split=protocol.selection.split,
                    scenario_seeds=list(protocol.selection.scenario_seeds),
                    mean_completion_rate=completion,
                    mean_makespan_s=100.0,
                    checkpoint_path=str(tmp_path / f"{checkpoint_id}.pt"),
                    checkpoint_sha256="a" * 64,
                    checkpoint_lineage=[checkpoint_id],
                    configuration_digest=DIGEST,
                    source_commit=SOURCE_COMMIT,
                    resume_provenance={},
                )
                episodes = [
                    _episode(
                        controller="mappo",
                        seed=scenario_seed,
                        split=protocol.selection.split,
                        failure_enabled=failure_enabled,
                        completion=completion if not failure_enabled else completion * 0.9,
                        makespan=100.0,
                    ).model_copy(
                        update={
                            "experiment_id": protocol.experiment_id,
                            "experiment_variant": variant,
                            "training_seed": training_seed,
                        }
                    )
                    for failure_enabled in protocol.selection.failure_modes
                    for scenario_seed in protocol.selection.scenario_seeds
                ]
                candidates.append(
                    CandidateEvaluation(result=result, episodes=episodes)
                )
            ranking = [
                candidate.result.checkpoint_id
                for candidate in reversed(candidates)
            ]
            selected = SelectedCheckpoint(
                **candidates[-1].result.model_dump(),
                required_checkpoint_fractions=fractions,
                candidate_ranking=ranking,
            )
            selections.append(
                RunSelectionEvidence(
                    matrix_id="matrix-fixture",
                    run_key=run_key,
                    candidates=candidates,
                    selected=selected,
                    evidence_path=tmp_path / run_key / "validation_selection.json",
                )
            )
    evidence_path = tmp_path / "matrix_selections.json"
    return MatrixSelectionEvidence(
        matrix_id="matrix-fixture",
        protocol_digest=protocol_digest(protocol),
        selections=selections,
        evidence_path=evidence_path,
    )


def _ablation_suite() -> EvaluationSuite:
    return EvaluationSuite(
        evaluation_id="ablation-fixture",
        seeds=[900, 901, 902, 903, 904],
        controllers=["mappo"],
        episodes=[],
        summaries=[
            _controller_summary("mappo", "mappo_full", False, 0.96, 110.0),
            _controller_summary("mappo", "mappo_full", True, 0.86, 130.0),
            _controller_summary("mappo", "mappo_no_bc", False, 0.95, 115.0),
            _controller_summary(
                "mappo",
                "mappo_no_failure_curriculum",
                False,
                0.97,
                108.0,
            ),
            _controller_summary(
                "mappo",
                "mappo_no_failure_curriculum",
                True,
                0.74,
                145.0,
            ),
        ],
        expected_split="test",
    )


def _candidate_run(tmp_path: Path) -> dict[str, object]:
    config = TrainingConfig.for_profile("unit").model_copy(
        update={
            "experiment_id": "construction_intelligence_v1",
            "experiment_variant": "mappo_full",
            "training_seed": 7,
            "checkpoint_fractions": [0.1, 0.25, 0.5, 0.75, 1.0],
            "configuration_digest": DIGEST,
            "source_commit": SOURCE_COMMIT,
            "resume_provenance": {},
        }
    )
    return {
        "run_key": "mappo-full-s7",
        "artifact_dir": str(tmp_path / "training"),
        "config": config.model_dump(mode="json"),
        "resume_provenance_history": [{"attempt": 2, "resumed": True}],
    }


def _write_policy_checkpoints(
    artifact_dir: Path,
    *,
    omit_fraction: float | None = None,
    metadata_fraction_overrides: dict[float, float] | None = None,
) -> list[Path]:
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parents[1]
    fixture_design = load_house_design(
        workspace / "configs" / "construction" / "cottage_v1.yaml"
    )
    paths = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        if omit_fraction == fraction:
            continue
        percentage = int(round(fraction * 100))
        path = checkpoint_dir / f"policy_{percentage:03d}pct.pt"
        metadata_fraction = (metadata_fraction_overrides or {}).get(fraction, fraction)
        torch.save(
            {
                "metadata": {
                    "transition_count": max(1, int(round(64 * fraction))),
                    "checkpoint_fraction": metadata_fraction,
                    "checkpoint_lineage": [
                        f"policy_{int(round(prior * 100)):03d}pct.pt"
                        for prior in (0.1, 0.25, 0.5, 0.75, 1.0)
                        if prior <= fraction
                    ],
                    "experiment_id": "construction_intelligence_v1",
                    "experiment_variant": "mappo_full",
                    "training_seed": 7,
                    "configuration_digest": DIGEST,
                    "source_commit": SOURCE_COMMIT,
                    "source_dirty": False,
                    "source_tree_digest": None,
                    "design_digest": design_digest(fixture_design),
                    "environment_fingerprint": {},
                    "resume_provenance": {
                        "attempt": 2,
                        "resumed": True,
                    },
                }
            },
            path,
        )
        paths.append(path)
    return paths


def _completed_matrix(
    tmp_path: Path,
    protocol: ExperimentProtocol,
    design: HouseDesign,
) -> LabRegistry:
    registry = LabRegistry(tmp_path / "matrix.sqlite")
    specs = expand_experiment_matrix(protocol, "unit")
    runs: list[tuple[str, dict[str, object], str | None, str | None]] = []
    for spec in specs:
        config_model = TrainingConfig.model_validate(
            {
                **spec.training_config_payload(),
                "source_commit": SOURCE_COMMIT,
                "resume_provenance": {},
                "environment_fingerprint": {"fixture": True},
            }
        )
        config_model.configuration_digest = configuration_digest(config_model)
        config = config_model.model_dump(mode="json")
        runs.append(
            (
                spec.run_id,
                config,
                config_model.configuration_digest,
                SOURCE_COMMIT,
            )
        )
    run_ids = registry.create_experiment_matrix(
        "matrix-fixture",
        protocol_digest=protocol_digest(protocol),
        protocol=protocol.model_dump(mode="json"),
        execution_profile="unit",
        design=design.model_dump(mode="json"),
        runs=runs,
    )
    for run_id in run_ids:
        registry.update_run(
            run_id,
            status="completed",
            progress=1.0,
            artifact_dir=str(tmp_path / "training" / run_id),
        )
    return registry


def _selection_for_run(
    run: dict[str, object],
    checkpoint_path: Path,
    *,
    checkpoint_sha: str | None = None,
) -> SelectedCheckpoint:
    run_key = cast(str, run["run_key"])
    config = cast(dict[str, object], run["config"])
    ranking = [
        f"{run_key}-checkpoint-{percentage:03d}pct"
        for percentage in (100, 75, 50, 25, 10)
    ]
    return SelectedCheckpoint(
        checkpoint_id=ranking[0],
        experiment_id=cast(str, config["experiment_id"]),
        experiment_variant=cast(str, config["experiment_variant"]),
        training_seed=cast(int, config["training_seed"]),
        checkpoint_fraction=1.0,
        transition_count=cast(int, config["transitions"]),
        split="validation",
        scenario_seeds=[800, 801, 802, 803, 804],
        mean_completion_rate=0.99,
        mean_makespan_s=100.0,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha or sha256(run_key.encode("utf-8")).hexdigest(),
        checkpoint_lineage=[str(checkpoint_path)],
        configuration_digest=cast(str, config["configuration_digest"]),
        source_commit=cast(str, config["source_commit"]),
        resume_provenance={},
        required_checkpoint_fractions=[0.1, 0.25, 0.5, 0.75, 1.0],
        candidate_ranking=ranking,
    )


def _episode(
    *,
    controller: ControllerName,
    seed: int,
    split: str,
    failure_enabled: bool,
    completion: float,
    makespan: float,
    manifest: PolicyManifest | None = None,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        scenario_id=f"cottage-{split}-{seed}",
        seed=seed,
        split=split,
        controller=controller,
        failure_enabled=failure_enabled,
        structure_completion_rate=completion,
        makespan_s=makespan,
        total_travel_m=30.0,
        total_energy_wh=8.0,
        idle_robot_seconds=5.0,
        mean_robot_utilization=0.8,
        collision_count=0,
        wasted_work_s=0.0,
        invalid_bid_count=0,
        drop_count=0,
        decision_count=10,
        policy_id=manifest.policy_id if manifest is not None else None,
        experiment_id=manifest.experiment_id if manifest is not None else None,
        experiment_variant=(
            manifest.experiment_variant if manifest is not None else None
        ),
        training_seed=manifest.training_seed if manifest is not None else None,
        transition_count=manifest.transition_count if manifest is not None else None,
        checkpoint_path=manifest.checkpoint_path if manifest is not None else None,
        checkpoint_sha256=(
            manifest.checkpoint_sha256 if manifest is not None else None
        ),
        checkpoint_lineage=(
            list(manifest.checkpoint_lineage) if manifest is not None else []
        ),
        configuration_digest=(
            manifest.configuration_digest if manifest is not None else None
        ),
        source_commit=manifest.source_commit if manifest is not None else None,
        resume_provenance=(
            dict(manifest.resume_provenance) if manifest is not None else {}
        ),
    )


def _acceptance_suite(
    *,
    mappo_nominal_completion: float,
    ippo_nominal_completion: float,
    mappo_nominal_makespan: float,
    cp_sat_nominal_makespan: float,
    mappo_failure_completion: float,
) -> EvaluationSuite:
    summaries = [
        _controller_summary(
            "mappo",
            "mappo_full",
            False,
            mappo_nominal_completion,
            mappo_nominal_makespan,
        ),
        _controller_summary(
            "ippo",
            "ippo_full",
            False,
            ippo_nominal_completion,
            120.0,
        ),
        _controller_summary(
            "mappo",
            "mappo_full",
            True,
            mappo_failure_completion,
            130.0,
        ),
        _controller_summary(
            "cp_sat",
            None,
            False,
            1.0,
            cp_sat_nominal_makespan,
        ),
    ]
    return EvaluationSuite(
        evaluation_id="acceptance-boundary-fixture",
        seeds=[900, 901, 902, 903, 904],
        controllers=["mappo", "ippo", "cp_sat"],
        episodes=[],
        summaries=summaries,
        expected_split="test",
    )


def _controller_summary(
    controller: ControllerName,
    experiment_variant: str | None,
    failure_enabled: bool,
    completion: float,
    makespan: float,
) -> ControllerEvaluation:
    def summary(value: float) -> MetricSummary:
        return MetricSummary(
            mean=value,
            std=0.0,
            bootstrap_ci95_low=value,
            bootstrap_ci95_high=value,
            median=value,
        )

    return ControllerEvaluation(
        controller=controller,
        failure_enabled=failure_enabled,
        episode_count=25,
        metrics={
            "structure_completion_rate": summary(completion),
            "makespan_s": summary(makespan),
        },
        experiment_id=(
            "construction_intelligence_v1"
            if experiment_variant is not None
            else None
        ),
        experiment_variant=experiment_variant,
        training_seed_count=5 if experiment_variant is not None else 0,
        scenario_seed_count=5,
    )
