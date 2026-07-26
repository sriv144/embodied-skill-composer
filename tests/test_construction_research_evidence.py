from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal, TypedDict, cast

from embodied_skill_composer.construction.evaluation import (
    EpisodeEvaluation,
    EvaluationGridValidation,
    EvaluationSuite,
)
from embodied_skill_composer.construction.experiment_execution import (
    AcceptanceResult,
    CandidateEvaluation,
    MatrixSelectionEvidence,
    PrimaryAcceptanceAudit,
    RunSelectionEvidence,
)
from embodied_skill_composer.construction.experiment_protocol import (
    AblationDecision,
    CheckpointValidationResult,
    ExperimentProtocol,
    SelectedCheckpoint,
    expand_experiment_matrix,
    load_experiment_protocol,
    protocol_digest,
)
from embodied_skill_composer.construction.intelligence_models import (
    PolicyManifest,
)
from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.policy import file_sha256
from embodied_skill_composer.construction.research_evidence import (
    audit_reproducibility_artifacts,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.training import TrainingConfig


SOURCE_COMMIT = "research-evidence-fixture"


class AuditArguments(TypedDict):
    registry: LabRegistry
    matrix_id: str
    protocol: ExperimentProtocol
    selection_evidence: MatrixSelectionEvidence
    suite: EvaluationSuite
    heldout_run_dir: Path
    acceptance_path: Path
    ablation_path: Path


def test_reproducibility_audit_passes_complete_fixture_and_detects_tampering(
    tmp_path: Path,
) -> None:
    fixture = _complete_research_fixture(tmp_path)

    complete = audit_reproducibility_artifacts(**fixture)

    assert complete.complete
    assert complete.missing_paths == []
    assert complete.invalid_artifacts == []
    assert complete.checked_file_count >= 20 * 16 + 5

    checkpoint = next(
        Path(candidate.result.checkpoint_path)
        for selection in fixture["selection_evidence"].selections
        for candidate in selection.candidates
    )
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(b"tampered-policy")
    tampered = audit_reproducibility_artifacts(**fixture)
    assert not tampered.complete
    assert any(
        "candidate checkpoint hash mismatch" in item
        for item in tampered.invalid_artifacts
    )

    checkpoint.write_bytes(original)
    fixture["acceptance_path"].write_text(
        PrimaryAcceptanceAudit(
            passed=False,
            results=_acceptance_results(passed=False),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    failed_acceptance = audit_reproducibility_artifacts(**fixture)
    assert not failed_acceptance.complete
    assert "primary acceptance thresholds did not pass" in (
        failed_acceptance.invalid_artifacts
    )

    fixture["acceptance_path"].write_text(
        PrimaryAcceptanceAudit(
            passed=True,
            results=_acceptance_results(passed=True),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    fixture["heldout_run_dir"].joinpath("episodes.csv").write_text(
        "seed\n900\n",
        encoding="utf-8",
    )
    incomplete_csv = audit_reproducibility_artifacts(**fixture)
    assert not incomplete_csv.complete
    assert "held-out episode CSV has 1 rows instead of 240" in (
        incomplete_csv.invalid_artifacts
    )


def _complete_research_fixture(tmp_path: Path) -> AuditArguments:
    protocol = load_experiment_protocol()
    design = load_house_design(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "construction"
        / "cottage_v1.yaml"
    )
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_configs: dict[str, TrainingConfig] = {}
    matrix_runs: list[
        tuple[str, dict[str, object], str | None, str | None]
    ] = []
    for spec in expand_experiment_matrix(protocol, "research"):
        config = TrainingConfig.model_validate(
            {
                **spec.training_config_payload(),
                "configuration_digest": spec.protocol_run_digest,
                "source_commit": SOURCE_COMMIT,
                "environment_fingerprint": {"fixture": True},
            }
        )
        run_configs[spec.run_id] = config
        matrix_runs.append(
            (
                spec.run_id,
                config.model_dump(mode="json"),
                config.configuration_digest,
                config.source_commit,
            )
        )
    run_ids = registry.create_experiment_matrix(
        "research-fixture",
        protocol_digest=protocol_digest(protocol),
        protocol=protocol.model_dump(mode="json"),
        execution_profile="research",
        design=design.model_dump(mode="json"),
        runs=matrix_runs,
    )

    selections: list[RunSelectionEvidence] = []
    for run_id in run_ids:
        run_key = run_id.removeprefix("research-fixture-")
        config = run_configs[run_key]
        artifact_dir = tmp_path / "training" / run_key
        checkpoint_dir = artifact_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        policy_path = artifact_dir / "policy.pt"
        policy_path.write_bytes(f"final:{run_key}".encode())
        (artifact_dir / "actor.onnx").write_bytes(b"fixture-onnx")
        (artifact_dir / "training_config.json").write_text(
            config.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (artifact_dir / "learning_curve.csv").write_text(
            "update,transitions\n1,1500000\n",
            encoding="utf-8",
        )
        manifest = PolicyManifest(
            policy_id=run_key,
            controller=config.algorithm,
            git_sha=SOURCE_COMMIT,
            seed=config.seed,
            experiment_id=config.experiment_id,
            experiment_variant=config.experiment_variant,
            training_seed=config.training_seed,
            transition_count=config.transitions,
            checkpoint_path=str(policy_path.resolve()),
            checkpoint_sha256=file_sha256(policy_path),
            configuration_digest=config.configuration_digest,
            source_commit=config.source_commit,
            environment_fingerprint=dict(config.environment_fingerprint),
            onnx_path=str((artifact_dir / "actor.onnx").resolve()),
            config=config.model_dump(mode="json"),
        )
        (artifact_dir / "policy_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        candidates: list[CandidateEvaluation] = []
        lineage: list[str] = []
        for fraction in protocol.checkpoint_fractions:
            percentage = int(round(fraction * 100))
            resumable_path = (
                checkpoint_dir / f"checkpoint_{percentage:03d}pct.pt"
            )
            candidate_path = (
                checkpoint_dir / f"policy_{percentage:03d}pct.pt"
            )
            resumable_path.write_bytes(
                f"resume:{run_key}:{percentage}".encode()
            )
            candidate_path.write_bytes(
                f"policy:{run_key}:{percentage}".encode()
            )
            lineage.append(str(resumable_path.resolve()))
            transition_count = int(config.transitions * fraction)
            checkpoint_id = (
                f"{run_key}-checkpoint-{percentage:03d}pct"
            )
            result = CheckpointValidationResult(
                checkpoint_id=checkpoint_id,
                experiment_id=config.experiment_id,
                experiment_variant=config.experiment_variant,
                training_seed=cast(int, config.training_seed),
                checkpoint_fraction=fraction,
                transition_count=transition_count,
                split="validation",
                scenario_seeds=list(protocol.selection.scenario_seeds),
                mean_completion_rate=fraction,
                mean_makespan_s=200.0 - percentage,
                checkpoint_path=str(candidate_path.resolve()),
                checkpoint_sha256=file_sha256(candidate_path),
                checkpoint_lineage=list(lineage),
                configuration_digest=cast(
                    str,
                    config.configuration_digest,
                ),
                source_commit=cast(str, config.source_commit),
                resume_provenance={},
            )
            candidates.append(
                CandidateEvaluation(
                    result=result,
                    episodes=[
                        _episode(
                            seed=seed,
                            split="validation",
                            failure_enabled=failure_enabled,
                            experiment_variant=config.experiment_variant,
                            training_seed=cast(int, config.training_seed),
                        )
                        for failure_enabled in protocol.selection.failure_modes
                        for seed in protocol.selection.scenario_seeds
                    ],
                )
            )
        selected_result = candidates[-1].result
        selected = SelectedCheckpoint(
            **selected_result.model_dump(),
            required_checkpoint_fractions=list(
                protocol.checkpoint_fractions
            ),
            candidate_ranking=[
                item.result.checkpoint_id for item in reversed(candidates)
            ],
        )
        evidence_path = (
            tmp_path
            / "validation"
            / run_key
            / "validation_selection.json"
        )
        evidence_path.parent.mkdir(parents=True)
        evidence = RunSelectionEvidence(
            matrix_id="research-fixture",
            run_key=run_key,
            candidates=candidates,
            selected=selected,
            evidence_path=evidence_path,
        )
        evidence_path.write_text(
            evidence.model_dump_json(indent=2),
            encoding="utf-8",
        )
        selections.append(evidence)
        registry.update_run(
            run_id,
            status="completed",
            progress=1.0,
            artifact_dir=str(artifact_dir),
        )
        registry.freeze_policy_selection(
            "research-fixture",
            run_key,
            selected.model_dump(mode="json"),
        )

    matrix_evidence_path = (
        tmp_path / "validation" / "matrix_selections.json"
    )
    selection_evidence = MatrixSelectionEvidence(
        matrix_id="research-fixture",
        protocol_digest=protocol_digest(protocol),
        selections=selections,
        evidence_path=matrix_evidence_path,
    )
    matrix_evidence_path.write_text(
        selection_evidence.model_dump_json(indent=2),
        encoding="utf-8",
    )

    heldout_run_dir = tmp_path / "heldout"
    heldout_run_dir.mkdir()
    episodes = [
        _episode(
            seed=900 + (index % 5),
            split="test",
            failure_enabled=bool(index % 2),
            experiment_variant=None,
            training_seed=None,
        )
        for index in range(240)
    ]
    suite = EvaluationSuite(
        evaluation_id="heldout-fixture",
        seeds=list(protocol.evaluation.scenario_seeds),
        controllers=list(protocol.evaluation.controllers),
        episodes=episodes,
        summaries=[],
        expected_split="test",
        grid_validation=EvaluationGridValidation(
            expected_episode_count=240,
            observed_episode_count=240,
            expected_split="test",
        ),
    )
    (heldout_run_dir / "evaluation.json").write_text(
        suite.model_dump_json(indent=2),
        encoding="utf-8",
    )
    with (heldout_run_dir / "episodes.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed"])
        writer.writeheader()
        writer.writerows({"seed": item.seed} for item in episodes)
    (heldout_run_dir / "report.md").write_text(
        "# Held-out report\n",
        encoding="utf-8",
    )
    acceptance_path = heldout_run_dir / "acceptance.json"
    acceptance_path.write_text(
        PrimaryAcceptanceAudit(
            passed=True,
            results=_acceptance_results(passed=True),
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    ablation_path = heldout_run_dir / "ablations.json"
    ablation_path.write_text(
        json.dumps(
            [
                AblationDecision(
                    hypothesis="behavior_cloning",
                    supported=False,
                    interpretation="fixture",
                ).model_dump(mode="json"),
                AblationDecision(
                    hypothesis="failure_curriculum",
                    supported=False,
                    interpretation="fixture",
                ).model_dump(mode="json"),
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "registry": registry,
        "matrix_id": "research-fixture",
        "protocol": protocol,
        "selection_evidence": selection_evidence,
        "suite": suite,
        "heldout_run_dir": heldout_run_dir,
        "acceptance_path": acceptance_path,
        "ablation_path": ablation_path,
    }


def _episode(
    *,
    seed: int,
    split: str,
    failure_enabled: bool,
    experiment_variant: str | None,
    training_seed: int | None,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        scenario_id=f"{split}-{seed}",
        seed=seed,
        split=split,
        controller="mappo" if experiment_variant else "sequential",
        failure_enabled=failure_enabled,
        structure_completion_rate=1.0,
        makespan_s=100.0,
        total_travel_m=20.0,
        total_energy_wh=5.0,
        idle_robot_seconds=2.0,
        mean_robot_utilization=0.8,
        collision_count=0,
        wasted_work_s=0.0,
        invalid_bid_count=0,
        drop_count=0,
        decision_count=10,
        experiment_id=(
            "construction_intelligence_v1"
            if experiment_variant
            else None
        ),
        experiment_variant=experiment_variant,
        training_seed=training_seed,
    )


def _acceptance_results(*, passed: bool) -> list[AcceptanceResult]:
    fixtures: tuple[
        tuple[str, float, Literal["min", "max"]],
        ...,
    ] = (
        ("mappo_no_failure_mean_completion", 0.95, "min"),
        ("ippo_no_failure_mean_completion", 0.95, "min"),
        ("mappo_median_makespan_cp_sat_ratio", 1.15, "max"),
        ("mappo_failure_mean_completion", 0.85, "min"),
    )
    return [
        AcceptanceResult(
            name=name,
            passed=passed,
            observed=threshold if passed else 0.0,
            threshold=threshold,
            comparison=comparison,
        )
        for name, threshold, comparison in fixtures
    ]
