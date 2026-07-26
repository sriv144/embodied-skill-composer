from __future__ import annotations

from embodied_skill_composer.construction.evaluation import (
    ControllerName,
    EpisodeEvaluation,
    EvaluationGridEntry,
    EvaluationSuite,
    render_evaluation_report,
    summarize_by_scenario_seed,
    summarize_by_training_seed,
    summarize_evaluations,
    validate_evaluation_grid,
)
from embodied_skill_composer.construction.experiment_execution import (
    audit_primary_acceptance,
)
from embodied_skill_composer.construction.experiment_protocol import (
    AblationEvidence,
    CheckpointValidationResult,
    ReleaseEvidence,
    decide_ablation_support,
    decide_release_completeness,
    expand_experiment_matrix,
    load_experiment_protocol,
    select_validation_checkpoint,
)


def test_complete_protocol_matrix_smoke_keeps_selection_isolated_from_heldout() -> None:
    protocol = load_experiment_protocol()
    unit_runs = expand_experiment_matrix(protocol, "unit")
    smoke_runs = expand_experiment_matrix(protocol, "smoke")
    research_runs = expand_experiment_matrix(protocol, "research")
    assert len(unit_runs) == len(smoke_runs) == len(research_runs) == 20

    selections = {}
    for run in research_runs:
        candidates = [
            CheckpointValidationResult(
                checkpoint_id=f"{run.run_id}-{int(fraction * 100):03d}",
                experiment_id=run.experiment_id,
                experiment_variant=run.experiment_variant,
                training_seed=run.training_seed,
                checkpoint_fraction=fraction,
                transition_count=int(run.transitions * fraction),
                split="validation",
                scenario_seeds=list(protocol.selection.scenario_seeds),
                mean_completion_rate=0.80 + 0.03 * index,
                mean_makespan_s=130.0 - 5.0 * index,
                checkpoint_path=f"C:/fixture/{run.run_id}/{fraction}.pt",
                checkpoint_sha256=f"{index + 1:064x}",
                checkpoint_lineage=[
                    f"C:/fixture/{run.run_id}/{earlier}.pt"
                    for earlier in protocol.checkpoint_fractions[: index + 1]
                ],
                configuration_digest=run.protocol_run_digest,
                source_commit="source-commit",
                resume_provenance={"attempt": 1},
            )
            for index, fraction in enumerate(protocol.checkpoint_fractions)
        ]
        selections[run.run_id] = select_validation_checkpoint(
            candidates,
            validation_seeds=protocol.selection.scenario_seeds,
            required_fractions=protocol.checkpoint_fractions,
        )
    selected_ids_before_heldout = {
        run_id: selection.checkpoint_id
        for run_id, selection in selections.items()
    }
    assert len(selections) == 20
    assert all(
        selection.checkpoint_fraction == 1.0 for selection in selections.values()
    )

    episodes = []
    policy_grid = []
    for controller in ("sequential", "greedy", "auction", "cp_sat"):
        policy_grid.append(EvaluationGridEntry(controller=controller))
        for failure_enabled in protocol.evaluation.failure_modes:
            for scenario_seed in protocol.evaluation.scenario_seeds:
                episodes.append(
                    _episode(
                        controller=controller,
                        scenario_seed=scenario_seed,
                        failure_enabled=failure_enabled,
                        completion=1.0,
                        makespan=100.0 if controller == "cp_sat" else 125.0,
                    )
                )
    for run in research_runs:
        policy_grid.append(
            EvaluationGridEntry(
                controller=run.algorithm,
                experiment_id=run.experiment_id,
                experiment_variant=run.experiment_variant,
                training_seed=run.training_seed,
            )
        )
        for failure_enabled in protocol.evaluation.failure_modes:
            for scenario_seed in protocol.evaluation.scenario_seeds:
                completion, makespan = _learned_fixture_metrics(
                    run.experiment_variant,
                    failure_enabled,
                )
                selection = selections[run.run_id]
                episodes.append(
                    _episode(
                        controller=run.algorithm,
                        scenario_seed=scenario_seed,
                        failure_enabled=failure_enabled,
                        completion=completion,
                        makespan=makespan,
                        experiment_id=run.experiment_id,
                        experiment_variant=run.experiment_variant,
                        training_seed=run.training_seed,
                        checkpoint_id=selection.checkpoint_id,
                        checkpoint_path=selection.checkpoint_path,
                        checkpoint_sha256=selection.checkpoint_sha256,
                        configuration_digest=selection.configuration_digest,
                        source_commit=selection.source_commit,
                    )
                )

    grid = validate_evaluation_grid(
        episodes,
        scenario_seeds=protocol.evaluation.scenario_seeds,
        policy_grid=policy_grid,
        failure_modes=protocol.evaluation.failure_modes,
        expected_split="test",
    )
    assert grid.expected_episode_count == grid.observed_episode_count == 240
    suite = EvaluationSuite(
        evaluation_id="protocol-smoke-heldout",
        seeds=list(protocol.evaluation.scenario_seeds),
        controllers=list(protocol.evaluation.controllers),
        episodes=episodes,
        summaries=summarize_evaluations(episodes),
        expected_split="test",
        grid_validation=grid,
        per_training_seed=summarize_by_training_seed(episodes),
        per_scenario_seed=summarize_by_scenario_seed(episodes),
    )
    acceptance = audit_primary_acceptance(suite, protocol)
    assert acceptance.passed
    report = render_evaluation_report(suite)
    assert "hierarchical bootstrap" in report
    assert "240/240 expected episodes" in report

    ablations = [
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="behavior_cloning",
                full_final_completion=0.98,
                ablated_final_completion=0.90,
            ),
        ),
        decide_ablation_support(
            protocol,
            AblationEvidence(
                hypothesis="failure_curriculum",
                full_failure_completion=0.90,
                ablated_failure_completion=0.75,
                full_no_failure_completion=0.98,
                ablated_no_failure_completion=0.97,
            ),
        ),
    ]
    release = decide_release_completeness(
        protocol,
        ReleaseEvidence(
            completed_training_run_ids=[run.run_id for run in research_runs],
            selected_policy_run_ids=list(selections),
            heldout_evaluated_run_ids=[run.run_id for run in research_runs],
            ablation_decisions=ablations,
            primary_acceptance_passed=acceptance.passed,
            report_generated=True,
            reproducibility_artifacts_complete=True,
        ),
    )
    assert release.complete
    assert {
        run_id: selection.checkpoint_id
        for run_id, selection in selections.items()
    } == selected_ids_before_heldout


def _learned_fixture_metrics(
    variant: str,
    failure_enabled: bool,
) -> tuple[float, float]:
    if variant == "mappo_full":
        return (0.90 if failure_enabled else 0.98, 110.0)
    if variant == "ippo_full":
        return (0.88 if failure_enabled else 0.96, 116.0)
    if variant == "mappo_no_bc":
        return (0.78 if failure_enabled else 0.90, 128.0)
    return (0.75 if failure_enabled else 0.97, 112.0)


def _episode(
    *,
    controller: ControllerName,
    scenario_seed: int,
    failure_enabled: bool,
    completion: float,
    makespan: float,
    experiment_id: str | None = None,
    experiment_variant: str | None = None,
    training_seed: int | None = None,
    checkpoint_id: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_sha256: str | None = None,
    configuration_digest: str | None = None,
    source_commit: str | None = None,
) -> EpisodeEvaluation:
    return EpisodeEvaluation(
        scenario_id=f"cottage-test-{scenario_seed:03d}",
        seed=scenario_seed,
        split="test",
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
        routing_backend="prioritized_astar",
        policy_id=checkpoint_id,
        experiment_id=experiment_id,
        experiment_variant=experiment_variant,
        training_seed=training_seed,
        transition_count=1_500_000 if checkpoint_id else None,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_lineage=[checkpoint_path] if checkpoint_path else [],
        configuration_digest=configuration_digest,
        source_commit=source_commit,
        resume_provenance={"attempt": 1} if checkpoint_id else {},
    )
