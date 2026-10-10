"""CLI-origin result integration from typed offline artifacts, never training/rollouts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import pytest

from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.workbench_evidence import build_local_research_summary


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def cli_results(tmp_path: Path, request):
    from embodied_skill_composer.construction.evaluation import (
        ControllerEvaluation,
        EpisodeEvaluation,
        EvaluationGridEntry,
        EvaluationSuite,
        MetricSummary,
        validate_evaluation_grid,
    )
    from embodied_skill_composer.construction.experiment_execution import (
        BASELINE_CONTROLLERS,
        CandidateEvaluation,
        MatrixSelectionEvidence,
        RunSelectionEvidence,
        audit_ablation_hypotheses,
        audit_primary_acceptance,
    )
    from embodied_skill_composer.construction.experiment_protocol import (
        CheckpointValidationResult,
        expand_experiment_matrix,
        load_experiment_protocol,
        protocol_digest,
        select_validation_checkpoint,
    )

    protocol = load_experiment_protocol()
    profile = getattr(request, "param", "unit")
    specs = expand_experiment_matrix(protocol, profile)
    matrix_id = f"cli-{profile}-fixture"
    registry = LabRegistry(tmp_path / "lab.sqlite")
    run_ids = registry.create_experiment_matrix(
        matrix_id, protocol_digest=protocol_digest(protocol), protocol=protocol.model_dump(mode="json"),
        execution_profile=profile, design={},
        runs=[(spec.run_id, spec.training_config_payload(), "a" * 64, "b" * 40) for spec in specs],
    )
    for run_id in run_ids:
        registry.update_run(run_id, status="completed", progress=1.0)

    def episode(controller, seed, failure, *, variant=None, training_seed=None, split="test", fraction=1.0):
        completion = 1.0
        if variant == "mappo_no_bc":
            completion = 0.92 if split == "test" else (1.0 if fraction == 1 else 0.8)
        elif variant == "mappo_no_failure_curriculum" and failure:
            completion = 0.8
        elif split == "validation" and fraction < 0.5:
            completion = 0.8
        return EpisodeEvaluation(
            scenario_id=f"offline-{seed}", seed=seed, split=split, controller=controller,
            failure_enabled=failure, structure_completion_rate=completion, makespan_s=100,
            total_travel_m=0, total_energy_wh=0, idle_robot_seconds=0, mean_robot_utilization=1,
            collision_count=0, wasted_work_s=0, invalid_bid_count=0, drop_count=0, decision_count=1,
            experiment_id=protocol.experiment_id if variant else None,
            experiment_variant=variant, training_seed=training_seed,
        )

    selections = []
    selection_path = tmp_path / "experiments" / "validation" / matrix_id / "matrix_selections.json"
    for spec in specs:
        candidates = []
        for fraction in protocol.checkpoint_fractions:
            episodes = [episode(
                spec.algorithm, seed, failure, variant=spec.experiment_variant,
                training_seed=spec.training_seed, split="validation", fraction=fraction,
            ) for failure in protocol.selection.failure_modes for seed in protocol.selection.scenario_seeds]
            result = CheckpointValidationResult(
                checkpoint_id=f"{spec.run_id}-{fraction}", experiment_id=protocol.experiment_id,
                experiment_variant=spec.experiment_variant, training_seed=spec.training_seed,
                checkpoint_fraction=fraction, transition_count=max(1, round(spec.transitions * fraction)),
                split="validation", scenario_seeds=protocol.selection.scenario_seeds,
                mean_completion_rate=mean(item.structure_completion_rate for item in episodes), mean_makespan_s=100,
                checkpoint_path=str(tmp_path / f"{spec.run_id}-{fraction}.pt"),
                checkpoint_sha256="c" * 64, checkpoint_lineage=[], configuration_digest="a" * 64,
                source_commit="b" * 40, resume_provenance={},
            )
            candidates.append(CandidateEvaluation(result=result, episodes=episodes))
        selected = select_validation_checkpoint(
            [item.result for item in candidates], validation_seeds=protocol.selection.scenario_seeds,
            required_fractions=protocol.checkpoint_fractions,
        )
        registry.freeze_policy_selection(matrix_id, spec.run_id, selected.model_dump(mode="json"))
        selections.append(RunSelectionEvidence(
            matrix_id=matrix_id, run_key=spec.run_id, candidates=candidates, selected=selected,
            evidence_path=selection_path.parent / spec.run_id / "validation_selection.json",
        ))
    selection_evidence = MatrixSelectionEvidence(
        matrix_id=matrix_id, protocol_digest=protocol_digest(protocol),
        selections=selections, evidence_path=selection_path,
    )
    _write(selection_path, selection_evidence.model_dump(mode="json"))
    policy_grid = [EvaluationGridEntry(controller=name) for name in BASELINE_CONTROLLERS]
    episodes = [episode(name, seed, failure) for name in BASELINE_CONTROLLERS
                for failure in protocol.evaluation.failure_modes for seed in protocol.evaluation.scenario_seeds]
    for spec in specs:
        policy_grid.append(EvaluationGridEntry(
            controller=spec.algorithm, experiment_id=protocol.experiment_id,
            experiment_variant=spec.experiment_variant, training_seed=spec.training_seed,
        ))
        episodes.extend(episode(spec.algorithm, seed, failure, variant=spec.experiment_variant,
                                training_seed=spec.training_seed)
                        for failure in protocol.evaluation.failure_modes for seed in protocol.evaluation.scenario_seeds)
    groups = defaultdict(list)
    for item in episodes:
        groups[(item.controller, item.experiment_variant, item.failure_enabled)].append(item)
    summaries = []
    for (controller, variant, failure), items in groups.items():
        metrics = {}
        for field in ("structure_completion_rate", "makespan_s"):
            values = [getattr(item, field) for item in items]
            metrics[field] = MetricSummary(mean=mean(values), std=0, bootstrap_ci95_low=min(values),
                                           bootstrap_ci95_high=max(values), median=median(values))
        summaries.append(ControllerEvaluation(
            controller=controller, experiment_variant=variant, failure_enabled=failure,
            experiment_id=protocol.experiment_id if variant else None, episode_count=len(items), metrics=metrics,
        ))
    suite = EvaluationSuite(
        evaluation_id="offline-cli-heldout", seeds=protocol.evaluation.scenario_seeds,
        controllers=protocol.evaluation.controllers, episodes=episodes, summaries=summaries, expected_split="test",
        grid_validation=validate_evaluation_grid(
            episodes, scenario_seeds=protocol.evaluation.scenario_seeds, policy_grid=policy_grid,
            failure_modes=protocol.evaluation.failure_modes, expected_split="test",
        ),
    )
    root = tmp_path / "experiments" / "heldout" / suite.evaluation_id
    payload = suite.model_dump(mode="json")
    _write(root / "evaluation.json", payload)
    acceptance = audit_primary_acceptance(suite, protocol).model_dump(mode="json")
    ablations = [item.model_dump(mode="json") for item in audit_ablation_hypotheses(
        suite, protocol, selection_evidence=selection_evidence,
    )]
    _write(root / "acceptance.json", acceptance)
    _write(root / "ablations.json", ablations)
    registry.upsert_evaluation(suite.evaluation_id, matrix_id=matrix_id, split="test", payload=payload, artifact_dir=str(root))
    return registry, root, selection_path, payload, acceptance, ablations


def test_completed_cli_matrix_exposes_verified_results_without_job_events_or_rollouts(cli_results, monkeypatch) -> None:
    registry, _, _, payload, acceptance, ablations = cli_results
    from embodied_skill_composer.construction import experiment_execution

    def forbidden(*args, **kwargs):
        pytest.fail("reading CLI results must not evaluate policies or write registry state")
    monkeypatch.setattr(experiment_execution, "evaluate_controller_episode", forbidden)
    monkeypatch.setattr(registry, "append_event", forbidden)
    monkeypatch.setattr(registry, "upsert_evaluation", forbidden)
    before = registry.list_runs()
    summary = build_local_research_summary(registry)
    assert summary["status"] == "in_progress"  # Unit evidence cannot claim research completion.
    assert summary["claim_allowed"] is False
    assert summary["training_run_count"] == summary["selection_count"] == 20
    assert summary["heldout_episode_count"] == 240
    assert summary["acceptance"] == acceptance
    assert summary["ablations"] == ablations
    assert summary["confidence_intervals"] == payload["summaries"]
    assert registry.list_runs() == before
    assert all(run["kind"] == "training" for run in before)


@pytest.mark.parametrize("cli_results", ["research"], indirect=True)
def test_research_profile_cli_artifacts_complete_the_local_view_without_a_public_claim(cli_results) -> None:
    registry, _, _, _, acceptance, ablations = cli_results
    summary = build_local_research_summary(registry)
    assert summary["status"] == "local_complete"
    assert summary["claim_allowed"] is False
    assert summary["training_run_count"] == summary["selection_count"] == 20
    assert summary["heldout_episode_count"] == 240
    assert summary["acceptance"] == acceptance
    assert summary["ablations"] == ablations
    assert all(run["kind"] == "training" for run in registry.list_runs())


@pytest.mark.parametrize("fault", [
    "missing_acceptance", "invalid_acceptance", "changed_acceptance", "missing_ablations", "changed_ablations",
    "wrong_evaluation_id", "disk_payload_differs", "registry_split", "episode_split", "missing_episode",
    "missing_selections", "changed_selection", "candidate_split", "candidate_controller", "candidate_source",
    "missing_fraction", "duplicate_selection",
    "wrong_protocol_digest", "linked_acceptance",
])
def test_cli_results_fail_closed_on_missing_changed_or_cross_split_artifacts(cli_results, monkeypatch, fault: str) -> None:
    registry, root, selection_path, payload, acceptance, ablations = cli_results
    if fault == "missing_acceptance":
        (root / "acceptance.json").unlink()
    elif fault == "invalid_acceptance":
        (root / "acceptance.json").write_text("not json", encoding="utf-8")
    elif fault == "changed_acceptance":
        acceptance["results"][0]["observed"] = 0
        _write(root / "acceptance.json", acceptance)
    elif fault == "missing_ablations":
        (root / "ablations.json").unlink()
    elif fault == "changed_ablations":
        ablations[0]["supported"] = not ablations[0]["supported"]
        _write(root / "ablations.json", ablations)
    elif fault == "disk_payload_differs":
        changed = dict(payload, evaluation_id="another")
        _write(root / "evaluation.json", changed)
    elif fault in {"wrong_evaluation_id", "registry_split", "episode_split", "missing_episode"}:
        if fault == "wrong_evaluation_id":
            payload["evaluation_id"] = "another"
        elif fault == "episode_split":
            payload["episodes"][0]["split"] = "validation"
        elif fault == "missing_episode":
            payload["episodes"].pop()
        _write(root / "evaluation.json", payload)
        registry.upsert_evaluation("offline-cli-heldout", matrix_id="cli-unit-fixture",
                                   split="validation" if fault == "registry_split" else "test",
                                   payload=payload, artifact_dir=str(root))
    elif fault == "missing_selections":
        selection_path.unlink()
    elif fault in {"changed_selection", "candidate_split", "candidate_controller", "candidate_source",
                   "missing_fraction", "duplicate_selection"}:
        selections = json.loads(selection_path.read_text(encoding="utf-8"))
        if fault == "changed_selection":
            selections["selections"][0]["selected"]["checkpoint_sha256"] = "d" * 64
        elif fault == "candidate_split":
            selections["selections"][0]["candidates"][0]["episodes"][0]["split"] = "test"
        elif fault == "candidate_controller":
            for episode in selections["selections"][0]["candidates"][0]["episodes"]:
                episode["controller"] = "ippo"
        elif fault == "candidate_source":
            selections["selections"][0]["candidates"][0]["result"]["source_commit"] = "e" * 40
        elif fault == "missing_fraction":
            selections["selections"][0]["candidates"].pop()
        else:
            selections["selections"].append(selections["selections"][0])
        _write(selection_path, selections)
    elif fault == "wrong_protocol_digest":
        original = registry.list_experiment_matrices
        def matrices():
            result = original()
            result[0]["protocol_digest"] = "f" * 64
            return result
        monkeypatch.setattr(registry, "list_experiment_matrices", matrices)
    else:
        original_path = root / "acceptance.json"
        original_path.rename(root / "redirected.json")
        try:
            original_path.symlink_to(root / "redirected.json")
        except OSError:
            pytest.skip("symlink creation unavailable")
    summary = build_local_research_summary(registry)
    assert summary["status"] == "in_progress"
    assert summary["claim_allowed"] is False
    assert summary["acceptance"] is None
    assert summary["ablations"] == []
