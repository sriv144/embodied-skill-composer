"""CLI-origin result integration from typed offline artifacts, never training/rollouts."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from statistics import mean
from time import perf_counter

import pytest

from embodied_skill_composer.construction.lab_registry import LabRegistry
from embodied_skill_composer.construction.workbench_evidence import build_local_research_summary

_CHECKPOINT_FIELDS = {
    "experiment_id",
    "experiment_variant",
    "training_seed",
    "transition_count",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_lineage",
    "configuration_digest",
    "source_commit",
    "resume_provenance",
}
_PROVENANCE_MUTATIONS = {
    "policy_id": "wrong-policy",
    "transition_count": 987654,
    "checkpoint_path": "different-checkpoint.pt",
    "checkpoint_sha256": "e" * 64,
    "checkpoint_lineage": ["different-ancestor"],
    "configuration_digest": "f" * 64,
    "source_commit": "d" * 40,
    "resume_provenance": {"run_id": "other", "attempt": 9, "checkpoint": "other.pt"},
}


@lru_cache(maxsize=1)
def _offline_aggregate_tables(episodes_json: str):
    """Use real deterministic summaries once, independent of verifier caching."""
    from pydantic import TypeAdapter

    from embodied_skill_composer.construction.evaluation import (
        EpisodeEvaluation,
        summarize_by_scenario_seed,
        summarize_by_training_seed,
        summarize_evaluations,
    )

    episodes = TypeAdapter(list[EpisodeEvaluation]).validate_json(episodes_json)
    return tuple(
        [item.model_dump(mode="json") for item in summarizer(episodes)]
        for summarizer in (
            summarize_evaluations,
            summarize_by_training_seed,
            summarize_by_scenario_seed,
        )
    )


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def cli_results(tmp_path: Path, request):
    from embodied_skill_composer.construction.evaluation import (
        EpisodeEvaluation,
        EvaluationGridEntry,
        EvaluationSuite,
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
        matrix_id,
        protocol_digest=protocol_digest(protocol),
        protocol=protocol.model_dump(mode="json"),
        execution_profile=profile,
        design={},
        runs=[(spec.run_id, spec.training_config_payload(), "a" * 64, "b" * 40) for spec in specs],
    )
    for run_id in run_ids:
        registry.update_run(run_id, status="completed", progress=1.0)

    def episode(
        controller, seed, failure, *, variant=None, training_seed=None, split="test", fraction=1.0
    ):
        completion = 1.0
        if variant == "mappo_no_bc":
            completion = 0.92 if split == "test" else (1.0 if fraction == 1 else 0.8)
        elif variant == "mappo_no_failure_curriculum" and failure:
            completion = 0.8
        elif split == "validation" and fraction < 0.5:
            completion = 0.8
        return EpisodeEvaluation(
            scenario_id=f"offline-{seed}",
            seed=seed,
            split=split,
            controller=controller,
            failure_enabled=failure,
            structure_completion_rate=completion,
            makespan_s=100,
            total_travel_m=0,
            total_energy_wh=0,
            idle_robot_seconds=0,
            mean_robot_utilization=1,
            collision_count=0,
            wasted_work_s=0,
            invalid_bid_count=0,
            drop_count=0,
            decision_count=1,
            experiment_id=protocol.experiment_id if variant else None,
            experiment_variant=variant,
            training_seed=training_seed,
        )

    selections = []
    selection_path = tmp_path / "experiments" / "validation" / matrix_id / "matrix_selections.json"
    for spec in specs:
        candidates = []
        for fraction in protocol.checkpoint_fractions:
            episodes = [
                episode(
                    spec.algorithm,
                    seed,
                    failure,
                    variant=spec.experiment_variant,
                    training_seed=spec.training_seed,
                    split="validation",
                    fraction=fraction,
                )
                for failure in protocol.selection.failure_modes
                for seed in protocol.selection.scenario_seeds
            ]
            result = CheckpointValidationResult(
                checkpoint_id=f"{spec.run_id}-{fraction}",
                experiment_id=protocol.experiment_id,
                experiment_variant=spec.experiment_variant,
                training_seed=spec.training_seed,
                checkpoint_fraction=fraction,
                transition_count=max(1, round(spec.transitions * fraction)),
                split="validation",
                scenario_seeds=protocol.selection.scenario_seeds,
                mean_completion_rate=mean(item.structure_completion_rate for item in episodes),
                mean_makespan_s=100,
                checkpoint_path=str(tmp_path / f"{spec.run_id}-{fraction}.pt"),
                checkpoint_sha256="c" * 64,
                checkpoint_lineage=[f"{spec.run_id}-previous"],
                configuration_digest="a" * 64,
                source_commit="b" * 40,
                resume_provenance={"run_id": spec.run_id, "attempt": 2, "checkpoint": "prior.pt"},
            )
            episodes = [
                item.model_copy(
                    update={
                        **result.model_dump(include=_CHECKPOINT_FIELDS),
                        "policy_id": result.checkpoint_id,
                    }
                )
                for item in episodes
            ]
            candidates.append(CandidateEvaluation(result=result, episodes=episodes))
        selected = select_validation_checkpoint(
            [item.result for item in candidates],
            validation_seeds=protocol.selection.scenario_seeds,
            required_fractions=protocol.checkpoint_fractions,
        )
        registry.freeze_policy_selection(matrix_id, spec.run_id, selected.model_dump(mode="json"))
        selections.append(
            RunSelectionEvidence(
                matrix_id=matrix_id,
                run_key=spec.run_id,
                candidates=candidates,
                selected=selected,
                evidence_path=selection_path.parent / spec.run_id / "validation_selection.json",
            )
        )
    selection_evidence = MatrixSelectionEvidence(
        matrix_id=matrix_id,
        protocol_digest=protocol_digest(protocol),
        selections=selections,
        evidence_path=selection_path,
    )
    _write(selection_path, selection_evidence.model_dump(mode="json"))
    policy_grid = [EvaluationGridEntry(controller=name) for name in BASELINE_CONTROLLERS]
    episodes = [
        episode(name, seed, failure)
        for name in BASELINE_CONTROLLERS
        for failure in protocol.evaluation.failure_modes
        for seed in protocol.evaluation.scenario_seeds
    ]
    for spec in specs:
        policy_grid.append(
            EvaluationGridEntry(
                controller=spec.algorithm,
                experiment_id=protocol.experiment_id,
                experiment_variant=spec.experiment_variant,
                training_seed=spec.training_seed,
            )
        )
        selected = next(item.selected for item in selections if item.run_key == spec.run_id)
        episodes.extend(
            episode(
                spec.algorithm,
                seed,
                failure,
                variant=spec.experiment_variant,
                training_seed=spec.training_seed,
            ).model_copy(
                update={
                    **selected.model_dump(include=_CHECKPOINT_FIELDS),
                    "policy_id": f"{matrix_id}-{spec.run_id}-selected",
                }
            )
            for failure in protocol.evaluation.failure_modes
            for seed in protocol.evaluation.scenario_seeds
        )
    aggregate_input = json.dumps(
        [
            item.model_dump(
                mode="json",
                exclude=_CHECKPOINT_FIELDS
                - {
                    "experiment_id",
                    "experiment_variant",
                    "training_seed",
                }
                | {"policy_id"},
            )
            for item in episodes
        ],
        sort_keys=True,
    )
    summaries, training_summaries, scenario_summaries = _offline_aggregate_tables(aggregate_input)
    suite = EvaluationSuite(
        evaluation_id="offline-cli-heldout",
        seeds=protocol.evaluation.scenario_seeds,
        controllers=protocol.evaluation.controllers,
        episodes=episodes,
        summaries=summaries,
        expected_split="test",
        grid_validation=validate_evaluation_grid(
            episodes,
            scenario_seeds=protocol.evaluation.scenario_seeds,
            policy_grid=policy_grid,
            failure_modes=protocol.evaluation.failure_modes,
            expected_split="test",
        ),
        per_training_seed=training_summaries,
        per_scenario_seed=scenario_summaries,
    )
    root = tmp_path / "experiments" / "heldout" / suite.evaluation_id
    payload = suite.model_dump(mode="json")
    _write(root / "evaluation.json", payload)
    acceptance = audit_primary_acceptance(suite, protocol).model_dump(mode="json")
    ablations = [
        item.model_dump(mode="json")
        for item in audit_ablation_hypotheses(
            suite,
            protocol,
            selection_evidence=selection_evidence,
        )
    ]
    _write(root / "acceptance.json", acceptance)
    _write(root / "ablations.json", ablations)
    registry.upsert_evaluation(
        suite.evaluation_id,
        matrix_id=matrix_id,
        split="test",
        payload=payload,
        artifact_dir=str(root),
    )
    return registry, root, selection_path, payload, acceptance, ablations


def test_completed_cli_matrix_exposes_verified_results_without_job_events_or_rollouts(
    cli_results, monkeypatch
) -> None:
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
def test_research_profile_cli_artifacts_complete_the_local_view_without_a_public_claim(
    cli_results,
) -> None:
    registry, _, _, _, acceptance, ablations = cli_results
    summary = build_local_research_summary(registry)
    assert summary["status"] == "local_complete"
    assert summary["claim_allowed"] is False
    assert summary["training_run_count"] == summary["selection_count"] == 20
    assert summary["heldout_episode_count"] == 240
    assert summary["acceptance"] == acceptance
    assert summary["ablations"] == ablations
    assert all(run["kind"] == "training" for run in registry.list_runs())


def test_cli_summary_caches_only_deterministic_aggregation_and_rechecks_provenance(
    cli_results, monkeypatch
) -> None:
    from embodied_skill_composer.construction import evaluation, workbench_evidence

    registry, root, _, payload, _, _ = cli_results
    workbench_evidence._cli_expected_aggregates.cache_clear()
    calls = []
    original = evaluation.summarize_evaluations

    def summarize(episodes):
        calls.append(len(episodes))
        return original(episodes)

    monkeypatch.setattr(evaluation, "summarize_evaluations", summarize)
    started = perf_counter()
    assert build_local_research_summary(registry)["acceptance"] is not None
    cold_seconds = perf_counter() - started
    started = perf_counter()
    assert build_local_research_summary(registry)["acceptance"] is not None
    warm_seconds = perf_counter() - started
    assert calls == [240]
    assert workbench_evidence._cli_expected_aggregates.cache_info().maxsize == 8
    assert workbench_evidence._cli_expected_aggregates.cache_info().hits == 1
    print(f"CLI summary aggregation cold={cold_seconds:.3f}s warm={warm_seconds:.3f}s")

    learned = next(item for item in payload["episodes"] if item["controller"] == "mappo")
    learned["checkpoint_sha256"] = "d" * 64
    _write(root / "evaluation.json", payload)
    registry.upsert_evaluation(
        "offline-cli-heldout",
        matrix_id="cli-unit-fixture",
        split="test",
        payload=payload,
        artifact_dir=str(root),
    )
    assert build_local_research_summary(registry)["acceptance"] is None
    assert calls == [240]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_acceptance",
        "invalid_acceptance",
        "changed_acceptance",
        "missing_ablations",
        "changed_ablations",
        "wrong_evaluation_id",
        "disk_payload_differs",
        "registry_split",
        "episode_split",
        "missing_episode",
        "missing_selections",
        "changed_selection",
        "candidate_split",
        "candidate_controller",
        "candidate_source",
        "missing_fraction",
        "duplicate_selection",
        "wrong_protocol_digest",
        "linked_acceptance",
        "summary_mean",
        "summary_interval",
        "summary_count",
        "summary_missing_metric",
        "training_summary",
        "scenario_summary",
        "missing_training_summary",
        "missing_scenario_summary",
        "candidate_completion_mean",
        "candidate_makespan_mean",
        "candidate_selected_ranking",
        "registered_run_identity",
        "registered_run_config_digest",
        "registered_run_source",
        *[f"heldout_{field}" for field in _PROVENANCE_MUTATIONS],
        *[f"candidate_episode_{field}" for field in _PROVENANCE_MUTATIONS],
        *[f"baseline_{field}" for field in _PROVENANCE_MUTATIONS],
    ],
)
def test_cli_results_fail_closed_on_missing_changed_or_cross_split_artifacts(
    cli_results, monkeypatch, fault: str
) -> None:
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
        registry.upsert_evaluation(
            "offline-cli-heldout",
            matrix_id="cli-unit-fixture",
            split="validation" if fault == "registry_split" else "test",
            payload=payload,
            artifact_dir=str(root),
        )
    elif fault == "missing_selections":
        selection_path.unlink()
    elif fault.startswith("heldout_") or fault.startswith("baseline_"):
        prefix = "heldout_" if fault.startswith("heldout_") else "baseline_"
        item = next(
            item
            for item in payload["episodes"]
            if (
                item["controller"] == "mappo"
                if prefix == "heldout_"
                else item["controller"] == "sequential"
            )
        )
        field = fault.removeprefix(prefix)
        item[field] = _PROVENANCE_MUTATIONS[field]
        _write(root / "evaluation.json", payload)
        registry.upsert_evaluation(
            "offline-cli-heldout",
            matrix_id="cli-unit-fixture",
            split="test",
            payload=payload,
            artifact_dir=str(root),
        )
    elif fault.startswith("summary_") or fault in {
        "training_summary",
        "scenario_summary",
        "missing_training_summary",
        "missing_scenario_summary",
    }:
        if fault == "summary_mean":
            payload["summaries"][0]["metrics"]["structure_completion_rate"]["mean"] = 0.5
        elif fault == "summary_interval":
            payload["summaries"][0]["metrics"]["makespan_s"]["bootstrap_ci95_low"] = 50
        elif fault == "summary_count":
            payload["summaries"][0]["episode_count"] += 1
        elif fault == "summary_missing_metric":
            del payload["summaries"][0]["metrics"]["total_travel_m"]
        elif fault == "training_summary":
            payload["per_training_seed"][0]["metrics"]["makespan_s"]["mean"] = 50
        elif fault == "scenario_summary":
            payload["per_scenario_seed"][0]["metrics"]["makespan_s"]["mean"] = 50
        elif fault == "missing_training_summary":
            payload["per_training_seed"].pop()
        else:
            payload["per_scenario_seed"].pop()
        _write(root / "evaluation.json", payload)
        registry.upsert_evaluation(
            "offline-cli-heldout",
            matrix_id="cli-unit-fixture",
            split="test",
            payload=payload,
            artifact_dir=str(root),
        )
    elif fault.startswith("candidate_episode_") or fault in {
        "candidate_completion_mean",
        "candidate_makespan_mean",
        "candidate_selected_ranking",
    }:
        selections = json.loads(selection_path.read_text(encoding="utf-8"))
        candidate = selections["selections"][0]["candidates"][0]
        if fault.startswith("candidate_episode_"):
            field = fault.removeprefix("candidate_episode_")
            candidate["episodes"][0][field] = _PROVENANCE_MUTATIONS[field]
        elif fault == "candidate_completion_mean":
            candidate["result"]["mean_completion_rate"] = 0.4
        elif fault == "candidate_makespan_mean":
            candidate["result"]["mean_makespan_s"] = 101
        else:
            # Disk and frozen DB agree on a valid-looking but false candidate ranking.
            selected = selections["selections"][0]["selected"]
            selected["candidate_ranking"] = list(reversed(selected["candidate_ranking"]))
            original = registry.list_policy_selections

            def frozen_selections(matrix_id):
                result = original(matrix_id)
                matching = next(
                    item
                    for item in result
                    if item["run_key"] == selections["selections"][0]["run_key"]
                )
                matching["selection"] = selected
                return result

            monkeypatch.setattr(registry, "list_policy_selections", frozen_selections)
        _write(selection_path, selections)
    elif fault in {
        "changed_selection",
        "candidate_split",
        "candidate_controller",
        "candidate_source",
        "missing_fraction",
        "duplicate_selection",
    }:
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
    elif fault == "wrong_protocol_digest" or fault.startswith("registered_run_"):
        original = registry.list_experiment_matrices

        def matrices():
            result = original()
            if fault == "wrong_protocol_digest":
                result[0]["protocol_digest"] = "f" * 64
            elif fault == "registered_run_identity":
                result[0]["runs"][0]["config"]["training_seed"] = 11
            elif fault == "registered_run_config_digest":
                result[0]["runs"][0]["config_digest"] = "f" * 64
            else:
                result[0]["runs"][0]["source_commit"] = "f" * 40
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
