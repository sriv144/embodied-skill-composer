from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal, TypeAlias

import numpy as np
from pydantic import BaseModel, Field

from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.marl_env_v1 import (
    TemporalConstructionCoordinationEnv,
    auction_temporal_actions,
    scripted_temporal_actions,
)
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.policy import TorchRLPolicyBundle, policy_actions
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
)
from embodied_skill_composer.construction.scheduler import schedule_build
from embodied_skill_composer.construction.training import cp_sat_expert_actions


ControllerName = Literal["sequential", "greedy", "auction", "ippo", "mappo", "cp_sat"]
EvaluationGroupKey: TypeAlias = tuple[ControllerName, str | None, str | None, bool]

_METRIC_FIELDS = (
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
_PRIMARY_VARIANTS: dict[ControllerName, str] = {
    "mappo": "mappo_full",
    "ippo": "ippo_full",
}


class EpisodeEvaluation(BaseModel):
    scenario_id: str
    seed: int
    split: str
    controller: ControllerName
    failure_enabled: bool
    structure_completion_rate: float = Field(ge=0, le=1)
    makespan_s: float = Field(ge=0)
    total_travel_m: float = Field(ge=0)
    total_energy_wh: float = Field(ge=0)
    idle_robot_seconds: float = Field(ge=0)
    mean_robot_utilization: float = Field(ge=0, le=1)
    collision_count: int = Field(ge=0)
    wasted_work_s: float = Field(ge=0)
    invalid_bid_count: int = Field(ge=0)
    drop_count: int = Field(ge=0)
    decision_count: int = Field(ge=0)
    routing_backend: str | None = None
    policy_id: str | None = None
    experiment_id: str | None = None
    experiment_variant: str | None = None
    training_seed: int | None = None
    transition_count: int | None = Field(default=None, ge=0)
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_lineage: list[str] = Field(default_factory=list)
    configuration_digest: str | None = None
    source_commit: str | None = None
    resume_provenance: dict[str, object] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    mean: float
    std: float
    bootstrap_ci95_low: float
    bootstrap_ci95_high: float
    median: float


class ControllerEvaluation(BaseModel):
    controller: ControllerName
    failure_enabled: bool
    episode_count: int
    metrics: dict[str, MetricSummary]
    experiment_id: str | None = None
    experiment_variant: str | None = None
    training_seed_count: int = Field(default=0, ge=0)
    scenario_seed_count: int = Field(default=0, ge=0)


class TrainingSeedEvaluation(BaseModel):
    controller: ControllerName
    failure_enabled: bool
    training_seed: int | None
    episode_count: int
    metrics: dict[str, MetricSummary]
    experiment_id: str | None = None
    experiment_variant: str | None = None


class ScenarioSeedEvaluation(BaseModel):
    controller: ControllerName
    failure_enabled: bool
    scenario_seed: int
    episode_count: int
    metrics: dict[str, MetricSummary]
    experiment_id: str | None = None
    experiment_variant: str | None = None


class EvaluationGridEntry(BaseModel):
    controller: ControllerName
    experiment_id: str | None = None
    experiment_variant: str | None = None
    training_seed: int | None = None


class EvaluationGridValidation(BaseModel):
    complete: Literal[True] = True
    expected_episode_count: int = Field(ge=0)
    observed_episode_count: int = Field(ge=0)
    expected_split: str | None = None


class EvaluationSuite(BaseModel):
    evaluation_id: str
    seeds: list[int]
    controllers: list[ControllerName]
    episodes: list[EpisodeEvaluation]
    summaries: list[ControllerEvaluation]
    expected_split: str | None = None
    grid_validation: EvaluationGridValidation | None = None
    per_training_seed: list[TrainingSeedEvaluation] = Field(default_factory=list)
    per_scenario_seed: list[ScenarioSeedEvaluation] = Field(default_factory=list)


class EvaluationArtifacts(BaseModel):
    run_dir: Path
    evaluation_json: Path
    episodes_csv: Path
    report_path: Path


class TemporalEpisodeMetrics(BaseModel):
    structure_completion_rate: float
    makespan_s: float
    total_travel_m: float
    total_energy_wh: float
    idle_robot_seconds: float
    robot_utilization: dict[str, float]
    collision_count: int
    wasted_work_s: float
    invalid_bid_count: int
    drop_count: int
    routing_backend: str | None = None


def evaluate_controller_episode(
    base_design: HouseDesign,
    *,
    seed: int,
    controller: ControllerName,
    bundle: TorchRLPolicyBundle | None = None,
    policy_manifest: PolicyManifest | None = None,
    failure_enabled: bool = False,
    device: str = "cpu",
    expected_split: str | None = None,
) -> EpisodeEvaluation:
    if policy_manifest is not None and policy_manifest.controller != controller:
        raise ValueError(
            "Policy manifest controller does not match evaluation controller: "
            f"{policy_manifest.controller!r} != {controller!r}."
        )
    scenario = generate_cottage_scenario(
        seed,
        base_design,
        config=CottageScenarioConfig(
            include_failures=failure_enabled,
            failure_probability=1.0 if failure_enabled else 0.0,
            obstacle_count_range=(0, 4),
        ),
    )
    scenario_split = scenario.split.value
    if expected_split is not None and scenario_split != expected_split:
        raise ValueError(
            f"Scenario seed {seed} belongs to split {scenario_split!r}; "
            f"expected {expected_split!r}."
        )
    env = TemporalConstructionCoordinationEnv(scenario)
    observations, _ = env.reset(seed=seed)
    priority = None
    if controller == "cp_sat":
        schedule = schedule_build(scenario.plan, "optimized")
        priority = {
            job.module_id: (job.start_s, job.end_s, tuple(job.robot_ids)) for job in schedule.jobs
        }
    while env.agents:
        diagnostics = None
        if controller == "sequential":
            actions = sequential_temporal_actions(env)
        elif controller == "greedy":
            actions = scripted_temporal_actions(env)
        elif controller == "auction":
            actions = auction_temporal_actions(env)
        elif controller == "cp_sat":
            assert priority is not None
            actions = cp_sat_expert_actions(env, priority)
        else:
            if bundle is None:
                raise ValueError(f"{controller} evaluation requires a policy bundle")
            actions, diagnostics = policy_actions(
                bundle.actor_model,
                observations,
                env.possible_agents,
                device=device,
                deterministic=True,
            )
        observations, _, _, _, _ = env.step(actions)
        if diagnostics:
            env.annotate_latest_decisions(controller, diagnostics)
    metrics = TemporalEpisodeMetrics.model_validate(env.metrics())
    return EpisodeEvaluation(
        scenario_id=scenario.scenario_id,
        seed=seed,
        split=scenario_split,
        controller=controller,
        failure_enabled=failure_enabled,
        structure_completion_rate=metrics.structure_completion_rate,
        makespan_s=metrics.makespan_s,
        total_travel_m=metrics.total_travel_m,
        total_energy_wh=metrics.total_energy_wh,
        idle_robot_seconds=metrics.idle_robot_seconds,
        mean_robot_utilization=float(np.mean(list(metrics.robot_utilization.values()))),
        collision_count=metrics.collision_count,
        wasted_work_s=metrics.wasted_work_s,
        invalid_bid_count=metrics.invalid_bid_count,
        drop_count=metrics.drop_count,
        decision_count=env.decision_count,
        routing_backend=metrics.routing_backend,
        policy_id=policy_manifest.policy_id if policy_manifest is not None else None,
        experiment_id=(
            policy_manifest.experiment_id if policy_manifest is not None else None
        ),
        experiment_variant=(
            policy_manifest.experiment_variant if policy_manifest is not None else None
        ),
        training_seed=(
            policy_manifest.training_seed if policy_manifest is not None else None
        ),
        transition_count=(
            policy_manifest.transition_count if policy_manifest is not None else None
        ),
        checkpoint_path=(
            policy_manifest.checkpoint_path if policy_manifest is not None else None
        ),
        checkpoint_sha256=(
            policy_manifest.checkpoint_sha256 if policy_manifest is not None else None
        ),
        checkpoint_lineage=(
            policy_manifest.checkpoint_lineage if policy_manifest is not None else []
        ),
        configuration_digest=(
            policy_manifest.configuration_digest if policy_manifest is not None else None
        ),
        source_commit=(
            policy_manifest.source_commit if policy_manifest is not None else None
        ),
        resume_provenance=(
            policy_manifest.resume_provenance if policy_manifest is not None else {}
        ),
    )


def run_evaluation_suite(
    base_design: HouseDesign,
    *,
    seeds: list[int],
    controllers: list[ControllerName],
    policies: dict[str, TorchRLPolicyBundle] | None = None,
    policy_manifests: dict[str, PolicyManifest] | None = None,
    include_failure_suite: bool = True,
    device: str = "cpu",
    evaluation_id: str | None = None,
    expected_split: str | None = None,
) -> EvaluationSuite:
    policies = policies or {}
    policy_manifests = policy_manifests or {}
    episodes = []
    failure_modes = [False, True] if include_failure_suite else [False]
    for failure_enabled in failure_modes:
        for controller in controllers:
            for seed in seeds:
                optional_args: dict[str, Any] = {}
                if controller in policy_manifests:
                    optional_args["policy_manifest"] = policy_manifests[controller]
                if expected_split is not None:
                    optional_args["expected_split"] = expected_split
                episodes.append(
                    evaluate_controller_episode(
                        base_design,
                        seed=seed,
                        controller=controller,
                        bundle=policies.get(controller),
                        failure_enabled=failure_enabled,
                        device=device,
                        **optional_args,
                    )
                )
    summaries = summarize_evaluations(episodes)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    resolved_evaluation_id = evaluation_id or f"{timestamp}-heldout-{len(seeds)}seed"
    policy_grid = [
        EvaluationGridEntry(
            controller=controller,
            experiment_id=manifest.experiment_id if manifest is not None else None,
            experiment_variant=(
                manifest.experiment_variant if manifest is not None else None
            ),
            training_seed=manifest.training_seed if manifest is not None else None,
        )
        for controller in controllers
        for manifest in [policy_manifests.get(controller)]
    ]
    grid_validation = validate_evaluation_grid(
        episodes,
        scenario_seeds=seeds,
        policy_grid=policy_grid,
        failure_modes=failure_modes,
        expected_split=expected_split,
    )
    return EvaluationSuite(
        evaluation_id=resolved_evaluation_id,
        seeds=seeds,
        controllers=controllers,
        episodes=episodes,
        summaries=summaries,
        expected_split=expected_split,
        grid_validation=grid_validation,
        per_training_seed=summarize_by_training_seed(episodes),
        per_scenario_seed=summarize_by_scenario_seed(episodes),
    )


def summarize_evaluations(
    episodes: list[EpisodeEvaluation],
) -> list[ControllerEvaluation]:
    summaries: list[ControllerEvaluation] = []
    for group_key, subset in _evaluation_groups(episodes):
        controller, experiment_id, experiment_variant, failure_enabled = group_key
        metrics = {
            field: _hierarchical_metric_summary(subset, field=field, seed=2027)
            for field in _METRIC_FIELDS
        }
        summaries.append(
            ControllerEvaluation(
                controller=controller,
                failure_enabled=failure_enabled,
                episode_count=len(subset),
                metrics=metrics,
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
                training_seed_count=len(
                    {
                        item.training_seed
                        for item in subset
                        if item.training_seed is not None
                    }
                ),
                scenario_seed_count=len({item.seed for item in subset}),
            )
        )
    return summaries


def summarize_by_training_seed(
    episodes: list[EpisodeEvaluation],
) -> list[TrainingSeedEvaluation]:
    groups: dict[
        tuple[ControllerName, str | None, str | None, bool, int | None],
        list[EpisodeEvaluation],
    ] = {}
    for item in episodes:
        key = (
            item.controller,
            item.experiment_id,
            item.experiment_variant,
            item.failure_enabled,
            item.training_seed,
        )
        groups.setdefault(key, []).append(item)
    summaries = []
    for key in sorted(groups, key=_sortable_group_key):
        controller, experiment_id, experiment_variant, failure_enabled, training_seed = key
        subset = groups[key]
        summaries.append(
            TrainingSeedEvaluation(
                controller=controller,
                failure_enabled=failure_enabled,
                training_seed=training_seed,
                episode_count=len(subset),
                metrics=_flat_metric_summaries(subset),
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
            )
        )
    return summaries


def summarize_by_scenario_seed(
    episodes: list[EpisodeEvaluation],
) -> list[ScenarioSeedEvaluation]:
    groups: dict[
        tuple[ControllerName, str | None, str | None, bool, int],
        list[EpisodeEvaluation],
    ] = {}
    for item in episodes:
        key = (
            item.controller,
            item.experiment_id,
            item.experiment_variant,
            item.failure_enabled,
            item.seed,
        )
        groups.setdefault(key, []).append(item)
    summaries = []
    for key in sorted(groups, key=_sortable_group_key):
        controller, experiment_id, experiment_variant, failure_enabled, scenario_seed = key
        subset = groups[key]
        summaries.append(
            ScenarioSeedEvaluation(
                controller=controller,
                failure_enabled=failure_enabled,
                scenario_seed=scenario_seed,
                episode_count=len(subset),
                metrics=_flat_metric_summaries(subset),
                experiment_id=experiment_id,
                experiment_variant=experiment_variant,
            )
        )
    return summaries


def validate_evaluation_grid(
    episodes: list[EpisodeEvaluation],
    *,
    scenario_seeds: list[int],
    policy_grid: list[EvaluationGridEntry],
    failure_modes: list[bool],
    expected_split: str | None,
) -> EvaluationGridValidation:
    if not scenario_seeds:
        raise ValueError("Evaluation scenario seed grid cannot be empty.")
    if not policy_grid:
        raise ValueError("Evaluation policy grid cannot be empty.")
    if not failure_modes:
        raise ValueError("Evaluation failure mode grid cannot be empty.")
    if len(scenario_seeds) != len(set(scenario_seeds)):
        raise ValueError("Evaluation scenario seeds must be unique.")
    policy_keys = [_grid_entry_key(item) for item in policy_grid]
    if len(policy_keys) != len(set(policy_keys)):
        raise ValueError("Evaluation policy grid entries must be unique.")
    if len(failure_modes) != len(set(failure_modes)):
        raise ValueError("Evaluation failure modes must be unique.")

    expected = {
        (*policy_key, scenario_seed, failure_enabled)
        for policy_key in policy_keys
        for scenario_seed in scenario_seeds
        for failure_enabled in failure_modes
    }
    observed_keys = [
        (
            item.controller,
            item.experiment_id,
            item.experiment_variant,
            item.training_seed,
            item.seed,
            item.failure_enabled,
        )
        for item in episodes
    ]
    observed = set(observed_keys)
    duplicates = sorted(
        {key for key, count in Counter(observed_keys).items() if count > 1},
        key=_sortable_group_key,
    )
    missing = sorted(expected - observed, key=_sortable_group_key)
    unexpected = sorted(observed - expected, key=_sortable_group_key)
    split_mismatches = (
        sorted(
            {
                (item.scenario_id, item.seed, item.split)
                for item in episodes
                if item.split != expected_split
            }
        )
        if expected_split is not None
        else []
    )
    if duplicates or missing or unexpected or split_mismatches:
        problems = []
        if duplicates:
            problems.append(f"duplicate cells={duplicates!r}")
        if missing:
            problems.append(f"missing cells={missing!r}")
        if unexpected:
            problems.append(f"unexpected cells={unexpected!r}")
        if split_mismatches:
            problems.append(f"split mismatches={split_mismatches!r}")
        raise ValueError("Incomplete evaluation grid: " + "; ".join(problems))
    return EvaluationGridValidation(
        expected_episode_count=len(expected),
        observed_episode_count=len(episodes),
        expected_split=expected_split,
    )


def write_evaluation_artifacts(
    suite: EvaluationSuite,
    output_root: Path,
) -> EvaluationArtifacts:
    run_dir = output_root.resolve() / suite.evaluation_id
    run_dir.mkdir(parents=True, exist_ok=False)
    evaluation_json = run_dir / "evaluation.json"
    evaluation_json.write_text(suite.model_dump_json(indent=2), encoding="utf-8")
    episodes_csv = run_dir / "episodes.csv"
    rows = [item.model_dump(mode="json") for item in suite.episodes]
    with episodes_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = run_dir / "report.md"
    report_path.write_text(render_evaluation_report(suite), encoding="utf-8")
    return EvaluationArtifacts(
        run_dir=run_dir,
        evaluation_json=evaluation_json,
        episodes_csv=episodes_csv,
        report_path=report_path,
    )


def render_evaluation_report(suite: EvaluationSuite) -> str:
    split_text = (
        f" The expected split is `{suite.expected_split}`."
        if suite.expected_split is not None
        else ""
    )
    lines = [
        "# Construction Intelligence Evaluation",
        "",
        f"Evaluation `{suite.evaluation_id}` covers scenario seeds "
        f"{', '.join(map(str, suite.seeds))}.{split_text}",
        "Confidence intervals are deterministic 2,000-replicate hierarchical bootstrap "
        "intervals: training seeds are resampled first, then scenario seeds within each "
        "sampled training seed.",
        "",
        "| Controller | Failures | Completion | Makespan (s) | Travel (m) | Utilization |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in suite.summaries:
        completion = summary.metrics["structure_completion_rate"]
        makespan = summary.metrics["makespan_s"]
        travel = summary.metrics["total_travel_m"]
        utilization = summary.metrics["mean_robot_utilization"]
        controller_label = _summary_label(summary)
        lines.append(
            f"| {controller_label} | {'yes' if summary.failure_enabled else 'no'} "
            f"| {completion.mean:.3f} [{completion.bootstrap_ci95_low:.3f}, "
            f"{completion.bootstrap_ci95_high:.3f}] | {makespan.mean:.1f} "
            f"| {travel.mean:.1f} | {utilization.mean:.3f} |"
        )
    if suite.grid_validation is not None:
        lines.extend(
            [
                "",
                "## Evaluation Grid",
                "",
                f"Complete: **yes** "
                f"({suite.grid_validation.observed_episode_count}/"
                f"{suite.grid_validation.expected_episode_count} expected episodes).",
            ]
        )
    if suite.per_training_seed:
        lines.extend(["", "## Per-training-seed Results", ""])
        lines.extend(_render_training_seed_rows(suite.per_training_seed))
    if suite.per_scenario_seed:
        lines.extend(["", "## Per-scenario-seed Results", ""])
        lines.extend(_render_scenario_seed_rows(suite.per_scenario_seed))
    lines.extend(["", "## Acceptance Audit", ""])
    lines.extend(_acceptance_audit(suite))
    lines.extend(
        [
            "",
            "## Fidelity Boundary",
            "",
            "These results are from `construction_coordination_v1`, the event simulator. "
            "They do not by themselves prove dynamic CoppeliaSim execution or physical grasping.",
        ]
    )
    return "\n".join(lines) + "\n"


def sequential_temporal_actions(
    env: TemporalConstructionCoordinationEnv,
) -> dict[str, int]:
    actions = {agent: 0 for agent in env.agents}
    available = [agent for agent in env.agents if env.robot_runtime[agent].status == "idle"]
    for module in sorted(env.ready_modules(), key=lambda item: item.module_id):
        team = env._select_capable_team(module, available)
        if team is None:
            continue
        action = env.module_index[module.module_id] + 1
        for agent in team:
            actions[agent] = action
        break
    return actions


def _evaluation_groups(
    episodes: list[EpisodeEvaluation],
) -> list[tuple[EvaluationGroupKey, list[EpisodeEvaluation]]]:
    groups: dict[EvaluationGroupKey, list[EpisodeEvaluation]] = {}
    for item in episodes:
        key = (
            item.controller,
            item.experiment_id,
            item.experiment_variant,
            item.failure_enabled,
        )
        groups.setdefault(key, []).append(item)
    return sorted(
        groups.items(),
        key=lambda item: (
            item[0][3],
            item[0][0],
            item[0][1] or "",
            item[0][2] or "",
        ),
    )


def _flat_metric_summaries(
    episodes: list[EpisodeEvaluation],
) -> dict[str, MetricSummary]:
    return {
        field: _metric_summary(
            np.array([float(getattr(item, field)) for item in episodes]),
            seed=2027,
        )
        for field in _METRIC_FIELDS
    }


def _hierarchical_metric_summary(
    episodes: list[EpisodeEvaluation],
    *,
    field: str,
    seed: int,
) -> MetricSummary:
    if not episodes:
        raise ValueError("Cannot summarize an empty evaluation group.")
    values = np.array([float(getattr(item, field)) for item in episodes])
    training_groups: dict[int | None, dict[int, list[float]]] = {}
    for item in episodes:
        scenario_groups = training_groups.setdefault(item.training_seed, {})
        scenario_groups.setdefault(item.seed, []).append(float(getattr(item, field)))
    training_seeds = sorted(training_groups, key=lambda value: (-1 if value is None else value))
    scenario_grids = {
        tuple(sorted(scenarios))
        for scenarios in training_groups.values()
    }
    if len(scenario_grids) != 1:
        raise ValueError(
            "Hierarchical aggregation requires the same scenario grid for "
            "every training seed."
        )
    scenario_seeds = list(next(iter(scenario_grids)))
    training_means = [
        float(
            np.mean(
                [
                    np.mean(training_groups[training_seed][scenario_seed])
                    for scenario_seed in sorted(training_groups[training_seed])
                ]
            )
        )
        for training_seed in training_seeds
    ]
    point_mean = float(np.mean(training_means))
    if len(episodes) == 1:
        low = high = point_mean
        std = 0.0
    else:
        rng = np.random.default_rng(seed)
        bootstrap = np.empty(2000, dtype=np.float64)
        for bootstrap_index in range(bootstrap.size):
            sampled_training_indices = rng.integers(
                0,
                len(training_seeds),
                size=len(training_seeds),
            )
            sampled_scenario_indices = rng.integers(
                0,
                len(scenario_seeds),
                size=len(scenario_seeds),
            )
            sampled_values: list[float] = []
            for training_index in sampled_training_indices:
                training_seed = training_seeds[int(training_index)]
                scenarios = training_groups[training_seed]
                sampled_values.extend(
                    float(
                        np.mean(
                            scenarios[
                                scenario_seeds[int(scenario_index)]
                            ]
                        )
                    )
                    for scenario_index in sampled_scenario_indices
                )
            bootstrap[bootstrap_index] = np.mean(sampled_values)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        std = float(values.std(ddof=1))
    return MetricSummary(
        mean=point_mean,
        std=std,
        bootstrap_ci95_low=float(low),
        bootstrap_ci95_high=float(high),
        median=float(median(values.tolist())),
    )


def _grid_entry_key(
    item: EvaluationGridEntry,
) -> tuple[ControllerName, str | None, str | None, int | None]:
    return (
        item.controller,
        item.experiment_id,
        item.experiment_variant,
        item.training_seed,
    )


def _sortable_group_key(values: tuple[object, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in values)


def _summary_label(summary: ControllerEvaluation) -> str:
    if summary.experiment_variant is None:
        return summary.controller
    return f"{summary.controller} ({summary.experiment_variant})"


def _render_training_seed_rows(
    summaries: list[TrainingSeedEvaluation],
) -> list[str]:
    lines = [
        "| Controller | Failures | Training seed | Completion | Makespan (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        label = (
            summary.controller
            if summary.experiment_variant is None
            else f"{summary.controller} ({summary.experiment_variant})"
        )
        training_seed = "n/a" if summary.training_seed is None else str(summary.training_seed)
        lines.append(
            f"| {label} | {'yes' if summary.failure_enabled else 'no'} "
            f"| {training_seed} "
            f"| {summary.metrics['structure_completion_rate'].mean:.3f} "
            f"| {summary.metrics['makespan_s'].mean:.1f} |"
        )
    return lines


def _render_scenario_seed_rows(
    summaries: list[ScenarioSeedEvaluation],
) -> list[str]:
    lines = [
        "| Controller | Failures | Scenario seed | Completion | Makespan (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        label = (
            summary.controller
            if summary.experiment_variant is None
            else f"{summary.controller} ({summary.experiment_variant})"
        )
        lines.append(
            f"| {label} | {'yes' if summary.failure_enabled else 'no'} "
            f"| {summary.scenario_seed} "
            f"| {summary.metrics['structure_completion_rate'].mean:.3f} "
            f"| {summary.metrics['makespan_s'].mean:.1f} |"
        )
    return lines


def _metric_summary(values: np.ndarray, *, seed: int) -> MetricSummary:
    rng = np.random.default_rng(seed)
    if values.size == 1:
        low = high = float(values[0])
        std = 0.0
    else:
        bootstrap = rng.choice(values, size=(2000, values.size), replace=True).mean(axis=1)
        low, high = np.quantile(bootstrap, [0.025, 0.975])
        std = float(values.std(ddof=1))
    return MetricSummary(
        mean=float(values.mean()),
        std=std,
        bootstrap_ci95_low=float(low),
        bootstrap_ci95_high=float(high),
        median=float(median(values.tolist())),
    )


def _acceptance_audit(suite: EvaluationSuite) -> list[str]:
    lines = []
    for controller in ("mappo", "ippo"):
        no_failure = _primary_summary(suite.summaries, controller, failure_enabled=False)
        if no_failure:
            completion = no_failure.metrics["structure_completion_rate"].mean
            lines.append(
                f"- `{controller}` no-failure completion >= 0.95: "
                f"{'PASS' if completion >= 0.95 else 'NOT YET'} ({completion:.3f})."
            )
    mappo = _primary_summary(suite.summaries, "mappo", failure_enabled=False)
    cp_sat = _primary_summary(suite.summaries, "cp_sat", failure_enabled=False)
    if mappo and cp_sat:
        ratio = mappo.metrics["makespan_s"].median / max(
            cp_sat.metrics["makespan_s"].median,
            1e-9,
        )
        lines.append(
            "- MAPPO median makespan within 15% of CP-SAT: "
            f"{'PASS' if ratio <= 1.15 else 'NOT YET'} ({ratio:.3f}x)."
        )
    failure = _primary_summary(suite.summaries, "mappo", failure_enabled=True)
    if failure:
        completion = failure.metrics["structure_completion_rate"].mean
        lines.append(
            "- MAPPO failure completion >= 0.85: "
            f"{'PASS' if completion >= 0.85 else 'NOT YET'} ({completion:.3f})."
        )
    if not lines:
        lines.append(
            "- Learned-policy acceptance cannot be audited until learned runs are included."
        )
    return lines


def _primary_summary(
    summaries: list[ControllerEvaluation],
    controller: ControllerName,
    *,
    failure_enabled: bool,
) -> ControllerEvaluation | None:
    candidates = [
        item
        for item in summaries
        if item.controller == controller and item.failure_enabled == failure_enabled
    ]
    if not candidates:
        return None
    primary_variant = _PRIMARY_VARIANTS.get(controller)
    for item in candidates:
        if item.experiment_variant == primary_variant:
            return item
    legacy = [item for item in candidates if item.experiment_variant is None]
    if len(legacy) == 1:
        return legacy[0]
    return None
