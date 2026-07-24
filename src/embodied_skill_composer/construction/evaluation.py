from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

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


class EvaluationSuite(BaseModel):
    evaluation_id: str
    seeds: list[int]
    controllers: list[ControllerName]
    episodes: list[EpisodeEvaluation]
    summaries: list[ControllerEvaluation]


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
    failure_enabled: bool = False,
    device: str = "cpu",
) -> EpisodeEvaluation:
    scenario = generate_cottage_scenario(
        seed,
        base_design,
        config=CottageScenarioConfig(
            include_failures=failure_enabled,
            failure_probability=1.0 if failure_enabled else 0.0,
            obstacle_count_range=(0, 4),
        ),
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
        split=scenario.split.value,
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
    )


def run_evaluation_suite(
    base_design: HouseDesign,
    *,
    seeds: list[int],
    controllers: list[ControllerName],
    policies: dict[str, TorchRLPolicyBundle] | None = None,
    include_failure_suite: bool = True,
    device: str = "cpu",
) -> EvaluationSuite:
    policies = policies or {}
    episodes = []
    failure_modes = [False, True] if include_failure_suite else [False]
    for failure_enabled in failure_modes:
        for controller in controllers:
            for seed in seeds:
                episodes.append(
                    evaluate_controller_episode(
                        base_design,
                        seed=seed,
                        controller=controller,
                        bundle=policies.get(controller),
                        failure_enabled=failure_enabled,
                        device=device,
                    )
                )
    summaries = summarize_evaluations(episodes)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return EvaluationSuite(
        evaluation_id=f"{timestamp}-heldout-{len(seeds)}seed",
        seeds=seeds,
        controllers=controllers,
        episodes=episodes,
        summaries=summaries,
    )


def summarize_evaluations(
    episodes: list[EpisodeEvaluation],
) -> list[ControllerEvaluation]:
    summaries = []
    controllers = sorted({item.controller for item in episodes})
    for failure_enabled in sorted({item.failure_enabled for item in episodes}):
        for controller in controllers:
            subset = [
                item
                for item in episodes
                if item.controller == controller and item.failure_enabled == failure_enabled
            ]
            if not subset:
                continue
            metrics = {}
            for field in (
                "structure_completion_rate",
                "makespan_s",
                "total_travel_m",
                "total_energy_wh",
                "idle_robot_seconds",
                "mean_robot_utilization",
                "collision_count",
                "wasted_work_s",
                "invalid_bid_count",
            ):
                values = np.array([float(getattr(item, field)) for item in subset])
                metrics[field] = _metric_summary(values, seed=2027)
            summaries.append(
                ControllerEvaluation(
                    controller=controller,
                    failure_enabled=failure_enabled,
                    episode_count=len(subset),
                    metrics=metrics,
                )
            )
    return summaries


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
    lines = [
        "# Construction Intelligence Evaluation",
        "",
        f"Evaluation `{suite.evaluation_id}` covers seeds {', '.join(map(str, suite.seeds))}.",
        "Confidence intervals are deterministic 2,000-sample bootstrap intervals.",
        "",
        "| Controller | Failures | Completion | Makespan (s) | Travel (m) | Utilization |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in suite.summaries:
        completion = summary.metrics["structure_completion_rate"]
        makespan = summary.metrics["makespan_s"]
        travel = summary.metrics["total_travel_m"]
        utilization = summary.metrics["mean_robot_utilization"]
        lines.append(
            f"| {summary.controller} | {'yes' if summary.failure_enabled else 'no'} "
            f"| {completion.mean:.3f} [{completion.bootstrap_ci95_low:.3f}, "
            f"{completion.bootstrap_ci95_high:.3f}] | {makespan.mean:.1f} "
            f"| {travel.mean:.1f} | {utilization.mean:.3f} |"
        )
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
    lookup = {(item.controller, item.failure_enabled): item for item in suite.summaries}
    lines = []
    for controller in ("mappo", "ippo"):
        no_failure = lookup.get((controller, False))
        if no_failure:
            completion = no_failure.metrics["structure_completion_rate"].mean
            lines.append(
                f"- `{controller}` no-failure completion >= 0.95: "
                f"{'PASS' if completion >= 0.95 else 'NOT YET'} ({completion:.3f})."
            )
    mappo = lookup.get(("mappo", False))
    cp_sat = lookup.get(("cp_sat", False))
    if mappo and cp_sat:
        ratio = mappo.metrics["makespan_s"].median / max(
            cp_sat.metrics["makespan_s"].median,
            1e-9,
        )
        lines.append(
            "- MAPPO median makespan within 15% of CP-SAT: "
            f"{'PASS' if ratio <= 1.15 else 'NOT YET'} ({ratio:.3f}x)."
        )
    failure = lookup.get(("mappo", True))
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
