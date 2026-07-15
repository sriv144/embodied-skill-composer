from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.benchmark import run_assembly_policy_benchmark
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    PolicyBenchmarkResult,
    TrainingConfig,
)
from embodied_skill_composer.assembly.scenario_generation import (
    ScenarioGenerationConfig,
    generate_assembly_scenarios,
    write_generated_scenario,
)


class AssemblyExperimentPolicyResult(BaseModel):
    scenario_id: str
    scenario_path: str
    seed: int
    policy_name: str
    success_rate: float
    mean_return: float
    mean_beams_installed: float
    mean_step_count: float = 0.0
    checkpoint_path: str | None = None
    train_metrics_path: str | None = None
    status: Literal["ok", "failed"] = "ok"
    notes: str = ""


class AssemblyExperimentSweepSummary(BaseModel):
    created_at: str
    runtime_profile: str
    backend: str
    scenario_count: int
    seeds: list[int]
    output_dir: str
    results: list[AssemblyExperimentPolicyResult] = Field(default_factory=list)


def default_experiment_output_dir(root: Path | None = None) -> Path:
    root = root or Path("logs") / "assembly_experiments"
    return root / datetime.now().strftime("%Y%m%d-%H%M%S")


def run_assembly_experiment_sweep(
    base_env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    scenario_count: int = 5,
    seeds: list[int] | None = None,
    output_dir: Path | None = None,
    beam_count: int = 2,
    evaluation_episodes: int | None = None,
) -> AssemblyExperimentSweepSummary:
    seeds = seeds or [7, 8, 9]
    output_dir = output_dir or default_experiment_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = ScenarioGenerationConfig(
        scenario_count=scenario_count,
        base_seed=training_config.seed,
        beam_count=beam_count,
    )
    scenarios = generate_assembly_scenarios(
        base_config=base_env_config,
        generation_config=generation,
        runtime_profile=runtime_profile,
    )
    summary = AssemblyExperimentSweepSummary(
        created_at=datetime.now().isoformat(timespec="seconds"),
        runtime_profile=runtime_profile.name,
        backend=runtime_profile.backend,
        scenario_count=len(scenarios),
        seeds=seeds,
        output_dir=str(output_dir),
    )

    for scenario in scenarios:
        scenario_dir = output_dir / scenario.scenario_id
        scenario_path = write_generated_scenario(scenario, scenario_dir / "scenario.yaml")
        for seed in seeds:
            seed_config = training_config.model_copy(
                update={"seed": seed, "evaluation_episodes": evaluation_episodes or training_config.evaluation_episodes}
            )
            run_dir = scenario_dir / f"seed_{seed}"
            try:
                option_checkpoint, option_metrics = _train_options(
                    scenario.config,
                    seed_config,
                    runtime_profile,
                    run_dir,
                )
                low_level_checkpoint, low_level_metrics = _train_low_level(
                    scenario.config,
                    seed_config,
                    runtime_profile,
                    run_dir,
                )
                benchmark = run_assembly_policy_benchmark(
                    env_config=scenario.config,
                    training_config=seed_config,
                    runtime_profile=runtime_profile,
                    options_checkpoint=option_checkpoint,
                    low_level_checkpoint=low_level_checkpoint,
                    episodes=seed_config.evaluation_episodes,
                )
                summary.results.extend(
                    [
                        _result_from_benchmark(
                            scenario.scenario_id,
                            scenario_path,
                            seed,
                            benchmark.scripted_options,
                        ),
                        _result_from_benchmark(
                            scenario.scenario_id,
                            scenario_path,
                            seed,
                            benchmark.learned_options,
                            checkpoint_path=option_checkpoint,
                            train_metrics_path=option_metrics,
                        ),
                        _result_from_benchmark(
                            scenario.scenario_id,
                            scenario_path,
                            seed,
                            benchmark.low_level_learned,
                            checkpoint_path=low_level_checkpoint,
                            train_metrics_path=low_level_metrics,
                        ),
                    ]
                )
            except Exception as exc:
                summary.results.extend(_failed_results(scenario.scenario_id, scenario_path, seed, exc))

    write_experiment_summary(summary, output_dir)
    return summary


def write_experiment_summary(summary: AssemblyExperimentSweepSummary, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    rows = [result.model_dump(mode="json") for result in summary.results]
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    metadata = summary.model_dump(mode="json", exclude={"results"})
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def _train_options(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    run_dir: Path,
) -> tuple[Path, Path]:
    from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer

    checkpoint_path = run_dir / "assembly_options.pt"
    metrics_path = run_dir / "assembly_option_training_metrics.json"
    env = build_assembly_backend(env_config, runtime_profile, seed=training_config.seed)
    trainer = HierarchicalOptionTrainer(env=env, config=training_config, device=runtime_profile.device)
    trainer.train(checkpoint_path=checkpoint_path, metrics_path=metrics_path)
    return checkpoint_path, metrics_path


def _train_low_level(
    env_config: AssemblyScenarioConfig,
    training_config: TrainingConfig,
    runtime_profile: AssemblyRuntimeProfile,
    run_dir: Path,
) -> tuple[Path, Path]:
    from embodied_skill_composer.assembly.trainer import MAPPOTrainer

    checkpoint_path = run_dir / "assembly_marl.pt"
    metrics_path = run_dir / "assembly_training_metrics.json"
    env = build_assembly_backend(env_config, runtime_profile, seed=training_config.seed)
    trainer = MAPPOTrainer(env=env, config=training_config, device=runtime_profile.device)
    trainer.train(checkpoint_path=checkpoint_path, metrics_path=metrics_path)
    return checkpoint_path, metrics_path


def _result_from_benchmark(
    scenario_id: str,
    scenario_path: Path,
    seed: int,
    result: PolicyBenchmarkResult,
    checkpoint_path: Path | None = None,
    train_metrics_path: Path | None = None,
) -> AssemblyExperimentPolicyResult:
    return AssemblyExperimentPolicyResult(
        scenario_id=scenario_id,
        scenario_path=str(scenario_path),
        seed=seed,
        policy_name=result.policy_name,
        success_rate=result.success_rate,
        mean_return=result.mean_return,
        mean_beams_installed=result.mean_beams_installed,
        mean_step_count=result.mean_step_count,
        checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
        train_metrics_path=None if train_metrics_path is None else str(train_metrics_path),
        notes=result.notes,
    )


def _failed_results(
    scenario_id: str,
    scenario_path: Path,
    seed: int,
    exc: Exception,
) -> list[AssemblyExperimentPolicyResult]:
    return [
        AssemblyExperimentPolicyResult(
            scenario_id=scenario_id,
            scenario_path=str(scenario_path),
            seed=seed,
            policy_name=policy_name,
            success_rate=0.0,
            mean_return=0.0,
            mean_beams_installed=0.0,
            status="failed",
            notes=str(exc),
        )
        for policy_name in ["scripted_options", "learned_options", "low_level_learned"]
    ]
