import importlib.util
from pathlib import Path

import yaml
import pytest

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.experiments import run_assembly_experiment_sweep
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BeamTask,
    TrainingConfig,
)
from embodied_skill_composer.assembly.scenario_generation import (
    ScenarioGenerationConfig,
    generate_assembly_scenarios,
    scenario_occupied_cells,
    write_generated_scenario,
)


torch_available = importlib.util.find_spec("torch") is not None


def build_base_config() -> AssemblyScenarioConfig:
    return AssemblyScenarioConfig(
        grid_size=12,
        max_steps=120,
        agent_starts=[(0, 2), (0, 3)],
        beams=[
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(8, 7),
                assembly_right=(8, 8),
            ),
            BeamTask(
                name="beam_beta",
                pickup_left=(2, 6),
                pickup_right=(2, 7),
                assembly_left=(9, 7),
                assembly_right=(9, 8),
            ),
        ],
        curriculum_beam_stages=[1, 2],
    )


def build_tiny_training_config() -> TrainingConfig:
    return TrainingConfig(
        total_iterations=1,
        episodes_per_iteration=1,
        update_epochs=1,
        minibatch_size=8,
        behavior_cloning_epochs=1,
        option_behavior_cloning_epochs=1,
        option_update_epochs=1,
        evaluation_episodes=1,
        seed=7,
    )


def test_generated_scenarios_are_valid_and_scripted_solvable(tmp_path: Path) -> None:
    profile = AssemblyRuntimeProfile()
    scenarios = generate_assembly_scenarios(
        base_config=build_base_config(),
        generation_config=ScenarioGenerationConfig(scenario_count=2, base_seed=11, beam_count=2),
        runtime_profile=profile,
    )

    assert [scenario.scenario_id for scenario in scenarios] == ["scenario_001", "scenario_002"]
    for scenario in scenarios:
        config = scenario.config
        occupied = scenario_occupied_cells(config)
        assert len(occupied) == len(config.agent_starts) + len(config.beams) * 4
        assert len(config.beams) == 2
        assert len(config.resources) == 2
        assert len(config.blueprint_slots) == 2
        assert {resource.resource_id for resource in config.resources} == {beam.name for beam in config.beams}
        assert {slot.required_resource_id for slot in config.blueprint_slots} == {beam.name for beam in config.beams}
        for x, y in occupied:
            assert 0 <= x < config.grid_size
            assert 0 <= y < config.grid_size

        env = build_assembly_backend(config, profile, seed=scenario.seed)
        env.reset(seed=scenario.seed)
        done = False
        while not done:
            result = env.execute_team_option(env.scripted_team_option())
            done = result.done
        artifact = env.build_artifact(policy_mode="scripted")
        assert artifact.metrics.success is True
        assert artifact.metrics.beams_installed == len(config.beams)
        assert artifact.metrics.structure_completion_rate == 1.0
        assert artifact.metrics.resource_delivery_accuracy == 1.0

        scenario_path = write_generated_scenario(scenario, tmp_path / f"{scenario.scenario_id}.yaml")
        loaded = AssemblyScenarioConfig.model_validate(yaml.safe_load(scenario_path.read_text(encoding="utf-8")))
        assert loaded == config


@pytest.mark.skipif(not torch_available, reason="torch is required for experiment sweep smoke tests")
def test_experiment_sweep_writes_summary_outputs(tmp_path: Path) -> None:
    summary = run_assembly_experiment_sweep(
        base_env_config=build_base_config(),
        training_config=build_tiny_training_config(),
        runtime_profile=AssemblyRuntimeProfile(device="cpu"),
        scenario_count=1,
        seeds=[7],
        output_dir=tmp_path / "sweep",
        evaluation_episodes=1,
    )

    assert summary.scenario_count == 1
    assert len(summary.results) == 3
    assert {result.policy_name for result in summary.results} == {
        "scripted_options",
        "learned_options",
        "low_level_learned",
    }
    assert all(result.status == "ok" for result in summary.results)
    assert (tmp_path / "sweep" / "summary.json").exists()
    assert (tmp_path / "sweep" / "summary.csv").exists()
    assert (tmp_path / "sweep" / "scenario_001" / "scenario.yaml").exists()
