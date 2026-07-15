from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.brain import (
    HeuristicConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.env import AssemblyAction, CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BeamTask,
)
from embodied_skill_composer.assembly.runtime import load_assembly_scenario
from embodied_skill_composer.assembly.scenario_generation import (
    ScenarioGenerationConfig,
    generate_assembly_scenarios,
    scenario_occupied_cells,
)


mujoco_available = importlib.util.find_spec("mujoco") is not None


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_single_beam_config(**updates: object) -> AssemblyScenarioConfig:
    payload = {
        "grid_size": 10,
        "max_steps": 80,
        "agent_starts": [(0, 2), (0, 3)],
        "beams": [
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(7, 6),
                assembly_right=(7, 7),
            )
        ],
        **updates,
    }
    return AssemblyScenarioConfig.model_validate(payload)


@pytest.mark.parametrize(
    "obstacles",
    [
        [(1, 1), (1, 1)],
        [(10, 1)],
        [(0, 2)],
        [(2, 2)],
    ],
)
def test_obstacle_geometry_validation_rejects_invalid_cells(
    obstacles: list[tuple[int, int]],
) -> None:
    with pytest.raises(ValidationError):
        build_single_beam_config(obstacle_cells=obstacles)


def test_manual_obstacle_contact_is_counted_without_entering_cell() -> None:
    env = CollaborativeAssemblyEnv(
        build_single_beam_config(obstacle_cells=[(1, 2)]),
        seed=7,
    )
    env.reset(seed=7)

    env.step([AssemblyAction.RIGHT, AssemblyAction.STAY])
    artifact = env.build_artifact(policy_mode="scripted")

    assert env.state.agent_positions == [(0, 2), (0, 3)]
    assert artifact.metrics.collision_count == 1
    assert artifact.metrics.obstacle_collision_count == 1
    assert artifact.metrics.wasted_step_count == 1


def test_construction_brain_solves_committed_obstacle_detour() -> None:
    config = load_assembly_scenario(workspace_root() / "configs" / "assembly_obstacles.yaml")
    env = CollaborativeAssemblyEnv(config, seed=7)

    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    assert episode.artifact.metrics.success is True
    assert episode.artifact.metrics.structure_completion_rate == 1.0
    assert episode.artifact.metrics.step_count > 32
    assert episode.artifact.metrics.collision_count == 0
    assert episode.artifact.metrics.obstacle_collision_count == 0
    assert episode.artifact.metrics.wasted_step_count == 0
    assert episode.steps[0].observation.obstacle_cells == config.obstacle_cells
    assert episode.diagnostics["obstacle_cells"] == [list(cell) for cell in config.obstacle_cells]


def test_generated_obstacle_scenario_is_valid_and_scripted_solvable() -> None:
    base = load_assembly_scenario(workspace_root() / "configs" / "assembly_env.yaml")
    scenario = generate_assembly_scenarios(
        base,
        ScenarioGenerationConfig(
            scenario_count=1,
            base_seed=31,
            beam_count=2,
            obstacle_count=4,
        ),
    )[0]

    assert len(scenario.config.obstacle_cells) == 4
    assert len(scenario_occupied_cells(scenario.config)) == 2 + 2 * 4 + 4
    env = CollaborativeAssemblyEnv(scenario.config, seed=scenario.seed)
    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=scenario.seed,
    )
    assert episode.artifact.metrics.success is True


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for obstacle parity")
def test_mujoco_obstacle_detour_matches_local_metrics() -> None:
    config = load_assembly_scenario(workspace_root() / "configs" / "assembly_obstacles.yaml")
    local_episode = run_construction_brain_episode(
        CollaborativeAssemblyEnv(config, seed=7),
        HeuristicConstructionBrain(),
        seed=7,
    )
    mujoco_env = build_assembly_backend(
        config,
        AssemblyRuntimeProfile(name="mujoco_obstacles", backend="mujoco_local"),
        seed=7,
    )
    mujoco_episode = run_construction_brain_episode(
        mujoco_env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    assert mujoco_episode.artifact.metrics == local_episode.artifact.metrics
    assert mujoco_episode.diagnostics["mujoco_model"]["obstacle_count"] == 6
    assert mujoco_env.model.geom("obstacle_000").id >= 0
    assert mujoco_episode.diagnostics["mujoco_physics_control"]["max_target_error"] < 0.02
