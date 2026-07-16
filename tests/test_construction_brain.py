from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from embodied_skill_composer.assembly.brain import (
    HeuristicConstructionBrain,
    ScriptedConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyScenarioConfig,
    AssemblyRuntimeProfile,
    BeamTask,
    BlueprintSlot,
    ConstructionResource,
    TeamOption,
)


mujoco_available = importlib.util.find_spec("mujoco") is not None


def build_config() -> AssemblyScenarioConfig:
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
    )


def test_construction_observation_exposes_brain_contract() -> None:
    env = CollaborativeAssemblyEnv(build_config(), seed=7)
    env.reset(seed=7)

    observation = env.get_construction_observation()

    assert observation.backend == "local_sandbox"
    assert observation.current_beam_name == "beam_alpha"
    assert len(observation.resources) == 2
    assert len(observation.blueprint_slots) == 2
    assert observation.progress.structure_completion_rate == 0.0
    assert TeamOption.GO_PICKUP in observation.available_options
    assert observation.physical_feedback is None


def test_scripted_construction_brain_matches_existing_oracle() -> None:
    config = build_config()
    oracle_env = CollaborativeAssemblyEnv(config, seed=7)
    oracle_env.reset(seed=7)
    oracle_options: list[TeamOption] = []
    done = False
    while not done:
        option = oracle_env.scripted_team_option()
        oracle_options.append(option)
        done = oracle_env.execute_team_option(option).done

    brain_env = CollaborativeAssemblyEnv(config, seed=7)
    episode = run_construction_brain_episode(
        brain_env,
        ScriptedConstructionBrain(),
        seed=7,
    )

    assert [step.decision.option for step in episode.steps] == oracle_options
    assert episode.artifact.metrics.total_reward == oracle_env.state.total_reward
    assert episode.artifact.metrics.success is True
    assert episode.artifact.metrics.structure_completion_rate == 1.0


def test_heuristic_brain_assigns_nearest_compatible_resources() -> None:
    config = build_config().model_copy(
        update={
            "resources": [
                ConstructionResource(
                    resource_id="beam_beta",
                    source_cells=[(9, 1), (9, 2)],
                ),
                ConstructionResource(
                    resource_id="beam_alpha",
                    source_cells=[(1, 1), (1, 2)],
                ),
            ],
            "blueprint_slots": [
                BlueprintSlot(
                    slot_id="near_slot",
                    target_cells=[(2, 1), (2, 2)],
                ),
                BlueprintSlot(
                    slot_id="far_slot",
                    target_cells=[(10, 1), (10, 2)],
                ),
            ],
        },
        deep=True,
    )
    env = CollaborativeAssemblyEnv(config, seed=7)
    env.reset(seed=7)
    observation = env.get_construction_observation()
    brain = HeuristicConstructionBrain()
    brain.reset(observation)

    assignments = brain.assignments(observation)

    assert [(item.resource_id, item.slot_id) for item in assignments] == [
        ("beam_alpha", "near_slot"),
        ("beam_beta", "far_slot"),
    ]


def test_heuristic_brain_completes_default_construction() -> None:
    env = CollaborativeAssemblyEnv(build_config(), seed=7)

    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    assert episode.artifact.metrics.success is True
    assert episode.artifact.metrics.resource_delivery_accuracy == 1.0
    assert episode.artifact.metrics.wasted_step_count == 0
    assert {assignment.status for assignment in episode.assignments} == {"completed"}
    assert all(step.decision.rationale for step in episode.steps)


def test_construction_brain_cli_writes_experiment_artifact(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "construction_brain.json"

    result = subprocess.run(
        [
            sys.executable,
            str(workspace / "scripts" / "run_construction_brain.py"),
            "--brain",
            "heuristic",
            "--episodes",
            "1",
            "--output",
            str(output_path),
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success_rate"] == 1.0
    assert payload["backend_status"]["is_ready"] is True
    assert payload["mean_structure_completion_rate"] == 1.0
    assert payload["mean_sensor_safety_holds"] == 0.0
    assert payload["mean_visual_safety_holds"] == 0.0
    assert payload["mean_terminal_safety_holds"] == 0.0
    assert payload["mean_visual_position_error_m"] is None
    assert payload["mean_visual_samples_per_episode"] == 0.0
    assert payload["mean_final_visible_blueprint_recall"] is None
    assert payload["mean_final_tracked_blueprint_recall"] is None
    assert payload["episodes"][0]["assignments"][0]["status"] == "completed"


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for backend parity")
def test_mujoco_brain_episode_matches_local_construction_metrics() -> None:
    config = build_config()
    local_env = CollaborativeAssemblyEnv(config, seed=7)
    mujoco_env = build_assembly_backend(
        config,
        AssemblyRuntimeProfile(name="mujoco_test", backend="mujoco_local"),
        seed=7,
    )

    local_episode = run_construction_brain_episode(
        local_env,
        HeuristicConstructionBrain(),
        seed=7,
    )
    mujoco_episode = run_construction_brain_episode(
        mujoco_env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    assert mujoco_env.get_backend_status().is_ready is True
    assert mujoco_episode.artifact.metrics == local_episode.artifact.metrics
    assert mujoco_episode.assignments == local_episode.assignments
    physical_feedback = mujoco_episode.steps[0].observation.physical_feedback
    assert physical_feedback is not None
    assert physical_feedback.backend == "mujoco_local"
    assert physical_feedback.gripper_state == "open"
    assert len(physical_feedback.gripper_joint_positions_m) == 4
    assert physical_feedback.required_minimum_grip_force_n == 25.0
    assert any(
        "Physical feedback:" in step.decision.rationale
        for step in mujoco_episode.steps
        if step.decision.option in {TeamOption.GRAB, TeamOption.INSTALL}
    )
    physics = mujoco_episode.diagnostics["mujoco_physics_control"]
    assert physics["physics_step_count"] > 0
    assert physics["trajectory_frame_count"] > len(mujoco_episode.steps)
    assert physics["max_target_error"] < 0.02
