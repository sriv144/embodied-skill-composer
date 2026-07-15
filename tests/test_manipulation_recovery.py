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
from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    TeamOption,
)
from embodied_skill_composer.assembly.runtime import load_assembly_scenario


mujoco_available = importlib.util.find_spec("mujoco") is not None


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_recovery_config() -> AssemblyScenarioConfig:
    return load_assembly_scenario(workspace_root() / "configs" / "assembly_recovery.yaml")


@pytest.mark.parametrize(
    "rules",
    [
        [{"beam_name": "missing", "phase": "grasp"}],
        [
            {"beam_name": "beam_alpha", "phase": "grasp"},
            {"beam_name": "beam_alpha", "phase": "grasp"},
        ],
    ],
)
def test_manipulation_failure_rules_validate_beam_and_uniqueness(
    rules: list[dict[str, str]],
) -> None:
    base = load_assembly_scenario(workspace_root() / "configs" / "assembly_env.yaml")
    payload = base.model_dump(mode="python")
    payload["manipulation_failures"] = rules

    with pytest.raises(ValidationError):
        AssemblyScenarioConfig.model_validate(payload)


def test_brain_retries_and_recovers_from_scheduled_manipulation_failures() -> None:
    env = CollaborativeAssemblyEnv(load_recovery_config(), seed=7)

    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    failed_steps = [
        step for step in episode.steps if step.execution.info.get("manipulation_failed")
    ]
    retry_steps = [step for step in episode.steps if "Retry" in step.decision.rationale]
    assert episode.artifact.metrics.success is True
    assert episode.artifact.metrics.step_count == 42
    assert episode.artifact.metrics.manipulation_failure_count == 2
    assert episode.artifact.metrics.manipulation_recovery_count == 2
    assert episode.artifact.metrics.wasted_step_count == 2
    assert [step.decision.option for step in failed_steps] == [
        TeamOption.GRAB,
        TeamOption.INSTALL,
    ]
    assert len(retry_steps) == 2
    assert episode.diagnostics["manipulation_attempts"] == {
        "beam_alpha:grasp": 2,
        "beam_alpha:install": 1,
        "beam_beta:grasp": 1,
        "beam_beta:install": 2,
    }


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for physical recovery")
def test_mujoco_rejects_misaligned_grasp_then_recovers() -> None:
    config = load_assembly_scenario(workspace_root() / "configs" / "assembly_env.yaml")
    env = build_assembly_backend(
        config,
        AssemblyRuntimeProfile(name="mujoco_recovery", backend="mujoco_local"),
        seed=7,
    )
    env.reset(seed=7)
    env.execute_team_option(TeamOption.GO_PICKUP)
    position = env._body_position("agent0")
    env._set_freejoint_pose(
        "agent0_free",
        position[0] + 0.10,
        position[1],
        position[2],
    )
    env._mujoco.mj_forward(env.model, env.data)

    failed = env.execute_team_option(TeamOption.GRAB)
    recovered = env.execute_team_option(TeamOption.GRAB)
    checks = env.get_physics_control_diagnostics()["physical_manipulation_checks"]

    assert failed.success is False
    assert failed.info["retryable"] is True
    assert str(failed.info["failure_reason"]).startswith("physical_alignment_error_")
    assert recovered.success is True
    assert env.logical_env.state.manipulation_failure_count == 1
    assert env.logical_env.state.manipulation_recovery_count == 1
    assert checks[0]["passed"] is False
    assert checks[1]["passed"] is True
    assert checks[1]["alignment_error_m"] < 0.001


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for recovery parity")
def test_scheduled_recovery_scenario_matches_local_and_mujoco() -> None:
    config = load_recovery_config()
    local_episode = run_construction_brain_episode(
        CollaborativeAssemblyEnv(config, seed=7),
        HeuristicConstructionBrain(),
        seed=7,
    )
    mujoco_env = build_assembly_backend(
        config,
        AssemblyRuntimeProfile(name="mujoco_recovery", backend="mujoco_local"),
        seed=7,
    )
    mujoco_episode = run_construction_brain_episode(
        mujoco_env,
        HeuristicConstructionBrain(),
        seed=7,
    )

    checks = mujoco_episode.diagnostics["mujoco_physics_control"][
        "physical_manipulation_checks"
    ]
    physical_retry_steps = [
        step
        for step in mujoco_episode.steps
        if "Retry" in step.decision.rationale
    ]
    assert mujoco_episode.artifact.metrics == local_episode.artifact.metrics
    assert mujoco_episode.artifact.metrics.manipulation_recovery_count == 2
    assert len(checks) == 6
    assert all(check["passed"] for check in checks)
    assert len(physical_retry_steps) == 2
    assert all(
        "Physical feedback:" in step.decision.rationale
        for step in physical_retry_steps
    )
    grasp_retry_feedback = physical_retry_steps[0].observation.physical_feedback
    assert grasp_retry_feedback is not None
    assert grasp_retry_feedback.last_check_phase == "grasp"
    assert min(grasp_retry_feedback.last_contact_forces_n.values()) >= 25.0
