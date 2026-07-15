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
from embodied_skill_composer.assembly.models import (
    PhysicalManipulationFeedback,
    PhysicalSensorConfig,
    TeamOption,
)
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
)
from embodied_skill_composer.assembly.sensing import PhysicalSensorSuite


mujoco_available = importlib.util.find_spec("mujoco") is not None
workspace = Path(__file__).resolve().parents[1]


def build_truth(
    alignment: float = 0.01,
    force: float = 30.0,
    joint_position: float = 0.02,
) -> PhysicalManipulationFeedback:
    return PhysicalManipulationFeedback(
        backend="mujoco_local",
        current_alignment_error_m=alignment,
        alignment_tolerance_m=0.03,
        required_minimum_grip_force_n=25.0,
        last_check_phase="grasp",
        last_check_passed=True,
        last_contact_forces_n={"agent0": force, "agent1": force},
        gripper_state="closed",
        gripper_joint_positions_m={
            "agent0_left_finger_slide": joint_position,
            "agent0_right_finger_slide": joint_position,
        },
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"dropout_probability": 1.01},
        {"dropout_probability": -0.01},
        {"ema_alpha": 0.0},
        {"alignment_noise_std_m": -0.001},
    ],
)
def test_physical_sensor_config_rejects_invalid_values(
    payload: dict[str, float],
) -> None:
    with pytest.raises(ValidationError):
        PhysicalSensorConfig.model_validate(payload)


def test_simulated_sensor_noise_is_seeded_and_reproducible() -> None:
    config = PhysicalSensorConfig(
        enabled=True,
        alignment_noise_std_m=0.01,
        force_noise_std_n=2.0,
        joint_position_noise_std_m=0.002,
    )
    first = PhysicalSensorSuite(config, seed=11).observe(build_truth(), physics_step=10)
    second = PhysicalSensorSuite(config, seed=11).observe(build_truth(), physics_step=10)

    assert first == second
    assert first.sensor_mode == "simulated"
    assert first.sensor_fresh is True
    assert first.current_alignment_error_m != 0.01
    assert first.last_contact_forces_n["agent0"] != 30.0


def test_sensor_dropout_marks_first_sample_unavailable() -> None:
    suite = PhysicalSensorSuite(
        PhysicalSensorConfig(enabled=True, dropout_probability=1.0),
        seed=7,
    )

    feedback = suite.observe(build_truth(), physics_step=20)

    assert feedback.sensor_dropped is True
    assert feedback.sensor_fresh is False
    assert feedback.current_alignment_error_m is None
    assert feedback.gripper_state == "unknown"
    assert feedback.gripper_joint_positions_m == {}
    assert suite.diagnostics()["dropout_count"] == 1


def test_sensor_ema_filters_successive_measurements() -> None:
    suite = PhysicalSensorSuite(
        PhysicalSensorConfig(enabled=True, ema_alpha=0.5),
        seed=7,
    )
    suite.observe(build_truth(alignment=0.0, force=10.0, joint_position=0.0), 0)

    feedback = suite.observe(
        build_truth(alignment=0.02, force=30.0, joint_position=0.02),
        10,
    )

    assert feedback.current_alignment_error_m == pytest.approx(0.01)
    assert feedback.last_contact_forces_n["agent0"] == pytest.approx(20.0)
    assert feedback.gripper_joint_positions_m[
        "agent0_left_finger_slide"
    ] == pytest.approx(0.01)


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for sensing tests")
def test_brain_waits_for_physical_realignment_before_grasp() -> None:
    config = load_assembly_scenario(workspace / "configs" / "assembly_env.yaml")
    profile = load_runtime_profile(
        workspace / "configs" / "assembly_profiles" / "mujoco_local.yaml"
    )
    env = build_assembly_backend(config, profile, seed=7)
    env.reset(seed=7)
    brain = HeuristicConstructionBrain()
    brain.reset(env.get_construction_observation())
    env.execute_team_option(TeamOption.GO_PICKUP)
    position = env._body_position("agent0")
    env._set_freejoint_pose(
        "agent0_free",
        position[0] + 0.10,
        position[1],
        position[2],
    )
    env._mujoco.mj_forward(env.model, env.data)

    misaligned = env.get_construction_observation()
    hold = brain.decide(misaligned)
    assert hold.option == TeamOption.WAIT
    assert hold.safety_hold_reason == "alignment_error"
    assert "reduces sensed alignment error" in hold.rationale
    assert env.execute_team_option(hold.option).success is True

    aligned = env.get_construction_observation()
    grasp = brain.decide(aligned)
    assert grasp.option == TeamOption.GRAB
    assert aligned.physical_feedback is not None
    assert aligned.physical_feedback.current_alignment_error_m is not None
    assert (
        aligned.physical_feedback.current_alignment_error_m
        <= aligned.physical_feedback.alignment_tolerance_m
    )
    stale_feedback = aligned.physical_feedback.model_copy(
        update={"sensor_fresh": False, "sensor_dropped": True}
    )
    stale = aligned.model_copy(update={"physical_feedback": stale_feedback})
    stale_hold = brain.decide(stale)
    assert stale_hold.option == TeamOption.WAIT
    assert stale_hold.safety_hold_reason == "sensor_unavailable"
    assert "dropped or stale physical sensor" in stale_hold.rationale


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for sensing tests")
def test_noisy_sensing_profile_completes_recovery_scenario() -> None:
    config = load_assembly_scenario(workspace / "configs" / "assembly_recovery.yaml")
    profile = load_runtime_profile(
        workspace / "configs" / "assembly_profiles" / "mujoco_sensing.yaml"
    )
    env = build_assembly_backend(config, profile, seed=7)

    episode = run_construction_brain_episode(
        env,
        HeuristicConstructionBrain(),
        seed=7,
    )
    sensor_diagnostics = episode.diagnostics["mujoco_physics_control"][
        "physical_sensors"
    ]

    assert episode.artifact.metrics.success is True
    assert sensor_diagnostics["enabled"] is True
    assert sensor_diagnostics["sample_count"] > 0
    assert sensor_diagnostics["dropout_count"] > 0
    assert (
        episode.diagnostics["construction_brain"]["sensor_safety_hold_count"]
        == sum(step.decision.safety_hold_reason is not None for step in episode.steps)
    )
    assert any(
        step.observation.physical_feedback is not None
        and step.observation.physical_feedback.sensor_dropped
        for step in episode.steps
    )
