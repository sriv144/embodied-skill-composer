from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    TeamOption,
)
from embodied_skill_composer.assembly.runtime import load_assembly_scenario


mujoco_available = importlib.util.find_spec("mujoco") is not None


def build_mujoco_env():
    workspace = Path(__file__).resolve().parents[1]
    config = load_assembly_scenario(workspace / "configs" / "assembly_env.yaml")
    env = build_assembly_backend(
        config,
        AssemblyRuntimeProfile(name="mujoco_attachment", backend="mujoco_local"),
        seed=7,
    )
    env.set_curriculum_stage(1)
    env.reset(seed=7)
    return env


def equality_active(env, name: str) -> bool:
    equality_id = env._mujoco.mj_name2id(
        env.model,
        env._mujoco.mjtObj.mjOBJ_EQUALITY,
        name,
    )
    return bool(env.data.eq_active[equality_id])


def site_position(env, name: str) -> np.ndarray:
    site_id = env._mujoco.mj_name2id(
        env.model,
        env._mujoco.mjtObj.mjOBJ_SITE,
        name,
    )
    return env.data.site_xpos[site_id].copy()


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for attachment tests")
def test_dual_gripper_contacts_activate_runtime_beam_weld() -> None:
    env = build_mujoco_env()
    env.execute_team_option(TeamOption.GO_PICKUP)

    result = env.execute_team_option(TeamOption.GRAB)
    physics = env.get_physics_control_diagnostics()
    check = physics["physical_manipulation_checks"][-1]

    assert result.success is True
    assert check["contact_agents"] == ["agent0", "agent1"]
    assert check["dual_gripper_contact"] is True
    assert check["grip_force_ready"] is True
    assert check["minimum_contact_force_n"] >= 25.0
    assert all(force >= 25.0 for force in check["contact_forces_n"].values())
    assert physics["active_attachment_beam"] == "beam_alpha"
    assert equality_active(env, "beam_alpha_track") is False
    assert equality_active(env, "beam_alpha_carry") is True
    assert physics["attachment_events"][-1]["event"] == "attached"


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for gripper tests")
def test_articulated_grippers_close_for_grasp_and_open_on_release() -> None:
    env = build_mujoco_env()
    initial = env.get_physics_control_diagnostics()["articulated_grippers"]

    assert initial["actuator_count"] == 4
    assert all(abs(position) < 1e-8 for position in initial["joint_positions"].values())
    env.execute_team_option(TeamOption.GO_PICKUP)
    assert env.execute_team_option(TeamOption.GRAB).success is True
    grasped = env.get_physics_control_diagnostics()["articulated_grippers"]

    assert grasped["events"][-1]["command"] == "close"
    assert all(position > 0.005 for position in grasped["joint_positions"].values())
    assert env.execute_team_option(TeamOption.GO_ASSEMBLY).success is True
    assert env.execute_team_option(TeamOption.INSTALL).success is True
    released = env.get_physics_control_diagnostics()["articulated_grippers"]

    assert released["events"][-1]["command"] == "open"
    assert all(abs(position) < 0.001 for position in released["joint_positions"].values())


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for contact tests")
def test_missing_second_gripper_contact_rejects_grasp_then_retries() -> None:
    env = build_mujoco_env()
    env.execute_team_option(TeamOption.GO_PICKUP)
    for side in ("left", "right"):
        agent1_geom = env.model.geom(f"agent1_{side}_finger_geom").id
        env.model.geom_contype[agent1_geom] = 0
        env.model.geom_conaffinity[agent1_geom] = 0

    failed = env.execute_team_option(TeamOption.GRAB)
    failed_check = env.get_physics_control_diagnostics()[
        "physical_manipulation_checks"
    ][-1]
    env._set_grasp_contacts_enabled("beam_alpha", True)
    recovered = env.execute_team_option(TeamOption.GRAB)

    assert failed.success is False
    assert failed.info["failure_reason"] == "missing_dual_gripper_contact"
    assert failed_check["contact_agents"] == ["agent0"]
    assert recovered.success is True
    assert env.logical_env.state.manipulation_recovery_count == 1


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for force tests")
def test_weak_dual_contact_force_rejects_grasp_then_recovers() -> None:
    env = build_mujoco_env()
    env._minimum_grip_force = 500.0
    env.execute_team_option(TeamOption.GO_PICKUP)

    failed = env.execute_team_option(TeamOption.GRAB)
    failed_check = env.get_physics_control_diagnostics()[
        "physical_manipulation_checks"
    ][-1]
    env._minimum_grip_force = 25.0
    recovered = env.execute_team_option(TeamOption.GRAB)

    assert failed.success is False
    assert failed_check["dual_gripper_contact"] is True
    assert failed_check["grip_force_ready"] is False
    assert failed_check["minimum_contact_force_n"] < 500.0
    assert str(failed.info["failure_reason"]).startswith("grip_force_")
    assert recovered.success is True
    assert env.logical_env.state.manipulation_recovery_count == 1


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for attachment tests")
def test_attachment_weld_recovers_forced_beam_displacement() -> None:
    env = build_mujoco_env()
    env.execute_team_option(TeamOption.GO_PICKUP)
    env.execute_team_option(TeamOption.GRAB)
    position = env._body_position("beam_alpha")
    env._set_freejoint_pose(
        "beam_alpha_free",
        position[0] + 0.12,
        position[1],
        position[2],
    )
    env._mujoco.mj_forward(env.model, env.data)

    displaced_error = float(
        np.linalg.norm(
            site_position(env, "beam_alpha_attach_site")
            - site_position(env, "agent0_gripper_site")
        )
    )
    env._advance_physics(30)
    recovered_error = float(
        np.linalg.norm(
            site_position(env, "beam_alpha_attach_site")
            - site_position(env, "agent0_gripper_site")
        )
    )

    assert displaced_error > 0.1
    assert recovered_error < 0.001
    assert env.get_physics_control_diagnostics()["active_attachment_beam"] == "beam_alpha"


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo is required for attachment tests")
def test_install_releases_attachment_and_restores_beam_tracking() -> None:
    env = build_mujoco_env()
    for option in [TeamOption.GO_PICKUP, TeamOption.GRAB, TeamOption.GO_ASSEMBLY]:
        assert env.execute_team_option(option).success is True

    installed = env.execute_team_option(TeamOption.INSTALL)
    physics = env.get_physics_control_diagnostics()

    assert installed.success is True
    assert physics["active_attachment_beam"] is None
    assert equality_active(env, "beam_alpha_track") is True
    assert equality_active(env, "beam_alpha_carry") is False
    assert [event["event"] for event in physics["attachment_events"]] == [
        "attached",
        "detached",
    ]
    target = np.asarray(env._beam_world_pose("beam_alpha"))
    assert np.linalg.norm(env._body_position("beam_alpha") - target) < 0.02
