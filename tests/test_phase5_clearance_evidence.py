from __future__ import annotations

from copy import deepcopy

import pytest

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    _verify_logical_transition_stop_proof,
)
from embodied_skill_composer.construction.intelligence_models import (
    RobotCommand,
    RobotTelemetry,
)
from embodied_skill_composer.construction.models import Pose3D, Vec3


def _zero_command(robot_id: str) -> RobotCommand:
    return RobotCommand(
        timestamp_s=1.0,
        robot_id=robot_id,
        linear_velocity_mps=0,
        angular_velocity_rps=0,
        wheel_target_velocity_rad_s=(0, 0, 0, 0),
        source="formation_hold",
    )


def _telemetry(
    robot_id: str,
    timestamp_s: float,
    *,
    linear_velocity_mps: float = 0,
) -> RobotTelemetry:
    return RobotTelemetry(
        timestamp_s=timestamp_s,
        robot_id=robot_id,
        measured_pose=Pose3D(
            position=Vec3(
                x=0 if robot_id == "robot_0" else 0.8,
                y=0,
                z=0,
            )
        ),
        linear_velocity_mps=linear_velocity_mps,
        angular_velocity_rps=0,
        battery_remaining_wh=100,
    )


def _proof_fixture() -> tuple[
    DynamicCoppeliaConfig,
    list[RobotCommand],
    list[RobotTelemetry],
    dict[str, object],
    list[dict[str, object]],
]:
    config = DynamicCoppeliaConfig(
        control_hz=10,
        logical_transition_settle_consecutive_samples=3,
        safety_distance_m=0.05,
    )
    robot_ids = ["robot_0", "robot_1"]
    commands = [_zero_command(robot_id) for robot_id in robot_ids]
    timestamps = [1.0, 1.1, 1.2]
    telemetry = [
        _telemetry(robot_id, timestamp_s)
        for timestamp_s in timestamps
        for robot_id in robot_ids
    ]
    command_records = {
        command.robot_id: {
            "present": True,
            "zero": True,
            "timestamp_s": command.timestamp_s,
            "source": command.source,
            "linear_velocity_mps": command.linear_velocity_mps,
            "angular_velocity_rps": command.angular_velocity_rps,
            "wheel_target_velocity_rad_s": list(
                command.wheel_target_velocity_rad_s
            ),
        }
        for command in commands
    }
    measured_records = {
        robot_id: {
            "timestamp_s": 1.2,
            "still": True,
            "linear_velocity_mps": 0.0,
            "angular_velocity_rps": 0.0,
        }
        for robot_id in robot_ids
    }
    proof: dict[str, object] = {
        "robot_ids": robot_ids,
        "latest_commands_zero": True,
        "latest_commands_by_robot": command_records,
        "measured_team_still": True,
        "measured_motion_by_robot": measured_records,
        "settled_linear_speed_mps": config.settled_linear_speed_mps,
        "settled_angular_speed_rps": config.settled_angular_speed_rps,
        "minimum_measured_team_to_scene_separation_m": 0.8,
        "minimum_measured_team_to_scene_pair": robot_ids,
        "safety_distance_m": config.safety_distance_m,
        "hold_command_index": 0,
        "command_count_at_settle": 2,
        "started_at_s": 1.0,
        "settled_at_s": 1.2,
        "physics_steps_elapsed": 2,
        "required_consecutive_still_samples": 3,
        "observed_consecutive_still_samples": 3,
        "no_nonzero_team_command_after_hold": True,
    }
    settle_samples = [
        {
            "timestamp_s": timestamp_s,
            "physics_step": 10 + index,
            "team_still": True,
            "consecutive_still_samples": index + 1,
            "robots": {
                robot_id: {
                    "timestamp_s": timestamp_s,
                    "linear_velocity_mps": 0.0,
                    "angular_velocity_rps": 0.0,
                    "still": True,
                }
                for robot_id in robot_ids
            },
        }
        for index, timestamp_s in enumerate(timestamps)
    ]
    event = {
        "timestamp_s": 1.2,
        "event": "logical_payload_lifted",
        "module_id": "module_0",
        "team_zero_motion_proof": proof,
    }
    trace = [
        {
            "timestamp_s": 1.2,
            "event": "logical_transition_settled",
            "module_id": "module_0",
            "transition": "logical_payload_lift",
            "team_zero_motion_proof": proof,
            "settle_samples": settle_samples,
        }
    ]
    return config, commands, telemetry, event, trace


def _verify_fixture(
    config: DynamicCoppeliaConfig,
    commands: list[RobotCommand],
    telemetry: list[RobotTelemetry],
    event: dict[str, object],
    trace: list[dict[str, object]],
) -> None:
    _verify_logical_transition_stop_proof(
        event=event,
        module_id="module_0",
        robot_ids=["robot_0", "robot_1"],
        transition="logical_payload_lift",
        timestamp_s=1.2,
        commands=commands,
        telemetry=telemetry,
        trace_records=trace,
        config=config,
    )


def test_transition_verifier_rescans_commands_and_stable_telemetry() -> None:
    config, commands, telemetry, event, trace = _proof_fixture()
    _verify_fixture(config, commands, telemetry, event, trace)

    nonzero_commands = [
        *commands,
        RobotCommand(
            timestamp_s=1.1,
            robot_id="robot_0",
            linear_velocity_mps=0.2,
            angular_velocity_rps=0,
            wheel_target_velocity_rad_s=(1, 1, 1, 1),
            source="path_follower",
        ),
    ]
    nonzero_event = deepcopy(event)
    nonzero_trace = deepcopy(trace)
    nonzero_proof = nonzero_event["team_zero_motion_proof"]
    assert isinstance(nonzero_proof, dict)
    nonzero_proof["command_count_at_settle"] = 3
    nonzero_trace[0]["team_zero_motion_proof"] = nonzero_proof
    with pytest.raises(ValueError, match="command slice"):
        _verify_fixture(
            config,
            nonzero_commands,
            telemetry,
            nonzero_event,
            nonzero_trace,
        )

    fabricated_telemetry = [
        sample.model_copy(
            update={"linear_velocity_mps": 0.2}
        )
        if sample.robot_id == "robot_0" and sample.timestamp_s == 1.1
        else sample
        for sample in telemetry
    ]
    with pytest.raises(ValueError, match="fabricated"):
        _verify_fixture(
            config,
            commands,
            fabricated_telemetry,
            event,
            trace,
        )

    gapped_trace = deepcopy(trace)
    gapped_samples = gapped_trace[0]["settle_samples"]
    assert isinstance(gapped_samples, list)
    assert isinstance(gapped_samples[0], dict)
    gapped_samples[0]["physics_step"] = 8
    with pytest.raises(ValueError, match="ordering"):
        _verify_fixture(
            config,
            commands,
            telemetry,
            event,
            gapped_trace,
        )
