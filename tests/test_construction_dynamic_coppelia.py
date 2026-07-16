from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaExecutor,
    world_error_to_youbot_body,
    youbot_wheel_targets,
)
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.skill_profiles import (
    skill_profile_from_mujoco_campaign,
)


WORKSPACE = Path(__file__).resolve().parents[1]


class FakeDynamicSim:
    primitiveshape_cuboid = 0
    colorcomponent_ambient_diffuse = 0
    shapeintparam_static = 1
    shapeintparam_respondable = 2
    simulation_stopped = 0
    floatparam_simulation_time_step = 100
    scriptintparam_enabled = 101

    def __init__(self) -> None:
        self.next_handle = 10
        self.state = self.simulation_stopped
        self.aliases: dict[int, str] = {}
        self.positions: dict[int, list[float]] = {}
        self.orientations: dict[int, list[float]] = {}
        self.trees: dict[int, list[int]] = {}
        self.target_velocities: list[tuple[int, float]] = []
        self.position_writes: list[tuple[int, int, list[float]]] = []
        self.time_step: float | None = None
        self.int_params: list[tuple[int, int, int]] = []

    def _handle(self) -> int:
        self.next_handle += 1
        return self.next_handle

    def createDummy(self, _size: float) -> int:
        return self._handle()

    def createPrimitiveShape(self, _kind: int, _dimensions, _options: int) -> int:
        return self._handle()

    def loadModel(self, _path: str) -> int:
        root = self._handle()
        controller = self._handle()
        self.aliases[controller] = "/robot/Script"
        descendants = [controller]
        for name in ("fl", "rl", "rr", "fr"):
            handle = self._handle()
            self.aliases[handle] = f"/robot/rollingjoint_{name}"
            script = self._handle()
            self.aliases[script] = f"/robot/rollingjoint_{name}/script"
            descendants.extend((handle, script))
        self.trees[root] = [root, *descendants]
        self.positions[root] = [0.0, 0.0, 0.05]
        self.orientations[root] = [0.0, 0.0, 0.0]
        return root

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[handle] = alias

    def getObjectAlias(self, handle: int, _options: int) -> str:
        return self.aliases.get(handle, f"object_{handle}")

    def getObjectsInTree(self, handle: int):
        return self.trees.get(handle, [handle])

    def scaleObjects(self, _handles, _scale: float, _positions_too: bool) -> None:
        pass

    def setObjectParent(self, _handle: int, _parent: int, _keep: bool) -> None:
        pass

    def setObjectPosition(self, handle: int, position) -> None:
        values = list(position)
        self.positions[handle] = values
        self.position_writes.append((self.state, handle, values))

    def getObjectPosition(self, handle: int):
        return self.positions.get(handle, [0.0, 0.0, 0.05])

    def setObjectOrientation(self, handle: int, orientation) -> None:
        self.orientations[handle] = list(orientation)

    def getObjectOrientation(self, handle: int):
        return self.orientations.get(handle, [0.0, 0.0, 0.0])

    def setShapeColor(self, _handle: int, _name, _component: int, _color) -> None:
        pass

    def setObjectInt32Param(self, _handle: int, _parameter: int, _value: int) -> None:
        self.int_params.append((_handle, _parameter, _value))

    def setJointTargetVelocity(self, handle: int, velocity: float) -> None:
        self.target_velocities.append((handle, velocity))

    def setFloatParam(self, _parameter: int, value: float) -> None:
        self.time_step = value

    def getSimulationState(self) -> int:
        return self.state

    def startSimulation(self) -> None:
        self.state = 1

    def stopSimulation(self) -> None:
        self.state = self.simulation_stopped


class FakeDynamicClient:
    def __init__(self) -> None:
        self.sim = FakeDynamicSim()
        self.stepping = False
        self.steps = 0

    def require(self, name: str):
        assert name == "sim"
        return self.sim

    def setStepping(self, enabled: bool) -> None:
        self.stepping = enabled

    def step(self) -> None:
        assert self.stepping
        self.steps += 1


@pytest.fixture(scope="module")
def plan():
    design = load_house_design(WORKSPACE / "configs" / "construction" / "cottage_v1.yaml")
    return compile_house_design(design)


def test_youbot_wheel_mapping_matches_coppelia_model_convention() -> None:
    assert youbot_wheel_targets(1.0, 0.0, 0.0, maximum=5.0) == (-1.0, -1.0, -1.0, -1.0)
    assert youbot_wheel_targets(0.0, 1.0, 0.0, maximum=5.0) == (-1.0, 1.0, -1.0, 1.0)
    assert max(abs(value) for value in youbot_wheel_targets(8, 4, 2, maximum=5)) == 5
    forward, lateral = world_error_to_youbot_body(0.0, 1.0, -math.pi / 2)
    assert forward == pytest.approx(1.0)
    assert lateral == pytest.approx(0.0)


def test_dynamic_executor_commands_wheels_without_post_start_pose_sync(plan) -> None:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(plan, client_factory=lambda _config: fake)
    executor.connect()
    assert executor.initial_robot_pose_writes == 4
    assert executor.disabled_bundled_motion_scripts == 4
    assert sum(item[1:] == (fake.sim.scriptintparam_enabled, 0) for item in fake.sim.int_params) == 4
    assert any(item[1:] == (fake.sim.shapeintparam_respondable, 1) for item in fake.sim.int_params)
    assert all(set(wheels) == {"fl", "rl", "rr", "fr"} for wheels in executor.wheel_handles.values())

    executor.start()
    command = executor.command_body_velocity("robot_1", 0.5, -0.2, 0.1)
    telemetry = executor.sample_telemetry("robot_1")
    executor.stop()

    assert fake.sim.time_step == pytest.approx(0.05)
    assert fake.steps == 10
    assert len(fake.sim.target_velocities) >= 8
    assert command.robot_id == "robot_1"
    assert telemetry.measured_pose.position.z == pytest.approx(0.12)
    assert executor.post_start_robot_pose_writes == 0
    robot_handles = set(executor.robot_handles.values())
    assert not any(
        state != fake.sim.simulation_stopped and handle in robot_handles
        for state, handle, _ in fake.sim.position_writes
    )


def test_mujoco_campaign_becomes_honest_skill_profile(tmp_path: Path) -> None:
    campaign = {
        "episodes": [
            {
                "steps": [
                    {
                        "observation": {
                            "physical_feedback": {
                                "last_check_phase": "grasp",
                                "last_check_passed": True,
                                "current_alignment_error_m": 0.004,
                                "last_contact_forces_n": {"left": 28.0, "right": 31.0},
                            }
                        }
                    },
                    {
                        "observation": {
                            "physical_feedback": {
                                "last_check_phase": "install",
                                "last_check_passed": False,
                                "current_alignment_error_m": 0.025,
                                "last_contact_forces_n": {"left": 19.0},
                            }
                        }
                    },
                ]
            }
        ]
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(campaign), encoding="utf-8")
    profile = skill_profile_from_mujoco_campaign(path)
    wall = profile.by_module_type["wall_panel"]
    assert profile.source_backend == "mujoco"
    assert wall.success_rate == 0.5
    assert wall.sample_count == 2
    assert wall.peak_force_mean_n == 25.0
    assert any("Duration is" in note for note in profile.notes)
