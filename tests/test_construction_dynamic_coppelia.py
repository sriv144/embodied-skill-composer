from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import (
    CONSTRUCTION_FLOOR_THICKNESS_M,
    CONSTRUCTION_FLOOR_TOP_Z_M,
    DynamicCoppeliaConfig,
    DynamicCoppeliaError,
    DynamicCoppeliaExecutor,
    world_error_to_youbot_body,
    youbot_wheel_targets,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5FullCottageRunner,
    prepare_phase5_physical_yard,
    verify_phase5_artifact_bundle,
    write_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.models import Vec2
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
)
from embodied_skill_composer.construction.skill_profiles import (
    skill_profile_from_mujoco_campaign,
)
from scripts.run_construction_phase5_coppelia import _simulator_version


WORKSPACE = Path(__file__).resolve().parents[1]


class FakeDynamicSim:
    primitiveshape_cuboid = 0
    colorcomponent_ambient_diffuse = 0
    shapeintparam_static = 1
    shapeintparam_respondable = 2
    simulation_stopped = 0
    floatparam_simulation_time_step = 100
    scriptintparam_enabled = 101
    object_script_type = 102
    object_shape_type = 0
    handle_tree = -1
    handle_single = -8
    handle_scene = -2
    handle_all = -3

    def __init__(self) -> None:
        self.next_handle = 10
        self.state = self.simulation_stopped
        self.aliases: dict[int, str] = {}
        self.positions: dict[int, list[float]] = {}
        self.orientations: dict[int, list[float]] = {}
        self.trees: dict[int, list[int]] = {}
        self.target_velocities: list[tuple[int, float]] = []
        self.latest_target_velocities: dict[int, float] = {}
        self.wheel_owner: dict[int, tuple[int, str]] = {}
        self.linear_velocities: dict[int, list[float]] = {}
        self.position_writes: list[tuple[int, int, list[float]]] = []
        self.time_step: float | None = None
        self.simulation_time = 0.0
        self.int_params: list[tuple[int, int, int]] = []
        self.int_param_values: dict[tuple[int, int], int] = {}
        self.script_handles: set[int] = set()
        self.script_sources: dict[int, str] = {}
        self.script_disabled: dict[int, bool] = {}
        self.bool_property_writes: list[tuple[int, str, bool]] = []
        self.bool_properties: dict[tuple[int, str], bool] = {}
        self.collections: dict[int, list[int]] = {}
        self.shape_handles: set[int] = set()
        self.shape_dimensions: dict[int, list[float]] = {}
        self.object_matrices: dict[tuple[int, int], list[float]] = {}
        self.parents: dict[int, int] = {}
        self.removed_object_batches: list[list[int]] = []

    def _handle(self) -> int:
        self.next_handle += 1
        return self.next_handle

    def createDummy(self, _size: float) -> int:
        return self._handle()

    def createPrimitiveShape(self, _kind: int, _dimensions, _options: int) -> int:
        handle = self._handle()
        self.shape_handles.add(handle)
        self.shape_dimensions[handle] = [float(value) for value in _dimensions]
        self.bool_properties[(handle, "collidable")] = True
        return handle

    def loadModel(self, _path: str) -> int:
        root = self._handle()
        self.shape_handles.add(root)
        self.shape_dimensions[root] = [0.07, 0.15, 0.24]
        self.bool_properties[(root, "collidable")] = False
        controller = self._handle()
        self.aliases[controller] = "/robot/Script"
        self.script_handles.add(controller)
        self.script_sources[controller] = """
            function setMovement()
                sim.setJointTargetVelocity(wheelJoints[1], 1)
            end
            wheelJoints[1] = sim.getObject('../rollingJoint_fl')
        """
        self.script_disabled[controller] = False
        descendants = [controller]
        for name in ("fl", "rl", "rr", "fr"):
            handle = self._handle()
            self.aliases[handle] = f"/robot/rollingjoint_{name}"
            self.wheel_owner[handle] = (root, name)
            script = self._handle()
            self.aliases[script] = f"/robot/rollingjoint_{name}/script"
            self.script_handles.add(script)
            self.script_sources[script] = f"""
                function sysCall_actuation()
                    slipping = sim.getObject('../slippingJoint_{name}')
                    sim.setObjectOrientation(slipping, {{0, 0, 0}}, rolling)
                end
            """
            self.script_disabled[script] = False
            descendants.extend((handle, script))
            respondable_wheel = self._handle()
            self.aliases[respondable_wheel] = f"/robot/wheel_respondable_{name}"
            self.shape_handles.add(respondable_wheel)
            self.shape_dimensions[respondable_wheel] = [0.042, 0.042, 0.042]
            wheel_x = -0.02 if name in {"rl", "rr"} else 0.02
            wheel_y = -0.067 if name in {"rl", "fl"} else 0.067
            self.object_matrices[(respondable_wheel, root)] = [
                1.0,
                0.0,
                0.0,
                wheel_x,
                0.0,
                1.0,
                0.0,
                wheel_y,
                0.0,
                0.0,
                1.0,
                -0.096,
            ]
            self.bool_properties[(respondable_wheel, "collidable")] = False
            descendants.append(respondable_wheel)
        arm_script = self._handle()
        self.aliases[arm_script] = "/robot/youBotArmJoint0/Script"
        self.script_handles.add(arm_script)
        self.script_sources[arm_script] = """
            gripperJoint = sim.getObject('../youBotGripperJoint1')
            sim.setJointTargetVelocity(gripperJoint, 0.04)
            sim.setJointTargetPosition(gripperJoint, 0.0)
        """
        self.script_disabled[arm_script] = False
        descendants.append(arm_script)
        for name in ("Rectangle13", "swedishWheel_rl"):
            shape = self._handle()
            self.aliases[shape] = f"/robot/{name}"
            self.shape_handles.add(shape)
            self.shape_dimensions[shape] = [0.03, 0.04, 0.04]
            self.bool_properties[(shape, "collidable")] = True
            descendants.append(shape)
        self.trees[root] = [root, *descendants]
        self.positions[root] = [0.0, 0.0, 0.05]
        self.orientations[root] = [0.0, 0.0, 0.0]
        return root

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[handle] = alias

    def getObjectAlias(self, handle: int, options: int) -> str:
        alias = self.aliases.get(handle, f"object_{handle}")
        if options == -1:
            return alias.rsplit("/", 1)[-1]
        if options in {1, 2} and not alias.startswith("/"):
            return f"/{alias}"
        return alias

    def getObjectsInTree(
        self,
        handle: int,
        object_type: int | None = None,
        _options: int = 0,
    ):
        if handle == self.handle_scene:
            descendants = sorted(
                set(self.aliases)
                | set(self.positions)
                | set(self.orientations)
                | set(self.parents)
                | set(self.trees)
                | {
                    item
                    for tree_handles in self.trees.values()
                    for item in tree_handles
                }
            )
        else:
            collected = set(self.trees.get(handle, [handle]))
            changed = True
            while changed:
                changed = False
                for child, parent in self.parents.items():
                    if parent not in collected or child in collected:
                        continue
                    collected.update(self.trees.get(child, [child]))
                    changed = True
            descendants = sorted(collected)
        if object_type == self.object_script_type:
            return [item for item in descendants if item in self.script_handles]
        if object_type == self.object_shape_type:
            return [item for item in descendants if item in self.shape_handles]
        return descendants

    def scaleObjects(self, _handles, _scale: float, _positions_too: bool) -> None:
        pass

    def setObjectParent(self, handle: int, parent: int, _keep: bool) -> None:
        self.parents[handle] = parent

    def removeObjects(self, handles, _delayed_removal: bool = False) -> None:
        removed = {int(handle) for handle in handles}
        self.removed_object_batches.append(sorted(removed))
        for handle in removed:
            self.aliases.pop(handle, None)
            self.positions.pop(handle, None)
            self.orientations.pop(handle, None)
            self.parents.pop(handle, None)
            self.trees.pop(handle, None)
            self.script_handles.discard(handle)
            self.script_sources.pop(handle, None)
            self.script_disabled.pop(handle, None)
            self.linear_velocities.pop(handle, None)
            self.shape_handles.discard(handle)
            self.shape_dimensions.pop(handle, None)
            self.bool_properties = {
                key: value
                for key, value in self.bool_properties.items()
                if key[0] != handle
            }
        self.parents = {
            child: parent
            for child, parent in self.parents.items()
            if parent not in removed
        }
        self.trees = {
            root: [item for item in tree if item not in removed]
            for root, tree in self.trees.items()
            if root not in removed
        }

    def setObjectPosition(self, handle: int, position) -> None:
        values = list(position)
        previous = self.positions.get(handle)
        self.positions[handle] = values
        self.position_writes.append((self.state, handle, values))
        if previous is None:
            return
        delta = [values[index] - previous[index] for index in range(3)]
        pending_parents = [handle]
        while pending_parents:
            parent = pending_parents.pop()
            for child, child_parent in tuple(self.parents.items()):
                if child_parent != parent:
                    continue
                child_position = self.positions.get(child)
                if child_position is not None:
                    self.positions[child] = [
                        child_position[index] + delta[index]
                        for index in range(3)
                    ]
                pending_parents.append(child)

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
        self.int_param_values[(_handle, _parameter)] = _value

    def getObjectInt32Param(self, handle: int, parameter: int) -> int:
        return self.int_param_values.get((handle, parameter), 1)

    def getBoolProperty(self, handle: int, name: str) -> bool:
        if name == "scriptDisabled":
            return self.script_disabled[handle]
        return self.bool_properties[(handle, name)]

    def getStringProperty(self, handle: int, name: str) -> str:
        assert name == "code"
        return self.script_sources[handle]

    def setBoolProperty(self, handle: int, name: str, value: bool) -> None:
        self.bool_property_writes.append((handle, name, value))
        if name == "scriptDisabled":
            self.script_disabled[handle] = value
        else:
            self.bool_properties[(handle, name)] = value

    def checkCollision(self, _first: int, _second: int):
        if self.aliases.get(_second, "").startswith(
            "collision_monitor_self_test_"
        ):
            return 1, [self.collections[_first][0], _second]
        return 0

    def createCollection(self, _options: int) -> int:
        handle = self._handle()
        self.collections[handle] = []
        return handle

    def addItemToCollection(
        self,
        collection: int,
        _what: int,
        object_handle: int,
        _options: int,
    ) -> None:
        assert _what == self.handle_single
        self.collections[collection].append(object_handle)

    def getCollectionObjects(self, collection: int) -> list[int]:
        return list(self.collections[collection])

    def getShapeBB(self, handle: int):
        return self.shape_dimensions[handle], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def getObjectMatrix(self, handle: int, relative_to: int):
        if handle == relative_to:
            return [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]
        return self.object_matrices.get(
            (handle, relative_to),
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

    def setJointTargetVelocity(self, handle: int, velocity: float) -> None:
        self.target_velocities.append((handle, velocity))
        self.latest_target_velocities[handle] = velocity

    def getObjectVelocity(self, handle: int):
        return self.linear_velocities.get(handle, [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]

    def setFloatParam(self, _parameter: int, value: float) -> None:
        self.time_step = value

    def getSimulationState(self) -> int:
        return self.state

    def startSimulation(self) -> None:
        self.state = 1

    def stopSimulation(self) -> None:
        self.state = self.simulation_stopped

    def integrate(self, dt: float, wheel_radius_m: float = 0.2) -> None:
        self.simulation_time += dt
        roots = {owner for owner, _ in self.wheel_owner.values()}
        for root in roots:
            wheels = {
                name: self.latest_target_velocities.get(handle, 0.0)
                for handle, (owner, name) in self.wheel_owner.items()
                if owner == root
            }
            if set(wheels) != {"fl", "rl", "rr", "fr"}:
                continue
            forward = -sum(wheels.values()) * wheel_radius_m / 4
            lateral = (
                -wheels["fl"] + wheels["rl"] - wheels["rr"] + wheels["fr"]
            ) * wheel_radius_m / 4
            yaw = self.orientations.get(root, [0.0, 0.0, 0.0])[2]
            velocity_x = -math.cos(yaw) * forward - math.sin(yaw) * lateral
            velocity_y = -math.sin(yaw) * forward + math.cos(yaw) * lateral
            position = self.positions[root]
            position[0] += velocity_x * dt
            position[1] += velocity_y * dt
            self.linear_velocities[root] = [velocity_x, velocity_y, 0.0]

    def getSimulationTime(self) -> float:
        return self.simulation_time

    def saveScene(self) -> bytes:
        return b"VREP" + b"\0" * 124


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
        self.sim.integrate(self.sim.time_step or 0.05)
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
    assert executor.bundled_motion_scripts_found == 24
    assert executor.disabled_bundled_motion_scripts == 8
    assert executor.verified_disabled_bundled_motion_scripts == 8
    assert executor.retained_bundled_maintenance_scripts == 16
    assert executor.verified_enabled_maintenance_scripts == 16
    assert executor.disabled_wheel_command_scripts == 4
    assert executor.disabled_arm_gripper_scripts == 4
    script_property_writes = [
        item
        for item in fake.sim.bool_property_writes
        if item[1] == "scriptDisabled"
    ]
    assert len(script_property_writes) == 24
    for handle, _name, _disabled in script_property_writes:
        alias = fake.sim.aliases[handle].casefold()
        expected_disabled = (
            alias.endswith("/script")
            and "rollingjoint_" not in alias
        )
        assert fake.sim.script_disabled[handle] is expected_disabled
    assert all(
        item["role_counts"]
        == {
            "wheel_command_writer": 1,
            "arm_gripper_writer": 1,
            "passive_omniwheel_maintenance": 4,
        }
        for item in executor.script_control_audit
    )
    floor_handle = next(
        handle
        for handle, alias in fake.sim.aliases.items()
        if alias == "construction_intelligence_floor"
    )
    floor_center_z = fake.sim.positions[floor_handle][2]
    assert floor_center_z == pytest.approx(
        CONSTRUCTION_FLOOR_TOP_Z_M - CONSTRUCTION_FLOOR_THICKNESS_M / 2
    )
    assert floor_center_z + CONSTRUCTION_FLOOR_THICKNESS_M / 2 == pytest.approx(0.0)
    assert any(item[1:] == (fake.sim.shapeintparam_respondable, 1) for item in fake.sim.int_params)
    assert all(set(wheels) == {"fl", "rl", "rr", "fr"} for wheels in executor.wheel_handles.values())
    assert executor.robot_base_contact_gate_passed
    assert len(executor.robot_base_physics_audit) == 4
    assert all(
        len(item["allowed_shape_handles"]) == 5
        and item["excluded_shape_handles"]
        and item["footprint_gate_passed"] is True
        for item in executor.robot_base_physics_audit
    )
    assert all(
        fake.sim.collections[executor.robot_collision_entities[robot_id]]
        == sorted(executor.robot_base_shape_handles[robot_id])
        for robot_id in executor.robot_handles
    )

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


def test_synchronized_carry_holds_faster_base_at_shared_waypoint(plan) -> None:
    fake = FakeDynamicClient()
    config = DynamicCoppeliaConfig(
        settle_steps=0,
        maximum_wheel_speed=10.0,
        effective_wheel_radius_m=0.2,
        position_gain=5.0,
        waypoint_tolerance_m=0.025,
        formation_tolerance_m=0.3,
        safety_distance_m=0.05,
        max_steps_per_waypoint=1_000,
    )
    executor = DynamicCoppeliaExecutor(
        plan,
        config=config,
        client_factory=lambda _config: fake,
    )
    executor.connect()
    fast_id, slow_id = "robot_1", "robot_2"
    slow_handle = executor.robot_handles[slow_id]
    original_integrate = fake.sim.integrate

    def integrate_with_slow_base(
        dt: float,
        wheel_radius_m: float = 0.2,
    ) -> None:
        before = list(fake.sim.positions[slow_handle])
        original_integrate(dt, wheel_radius_m)
        after = fake.sim.positions[slow_handle]
        fake.sim.positions[slow_handle] = [
            before[index] + 0.25 * (after[index] - before[index])
            for index in range(3)
        ]

    fake.sim.integrate = integrate_with_slow_base  # type: ignore[method-assign]
    starts = {
        robot_id: Vec2(
            x=fake.sim.positions[executor.robot_handles[robot_id]][0],
            y=fake.sim.positions[executor.robot_handles[robot_id]][1],
        )
        for robot_id in (fast_id, slow_id)
    }
    routes = {
        robot_id: [
            Vec2(x=start.x + delta, y=start.y)
            for delta in (0.0, 0.1, 0.2, 0.3)
        ]
        for robot_id, start in starts.items()
    }

    executor.start()
    executor.follow_synchronized_routes(routes)
    executor.stop()

    held_targets = [
        command
        for command in executor.commands
        if command.robot_id == fast_id
        and command.source == "formation_hold"
        and command.target_position is not None
    ]
    assert any(
        slow.robot_id == slow_id
        and slow.source == "path_follower"
        and slow.timestamp_s == held.timestamp_s
        and slow.target_position is not None
        and (
            slow.target_position.x - starts[slow_id].x
            == pytest.approx(
                held.target_position.x - starts[fast_id].x
            )
        )
        for held in held_targets
        for slow in executor.commands
    )
    assert max(executor.synchronized_formation_errors_m) <= (
        config.formation_tolerance_m
    )
    for robot_id, route in routes.items():
        measured = fake.sim.positions[executor.robot_handles[robot_id]]
        assert math.hypot(
            measured[0] - route[-1].x,
            measured[1] - route[-1].y,
        ) <= config.waypoint_tolerance_m
    assert executor.post_start_robot_pose_writes == 0


def test_dynamic_executor_replaces_only_its_prior_generated_scene(plan) -> None:
    fake = FakeDynamicClient()
    first = DynamicCoppeliaExecutor(plan, client_factory=lambda _config: fake)
    first.connect()
    first_root = first.root_handle
    assert first_root is not None
    unrelated = fake.sim.createDummy(0.01)
    fake.sim.setObjectAlias(unrelated, "user-owned-scene-object")

    second = DynamicCoppeliaExecutor(plan, client_factory=lambda _config: fake)
    second.connect()

    assert second.prior_generated_scene_root_count == 1
    assert second.prior_generated_scene_object_count > 1
    assert second.generated_scene_cleanup_verified
    assert first_root not in fake.sim.aliases
    assert fake.sim.aliases[unrelated] == "user-owned-scene-object"
    generated_roots = [
        handle
        for handle, alias in fake.sim.aliases.items()
        if alias == "ESCConstructionIntelligenceV1"
    ]
    assert generated_roots == [second.root_handle]
    assert all(
        fake.sim.parents[carrier] == second.root_handle
        for carrier in second.payload_carriers.values()
    )
    assert any(
        item["event"] == "prior_generated_scene_cleanup"
        and item["removed_root_count"] == 1
        and item["verified"] is True
        for item in second.runtime_events
    )


def test_dynamic_executor_rejects_indeterminate_script_disable_readback(plan) -> None:
    fake = FakeDynamicClient()

    def indeterminate_readback(_handle: int, _name: str) -> None:
        return None

    fake.sim.getBoolProperty = indeterminate_readback  # type: ignore[method-assign]
    executor = DynamicCoppeliaExecutor(plan, client_factory=lambda _config: fake)

    with pytest.raises(DynamicCoppeliaError, match="indeterminate disabled state"):
        executor.connect()


def test_dynamic_executor_rejects_unclassified_retained_script(plan) -> None:
    fake = FakeDynamicClient()
    original_load_model = fake.sim.loadModel

    def load_model_with_unknown_script(path: str) -> int:
        root = original_load_model(path)
        handle = fake.sim._handle()
        fake.sim.aliases[handle] = "/robot/unknown/Script"
        fake.sim.script_handles.add(handle)
        fake.sim.script_sources[handle] = "function sysCall_actuation() end"
        fake.sim.script_disabled[handle] = False
        fake.sim.trees[root].append(handle)
        return root

    fake.sim.loadModel = load_model_with_unknown_script  # type: ignore[method-assign]
    executor = DynamicCoppeliaExecutor(plan, client_factory=lambda _config: fake)

    with pytest.raises(DynamicCoppeliaError, match="unrecognized control role"):
        executor.connect()


def test_dynamic_executor_rejects_undersized_planned_base_footprint(plan) -> None:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(planned_robot_footprint_radius_m=0.05),
        client_factory=lambda _config: fake,
    )

    with pytest.raises(DynamicCoppeliaError, match="exceeds planned radius"):
        executor.connect()


def test_dynamic_executor_rejects_blind_base_collision_collection(plan) -> None:
    fake = FakeDynamicClient()
    fake.sim.checkCollision = lambda _first, _second: 0  # type: ignore[method-assign]
    executor = DynamicCoppeliaExecutor(
        plan,
        client_factory=lambda _config: fake,
    )

    with pytest.raises(DynamicCoppeliaError, match="positive self-test"):
        executor.connect()


def test_dynamic_executor_rejects_collision_from_excluded_youbot_shape(plan) -> None:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
        client_factory=lambda _config: fake,
    )
    executor.connect()
    robot_id = plan.robots[0].robot_id
    collection = executor.robot_world_collision_entity
    world = executor.world_collision_entity
    assert collection is not None
    assert world is not None
    module_handle = executor.module_handles[plan.modules[0].module_id]
    excluded_handle = min(executor.robot_excluded_shape_handles[robot_id])

    def excluded_collision(first: int, second: int):
        if first == collection and second == world:
            return 1, [excluded_handle, module_handle]
        return 0

    fake.sim.checkCollision = excluded_collision  # type: ignore[method-assign]
    executor.start()
    with pytest.raises(DynamicCoppeliaError, match="excluded YouBot geometry"):
        executor._step_physics()
    executor.stop()


def test_phase5_verifier_rejects_tampered_base_physics_readback(plan) -> None:
    import embodied_skill_composer.construction.coppelia_phase5 as phase5

    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        client_factory=lambda _config: fake,
    )
    executor.connect()
    diagnostics = deepcopy(executor.diagnostics())
    audit = diagnostics["robot_base_physics_audit"]
    assert isinstance(audit, list)
    first_robot = audit[0]
    assert isinstance(first_robot, dict)
    records = first_robot["shape_records"]
    assert isinstance(records, list)
    excluded = next(
        record
        for record in records
        if record["role"] == "excluded_v1_geometry"
    )
    excluded["after"]["collidable"] = True
    diagnostics["robot_base_physics_audit_sha256"] = phase5._sha256_json(
        audit
    )

    with pytest.raises(ValueError, match="policy readback"):
        phase5._verify_robot_base_physics_audit(
            diagnostics=diagnostics,
            trace=executor.runtime_events,
            robot_ids={robot.robot_id for robot in plan.robots},
            planned_radius_m=executor.config.planned_robot_footprint_radius_m,
        )


def test_logically_attached_module_collision_is_never_exempted(plan) -> None:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
        client_factory=lambda _config: fake,
    )
    executor.connect()
    executor.start()
    module_id = plan.modules[0].module_id
    robot_id = plan.robots[0].robot_id
    robot_collection = executor.robot_world_collision_entity
    world_collection = executor.world_collision_entity
    assert robot_collection is not None
    assert world_collection is not None
    module_handle = executor.module_handles[module_id]

    def attached_module_collision(first: int, second: int):
        if first == robot_collection and second == world_collection:
            return 1, [
                min(executor.robot_base_shape_handles[robot_id]),
                module_handle,
            ]
        return 0

    fake.sim.checkCollision = attached_module_collision  # type: ignore[method-assign]
    executor.logical_attachments[module_id] = [robot_id]
    executor._step_physics()
    executor.stop()

    assert executor.permitted_logical_payload_overlaps == 0
    assert executor.physical_collision_stops == 1
    assert executor.physical_collision_events[-1]["entity_2"] == (
        f"module:{module_id}"
    )
    assert (
        executor.physical_collision_events[-1][
            "permitted_logical_payload_overlap"
        ]
        is False
    )


def test_collision_monitor_uses_bounded_collection_query_groups(plan) -> None:
    fake = FakeDynamicClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
        client_factory=lambda _config: fake,
    )
    executor.connect()

    expected_group_count = math.comb(len(plan.robots), 2) + 1
    expected_logical_pairs = (
        math.comb(len(plan.robots), 2)
        + len(plan.robots) * len(plan.modules)
        + len(plan.robots) * len(plan.site_grid.obstacle_cells)
    )
    ready = next(
        item
        for item in executor.runtime_events
        if item["event"] == "collision_monitor_ready"
    )
    assert len(executor.collision_pairs) == expected_group_count
    assert ready["expected_queries_per_step"] == expected_group_count
    assert len(ready["query_group_inventory"]) == expected_group_count
    assert sum(executor.collision_pair_category_counts.values()) == (
        expected_logical_pairs
    )

    executor.start()
    executor._step_physics()
    executor.stop()

    assert executor.collision_query_count == expected_group_count
    assert executor.collision_query_rounds == 1
    assert executor.diagnostics()[
        "collision_queries_cover_every_physics_step"
    ] is True


def test_remote_step_handshake_reconciles_only_a_measured_completed_step(plan) -> None:
    class CompletedStepHandshakeClient(FakeDynamicClient):
        def step(self) -> None:
            super().step()
            raise RuntimeError("No such function: _*executed*_")

    fake = CompletedStepHandshakeClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
        client_factory=lambda _config: fake,
    )
    executor.connect()
    executor.start()
    executor._step_physics()
    executor.stop()

    assert executor.physics_steps == 1
    assert executor.remote_step_handshake_reconciliations == 1
    assert executor.collision_query_rounds == 1
    event = next(
        item
        for item in executor.runtime_events
        if item["event"] == "remote_step_handshake_reconciled"
    )
    assert event["physics_step"] == 1
    assert event["observed_simulation_time_s"] == pytest.approx(0.05)
    assert event["expected_simulation_time_s"] == pytest.approx(0.05)
    assert executor.diagnostics()["remote_step_handshake_reconciliations"] == 1


def test_remote_step_handshake_fails_closed_without_measured_advance(plan) -> None:
    class UnadvancedStepHandshakeClient(FakeDynamicClient):
        def step(self) -> None:
            raise RuntimeError("No such function: _*executed*_")

    fake = UnadvancedStepHandshakeClient()
    executor = DynamicCoppeliaExecutor(
        plan,
        config=DynamicCoppeliaConfig(settle_steps=0),
        client_factory=lambda _config: fake,
    )
    executor.connect()
    executor.start()
    with pytest.raises(
        DynamicCoppeliaError,
        match="without exactly one measured simulator step",
    ):
        executor._step_physics()
    executor.stop()

    assert executor.physics_steps == 0
    assert executor.remote_step_handshake_reconciliations == 0
    assert executor.collision_query_rounds == 0


def test_remote_step_handshake_evidence_is_bound_to_measured_time() -> None:
    from embodied_skill_composer.construction import coppelia_phase5 as phase5

    diagnostics = {
        "remote_step_handshake_reconciliations": 1,
        "executor_config": DynamicCoppeliaConfig().model_dump(mode="json"),
    }
    trace: list[object] = [
        {
            "timestamp_s": 0.05,
            "event": "remote_step_handshake_reconciled",
            "physics_step": 1,
            "observed_simulation_time_s": 0.05,
            "expected_simulation_time_s": 0.05,
            "error": "No such function: _*executed*_",
        }
    ]
    phase5._verify_remote_step_handshake_reconciliations(
        trace=trace,
        diagnostics=diagnostics,
        physics_steps=2,
    )

    tampered = deepcopy(trace)
    assert isinstance(tampered[0], dict)
    tampered[0]["observed_simulation_time_s"] = 0.1
    with pytest.raises(
        ValueError,
        match="reconciliation trace is invalid",
    ):
        phase5._verify_remote_step_handshake_reconciliations(
            trace=tampered,
            diagnostics=diagnostics,
            physics_steps=2,
        )


def test_phase5_simulator_version_uses_definitive_current_integer_api() -> None:
    class CurrentVersionSim:
        intparam_program_version = 1
        intparam_program_revision = 30

        @staticmethod
        def getInt32Param(parameter: int) -> int:
            return {1: 41000, 30: 0}[parameter]

    class IndeterminateVersionSim(CurrentVersionSim):
        @staticmethod
        def getInt32Param(_parameter: int) -> None:
            return None

    assert _simulator_version(CurrentVersionSim()) == "4.10.0 rev 0"
    assert _simulator_version(IndeterminateVersionSim()) is None


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


def _phase5_scenario():
    design = load_house_design(WORKSPACE / "configs" / "construction" / "cottage_v1.yaml")
    generated = generate_cottage_scenario(
        900,
        design,
        config=CottageScenarioConfig(
            widths_m=(6.0,),
            depths_m=(6.0,),
            interior_panel_range=(0, 0),
            obstacle_count_range=(0, 0),
        ),
    )
    return prepare_phase5_physical_yard(generated)


def _phase5_executor(scenario):
    fake = FakeDynamicClient()
    config = DynamicCoppeliaConfig(
        control_hz=5,
        settle_steps=0,
        maximum_wheel_speed=10.0,
        effective_wheel_radius_m=0.2,
        position_gain=5.0,
        waypoint_tolerance_m=0.22,
        formation_tolerance_m=0.5,
        install_tolerance_m=0.3,
        safety_distance_m=0.05,
        max_steps_per_waypoint=1_000,
    )
    executor = DynamicCoppeliaExecutor(
        scenario.plan,
        config=config,
        client_factory=lambda _config: fake,
    )
    executor.connect()
    return executor


def test_phase5_physical_yard_is_deterministic_clear_and_hash_pinned() -> None:
    design = load_house_design(
        WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
    )
    generated = generate_cottage_scenario(900, design)
    original = generated.model_copy(deep=True)

    first, first_manifest = prepare_phase5_physical_yard(generated)
    second, second_manifest = prepare_phase5_physical_yard(generated)

    assert first == second
    assert first_manifest == second_manifest
    assert generated == original
    assert first_manifest.configuration.phase5_team_size == 2
    assert first_manifest.initial_clearance_preflight_passed
    assert first_manifest.sequential_route_preflight_passed
    assert first_manifest.all_robots_active_preflight_passed
    assert first_manifest.source_plan_digest != (
        first_manifest.transformed_plan_digest
    )
    assert first_manifest.maximum_preflight_formation_offset_m > 0
    maximum_configured_offset = max(
        math.hypot(
            module.dimensions.width / 2,
            module.dimensions.depth / 2,
        )
        + first_manifest.configuration.formation_clearance_m
        + first_manifest.configuration.maximum_formation_expansion_m
        for module in first.plan.modules
    )
    assert (
        first_manifest.maximum_preflight_formation_offset_m
        <= maximum_configured_offset
    )
    assert first.plan.site_grid.obstacle_cells

    staging_bounds = []
    for module in first.plan.modules:
        position = module.staging_pose.position
        assert module.staging_pose.rotation_rpy_degrees.z == pytest.approx(
            module.target_pose.rotation_rpy_degrees.z
        )
        yaw = math.radians(
            module.staging_pose.rotation_rpy_degrees.z
        )
        half_x = (
            abs(math.cos(yaw)) * module.dimensions.width / 2
            + abs(math.sin(yaw)) * module.dimensions.depth / 2
        )
        half_y = (
            abs(math.sin(yaw)) * module.dimensions.width / 2
            + abs(math.cos(yaw)) * module.dimensions.depth / 2
        )
        staging_bounds.append(
            (
                module.module_id,
                position.x - half_x,
                position.x + half_x,
                position.y - half_y,
                position.y + half_y,
            )
        )
    for index, left in enumerate(staging_bounds):
        for right in staging_bounds[index + 1 :]:
            assert not (
                left[1] < right[2]
                and right[1] < left[2]
                and left[3] < right[4]
                and right[3] < left[4]
            ), (left[0], right[0])
    for robot in first.plan.robots:
        for module in first.plan.modules:
            assert not (
                abs(
                    robot.start_pose.position.x
                    - module.staging_pose.position.x
                )
                <= module.dimensions.width / 2
                and abs(
                    robot.start_pose.position.y
                    - module.staging_pose.position.y
                )
                <= module.dimensions.depth / 2
            )


def test_phase5_runner_rejects_tampered_physical_yard_manifest() -> None:
    scenario, physical_yard = _phase5_scenario()
    executor = DynamicCoppeliaExecutor(
        scenario.plan,
        client_factory=lambda _config: FakeDynamicClient(),
    )
    tampered = physical_yard.model_copy(
        update={"transformed_plan_digest": "0" * 64}
    )

    with pytest.raises(ValueError, match="does not match"):
        Phase5FullCottageRunner(
            executor,
            scenario,
            physical_yard=tampered,
        )


def test_phase5_nominal_full_cottage_offline_harness_proves_invariants(
    tmp_path: Path,
) -> None:
    scenario, physical_yard = _phase5_scenario()
    executor = _phase5_executor(scenario)
    result = Phase5FullCottageRunner(
        executor,
        scenario,
        physical_yard=physical_yard,
    ).run("nominal")

    assert result.status == "completed"
    assert not result.live_gate_passed
    assert result.evidence_kind == "offline_harness"
    assert len(result.installed_module_ids) == len(scenario.plan.modules)
    assert len(result.replay) == len(scenario.plan.modules)
    assert all(result.acceptance.values())
    assert result.acceptance["zero_nominal_collision_stops"]
    assert executor.post_start_robot_pose_writes == 0
    assert all(item.measured_poses for item in result.replay)
    assert all(
        item.planned_start_s <= item.planned_pickup_s <= item.planned_end_s
        and item.started_at_s <= item.pickup_at_s <= item.installed_at_s
        for item in result.replay
    )
    modules = {module.module_id: module for module in scenario.plan.modules}
    robot_radius = physical_yard.configuration.robot_footprint_radius_m
    required_payload_clearance = (
        robot_radius + physical_yard.configuration.route_clearance_m
    )
    for item in result.replay:
        module = modules[item.module_id]
        center = module.staging_pose.position
        yaw = math.radians(
            module.staging_pose.rotation_rpy_degrees.z
        )
        half_x = (
            abs(math.cos(yaw)) * module.dimensions.width / 2
            + abs(math.sin(yaw)) * module.dimensions.depth / 2
        )
        half_y = (
            abs(math.sin(yaw)) * module.dimensions.width / 2
            + abs(math.cos(yaw)) * module.dimensions.depth / 2
        )
        minimum_x = center.x - half_x - robot_radius
        maximum_x = center.x + half_x + robot_radius
        minimum_y = center.y - half_y - robot_radius
        maximum_y = center.y + half_y + robot_radius
        for route in item.approach_routes.values():
            assert all(
                not (
                    minimum_x <= waypoint.x <= maximum_x
                    and minimum_y <= waypoint.y <= maximum_y
                )
                for waypoint in route
            ), item.module_id
        assert {
            len(route) for route in item.assigned_carry_routes.values()
        } == {len(item.carry_route)}
        assert all(
            math.dist(
                (left.x, left.y),
                (right.x, right.y),
            )
            <= physical_yard.configuration.carry_sample_spacing_m + 1e-9
            for left, right in zip(
                item.carry_route,
                item.carry_route[1:],
                strict=False,
            )
        )
        for robot_id, route in item.assigned_carry_routes.items():
            offset = item.formation_offsets[robot_id]
            assert all(
                assigned.x
                == pytest.approx(
                    carrier.x
                    - item.logical_carrier_offset.x
                    + offset.x
                )
                and assigned.y
                == pytest.approx(
                    carrier.y
                    - item.logical_carrier_offset.y
                    + offset.y
                )
                for carrier, assigned in zip(
                    item.carry_route,
                    route,
                    strict=True,
                )
            )
        assert (
            item.minimum_planned_payload_clearance_m
            >= required_payload_clearance
        )
        assert (
            item.minimum_measured_payload_clearance_m
            >= required_payload_clearance
        )

    bundle_dir = tmp_path / "nominal"
    manifest = write_phase5_artifact_bundle(
        bundle_dir,
        run_id="offline-nominal-test",
        result=result,
        scenario=scenario,
        executor=executor,
        source_commit="1" * 40,
        source_tree_digest="2" * 64,
        source_dirty=False,
        approval_gate_confirmed=False,
    )
    verified = verify_phase5_artifact_bundle(bundle_dir)
    assert not manifest.live_evidence
    assert not manifest.approval_gate_confirmed
    assert verified == manifest
    assert (bundle_dir / "construction_intelligence.ttt").is_file()
    report = (bundle_dir / "report.md").read_text(encoding="utf-8")
    assert "does not claim arm motion" in report

    live_metrics = dict(result.metrics)
    live_metrics.update(
        {
            "payload_transport": "logical_carrier",
            "live_evidence": True,
            "live_gate_passed": True,
        }
    )
    live_result = result.model_copy(
        update={
            "evidence_kind": "live_coppelia",
            "live_gate_passed": True,
            "metrics": live_metrics,
        }
    )
    import embodied_skill_composer.construction.coppelia_phase5 as phase5

    phase5._verify_passing_phase5_evidence(
        scenario=scenario,
        result=live_result,
        replay=live_result.replay,
        commands=executor.commands,
        telemetry=executor.telemetry,
        trace=live_result.trace,
        config=executor.config,
        physical_yard=physical_yard,
    )
    live_dir = tmp_path / "synthetic-live-nominal"
    with pytest.raises(ValueError, match="runtime-origin attestation"):
        write_phase5_artifact_bundle(
            live_dir,
            run_id="synthetic-live-nominal-test",
            result=live_result,
            scenario=scenario,
            executor=executor,
            source_commit="1" * 40,
            source_tree_digest="2" * 64,
            source_dirty=False,
            approval_gate_confirmed=True,
            simulator_version="CoppeliaSim coherent test fixture",
        )
    assert not live_dir.exists()

    replay_path = bundle_dir / "planned_vs_measured_replay.json"
    replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
    robot_id = replay_payload[0]["executed_robot_ids"][0]
    replay_payload[0]["assigned_carry_routes"][robot_id][1]["x"] += 0.05
    replay_path.write_text(
        json.dumps(replay_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = bundle_dir / "manifest.json"
    manifest_payload = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    artifact = next(
        item
        for item in manifest_payload["artifacts"]
        if item["path"] == replay_path.name
    )
    artifact["bytes"] = replay_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(
        replay_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rigidly derived"):
        verify_phase5_artifact_bundle(bundle_dir)


def test_phase5_recovery_disables_after_quarter_and_reassigns_remaining_work(
    tmp_path: Path,
) -> None:
    scenario, physical_yard = _phase5_scenario()
    executor = _phase5_executor(scenario)
    result = Phase5FullCottageRunner(
        executor,
        scenario,
        physical_yard=physical_yard,
    ).run(
        "unavailable_robot_recovery"
    )

    assert result.status == "completed"
    assert result.recovery is not None
    assert result.recovery.completion_fraction_at_disable >= 0.25
    assert result.recovery.commands_after_stop == 0
    assert result.recovery.reassigned_job_ids
    assert len(result.installed_module_ids) == len(scenario.plan.modules)
    assert all(result.acceptance.values())
    disabled = result.recovery.robot_id
    cutoff = result.recovery.stop_command_index
    assert not any(
        command.robot_id == disabled for command in executor.commands[cutoff:]
    )
    assert any(
        item.reassigned_after_unavailability
        and disabled in item.planned_robot_ids
        and disabled not in item.executed_robot_ids
        for item in result.replay
    )
    settled = executor.disabled_settle_samples[disabled][-1].measured_pose.position
    disabled_clearance = (
        2 * physical_yard.configuration.robot_footprint_radius_m
        + physical_yard.configuration.route_clearance_m
    )
    post_disable_replays = [
        item
        for item in result.replay
        if item.pickup_at_s >= result.recovery.disabled_at_s
    ]
    assert post_disable_replays
    assert all(
        math.hypot(
            waypoint.x - settled.x,
            waypoint.y - settled.y,
        )
        >= disabled_clearance
        for item in post_disable_replays
        for route in item.assigned_carry_routes.values()
        for waypoint in route
    )

    executor.started = True
    with pytest.raises(DynamicCoppeliaError, match="wheel commands are forbidden"):
        executor.command_body_velocity(disabled, 0.1, 0.0, 0.0)
    executor.started = False

    live_metrics = dict(result.metrics)
    live_metrics.update(
        {
            "payload_transport": "logical_carrier",
            "live_evidence": True,
            "live_gate_passed": True,
        }
    )
    live_result = result.model_copy(
        update={
            "evidence_kind": "live_coppelia",
            "live_gate_passed": True,
            "metrics": live_metrics,
        }
    )
    live_dir = tmp_path / "synthetic-live-recovery"
    with pytest.raises(ValueError, match="runtime-origin attestation"):
        write_phase5_artifact_bundle(
            live_dir,
            run_id="synthetic-live-recovery-test",
            result=live_result,
            scenario=scenario,
            executor=executor,
            source_commit="1" * 40,
            source_tree_digest="2" * 64,
            source_dirty=False,
            approval_gate_confirmed=True,
            simulator_version="CoppeliaSim coherent test fixture",
        )
    assert not live_dir.exists()


def test_fake_client_cannot_be_labeled_as_live_coppelia_evidence() -> None:
    scenario, _physical_yard = _phase5_scenario()
    executor = _phase5_executor(scenario)
    with pytest.raises(ValueError, match="cannot be attested as live"):
        Phase5FullCottageRunner(
            executor,
            scenario,
            evidence_kind="live_coppelia",
        )
