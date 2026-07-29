from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from embodied_skill_composer.construction.compiler import compile_house_design
from embodied_skill_composer.construction.coppelia_dynamic import (
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
    handle_tree = -1
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
        self.int_params: list[tuple[int, int, int]] = []
        self.int_param_values: dict[tuple[int, int], int] = {}
        self.script_handles: set[int] = set()
        self.script_sources: dict[int, str] = {}
        self.script_disabled: dict[int, bool] = {}
        self.bool_property_writes: list[tuple[int, str, bool]] = []
        self.collections: dict[int, int] = {}
        self.parents: dict[int, int] = {}
        self.removed_object_batches: list[list[int]] = []

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
        self.trees[root] = [root, *descendants]
        self.positions[root] = [0.0, 0.0, 0.05]
        self.orientations[root] = [0.0, 0.0, 0.0]
        return root

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[handle] = alias

    def getObjectAlias(self, handle: int, _options: int) -> str:
        return self.aliases.get(handle, f"object_{handle}")

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
        self.int_param_values[(_handle, _parameter)] = _value

    def getObjectInt32Param(self, handle: int, parameter: int) -> int:
        return self.int_param_values.get((handle, parameter), 1)

    def getBoolProperty(self, handle: int, name: str) -> bool:
        assert name == "scriptDisabled"
        return self.script_disabled[handle]

    def getStringProperty(self, handle: int, name: str) -> str:
        assert name == "code"
        return self.script_sources[handle]

    def setBoolProperty(self, handle: int, name: str, value: bool) -> None:
        assert name == "scriptDisabled"
        self.bool_property_writes.append((handle, name, value))
        self.script_disabled[handle] = value

    def checkCollision(self, _first: int, _second: int) -> int:
        return 0

    def createCollection(self, _options: int) -> int:
        handle = self._handle()
        self.collections[handle] = -1
        return handle

    def addItemToCollection(
        self,
        collection: int,
        _what: int,
        object_handle: int,
        _options: int,
    ) -> None:
        self.collections[collection] = object_handle

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
    assert len(fake.sim.bool_property_writes) == 24
    for handle, _name, _disabled in fake.sim.bool_property_writes:
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
    robot_collection = executor.robot_collision_entities[robot_id]
    module_handle = executor.module_handles[module_id]

    def attached_module_collision(first: int, second: int) -> int:
        return int(first == robot_collection and second == module_handle)

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
    assert first.plan.site_grid.obstacle_cells

    staging_bounds = []
    for module in first.plan.modules:
        position = module.staging_pose.position
        staging_bounds.append(
            (
                module.module_id,
                position.x - module.dimensions.width / 2,
                position.x + module.dimensions.width / 2,
                position.y - module.dimensions.depth / 2,
                position.y + module.dimensions.depth / 2,
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
    for item in result.replay:
        module = modules[item.module_id]
        center = module.staging_pose.position
        minimum_x = center.x - module.dimensions.width / 2 - robot_radius
        maximum_x = center.x + module.dimensions.width / 2 + robot_radius
        minimum_y = center.y - module.dimensions.depth / 2 - robot_radius
        maximum_y = center.y + module.dimensions.depth / 2 + robot_radius
        for route in item.approach_routes.values():
            assert all(
                not (
                    minimum_x <= waypoint.x <= maximum_x
                    and minimum_y <= waypoint.y <= maximum_y
                )
                for waypoint in route
            ), item.module_id

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
    live_dir = tmp_path / "synthetic-live-nominal"
    live_manifest = write_phase5_artifact_bundle(
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
    assert live_manifest.live_evidence
    assert live_manifest.live_gate_passed
    assert verify_phase5_artifact_bundle(live_dir) == live_manifest


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
    live_manifest = write_phase5_artifact_bundle(
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
    assert live_manifest.live_evidence
    assert live_manifest.live_gate_passed
    assert verify_phase5_artifact_bundle(live_dir) == live_manifest


def test_fake_client_cannot_be_labeled_as_live_coppelia_evidence() -> None:
    scenario, _physical_yard = _phase5_scenario()
    executor = _phase5_executor(scenario)
    with pytest.raises(ValueError, match="cannot be attested as live"):
        Phase5FullCottageRunner(
            executor,
            scenario,
            evidence_kind="live_coppelia",
        )
