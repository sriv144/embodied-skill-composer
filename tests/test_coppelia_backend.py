from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from embodied_skill_composer.assembly.coppelia_backend import (
    CoppeliaSimAssemblyBackend,
    build_coppelia_scene_spec,
    inspect_coppelia_runtime,
)
from embodied_skill_composer.assembly.blueprint import compile_modular_blueprint
from embodied_skill_composer.assembly.brain import (
    PrecedenceConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BeamTask,
    CoppeliaSimConfig,
)
from embodied_skill_composer.assembly.runtime import (
    load_asset_catalog,
    load_modular_blueprint,
    load_runtime_profile,
)


class FakeSim:
    simulation_stopped = 0
    simulation_advancing = 17
    primitiveshape_cuboid = 3
    colorcomponent_ambient_diffuse = 0
    shapeintparam_static = 3003
    intparam_dynamic_engine = 8
    physics_mujoco = 4
    handle_scene = -12
    stringparam_scene_path_and_name = 13
    sceneobject_shape = 0
    sceneobject_script = 17

    def __init__(self) -> None:
        self.state = self.simulation_stopped
        self.engine = 0
        self.time = 0.0
        self.next_handle = 1
        self.aliases: dict[int, str] = {}
        self.positions: dict[int, list[float]] = {}
        self.matrices: dict[int, list[float]] = {}
        self.object_types: dict[int, int] = {}
        self.model_trees: dict[int, list[int]] = {}
        self.scaled_models: list[tuple[list[int], float]] = []
        self.imported_shapes: list[tuple[str, int, float]] = []
        self.camera_resolution = [64, 64]

    def getSimulationState(self) -> int:
        return self.state

    def startSimulation(self) -> None:
        self.state = self.simulation_advancing

    def stopSimulation(self) -> None:
        self.state = self.simulation_stopped

    def getSimulationTime(self) -> float:
        return self.time

    def getSimulationTimeStep(self) -> float:
        return 0.05

    def getStringParam(self, parameter: int) -> str:
        _ = parameter
        return ""

    def getInt32Param(self, parameter: int) -> int:
        _ = parameter
        return self.engine

    def setInt32Param(self, parameter: int, value: int) -> None:
        _ = parameter
        self.engine = value

    def getObject(self, path: str, options: dict | None = None) -> int:
        _ = options
        alias = path.rsplit("/", 1)[-1]
        for handle, current_alias in self.aliases.items():
            if current_alias == alias:
                return handle
        return -1

    def getObjectsInTree(self, base: int) -> list[int]:
        if base in self.model_trees:
            return [
                handle
                for handle in self.model_trees[base]
                if handle in self.object_types
            ]
        return list(self.aliases)

    def removeObjects(self, handles: list[int], delayed: bool) -> None:
        _ = delayed
        for handle in handles:
            self.aliases.pop(handle, None)
            self.positions.pop(handle, None)
            self.object_types.pop(handle, None)

    def createDummy(self, size: float) -> int:
        _ = size
        return self._new_handle(object_type=1)

    def createPrimitiveShape(
        self,
        primitive_type: int,
        sizes: list[float],
        options: int,
    ) -> int:
        _ = (primitive_type, sizes, options)
        return self._new_handle(object_type=self.sceneobject_shape)

    def createVisionSensor(
        self,
        options: int,
        int_parameters: list[int],
        float_parameters: list[float],
    ) -> int:
        _ = (options, float_parameters)
        self.camera_resolution = int_parameters[:2]
        return self._new_handle(object_type=9)

    def loadModel(self, path: str) -> int:
        assert Path(path).is_file()
        root = self._new_handle(object_type=self.sceneobject_shape)
        script = self._new_handle(object_type=self.sceneobject_script)
        shape = self._new_handle(object_type=self.sceneobject_shape)
        self.model_trees[root] = [root, script, shape]
        return root

    def importShape(
        self,
        file_format: int,
        path: str,
        options: int,
        tolerance: float,
        scale: float,
    ) -> int:
        assert file_format == 0
        assert tolerance == 0.0
        assert Path(path).is_file()
        self.imported_shapes.append((path, options, scale))
        return self._new_handle(object_type=self.sceneobject_shape)

    def getObjectType(self, handle: int) -> int:
        return self.object_types[handle]

    def scaleObjects(
        self,
        handles: list[int],
        scale: float,
        scale_positions: bool,
    ) -> None:
        assert scale_positions is True
        self.scaled_models.append((list(handles), scale))

    def setObjectAlias(self, handle: int, alias: str) -> None:
        self.aliases[handle] = alias

    def setObjectParent(self, handle: int, parent: int, keep_in_place: bool) -> None:
        _ = (handle, parent, keep_in_place)

    def setObjectPosition(self, handle: int, position: list[float]) -> None:
        self.positions[handle] = position

    def setObjectOrientation(self, handle: int, orientation: list[float]) -> None:
        _ = (handle, orientation)

    def setObjectMatrix(self, handle: int, matrix: list[float]) -> None:
        self.matrices[handle] = matrix
        self.positions[handle] = [matrix[3], matrix[7], matrix[11]]

    def setShapeColor(
        self,
        handle: int,
        color_name: str | None,
        component: int,
        color: list[float],
    ) -> None:
        _ = (handle, color_name, component, color)

    def setObjectInt32Param(self, handle: int, parameter: int, value: int) -> None:
        _ = (handle, parameter, value)

    def getVisionSensorImg(self, handle: int) -> tuple[bytes, list[int]]:
        _ = handle
        width, height = self.camera_resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[:, :, 1] = 180
        return image.tobytes(), self.camera_resolution

    def saveScene(self, path: str) -> None:
        Path(path).write_bytes(b"fake-coppelia-scene")

    def _new_handle(self, *, object_type: int = 0) -> int:
        handle = self.next_handle
        self.next_handle += 1
        self.object_types[handle] = object_type
        return handle


class FakeClient:
    def __init__(self, sim: FakeSim | None = None) -> None:
        self.sim = sim or FakeSim()
        self.stepping = False

    def require(self, name: str) -> FakeSim:
        assert name == "sim"
        return self.sim

    def setStepping(self, stepping: bool) -> None:
        self.stepping = stepping

    def step(self) -> None:
        assert self.stepping
        self.sim.time += self.sim.getSimulationTimeStep()


def build_config() -> AssemblyScenarioConfig:
    return AssemblyScenarioConfig(
        grid_size=12,
        max_steps=120,
        agent_starts=[(0, 2), (0, 3)],
        obstacle_cells=[(5, 5)],
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


def build_profile(
    *,
    use_bundled_robot_model: bool = False,
    robot_model_path: Path | None = None,
) -> AssemblyRuntimeProfile:
    return AssemblyRuntimeProfile(
        name="coppelia_test",
        backend="coppelia_sim",
        coppelia=CoppeliaSimConfig(
            camera_width=64,
            camera_height=64,
            use_bundled_robot_model=use_bundled_robot_model,
            robot_model_path=(
                str(robot_model_path)
                if robot_model_path is not None
                else CoppeliaSimConfig().robot_model_path
            ),
        ),
    )


def test_scene_spec_contains_construction_objects() -> None:
    specs = build_coppelia_scene_spec(build_config(), scale=0.35)
    categories = [spec.category for spec in specs]

    assert categories.count("floor") == 1
    assert categories.count("agent") == 2
    assert categories.count("resource") == 2
    assert categories.count("blueprint") == 4
    assert categories.count("obstacle") == 1
    assert len({spec.alias for spec in specs}) == len(specs)


def test_runtime_probe_reports_fake_live_connection() -> None:
    fake = FakeClient()
    result = inspect_coppelia_runtime(build_profile(), client_factory=lambda _: fake)

    assert result["connected"] is True
    assert result["simulation_state"] == fake.sim.simulation_stopped
    assert result["physics_engine"] == 0


def test_coppelia_backend_runs_scripted_episode_with_fake_transport(
    tmp_path: Path,
) -> None:
    fake = FakeClient()
    backend = CoppeliaSimAssemblyBackend(
        config=build_config(),
        runtime_profile=build_profile(),
        seed=7,
        client_factory=lambda _: fake,
    )

    assert backend.is_ready is True
    assert fake.sim.engine == fake.sim.physics_mujoco
    backend.reset(seed=7)
    done = False
    while not done:
        result = backend.execute_team_option(backend.scripted_team_option())
        done = result.done

    artifact = backend.build_artifact(policy_mode="scripted")
    diagnostics = backend.get_option_episode_diagnostics()["coppelia_sim"]
    image_path = backend.capture_camera(tmp_path / "final.png")
    scene_path = backend.save_scene(tmp_path / "scene.ttt")

    assert artifact.metrics.success is True
    assert artifact.metrics.beams_installed == 2
    assert diagnostics["control_mode"] == "kinematic_cooperative_pose_sync"
    assert diagnostics["simulation_step_count"] > 0
    assert diagnostics["camera_ready"] is True
    assert image_path.exists()
    assert scene_path.exists()


def test_coppelia_backend_loads_runtime_robot_models(tmp_path: Path) -> None:
    model_path = tmp_path / "KUKA YouBot.ttm"
    model_path.write_bytes(b"fake-model")
    fake = FakeClient()
    backend = CoppeliaSimAssemblyBackend(
        config=build_config(),
        runtime_profile=build_profile(
            use_bundled_robot_model=True,
            robot_model_path=model_path,
        ),
        seed=7,
        client_factory=lambda _: fake,
    )

    diagnostics = backend.get_option_episode_diagnostics()["coppelia_sim"]

    assert diagnostics["loaded_robot_models"] == 2
    assert diagnostics["robot_model_path"] == str(model_path)
    assert len(fake.sim.scaled_models) == 2
    assert all(scale == 0.55 for _, scale in fake.sim.scaled_models)
    assert fake.sim.sceneobject_script not in fake.sim.object_types.values()


def test_coppelia_backend_imports_room_assets_and_records_replay(
    tmp_path: Path,
) -> None:
    workspace = Path(__file__).resolve().parents[1]
    compiled = compile_modular_blueprint(
        load_modular_blueprint(
            workspace / "configs" / "blueprints" / "modular_room_v0.yaml"
        ),
        load_asset_catalog(
            workspace / "configs" / "construction_asset_catalog.yaml"
        ),
        workspace_root=workspace,
    )
    fake = FakeClient()
    profile = build_profile().model_copy(
        update={
            "coppelia": build_profile().coppelia.model_copy(
                update={"use_construction_meshes": True}
            )
        }
    )
    backend = CoppeliaSimAssemblyBackend(
        config=compiled.scenario,
        runtime_profile=profile,
        seed=7,
        client_factory=lambda _: fake,
    )

    episode = run_construction_brain_episode(
        backend,
        PrecedenceConstructionBrain(),
        seed=7,
    )
    diagnostics = backend.get_option_episode_diagnostics()
    recording = backend.record_episode(
        tmp_path / "room.gif",
        diagnostics=diagnostics,
    )
    final_overview = backend.capture_camera(
        tmp_path / "overview.png",
        camera_name="overview",
    )
    final_topdown = backend.capture_camera(
        tmp_path / "topdown.png",
        camera_name="topdown",
    )
    coppelia = backend.get_option_episode_diagnostics()["coppelia_sim"]

    assert episode.artifact.metrics.success is True
    assert len(fake.sim.imported_shapes) == 10
    assert all(options == 16 for _, options, _ in fake.sim.imported_shapes)
    assert coppelia["loaded_asset_meshes"] == 10
    assert coppelia["asset_fallbacks"] == []
    assert coppelia["cameras"] == ["overview", "topdown"]
    assert coppelia["recording_frame_count"] > 1
    assert recording.exists()
    assert final_overview.exists()
    assert final_topdown.exists()

    first_frame = backend.logical_env.frame_history[0]
    backend._sync_from_playback_frame(first_frame)
    first_resource = compiled.scenario.resources[0]
    handle = fake.sim.getObject(f"/construction_resource_{first_resource.resource_id}")
    target_pose = compiled.scenario.blueprint_slots[0].target_pose
    assert target_pose is not None
    assert fake.sim.positions[handle] != list(target_pose.position_m)


@pytest.mark.skipif(
    os.environ.get("RUN_COPPELIA_LIVE_TESTS") != "1",
    reason="Set RUN_COPPELIA_LIVE_TESTS=1 with CoppeliaSim running.",
)
def test_live_coppelia_connection() -> None:
    profile = build_profile()
    profile = profile.model_copy(
        update={
            "coppelia": profile.coppelia.model_copy(
                update={"port": int(os.environ.get("COPPELIA_PORT", "23000"))}
            )
        }
    )
    result = inspect_coppelia_runtime(profile)
    assert result["connected"] is True, result["error"]


@pytest.mark.skipif(
    os.environ.get("RUN_COPPELIA_LIVE_TESTS") != "1",
    reason="Set RUN_COPPELIA_LIVE_TESTS=1 with CoppeliaSim running.",
)
def test_live_coppelia_builds_modular_room(tmp_path: Path) -> None:
    workspace = Path(__file__).resolve().parents[1]
    compiled = compile_modular_blueprint(
        load_modular_blueprint(
            workspace / "configs" / "blueprints" / "modular_room_v0.yaml"
        ),
        load_asset_catalog(
            workspace / "configs" / "construction_asset_catalog.yaml"
        ),
        workspace_root=workspace,
    )
    profile = load_runtime_profile(
        workspace / "configs" / "assembly_profiles" / "coppelia_local.yaml"
    )
    profile = profile.model_copy(
        update={
            "coppelia": profile.coppelia.model_copy(
                update={"port": int(os.environ.get("COPPELIA_PORT", "23000"))}
            )
        }
    )
    backend = CoppeliaSimAssemblyBackend(
        config=compiled.scenario,
        runtime_profile=profile,
        seed=7,
    )
    try:
        episode = run_construction_brain_episode(
            backend,
            PrecedenceConstructionBrain(),
            seed=7,
        )
        backend.focus_cameras_on_structure()
        scene_path = backend.save_scene(tmp_path / "modular_room.ttt")
        diagnostics = backend.get_option_episode_diagnostics()["coppelia_sim"]

        assert episode.artifact.metrics.success is True
        assert episode.artifact.metrics.beams_installed == 10
        assert diagnostics["loaded_asset_meshes"] == 10
        assert diagnostics["loaded_robot_models"] == 2
        assert diagnostics["cameras"] == ["overview", "topdown"]
        assert scene_path.exists()
    finally:
        backend.close()
