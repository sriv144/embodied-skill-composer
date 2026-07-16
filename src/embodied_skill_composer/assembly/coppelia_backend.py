from __future__ import annotations

import importlib.util
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, cast

import numpy as np

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.blueprint import validate_asset_catalog
from embodied_skill_composer.assembly.models import (
    AssemblyPlaybackFrame,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    AssetCatalog,
    BackendStatus,
    BlueprintSlot,
    ConstructionBrainObservation,
    ConstructionResource,
    CoppeliaSimConfig,
    EpisodeArtifact,
    OptionExecutionResult,
    TeamOption,
)
from embodied_skill_composer.assembly.runtime import load_asset_catalog


class CoppeliaSimBackendUnavailableError(RuntimeError):
    """Raised when live CoppeliaSim execution is requested without a connection."""


@dataclass(frozen=True)
class CoppeliaSceneObjectSpec:
    alias: str
    category: Literal["floor", "agent", "resource", "blueprint", "obstacle"]
    position: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]
    yaw_radians: float = 0.0


def coppelia_client_available() -> bool:
    return importlib.util.find_spec("coppeliasim_zmqremoteapi_client") is not None


def grid_to_coppelia_world(
    coordinate: tuple[int, int],
    *,
    grid_size: int,
    scale: float,
    z: float,
) -> tuple[float, float, float]:
    center = (grid_size - 1) / 2.0
    return (
        (coordinate[0] - center) * scale,
        (coordinate[1] - center) * scale,
        z,
    )


def build_coppelia_scene_spec(
    config: AssemblyScenarioConfig,
    *,
    scale: float,
) -> list[CoppeliaSceneObjectSpec]:
    world_size = config.grid_size * scale
    specs = [
        CoppeliaSceneObjectSpec(
            alias="construction_floor",
            category="floor",
            # Keep the generated floor just above CoppeliaSim's default floor.
            # Coplanar surfaces produce severe z-fighting in angled recordings.
            position=(0.0, 0.0, 0.001),
            size=(world_size, world_size, 0.06),
            color=(0.22, 0.24, 0.27),
        )
    ]
    agent_colors = [(0.08, 0.72, 0.92), (0.93, 0.36, 0.18)]
    for index, coordinate in enumerate(config.agent_starts):
        specs.append(
            CoppeliaSceneObjectSpec(
                alias=f"construction_agent_{index}",
                category="agent",
                position=grid_to_coppelia_world(
                    coordinate,
                    grid_size=config.grid_size,
                    scale=scale,
                    z=0.11,
                ),
                size=(0.28, 0.24, 0.18),
                color=agent_colors[index % len(agent_colors)],
            )
        )
    for beam in config.beams:
        left = grid_to_coppelia_world(
            beam.pickup_left,
            grid_size=config.grid_size,
            scale=scale,
            z=0.09,
        )
        right = grid_to_coppelia_world(
            beam.pickup_right,
            grid_size=config.grid_size,
            scale=scale,
            z=0.09,
        )
        delta_x = right[0] - left[0]
        delta_y = right[1] - left[1]
        length = max(0.30, math.hypot(delta_x, delta_y) + 0.22)
        specs.append(
            CoppeliaSceneObjectSpec(
                alias=f"construction_resource_{beam.name}",
                category="resource",
                position=(
                    (left[0] + right[0]) / 2.0,
                    (left[1] + right[1]) / 2.0,
                    0.09,
                ),
                size=(length, 0.14, 0.12),
                color=(0.95, 0.67, 0.12),
                yaw_radians=math.atan2(delta_y, delta_x),
            )
        )
    for slot in config.derived_blueprint_slots():
        for index, coordinate in enumerate(slot.target_cells):
            specs.append(
                CoppeliaSceneObjectSpec(
                    alias=f"construction_blueprint_{slot.slot_id}_{index}",
                    category="blueprint",
                    position=grid_to_coppelia_world(
                        coordinate,
                        grid_size=config.grid_size,
                        scale=scale,
                        z=0.025,
                    ),
                    size=(scale * 0.72, scale * 0.72, 0.025),
                    color=(0.26, 0.86, 0.38),
                )
            )
    for index, coordinate in enumerate(config.obstacle_cells):
        specs.append(
            CoppeliaSceneObjectSpec(
                alias=f"construction_obstacle_{index}",
                category="obstacle",
                position=grid_to_coppelia_world(
                    coordinate,
                    grid_size=config.grid_size,
                    scale=scale,
                    z=0.22,
                ),
                size=(scale * 0.82, scale * 0.82, 0.44),
                color=(0.48, 0.50, 0.54),
            )
        )
    return specs


def _connect_client(config: CoppeliaSimConfig):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    client = RemoteAPIClient(host=config.host, port=config.port)
    try:
        import zmq

        timeout_ms = int(config.connection_timeout_s * 1000)
        client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        client.socket.setsockopt(zmq.LINGER, 0)
    except (AttributeError, ImportError):
        pass
    return client


def inspect_coppelia_runtime(
    runtime_profile: AssemblyRuntimeProfile,
    client_factory: Callable[[CoppeliaSimConfig], Any] | None = None,
) -> dict[str, object]:
    config = runtime_profile.coppelia
    executable = Path(config.executable_path)
    result: dict[str, object] = {
        "runtime_profile": runtime_profile.name,
        "backend": runtime_profile.backend,
        "client_installed": coppelia_client_available(),
        "executable_path": str(executable),
        "executable_exists": executable.exists(),
        "host": config.host,
        "port": config.port,
        "connected": False,
        "simulation_state": None,
        "scene_path": None,
        "object_count": 0,
        "physics_engine": None,
        "error": None,
    }
    if client_factory is None and not result["client_installed"]:
        result["error"] = "Install requirements-sim-coppelia.txt."
        return result
    try:
        client = (client_factory or _connect_client)(config)
        sim = client.require("sim")
        result.update(
            connected=True,
            simulation_state=int(sim.getSimulationState()),
            scene_path=str(sim.getStringParam(sim.stringparam_scene_path_and_name)),
            object_count=len(sim.getObjectsInTree(sim.handle_scene)),
            physics_engine=int(sim.getInt32Param(sim.intparam_dynamic_engine)),
        )
    except Exception as exc:  # pragma: no cover - exact transport errors vary
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


@dataclass
class CoppeliaSimAssemblyBackend:
    config: AssemblyScenarioConfig
    runtime_profile: AssemblyRuntimeProfile
    seed: int = 7
    client_factory: Callable[[CoppeliaSimConfig], Any] | None = field(
        default=None,
        repr=False,
    )
    backend_name: str = field(default="coppelia_sim", init=False)
    is_ready: bool = field(default=False, init=False)
    readiness_notes: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.logical_env = CollaborativeAssemblyEnv(self.config, seed=self.seed)
        self._workspace_root = Path(__file__).resolve().parents[3]
        # The ZeroMQ remote API is dynamically imported and intentionally
        # untyped at this optional dependency boundary.
        self._client: Any = None
        self._sim: Any = None
        self._root_handle: int | None = None
        self._camera_handle: int | None = None
        self._camera_handles: dict[str, int] = {}
        self._object_handles: dict[str, int] = {}
        self._simulation_step_count = 0
        self._last_seed: int | None = None
        self._connection_error: str | None = None
        self._scene_build_count = 0
        self._loaded_robot_models = 0
        self._loaded_asset_meshes = 0
        self._asset_fallbacks: list[str] = []
        self._asset_catalog: AssetCatalog | None = None
        self._last_recording_path: str | None = None
        self._last_recording_frame_count = 0
        self._recording_fallback_reason: str | None = None
        self.readiness_notes = [
            "CoppeliaSim spike delegates task semantics to the local construction environment.",
            "The v0 scene uses deterministic kinematic pose updates and ZeroMQ stepping.",
            "Real wheel, gripper, contact, and force-sensor dynamics remain future parity work.",
        ]
        self._load_asset_catalog()
        if not coppelia_client_available() and self.client_factory is None:
            self.readiness_notes.append(
                "Install optional dependencies with `pip install -r requirements-sim-coppelia.txt`."
            )
            return
        try:
            self._client = (self.client_factory or _connect_client)(
                self.runtime_profile.coppelia
            )
            self._sim = self._client.require("sim")
            self._sim.getSimulationState()
            self.is_ready = True
            if self.runtime_profile.coppelia.rebuild_scene_on_connect:
                self.rebuild_scene()
        except Exception as exc:  # pragma: no cover - transport-specific
            self._connection_error = f"{type(exc).__name__}: {exc}"
            self.readiness_notes.append(
                f"Could not connect to {self.runtime_profile.coppelia.host}:"
                f"{self.runtime_profile.coppelia.port}: {self._connection_error}"
            )

    @property
    def num_agents(self) -> int:
        return self.logical_env.num_agents

    @property
    def action_size(self) -> int:
        return self.logical_env.action_size

    @property
    def option_size(self) -> int:
        return self.logical_env.option_size

    @property
    def obs_dim(self) -> int:
        return self.logical_env.obs_dim

    @property
    def state_dim(self) -> int:
        return self.logical_env.state_dim

    @property
    def team_option_obs_dim(self) -> int:
        return self.logical_env.team_option_obs_dim

    @property
    def active_beam_count(self) -> int:
        return self.logical_env.active_beam_count

    @property
    def active_stage_index(self) -> int | None:
        return self.logical_env.active_stage_index

    @property
    def recovery_option_usage(self) -> dict[str, int]:
        return self.logical_env.recovery_option_usage

    def rebuild_scene(self) -> None:
        self._require_ready()
        self._ensure_stopped()
        existing = self._sim.getObject("/ESCConstruction", {"noError": True})
        if existing != -1:
            handles = self._sim.getObjectsInTree(existing)
            self._sim.removeObjects(list(reversed(handles)), False)
        if self.runtime_profile.coppelia.use_mujoco_physics:
            self._sim.setInt32Param(
                self._sim.intparam_dynamic_engine,
                self._sim.physics_mujoco,
            )
        self._root_handle = self._sim.createDummy(0.01)
        self._sim.setObjectAlias(self._root_handle, "ESCConstruction")
        self._object_handles = {}
        self._loaded_robot_models = 0
        self._loaded_asset_meshes = 0
        self._asset_fallbacks = []
        resources = {
            resource.resource_id: resource
            for resource in self.config.derived_resources()
        }
        for spec in build_coppelia_scene_spec(
            self.config,
            scale=self.runtime_profile.coppelia.grid_scale_m,
        ):
            if spec.category == "agent" and self._robot_model_available():
                self._create_robot_model(spec)
            elif spec.category == "resource" and self._resource_mesh_available(
                resources.get(spec.alias.removeprefix("construction_resource_"))
            ):
                self._create_asset_shape(
                    spec,
                    resources[spec.alias.removeprefix("construction_resource_")],
                )
            else:
                self._create_shape(spec)
        self._camera_handles = {
            "overview": self._create_camera("overview"),
            "topdown": self._create_camera("topdown"),
        }
        self._camera_handle = self._camera_handles["overview"]
        self._scene_build_count += 1

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._require_ready()
        self._last_seed = self.seed if seed is None else seed
        observations, state = self.logical_env.reset(seed=seed)
        self._ensure_stopped()
        self._sync_from_logical_state()
        self._client.setStepping(True)
        self._sim.startSimulation()
        self._simulation_step_count = 0
        self._advance_simulation()
        return observations, state

    def set_curriculum_stage(
        self,
        beam_count: int | None = None,
        stage_index: int | None = None,
    ) -> None:
        self.logical_env.set_curriculum_stage(
            beam_count=beam_count,
            stage_index=stage_index,
        )
        if self.is_ready:
            self._sync_from_logical_state()

    def step(
        self,
        actions: list[int],
    ) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        self._require_ready()
        result = self.logical_env.step(actions)
        self._sync_from_logical_state()
        self._advance_simulation()
        return result

    def build_artifact(
        self,
        policy_mode: Literal["scripted", "learned", "brain"],
    ) -> EpisodeArtifact:
        return self.logical_env.build_artifact(policy_mode=policy_mode)

    def get_agent_observations(self) -> np.ndarray:
        return self.logical_env.get_agent_observations()

    def get_action_masks(self) -> np.ndarray:
        return self.logical_env.get_action_masks()

    def get_privileged_state(self) -> np.ndarray:
        return self.logical_env.get_privileged_state()

    def get_team_option_observation(self) -> np.ndarray:
        return self.logical_env.get_team_option_observation()

    def get_team_option_mask(self) -> np.ndarray:
        return self.logical_env.get_team_option_mask()

    def scripted_team_option(self) -> TeamOption:
        return self.logical_env.scripted_team_option()

    def execute_team_option(
        self,
        option: int | TeamOption,
        max_primitive_steps: int | None = None,
    ) -> OptionExecutionResult:
        self._require_ready()
        first_new_frame = len(self.logical_env.frame_history)
        result = self.logical_env.execute_team_option(
            option,
            max_primitive_steps=max_primitive_steps,
        )
        frames = self.logical_env.frame_history[first_new_frame:]
        if frames:
            for frame in frames:
                self._sync_from_playback_frame(frame)
                self._advance_simulation()
        else:
            self._sync_from_logical_state()
            self._advance_simulation()
        return result

    def get_option_episode_diagnostics(self) -> dict[str, object]:
        diagnostics = self.logical_env.get_option_episode_diagnostics()
        diagnostics["backend"] = self.backend_name
        diagnostics["runtime_profile"] = self.runtime_profile.name
        diagnostics["backend_status"] = self.get_backend_status().model_dump(
            mode="json"
        )
        diagnostics["coppelia_sim"] = {
            "connected": self.is_ready,
            "host": self.runtime_profile.coppelia.host,
            "port": self.runtime_profile.coppelia.port,
            "scene_build_count": self._scene_build_count,
            "scene_object_count": len(self._object_handles),
            "simulation_step_count": self._simulation_step_count,
            "simulation_time_s": self._simulation_time(),
            "time_step_s": self._time_step(),
            "physics_engine": self._physics_engine(),
            "control_mode": "kinematic_cooperative_pose_sync",
            "camera_ready": bool(self._camera_handles),
            "cameras": sorted(self._camera_handles),
            "robot_model_path": self.runtime_profile.coppelia.robot_model_path,
            "loaded_robot_models": self._loaded_robot_models,
            "loaded_asset_meshes": self._loaded_asset_meshes,
            "asset_fallbacks": list(self._asset_fallbacks),
            "recording_path": self._last_recording_path,
            "recording_frame_count": self._last_recording_frame_count,
            "recording_fallback_reason": self._recording_fallback_reason,
            "connection_error": self._connection_error,
        }
        return diagnostics

    def get_construction_observation(self) -> ConstructionBrainObservation:
        return self.logical_env.get_construction_observation().model_copy(
            update={"backend": self.backend_name}
        )

    def get_backend_status(self) -> BackendStatus:
        notes = list(self.readiness_notes)
        if self.runtime_profile.notes:
            notes.append(self.runtime_profile.notes)
        return BackendStatus(
            backend_name=self.backend_name,
            is_ready=self.is_ready,
            readiness_notes=notes,
        )

    def capture_camera(
        self,
        output_path: Path,
        *,
        camera_name: str = "overview",
    ) -> Path:
        self._require_ready()
        if camera_name not in self._camera_handles:
            raise CoppeliaSimBackendUnavailableError(
                f"CoppeliaSim scene has no '{camera_name}' construction camera."
            )
        array = self._camera_array(camera_name)
        import imageio.v3 as imageio

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(output_path, array)
        return output_path

    def focus_cameras_on_structure(self) -> None:
        """Frame installed metric targets for the final construction images."""
        target_poses = [
            slot.target_pose.position_m
            for slot in self.config.derived_blueprint_slots()
            if slot.target_pose is not None
        ]
        if not target_poses:
            return
        min_x = min(pose[0] for pose in target_poses)
        max_x = max(pose[0] for pose in target_poses)
        min_y = min(pose[1] for pose in target_poses)
        max_y = max(pose[1] for pose in target_poses)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        span = max(max_x - min_x, max_y - min_y, 2.0)
        camera_position = (
            center[0] - span * 2.1,
            center[1] - span * 2.1,
            max(3.6, span * 1.8),
        )
        target = (center[0], center[1], 0.5)
        self._ensure_stopped()
        self._sync_from_logical_state()
        self._set_camera_look_at("overview", camera_position, target)
        self._sim.setObjectPosition(
            self._camera_handles["topdown"],
            [center[0], center[1], max(4.8, span * 2.4)],
        )
        self._sim.setObjectOrientation(
            self._camera_handles["topdown"],
            [math.pi, 0.0, 0.0],
        )
        self._client.setStepping(True)
        self._sim.startSimulation()
        self._advance_simulation()

    def record_episode(
        self,
        output_path: Path,
        *,
        diagnostics: dict[str, object] | None = None,
        camera_name: str = "overview",
        fps: int = 12,
    ) -> Path:
        self._require_ready()
        if camera_name not in self._camera_handles:
            raise CoppeliaSimBackendUnavailableError(
                f"CoppeliaSim scene has no '{camera_name}' construction camera."
            )
        import imageio.v2 as imageio

        payload = diagnostics or self.get_option_episode_diagnostics()
        raw_frames = payload.get("state_snapshots", [])
        if not isinstance(raw_frames, list) or not raw_frames:
            raise ValueError("CoppeliaSim recording requires playback state snapshots.")
        frames = [AssemblyPlaybackFrame.model_validate(item) for item in raw_frames]
        rendered_frames: list[np.ndarray] = []
        self._ensure_stopped()
        self._client.setStepping(True)
        self._sim.startSimulation()
        try:
            for frame in frames:
                self._sync_from_playback_frame(frame)
                self._advance_simulation()
                rendered_frames.append(
                    self._annotate_frame(self._camera_array(camera_name), frame)
                )
        finally:
            self._ensure_stopped()
        self._sync_from_playback_frame(frames[-1])

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if output_path.suffix.lower() == ".gif":
                imageio.mimsave(
                    output_path,
                    cast(Any, rendered_frames),
                    duration=1000 / fps,
                )
            else:
                imageio.mimsave(output_path, cast(Any, rendered_frames), fps=fps)
            recording_path = output_path
            self._recording_fallback_reason = None
        except Exception as exc:
            fallback_dir = output_path.with_suffix("")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            for index, rendered in enumerate(rendered_frames):
                imageio.imwrite(fallback_dir / f"frame_{index:04d}.png", rendered)
            recording_path = fallback_dir
            self._recording_fallback_reason = f"{type(exc).__name__}: {exc}"
        self._last_recording_path = str(recording_path)
        self._last_recording_frame_count = len(rendered_frames)
        return recording_path

    def save_scene(self, output_path: Path) -> Path:
        self._require_ready()
        self._ensure_stopped()
        self._sync_from_logical_state()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._sim.saveScene(str(output_path))
        return output_path

    def close(self, *, stop_simulation: bool = True) -> None:
        if stop_simulation and self.is_ready:
            try:
                self._ensure_stopped()
            except Exception as exc:  # pragma: no cover - best-effort transport cleanup
                self.readiness_notes.append(
                    f"CoppeliaSim cleanup could not confirm a stopped simulation: "
                    f"{type(exc).__name__}: {exc}"
                )

    def _create_shape(self, spec: CoppeliaSceneObjectSpec) -> int:
        handle = self._sim.createPrimitiveShape(
            self._sim.primitiveshape_cuboid,
            list(spec.size),
            2,
        )
        self._sim.setObjectAlias(handle, spec.alias)
        self._sim.setObjectParent(handle, self._root_handle, True)
        self._sim.setObjectPosition(handle, list(spec.position))
        self._sim.setObjectOrientation(handle, [0.0, 0.0, spec.yaw_radians])
        self._sim.setShapeColor(
            handle,
            None,
            self._sim.colorcomponent_ambient_diffuse,
            list(spec.color),
        )
        self._sim.setObjectInt32Param(handle, self._sim.shapeintparam_static, 1)
        self._object_handles[spec.alias] = handle
        return int(handle)

    def _load_asset_catalog(self) -> None:
        config = self.runtime_profile.coppelia
        if not config.use_construction_meshes:
            return
        catalog_path = Path(config.asset_catalog_path)
        if not catalog_path.is_absolute():
            catalog_path = self._workspace_root / catalog_path
        try:
            catalog = load_asset_catalog(catalog_path)
            validate_asset_catalog(catalog, self._workspace_root)
            self._asset_catalog = catalog
        except Exception as exc:
            self.readiness_notes.append(
                "Construction asset catalog is unavailable; primitive resource "
                f"geometry will be used ({type(exc).__name__}: {exc})."
            )

    def _resource_mesh_available(
        self,
        resource: ConstructionResource | None,
    ) -> bool:
        return (
            resource is not None
            and resource.asset_key is not None
            and self._asset_catalog is not None
            and resource.asset_key in self._asset_catalog.components
        )

    def _create_asset_shape(
        self,
        spec: CoppeliaSceneObjectSpec,
        resource: ConstructionResource,
    ) -> int:
        if self._asset_catalog is None or resource.asset_key is None:
            return self._create_shape(spec)
        asset = self._asset_catalog.components[resource.asset_key]
        mesh_path = (self._workspace_root / asset.visual_mesh).resolve()
        try:
            handle = self._sim.importShape(
                0,
                str(mesh_path),
                16,
                0.0,
                asset.mesh_scale,
            )
            self._loaded_asset_meshes += 1
        except Exception as exc:
            self._asset_fallbacks.append(
                f"{resource.resource_id}: {type(exc).__name__}: {exc}"
            )
            fallback = CoppeliaSceneObjectSpec(
                alias=spec.alias,
                category=spec.category,
                position=spec.position,
                size=asset.dimensions_m,
                color=spec.color,
                yaw_radians=spec.yaw_radians,
            )
            return self._create_shape(fallback)

        self._sim.setObjectAlias(handle, spec.alias)
        self._sim.setObjectParent(handle, self._root_handle, True)
        position = (
            spec.position
            if resource.source_pose is None
            else resource.source_pose.position_m
        )
        source_rotation = (
            (0.0, 0.0, math.degrees(spec.yaw_radians))
            if resource.source_pose is None
            else resource.source_pose.rotation_rpy_degrees
        )
        orientation = tuple(
            math.radians(source_rotation[index] + asset.orientation_rpy_degrees[index])
            for index in range(3)
        )
        self._sim.setObjectPosition(handle, list(position))
        self._sim.setObjectOrientation(handle, list(orientation))
        self._sim.setObjectInt32Param(handle, self._sim.shapeintparam_static, 1)
        self._object_handles[spec.alias] = handle
        return int(handle)

    def _create_robot_model(self, spec: CoppeliaSceneObjectSpec) -> int:
        config = self.runtime_profile.coppelia
        handle = self._sim.loadModel(str(Path(config.robot_model_path).resolve()))
        self._sim.setObjectAlias(handle, spec.alias)
        scripts = [
            object_handle
            for object_handle in self._sim.getObjectsInTree(handle)
            if self._sim.getObjectType(object_handle) == self._sim.sceneobject_script
        ]
        if scripts:
            self._sim.removeObjects(scripts, False)
        model_objects = self._sim.getObjectsInTree(handle)
        self._sim.scaleObjects(model_objects, config.robot_model_scale, True)
        for object_handle in model_objects:
            if self._sim.getObjectType(object_handle) == self._sim.sceneobject_shape:
                self._sim.setObjectInt32Param(
                    object_handle,
                    self._sim.shapeintparam_static,
                    1,
                )
        self._sim.setObjectParent(handle, self._root_handle, True)
        self._sim.setObjectPosition(handle, list(spec.position))
        self._sim.setObjectOrientation(handle, [0.0, 0.0, spec.yaw_radians])
        self._object_handles[spec.alias] = handle
        self._loaded_robot_models += 1
        return int(handle)

    def _robot_model_available(self) -> bool:
        config = self.runtime_profile.coppelia
        return config.use_bundled_robot_model and Path(config.robot_model_path).is_file()

    def _create_camera(self, camera_name: Literal["overview", "topdown"]) -> int:
        config = self.runtime_profile.coppelia
        options = 2 + 4 + 64 + 128
        world_size = self.config.grid_size * config.grid_scale_m
        site_center, site_span, site_min_y = self._construction_site_bounds()
        handle = self._sim.createVisionSensor(
            options,
            [config.camera_width, config.camera_height, 0, 0],
            [
                0.05,
                20.0,
                math.radians(
                    config.overview_camera_fov_degrees
                    if camera_name == "overview"
                    else 55.0
                ),
                0.08,
                0.0,
                0.0,
                0.92,
                0.94,
                0.97,
                0.0,
                0.0,
            ],
        )
        alias = f"construction_camera_{camera_name}"
        self._sim.setObjectAlias(handle, alias)
        self._sim.setObjectParent(handle, self._root_handle, True)
        if camera_name == "overview":
            camera_position = (
                site_center[0] - site_span * 0.75,
                site_min_y - site_span * 0.25,
                max(3.0, site_span * 0.72),
            )
            self._sim.setObjectMatrix(
                handle,
                self._look_at_matrix(
                    camera_position,
                    (site_center[0], site_center[1], 0.25),
                ),
            )
        else:
            self._sim.setObjectPosition(
                handle,
                [0.0, 0.0, max(5.0, world_size * 1.35)],
            )
            self._sim.setObjectOrientation(handle, [math.pi, 0.0, 0.0])
        self._object_handles[alias] = handle
        return int(handle)

    def _set_camera_look_at(
        self,
        camera_name: str,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> None:
        self._sim.setObjectMatrix(
            self._camera_handles[camera_name],
            self._look_at_matrix(position, target),
        )

    @staticmethod
    def _look_at_matrix(
        position: tuple[float, float, float],
        target: tuple[float, float, float],
    ) -> list[float]:
        origin = np.asarray(position, dtype=np.float64)
        forward = np.asarray(target, dtype=np.float64) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        return [
            float(right[0]),
            float(down[0]),
            float(forward[0]),
            float(origin[0]),
            float(right[1]),
            float(down[1]),
            float(forward[1]),
            float(origin[1]),
            float(right[2]),
            float(down[2]),
            float(forward[2]),
            float(origin[2]),
        ]

    def _construction_site_bounds(
        self,
    ) -> tuple[tuple[float, float], float, float]:
        scale = self.runtime_profile.coppelia.grid_scale_m
        world_size = self.config.grid_size * scale
        points = [
            grid_to_coppelia_world(
                coordinate,
                grid_size=self.config.grid_size,
                scale=scale,
                z=0.0,
            )[:2]
            for coordinate in self.config.agent_starts
        ]
        slots = {
            slot.required_resource_id: slot
            for slot in self.config.derived_blueprint_slots()
            if slot.required_resource_id is not None
        }
        for beam in self.config.beams:
            points.extend(
                grid_to_coppelia_world(
                    coordinate,
                    grid_size=self.config.grid_size,
                    scale=scale,
                    z=0.0,
                )[:2]
                for coordinate in (beam.pickup_left, beam.pickup_right)
            )
            slot = slots.get(beam.name)
            if slot is not None and slot.target_pose is not None:
                points.append(slot.target_pose.position_m[:2])
            else:
                points.extend(
                    grid_to_coppelia_world(
                        coordinate,
                        grid_size=self.config.grid_size,
                        scale=scale,
                        z=0.0,
                    )[:2]
                    for coordinate in (beam.assembly_left, beam.assembly_right)
                )
        min_x = min(point[0] for point in points)
        max_x = max(point[0] for point in points)
        min_y = min(point[1] for point in points)
        max_y = max(point[1] for point in points)
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        span = max(max_x - min_x, max_y - min_y, world_size * 0.35)
        return center, span, min_y

    def _sync_from_logical_state(self) -> None:
        frame = self.logical_env._snapshot_frame()
        self._sync_from_playback_frame(frame)

    def _sync_from_playback_frame(self, frame: AssemblyPlaybackFrame) -> None:
        scale = self.runtime_profile.coppelia.grid_scale_m
        for index, coordinate in enumerate(frame.agent_positions):
            alias = f"construction_agent_{index}"
            self._set_position(
                alias,
                grid_to_coppelia_world(
                    coordinate,
                    grid_size=self.config.grid_size,
                    scale=scale,
                    z=0.11,
                ),
            )
        installed = set(frame.completed_beams)
        resources = {
            resource.resource_id: resource
            for resource in self.config.derived_resources()
        }
        slots = {
            slot.required_resource_id: slot
            for slot in self.config.derived_blueprint_slots()
            if slot.required_resource_id is not None
        }
        for slot in self.config.derived_blueprint_slots():
            is_completed = slot.required_resource_id in installed
            for index, coordinate in enumerate(slot.target_cells):
                marker_position = grid_to_coppelia_world(
                    coordinate,
                    grid_size=self.config.grid_size,
                    scale=scale,
                    z=-0.05 if is_completed else 0.025,
                )
                self._set_position(
                    f"construction_blueprint_{slot.slot_id}_{index}",
                    marker_position,
                )
        for beam in self.config.beams:
            alias = f"construction_resource_{beam.name}"
            if beam.name in installed:
                target_slot = slots.get(beam.name)
                if target_slot is not None and target_slot.target_pose is not None:
                    position = target_slot.target_pose.position_m
                    self._set_orientation(
                        alias,
                        self._resource_target_orientation(
                            resources.get(beam.name), target_slot
                        ),
                    )
                else:
                    position = self._beam_center(
                        beam.assembly_left,
                        beam.assembly_right,
                        z=0.10,
                    )
            elif frame.carrying and frame.current_beam_name == beam.name:
                agent_world = [
                    grid_to_coppelia_world(
                        coordinate,
                        grid_size=self.config.grid_size,
                        scale=scale,
                        z=0.20,
                    )
                    for coordinate in frame.agent_positions
                ]
                position = (
                    (agent_world[0][0] + agent_world[1][0]) / 2.0,
                    (agent_world[0][1] + agent_world[1][1]) / 2.0,
                    0.20,
                )
            else:
                position = self._beam_center(
                    beam.pickup_left,
                    beam.pickup_right,
                    z=0.09,
                )
            self._set_position(alias, position)

    def _resource_target_orientation(
        self,
        resource: ConstructionResource | None,
        slot: BlueprintSlot,
    ) -> tuple[float, float, float]:
        if slot.target_pose is None:
            raise ValueError("target orientation requires a blueprint target pose")
        target = slot.target_pose.rotation_rpy_degrees
        asset_rotation = (0.0, 0.0, 0.0)
        if (
            resource is not None
            and resource.asset_key is not None
            and self._asset_catalog is not None
            and resource.asset_key in self._asset_catalog.components
        ):
            asset_rotation = self._asset_catalog.components[
                resource.asset_key
            ].orientation_rpy_degrees
        return (
            math.radians(target[0] + asset_rotation[0]),
            math.radians(target[1] + asset_rotation[1]),
            math.radians(target[2] + asset_rotation[2]),
        )

    def _beam_center(
        self,
        left: tuple[int, int],
        right: tuple[int, int],
        *,
        z: float,
    ) -> tuple[float, float, float]:
        scale = self.runtime_profile.coppelia.grid_scale_m
        left_world = grid_to_coppelia_world(
            left,
            grid_size=self.config.grid_size,
            scale=scale,
            z=z,
        )
        right_world = grid_to_coppelia_world(
            right,
            grid_size=self.config.grid_size,
            scale=scale,
            z=z,
        )
        return (
            (left_world[0] + right_world[0]) / 2.0,
            (left_world[1] + right_world[1]) / 2.0,
            z,
        )

    def _set_position(
        self,
        alias: str,
        position: tuple[float, float, float],
    ) -> None:
        handle = self._object_handles.get(alias)
        if handle is not None:
            self._sim.setObjectPosition(handle, list(position))

    def _set_orientation(
        self,
        alias: str,
        orientation: tuple[float, float, float],
    ) -> None:
        handle = self._object_handles.get(alias)
        if handle is not None:
            self._sim.setObjectOrientation(handle, list(orientation))

    def _camera_array(self, camera_name: str) -> np.ndarray:
        handle = self._camera_handles[camera_name]
        image, resolution = self._sim.getVisionSensorImg(handle)
        array = np.frombuffer(image, dtype=np.uint8).reshape(
            int(resolution[1]),
            int(resolution[0]),
            3,
        )
        return array

    def _annotate_frame(
        self,
        frame_array: np.ndarray,
        frame: AssemblyPlaybackFrame,
    ) -> np.ndarray:
        from PIL import Image, ImageDraw

        image = Image.fromarray(frame_array)
        draw = ImageDraw.Draw(image)
        completed = len(frame.completed_component_ids)
        total = len(self.config.beams)
        component = frame.current_beam_name or "complete"
        option = frame.selected_option or "reset"
        label = (
            f"Modular Room v0 | {completed}/{total} installed | "
            f"{component} | {option}"
        )
        draw.rectangle((0, 0, image.width, 28), fill=(18, 22, 27))
        draw.text((10, 8), label, fill=(245, 247, 250))
        return np.asarray(image)

    def _advance_simulation(self) -> None:
        for _ in range(self.runtime_profile.coppelia.control_steps_per_frame):
            self._client.step()
            self._simulation_step_count += 1

    def _ensure_stopped(self) -> None:
        if self._sim is None:
            return
        if self._sim.getSimulationState() == self._sim.simulation_stopped:
            return
        self._sim.stopSimulation()
        deadline = time.monotonic() + self.runtime_profile.coppelia.connection_timeout_s
        while time.monotonic() < deadline:
            if self._sim.getSimulationState() == self._sim.simulation_stopped:
                return
            time.sleep(0.02)
        raise CoppeliaSimBackendUnavailableError(
            "CoppeliaSim did not stop before the connection timeout."
        )

    def _simulation_time(self) -> float | None:
        if not self.is_ready:
            return None
        try:
            return float(self._sim.getSimulationTime())
        except Exception:
            return None

    def _time_step(self) -> float | None:
        if not self.is_ready:
            return None
        try:
            return float(self._sim.getSimulationTimeStep())
        except Exception:
            return None

    def _physics_engine(self) -> int | None:
        if not self.is_ready:
            return None
        try:
            return int(self._sim.getInt32Param(self._sim.intparam_dynamic_engine))
        except Exception:
            return None

    def _require_ready(self) -> None:
        if not self.is_ready or self._sim is None or self._client is None:
            raise CoppeliaSimBackendUnavailableError(
                "CoppeliaSim backend is not connected. Start CoppeliaSim with the ZMQ "
                "Remote API server and install requirements-sim-coppelia.txt."
            )
