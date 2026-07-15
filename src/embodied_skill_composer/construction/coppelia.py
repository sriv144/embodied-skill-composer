from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from embodied_skill_composer.construction.models import BuildPlan, ExecutionFrame, ExecutionTrace


class CoppeliaConstructionConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=23000, ge=1, le=65535)
    robot_model_path: str = (
        "C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/robots/mobile/"
        "KUKA YouBot.ttm"
    )
    robot_model_scale: float = Field(default=0.42, gt=0, le=2)
    use_robot_models: bool = True
    control_steps_per_frame: int = Field(default=1, ge=1, le=20)


class CoppeliaConstructionAdapter:
    """Kinematic Coppelia playback for the canonical Construction v2 trace."""

    controller_name = "kinematic_cooperative_pose_sync_v2"

    def __init__(
        self,
        plan: BuildPlan,
        trace: ExecutionTrace,
        *,
        config: CoppeliaConstructionConfig | None = None,
        client_factory: Callable[[CoppeliaConstructionConfig], Any] | None = None,
    ) -> None:
        self.plan = plan
        self.trace = trace
        self.config = config or CoppeliaConstructionConfig()
        self.client_factory = client_factory
        self.client: Any = None
        self.sim: Any = None
        self.root_handle: int | None = None
        self.module_handles: dict[str, int] = {}
        self.robot_handles: dict[str, int] = {}
        self.loaded_robot_models = 0
        self.physics_steps = 0
        self.is_ready = False

    def connect(self) -> None:
        self.client = (self.client_factory or _connect_client)(self.config)
        self.sim = self.client.require("sim")
        self._ensure_stopped()
        self._build_scene()
        self.is_ready = True

    def play(self, *, max_frames: int | None = None) -> dict[str, object]:
        self._require_ready()
        frames = self.trace.frames[:max_frames]
        self.client.setStepping(True)
        self.sim.startSimulation()
        try:
            for frame in frames:
                self.sync_frame(frame)
                for _ in range(self.config.control_steps_per_frame):
                    self.client.step()
                    self.physics_steps += 1
        finally:
            self.sim.stopSimulation()
        return self.diagnostics(frame_count=len(frames))

    def sync_frame(self, frame: ExecutionFrame) -> None:
        self._require_ready()
        module_by_id = {item.module_id: item for item in self.plan.modules}
        for state in frame.modules:
            handle = self.module_handles[state.module_id]
            module = module_by_id[state.module_id]
            self.sim.setObjectPosition(
                handle,
                [state.position.x, state.position.y, max(state.position.z, 0.04)],
            )
            rotation = (
                module.target_pose.rotation_rpy_degrees
                if state.status == "installed"
                else module.staging_pose.rotation_rpy_degrees
            )
            self.sim.setObjectOrientation(
                handle,
                [math.radians(rotation.x), math.radians(rotation.y), math.radians(rotation.z)],
            )
        for state in frame.robots:
            self.sim.setObjectPosition(
                self.robot_handles[state.robot_id],
                [state.position.x, state.position.y, 0.12],
            )

    def save_scene(self, output_path: Path) -> Path:
        self._require_ready()
        self._ensure_stopped()
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_frame(self.trace.frames[-1])
        self.sim.saveScene(str(output_path))
        return output_path

    def diagnostics(self, *, frame_count: int | None = None) -> dict[str, object]:
        return {
            "backend": "coppelia_sim",
            "controller": self.controller_name,
            "connected": self.is_ready,
            "plan_id": self.plan.plan_id,
            "trace_controller": self.trace.schedule.controller,
            "module_count": len(self.module_handles),
            "robot_count": len(self.robot_handles),
            "loaded_robot_models": self.loaded_robot_models,
            "frame_count": frame_count if frame_count is not None else len(self.trace.frames),
            "physics_steps": self.physics_steps,
            "logical_completion": self.trace.metrics.structure_completion_rate,
            "backend_limitations": [
                "Trace playback is kinematic pose synchronization.",
                "Wheel, arm, gripper, force, and contact dynamics are not controlled in v2.",
            ],
        }

    def _build_scene(self) -> None:
        self.root_handle = self.sim.createDummy(0.01)
        self.sim.setObjectAlias(self.root_handle, "ESCConstructionV2")
        floor = self._create_box(
            "construction_v2_floor",
            (0.0, 0.0, -0.08),
            (22.0, 18.0, 0.12),
            (0.12, 0.15, 0.14),
        )
        self.sim.setObjectInt32Param(floor, self.sim.shapeintparam_static, 1)
        first_frame = self.trace.frames[0]
        first_modules = {item.module_id: item for item in first_frame.modules}
        colors = {
            "foundation": (0.28, 0.31, 0.31),
            "roof_panel": (0.10, 0.14, 0.15),
            "door_panel": (0.45, 0.23, 0.10),
            "window_panel": (0.08, 0.45, 0.52),
        }
        for module in self.plan.modules:
            state = first_modules[module.module_id]
            handle = self._create_box(
                f"construction_v2_module_{module.module_id}",
                (state.position.x, state.position.y, max(state.position.z, 0.04)),
                (
                    module.dimensions.width,
                    module.dimensions.depth,
                    module.dimensions.height,
                ),
                colors.get(module.module_type.value, (0.84, 0.85, 0.81)),
            )
            self.module_handles[module.module_id] = handle
        for index, robot in enumerate(self.plan.robots):
            handle = self._create_robot(robot.robot_id, robot.start_pose.position)
            self.robot_handles[robot.robot_id] = handle
            if not self._robot_model_available():
                color = [(0.95, 0.31, 0.10), (0.92, 0.66, 0.10), (0.10, 0.58, 0.55), (0.20, 0.38, 0.80)][index]
                self.sim.setShapeColor(
                    handle,
                    None,
                    self.sim.colorcomponent_ambient_diffuse,
                    list(color),
                )

    def _create_box(
        self,
        alias: str,
        position: tuple[float, float, float],
        dimensions: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> int:
        handle = self.sim.createPrimitiveShape(
            self.sim.primitiveshape_cuboid,
            list(dimensions),
            2,
        )
        self.sim.setObjectAlias(handle, alias)
        self.sim.setObjectParent(handle, self.root_handle, True)
        self.sim.setObjectPosition(handle, list(position))
        self.sim.setShapeColor(
            handle,
            None,
            self.sim.colorcomponent_ambient_diffuse,
            list(color),
        )
        self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_static, 1)
        return handle

    def _create_robot(self, robot_id: str, position: Any) -> int:
        if not self._robot_model_available():
            return self._create_box(
                f"construction_v2_{robot_id}",
                (position.x, position.y, 0.15),
                (0.55, 0.42, 0.24),
                (0.95, 0.31, 0.10),
            )
        handle = self.sim.loadModel(str(Path(self.config.robot_model_path).resolve()))
        self.sim.setObjectAlias(handle, f"construction_v2_{robot_id}")
        descendants = self.sim.getObjectsInTree(handle)
        self.sim.scaleObjects(descendants, self.config.robot_model_scale, True)
        self.sim.setObjectParent(handle, self.root_handle, True)
        self.sim.setObjectPosition(handle, [position.x, position.y, 0.05])
        self.loaded_robot_models += 1
        return handle

    def _robot_model_available(self) -> bool:
        return self.config.use_robot_models and Path(self.config.robot_model_path).is_file()

    def _ensure_stopped(self) -> None:
        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()

    def _require_ready(self) -> None:
        if not self.is_ready:
            raise RuntimeError("Coppelia Construction v2 adapter is not connected")


def _connect_client(config: CoppeliaConstructionConfig):
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

    return RemoteAPIClient(host=config.host, port=config.port)
