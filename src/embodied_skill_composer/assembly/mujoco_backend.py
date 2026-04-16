from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyPlaybackFrame,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BackendStatus,
    EpisodeArtifact,
    OptionExecutionResult,
    TeamOption,
)


class MuJoCoBackendUnavailableError(RuntimeError):
    """Raised when the MuJoCo backend is used without optional sim dependencies."""


def mujoco_available() -> bool:
    return importlib.util.find_spec("mujoco") is not None


@dataclass
class MuJoCoAssemblyBackend:
    config: AssemblyScenarioConfig
    runtime_profile: AssemblyRuntimeProfile
    seed: int = 7
    backend_name: str = field(default="mujoco_local", init=False)
    is_ready: bool = field(default_factory=mujoco_available, init=False)
    readiness_notes: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.logical_env = CollaborativeAssemblyEnv(self.config, seed=self.seed)
        self.model = None
        self.data = None
        self._scale = 0.35
        self._agent_joint_names = ["agent0_free", "agent1_free"]
        self._beam_joint_names = [f"{beam.name}_free" for beam in self.config.beams]
        self._mujoco = None
        self._renderer = None
        self._renderer_size = (960, 720)
        self._last_seed: int | None = None
        self.readiness_notes = [
            "MuJoCo local backend mirrors the current assembly option task in a 3D scene.",
            "Physics/render stepping happens in MuJoCo; v1 grasp/install semantics remain option-level and deterministic.",
            "Use this backend for Windows visual playback and portfolio video artifacts before ROS 2/Gazebo/Isaac.",
        ]
        if self.is_ready:
            self._initialize_mujoco()
        else:
            self.readiness_notes.append("Install optional dependencies with `pip install -r requirements-sim-mujoco.txt`.")

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

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._last_seed = self.seed if seed is None else seed
        observations, state = self.logical_env.reset(seed=seed)
        self._sync_from_logical_state()
        return observations, state

    def set_curriculum_stage(self, beam_count: int | None = None, stage_index: int | None = None) -> None:
        self.logical_env.set_curriculum_stage(beam_count=beam_count, stage_index=stage_index)
        self._sync_from_logical_state()

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        result = self.logical_env.step(actions)
        self._sync_from_logical_state()
        return result

    def build_artifact(self, policy_mode: str) -> EpisodeArtifact:
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

    def execute_team_option(self, option: int | TeamOption, max_primitive_steps: int | None = None) -> OptionExecutionResult:
        result = self.logical_env.execute_team_option(option, max_primitive_steps=max_primitive_steps)
        self._sync_from_logical_state()
        return result

    def get_option_episode_diagnostics(self) -> dict[str, object]:
        diagnostics = self.logical_env.get_option_episode_diagnostics()
        diagnostics["backend"] = self.backend_name
        diagnostics["runtime_profile"] = self.runtime_profile.name
        diagnostics["backend_status"] = self.get_backend_status().model_dump(mode="json")
        diagnostics["mujoco_model"] = {"available": self.is_ready, "nq": 0 if self.model is None else int(self.model.nq)}
        return diagnostics

    def get_backend_status(self) -> BackendStatus:
        notes = list(self.readiness_notes)
        if self.runtime_profile.notes:
            notes.append(self.runtime_profile.notes)
        return BackendStatus(backend_name=self.backend_name, is_ready=self.is_ready, readiness_notes=notes)

    def render_frame(self, frame: AssemblyPlaybackFrame | None = None, width: int = 960, height: int = 720) -> np.ndarray:
        self._require_ready()
        if frame is not None:
            self._sync_from_playback_frame(frame)
        if self._renderer is None or self._renderer_size != (width, height):
            self._renderer = self._mujoco.Renderer(self.model, height=height, width=width)
            self._renderer_size = (width, height)
        self._renderer.update_scene(self.data, camera="portfolio_cam")
        return self._renderer.render()

    def record_episode(
        self,
        output_path: Path,
        diagnostics: dict[str, object] | None = None,
        fps: int = 12,
        width: int = 960,
        height: int = 720,
    ) -> Path:
        self._require_ready()
        import imageio.v2 as imageio

        diagnostics = diagnostics or self.get_option_episode_diagnostics()
        frames = self._diagnostic_frames(diagnostics)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_frames = [self.render_frame(frame, width=width, height=height) for frame in frames]
        try:
            imageio.mimsave(output_path, rendered_frames, fps=fps)
        except Exception:
            fallback_dir = output_path.with_suffix("")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            for index, rendered in enumerate(rendered_frames):
                imageio.imwrite(fallback_dir / f"frame_{index:03d}.png", rendered)
            raise
        return output_path

    def launch_viewer_playback(
        self,
        diagnostics: dict[str, object] | None = None,
        seconds_per_frame: float = 0.08,
    ) -> None:
        self._require_ready()
        import mujoco.viewer

        frames = self._diagnostic_frames(diagnostics or self.get_option_episode_diagnostics())
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for frame in frames:
                self._sync_from_playback_frame(frame)
                self._mujoco.mj_forward(self.model, self.data)
                viewer.sync()
                time.sleep(seconds_per_frame)

    def render_ascii(self) -> str:
        return self.logical_env.render_ascii()

    def _initialize_mujoco(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._sync_from_logical_state()

    def _require_ready(self) -> None:
        if not self.is_ready or self.model is None or self.data is None:
            raise MuJoCoBackendUnavailableError(
                "MuJoCo backend is not ready. Install optional dependencies with "
                "`pip install -r requirements-sim-mujoco.txt`."
            )

    def _build_xml(self) -> str:
        grid_extent = self.config.grid_size * self._scale
        half = grid_extent / 2.0
        pickup_geoms = []
        assembly_geoms = []
        for beam in self.config.beams:
            for name, coord in [(f"{beam.name}_pickup_l", beam.pickup_left), (f"{beam.name}_pickup_r", beam.pickup_right)]:
                x, y = self._world_xy(coord)
                pickup_geoms.append(
                    f'<geom name="{name}" type="box" pos="{x:.3f} {y:.3f} 0.011" '
                    'size="0.135 0.135 0.01" rgba="1.0 0.72 0.18 0.72" contype="0" conaffinity="0"/>'
                )
            for name, coord in [(f"{beam.name}_assembly_l", beam.assembly_left), (f"{beam.name}_assembly_r", beam.assembly_right)]:
                x, y = self._world_xy(coord)
                assembly_geoms.append(
                    f'<geom name="{name}" type="box" pos="{x:.3f} {y:.3f} 0.014" '
                    'size="0.15 0.15 0.012" rgba="0.25 0.55 1.0 0.72" contype="0" conaffinity="0"/>'
                )
        beam_bodies = []
        for beam in self.config.beams:
            x, y, z = self._beam_world_pose(beam.name)
            beam_bodies.append(
                f'''
                <body name="{beam.name}" pos="{x:.3f} {y:.3f} {z:.3f}">
                  <freejoint name="{beam.name}_free"/>
                  <geom type="box" size="0.10 0.31 0.055" rgba="0.73 0.34 0.14 1"/>
                </body>
                '''
            )
        return f"""
        <mujoco model="collaborative_assembly">
          <compiler angle="degree"/>
          <option timestep="0.01" gravity="0 0 -9.81"/>
          <visual>
            <global offwidth="1280" offheight="960"/>
            <quality shadowsize="2048"/>
            <headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25"/>
          </visual>
          <asset>
            <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.9 0.93 0.96" rgb2="0.78 0.84 0.9"/>
            <material name="floor_grid" texture="grid" texrepeat="10 10" reflectance="0.12"/>
          </asset>
          <worldbody>
            <light name="key" pos="0 -3 5" dir="0 1 -1" directional="true"/>
            <camera name="portfolio_cam" pos="0 -5.2 5.4" xyaxes="1 0 0 0 0.72 0.69" fovy="45"/>
            <geom name="floor" type="plane" size="{half:.3f} {half:.3f} 0.05" material="floor_grid"/>
            {''.join(pickup_geoms)}
            {''.join(assembly_geoms)}
            <body name="agent0" pos="0 0 0.18">
              <freejoint name="agent0_free"/>
              <geom type="cylinder" size="0.14 0.16" rgba="0.85 0.12 0.12 1"/>
              <geom type="sphere" pos="0 0 0.20" size="0.11" rgba="1 0.45 0.38 1"/>
            </body>
            <body name="agent1" pos="0 0 0.18">
              <freejoint name="agent1_free"/>
              <geom type="cylinder" size="0.14 0.16" rgba="0.0 0.55 0.34 1"/>
              <geom type="sphere" pos="0 0 0.20" size="0.11" rgba="0.27 0.84 0.62 1"/>
            </body>
            {''.join(beam_bodies)}
          </worldbody>
        </mujoco>
        """

    def _sync_from_logical_state(self) -> None:
        if not self.is_ready or self.model is None or self.data is None:
            return
        current_beam = self.logical_env._current_beam().name
        for index, coord in enumerate(self.logical_env.state.agent_positions):
            x, y = self._world_xy(coord)
            self._set_freejoint_pose(self._agent_joint_names[index], x, y, 0.18)
        for beam in self.config.beams:
            x, y, z = self._beam_world_pose(beam.name, current_beam=current_beam)
            self._set_freejoint_pose(f"{beam.name}_free", x, y, z)
        self._mujoco.mj_forward(self.model, self.data)

    def _sync_from_playback_frame(self, frame: AssemblyPlaybackFrame) -> None:
        self._require_ready()
        for index, coord in enumerate(frame.agent_positions):
            x, y = self._world_xy(coord)
            self._set_freejoint_pose(self._agent_joint_names[index], x, y, 0.18)
        for beam in self.config.beams:
            x, y, z = self._beam_world_pose(beam.name, frame=frame)
            self._set_freejoint_pose(f"{beam.name}_free", x, y, z)
        self._mujoco.mj_forward(self.model, self.data)

    def _beam_world_pose(
        self,
        beam_name: str,
        current_beam: str | None = None,
        frame: AssemblyPlaybackFrame | None = None,
    ) -> tuple[float, float, float]:
        beam = next(item for item in self.config.beams if item.name == beam_name)
        installed = beam_name in self.logical_env.state.installed_beams
        carrying = self.logical_env.state.carrying and current_beam == beam_name
        agent_positions = self.logical_env.state.agent_positions
        if frame is not None:
            installed = beam_name in self._installed_beams_for_frame(frame)
            carrying = bool(frame.carrying and frame.current_beam_name == beam_name)
            agent_positions = frame.agent_positions
        if carrying:
            x0, y0 = self._world_xy(agent_positions[0])
            x1, y1 = self._world_xy(agent_positions[1])
            return (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.40)
        targets = [beam.assembly_left, beam.assembly_right] if installed else [beam.pickup_left, beam.pickup_right]
        x0, y0 = self._world_xy(targets[0])
        x1, y1 = self._world_xy(targets[1])
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.08 if installed else 0.07)

    def _installed_beams_for_frame(self, frame: AssemblyPlaybackFrame) -> set[str]:
        installed_count = min(frame.current_beam_index, len(self.config.beams))
        return {beam.name for beam in self.config.beams[:installed_count]}

    def _set_freejoint_pose(self, joint_name: str, x: float, y: float, z: float) -> None:
        joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_address : qpos_address + 7] = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[self.model.jnt_dofadr[joint_id] : self.model.jnt_dofadr[joint_id] + 6] = 0.0

    def _world_xy(self, coord: tuple[int, int]) -> tuple[float, float]:
        center = (self.config.grid_size - 1) / 2.0
        return ((coord[0] - center) * self._scale, (coord[1] - center) * self._scale)

    def _diagnostic_frames(self, diagnostics: dict[str, object]) -> list[AssemblyPlaybackFrame]:
        raw_frames = diagnostics.get("state_snapshots", [])
        if not isinstance(raw_frames, list):
            return []
        return [AssemblyPlaybackFrame.model_validate(frame) for frame in raw_frames]
