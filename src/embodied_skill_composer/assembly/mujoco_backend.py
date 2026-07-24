from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from embodied_skill_composer.assembly.env import AssemblyAction, CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyPlaybackFrame,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BackendStatus,
    ConstructionBrainObservation,
    EpisodeArtifact,
    OptionExecutionResult,
    PhysicalManipulationFeedback,
    TeamOption,
    VisualPerceptionEvaluation,
    VisualPerceptionFeedback,
)
from embodied_skill_composer.assembly.perception import (
    AssemblyVisualFrame,
    ClassicalAssemblyPerception,
    MultiObjectVisualTracker,
    assess_visual_terminal_readiness,
    evaluate_visual_perception,
)
from embodied_skill_composer.assembly.sensing import PhysicalSensorSuite


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
        # MuJoCo is an optional, dynamically imported dependency.  Keep its
        # runtime objects at the untyped boundary rather than importing it at
        # module load time (which would make the optional backend mandatory).
        self.model: Any = None
        self.data: Any = None
        self._scale = 0.35
        self._agent_joint_names = ["agent0_free", "agent1_free"]
        self._beam_joint_names = [f"{beam.name}_free" for beam in self.config.beams]
        self._finger_joint_names = [
            f"agent{agent}_{side}_finger_slide"
            for agent in range(2)
            for side in ("left", "right")
        ]
        self._finger_actuator_names = [
            f"agent{agent}_{side}_finger_servo"
            for agent in range(2)
            for side in ("left", "right")
        ]
        self._mujoco: Any = None
        self._renderer: Any = None
        self._renderer_size = (960, 720)
        self._perception_renderer: Any = None
        self._perception_renderer_size = (256, 256)
        self._last_seed: int | None = None
        self._control_substeps = 10
        self._settle_substeps = 8
        self._trajectory_capture_stride = 2
        self._physics_step_count = 0
        self._control_errors: list[float] = []
        self._physics_qpos_frames: list[np.ndarray] = []
        self._last_recording_source = "none"
        self._manipulation_alignment_tolerance = (
            self.runtime_profile.manipulation_alignment_tolerance_m
        )
        self._minimum_grip_force = self.runtime_profile.manipulation_min_grip_force_n
        self._physical_manipulation_checks: list[dict[str, object]] = []
        self._active_attachment_beam: str | None = None
        self._attachment_events: list[dict[str, object]] = []
        self._finger_closed_position = 0.035
        self._gripper_control_steps = 16
        self._gripper_events: list[dict[str, object]] = []
        self._physical_sensor_suite = PhysicalSensorSuite(
            self.runtime_profile.physical_sensors,
            seed=self.seed,
        )
        self._visual_estimator = ClassicalAssemblyPerception(
            self.runtime_profile.visual_perception
        )
        self._visual_tracker = MultiObjectVisualTracker(
            self.runtime_profile.visual_perception
        )
        self._visual_sample_count = 0
        self._visual_evaluations: list[VisualPerceptionEvaluation] = []
        self._visual_feedback_history: list[VisualPerceptionFeedback] = []
        self.readiness_notes = [
            "MuJoCo local backend executes assembly motion with physics-stepped mocap-weld pose tracking.",
            "Dual gripper contacts gate grasping and runtime weld constraints attach carried beams.",
            "Use this backend for Windows visual playback and portfolio video artifacts before ROS 2/Gazebo/Isaac.",
        ]
        if self.runtime_profile.physical_sensors.enabled:
            self.readiness_notes.append(
                "Simulated physical sensing adds deterministic noise, filtering, and dropout metadata."
            )
        if self.runtime_profile.visual_perception.enabled:
            self.readiness_notes.append(
                "Top-down RGB/depth perception estimates agents, resources, and blueprint cells."
            )
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
        self._physical_sensor_suite.reset(self._last_seed)
        self._reset_physics_control()
        self._sync_from_logical_state()
        return observations, state

    def set_curriculum_stage(self, beam_count: int | None = None, stage_index: int | None = None) -> None:
        self.logical_env.set_curriculum_stage(beam_count=beam_count, stage_index=stage_index)
        self._reset_attachment_constraints()
        self._sync_from_logical_state()

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        beam_name = self.logical_env._current_beam().name
        self._queue_physical_failure_for_actions(actions)
        result = self.logical_env.step(actions)
        info = result[4]
        if info.get("installed"):
            self._deactivate_beam_attachment(beam_name, reason="installed")
        self._drive_to_logical_state()
        if info.get("installed"):
            self._set_grasp_contacts_enabled(beam_name, True)
        if info.get("picked"):
            self._activate_beam_attachment(beam_name)
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

    def execute_team_option(self, option: int | TeamOption, max_primitive_steps: int | None = None) -> OptionExecutionResult:
        team_option = TeamOption(option)
        beam_name = self.logical_env._current_beam().name
        self._queue_physical_failure_for_option(team_option)
        first_new_frame = len(self.logical_env.frame_history)
        result = self.logical_env.execute_team_option(
            team_option,
            max_primitive_steps=max_primitive_steps,
        )
        if result.info.get("installed"):
            self._deactivate_beam_attachment(beam_name, reason="installed")
        new_frames = self.logical_env.frame_history[first_new_frame:]
        if new_frames:
            for frame in new_frames:
                self._drive_to_playback_frame(frame)
        else:
            self._drive_to_logical_state()
        if result.info.get("installed"):
            self._set_grasp_contacts_enabled(beam_name, True)
        if result.info.get("picked"):
            self._activate_beam_attachment(beam_name)
        return result

    def get_option_episode_diagnostics(self) -> dict[str, object]:
        diagnostics = self.logical_env.get_option_episode_diagnostics()
        diagnostics["backend"] = self.backend_name
        diagnostics["runtime_profile"] = self.runtime_profile.name
        diagnostics["backend_status"] = self.get_backend_status().model_dump(mode="json")
        diagnostics["mujoco_model"] = {
            "available": self.is_ready,
            "nq": 0 if self.model is None else int(self.model.nq),
            "obstacle_count": len(self.config.obstacle_cells),
        }
        diagnostics["mujoco_physics_control"] = self.get_physics_control_diagnostics()
        diagnostics["mujoco_visual_perception"] = (
            self.get_visual_perception_diagnostics()
        )
        return diagnostics

    def get_physics_control_diagnostics(self) -> dict[str, object]:
        errors = self._control_errors
        grip_force_samples: list[float] = []
        for check in self._physical_manipulation_checks:
            if not check.get("contact_required") or not check.get("dual_gripper_contact"):
                continue
            force = check.get("minimum_contact_force_n")
            if isinstance(force, (int, float)):
                grip_force_samples.append(float(force))
        return {
            "mode": "mocap_weld_pose_tracking",
            "control_substeps": self._control_substeps,
            "settle_substeps": self._settle_substeps,
            "trajectory_capture_stride": self._trajectory_capture_stride,
            "physics_step_count": self._physics_step_count,
            "trajectory_frame_count": len(self._physics_qpos_frames),
            "mean_target_error": 0.0 if not errors else float(np.mean(errors)),
            "max_target_error": 0.0 if not errors else float(np.max(errors)),
            "final_target_error": 0.0 if not errors else float(errors[-1]),
            "recording_source": self._last_recording_source,
            "manipulation_alignment_tolerance_m": self._manipulation_alignment_tolerance,
            "manipulation_min_grip_force_n": self._minimum_grip_force,
            "grip_force_summary_n": {
                "sample_count": len(grip_force_samples),
                "minimum": 0.0 if not grip_force_samples else float(np.min(grip_force_samples)),
                "mean": 0.0 if not grip_force_samples else float(np.mean(grip_force_samples)),
                "maximum": 0.0 if not grip_force_samples else float(np.max(grip_force_samples)),
            },
            "physical_manipulation_checks": list(self._physical_manipulation_checks),
            "active_attachment_beam": self._active_attachment_beam,
            "attachment_events": list(self._attachment_events),
            "articulated_grippers": {
                "actuator_count": 0 if self.model is None else int(self.model.nu),
                "closed_target": self._finger_closed_position,
                "control_steps": self._gripper_control_steps,
                "joint_positions": self._finger_positions(),
                "events": list(self._gripper_events),
            },
            "physical_sensors": self._physical_sensor_suite.diagnostics(),
        }

    def get_construction_observation(self) -> ConstructionBrainObservation:
        observation = self.logical_env.get_construction_observation()
        visual_feedback = None
        if self.runtime_profile.visual_perception.enabled:
            visual_feedback = self._observe_visual_perception()
        return observation.model_copy(
            update={
                "backend": self.backend_name,
                "physical_feedback": self._physical_sensor_suite.observe(
                    self._physical_manipulation_truth(),
                    physics_step=self._physics_step_count,
                ),
                "visual_feedback": visual_feedback,
            }
        )

    def _observe_visual_perception(self) -> VisualPerceptionFeedback:
        self._visual_sample_count += 1
        config = self.runtime_profile.visual_perception
        frame = self.capture_visual_frame(
            width=config.width,
            height=config.height,
            camera_name=config.camera_name,
        )
        feedback = self._visual_estimator.estimate(
            frame,
            sample_index=self._visual_sample_count,
        )
        feedback = self._visual_tracker.update(feedback)
        feedback = feedback.model_copy(
            update={
                "terminal_assessment": assess_visual_terminal_readiness(
                    feedback,
                    carrying=self.logical_env.state.carrying,
                    config=config,
                )
            }
        )
        self._visual_feedback_history.append(feedback)
        self._visual_evaluations.append(
            evaluate_visual_perception(feedback, self._visual_truth_positions())
        )
        return feedback

    def _visual_truth_positions(
        self,
    ) -> dict[str, list[tuple[float, float, float]]]:
        return {
            "agent": [
                self._xyz_tuple(self._body_position(f"agent{index}"))
                for index in range(self.num_agents)
            ],
            "resource": [
                self._xyz_tuple(self._body_position(beam.name))
                for beam in self.config.beams
            ],
            "blueprint_cell": [
                (*self._world_xy(cell), 0.014)
                for beam in self.config.beams
                for cell in (beam.assembly_left, beam.assembly_right)
            ],
        }

    def get_visual_perception_diagnostics(self) -> dict[str, object]:
        mean_errors = [
            evaluation.mean_position_error_m
            for evaluation in self._visual_evaluations
            if evaluation.position_errors_m
        ]
        latest = self._visual_evaluations[-1] if self._visual_evaluations else None
        latest_feedback = (
            self._visual_feedback_history[-1]
            if self._visual_feedback_history
            else None
        )
        assessment_reasons: dict[str, int] = {}
        for feedback in self._visual_feedback_history:
            assessment = feedback.terminal_assessment
            if assessment is None:
                continue
            assessment_reasons[assessment.reason] = (
                assessment_reasons.get(assessment.reason, 0) + 1
            )
        return {
            "enabled": self.runtime_profile.visual_perception.enabled,
            "tracking_enabled": (
                self.runtime_profile.visual_perception.tracking_enabled
            ),
            "estimated_state_control_enabled": (
                self.runtime_profile.visual_perception.estimated_state_control_enabled
            ),
            "camera_name": self.runtime_profile.visual_perception.camera_name,
            "sample_count": self._visual_sample_count,
            "samples_with_predictions": sum(
                feedback.predicted_estimate_count > 0
                for feedback in self._visual_feedback_history
            ),
            "predicted_estimate_count": sum(
                feedback.predicted_estimate_count
                for feedback in self._visual_feedback_history
            ),
            "terminal_assessment_reasons": assessment_reasons,
            "ready_assessment_count": assessment_reasons.get("ready", 0),
            "prediction_backed_ready_count": sum(
                feedback.terminal_assessment is not None
                and feedback.terminal_assessment.ready
                and feedback.terminal_assessment.uses_predicted_tracks
                for feedback in self._visual_feedback_history
            ),
            "mean_position_error_m": (
                0.0 if not mean_errors else float(np.mean(mean_errors))
            ),
            "latest_feedback": (
                None
                if latest_feedback is None
                else latest_feedback.model_dump(mode="json")
            ),
            "latest_evaluation": (
                None if latest is None else latest.model_dump(mode="json")
            ),
        }

    def _physical_manipulation_truth(self) -> PhysicalManipulationFeedback:
        latest_check = (
            self._physical_manipulation_checks[-1]
            if self._physical_manipulation_checks
            else None
        )
        raw_forces = {} if latest_check is None else latest_check.get("contact_forces_n", {})
        if not isinstance(raw_forces, dict):
            raw_forces = {}
        positions = self._finger_positions()
        gripper_state: Literal["open", "closed", "transitioning", "unknown"]
        if not positions or max(abs(position) for position in positions.values()) < 0.001:
            gripper_state = "open"
        elif min(positions.values()) > 0.005:
            gripper_state = "closed"
        else:
            gripper_state = "transitioning"
        last_check_phase: Literal["grasp", "install"] | None = None
        if latest_check is not None:
            raw_phase = latest_check.get("phase")
            if raw_phase == "grasp":
                last_check_phase = "grasp"
            elif raw_phase == "install":
                last_check_phase = "install"
        return PhysicalManipulationFeedback(
            backend=self.backend_name,
            current_alignment_error_m=self._terminal_alignment_error(),
            alignment_tolerance_m=self._manipulation_alignment_tolerance,
            required_minimum_grip_force_n=self._minimum_grip_force,
            last_check_phase=last_check_phase,
            last_check_passed=(
                None if latest_check is None else bool(latest_check.get("passed"))
            ),
            last_contact_forces_n={
                str(agent): float(force) for agent, force in raw_forces.items()
            },
            active_attachment_beam=self._active_attachment_beam,
            gripper_state=gripper_state,
            gripper_joint_positions_m=positions,
        )

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
        return np.asarray(self._renderer.render())

    def capture_visual_frame(
        self,
        width: int = 256,
        height: int = 256,
        camera_name: str = "perception_cam",
    ) -> AssemblyVisualFrame:
        self._require_ready()
        if (
            self._perception_renderer is None
            or self._perception_renderer_size != (width, height)
        ):
            self._perception_renderer = self._mujoco.Renderer(
                self.model,
                height=height,
                width=width,
            )
            self._perception_renderer_size = (width, height)
        renderer = self._perception_renderer
        renderer.disable_depth_rendering()
        renderer.disable_segmentation_rendering()
        renderer.update_scene(self.data, camera=camera_name)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()

        camera_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        return AssemblyVisualFrame(
            camera_name=camera_name,
            rgb=rgb,
            depth_m=depth,
            segmentation=segmentation,
            camera_position_m=self.data.cam_xpos[camera_id].copy(),
            camera_rotation=self.data.cam_xmat[camera_id].reshape(3, 3).copy(),
            vertical_fov_degrees=float(self.model.cam_fovy[camera_id]),
        )

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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if len(self._physics_qpos_frames) > 1:
            rendered_frames = self._render_physics_trajectory(width=width, height=height)
            self._last_recording_source = "physics_trajectory"
        else:
            frames = self._diagnostic_frames(diagnostics)
            rendered_frames = [self.render_frame(frame, width=width, height=height) for frame in frames]
            self._last_recording_source = "logical_snapshots"
        try:
            imageio.mimsave(output_path, cast(Any, rendered_frames), fps=fps)
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
        obstacle_geoms = []
        for index, coord in enumerate(self.config.obstacle_cells):
            x, y = self._world_xy(coord)
            obstacle_geoms.append(
                f'<geom name="obstacle_{index:03d}" type="box" '
                f'pos="{x:.3f} {y:.3f} 0.18" size="0.15 0.15 0.18" '
                'rgba="0.28 0.31 0.35 1" contype="8" conaffinity="6"/>'
            )
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
        beam_targets = []
        tracking_constraints = [
            '<weld name="agent0_track" body1="agent0_target" body2="agent0" solref="0.02 1"/>',
            '<weld name="agent1_track" body1="agent1_target" body2="agent1" solref="0.02 1"/>',
        ]
        for beam in self.config.beams:
            x, y, z = self._beam_world_pose(beam.name)
            beam_bodies.append(
                f'''
                <body name="{beam.name}" pos="{x:.3f} {y:.3f} {z:.3f}">
                  <freejoint name="{beam.name}_free"/>
                  <geom type="box" size="0.10 0.175 0.055" rgba="0.73 0.34 0.14 1"
                        contype="4" conaffinity="9"/>
                  <geom name="{beam.name}_grasp_geom" type="sphere" size="0.015"
                        rgba="0.95 0.82 0.2 0.35" contype="32" conaffinity="16"
                        condim="1" friction="0 0 0" solref="0.08 1"
                        solimp="0.7 0.9 0.01"/>
                  <site name="{beam.name}_attach_site" size="0.025" rgba="1 0.9 0.1 1"/>
                </body>
                '''
            )
            beam_targets.append(
                f'<body name="{beam.name}_target" mocap="true" '
                f'pos="{x:.3f} {y:.3f} {z:.3f}"/>'
            )
            tracking_constraints.append(
                f'<weld name="{beam.name}_track" body1="{beam.name}_target" '
                f'body2="{beam.name}" solref="0.02 1"/>'
            )
            tracking_constraints.append(
                f'<weld name="{beam.name}_carry" site1="{beam.name}_attach_site" '
                'site2="agent0_gripper_site" active="false" solref="0.02 1"/>'
            )
        return f"""
        <mujoco model="collaborative_assembly">
          <compiler angle="degree"/>
          <option timestep="0.01" gravity="0 0 -9.81" integrator="implicitfast"/>
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
            <camera name="perception_cam" pos="0 0 5.5" xyaxes="1 0 0 0 1 0" fovy="45"/>
            <geom name="floor" type="plane" size="{half:.3f} {half:.3f} 0.05"
                  material="floor_grid" contype="1" conaffinity="6"/>
            {''.join(pickup_geoms)}
            {''.join(assembly_geoms)}
            {''.join(obstacle_geoms)}
            <body name="agent0" pos="0 0 0.18">
              <freejoint name="agent0_free"/>
              <geom type="cylinder" size="0.14 0.16" rgba="0.85 0.12 0.12 1"
                    contype="2" conaffinity="11"/>
              <geom type="sphere" pos="0 0 0.20" size="0.11" rgba="1 0.45 0.38 1"
                    contype="2" conaffinity="11"/>
              <site name="agent0_gripper_site" pos="0 0.175 0.22" size="0.025"
                    rgba="1 0.95 0.2 1"/>
              <body name="agent0_left_finger" pos="-0.045 0.175 0.22">
                <joint name="agent0_left_finger_slide" type="slide" axis="1 0 0"
                       range="0 0.035" damping="0.2" armature="0.001"/>
                <geom name="agent0_left_finger_geom" type="box" size="0.008 0.018 0.025"
                      rgba="1 0.86 0.16 1" contype="16" conaffinity="32" condim="1"
                      friction="0 0 0" solref="0.08 1" solimp="0.7 0.9 0.01"/>
              </body>
              <body name="agent0_right_finger" pos="0.045 0.175 0.22">
                <joint name="agent0_right_finger_slide" type="slide" axis="-1 0 0"
                       range="0 0.035" damping="0.2" armature="0.001"/>
                <geom name="agent0_right_finger_geom" type="box" size="0.008 0.018 0.025"
                      rgba="1 0.86 0.16 1" contype="16" conaffinity="32" condim="1"
                      friction="0 0 0" solref="0.08 1" solimp="0.7 0.9 0.01"/>
              </body>
            </body>
            <body name="agent1" pos="0 0 0.18">
              <freejoint name="agent1_free"/>
              <geom type="cylinder" size="0.14 0.16" rgba="0.0 0.55 0.34 1"
                    contype="2" conaffinity="11"/>
              <geom type="sphere" pos="0 0 0.20" size="0.11" rgba="0.27 0.84 0.62 1"
                    contype="2" conaffinity="11"/>
              <site name="agent1_gripper_site" pos="0 -0.175 0.22" size="0.025"
                    rgba="1 0.95 0.2 1"/>
              <body name="agent1_left_finger" pos="-0.045 -0.175 0.22">
                <joint name="agent1_left_finger_slide" type="slide" axis="1 0 0"
                       range="0 0.035" damping="0.2" armature="0.001"/>
                <geom name="agent1_left_finger_geom" type="box" size="0.008 0.018 0.025"
                      rgba="1 0.86 0.16 1" contype="16" conaffinity="32" condim="1"
                      friction="0 0 0" solref="0.08 1" solimp="0.7 0.9 0.01"/>
              </body>
              <body name="agent1_right_finger" pos="0.045 -0.175 0.22">
                <joint name="agent1_right_finger_slide" type="slide" axis="-1 0 0"
                       range="0 0.035" damping="0.2" armature="0.001"/>
                <geom name="agent1_right_finger_geom" type="box" size="0.008 0.018 0.025"
                      rgba="1 0.86 0.16 1" contype="16" conaffinity="32" condim="1"
                      friction="0 0 0" solref="0.08 1" solimp="0.7 0.9 0.01"/>
              </body>
            </body>
            <body name="agent0_target" mocap="true" pos="0 0 0.18"/>
            <body name="agent1_target" mocap="true" pos="0 0 0.18"/>
            {''.join(beam_targets)}
            {''.join(beam_bodies)}
          </worldbody>
          <equality>
            {''.join(tracking_constraints)}
          </equality>
          <actuator>
            <position name="agent0_left_finger_servo" joint="agent0_left_finger_slide"
                      kp="2400" dampratio="1" ctrlrange="0 0.035" forcerange="-200 200"/>
            <position name="agent0_right_finger_servo" joint="agent0_right_finger_slide"
                      kp="2400" dampratio="1" ctrlrange="0 0.035" forcerange="-200 200"/>
            <position name="agent1_left_finger_servo" joint="agent1_left_finger_slide"
                      kp="2400" dampratio="1" ctrlrange="0 0.035" forcerange="-200 200"/>
            <position name="agent1_right_finger_servo" joint="agent1_right_finger_slide"
                      kp="2400" dampratio="1" ctrlrange="0 0.035" forcerange="-200 200"/>
          </actuator>
        </mujoco>
        """

    def _sync_from_logical_state(self) -> None:
        if not self.is_ready or self.model is None or self.data is None:
            return
        for body_name, pose in self._logical_body_poses().items():
            self._set_freejoint_pose(f"{body_name}_free", *pose)
            self._set_mocap_pose(f"{body_name}_target", *pose)
        self._mujoco.mj_forward(self.model, self.data)
        if not self._physics_qpos_frames:
            self._physics_qpos_frames.append(self.data.qpos.copy())

    def _sync_from_playback_frame(self, frame: AssemblyPlaybackFrame) -> None:
        self._require_ready()
        for body_name, pose in self._frame_body_poses(frame).items():
            self._set_freejoint_pose(f"{body_name}_free", *pose)
            self._set_mocap_pose(f"{body_name}_target", *pose)
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_physics_control(self) -> None:
        self._physics_step_count = 0
        self._control_errors = []
        self._physics_qpos_frames = []
        self._last_recording_source = "none"
        self._physical_manipulation_checks = []
        self._visual_sample_count = 0
        self._visual_evaluations = []
        self._visual_feedback_history = []
        self._visual_tracker.reset()
        self._reset_attachment_constraints()
        self._reset_grippers()

    def _queue_physical_failure_for_actions(self, actions: list[int]) -> None:
        if all(AssemblyAction(action) == AssemblyAction.GRAB for action in actions):
            self._queue_physical_manipulation_failure("grasp")
        elif all(AssemblyAction(action) == AssemblyAction.INSTALL for action in actions):
            self._queue_physical_manipulation_failure("install")

    def _queue_physical_failure_for_option(self, option: TeamOption) -> None:
        if option == TeamOption.GRAB:
            self._queue_physical_manipulation_failure("grasp")
        elif option == TeamOption.INSTALL:
            self._queue_physical_manipulation_failure("install")

    def _queue_physical_manipulation_failure(self, phase: str) -> None:
        if not self.is_ready or self.model is None or self.data is None:
            return
        if phase == "grasp":
            terminal_ready = not self.logical_env.state.carrying and self.logical_env._at_pickup()
        else:
            terminal_ready = self.logical_env.state.carrying and self.logical_env._at_assembly()
        if not terminal_ready:
            return

        beam_name = self.logical_env._current_beam().name
        beam_represented = beam_name in {beam.name for beam in self.config.beams}
        error = self._terminal_alignment_error()
        alignment_passed = error <= self._manipulation_alignment_tolerance
        contact_forces: dict[str, float] = {}
        contact_required = phase == "grasp" and beam_represented
        if phase == "grasp" and alignment_passed and beam_represented:
            self._prepare_grasp_contact(beam_name)
            contact_forces = self._gripper_contact_forces(beam_name)
        contact_agents = sorted(contact_forces)
        dual_contact = set(contact_agents) == {"agent0", "agent1"}
        minimum_measured_force = min(contact_forces.values(), default=0.0)
        grip_force_ready = not contact_required or (
            dual_contact and minimum_measured_force >= self._minimum_grip_force
        )
        attachment_ready = phase != "install" or (
            not beam_represented or self._active_attachment_beam == beam_name
        )
        passed = alignment_passed and grip_force_ready and attachment_ready
        self._physical_manipulation_checks.append(
            {
                "beam_name": beam_name,
                "phase": phase,
                "beam_represented_in_model": beam_represented,
                "alignment_error_m": error,
                "tolerance_m": self._manipulation_alignment_tolerance,
                "contact_required": contact_required,
                "contact_agents": contact_agents,
                "contact_forces_n": contact_forces,
                "dual_gripper_contact": dual_contact,
                "minimum_contact_force_n": minimum_measured_force,
                "required_minimum_grip_force_n": self._minimum_grip_force,
                "grip_force_ready": grip_force_ready,
                "attachment_ready": attachment_ready,
                "passed": passed,
            }
        )
        if not passed:
            if not alignment_passed:
                reason = (
                    f"physical_alignment_error_{error:.4f}m_exceeds_"
                    f"{self._manipulation_alignment_tolerance:.4f}m"
                )
            elif contact_required and not dual_contact:
                reason = "missing_dual_gripper_contact"
            elif contact_required and not grip_force_ready:
                reason = (
                    f"grip_force_{minimum_measured_force:.2f}N_below_"
                    f"{self._minimum_grip_force:.2f}N"
                )
            else:
                reason = "beam_attachment_not_active"
            self.logical_env.queue_manipulation_failure(
                phase,
                reason,
            )

    def _prepare_grasp_contact(self, beam_name: str) -> None:
        self._command_grippers(closed=False)
        poses = self._logical_body_poses()
        beam_pose = poses[beam_name]
        poses[beam_name] = (beam_pose[0], beam_pose[1], 0.40)
        self._drive_to_body_poses(poses)
        self._command_grippers(closed=True)

    def _dual_gripper_contacts(self, beam_name: str) -> set[str]:
        return set(self._gripper_contact_forces(beam_name))

    def _gripper_contact_forces(self, beam_name: str) -> dict[str, float]:
        beam_geom = f"{beam_name}_grasp_geom"
        detected: dict[str, float] = {}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_names = {
                self._mujoco.mj_id2name(
                    self.model,
                    self._mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom1),
                ),
                self._mujoco.mj_id2name(
                    self.model,
                    self._mujoco.mjtObj.mjOBJ_GEOM,
                    int(contact.geom2),
                ),
            }
            if beam_geom not in geom_names:
                continue
            contact_force = np.zeros(6, dtype=np.float64)
            self._mujoco.mj_contactForce(
                self.model,
                self.data,
                index,
                contact_force,
            )
            normal_force = abs(float(contact_force[0]))
            for agent_name in ("agent0", "agent1"):
                finger_geoms = {
                    f"{agent_name}_left_finger_geom",
                    f"{agent_name}_right_finger_geom",
                }
                if geom_names & finger_geoms:
                    detected[agent_name] = (
                        detected.get(agent_name, 0.0) + normal_force
                    )
        return detected

    def _reset_attachment_constraints(self) -> None:
        self._active_attachment_beam = None
        self._attachment_events = []
        if not self.is_ready or self.model is None or self.data is None:
            return
        for beam in self.config.beams:
            self._set_equality_active(f"{beam.name}_track", True)
            self._set_equality_active(f"{beam.name}_carry", False)
            self._set_grasp_contacts_enabled(beam.name, True)

    def _activate_beam_attachment(self, beam_name: str) -> None:
        if beam_name not in {beam.name for beam in self.config.beams}:
            return
        self._set_grasp_contacts_enabled(beam_name, False)
        self._set_equality_active(f"{beam_name}_track", False)
        self._set_equality_active(f"{beam_name}_carry", True)
        self._active_attachment_beam = beam_name
        self._attachment_events.append(
            {
                "beam_name": beam_name,
                "event": "attached",
                "physics_step": self._physics_step_count,
            }
        )
        self._mujoco.mj_forward(self.model, self.data)
        self._advance_physics(self._settle_substeps)

    def _deactivate_beam_attachment(self, beam_name: str, reason: str) -> None:
        if beam_name not in {beam.name for beam in self.config.beams}:
            return
        self._set_equality_active(f"{beam_name}_carry", False)
        self._set_equality_active(f"{beam_name}_track", True)
        self._active_attachment_beam = None
        self._attachment_events.append(
            {
                "beam_name": beam_name,
                "event": "detached",
                "reason": reason,
                "physics_step": self._physics_step_count,
            }
        )
        self._mujoco.mj_forward(self.model, self.data)
        self._command_grippers(closed=False)

    def _set_equality_active(self, equality_name: str, active: bool) -> None:
        equality_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_EQUALITY,
            equality_name,
        )
        self.data.eq_active[equality_id] = active

    def _set_grasp_contacts_enabled(self, beam_name: str, enabled: bool) -> None:
        settings = {
            "agent0_left_finger_geom": (16, 32),
            "agent0_right_finger_geom": (16, 32),
            "agent1_left_finger_geom": (16, 32),
            "agent1_right_finger_geom": (16, 32),
            f"{beam_name}_grasp_geom": (32, 16),
        }
        for geom_name, (contype, conaffinity) in settings.items():
            geom_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            self.model.geom_contype[geom_id] = contype if enabled else 0
            self.model.geom_conaffinity[geom_id] = conaffinity if enabled else 0

    def _reset_grippers(self) -> None:
        self._gripper_events = []
        if not self.is_ready or self.model is None or self.data is None:
            return
        for joint_name in self._finger_joint_names:
            joint_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = 0.0
        for actuator_name in self._finger_actuator_names:
            actuator_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            self.data.ctrl[actuator_id] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def _command_grippers(self, closed: bool) -> None:
        target = self._finger_closed_position if closed else 0.0
        positions = self._finger_positions()
        if positions and max(abs(value - target) for value in positions.values()) < 1e-4:
            return
        for actuator_name in self._finger_actuator_names:
            actuator_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            self.data.ctrl[actuator_id] = target
        self._advance_physics(self._gripper_control_steps)
        self._gripper_events.append(
            {
                "command": "close" if closed else "open",
                "target": target,
                "physics_step": self._physics_step_count,
                "joint_positions": self._finger_positions(),
            }
        )

    def _finger_positions(self) -> dict[str, float]:
        if not self.is_ready or self.model is None or self.data is None:
            return {}
        positions: dict[str, float] = {}
        for joint_name in self._finger_joint_names:
            joint_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            positions[joint_name] = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return positions

    def _advance_physics(self, steps: int) -> None:
        for _ in range(steps):
            self._mujoco.mj_step(self.model, self.data)
            self._physics_step_count += 1
            if self._physics_step_count % self._trajectory_capture_stride == 0:
                self._physics_qpos_frames.append(self.data.qpos.copy())

    def _terminal_alignment_error(self) -> float:
        poses = self._logical_body_poses()
        beam_name = self.logical_env._current_beam().name
        relevant_bodies = ["agent0", "agent1"]
        if beam_name in poses:
            relevant_bodies.append(beam_name)
        return max(
            float(
                np.linalg.norm(
                    self._body_position(body_name) - np.asarray(poses[body_name])
                )
            )
            for body_name in relevant_bodies
        )

    def _logical_body_poses(self) -> dict[str, tuple[float, float, float]]:
        poses = {
            f"agent{index}": (*self._world_xy(coord), 0.18)
            for index, coord in enumerate(self.logical_env.state.agent_positions)
        }
        current_beam = self.logical_env._current_beam().name
        poses.update(
            {
                beam.name: self._beam_world_pose(beam.name, current_beam=current_beam)
                for beam in self.config.beams
            }
        )
        return poses

    def _frame_body_poses(
        self,
        frame: AssemblyPlaybackFrame,
    ) -> dict[str, tuple[float, float, float]]:
        poses = {
            f"agent{index}": (*self._world_xy(coord), 0.18)
            for index, coord in enumerate(frame.agent_positions)
        }
        poses.update(
            {
                beam.name: self._beam_world_pose(beam.name, frame=frame)
                for beam in self.config.beams
            }
        )
        return poses

    def _drive_to_logical_state(self) -> None:
        self._drive_to_body_poses(self._logical_body_poses())

    def _drive_to_playback_frame(self, frame: AssemblyPlaybackFrame) -> None:
        self._drive_to_body_poses(self._frame_body_poses(frame))

    def _drive_to_body_poses(
        self,
        target_poses: dict[str, tuple[float, float, float]],
    ) -> None:
        self._require_ready()
        start_poses = {
            body_name: self._mocap_position(f"{body_name}_target")
            for body_name in target_poses
        }
        total_substeps = self._control_substeps + self._settle_substeps
        for substep in range(total_substeps):
            alpha = min(1.0, (substep + 1) / self._control_substeps)
            for body_name, target in target_poses.items():
                start = start_poses[body_name]
                interpolated = tuple(
                    start[axis] + alpha * (target[axis] - start[axis])
                    for axis in range(3)
                )
                self._set_mocap_pose(f"{body_name}_target", *interpolated)
            self._mujoco.mj_step(self.model, self.data)
            self._physics_step_count += 1
            if self._physics_step_count % self._trajectory_capture_stride == 0:
                self._physics_qpos_frames.append(self.data.qpos.copy())

        error = max(
            float(np.linalg.norm(self._body_position(body_name) - np.asarray(target)))
            for body_name, target in target_poses.items()
        )
        self._control_errors.append(error)

    def _render_physics_trajectory(self, width: int, height: int) -> list[np.ndarray]:
        self._require_ready()
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        rendered_frames: list[np.ndarray] = []
        try:
            for frame_qpos in self._physics_qpos_frames:
                self.data.qpos[:] = frame_qpos
                self.data.qvel[:] = 0.0
                self._mujoco.mj_forward(self.model, self.data)
                rendered_frames.append(self.render_frame(width=width, height=height).copy())
        finally:
            self.data.qpos[:] = qpos
            self.data.qvel[:] = qvel
            self._mujoco.mj_forward(self.model, self.data)
        return rendered_frames

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

    def _set_mocap_pose(self, body_name: str, x: float, y: float, z: float) -> None:
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = [x, y, z]
        self.data.mocap_quat[mocap_id] = [1.0, 0.0, 0.0, 0.0]

    def _mocap_position(self, body_name: str) -> np.ndarray:
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        mocap_id = self.model.body_mocapid[body_id]
        return np.asarray(self.data.mocap_pos[mocap_id]).copy()

    def _body_position(self, body_name: str) -> np.ndarray:
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        return np.asarray(self.data.xpos[body_id]).copy()

    @staticmethod
    def _xyz_tuple(values: np.ndarray) -> tuple[float, float, float]:
        return (float(values[0]), float(values[1]), float(values[2]))

    def _world_xy(self, coord: tuple[int, int]) -> tuple[float, float]:
        center = (self.config.grid_size - 1) / 2.0
        return ((coord[0] - center) * self._scale, (coord[1] - center) * self._scale)

    def _diagnostic_frames(self, diagnostics: dict[str, object]) -> list[AssemblyPlaybackFrame]:
        raw_frames = diagnostics.get("state_snapshots", [])
        if not isinstance(raw_frames, list):
            return []
        return [AssemblyPlaybackFrame.model_validate(frame) for frame in raw_frames]
