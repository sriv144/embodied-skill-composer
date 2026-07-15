from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embodied_skill_composer.assembly.models import (
    PhysicalManipulationFeedback,
    PhysicalSensorConfig,
)


@dataclass
class PhysicalSensorSuite:
    config: PhysicalSensorConfig
    seed: int = 7
    _sample_count: int = field(default=0, init=False)
    _fresh_sample_count: int = field(default=0, init=False)
    _dropout_count: int = field(default=0, init=False)
    _last_fresh_physics_step: int | None = field(default=None, init=False)
    _last_feedback: PhysicalManipulationFeedback | None = field(default=None, init=False)
    _alignment_estimate: float | None = field(default=None, init=False)
    _force_estimates: dict[str, float] = field(default_factory=dict, init=False)
    _joint_estimates: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
        self._sample_count = 0
        self._fresh_sample_count = 0
        self._dropout_count = 0
        self._last_fresh_physics_step = None
        self._last_feedback = None
        self._alignment_estimate = None
        self._force_estimates = {}
        self._joint_estimates = {}

    def observe(
        self,
        truth: PhysicalManipulationFeedback,
        physics_step: int,
    ) -> PhysicalManipulationFeedback:
        self._sample_count += 1
        if not self.config.enabled:
            self._fresh_sample_count += 1
            self._last_fresh_physics_step = physics_step
            feedback = truth.model_copy(
                update={
                    "sensor_mode": "privileged",
                    "sensor_fresh": True,
                    "sensor_dropped": False,
                    "sensor_age_physics_steps": 0,
                    "sensor_sample_index": self._sample_count,
                }
            )
            self._last_feedback = feedback
            return feedback

        if self._rng.random() < self.config.dropout_probability:
            return self._dropout_feedback(truth, physics_step)
        return self._fresh_feedback(truth, physics_step)

    def diagnostics(self) -> dict[str, object]:
        return {
            "enabled": self.config.enabled,
            "sample_count": self._sample_count,
            "fresh_sample_count": self._fresh_sample_count,
            "dropout_count": self._dropout_count,
            "dropout_rate": (
                0.0 if self._sample_count == 0 else self._dropout_count / self._sample_count
            ),
            "alignment_noise_std_m": self.config.alignment_noise_std_m,
            "force_noise_std_n": self.config.force_noise_std_n,
            "joint_position_noise_std_m": self.config.joint_position_noise_std_m,
            "configured_dropout_probability": self.config.dropout_probability,
            "ema_alpha": self.config.ema_alpha,
        }

    def _dropout_feedback(
        self,
        truth: PhysicalManipulationFeedback,
        physics_step: int,
    ) -> PhysicalManipulationFeedback:
        self._dropout_count += 1
        if self._last_feedback is None:
            return truth.model_copy(
                update={
                    "current_alignment_error_m": None,
                    "last_contact_forces_n": {},
                    "gripper_state": "unknown",
                    "gripper_joint_positions_m": {},
                    "sensor_mode": "simulated",
                    "sensor_fresh": False,
                    "sensor_dropped": True,
                    "sensor_age_physics_steps": 0,
                    "sensor_sample_index": self._sample_count,
                }
            )
        age = 0
        if self._last_fresh_physics_step is not None:
            age = max(0, physics_step - self._last_fresh_physics_step)
        feedback = self._last_feedback.model_copy(
            update={
                "active_attachment_beam": truth.active_attachment_beam,
                "last_check_phase": truth.last_check_phase,
                "last_check_passed": truth.last_check_passed,
                "sensor_fresh": False,
                "sensor_dropped": True,
                "sensor_age_physics_steps": age,
                "sensor_sample_index": self._sample_count,
            }
        )
        self._last_feedback = feedback
        return feedback

    def _fresh_feedback(
        self,
        truth: PhysicalManipulationFeedback,
        physics_step: int,
    ) -> PhysicalManipulationFeedback:
        self._fresh_sample_count += 1
        alignment = self._noisy_nonnegative(
            truth.current_alignment_error_m,
            self.config.alignment_noise_std_m,
        )
        if alignment is not None:
            self._alignment_estimate = self._ema(self._alignment_estimate, alignment)

        forces = {
            agent: self._noisy_nonnegative(force, self.config.force_noise_std_n) or 0.0
            for agent, force in truth.last_contact_forces_n.items()
        }
        filtered_forces = {
            agent: self._ema(self._force_estimates.get(agent), force)
            for agent, force in forces.items()
        }
        self._force_estimates.update(filtered_forces)

        joints = {
            name: self._noisy_nonnegative(
                position,
                self.config.joint_position_noise_std_m,
            )
            or 0.0
            for name, position in truth.gripper_joint_positions_m.items()
        }
        filtered_joints = {
            name: self._ema(self._joint_estimates.get(name), position)
            for name, position in joints.items()
        }
        self._joint_estimates.update(filtered_joints)

        feedback = truth.model_copy(
            update={
                "current_alignment_error_m": self._alignment_estimate,
                "last_contact_forces_n": filtered_forces,
                "gripper_state": self._infer_gripper_state(filtered_joints),
                "gripper_joint_positions_m": filtered_joints,
                "sensor_mode": "simulated",
                "sensor_fresh": True,
                "sensor_dropped": False,
                "sensor_age_physics_steps": 0,
                "sensor_sample_index": self._sample_count,
            }
        )
        self._last_fresh_physics_step = physics_step
        self._last_feedback = feedback
        return feedback

    def _noisy_nonnegative(self, value: float | None, std: float) -> float | None:
        if value is None:
            return None
        return max(0.0, float(value + self._rng.normal(0.0, std)))

    def _ema(self, previous: float | None, current: float) -> float:
        if previous is None:
            return current
        alpha = self.config.ema_alpha
        return float(alpha * current + (1.0 - alpha) * previous)

    @staticmethod
    def _infer_gripper_state(
        positions: dict[str, float],
    ) -> str:
        if not positions:
            return "unknown"
        if max(abs(position) for position in positions.values()) < 0.001:
            return "open"
        if min(positions.values()) > 0.005:
            return "closed"
        return "transitioning"
