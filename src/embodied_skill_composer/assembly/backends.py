from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyMetrics,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    BackendStatus,
    EpisodeArtifact,
    OptionExecutionResult,
)
from embodied_skill_composer.assembly.mujoco_backend import MuJoCoAssemblyBackend


@runtime_checkable
class AssemblyTaskBackend(Protocol):
    config: AssemblyScenarioConfig
    obs_dim: int
    state_dim: int
    team_option_obs_dim: int
    action_size: int
    option_size: int
    backend_name: str
    is_ready: bool
    readiness_notes: list[str]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]: ...

    def set_curriculum_stage(self, beam_count: int | None = None, stage_index: int | None = None) -> None: ...

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, float, bool, dict]: ...

    def build_artifact(self, policy_mode: str) -> EpisodeArtifact: ...

    def get_agent_observations(self) -> np.ndarray: ...

    def get_action_masks(self) -> np.ndarray: ...

    def get_privileged_state(self) -> np.ndarray: ...

    def get_team_option_observation(self) -> np.ndarray: ...

    def get_team_option_mask(self) -> np.ndarray: ...

    def scripted_team_option(self): ...

    def execute_team_option(self, option: int, max_primitive_steps: int | None = None) -> OptionExecutionResult: ...

    def get_option_episode_diagnostics(self) -> dict[str, object]: ...

    def get_backend_status(self) -> BackendStatus: ...


class IsaacBackendNotReadyError(RuntimeError):
    """Raised when the Isaac backend contract is selected before simulator implementation exists."""


@dataclass
class IsaacLabAssemblyBackend:
    config: AssemblyScenarioConfig
    runtime_profile: AssemblyRuntimeProfile
    seed: int = 7
    obs_dim: int = 25
    state_dim: int = 16
    team_option_obs_dim: int = 25
    action_size: int = 7
    option_size: int = 8
    _last_seed: int | None = field(default=None, init=False)
    backend_name: str = field(default="isaac_lab", init=False)
    is_ready: bool = field(default=False, init=False)
    readiness_notes: list[str] = field(
        default_factory=lambda: [
            "Isaac Lab backend is a contract-preserving stub in this repository.",
            "Use Linux plus NVIDIA GPU for real Isaac work; keep Windows local_sandbox for regression.",
            "First bring-up target: scripted options on the same two-beam assembly task.",
            "Only simulator-dependent execution methods are blocked right now.",
        ],
        init=False,
    )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._last_seed = self.seed if seed is None else seed
        return (
            np.zeros(self.team_option_obs_dim, dtype=np.float32),
            np.zeros(self.state_dim, dtype=np.float32),
        )

    def set_curriculum_stage(self, beam_count: int | None = None, stage_index: int | None = None) -> None:
        _ = (beam_count, stage_index)

    def step(self, actions: list[int]) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        _ = actions
        self._raise_not_ready("step")

    def build_artifact(self, policy_mode: str) -> EpisodeArtifact:
        _ = policy_mode
        return EpisodeArtifact(
            metrics=AssemblyMetrics(
                success=False,
                beams_installed=0,
                total_beams=len(self.config.beams),
                step_count=0,
                total_reward=0.0,
                collision_count=0,
                invalid_action_count=0,
                deadlock_steps=0,
                coordination_efficiency=0.0,
            ),
            final_positions=list(self.config.agent_starts),
            carrying=False,
            completed_beams=[],
            policy_mode="scripted",
        )

    def get_agent_observations(self) -> np.ndarray:
        self._raise_not_ready("get_agent_observations")

    def get_action_masks(self) -> np.ndarray:
        self._raise_not_ready("get_action_masks")

    def get_privileged_state(self) -> np.ndarray:
        return np.zeros(self.state_dim, dtype=np.float32)

    def get_team_option_observation(self) -> np.ndarray:
        return np.zeros(self.team_option_obs_dim, dtype=np.float32)

    def get_team_option_mask(self) -> np.ndarray:
        return np.ones(self.option_size, dtype=np.float32)

    def scripted_team_option(self):
        self._raise_not_ready("scripted_team_option")

    def execute_team_option(self, option: int, max_primitive_steps: int | None = None) -> OptionExecutionResult:
        _ = (option, max_primitive_steps)
        self._raise_not_ready("execute_team_option")

    def get_option_episode_diagnostics(self) -> dict[str, object]:
        return {
            "backend": self.backend_name,
            "runtime_profile": self.runtime_profile.name,
            "status": "stub",
            "message": self._error_message("get_option_episode_diagnostics"),
            "last_seed": self._last_seed,
            "backend_status": self.get_backend_status().model_dump(mode="json"),
        }

    def get_backend_status(self) -> BackendStatus:
        notes = list(self.readiness_notes)
        notes.append(f"Runtime profile: {self.runtime_profile.name}")
        if self.runtime_profile.requires_linux:
            notes.append("Requires a Linux host before simulator bring-up.")
        if self.runtime_profile.requires_nvidia_gpu:
            notes.append("Requires a CUDA-capable NVIDIA GPU environment.")
        if self.runtime_profile.notes:
            notes.append(self.runtime_profile.notes)
        notes.extend(
            [
                "Bring-up checklist:",
                "1. Prepare Ubuntu LTS with recent NVIDIA drivers.",
                "2. Install Isaac Sim / Isaac Lab in an isolated conda or micromamba environment.",
                "3. Recreate the two-beam assembly task semantics and scripted option execution.",
                "4. Match artifacts and diagnostics before training learned policies.",
            ]
        )
        return BackendStatus(
            backend_name=self.backend_name,
            is_ready=self.is_ready,
            readiness_notes=notes,
        )

    def _raise_not_ready(self, method_name: str):
        raise IsaacBackendNotReadyError(self._error_message(method_name))

    def _error_message(self, method_name: str) -> str:
        return (
            f"Isaac Lab backend stub selected via runtime profile '{self.runtime_profile.name}', but method "
            f"'{method_name}' requires a real Isaac simulator implementation. Keep using 'local_sandbox' for regression "
            "runs and implement the Isaac backend against the same assembly task contract first."
        )


def build_assembly_backend(
    config: AssemblyScenarioConfig,
    runtime_profile: AssemblyRuntimeProfile,
    seed: int = 7,
) -> AssemblyTaskBackend:
    if runtime_profile.backend == "local_sandbox":
        return CollaborativeAssemblyEnv(config=config, seed=seed)
    if runtime_profile.backend == "mujoco_local":
        return MuJoCoAssemblyBackend(config=config, runtime_profile=runtime_profile, seed=seed)
    return IsaacLabAssemblyBackend(config=config, runtime_profile=runtime_profile, seed=seed)
