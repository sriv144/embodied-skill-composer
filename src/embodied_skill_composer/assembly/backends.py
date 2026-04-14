from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import (
    AssemblyMetrics,
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    EpisodeArtifact,
    OptionExecutionResult,
)


@runtime_checkable
class AssemblyTaskBackend(Protocol):
    config: AssemblyScenarioConfig
    obs_dim: int
    state_dim: int
    team_option_obs_dim: int
    action_size: int
    option_size: int

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
            "backend": "isaac_lab",
            "runtime_profile": self.runtime_profile.name,
            "status": "stub",
            "message": self._error_message("get_option_episode_diagnostics"),
            "last_seed": self._last_seed,
        }

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
    return IsaacLabAssemblyBackend(config=config, runtime_profile=runtime_profile, seed=seed)
