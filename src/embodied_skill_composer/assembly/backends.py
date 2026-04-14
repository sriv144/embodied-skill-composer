from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from embodied_skill_composer.assembly.env import CollaborativeAssemblyEnv
from embodied_skill_composer.assembly.models import AssemblyRuntimeProfile, AssemblyScenarioConfig, EpisodeArtifact, OptionExecutionResult


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


@dataclass
class IsaacLabAssemblyBackend:
    config: AssemblyScenarioConfig
    runtime_profile: AssemblyRuntimeProfile
    seed: int = 7

    def __post_init__(self) -> None:
        raise NotImplementedError(
            "Isaac Lab backend is not implemented yet. The current milestone is to preserve the assembly task contract "
            "through the local_sandbox backend so the Isaac port can match it later."
        )


def build_assembly_backend(
    config: AssemblyScenarioConfig,
    runtime_profile: AssemblyRuntimeProfile,
    seed: int = 7,
) -> AssemblyTaskBackend:
    if runtime_profile.backend == "local_sandbox":
        return CollaborativeAssemblyEnv(config=config, seed=seed)
    return IsaacLabAssemblyBackend(config=config, runtime_profile=runtime_profile, seed=seed)
