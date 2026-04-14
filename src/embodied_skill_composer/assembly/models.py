from __future__ import annotations

from enum import IntEnum
from typing import Literal

from pydantic import BaseModel, Field


GridCoord = tuple[int, int]


class BeamTask(BaseModel):
    name: str
    pickup_left: GridCoord
    pickup_right: GridCoord
    assembly_left: GridCoord
    assembly_right: GridCoord


class AssemblyScenarioConfig(BaseModel):
    grid_size: int = 12
    max_steps: int = 120
    agent_starts: list[GridCoord] = Field(default_factory=lambda: [(0, 0), (0, 1)])
    beams: list[BeamTask]
    collision_penalty: float = 0.2
    invalid_action_penalty: float = 0.1
    step_penalty: float = 0.01
    grasp_reward: float = 0.5
    install_reward: float = 1.5
    completion_reward: float = 5.0
    distance_shaping: float = 0.02
    second_beam_pickup_bonus: float = 0.05
    second_beam_install_bonus: float = 0.08
    curriculum_beam_stages: list[int] = Field(default_factory=lambda: [1, 2])
    curriculum_stage_beams: list[list[BeamTask]] = Field(default_factory=list)
    option_max_primitive_steps: int = 24


class TrainingConfig(BaseModel):
    total_iterations: int = 20
    episodes_per_iteration: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 7
    behavior_cloning_epochs: int = 60
    behavior_cloning_lr: float = 1e-3
    scripted_mixing_start: float = 0.8
    scripted_mixing_end: float = 0.1
    evaluation_episodes: int = 5
    behavior_cloning_aux_coef: float = 0.2
    curriculum_stage_iterations: list[int] = Field(default_factory=lambda: [0, 6])
    option_actor_lr: float = 3e-4
    option_critic_lr: float = 1e-3
    option_behavior_cloning_epochs: int = 120
    option_behavior_cloning_lr: float = 8e-4
    option_update_epochs: int = 4
    option_entropy_coef: float = 0.01
    option_behavior_cloning_aux_coef: float = 0.1
    option_scripted_mixing_start: float = 0.7
    option_scripted_mixing_end: float = 0.05
    option_switch_penalty: float = 0.0
    option_recovery_limit: int = 3


class AssemblyRuntimeProfile(BaseModel):
    name: str = "local_dev"
    backend: Literal["local_sandbox", "isaac_lab"] = "local_sandbox"
    device: str | None = None
    requires_linux: bool = False
    requires_nvidia_gpu: bool = False
    notes: str = ""


class BackendStatus(BaseModel):
    backend_name: str
    is_ready: bool
    readiness_notes: list[str] = Field(default_factory=list)


class GpuRuntimeStatus(BaseModel):
    runtime_profile: str
    backend: str
    requested_device: str | None
    torch_installed: bool
    cuda_available: bool
    selected_device: str
    device_name: str | None = None
    tensor_allocation_ok: bool = False
    notes: list[str] = Field(default_factory=list)


class TeamOption(IntEnum):
    GO_PICKUP = 0
    GRAB = 1
    GO_ASSEMBLY = 2
    INSTALL = 3
    RESET_TO_PICKUP_ROUTE = 4
    REPOSITION_AFTER_INSTALL = 5
    WAIT = 6
    ALIGN_FOR_TERMINAL_ACTION = 7


class OptionExecutionResult(BaseModel):
    option: TeamOption
    reward: float
    primitive_steps: int
    done: bool
    success: bool
    info: dict[str, str | float | int | bool | None] = Field(default_factory=dict)


class OptionTrainingSample(BaseModel):
    observation: list[float]
    action_mask: list[float]
    action: TeamOption
    stage_index: int


class OptionPolicyMetrics(BaseModel):
    success_rate: float
    mean_return: float
    mean_beams_installed: float
    mean_option_switches: float
    mean_recovery_usage: float


class AssemblyPlaybackFrame(BaseModel):
    step_count: int
    current_beam_index: int
    current_beam_name: str | None = None
    carrying: bool
    agent_positions: list[GridCoord]
    pickup_targets: list[GridCoord] = Field(default_factory=list)
    assembly_targets: list[GridCoord] = Field(default_factory=list)
    selected_option: str | None = None
    primitive_step_index: int = 0
    option_reward: float = 0.0
    option_success: bool | None = None


class PolicyBenchmarkResult(BaseModel):
    policy_name: str
    success_rate: float
    mean_return: float
    mean_beams_installed: float
    notes: str = ""


class AssemblyBenchmarkSummary(BaseModel):
    backend: str
    runtime_profile: str
    scripted_options: PolicyBenchmarkResult
    learned_options: PolicyBenchmarkResult
    low_level_learned: PolicyBenchmarkResult


class AssemblyMetrics(BaseModel):
    success: bool
    beams_installed: int
    total_beams: int
    step_count: int
    total_reward: float
    collision_count: int
    invalid_action_count: int
    deadlock_steps: int
    coordination_efficiency: float


class EpisodeArtifact(BaseModel):
    metrics: AssemblyMetrics
    final_positions: list[GridCoord]
    carrying: bool
    completed_beams: list[str]
    policy_mode: Literal["scripted", "learned"]
