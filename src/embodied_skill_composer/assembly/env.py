from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from random import Random
from typing import Callable, Literal

import numpy as np

from embodied_skill_composer.assembly.models import (
    AssemblyMetrics,
    AssemblyPlaybackFrame,
    AssemblyScenarioConfig,
    BackendStatus,
    BeamTask,
    BlueprintSlot,
    BlueprintSlotState,
    ConstructionBrainObservation,
    ConstructionProgress,
    ConstructionResource,
    ConstructionResourceState,
    EpisodeArtifact,
    OptionExecutionResult,
    TeamOption,
)


class AssemblyAction(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    GRAB = 5
    INSTALL = 6


@dataclass
class AssemblyState:
    agent_positions: list[tuple[int, int]]
    current_beam_index: int
    carrying: bool
    installed_beams: list[str]
    step_count: int
    total_reward: float
    collision_count: int
    invalid_action_count: int
    deadlock_steps: int
    energy_cost: float
    idle_step_count: int
    wasted_step_count: int
    obstacle_collision_count: int
    manipulation_attempts: dict[str, int]
    manipulation_failure_count: int
    manipulation_recovery_count: int
    last_manipulation_failure: str | None


class CollaborativeAssemblyEnv:
    num_agents = 2
    action_size = len(AssemblyAction)
    option_size = len(TeamOption)
    backend_name = "local_sandbox"
    is_ready = True
    readiness_notes = [
        "Local sandbox backend is the regression oracle for collaborative assembly.",
        "Use this backend for fast training, playback, and benchmark checks on Windows or Linux.",
    ]

    def __init__(self, config: AssemblyScenarioConfig, seed: int = 7) -> None:
        self.config = config
        self.random = Random(seed)
        self.active_beam_count = len(self.config.beams)
        self.active_stage_index: int | None = None
        self.state = self._initial_state()
        self._queued_manipulation_failures: dict[str, list[str]] = {}
        self._failed_manipulations: set[str] = set()
        self._recovered_manipulations: set[str] = set()
        self.manipulation_failure_history: list[dict[str, str | int | bool]] = []
        self.option_history: list[OptionExecutionResult] = []
        self.option_switch_count = 0
        self.recovery_option_usage = {"reset_to_pickup_route": 0, "reposition_after_install": 0}
        self.milestones: dict[str, int | None] = {
            "first_beam_completion_step": None,
            "second_beam_pickup_step": None,
            "second_beam_install_step": None,
        }
        self._last_option: TeamOption | None = None
        self.frame_history: list[AssemblyPlaybackFrame] = [self._snapshot_frame()]

    @property
    def obs_dim(self) -> int:
        return len(self.get_agent_observations()[0])

    @property
    def state_dim(self) -> int:
        return len(self.get_privileged_state())

    @property
    def team_option_obs_dim(self) -> int:
        return len(self.get_team_option_observation())

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.random.seed(seed)
        self.state = self._initial_state()
        self._queued_manipulation_failures = {}
        self._failed_manipulations = set()
        self._recovered_manipulations = set()
        self.manipulation_failure_history = []
        self.option_history = []
        self.option_switch_count = 0
        self.recovery_option_usage = {"reset_to_pickup_route": 0, "reposition_after_install": 0}
        self.milestones = {
            "first_beam_completion_step": None,
            "second_beam_pickup_step": None,
            "second_beam_install_step": None,
        }
        self._last_option = None
        self.frame_history = [self._snapshot_frame()]
        return self.get_agent_observations(), self.get_privileged_state()

    def set_curriculum_stage(self, beam_count: int | None = None, stage_index: int | None = None) -> None:
        if beam_count is None and stage_index is None:
            self.active_beam_count = len(self.config.beams)
            self.active_stage_index = None
            return
        if stage_index is not None and self.config.curriculum_stage_beams:
            self.active_stage_index = max(0, min(int(stage_index), len(self.config.curriculum_stage_beams) - 1))
            self.active_beam_count = len(self._available_beams())
            return
        self.active_stage_index = None
        requested_beam_count = len(self.config.beams) if beam_count is None else beam_count
        self.active_beam_count = max(1, min(requested_beam_count, len(self.config.beams)))

    def step(
        self,
        actions: list[int],
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        bool,
        dict[str, str | float | int | bool | None],
    ]:
        if len(actions) != self.num_agents:
            raise ValueError(f"expected {self.num_agents} actions, got {len(actions)}")
        beam = self._current_beam()
        reward = -self.config.step_penalty
        before_distance = self._distance_to_objective()
        before_second_pickup = self._second_beam_pickup_distance()
        before_second_install = self._second_beam_install_distance()
        before_collision_count = self.state.collision_count
        before_invalid_action_count = self.state.invalid_action_count
        before_deadlock_steps = self.state.deadlock_steps
        before_manipulation_failures = self.state.manipulation_failure_count
        self.state.energy_cost += float(sum(1 for action in actions if AssemblyAction(action) != AssemblyAction.STAY))
        self.state.idle_step_count += sum(1 for action in actions if AssemblyAction(action) == AssemblyAction.STAY)
        info: dict[str, str | float | int | bool | None] = {
            "picked": False,
            "installed": False,
            "manipulation_failed": False,
            "failure_reason": None,
            "retryable": False,
        }

        if self.state.carrying:
            reward += self._step_carrying(actions)
        else:
            reward += self._step_independent(actions)
            if all(action == AssemblyAction.GRAB for action in actions):
                if self._at_positions([beam.pickup_left, beam.pickup_right]):
                    failure_reason = self._attempt_manipulation(beam.name, "grasp")
                    if failure_reason is None:
                        self.state.carrying = True
                        reward += self.config.grasp_reward
                        info["picked"] = True
                    else:
                        reward -= self.config.manipulation_failure_penalty
                        info["manipulation_failed"] = True
                        info["failure_reason"] = failure_reason
                        info["retryable"] = True
                else:
                    reward -= self.config.invalid_action_penalty
                    self.state.invalid_action_count += 1

        if self.state.carrying and all(action == AssemblyAction.INSTALL for action in actions):
            if self._at_positions([beam.assembly_left, beam.assembly_right]):
                failure_reason = self._attempt_manipulation(beam.name, "install")
                if failure_reason is None:
                    self.state.carrying = False
                    self.state.installed_beams.append(beam.name)
                    self.state.current_beam_index += 1
                    reward += self.config.install_reward
                    info["installed"] = True
                else:
                    reward -= self.config.manipulation_failure_penalty
                    info["manipulation_failed"] = True
                    info["failure_reason"] = failure_reason
                    info["retryable"] = True
            else:
                reward -= self.config.invalid_action_penalty
                self.state.invalid_action_count += 1

        after_distance = self._distance_to_objective()
        reward += self.config.distance_shaping * (before_distance - after_distance)
        reward += self._phase_two_bonus(before_second_pickup, before_second_install)
        if (
            self.state.collision_count > before_collision_count
            or self.state.invalid_action_count > before_invalid_action_count
            or self.state.deadlock_steps > before_deadlock_steps
            or self.state.manipulation_failure_count > before_manipulation_failures
        ):
            self.state.wasted_step_count += 1
        self.state.step_count += 1
        self.state.total_reward += reward

        done = self.state.current_beam_index >= self.active_beam_count or self.state.step_count >= self.config.max_steps
        if self.state.current_beam_index >= self.active_beam_count:
            reward += self.config.completion_reward
            self.state.total_reward += self.config.completion_reward

        return self.get_agent_observations(), self.get_privileged_state(), reward, done, info

    def build_artifact(
        self,
        policy_mode: Literal["scripted", "learned", "brain"],
    ) -> EpisodeArtifact:
        construction = self._construction_summaries()
        metrics = AssemblyMetrics(
            success=self.state.current_beam_index >= self.active_beam_count,
            beams_installed=len(self.state.installed_beams),
            total_beams=self.active_beam_count,
            step_count=self.state.step_count,
            total_reward=self.state.total_reward,
            collision_count=self.state.collision_count,
            invalid_action_count=self.state.invalid_action_count,
            deadlock_steps=self.state.deadlock_steps,
            coordination_efficiency=len(self.state.installed_beams) / max(1, self.state.step_count),
            structure_completion_rate=_metric_float(
                construction["structure_completion_rate"],
                "structure_completion_rate",
            ),
            resource_delivery_accuracy=_metric_float(
                construction["resource_delivery_accuracy"],
                "resource_delivery_accuracy",
            ),
            energy_cost=self.state.energy_cost,
            idle_step_count=self.state.idle_step_count,
            wasted_step_count=self.state.wasted_step_count,
            obstacle_collision_count=self.state.obstacle_collision_count,
            manipulation_failure_count=self.state.manipulation_failure_count,
            manipulation_recovery_count=self.state.manipulation_recovery_count,
        )
        return EpisodeArtifact(
            metrics=metrics,
            final_positions=self.state.agent_positions,
            carrying=self.state.carrying,
            completed_beams=self.state.installed_beams,
            policy_mode=policy_mode,
        )

    def get_construction_observation(self) -> ConstructionBrainObservation:
        construction = self._construction_summaries()
        resource_inventory = construction["resource_inventory"]
        blueprint_slots = construction["blueprint_slots"]
        if not isinstance(resource_inventory, list) or not isinstance(blueprint_slots, list):
            raise RuntimeError("construction summaries must contain list-based resource and slot state")

        available_options = [
            TeamOption(index)
            for index, is_available in enumerate(self.get_team_option_mask())
            if is_available > 0
        ]
        current_beam_name = None
        if self.state.current_beam_index < self.active_beam_count:
            current_beam_name = self._current_beam().name

        return ConstructionBrainObservation(
            backend=self.backend_name,
            step_count=self.state.step_count,
            current_beam_index=self.state.current_beam_index,
            current_beam_name=current_beam_name,
            agent_positions=list(self.state.agent_positions),
            carrying=self.state.carrying,
            completed_beams=list(self.state.installed_beams),
            obstacle_cells=list(self.config.obstacle_cells),
            manipulation_attempts=dict(self.state.manipulation_attempts),
            last_manipulation_failure=self.state.last_manipulation_failure,
            resources=[ConstructionResourceState.model_validate(item) for item in resource_inventory],
            blueprint_slots=[BlueprintSlotState.model_validate(item) for item in blueprint_slots],
            progress=ConstructionProgress(
                structure_completion_rate=_metric_float(
                    construction["structure_completion_rate"],
                    "structure_completion_rate",
                ),
                resource_delivery_accuracy=_metric_float(
                    construction["resource_delivery_accuracy"],
                    "resource_delivery_accuracy",
                ),
                energy_cost=self.state.energy_cost,
                idle_step_count=self.state.idle_step_count,
                wasted_step_count=self.state.wasted_step_count,
                collision_count=self.state.collision_count,
                obstacle_collision_count=self.state.obstacle_collision_count,
                manipulation_failure_count=self.state.manipulation_failure_count,
                manipulation_recovery_count=self.state.manipulation_recovery_count,
                coordination_efficiency=len(self.state.installed_beams) / max(1, self.state.step_count),
            ),
            available_options=available_options,
        )

    def get_agent_observations(self) -> np.ndarray:
        beam = self._current_beam()
        observations: list[list[float]] = []
        for index, (x, y) in enumerate(self.state.agent_positions):
            mate_x, mate_y = self.state.agent_positions[1 - index]
            targets = [beam.assembly_left, beam.assembly_right] if self.state.carrying else [beam.pickup_left, beam.pickup_right]
            target = targets[index]
            observations.append(
                [
                    x / self.config.grid_size,
                    y / self.config.grid_size,
                    (target[0] - x) / self.config.grid_size,
                    (target[1] - y) / self.config.grid_size,
                    (mate_x - x) / self.config.grid_size,
                    (mate_y - y) / self.config.grid_size,
                    target[0] / self.config.grid_size,
                    target[1] / self.config.grid_size,
                    float((x, y) == target),
                    float((not self.state.carrying) and self._at_pickup()),
                    float(self.state.carrying and self._at_assembly()),
                    float(self.state.carrying),
                    self.state.current_beam_index / max(1, self.active_beam_count),
                    len(self.state.installed_beams) / max(1, self.active_beam_count),
                    self.state.step_count / max(1, self.config.max_steps),
                ]
            )
        return np.asarray(observations, dtype=np.float32)

    def get_action_masks(self) -> np.ndarray:
        beam = self._current_beam()
        masks: list[list[float]] = []
        for _index, _position in enumerate(self.state.agent_positions):
            mask = [1.0] * self.action_size
            if self.state.carrying:
                if not self._at_positions([beam.assembly_left, beam.assembly_right]):
                    mask[AssemblyAction.INSTALL] = 0.0
                mask[AssemblyAction.GRAB] = 0.0
            else:
                if not self._at_positions([beam.pickup_left, beam.pickup_right]):
                    mask[AssemblyAction.GRAB] = 0.0
                mask[AssemblyAction.INSTALL] = 0.0
            masks.append(mask)
        return np.asarray(masks, dtype=np.float32)

    def get_privileged_state(self) -> np.ndarray:
        beam = self._current_beam()
        values = [
            self.state.agent_positions[0][0] / self.config.grid_size,
            self.state.agent_positions[0][1] / self.config.grid_size,
            self.state.agent_positions[1][0] / self.config.grid_size,
            self.state.agent_positions[1][1] / self.config.grid_size,
            beam.pickup_left[0] / self.config.grid_size,
            beam.pickup_left[1] / self.config.grid_size,
            beam.pickup_right[0] / self.config.grid_size,
            beam.pickup_right[1] / self.config.grid_size,
            beam.assembly_left[0] / self.config.grid_size,
            beam.assembly_left[1] / self.config.grid_size,
            beam.assembly_right[0] / self.config.grid_size,
            beam.assembly_right[1] / self.config.grid_size,
            float(self.state.carrying),
            self.state.current_beam_index / max(1, self.active_beam_count),
            len(self.state.installed_beams) / max(1, self.active_beam_count),
            self.state.step_count / max(1, self.config.max_steps),
        ]
        return np.asarray(values, dtype=np.float32)

    def get_team_option_observation(self) -> np.ndarray:
        beam = self._current_beam()
        pickup_targets = [beam.pickup_left, beam.pickup_right]
        assembly_targets = [beam.assembly_left, beam.assembly_right]
        last_option_value = -1.0 if self._last_option is None else float(int(self._last_option) / max(1, self.option_size - 1))
        values = [
            self.state.agent_positions[0][0] / self.config.grid_size,
            self.state.agent_positions[0][1] / self.config.grid_size,
            self.state.agent_positions[1][0] / self.config.grid_size,
            self.state.agent_positions[1][1] / self.config.grid_size,
            pickup_targets[0][0] / self.config.grid_size,
            pickup_targets[0][1] / self.config.grid_size,
            pickup_targets[1][0] / self.config.grid_size,
            pickup_targets[1][1] / self.config.grid_size,
            assembly_targets[0][0] / self.config.grid_size,
            assembly_targets[0][1] / self.config.grid_size,
            assembly_targets[1][0] / self.config.grid_size,
            assembly_targets[1][1] / self.config.grid_size,
            float(self.state.carrying),
            float(self._at_pickup()),
            float(self._at_assembly()),
            self._distance_to_positions(pickup_targets) / max(1, self.config.grid_size * self.num_agents),
            self._distance_to_positions(assembly_targets) / max(1, self.config.grid_size * self.num_agents),
            float(self._should_reset_to_pickup_route()),
            float(self._should_reposition_after_install()),
            self.state.current_beam_index / max(1, self.active_beam_count),
            len(self.state.installed_beams) / max(1, self.active_beam_count),
            self.state.step_count / max(1, self.config.max_steps),
            last_option_value,
            self.recovery_option_usage["reset_to_pickup_route"] / max(1, self.active_beam_count),
            self.recovery_option_usage["reposition_after_install"] / max(1, self.active_beam_count),
        ]
        return np.asarray(values, dtype=np.float32)

    def get_team_option_mask(self) -> np.ndarray:
        mask = np.zeros(self.option_size, dtype=np.float32)
        if self.state.current_beam_index >= self.active_beam_count:
            mask[TeamOption.WAIT] = 1.0
            return mask

        pickup_targets = [self._current_beam().pickup_left, self._current_beam().pickup_right]
        assembly_targets = [self._current_beam().assembly_left, self._current_beam().assembly_right]
        at_pickup = self._at_positions(pickup_targets)
        at_assembly = self._at_positions(assembly_targets)

        mask[TeamOption.WAIT] = 1.0
        if self.state.carrying:
            if at_assembly:
                mask[TeamOption.INSTALL] = 1.0
            else:
                mask[TeamOption.GO_ASSEMBLY] = 1.0
                if self._distance_to_positions(assembly_targets) <= 2:
                    mask[TeamOption.ALIGN_FOR_TERMINAL_ACTION] = 1.0
        else:
            if self._should_reposition_after_install():
                mask[TeamOption.REPOSITION_AFTER_INSTALL] = 1.0
            if self._should_reset_to_pickup_route():
                mask[TeamOption.RESET_TO_PICKUP_ROUTE] = 1.0
            if at_pickup:
                mask[TeamOption.GRAB] = 1.0
            else:
                mask[TeamOption.GO_PICKUP] = 1.0
                if self._distance_to_positions(pickup_targets) <= 2:
                    mask[TeamOption.ALIGN_FOR_TERMINAL_ACTION] = 1.0

        if mask.sum() <= 0:
            mask[TeamOption.WAIT] = 1.0
        return mask

    def scripted_team_option(self) -> TeamOption:
        if self.state.current_beam_index >= self.active_beam_count:
            return TeamOption.WAIT
        if self.state.carrying:
            if self._at_assembly():
                return TeamOption.INSTALL
            if self._distance_to_positions([self._current_beam().assembly_left, self._current_beam().assembly_right]) <= 2:
                return TeamOption.ALIGN_FOR_TERMINAL_ACTION
            return TeamOption.GO_ASSEMBLY

        if self._should_reposition_after_install():
            return TeamOption.REPOSITION_AFTER_INSTALL
        if self._should_reset_to_pickup_route():
            return TeamOption.RESET_TO_PICKUP_ROUTE
        if self._at_pickup():
            return TeamOption.GRAB
        if self._distance_to_positions([self._current_beam().pickup_left, self._current_beam().pickup_right]) <= 2:
            return TeamOption.ALIGN_FOR_TERMINAL_ACTION
        return TeamOption.GO_PICKUP

    def execute_team_option(self, option: int | TeamOption, max_primitive_steps: int | None = None) -> OptionExecutionResult:
        team_option = TeamOption(option)
        primitive_budget = max_primitive_steps or self.config.option_max_primitive_steps
        reward = 0.0
        primitive_steps = 0
        done = False
        success = False
        pre_beam_index = self.state.current_beam_index
        option_info: dict[str, str | float | int | bool | None] = {
            "picked": False,
            "installed": False,
            "terminated_by_limit": False,
        }

        if team_option == TeamOption.WAIT:
            _, _, step_reward, done, info = self.step([AssemblyAction.STAY, AssemblyAction.STAY])
            reward += step_reward
            primitive_steps = 1
            success = True
            option_info.update(info)
            self.frame_history.append(
                self._snapshot_frame(
                    option=team_option,
                    primitive_step_index=primitive_steps,
                    option_reward=reward,
                    option_success=success,
                )
            )
        elif team_option == TeamOption.GRAB:
            _, _, step_reward, done, info = self.step([AssemblyAction.GRAB, AssemblyAction.GRAB])
            reward += step_reward
            primitive_steps = 1
            success = bool(info["picked"])
            option_info.update(info)
            self.frame_history.append(
                self._snapshot_frame(
                    option=team_option,
                    primitive_step_index=primitive_steps,
                    option_reward=reward,
                    option_success=success,
                )
            )
        elif team_option == TeamOption.INSTALL:
            _, _, step_reward, done, info = self.step([AssemblyAction.INSTALL, AssemblyAction.INSTALL])
            reward += step_reward
            primitive_steps = 1
            success = bool(info["installed"])
            option_info.update(info)
            self.frame_history.append(
                self._snapshot_frame(
                    option=team_option,
                    primitive_step_index=primitive_steps,
                    option_reward=reward,
                    option_success=success,
                )
            )
        else:
            while primitive_steps < primitive_budget and not done:
                if self._option_completed(team_option):
                    success = True
                    break
                actions = self._actions_for_option(team_option)
                _, _, step_reward, done, info = self.step(actions)
                reward += step_reward
                primitive_steps += 1
                option_info["picked"] = bool(option_info["picked"]) or bool(info["picked"])
                option_info["installed"] = bool(option_info["installed"]) or bool(info["installed"])
                self.frame_history.append(
                    self._snapshot_frame(
                        option=team_option,
                        primitive_step_index=primitive_steps,
                        option_reward=reward,
                        option_success=self._option_completed(team_option),
                    )
                )
                if self._option_completed(team_option):
                    success = True
                    break
            if primitive_steps >= primitive_budget and not success and not done:
                option_info["terminated_by_limit"] = True

        result = OptionExecutionResult(
            option=team_option,
            reward=reward,
            primitive_steps=primitive_steps,
            done=done,
            success=success,
            info=option_info,
        )
        self._record_option_result(result, pre_beam_index)
        return result

    def get_option_episode_diagnostics(self) -> dict[str, object]:
        construction = self._construction_summaries()
        return {
            "backend": self.backend_name,
            "backend_status": self.get_backend_status().model_dump(mode="json"),
            "selected_options": [result.option.name.lower() for result in self.option_history],
            "option_switch_count": self.option_switch_count,
            "option_results": [result.model_dump(mode="json") for result in self.option_history],
            "recovery_option_usage": dict(self.recovery_option_usage),
            "first_beam_completion_step": self.milestones["first_beam_completion_step"],
            "second_beam_pickup_step": self.milestones["second_beam_pickup_step"],
            "second_beam_install_step": self.milestones["second_beam_install_step"],
            "resource_inventory": construction["resource_inventory"],
            "blueprint_slots": construction["blueprint_slots"],
            "construction_metrics": {
                "structure_completion_rate": construction["structure_completion_rate"],
                "resource_delivery_accuracy": construction["resource_delivery_accuracy"],
                "energy_cost": self.state.energy_cost,
                "idle_step_count": self.state.idle_step_count,
                "wasted_step_count": self.state.wasted_step_count,
                "collision_count": self.state.collision_count,
                "obstacle_collision_count": self.state.obstacle_collision_count,
                "manipulation_failure_count": self.state.manipulation_failure_count,
                "manipulation_recovery_count": self.state.manipulation_recovery_count,
            },
            "obstacle_cells": [list(cell) for cell in self.config.obstacle_cells],
            "manipulation_attempts": dict(self.state.manipulation_attempts),
            "last_manipulation_failure": self.state.last_manipulation_failure,
            "manipulation_failure_history": list(self.manipulation_failure_history),
            "state_snapshots": [frame.model_dump(mode="json") for frame in self.frame_history],
        }

    def get_backend_status(self) -> BackendStatus:
        return BackendStatus(
            backend_name=self.backend_name,
            is_ready=self.is_ready,
            readiness_notes=list(self.readiness_notes),
        )

    def render_ascii(self) -> str:
        grid = [["." for _ in range(self.config.grid_size)] for _ in range(self.config.grid_size)]
        for x, y in self.config.obstacle_cells:
            grid[y][x] = "#"
        for beam in self._available_beams():
            for x, y in [beam.pickup_left, beam.pickup_right]:
                grid[y][x] = "P"
            for x, y in [beam.assembly_left, beam.assembly_right]:
                grid[y][x] = "A"
        for index, (x, y) in enumerate(self.state.agent_positions):
            grid[y][x] = str(index)
        return "\n".join(" ".join(row) for row in grid)

    def _initial_state(self) -> AssemblyState:
        return AssemblyState(
            agent_positions=list(self.config.agent_starts),
            current_beam_index=0,
            carrying=False,
            installed_beams=[],
            step_count=0,
            total_reward=0.0,
            collision_count=0,
            invalid_action_count=0,
            deadlock_steps=0,
            energy_cost=0.0,
            idle_step_count=0,
            wasted_step_count=0,
            obstacle_collision_count=0,
            manipulation_attempts={},
            manipulation_failure_count=0,
            manipulation_recovery_count=0,
            last_manipulation_failure=None,
        )

    def queue_manipulation_failure(self, phase: str, reason: str) -> None:
        if phase not in {"grasp", "install"}:
            raise ValueError(f"unsupported manipulation phase: {phase}")
        if self.state.current_beam_index >= self.active_beam_count:
            raise RuntimeError("cannot queue a manipulation failure after task completion")
        key = self._manipulation_key(self._current_beam().name, phase)
        self._queued_manipulation_failures.setdefault(key, []).append(reason)

    def _attempt_manipulation(self, beam_name: str, phase: str) -> str | None:
        key = self._manipulation_key(beam_name, phase)
        attempt = self.state.manipulation_attempts.get(key, 0) + 1
        self.state.manipulation_attempts[key] = attempt

        failure_reason = None
        queued = self._queued_manipulation_failures.get(key, [])
        if queued:
            failure_reason = queued.pop(0)
        else:
            rule = next(
                (
                    item
                    for item in self.config.manipulation_failures
                    if item.beam_name == beam_name and item.phase == phase
                ),
                None,
            )
            if rule is not None and attempt <= rule.fail_first_attempts:
                failure_reason = rule.reason

        if failure_reason is not None:
            self.state.manipulation_failure_count += 1
            self.state.last_manipulation_failure = (
                f"{beam_name} {phase} attempt {attempt}: {failure_reason}"
            )
            self._failed_manipulations.add(key)
            self.manipulation_failure_history.append(
                {
                    "beam_name": beam_name,
                    "phase": phase,
                    "attempt": attempt,
                    "success": False,
                    "reason": failure_reason,
                }
            )
            return failure_reason

        recovered = key in self._failed_manipulations and key not in self._recovered_manipulations
        if recovered:
            self.state.manipulation_recovery_count += 1
            self._recovered_manipulations.add(key)
        self.state.last_manipulation_failure = None
        self.manipulation_failure_history.append(
            {
                "beam_name": beam_name,
                "phase": phase,
                "attempt": attempt,
                "success": True,
                "reason": "recovered" if recovered else "completed",
            }
        )
        return None

    @staticmethod
    def _manipulation_key(beam_name: str, phase: str) -> str:
        return f"{beam_name}:{phase}"

    def _step_independent(self, actions: list[int]) -> float:
        reward = 0.0
        proposed = [self._apply_motion(position, AssemblyAction(action)) for position, action in zip(self.state.agent_positions, actions)]
        for index, position in enumerate(proposed):
            if position in self.config.obstacle_cells and position != self.state.agent_positions[index]:
                proposed[index] = self.state.agent_positions[index]
                self.state.collision_count += 1
                self.state.obstacle_collision_count += 1
                reward -= self.config.collision_penalty
        positions_swapped = (
            proposed[0] == self.state.agent_positions[1]
            and proposed[1] == self.state.agent_positions[0]
        )
        if proposed[0] == proposed[1] or positions_swapped:
            self.state.collision_count += 1
            reward -= self.config.collision_penalty
            return reward
        moved = False
        next_positions = list(self.state.agent_positions)
        for index, action in enumerate(actions):
            if action in {AssemblyAction.GRAB, AssemblyAction.INSTALL}:
                continue
            if proposed[index] != self.state.agent_positions[index]:
                moved = True
            next_positions[index] = proposed[index]
        self.state.agent_positions = next_positions
        terminal_actions = {AssemblyAction.GRAB, AssemblyAction.INSTALL}
        if not moved and any(AssemblyAction(action) not in terminal_actions for action in actions):
            self.state.deadlock_steps += 1
        return reward

    def _step_carrying(self, actions: list[int]) -> float:
        motions = {AssemblyAction(action) for action in actions}
        if len(motions) != 1:
            self.state.deadlock_steps += 1
            return -self.config.invalid_action_penalty
        motion = AssemblyAction(actions[0])
        if motion in {AssemblyAction.GRAB, AssemblyAction.INSTALL}:
            return 0.0
        proposed = [self._apply_motion(position, motion) for position in self.state.agent_positions]
        if any(position in self.config.obstacle_cells for position in proposed):
            self.state.collision_count += 1
            self.state.obstacle_collision_count += 1
            return -self.config.collision_penalty
        if proposed[0] == proposed[1]:
            self.state.collision_count += 1
            return -self.config.collision_penalty
        original_delta = (
            self.state.agent_positions[1][0] - self.state.agent_positions[0][0],
            self.state.agent_positions[1][1] - self.state.agent_positions[0][1],
        )
        new_delta = (proposed[1][0] - proposed[0][0], proposed[1][1] - proposed[0][1])
        if new_delta != original_delta:
            self.state.invalid_action_count += 1
            return -self.config.invalid_action_penalty
        self.state.agent_positions = proposed
        return 0.0

    def _apply_motion(self, position: tuple[int, int], action: AssemblyAction) -> tuple[int, int]:
        x, y = position
        if action == AssemblyAction.UP:
            y = max(0, y - 1)
        elif action == AssemblyAction.DOWN:
            y = min(self.config.grid_size - 1, y + 1)
        elif action == AssemblyAction.LEFT:
            x = max(0, x - 1)
        elif action == AssemblyAction.RIGHT:
            x = min(self.config.grid_size - 1, x + 1)
        return (x, y)

    def _distance_to_objective(self) -> float:
        targets = [self._current_beam().assembly_left, self._current_beam().assembly_right] if self.state.carrying else [self._current_beam().pickup_left, self._current_beam().pickup_right]
        return self._distance_to_positions(targets)

    def _second_beam_pickup_distance(self) -> float:
        if self.active_beam_count < 2 or self.state.current_beam_index != 1:
            return 0.0
        beam = self._current_beam()
        return self._distance_to_positions([beam.pickup_left, beam.pickup_right])

    def _second_beam_install_distance(self) -> float:
        if self.active_beam_count < 2 or self.state.current_beam_index != 1 or not self.state.carrying:
            return 0.0
        beam = self._current_beam()
        return self._distance_to_positions([beam.assembly_left, beam.assembly_right])

    def _phase_two_bonus(self, before_second_pickup: float, before_second_install: float) -> float:
        if self.active_beam_count < 2 or self.state.current_beam_index != 1:
            return 0.0
        bonus = 0.0
        if not self.state.carrying:
            after_pickup = self._second_beam_pickup_distance()
            bonus += self.config.second_beam_pickup_bonus * (before_second_pickup - after_pickup)
        else:
            after_install = self._second_beam_install_distance()
            bonus += self.config.second_beam_install_bonus * (before_second_install - after_install)
        return bonus

    def _current_beam(self) -> BeamTask:
        available_beams = self._available_beams()
        return available_beams[min(self.state.current_beam_index, len(available_beams) - 1)]

    def _available_beams(self) -> list[BeamTask]:
        if self.active_stage_index is not None and self.config.curriculum_stage_beams:
            return self.config.curriculum_stage_beams[self.active_stage_index]
        return self.config.beams[: self.active_beam_count]

    def _active_construction_resources(self) -> list[ConstructionResource]:
        beams = self._available_beams()
        defaults = [ConstructionResource.from_beam(beam) for beam in beams]
        if not self.config.resources:
            return defaults
        if self.active_stage_index is None and self.active_beam_count == len(self.config.beams):
            return list(self.config.resources)

        active_beam_names = {beam.name for beam in beams}
        active_slot_ids = {f"{beam.name}_slot" for beam in beams}
        resources = [
            resource
            for resource in self.config.resources
            if resource.resource_id in active_beam_names or resource.assigned_slot_id in active_slot_ids
        ]
        return resources or defaults

    def _active_blueprint_slots(self) -> list[BlueprintSlot]:
        beams = self._available_beams()
        defaults = [BlueprintSlot.from_beam(beam) for beam in beams]
        if not self.config.blueprint_slots:
            return defaults
        if self.active_stage_index is None and self.active_beam_count == len(self.config.beams):
            return list(self.config.blueprint_slots)

        active_beam_names = {beam.name for beam in beams}
        active_slot_ids = {f"{beam.name}_slot" for beam in beams}
        slots = [
            slot
            for slot in self.config.blueprint_slots
            if slot.slot_id in active_slot_ids or slot.required_resource_id in active_beam_names
        ]
        return slots or defaults

    def _construction_summaries(self) -> dict[str, object]:
        resources = self._active_construction_resources()
        slots = self._active_blueprint_slots()
        completed_slot_ids = {slot.slot_id for slot in slots if self._blueprint_slot_completed(slot)}
        delivered_resource_ids = {
            resource.resource_id
            for resource in resources
            if resource.assigned_slot_id in completed_slot_ids or resource.resource_id in self.state.installed_beams
        }

        return {
            "resource_inventory": [
                {
                    **resource.model_dump(mode="json"),
                    "delivered": resource.resource_id in delivered_resource_ids,
                }
                for resource in resources
            ],
            "blueprint_slots": [
                {
                    **slot.model_dump(mode="json"),
                    "completed": slot.slot_id in completed_slot_ids,
                }
                for slot in slots
            ],
            "structure_completion_rate": len(completed_slot_ids) / max(1, len(slots)),
            "resource_delivery_accuracy": len(delivered_resource_ids) / max(1, len(resources)),
        }

    def _blueprint_slot_completed(self, slot: BlueprintSlot) -> bool:
        installed_beams = set(self.state.installed_beams)
        if slot.required_resource_id in installed_beams:
            return True
        slot_targets = {tuple(cell) for cell in slot.target_cells}
        for beam in self._available_beams():
            if beam.name not in installed_beams:
                continue
            if slot_targets == {beam.assembly_left, beam.assembly_right}:
                return True
            if slot.slot_id == f"{beam.name}_slot":
                return True
        return False

    def _at_pickup(self) -> bool:
        beam = self._current_beam()
        return self._at_positions([beam.pickup_left, beam.pickup_right])

    def _at_assembly(self) -> bool:
        beam = self._current_beam()
        return self._at_positions([beam.assembly_left, beam.assembly_right])

    def _at_positions(self, targets: list[tuple[int, int]]) -> bool:
        return all(position == target for position, target in zip(self.state.agent_positions, targets))

    def _distance_to_positions(self, targets: list[tuple[int, int]]) -> float:
        return float(
            sum(
                abs(position[0] - target[0]) + abs(position[1] - target[1])
                for position, target in zip(self.state.agent_positions, targets)
            )
        )

    def _should_reposition_after_install(self) -> bool:
        if self.state.carrying or self.state.current_beam_index <= 0:
            return False
        previous_beam = self._available_beams()[self.state.current_beam_index - 1]
        return self._at_positions([previous_beam.assembly_left, previous_beam.assembly_right])

    def _should_reset_to_pickup_route(self) -> bool:
        if self.state.carrying or self.state.current_beam_index <= 0:
            return False
        staging_targets = self._pickup_staging_targets()
        if self._at_positions(staging_targets):
            return False
        pickup_x = self._current_beam().pickup_left[0]
        furthest_x = max(int(position[0]) for position in self.state.agent_positions)
        return furthest_x > pickup_x + 1

    def _pickup_staging_targets(self) -> list[tuple[int, int]]:
        beam = self._current_beam()
        staging_x = min(self.config.grid_size - 1, beam.pickup_left[0] + 1)
        return [(staging_x, beam.pickup_left[1]), (staging_x, beam.pickup_right[1])]

    def _reposition_targets(self) -> list[tuple[int, int]]:
        if self.state.current_beam_index <= 0:
            return self._pickup_staging_targets()
        previous_beam = self._available_beams()[self.state.current_beam_index - 1]
        beam = self._current_beam()
        clear_x = max(beam.pickup_left[0] + 1, previous_beam.assembly_left[0] - 1)
        return [(clear_x, previous_beam.assembly_left[1]), (clear_x, previous_beam.assembly_right[1])]

    def _option_completed(self, option: TeamOption) -> bool:
        if option == TeamOption.GO_PICKUP:
            return self._at_pickup()
        if option == TeamOption.GO_ASSEMBLY:
            return self._at_assembly()
        if option == TeamOption.RESET_TO_PICKUP_ROUTE:
            return self._at_positions(self._pickup_staging_targets())
        if option == TeamOption.REPOSITION_AFTER_INSTALL:
            return self._at_positions(self._reposition_targets())
        if option == TeamOption.ALIGN_FOR_TERMINAL_ACTION:
            return self._at_assembly() if self.state.carrying else self._at_pickup()
        return False

    def _actions_for_option(self, option: TeamOption) -> list[int]:
        beam = self._current_beam()
        if option == TeamOption.GO_PICKUP:
            return self._joint_actions_towards([beam.pickup_left, beam.pickup_right], carrying=False)
        if option == TeamOption.GO_ASSEMBLY:
            return self._joint_actions_towards([beam.assembly_left, beam.assembly_right], carrying=True)
        if option == TeamOption.RESET_TO_PICKUP_ROUTE:
            return self._joint_actions_towards(self._pickup_staging_targets(), carrying=False)
        if option == TeamOption.REPOSITION_AFTER_INSTALL:
            return self._joint_actions_towards(self._reposition_targets(), carrying=False)
        if option == TeamOption.ALIGN_FOR_TERMINAL_ACTION:
            targets = [beam.assembly_left, beam.assembly_right] if self.state.carrying else [beam.pickup_left, beam.pickup_right]
            return self._joint_actions_towards(targets, carrying=self.state.carrying)
        return [int(AssemblyAction.STAY), int(AssemblyAction.STAY)]

    def _joint_actions_towards(self, targets: list[tuple[int, int]], carrying: bool) -> list[int]:
        if carrying:
            path = self._rigid_shortest_path(targets)
            if path is None or len(path) < 2:
                return [int(AssemblyAction.STAY), int(AssemblyAction.STAY)]
            action = self._motion_between(self.state.agent_positions[0], path[1])
            return [int(action), int(action)]

        at_target = [
            position == target
            for position, target in zip(self.state.agent_positions, targets)
        ]
        if at_target.count(True) == 1:
            parked_index = at_target.index(True)
            moving_index = 1 - parked_index
            parked_cell = self.state.agent_positions[parked_index]
            path = self._shortest_path(
                self.state.agent_positions[moving_index],
                targets[moving_index],
                lambda cell: self._cell_is_open(cell) and cell != parked_cell,
            )
            actions = [AssemblyAction.STAY, AssemblyAction.STAY]
            if path is not None and len(path) >= 2:
                actions[moving_index] = self._motion_between(
                    self.state.agent_positions[moving_index],
                    path[1],
                )
            return [int(action) for action in actions]

        paths = [
            self._shortest_path(position, target, self._cell_is_open)
            for position, target in zip(self.state.agent_positions, targets)
        ]
        actions = [
            AssemblyAction.STAY
            if path is None or len(path) < 2
            else self._motion_between(self.state.agent_positions[index], path[1])
            for index, path in enumerate(paths)
        ]
        return [int(action) for action in self._resolve_independent_conflict(actions, targets)]

    def _rigid_shortest_path(
        self,
        targets: list[tuple[int, int]],
    ) -> list[tuple[int, int]] | None:
        start = self.state.agent_positions[0]
        companion_offset = (
            self.state.agent_positions[1][0] - start[0],
            self.state.agent_positions[1][1] - start[1],
        )
        target_offset = (
            targets[1][0] - targets[0][0],
            targets[1][1] - targets[0][1],
        )
        if companion_offset != target_offset:
            return None

        def formation_is_open(anchor: tuple[int, int]) -> bool:
            companion = (
                anchor[0] + companion_offset[0],
                anchor[1] + companion_offset[1],
            )
            return self._cell_is_open(anchor) and self._cell_is_open(companion)

        return self._shortest_path(start, targets[0], formation_is_open)

    def _shortest_path(
        self,
        start: tuple[int, int],
        target: tuple[int, int],
        is_valid: Callable[[tuple[int, int]], bool],
    ) -> list[tuple[int, int]] | None:
        if start == target:
            return [start]
        if not is_valid(target):
            return None

        frontier = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            current = frontier.popleft()
            for neighbor in self._ordered_neighbors(current, target):
                if neighbor in previous or not is_valid(neighbor):
                    continue
                previous[neighbor] = current
                if neighbor == target:
                    path = [target]
                    parent = previous[target]
                    while parent is not None:
                        path.append(parent)
                        parent = previous[parent]
                    return list(reversed(path))
                frontier.append(neighbor)
        return None

    def _ordered_neighbors(
        self,
        position: tuple[int, int],
        target: tuple[int, int],
    ) -> list[tuple[int, int]]:
        preferred: list[AssemblyAction] = []
        if target[0] != position[0]:
            preferred.append(AssemblyAction.RIGHT if target[0] > position[0] else AssemblyAction.LEFT)
        if target[1] != position[1]:
            preferred.append(AssemblyAction.DOWN if target[1] > position[1] else AssemblyAction.UP)
        for action in [AssemblyAction.RIGHT, AssemblyAction.LEFT, AssemblyAction.DOWN, AssemblyAction.UP]:
            if action not in preferred:
                preferred.append(action)
        return [
            neighbor
            for action in preferred
            if (neighbor := self._apply_motion(position, action)) != position
        ]

    def _cell_is_open(self, cell: tuple[int, int]) -> bool:
        return (
            0 <= cell[0] < self.config.grid_size
            and 0 <= cell[1] < self.config.grid_size
            and cell not in self.config.obstacle_cells
        )

    def _resolve_independent_conflict(
        self,
        actions: list[AssemblyAction],
        targets: list[tuple[int, int]],
    ) -> list[AssemblyAction]:
        proposed = [
            self._apply_motion(position, action)
            for position, action in zip(self.state.agent_positions, actions)
        ]
        same_destination = proposed[0] == proposed[1]
        swapped = (
            proposed[0] == self.state.agent_positions[1]
            and proposed[1] == self.state.agent_positions[0]
        )
        if not same_destination and not swapped:
            return actions

        at_target = [
            position == target
            for position, target in zip(self.state.agent_positions, targets)
        ]
        priority = [1, 0] if at_target[0] and not at_target[1] else [0, 1]
        if at_target[1] and not at_target[0]:
            priority = [0, 1]
        for index in priority:
            other_index = 1 - index
            other_position = self.state.agent_positions[other_index]

            def is_open_away_from_other(cell: tuple[int, int]) -> bool:
                return self._cell_is_open(cell) and cell != other_position

            path = self._shortest_path(
                self.state.agent_positions[index],
                targets[index],
                is_open_away_from_other,
            )
            if path is None or len(path) < 2:
                continue
            replanned = self._motion_between(self.state.agent_positions[index], path[1])
            result = [AssemblyAction.STAY, AssemblyAction.STAY]
            result[index] = replanned
            return result
        return [AssemblyAction.STAY, AssemblyAction.STAY]

    def _motion_between(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> AssemblyAction:
        delta = (end[0] - start[0], end[1] - start[1])
        actions = {
            (0, -1): AssemblyAction.UP,
            (0, 1): AssemblyAction.DOWN,
            (-1, 0): AssemblyAction.LEFT,
            (1, 0): AssemblyAction.RIGHT,
            (0, 0): AssemblyAction.STAY,
        }
        if delta not in actions:
            raise ValueError(f"path contains non-adjacent cells: {start} -> {end}")
        return actions[delta]
    def _motion_towards(self, position: tuple[int, int], target: tuple[int, int]) -> AssemblyAction:
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if dx != 0:
            return AssemblyAction.RIGHT if dx > 0 else AssemblyAction.LEFT
        if dy != 0:
            return AssemblyAction.DOWN if dy > 0 else AssemblyAction.UP
        return AssemblyAction.STAY

    def _distance_from_positions(self, positions: list[tuple[int, int]], targets: list[tuple[int, int]]) -> float:
        return float(
            sum(abs(position[0] - target[0]) + abs(position[1] - target[1]) for position, target in zip(positions, targets))
        )

    def _record_option_result(self, result: OptionExecutionResult, pre_beam_index: int) -> None:
        if self._last_option is not None and self._last_option != result.option:
            self.option_switch_count += 1
        self._last_option = result.option
        if result.option == TeamOption.RESET_TO_PICKUP_ROUTE:
            self.recovery_option_usage["reset_to_pickup_route"] += 1
        if result.option == TeamOption.REPOSITION_AFTER_INSTALL:
            self.recovery_option_usage["reposition_after_install"] += 1
        if result.info.get("installed") and pre_beam_index == 0 and self.milestones["first_beam_completion_step"] is None:
            self.milestones["first_beam_completion_step"] = self.state.step_count
        if result.info.get("picked") and pre_beam_index == 1 and self.milestones["second_beam_pickup_step"] is None:
            self.milestones["second_beam_pickup_step"] = self.state.step_count
        if result.info.get("installed") and pre_beam_index == 1 and self.milestones["second_beam_install_step"] is None:
            self.milestones["second_beam_install_step"] = self.state.step_count
        self.option_history.append(result)

    def _snapshot_frame(
        self,
        option: TeamOption | None = None,
        primitive_step_index: int = 0,
        option_reward: float = 0.0,
        option_success: bool | None = None,
    ) -> AssemblyPlaybackFrame:
        beam = self._available_beams()[min(self.state.current_beam_index, len(self._available_beams()) - 1)]
        return AssemblyPlaybackFrame(
            step_count=self.state.step_count,
            current_beam_index=self.state.current_beam_index,
            current_beam_name=beam.name if self.state.current_beam_index < self.active_beam_count else None,
            carrying=self.state.carrying,
            agent_positions=list(self.state.agent_positions),
            pickup_targets=[beam.pickup_left, beam.pickup_right],
            assembly_targets=[beam.assembly_left, beam.assembly_right],
            selected_option=None if option is None else option.name.lower(),
            primitive_step_index=primitive_step_index,
            option_reward=option_reward,
            option_success=option_success,
            completed_beams=list(self.state.installed_beams),
            completed_component_ids=[
                resource.component_id or resource.resource_id
                for resource in self._active_construction_resources()
                if resource.resource_id in self.state.installed_beams
            ],
        )


def _metric_float(value: object, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"construction summary '{name}' must be numeric")
    return float(value)
