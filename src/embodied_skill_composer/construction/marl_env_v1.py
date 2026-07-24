from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import hypot, inf

import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, MultiBinary
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from embodied_skill_composer.construction.intelligence_models import (
    ConstructionFailure,
    DecisionCandidate,
    FailureKind,
    ScenarioManifest,
    SkillDistribution,
    SkillProfile,
    SwarmBrainEvent,
)
from embodied_skill_composer.construction.models import BuildModule, BuildPlan, Vec2
from embodied_skill_composer.construction.routing import (
    RoutePlan,
    RoutingAdapter,
    RoutingError,
    create_routing_adapter,
)


MAX_MODULES = 32
FLEET_SIZE = 4
MODULE_FEATURES = 14
ROBOT_FEATURES = 11


@dataclass
class RobotRuntime:
    position: Vec2
    battery_remaining_wh: float
    status: str = "idle"
    active_module_id: str | None = None
    available_at_s: float = 0.0


@dataclass
class ActiveJob:
    module_id: str
    robot_ids: tuple[str, ...]
    start_s: float
    end_s: float
    travel_distance_m: float
    energy_wh: float
    routes: RoutePlan
    carry_route: list[Vec2]
    skill_success: bool


class TemporalConstructionCoordinationEnv(ParallelEnv):
    """Event-driven four-robot construction allocation environment.

    Actions are WAIT (0) or a bid for one of 32 padded module slots. The
    environment advances to the next completion or disruption event, so work
    consumes simulated time and robots remain busy across decision epochs.
    """

    metadata = {
        "name": "construction_coordination_v1",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        scenario: ScenarioManifest | BuildPlan,
        *,
        skill_profile: SkillProfile | None = None,
        router: RoutingAdapter | None = None,
        max_decisions: int = 256,
        max_sim_time_s: float = 1800.0,
    ) -> None:
        self.scenario: ScenarioManifest | None
        if isinstance(scenario, ScenarioManifest):
            scenario_manifest = scenario.model_copy(deep=True)
            self.scenario = scenario_manifest
            self.plan = scenario_manifest.plan
            self.failures = list(scenario_manifest.failures)
            route_seed = scenario_manifest.seed
        else:
            self.scenario = None
            self.plan = scenario.model_copy(deep=True)
            self.failures = []
            route_seed = 0
        if len(self.plan.robots) != FLEET_SIZE:
            raise ValueError(f"construction_coordination_v1 requires {FLEET_SIZE} robots")
        if len(self.plan.modules) > MAX_MODULES:
            raise ValueError(f"construction_coordination_v1 supports at most {MAX_MODULES} modules")
        self.max_decisions = max_decisions
        self.max_sim_time_s = max_sim_time_s
        self.skill_profile = skill_profile or default_skill_profile()
        self.router = router or create_routing_adapter(seed=route_seed)
        self.possible_agents = [robot.robot_id for robot in self.plan.robots]
        self.agent_name_mapping = {agent: index for index, agent in enumerate(self.possible_agents)}
        self.modules = list(self.plan.modules)
        self.module_index = {module.module_id: index for index, module in enumerate(self.modules)}
        self.module_by_id = {module.module_id: module for module in self.modules}
        self.robots = {robot.robot_id: robot for robot in self.plan.robots}
        self.np_random, self.np_random_seed = seeding.np_random(None)
        self.agents: list[str] = []
        self.robot_runtime: dict[str, RobotRuntime] = {}
        self.completed: set[str] = set()
        self.active_jobs: dict[str, ActiveJob] = {}
        self.decision_count = 0
        self.sim_time_s = 0.0
        self.assignment_history: list[dict[str, object]] = []
        self.brain_events: list[SwarmBrainEvent] = []
        self.event_log: list[dict[str, object]] = []
        self.pending_failures: list[ConstructionFailure] = []
        self.total_travel_m = 0.0
        self.total_energy_wh = 0.0
        self.idle_robot_seconds = 0.0
        self.busy_robot_seconds = {agent: 0.0 for agent in self.possible_agents}
        self.collision_count = 0
        self.wasted_work_s = 0.0
        self.invalid_bid_count = 0
        self.drop_count = 0

    @lru_cache(maxsize=None)
    def observation_space(self, _agent: str):
        return DictSpace(
            {
                "self": Box(-1.0, 1.0, shape=(ROBOT_FEATURES + 1,), dtype=np.float32),
                "robots": Box(
                    -1.0,
                    1.0,
                    shape=(FLEET_SIZE, ROBOT_FEATURES),
                    dtype=np.float32,
                ),
                "modules": Box(
                    -1.0,
                    1.0,
                    shape=(MAX_MODULES, MODULE_FEATURES),
                    dtype=np.float32,
                ),
                "dependencies": MultiBinary((MAX_MODULES, MAX_MODULES)),
                "action_mask": MultiBinary(MAX_MODULES + 1),
            }
        )

    @lru_cache(maxsize=None)
    def action_space(self, _agent: str):
        return Discrete(MAX_MODULES + 1)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        self.robot_runtime = {
            robot.robot_id: RobotRuntime(
                position=Vec2(x=robot.start_pose.position.x, y=robot.start_pose.position.y),
                battery_remaining_wh=robot.battery_capacity_wh,
            )
            for robot in self.plan.robots
        }
        self.completed = set()
        self.active_jobs = {}
        self.decision_count = 0
        self.sim_time_s = 0.0
        self.assignment_history = []
        self.brain_events = []
        self.event_log = []
        self.pending_failures = sorted(
            (failure.model_copy(deep=True) for failure in self.failures),
            key=lambda item: (item.trigger_time_s, item.failure_id),
        )
        self.total_travel_m = 0.0
        self.total_energy_wh = 0.0
        self.idle_robot_seconds = 0.0
        self.busy_robot_seconds = {agent: 0.0 for agent in self.possible_agents}
        self.collision_count = 0
        self.wasted_work_s = 0.0
        self.invalid_bid_count = 0
        self.drop_count = 0
        return self._observations(), self._infos()

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        acting_agents = list(self.agents)
        self.decision_count += 1
        completed_before = len(self.completed)
        travel_before = self.total_travel_m
        invalid_before = self.invalid_bid_count
        dropped_before = self.drop_count
        start_time = self.sim_time_s

        assignments = self._allocate(actions)
        self._advance_to_next_event()
        elapsed = self.sim_time_s - start_time
        completion_gain = len(self.completed) - completed_before
        team_reward = (
            4.0 * completion_gain / len(self.modules)
            - 0.15 * elapsed / self.max_sim_time_s
            - 0.002 * (self.total_travel_m - travel_before)
            - 0.04 * (self.invalid_bid_count - invalid_before)
            - 0.08 * (self.drop_count - dropped_before)
        )
        done = len(self.completed) == len(self.modules)
        if done:
            team_reward += 2.0
        truncated = (
            self.decision_count >= self.max_decisions or self.sim_time_s >= self.max_sim_time_s
        ) and not done
        rewards = {agent: float(team_reward) for agent in acting_agents}
        terminations = {agent: done for agent in acting_agents}
        truncations = {agent: truncated for agent in acting_agents}
        observations = self._observations()
        infos = self._infos(assignments=assignments)
        if done or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def ready_modules(self) -> list[BuildModule]:
        return [
            module
            for module in self.modules
            if module.module_id not in self.completed
            and module.module_id not in self.active_jobs
            and set(module.dependencies) <= self.completed
        ]

    def action_mask(self, agent: str) -> np.ndarray:
        mask = np.zeros(MAX_MODULES + 1, dtype=np.int8)
        mask[0] = 1
        runtime = self.robot_runtime.get(agent)
        if runtime is None or runtime.status != "idle":
            return mask
        robot = self.robots[agent]
        for module in self.ready_modules():
            if module.required_team_size == 1 and robot.payload_capacity_kg < module.mass_kg:
                continue
            mask[self.module_index[module.module_id] + 1] = 1
        return mask

    def state(self) -> np.ndarray:
        modules = self._module_features().reshape(-1)
        dependencies = self._dependency_matrix().astype(np.float32).reshape(-1)
        robots = self._robot_features().reshape(-1)
        clock = np.array(
            [
                self.sim_time_s / self.max_sim_time_s,
                self.decision_count / self.max_decisions,
            ],
            dtype=np.float32,
        )
        return np.concatenate([modules, dependencies, robots, clock])

    def metrics(self) -> dict[str, object]:
        elapsed = max(self.sim_time_s, 1e-9)
        return {
            "environment": str(self.metadata["name"]),
            "structure_completion_rate": len(self.completed) / len(self.modules),
            "completed_modules": len(self.completed),
            "total_modules": len(self.modules),
            "makespan_s": self.sim_time_s,
            "total_travel_m": self.total_travel_m,
            "total_energy_wh": self.total_energy_wh,
            "idle_robot_seconds": self.idle_robot_seconds,
            "robot_utilization": {
                agent: min(self.busy_robot_seconds[agent] / elapsed, 1.0)
                for agent in self.possible_agents
            },
            "collision_count": self.collision_count,
            "wasted_work_s": self.wasted_work_s,
            "invalid_bid_count": self.invalid_bid_count,
            "drop_count": self.drop_count,
            "routing_backend": self._last_routing_backend(),
        }

    def annotate_latest_decisions(
        self,
        controller: str,
        diagnostics: dict[str, dict[str, object]],
    ) -> None:
        by_robot = {
            event.robot_id: event
            for event in self.brain_events
            if event.decision_index == self.decision_count
        }
        for robot_id, values in diagnostics.items():
            event = by_robot.get(robot_id)
            if event is None:
                continue
            event.controller = controller
            event.action_probability = _diagnostic_float(
                values["selected_probability"],
                field="selected_probability",
            )
            event.uncertainty = _diagnostic_float(
                values["uncertainty"],
                field="uncertainty",
            )
            probabilities = values["action_probabilities"]
            if not isinstance(probabilities, (list, tuple)):
                raise TypeError("action_probabilities must be a sequence")
            for candidate in event.candidates:
                action = self.module_index[candidate.module_id] + 1
                candidate.probability = _diagnostic_float(
                    probabilities[action],
                    field=f"action_probabilities[{action}]",
                )

    def _allocate(self, actions: dict[str, int]) -> list[dict[str, object]]:
        ready = self.ready_modules()
        ready_ids = {module.module_id for module in ready}
        bids: dict[str, list[str]] = defaultdict(list)
        for agent in self.agents:
            action = int(actions.get(agent, 0))
            mask = self.action_mask(agent)
            if not 0 <= action <= MAX_MODULES or not mask[action]:
                self.invalid_bid_count += 1
                continue
            self._record_decision(agent, action, ready)
            if action == 0:
                continue
            module_index = action - 1
            if module_index >= len(self.modules):
                self.invalid_bid_count += 1
                continue
            module = self.modules[module_index]
            if module.module_id in ready_ids:
                bids[module.module_id].append(agent)

        used_agents: set[str] = set()
        selected: list[tuple[BuildModule, list[str]]] = []
        for module in sorted(ready, key=lambda item: (-item.required_team_size, item.module_id)):
            candidates = [agent for agent in bids[module.module_id] if agent not in used_agents]
            team = self._select_capable_team(module, candidates)
            if team is None:
                continue
            selected.append((module, team))
            used_agents.update(team)
        if not selected:
            return []

        starts: dict[str, Vec2] = {}
        goals: dict[str, Vec2] = {}
        for module, team in selected:
            for team_index, agent in enumerate(team):
                starts[agent] = self.robot_runtime[agent].position
                goals[agent] = _formation_point(
                    module.staging_pose.position,
                    team_index,
                    len(team),
                )
        try:
            approach_routes = self.router.route_many(self.plan.site_grid, starts, goals)
        except RoutingError as exc:
            self.invalid_bid_count += len(selected)
            self.event_log.append(
                {"timestamp_s": self.sim_time_s, "event": "routing_blocked", "reason": str(exc)}
            )
            return []

        self.collision_count += approach_routes.conflict_count
        assignments: list[dict[str, object]] = []
        for module, team in selected:
            carrier_id = f"carrier::{module.module_id}"
            try:
                carry_route = self.router.route_many(
                    self.plan.site_grid,
                    {carrier_id: Vec2.model_validate(module.staging_pose.position)},
                    {carrier_id: Vec2.model_validate(module.target_pose.position)},
                )
            except RoutingError as exc:
                self.invalid_bid_count += 1
                self.event_log.append(
                    {
                        "timestamp_s": self.sim_time_s,
                        "event": "payload_route_blocked",
                        "module_id": module.module_id,
                        "reason": str(exc),
                    }
                )
                continue
            approach_distance = sum(
                _path_distance(approach_routes.world_paths[agent]) for agent in team
            )
            carry_distance = _path_distance(carry_route.world_paths[carrier_id])
            travel_distance = approach_distance + carry_distance * len(team)
            approach_duration = max(
                _path_distance(approach_routes.world_paths[agent]) / self.robots[agent].speed_mps
                for agent in team
            )
            carry_duration = carry_distance / min(self.robots[agent].speed_mps for agent in team)
            distribution = self._skill_distribution(module)
            install_duration = max(
                1.0,
                float(
                    self.np_random.normal(
                        distribution.duration_mean_s,
                        distribution.duration_std_s,
                    )
                ),
            )
            duration = 3.0 + approach_duration + carry_duration + install_duration
            energy = travel_distance * 0.35 + install_duration * len(team) * 0.03
            job = ActiveJob(
                module_id=module.module_id,
                robot_ids=tuple(team),
                start_s=self.sim_time_s,
                end_s=self.sim_time_s + duration,
                travel_distance_m=travel_distance,
                energy_wh=energy,
                routes=approach_routes,
                carry_route=carry_route.world_paths[carrier_id],
                skill_success=bool(self.np_random.random() <= distribution.success_rate),
            )
            self.active_jobs[module.module_id] = job
            self.total_travel_m += travel_distance
            self.total_energy_wh += energy
            for agent in team:
                runtime = self.robot_runtime[agent]
                runtime.status = "busy"
                runtime.active_module_id = module.module_id
            assignment = {
                "module_id": module.module_id,
                "robot_ids": list(team),
                "start_s": self.sim_time_s,
                "planned_end_s": job.end_s,
                "travel_distance_m": travel_distance,
                "routing_backend": approach_routes.backend,
                "approach_routes": {
                    agent: [
                        point.model_dump(mode="json")
                        for point in approach_routes.world_paths[agent]
                    ]
                    for agent in team
                },
                "carry_route": [
                    point.model_dump(mode="json") for point in carry_route.world_paths[carrier_id]
                ],
            }
            assignments.append(assignment)
            self.assignment_history.append(assignment)
            self.event_log.append(
                {"timestamp_s": self.sim_time_s, "event": "assignment", **assignment}
            )
        return assignments

    def _advance_to_next_event(self) -> None:
        event_times = [job.end_s for job in self.active_jobs.values()]
        event_times.extend(
            failure.trigger_time_s
            for failure in self.pending_failures
            if failure.trigger_time_s >= self.sim_time_s
        )
        event_times.extend(
            runtime.available_at_s
            for runtime in self.robot_runtime.values()
            if runtime.status == "unavailable" and runtime.available_at_s > self.sim_time_s
        )
        next_time = min(event_times, default=self.sim_time_s + 1.0)
        next_time = min(next_time, self.max_sim_time_s)
        elapsed = max(next_time - self.sim_time_s, 0.0)
        for agent, runtime in self.robot_runtime.items():
            if runtime.status == "busy":
                self.busy_robot_seconds[agent] += elapsed
            elif runtime.status == "idle":
                self.idle_robot_seconds += elapsed
        self.sim_time_s = next_time

        due_failures = [
            failure
            for failure in self.pending_failures
            if failure.trigger_time_s <= self.sim_time_s + 1e-9
        ]
        for failure in due_failures:
            self._apply_failure(failure)
            self.pending_failures.remove(failure)

        for agent, runtime in self.robot_runtime.items():
            if runtime.status == "unavailable" and runtime.available_at_s <= self.sim_time_s:
                runtime.status = "idle"
                runtime.available_at_s = 0.0
                self.event_log.append(
                    {
                        "timestamp_s": self.sim_time_s,
                        "event": "robot_recovered",
                        "robot_id": agent,
                    }
                )

        completed_jobs = [
            job for job in self.active_jobs.values() if job.end_s <= self.sim_time_s + 1e-9
        ]
        for job in completed_jobs:
            if job.skill_success:
                self._complete_job(job)
            else:
                self.drop_count += 1
                self._cancel_job(job, reason="skill_failure")

    def _complete_job(self, job: ActiveJob) -> None:
        module = self.module_by_id[job.module_id]
        self.completed.add(job.module_id)
        self.active_jobs.pop(job.module_id, None)
        for team_index, agent in enumerate(job.robot_ids):
            runtime = self.robot_runtime[agent]
            runtime.position = _formation_point(
                module.target_pose.position,
                team_index,
                len(job.robot_ids),
            )
            runtime.battery_remaining_wh = max(
                0.0,
                runtime.battery_remaining_wh - job.energy_wh / len(job.robot_ids),
            )
            runtime.active_module_id = None
            runtime.status = "idle" if runtime.battery_remaining_wh > 0 else "unavailable"
            runtime.available_at_s = 0.0 if runtime.status == "idle" else inf
        self.event_log.append(
            {
                "timestamp_s": self.sim_time_s,
                "event": "completion",
                "module_id": job.module_id,
                "robot_ids": list(job.robot_ids),
            }
        )

    def _cancel_job(self, job: ActiveJob, *, reason: str) -> None:
        self.active_jobs.pop(job.module_id, None)
        self.wasted_work_s += max(self.sim_time_s - job.start_s, 0.0)
        for agent in job.robot_ids:
            runtime = self.robot_runtime[agent]
            runtime.active_module_id = None
            if runtime.status != "unavailable":
                runtime.status = "idle"
        self.event_log.append(
            {
                "timestamp_s": self.sim_time_s,
                "event": "job_cancelled",
                "module_id": job.module_id,
                "robot_ids": list(job.robot_ids),
                "reason": reason,
            }
        )

    def _apply_failure(self, failure: ConstructionFailure) -> None:
        if failure.kind == FailureKind.ROBOT_UNAVAILABLE:
            assert failure.robot_id is not None
            runtime = self.robot_runtime[failure.robot_id]
            if runtime.active_module_id:
                self._cancel_job(
                    self.active_jobs[runtime.active_module_id],
                    reason=failure.failure_id,
                )
            runtime.status = "unavailable"
            runtime.available_at_s = self.sim_time_s + failure.duration_s
        elif failure.kind == FailureKind.OBSTACLE:
            assert failure.obstacle_cell is not None
            if failure.obstacle_cell not in self.plan.site_grid.obstacle_cells:
                self.plan.site_grid.obstacle_cells.append(failure.obstacle_cell)
            affected = [
                job
                for job in self.active_jobs.values()
                if any(failure.obstacle_cell in path for path in job.routes.cell_paths.values())
            ]
            for job in affected:
                self._cancel_job(job, reason=failure.failure_id)
        else:
            failed_job = self.active_jobs.get(failure.module_id or "")
            if failed_job is None and self.active_jobs:
                failed_job = sorted(
                    self.active_jobs.values(),
                    key=lambda item: item.module_id,
                )[0]
            if failed_job is not None:
                self.drop_count += 1
                self._cancel_job(failed_job, reason=failure.failure_id)
        self.event_log.append(
            {
                "timestamp_s": self.sim_time_s,
                "event": "failure",
                "failure": failure.model_dump(mode="json"),
            }
        )

    def _select_capable_team(
        self,
        module: BuildModule,
        candidates: list[str],
    ) -> list[str] | None:
        eligible = [agent for agent in candidates if self.robot_runtime[agent].status == "idle"]
        for team in combinations(sorted(eligible), module.required_team_size):
            if sum(self.robots[agent].payload_capacity_kg for agent in team) >= module.mass_kg:
                return list(team)
        return None

    def _skill_distribution(self, module: BuildModule) -> SkillDistribution:
        return self.skill_profile.by_module_type.get(
            module.module_type.value,
            SkillDistribution(
                success_rate=1.0,
                duration_mean_s=module.install_duration_s,
            ),
        )

    def _record_decision(
        self,
        agent: str,
        action: int,
        ready: list[BuildModule],
    ) -> None:
        runtime = self.robot_runtime[agent]
        candidates = []
        for module in ready:
            distance = hypot(
                runtime.position.x - module.staging_pose.position.x,
                runtime.position.y - module.staging_pose.position.y,
            )
            selected = action == self.module_index[module.module_id] + 1
            candidates.append(
                DecisionCandidate(
                    module_id=module.module_id,
                    score=-distance,
                    selected=selected,
                    rejection_reason=None if selected else "lower controller preference",
                )
            )
        remaining_duration = (
            sum(
                module.install_duration_s
                for module in self.modules
                if module.module_id not in self.completed
            )
            / FLEET_SIZE
        )
        selected_id = None if action == 0 else self.modules[action - 1].module_id
        self.brain_events.append(
            SwarmBrainEvent(
                timestamp_s=self.sim_time_s,
                decision_index=self.decision_count,
                controller="external_bid_policy",
                robot_id=agent,
                selected_module_id=selected_id,
                candidates=candidates,
                predicted_remaining_s=remaining_duration,
                reason="wait" if action == 0 else "valid bid",
            )
        )

    def _observations(self) -> dict[str, dict[str, np.ndarray]]:
        module_features = self._module_features()
        dependencies = self._dependency_matrix()
        robot_features = self._robot_features()
        result = {}
        for agent in self.agents:
            index = self.agent_name_mapping[agent]
            self_features = np.concatenate(
                [robot_features[index], np.array([index / (FLEET_SIZE - 1)], dtype=np.float32)]
            )
            result[agent] = {
                "self": self_features.astype(np.float32),
                "robots": robot_features.copy(),
                "modules": module_features.copy(),
                "dependencies": dependencies.copy(),
                "action_mask": self.action_mask(agent),
            }
        return result

    def _module_features(self) -> np.ndarray:
        features = np.zeros((MAX_MODULES, MODULE_FEATURES), dtype=np.float32)
        ready_ids = {module.module_id for module in self.ready_modules()}
        module_types = list(type(self.modules[0].module_type))
        for index, module in enumerate(self.modules):
            completed_dependencies = sum(dep in self.completed for dep in module.dependencies)
            dependency_fraction = completed_dependencies / max(len(module.dependencies), 1)
            features[index] = np.array(
                [
                    1.0,
                    float(module.module_id in self.completed),
                    float(module.module_id in ready_ids),
                    float(module.module_id in self.active_jobs),
                    module.required_team_size / 2.0,
                    min(module.mass_kg / 100.0, 1.0),
                    min(module.install_duration_s / 30.0, 1.0),
                    self._normalize_x(module.staging_pose.position.x),
                    self._normalize_y(module.staging_pose.position.y),
                    self._normalize_x(module.target_pose.position.x),
                    self._normalize_y(module.target_pose.position.y),
                    len(module.dependencies) / MAX_MODULES,
                    dependency_fraction,
                    module_types.index(module.module_type) / max(len(module_types) - 1, 1),
                ],
                dtype=np.float32,
            )
        return features

    def _dependency_matrix(self) -> np.ndarray:
        matrix = np.zeros((MAX_MODULES, MAX_MODULES), dtype=np.int8)
        for module in self.modules:
            row = self.module_index[module.module_id]
            for dependency in module.dependencies:
                matrix[row, self.module_index[dependency]] = 1
        return matrix

    def _robot_features(self) -> np.ndarray:
        features = np.zeros((FLEET_SIZE, ROBOT_FEATURES), dtype=np.float32)
        for index, robot in enumerate(self.plan.robots):
            runtime = self.robot_runtime[robot.robot_id]
            features[index] = np.array(
                [
                    min(robot.payload_capacity_kg / 100.0, 1.0),
                    min(robot.speed_mps / 2.0, 1.0),
                    runtime.battery_remaining_wh / robot.battery_capacity_wh,
                    self._normalize_x(runtime.position.x),
                    self._normalize_y(runtime.position.y),
                    float(runtime.status == "idle"),
                    float(runtime.status == "busy"),
                    float(runtime.status == "unavailable"),
                    float(robot.role == "heavy"),
                    float(robot.role == "precision"),
                    min(runtime.available_at_s / self.max_sim_time_s, 1.0)
                    if runtime.status == "unavailable"
                    else 0.0,
                ],
                dtype=np.float32,
            )
        return features

    def _infos(
        self,
        *,
        assignments: list[dict[str, object]] | None = None,
    ) -> dict[str, dict[str, object]]:
        metrics = self.metrics()
        return {
            agent: {
                "action_mask": self.action_mask(agent),
                "sim_time_s": self.sim_time_s,
                "completed_modules": len(self.completed),
                "total_modules": len(self.modules),
                "assignments": assignments or [],
                "active_jobs": sorted(self.active_jobs),
                "metrics": metrics,
            }
            for agent in self.agents
        }

    def _normalize_x(self, value: float) -> float:
        width_m = (self.plan.site_grid.width - 1) * self.plan.site_grid.resolution_m
        return float(np.clip(2 * (value - self.plan.site_grid.origin.x) / width_m - 1, -1, 1))

    def _normalize_y(self, value: float) -> float:
        height_m = (self.plan.site_grid.height - 1) * self.plan.site_grid.resolution_m
        return float(np.clip(2 * (value - self.plan.site_grid.origin.y) / height_m - 1, -1, 1))

    def _last_routing_backend(self) -> str | None:
        if not self.assignment_history:
            return None
        return str(self.assignment_history[-1]["routing_backend"])


def default_skill_profile() -> SkillProfile:
    profiles = {
        module_type: SkillDistribution(success_rate=1.0, duration_mean_s=duration)
        for module_type, duration in {
            "foundation": 16.0,
            "wall_panel": 11.0,
            "door_panel": 11.0,
            "window_panel": 11.0,
            "interior_panel": 10.0,
            "roof_panel": 18.0,
        }.items()
    }
    return SkillProfile(
        profile_id="deterministic-fixture-v1",
        source_backend="fixture",
        by_module_type=profiles,
        notes=["Deterministic fallback used until MuJoCo campaign metrics are imported."],
    )


def _diagnostic_float(value: object, *, field: str) -> float:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise TypeError(f"{field} must be numeric")
    return float(value)


def scripted_temporal_actions(
    env: TemporalConstructionCoordinationEnv,
) -> dict[str, int]:
    actions = {agent: 0 for agent in env.agents}
    available = [agent for agent in env.agents if env.robot_runtime[agent].status == "idle"]
    for module in sorted(
        env.ready_modules(),
        key=lambda item: (-item.required_team_size, item.module_id),
    ):
        candidates = sorted(
            available,
            key=lambda agent: (-env.robots[agent].payload_capacity_kg, agent),
        )
        team = env._select_capable_team(module, candidates)
        if team is None:
            continue
        action = env.module_index[module.module_id] + 1
        for agent in team:
            actions[agent] = action
            available.remove(agent)
    return actions


def auction_temporal_actions(
    env: TemporalConstructionCoordinationEnv,
) -> dict[str, int]:
    actions = {agent: 0 for agent in env.agents}
    available = {agent for agent in env.agents if env.robot_runtime[agent].status == "idle"}
    scored_modules = []
    for module in env.ready_modules():
        scored_teams = []
        for team in combinations(sorted(available), module.required_team_size):
            if sum(env.robots[agent].payload_capacity_kg for agent in team) < module.mass_kg:
                continue
            travel = sum(
                hypot(
                    env.robot_runtime[agent].position.x - module.staging_pose.position.x,
                    env.robot_runtime[agent].position.y - module.staging_pose.position.y,
                )
                for agent in team
            )
            scored_teams.append((travel, team))
        if scored_teams:
            travel, team = min(scored_teams, key=lambda item: (item[0], item[1]))
            scored_modules.append((travel, module.module_id, team))
    for _, module_id, team in sorted(scored_modules):
        if not set(team) <= available:
            continue
        action = env.module_index[module_id] + 1
        for agent in team:
            actions[agent] = action
            available.remove(agent)
    return actions


def _formation_point(position, index: int, team_size: int) -> Vec2:
    if team_size == 1:
        return Vec2(x=position.x, y=position.y)
    offset = -0.35 if index == 0 else 0.35
    return Vec2(x=position.x, y=position.y + offset)


def _path_distance(path: list[Vec2]) -> float:
    return sum(
        hypot(right.x - left.x, right.y - left.y)
        for left, right in zip(path, path[1:], strict=False)
    )
