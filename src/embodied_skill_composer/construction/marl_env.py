from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import numpy as np
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from embodied_skill_composer.construction.models import BuildModule, BuildPlan


class ConstructionCoordinationEnv(ParallelEnv):
    """Parallel high-level team formation environment for Construction v2."""

    metadata = {
        "name": "construction_coordination_v0",
        "render_modes": [],
        "is_parallelizable": True,
    }

    def __init__(self, plan: BuildPlan, *, max_decisions: int = 128) -> None:
        self.plan = plan.model_copy(deep=True)
        self.max_decisions = max_decisions
        self.possible_agents = [robot.robot_id for robot in self.plan.robots]
        self.agent_name_mapping = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }
        self.modules = list(self.plan.modules)
        self.module_index = {
            module.module_id: index for index, module in enumerate(self.modules)
        }
        self.robots = {robot.robot_id: robot for robot in self.plan.robots}
        self.agents: list[str] = []
        self.completed: set[str] = set()
        self.decision_count = 0
        self.assignment_history: list[dict[str, object]] = []
        self.np_random, self.np_random_seed = seeding.np_random(None)

    @lru_cache(maxsize=None)
    def observation_space(self, _agent: str):
        return Box(low=0.0, high=1.0, shape=(len(self.modules) * 3 + 4,), dtype=np.float32)

    @lru_cache(maxsize=None)
    def action_space(self, _agent: str):
        return Discrete(len(self.modules) + 1)

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        self.completed = set()
        self.decision_count = 0
        self.assignment_history = []
        return self._observations(), self._infos()

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        self.decision_count += 1
        ready = self.ready_modules()
        ready_ids = {module.module_id for module in ready}
        bids: dict[str, list[str]] = defaultdict(list)
        rewards = {agent: -0.01 for agent in self.agents}
        for agent, action in actions.items():
            if action == 0:
                continue
            module_index = action - 1
            if not 0 <= module_index < len(self.modules):
                rewards[agent] -= 0.1
                continue
            module = self.modules[module_index]
            if module.module_id not in ready_ids:
                rewards[agent] -= 0.05
                continue
            bids[module.module_id].append(agent)

        used_agents: set[str] = set()
        assignments: list[dict[str, object]] = []
        for module in sorted(ready, key=lambda item: (-item.required_team_size, item.module_id)):
            candidates = [agent for agent in bids[module.module_id] if agent not in used_agents]
            team = self._select_capable_team(module, candidates)
            if team is None:
                continue
            used_agents.update(team)
            self.completed.add(module.module_id)
            for agent in team:
                rewards[agent] += 1.0 / module.required_team_size
            assignments.append({"module_id": module.module_id, "robot_ids": team})
        self.assignment_history.extend(assignments)

        done = len(self.completed) == len(self.modules)
        truncated = self.decision_count >= self.max_decisions and not done
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        observations = self._observations()
        infos = self._infos(assignments=assignments)
        if done or truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None

    def state(self) -> np.ndarray:
        completed = np.array(
            [module.module_id in self.completed for module in self.modules], dtype=np.float32
        )
        ready_ids = {module.module_id for module in self.ready_modules()}
        ready = np.array(
            [module.module_id in ready_ids for module in self.modules], dtype=np.float32
        )
        team_sizes = np.array(
            [module.required_team_size / 2 for module in self.modules], dtype=np.float32
        )
        progress = np.array(
            [len(self.completed) / len(self.modules), self.decision_count / self.max_decisions],
            dtype=np.float32,
        )
        return np.concatenate([completed, ready, team_sizes, progress])

    def ready_modules(self) -> list[BuildModule]:
        return [
            module
            for module in self.modules
            if module.module_id not in self.completed
            and set(module.dependencies) <= self.completed
        ]

    def action_mask(self, _agent: str) -> np.ndarray:
        ready_ids = {module.module_id for module in self.ready_modules()}
        return np.array(
            [1] + [int(module.module_id in ready_ids) for module in self.modules],
            dtype=np.int8,
        )

    def _observations(self) -> dict[str, np.ndarray]:
        global_state = self.state()
        observations = {}
        for agent in self.agents:
            robot = self.robots[agent]
            local = np.array(
                [
                    robot.payload_capacity_kg / 100,
                    robot.speed_mps / 2,
                    robot.battery_capacity_wh / 1000,
                    self.agent_name_mapping[agent] / max(len(self.possible_agents) - 1, 1),
                ],
                dtype=np.float32,
            )
            observations[agent] = np.concatenate([global_state[:-2], local])
        return observations

    def _infos(
        self,
        *,
        assignments: list[dict[str, object]] | None = None,
    ) -> dict[str, dict[str, object]]:
        return {
            agent: {
                "action_mask": self.action_mask(agent),
                "completed_modules": len(self.completed),
                "total_modules": len(self.modules),
                "assignments": assignments or [],
            }
            for agent in self.agents
        }

    def _select_capable_team(
        self,
        module: BuildModule,
        candidates: list[str],
    ) -> list[str] | None:
        candidates = sorted(
            candidates,
            key=lambda agent: (-self.robots[agent].payload_capacity_kg, agent),
        )
        team = candidates[: module.required_team_size]
        if len(team) != module.required_team_size:
            return None
        if sum(self.robots[agent].payload_capacity_kg for agent in team) < module.mass_kg:
            return None
        return team


def scripted_coordination_actions(env: ConstructionCoordinationEnv) -> dict[str, int]:
    actions = {agent: 0 for agent in env.agents}
    available = list(env.agents)
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
