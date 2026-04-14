from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from embodied_skill_composer.assembly.backends import AssemblyTaskBackend
from embodied_skill_composer.assembly.models import OptionPolicyMetrics, TeamOption, TrainingConfig


class OptionActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class OptionCritic(nn.Module):
    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


@dataclass
class OptionTrainingSummary:
    iterations: int
    warmstart_success_rate: float
    last_success_rate: float
    last_mean_return: float
    last_mean_beams_installed: float
    scripted_success_rate: float
    checkpoint_path: str
    metrics_path: str


class HierarchicalOptionTrainer:
    def __init__(self, env: AssemblyTaskBackend, config: TrainingConfig, device: str | None = None) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.actor = OptionActor(env.team_option_obs_dim, env.option_size).to(self.device)
        self.critic = OptionCritic(env.team_option_obs_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.option_actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.option_critic_lr)
        self.bc_optim = torch.optim.Adam(self.actor.parameters(), lr=config.option_behavior_cloning_lr)
        torch.manual_seed(config.seed)
        self.bc_observations, self.bc_masks, self.bc_actions = self._collect_scripted_dataset()

    def train(self, checkpoint_path: Path, metrics_path: Path) -> OptionTrainingSummary:
        history: list[dict[str, float]] = []
        scripted_success = self.evaluate_scripted_baseline(episodes=5)
        warmstart_stats = self._behavior_clone_warmstart()
        best_stats = dict(warmstart_stats)
        best_actor_state = deepcopy(self.actor.state_dict())
        best_critic_state = deepcopy(self.critic.state_dict())

        for iteration in range(self.config.total_iterations):
            self._set_curriculum(iteration)
            batch = self._collect_rollouts(iteration)
            stats = self._update(batch, iteration)
            stats["curriculum_stage"] = float(self._curriculum_stage_index(iteration))
            history.append(stats)
            if self._is_better(stats, best_stats):
                best_stats = dict(stats)
                best_actor_state = deepcopy(self.actor.state_dict())
                best_critic_state = deepcopy(self.critic.state_dict())

        self.env.set_curriculum_stage(None)
        self.actor.load_state_dict(best_actor_state)
        self.critic.load_state_dict(best_critic_state)

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "team_option_obs_dim": self.env.team_option_obs_dim,
                "option_size": self.env.option_size,
            },
            checkpoint_path,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return OptionTrainingSummary(
            iterations=self.config.total_iterations,
            warmstart_success_rate=warmstart_stats["success_rate"],
            last_success_rate=best_stats["success_rate"],
            last_mean_return=best_stats["mean_return"],
            last_mean_beams_installed=best_stats["mean_beams_installed"],
            scripted_success_rate=scripted_success,
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
        )

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])

    def evaluate_policy(self, episodes: int | None = None) -> OptionPolicyMetrics:
        episodes = episodes or self.config.evaluation_episodes
        returns: list[float] = []
        beams_installed: list[int] = []
        option_switches: list[int] = []
        recovery_usage: list[int] = []
        successes = 0
        self.env.set_curriculum_stage(None)

        for episode in range(episodes):
            self.env.reset(seed=self.config.seed + episode)
            done = False
            while not done:
                observation = torch.as_tensor(
                    self.env.get_team_option_observation(), dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                mask = torch.as_tensor(self._masked_option_array(), dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    logits = self._masked_logits(self.actor(observation), mask)
                    option = torch.argmax(logits, dim=-1).item()
                result = self.env.execute_team_option(option, max_primitive_steps=self.env.config.option_max_primitive_steps)
                done = result.done
            artifact = self.env.build_artifact(policy_mode="learned")
            diagnostics = self.env.get_option_episode_diagnostics()
            returns.append(artifact.metrics.total_reward)
            beams_installed.append(artifact.metrics.beams_installed)
            option_switches.append(int(diagnostics["option_switch_count"]))
            recovery_usage.append(
                int(sum(diagnostics["recovery_option_usage"].values()))  # type: ignore[union-attr]
            )
            successes += int(artifact.metrics.success)

        return OptionPolicyMetrics(
            success_rate=successes / max(1, episodes),
            mean_return=float(sum(returns) / max(1, len(returns))),
            mean_beams_installed=float(sum(beams_installed) / max(1, len(beams_installed))),
            mean_option_switches=float(sum(option_switches) / max(1, len(option_switches))),
            mean_recovery_usage=float(sum(recovery_usage) / max(1, len(recovery_usage))),
        )

    def evaluate_scripted_baseline(self, episodes: int = 5) -> float:
        successes = 0
        self.env.set_curriculum_stage(None)
        for episode in range(episodes):
            self.env.reset(seed=self.config.seed + episode)
            done = False
            while not done:
                result = self.env.execute_team_option(self.env.scripted_team_option())
                done = result.done
            artifact = self.env.build_artifact(policy_mode="scripted")
            successes += int(artifact.metrics.success)
        return successes / max(1, episodes)

    def _behavior_clone_warmstart(self) -> dict[str, float]:
        if self.config.option_behavior_cloning_epochs <= 0:
            metrics = self.evaluate_policy()
            return metrics.model_dump()
        for _ in range(self.config.option_behavior_cloning_epochs):
            logits = self._masked_logits(self.actor(self.bc_observations), self.bc_masks)
            loss = F.cross_entropy(logits, self.bc_actions)
            self.bc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.bc_optim.step()
        metrics = self.evaluate_policy()
        return metrics.model_dump()

    def _collect_rollouts(self, iteration: int) -> dict[str, torch.Tensor]:
        observations: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        old_logprobs: list[torch.Tensor] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[torch.Tensor] = []
        episode_returns: list[float] = []
        successes = 0
        beams_installed: list[float] = []
        mixing_ratio = self._scripted_mixing_ratio(iteration)

        for episode in range(self.config.episodes_per_iteration):
            self.env.reset(seed=self.config.seed + episode)
            done = False
            episode_return = 0.0
            previous_option: int | None = None
            while not done:
                observation = torch.as_tensor(
                    self.env.get_team_option_observation(), dtype=torch.float32, device=self.device
                )
                mask = torch.as_tensor(self._masked_option_array(), dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = self._masked_logits(self.actor(observation.unsqueeze(0)), mask.unsqueeze(0)).squeeze(0)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    if torch.rand(1, device=self.device).item() < mixing_ratio:
                        action = torch.as_tensor(int(self.env.scripted_team_option()), dtype=torch.int64, device=self.device)
                    logprob = dist.log_prob(action)
                    value = self.critic(observation.unsqueeze(0)).squeeze(0)

                result = self.env.execute_team_option(
                    int(action.item()), max_primitive_steps=self.env.config.option_max_primitive_steps
                )
                reward = result.reward
                if previous_option is not None and previous_option != int(action.item()):
                    reward -= self.config.option_switch_penalty

                observations.append(observation)
                masks.append(mask)
                actions.append(action)
                old_logprobs.append(logprob)
                rewards.append(reward)
                dones.append(float(result.done))
                values.append(value)
                episode_return += reward
                previous_option = int(action.item())
                done = result.done

            artifact = self.env.build_artifact(policy_mode="learned")
            episode_returns.append(episode_return)
            beams_installed.append(float(artifact.metrics.beams_installed))
            successes += int(artifact.metrics.success)

        reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        value_tensor = torch.stack(values)
        advantages = torch.zeros_like(reward_tensor)
        gae = torch.tensor(0.0, device=self.device)
        for index in reversed(range(len(reward_tensor))):
            mask = 1.0 - done_tensor[index]
            bootstrap = (
                value_tensor[index + 1]
                if index < len(reward_tensor) - 1 and done_tensor[index] == 0
                else torch.tensor(0.0, device=self.device)
            )
            delta = reward_tensor[index] + self.config.gamma * bootstrap * mask - value_tensor[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae
        returns = advantages + value_tensor

        return {
            "observations": torch.stack(observations),
            "masks": torch.stack(masks),
            "actions": torch.stack(actions),
            "old_logprobs": torch.stack(old_logprobs),
            "advantages": advantages,
            "returns": returns,
            "episode_returns": torch.as_tensor(episode_returns, dtype=torch.float32, device=self.device),
            "success_rate": torch.tensor(successes / max(1, self.config.episodes_per_iteration), dtype=torch.float32),
            "mean_beams_installed": torch.tensor(
                sum(beams_installed) / max(1, len(beams_installed)), dtype=torch.float32, device=self.device
            ),
        }

    def _update(self, batch: dict[str, torch.Tensor], iteration: int) -> dict[str, float]:
        observations = batch["observations"]
        masks = batch["masks"]
        actions = batch["actions"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.config.option_update_epochs):
            logits = self._masked_logits(self.actor(observations), masks)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprobs - old_logprobs)
            surrogate_1 = ratio * normalized_advantages
            surrogate_2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * normalized_advantages
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
            bc_logits = self._masked_logits(self.actor(self.bc_observations), self.bc_masks)
            bc_loss = F.cross_entropy(bc_logits, self.bc_actions)

            values = self.critic(observations)
            critic_loss = torch.mean((values - returns) ** 2)

            self.actor_optim.zero_grad()
            (
                actor_loss
                - self.config.option_entropy_coef * entropy
                + self.config.option_behavior_cloning_aux_coef * bc_loss
            ).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            (self.config.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optim.step()

        evaluation = self.evaluate_policy()
        return {
            "iteration": float(iteration),
            "mean_return": evaluation.mean_return,
            "success_rate": evaluation.success_rate,
            "mean_beams_installed": evaluation.mean_beams_installed,
            "mean_option_switches": evaluation.mean_option_switches,
            "mean_recovery_usage": evaluation.mean_recovery_usage,
            "batch_mean_return": float(batch["episode_returns"].mean().item()),
            "scripted_baseline_success_rate": self.evaluate_scripted_baseline(episodes=3),
        }

    def _collect_scripted_dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        action_rows: list[int] = []
        original_beam_count = self.env.active_beam_count
        original_stage_index = self.env.active_stage_index
        stage_count = len(self.env.config.curriculum_stage_beams) if self.env.config.curriculum_stage_beams else 1

        for stage_index in range(stage_count):
            self.env.set_curriculum_stage(stage_index=stage_index if self.env.config.curriculum_stage_beams else 1)
            for episode in range(self.config.episodes_per_iteration * 3):
                self.env.reset(seed=self.config.seed + (stage_index * 100) + episode)
                done = False
                while not done:
                    option = self.env.scripted_team_option()
                    observation_rows.append(
                        torch.as_tensor(self.env.get_team_option_observation(), dtype=torch.float32, device=self.device)
                    )
                    mask_rows.append(
                        torch.as_tensor(self._masked_option_array(), dtype=torch.float32, device=self.device)
                    )
                    action_rows.append(int(option))
                    result = self.env.execute_team_option(option, max_primitive_steps=self.env.config.option_max_primitive_steps)
                    done = result.done

        if original_stage_index is not None and self.env.config.curriculum_stage_beams:
            self.env.set_curriculum_stage(stage_index=original_stage_index)
        else:
            self.env.set_curriculum_stage(original_beam_count)

        return (
            torch.stack(observation_rows),
            torch.stack(mask_rows),
            torch.as_tensor(action_rows, dtype=torch.int64, device=self.device),
        )

    def _curriculum_stage_index(self, iteration: int) -> int:
        thresholds = self.config.curriculum_stage_iterations
        selected = 0
        for index, threshold in enumerate(thresholds):
            if iteration >= threshold:
                selected = index
        return selected

    def _set_curriculum(self, iteration: int) -> None:
        if self.env.config.curriculum_stage_beams:
            self.env.set_curriculum_stage(stage_index=self._curriculum_stage_index(iteration))
            return
        stages = self.env.config.curriculum_beam_stages
        thresholds = self.config.curriculum_stage_iterations
        selected = stages[0]
        for threshold, beam_count in zip(thresholds, stages):
            if iteration >= threshold:
                selected = beam_count
        self.env.set_curriculum_stage(selected)

    def _scripted_mixing_ratio(self, iteration: int) -> float:
        if self.config.total_iterations <= 1:
            return self.config.option_scripted_mixing_end
        progress = iteration / max(1, self.config.total_iterations - 1)
        return (
            self.config.option_scripted_mixing_start
            + (self.config.option_scripted_mixing_end - self.config.option_scripted_mixing_start) * progress
        )

    def _masked_option_array(self) -> np.ndarray:
        mask = self.env.get_team_option_mask().copy()
        if self.env.recovery_option_usage["reset_to_pickup_route"] >= self.config.option_recovery_limit:
            mask[TeamOption.RESET_TO_PICKUP_ROUTE] = 0.0
        if self.env.recovery_option_usage["reposition_after_install"] >= self.config.option_recovery_limit:
            mask[TeamOption.REPOSITION_AFTER_INSTALL] = 0.0
        if mask.sum() <= 0:
            mask[TeamOption.WAIT] = 1.0
        return np.asarray(mask, dtype=np.float32)

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        zero_rows = safe_mask.sum(dim=-1, keepdim=True) == 0
        if zero_rows.any():
            safe_mask = torch.where(zero_rows, torch.zeros_like(safe_mask), safe_mask)
            safe_mask[:, int(TeamOption.WAIT)] = torch.where(
                zero_rows.squeeze(-1),
                torch.ones_like(safe_mask[:, int(TeamOption.WAIT)]),
                safe_mask[:, int(TeamOption.WAIT)],
            )
        return logits.masked_fill(safe_mask <= 0, -1e4)

    def _is_better(self, candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
        if candidate["success_rate"] != incumbent["success_rate"]:
            return candidate["success_rate"] > incumbent["success_rate"]
        if candidate["mean_beams_installed"] != incumbent["mean_beams_installed"]:
            return candidate["mean_beams_installed"] > incumbent["mean_beams_installed"]
        return candidate["mean_return"] > incumbent["mean_return"]
