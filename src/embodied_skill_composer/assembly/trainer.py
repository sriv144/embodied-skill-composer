from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from embodied_skill_composer.assembly.backends import AssemblyTaskBackend
from embodied_skill_composer.assembly.baseline import scripted_joint_policy
from embodied_skill_composer.assembly.models import TrainingConfig


class SharedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, num_phases: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.phase_heads = nn.ModuleList(nn.Linear(128, action_dim) for _ in range(num_phases))

    def forward(self, observations: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(observations)
        outputs = []
        for row, phase_index in zip(hidden, phase_indices.tolist()):
            outputs.append(self.phase_heads[int(phase_index)](row))
        return torch.stack(outputs)


class CentralCritic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


@dataclass
class TrainingSummary:
    iterations: int
    last_mean_return: float
    last_success_rate: float
    baseline_success_rate: float
    warmstart_success_rate: float
    checkpoint_path: str
    metrics_path: str


class MAPPOTrainer:
    def __init__(self, env: AssemblyTaskBackend, config: TrainingConfig, device: str | None = None) -> None:
        self.env = env
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_phases = max(1, len(env.config.beams))
        self.actor = SharedActor(env.obs_dim, env.action_size, self.num_phases).to(self.device)
        self.critic = CentralCritic(env.state_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.bc_optim = torch.optim.Adam(self.actor.parameters(), lr=config.behavior_cloning_lr)
        torch.manual_seed(config.seed)
        self.bc_observations, self.bc_phase_indices, self.bc_actions = self._collect_scripted_dataset()

    def train(self, checkpoint_path: Path, metrics_path: Path) -> TrainingSummary:
        history: list[dict[str, float]] = []
        baseline_success = self.evaluate_scripted_baseline(episodes=5)
        warmstart_stats = self._behavior_clone_warmstart()
        best_stats = dict(warmstart_stats)
        best_actor_state = deepcopy(self.actor.state_dict())
        best_critic_state = deepcopy(self.critic.state_dict())

        for iteration in range(self.config.total_iterations):
            curriculum_stage_index = self._curriculum_stage_index(iteration)
            if self.env.config.curriculum_stage_beams:
                self.env.set_curriculum_stage(stage_index=curriculum_stage_index)
                curriculum_marker = float(curriculum_stage_index)
            else:
                curriculum_beams = self._curriculum_beam_count(iteration)
                self.env.set_curriculum_stage(curriculum_beams)
                curriculum_marker = float(curriculum_beams)
            batch = self._collect_rollouts(iteration)
            iteration_stats = self._update(batch, iteration)
            iteration_stats["curriculum_stage"] = curriculum_marker
            history.append(iteration_stats)
            if self._is_better(iteration_stats, best_stats):
                best_stats = dict(iteration_stats)
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
                "obs_dim": self.env.obs_dim,
                "action_size": self.env.action_size,
                "state_dim": self.env.state_dim,
                "num_phases": self.num_phases,
            },
            checkpoint_path,
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        return TrainingSummary(
            iterations=self.config.total_iterations,
            last_mean_return=best_stats["mean_return"],
            last_success_rate=best_stats["success_rate"],
            baseline_success_rate=baseline_success,
            warmstart_success_rate=warmstart_stats["success_rate"],
            checkpoint_path=str(checkpoint_path),
            metrics_path=str(metrics_path),
        )

    def evaluate_policy(self, episodes: int | None = None) -> dict[str, float]:
        episodes = episodes or self.config.evaluation_episodes
        returns = []
        successes = 0
        self.env.set_curriculum_stage(None)
        for episode in range(episodes):
            obs, _ = self.env.reset(seed=self.config.seed + episode)
            done = False
            total_reward = 0.0
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                mask_tensor = torch.as_tensor(self.env.get_action_masks(), dtype=torch.float32, device=self.device)
                phase_tensor = self._current_phase_indices()
                with torch.no_grad():
                    logits = self._masked_logits(self.actor(obs_tensor, phase_tensor), mask_tensor)
                    actions = torch.argmax(logits, dim=-1).cpu().tolist()
                obs, _, reward, done, _ = self.env.step(actions)
                total_reward += reward
            artifact = self.env.build_artifact(policy_mode="learned")
            returns.append(total_reward)
            successes += int(artifact.metrics.success)
        return {"mean_return": float(sum(returns) / max(1, len(returns))), "success_rate": successes / max(1, episodes)}

    def evaluate_scripted_baseline(self, episodes: int = 5) -> float:
        successes = 0
        self.env.set_curriculum_stage(None)
        for episode in range(episodes):
            self.env.reset(seed=self.config.seed + episode)
            done = False
            while not done:
                _, _, _, done, _ = self.env.step(scripted_joint_policy(self.env))
            artifact = self.env.build_artifact(policy_mode="scripted")
            successes += int(artifact.metrics.success)
        return successes / max(1, episodes)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])

    def _collect_rollouts(self, iteration: int) -> dict[str, torch.Tensor]:
        obs_buffer: list[torch.Tensor] = []
        state_buffer: list[torch.Tensor] = []
        mask_buffer: list[torch.Tensor] = []
        phase_buffer: list[torch.Tensor] = []
        actions_buffer: list[torch.Tensor] = []
        old_logprobs_buffer: list[torch.Tensor] = []
        rewards_buffer: list[float] = []
        dones_buffer: list[float] = []
        values_buffer: list[torch.Tensor] = []
        episode_returns: list[float] = []
        successes = 0
        mixing_ratio = self._scripted_mixing_ratio(iteration)

        for episode in range(self.config.episodes_per_iteration):
            obs, state = self.env.reset(seed=self.config.seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                mask_tensor = torch.as_tensor(self.env.get_action_masks(), dtype=torch.float32, device=self.device)
                phase_tensor = self._current_phase_indices()
                with torch.no_grad():
                    logits = self._masked_logits(self.actor(obs_tensor, phase_tensor), mask_tensor)
                    dist = Categorical(logits=logits)
                    actions = dist.sample()
                    if torch.rand(1, device=self.device).item() < mixing_ratio:
                        actions = torch.as_tensor(scripted_joint_policy(self.env), dtype=torch.int64, device=self.device)
                    logprobs = dist.log_prob(actions)
                    value = self.critic(state_tensor)
                next_obs, next_state, reward, done, _ = self.env.step(actions.cpu().tolist())
                obs_buffer.append(obs_tensor)
                state_buffer.append(state_tensor)
                mask_buffer.append(mask_tensor)
                phase_buffer.append(phase_tensor)
                actions_buffer.append(actions)
                old_logprobs_buffer.append(logprobs)
                rewards_buffer.append(reward)
                dones_buffer.append(float(done))
                values_buffer.append(value)
                obs, state = next_obs, next_state
                episode_return += reward

            artifact = self.env.build_artifact(policy_mode="learned")
            successes += int(artifact.metrics.success)
            episode_returns.append(episode_return)

        rewards = torch.as_tensor(rewards_buffer, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones_buffer, dtype=torch.float32, device=self.device)
        values = torch.stack(values_buffer)
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=self.device)
        for index in reversed(range(len(rewards_buffer))):
            mask = 1.0 - dones[index]
            bootstrap = values[index + 1] if index < len(rewards_buffer) - 1 and dones[index] == 0 else torch.tensor(0.0, device=self.device)
            delta = rewards[index] + self.config.gamma * bootstrap * mask - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae
        returns = advantages + values

        return {
            "observations": torch.stack(obs_buffer),
            "states": torch.stack(state_buffer),
            "action_masks": torch.stack(mask_buffer),
            "phase_indices": torch.stack(phase_buffer),
            "actions": torch.stack(actions_buffer),
            "old_logprobs": torch.stack(old_logprobs_buffer),
            "advantages": advantages,
            "returns": returns,
            "episode_returns": torch.as_tensor(episode_returns, dtype=torch.float32),
            "success_rate": torch.tensor(successes / max(1, self.config.episodes_per_iteration), dtype=torch.float32),
        }

    def _update(self, batch: dict[str, torch.Tensor], iteration: int) -> dict[str, float]:
        observations = batch["observations"]
        states = batch["states"]
        action_masks = batch["action_masks"]
        phase_indices = batch["phase_indices"]
        actions = batch["actions"]
        old_logprobs = batch["old_logprobs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        flat_obs = observations.reshape(-1, self.env.obs_dim)
        flat_masks = action_masks.reshape(-1, self.env.action_size)
        flat_phase_indices = phase_indices.reshape(-1)
        flat_actions = actions.reshape(-1)
        flat_old_logprobs = old_logprobs.reshape(-1)
        flat_advantages = advantages.repeat_interleave(self.env.num_agents)
        flat_returns = returns.repeat_interleave(self.env.num_agents)
        flat_states = states.repeat_interleave(self.env.num_agents, dim=0)

        normalized_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        for _ in range(self.config.update_epochs):
            logits = self._masked_logits(self.actor(flat_obs, flat_phase_indices), flat_masks)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(flat_actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_logprobs - flat_old_logprobs)
            surrogate_1 = ratio * normalized_advantages
            surrogate_2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * normalized_advantages
            actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
            bc_loss = F.cross_entropy(self.actor(self.bc_observations, self.bc_phase_indices), self.bc_actions)

            values = self.critic(flat_states)
            critic_loss = torch.mean((values - flat_returns) ** 2)

            self.actor_optim.zero_grad()
            (
                actor_loss
                - self.config.entropy_coef * entropy
                + self.config.behavior_cloning_aux_coef * bc_loss
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
            "mean_return": evaluation["mean_return"],
            "success_rate": evaluation["success_rate"],
            "batch_mean_return": float(batch["episode_returns"].mean().item()),
            "scripted_baseline_success_rate": self.evaluate_scripted_baseline(episodes=3),
        }

    def _behavior_clone_warmstart(self) -> dict[str, float]:
        if self.config.behavior_cloning_epochs <= 0:
            return self.evaluate_policy()
        for _ in range(self.config.behavior_cloning_epochs):
            logits = self.actor(self.bc_observations, self.bc_phase_indices)
            loss = F.cross_entropy(logits, self.bc_actions)
            self.bc_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.bc_optim.step()
        return self.evaluate_policy()

    def _collect_scripted_dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observation_rows: list[torch.Tensor] = []
        phase_rows: list[torch.Tensor] = []
        action_rows: list[int] = []
        original_beam_count = self.env.active_beam_count
        original_stage_index = self.env.active_stage_index
        for episode in range(self.config.episodes_per_iteration * 4):
            if self.env.config.curriculum_stage_beams:
                self.env.set_curriculum_stage(stage_index=0)
            else:
                self.env.set_curriculum_stage(1)
            obs, _ = self.env.reset(seed=self.config.seed + episode)
            done = False
            while not done:
                scripted_actions = scripted_joint_policy(self.env)
                observation_rows.extend(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                )
                phase_rows.extend(self._current_phase_indices())
                action_rows.extend(scripted_actions)
                obs, _, _, done, _ = self.env.step(scripted_actions)
        if original_stage_index is not None and self.env.config.curriculum_stage_beams:
            self.env.set_curriculum_stage(stage_index=original_stage_index)
        else:
            self.env.set_curriculum_stage(original_beam_count)
        return (
            torch.stack(list(observation_rows)),
            torch.stack(list(phase_rows)),
            torch.as_tensor(action_rows, dtype=torch.int64, device=self.device),
        )

    def _scripted_mixing_ratio(self, iteration: int) -> float:
        if self.config.total_iterations <= 1:
            return self.config.scripted_mixing_end
        progress = iteration / max(1, self.config.total_iterations - 1)
        return (
            self.config.scripted_mixing_start
            + (self.config.scripted_mixing_end - self.config.scripted_mixing_start) * progress
        )

    def _curriculum_beam_count(self, iteration: int) -> int:
        stages = self.env.config.curriculum_beam_stages
        thresholds = self.config.curriculum_stage_iterations
        selected = stages[0]
        for threshold, beam_count in zip(thresholds, stages):
            if iteration >= threshold:
                selected = beam_count
        return selected

    def _curriculum_stage_index(self, iteration: int) -> int:
        thresholds = self.config.curriculum_stage_iterations
        selected = 0
        for index, threshold in enumerate(thresholds):
            if iteration >= threshold:
                selected = index
        return selected

    def _masked_logits(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        safe_mask = action_mask.clone()
        zero_rows = safe_mask.sum(dim=-1, keepdim=True) == 0
        if zero_rows.any():
            safe_mask = torch.where(zero_rows, torch.zeros_like(safe_mask), safe_mask)
            safe_mask[:, 0] = torch.where(zero_rows.squeeze(-1), torch.ones_like(safe_mask[:, 0]), safe_mask[:, 0])
        invalid = safe_mask <= 0
        return logits.masked_fill(invalid, -1e4)

    def _current_phase_indices(self) -> torch.Tensor:
        phase = min(self.env.state.current_beam_index, self.num_phases - 1)
        return torch.full((self.env.num_agents,), phase, dtype=torch.int64, device=self.device)

    def _is_better(self, candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
        if candidate["success_rate"] != incumbent["success_rate"]:
            return candidate["success_rate"] > incumbent["success_rate"]
        return candidate["mean_return"] > incumbent["mean_return"]
