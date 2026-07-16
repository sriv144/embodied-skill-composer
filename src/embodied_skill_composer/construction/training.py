from __future__ import annotations

import csv
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from tensordict import TensorDict
from torch import Tensor
from torch.nn import functional as functional
from torch.utils.tensorboard import SummaryWriter
from torchrl.objectives.multiagent import IPPOLoss, MAPPOLoss

from embodied_skill_composer.construction.intelligence_models import PolicyManifest
from embodied_skill_composer.construction.marl_env_v1 import (
    FLEET_SIZE,
    TemporalConstructionCoordinationEnv,
)
from embodied_skill_composer.construction.models import HouseDesign
from embodied_skill_composer.construction.policy import (
    TorchRLPolicyBundle,
    build_torchrl_policy,
    export_actor_onnx,
    observations_to_tensors,
    save_policy_checkpoint,
)
from embodied_skill_composer.construction.scenarios import (
    CottageScenarioConfig,
    generate_cottage_scenario,
)
from embodied_skill_composer.construction.scheduler import schedule_build


class TrainingConfig(BaseModel):
    algorithm: Literal["mappo", "ippo"] = "mappo"
    profile: Literal["unit", "smoke", "research"] = "smoke"
    seed: int = 7
    transitions: int = Field(default=50_000, gt=0)
    expert_episodes: int = Field(default=24, ge=0)
    behavior_clone_epochs: int = Field(default=20, ge=0)
    rollout_decisions: int = Field(default=256, gt=0)
    ppo_epochs: int = Field(default=4, gt=0)
    minibatch_size: int = Field(default=256, gt=0)
    hidden_dim: int = Field(default=128, ge=32, le=512)
    learning_rate: float = Field(default=3e-4, gt=0)
    gamma: float = Field(default=0.99, gt=0, le=1)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    clip_epsilon: float = Field(default=0.2, gt=0)
    entropy_coefficient: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    include_training_failures: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    output_root: Path = Path("logs/construction_intelligence/training")

    @classmethod
    def for_profile(
        cls,
        profile: Literal["unit", "smoke", "research"],
        *,
        algorithm: Literal["mappo", "ippo"] = "mappo",
        seed: int = 7,
    ) -> "TrainingConfig":
        if profile == "unit":
            return cls(
                algorithm=algorithm,
                profile=profile,
                seed=seed,
                transitions=64,
                expert_episodes=1,
                behavior_clone_epochs=1,
                rollout_decisions=8,
                ppo_epochs=1,
                minibatch_size=8,
                hidden_dim=32,
                include_training_failures=False,
                device="cpu",
            )
        if profile == "research":
            return cls(
                algorithm=algorithm,
                profile=profile,
                seed=seed,
                transitions=1_500_000,
                expert_episodes=128,
                behavior_clone_epochs=40,
                rollout_decisions=2048,
                ppo_epochs=6,
                minibatch_size=512,
            )
        return cls(algorithm=algorithm, profile=profile, seed=seed)


class TrainingArtifacts(BaseModel):
    run_id: str
    run_dir: Path
    config_path: Path
    checkpoint_path: Path
    onnx_path: Path
    policy_manifest_path: Path
    learning_curve_path: Path
    tensorboard_dir: Path
    transitions: int
    updates: int


@dataclass
class RolloutStep:
    observations: dict[str, Tensor]
    state: Tensor
    actions: Tensor
    action_log_prob: Tensor
    values: Tensor
    reward: float
    done: bool


@dataclass
class RolloutEpisode:
    steps: list[RolloutStep]
    bootstrap_values: Tensor
    completed: bool


def train_swarm_policy(
    base_design: HouseDesign,
    config: TrainingConfig,
    *,
    progress_callback=None,
    cancel_check=None,
) -> TrainingArtifacts:
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    run_id = _run_id(config)
    run_dir = config.output_root.resolve() / run_id
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=False)
    config_path = run_dir / "training_config.json"
    config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    bundle = build_torchrl_policy(config.algorithm, hidden_dim=config.hidden_dim).to(device)
    loss_module = build_ppo_loss(bundle, config).to(device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    curve_rows: list[dict[str, float | int]] = []
    try:
        if config.expert_episodes and config.behavior_clone_epochs:
            expert = collect_cp_sat_expert_samples(
                base_design,
                episode_count=config.expert_episodes,
                seed=config.seed,
                device=device,
            )
            bc_losses = behavior_clone_actor(
                bundle,
                expert,
                epochs=config.behavior_clone_epochs,
                learning_rate=config.learning_rate,
                max_grad_norm=config.max_grad_norm,
            )
            for epoch, loss in enumerate(bc_losses, start=1):
                writer.add_scalar("behavior_cloning/loss", loss, epoch)
            _notify(progress_callback, "behavior_cloning_complete", 0, {"loss": bc_losses[-1]})

        transitions = 0
        updates = 0
        episode_cursor = 0
        while transitions < config.transitions:
            if cancel_check and cancel_check():
                raise RuntimeError("training cancelled")
            remaining_decisions = max((config.transitions - transitions) // FLEET_SIZE, 1)
            decision_target = min(config.rollout_decisions, remaining_decisions)
            episodes, episode_cursor = collect_policy_rollouts(
                bundle,
                base_design,
                config,
                decision_target=decision_target,
                episode_cursor=episode_cursor,
                device=device,
            )
            batch = build_ppo_batch(
                episodes,
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                device=device,
            )
            losses = optimize_ppo_batch(
                loss_module,
                optimizer,
                batch,
                epochs=config.ppo_epochs,
                minibatch_size=config.minibatch_size,
                max_grad_norm=config.max_grad_norm,
            )
            decisions = int(batch.batch_size[0])
            transitions += decisions * FLEET_SIZE
            updates += 1
            completion = float(np.mean([episode.completed for episode in episodes]))
            mean_return = float(
                np.mean([sum(step.reward for step in episode.steps) for episode in episodes])
            )
            row = {
                "update": updates,
                "transitions": min(transitions, config.transitions),
                "loss_objective": losses["loss_objective"],
                "loss_critic": losses["loss_critic"],
                "loss_entropy": losses["loss_entropy"],
                "mean_episode_return": mean_return,
                "rollout_terminal_fraction": completion,
            }
            curve_rows.append(row)
            for key, value in row.items():
                if key not in {"update", "transitions"}:
                    writer.add_scalar(f"training/{key}", value, transitions)
            _notify(progress_callback, "ppo_update", min(transitions, config.transitions), row)
    finally:
        writer.close()

    learning_curve_path = run_dir / "learning_curve.csv"
    _write_curve(learning_curve_path, curve_rows)
    checkpoint_path = run_dir / "policy.pt"
    checkpoint_sha = save_policy_checkpoint(
        bundle,
        checkpoint_path,
        metadata={
            "run_id": run_id,
            "hidden_dim": config.hidden_dim,
            "transitions": min(transitions, config.transitions),
            "environment_schema": TemporalConstructionCoordinationEnv.metadata["name"],
        },
    )
    onnx_path = export_actor_onnx(bundle.actor_model, run_dir / "actor.onnx", device="cpu")
    manifest = PolicyManifest(
        policy_id=run_id,
        controller=config.algorithm,
        git_sha=_git_sha(),
        seed=config.seed,
        transition_count=min(transitions, config.transitions),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        onnx_path=str(onnx_path),
        config=config.model_dump(mode="json"),
    )
    policy_manifest_path = run_dir / "policy_manifest.json"
    policy_manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return TrainingArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        onnx_path=onnx_path,
        policy_manifest_path=policy_manifest_path,
        learning_curve_path=learning_curve_path,
        tensorboard_dir=tensorboard_dir,
        transitions=min(transitions, config.transitions),
        updates=updates,
    )


def collect_cp_sat_expert_samples(
    base_design: HouseDesign,
    *,
    episode_count: int,
    seed: int,
    device: torch.device | str,
) -> dict[str, Tensor]:
    samples: dict[str, list[Tensor]] = defaultdict_tensor_lists()
    for episode_index in range(episode_count):
        scenario_seed = (seed + episode_index) % 800
        scenario = generate_cottage_scenario(
            scenario_seed,
            base_design,
            config=CottageScenarioConfig(obstacle_count_range=(0, 2)),
        )
        env = TemporalConstructionCoordinationEnv(scenario)
        observations, _ = env.reset(seed=scenario_seed)
        schedule = schedule_build(scenario.plan, "optimized")
        priority = {
            job.module_id: (job.start_s, job.end_s, tuple(job.robot_ids))
            for job in schedule.jobs
        }
        while env.agents:
            tensors = observations_to_tensors(
                observations,
                env.possible_agents,
                device=device,
            )
            actions = cp_sat_expert_actions(env, priority)
            for key, value in tensors.items():
                samples[key].append(value)
            samples["actions"].append(
                torch.tensor(
                    [actions[agent] for agent in env.possible_agents],
                    dtype=torch.long,
                    device=device,
                )
            )
            observations, _, _, _, _ = env.step(actions)
    return {key: torch.stack(value) for key, value in samples.items()}


def behavior_clone_actor(
    bundle: TorchRLPolicyBundle,
    samples: dict[str, Tensor],
    *,
    epochs: int,
    learning_rate: float,
    max_grad_norm: float,
) -> list[float]:
    optimizer = torch.optim.Adam(bundle.actor_model.parameters(), lr=learning_rate)
    losses: list[float] = []
    for _ in range(epochs):
        logits = bundle.actor_model(
            samples["self"],
            samples["robots"],
            samples["modules"],
            samples["dependencies"],
            samples["action_mask"],
        )
        targets = samples["actions"]
        per_action = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        weights = torch.where(targets.reshape(-1) == 0, 0.5, 1.5)
        loss = (per_action * weights).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.actor_model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def collect_policy_rollouts(
    bundle: TorchRLPolicyBundle,
    base_design: HouseDesign,
    config: TrainingConfig,
    *,
    decision_target: int,
    episode_cursor: int,
    device: torch.device | str,
) -> tuple[list[RolloutEpisode], int]:
    episodes: list[RolloutEpisode] = []
    decisions = 0
    while decisions < decision_target:
        scenario_seed = (config.seed + episode_cursor) % 800
        scenario = generate_cottage_scenario(
            scenario_seed,
            base_design,
            config=CottageScenarioConfig(
                include_failures=config.include_training_failures,
                obstacle_count_range=(0, 4),
            ),
        )
        env = TemporalConstructionCoordinationEnv(scenario)
        observations, _ = env.reset(seed=scenario_seed)
        trajectory: list[RolloutStep] = []
        while env.agents and decisions < decision_target:
            tensors = observations_to_tensors(
                observations,
                env.possible_agents,
                device=device,
            )
            state = torch.as_tensor(env.state(), dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = bundle.actor_model(
                    tensors["self"].unsqueeze(0),
                    tensors["robots"].unsqueeze(0),
                    tensors["modules"].unsqueeze(0),
                    tensors["dependencies"].unsqueeze(0),
                    tensors["action_mask"].unsqueeze(0),
                ).squeeze(0)
                distribution = torch.distributions.Categorical(logits=logits)
                sampled_actions = distribution.sample()
                log_prob = distribution.log_prob(sampled_actions)
                values = _critic_values(bundle, tensors, state)
            actions = {
                agent: int(sampled_actions[index].item())
                for index, agent in enumerate(env.possible_agents)
            }
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            done = all(terminations.values()) or all(truncations.values())
            trajectory.append(
                RolloutStep(
                    observations={key: value.detach() for key, value in tensors.items()},
                    state=state.detach(),
                    actions=sampled_actions.detach(),
                    action_log_prob=log_prob.detach(),
                    values=values.detach(),
                    reward=float(np.mean(list(rewards.values()))),
                    done=done,
                )
            )
            observations = next_observations
            decisions += 1
        if env.agents:
            bootstrap_tensors = observations_to_tensors(
                observations,
                env.possible_agents,
                device=device,
            )
            bootstrap_state = torch.as_tensor(
                env.state(),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                bootstrap_values = _critic_values(
                    bundle,
                    bootstrap_tensors,
                    bootstrap_state,
                ).detach()
        else:
            bootstrap_values = torch.zeros(FLEET_SIZE, 1, device=device)
        episodes.append(
            RolloutEpisode(
                steps=trajectory,
                bootstrap_values=bootstrap_values,
                completed=bool(trajectory and trajectory[-1].done),
            )
        )
        episode_cursor += 1
    return episodes, episode_cursor


def build_ppo_batch(
    episodes: list[RolloutEpisode],
    *,
    gamma: float,
    gae_lambda: float,
    device: torch.device | str,
) -> TensorDict:
    flattened: list[RolloutStep] = []
    advantages: list[Tensor] = []
    value_targets: list[Tensor] = []
    for episode in episodes:
        episode_advantages = [
            torch.zeros(FLEET_SIZE, 1, device=device) for _ in episode.steps
        ]
        next_advantage = torch.zeros(FLEET_SIZE, 1, device=device)
        next_value = episode.bootstrap_values
        for index in range(len(episode.steps) - 1, -1, -1):
            step = episode.steps[index]
            continuation = 0.0 if step.done else 1.0
            reward = torch.full((FLEET_SIZE, 1), step.reward, device=device)
            delta = reward + gamma * next_value * continuation - step.values
            next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
            episode_advantages[index] = next_advantage
            next_value = step.values
        flattened.extend(episode.steps)
        advantages.extend(episode_advantages)
        value_targets.extend(
            advantage + step.values
            for advantage, step in zip(episode_advantages, episode.steps, strict=True)
        )

    observations = {
        key: torch.stack([step.observations[key] for step in flattened])
        for key in ("self", "robots", "modules", "dependencies", "action_mask")
    }
    count = len(flattened)
    data = TensorDict(batch_size=[count], device=device)
    for key, value in observations.items():
        data.set(("agents", key), value)
    data.set("state", torch.stack([step.state for step in flattened]))
    data.set(("agents", "action"), torch.stack([step.actions for step in flattened]))
    data.set(
        ("agents", "action_log_prob"),
        torch.stack([step.action_log_prob for step in flattened]),
    )
    data.set(("agents", "advantage"), torch.stack(advantages))
    data.set(("agents", "value_target"), torch.stack(value_targets))
    return data


def optimize_ppo_batch(
    loss_module,
    optimizer: torch.optim.Optimizer,
    batch: TensorDict,
    *,
    epochs: int,
    minibatch_size: int,
    max_grad_norm: float,
) -> dict[str, float]:
    latest = {"loss_objective": 0.0, "loss_critic": 0.0, "loss_entropy": 0.0}
    for _ in range(epochs):
        permutation = torch.randperm(batch.batch_size[0], device=batch.device)
        for start in range(0, batch.batch_size[0], minibatch_size):
            indices = permutation[start : start + minibatch_size]
            losses = loss_module(batch[indices])
            objective = sum(
                value
                for key, value in losses.items()
                if key.startswith("loss_")
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optimizer.step()
            for key in latest:
                if key in losses:
                    latest[key] = float(losses[key].detach().cpu())
    return latest


def cp_sat_expert_actions(
    env: TemporalConstructionCoordinationEnv,
    priority: dict[str, tuple[int, int, tuple[str, ...]]],
) -> dict[str, int]:
    actions = {agent: 0 for agent in env.agents}
    available = {
        agent for agent in env.agents if env.robot_runtime[agent].status == "idle"
    }
    for module in sorted(
        env.ready_modules(),
        key=lambda item: (*priority[item.module_id][:2], item.module_id),
    ):
        preferred = priority[module.module_id][2]
        team = list(preferred) if set(preferred) <= available else None
        if team is not None and (
            len(team) != module.required_team_size
            or sum(env.robots[agent].payload_capacity_kg for agent in team) < module.mass_kg
        ):
            team = None
        if team is None:
            continue
        action = env.module_index[module.module_id] + 1
        for agent in team:
            actions[agent] = action
            available.remove(agent)
    return actions


def build_ppo_loss(bundle: TorchRLPolicyBundle, config: TrainingConfig):
    loss_class = MAPPOLoss if config.algorithm == "mappo" else IPPOLoss
    loss_module = loss_class(
        bundle.actor,
        bundle.critic,
        clip_epsilon=config.clip_epsilon,
        entropy_coeff=config.entropy_coefficient,
        critic_coeff=1.0,
        normalize_advantage=True,
    )
    loss_module.set_keys(
        action=("agents", "action"),
        sample_log_prob=("agents", "action_log_prob"),
        value=("agents", "state_value"),
        advantage=("agents", "advantage"),
        value_target=("agents", "value_target"),
    )
    return loss_module


def _critic_values(
    bundle: TorchRLPolicyBundle,
    observations: dict[str, Tensor],
    state: Tensor,
) -> Tensor:
    if bundle.algorithm == "mappo":
        return bundle.critic_model(state.unsqueeze(0)).squeeze(0)
    return bundle.critic_model(
        observations["self"].unsqueeze(0),
        observations["modules"].unsqueeze(0),
    ).squeeze(0)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_id(config: TrainingConfig) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{config.algorithm}-{config.profile}-s{config.seed}"


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0]) if rows else ["update", "transitions"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _notify(callback, event: str, transitions: int, payload: dict[str, object]) -> None:
    if callback:
        callback({"event": event, "transitions": transitions, **payload})


def defaultdict_tensor_lists() -> dict[str, list[Tensor]]:
    return {
        "self": [],
        "robots": [],
        "modules": [],
        "dependencies": [],
        "action_mask": [],
        "actions": [],
    }
