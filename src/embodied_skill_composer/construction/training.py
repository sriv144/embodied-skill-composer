from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator
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
    experiment_id: str = "ad_hoc"
    experiment_variant: str = "default"
    training_seed: int | None = None
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
    checkpoint_fractions: list[float] = Field(default_factory=lambda: [1.0])
    checkpoint_lineage: list[str] = Field(default_factory=list)
    configuration_digest: str | None = None
    source_commit: str | None = None
    source_dirty: bool = False
    source_tree_digest: str | None = None
    resume_checkpoint: Path | None = None
    resume_provenance: dict[str, object] = Field(default_factory=dict)
    environment_fingerprint: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reproducibility_fields(self) -> "TrainingConfig":
        if self.training_seed is None:
            self.training_seed = self.seed
        if self.training_seed != self.seed:
            raise ValueError("training_seed must match seed")
        fractions = sorted(set(self.checkpoint_fractions))
        if not fractions or fractions[-1] != 1.0:
            fractions.append(1.0)
        if fractions[0] <= 0 or fractions[-1] > 1:
            raise ValueError("checkpoint_fractions must be in (0, 1]")
        self.checkpoint_fractions = fractions
        return self

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
    current_environment = environment_fingerprint()
    if (
        config.environment_fingerprint
        and config.environment_fingerprint != current_environment
    ):
        raise ValueError(
            "incompatible training execution (environment fingerprint changed)"
        )
    if not config.environment_fingerprint:
        config.environment_fingerprint = current_environment
    current_source = source_fingerprint()
    current_commit = str(current_source["commit"])
    current_dirty = bool(current_source["dirty"])
    current_tree_digest = str(current_source["tree_digest"])
    if config.profile == "research" and current_dirty:
        raise ValueError("research training requires a clean source worktree")
    if (
        config.source_commit not in {None, current_commit}
        or (
            config.source_tree_digest is not None
            and (
                config.source_dirty != current_dirty
                or config.source_tree_digest != current_tree_digest
            )
        )
    ):
        raise ValueError(
            "incompatible training execution (source worktree fingerprint changed)"
        )
    resolved_commit = config.source_commit or current_commit
    config.source_dirty = current_dirty
    config.source_tree_digest = current_tree_digest
    resolved_digest = configuration_digest(config)
    if config.configuration_digest not in {None, resolved_digest}:
        raise ValueError("training configuration digest does not match the supplied digest")
    config.source_commit = resolved_commit
    config.configuration_digest = resolved_digest
    design_digest = _design_digest(base_design)
    _seed_everything(config.seed)
    device = _resolve_device(config.device)
    if config.resume_checkpoint is not None:
        resume_path = config.resume_checkpoint.resolve()
        if not resume_path.is_file():
            raise ValueError(f"resume checkpoint does not exist: {resume_path}")
        run_dir = resume_path.parent.parent if resume_path.parent.name == "checkpoints" else resume_path.parent
        run_id = run_dir.name
    else:
        run_id = _run_id(config)
        run_dir = config.output_root.resolve() / run_id
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=config.resume_checkpoint is not None)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    bundle = build_torchrl_policy(config.algorithm, hidden_dim=config.hidden_dim).to(device)
    loss_module = build_ppo_loss(bundle, config).to(device)
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=config.learning_rate)
    bc_optimizer = torch.optim.Adam(bundle.actor_model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    curve_rows: list[dict[str, float | int]] = []
    transitions = 0
    updates = 0
    episode_cursor = 0
    bc_epoch = 0
    checkpoint_lineage = list(config.checkpoint_lineage)
    if config.resume_checkpoint is not None:
        resumed = load_training_checkpoint(
            config.resume_checkpoint,
            bundle=bundle,
            ppo_optimizer=optimizer,
            bc_optimizer=bc_optimizer,
            expected_configuration_digest=resolved_digest,
            expected_design_digest=design_digest,
            expected_source_commit=resolved_commit,
            expected_source_dirty=current_dirty,
            expected_source_tree_digest=current_tree_digest,
            device=device,
        )
        transitions = _checkpoint_int(resumed, "transitions")
        updates = _checkpoint_int(resumed, "updates")
        episode_cursor = _checkpoint_int(resumed, "episode_cursor")
        bc_epoch = _checkpoint_int(resumed, "bc_epoch")
        curve_rows = cast(list[dict[str, float | int]], resumed["curve_rows"])
        previous_lineage = resumed.get("checkpoint_lineage", [])
        if isinstance(previous_lineage, list):
            checkpoint_lineage = [str(item) for item in previous_lineage]
        checkpoint_lineage.append(str(config.resume_checkpoint.resolve()))
        _restore_rng_state(cast(dict[str, object], resumed["rng_state"]))
        _notify(
            progress_callback,
            "training_resumed",
            transitions,
            {
                "checkpoint_path": str(config.resume_checkpoint.resolve()),
                "updates": updates,
                "episode_cursor": episode_cursor,
            },
        )

    checkpoint_targets = {
        max(1, int(round(config.transitions * fraction))): fraction
        for fraction in config.checkpoint_fractions
    }
    saved_targets = {target for target in checkpoint_targets if target <= transitions}

    def persist_checkpoint(*, event: str, fraction: float | None = None) -> Path:
        latest_path = checkpoints_dir / "latest.pt"
        published_path = latest_path
        if fraction is not None:
            published_path = checkpoints_dir / f"checkpoint_{int(round(fraction * 100)):03d}pct.pt"
            if str(published_path) not in checkpoint_lineage:
                checkpoint_lineage.append(str(published_path))
        save_training_checkpoint(
            latest_path,
            bundle=bundle,
            ppo_optimizer=optimizer,
            bc_optimizer=bc_optimizer,
            config=config,
            configuration_digest_value=resolved_digest,
            design_digest=design_digest,
            source_commit_value=resolved_commit,
            transitions=transitions,
            updates=updates,
            episode_cursor=episode_cursor,
            bc_epoch=bc_epoch,
            curve_rows=curve_rows,
            checkpoint_lineage=checkpoint_lineage,
        )
        if fraction is not None:
            save_training_checkpoint(
                published_path,
                bundle=bundle,
                ppo_optimizer=optimizer,
                bc_optimizer=bc_optimizer,
                config=config,
                configuration_digest_value=resolved_digest,
                design_digest=design_digest,
                source_commit_value=resolved_commit,
                transitions=transitions,
                updates=updates,
                episode_cursor=episode_cursor,
                bc_epoch=bc_epoch,
                curve_rows=curve_rows,
                checkpoint_lineage=checkpoint_lineage,
            )
        _notify(
            progress_callback,
            event,
            min(transitions, config.transitions),
            {
                "checkpoint_path": str(latest_path),
                "snapshot_path": str(published_path),
                "fraction": fraction,
            },
        )
        return latest_path

    try:
        if not (checkpoints_dir / "latest.pt").exists():
            persist_checkpoint(event="checkpoint_saved")
        if config.expert_episodes and bc_epoch < config.behavior_clone_epochs:
            expert = collect_cp_sat_expert_samples(
                base_design,
                episode_count=config.expert_episodes,
                seed=config.seed,
                device=device,
                cancel_check=cancel_check,
            )
            latest_bc_loss = 0.0
            while bc_epoch < config.behavior_clone_epochs:
                if cancel_check and cancel_check():
                    persist_checkpoint(event="checkpoint_saved")
                    raise RuntimeError("training cancelled")
                latest_bc_loss = behavior_clone_actor(
                    bundle,
                    expert,
                    epochs=1,
                    learning_rate=config.learning_rate,
                    max_grad_norm=config.max_grad_norm,
                    optimizer=bc_optimizer,
                )[0]
                bc_epoch += 1
                persist_checkpoint(event="behavior_cloning_checkpoint")
                epoch = bc_epoch
                loss = latest_bc_loss
                writer.add_scalar("behavior_cloning/loss", loss, epoch)
            _notify(
                progress_callback,
                "behavior_cloning_complete",
                transitions,
                {"loss": latest_bc_loss, "epoch": bc_epoch},
            )

        while transitions < config.transitions:
            if cancel_check and cancel_check():
                raise RuntimeError("training cancelled")
            future_targets = [
                target for target in checkpoint_targets if target > transitions
            ]
            next_stop = min([config.transitions, *future_targets])
            remaining_decisions = max(
                (next_stop - transitions + FLEET_SIZE - 1) // FLEET_SIZE,
                1,
            )
            decision_target = min(config.rollout_decisions, remaining_decisions)
            episodes, episode_cursor = collect_policy_rollouts(
                bundle,
                base_design,
                config,
                decision_target=decision_target,
                episode_cursor=episode_cursor,
                device=device,
                cancel_check=cancel_check,
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
            crossed = sorted(
                target
                for target in checkpoint_targets
                if target <= transitions and target not in saved_targets
            )
            if crossed:
                for target in crossed:
                    persist_checkpoint(
                        event="checkpoint_saved",
                        fraction=checkpoint_targets[target],
                    )
            else:
                persist_checkpoint(event="checkpoint_saved")
            saved_targets.update(crossed)
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
        git_sha=resolved_commit,
        seed=config.seed,
        experiment_id=config.experiment_id,
        experiment_variant=config.experiment_variant,
        training_seed=config.training_seed,
        transition_count=min(transitions, config.transitions),
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        checkpoint_lineage=checkpoint_lineage,
        configuration_digest=resolved_digest,
        source_commit=resolved_commit,
        source_dirty=current_dirty,
        source_tree_digest=current_tree_digest,
        resume_provenance=config.resume_provenance,
        environment_fingerprint=config.environment_fingerprint,
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
    cancel_check=None,
) -> dict[str, Tensor]:
    samples: dict[str, list[Tensor]] = defaultdict_tensor_lists()
    for episode_index in range(episode_count):
        if cancel_check and cancel_check():
            raise RuntimeError("training cancelled")
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
            if cancel_check and cancel_check():
                raise RuntimeError("training cancelled")
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
    optimizer: torch.optim.Optimizer | None = None,
) -> list[float]:
    optimizer = optimizer or torch.optim.Adam(
        bundle.actor_model.parameters(), lr=learning_rate
    )
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
    cancel_check=None,
) -> tuple[list[RolloutEpisode], int]:
    episodes: list[RolloutEpisode] = []
    decisions = 0
    while decisions < decision_target:
        if cancel_check and cancel_check():
            raise RuntimeError("training cancelled")
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
            if cancel_check and cancel_check():
                raise RuntimeError("training cancelled")
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


def save_training_checkpoint(
    path: Path,
    *,
    bundle: TorchRLPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    bc_optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    configuration_digest_value: str,
    design_digest: str,
    source_commit_value: str,
    transitions: int,
    updates: int,
    episode_cursor: int,
    bc_epoch: int,
    curve_rows: list[dict[str, float | int]],
    checkpoint_lineage: list[str],
) -> Path:
    """Atomically persist all state required for an exact local resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema_version": 2,
        "algorithm": bundle.algorithm,
        "environment_schema": TemporalConstructionCoordinationEnv.metadata["name"],
        "actor_state_dict": bundle.actor_model.state_dict(),
        "critic_state_dict": bundle.critic_model.state_dict(),
        "ppo_optimizer_state_dict": ppo_optimizer.state_dict(),
        "bc_optimizer_state_dict": bc_optimizer.state_dict(),
        "transitions": transitions,
        "updates": updates,
        "episode_cursor": episode_cursor,
        "bc_epoch": bc_epoch,
        "curve_rows": curve_rows,
        "rng_state": _capture_rng_state(),
        "config": config.model_dump(mode="json"),
        "configuration_digest": configuration_digest_value,
        "design_digest": design_digest,
        "source_commit": source_commit_value,
        "source_dirty": config.source_dirty,
        "source_tree_digest": config.source_tree_digest,
        "checkpoint_lineage": checkpoint_lineage,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_training_checkpoint(
    path: Path,
    *,
    bundle: TorchRLPolicyBundle,
    ppo_optimizer: torch.optim.Optimizer,
    bc_optimizer: torch.optim.Optimizer,
    expected_configuration_digest: str,
    expected_design_digest: str,
    expected_source_commit: str,
    expected_source_dirty: bool,
    expected_source_tree_digest: str,
    device: torch.device | str,
) -> dict[str, object]:
    payload_value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload_value, dict):
        raise ValueError("resume checkpoint payload is not a dictionary")
    payload = cast(dict[str, object], payload_value)
    compatibility = {
        "schema_version": (payload.get("schema_version"), 2),
        "algorithm": (payload.get("algorithm"), bundle.algorithm),
        "environment_schema": (
            payload.get("environment_schema"),
            TemporalConstructionCoordinationEnv.metadata["name"],
        ),
        "configuration_digest": (
            payload.get("configuration_digest"),
            expected_configuration_digest,
        ),
        "design_digest": (payload.get("design_digest"), expected_design_digest),
        "source_commit": (payload.get("source_commit"), expected_source_commit),
        "source_dirty": (payload.get("source_dirty"), expected_source_dirty),
        "source_tree_digest": (
            payload.get("source_tree_digest"),
            expected_source_tree_digest,
        ),
    }
    mismatches = [
        f"{name}: checkpoint={actual!r}, current={expected!r}"
        for name, (actual, expected) in compatibility.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("incompatible resume checkpoint (" + "; ".join(mismatches) + ")")
    actor_state = payload.get("actor_state_dict")
    critic_state = payload.get("critic_state_dict")
    ppo_state = payload.get("ppo_optimizer_state_dict")
    bc_state = payload.get("bc_optimizer_state_dict")
    if not all(isinstance(value, dict) for value in (actor_state, critic_state, ppo_state, bc_state)):
        raise ValueError("resume checkpoint is missing model or optimizer state")
    bundle.actor_model.load_state_dict(cast(dict[str, Tensor], actor_state))
    bundle.critic_model.load_state_dict(cast(dict[str, Tensor], critic_state))
    ppo_optimizer.load_state_dict(cast(dict[str, object], ppo_state))
    bc_optimizer.load_state_dict(cast(dict[str, object], bc_state))
    return payload


def configuration_digest(config: TrainingConfig) -> str:
    payload = config.model_dump(
        mode="json",
        exclude={
            "checkpoint_lineage",
            "configuration_digest",
            "output_root",
            "resume_checkpoint",
            "resume_provenance",
            "source_commit",
            "source_dirty",
            "source_tree_digest",
        },
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def source_commit() -> str:
    return _git_sha()


def source_fingerprint() -> dict[str, object]:
    workspace = Path(__file__).resolve().parents[3]
    commit = source_commit()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=workspace,
        check=False,
        capture_output=True,
        timeout=10,
    )
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=workspace,
        check=False,
        capture_output=True,
        timeout=10,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=workspace,
        check=False,
        capture_output=True,
        timeout=10,
    )
    if status.returncode or diff.returncode or untracked.returncode:
        return {"commit": commit, "dirty": True, "tree_digest": "unknown"}
    digest = hashlib.sha256()
    digest.update(diff.stdout)
    for raw_path in sorted(item for item in untracked.stdout.split(b"\0") if item):
        relative = os.fsdecode(raw_path)
        path = workspace / relative
        digest.update(raw_path)
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "commit": commit,
        "dirty": bool(status.stdout),
        "tree_digest": digest.hexdigest(),
    }


def environment_fingerprint() -> dict[str, object]:
    packages = {}
    for name in (
        "gymnasium",
        "numpy",
        "ortools",
        "pettingzoo",
        "pydantic",
        "tensordict",
        "torch",
        "torchrl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


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
        return cast(Tensor, bundle.critic_model(state.unsqueeze(0)).squeeze(0))
    return cast(
        Tensor,
        bundle.critic_model(
            observations["self"].unsqueeze(0),
            observations["modules"].unsqueeze(0),
        ).squeeze(0),
    )


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
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
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        timeout=5,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, object]) -> None:
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu = state.get("torch_cpu")
    torch_cuda = state.get("torch_cuda")
    if python_state is None or numpy_state is None or not isinstance(torch_cpu, Tensor):
        raise ValueError("resume checkpoint has incomplete RNG state")
    random.setstate(cast(tuple[object, ...], python_state))
    np.random.set_state(cast(tuple[str, np.ndarray, int, int, float], numpy_state))
    torch.set_rng_state(torch_cpu.cpu())
    if torch.cuda.is_available() and isinstance(torch_cuda, list):
        torch.cuda.set_rng_state_all(
            [item.cpu() for item in torch_cuda if isinstance(item, Tensor)]
        )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _design_digest(design: HouseDesign) -> str:
    serialized = _canonical_json(design.model_dump(mode="json"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_curve(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0]) if rows else ["update", "transitions"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _notify(callback, event: str, transitions: int, payload: Mapping[str, object]) -> None:
    if callback:
        callback({"event": event, "transitions": transitions, **payload})


def _checkpoint_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"resume checkpoint field {key!r} must be an integer")
    return value


def defaultdict_tensor_lists() -> dict[str, list[Tensor]]:
    return {
        "self": [],
        "robots": [],
        "modules": [],
        "dependencies": [],
        "action_mask": [],
        "actions": [],
    }
