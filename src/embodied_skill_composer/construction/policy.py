from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch import Tensor, nn
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import MaskedCategorical

from embodied_skill_composer.construction.marl_env_v1 import (
    FLEET_SIZE,
    MAX_MODULES,
    MODULE_FEATURES,
    ROBOT_FEATURES,
)


SELF_FEATURES = ROBOT_FEATURES + 1
GLOBAL_STATE_FEATURES = (
    MAX_MODULES * MODULE_FEATURES
    + MAX_MODULES * MAX_MODULES
    + FLEET_SIZE * ROBOT_FEATURES
    + 2
)


class SwarmPointerActor(nn.Module):
    """Parameter-shared robot policy with dependency-aware module attention."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_encoder = nn.Sequential(
            nn.Linear(SELF_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.fleet_encoder = nn.Sequential(
            nn.Linear(ROBOT_FEATURES, hidden_dim),
            nn.SiLU(),
        )
        self.module_encoder = nn.Sequential(
            nn.Linear(MODULE_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.dependency_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )
        self.query = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        self_features: Tensor,
        fleet_features: Tensor,
        module_features: Tensor,
        dependencies: Tensor,
        action_mask: Tensor,
    ) -> Tensor:
        own = self.self_encoder(self_features)
        fleet_context = self.fleet_encoder(fleet_features).mean(dim=-2)
        modules = self.module_encoder(module_features)
        dependency_count = dependencies.sum(dim=-1, keepdim=True).clamp_min(1.0)
        dependency_context = torch.matmul(dependencies.float(), modules) / dependency_count
        modules = self.dependency_projection(torch.cat([modules, dependency_context], dim=-1))
        query = self.query(torch.cat([own, fleet_context], dim=-1))
        job_logits = torch.einsum("...ah,...amh->...am", query, self.key(modules))
        job_logits = job_logits / sqrt(self.hidden_dim)
        wait_logit = self.wait_head(torch.cat([own, fleet_context], dim=-1))
        logits = torch.cat([wait_logit, job_logits], dim=-1)
        return logits.masked_fill(~action_mask.bool(), -1e9)


class CentralValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(GLOBAL_STATE_FEATURES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: Tensor) -> Tensor:
        value = self.network(state)
        return value.unsqueeze(-2).expand(*value.shape[:-1], FLEET_SIZE, 1)


class IndependentValueNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.self_encoder = nn.Sequential(nn.Linear(SELF_FEATURES, hidden_dim), nn.SiLU())
        self.module_encoder = nn.Sequential(
            nn.Linear(MODULE_FEATURES, hidden_dim),
            nn.SiLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, self_features: Tensor, module_features: Tensor) -> Tensor:
        own = self.self_encoder(self_features)
        module_mask = module_features[..., 0:1]
        module_count = module_mask.sum(dim=-2).clamp_min(1.0)
        module_context = (self.module_encoder(module_features) * module_mask).sum(dim=-2)
        module_context = module_context / module_count
        return self.value(torch.cat([own, module_context], dim=-1))


@dataclass
class TorchRLPolicyBundle:
    actor_model: SwarmPointerActor
    critic_model: nn.Module
    actor: ProbabilisticActor
    critic: TensorDictModule
    algorithm: Literal["mappo", "ippo"]

    def to(self, device: torch.device | str) -> "TorchRLPolicyBundle":
        self.actor.to(device)
        self.critic.to(device)
        return self


def build_torchrl_policy(
    algorithm: Literal["mappo", "ippo"],
    *,
    hidden_dim: int = 128,
) -> TorchRLPolicyBundle:
    actor_model = SwarmPointerActor(hidden_dim=hidden_dim)
    actor_module = TensorDictModule(
        actor_model,
        in_keys=[
            ("agents", "self"),
            ("agents", "robots"),
            ("agents", "modules"),
            ("agents", "dependencies"),
            ("agents", "action_mask"),
        ],
        out_keys=[("agents", "logits")],
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys={
            "logits": ("agents", "logits"),
            "mask": ("agents", "action_mask"),
        },
        out_keys=[("agents", "action")],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    if algorithm == "mappo":
        critic_model: nn.Module = CentralValueNetwork(hidden_dim=hidden_dim * 2)
        critic = TensorDictModule(
            critic_model,
            in_keys=["state"],
            out_keys=[("agents", "state_value")],
        )
    else:
        critic_model = IndependentValueNetwork(hidden_dim=hidden_dim)
        critic = TensorDictModule(
            critic_model,
            in_keys=[("agents", "self"), ("agents", "modules")],
            out_keys=[("agents", "state_value")],
        )
    return TorchRLPolicyBundle(
        actor_model=actor_model,
        critic_model=critic_model,
        actor=actor,
        critic=critic,
        algorithm=algorithm,
    )


def observations_to_tensors(
    observations: dict[str, dict[str, np.ndarray]],
    agent_ids: list[str],
    *,
    device: torch.device | str,
) -> dict[str, Tensor]:
    return {
        "self": torch.as_tensor(
            np.stack([observations[agent]["self"] for agent in agent_ids]),
            dtype=torch.float32,
            device=device,
        ),
        "robots": torch.as_tensor(
            np.stack([observations[agent]["robots"] for agent in agent_ids]),
            dtype=torch.float32,
            device=device,
        ),
        "modules": torch.as_tensor(
            np.stack([observations[agent]["modules"] for agent in agent_ids]),
            dtype=torch.float32,
            device=device,
        ),
        "dependencies": torch.as_tensor(
            np.stack([observations[agent]["dependencies"] for agent in agent_ids]),
            dtype=torch.float32,
            device=device,
        ),
        "action_mask": torch.as_tensor(
            np.stack([observations[agent]["action_mask"] for agent in agent_ids]),
            dtype=torch.bool,
            device=device,
        ),
    }


@torch.no_grad()
def policy_actions(
    model: SwarmPointerActor,
    observations: dict[str, dict[str, np.ndarray]],
    agent_ids: list[str],
    *,
    device: torch.device | str,
    deterministic: bool = True,
) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    tensors = observations_to_tensors(observations, agent_ids, device=device)
    logits = model(
        tensors["self"].unsqueeze(0),
        tensors["robots"].unsqueeze(0),
        tensors["modules"].unsqueeze(0),
        tensors["dependencies"].unsqueeze(0),
        tensors["action_mask"].unsqueeze(0),
    ).squeeze(0)
    distribution = Categorical(logits=logits)
    selected = logits.argmax(dim=-1) if deterministic else distribution.sample()
    probabilities = distribution.probs
    entropy = distribution.entropy()
    actions = {agent: int(selected[index].item()) for index, agent in enumerate(agent_ids)}
    diagnostics = {
        agent: {
            "selected_action": int(selected[index].item()),
            "selected_probability": float(probabilities[index, selected[index]].item()),
            "uncertainty": float(entropy[index].item()),
            "action_probabilities": probabilities[index].cpu().tolist(),
        }
        for index, agent in enumerate(agent_ids)
    }
    return actions, diagnostics


def save_policy_checkpoint(
    bundle: TorchRLPolicyBundle,
    path: Path,
    *,
    metadata: dict[str, object],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": bundle.algorithm,
            "actor_state_dict": bundle.actor_model.state_dict(),
            "critic_state_dict": bundle.critic_model.state_dict(),
            "metadata": metadata,
        },
        path,
    )
    return file_sha256(path)


def load_policy_checkpoint(
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> TorchRLPolicyBundle:
    payload = torch.load(path, map_location=device, weights_only=True)
    hidden_dim = int(payload.get("metadata", {}).get("hidden_dim", 128))
    bundle = build_torchrl_policy(payload["algorithm"], hidden_dim=hidden_dim)
    bundle.actor_model.load_state_dict(payload["actor_state_dict"])
    bundle.critic_model.load_state_dict(payload["critic_state_dict"])
    return bundle.to(device)


def export_actor_onnx(
    model: SwarmPointerActor,
    path: Path,
    *,
    device: torch.device | str = "cpu",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    sample_inputs = (
        torch.zeros(1, FLEET_SIZE, SELF_FEATURES, device=device),
        torch.zeros(1, FLEET_SIZE, FLEET_SIZE, ROBOT_FEATURES, device=device),
        torch.zeros(1, FLEET_SIZE, MAX_MODULES, MODULE_FEATURES, device=device),
        torch.zeros(1, FLEET_SIZE, MAX_MODULES, MAX_MODULES, device=device),
        torch.ones(1, FLEET_SIZE, MAX_MODULES + 1, dtype=torch.bool, device=device),
    )
    batch_dimension = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        model,
        sample_inputs,
        path,
        input_names=["self", "robots", "modules", "dependencies", "action_mask"],
        output_names=["masked_logits"],
        dynamic_shapes=tuple({0: batch_dimension} for _ in sample_inputs),
        opset_version=18,
        dynamo=True,
    )
    return path


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
