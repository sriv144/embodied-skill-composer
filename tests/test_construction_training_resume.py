from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchrl")

from embodied_skill_composer.construction.marl_env_v1 import (  # noqa: E402
    TemporalConstructionCoordinationEnv,
)
from embodied_skill_composer.construction.policy import (  # noqa: E402
    TorchRLPolicyBundle,
    build_torchrl_policy,
)
from embodied_skill_composer.construction.training import (  # noqa: E402
    TrainingConfig,
    _restore_rng_state,
    configuration_digest,
    load_training_checkpoint,
    save_training_checkpoint,
    train_swarm_policy,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402


CONFIGURATION_DIGEST = "configuration-v1"
DESIGN_DIGEST = "design-v1"
SOURCE_COMMIT = "0123456789abcdef"
SOURCE_TREE_DIGEST = "source-tree-v1"


def test_configuration_digest_is_stable_and_excludes_resume_metadata(
    tmp_path: Path,
) -> None:
    config = TrainingConfig.for_profile("unit", algorithm="mappo", seed=19)
    baseline = configuration_digest(config)

    round_tripped = TrainingConfig.model_validate(config.model_dump(mode="python"))
    metadata_changed = config.model_copy(
        update={
            "checkpoint_lineage": ["checkpoint-10.pt", "checkpoint-25.pt"],
            "configuration_digest": "already-recorded",
            "output_root": tmp_path / "different-output-root",
            "resume_checkpoint": tmp_path / "checkpoint-latest.pt",
            "resume_provenance": {"attempt": 3, "reason": "worker_restart"},
            "source_commit": "fedcba9876543210",
        }
    )
    substantive_change = config.model_copy(
        update={"learning_rate": config.learning_rate * 2.0}
    )

    assert configuration_digest(round_tripped) == baseline
    assert configuration_digest(metadata_changed) == baseline
    assert configuration_digest(substantive_change) != baseline


def test_fresh_worker_refuses_environment_changed_after_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodied_skill_composer.construction.training as training_module

    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    config = TrainingConfig.for_profile("unit", seed=17)
    config.environment_fingerprint = {"python": "queued-environment"}
    monkeypatch.setattr(
        training_module,
        "environment_fingerprint",
        lambda: {"python": "worker-environment"},
    )

    with pytest.raises(ValueError, match="environment fingerprint changed"):
        train_swarm_policy(design, config)


def test_fresh_worker_refuses_source_changed_after_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodied_skill_composer.construction.training as training_module

    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    config = TrainingConfig.for_profile("unit", seed=18)
    config.environment_fingerprint = {"python": "same-environment"}
    config.source_commit = "queued-commit"
    config.source_dirty = False
    config.source_tree_digest = "queued-tree"
    monkeypatch.setattr(
        training_module,
        "environment_fingerprint",
        lambda: {"python": "same-environment"},
    )
    monkeypatch.setattr(
        training_module,
        "source_fingerprint",
        lambda: {
            "commit": "worker-commit",
            "dirty": False,
            "tree_digest": "worker-tree",
        },
    )

    with pytest.raises(ValueError, match="source worktree fingerprint changed"):
        train_swarm_policy(design, config)


def test_checkpoint_round_trip_restores_training_and_rng_state(
    tmp_path: Path,
) -> None:
    config = TrainingConfig.for_profile("unit", algorithm="mappo", seed=23)
    config.source_tree_digest = SOURCE_TREE_DIGEST
    source_bundle = build_torchrl_policy("mappo", hidden_dim=config.hidden_dim)
    source_ppo = _optimizer_for_bundle(source_bundle)
    source_bc = torch.optim.Adam(
        source_bundle.actor_model.parameters(),
        lr=config.learning_rate,
    )
    _prime_optimizer(
        source_ppo,
        [
            *source_bundle.actor_model.parameters(),
            *source_bundle.critic_model.parameters(),
        ],
    )
    _prime_optimizer(source_bc, list(source_bundle.actor_model.parameters()))

    actor_state = _clone_tensor_mapping(source_bundle.actor_model.state_dict())
    critic_state = _clone_tensor_mapping(source_bundle.critic_model.state_dict())
    ppo_state = copy.deepcopy(source_ppo.state_dict())
    bc_state = copy.deepcopy(source_bc.state_dict())
    curve_rows: list[dict[str, float | int]] = [
        {"update": 1, "transitions": 8, "completion_rate": 0.5},
        {"update": 2, "transitions": 16, "completion_rate": 0.75},
    ]

    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoints" / "checkpoint-latest.pt",
        bundle=source_bundle,
        ppo_optimizer=source_ppo,
        bc_optimizer=source_bc,
        config=config,
        configuration_digest_value=CONFIGURATION_DIGEST,
        design_digest=DESIGN_DIGEST,
        source_commit_value=SOURCE_COMMIT,
        transitions=16,
        updates=2,
        episode_cursor=5,
        bc_epoch=1,
        curve_rows=curve_rows,
        checkpoint_lineage=["checkpoint-10.pt"],
    )
    expected_rng_values = (
        random.random(),
        float(np.random.random()),
        torch.rand(4),
    )

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored_bundle = build_torchrl_policy("mappo", hidden_dim=config.hidden_dim)
    restored_ppo = _optimizer_for_bundle(restored_bundle)
    restored_bc = torch.optim.Adam(
        restored_bundle.actor_model.parameters(),
        lr=config.learning_rate,
    )
    payload = load_training_checkpoint(
        checkpoint,
        bundle=restored_bundle,
        ppo_optimizer=restored_ppo,
        bc_optimizer=restored_bc,
        expected_configuration_digest=CONFIGURATION_DIGEST,
        expected_design_digest=DESIGN_DIGEST,
        expected_source_commit=SOURCE_COMMIT,
        expected_source_dirty=False,
        expected_source_tree_digest=SOURCE_TREE_DIGEST,
        device="cpu",
    )

    _assert_tensor_mapping_equal(
        actor_state,
        restored_bundle.actor_model.state_dict(),
    )
    _assert_tensor_mapping_equal(
        critic_state,
        restored_bundle.critic_model.state_dict(),
    )
    _assert_nested_equal(ppo_state, restored_ppo.state_dict())
    _assert_nested_equal(bc_state, restored_bc.state_dict())
    assert payload["transitions"] == 16
    assert payload["updates"] == 2
    assert payload["episode_cursor"] == 5
    assert payload["bc_epoch"] == 1
    assert payload["curve_rows"] == curve_rows
    assert payload["checkpoint_lineage"] == ["checkpoint-10.pt"]
    assert payload["config"] == config.model_dump(mode="json")

    rng_state = payload["rng_state"]
    assert isinstance(rng_state, dict)
    assert set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    _restore_rng_state(rng_state)
    assert random.random() == expected_rng_values[0]
    assert float(np.random.random()) == expected_rng_values[1]
    torch.testing.assert_close(torch.rand(4), expected_rng_values[2])


def test_checkpoint_refuses_incompatible_resume_provenance(tmp_path: Path) -> None:
    config = TrainingConfig.for_profile("unit", algorithm="mappo", seed=29)
    config.source_tree_digest = SOURCE_TREE_DIGEST
    source_bundle = build_torchrl_policy("mappo", hidden_dim=config.hidden_dim)
    source_ppo = _optimizer_for_bundle(source_bundle)
    source_bc = torch.optim.Adam(
        source_bundle.actor_model.parameters(),
        lr=config.learning_rate,
    )
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        bundle=source_bundle,
        ppo_optimizer=source_ppo,
        bc_optimizer=source_bc,
        config=config,
        configuration_digest_value=CONFIGURATION_DIGEST,
        design_digest=DESIGN_DIGEST,
        source_commit_value=SOURCE_COMMIT,
        transitions=0,
        updates=0,
        episode_cursor=0,
        bc_epoch=0,
        curve_rows=[],
        checkpoint_lineage=[],
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    schema_checkpoint = tmp_path / "checkpoint-schema-mismatch.pt"
    torch.save({**checkpoint_payload, "schema_version": 999}, schema_checkpoint)
    environment_checkpoint = tmp_path / "checkpoint-environment-mismatch.pt"
    torch.save(
        {**checkpoint_payload, "environment_schema": "different_environment"},
        environment_checkpoint,
    )
    cases = [
        (
            "configuration_digest",
            checkpoint,
            "configuration-v2",
            DESIGN_DIGEST,
            SOURCE_COMMIT,
        ),
        (
            "design_digest",
            checkpoint,
            CONFIGURATION_DIGEST,
            "design-v2",
            SOURCE_COMMIT,
        ),
        (
            "source_commit",
            checkpoint,
            CONFIGURATION_DIGEST,
            DESIGN_DIGEST,
            "different-commit",
        ),
        (
            "schema_version",
            schema_checkpoint,
            CONFIGURATION_DIGEST,
            DESIGN_DIGEST,
            SOURCE_COMMIT,
        ),
        (
            "environment_schema",
            environment_checkpoint,
            CONFIGURATION_DIGEST,
            DESIGN_DIGEST,
            SOURCE_COMMIT,
        ),
    ]

    for (
        mismatch_name,
        candidate,
        expected_config,
        expected_design,
        expected_commit,
    ) in cases:
        restored_bundle = build_torchrl_policy(
            "mappo",
            hidden_dim=config.hidden_dim,
        )
        restored_ppo = _optimizer_for_bundle(restored_bundle)
        restored_bc = torch.optim.Adam(
            restored_bundle.actor_model.parameters(),
            lr=config.learning_rate,
        )
        with pytest.raises(ValueError, match=mismatch_name):
            load_training_checkpoint(
                candidate,
                bundle=restored_bundle,
                ppo_optimizer=restored_ppo,
                bc_optimizer=restored_bc,
                expected_configuration_digest=expected_config,
                expected_design_digest=expected_design,
                expected_source_commit=expected_commit,
                expected_source_dirty=False,
                expected_source_tree_digest=SOURCE_TREE_DIGEST,
                device="cpu",
            )

    assert checkpoint_payload["environment_schema"] == (
        TemporalConstructionCoordinationEnv.metadata["name"]
    )


def test_training_interrupts_then_resumes_from_exact_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embodied_skill_composer.construction.training as training_module

    design = load_house_design(
        Path(__file__).resolve().parents[1] / "configs" / "construction" / "cottage_v1.yaml"
    )
    config = TrainingConfig.for_profile("unit", algorithm="mappo", seed=31)
    config.output_root = tmp_path / "training"
    config.transitions = 32
    config.rollout_decisions = 4
    config.expert_episodes = 0
    config.behavior_clone_epochs = 0
    config.checkpoint_fractions = [0.5, 1.0]
    config.source_commit = "test-source-commit"

    def fake_export(_model, path: Path, *, device: str = "cpu") -> Path:
        del device
        path.write_bytes(b"fixture-onnx")
        return path

    monkeypatch.setattr(training_module, "export_actor_onnx", fake_export)
    monkeypatch.setattr(training_module, "source_commit", lambda: "test-source-commit")
    def interrupt_after_first_update() -> bool:
        return any(
            config.output_root.glob("*/checkpoints/checkpoint_050pct.pt")
        )

    with pytest.raises(RuntimeError, match="training cancelled"):
        train_swarm_policy(design, config, cancel_check=interrupt_after_first_update)

    run_dirs = list(config.output_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    latest = run_dir / "checkpoints" / "latest.pt"
    first_payload = torch.load(latest, map_location="cpu", weights_only=False)
    assert first_payload["transitions"] == 16
    assert (run_dir / "checkpoints" / "checkpoint_050pct.pt").is_file()

    config.resume_checkpoint = latest
    config.resume_provenance = {"reason": "test-interruption"}
    artifacts = train_swarm_policy(design, config, cancel_check=lambda: False)

    final_payload = torch.load(latest, map_location="cpu", weights_only=False)
    manifest = training_module.PolicyManifest.model_validate_json(
        artifacts.policy_manifest_path.read_text(encoding="utf-8")
    )
    assert artifacts.transitions == 32
    assert artifacts.updates == 2
    assert final_payload["transitions"] == 32
    assert final_payload["updates"] == 2
    assert (run_dir / "checkpoints" / "checkpoint_100pct.pt").is_file()
    assert str(latest) in manifest.checkpoint_lineage
    assert manifest.resume_provenance == {"reason": "test-interruption"}
    assert manifest.configuration_digest == configuration_digest(config)


def _optimizer_for_bundle(
    bundle: TorchRLPolicyBundle,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            *bundle.actor_model.parameters(),
            *bundle.critic_model.parameters(),
        ],
        lr=3e-4,
    )


def _prime_optimizer(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = torch.stack([parameter.square().mean() for parameter in parameters]).sum()
    loss.backward()
    optimizer.step()


def _clone_tensor_mapping(
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in values.items()}


def _assert_tensor_mapping_equal(
    expected: dict[str, torch.Tensor],
    actual: dict[str, torch.Tensor],
) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, rtol=0, atol=0)


def _assert_nested_equal(expected: Any, actual: Any) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_nested_equal(expected_value, actual[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for expected_value, actual_value in zip(expected, actual, strict=True):
            _assert_nested_equal(expected_value, actual_value)
        return
    assert actual == expected
