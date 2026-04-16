import json
import importlib.util
from pathlib import Path

import matplotlib
import pytest

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.gpu import inspect_gpu_runtime
from embodied_skill_composer.assembly.mujoco_backend import MuJoCoAssemblyBackend
from embodied_skill_composer.assembly.models import AssemblyRuntimeProfile, AssemblyScenarioConfig, BeamTask

matplotlib.use("Agg")

from embodied_skill_composer.assembly.visualizer import render_playback_frames, render_summary_figure


mujoco_available = importlib.util.find_spec("mujoco") is not None
torch_available = importlib.util.find_spec("torch") is not None


def build_default_assembly_config() -> AssemblyScenarioConfig:
    return AssemblyScenarioConfig(
        grid_size=12,
        max_steps=120,
        agent_starts=[(0, 2), (0, 3)],
        beams=[
            BeamTask(
                name="beam_alpha",
                pickup_left=(2, 2),
                pickup_right=(2, 3),
                assembly_left=(8, 7),
                assembly_right=(8, 8),
            ),
            BeamTask(
                name="beam_beta",
                pickup_left=(2, 6),
                pickup_right=(2, 7),
                assembly_left=(9, 7),
                assembly_right=(9, 8),
            ),
        ],
    )


def test_local_backend_status_reports_ready() -> None:
    backend = build_assembly_backend(
        config=build_default_assembly_config(),
        runtime_profile=AssemblyRuntimeProfile(name="local_dev", backend="local_sandbox"),
        seed=7,
    )
    status = backend.get_backend_status()
    assert status.backend_name == "local_sandbox"
    assert status.is_ready is True
    assert status.readiness_notes


def test_isaac_backend_status_reports_assumptions() -> None:
    backend = build_assembly_backend(
        config=build_default_assembly_config(),
        runtime_profile=AssemblyRuntimeProfile(
            name="isaac_gpu",
            backend="isaac_lab",
            device="cuda",
            requires_linux=True,
            requires_nvidia_gpu=True,
            notes="Planned Linux-only profile.",
        ),
        seed=7,
    )
    status = backend.get_backend_status()
    assert status.backend_name == "isaac_lab"
    assert status.is_ready is False
    joined_notes = " ".join(status.readiness_notes).lower()
    assert "linux" in joined_notes
    assert "nvidia" in joined_notes or "cuda" in joined_notes


@pytest.mark.skipif(not mujoco_available, reason="MuJoCo optional dependency is not installed")
def test_mujoco_backend_factory_and_scripted_episode(tmp_path: Path) -> None:
    backend = build_assembly_backend(
        config=build_default_assembly_config(),
        runtime_profile=AssemblyRuntimeProfile(name="mujoco_local", backend="mujoco_local", device="cpu"),
        seed=7,
    )
    assert isinstance(backend, MuJoCoAssemblyBackend)
    assert backend.get_backend_status().is_ready is True

    backend.reset(seed=7)
    done = False
    while not done:
        result = backend.execute_team_option(backend.scripted_team_option())
        done = result.done

    artifact = backend.build_artifact(policy_mode="scripted")
    diagnostics = backend.get_option_episode_diagnostics()
    recording = backend.record_episode(tmp_path / "mujoco_scripted.mp4", diagnostics=diagnostics, width=640, height=480)

    assert artifact.metrics.success is True
    assert artifact.metrics.beams_installed == 2
    assert diagnostics["backend"] == "mujoco_local"
    assert diagnostics["selected_options"]
    assert recording.exists()


@pytest.mark.skipif(
    not (mujoco_available and torch_available and (Path(__file__).resolve().parents[1] / "logs" / "assembly_options.pt").exists()),
    reason="MuJoCo, torch, and the learned options checkpoint are required",
)
def test_learned_policy_completes_mujoco_episode() -> None:
    import torch

    from embodied_skill_composer.assembly.options_trainer import HierarchicalOptionTrainer
    from embodied_skill_composer.assembly.runtime import load_assembly_scenario, load_training_config

    workspace = Path(__file__).resolve().parents[1]
    config = load_assembly_scenario(workspace / "configs" / "assembly_env.yaml")
    training = load_training_config(workspace / "configs" / "assembly_training.yaml")
    backend = build_assembly_backend(
        config=config,
        runtime_profile=AssemblyRuntimeProfile(name="mujoco_local", backend="mujoco_local", device="cuda"),
        seed=training.seed,
    )
    trainer = HierarchicalOptionTrainer(backend, training, device="cuda" if torch.cuda.is_available() else "cpu")
    trainer.load_checkpoint(workspace / "logs" / "assembly_options.pt")

    backend.reset(seed=training.seed)
    done = False
    while not done:
        observation = torch.as_tensor(
            backend.get_team_option_observation(), dtype=torch.float32, device=trainer.device
        ).unsqueeze(0)
        mask = torch.as_tensor(trainer._masked_option_array(), dtype=torch.float32, device=trainer.device).unsqueeze(0)
        with torch.no_grad():
            logits = trainer._masked_logits(trainer.actor(observation), mask)
            option = int(torch.argmax(logits, dim=-1).item())
        result = backend.execute_team_option(option, max_primitive_steps=backend.config.option_max_primitive_steps)
        done = result.done

    artifact = backend.build_artifact(policy_mode="learned")
    assert artifact.metrics.success is True
    assert artifact.metrics.beams_installed == 2


def test_visualizer_renders_playback_frames(tmp_path: Path) -> None:
    env = build_assembly_backend(
        config=build_default_assembly_config(),
        runtime_profile=AssemblyRuntimeProfile(name="local_dev", backend="local_sandbox"),
        seed=7,
    )
    env.reset(seed=7)
    done = False
    while not done:
        result = env.execute_team_option(env.scripted_team_option())
        done = result.done
    diagnostics = env.get_option_episode_diagnostics()

    frame_paths = render_playback_frames(build_default_assembly_config(), diagnostics, tmp_path / "frames")
    summary_path = render_summary_figure(build_default_assembly_config(), diagnostics, tmp_path / "summary.png")

    assert frame_paths
    assert all(path.exists() for path in frame_paths)
    assert summary_path.exists()
    assert len(frame_paths) == len(diagnostics["state_snapshots"])


def test_gpu_status_shape() -> None:
    status = inspect_gpu_runtime(
        AssemblyRuntimeProfile(
            name="local_gpu",
            backend="local_sandbox",
            device="cuda",
            requires_nvidia_gpu=True,
        )
    )
    payload = status.model_dump(mode="json")
    assert payload["runtime_profile"] == "local_gpu"
    assert payload["backend"] == "local_sandbox"
    assert "torch_installed" in payload
    assert "cuda_available" in payload
    assert "tensor_allocation_ok" in payload
    assert isinstance(payload["notes"], list)


def test_vscode_tasks_and_docs_stay_in_sync() -> None:
    workspace = Path(__file__).resolve().parents[1]
    tasks_payload = json.loads((workspace / ".vscode" / "tasks.json").read_text(encoding="utf-8"))
    windows_doc = (workspace / "docs" / "setup" / "windows-vscode.md").read_text(encoding="utf-8")
    readme = (workspace / "README.md").read_text(encoding="utf-8")

    tasks = {task["label"]: task for task in tasks_payload["tasks"]}
    required_commands = {
        "Assembly Benchmark": "scripts\\benchmark_assembly_policies.py --runtime-profile configs\\assembly_profiles\\local_dev.yaml --episodes 3",
        "Hierarchical Eval": "scripts\\eval_assembly_options.py --policy learned --runtime-profile configs\\assembly_profiles\\local_dev.yaml --episodes 3",
        "Low-Level Eval": "scripts\\eval_assembly_policy.py --policy learned --runtime-profile configs\\assembly_profiles\\local_dev.yaml --episodes 3",
        "Pytest": "-m pytest -q --basetemp .pytest_tmp",
    }

    for label, command_fragment in required_commands.items():
        task = tasks[label]
        rendered_args = " ".join(task["args"])
        assert command_fragment in rendered_args
        assert command_fragment in windows_doc or command_fragment in readme
