from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from embodied_skill_composer.copilot import agent, cli, reports, runner
from embodied_skill_composer.copilot.registry import CopilotRegistry
from embodied_skill_composer.core.models import (
    ObjectState,
    RobotState,
    TaskSpec,
    TaskType,
    WorldState,
    ZoneState,
)
from embodied_skill_composer.core.planner import RuleBasedPlanner
from embodied_skill_composer.rl.grasp_policy import (
    GraspPolicy,
    default_grasp_policy,
    load_grasp_policy,
    save_grasp_policy,
)
from embodied_skill_composer.rl.trainer import GraspPolicyTrainer
from embodied_skill_composer.tasks.catalog import load_tasks


def _copilot_result(tmp_path: Path, *, exit_code: int = 3) -> runner.CopilotRunResult:
    return runner.CopilotRunResult(
        run_id="run-coverage",
        run_dir=tmp_path,
        report_path=tmp_path / "report.md",
        exit_code=exit_code,
        summary={"status": "tested"},
    )


def _install_fake_agents(
    monkeypatch: pytest.MonkeyPatch,
    tool_outputs: dict[str, str],
    run_state: dict[str, object],
    *,
    fail_run: bool = False,
) -> None:
    fake_agents = ModuleType("agents")

    class FakeAgent:
        def __init__(self, *, name: str, instructions: str, tools: list[object]) -> None:
            run_state["agent_name"] = name
            run_state["instructions"] = instructions
            self.tools = tools

    class FakeRunConfig:
        def __init__(self, *, model: str) -> None:
            self.model = model

    class FakeRunner:
        @staticmethod
        def run_sync(
            fake_agent: FakeAgent,
            prompt: str,
            *,
            run_config: FakeRunConfig | None,
        ) -> SimpleNamespace:
            if fail_run:
                raise ValueError("synthetic runner failure")
            run_state["prompt"] = prompt
            run_state["run_config"] = run_config
            arguments = {
                "list_recent_runs": {"limit": 2},
                "benchmark": {"episodes": 1},
                "nvidia_check": {},
                "train_options": {},
                "train_marl": {},
                "sweep": {"scenarios": 2, "seeds": "1,2"},
            }
            for tool in fake_agent.tools:
                name = tool.__name__
                tool_outputs[name] = tool(**arguments[name])
            return SimpleNamespace(final_output="fake-agent-output")

    fake_agents.Agent = FakeAgent
    fake_agents.RunConfig = FakeRunConfig
    fake_agents.Runner = FakeRunner
    fake_agents.function_tool = lambda function: function
    monkeypatch.setitem(sys.modules, "agents", fake_agents)


def test_dotenv_loader_honors_existing_key_and_parses_quotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent.load_dotenv_if_needed(tmp_path / "missing.env")
    assert "OPENAI_API_KEY" not in agent.os.environ

    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\nIGNORED=value\ninvalid-line\nOPENAI_API_KEY='test-key'\n",
        encoding="utf-8",
    )
    agent.load_dotenv_if_needed(dotenv)
    assert agent.os.environ["OPENAI_API_KEY"] == "test-key"

    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    dotenv.write_text("OPENAI_API_KEY=replacement\n", encoding="utf-8")
    agent.load_dotenv_if_needed(dotenv)
    assert agent.os.environ["OPENAI_API_KEY"] == "existing-key"


def test_agent_tools_report_results_and_confirmation_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _copilot_result(tmp_path)
    registry = SimpleNamespace(recent_runs=lambda limit: [{"id": f"recent-{limit}"}])
    monkeypatch.setattr(agent, "load_dotenv_if_needed", lambda _path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(agent, "default_registry", lambda: registry)
    monkeypatch.setattr(agent, "run_benchmark", lambda **_kwargs: result)
    monkeypatch.setattr(agent, "run_nvidia_readiness_check", lambda: {"ready": False})

    def blocked(**_kwargs: object) -> runner.CopilotRunResult:
        raise runner.ConfirmationRequired("training blocked")

    monkeypatch.setattr(agent, "run_train_options", blocked)
    monkeypatch.setattr(agent, "run_train_marl", blocked)
    monkeypatch.setattr(agent, "run_sweep", blocked)
    outputs: dict[str, str] = {}
    state: dict[str, object] = {}
    _install_fake_agents(monkeypatch, outputs, state)

    response = agent.run_agent_prompt(
        "summarize the lab",
        model="test-model",
        allow_training=False,
    )

    assert response == "fake-agent-output"
    assert json.loads(outputs["list_recent_runs"])[0]["id"] == "recent-2"
    assert json.loads(outputs["benchmark"])["run_id"] == result.run_id
    assert json.loads(outputs["nvidia_check"])["ready"] is False
    assert outputs["train_options"] == "training blocked"
    assert outputs["train_marl"] == "training blocked"
    assert outputs["sweep"] == "training blocked"
    assert state["run_config"].model == "test-model"


def test_agent_training_tools_return_artifacts_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _copilot_result(tmp_path, exit_code=0)
    monkeypatch.setattr(agent, "load_dotenv_if_needed", lambda _path: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        agent,
        "default_registry",
        lambda: SimpleNamespace(recent_runs=lambda limit: []),
    )
    monkeypatch.setattr(agent, "run_benchmark", lambda **_kwargs: result)
    monkeypatch.setattr(agent, "run_nvidia_readiness_check", lambda: {"ready": True})
    monkeypatch.setattr(agent, "run_train_options", lambda **_kwargs: result)
    monkeypatch.setattr(agent, "run_train_marl", lambda **_kwargs: result)
    monkeypatch.setattr(agent, "run_sweep", lambda **_kwargs: result)
    outputs: dict[str, str] = {}
    state: dict[str, object] = {}
    _install_fake_agents(monkeypatch, outputs, state)

    assert agent.run_agent_prompt("run approved tools", allow_training=True) == "fake-agent-output"
    assert state["run_config"] is None
    for name in ("train_options", "train_marl", "sweep"):
        assert json.loads(outputs[name])["exit_code"] == 0


def test_agent_reports_missing_key_sdk_and_runner_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "load_dotenv_if_needed", lambda _path: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        agent.run_agent_prompt("hello")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "agents", None)
    with pytest.raises(RuntimeError, match="Agents SDK"):
        agent.run_agent_prompt("hello")

    outputs: dict[str, str] = {}
    state: dict[str, object] = {}
    _install_fake_agents(monkeypatch, outputs, state, fail_run=True)
    monkeypatch.setattr(
        agent,
        "default_registry",
        lambda: SimpleNamespace(recent_runs=lambda limit: []),
    )
    with pytest.raises(RuntimeError, match="synthetic runner failure"):
        agent.run_agent_prompt("hello")


@pytest.mark.parametrize(
    ("argv", "target", "expected_args", "expected_kwargs"),
    [
        (["benchmark", "--episodes", "2", "--runtime-profile", "p"], "run_benchmark", (2, "p"), {}),
        (
            ["eval-options", "--policy", "learned", "--episodes", "3", "--runtime-profile", "p"],
            "run_eval_options",
            ("learned", 3, "p"),
            {},
        ),
        (
            ["train-options", "--runtime-profile", "p", "--yes"],
            "run_train_options",
            ("p",),
            {"yes": True},
        ),
        (
            ["train-marl", "--runtime-profile", "p", "--yes"],
            "run_train_marl",
            ("p",),
            {"yes": True},
        ),
        (
            [
                "sweep",
                "--scenarios",
                "2",
                "--seeds",
                "1,2",
                "--beam-count",
                "3",
                "--runtime-profile",
                "p",
                "--yes",
            ],
            "run_sweep",
            (2, "1,2", 3, "p"),
            {"yes": True},
        ),
    ],
)
def test_cli_dispatches_experiment_commands(
    argv: list[str],
    target: str,
    expected_args: tuple[object, ...],
    expected_kwargs: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_run(*args: object, **kwargs: object) -> runner.CopilotRunResult:
        calls.append((args, kwargs))
        return _copilot_result(tmp_path)

    monkeypatch.setattr(cli, target, fake_run)
    monkeypatch.setattr(sys, "argv", ["esc-copilot", *argv])

    assert cli.main() == 3
    assert calls == [(expected_args, expected_kwargs)]
    assert "Run ID: run-coverage" in capsys.readouterr().out


def test_cli_dispatches_ask_nvidia_and_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ask_calls: list[tuple[str, str | None, bool]] = []

    def fake_prompt(prompt: str, *, model: str | None, allow_training: bool) -> str:
        ask_calls.append((prompt, model, allow_training))
        return "answer"

    monkeypatch.setattr(cli, "run_agent_prompt", fake_prompt)
    monkeypatch.setattr(
        sys,
        "argv",
        ["esc-copilot", "ask", "status", "--model", "model-x", "--yes"],
    )
    assert cli.main() == 0
    assert ask_calls == [("status", "model-x", True)]
    assert "answer" in capsys.readouterr().out

    monkeypatch.setattr(cli, "_run_nvidia_check", lambda profile, url: 6)
    monkeypatch.setattr(
        sys,
        "argv",
        ["esc-copilot", "nvidia-check", "--runtime-profile", "gpu.yaml", "--aiq-url", "u"],
    )
    assert cli.main() == 6

    def confirmation(*_args: object, **_kwargs: object) -> runner.CopilotRunResult:
        raise runner.ConfirmationRequired("approve me")

    monkeypatch.setattr(cli, "run_benchmark", confirmation)
    monkeypatch.setattr(sys, "argv", ["esc-copilot", "benchmark"])
    assert cli.main() == 2
    assert "Confirmation required: approve me" in capsys.readouterr().out

    def unavailable(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("offline")

    monkeypatch.setattr(cli, "run_agent_prompt", unavailable)
    monkeypatch.setattr(sys, "argv", ["esc-copilot", "ask", "status"])
    assert cli.main() == 1
    assert "Copilot unavailable: offline" in capsys.readouterr().out

    fake_parser = SimpleNamespace(parse_args=lambda: Namespace(command="unknown"))
    monkeypatch.setattr(cli, "build_parser", lambda: fake_parser)
    with pytest.raises(AssertionError, match="Unhandled command"):
        cli.main()


def test_cli_nvidia_helper_persists_summary_and_registry_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = SimpleNamespace(id="nvidia-run", run_dir=tmp_path)

    class FakeRegistry:
        def __init__(self) -> None:
            self.artifacts: list[tuple[object, ...]] = []
            self.completed: tuple[object, ...] | None = None

        def create_run(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return record

        def add_artifact(self, *args: object) -> None:
            self.artifacts.append(args)

        def complete_run(self, *args: object) -> None:
            self.completed = args

    registry = FakeRegistry()
    summary = {"runtime": {"cuda_available": False}, "aiq": {"reachable": False}}
    monkeypatch.setattr(cli, "default_registry", lambda: registry)
    monkeypatch.setattr(
        cli,
        "run_nvidia_readiness_check",
        lambda profile, *, aiq_url: summary,
    )

    def fake_report(run_dir: Path, *_args: object) -> Path:
        path = run_dir / "report.md"
        path.write_text("report", encoding="utf-8")
        return path

    monkeypatch.setattr(cli, "write_report", fake_report)

    assert cli._run_nvidia_check("gpu.yaml", "http://aiq") == 0
    assert json.loads((tmp_path / "nvidia_readiness.json").read_text())["aiq"] == {
        "reachable": False
    }
    assert len(registry.artifacts) == 2
    assert registry.completed is not None
    assert "Run ID: nvidia-run" in capsys.readouterr().out


def test_runner_confirmation_interactive_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: "YES")
    runner.require_confirmation("training", yes=False)
    runner.require_confirmation("training", yes=True)

    monkeypatch.setattr("builtins.input", lambda _prompt: "no")
    with pytest.raises(runner.ConfirmationRequired, match="not confirmed"):
        runner.require_confirmation("training", yes=False)


def test_runner_builds_all_command_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeRegistry:
        def create_run(
            self,
            kind: str,
            **_kwargs: object,
        ) -> SimpleNamespace:
            run_dir = tmp_path / kind
            run_dir.mkdir(exist_ok=True)
            return SimpleNamespace(id=f"{kind}-run", run_dir=run_dir)

    calls: list[tuple[list[str], str, Path, bool]] = []

    def fake_subprocess(
        _record: object,
        _registry: object,
        command: list[str],
        title: str,
        summary_path: Path,
        parse_stdout_json: bool = False,
    ) -> object:
        calls.append((command, title, summary_path, parse_stdout_json))
        return object()

    registry = FakeRegistry()
    monkeypatch.setattr(runner, "_run_subprocess", fake_subprocess)
    runner.run_eval_options("learned", 2, "profile.yaml", registry=registry)
    runner.run_train_options("profile.yaml", yes=True, registry=registry)
    runner.run_train_marl("profile.yaml", yes=True, registry=registry)
    runner.run_sweep(3, "4,5", 2, "profile.yaml", yes=True, registry=registry)

    scripts = [Path(call[0][1]).name for call in calls]
    assert scripts == [
        "eval_assembly_options.py",
        "train_assembly_options.py",
        "train_assembly_marl.py",
        "run_assembly_experiment_sweep.py",
    ]
    assert calls[0][3] is True
    assert all("profile.yaml" in call[0] for call in calls)


def test_runner_eval_summary_handles_missing_invalid_and_valid_json() -> None:
    missing = runner._extract_eval_stdout_summary("plain log")
    assert "No JSON" in missing["notes"][0]

    invalid = runner._extract_eval_stdout_summary("log\n[\ninvalid")
    assert "Could not parse" in invalid["notes"][0]

    episodes = [
        {
            "artifact": {
                "metrics": {
                    "success": True,
                    "beams_installed": 2,
                    "total_reward": 4.0,
                }
            }
        },
        {"artifact": {"metrics": {"success": False}}},
        "ignored",
    ]
    valid = runner._extract_eval_stdout_summary("progress\n" + json.dumps(episodes, indent=2))
    assert valid["success_rate"] == 0.5
    assert valid["mean_beams_installed"] == 1.0
    assert valid["mean_return"] == 2.0


def test_runner_failed_subprocess_captures_eval_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = CopilotRegistry(tmp_path / "runs.sqlite", tmp_path / "runs")
    record = registry.create_run("eval-options", command=[])
    episodes = [{"artifact": {"metrics": {"success": True, "beams_installed": 1}}}]

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            ["python", "fake.py"],
            4,
            "log\n" + json.dumps(episodes, indent=2),
            "synthetic stderr",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._run_subprocess(
        record,
        registry,
        ["python", "fake.py"],
        "Evaluation",
        record.run_dir / "summary.json",
        parse_stdout_json=True,
    )

    assert result.exit_code == 4
    assert result.summary["success_rate"] == 1.0
    assert (record.run_dir / "stdout.txt").exists()
    assert (record.run_dir / "stderr.txt").read_text() == "synthetic stderr"
    assert registry.recent_runs()[0]["status"] == "failed"


def test_reports_cover_missing_invalid_nvidia_notes_and_generic_payloads(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    assert "not found" in reports.summarize_json_file(missing)["notes"][0]

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert "Could not parse" in reports.summarize_json_file(invalid)["notes"][0]

    sequence = tmp_path / "sequence.json"
    sequence.write_text("[]", encoding="utf-8")
    assert "Expected a JSON object" in reports.summarize_json_file(sequence)["notes"][0]

    payload_path = tmp_path / "payload.json"
    payload_path.write_text('{"value": 3}', encoding="utf-8")
    assert reports.summarize_json_file(payload_path) == {"value": 3}

    nvidia_payload = {
        "runtime": {"cuda_available": False, "tensor_allocation_ok": False},
        "isaac_backend": {"is_ready": False},
        "aiq": {"reachable": False},
    }
    nvidia_lines = reports._summarize_payload(nvidia_payload)
    assert any("CUDA available" in line for line in nvidia_lines)
    assert reports._summarize_payload({"notes": ["one", "two"]}) == [
        "- one",
        "- two",
    ]
    assert reports._summarize_payload({"value": 4})[0] == "```json"

    report_path = reports.write_report(
        tmp_path,
        "Readiness",
        [],
        1,
        nvidia_payload,
        tmp_path / "stdout.txt",
        tmp_path / "stderr.txt",
    )
    report = report_path.read_text(encoding="utf-8")
    assert "(no subprocess command)" in report
    assert "Stdout:" in report
    assert "Stderr:" in report


def test_grasp_policy_roundtrip_defaults_and_training(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert load_grasp_policy(missing) == default_grasp_policy()
    assert default_grasp_policy().success_threshold(0) == 0.88
    assert default_grasp_policy().success_threshold(99) == 0.66
    assert GraspPolicy(thresholds={}).success_threshold(2) == 0.75

    policy_path = tmp_path / "nested" / "policy.json"
    save_grasp_policy(GraspPolicy(thresholds={"1": 0.5}), policy_path)
    assert load_grasp_policy(policy_path).thresholds == {"1": 0.5}

    trained_path = tmp_path / "trained.json"
    summary = GraspPolicyTrainer(seed=7).train(episodes=400, save_path=trained_path)
    assert summary.episodes == 400
    assert summary.save_path == str(trained_path)
    assert load_grasp_policy(trained_path).thresholds == summary.learned_thresholds
    assert all(0.45 <= value <= 0.95 for value in summary.learned_thresholds.values())


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- not-a-mapping\n", "must be a mapping"),
        ("tasks: []\n", "'tasks' must be a mapping"),
        ("tasks:\n  broken: 3\n", "entries must be named mappings"),
    ],
)
def test_task_catalog_rejects_invalid_yaml_shapes(
    content: str,
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_tasks(path)


def test_planner_stack_and_collection_validation_branches() -> None:
    world = WorldState(
        robot=RobotState(end_effector_position=(0.0, 0.0, 0.4), gripper_opening=0.08),
        objects={
            "top": ObjectState(
                name="top",
                color_name="red",
                position=(0.0, 0.0, 0.03),
                size=(0.03, 0.03, 0.03),
            ),
            "base": ObjectState(
                name="base",
                color_name="blue",
                position=(0.2, 0.0, 0.03),
                size=(0.04, 0.04, 0.04),
            ),
        },
        zones={
            "tote": ZoneState(
                name="tote",
                center=(0.0, -0.5, 0.0),
                size=(0.2, 0.2, 0.05),
            )
        },
    )
    planner = RuleBasedPlanner()
    stack = TaskSpec(
        name="stack",
        task_type=TaskType.STACK_BLOCKS,
        source_object="top",
        target_object="base",
    )
    plan = planner.plan(stack, world)
    assert plan[-1].params == {"target_object": "base"}

    no_targets = TaskSpec(
        name="empty",
        task_type=TaskType.MULTI_OBJECT_COLLECTION,
        source_object="top",
        drop_zone="tote",
    )
    with pytest.raises(ValueError, match="target_objects"):
        planner.plan(no_targets, world)

    no_zone = no_targets.model_copy(update={"target_objects": ["top"], "drop_zone": None})
    with pytest.raises(ValueError, match="drop_zone"):
        planner.plan(no_zone, world)

    unstationed = no_targets.model_copy(update={"target_objects": ["top"]})
    assert [step.name for step in planner.plan(unstationed, world)] == ["observe_scene"]
