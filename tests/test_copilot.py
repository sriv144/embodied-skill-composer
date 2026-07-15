import json
import sqlite3
import subprocess
import sys

import pytest

from embodied_skill_composer.copilot.nvidia import check_aiq_health
from embodied_skill_composer.copilot.registry import CopilotRegistry
from embodied_skill_composer.copilot.runner import ConfirmationRequired, require_confirmation, run_benchmark


def test_registry_creates_runs_and_metrics(tmp_path):
    registry = CopilotRegistry(tmp_path / "experiments.sqlite", tmp_path / "runs")
    run = registry.create_run("benchmark", command=["python", "x.py"], runtime_profile="local_dev")
    registry.add_metric(run.id, "scripted_options", 1.0, 10.5, 2.0, 32.0)
    registry.complete_run(run.id, "completed", 0, run.run_dir / "report.md")

    rows = registry.recent_runs()
    assert rows[0]["id"] == run.id
    assert rows[0]["status"] == "completed"

    with sqlite3.connect(tmp_path / "experiments.sqlite") as conn:
        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert metric_count == 1


def test_training_requires_yes_in_non_interactive(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ConfirmationRequired):
        require_confirmation("training", yes=False)


def test_benchmark_runner_records_metrics_with_mocked_subprocess(tmp_path, monkeypatch):
    registry = CopilotRegistry(tmp_path / "experiments.sqlite", tmp_path / "runs")

    def fake_run(command, cwd, capture_output, text, env):
        output_path = command[command.index("--output") + 1]
        payload = {
            "backend": "local_sandbox",
            "runtime_profile": "local_dev",
            "scripted_options": {
                "policy_name": "scripted_options",
                "success_rate": 1.0,
                "mean_return": 10.52,
                "mean_beams_installed": 2.0,
                "mean_step_count": 32.0,
            },
            "learned_options": {
                "policy_name": "learned_options",
                "success_rate": 1.0,
                "mean_return": 10.52,
                "mean_beams_installed": 2.0,
                "mean_step_count": 32.0,
            },
            "low_level_learned": {
                "policy_name": "low_level_learned",
                "success_rate": 0.0,
                "mean_return": -0.02,
                "mean_beams_installed": 1.0,
                "mean_step_count": 120.0,
            },
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_benchmark(episodes=1, registry=registry)

    assert result.exit_code == 0
    assert result.report_path.exists()
    with sqlite3.connect(tmp_path / "experiments.sqlite") as conn:
        metric_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert metric_count == 3


def test_aiq_health_reports_unreachable():
    result = check_aiq_health("http://127.0.0.1:9", timeout_seconds=0.1)
    assert result["reachable"] is False
