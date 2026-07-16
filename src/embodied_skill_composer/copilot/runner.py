from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_skill_composer.copilot.paths import PROJECT_ROOT
from embodied_skill_composer.copilot.registry import CopilotRegistry, default_registry
from embodied_skill_composer.copilot.reports import summarize_json_file, write_report


@dataclass(frozen=True)
class CopilotRunResult:
    run_id: str
    run_dir: Path
    report_path: Path
    exit_code: int
    summary: dict[str, Any]


class ConfirmationRequired(RuntimeError):
    pass


def require_confirmation(action: str, yes: bool) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise ConfirmationRequired(f"{action} requires --yes in non-interactive mode.")
    response = input(f"{action} may write checkpoints and take time. Type YES to continue: ")
    if response.strip() != "YES":
        raise ConfirmationRequired(f"{action} was not confirmed.")


def run_benchmark(
    episodes: int = 5,
    runtime_profile: str = "configs/assembly_profiles/local_dev.yaml",
    registry: CopilotRegistry | None = None,
) -> CopilotRunResult:
    registry = registry or default_registry()
    record = registry.create_run("benchmark", command=[], runtime_profile=runtime_profile)
    output_path = record.run_dir / "assembly_policy_benchmark.json"
    command = [
        sys.executable,
        "scripts/benchmark_assembly_policies.py",
        "--runtime-profile",
        runtime_profile,
        "--episodes",
        str(episodes),
        "--output",
        str(output_path),
    ]
    return _run_subprocess(record, registry, command, "Assembly Policy Benchmark", output_path)


def run_eval_options(
    policy: str = "scripted",
    episodes: int = 1,
    runtime_profile: str = "configs/assembly_profiles/local_dev.yaml",
    registry: CopilotRegistry | None = None,
) -> CopilotRunResult:
    registry = registry or default_registry()
    record = registry.create_run("eval-options", command=[], runtime_profile=runtime_profile)
    output_path = record.run_dir / "eval_options_stdout.json"
    command = [
        sys.executable,
        "scripts/eval_assembly_options.py",
        "--policy",
        policy,
        "--runtime-profile",
        runtime_profile,
        "--episodes",
        str(episodes),
    ]
    return _run_subprocess(record, registry, command, f"Option Evaluation: {policy}", output_path, parse_stdout_json=True)


def run_train_options(
    runtime_profile: str = "configs/assembly_profiles/local_dev.yaml",
    yes: bool = False,
    registry: CopilotRegistry | None = None,
) -> CopilotRunResult:
    require_confirmation("Hierarchical options training", yes)
    registry = registry or default_registry()
    record = registry.create_run("train-options", command=[], runtime_profile=runtime_profile)
    checkpoint = record.run_dir / "assembly_options.pt"
    metrics = record.run_dir / "assembly_option_training_metrics.json"
    command = [
        sys.executable,
        "scripts/train_assembly_options.py",
        "--runtime-profile",
        runtime_profile,
        "--checkpoint",
        str(checkpoint),
        "--metrics",
        str(metrics),
    ]
    return _run_subprocess(record, registry, command, "Hierarchical Options Training", metrics)


def run_train_marl(
    runtime_profile: str = "configs/assembly_profiles/local_dev.yaml",
    yes: bool = False,
    registry: CopilotRegistry | None = None,
) -> CopilotRunResult:
    require_confirmation("Low-level MARL training", yes)
    registry = registry or default_registry()
    record = registry.create_run("train-marl", command=[], runtime_profile=runtime_profile)
    checkpoint = record.run_dir / "assembly_marl.pt"
    metrics = record.run_dir / "assembly_training_metrics.json"
    command = [
        sys.executable,
        "scripts/train_assembly_marl.py",
        "--runtime-profile",
        runtime_profile,
        "--checkpoint",
        str(checkpoint),
        "--metrics",
        str(metrics),
    ]
    return _run_subprocess(record, registry, command, "Low-Level MARL Training", metrics)


def run_sweep(
    scenarios: int = 5,
    seeds: str = "7,8,9",
    beam_count: int = 2,
    runtime_profile: str = "configs/assembly_profiles/local_dev.yaml",
    yes: bool = False,
    registry: CopilotRegistry | None = None,
) -> CopilotRunResult:
    require_confirmation("Assembly experiment sweep", yes)
    registry = registry or default_registry()
    record = registry.create_run("sweep", command=[], runtime_profile=runtime_profile)
    sweep_dir = record.run_dir / "sweep"
    command = [
        sys.executable,
        "scripts/run_assembly_experiment_sweep.py",
        "--runtime-profile",
        runtime_profile,
        "--scenarios",
        str(scenarios),
        "--seeds",
        seeds,
        "--beam-count",
        str(beam_count),
        "--output-dir",
        str(sweep_dir),
    ]
    return _run_subprocess(record, registry, command, "Assembly Experiment Sweep", sweep_dir / "summary.json")


def _run_subprocess(
    record,
    registry: CopilotRegistry,
    command: list[str],
    title: str,
    summary_path: Path,
    parse_stdout_json: bool = False,
) -> CopilotRunResult:
    (record.run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    process = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, env=os.environ.copy())
    stdout_path = record.run_dir / "stdout.txt"
    stderr_path = record.run_dir / "stderr.txt"
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    registry.add_artifact(record.id, "stdout", stdout_path, "Captured subprocess stdout.")
    registry.add_artifact(record.id, "stderr", stderr_path, "Captured subprocess stderr.")
    registry.add_artifact(record.id, "command", record.run_dir / "command.json", "Exact command executed.")

    if parse_stdout_json:
        summary = _extract_eval_stdout_summary(process.stdout)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        summary = summarize_json_file(summary_path)

    if summary_path.exists():
        registry.add_artifact(record.id, "summary", summary_path, "Primary structured run output.")
    _record_metrics(record.id, registry, summary)
    report_path = write_report(record.run_dir, title, command, process.returncode, summary, stdout_path, stderr_path)
    registry.add_artifact(record.id, "report", report_path, "Markdown research/debug report.")
    status = "completed" if process.returncode == 0 else "failed"
    registry.complete_run(record.id, status, process.returncode, report_path)
    return CopilotRunResult(record.id, record.run_dir, report_path, process.returncode, summary)


def _record_metrics(run_id: str, registry: CopilotRegistry, summary: dict[str, Any]) -> None:
    for key in ("scripted_options", "learned_options", "low_level_learned"):
        row = summary.get(key)
        if isinstance(row, dict):
            registry.add_metric(
                run_id,
                str(row.get("policy_name", key)),
                float(row.get("success_rate", 0.0)),
                float(row.get("mean_return", 0.0)),
                float(row.get("mean_beams_installed", 0.0)),
                float(row.get("mean_step_count", 0.0)),
            )


def _extract_eval_stdout_summary(stdout: str) -> dict[str, Any]:
    marker = "[\n"
    index = stdout.find(marker)
    if index == -1:
        return {"notes": ["No JSON episode payload found in eval stdout."], "raw": stdout}
    try:
        episodes = json.loads(stdout[index:])
    except json.JSONDecodeError as exc:
        return {"notes": [f"Could not parse eval stdout JSON: {exc}"], "raw": stdout}
    artifacts = [episode.get("artifact", {}) for episode in episodes if isinstance(episode, dict)]
    successes = [artifact.get("metrics", {}).get("success", False) for artifact in artifacts]
    beams = [artifact.get("metrics", {}).get("beams_installed", 0) for artifact in artifacts]
    returns = [artifact.get("metrics", {}).get("total_reward", 0.0) for artifact in artifacts]
    return {
        "episodes": episodes,
        "success_rate": sum(int(bool(item)) for item in successes) / max(1, len(successes)),
        "mean_beams_installed": sum(float(item) for item in beams) / max(1, len(beams)),
        "mean_return": sum(float(item) for item in returns) / max(1, len(returns)),
    }

