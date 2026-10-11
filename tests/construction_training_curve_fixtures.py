"""Small synthetic source artifacts, distinct from five validation checkpoints."""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path
from typing import Any


def training_csv() -> bytes:
    rows = [{
        "update": update, "transitions": transitions,
        "loss_objective": -0.1, "loss_critic": 0.2, "loss_entropy": -0.01,
        "mean_episode_return": 0.5, "rollout_terminal_fraction": 0.9,
        "rollout_episode_count": 4, "rollout_termination_fraction": 0.25,
        "rollout_truncation_fraction": 0.65, "rollout_collector_cutoff_fraction": 0.1,
        "rollout_success_fraction": 0.25, "rollout_structure_completion_rate": 0.6,
        "rollout_failure_probability_mean": 0.1, "rollout_failure_probability_min": 0.0,
        "rollout_failure_probability_max": 0.2, "rollout_failure_scenario_fraction": 0.25,
    } for update, transitions in enumerate(
        (150_000, 375_000, 625_000, 750_000, 1_000_000, 1_125_000, 1_375_000, 1_500_000),
        start=1,
    )]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def attach_training_curve_sources(root: Path, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for run in runs:
        run["config"].setdefault("rollout_decisions", 62_500)
        run["config"].setdefault("ppo_epochs", 6)
        run["config"].setdefault("checkpoint_fractions", [0.1, 0.25, 0.5, 0.75, 1.0])
    private = copy.deepcopy(runs)
    for run in private:
        directory = root / run["run_key"]
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "learning_curve.csv").write_bytes(training_csv())
        (directory / "training_config.json").write_text(json.dumps(run["config"]), encoding="utf-8")
        run["artifact_dir"] = str(directory)
    return private
