"""Bounded, public-safe per-update training evidence; never validation estimates."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any


SCHEMA = "construction-intelligence-training-curves-v1"
INDEX_ROLE = "training_curves"
INDEX_PATH = "evidence/research/training-curves.json"
CURVE_PREFIX = "training_curve:"
MAX_CURVE_BYTES = 2 * 1024**2
_RUN_KEY = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,159}")
_CORE = {
    "update", "transitions", "loss_objective", "loss_critic", "loss_entropy",
    "mean_episode_return", "rollout_terminal_fraction",
}
_FRACTIONS = {
    "rollout_terminal_fraction", "rollout_termination_fraction", "rollout_truncation_fraction",
    "rollout_collector_cutoff_fraction", "rollout_success_fraction",
    "rollout_structure_completion_rate", "rollout_failure_probability_mean",
    "rollout_failure_probability_min", "rollout_failure_probability_max",
    "rollout_failure_scenario_fraction", "clip_fraction", "ppo_clip_fraction",
}
_ALLOWED = _CORE | _FRACTIONS | {
    "rollout_episode_count", "kl_approx", "ppo_kl_approx", "ppo_optimizer_steps",
}
_IDENTITY = (
    "algorithm", "experiment_variant", "training_seed", "configuration_digest",
    "source_commit", "source_tree_digest",
)
SEMANTICS = {
    "curve_kind": "training_update_diagnostics",
    "validation_curve_kind": "validation_checkpoint_completion",
    "rollout_terminal_fraction": "Termination OR time-limit truncation; not task success.",
    "rollout_success_fraction": "Actual full-structure completion, when recorded by the source.",
    "rollout_structure_completion_rate": "Mean collected partial-trajectory completion; not validation.",
    "rollout_failure_scenario_fraction": "Scheduled failures; they may not trigger before collector cutoff.",
    "ppo_losses": "Source-recorded final minibatch losses per update; not epoch means.",
    "missing_diagnostics": "Unavailable fields are not inferred, backfilled, or relabelled.",
}


def curve_contract(role: str) -> tuple[str, str] | None:
    if role == INDEX_ROLE:
        return INDEX_PATH, "application/json"
    if not role.startswith(CURVE_PREFIX):
        return None
    key = role.removeprefix(CURVE_PREFIX)
    if _RUN_KEY.fullmatch(key) is None:
        raise ValueError("training curve run key is not public-safe")
    return f"evidence/research/training/{key}/learning_curve.csv", "text/csv"


def validate_curve(
    data: bytes, *, transitions: int, rollout_decisions: int | None = None,
    checkpoint_fractions: list[float] | None = None,
) -> dict[str, Any]:
    """Reject nonnumeric data, ambiguous columns, incomplete counters, and nonfinite diagnostics."""
    if not data or len(data) > MAX_CURVE_BYTES:
        raise ValueError("training curve is empty or exceeds its byte budget")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 1:
        raise ValueError("training curve source transition budget is missing or invalid")
    if rollout_decisions is not None and (
        isinstance(rollout_decisions, bool) or not isinstance(rollout_decisions, int)
        or rollout_decisions < 1
    ):
        raise ValueError("training curve source rollout size is missing or invalid")
    checkpoints = {transitions}
    if checkpoint_fractions is not None:
        if rollout_decisions is None:
            raise ValueError("training curve source rollout size is missing or invalid")
        if not isinstance(checkpoint_fractions, list) or not checkpoint_fractions:
            raise ValueError("training curve checkpoint fractions are invalid")
        for fraction in checkpoint_fractions:
            if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
                    or not math.isfinite(fraction) or not 0 < fraction <= 1):
                raise ValueError("training curve checkpoint fractions are invalid")
            target = int((Decimal(transitions) * Decimal(str(fraction))).to_integral_value(
                rounding=ROUND_HALF_UP,
            ))
            checkpoints.add(max(1, target))
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
        columns = reader.fieldnames
        if (not columns or len(columns) != len(set(columns))
                or not _CORE.issubset(columns) or not set(columns).issubset(_ALLOWED)):
            raise ValueError("training curve columns are missing, duplicate, or unsafe")
        count, previous = 0, 0
        for row in reader:
            count += 1
            if count > 50_000 or set(row) != set(columns) or any(value is None for value in row.values()):
                raise ValueError("training curve row shape/count is invalid")
            values = {key: float(value) for key, value in row.items()}
            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError("training curve contains nonfinite diagnostics")
            if values["update"] != count or not values["transitions"].is_integer():
                raise ValueError("training curve update/transition counters are invalid")
            current = int(values["transitions"])
            if current <= previous or current > transitions or current % 4:
                raise ValueError("training curve transition counters are not increasing four-robot transitions")
            if rollout_decisions is not None and current - previous > 4 * rollout_decisions:
                raise ValueError("training curve skips source update transitions")
            if checkpoint_fractions is not None:
                assert rollout_decisions is not None
                next_stop = min(target for target in checkpoints if target > previous)
                decisions = min(rollout_decisions, (next_stop - previous + 3) // 4)
                if current != min(previous + decisions * 4, transitions):
                    raise ValueError("training curve does not contain every configured source update")
            for field in _FRACTIONS & values.keys():
                if not 0 <= values[field] <= 1:
                    raise ValueError(f"training curve fraction outside [0,1]: {field}")
            for field in {"rollout_episode_count", "ppo_optimizer_steps"} & values.keys():
                if values[field] < 1 or not values[field].is_integer():
                    raise ValueError(f"training curve count must be a positive integer: {field}")
            previous = current
    except (UnicodeError, csv.Error, TypeError, OverflowError) as exc:
        raise ValueError(f"training curve is not bounded numeric UTF-8 CSV: {exc}") from exc
    if not count or previous != transitions:
        raise ValueError("training curve is truncated before the full transition budget")
    return {
        "columns": columns, "row_count": count, "first_update": 1, "last_update": count,
        "final_transitions": previous,
        "unavailable_diagnostics": sorted((_ALLOWED - _CORE) - set(columns)),
    }


def stage_training_curves(staging: Path, matrix: dict[str, Any]) -> dict[str, tuple[str, str, str]]:
    """Copy exactly the twenty completed source CSVs; retain bytes, never raw run directories."""
    records = []
    artifacts: dict[str, tuple[str, str, str]] = {
        INDEX_ROLE: ("training-curves.json", INDEX_PATH, "application/json"),
    }
    runs = matrix.get("runs")
    if not isinstance(runs, list) or len(runs) != 20:
        raise ValueError("training curves require the complete twenty-run matrix")
    seen: set[str] = set()
    for run in runs:
        key = str(run["run_key"])
        role = f"{CURVE_PREFIX}{key}"
        contract = curve_contract(role)
        if contract is None or key in seen or run.get("status") != "completed":
            raise ValueError("training curve membership must be twenty distinct completed runs")
        seen.add(key)
        raw_dir = run.get("artifact_dir")
        if not isinstance(raw_dir, str) or not raw_dir:
            raise ValueError(f"training curve source artifact directory is missing: {key}")
        source = Path(raw_dir) / "learning_curve.csv"
        if source.is_symlink() or not source.is_file() or source.stat().st_size > MAX_CURVE_BYTES:
            raise ValueError(f"training curve source is missing, unsafe, or too large: {key}")
        config = run["config"]
        metadata_path = Path(raw_dir) / "training_config.json"
        if (metadata_path.is_symlink() or not metadata_path.is_file()
                or metadata_path.stat().st_size > MAX_CURVE_BYTES):
            raise ValueError(f"training curve source configuration is missing or unsafe: {key}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (not isinstance(metadata, dict) or metadata.get("source_dirty") is not False or any(
            metadata.get(field) != config.get(field)
            for field in (*_IDENTITY, "transitions", "rollout_decisions", "ppo_epochs", "checkpoint_fractions")
        )):
            raise ValueError(f"training curve source configuration/provenance mismatch: {key}")
        data = source.read_bytes()
        summary = validate_curve(data, transitions=config.get("transitions", 0),
                                 rollout_decisions=config.get("rollout_decisions"),
                                 checkpoint_fractions=config.get("checkpoint_fractions"))
        relative = contract[0].removeprefix("evidence/research/")
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        artifacts[role] = relative, contract[0], contract[1]
        records.append({
            "run_key": key, "role": role, "path": contract[0],
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            **{field: config.get(field) for field in _IDENTITY}, **summary,
        })
    index = {
        "schema_version": SCHEMA, "matrix_id": matrix["id"], "semantics": SEMANTICS,
        "run_count": 20, "runs": sorted(records, key=lambda item: item["run_key"]),
    }
    (staging / "training-curves.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return artifacts


def verify_training_curves(
    files: dict[str, Path], configs: dict[str, dict[str, Any]], *, matrix_id: str,
) -> None:
    """Independently bind the index, twenty curves, source/config identities, and byte hashes."""
    index_path = files[INDEX_ROLE]
    if index_path.stat().st_size > MAX_CURVE_BYTES:
        raise ValueError("training curves index exceeds its byte budget")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (not isinstance(index, dict) or index.get("schema_version") != SCHEMA
            or index.get("matrix_id") != matrix_id or index.get("semantics") != SEMANTICS
            or index.get("run_count") != 20):
        raise ValueError("training curves index identity/semantics is invalid")
    records = index.get("runs")
    if not isinstance(records, list) or len(records) != 20 or not all(isinstance(row, dict) for row in records):
        raise ValueError("training curves index must contain twenty run records")
    keys = [record.get("run_key") for record in records]
    if keys != sorted(configs) or len(configs) != 20:
        raise ValueError("training curves do not cover the exact twenty research runs")
    observed_roles = {role for role in files if role.startswith(CURVE_PREFIX)}
    if observed_roles != {f"{CURVE_PREFIX}{key}" for key in configs}:
        raise ValueError("training curve artifact membership is incomplete or excessive")
    for record in records:
        key = record["run_key"]
        config = configs[key]
        role = f"{CURVE_PREFIX}{key}"
        contract = curve_contract(role)
        assert contract is not None
        path = files[role]
        if path.stat().st_size > MAX_CURVE_BYTES:
            raise ValueError("training curve exceeds its byte budget")
        data = path.read_bytes()
        summary = validate_curve(data, transitions=config.get("transitions", 0),
                                 rollout_decisions=config.get("rollout_decisions"),
                                 checkpoint_fractions=config.get("checkpoint_fractions"))
        expected = {
            "run_key": key, "role": role, "path": contract[0], "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            **{field: config.get(field) for field in _IDENTITY}, **summary,
        }
        if record != expected:
            raise ValueError(f"training curve index hash/content/provenance mismatch: {key}")
