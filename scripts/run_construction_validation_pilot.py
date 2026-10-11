"""Durable, validation-only PPO update-budget pilots; never a release matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
SCHEMA = "construction-intelligence-validation-pilot-v1"
EXPERIMENT_ID = "construction_intelligence_validation_pilot"
VALIDATION_SEEDS = [800, 801, 802, 803, 804]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
TRANSITIONS = 150_000


@dataclass(frozen=True)
class PilotRuntime:
    """Lazy production bindings also permit workflow tests without Torch imports."""

    source: Callable[[], dict[str, object]]
    environment: Callable[[], dict[str, object]]
    load_design: Callable[..., Any]
    design_digest: Callable[..., str]
    load_protocol: Callable[..., Any]
    protocol_digest: Callable[..., str]
    config: Callable[..., Any]
    config_digest: Callable[..., str]
    registry: Callable[..., Any]
    service: Callable[..., Any]
    candidates: Callable[..., Any]
    episode: Callable[..., Any]


def _runtime() -> PilotRuntime:
    source_root = str(WORKSPACE / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from embodied_skill_composer.construction.evaluation import evaluate_controller_episode
    from embodied_skill_composer.construction.experiment_execution import (
        evaluate_run_checkpoint_candidates,
    )
    from embodied_skill_composer.construction.experiment_protocol import (
        load_experiment_protocol,
        protocol_digest,
    )
    from embodied_skill_composer.construction.lab_registry import LabRegistry
    from embodied_skill_composer.construction.lab_service import LabService
    from embodied_skill_composer.construction.runtime import load_house_design
    from embodied_skill_composer.construction.training import (
        TrainingConfig,
        configuration_digest,
        design_digest,
        environment_fingerprint,
        source_fingerprint,
    )

    return PilotRuntime(
        source_fingerprint, environment_fingerprint, load_house_design, design_digest,
        load_experiment_protocol, protocol_digest, TrainingConfig.model_validate,
        configuration_digest, LabRegistry, LabService, evaluate_run_checkpoint_candidates,
        evaluate_controller_episode,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"), nargs="?", default="plan")
    parser.add_argument(
        "--registry", type=Path,
        default=WORKSPACE / "logs/construction_intelligence/validation-pilot.sqlite",
        help="Dedicated pilot registry, not the frozen research registry.",
    )
    parser.add_argument(
        "--evidence-root", type=Path,
        default=WORKSPACE / "logs/construction_intelligence/validation_pilot",
    )
    parser.add_argument(
        "--design", type=Path, default=WORKSPACE / "configs/construction/cottage_v1.yaml",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-resume-attempts", type=int, default=3)
    return parser


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_clean_source(source: dict[str, object]) -> None:
    if (
        source.get("dirty") is not False
        or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("tree_digest", "")))
    ):
        raise ValueError("validation pilots require clean committed source fingerprints")


def build_pilot_protocol(
    source: dict[str, object], *, design_digest: str, parent_protocol_digest: str, device: str,
) -> tuple[str, str, dict[str, Any]]:
    _require_clean_source(source)
    if device not in {"cpu", "cuda"}:
        raise ValueError("pilot device must be cpu or cuda")
    specs: list[dict[str, Any]] = []
    for algorithm in ("mappo", "ippo"):
        for epochs in (6, 2):
            key = f"{algorithm}-ppo{epochs}-seed7"
            config = {
                "algorithm": algorithm, "profile": "research", "seed": 7,
                "training_seed": 7, "experiment_id": EXPERIMENT_ID,
                "experiment_variant": f"{algorithm}_ppo{epochs}",
                "transitions": TRANSITIONS, "expert_episodes": 128,
                "behavior_clone_epochs": 40, "rollout_decisions": 2048,
                "ppo_epochs": epochs, "minibatch_size": 512, "hidden_dim": 128,
                "include_training_failures": True, "failure_curriculum_mode": "linear",
                "failure_curriculum_start_probability": 0.0,
                "failure_curriculum_final_probability": 0.35,
                "failure_curriculum_ramp_fraction": 0.5,
                "failure_curriculum_reference_transitions": 1_500_000,
                "checkpoint_fractions": list(FRACTIONS), "device": device,
            }
            specs.append({"run_key": key, "config": config})
    protocol: dict[str, Any] = {
        "schema_version": SCHEMA, "experiment_id": EXPERIMENT_ID,
        "purpose": "validation_only_non_release", "release_eligible": False,
        "source": dict(source), "design_digest": design_digest,
        "parent_protocol_digest": parent_protocol_digest,
        "selection": {
            "split": "validation", "scenario_seeds": list(VALIDATION_SEEDS),
            "failure_modes": [False, True], "deterministic": True,
        },
        "checkpoint_fractions": list(FRACTIONS), "run_specs": specs,
        "total_training_transitions": 4 * TRANSITIONS,
        "comparison": "PPO epochs 6 versus 2; all other training fields fixed",
        "comparison_checkpoint": "final 150000-transition checkpoint only",
        "winner_rule": {
            "eligibility": {
                "both_algorithms_nominal_completion_min": 0.95,
                "mappo_failure_completion_min": 0.85,
                "mappo_median_makespan_cp_sat_ratio_max": 1.15,
            },
            "lexicographic_order": [
                "min_primary_nominal_completion_desc", "mean_primary_nominal_completion_desc",
                "mappo_failure_completion_desc", "mappo_nominal_makespan_ratio_asc",
                "pooled_nominal_mean_makespan_asc", "ppo_epochs_asc",
            ],
            "no_eligible_winner": None,
        },
        "research_launch_authorized": False,
    }
    identity = _digest(protocol)
    for spec in specs:
        spec["config"]["protocol_run_digest"] = _digest({"pilot": identity, "run": spec})
    digest = _digest(protocol)
    return f"validation-pilot-{str(source['commit'])[:12]}-{digest[:12]}", digest, protocol


@contextmanager
def executor_lock(path: Path) -> Iterator[None]:
    """Kernel-owned lock: crashes release ownership without trusting PID files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("pilot executor lock cannot be a symlink")
    handle = path.open("a+b")
    locked = False
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                lock_module: Any = importlib.import_module("fcntl")
                lock_module.flock(handle.fileno(), lock_module.LOCK_EX | lock_module.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError("another pilot executor already owns this registry") from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                lock_module = importlib.import_module("fcntl")
                lock_module.flock(handle.fileno(), lock_module.LOCK_UN)
        handle.close()


def _dump(value: Any) -> dict[str, Any]:
    result = value if isinstance(value, dict) else value.model_dump(mode="json")
    if not isinstance(result, dict):
        raise ValueError("pilot evidence must contain JSON objects")
    return result


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _matrix_runs(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    runs = matrix.get("runs")
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        raise ValueError("pilot matrix run records are malformed")
    return runs


def _assert_matrix(
    matrix: dict[str, Any], *, matrix_id: str, digest: str, protocol: dict[str, Any],
    configs: dict[str, Any], design: Any, runtime: PilotRuntime,
) -> list[dict[str, Any]]:
    if (
        matrix.get("id") != matrix_id or matrix.get("protocol_digest") != digest
        or matrix.get("protocol") != protocol
        or matrix.get("execution_profile") != "validation_pilot"
        or matrix.get("expected_run_count") != 4
    ):
        raise ValueError("persisted pilot matrix identity/protocol has changed")
    runs = _matrix_runs(matrix)
    if len(runs) != 4 or {run.get("run_key") for run in runs} != set(configs):
        raise ValueError("pilot must contain exactly its four distinct training runs")
    mutable = {"configuration_digest", "checkpoint_lineage", "resume_checkpoint", "resume_provenance"}
    for run in runs:
        key = run["run_key"]
        expected = _dump(configs[key])
        actual = _dump(runtime.config(run["config"]))
        if (
            run.get("id") != f"{matrix_id}-{key}"
            or {k: v for k, v in actual.items() if k not in mutable}
            != {k: v for k, v in expected.items() if k not in mutable}
            or run.get("input", {}).get("design") != _dump(design)
        ):
            raise ValueError(f"pilot run configuration/source/design changed: {key}")
        recomputed = runtime.config_digest(runtime.config(actual))
        if (
            run.get("config_digest") != recomputed
            or actual.get("configuration_digest") != recomputed
            or expected.get("configuration_digest") != recomputed
            or run.get("source_commit") != protocol["source"]["commit"]
        ):
            raise ValueError(f"pilot run fingerprint mismatch: {key}")
    return runs


def _assert_validation_grid(episodes: list[dict[str, Any]], controller: str) -> None:
    expected = {(seed, failure) for seed in VALIDATION_SEEDS for failure in (False, True)}
    cells = [(episode.get("seed"), episode.get("failure_enabled")) for episode in episodes]
    if len(cells) != 10 or set(cells) != expected:
        raise ValueError("pilot episodes must cover exactly validation seeds 800-804 in both modes")
    for episode in episodes:
        if episode.get("split") != "validation" or episode.get("controller") != controller:
            raise ValueError("non-validation or wrong-controller episode in pilot evidence")
        for field in ("structure_completion_rate", "makespan_s"):
            value = episode.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid pilot episode metric: {field}")
        if episode["structure_completion_rate"] > 1:
            raise ValueError("pilot completion fraction exceeds one")


def _candidate_summary(
    candidate: dict[str, Any], baseline: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for failure in (False, True):
        rows = [row for row in candidate["episodes"] if row["failure_enabled"] == failure]
        reference = [row for row in baseline if row["failure_enabled"] == failure]
        baseline_makespan = median(row["makespan_s"] for row in reference)
        if baseline_makespan <= 0:
            raise ValueError("CP-SAT validation baseline has a zero median makespan")
        makespan = median(row["makespan_s"] for row in rows)
        metrics["failure" if failure else "nominal"] = {
            "mean_completion": mean(row["structure_completion_rate"] for row in rows),
            "median_makespan_s": makespan,
            "mean_makespan_s": mean(row["makespan_s"] for row in rows),
            "median_makespan_cp_sat_ratio": makespan / baseline_makespan,
        }
    return {
        "checkpoint_fraction": candidate["result"]["checkpoint_fraction"],
        "transition_count": candidate["result"]["transition_count"], **metrics,
    }


def choose_common_epochs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered FINAL-checkpoint validation pilot rule, not release gates."""
    rows = []
    preferences = {}
    for algorithm in ("mappo", "ippo"):
        candidates = [row for row in summaries if row["algorithm"] == algorithm]
        if len(candidates) != 2 or {row["ppo_epochs"] for row in candidates} != {6, 2}:
            raise ValueError("pilot winner requires both epoch budgets for both algorithms")
        preferred = min(candidates, key=lambda row: (
            -row["final"]["nominal"]["mean_completion"],
            -row["final"]["failure"]["mean_completion"],
            row["final"]["nominal"]["median_makespan_cp_sat_ratio"],
            row["final"]["nominal"]["mean_makespan_s"], row["ppo_epochs"],
        ))
        preferences[algorithm] = preferred["ppo_epochs"]
    for epochs in (6, 2):
        mappo = next(row["final"] for row in summaries
                     if row["algorithm"] == "mappo" and row["ppo_epochs"] == epochs)
        ippo = next(row["final"] for row in summaries
                    if row["algorithm"] == "ippo" and row["ppo_epochs"] == epochs)
        completions = [mappo["nominal"]["mean_completion"], ippo["nominal"]["mean_completion"]]
        eligible = (
            min(completions) >= 0.95
            and mappo["failure"]["mean_completion"] >= 0.85
            and mappo["nominal"]["median_makespan_cp_sat_ratio"] <= 1.15
        )
        rows.append({
            "ppo_epochs": epochs, "validation_pilot_eligible": eligible,
            "minimum_nominal_completion": min(completions),
            "mean_nominal_completion": mean(completions),
            "mappo_failure_completion": mappo["failure"]["mean_completion"],
            "mappo_nominal_makespan_ratio": mappo["nominal"]["median_makespan_cp_sat_ratio"],
            "pooled_nominal_mean_makespan_s": mean([
                mappo["nominal"]["mean_makespan_s"], ippo["nominal"]["mean_makespan_s"],
            ]),
        })
    eligible_rows = [row for row in rows if row["validation_pilot_eligible"]]
    winner = min(eligible_rows, key=lambda row: (
        -row["minimum_nominal_completion"], -row["mean_nominal_completion"],
        -row["mappo_failure_completion"], row["mappo_nominal_makespan_ratio"],
        row["pooled_nominal_mean_makespan_s"], row["ppo_epochs"],
    )) if eligible_rows else None
    return {
        "comparison_checkpoint": "final 150000-transition checkpoint only",
        "winner_ppo_epochs": winner["ppo_epochs"] if winner else None,
        "per_algorithm_preferences": preferences,
        "per_algorithm_preference_rule": (
            "final nominal completion, final failure completion, nominal makespan ratio, "
            "nominal mean makespan, fewer epochs; diagnostic only"
        ),
        "epoch_options": rows, "release_eligible": False,
        "research_launch_authorized": False,
    }


def _evaluate(
    registry: Any, matrix_id: str, runs: list[dict[str, Any]], design: Any,
    protocol: Any, pilot_protocol: dict[str, Any], root: Path, device: str,
    runtime: PilotRuntime,
) -> dict[str, Any]:
    if runtime.source() != pilot_protocol["source"]:
        raise ValueError("pilot source changed before validation evaluation")
    baseline = [
        _dump(runtime.episode(
            design, seed=seed, controller="cp_sat", failure_enabled=failure,
            expected_split="validation", device=device,
        ))
        for failure in (False, True) for seed in VALIDATION_SEEDS
    ]
    _assert_validation_grid(baseline, "cp_sat")
    _write_json(root / "baseline_validation.json", {
        "schema_version": SCHEMA, "release_eligible": False, "episodes": baseline,
        "source": pilot_protocol["source"],
    })
    evidence = []
    summaries = []
    for run in runs:
        if runtime.source() != pilot_protocol["source"]:
            raise ValueError("pilot source changed before validation evaluation")
        selection = _dump(runtime.candidates(
            design, run, protocol, matrix_id=matrix_id,
            output_root=root / "validation", device=device,
        ))
        candidates = selection.get("candidates", [])
        if (
            selection.get("matrix_id") != matrix_id
            or selection.get("run_key") != run["run_key"]
            or len(candidates) != 5
            or [candidate["result"]["checkpoint_fraction"] for candidate in candidates] != FRACTIONS
        ):
            raise ValueError("pilot candidate evidence identity/grid mismatch")
        config = run["config"]
        for candidate in candidates:
            result = candidate["result"]
            if (
                result.get("scenario_seeds") != VALIDATION_SEEDS
                or result.get("split") != "validation"
                or result.get("experiment_id") != EXPERIMENT_ID
                or result.get("experiment_variant") != config["experiment_variant"]
                or result.get("training_seed") != 7
                or result.get("configuration_digest") != run["config_digest"]
                or result.get("source_commit") != pilot_protocol["source"]["commit"]
                or result.get("transition_count")
                != int(TRANSITIONS * result["checkpoint_fraction"])
            ):
                raise ValueError("pilot candidate provenance mismatch")
            _assert_validation_grid(candidate["episodes"], config["algorithm"])
            for episode in candidate["episodes"]:
                if any(episode.get(field) != result.get(field) for field in (
                    "experiment_id", "experiment_variant", "training_seed",
                    "configuration_digest", "source_commit", "checkpoint_sha256",
                )):
                    raise ValueError("pilot candidate episode provenance mismatch")
        selected_id = selection["selected"]["checkpoint_id"]
        selected = [item for item in candidates if item["result"]["checkpoint_id"] == selected_id]
        if len(selected) != 1 or any(
            selection["selected"].get(key) != value
            for key, value in selected[0]["result"].items()
        ):
            raise ValueError("pilot selection is not one of its validated candidates")
        evidence.append(selection)
        summaries.append({
            "run_key": run["run_key"], "algorithm": config["algorithm"],
            "ppo_epochs": config["ppo_epochs"], "training_seed": 7,
            "selected": _candidate_summary(selected[0], baseline),
            "final": _candidate_summary(candidates[-1], baseline),
            "learning_checkpoints": [_candidate_summary(item, baseline) for item in candidates],
        })
    comparisons = []
    for algorithm in ("mappo", "ippo"):
        control = next(row for row in summaries if row["algorithm"] == algorithm and row["ppo_epochs"] == 6)
        treatment = next(row for row in summaries if row["algorithm"] == algorithm and row["ppo_epochs"] == 2)
        comparisons.append({
            "algorithm": algorithm,
            "final_nominal_completion_delta_points": 100 * (
                treatment["final"]["nominal"]["mean_completion"]
                - control["final"]["nominal"]["mean_completion"]
            ),
            "final_failure_completion_delta_points": 100 * (
                treatment["final"]["failure"]["mean_completion"]
                - control["final"]["failure"]["mean_completion"]
            ),
            "final_nominal_makespan_ratio_epoch2_over_epoch6": (
                treatment["final"]["nominal"]["median_makespan_s"]
                / control["final"]["nominal"]["median_makespan_s"]
            ),
        })
    summary = {
        "schema_version": SCHEMA, "matrix_id": matrix_id,
        "purpose": "validation_only_non_release", "release_eligible": False,
        "research_launch_authorized": False, "source": pilot_protocol["source"],
        "runs": summaries, "comparisons": comparisons,
        "final_checkpoint_decision": choose_common_epochs(summaries),
        "limitations": "Single training seed; validation only; no release acceptance or confidence claim.",
    }
    if runtime.source() != pilot_protocol["source"]:
        raise ValueError("pilot source changed before validation evidence persistence")
    _write_json(root / "pilot_evidence.json", {
        "protocol": pilot_protocol, "release_eligible": False,
        "selections": evidence, "baseline_episodes": baseline,
    })
    _write_json(root / "validation_pilot_summary.json", summary)
    if runtime.source() != pilot_protocol["source"]:
        raise ValueError("pilot source changed before validation evidence registration")
    registry.upsert_evaluation(
        f"{matrix_id}-validation", matrix_id=matrix_id, split="validation",
        payload=summary, artifact_dir=str(root),
    )
    return summary


def execute(args: argparse.Namespace, runtime: PilotRuntime) -> int:
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError("poll seconds must be finite and positive")
    if args.max_resume_attempts < 1:
        raise ValueError("maximum resume attempts must be positive")
    source = runtime.source()
    _require_clean_source(source)
    design = runtime.load_design(args.design)
    parent = runtime.load_protocol()
    selection = _dump(parent.selection)
    if (
        parent.protocol_version != "1.1.0"
        or selection.get("scenario_seeds") != VALIDATION_SEEDS
        or selection.get("split") != "validation"
        or selection.get("failure_modes") != [False, True]
        or selection.get("deterministic") is not True
        or list(parent.checkpoint_fractions) != FRACTIONS
    ):
        raise ValueError("pilot requires corrected protocol 1.1.0 and exact validation-only grid")
    matrix_id, digest, pilot = build_pilot_protocol(
        source, design_digest=runtime.design_digest(design),
        parent_protocol_digest=runtime.protocol_digest(parent), device=args.device,
    )
    if args.command == "plan":
        print(json.dumps({"matrix_id": matrix_id, "protocol_digest": digest, "protocol": pilot}, indent=2))
        return 0
    registry_path = args.registry.resolve()
    with executor_lock(registry_path.with_suffix(".pilot-executor.lock")):
        registry = runtime.registry(registry_path)
        environment = runtime.environment()
        configs = {}
        for spec in pilot["run_specs"]:
            payload = {
                **spec["config"], "environment_fingerprint": environment,
                "source_commit": source["commit"], "source_tree_digest": source["tree_digest"],
                "source_dirty": False, "output_root": str(registry_path.parent / "pilot_training"),
            }
            config = runtime.config(payload)
            if _dump(config).get("failure_curriculum_reference_transitions") != 1_500_000:
                raise ValueError("training runtime cannot preserve the research curriculum prefix")
            configs[spec["run_key"]] = runtime.config({
                **_dump(config), "configuration_digest": runtime.config_digest(config),
            })
        matrix = registry.get_experiment_matrix(matrix_id)
        for other in registry.list_experiment_matrices():
            if other.get("protocol", {}).get("schema_version") != SCHEMA:
                raise ValueError("pilot requires a dedicated registry without research matrices")
        expected_ids = {f"{matrix_id}-{key}" for key in configs}
        for other in registry.list_runs(limit=100_000):
            if other["id"] not in expected_ids and other["status"] not in {"completed", "cancelled", "failed"}:
                raise ValueError("another matrix owns pending work in the pilot registry")
        if matrix is not None:
            _assert_matrix(matrix, matrix_id=matrix_id, digest=digest, protocol=pilot,
                           configs=configs, design=design, runtime=runtime)
        if runtime.source() != source:
            raise ValueError("pilot source fingerprint changed before dispatch")
        service = runtime.service(registry)
        try:
            if matrix is None:
                service.launch_training_matrix(
                    design, matrix_id=matrix_id, protocol_digest=digest, protocol=pilot,
                    execution_profile="validation_pilot", runs=list(configs.items()),
                )
            previous = None
            while True:
                if runtime.source() != source:
                    raise ValueError("pilot source fingerprint changed during execution")
                matrix = registry.get_experiment_matrix(matrix_id)
                if matrix is None:
                    raise RuntimeError("pilot matrix disappeared")
                runs = _assert_matrix(matrix, matrix_id=matrix_id, digest=digest, protocol=pilot,
                                      configs=configs, design=design, runtime=runtime)
                status = [(run["run_key"], run["status"], run.get("attempt", 0)) for run in runs]
                if status != previous:
                    print(json.dumps({"matrix_id": matrix_id, "runs": status}), flush=True)
                    previous = status
                if all(run["status"] == "completed" for run in runs):
                    break
                for run in runs:
                    if run["status"] in {"cancelled", "cancel_requested"}:
                        raise RuntimeError(f"pilot cancellation is terminal: {run['run_key']}")
                    if run["status"] not in {"failed", "interrupted"}:
                        continue
                    checkpoint = run.get("latest_checkpoint")
                    if (
                        int(run.get("attempt", 0)) >= args.max_resume_attempts
                        or (checkpoint is not None and not Path(checkpoint).is_file())
                        or (checkpoint is None and float(run.get("progress", 0)) > 0)
                    ):
                        raise RuntimeError(f"pilot cannot safely resume: {run['run_key']}")
                    service.resume(run["id"])
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            print("Pilot executor detached; the existing durable worker retains its claim.", file=sys.stderr)
            return 130
        finally:
            service.shutdown()
        root = args.evidence_root.resolve() / matrix_id
        _write_json(root / "pilot_protocol.json", pilot)
        summary = _evaluate(registry, matrix_id, runs, design, parent, pilot,
                            root, args.device, runtime)
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> int:
    return execute(build_parser().parse_args(), _runtime())


if __name__ == "__main__":
    raise SystemExit(main())
