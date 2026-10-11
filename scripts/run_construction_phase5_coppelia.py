# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.construction.coppelia_dynamic import (
    DynamicCoppeliaConfig,
    DynamicCoppeliaExecutor,
)
from embodied_skill_composer.construction.coppelia_phase5 import (
    Phase5FullCottageRunner,
    Phase5PhysicalYardManifest,
    Phase5RunResult,
    Phase5Scenario,
    prepare_phase5_physical_yard,
    write_phase5_artifact_bundle,
)
from embodied_skill_composer.construction.intelligence_models import ScenarioManifest
from embodied_skill_composer.construction.runtime import load_house_design
from embodied_skill_composer.construction.scenarios import generate_cottage_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manually approval-gated Construction Intelligence v1 "
            "full-cottage Coppelia evidence scenarios."
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=WORKSPACE / "configs" / "construction" / "cottage_v1.yaml",
    )
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument(
        "--scenario",
        choices=("nominal", "unavailable-robot-recovery"),
        required=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--robot-model", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE / "logs" / "construction_intelligence" / "coppelia_phase5",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm that CoppeliaSim is running and this live evidence run is approved.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.yes:
        print(
            "Phase 5 live evidence is approval-gated and no offline result will be "
            "substituted. Start CoppeliaSim with the ZeroMQ remote API on port "
            f"{args.port}, then re-run with --yes."
        )
        return 2

    try:
        source_commit = _git_output("rev-parse", "--verify", "HEAD")
        source_status = _git_output(
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        source_diff = _git_output("diff", "--binary", "HEAD")
    except RuntimeError as exc:
        print(f"Phase 5 requires a verifiable Git source identity: {exc}")
        return 2
    if re.fullmatch(r"[0-9a-f]{40,64}", source_commit) is None:
        print("Phase 5 requires a full hexadecimal Git commit identity.")
        return 2
    if source_status:
        print(
            "Phase 5 release evidence requires a clean source worktree. "
            "Commit or remove every tracked and untracked change before the live run."
        )
        return 2
    source_tree_digest = hashlib.sha256(
        (
            source_commit
            + "\0"
            + source_status
            + "\0"
            + source_diff
            + "\0"
            + _untracked_content_fingerprint()
        ).encode()
    ).hexdigest()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-{args.scenario}-s{args.seed}"
    run_dir = args.output_root.resolve() / run_id

    design = load_house_design(args.design)
    generated_scenario = generate_cottage_scenario(args.seed, design)
    scenario, physical_yard = prepare_phase5_physical_yard(generated_scenario)
    config = DynamicCoppeliaConfig(
        host=args.host,
        port=args.port,
        planned_robot_footprint_radius_m=(physical_yard.configuration.robot_footprint_radius_m),
    )
    if args.robot_model is not None:
        config.robot_model_path = str(args.robot_model.resolve())
    executor = DynamicCoppeliaExecutor(scenario.plan, config=config)
    try:
        executor.connect()
    except Exception as exc:
        failure_path = _persist_gate_failure(
            run_dir,
            status="offline_or_connection_failed",
            phase="connection",
            scenario_name=args.scenario,
            scenario_seed=args.seed,
            source_commit=source_commit,
            source_tree_digest=source_tree_digest,
            executor=executor,
            error=exc,
            required_resource=(
                "A running CoppeliaSim instance with the ZeroMQ remote API reachable at "
                f"{args.host}:{args.port} and the configured YouBot model installed."
            ),
        )
        print(
            f"CoppeliaSim was not reachable; no live evidence was claimed. "
            f"Failure record: {failure_path}"
        )
        return 3

    phase5_scenario: Phase5Scenario = (
        "unavailable_robot_recovery" if args.scenario == "unavailable-robot-recovery" else "nominal"
    )
    return _run_phase5_evidence(
        executor,
        scenario,
        physical_yard,
        phase5_scenario,
        run_dir=run_dir,
        run_id=run_id,
        source_commit=source_commit,
        source_tree_digest=source_tree_digest,
    )


def _run_phase5_evidence(
    executor: DynamicCoppeliaExecutor,
    scenario: ScenarioManifest,
    physical_yard: Phase5PhysicalYardManifest,
    phase5_scenario: Phase5Scenario,
    *,
    run_dir: Path,
    run_id: str,
    source_commit: str,
    source_tree_digest: str,
) -> int:
    """Retain bounded diagnostics even when runtime attestation cannot finalize."""
    result: Phase5RunResult | None = None
    phase = "runner_initialization"
    try:
        runner = Phase5FullCottageRunner(
            executor,
            scenario,
            evidence_kind="live_coppelia",
            physical_yard=physical_yard,
        )
        phase = "runner_execution"
        result = runner.run(phase5_scenario)
        phase = "artifact_write_or_verification"
        # A finalized runtime origin can prove a measured failed attempt too.
        # Preserve its native diagnostic artifacts without promoting its gate.
        manifest = write_phase5_artifact_bundle(
            run_dir,
            run_id=run_id,
            result=result,
            scenario=scenario,
            executor=executor,
            source_commit=source_commit,
            source_tree_digest=source_tree_digest,
            source_dirty=False,
            approval_gate_confirmed=True,
            simulator_version=_simulator_version(executor.sim),
        )
        if not manifest.live_evidence:
            raise ValueError("artifact writer did not accept the result as live evidence")
    except Exception as exc:
        failure_path = _persist_gate_failure(
            run_dir,
            status="runtime_or_artifact_failed",
            phase=phase,
            scenario_name=phase5_scenario,
            scenario_seed=scenario.seed,
            source_commit=source_commit,
            source_tree_digest=source_tree_digest,
            executor=executor,
            result=result,
            error=exc,
        )
        print(f"Phase 5 evidence was not accepted. Diagnostic record: {failure_path}")
        return 1
    if result.status != "completed" or not result.live_gate_passed:
        failure_path = _persist_gate_failure(
            run_dir,
            status="run_failed",
            phase="runner_execution",
            scenario_name=phase5_scenario,
            scenario_seed=scenario.seed,
            source_commit=source_commit,
            source_tree_digest=source_tree_digest,
            executor=executor,
            result=result,
        )
        print(f"Phase 5 live gate failed; measured native artifacts retained at {run_dir}.")
        print(f"Diagnostic record: {failure_path}")
        return 1
    print(f"Phase 5 evidence bundle written to {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Live gate passed: {result.live_gate_passed}")
    return 0 if result.live_gate_passed and manifest.live_evidence else 1


def _persist_gate_failure(
    run_dir: Path,
    *,
    status: str,
    phase: str,
    scenario_name: str,
    scenario_seed: int,
    source_commit: str,
    source_tree_digest: str,
    executor: DynamicCoppeliaExecutor,
    result: Phase5RunResult | None = None,
    error: Exception | None = None,
    required_resource: str | None = None,
) -> Path:
    """Write a non-release diagnostic; never mint or repair live attestation."""
    acceptance = dict(result.acceptance) if result is not None else {}
    trace_errors: deque[dict[str, object]] = deque(maxlen=32)
    trace_error_count = 0
    if result is not None:
        for event in result.trace:
            if not isinstance(event.get("error"), str):
                continue
            trace_error_count += 1
            diagnostic: dict[str, object] = {
                key: value[:8192]
                for key in ("event", "error_type", "error")
                if isinstance(value := event.get(key), str)
            }
            timestamp = event.get("timestamp_s")
            if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
                diagnostic["timestamp_s"] = timestamp
            trace_errors.append(diagnostic)
    failure = {
        "schema_version": "construction_intelligence.coppelia_gate_failure.v1",
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": status,
        "phase": phase,
        "live_evidence": False,
        "live_gate_passed": False,
        "release_eligible": False,
        "scenario": scenario_name,
        "scenario_seed": scenario_seed,
        "source_commit": source_commit,
        "source_tree_digest": source_tree_digest,
        "source_dirty": False,
        "runner_status": result.status if result is not None else None,
        "runner_error_type": result.error_type if result is not None else None,
        "runner_error": result.error[:8192] if result is not None and result.error else None,
        "error_type": (
            type(error).__name__
            if error is not None
            else result.error_type
            if result is not None
            else None
        ),
        "error": (
            str(error)[:8192]
            if error is not None
            else result.error[:8192]
            if result is not None and result.error
            else None
        ),
        "trace_errors": list(trace_errors),
        "trace_error_count": trace_error_count,
        "trace_errors_truncated": trace_error_count > len(trace_errors),
        "acceptance": acceptance,
        "failed_gates": sorted(name for name, passed in acceptance.items() if not passed),
        "command_count": len(executor.commands),
        "telemetry_count": len(executor.telemetry),
        "installed_module_ids": result.installed_module_ids if result is not None else [],
        "expected_module_count": result.expected_module_count if result is not None else None,
        "replay_job_count": len(result.replay) if result is not None else 0,
        "required_resource": required_resource,
        "limitation": "Diagnostic only. This record is not a native or release evidence bundle.",
    }
    # Native manifests bind an exact inventory. Never append diagnostics to a
    # complete bundle, even when its measured result failed acceptance.
    diagnostic_dir = (
        run_dir.with_name(f"{run_dir.name}-diagnostics")
        if (run_dir / "manifest.json").exists()
        else run_dir
    )
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostic_dir / "gate_failure.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(failure, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "Git command failed without diagnostics"
        raise RuntimeError(f"git {' '.join(args)}: {detail}")
    return completed.stdout.strip()


def _simulator_version(sim: object) -> str | None:
    get_string = getattr(sim, "getStringParam", None)
    version_parameter = getattr(sim, "stringparam_application_version", None)
    if callable(get_string) and version_parameter is not None:
        try:
            version = get_string(version_parameter)
        except Exception:
            version = None
        if isinstance(version, str) and version.strip():
            return version.strip()

    get_integer = getattr(sim, "getInt32Param", None)
    program_version_parameter = getattr(sim, "intparam_program_version", None)
    program_revision_parameter = getattr(sim, "intparam_program_revision", None)
    if (
        not callable(get_integer)
        or program_version_parameter is None
        or program_revision_parameter is None
    ):
        return None
    try:
        program_version = get_integer(program_version_parameter)
        program_revision = get_integer(program_revision_parameter)
    except Exception:
        return None
    if (
        not isinstance(program_version, int)
        or isinstance(program_version, bool)
        or program_version <= 0
        or not isinstance(program_revision, int)
        or isinstance(program_revision, bool)
        or program_revision < 0
    ):
        return None
    major = program_version // 10_000
    minor = (program_version // 100) % 100
    patch = program_version % 100
    return f"{major}.{minor}.{patch} rev {program_revision}"


def _untracked_content_fingerprint() -> str:
    paths = _git_output("ls-files", "--others", "--exclude-standard").splitlines()
    records: list[str] = []
    for relative in sorted(paths):
        path = (WORKSPACE / relative).resolve()
        try:
            path.relative_to(WORKSPACE.resolve())
        except ValueError:
            continue
        if path.is_file():
            records.append(f"{relative}\0{hashlib.sha256(path.read_bytes()).hexdigest()}")
    return "\n".join(records)


if __name__ == "__main__":
    raise SystemExit(main())
