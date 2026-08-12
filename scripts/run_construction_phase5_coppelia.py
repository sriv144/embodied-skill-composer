# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
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
    Phase5Scenario,
    prepare_phase5_physical_yard,
    write_phase5_artifact_bundle,
)
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
    scenario, physical_yard = prepare_phase5_physical_yard(
        generated_scenario
    )
    config = DynamicCoppeliaConfig(
        host=args.host,
        port=args.port,
        planned_robot_footprint_radius_m=(
            physical_yard.configuration.robot_footprint_radius_m
        ),
    )
    if args.robot_model is not None:
        config.robot_model_path = str(args.robot_model.resolve())
    executor = DynamicCoppeliaExecutor(scenario.plan, config=config)
    try:
        executor.connect()
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=False)
        failure = {
            "schema_version": "construction_intelligence.coppelia_gate_failure.v1",
            "status": "offline_or_connection_failed",
            "live_evidence": False,
            "scenario": args.scenario,
            "scenario_seed": args.seed,
            "source_commit": source_commit,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "required_resource": (
                "A running CoppeliaSim instance with the ZeroMQ remote API reachable at "
                f"{args.host}:{args.port} and the configured YouBot model installed."
            ),
        }
        (run_dir / "gate_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"CoppeliaSim was not reachable; no live evidence was claimed. "
            f"Failure record: {run_dir / 'gate_failure.json'}"
        )
        return 3

    phase5_scenario: Phase5Scenario = (
        "unavailable_robot_recovery"
        if args.scenario == "unavailable-robot-recovery"
        else "nominal"
    )
    runner = Phase5FullCottageRunner(
        executor,
        scenario,
        evidence_kind="live_coppelia",
        physical_yard=physical_yard,
    )
    result = runner.run(phase5_scenario)
    manifest = write_phase5_artifact_bundle(
        run_dir,
        run_id=run_id,
        result=result,
        scenario=scenario,
        executor=executor,
        source_commit=source_commit,
        source_tree_digest=source_tree_digest,
        source_dirty=bool(source_status),
        approval_gate_confirmed=True,
        simulator_version=_simulator_version(executor.sim),
    )
    print(f"Phase 5 evidence bundle written to {run_dir}")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Live gate passed: {result.live_gate_passed}")
    return 0 if result.live_gate_passed and manifest.live_evidence else 1


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
