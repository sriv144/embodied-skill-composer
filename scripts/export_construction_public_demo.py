from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Literal, cast

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.compiler import compile_house_design  # noqa: E402
from embodied_skill_composer.construction.public_demo_provenance import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    SourceIdentity,
    export_public_demo_bundle,
    verify_public_demo_export,
    verify_public_demo_regeneration_identity,
)
from embodied_skill_composer.construction.release_identity import (  # noqa: E402
    repository_release_identity,
    validate_release_identity,
)
from embodied_skill_composer.construction.recovery import (  # noqa: E402
    Disruption,
    inject_disruption,
)
from embodied_skill_composer.construction.reporting import (  # noqa: E402
    render_research_report,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402
from embodied_skill_composer.construction.scheduler import compare_controllers  # noqa: E402
from embodied_skill_composer.construction.trace import build_execution_trace  # noqa: E402
from embodied_skill_composer.construction.training import source_fingerprint  # noqa: E402


def export_public_demo(
    output_dir: Path,
    *,
    regenerate_robot: bool = True,
    channel: Literal["preview", "release"] = "preview",
    deterministic_bundle: Path | None = None,
    research_bundle: Path | None = None,
    simulator_bundle: Path | None = None,
    source: SourceIdentity | None = None,
    release_version: str | None = None,
    release_tag: str | None = None,
) -> dict[str, object]:
    """Export a fixture preview or package explicit canonical evidence."""

    current_source = _current_source()
    if channel == "release" and source is not None and source != current_source:
        raise ValueError(
            "explicit release source identity does not match the current Git worktree"
        )
    resolved_source = source or current_source
    if deterministic_bundle is not None:
        return export_public_demo_bundle(
            output_dir,
            deterministic_bundle=deterministic_bundle,
            research_bundle=research_bundle,
            simulator_bundle=simulator_bundle,
            source=resolved_source,
            channel=channel,
            release_version=release_version,
            release_tag=release_tag,
        )
    if channel == "release":
        raise ValueError(
            "release mode requires --deterministic-bundle, --research-bundle, "
            "and --simulator-bundle"
        )
    with tempfile.TemporaryDirectory(prefix="construction-public-preview-") as raw:
        reusable_robot = output_dir.resolve() / "construction_robot.glb"
        fixture_manifest = _build_fixture_bundle(
            Path(raw),
            regenerate_robot=regenerate_robot,
            reusable_robot=(
                reusable_robot
                if not regenerate_robot and reusable_robot.is_file()
                else None
            ),
            source=resolved_source,
        )
        return export_public_demo_bundle(
            output_dir,
            deterministic_bundle=fixture_manifest,
            source=resolved_source,
            channel="preview",
            release_version=release_version,
            release_tag=release_tag,
        )


def _build_fixture_bundle(
    root: Path,
    *,
    regenerate_robot: bool,
    reusable_robot: Path | None,
    source: SourceIdentity,
) -> Path:
    """Materialize the reviewed deterministic fixture without making research claims."""

    design_path = WORKSPACE / "configs" / "construction" / "cottage_v1.yaml"
    design = load_house_design(design_path)
    plan = compile_house_design(design)
    schedules = compare_controllers(plan)
    traces = {
        name: build_execution_trace(plan, schedule)
        for name, schedule in schedules.items()
    }
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    house_source = (
        WORKSPACE / "artifacts" / "construction_v2" / "cottage_v1" / "house.glb"
    )
    if not house_source.is_file():
        raise FileNotFoundError(
            "Generate the cottage first with scripts/generate_construction_assets.py"
        )
    shutil.copyfile(house_source, root / "house.glb")
    robot_path = root / "construction_robot.glb"
    canonical_robot = (
        WORKSPACE / "workbench" / "public" / "demo" / "construction_robot.glb"
    )
    robot_source = (
        canonical_robot
        if regenerate_robot or reusable_robot is None
        else reusable_robot
    )
    if not robot_source.is_file():
        raise FileNotFoundError(
            "The reviewed canonical construction robot asset is missing: "
            f"{robot_source}"
        )
    shutil.copyfile(robot_source, robot_path)

    sequential = schedules["sequential"].makespan_s
    optimized = schedules["optimized"].makespan_s
    project = {
        "design": design.model_dump(mode="json"),
        "plan": plan.model_dump(mode="json"),
        "controllers": {
            name: trace.metrics.model_dump(mode="json")
            for name, trace in traces.items()
        },
        "optimized_improvement_percent": round(
            100 * (1 - optimized / sequential), 1
        ),
        "geometry_asset_url": "house.glb",
        "robot_asset_url": "construction_robot.glb",
    }
    _write_json(root / "project.json", project)
    for name, trace in traces.items():
        _write_json(trace_dir / f"{name}.json", trace.model_dump(mode="json"))
    recovery = inject_disruption(
        plan,
        schedules["optimized"],
        Disruption(failure_type="obstacle", timestamp_s=72),
    )
    _write_json(trace_dir / "recovery.json", recovery.model_dump(mode="json"))
    _write_json(
        root / "scenarios.json",
        [
            {
                "id": design.design_id,
                "seed": None,
                "split": "fixture",
                "payload": {
                    "module_count": len(plan.modules),
                    "title": design.title,
                },
                "created_at": "2026-07-15T00:00:00Z",
            }
        ],
    )
    _write_json(root / "policies.json", [])
    _write_json(
        root / "runs.json",
        [
            {
                "id": "fixture_baseline_evaluation",
                "kind": "evaluation",
                "status": "completed",
                "config": {
                    "controllers": ["sequential", "greedy", "cp_sat"],
                    "scope": "deterministic_cottage_fixture",
                },
                "created_at": "2026-07-15T00:00:00Z",
                "started_at": "2026-07-15T00:00:00Z",
                "ended_at": "2026-07-15T00:00:01Z",
                "progress": 1.0,
                "artifact_dir": "demo",
                "error": None,
            }
        ],
    )
    report = render_research_report(plan, traces)
    report += (
        "\n## Public Demo Provenance\n\n"
        "This is a reviewed deterministic fixture preview. It contains no "
        "canonical MAPPO/IPPO research result and no live Coppelia evidence. "
        "The generated `release-status.json` and `provenance.json` make those "
        "boundaries machine-readable.\n"
    )
    (root / "report.md").write_text(
        report,
        encoding="utf-8",
        newline="\n",
    )

    role_targets = {
        "project": "project.json",
        "scenarios": "scenarios.json",
        "policies": "policies.json",
        "runs": "runs.json",
        "report": "report.md",
        "house": "house.glb",
        "robot": "construction_robot.glb",
        "trace_sequential": "traces/sequential.json",
        "trace_greedy": "traces/greedy.json",
        "trace_optimized": "traces/optimized.json",
        "trace_recovery": "traces/recovery.json",
    }
    artifacts = [
        {
            "role": role,
            "path": target,
            "target": target,
            "sha256": _sha256(root / target),
            "media_type": _media_type(target),
        }
        for role, target in sorted(role_targets.items())
    ]
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": "deterministic",
        "evidence_status": "fixture",
        "source": source.model_dump(mode="json"),
        "created_at": "2026-07-15T00:00:00Z",
        "configuration_digests": [_sha256(design_path)],
        "protocol_digest": None,
        "profile": None,
        "matrix_id": None,
        "artifacts": artifacts,
    }
    manifest_path = root / "fixture-bundle.json"
    _write_json(manifest_path, manifest)
    return manifest_path


def _current_source() -> SourceIdentity:
    payload = source_fingerprint()
    return SourceIdentity(
        commit=str(payload["commit"]),
        dirty=bool(payload["dirty"]),
        tree_digest=str(payload["tree_digest"]),
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_type(path: str) -> str:
    suffix = Path(path).suffix
    return {
        ".glb": "model/gltf-binary",
        ".json": "application/json",
        ".md": "text/markdown",
    }.get(suffix, "application/octet-stream")


def _explicit_source(args: argparse.Namespace) -> SourceIdentity | None:
    values = (
        args.source_commit,
        args.source_dirty,
        args.source_tree_digest,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "--source-commit, --source-dirty, and --source-tree-digest "
            "must be supplied together"
        )
    return SourceIdentity(
        commit=cast(str, args.source_commit),
        dirty=cast(bool, args.source_dirty),
        tree_digest=cast(str, args.source_tree_digest),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a hash-pinned read-only public demo. Preview mode may build "
            "the reviewed fixture; release mode requires all canonical bundles."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE / "workbench" / "public" / "demo",
    )
    parser.add_argument(
        "--channel",
        choices=("preview", "release"),
        default="preview",
    )
    parser.add_argument("--deterministic-bundle", type=Path)
    parser.add_argument("--research-bundle", type=Path)
    parser.add_argument("--simulator-bundle", type=Path)
    parser.add_argument("--source-commit")
    dirty_group = parser.add_mutually_exclusive_group()
    dirty_group.add_argument(
        "--source-dirty",
        action="store_true",
        dest="source_dirty",
    )
    dirty_group.add_argument(
        "--source-clean",
        action="store_false",
        dest="source_dirty",
    )
    parser.set_defaults(source_dirty=None)
    parser.add_argument("--source-tree-digest")
    parser.add_argument("--release-version")
    parser.add_argument("--release-tag")
    parser.add_argument("--reuse-robot", action="store_true")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-hash an existing export instead of writing it.",
    )
    parser.add_argument(
        "--expected-channel",
        choices=("preview", "release"),
        help="With --verify-only, require the recorded channel.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        help=(
            "With --verify-only, require a second verified export to have the "
            "same canonical regeneration identity."
        ),
    )
    args = parser.parse_args()
    if args.verify_only:
        verify_public_demo_export(
            args.output,
            expected_channel=cast(
                Literal["preview", "release"] | None,
                args.expected_channel,
            ),
            expected_source=_explicit_source(args),
            expected_release_version=args.release_version,
            expected_release_tag=args.release_tag,
        )
        if args.compare_output is not None:
            verify_public_demo_regeneration_identity(
                args.compare_output,
                args.output,
            )
        print(args.output)
        return
    if args.expected_channel is not None or args.compare_output is not None:
        parser.error("--expected-channel and --compare-output require --verify-only")
    release_version = cast(str | None, args.release_version)
    release_tag = cast(str | None, args.release_tag)
    if args.channel == "release":
        try:
            repository_identity = repository_release_identity(WORKSPACE)
            if release_version is None and release_tag is None:
                release_version = repository_identity.version
                release_tag = repository_identity.tag
            elif release_version is None or release_tag is None:
                parser.error(
                    "--release-version and --release-tag must be supplied together"
                )
            else:
                validate_release_identity(release_version, release_tag)
                if (
                    release_version != repository_identity.version
                    or release_tag != repository_identity.tag
                ):
                    parser.error(
                        "release version/tag do not match the repository "
                        "package versions"
                    )
        except ValueError as exc:
            parser.error(str(exc))
    provenance = export_public_demo(
        args.output,
        regenerate_robot=not args.reuse_robot,
        channel=cast(Literal["preview", "release"], args.channel),
        deterministic_bundle=args.deterministic_bundle,
        research_bundle=args.research_bundle,
        simulator_bundle=args.simulator_bundle,
        source=_explicit_source(args),
        release_version=release_version,
        release_tag=release_tag,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "channel": provenance["channel"],
                "artifact_count": provenance["artifact_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
