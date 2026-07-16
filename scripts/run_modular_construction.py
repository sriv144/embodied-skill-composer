# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.blueprint import compile_modular_blueprint
from embodied_skill_composer.assembly.brain import (
    HeuristicConstructionBrain,
    PrecedenceConstructionBrain,
    ScriptedConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.coppelia_backend import (
    CoppeliaSimAssemblyBackend,
)
from embodied_skill_composer.assembly.reporting import render_construction_lab_report
from embodied_skill_composer.assembly.runtime import (
    load_asset_catalog,
    load_modular_blueprint,
    load_runtime_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile and execute a modular construction blueprint."
    )
    parser.add_argument(
        "--blueprint",
        type=Path,
        default=PROJECT_ROOT / "configs" / "blueprints" / "modular_room_v0.yaml",
    )
    parser.add_argument(
        "--asset-catalog",
        type=Path,
        default=PROJECT_ROOT / "configs" / "construction_asset_catalog.yaml",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=(
            PROJECT_ROOT / "configs" / "assembly_profiles" / "coppelia_local.yaml"
        ),
    )
    parser.add_argument(
        "--brain",
        choices=["precedence", "heuristic", "scripted"],
        default="precedence",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-id")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blueprint = load_modular_blueprint(args.blueprint.resolve())
    catalog = load_asset_catalog(args.asset_catalog.resolve())
    compiled = compile_modular_blueprint(
        blueprint,
        catalog,
        workspace_root=PROJECT_ROOT,
    )
    runtime_profile = load_runtime_profile(args.runtime_profile.resolve())
    brain_types = {
        "precedence": PrecedenceConstructionBrain,
        "heuristic": HeuristicConstructionBrain,
        "scripted": ScriptedConstructionBrain,
    }
    run_id = args.run_id or (
        f"{blueprint.blueprint_id}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_root = args.output_root.resolve()
    log_dir = output_root / "logs" / "construction_runs" / run_id
    artifact_dir = output_root / "artifacts" / "construction_runs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = build_assembly_backend(
        config=compiled.scenario,
        runtime_profile=runtime_profile,
        seed=args.seed,
    )
    status = backend.get_backend_status()
    if runtime_profile.backend == "coppelia_sim" and not status.is_ready:
        print("CoppeliaSim backend is not ready.")
        for note in status.readiness_notes:
            print(f"- {note}")
        return 1

    recording_path: Path | None = None
    final_overview: Path | None = None
    final_topdown: Path | None = None
    scene_path: Path | None = None
    try:
        episode = run_construction_brain_episode(
            backend,
            brain_types[args.brain](),
            seed=args.seed,
        )
        if isinstance(backend, CoppeliaSimAssemblyBackend):
            if args.record:
                recording_path = backend.record_episode(
                    artifact_dir / "overview.mp4",
                    diagnostics=episode.diagnostics,
                )
            backend.focus_cameras_on_structure()
            final_overview = backend.capture_camera(
                artifact_dir / "final_overview.png",
                camera_name="overview",
            )
            final_topdown = backend.capture_camera(
                artifact_dir / "final_topdown.png",
                camera_name="topdown",
            )
            scene_path = backend.save_scene(artifact_dir / "modular_room.ttt")
            episode = episode.model_copy(
                update={"diagnostics": backend.get_option_episode_diagnostics()}
            )
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    episode_path = log_dir / "episode.json"
    episode_path.write_text(
        json.dumps(episode.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    blueprint_path = log_dir / "blueprint.json"
    blueprint_path.write_text(
        json.dumps(compiled.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    report_path = log_dir / "report.md"
    report_path.write_text(
        render_construction_lab_report(compiled, episode),
        encoding="utf-8",
    )

    metrics = episode.artifact.metrics
    print(f"Blueprint: {blueprint.blueprint_id}")
    print(f"Brain: {episode.brain_name}")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Success: {metrics.success}")
    print(f"Components installed: {metrics.beams_installed}/{metrics.total_beams}")
    print(f"Structure completion: {metrics.structure_completion_rate:.3f}")
    if recording_path is not None:
        print(f"Replay: {recording_path}")
    if final_overview is not None:
        print(f"Final overview: {final_overview}")
    if final_topdown is not None:
        print(f"Final top-down: {final_topdown}")
    if scene_path is not None:
        print(f"Scene: {scene_path}")
    print(f"Episode: {episode_path}")
    print(f"Report: {report_path}")
    return 0 if metrics.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
