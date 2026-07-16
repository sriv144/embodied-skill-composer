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
from embodied_skill_composer.assembly.brain import (
    HeuristicConstructionBrain,
    ScriptedConstructionBrain,
    run_construction_brain_episode,
)
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a layered construction brain against an assembly backend."
    )
    parser.add_argument("--brain", choices=["scripted", "heuristic"], default="heuristic")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--env-config",
        default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"),
    )
    parser.add_argument(
        "--runtime-profile",
        default=str(PROJECT_ROOT / "configs" / "assembly_profiles" / "local_dev.yaml"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")

    config = load_assembly_scenario(Path(args.env_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    brain_type = (
        ScriptedConstructionBrain
        if args.brain == "scripted"
        else HeuristicConstructionBrain
    )
    episodes = []
    for episode_index in range(args.episodes):
        env = build_assembly_backend(
            config=config,
            runtime_profile=runtime_profile,
            seed=args.seed + episode_index,
        )
        result = run_construction_brain_episode(
            env=env,
            brain=brain_type(),
            seed=args.seed + episode_index,
        )
        episodes.append(result)

    success_rate = sum(int(item.artifact.metrics.success) for item in episodes) / len(episodes)
    mean_completion = sum(
        item.artifact.metrics.structure_completion_rate for item in episodes
    ) / len(episodes)
    mean_sensor_safety_holds = sum(
        int(item.diagnostics.get("construction_brain", {}).get("sensor_safety_hold_count", 0))
        for item in episodes
    ) / len(episodes)
    mean_visual_safety_holds = sum(
        int(item.diagnostics.get("construction_brain", {}).get("visual_safety_hold_count", 0))
        for item in episodes
    ) / len(episodes)
    mean_terminal_safety_holds = sum(
        int(item.diagnostics.get("construction_brain", {}).get("terminal_safety_hold_count", 0))
        for item in episodes
    ) / len(episodes)
    visual_diagnostics = [
        value
        for item in episodes
        if isinstance(
            value := item.diagnostics.get("mujoco_visual_perception"),
            dict,
        )
        and value.get("enabled")
    ]
    mean_visual_position_error = (
        None
        if not visual_diagnostics
        else sum(float(item["mean_position_error_m"]) for item in visual_diagnostics)
        / len(visual_diagnostics)
    )
    mean_visual_samples = (
        0.0
        if not visual_diagnostics
        else sum(int(item["sample_count"]) for item in visual_diagnostics)
        / len(visual_diagnostics)
    )
    final_visible_blueprint_recall = [
        float(item["latest_evaluation"]["visible_recall_by_category"]["blueprint_cell"])
        for item in visual_diagnostics
        if item.get("latest_evaluation")
    ]
    final_tracked_blueprint_recall = [
        float(item["latest_evaluation"]["recall_by_category"]["blueprint_cell"])
        for item in visual_diagnostics
        if item.get("latest_evaluation")
    ]
    backend_status = episodes[0].diagnostics.get("backend_status", {})
    backend_ready = bool(
        isinstance(backend_status, dict) and backend_status.get("is_ready", False)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        PROJECT_ROOT / "logs" / "construction_brain" / f"{args.brain}-{timestamp}.json"
    )
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "brain": episodes[0].brain_name,
        "runtime_profile": runtime_profile.model_dump(mode="json"),
        "backend_status": backend_status,
        "episode_count": len(episodes),
        "success_rate": success_rate,
        "mean_structure_completion_rate": mean_completion,
        "mean_sensor_safety_holds": mean_sensor_safety_holds,
        "mean_visual_safety_holds": mean_visual_safety_holds,
        "mean_terminal_safety_holds": mean_terminal_safety_holds,
        "mean_visual_position_error_m": mean_visual_position_error,
        "mean_visual_samples_per_episode": mean_visual_samples,
        "mean_final_visible_blueprint_recall": (
            None
            if not final_visible_blueprint_recall
            else sum(final_visible_blueprint_recall) / len(final_visible_blueprint_recall)
        ),
        "mean_final_tracked_blueprint_recall": (
            None
            if not final_tracked_blueprint_recall
            else sum(final_tracked_blueprint_recall) / len(final_tracked_blueprint_recall)
        ),
        "episodes": [item.model_dump(mode="json") for item in episodes],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Brain: {episodes[0].brain_name}")
    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Backend ready: {backend_ready}")
    if not backend_ready:
        print("Backend note: execution used logical task semantics without a ready simulator model.")
    print(f"Episodes: {len(episodes)}")
    print(f"Success rate: {success_rate:.3f}")
    print(f"Mean structure completion: {mean_completion:.3f}")
    print(f"Mean sensor safety holds: {mean_sensor_safety_holds:.2f}")
    print(f"Mean visual safety holds: {mean_visual_safety_holds:.2f}")
    print(f"Mean terminal safety holds: {mean_terminal_safety_holds:.2f}")
    if mean_visual_position_error is not None:
        print(f"Mean visual position error: {mean_visual_position_error:.4f} m")
        print(f"Mean visual samples per episode: {mean_visual_samples:.2f}")
        print(
            "Mean final blueprint recall: "
            f"visible={sum(final_visible_blueprint_recall) / len(final_visible_blueprint_recall):.3f}, "
            f"tracked={sum(final_tracked_blueprint_recall) / len(final_tracked_blueprint_recall):.3f}"
        )
    print(f"Artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
