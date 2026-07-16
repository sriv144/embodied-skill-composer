# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.coppelia_backend import (
    CoppeliaSimAssemblyBackend,
)
from embodied_skill_composer.assembly.runtime import (
    load_assembly_scenario,
    load_runtime_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the scripted construction task through CoppeliaSim."
    )
    parser.add_argument(
        "--env-config",
        default=str(PROJECT_ROOT / "configs" / "assembly_env.yaml"),
    )
    parser.add_argument(
        "--runtime-profile",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "assembly_profiles"
            / "coppelia_local.yaml"
        ),
    )
    parser.add_argument("--host", help="Override the profile ZeroMQ host.")
    parser.add_argument("--port", type=int, help="Override the profile ZeroMQ port.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--diagnostics-output",
        default=str(PROJECT_ROOT / "logs" / "coppelia" / "scripted_episode.json"),
    )
    parser.add_argument(
        "--image-output",
        default=str(PROJECT_ROOT / "artifacts" / "coppelia" / "scripted_final.png"),
    )
    parser.add_argument(
        "--scene-output",
        default=str(
            PROJECT_ROOT / "artifacts" / "coppelia" / "construction_workbench.ttt"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_config = load_assembly_scenario(Path(args.env_config))
    runtime_profile = load_runtime_profile(Path(args.runtime_profile))
    if args.host or args.port:
        runtime_profile = runtime_profile.model_copy(
            update={
                "coppelia": runtime_profile.coppelia.model_copy(
                    update={
                        "host": args.host or runtime_profile.coppelia.host,
                        "port": args.port or runtime_profile.coppelia.port,
                    }
                )
            }
        )
    backend = build_assembly_backend(
        config=env_config,
        runtime_profile=runtime_profile,
        seed=args.seed,
    )
    if not isinstance(backend, CoppeliaSimAssemblyBackend):
        print(
            f"Runtime profile '{runtime_profile.name}' must use backend 'coppelia_sim'."
        )
        return 1
    status = backend.get_backend_status()
    if not status.is_ready:
        print("CoppeliaSim backend is not ready.")
        for note in status.readiness_notes:
            print(f"- {note}")
        return 1

    try:
        backend.reset(seed=args.seed)
        done = False
        while not done:
            result = backend.execute_team_option(backend.scripted_team_option())
            done = result.done

        artifact = backend.build_artifact(policy_mode="scripted")
        image_path = backend.capture_camera(
            Path(args.image_output),
            camera_name="topdown",
        )
        diagnostics = backend.get_option_episode_diagnostics()
        scene_path = backend.save_scene(Path(args.scene_output))
    finally:
        backend.close()

    diagnostics_path = Path(args.diagnostics_output).resolve()
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        json.dumps(
            {
                "artifact": artifact.model_dump(mode="json"),
                "diagnostics": diagnostics,
                "image_path": str(image_path),
                "scene_path": str(scene_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Runtime profile: {runtime_profile.name} ({runtime_profile.backend})")
    print(f"Success: {artifact.metrics.success}")
    print(
        "Beams installed: "
        f"{artifact.metrics.beams_installed}/{artifact.metrics.total_beams}"
    )
    print(f"Steps: {artifact.metrics.step_count}")
    print(f"Simulation steps: {diagnostics['coppelia_sim']['simulation_step_count']}")
    print(f"Final image: {image_path}")
    print(f"Scene: {scene_path}")
    print(f"Diagnostics: {diagnostics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
