from __future__ import annotations

import json
import subprocess
from pathlib import Path

from embodied_skill_composer.construction.models import BuildPlan


DEFAULT_BLENDER_PATH = Path(
    "C:/Program Files/Blender Foundation/Blender 5.1/blender.exe"
)


def generate_blender_assets(
    plan: BuildPlan,
    output_dir: Path,
    *,
    blender_path: Path = DEFAULT_BLENDER_PATH,
    timeout_s: int = 180,
) -> dict[str, Path]:
    if not blender_path.is_file():
        raise FileNotFoundError(
            f"Blender was not found at {blender_path}. Core planning does not require Blender."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "build_plan.json"
    plan_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2), encoding="utf-8"
    )
    script_path = Path(__file__).with_name("blender_generate.py")
    command = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--plan",
        str(plan_path),
        "--output",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    (output_dir / "blender.log").write_text(
        result.stdout + "\n" + result.stderr, encoding="utf-8"
    )
    if result.returncode != 0 or "Traceback (most recent call last)" in result.stderr:
        raise RuntimeError(
            f"Blender generation failed with exit code {result.returncode}; "
            f"see {output_dir / 'blender.log'}"
        )
    artifacts = {
        "blend": output_dir / "house.blend",
        "glb": output_dir / "house.glb",
        "assembled_preview": output_dir / "assembled_preview.png",
        "exploded_preview": output_dir / "exploded_modules.png",
        "manifest": output_dir / "geometry_manifest.json",
    }
    missing = [str(path) for path in artifacts.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Blender did not produce expected artifacts: {missing}")
    return artifacts
