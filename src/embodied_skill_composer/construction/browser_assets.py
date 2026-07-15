from __future__ import annotations

import subprocess
from pathlib import Path

from embodied_skill_composer.construction.blender import DEFAULT_BLENDER_PATH


def generate_browser_robot_asset(
    output_path: Path,
    *,
    blender_path: Path = DEFAULT_BLENDER_PATH,
    timeout_s: int = 120,
) -> Path:
    """Generate the project's redistributable procedural construction rover."""
    if not blender_path.is_file():
        raise FileNotFoundError(f"Blender was not found at {blender_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).with_name("blender_robot_generate.py")
    command = [
        str(blender_path),
        "--background",
        "--python",
        str(script_path),
        "--",
        "--output",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    log_path = output_path.with_suffix(".log")
    if result.returncode != 0 or "Traceback (most recent call last)" in result.stderr:
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        raise RuntimeError(
            f"Blender robot generation failed with exit code {result.returncode}; "
            f"see {log_path}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"Blender did not produce {output_path}")
    log_path.unlink(missing_ok=True)
    return output_path
