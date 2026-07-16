from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from embodied_skill_composer.construction.intelligence_models import (
    SkillDistribution,
    SkillProfile,
)


DEFAULT_INSTALL_DURATIONS = {
    "foundation": 16.0,
    "wall_panel": 11.0,
    "door_panel": 11.0,
    "window_panel": 11.0,
    "interior_panel": 10.0,
    "roof_panel": 18.0,
}


def skill_profile_from_mujoco_campaign(path: Path) -> SkillProfile:
    """Calibrate high-level skill randomization from MuJoCo terminal checks.

    Existing campaigns measure pass/fail, alignment, and contact force. They do
    not expose wall-clock manipulation duration, so duration remains a typed
    prior widened by the observed failure rate instead of being misrepresented
    as a measured quantity.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes", [])
    passes: list[bool] = []
    alignment_errors: list[float] = []
    peak_forces: list[float] = []
    for episode in episodes:
        for step in episode.get("steps", []):
            feedback = step.get("observation", {}).get("physical_feedback") or {}
            passed = feedback.get("last_check_passed")
            if passed is None:
                continue
            passes.append(bool(passed))
            alignment = feedback.get("current_alignment_error_m")
            if alignment is not None:
                alignment_errors.append(float(alignment))
            forces = [float(value) for value in feedback.get("last_contact_forces_n", {}).values()]
            if forces:
                peak_forces.append(max(forces))
    if not passes:
        raise ValueError(f"MuJoCo campaign contains no terminal manipulation checks: {path}")

    success_rate = sum(passes) / len(passes)
    failure_rate = 1.0 - success_rate
    alignment_mean, alignment_std = _mean_std(alignment_errors)
    force_mean, force_std = _mean_std(peak_forces)
    distributions = {}
    for module_type, prior_duration in DEFAULT_INSTALL_DURATIONS.items():
        heavy_multiplier = 1.15 if module_type in {"foundation", "roof_panel"} else 1.0
        duration_mean = prior_duration * heavy_multiplier * (1.0 + failure_rate * 0.5)
        duration_std = max(duration_mean * (0.05 + failure_rate), 0.25)
        distributions[module_type] = SkillDistribution(
            success_rate=success_rate,
            duration_mean_s=duration_mean,
            duration_std_s=duration_std,
            peak_force_mean_n=force_mean,
            peak_force_std_n=force_std,
            alignment_error_mean_m=alignment_mean,
            alignment_error_std_m=alignment_std,
            sample_count=len(passes),
        )
    profile_id = f"mujoco-{path.stem}-v1"
    return SkillProfile(
        profile_id=profile_id,
        source_backend="mujoco",
        source_artifacts=[str(path.resolve())],
        by_module_type=distributions,
        notes=[
            f"Parsed {len(passes)} MuJoCo terminal manipulation checks.",
            "Success, force, and alignment are empirical campaign statistics.",
            "Duration is a construction-module prior widened by observed failure rate; "
            "the source campaign does not record wall-clock skill duration.",
        ],
    )


def write_skill_profile(profile: SkillProfile, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return path


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0
