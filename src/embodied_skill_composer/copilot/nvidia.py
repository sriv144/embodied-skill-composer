from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from embodied_skill_composer.assembly.backends import build_assembly_backend
from embodied_skill_composer.assembly.gpu import inspect_gpu_runtime
from embodied_skill_composer.assembly.runtime import load_assembly_scenario, load_runtime_profile
from embodied_skill_composer.copilot.paths import PROJECT_ROOT


def run_nvidia_readiness_check(
    runtime_profile_path: Path | None = None,
    env_config_path: Path | None = None,
    aiq_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    profile_path = runtime_profile_path or PROJECT_ROOT / "configs" / "assembly_profiles" / "local_gpu.yaml"
    env_path = env_config_path or PROJECT_ROOT / "configs" / "assembly_env.yaml"
    runtime_profile = load_runtime_profile(profile_path)
    env_config = load_assembly_scenario(env_path)
    gpu_status = inspect_gpu_runtime(runtime_profile)

    isaac_profile = load_runtime_profile(PROJECT_ROOT / "configs" / "assembly_profiles" / "isaac_gpu.yaml")
    isaac_backend = build_assembly_backend(env_config, isaac_profile)
    backend_status = isaac_backend.get_backend_status()

    return {
        "runtime": gpu_status.model_dump(mode="json"),
        "isaac_backend": backend_status.model_dump(mode="json"),
        "aiq": check_aiq_health(aiq_url),
    }


def check_aiq_health(aiq_url: str, timeout_seconds: float = 2.0) -> dict[str, Any]:
    endpoints = [f"{aiq_url.rstrip('/')}/health", f"{aiq_url.rstrip('/')}/v1/health"]
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            with urllib.request.urlopen(endpoint, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
                parsed: object
                try:
                    parsed = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    parsed = body
                return {
                    "reachable": True,
                    "url": endpoint,
                    "status_code": response.status,
                    "response": parsed,
                }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{endpoint}: {exc}")
    return {
        "reachable": False,
        "base_url": aiq_url,
        "errors": errors,
        "notes": ["No AI-Q backend was started; this is a readiness probe only."],
    }

