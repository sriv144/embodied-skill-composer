from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from embodied_skill_composer.core.models import TaskSpec


def load_tasks(config_path: Path) -> dict[str, TaskSpec]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", {})
    return {name: TaskSpec(name=name, **payload) for name, payload in tasks.items()}
