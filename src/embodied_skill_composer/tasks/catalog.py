from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from embodied_skill_composer.core.models import TaskSpec


class _YamlModule(Protocol):
    def safe_load(self, stream: str) -> object: ...


yaml = cast(_YamlModule, import_module("yaml"))


def load_tasks(config_path: Path) -> dict[str, TaskSpec]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"task catalog must be a mapping: {config_path}")
    tasks = data.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError(f"task catalog 'tasks' must be a mapping: {config_path}")
    result: dict[str, TaskSpec] = {}
    for name, payload in tasks.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            raise ValueError(f"task catalog entries must be named mappings: {config_path}")
        result[name] = TaskSpec.model_validate({"name": name, **payload})
    return result
