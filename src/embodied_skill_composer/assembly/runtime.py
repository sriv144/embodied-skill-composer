from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from embodied_skill_composer.assembly.models import (
    AssemblyRuntimeProfile,
    AssemblyScenarioConfig,
    AssetCatalog,
    ModularBlueprint,
    TrainingConfig,
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_runtime_profile(path: Path | None = None) -> AssemblyRuntimeProfile:
    if path is None:
        return AssemblyRuntimeProfile()
    return AssemblyRuntimeProfile.model_validate(load_yaml(path))


def load_assembly_scenario(path: Path) -> AssemblyScenarioConfig:
    return AssemblyScenarioConfig.model_validate(load_yaml(path))


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig.model_validate(load_yaml(path))


def load_modular_blueprint(path: Path) -> ModularBlueprint:
    return ModularBlueprint.model_validate(load_yaml(path))


def load_asset_catalog(path: Path) -> AssetCatalog:
    return AssetCatalog.model_validate(load_yaml(path))
