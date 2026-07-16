from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COPILOT_DIR = PROJECT_ROOT / "logs" / "copilot"
DEFAULT_REGISTRY_PATH = DEFAULT_COPILOT_DIR / "experiments.sqlite"

