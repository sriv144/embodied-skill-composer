from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraspPolicy:
    thresholds: dict[str, float]

    def success_threshold(self, clutter_level: int) -> float:
        key = str(max(1, min(4, clutter_level)))
        return float(self.thresholds.get(key, 0.75))

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"thresholds": self.thresholds}


def default_grasp_policy() -> GraspPolicy:
    return GraspPolicy(thresholds={"1": 0.88, "2": 0.81, "3": 0.74, "4": 0.66})


def load_grasp_policy(path: Path) -> GraspPolicy:
    if not path.exists():
        return default_grasp_policy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return GraspPolicy(thresholds={key: float(value) for key, value in payload["thresholds"].items()})


def save_grasp_policy(policy: GraspPolicy, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
