from __future__ import annotations

import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SRC = WORKSPACE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from embodied_skill_composer.construction.compiler import compile_house_design  # noqa: E402
from embodied_skill_composer.construction.marl_env import (  # noqa: E402
    ConstructionCoordinationEnv,
    scripted_coordination_actions,
)
from embodied_skill_composer.construction.runtime import load_house_design  # noqa: E402


def main() -> None:
    design = load_house_design(WORKSPACE / "configs" / "construction" / "cottage_v1.yaml")
    env = ConstructionCoordinationEnv(compile_house_design(design))
    env.reset(seed=7)
    total_rewards = {agent: 0.0 for agent in env.agents}
    while env.agents:
        actions = scripted_coordination_actions(env)
        _, rewards, _, _, _ = env.step(actions)
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
    print(
        json.dumps(
            {
                "environment": env.metadata["name"],
                "decisions": env.decision_count,
                "completed_modules": len(env.completed),
                "total_modules": len(env.modules),
                "assignments": env.assignment_history,
                "rewards": total_rewards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
