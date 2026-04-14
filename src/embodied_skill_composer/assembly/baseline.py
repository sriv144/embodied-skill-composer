from __future__ import annotations

from embodied_skill_composer.assembly.backends import AssemblyTaskBackend
from embodied_skill_composer.assembly.env import AssemblyAction


def scripted_joint_policy(env: AssemblyTaskBackend) -> list[int]:
    beam = env._current_beam()
    if env.state.carrying:
        targets = [beam.assembly_left, beam.assembly_right]
        terminal_action = AssemblyAction.INSTALL
    else:
        targets = [beam.pickup_left, beam.pickup_right]
        terminal_action = AssemblyAction.GRAB

    actions: list[int] = []
    for position, target in zip(env.state.agent_positions, targets):
        if position == target:
            actions.append(int(terminal_action))
            continue
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        if dx != 0:
            actions.append(int(AssemblyAction.RIGHT if dx > 0 else AssemblyAction.LEFT))
        elif dy != 0:
            actions.append(int(AssemblyAction.DOWN if dy > 0 else AssemblyAction.UP))
        else:
            actions.append(int(AssemblyAction.STAY))
    if env.state.carrying and actions[0] != actions[1]:
        actions[1] = actions[0]
    if all(position == target for position, target in zip(env.state.agent_positions, targets)):
        return [int(terminal_action), int(terminal_action)]
    return actions
