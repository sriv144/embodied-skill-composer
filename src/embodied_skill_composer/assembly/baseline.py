from __future__ import annotations

from embodied_skill_composer.assembly.backends import AssemblyTaskBackend
from embodied_skill_composer.assembly.env import AssemblyAction


def scripted_joint_policy(env: AssemblyTaskBackend) -> list[int]:
    observation = env.get_construction_observation()
    if observation.current_beam_name is None:
        return [int(AssemblyAction.STAY)] * env.num_agents
    candidate_beams = env.config.beams
    if env.active_stage_index is not None and env.config.curriculum_stage_beams:
        candidate_beams = env.config.curriculum_stage_beams[env.active_stage_index]
    beam = next(
        (
            candidate
            for candidate in candidate_beams
            if candidate.name == observation.current_beam_name
        ),
        None,
    )
    if beam is None:
        return [int(AssemblyAction.STAY)] * env.num_agents
    if observation.carrying:
        targets = [beam.assembly_left, beam.assembly_right]
        terminal_action = AssemblyAction.INSTALL
    else:
        targets = [beam.pickup_left, beam.pickup_right]
        terminal_action = AssemblyAction.GRAB

    actions: list[int] = []
    for position, target in zip(observation.agent_positions, targets):
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
    if observation.carrying and actions[0] != actions[1]:
        actions[1] = actions[0]
    if all(
        position == target
        for position, target in zip(observation.agent_positions, targets)
    ):
        return [int(terminal_action), int(terminal_action)]
    return actions
