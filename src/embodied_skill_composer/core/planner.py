from __future__ import annotations

from typing import cast

import networkx as nx

from embodied_skill_composer.core.models import SkillStep, TaskSpec, TaskType, WorldState


class RuleBasedPlanner:
    """Builds a linear skill chain through a small DAG so sequencing stays explicit."""

    def plan(self, task: TaskSpec, world: WorldState) -> list[SkillStep]:
        graph = nx.DiGraph()
        steps = self._build_steps(task, world)
        for index, step in enumerate(steps):
            graph.add_node(index, step=step)
            if index:
                graph.add_edge(index - 1, index)
        return [cast(SkillStep, graph.nodes[index]["step"]) for index in nx.topological_sort(graph)]

    def _build_steps(self, task: TaskSpec, world: WorldState) -> list[SkillStep]:
        if task.task_type is TaskType.MULTI_OBJECT_COLLECTION:
            return self._build_collection_steps(task, world)

        source_name = task.source_object
        source = world.objects[source_name]
        base_steps = [
            SkillStep(name="open_gripper"),
            SkillStep(
                name="move_to_pose",
                params={"target_object": source_name, "z_offset": source.size[2] + 0.09},
            ),
            SkillStep(
                name="grasp_object",
                params={"object_name": source_name, "approach_offset": source.size[2] + 0.02},
                max_retries=2,
            ),
            SkillStep(name="lift_object", params={"height": 0.16}),
        ]

        if task.task_type is TaskType.PICK_AND_PLACE:
            assert task.target_zone is not None
            return base_steps + [
                SkillStep(name="move_to_pose", params={"target_zone": task.target_zone, "z_offset": 0.12}),
                SkillStep(name="place_object", params={"target_zone": task.target_zone}),
            ]

        if task.task_type is TaskType.SORT_TO_ZONE:
            source_color = world.objects[source_name].color_name
            target_zone = task.color_routing[source_color]
            return base_steps + [
                SkillStep(name="move_to_pose", params={"target_zone": target_zone, "z_offset": 0.12}),
                SkillStep(name="place_object", params={"target_zone": target_zone}),
            ]

        if task.task_type is TaskType.STACK_BLOCKS:
            assert task.target_object is not None
            target = world.objects[task.target_object]
            return base_steps + [
                SkillStep(
                    name="move_to_pose",
                    params={
                        "target_object": task.target_object,
                        "z_offset": target.size[2] * 2 + source.size[2] + 0.08,
                    },
                ),
                SkillStep(name="place_object", params={"target_object": task.target_object}),
            ]

        raise ValueError(f"Unsupported task type: {task.task_type}")

    def _build_collection_steps(self, task: TaskSpec, world: WorldState) -> list[SkillStep]:
        if not task.target_objects:
            raise ValueError("multi_object_collection tasks require target_objects")
        if not task.drop_zone:
            raise ValueError("multi_object_collection tasks require drop_zone")

        pending_targets = [
            world.objects[name]
            for name in task.target_objects
            if name in world.objects and not world.objects[name].collected
        ]
        pending_targets.sort(
            key=lambda obj: (
                obj.station_name or "",
                obj.position[0],
                obj.position[1],
                obj.name,
            )
        )

        steps: list[SkillStep] = [SkillStep(name="observe_scene", params={"mode": task.perception_mode})]
        for target in pending_targets:
            if not target.station_name:
                continue
            steps.extend(
                [
                    SkillStep(name="navigate_to_waypoint", params={"waypoint": target.station_name}),
                    SkillStep(
                        name="pick_object",
                        params={"object_name": target.name, "policy_mode": task.policy_mode},
                        max_retries=2,
                    ),
                    SkillStep(name="navigate_to_waypoint", params={"waypoint": task.drop_zone}),
                    SkillStep(name="deliver_object", params={"zone_name": task.drop_zone}),
                ]
            )
        return steps
