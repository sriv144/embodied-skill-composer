from __future__ import annotations

from dataclasses import dataclass

from embodied_skill_composer.core.interfaces import SimulationAdapter, Skill
from embodied_skill_composer.core.models import SkillResult, SkillStep, WorldState


@dataclass
class BaseSkill:
    name: str

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        return SkillResult(success=True, message="preconditions satisfied")

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        raise NotImplementedError


class MoveToPoseSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("move_to_pose")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        if not any(key in step.params for key in ("target_object", "target_zone", "target_position")):
            return SkillResult(success=False, message="missing target for move", error_code="missing_target")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        z_offset = float(step.params.get("z_offset", 0.1))
        yaw = float(step.params.get("yaw", 0.0))
        if "target_position" in step.params:
            x, y, z = step.params["target_position"]
        elif "target_object" in step.params:
            x, y, z = adapter.resolve_object_position(step.params["target_object"])
        else:
            x, y, z = adapter.resolve_zone_center(step.params["target_zone"])
        success = adapter.move_to((x, y, z + z_offset), yaw=yaw)
        if not success:
            return SkillResult(success=False, message="failed to move arm", error_code="move_failed")
        return SkillResult(success=True, message="arm moved to target pose")


class OpenGripperSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("open_gripper")

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        success = adapter.open_gripper()
        return SkillResult(
            success=success,
            message="gripper opened" if success else "failed to open gripper",
            error_code=None if success else "open_gripper_failed",
        )


class CloseGripperSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("close_gripper")

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        success = adapter.close_gripper()
        return SkillResult(
            success=success,
            message="gripper closed" if success else "failed to close gripper",
            error_code=None if success else "close_gripper_failed",
        )


class GraspObjectSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("grasp_object")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        object_name = step.params.get("object_name")
        if not object_name or object_name not in world.objects:
            return SkillResult(success=False, message="object not found in world", error_code="unknown_object")
        if world.robot.holding_object is not None:
            return SkillResult(success=False, message="robot already holding object", error_code="already_holding")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        object_name = str(step.params["object_name"])
        approach_offset = float(step.params.get("approach_offset", 0.04))
        success = adapter.attempt_grasp(object_name, approach_offset=approach_offset)
        return SkillResult(
            success=success,
            message=f"grasped {object_name}" if success else f"failed to grasp {object_name}",
            error_code=None if success else "grasp_failed",
        )


class LiftObjectSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("lift_object")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        if world.robot.holding_object is None:
            return SkillResult(success=False, message="no object to lift", error_code="nothing_held")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        success = adapter.lift_object(float(step.params.get("height", 0.14)))
        return SkillResult(
            success=success,
            message="lifted held object" if success else "failed to lift object",
            error_code=None if success else "lift_failed",
        )


class PlaceObjectSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("place_object")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        if world.robot.holding_object is None:
            return SkillResult(success=False, message="robot is not holding anything", error_code="nothing_held")
        if not any(key in step.params for key in ("target_zone", "target_object", "target_position")):
            return SkillResult(success=False, message="missing placement target", error_code="missing_target")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        if "target_position" in step.params:
            target = tuple(step.params["target_position"])
        elif "target_zone" in step.params:
            target = adapter.resolve_zone_center(str(step.params["target_zone"]))
        else:
            target = adapter.resolve_stack_position(str(step.params["target_object"]))
        success = adapter.place_held_object(target)
        return SkillResult(
            success=success,
            message="placed held object" if success else "failed to place object",
            error_code=None if success else "place_failed",
        )


class ObserveSceneSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("observe_scene")

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        # Perception runs outside the executor; this step keeps the plan explicit in logs.
        return SkillResult(success=True, message="scene observation checkpoint recorded")


class NavigateToWaypointSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("navigate_to_waypoint")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        waypoint = step.params.get("waypoint")
        if not waypoint or waypoint not in world.stations and waypoint not in world.zones:
            return SkillResult(success=False, message="unknown waypoint", error_code="unknown_waypoint")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        warehouse_adapter = adapter  # runtime check below keeps compatibility with tabletop adapters
        if not hasattr(warehouse_adapter, "navigate_to_waypoint"):
            return SkillResult(
                success=False,
                message="adapter does not support waypoint navigation",
                error_code="navigation_unsupported",
            )
        success = getattr(warehouse_adapter, "navigate_to_waypoint")(str(step.params["waypoint"]))
        return SkillResult(
            success=success,
            message="navigated to waypoint" if success else "failed to navigate",
            error_code=None if success else "navigation_failed",
        )


class PickObjectSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("pick_object")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        object_name = step.params.get("object_name")
        if not object_name or object_name not in world.objects:
            return SkillResult(success=False, message="object not found in world", error_code="unknown_object")
        if world.objects[object_name].collected:
            return SkillResult(success=False, message="object already collected", error_code="already_collected")
        if world.robot.holding_object is not None:
            return SkillResult(success=False, message="robot already holding object", error_code="already_holding")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        if not hasattr(adapter, "pick_object"):
            return SkillResult(
                success=False,
                message="adapter does not support warehouse pickup",
                error_code="pickup_unsupported",
            )
        object_name = str(step.params["object_name"])
        policy_mode = str(step.params.get("policy_mode", "scripted"))
        success = getattr(adapter, "pick_object")(object_name, policy_mode=policy_mode)
        return SkillResult(
            success=success,
            message=f"picked {object_name}" if success else f"failed to pick {object_name}",
            error_code=None if success else "pickup_failed",
        )


class DeliverObjectSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("deliver_object")

    def check_preconditions(self, world: WorldState, step: SkillStep) -> SkillResult:
        if world.robot.holding_object is None:
            return SkillResult(success=False, message="robot is not holding anything", error_code="nothing_held")
        zone_name = step.params.get("zone_name")
        if not zone_name or zone_name not in world.zones:
            return SkillResult(success=False, message="unknown delivery zone", error_code="unknown_zone")
        return super().check_preconditions(world, step)

    def execute(self, adapter: SimulationAdapter, world: WorldState, step: SkillStep) -> SkillResult:
        if not hasattr(adapter, "deliver_held_object"):
            return SkillResult(
                success=False,
                message="adapter does not support warehouse delivery",
                error_code="delivery_unsupported",
            )
        zone_name = str(step.params["zone_name"])
        success = getattr(adapter, "deliver_held_object")(zone_name)
        return SkillResult(
            success=success,
            message="delivered held object" if success else "failed to deliver object",
            error_code=None if success else "delivery_failed",
        )


def build_skill_registry() -> dict[str, Skill]:
    skills = [
        MoveToPoseSkill(),
        OpenGripperSkill(),
        CloseGripperSkill(),
        GraspObjectSkill(),
        LiftObjectSkill(),
        PlaceObjectSkill(),
        ObserveSceneSkill(),
        NavigateToWaypointSkill(),
        PickObjectSkill(),
        DeliverObjectSkill(),
    ]
    return {skill.name: skill for skill in skills}
