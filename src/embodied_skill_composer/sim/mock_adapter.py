from __future__ import annotations

from typing import Any

from embodied_skill_composer.core.models import ObjectState, RobotState, WorldState, ZoneState


def _as_vector3(values: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


class MockTabletopAdapter:
    """Deterministic fallback backend for local demos and tests."""

    def __init__(self, runtime_config: dict[str, Any], scene_config: dict[str, Any]) -> None:
        self.runtime_config = runtime_config
        self.scene_config = scene_config
        self.robot = RobotState(end_effector_position=(0.35, 0.0, 0.35), gripper_opening=0.08)
        self.objects: dict[str, ObjectState] = {}
        self.zones: dict[str, ZoneState] = {}
        self.reset(runtime_config.get("seed", 0))

    def reset(self, seed: int | None = None) -> None:
        self.robot = RobotState(end_effector_position=(0.35, 0.0, 0.35), gripper_opening=0.08)
        self.zones = {
            name: ZoneState(
                name=name,
                center=_as_vector3(payload["center"]),
                size=_as_vector3(payload["size"]),
            )
            for name, payload in self.scene_config["zones"].items()
        }
        self.objects = {
            name: ObjectState(
                name=name,
                color_name=payload["color_name"],
                position=_as_vector3(payload["position"]),
                size=_as_vector3(payload["size"]),
            )
            for name, payload in self.scene_config["objects"].items()
        }

    def get_world_state(self) -> WorldState:
        return WorldState(robot=self.robot, objects=self.objects, zones=self.zones)

    def move_to(self, target_position: tuple[float, float, float], yaw: float = 0.0) -> bool:
        self.robot.end_effector_position = target_position
        if self.robot.holding_object is not None:
            held_name = self.robot.holding_object
            held = self.objects[held_name]
            self.objects[held_name] = held.model_copy(update={"position": target_position, "held": True})
        return True

    def open_gripper(self) -> bool:
        self.robot.gripper_opening = 0.08
        return True

    def close_gripper(self) -> bool:
        self.robot.gripper_opening = 0.0
        return True

    def attempt_grasp(self, object_name: str, approach_offset: float = 0.04) -> bool:
        target = self.objects[object_name]
        ee = self.robot.end_effector_position
        aligned = abs(target.position[0] - ee[0]) < 0.08 and abs(target.position[1] - ee[1]) < 0.08
        if not aligned:
            return False
        self.robot.holding_object = object_name
        self.objects[object_name] = target.model_copy(
            update={"held": True, "position": (ee[0], ee[1], ee[2] - 0.03)}
        )
        return True

    def lift_object(self, height: float) -> bool:
        if self.robot.holding_object is None:
            return False
        current = self.robot.end_effector_position
        return self.move_to((current[0], current[1], current[2] + height))

    def place_held_object(self, target_position: tuple[float, float, float]) -> bool:
        held_name = self.robot.holding_object
        if held_name is None:
            return False
        held = self.objects[held_name]
        self.objects[held_name] = held.model_copy(
            update={"position": target_position, "held": False}
        )
        self.robot.holding_object = None
        self.robot.gripper_opening = 0.08
        return True

    def resolve_zone_center(self, zone_name: str) -> tuple[float, float, float]:
        return self.zones[zone_name].center

    def resolve_object_position(self, object_name: str) -> tuple[float, float, float]:
        return self.objects[object_name].position

    def resolve_stack_position(self, object_name: str) -> tuple[float, float, float]:
        base = self.objects[object_name]
        return (base.position[0], base.position[1], base.position[2] + base.size[2] * 2)

    def close(self) -> None:
        return None
