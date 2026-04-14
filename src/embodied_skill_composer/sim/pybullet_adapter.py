from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import pybullet as p
import pybullet_data

from embodied_skill_composer.core.models import ObjectState, RobotState, SensorObservation, WorldState, ZoneState


def _as_vector3(values: list[float] | tuple[float, ...]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


class PyBulletTabletopAdapter:
    PANDA_JOINTS = list(range(7))
    FINGER_JOINTS = [9, 10]
    END_EFFECTOR_LINK = 11
    HOME_JOINT_POSITIONS = [0.0, -0.35, 0.0, -2.25, 0.0, 2.0, 0.8]

    def __init__(
        self,
        runtime_config: dict[str, Any],
        scene_config: dict[str, Any],
        gui: bool = False,
    ) -> None:
        self.runtime_config = runtime_config
        self.scene_config = scene_config
        self.gui = gui
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        self.robot_id: int | None = None
        self.table_id: int | None = None
        self.held_constraint_id: int | None = None
        self.held_object_name: str | None = None
        self.object_ids: dict[str, int] = {}
        self.object_sizes: dict[str, tuple[float, float, float]] = {}
        self.object_colors: dict[str, str] = {}
        self.zone_centers: dict[str, tuple[float, float, float]] = {}
        self.zone_sizes: dict[str, tuple[float, float, float]] = {}
        self.gripper_opening = 0.08
        self.reset(runtime_config.get("seed", 0))

    def close(self) -> None:
        if p.isConnected(self.client):
            p.disconnect(self.client)

    def reset(self, seed: int | None = None) -> None:
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(
            float(self.runtime_config["simulation"]["time_step"]), physicsClientId=self.client
        )
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._create_table()
        self._load_robot()
        self._create_zones()
        self._spawn_objects()
        self.held_constraint_id = None
        self.held_object_name = None
        self._step(self.runtime_config["simulation"]["settle_steps"])

    def get_world_state(self) -> WorldState:
        assert self.robot_id is not None
        end_effector_state = p.getLinkState(
            self.robot_id, self.END_EFFECTOR_LINK, physicsClientId=self.client
        )
        objects = {}
        for name, body_id in self.object_ids.items():
            position, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)
            objects[name] = ObjectState(
                name=name,
                color_name=self.object_colors[name],
                position=_as_vector3(position),
                size=self.object_sizes[name],
                held=name == self.held_object_name,
            )
        zones = {
            name: ZoneState(name=name, center=center, size=self.zone_sizes[name])
            for name, center in self.zone_centers.items()
        }
        robot = RobotState(
            end_effector_position=_as_vector3(end_effector_state[4]),
            gripper_opening=self.gripper_opening,
            holding_object=self.held_object_name,
        )
        return WorldState(robot=robot, objects=objects, zones=zones)

    def move_to(self, target_position: tuple[float, float, float], yaw: float = 0.0) -> bool:
        assert self.robot_id is not None
        target_orientation = p.getQuaternionFromEuler([math.pi, 0.0, yaw])
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            self.END_EFFECTOR_LINK,
            target_position,
            targetOrientation=target_orientation,
            maxNumIterations=150,
            residualThreshold=1e-4,
            physicsClientId=self.client,
        )
        for joint_index, joint_target in zip(self.PANDA_JOINTS, joint_targets[:7]):
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_target,
                force=800,
                physicsClientId=self.client,
            )
        self._step(self.runtime_config["simulation"]["move_steps"])
        end_effector_state = p.getLinkState(
            self.robot_id, self.END_EFFECTOR_LINK, physicsClientId=self.client
        )
        actual = end_effector_state[4]
        error = math.dist(target_position, actual)
        return error < 0.06

    def open_gripper(self) -> bool:
        return self._set_gripper(0.04)

    def close_gripper(self) -> bool:
        return self._set_gripper(0.0)

    def attempt_grasp(self, object_name: str, approach_offset: float = 0.04) -> bool:
        target_x, target_y, target_z = self.resolve_object_position(object_name)
        self.open_gripper()
        if not self.move_to((target_x, target_y, target_z + 0.10)):
            return False
        if not self.move_to((target_x, target_y, target_z + approach_offset)):
            return False
        self.close_gripper()
        if not self._attach_if_close(object_name):
            self.open_gripper()
            self.move_to((target_x, target_y, target_z + 0.10))
            return False
        self.move_to((target_x, target_y, target_z + 0.12))
        return True

    def lift_object(self, height: float) -> bool:
        if self.held_object_name is None:
            return False
        current = self.get_world_state().robot.end_effector_position
        return self.move_to((current[0], current[1], current[2] + height))

    def place_held_object(self, target_position: tuple[float, float, float]) -> bool:
        if self.held_object_name is None:
            return False
        hover = (target_position[0], target_position[1], target_position[2] + 0.12)
        near = (target_position[0], target_position[1], target_position[2] + 0.05)
        if not self.move_to(hover):
            return False
        if not self.move_to(near):
            return False
        held_name = self.held_object_name
        body_id = self.object_ids[held_name]
        self.open_gripper()
        if self.held_constraint_id is not None:
            p.removeConstraint(self.held_constraint_id, physicsClientId=self.client)
            self.held_constraint_id = None
        final_height = target_position[2] + self.object_sizes[held_name][2]
        p.resetBasePositionAndOrientation(
            body_id,
            [target_position[0], target_position[1], final_height],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client,
        )
        self.held_object_name = None
        self._step(self.runtime_config["simulation"]["settle_steps"])
        self.move_to(hover)
        return True

    def capture_observation(self) -> SensorObservation:
        width = 320
        height = 240
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.55, 0.0, 1.15],
            cameraTargetPosition=[0.55, 0.0, 0.0],
            cameraUpVector=[0.0, 1.0, 0.0],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=width / height,
            nearVal=0.05,
            farVal=3.0,
        )
        _, _, rgba_pixels, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )
        rgb: list[list[list[int]]] = []
        for row in rgba_pixels:
            rgb_row: list[list[int]] = []
            for pixel in row:
                rgb_row.append([int(pixel[0]), int(pixel[1]), int(pixel[2])])
            rgb.append(rgb_row)
        return SensorObservation(camera_name="tabletop_overhead", rgb=rgb, resolution=(width, height))

    def resolve_zone_center(self, zone_name: str) -> tuple[float, float, float]:
        return self.zone_centers[zone_name]

    def resolve_object_position(self, object_name: str) -> tuple[float, float, float]:
        position, _ = p.getBasePositionAndOrientation(
            self.object_ids[object_name], physicsClientId=self.client
        )
        return _as_vector3(position)

    def resolve_stack_position(self, object_name: str) -> tuple[float, float, float]:
        position = self.resolve_object_position(object_name)
        size = self.object_sizes[object_name]
        return (position[0], position[1], position[2] + size[2] + 0.01)

    def _create_table(self) -> None:
        collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.35, 0.45, 0.02], physicsClientId=self.client
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.35, 0.45, 0.02],
            rgbaColor=[0.55, 0.45, 0.35, 1.0],
            physicsClientId=self.client,
        )
        self.table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[0.55, 0.0, -0.02],
            physicsClientId=self.client,
        )

    def _load_robot(self) -> None:
        self.robot_id = p.loadURDF(
            str(Path(pybullet_data.getDataPath()) / "franka_panda/panda.urdf"),
            useFixedBase=True,
            basePosition=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )
        for joint_index, joint_target in enumerate(self.HOME_JOINT_POSITIONS):
            p.resetJointState(self.robot_id, joint_index, joint_target, physicsClientId=self.client)
            p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=joint_target,
                force=800,
                physicsClientId=self.client,
            )
        self._set_gripper(0.04)
        self._step(self.runtime_config["simulation"]["settle_steps"])

    def _create_zones(self) -> None:
        self.zone_centers.clear()
        self.zone_sizes.clear()
        for name, payload in self.scene_config["zones"].items():
            center = _as_vector3(payload["center"])
            size = _as_vector3(payload["size"])
            self.zone_centers[name] = center
            self.zone_sizes[name] = size
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=list(size), physicsClientId=self.client
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=list(size),
                rgbaColor=payload["color"],
                physicsClientId=self.client,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=list(center),
                physicsClientId=self.client,
            )

    def _spawn_objects(self) -> None:
        self.object_ids.clear()
        self.object_sizes.clear()
        self.object_colors.clear()
        for name, payload in self.scene_config["objects"].items():
            size = _as_vector3(payload["size"])
            position = payload["position"]
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=list(size), physicsClientId=self.client
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=list(size),
                rgbaColor=payload["rgba"],
                physicsClientId=self.client,
            )
            body_id = p.createMultiBody(
                baseMass=0.15,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=position,
                physicsClientId=self.client,
            )
            p.changeDynamics(body_id, -1, lateralFriction=1.0, physicsClientId=self.client)
            self.object_ids[name] = body_id
            self.object_sizes[name] = size
            self.object_colors[name] = payload["color_name"]

    def _set_gripper(self, opening: float) -> bool:
        assert self.robot_id is not None
        for finger_joint in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot_id,
                finger_joint,
                p.POSITION_CONTROL,
                targetPosition=opening,
                force=150,
                physicsClientId=self.client,
            )
        self.gripper_opening = opening * 2
        self._step(self.runtime_config["simulation"]["settle_steps"])
        return True

    def _attach_if_close(self, object_name: str) -> bool:
        assert self.robot_id is not None
        object_id = self.object_ids[object_name]
        object_position, _ = p.getBasePositionAndOrientation(object_id, physicsClientId=self.client)
        ee_state = p.getLinkState(self.robot_id, self.END_EFFECTOR_LINK, physicsClientId=self.client)
        ee_position = ee_state[4]
        threshold = float(self.runtime_config["simulation"]["grasp_threshold"])
        if math.dist(object_position, ee_position) > threshold:
            return False
        if self.held_constraint_id is not None:
            p.removeConstraint(self.held_constraint_id, physicsClientId=self.client)
        parent_pos, parent_orn = p.invertTransform(ee_position, ee_state[5])
        child_frame_pos, _ = p.multiplyTransforms(
            parent_pos, parent_orn, object_position, [0.0, 0.0, 0.0, 1.0]
        )
        self.held_constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.END_EFFECTOR_LINK,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=child_frame_pos,
            childFramePosition=[0.0, 0.0, 0.0],
            physicsClientId=self.client,
        )
        self.held_object_name = object_name
        self._step(self.runtime_config["simulation"]["settle_steps"])
        return True

    def _step(self, count: int) -> None:
        for _ in range(count):
            p.stepSimulation(physicsClientId=self.client)
            if self.gui:
                time_step = float(self.runtime_config["simulation"]["time_step"])
                time.sleep(time_step)
