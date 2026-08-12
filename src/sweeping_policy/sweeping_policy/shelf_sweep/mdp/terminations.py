"""Termination terms adapted from the reference random shelf-sweep task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


LEFT_FINGER_OFFSET = (0.13, 0.07, 0.0)
RIGHT_FINGER_OFFSET = (0.13, -0.07, 0.0)
WRIST_OFFSET = (-0.14, 0.0, 0.0)


def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _offset_body_position(
    robot: Articulation,
    body_id: int,
    offset: tuple[float, float, float],
) -> torch.Tensor:
    offset_tensor = torch.tensor(
        offset, dtype=torch.float32, device=robot.device
    ).repeat(robot.num_instances, 1)
    return robot.data.body_pos_w[:, body_id] + math_utils.quat_apply(
        robot.data.body_quat_w[:, body_id], offset_tensor
    )


def drop_object_termination(
    env: ManagerBasedRLEnv,
    height_condition: float,
    rotation_condition: float,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Terminate when any work or waiting object drops or flips."""
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    states = objects.data.object_link_state_w
    heights = states[..., 2]
    is_dropped = heights < height_condition
    num_envs, num_objects = states.shape[:2]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(
        states[..., 3:7].reshape(-1, 4)
    )
    roll = _normalize_angle(roll.view(num_envs, num_objects))
    pitch = _normalize_angle(pitch.view(num_envs, num_objects))
    is_flipped = (torch.abs(roll) > rotation_condition) | (
        torch.abs(pitch) > rotation_condition
    )
    return torch.any(is_dropped | is_flipped, dim=1)


def push_fast_termination(
    env: ManagerBasedRLEnv,
    speed_condition: float,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Terminate when the selected target exceeds the reference speed limit."""
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()
    target_velocity = objects.data.object_lin_vel_w[rows, target_ids]
    return torch.linalg.norm(target_velocity, dim=-1) > speed_condition


def shelf_collision_termination(
    env: ManagerBasedRLEnv,
    threshold: float,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Reference shelf velocity and finger/wrist height termination."""
    shelf: RigidObject = env.scene[shelf_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(eef_body_name)
    body_id = body_ids[0]
    shelf_height = shelf.data.root_pos_w[:, 2] + 1.05
    left_finger = _offset_body_position(robot, body_id, LEFT_FINGER_OFFSET)
    right_finger = _offset_body_position(robot, body_id, RIGHT_FINGER_OFFSET)
    wrist = _offset_body_position(robot, body_id, WRIST_OFFSET)
    return (
        (torch.linalg.norm(shelf.data.root_vel_w, dim=-1) > threshold)
        | (left_finger[:, 2] - shelf_height < 0.01)
        | (right_finger[:, 2] - shelf_height < 0.01)
        | (wrist[:, 2] - shelf_height < 0.07)
    )


def hand_velocity_termination(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any UR5e arm joint exceeds the reference speed limit."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(torch.abs(robot.data.joint_vel[:, :6]) > threshold, dim=1)
