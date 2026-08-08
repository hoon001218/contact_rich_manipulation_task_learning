"""Task-space observations for shelf grasping."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def tcp_pose_in_robot_root_frame(
    env,
    tcp_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["tcp"]),
) -> torch.Tensor:
    """TCP position and quaternion in the UR5e root frame."""
    robot: Articulation = env.scene[tcp_cfg.name]
    body_id = tcp_cfg.body_ids[0]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    return torch.cat((pos_b, math_utils.quat_unique(quat_b)), dim=-1)


def object_pose_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position and quaternion in the UR5e root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((pos_b, math_utils.quat_unique(quat_b)), dim=-1)


def object_to_tcp_vector(
    env,
    tcp_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["tcp"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Vector from TCP to object, expressed in robot-root axes."""
    robot: Articulation = env.scene[tcp_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    body_id = tcp_cfg.body_ids[0]
    tcp_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
    )
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_asset.data.root_pos_w,
    )
    return object_pos_b - tcp_pos_b


def object_velocity_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear and angular velocity expressed in robot-root axes."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    root_quat = robot.data.root_quat_w
    linear_b = math_utils.quat_apply_inverse(root_quat, object_asset.data.root_lin_vel_w)
    angular_b = math_utils.quat_apply_inverse(root_quat, object_asset.data.root_ang_vel_w)
    return torch.cat((linear_b, angular_b), dim=-1)

