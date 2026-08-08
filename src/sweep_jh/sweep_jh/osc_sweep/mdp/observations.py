"""Policy observations for the OSC sweep task."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .common import (
    pose_w_to_root_rpy,
    target_contact_data_w,
    virtual_ft_wrench_in_base_frame,
)


def end_effector_pose_b(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Six-dimensional EEF pose in the robot base frame."""
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    return pose_w_to_root_rpy(
        robot,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )


def end_effector_twist_b(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EEF linear and angular velocity in the robot base frame.

    The returned order is ``[vx, vy, vz, wx, wy, wz]``.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    relative_velocity_w = (
        robot.data.body_vel_w[:, body_id] - robot.data.root_vel_w
    )
    linear_velocity_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, :3]
    )
    angular_velocity_b = math_utils.quat_apply_inverse(
        robot.data.root_quat_w, relative_velocity_w[:, 3:]
    )
    return torch.cat((linear_velocity_b, angular_velocity_b), dim=-1)


def virtual_ft_wrench_b(
    env,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Virtual F/T wrench in robot-base axes at the sensor origin."""
    return virtual_ft_wrench_in_base_frame(env, asset_cfg)


def target_contact_point_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Force-weighted gripper/cube contact point in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    point_w, _, contact_mask = target_contact_data_w(env, sensor_names, force_threshold)
    point_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        point_w,
    )
    return torch.where(contact_mask.unsqueeze(-1), point_b, torch.zeros_like(point_b))


def initial_target_pose_b(env, command_name: str) -> torch.Tensor:
    """Initial target pose captured after reset, as ``xyz + RPY``."""
    command = env.command_manager.get_term(command_name)
    return command.initial_pose_b


def object_linear_velocity_b(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Target-object linear velocity expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    return math_utils.quat_apply_inverse(
        robot.data.root_quat_w, target.data.root_lin_vel_w
    )


def task_phase(env, command_name: str) -> torch.Tensor:
    """Return 0 for sweeping and 1 for the Home-return phase."""
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "task_phase"):
        raise RuntimeError(f"Command '{command_name}' does not expose task_phase.")
    return command.task_phase.float().unsqueeze(-1)
