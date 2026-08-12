"""Policy observations for the shelf-sweeping task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


EEF_OFFSET = (0.13, 0.0, 0.0)


def _eef_pose_w(
    robot: Articulation,
    eef_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_id = eef_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[:, body_id]
    body_quat_w = robot.data.body_quat_w[:, body_id]
    offset = torch.tensor(
        EEF_OFFSET,
        dtype=torch.float32,
        device=robot.device,
    ).repeat(robot.num_instances, 1)
    eef_pos_w = body_pos_w + math_utils.quat_apply(body_quat_w, offset)
    return eef_pos_w, body_quat_w


def sweep_joint_pos_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """First eight relative joint positions, matching the reference task."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, :8] - robot.data.default_joint_pos[:, :8]


def sweep_joint_vel_rel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Six relative UR5e arm joint velocities, matching the reference task."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_vel[:, :6] - robot.data.default_joint_vel[:, :6]


def target_position_in_eef_frame(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Selected target position expressed in the current EEF frame."""
    robot: Articulation = env.scene[eef_cfg.name]
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()
    target_pos_w = objects.data.object_pos_w[rows, target_ids]
    eef_pos_w, eef_quat_w = _eef_pose_w(robot, eef_cfg)
    target_pos_eef, _ = math_utils.subtract_frame_transforms(
        eef_pos_w,
        eef_quat_w,
        target_pos_w,
    )
    return target_pos_eef


def target_object_width(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Width of the selected target object."""
    return env.target_width


def eef_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """EEF position and quaternion expressed in the robot root frame."""
    robot: Articulation = env.scene[eef_cfg.name]
    eef_pos_w, eef_quat_w = _eef_pose_w(robot, eef_cfg)
    eef_pos_b, eef_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        eef_pos_w,
        eef_quat_w,
    )
    return torch.cat((eef_pos_b, math_utils.quat_unique(eef_quat_b)), dim=-1)


def goal_position_in_eef_frame(
    env: ManagerBasedRLEnv,
    eef_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sweep goal position expressed in the current EEF frame."""
    robot: Articulation = env.scene[eef_cfg.name]
    eef_pos_w, eef_quat_w = _eef_pose_w(robot, eef_cfg)
    goal_pos_eef, _ = math_utils.subtract_frame_transforms(
        eef_pos_w,
        eef_quat_w,
        env.target_goal_pos_w,
    )
    return goal_pos_eef
