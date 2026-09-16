"""Policy observations for the shelf-sweeping task."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _eef_pose_w_from_frame_transformer(
    eef_frame: FrameTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the configured EEF world pose from the FrameTransformer sensor."""
    return (
        eef_frame.data.target_pos_w[..., 0, :],
        eef_frame.data.target_quat_w[..., 0, :],
    )


def _eef_pose_in_robot_root_frame(
    robot: Articulation,
    eef_frame: FrameTransformer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the EEF pose expressed in the robot root frame.

    This follows the reference task's ``ee_pos_r`` conversion: obtain the EEF
    pose in world coordinates first, then transform that pose into the robot
    root frame.
    """
    eef_pos_w, eef_quat_w = _eef_pose_w_from_frame_transformer(eef_frame)
    return math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        eef_pos_w,
        eef_quat_w,
    )


def _position_w_in_robot_root_frame(
    robot: Articulation,
    position_w: torch.Tensor,
) -> torch.Tensor:
    """Transform a world-frame position into the robot root frame."""
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        position_w,
    )
    return position_b


def _position_in_eef_frame_from_robot_root(
    robot: Articulation,
    eef_frame: FrameTransformer,
    position_w: torch.Tensor,
) -> torch.Tensor:
    """Express a world position in EEF coordinates via the robot root frame."""
    eef_pos_b, eef_quat_b = _eef_pose_in_robot_root_frame(robot, eef_frame)
    position_b = _position_w_in_robot_root_frame(robot, position_w)
    position_eef, _ = math_utils.subtract_frame_transforms(
        eef_pos_b,
        eef_quat_b,
        position_b,
    )
    return position_eef


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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Selected target position expressed in the current EEF frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    eef_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()
    target_state_w = objects.data.object_state_w[rows, target_ids]
    target_pos_w = target_state_w[:, :3]

    if not hasattr(env, "_target_frame_marker"):
        target_marker_cfg = FRAME_MARKER_CFG.copy()
        target_marker_cfg.prim_path = "/Visuals/TargetFrame"
        target_marker_cfg.markers["frame"].scale = (0.10, 0.10, 0.10)

        env._target_frame_marker = VisualizationMarkers(target_marker_cfg)

    object_pos_w = objects.data.object_link_pos_w
    object_quat_w = objects.data.object_link_quat_w

    env._target_frame_marker.visualize(
        translations=object_pos_w.reshape(-1, 3),
        orientations=object_quat_w.reshape(-1, 4),
    )

    # Match the reference observation path: world -> robot root -> EEF.
    target_pos_eef = _position_in_eef_frame_from_robot_root(
        robot,
        eef_frame,
        target_pos_w,
    )

    print("Target position in EEF frame:", target_pos_eef)
    print("Target position in world frame:", target_pos_w)
    print("EEF position in world frame:", eef_frame.data.target_pos_w[..., 0, :])

    return target_pos_eef


def target_object_width(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Width of the selected target object."""
    return env.target_width


def eef_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EEF position and quaternion expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    eef_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    eef_pos_b, eef_quat_b = _eef_pose_in_robot_root_frame(robot, eef_frame)
    return torch.cat((eef_pos_b, math_utils.quat_unique(eef_quat_b)), dim=-1)


def goal_position_in_eef_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Sweep goal position expressed in the current EEF frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    eef_frame: FrameTransformer = env.scene[eef_frame_cfg.name]
    goal_pos_eef = _position_in_eef_frame_from_robot_root(
        robot,
        eef_frame,
        env.target_goal_pos_w,
    )
    return goal_pos_eef
