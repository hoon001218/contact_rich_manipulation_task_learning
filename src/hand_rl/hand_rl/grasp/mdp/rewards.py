"""Reward terms for reaching, enclosing, and lifting the primitive object."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def tcp_object_proximity(
    env,
    std: float,
    tcp_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["tcp"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Smooth reward for moving the palm TCP toward the object."""
    robot: Articulation = env.scene[tcp_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    tcp_pos = robot.data.body_pos_w[:, tcp_cfg.body_ids[0]]
    distance = torch.linalg.norm(tcp_pos - object_asset.data.root_pos_w, dim=-1)
    return 1.0 - torch.tanh(distance / std)


def fingertip_enclosure(
    env,
    std: float,
    fingertip_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward all five fingertips for approaching the object surface."""
    robot: Articulation = env.scene[fingertip_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    tip_positions = robot.data.body_pos_w[:, fingertip_cfg.body_ids]
    distances = torch.linalg.norm(tip_positions - object_asset.data.root_pos_w.unsqueeze(1), dim=-1)
    mean_distance = distances.mean(dim=-1)
    return 1.0 - torch.tanh(mean_distance / std)


def lift_progress(
    env,
    shelf_top_height: float,
    object_half_height: float,
    target_lift_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous normalized object lift height above the shelf."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    height = (
        object_asset.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
        - shelf_top_height
        - object_half_height
    )
    return torch.clamp(height / target_lift_height, min=0.0, max=1.0)


def lifted_and_held(
    env,
    shelf_top_height: float,
    object_half_height: float,
    minimum_lift: float,
    maximum_tcp_distance: float,
    tcp_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["tcp"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse grasp reward when the lifted object remains close to the hand."""
    robot: Articulation = env.scene[tcp_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_height = (
        object_asset.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
        - shelf_top_height
        - object_half_height
    )
    tcp_pos = robot.data.body_pos_w[:, tcp_cfg.body_ids[0]]
    held_close = torch.linalg.norm(tcp_pos - object_asset.data.root_pos_w, dim=-1) < maximum_tcp_distance
    return torch.logical_and(relative_height > minimum_lift, held_close).float()


def success_bonus(
    env,
    shelf_top_height: float,
    object_half_height: float,
    success_lift_height: float,
    maximum_tcp_distance: float,
    maximum_tip_distance: float,
    minimum_close_fingertips: int,
    tcp_cfg: SceneEntityCfg,
    fingertip_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success bonus for a lifted object that remains enclosed by the hand."""
    robot: Articulation = env.scene[tcp_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    height = (
        object_asset.data.root_pos_w[:, 2]
        - env.scene.env_origins[:, 2]
        - shelf_top_height
        - object_half_height
    )
    tcp_distance = torch.linalg.norm(
        robot.data.body_pos_w[:, tcp_cfg.body_ids[0]] - object_asset.data.root_pos_w, dim=-1
    )
    tip_positions = robot.data.body_pos_w[:, fingertip_cfg.body_ids]
    close_tip_count = (
        torch.linalg.norm(tip_positions - object_asset.data.root_pos_w.unsqueeze(1), dim=-1)
        < maximum_tip_distance
    ).sum(dim=-1)
    success = (height > success_lift_height) & (tcp_distance < maximum_tcp_distance)
    success &= close_tip_count >= minimum_close_fingertips
    return success.float()
