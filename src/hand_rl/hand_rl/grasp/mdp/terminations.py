"""Termination conditions for shelf grasping."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


def object_lifted_to_goal(
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
    """End successfully only when the lifted object remains enclosed by the hand."""
    robot: Articulation = env.scene[tcp_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_height = (
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
    success = (relative_height > success_lift_height) & (tcp_distance < maximum_tcp_distance)
    return success & (close_tip_count >= minimum_close_fingertips)


def object_fell_from_shelf(
    env,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """End unsuccessfully if the object falls below the shelf workspace."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_height = object_asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return relative_height < minimum_height
