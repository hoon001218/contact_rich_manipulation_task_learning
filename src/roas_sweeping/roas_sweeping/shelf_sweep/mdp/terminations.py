"""Termination terms for initial environment validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def any_object_below_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Terminate an environment if any shelf object falls below the shelf."""
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    relative_height = (
        objects.data.object_pos_w[..., 2]
        - env.scene.env_origins[:, None, 2]
    )
    return torch.any(relative_height < minimum_height, dim=1)
