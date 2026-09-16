"""Small baseline reward set for validating the assembled environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

from .events import ensure_task_state_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_goal_proximity(
    env: ManagerBasedRLEnv,
    std: float,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Dense validation reward for moving the selected object to its sweep goal."""
    ensure_task_state_buffers(env)
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_pos_w = objects.data.object_pos_w[rows, env.target_id[:, 0].long()]
    distance = torch.linalg.norm(env.target_goal_pos_w - target_pos_w, dim=-1)
    return 1.0 - torch.tanh(distance / std)
