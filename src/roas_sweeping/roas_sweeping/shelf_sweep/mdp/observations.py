"""Observations for the ROAS-hand shelf-sweeping environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .events import ensure_task_state_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_position_in_eef_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    eef_cfg: SceneEntityCfg,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Selected shelf-object position expressed in the palm sensor frame."""
    ensure_task_state_buffers(env)
    robot: Articulation = env.scene[robot_cfg.name]
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id[:, 0].long()
    target_pos_w = objects.data.object_pos_w[rows, target_ids]
    target_pos_eef, _ = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, eef_cfg.body_ids[0]],
        robot.data.body_quat_w[:, eef_cfg.body_ids[0]],
        target_pos_w,
    )
    return target_pos_eef


def target_object_width(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Width of the object selected for the current sweep."""
    ensure_task_state_buffers(env)
    return env.target_width


def force_sensor_contacts(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Return one binary contact state for every tactile link.

    Each configured contact sensor owns exactly one URDF force-sensor body, so
    the returned order follows ``sensor_names``. A sensor is on when the norm
    of its world-frame net contact force is greater than ``force_threshold``.
    """
    contacts: list[torch.Tensor] = []
    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        force = sensor.data.net_forces_w
        if force.shape[1] != 1:
            raise RuntimeError(
                f"Contact sensor {name!r} must resolve one body, got {force.shape[1]}."
            )
        is_contact = torch.linalg.norm(force[:, 0, :], dim=-1) > force_threshold
        contacts.append(is_contact.to(dtype=torch.float32).unsqueeze(-1))
    return torch.cat(contacts, dim=-1)
