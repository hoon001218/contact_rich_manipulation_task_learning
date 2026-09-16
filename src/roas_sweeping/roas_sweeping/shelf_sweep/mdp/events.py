"""Reset randomization for the shelf objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ensure_task_state_buffers(env: ManagerBasedRLEnv) -> None:
    """Allocate task state before managers perform observation shape inference."""
    if not hasattr(env, "target_id"):
        env.target_id = torch.zeros(
            (env.num_envs, 1),
            dtype=torch.long,
            device=env.device,
        )
    if not hasattr(env, "target_width"):
        env.target_width = torch.zeros(
            (env.num_envs, 1),
            dtype=torch.float32,
            device=env.device,
        )
    if not hasattr(env, "sweep_dir"):
        env.sweep_dir = torch.zeros(
            (env.num_envs, 3),
            dtype=torch.float32,
            device=env.device,
        )
    if not hasattr(env, "target_goal_pos_w"):
        env.target_goal_pos_w = torch.zeros(
            (env.num_envs, 3),
            dtype=torch.float32,
            device=env.device,
        )


def randomize_working_and_waiting_objects(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    working_x_range: tuple[float, float],
    working_y_range: tuple[float, float],
    working_height: float,
    waiting_height: float,
    waiting_xy: tuple[tuple[float, float], ...],
    object_widths: tuple[float, ...],
    push_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> None:
    """Select one object for sweeping and move the others to waiting slots."""
    objects: RigidObjectCollection = env.scene[asset_cfg.name]
    if len(waiting_xy) != objects.num_objects:
        raise ValueError(
            "waiting_xy must contain one slot per object: "
            f"{len(waiting_xy)} != {objects.num_objects}"
        )
    if len(object_widths) != objects.num_objects:
        raise ValueError(
            "object_widths must contain one width per object: "
            f"{len(object_widths)} != {objects.num_objects}"
        )
    if push_distance > working_y_range[1] - working_y_range[0]:
        raise ValueError("push_distance is larger than the working y range.")

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    ensure_task_state_buffers(env)
    target_ids = torch.randint(
        low=0,
        high=objects.num_objects,
        size=(num_resets,),
        device=env.device,
    )
    env.target_id[env_ids, 0] = target_ids
    widths = torch.tensor(object_widths, dtype=torch.float32, device=env.device)
    env.target_width[env_ids, 0] = widths[target_ids]

    states = torch.zeros(
        (num_resets, objects.num_objects, 13),
        dtype=torch.float32,
        device=env.device,
    )
    states[..., :2] = torch.tensor(
        waiting_xy,
        dtype=torch.float32,
        device=env.device,
    ).unsqueeze(0)
    states[..., 2] = waiting_height
    states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    states[..., 3:7] = math_utils.random_yaw_orientation(
        num=num_resets * objects.num_objects,
        device=env.device,
    ).reshape(num_resets, objects.num_objects, 4)

    rows = torch.arange(num_resets, device=env.device)
    random_xy = torch.rand((num_resets, 2), device=env.device)
    random_xy[:, 0] = working_x_range[0] + random_xy[:, 0] * (
        working_x_range[1] - working_x_range[0]
    )
    random_xy[:, 1] = working_y_range[0] + random_xy[:, 1] * (
        working_y_range[1] - working_y_range[0]
    )
    states[rows, target_ids, :2] = random_xy + env.scene.env_origins[env_ids, :2]
    states[rows, target_ids, 2] = working_height + env.scene.env_origins[env_ids, 2]

    can_push_positive = random_xy[:, 1] + push_distance <= working_y_range[1]
    can_push_negative = random_xy[:, 1] - push_distance >= working_y_range[0]
    if not torch.all(can_push_positive | can_push_negative):
        raise RuntimeError("Sampled an object position with no feasible push direction.")

    both_directions = can_push_positive & can_push_negative
    random_direction = torch.where(
        torch.rand(num_resets, device=env.device) < 0.5,
        -torch.ones(num_resets, device=env.device),
        torch.ones(num_resets, device=env.device),
    )
    direction = torch.where(
        both_directions,
        random_direction,
        torch.where(
            can_push_positive,
            torch.ones_like(random_direction),
            -torch.ones_like(random_direction),
        ),
    )

    env.sweep_dir[env_ids] = 0.0
    env.sweep_dir[env_ids, 1] = direction * push_distance
    target_pos_w = states[rows, target_ids, :3]
    env.target_goal_pos_w[env_ids] = target_pos_w + env.sweep_dir[env_ids]

    objects.write_object_link_state_to_sim(states, env_ids=env_ids)
