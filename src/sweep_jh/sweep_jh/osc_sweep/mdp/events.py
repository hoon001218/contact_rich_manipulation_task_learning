"""Reset randomization for the sweep task."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.events import randomize_rigid_body_scale
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.utils.stage import get_current_stage


VARIABLE_CUBE_SIZE_BUFFER = "_sweep_target_cube_side_lengths"


def randomize_target_cube_size(
    env,
    env_ids: torch.Tensor | None,
    size_range: tuple[float, float],
    base_size: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Apply isotropic per-environment cube scaling before physics starts.

    Isaac Lab only supports rigid-body scale randomization before simulation
    startup.  The sampled side lengths are cached so reset can place every
    differently sized cube directly on the table surface.
    """
    if base_size <= 0.0:
        raise ValueError("base_size must be positive.")
    if size_range[0] <= 0.0 or size_range[1] < size_range[0]:
        raise ValueError("size_range must be positive and ordered.")

    randomize_rigid_body_scale(
        env,
        env_ids,
        scale_range=(size_range[0] / base_size, size_range[1] / base_size),
        asset_cfg=asset_cfg,
    )

    if env_ids is None:
        selected_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        selected_ids = env_ids.to(device="cpu", dtype=torch.long)
    prim_paths = sim_utils.find_matching_prim_paths(env.scene[asset_cfg.name].cfg.prim_path)
    stage = get_current_stage()
    sampled_sizes = []
    for env_id in selected_ids.tolist():
        scale = stage.GetPrimAtPath(prim_paths[env_id]).GetAttribute("xformOp:scale").Get()
        if scale is None:
            raise RuntimeError(f"Missing scale attribute on '{prim_paths[env_id]}'.")
        sampled_sizes.append(base_size * float(scale[0]))

    if not hasattr(env, VARIABLE_CUBE_SIZE_BUFFER):
        setattr(
            env,
            VARIABLE_CUBE_SIZE_BUFFER,
            torch.full(
                (env.scene.num_envs,),
                base_size,
                dtype=torch.float32,
                device=env.device,
            ),
        )
    size_buffer = getattr(env, VARIABLE_CUBE_SIZE_BUFFER)
    size_buffer[selected_ids.to(env.device)] = torch.tensor(
        sampled_sizes, dtype=torch.float32, device=env.device
    )


def reset_variable_size_target_object(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    table_top_height: float,
    clearance: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Reset a randomized cube with its bottom face on the table."""
    if clearance < 0.0:
        raise ValueError("clearance must be non-negative.")
    if not hasattr(env, VARIABLE_CUBE_SIZE_BUFFER):
        raise RuntimeError("Cube sizes were not initialized by the prestartup event.")

    target: RigidObject = env.scene[asset_cfg.name]
    count = len(env_ids)
    if count == 0:
        return

    default_state = target.data.default_root_state[env_ids].clone()
    positions = default_state[:, :3]
    positions += env.scene.env_origins[env_ids]
    positions[:, 0] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("x", (0.0, 0.0))
    )
    positions[:, 1] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("y", (0.0, 0.0))
    )
    side_lengths = getattr(env, VARIABLE_CUBE_SIZE_BUFFER)[env_ids]
    positions[:, 2] = table_top_height + 0.5 * side_lengths + clearance
    positions[:, 2] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("z", (0.0, 0.0))
    )

    yaw = torch.empty(count, device=env.device).uniform_(
        *pose_range.get("yaw", (0.0, 0.0))
    )
    zero = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zero, zero, yaw)

    velocities = default_state[:, 7:13]
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        velocities[:, index] += torch.empty(count, device=env.device).uniform_(low, high)

    target.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=env_ids
    )
    target.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_target_object(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> None:
    """Randomize cube XY position/yaw and clear its velocities."""
    target: RigidObject = env.scene[asset_cfg.name]
    count = len(env_ids)
    if count == 0:
        return

    default_state = target.data.default_root_state[env_ids].clone()
    positions = default_state[:, :3]
    positions += env.scene.env_origins[env_ids]
    positions[:, 0] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("x", (0.0, 0.0))
    )
    positions[:, 1] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("y", (0.0, 0.0))
    )
    positions[:, 2] += torch.empty(count, device=env.device).uniform_(
        *pose_range.get("z", (0.0, 0.0))
    )

    yaw = torch.empty(count, device=env.device).uniform_(
        *pose_range.get("yaw", (0.0, 0.0))
    )
    zero = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zero, zero, yaw)

    velocities = default_state[:, 7:13]
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        low, high = velocity_range.get(key, (0.0, 0.0))
        velocities[:, index] += torch.empty(
            count, device=env.device
        ).uniform_(low, high)

    target.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=env_ids
    )
    target.write_root_velocity_to_sim(velocities, env_ids=env_ids)

