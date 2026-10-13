"""Batched research rewards and thin Isaac Lab RewardManager adapters.

The numerical kernels use only PyTorch and do not read simulator state. Every
argument carries an environment leading dimension; there are no per-environment
Python loops or device-to-host reads. Native RewardTermCfg weights remain the
only reward weights. The adapters divide discrete transition rewards by dt
because Isaac Lab's RewardManager multiplies each term by dt.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "corner_path_increment", "reward_components", "clearance", "rearrangement",
    "motion", "time_penalty", "terminal",
]


def corner_path_increment(previous_corners, corners, active):
    """Sum active objects' mean eight-corner displacement, in meters.

    Corners have fixed object-local correspondence and shape (N, O, 8, 3),
    expressed in the fixed environment frame. Translation, rotation, pivoting,
    and out-and-back movement all contribute. Averaging the eight corners makes
    a pure translation of one object contribute exactly its translation length;
    objects themselves are summed, including objects absent from initial vision.
    Inactive padded slots are removed before arithmetic, even if they hold NaN.
    """
    if corners.shape != previous_corners.shape or corners.shape[-2:] != (8, 3):
        raise ValueError("Current and previous corners must have matching (..., objects, 8, 3) shapes.")
    if active.shape != corners.shape[:-2]:
        raise ValueError("Active mask must contain one flag per object.")
    delta = torch.where(active[..., None, None], corners - previous_corners, 0.0)
    return torch.linalg.vector_norm(delta, dim=-1).mean(-1).sum(-1)


def reward_components(cfg, blocked, blocker_area, path_length, joint_velocity,
                      previous_joint_velocity, wrench, previous_wrench, dt, *, elapsed_time):
    """Return unweighted (clearance, rearrangement, motion, time) and two costs.

    Clearance, rearrangement, and both motion scores use exp(-a * x). Motion
    penalizes acceleration of six arm and six independent hand joints, plus
    increases in force/torque magnitude. Constant speed, constant load, releasing
    load, and changing force direction at constant magnitude add no motion cost.
    The compensated wrench may use any consistently expressed frame: each force
    and torque vector is reduced to its norm before differencing.

    Time is the exact integral of scale * (exp(growth * t) - 1) over
    [elapsed_time, elapsed_time + dt]. It has no T_ref and starts at zero rate.
    Accumulating this term is independent of control frequency. Nonfinite
    numerical results are replaced by zero; termination separately marks invalid
    active physical state as failure so a failed simulation cannot poison PPO.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Control dt must be finite and positive.")
    if joint_velocity.shape[-1] != 12 or previous_joint_velocity.shape != joint_velocity.shape:
        raise ValueError("Motion requires current and previous velocities of 12 independent joints.")
    batch_shape = joint_velocity.shape[:-1]
    if (blocked.shape != batch_shape or blocker_area.shape != batch_shape
            or path_length.shape != batch_shape or elapsed_time.shape != batch_shape):
        raise ValueError("Areas, path length, and elapsed time must contain one value per environment.")
    if wrench.shape != (*batch_shape, 6) or previous_wrench.shape != wrench.shape:
        raise ValueError("Current and previous wrench must contain six components per environment.")

    ratio = (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    clear = torch.exp(-cfg.clearance_decay * ratio)
    rearr = torch.exp(-cfg.rearrangement_decay * path_length.clamp_min(0.0))

    acceleration = (joint_velocity - previous_joint_velocity) / dt
    # Scalars broadcast on-device; no per-step device tensor for joint scales.
    acc_cost = ((acceleration[..., :6] / cfg.arm_joint_acceleration_ref).square().sum(-1)
                + (acceleration[..., 6:] / cfg.hand_joint_acceleration_ref).square().sum(-1))
    force_increase = (torch.linalg.vector_norm(wrench[..., :3], dim=-1)
                      - torch.linalg.vector_norm(previous_wrench[..., :3], dim=-1)).clamp_min(0.0)
    torque_increase = (torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
                       - torch.linalg.vector_norm(previous_wrench[..., 3:], dim=-1)).clamp_min(0.0)
    rate_cost = ((force_increase / (dt * cfg.force_rate_ref)).square()
                 + (torque_increase / (dt * cfg.torque_rate_ref)).square())
    motion_score = 0.5 * (torch.exp(-cfg.motion_acceleration_decay * acc_cost)
                          + torch.exp(-cfg.motion_wrench_increase_decay * rate_cost))

    if cfg.time_penalty_scale == 0.0:
        time_score = torch.zeros_like(clear)
    else:
        growth_dt = cfg.time_penalty_growth_rate * dt
        exponential_dt = math.expm1(growth_dt)
        # Stable expm1(x) - x avoids cancellation at the first/smallest step.
        excess = (growth_dt ** 2 * (0.5 + growth_dt * (1 / 6 + growth_dt * (1 / 24 + growth_dt / 120)))
                  if growth_dt < 1.0e-4 else exponential_dt - growth_dt)
        integral = torch.expm1(cfg.time_penalty_growth_rate * elapsed_time) * exponential_dt + excess
        time_score = (-cfg.time_penalty_scale * integral / cfg.time_penalty_growth_rate).to(clear)

    components = torch.stack((clear, rearr, motion_score, time_score), -1)
    costs = torch.stack((acc_cost, rate_cost), -1)
    return (torch.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(costs, nan=0.0, posinf=0.0, neginf=0.0))


def _reward_rate(env, index):
    # Lazy import keeps the numerical kernels independent of scene integration.
    from .events import get_task_state

    return get_task_state(env).components[:, index] / env.step_dt


def clearance(env):
    """Blocker overlap quality as a native manager reward rate."""
    return _reward_rate(env, 0)


def rearrangement(env):
    """Cumulative object pose-change quality as a native manager reward rate."""
    return _reward_rate(env, 1)


def motion(env):
    """Robot acceleration and increasing load quality as a reward rate."""
    return _reward_rate(env, 2)


def time_penalty(env):
    """Integrated elapsed-time penalty converted to a native reward rate."""
    return _reward_rate(env, 3)


def terminal(env):
    """One-time success/failure payment converted to a native reward rate."""
    return _reward_rate(env, 4)
