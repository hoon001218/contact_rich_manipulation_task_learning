"""Batched research rewards and thin Isaac Lab RewardManager adapters.

The numerical kernels use only PyTorch and do not read simulator state. Every
argument carries an environment leading dimension; there are no per-environment
Python loops or device-to-host reads. Native RewardTermCfg weights remain the
only reward weights. Clearance, rearrangement and motion keep exp(-a*x) quality
scores. The adapters divide discrete scores/payments by dt to cancel the native
RewardManager's dt factor; Time retains its exact elapsed-time integral.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "corner_path_increment", "reward_components", "clearance", "rearrangement",
    "motion_action_rate", "motion_joint_velocity", "motion_force_increase", "motion_torque_increase",
    "time_penalty", "terminal",
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


def reward_components(cfg, blocked, blocker_area, path_increment,
                      wrench, previous_wrench, dt, *, elapsed_time):
    """Return unweighted (clearance, rearrangement, force quality, torque quality, time).

    Clearance evaluates only the designated Blocker. Rearrangement uses this
    transition's corner path in exp(-a*x), including rotation and return motion.
    Its score returns to one when objects stop; past motion is not charged again.

    Motion combines native action-rate and joint-velocity qualities with two
    independent force/torque growth qualities. Cache the two scores separately
    for native per-term logging; each exponent uses squared positive growth above a
    small dead band. Constant load, release and direction-only changes have
    quality one. Wrenches are self-load compensated at the current sensor center:
    rotations of their axes preserve norms, but translating the torque origin
    would change the measured load. Return the two excess-rate squared costs
    separately for diagnostics, in (N/s)^2 and (N*m/s)^2.

    Time is the exact integral of scale * (exp(growth * t) - 1) over
    [elapsed_time, elapsed_time + dt]. It has no T_ref and starts at zero rate.
    Accumulating this term is independent of control frequency. Nonfinite
    numerical results are replaced by zero; termination separately marks invalid
    active physical state as failure so a failed simulation cannot poison PPO.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Control dt must be finite and positive.")
    batch_shape = blocked.shape
    if (blocker_area.shape != batch_shape
            or path_increment.shape != batch_shape or elapsed_time.shape != batch_shape):
        raise ValueError("Areas, path length, and elapsed time must contain one value per environment.")
    if wrench.shape != (*batch_shape, 6) or previous_wrench.shape != wrench.shape:
        raise ValueError("Current and previous wrench must contain six components per environment.")

    ratio = (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    clear = torch.exp(-cfg.clearance_decay * ratio)
    rearr = torch.exp(-cfg.rearrangement_decay * path_increment.clamp_min(0.0))

    force_increase = (torch.linalg.vector_norm(wrench[..., :3], dim=-1)
                      - torch.linalg.vector_norm(previous_wrench[..., :3], dim=-1)).clamp_min(0.0)
    torque_increase = (torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
                       - torch.linalg.vector_norm(previous_wrench[..., 3:], dim=-1)).clamp_min(0.0)
    force_cost = (force_increase / dt - cfg.force_rate_deadband).clamp_min(0.0).square()
    torque_cost = (torque_increase / dt - cfg.torque_rate_deadband).clamp_min(0.0).square()
    rate_cost = torch.stack((force_cost, torque_cost), -1)
    force_score = torch.exp(-cfg.motion_force_rate_decay * force_cost)
    torque_score = torch.exp(-cfg.motion_torque_rate_decay * torque_cost)

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

    components = torch.stack((clear, rearr, force_score, torque_score, time_score), -1)
    return (torch.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(rate_cost, nan=0.0, posinf=0.0, neginf=0.0))


def _reward_rate(env, index):
    # Lazy import keeps the numerical kernels independent of scene integration.
    from .events import get_task_state

    return get_task_state(env).components[:, index] / env.step_dt


def clearance(env):
    """Blocker overlap quality as a native manager reward rate."""
    return _reward_rate(env, 0)


def rearrangement(env):
    """Exponential quality of the current step's object corner displacement."""
    return _reward_rate(env, 1)


def motion_action_rate(env):
    """Exp quality of native raw 18D action differences, without /dt in the cost."""
    from isaaclab.envs.mdp.rewards import action_rate_l2
    from .events import get_task_state

    cfg = get_task_state(env).cfg
    score = torch.exp(-cfg.motion_action_rate_decay * action_rate_l2(env))
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) / env.step_dt


def motion_joint_velocity(env, arm_cfg, hand_cfg):
    """Exp quality of native independent-joint velocity sums at the final substep."""
    from isaaclab.envs.mdp.rewards import joint_vel_l2
    from .events import get_task_state

    cfg = get_task_state(env).cfg
    velocity_cost = joint_vel_l2(env, arm_cfg) + joint_vel_l2(env, hand_cfg)
    score = torch.exp(-cfg.motion_joint_velocity_decay * velocity_cost)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) / env.step_dt


def motion_force_increase(env):
    """Cached exp quality of positive compensated sensor-force magnitude growth."""
    return _reward_rate(env, 2)


def motion_torque_increase(env):
    """Cached exp quality of positive F/T sensor-moment growth, not joint torque."""
    return _reward_rate(env, 3)


def time_penalty(env):
    """Integrated elapsed-time penalty converted to a native reward rate."""
    return _reward_rate(env, 4)


def terminal(env):
    """One-time success/failure payment converted to a native reward rate."""
    return _reward_rate(env, 5)
