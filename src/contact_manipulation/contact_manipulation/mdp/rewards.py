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


def reward_components(cfg, blocked, blocker_area, path_increment,
                      wrench, previous_wrench, dt, *, elapsed_time):
    """Return unweighted (clearance, rearrangement, wrench quality, time).

    Clearance evaluates only the designated Blocker. Rearrangement uses this
    transition's corner path in exp(-a*x), including rotation and return motion.
    Its score returns to one when objects stop; past motion is not charged again.

    Motion combines a native joint_acc_l2 based RMS acceleration score with this
    wrench score. Its exponent uses the RMS of normalized positive force/torque
    magnitude growth rates, not their squared sum. Constant load, release and
    direction-only changes have quality one. Wrenches must use a
    consistent frame/origin. Return the raw squared rate separately for diagnostics.

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
    rate_cost = ((force_increase / (dt * cfg.force_rate_ref)).square()
                 + (torque_increase / (dt * cfg.torque_rate_ref)).square())
    wrench_score = torch.exp(-cfg.motion_wrench_increase_decay * torch.sqrt(rate_cost / 2.0))

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

    components = torch.stack((clear, rearr, wrench_score, time_score), -1)
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


def _joint_count(env, asset_cfg):
    """Count resolved joints using metadata only, including native slice selection."""
    ids = asset_cfg.joint_ids
    return (len(range(env.scene[asset_cfg.name].num_joints)[ids])
            if isinstance(ids, slice) else len(ids))


def motion(env, arm_cfg, hand_cfg):
    """Average exp scores of normalized RMS acceleration and load increase.

    Native ArticulationData already updates acceleration every physics step and
    clears it when resetting joint velocity. Use its official joint_acc_l2 on the
    independent arm/hand joints and normalize by their total count, avoiding a
    larger exponent solely because more joints are included. No velocity history
    or per-environment loop is needed. This is the final physics-substep sample,
    not an average over the whole policy interval.
    """
    from isaaclab.envs.mdp.rewards import joint_acc_l2
    from .events import get_task_state

    state = get_task_state(env)
    cfg = state.cfg
    count = _joint_count(env, arm_cfg) + _joint_count(env, hand_cfg)
    if count <= 0:
        raise ValueError("Motion requires at least one selected independent joint.")
    squared = (joint_acc_l2(env, arm_cfg) / cfg.arm_joint_acceleration_ref**2
               + joint_acc_l2(env, hand_cfg) / cfg.hand_joint_acceleration_ref**2)
    acceleration_score = torch.exp(-cfg.motion_acceleration_decay * torch.sqrt(squared / count))
    score = 0.5 * (acceleration_score + state.components[:, 2])
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) / env.step_dt


def time_penalty(env):
    """Integrated elapsed-time penalty converted to a native reward rate."""
    return _reward_rate(env, 3)


def terminal(env):
    """One-time success/failure payment converted to a native reward rate."""
    return _reward_rate(env, 4)
