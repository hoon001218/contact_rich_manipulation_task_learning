"""Formulation-v2 reward terms for completed control transitions."""

from __future__ import annotations

import math

import torch

from .state import update_mdp


def reward_components(cfg, blocked, blocker_area, path_length, joint_velocity, previous_joint_velocity,
                      wrench, previous_wrench, dt, *, elapsed_time):
    """Formulation v2 with robot motion measured at 12 independent joints.

    The first six measured velocities belong to the UR arm and the last six
    to the hand. Mimic joints are excluded. Motion measures acceleration
    and increasing wrench magnitude; constant joint speed and constant or
    decreasing load incur no cost. Direction changes at equal magnitude are
    deliberately excluded from the load-increase term.
    ``path_length`` is the cumulative sum of each active object's mean OBB
    corner displacement in meters. Fixed corner correspondence incorporates
    translation and rotation, including pivoting and out-and-back movement.
    Clearance, rearrangement, and both motion scores use exp(-decay * cost).
    ``elapsed_time`` is the episode time before this transition. Time charges
    an exponentially increasing cost rate that starts at zero. Integrating
    that rate over the transition keeps the accumulated time cost independent
    of the control frequency.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Control-transition dt must be finite and positive.")
    if joint_velocity.shape[-1] != 12 or previous_joint_velocity.shape != joint_velocity.shape:
        raise ValueError("Motion costs require current and previous velocities of 12 independent joints.")
    clear = torch.exp(-cfg.clearance_decay * (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0, 1))
    if elapsed_time.shape != clear.shape:
        raise ValueError("Elapsed time must contain one value per environment.")
    rearr = torch.exp(-cfg.rearrangement_decay * path_length)
    acceleration_ref = joint_velocity.new_tensor([cfg.arm_joint_acceleration_ref] * 6
                                                 + [cfg.hand_joint_acceleration_ref] * 6)
    acceleration = (joint_velocity - previous_joint_velocity) / dt
    acc_cost = (acceleration / acceleration_ref).square().sum(-1)
    force = torch.linalg.vector_norm(wrench[..., :3], dim=-1)
    previous_force = torch.linalg.vector_norm(previous_wrench[..., :3], dim=-1)
    torque = torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
    previous_torque = torch.linalg.vector_norm(previous_wrench[..., 3:], dim=-1)
    force_increase_rate = (force - previous_force).clamp_min(0) / dt
    torque_increase_rate = (torque - previous_torque).clamp_min(0) / dt
    rate_cost = ((force_increase_rate / cfg.force_rate_ref).square()
                 + (torque_increase_rate / cfg.torque_rate_ref).square())
    costs = torch.stack((acc_cost, rate_cost), -1)
    motion_decay = costs.new_tensor((cfg.motion_acceleration_decay, cfg.motion_wrench_increase_decay))
    motion = torch.exp(-motion_decay * costs).mean(-1)
    if cfg.time_penalty_scale == 0:
        time = torch.zeros_like(clear)
    else:
        # Integral of scale * (exp(growth * t) - 1). Separate the positive
        # expm1 parts to avoid subtracting nearly equal values at episode start.
        scaled_dt = cfg.time_penalty_growth_rate * dt
        expm1_dt = math.expm1(scaled_dt)
        excess = (scaled_dt ** 2 * (.5 + scaled_dt * (1 / 6 + scaled_dt * (1 / 24 + scaled_dt / 120)))
                  if scaled_dt < 1.0e-4 else expm1_dt - scaled_dt)
        integral = (torch.expm1(cfg.time_penalty_growth_rate * elapsed_time) * expm1_dt + excess)
        time = (-cfg.time_penalty_scale * integral / cfg.time_penalty_growth_rate).to(clear)
    return torch.stack((clear, rearr, motion, time), -1), costs


def reward_rate(env):
    """Return the total reward rate for diagnostics and numerical validation."""
    return update_mdp(env).discrete_reward / float(env.step_dt)


def clearance(env):
    """Clearance quality, converted to a rate for Isaac Lab's RewardManager."""
    return update_mdp(env).components[:, 0] / float(env.step_dt)


def rearrangement(env):
    """Cumulative OBB-corner movement quality for the completed transition."""
    return update_mdp(env).components[:, 1] / float(env.step_dt)


def motion(env):
    """Joint acceleration and increasing wrench quality for this transition."""
    return update_mdp(env).components[:, 2] / float(env.step_dt)


def time_penalty(env):
    """Integrated increasing time penalty, charged once per transition."""
    return update_mdp(env).components[:, 3] / float(env.step_dt)


def terminal(env):
    """One-time success or failure reward for the completed transition."""
    return update_mdp(env).components[:, 4] / float(env.step_dt)


def sync_reward_weights(cfg):
    """Use native reward-term weights in the shared transition diagnostics.

    Native ``cfg.rewards.<term>.weight`` is the environment's authoritative
    configuration. The numerical state keeps matching weights so its total
    agrees with RewardManager, including Hydra overrides and disabled terms.
    """
    for term, field in (("clearance", "w_clear"), ("rearrangement", "w_rearr"),
                        ("motion", "w_motion"), ("time", "w_time"),
                        ("terminal", "w_terminal")):
        term_cfg = getattr(cfg.rewards, term)
        setattr(cfg.mdp, field, 0.0 if term_cfg is None else term_cfg.weight)
    cfg.mdp.__post_init__()

