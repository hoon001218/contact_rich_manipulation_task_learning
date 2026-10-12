"""Formulation-v2 reward terms for completed control transitions."""

from __future__ import annotations

import math

import torch

from .state import update_mdp


def reward_components(cfg, blocked, blocker_area, path_length, joint_velocity, previous_joint_velocity,
                      wrench, previous_wrench, dt):
    """Formulation v2 with robot motion measured at 12 independent joints.

    The first six measured velocities belong to the UR arm and the last six
    to the hand. Mimic joints are excluded. Motion measures acceleration
    and increasing wrench magnitude; constant joint speed and constant or
    decreasing load incur no cost. Direction changes at equal magnitude are
    deliberately excluded from the load-increase term.
    Object movement contributes to the cumulative rearrangement term.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Control-transition dt must be finite and positive.")
    if joint_velocity.shape[-1] != 12 or previous_joint_velocity.shape != joint_velocity.shape:
        raise ValueError("Motion costs require current and previous velocities of 12 independent joints.")
    clear = torch.exp(-(blocked / blocker_area.clamp_min(1.0e-12)).clamp(0, 1))
    rearr = torch.exp(-path_length / cfg.displacement_ref)
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
    motion = torch.exp(-costs).mean(-1)
    time = torch.full_like(clear, -dt / cfg.time_ref)
    return torch.stack((clear, rearr, motion, time), -1), costs


def reward_rate(env):
    """RewardManager multiplies by dt: undo it to deliver the v2 discrete reward."""
    return update_mdp(env).discrete_reward / float(env.step_dt)

