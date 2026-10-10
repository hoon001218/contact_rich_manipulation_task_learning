"""Formulation-v2 reward terms for completed control transitions."""

from __future__ import annotations

import math

import torch

from .state import update_mdp


def reward_components(cfg, blocked, blocker_area, path_length, velocity, previous_velocity,
                      wrench, previous_wrench, active, dt):
    """Formulation v2, per completed control transition, with no hidden averages."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Control-transition dt must be finite and positive.")
    clear = torch.exp(-(blocked / blocker_area.clamp_min(1.0e-12)).clamp(0, 1))
    rearr = torch.exp(-path_length / cfg.displacement_ref)
    linear = torch.linalg.vector_norm(velocity[..., :2], dim=-1)
    angular = torch.linalg.vector_norm(velocity[..., 3:], dim=-1)
    vel_cost = (((linear - cfg.linear_velocity_soft).clamp_min(0) / cfg.linear_velocity_ref).square()
                + ((angular - cfg.angular_velocity_soft).clamp_min(0) / cfg.angular_velocity_ref).square())
    acceleration = (velocity - previous_velocity) / dt
    acc_cost = ((torch.linalg.vector_norm(acceleration[..., :3], dim=-1) / cfg.linear_acceleration_ref).square()
                + (torch.linalg.vector_norm(acceleration[..., 3:], dim=-1) / cfg.angular_acceleration_ref).square())
    force, torque = torch.linalg.vector_norm(wrench[..., :3], dim=-1), torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
    load_cost = ((force - cfg.force_soft).clamp_min(0) / cfg.force_ref
                 + (torque - cfg.torque_soft).clamp_min(0) / cfg.torque_ref)
    rate = (wrench - previous_wrench) / dt
    rate_cost = ((torch.linalg.vector_norm(rate[..., :3], dim=-1) / cfg.force_rate_ref).square()
                 + (torch.linalg.vector_norm(rate[..., 3:], dim=-1) / cfg.torque_rate_ref).square())
    costs = torch.stack(((vel_cost * active).sum(-1), (acc_cost * active).sum(-1), load_cost, rate_cost), -1)
    motion = torch.exp(-costs).mean(-1)
    time = torch.full_like(clear, -dt / cfg.time_ref)
    return torch.stack((clear, rearr, motion, time), -1), costs


def reward_rate(env):
    """RewardManager multiplies by dt: undo it to deliver the v2 discrete reward."""
    return update_mdp(env).discrete_reward / float(env.step_dt)

