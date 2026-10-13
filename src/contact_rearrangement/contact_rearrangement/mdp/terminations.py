"""Target protection, stable clearance, and physical task failures."""

from __future__ import annotations

import math

import torch

from .state import quat_apply, quat_conjugate, quat_mul, update_mdp


def terminal_transition(success_candidate, failure, stable_count, paid, cfg):
    success = success_candidate & (stable_count >= cfg.stable_steps) & ~failure
    terminal = torch.where(failure, -cfg.failure_penalty,
                           torch.where(success, cfg.success_reward, 0.0))
    terminal = terminal * ~paid
    return success, terminal, paid | success | failure



def task_conditions(env, state, center, corners, rotation, velocity, active,
                    target, total_overlap, sensor_wrench):
    """Evaluate clearance and failure predicates without mutating episode state.

    All physical objects must settle and every non-target footprint must clear
    the fixed initial access region. Dropping, tilting, leaving the workspace,
    lifting, hard external loads or total sensor overrange, and forbidden robot
    contact remain independent failure channels, in the order exposed by
    training diagnostics.
    """
    cfg = env.cfg.mdp
    initial_target_distance = torch.linalg.vector_norm(center - state.initial_center, dim=-1)
    delta_q = quat_mul(rotation, quat_conjugate(state.initial_rotation))
    target_angle = 2 * torch.acos(delta_q[..., 0].abs().clamp(0, 1))
    protected = ~(((initial_target_distance > cfg.target_translation_tolerance)
                   | (target_angle > cfg.target_rotation_tolerance)) & target & active).any(-1)
    stable = ~(((torch.linalg.vector_norm(velocity[..., :3], dim=-1) > cfg.stable_linear_velocity)
                | (torch.linalg.vector_norm(velocity[..., 3:], dim=-1) > cfg.stable_angular_velocity)) & active).any(-1)
    success_candidate = (total_overlap <= cfg.success_overlap_area) & protected & stable
    stable_count = torch.where(success_candidate, state.stable_count + 1, 0)
    bottom, top = corners[..., 2].amin(-1), corners[..., 2].amax(-1)
    xy_min, xy_max = corners[..., :2].amin(-2), corners[..., :2].amax(-2)
    rules = env.spawn_rules
    workspace = ((xy_min[..., 0] < rules.x_range[0] - cfg.workspace_tolerance)
                 | (xy_max[..., 0] > rules.x_range[1] + cfg.workspace_tolerance)
                 | (xy_min[..., 1] < rules.y_range[0] - cfg.workspace_tolerance)
                 | (xy_max[..., 1] > rules.y_range[1] + cfg.workspace_tolerance)
                 | (top > rules.ceiling + cfg.workspace_tolerance))
    up = torch.zeros_like(center)
    up[..., 2] = 1
    tilted = quat_apply(rotation, up)[..., 2] < math.cos(cfg.max_tilt)
    hard = ((torch.linalg.vector_norm(sensor_wrench[:, :3], dim=-1) > cfg.force_hard)
            | (torch.linalg.vector_norm(sensor_wrench[:, 3:], dim=-1) > cfg.torque_hard))
    sensor = getattr(env, "tool_force_torque_sensor", None)
    if sensor is not None:
        # Range applies to total measured reaction before self-load removal.
        # A clipped or compensated value cannot recover that overload signal.
        hard |= sensor.valid & sensor.range_exceeded.any(-1)
    forbidden = getattr(env, "forbidden_contact", torch.zeros_like(state.failure))
    if callable(forbidden):
        forbidden = forbidden(env)
    reasons = torch.stack((((bottom < rules.shelf_top - cfg.drop_tolerance) & active).any(-1),
                           (tilted & active).any(-1), (workspace & active).any(-1),
                           ((bottom > rules.shelf_top + cfg.lift_tolerance) & active).any(-1),
                           hard, forbidden.bool()), -1)
    failure = reasons.any(-1) | ~torch.isfinite(sensor_wrench).all(-1)
    return success_candidate, stable_count, reasons, failure


def success(env):
    return update_mdp(env).success.clone()


def failure(env):
    return update_mdp(env).failure.clone()

