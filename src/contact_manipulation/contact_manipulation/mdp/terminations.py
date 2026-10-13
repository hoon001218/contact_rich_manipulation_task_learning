"""Batched physical task predicates; timeout remains Isaac Lab's native term."""

from __future__ import annotations

import math

import torch

__all__ = ["task_conditions", "terminal_transition", "success", "failure"]


def terminal_transition(success_candidate, failure, stable_count, paid, cfg):
    """Resolve failure before success and pay a terminal reward only once."""
    successful = success_candidate & (stable_count >= cfg.stable_steps) & ~failure
    payment = torch.where(failure, -cfg.failure_penalty,
                          torch.where(successful, cfg.success_reward, 0.0))
    payment = torch.where(paid, 0.0, payment)
    return successful, payment, paid | successful | failure


def task_conditions(cfg, *, center, corners, rotation, velocity, active, target,
                    initial_center, initial_rotation, total_overlap, sensor_wrench,
                    stable_count, workspace_bounds, shelf_top, ceiling,
                    sensor_range_exceeded=None, forbidden_contact=None, state_valid=None):
    """Evaluate a full parallel batch without reading tensors back to Python.

    Object tensors have shapes (N, O, 3), (N, O, 8, 3), (N, O, 4), and
    (N, O, 6); quaternions are scalar-first relative to each object's upright
    pose. Geometry and initial geometry use the fixed environment frame.
    ``workspace_bounds`` is [xmin, xmax, ymin, ymax], shape (4,) or (N, 4).
    Shelf/ceiling heights may be scalars or (N,) tensors. ``target`` and
    ``active`` are boolean object masks; total_overlap excludes the target.

    Return success candidate, updated stability count, failure reasons, failure.
    Reasons are drop, tilt, workspace exit, lift, hard load/overrange, forbidden
    robot contact, invalid state. Inactive slots never trigger a failure, even
    when padded with NaN. state_valid optionally incorporates robot, tactile, or
    other integration-specific finite checks supplied by the environment.
    """
    if corners.shape != (*center.shape[:-1], 8, 3):
        raise ValueError("Corners must have shape (environments, objects, 8, 3).")
    if (rotation.shape != (*center.shape[:-1], 4) or velocity.shape != (*center.shape[:-1], 6)
            or initial_center.shape != center.shape or initial_rotation.shape != rotation.shape
            or active.shape != center.shape[:-1] or target.shape != active.shape):
        raise ValueError("Object poses, velocities, and masks must have matching environment/object dimensions.")
    if (sensor_wrench.shape != (center.shape[0], 6)
            or total_overlap.shape != (center.shape[0],) or stable_count.shape != total_overlap.shape):
        raise ValueError("Overlap, wrench, and stability count must have one entry per environment.")

    rotation_norm = torch.linalg.vector_norm(rotation, dim=-1)
    initial_norm = torch.linalg.vector_norm(initial_rotation, dim=-1)
    valid_objects = (torch.isfinite(center).all(-1) & torch.isfinite(corners).all((-2, -1))
                     & torch.isfinite(rotation).all(-1) & torch.isfinite(velocity).all(-1)
                     & torch.isfinite(initial_center).all(-1) & torch.isfinite(initial_rotation).all(-1)
                     & (rotation_norm > 1.0e-8) & (initial_norm > 1.0e-8))
    invalid = ((~valid_objects & active).any(-1) | ~torch.isfinite(sensor_wrench).all(-1)
               | ~torch.isfinite(total_overlap) | (total_overlap < 0.0)
               | ((target & active).sum(-1) != 1))
    if state_valid is not None:
        invalid = invalid | ~state_valid

    distance = torch.linalg.vector_norm(center - initial_center, dim=-1)
    cosine_half_angle = ((rotation * initial_rotation).sum(-1).abs()
                         / (rotation_norm * initial_norm).clamp_min(1.0e-12)).clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(cosine_half_angle)
    protected = ~(((distance > cfg.target_translation_tolerance)
                   | (angle > cfg.target_rotation_tolerance)) & target & active).any(-1)
    settled = ~(((torch.linalg.vector_norm(velocity[..., :3], dim=-1) > cfg.stable_linear_velocity)
                 | (torch.linalg.vector_norm(velocity[..., 3:], dim=-1) > cfg.stable_angular_velocity))
                & active).any(-1)

    bottom, top = corners[..., 2].amin(-1), corners[..., 2].amax(-1)
    xy_min, xy_max = corners[..., :2].amin(-2), corners[..., :2].amax(-2)
    # Expand only tensor dimensions. Scalars naturally broadcast without a device copy.
    floor = shelf_top[..., None] if isinstance(shelf_top, torch.Tensor) and shelf_top.ndim else shelf_top
    roof = ceiling[..., None] if isinstance(ceiling, torch.Tensor) and ceiling.ndim else ceiling
    bounds = workspace_bounds[..., None, :]
    outside = ((xy_min[..., 0] < bounds[..., 0] - cfg.workspace_tolerance)
               | (xy_max[..., 0] > bounds[..., 1] + cfg.workspace_tolerance)
               | (xy_min[..., 1] < bounds[..., 2] - cfg.workspace_tolerance)
               | (xy_max[..., 1] > bounds[..., 3] + cfg.workspace_tolerance)
               | (top > roof + cfg.workspace_tolerance))
    up_z = 1.0 - 2.0 * rotation[..., 1:3].square().sum(-1) / rotation_norm.square().clamp_min(1.0e-12)
    tilted = up_z < math.cos(cfg.max_tilt)
    hard = ((torch.linalg.vector_norm(sensor_wrench[..., :3], dim=-1) > cfg.force_hard)
            | (torch.linalg.vector_norm(sensor_wrench[..., 3:], dim=-1) > cfg.torque_hard))
    if sensor_range_exceeded is not None:
        hard = hard | sensor_range_exceeded.any(-1) if sensor_range_exceeded.ndim == 2 else hard | sensor_range_exceeded
    forbidden = torch.zeros_like(hard) if forbidden_contact is None else forbidden_contact
    reasons = torch.stack((((bottom < floor - cfg.drop_tolerance) & active).any(-1),
                           (tilted & active).any(-1), (outside & active).any(-1),
                           ((bottom > floor + cfg.lift_tolerance) & active).any(-1),
                           hard, forbidden, invalid), -1)
    failed = reasons.any(-1)
    candidate = (total_overlap <= cfg.success_overlap_area) & protected & settled & ~failed
    next_stable_count = torch.where(candidate, stable_count + 1, 0)
    return candidate, next_stable_count, reasons, failed


def success(env):
    """Native TerminationTerm: stable clearance with a protected target."""
    from .events import get_task_state

    return get_task_state(env).success


def failure(env):
    """Native TerminationTerm: physical, sensor, or invalid-state failure."""
    from .events import get_task_state

    return get_task_state(env).failure
