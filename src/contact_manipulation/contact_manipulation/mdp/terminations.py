"""Batched physical task predicates and shared-scene round truncation."""

from __future__ import annotations

import math

import torch

__all__ = ["task_conditions", "terminal_transition", "scene_time_out", "success", "failure"]


def scene_time_out(env):
    """Native episode timeout plus a common boundary for replacing the shared scene.

    Early reset counters remain episode-local. Scene expiry truncates every row,
    including recently restarted rows, so PPO bootstraps instead of marking failure.
    A native shared round starts with every episode counter at zero. Until its
    deadline, no episode can reach the equally long episode limit. Publish that
    host-side fact so PPO can skip final-state inference without a CUDA scalar
    read. Unrecognized/external environments retain the generic bootstrap path.
    """
    expired = env.episode_length_buf >= env.max_episode_length
    runtime = getattr(env, "_episode_reset", None)
    deadline = getattr(runtime, "next_scene_step", math.inf)
    start = getattr(runtime, "scene_start_step", None)
    bounded_round = (start is not None and hasattr(runtime, "shared_template")
                     and 0 <= env.common_step_counter - start < deadline - start <= env.max_episode_length)
    if hasattr(env, "extras"):
        env.extras["time_outs_possible"] = not bounded_round
    if runtime is not None and env.common_step_counter >= deadline:
        return torch.ones_like(expired)
    return expired


def terminal_transition(success_candidate, failure, stable_count, paid, cfg):
    """Resolve failure before success and pay a terminal reward only once."""
    successful = success_candidate & (stable_count >= cfg.stable_steps) & ~failure
    payment = torch.where(failure, -cfg.failure_penalty,
                          torch.where(successful, cfg.success_reward, 0.0))
    payment = torch.where(paid, 0.0, payment)
    return successful, payment, paid | successful | failure


def task_conditions(cfg, *, center, corners, rotation, velocity, active, target,
                    initial_center, initial_rotation, blocker_overlap, sensor_wrench,
                    stable_count, workspace_bounds, shelf_top, ceiling,
                    sensor_range_exceeded=None, shelf_pose=None, initial_shelf_pose=None, state_valid=None):
    """Evaluate a full parallel batch without reading tensors back to Python.

    Object tensors have shapes (N, O, 3), (N, O, 8, 3), (N, O, 4), and
    (N, O, 6); quaternions are scalar-first relative to each object's upright
    pose. Geometry and initial geometry use the fixed environment frame.
    ``workspace_bounds`` is [xmin, xmax, ymin, ymax], shape (4,) or (N, 4).
    Shelf/ceiling heights may be scalars or (N,) tensors. ``target`` and
    ``active`` are boolean object masks. ``blocker_overlap`` is the designated
    blocker's overlap area with the target approach region, in square meters.
    Other objects do not contribute to clearance; their physical safety and
    settlement checks still apply.

    Return success candidate, updated stability count, failure reasons, failure.
    Reasons are drop, tilt, workspace exit, lift, hard load/overrange, shelf
    displacement, invalid state. Inactive slots never trigger a failure, even
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
            or blocker_overlap.shape != (center.shape[0],) or stable_count.shape != blocker_overlap.shape):
        raise ValueError("Overlap, wrench, and stability count must have one entry per environment.")

    rotation_norm = torch.linalg.vector_norm(rotation, dim=-1)
    initial_norm = torch.linalg.vector_norm(initial_rotation, dim=-1)
    valid_objects = (torch.isfinite(center).all(-1) & torch.isfinite(corners).all((-2, -1))
                     & torch.isfinite(rotation).all(-1) & torch.isfinite(velocity).all(-1)
                     & torch.isfinite(initial_center).all(-1) & torch.isfinite(initial_rotation).all(-1)
                     & (rotation_norm > 1.0e-8) & (initial_norm > 1.0e-8))
    invalid = ((~valid_objects & active).any(-1) | ~torch.isfinite(sensor_wrench).all(-1)
               | ~torch.isfinite(blocker_overlap) | (blocker_overlap < 0.0)
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
    displaced = torch.zeros_like(hard)
    if shelf_pose is not None or initial_shelf_pose is not None:
        if (shelf_pose is None or initial_shelf_pose is None
                or shelf_pose.shape != (center.shape[0], 7) or initial_shelf_pose.shape != shelf_pose.shape):
            raise ValueError("Shelf poses must both have shape [environments, 7].")
        q, q0 = shelf_pose[:, 3:], initial_shelf_pose[:, 3:]
        norm, norm0 = q.norm(dim=-1), q0.norm(dim=-1)
        invalid = invalid | (~torch.isfinite(shelf_pose).all(-1) | ~torch.isfinite(initial_shelf_pose).all(-1)
                             | (norm < 1.e-8) | (norm0 < 1.e-8))
        # Compare cos(theta/2) directly: q and -q are the same rotation; no acos needed.
        cosine = ((q * q0).sum(-1).abs() / (norm * norm0).clamp_min(1.e-12)).clamp(0., 1.)
        displaced = ((shelf_pose[:, :3] - initial_shelf_pose[:, :3]).square().sum(-1)
                     > cfg.shelf_translation_tolerance ** 2) | (cosine < math.cos(cfg.shelf_rotation_tolerance / 2))
    reasons = torch.stack((((bottom < floor - cfg.drop_tolerance) & active).any(-1),
                           (tilted & active).any(-1), (outside & active).any(-1),
                           ((bottom > floor + cfg.lift_tolerance) & active).any(-1),
                           hard, displaced, invalid), -1)
    failed = reasons.any(-1)
    candidate = (blocker_overlap <= cfg.success_overlap_area) & protected & settled & ~failed
    next_stable_count = torch.where(candidate, stable_count + 1, 0)
    return candidate, next_stable_count, reasons, failed


def success(env):
    """Native TerminationTerm: stable blocker clearance with a protected target."""
    from .events import get_task_state

    return get_task_state(env).success


def failure(env):
    """Native TerminationTerm: physical, sensor, or invalid-state failure."""
    from .events import get_task_state

    return get_task_state(env).failure
