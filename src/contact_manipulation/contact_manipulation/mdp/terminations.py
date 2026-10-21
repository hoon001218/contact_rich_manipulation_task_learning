"""Batched physical task predicates and shared-scene round truncation."""

from __future__ import annotations

import math

import torch

__all__ = ["FAILURE_REASON_NAMES", "task_conditions", "terminal_transition", "scene_time_out", "success", "failure"]


# Ordered columns of TaskState.failure_reasons, shared by predicates and logging.
FAILURE_REASON_NAMES = (
    "object_drop", "object_tilt", "hard_load_or_overrange", "shelf_displacement", "invalid_state", "singularity",
)
_FAILURE_REASON_INDEX = {name: index for index, name in enumerate(FAILURE_REASON_NAMES)}


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


def terminal_transition(success_candidate, failure, stable_count, paid, cfg, *, success_enabled=True):
    """Report success independently and pay its bonus once; all other cases pay zero."""
    successful = success_candidate & (stable_count >= cfg.stable_steps)
    completed = successful & success_enabled
    payment = torch.where(completed, cfg.success_reward, 0.0)
    payment = torch.where(paid, 0.0, payment)
    return successful, payment, paid | completed | failure


def configured_terminations(cfg, *, device):
    """Bind native enabled terms once; raw diagnostic causes remain available.

    Standalone tensor callers without native configuration retain every cause.
    A configured cause contributes only if its non-timeout TerminationTerm is
    enabled. This also handles renamed terms and the aggregate failure term.
    """
    if cfg is None:
        return torch.ones(len(FAILURE_REASON_NAMES), device=device, dtype=torch.bool), True
    terms = cfg.values() if isinstance(cfg, dict) else vars(cfg).values()
    enabled, success_enabled = set(), False
    for term in terms:
        if term is None or getattr(term, "time_out", False):
            continue
        function = getattr(term, "func", None)
        if function is success:
            success_enabled = True
        elif function is failure:
            reason = getattr(term, "params", {}).get("reason")
            if reason is None:
                enabled.update(FAILURE_REASON_NAMES)
            elif reason in _FAILURE_REASON_INDEX:
                enabled.add(reason)
    return torch.tensor([name in enabled for name in FAILURE_REASON_NAMES],
                        device=device, dtype=torch.bool), success_enabled


def task_conditions(cfg, *, center, corners, rotation, velocity, active, target,
                    initial_center, initial_rotation, blocker_overlap, sensor_wrench,
                    stable_count, tilt_count, step_dt, workspace_bounds, shelf_top, ceiling,
                    sensor_range_exceeded=None, shelf_pose=None, initial_shelf_pose=None, state_valid=None,
                    failure_mask=None, singularity=None):
    """Evaluate a full parallel batch without reading tensors back to Python.

    Object tensors have shapes (N, O, 3), (N, O, 8, 3), (N, O, 4), and
    (N, O, 6); quaternions are scalar-first relative to each object's upright
    pose. Geometry and initial geometry use the fixed environment frame.
    Workspace/ceiling metadata is retained for callers but is not a failure gate.
    Shelf height may be a scalar or (N,) tensor. ``target`` and
    ``active`` are boolean object masks. ``blocker_overlap`` is the designated
    blocker's overlap area with the target approach region, in square meters.
    Task success requires only a protected Target and cleared Blocker, held for
    ``stable_steps`` samples. Object velocity, other objects, and failure causes
    do not gate that task predicate. Native enabled failures remain separate.

    ``tilt_count`` tracks consecutive tipped samples separately for each object;
    recovering upright or becoming inactive clears only that object's count.
    Return success candidate, updated stability/tilt counts, failure reasons, failure.
    Reason columns follow ``FAILURE_REASON_NAMES``; ``failure_mask`` selects only
    enabled native causes for aggregate failure without erasing diagnostics.
    Inactive slots never trigger a failure, even
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
    if tilt_count.shape != active.shape or tilt_count.dtype != torch.long:
        raise ValueError("Tilt counts must have one int64 entry per environment/object.")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive.")

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
    within_target_tolerance = ((distance <= cfg.target_translation_tolerance)
                               & (angle <= cfg.target_rotation_tolerance)
                               & (rotation_norm > 1.e-8) & (initial_norm > 1.e-8))
    target_active = target & active
    protected = ((target_active.sum(-1) == 1)
                 & (within_target_tolerance | ~target_active).all(-1))

    bottom = corners[..., 2].amin(-1)
    # Expand only tensor dimensions. Scalars naturally broadcast without a device copy.
    floor = shelf_top[..., None] if isinstance(shelf_top, torch.Tensor) and shelf_top.ndim else shelf_top
    up_z = 1.0 - 2.0 * rotation[..., 1:3].square().sum(-1) / rotation_norm.square().clamp_min(1.0e-12)
    # Inclusive angular boundary, allowing only floating-point round-off at 80 degrees.
    tilted = (up_z <= math.cos(cfg.max_tilt) + 4 * torch.finfo(rotation.dtype).eps) & active & valid_objects
    tilt_steps = max(1, math.ceil(cfg.tilt_duration / step_dt - 1.e-9))
    next_tilt_count = torch.where(tilted, (tilt_count + 1).clamp_max(tilt_steps), 0)
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
    singular = torch.zeros_like(hard)
    if singularity is not None:
        if singularity.shape != blocker_overlap.shape:
            raise ValueError("Singularity margin must contain one value per environment.")
        singular = ~torch.isfinite(singularity) | (singularity <= cfg.singularity_termination_threshold)
    reasons = torch.stack((((bottom < floor - cfg.drop_tolerance) & active).any(-1),
                           (next_tilt_count >= tilt_steps).any(-1),
                           hard, displaced, invalid, singular), -1)
    failed = (reasons if failure_mask is None else reasons & failure_mask).any(-1)
    candidate = ((blocker_overlap >= 0.) & (blocker_overlap <= cfg.success_overlap_area)
                 & protected)
    next_stable_count = torch.where(candidate, stable_count + 1, 0)
    return candidate, next_stable_count, next_tilt_count, reasons, failed


def success(env):
    """Native TerminationTerm: sustained blocker clearance with a protected target."""
    from .events import get_task_state

    return get_task_state(env).success


def failure(env, reason: str | None = None):
    """Read an aggregate failure or one cached cause for native termination logs.

    Several causes may be true in the same environment. The termination manager
    combines them with OR; failure alone does not produce a terminal payment.
    """
    from .events import get_task_state

    if reason is None:
        return get_task_state(env).failure
    if reason not in _FAILURE_REASON_INDEX:
        raise ValueError(f"Unknown failure reason {reason!r}; expected one of {FAILURE_REASON_NAMES}.")
    return get_task_state(env).failure_reasons[:, _FAILURE_REASON_INDEX[reason]]
