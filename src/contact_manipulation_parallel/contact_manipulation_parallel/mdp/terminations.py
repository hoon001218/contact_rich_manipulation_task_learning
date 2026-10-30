"""Batched physical task predicates and shared-scene round truncation."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

__all__ = ["FAILURE_REASON_NAMES", "ObjectSupportValidity", "object_support_validity", "task_conditions",
           "terminal_transition", "scene_time_out", "success", "failure"]


# Ordered columns of TaskState.failure_reasons, shared by predicates and logging.
FAILURE_REASON_NAMES = (
    "object_support_lost", "object_tilt", "shelf_displacement", "invalid_state", "singularity",
)
_FAILURE_REASON_INDEX = {name: index for index, name in enumerate(FAILURE_REASON_NAMES)}


def scene_time_out(env):
    """Native episode timeout plus a common boundary for replacing the shared scene.

    Early reset counters remain episode-local. Scene expiry truncates every row,
    including recently restarted rows, so PPO bootstraps instead of marking failure.
    """
    expired = env.episode_length_buf >= env.max_episode_length
    runtime = getattr(env, "_episode_reset", None)
    deadline = getattr(runtime, "next_scene_step", math.inf)
    if runtime is not None and env.common_step_counter >= deadline:
        return torch.ones_like(expired)
    return expired


def terminal_transition(success_candidate, failure, stable_count, success_paid, failure_paid, cfg,
                        *, step_dt=0.01):
    """Pay success once independently of termination, and a later failure once."""
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive.")
    stable_steps = max(1, math.ceil(cfg.success_duration / step_dt - 1.e-9))
    successful = success_candidate & (stable_count >= stable_steps) & ~failure
    fresh_success = successful & ~success_paid & ~failure_paid
    fresh_failure = failure & ~failure_paid
    payment = torch.where(fresh_success, cfg.success_reward, 0.0)
    payment = torch.where(fresh_failure, -cfg.failure_penalty, payment)
    return successful, payment, success_paid | fresh_success, failure_paid | fresh_failure


class ObjectSupportValidity(NamedTuple):
    """Shared geometric gates and debounced per-object failure state, all on device."""

    progress_valid: torch.Tensor
    completion_valid: torch.Tensor
    support_loss_count: torch.Tensor
    lost: torch.Tensor
    support_distance: torch.Tensor
    stability_distance: torch.Tensor
    tilt_angle: torch.Tensor


def object_support_validity(cfg, *, center_shelf, corners_shelf, velocity, active, upright_z,
                            support_bounds, support_height, support_loss_count,
                            step_dt, support_bottom=None):
    """Evaluate shelf retention without imposing a velocity gate on progress.

    Geometry is already expressed in the *current* shelf link frame. Bounds are
    the measured work panel's [xmin,xmax,ymin,ymax], not the full shelf OBB.
    ``support_bottom`` may provide the exact collision geometry minimum in that
    frame; the conservative OBB minimum is the fallback. A signed lower bound
    rejects below-panel penetration rather than treating every negative gap as
    supported. None of these geometric checks proves a contact force exists.

    The support representative is (OBB center x/y, collision geometry bottom).
    Its Euclidean distance to the finite panel rectangle provides ONE support
    loss measure for lifting, falling, and leaving the panel. These events are
    not separately thresholded and OR-ed. Tipping remains an independent check:
    a tipped object can still touch the panel at zero support distance.
    Reward stability additionally measures the most protruding OBB corner in
    XY and canonical-up tilt. These do not change the existing failure timers.

    All object predicates, including finite checks and timers, ignore inactive
    slots. Only final completion requires settled objects. Failure timers belong
    to individual objects, so alternating transients do not accumulate together.
    """
    if (center_shelf.ndim != 3 or center_shelf.shape[-1] != 3
            or corners_shelf.shape != (*center_shelf.shape[:-1], 8, 3)
            or velocity.shape != (*center_shelf.shape[:-1], 6)
            or active.shape != center_shelf.shape[:-1] or upright_z.shape != active.shape):
        raise ValueError("Shelf-frame object geometry, velocity, and masks must have matching dimensions.")
    if support_loss_count.shape != active.shape or support_loss_count.dtype != torch.long:
        raise ValueError("Support failure counts must have one int64 entry per environment/object.")
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("step_dt must be finite and positive.")
    bounds = torch.as_tensor(support_bounds, device=center_shelf.device, dtype=center_shelf.dtype)
    height = torch.as_tensor(support_height, device=center_shelf.device, dtype=center_shelf.dtype)
    if bounds.shape not in ((4,), (center_shelf.shape[0], 4)):
        raise ValueError("Support bounds must have shape (4,) or (environments, 4).")
    if height.ndim > 1 or (height.ndim == 1 and height.shape[0] != center_shelf.shape[0]):
        raise ValueError("Support height must be a scalar or one value per environment.")
    lower = corners_shelf.amin(-2)
    upper = corners_shelf.amax(-2)
    bottom = lower[..., 2] if support_bottom is None else support_bottom
    if bottom.shape != active.shape:
        raise ValueError("Support bottom must contain one value per environment/object.")
    floor = height[..., None] if height.ndim else height
    signed_gap = bottom - floor
    # Strided views avoid an implicit CPU index tensor/copy under CUDA capture.
    bound_lo, bound_hi = bounds[..., 0::2].unsqueeze(-2), bounds[..., 1::2].unsqueeze(-2)
    valid_metadata = (torch.isfinite(bounds).all(-1) & (bounds[..., 1] > bounds[..., 0])
                      & (bounds[..., 3] > bounds[..., 2]) & torch.isfinite(height))
    finite = (torch.isfinite(center_shelf).all(-1) & torch.isfinite(corners_shelf).all((-2, -1))
              & torch.isfinite(velocity).all(-1) & torch.isfinite(upright_z) & torch.isfinite(bottom))
    xy_lo, xy_hi = lower[..., :2], upper[..., :2]
    centered = ((center_shelf[..., :2] >= bound_lo) & (center_shelf[..., :2] <= bound_hi)).all(-1)
    overlaps = ((xy_hi >= bound_lo) & (xy_lo <= bound_hi)).all(-1)
    upright = upright_z > math.cos(cfg.max_tilt) + 4 * torch.finfo(center_shelf.dtype).eps
    above_lower_limit = signed_gap >= -cfg.support_penetration_tolerance
    progress_object = (finite & centered & overlaps & upright & above_lower_limit
                       & (signed_gap <= cfg.progress_support_gap))
    contained = ((xy_lo >= bound_lo - cfg.success_xy_tolerance)
                 & (xy_hi <= bound_hi + cfg.success_xy_tolerance)).all(-1)
    settled = ((velocity[..., :3].square().sum(-1) <= cfg.stable_linear_velocity ** 2)
               & (velocity[..., 3:].square().sum(-1) <= cfg.stable_angular_velocity ** 2))
    complete_object = (finite & contained & upright & above_lower_limit & settled
                       & (signed_gap <= cfg.success_support_gap))

    closest_xy = torch.maximum(torch.minimum(center_shelf[..., :2], bound_hi), bound_lo)
    distance = torch.sqrt((center_shelf[..., :2] - closest_xy).square().sum(-1) + signed_gap.square())
    distance = torch.where(active, torch.where(finite, distance, math.inf), 0.)
    corner_xy = corners_shelf[..., :2]
    outside_xy = torch.maximum(bound_lo.unsqueeze(-2) - corner_xy,
                               corner_xy - bound_hi.unsqueeze(-2)).clamp_min(0.)
    outside_squared = outside_xy.square().sum(-1).amax(-1)
    stability_distance = torch.sqrt(outside_squared + signed_gap.square())
    stability_distance = torch.where(active, torch.where(finite, stability_distance, math.inf), 0.)
    tilt_angle = torch.acos(upright_z.clamp(-1., 1.))
    tilt_angle = torch.where(active, torch.where(finite, tilt_angle, math.inf), 0.)
    unsupported = (distance >= cfg.support_loss_distance) & active & finite
    support_steps = max(1, math.ceil(cfg.support_loss_duration / step_dt - 1.e-9))
    next_loss = torch.where(unsupported, (support_loss_count + 1).clamp_max(support_steps), 0)
    return ObjectSupportValidity(
        (progress_object | ~active).all(-1) & valid_metadata,
        (complete_object | ~active).all(-1) & valid_metadata,
        next_loss, (next_loss >= support_steps).any(-1), distance,
        stability_distance, tilt_angle,
    )


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
                    shelf_pose=None, initial_shelf_pose=None, state_valid=None,
                    failure_mask=None, singularity=None, support_validity=None, raw_goal_out=None,
                    target_protected_out=None):
    """Evaluate a full parallel batch without reading tensors back to Python.

    Object tensors have shapes (N, O, 3), (N, O, 8, 3), (N, O, 4), and
    (N, O, 6); quaternions are scalar-first relative to each object's upright
    pose. Geometry and initial geometry use the fixed environment frame.
    Native callers provide cached ``support_validity`` computed in the current
    shelf frame. Standalone callers fall back to these fixed-frame corners and
    workspace bounds; they must supply a persistent helper result to debounce
    support loss across samples. Ceiling is retained as reset metadata.
    Shelf height may be a scalar or (N,) tensor. ``target`` and
    ``active`` are boolean object masks. ``blocker_overlap`` is the designated
    blocker's overlap area with the target approach region, in square meters.
    Task success requires a protected Target, cleared Blocker, supported and
    retained settled active objects, and no enabled failure. The terminal
    transition confirms that predicate for ``success_duration`` seconds.

    ``tilt_count`` tracks consecutive tipped samples separately for each object;
    recovering upright or becoming inactive clears only that object's count.
    Return success candidate, updated stability/tilt counts, failure reasons, failure.
    Reason columns follow ``FAILURE_REASON_NAMES``; ``failure_mask`` selects only
    enabled native causes for aggregate failure without erasing diagnostics.
    ``raw_goal_out`` optionally receives the protected-target/cleared-blocker
    predicate before completion validity, so diagnostics do not recompute poses.
    ``target_protected_out`` exposes the same pose predicate to reward gates.
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
    if target_protected_out is not None:
        if target_protected_out.shape != blocker_overlap.shape or target_protected_out.dtype != torch.bool:
            raise ValueError("Target-protection output must have one boolean entry per environment.")
        target_protected_out.copy_(protected)

    up_z = 1.0 - 2.0 * rotation[..., 1:3].square().sum(-1) / rotation_norm.square().clamp_min(1.0e-12)
    # Inclusive angular boundary, allowing only floating-point round-off at 80 degrees.
    tilted = (up_z <= math.cos(cfg.max_tilt) + 4 * torch.finfo(rotation.dtype).eps) & active & valid_objects
    tilt_steps = max(1, math.ceil(cfg.tilt_duration / step_dt - 1.e-9))
    next_tilt_count = torch.where(tilted, (tilt_count + 1).clamp_max(tilt_steps), 0)
    displaced = torch.zeros_like(invalid)
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
    singular = torch.zeros_like(invalid)
    if singularity is not None:
        if singularity.shape != blocker_overlap.shape:
            raise ValueError("Singularity margin must contain one value per environment.")
        singular = ~torch.isfinite(singularity) | (singularity <= cfg.singularity_termination_threshold)
    if support_validity is None:
        support_validity = object_support_validity(
            cfg, center_shelf=center, corners_shelf=corners, velocity=velocity, active=active,
            upright_z=up_z, support_bounds=workspace_bounds, support_height=shelf_top,
            support_loss_count=torch.zeros_like(tilt_count),
            step_dt=step_dt,
        )
    reasons = torch.stack((support_validity.lost, (next_tilt_count >= tilt_steps).any(-1),
                           displaced, invalid, singular), -1)
    failed = (reasons if failure_mask is None else reasons & failure_mask).any(-1)
    raw_goal = (blocker_overlap >= 0.) & (blocker_overlap <= cfg.success_overlap_area) & protected
    if raw_goal_out is not None:
        if raw_goal_out.shape != blocker_overlap.shape or raw_goal_out.dtype != torch.bool:
            raise ValueError("Raw-goal output must have one boolean entry per environment.")
        raw_goal_out.copy_(raw_goal)
    candidate = (raw_goal & support_validity.completion_valid & ~invalid & ~failed
                 & ~tilted.any(-1))
    next_stable_count = torch.where(candidate, stable_count + 1, 0)
    return candidate, next_stable_count, next_tilt_count, reasons, failed


def success(env):
    """Native TerminationTerm: a sustained, settled valid shelf rearrangement."""
    from .events import get_task_state

    return get_task_state(env).success


def failure(env, reason: str | None = None):
    """Read an aggregate failure or one cached cause for native termination logs.

    Several causes may be true in the same environment. Native cause logs stay
    separate; the shared terminal transition pays their aggregate failure once.
    """
    from .events import get_task_state

    if reason is None:
        return get_task_state(env).failure
    if reason not in _FAILURE_REASON_INDEX:
        raise ValueError(f"Unknown failure reason {reason!r}; expected one of {FAILURE_REASON_NAMES}.")
    return get_task_state(env).failure_reasons[:, _FAILURE_REASON_INDEX[reason]]
