"""Batched policy measurements; scene roles and validity are never features.

Initial camera geometry is frozen in robot-base coordinates. Only its reference
frame changes with the measured EEF pose. Slots 0/1 are target/blocker; history
contains the current and previous sample, newest first. The critic tracks the
current true geometry of these same initially recognized objects. All quaternions use
``(w, x, y, z)``. Quaternion operations use Isaac Lab tensor utilities.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply as _lab_quat_apply, quat_conjugate, quat_mul as _lab_quat_mul, quat_unique

__all__ = [
    "prepare_initial_geometry", "select_initial_geometry_indices", "visible_object_image_x",
    "adjacent_geometry_permutation", "geometry_in_eef_frame", "pack_policy_observation",
    "dynamic_sample", "calibrate_wrench",
    "policy_observation", "initial_scene_geometry", "critic_observation", "critic_scene_geometry",
    "current_robot_sample", "previous_robot_sample",
    "quat_apply", "quat_conjugate", "quat_mul",
]

SI_200_8_RANGES = (200.0, 200.0, 360.0, 8.0, 8.0, 8.0)
SI_200_8_RESOLUTION = (0.1, 0.1, 0.1, 0.005, 0.005, 0.005)


# Preserve broadcastable task tensors while using Isaac Lab's quaternion math.


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    shape = torch.broadcast_shapes(quaternion.shape[:-1], vector.shape[:-1])
    return _lab_quat_apply(quaternion.expand(*shape, 4), vector.expand(*shape, 3))


def quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _lab_quat_mul(*torch.broadcast_tensors(left, right))


def select_initial_geometry_indices(
    geometry_b: torch.Tensor, valid: torch.Tensor, roles: torch.Tensor, capacity: int,
    *, image_x: torch.Tensor | None = None, preserve_distractor_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select nearby visible objects, then order their slots by image column.

    The inputs have already been validated or produced by the visible-surface
    fitter. Distances use only observed OBB centers, to the nearer visible role.
    An unseen role contributes no reference point. With neither role visible,
    source order breaks the otherwise undefined proximity ranking. Ties also
    keep source order; hidden objects never fill an observation slot. ``image_x``
    contains visible-pixel centroids, never privileged object projections.
    Reset consumes an already selected/sorted snapshot with
    ``preserve_distractor_order=True`` instead of ranking it a second time.
    """
    batch, source_capacity = valid.shape
    role_sources = roles.clamp_min(0)
    role_valid = (roles >= 0) & valid.gather(1, role_sources)
    role_centers = geometry_b[..., :3].gather(1, role_sources[..., None].expand(-1, -1, 3))
    distances = (geometry_b[:, :, None, :3] - role_centers[:, None]).square().sum(-1)
    distances = torch.where(role_valid[:, None], distances, float("inf")).amin(-1)
    distances = torch.where(role_valid.any(-1, keepdim=True), distances, 0.)
    source_indices = torch.arange(source_capacity, device=geometry_b.device).expand(batch, -1)
    if preserve_distractor_order:
        distances = source_indices.to(geometry_b.dtype)
    distractor_valid = valid & (source_indices != roles[:, :1]) & (source_indices != roles[:, 1:2])
    nearest = torch.where(distractor_valid, distances, float("inf")).argsort(dim=-1, stable=True)
    nearest = nearest[:, :max(0, capacity - 2)]
    nearest_valid = distractor_valid.gather(1, nearest)
    if image_x is not None:
        if image_x.shape != valid.shape:
            raise ValueError("Image columns must have the same shape as geometry validity.")
        columns = torch.where(nearest_valid, image_x.gather(1, nearest), float("inf"))
        order = columns.argsort(dim=-1, stable=True)
        nearest = nearest.gather(1, order)
        nearest_valid = nearest_valid.gather(1, order)
    selected = roles.new_full((batch, capacity), -1)
    selected_valid = valid.new_zeros((batch, capacity))
    selected[:, :2] = torch.where(role_valid, roles, -1)
    selected_valid[:, :2] = role_valid
    selected[:, 2:2 + nearest.shape[1]] = torch.where(nearest_valid, nearest, -1)
    selected_valid[:, 2:2 + nearest.shape[1]] = nearest_valid
    return selected, selected_valid


def visible_object_image_x(
    labels: torch.Tensor, object_count: int, image_width: int, *, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reduce ray first-hit labels [N,H*W] to visible-pixel centroid columns.

    Background contributes neither counts nor columns. Scatter reductions keep
    the cost linear in pixels, without a pixel-by-object tensor or host reads.
    Objects with no visible pixel receive +inf and cannot lead the ordering.
    """
    if labels.ndim != 2 or image_width < 1 or labels.shape[1] % image_width or object_count < 1:
        raise ValueError("Expected flattened ray images, a matching width and a positive object count.")
    valid = (labels >= 0) & (labels < object_count)
    indices = labels.long().clamp(0, object_count - 1)
    columns = torch.arange(labels.shape[1], device=labels.device).remainder(image_width).to(dtype)
    weights = valid.to(dtype)
    counts = torch.zeros((labels.shape[0], object_count), device=labels.device, dtype=dtype)
    sums = torch.zeros_like(counts)
    counts.scatter_add_(1, indices, weights)
    sums.scatter_add_(1, indices, weights * columns)
    return torch.where(counts > 0, sums / counts.clamp_min(1), float("inf"))


def adjacent_geometry_permutation(
    valid: torch.Tensor, probability: float = 0.2, *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw reset-only neighboring swaps, preserving roles, padding and IDs.

    A random even/odd edge partition prevents overlapping swaps. An eligible
    edge is selected with probability 2*p, making each valid adjacent pair's
    marginal swap probability p. Every object moves at most one slot. The same
    returned permutation must be applied to actor geometry and critic IDs.
    """
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("Geometry validity must be a boolean [N,slots] tensor.")
    if not math.isfinite(probability) or not 0.0 <= probability <= 0.5:
        raise ValueError("Adjacent swap probability must be in [0, 0.5].")
    batch, count = valid.shape
    indices = torch.arange(count, device=valid.device).expand(batch, -1)
    if count <= 3 or probability == 0.0:
        return indices
    edges = torch.arange(count - 3, device=valid.device)
    parity = torch.randint(2, (batch, 1), device=valid.device, generator=generator)
    draws = torch.rand((batch, count - 3), device=valid.device, generator=generator)
    swaps = ((edges.remainder(2) == parity) & (draws < 2.0 * probability)
             & valid[:, 2:-1] & valid[:, 3:]).long()
    shift = torch.nn.functional.pad(swaps, (2, 1)) - torch.nn.functional.pad(swaps, (3, 0))
    return indices + shift


def prepare_initial_geometry(
    geometry_b: torch.Tensor,
    valid: torch.Tensor,
    target_indices: torch.Tensor,
    blocker_indices: torch.Tensor,
    *,
    capacity: int = 10,
    return_indices: bool = False,
    preserve_distractor_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reserve role slots and select nearby observed distractors once on reset.

    Inputs are ``[N, M, 10]`` geometry, ``[N, M]`` validity and ``[N]``
    source-slot indices. Index -1 denotes an unassociated role. An associated
    but invisible role also leaves its reserved slot zero. The remaining slots
    contain recognized objects nearest either visible role's observed OBB center.
    Excess recognized objects are omitted from the policy input, not physics or
    the ray scene. No hidden ground-truth position enters the ranking. Valid OBBs
    must be finite, have positive extents and unit quaternions (norm tolerance
    1e-3); accepted floating-point drift is normalized in the stored copy.

    Semantic checks synchronize the device here, at initial-snapshot acceptance;
    observation construction never repeats this sort or these checks. Optional
    source indices bind the critic to the exact same objects and slot order.
    """
    if geometry_b.ndim != 3 or geometry_b.shape[-1] != 10 or valid.shape != geometry_b.shape[:2]:
        raise ValueError("Expected geometry [N, M, 10] and validity [N, M].")
    if valid.dtype != torch.bool:
        raise ValueError("Geometry validity must be boolean.")
    if not geometry_b.is_floating_point():
        raise ValueError("Geometry must have floating-point dtype.")
    quaternion_norm = torch.linalg.vector_norm(geometry_b[..., 3:7], dim=-1)
    geometry_well_formed = (torch.isfinite(geometry_b).all(dim=-1)
                           & (geometry_b[..., 7:10] > 0).all(dim=-1)
                           & ((quaternion_norm - 1.0).abs() <= 1e-3))
    if bool((valid & ~geometry_well_formed).any()):
        raise ValueError("Valid geometry must be finite, with positive extents and unit quaternions.")
    geometry_b = torch.where(valid[..., None], geometry_b, 0.0)
    geometry_b[..., 3:7] /= torch.where(valid, quaternion_norm, 1.0)[..., None]
    batch, source_capacity = valid.shape
    if capacity < 2 or source_capacity < 1:
        raise ValueError("At least two output role slots and one source slot are required.")
    if target_indices.shape != (batch,) or blocker_indices.shape != (batch,):
        raise ValueError("Target and blocker indices must each have shape [N].")
    if target_indices.dtype != torch.long or blocker_indices.dtype != torch.long:
        raise ValueError("Target and blocker indices must use torch.long.")
    roles = torch.stack((target_indices, blocker_indices), dim=-1)
    if bool(((roles < -1) | (roles >= source_capacity)).any()):
        raise ValueError("Role indices must be -1 or refer to a source geometry slot.")
    if bool(((target_indices == blocker_indices) & (target_indices >= 0)).any()):
        raise ValueError("Target and blocker must be different objects.")
    indices, output_valid = select_initial_geometry_indices(
        geometry_b, valid, roles, capacity, preserve_distractor_order=preserve_distractor_order)
    output = geometry_b.gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, 10))
    result = torch.where(output_valid[..., None], output, 0.0)
    return (result, output_valid, indices) if return_indices else (result, output_valid)


def geometry_in_eef_frame(
    geometry: torch.Tensor, valid: torch.Tensor, tcp_pose_b: torch.Tensor,
) -> torch.Tensor:
    """Re-express initial OBBs in the current EEF frame; leave padding zero."""
    inverse_rotation = quat_conjugate(tcp_pose_b[:, None, 3:7])
    position = quat_apply(inverse_rotation, geometry[..., :3] - tcp_pose_b[:, None, :3])
    orientation = quat_unique(quat_mul(inverse_rotation, geometry[..., 3:7]))
    transformed = torch.cat((position, orientation, geometry[..., 7:10]), dim=-1)
    return torch.where(valid[..., None], transformed, 0.0)


def pack_policy_observation(geometry_eef: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Concatenate preordered/padded geometry and newest-first [N, 2, 72] history.

    The producer zeroes missing geometry/history before this operation. There
    are no mask or role channels. With 10 geometry slots the result is 244D.
    """
    if geometry_eef.ndim != 3 or geometry_eef.shape[-1] != 10:
        raise ValueError("Geometry must have shape [N, M, 10].")
    if history.shape != (geometry_eef.shape[0], 2, 72):
        raise ValueError("History must contain current and previous samples with shape [N, 2, 72].")
    return torch.cat((geometry_eef.flatten(1), history.flatten(1)), dim=-1)


def dynamic_sample(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    tcp_pose_b: torch.Tensor,
    measured_wrench: torch.Tensor,
    tactile_contacts: torch.Tensor,
    tactile_valid: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    joint_position_scale: float = math.pi,
    joint_velocity_scale: float = 2.0,
    wrench_ranges: torch.Tensor | None = None,
) -> torch.Tensor:
    """Assemble one 72D robot/sensor/action sample with no device-to-host reads.

    The 12 independent joints are arm 6 then hand 6; mimic joints are excluded.
    F/T remains in the physical sensor frame, including hand weight and inertia.
    Pass cached ``wrench_ranges`` in the environment hot path to avoid rebuilding
    a device constant. On reset,
    the caller supplies zero previous action and false tactile validity.
    """
    batch = joint_pos.shape[0]
    if joint_pos.shape != (batch, 12) or joint_vel.shape != (batch, 12):
        raise ValueError("Joint measurements must contain 12 independent joints.")
    if tcp_pose_b.shape != (batch, 7) or measured_wrench.shape != (batch, 6):
        raise ValueError("EEF pose and measured wrench must have shapes [N, 7] and [N, 6].")
    if tactile_contacts.shape != (batch, 17) or tactile_valid.shape != (batch, 17):
        raise ValueError("Tactile measurements and validity must each contain 17 channels.")
    if previous_action.shape != (batch, 18):
        raise ValueError("The preceding action must have 18 channels.")
    if joint_position_scale <= 0 or joint_velocity_scale <= 0:
        raise ValueError("Joint normalization scales must be positive.")
    if wrench_ranges is None:
        wrench_ranges = measured_wrench.new_tensor(SI_200_8_RANGES)
    normalized_wrench = (measured_wrench / wrench_ranges).clamp(-5.0, 5.0)
    tactile = torch.where(tactile_valid, tactile_contacts.to(joint_pos.dtype), -1.0)
    return torch.cat((joint_pos / joint_position_scale, joint_vel / joint_velocity_scale,
                      tcp_pose_b, normalized_wrench, tactile, previous_action), dim=-1)


def calibrate_wrench(
    ideal_wrench: torch.Tensor,
    ranges: torch.Tensor | None = None,
    resolution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SI-200-8 clipping and deterministic resolution to the joint reaction.

    Return measured raw wrench and per-axis overload flags. A sensor provider
    that already calibrated its reading must not apply this operation again.
    Cache ranges/resolution tensors in the runtime for parallel stepping.
    """
    if ranges is None:
        ranges = ideal_wrench.new_tensor(SI_200_8_RANGES)
    if resolution is None:
        resolution = ideal_wrench.new_tensor(SI_200_8_RESOLUTION)
    exceeded = ideal_wrench.abs() > ranges
    limited = ideal_wrench.clamp(min=-ranges, max=ranges)
    measured = torch.round(limited / resolution) * resolution
    return measured, exceeded


def policy_observation(env) -> torch.Tensor:
    """Shared packed snapshot for native observation terms and timeout bootstrap."""
    from .events import get_task_state

    state = get_task_state(env)
    if state.observation is None:
        geometry = geometry_in_eef_frame(state.geometry_b, state.geometry_valid, state.tcp_pose_b)
        state.observation = pack_policy_observation(geometry, state.history)
    return state.observation


def initial_scene_geometry(env) -> torch.Tensor:
    """Initial visible geometry in the current EEF frame; target/blocker first."""
    return policy_observation(env)[:, :100]


def critic_observation(env) -> torch.Tensor:
    """Current GT for reset-time recognized IDs, with the same robot history.

    Membership, role slots and padding stay fixed until reset. No visibility
    query runs here, and objects absent from initial vision are never inserted.
    A fresh packed tensor preserves completed observations across auto-reset.
    """
    from .events import get_task_state

    state = get_task_state(env)
    if state.critic_observation is None:
        valid = state.geometry_valid & (state.critic_source_indices >= 0)
        geometry = geometry_in_eef_frame(state.critic_geometry_b, valid, state.tcp_pose_b)
        state.critic_observation = pack_policy_observation(geometry, state.history)
    return state.critic_observation


def critic_scene_geometry(env) -> torch.Tensor:
    """True current OBBs in the current EEF frame for initial vision slots."""
    return critic_observation(env)[:, :100]


def current_robot_sample(env) -> torch.Tensor:
    """Current robot, SI-200-8 F/T reaction, tactile and action measurements."""
    return policy_observation(env)[:, 100:172]


def previous_robot_sample(env) -> torch.Tensor:
    """Previous completed sample, zeroed for rows beginning a new episode."""
    return policy_observation(env)[:, 172:244]
