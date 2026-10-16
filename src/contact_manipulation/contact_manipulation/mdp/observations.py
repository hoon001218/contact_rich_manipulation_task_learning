"""Batched policy measurements; scene roles and validity are never features.

Initial camera geometry is frozen in robot-base coordinates. Only its reference
frame changes with the measured EEF pose. Slots 0/1 are target/blocker; history
contains the current and previous sample, newest first. All quaternions use
``(w, x, y, z)``. Quaternion operations use Isaac Lab tensor utilities.
"""

from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_apply as _lab_quat_apply, quat_conjugate, quat_mul as _lab_quat_mul, quat_unique

__all__ = [
    "prepare_initial_geometry", "geometry_in_eef_frame", "pack_policy_observation",
    "dynamic_sample", "self_motion_wrench", "calibrate_wrench", "compensate_wrench",
    "policy_observation", "initial_scene_geometry", "current_robot_sample", "previous_robot_sample",
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


def prepare_initial_geometry(
    geometry_b: torch.Tensor,
    valid: torch.Tensor,
    target_indices: torch.Tensor,
    blocker_indices: torch.Tensor,
    *,
    capacity: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve role slots and compact observed distractors **once on reset**.

    Inputs are ``[N, M, 10]`` geometry, ``[N, M]`` validity and ``[N]``
    source-slot indices. Index -1 denotes an unassociated role. An associated
    but invisible role also leaves its reserved slot zero. Distractors retain
    their original order. No observed object is silently discarded. Valid OBBs
    must be finite, have positive extents and unit quaternions (norm tolerance
    1e-3); accepted floating-point drift is normalized in the stored copy.

    Semantic checks synchronize the device here, at initial-snapshot acceptance;
    observation construction never repeats this sort or these checks.
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
    role_sources = roles.clamp_min(0)
    role_valid = (roles >= 0) & valid.gather(1, role_sources)
    role_geometry = geometry_b.gather(1, role_sources[..., None].expand(-1, -1, 10))
    role_geometry = torch.where(role_valid[..., None], role_geometry, 0.0)

    source_indices = torch.arange(source_capacity, device=geometry_b.device).expand(batch, -1)
    distractor_valid = valid & (source_indices != target_indices[:, None]) & (
        source_indices != blocker_indices[:, None]
    )
    if bool((distractor_valid.sum(-1) > capacity - 2).any()):
        raise ValueError("Geometry capacity cannot hold both reserved roles and all observed distractors.")
    distractor_indices = torch.where(
        distractor_valid, source_indices, source_indices + source_capacity
    ).argsort(dim=-1)[:, :capacity - 2]
    count = distractor_indices.shape[-1]
    compact_valid = distractor_valid.gather(1, distractor_indices)
    compact_geometry = geometry_b.gather(1, distractor_indices[..., None].expand(-1, -1, 10))
    compact_geometry = torch.where(compact_valid[..., None], compact_geometry, 0.0)
    output = geometry_b.new_zeros((batch, capacity, 10))
    output_valid = valid.new_zeros((batch, capacity))
    output[:, :2], output_valid[:, :2] = role_geometry, role_valid
    output[:, 2:2 + count], output_valid[:, 2:2 + count] = compact_geometry, compact_valid
    return output, output_valid


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
    compensated_wrench: torch.Tensor,
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
    F/T remains in the physical sensor frame. Pass cached ``wrench_ranges`` in
    the environment hot path to avoid rebuilding a device constant. On reset,
    the caller supplies zero previous action and false tactile validity.
    """
    batch = joint_pos.shape[0]
    if joint_pos.shape != (batch, 12) or joint_vel.shape != (batch, 12):
        raise ValueError("Joint measurements must contain 12 independent joints.")
    if tcp_pose_b.shape != (batch, 7) or compensated_wrench.shape != (batch, 6):
        raise ValueError("EEF pose and compensated wrench must have shapes [N, 7] and [N, 6].")
    if tactile_contacts.shape != (batch, 17) or tactile_valid.shape != (batch, 17):
        raise ValueError("Tactile measurements and validity must each contain 17 channels.")
    if previous_action.shape != (batch, 18):
        raise ValueError("The preceding action must have 18 channels.")
    if joint_position_scale <= 0 or joint_velocity_scale <= 0:
        raise ValueError("Joint normalization scales must be positive.")
    if wrench_ranges is None:
        wrench_ranges = compensated_wrench.new_tensor(SI_200_8_RANGES)
    normalized_wrench = (compensated_wrench / wrench_ranges).clamp(-5.0, 5.0)
    tactile = torch.where(tactile_valid, tactile_contacts.to(joint_pos.dtype), -1.0)
    return torch.cat((joint_pos / joint_position_scale, joint_vel / joint_velocity_scale,
                      tcp_pose_b, normalized_wrench, tactile, previous_action.clamp(-1.0, 1.0)), dim=-1)


def self_motion_wrench(
    masses: torch.Tensor,
    inertias: torch.Tensor,
    com_position: torch.Tensor,
    body_rotation: torch.Tensor,
    com_acceleration: torch.Tensor,
    angular_velocity: torch.Tensor,
    sensor_position: torch.Tensor,
    sensor_rotation: torch.Tensor,
    gravity: torch.Tensor,
) -> torch.Tensor:
    """Sum Newton--Euler distal-body self-load about the sensor center.

    Body arrays have shape [N, B, ...]; masses [N, B], inertia [N, B, 3, 3]
    is the COM tensor in the actor/link frame. Native COM accelerations are
    [linear, angular] in world coordinates, and angular velocities are world
    measurements from the same completed physics step. Sensor pose is world
    [N, 3]/[N, 4]. Output is parent-on-child reaction [N, 6] in sensor axes.
    Both articulated-finger motion and the gyroscopic term are included.
    """
    forces = masses[..., None] * (com_acceleration[..., :3] - gravity)
    inverse_body = quat_conjugate(body_rotation)
    alpha = quat_apply(inverse_body, com_acceleration[..., 3:6])
    omega = quat_apply(inverse_body, angular_velocity)
    angular_momentum = (inertias @ omega[..., None]).squeeze(-1)
    moment_body = (inertias @ alpha[..., None]).squeeze(-1) + torch.linalg.cross(
        omega, angular_momentum, dim=-1
    )
    moments = torch.linalg.cross(com_position - sensor_position[:, None, :], forces, dim=-1) + quat_apply(
        body_rotation, moment_body
    )
    inverse_sensor = quat_conjugate(sensor_rotation)
    return torch.cat((quat_apply(inverse_sensor, forces.sum(dim=-2)),
                      quat_apply(inverse_sensor, moments.sum(dim=-2))), dim=-1)


def calibrate_wrench(
    ideal_wrench: torch.Tensor,
    ranges: torch.Tensor | None = None,
    resolution: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply SI-200-8 clipping and deterministic resolution before compensation.

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


def compensate_wrench(
    measured_raw: torch.Tensor, self_load: torch.Tensor, bias: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """Remove self-load and residual bias; invalid readings contribute zero.

    Compensation is not quantized again. Validity accepts [N] or [N, 1]; masking
    all rows directly avoids dynamic selection and GPU synchronization.
    """
    if valid.ndim == 1:
        valid = valid[:, None]
    return torch.where(valid, measured_raw - self_load - bias, 0.0)


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


def current_robot_sample(env) -> torch.Tensor:
    """Current robot, compensated F/T, tactile and action measurements."""
    return policy_observation(env)[:, 100:172]


def previous_robot_sample(env) -> torch.Tensor:
    """Previous completed sample, zeroed for rows beginning a new episode."""
    return policy_observation(env)[:, 172:244]
