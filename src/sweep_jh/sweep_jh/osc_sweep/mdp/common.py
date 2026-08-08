"""Shared tensor helpers for sweep observations, rewards, and terminations."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import ContactSensor


def virtual_ft_wrench_in_base_frame(env, asset_cfg) -> torch.Tensor:
    """Return wrist wrench expressed in robot-base axes.

    The force and torque reference point remains the virtual F/T sensor
    origin; only their coordinate axes are rotated from sensor to robot base.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    wrench_sensor = -robot.data.body_incoming_joint_wrench_b[:, body_id, :]
    _, sensor_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, body_id],
        robot.data.body_quat_w[:, body_id],
    )
    force_b = math_utils.quat_apply(sensor_quat_b, wrench_sensor[:, :3])
    torque_b = math_utils.quat_apply(sensor_quat_b, wrench_sensor[:, 3:])
    return torch.cat((force_b, torque_b), dim=-1)


def pose_w_to_root_rpy(
    robot: Articulation,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
) -> torch.Tensor:
    """Convert a world pose to robot-root ``xyz + roll/pitch/yaw``."""
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        pos_w,
        quat_w,
    )
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_b)
    return torch.cat(
        (pos_b, torch.stack((roll, pitch, yaw), dim=-1)), dim=-1
    )


def target_contact_data_w(
    env,
    sensor_names: tuple[str, ...] = (
        "left_contact",
        "right_contact",
    ),
    force_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return force-weighted contact point, summed normal force, and contact mask.

    Contact sensors are filtered against the target cube. A zero point is
    returned when neither gripper pad is in contact.
    """
    point_numerator = torch.zeros(env.num_envs, 3, device=env.device)
    force_sum_w = torch.zeros_like(point_numerator)
    weight_sum = torch.zeros(env.num_envs, 1, device=env.device)

    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        force_matrix_w = sensor.data.force_matrix_w
        contact_pos_w = sensor.data.contact_pos_w
        if force_matrix_w is None or contact_pos_w is None:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' must track filtered forces "
                "and contact points."
            )

        force_sum_w += force_matrix_w.sum(dim=(1, 2))
        weights = torch.linalg.norm(force_matrix_w, dim=-1)
        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        weights = torch.where(valid, weights, 0.0)
        safe_points = torch.nan_to_num(contact_pos_w, nan=0.0)
        point_numerator += (safe_points * weights.unsqueeze(-1)).sum(
            dim=(1, 2)
        )
        weight_sum += weights.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)

    contact_mask = weight_sum.squeeze(-1) > force_threshold
    contact_point_w = point_numerator / torch.clamp(weight_sum, min=1.0e-6)
    contact_point_w = torch.where(
        contact_mask.unsqueeze(-1),
        contact_point_w,
        torch.zeros_like(contact_point_w),
    )
    return contact_point_w, force_sum_w, contact_mask


def filtered_contact_mask(
    env,
    sensor_name: str,
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Return whether a filtered contact sensor detects meaningful contact."""
    if force_threshold < 0.0:
        raise ValueError("force_threshold must be non-negative.")
    sensor: ContactSensor = env.scene[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(
            f"Contact sensor '{sensor_name}' must use filter_prim_paths_expr."
        )
    force_magnitudes = torch.linalg.norm(force_matrix_w, dim=-1)
    reduce_dims = tuple(range(1, force_magnitudes.ndim))
    total_filtered_force = force_magnitudes.sum(dim=reduce_dims)
    return total_filtered_force > force_threshold


def object_displacement_b(env, command_name: str) -> torch.Tensor:
    """Current target-object displacement from the command's initial pose."""
    command = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[command.cfg.robot_name]
    target: RigidObject = env.scene[command.cfg.object_name]
    current_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
    )
    return current_pos_b - command.initial_pose_b[:, :3]


def desired_direction_b(env, command_name: str) -> torch.Tensor:
    """Desired unit sweep direction in the robot base XY plane."""
    command = env.command_manager.get_command(command_name)
    return torch.cat(
        (
            command[:, :2],
            torch.zeros(env.num_envs, 1, device=env.device),
        ),
        dim=-1,
    )


def active_gripper_side_direction_b(
    env,
    eef_cfg,
    object_cfg,
    side_axis_local: tuple[float, float, float],
) -> torch.Tensor:
    """Return the broad-side normal that points from the gripper to the object.

    A pad plane has two possible normals. The sign is selected per environment
    using the current vector from ``SweepToolCenter`` to the target object.
    """
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    eef_pos_b, eef_quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, eef_body_id],
        robot.data.body_quat_w[:, eef_body_id],
    )
    object_pos_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        target.data.root_pos_w,
    )
    local_axis = torch.tensor(
        side_axis_local, dtype=eef_pos_b.dtype, device=env.device
    ).repeat(env.num_envs, 1)
    unsigned_side_b = math_utils.quat_apply(eef_quat_b, local_axis)
    to_object_b = object_pos_b - eef_pos_b
    sign = torch.where(
        torch.sum(unsigned_side_b * to_object_b, dim=-1, keepdim=True) >= 0.0,
        1.0,
        -1.0,
    )
    return unsigned_side_b * sign


def side_pad_contact_quality(
    env,
    sensor_names: tuple[str, ...],
    pad_size: tuple[float, float, float],
    face_normal_axis: int,
    center_sigma: float,
    face_sigma: float,
    force_threshold: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure whether target contact lies near the center of a broad pad face.

    The contact point is transformed into each pad's local frame. The score is
    high only when:

    1. the point lies near ``+/- half_extent`` on ``face_normal_axis``; and
    2. the two in-plane coordinates are near the pad center.

    Scores from both pads are averaged using target-contact normal force.
    """
    if face_normal_axis not in (0, 1, 2):
        raise ValueError("face_normal_axis must be 0, 1, or 2.")
    if center_sigma <= 0.0 or face_sigma <= 0.0:
        raise ValueError("Contact-quality sigmas must be positive.")

    pad_half_size = 0.5 * torch.tensor(
        pad_size, dtype=torch.float32, device=env.device
    )
    quality_numerator = torch.zeros(env.num_envs, device=env.device)
    force_weight_sum = torch.zeros(env.num_envs, device=env.device)

    for sensor_name in sensor_names:
        sensor: ContactSensor = env.scene[sensor_name]
        force_matrix_w = sensor.data.force_matrix_w
        contact_pos_w = sensor.data.contact_pos_w
        pad_pos_w = sensor.data.pos_w
        pad_quat_w = sensor.data.quat_w
        if (
            force_matrix_w is None
            or contact_pos_w is None
            or pad_pos_w is None
            or pad_quat_w is None
        ):
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' must track pose, filtered "
                "forces, and contact points."
            )

        valid = torch.isfinite(contact_pos_w).all(dim=-1)
        force_weights = torch.linalg.norm(force_matrix_w, dim=-1)
        force_weights = torch.where(valid, force_weights, 0.0)
        safe_points_w = torch.nan_to_num(contact_pos_w, nan=0.0)

        pad_pos_w = pad_pos_w.unsqueeze(2).expand_as(safe_points_w)
        pad_quat_w = pad_quat_w.unsqueeze(2).expand(
            *safe_points_w.shape[:-1], 4
        )
        point_delta_w = safe_points_w - pad_pos_w
        point_delta_local = math_utils.quat_apply_inverse(
            pad_quat_w.reshape(-1, 4),
            point_delta_w.reshape(-1, 3),
        ).view_as(point_delta_w)

        normalized_point = point_delta_local / torch.clamp(
            pad_half_size, min=1.0e-6
        )
        normal_ratio = torch.abs(normalized_point[..., face_normal_axis])
        face_score = torch.exp(
            -0.5 * torch.square((normal_ratio - 1.0) / face_sigma)
        )

        tangent_mask = torch.ones(3, device=env.device)
        tangent_mask[face_normal_axis] = 0.0
        tangent_radius_sq = torch.sum(
            torch.square(normalized_point * tangent_mask), dim=-1
        )
        center_score = torch.exp(
            -0.5 * tangent_radius_sq / (center_sigma**2)
        )
        contact_quality = face_score * center_score

        quality_numerator += torch.sum(
            contact_quality * force_weights, dim=(1, 2)
        )
        force_weight_sum += torch.sum(force_weights, dim=(1, 2))

    contact_mask = force_weight_sum > force_threshold
    quality = quality_numerator / torch.clamp(
        force_weight_sum, min=1.0e-6
    )
    quality = torch.where(contact_mask, quality, torch.zeros_like(quality))
    return torch.clamp(quality, 0.0, 1.0), contact_mask
