"""Cartesian-space reward formulation for the Sweep JH task."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .common import (
    active_gripper_side_direction_b,
    desired_direction_b,
    object_displacement_b,
    virtual_ft_wrench_in_base_frame,
)


def _smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _position_w_to_b(robot: Articulation, position_w: torch.Tensor) -> torch.Tensor:
    position_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        position_w,
    )
    return position_b


def _eef_object_distance(
    env,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    return torch.linalg.norm(
        robot.data.body_pos_w[:, eef_cfg.body_ids[0]] - target.data.root_pos_w,
        dim=-1,
    )


def proximity_gate(
    env,
    near_distance: float,
    far_distance: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Smoothly increase from zero at ``far_distance`` to one nearby."""
    if not 0.0 < near_distance < far_distance:
        raise ValueError("Expected 0 < near_distance < far_distance.")
    distance = _eef_object_distance(env, eef_cfg, object_cfg)
    return _smoothstep01(
        (far_distance - distance) / (far_distance - near_distance)
    )


def wrist_force_components_b(
    env,
    command_name: str,
    ft_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return axial, tangential, and total force in the base XY plane.

    Excluding base-Z prevents the downstream tool's gravity load from being
    mistaken for planar object contact.
    """
    force_b = virtual_ft_wrench_in_base_frame(env, ft_cfg)[:, :3]
    direction_b = desired_direction_b(env, command_name)
    planar_force_b = force_b.clone()
    planar_force_b[:, 2] = 0.0
    signed_axial = torch.sum(planar_force_b * direction_b, dim=-1)
    # The incoming-joint-wrench sign depends on the parent/child convention.
    # Contact cannot pull the rigid object, so use the axial magnitude until
    # the hardware/simulator sign convention is explicitly calibrated.
    axial = torch.abs(signed_axial)
    tangential_vector = (
        planar_force_b - signed_axial.unsqueeze(-1) * direction_b
    )
    tangential = torch.linalg.norm(tangential_vector, dim=-1)
    total = torch.linalg.norm(planar_force_b, dim=-1)
    return axial, tangential, total


def contact_gate(
    env,
    command_name: str,
    near_distance: float,
    far_distance: float,
    force_low: float,
    force_high: float,
    eef_cfg: SceneEntityCfg,
    ft_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Estimate contact smoothly from Cartesian proximity and wrist force."""
    if not 0.0 <= force_low < force_high:
        raise ValueError("Expected 0 <= force_low < force_high.")
    proximity = proximity_gate(
        env, near_distance, far_distance, eef_cfg, object_cfg
    )
    _, _, total_force = wrist_force_components_b(env, command_name, ft_cfg)
    force = _smoothstep01((total_force - force_low) / (force_high - force_low))
    return proximity * force


def push_pose_error(
    env,
    command_name: str,
    stand_off: float,
    position_scale: float,
    near_distance: float,
    far_distance: float,
    force_low: float,
    force_high: float,
    eef_cfg: SceneEntityCfg,
    ft_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Normalized EEF error from the moving Cartesian push pose."""
    if stand_off <= 0.0 or position_scale <= 0.0:
        raise ValueError("Push-pose distances must be positive.")
    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    object_pos_b = _position_w_to_b(robot, target.data.root_pos_w)
    eef_pos_b = _position_w_to_b(
        robot, robot.data.body_pos_w[:, eef_cfg.body_ids[0]]
    )
    desired_push_pos_b = object_pos_b - stand_off * desired_direction_b(
        env, command_name
    )
    error = torch.linalg.norm(eef_pos_b - desired_push_pos_b, dim=-1)
    gate = contact_gate(
        env,
        command_name,
        near_distance,
        far_distance,
        force_low,
        force_high,
        eef_cfg,
        ft_cfg,
        object_cfg,
    )
    return (1.0 - gate) * torch.tanh(error / position_scale)


def push_axis_alignment_error(
    env,
    command_name: str,
    side_axis_local: tuple[float, float, float],
    near_distance: float,
    far_distance: float,
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize disagreement between gripper push axis and sweep direction."""
    side_b = active_gripper_side_direction_b(
        env, eef_cfg, object_cfg, side_axis_local
    )
    alignment = torch.sum(side_b * desired_direction_b(env, command_name), dim=-1)
    gate = proximity_gate(env, near_distance, far_distance, eef_cfg, object_cfg)
    return gate * 0.5 * (1.0 - torch.clamp(alignment, -1.0, 1.0))


def normal_force_tracking(
    env,
    command_name: str,
    near_distance: float,
    far_distance: float,
    force_low: float,
    force_high: float,
    eef_cfg: SceneEntityCfg,
    ft_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Track desired axial push force in robot-base Cartesian axes."""
    axial, _, _ = wrist_force_components_b(env, command_name, ft_cfg)
    command = env.command_manager.get_command(command_name)
    tolerance = torch.clamp(command[:, 4], min=1.0e-3)
    tracking = torch.exp(-torch.square((axial - command[:, 3]) / tolerance))
    gate = contact_gate(
        env,
        command_name,
        near_distance,
        far_distance,
        force_low,
        force_high,
        eef_cfg,
        ft_cfg,
        object_cfg,
    )
    return gate * tracking


def tangential_force_ratio(
    env,
    command_name: str,
    near_distance: float,
    far_distance: float,
    force_scale: float,
    eef_cfg: SceneEntityCfg,
    ft_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize wrist force perpendicular to the requested sweep direction."""
    if force_scale <= 0.0:
        raise ValueError("force_scale must be positive.")
    _, tangential, _ = wrist_force_components_b(env, command_name, ft_cfg)
    gate = proximity_gate(env, near_distance, far_distance, eef_cfg, object_cfg)
    return gate * torch.tanh(tangential / force_scale)


class CartesianDeltaProgress(ManagerTermBase):
    """Reward only newly achieved longitudinal object progress."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._previous_progress = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_progress[env_ids] = 0.0
        self._initialized[env_ids] = False

    def __call__(
        self,
        env,
        command_name: str,
        rate_scale: float = 1.0,
    ) -> torch.Tensor:
        if rate_scale <= 0.0:
            raise ValueError("rate_scale must be positive.")
        displacement_b = object_displacement_b(env, command_name)
        direction_b = desired_direction_b(env, command_name)
        progress = torch.sum(displacement_b * direction_b, dim=-1)
        distance = torch.clamp(
            env.command_manager.get_command(command_name)[:, 2], min=1.0e-6
        )
        delta = progress - self._previous_progress
        value = delta / (distance * env.step_dt)
        value = torch.where(self._initialized, value, torch.zeros_like(value))
        self._previous_progress[:] = progress
        self._initialized[:] = True
        return torch.tanh(value / rate_scale)


def normalized_endpoint_error(
    env,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Endpoint distance divided by commanded travel distance."""
    target: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    distance = torch.clamp(
        env.command_manager.get_command(command_name)[:, 2], min=1.0e-6
    )
    error = torch.linalg.norm(
        target.data.root_pos_w - command_term.goal_pos_w, dim=-1
    )
    return torch.tanh(error / distance)


def normalized_lateral_error(env, command_name: str) -> torch.Tensor:
    """Absolute object displacement perpendicular to the sweep direction."""
    displacement_b = object_displacement_b(env, command_name)
    direction = env.command_manager.get_command(command_name)[:, :2]
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    lateral = torch.abs(torch.sum(displacement_b[:, :2] * perpendicular, dim=-1))
    distance = torch.clamp(
        env.command_manager.get_command(command_name)[:, 2], min=1.0e-6
    )
    return torch.tanh(lateral / distance)


def normalized_overshoot(env, command_name: str) -> torch.Tensor:
    """Longitudinal travel beyond the commanded distance."""
    displacement_b = object_displacement_b(env, command_name)
    direction_b = desired_direction_b(env, command_name)
    command = env.command_manager.get_command(command_name)
    progress = torch.sum(displacement_b * direction_b, dim=-1)
    normalized = torch.relu(progress - command[:, 2]) / torch.clamp(
        command[:, 2], min=1.0e-6
    )
    return torch.tanh(normalized)


def near_goal_speed(
    env,
    command_name: str,
    goal_region_scale: float,
    speed_scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Penalize object speed only near the desired endpoint."""
    if goal_region_scale <= 0.0 or speed_scale <= 0.0:
        raise ValueError("Goal and speed scales must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_term(command_name)
    error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    gate = torch.exp(-torch.square(error / goal_region_scale))
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return gate * torch.tanh(speed / speed_scale)


def stopped_at_goal(
    env,
    command_name: str,
    position_tolerance: float,
    speed_tolerance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Continuous reward for precise placement with low object speed."""
    if position_tolerance <= 0.0 or speed_tolerance <= 0.0:
        raise ValueError("Stopping tolerances must be positive.")
    target: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_term(command_name)
    error = torch.linalg.norm(target.data.root_pos_w - command.goal_pos_w, dim=-1)
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return torch.exp(-torch.square(error / position_tolerance)) * torch.exp(
        -torch.square(speed / speed_tolerance)
    )


def pose_action_rate(env) -> torch.Tensor:
    """Unit-normalized squared rate of Cartesian relative-pose actions."""
    delta = env.action_manager.action[:, 6:12] - env.action_manager.prev_action[:, 6:12]
    delta = torch.clamp(delta, -2.0, 2.0)
    return 0.25 * torch.mean(torch.square(delta), dim=-1)


def stiffness_action_rate(env) -> torch.Tensor:
    """Unit-normalized squared rate of task-space stiffness actions."""
    delta = env.action_manager.action[:, :6] - env.action_manager.prev_action[:, :6]
    delta = torch.clamp(delta, -2.0, 2.0)
    return 0.25 * torch.mean(torch.square(delta), dim=-1)


def force_limit_excess(
    env,
    soft_limit: float,
    hard_limit: float,
    ft_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalized soft barrier before the hard wrist-force termination."""
    if not 0.0 <= soft_limit < hard_limit:
        raise ValueError("Expected 0 <= soft_limit < hard_limit.")
    total = torch.linalg.norm(
        virtual_ft_wrench_in_base_frame(env, ft_cfg)[:, :3], dim=-1
    )
    return torch.clamp(
        (total - soft_limit) / (hard_limit - soft_limit), 0.0, 1.0
    )


def ft_torque_excess(
    env,
    deadband: float,
    hard_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Normalize wrist torque between a deadband and hard limit."""
    if not 0.0 <= deadband < hard_limit:
        raise ValueError("Expected 0 <= deadband < hard_limit.")
    torque_b = virtual_ft_wrench_in_base_frame(env, asset_cfg)[:, 3:]
    torque = torch.linalg.norm(torque_b, dim=-1)
    return torch.clamp(
        (torque - deadband) / (hard_limit - deadband), 0.0, 1.0
    )


def torque_saturation(
    env,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Indicate invalid/clipped policy action or saturated OSC torque."""
    return env.action_manager.get_term(action_name).torque_saturated.float()
