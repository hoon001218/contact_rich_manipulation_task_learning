"""Termination conditions for safe sweep training."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .common import filtered_contact_mask, virtual_ft_wrench_in_base_frame
from .rewards import normalized_lateral_error


def target_reached(
    env,
    command_name: str,
    endpoint_threshold: float,
    lateral_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate successfully after reaching the requested endpoint."""
    command = env.command_manager.get_term(command_name)
    target: RigidObject = env.scene[object_cfg.name]
    endpoint_error = torch.linalg.norm(
        target.data.root_pos_w - command.goal_pos_w, dim=-1
    )
    return (endpoint_error < endpoint_threshold) & (
        normalized_lateral_error(env, command_name) < lateral_threshold
    )


def target_invalid_pose(
    env,
    minimum_height: float,
    maximum_tilt: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the cube falls off the table or tips excessively."""
    target: RigidObject = env.scene[object_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(target.data.root_quat_w)
    return (
        (target.data.root_pos_w[:, 2] < minimum_height)
        | (torch.abs(roll) > maximum_tilt)
        | (torch.abs(pitch) > maximum_tilt)
    )


def excessive_ft_wrench(
    env,
    force_limit: float,
    torque_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate unsafe wrist loads."""
    wrench = virtual_ft_wrench_in_base_frame(env, asset_cfg)
    return (torch.linalg.norm(wrench[:, :3], dim=-1) > force_limit) | (
        torch.linalg.norm(wrench[:, 3:], dim=-1) > torque_limit
    )


def arm_joint_speed_limit(
    env,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate numerically unsafe arm motion."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.any(
        torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]) > maximum_speed,
        dim=-1,
    )


def object_inside_gripper(
    env,
    center_half_extents: tuple[float, float, float],
    eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Terminate when the object center enters the open gripper volume.

    The target position is expressed in the EEF frame, so the exclusion box
    rotates freely with the gripper.  Consequently this condition constrains
    only object placement between the fingers and does not prescribe a world
    orientation for the gripper itself.
    """
    if len(center_half_extents) != 3 or any(
        half_extent <= 0.0 for half_extent in center_half_extents
    ):
        raise ValueError("center_half_extents must contain three positive values.")

    robot: Articulation = env.scene[eef_cfg.name]
    target: RigidObject = env.scene[object_cfg.name]
    eef_body_id = eef_cfg.body_ids[0]
    object_pos_eef, _ = math_utils.subtract_frame_transforms(
        robot.data.body_pos_w[:, eef_body_id],
        robot.data.body_quat_w[:, eef_body_id],
        target.data.root_pos_w,
    )
    half_extents = torch.tensor(
        center_half_extents,
        dtype=object_pos_eef.dtype,
        device=env.device,
    )
    return torch.all(torch.abs(object_pos_eef) <= half_extents, dim=-1)


class TargetStoppedAtGoal(ManagerTermBase):
    """Terminate after the object remains stopped at the goal for a dwell time."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dwell_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_elapsed[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        endpoint_threshold: float,
        lateral_threshold: float,
        speed_threshold: float,
        dwell_time: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> torch.Tensor:
        if dwell_time <= 0.0:
            raise ValueError("dwell_time must be positive.")

        command = env.command_manager.get_term(command_name)
        target: RigidObject = env.scene[object_cfg.name]
        endpoint_error = torch.linalg.norm(
            target.data.root_pos_w - command.goal_pos_w, dim=-1
        )
        speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
        stopped_at_goal = (
            (endpoint_error < endpoint_threshold)
            & (normalized_lateral_error(env, command_name) < lateral_threshold)
            & (speed < speed_threshold)
        )
        self._dwell_elapsed[:] = torch.where(
            stopped_at_goal,
            self._dwell_elapsed + env.step_dt,
            torch.zeros_like(self._dwell_elapsed),
        )
        return self._dwell_elapsed >= dwell_time


class HomeAfterSweepSuccess(ManagerTermBase):
    """Succeed after stable Home return without touching the parked object."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dwell_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._dwell_elapsed[env_ids] = 0.0

    def __call__(
        self,
        env,
        command_name: str,
        joint_position_threshold: float,
        joint_speed_threshold: float,
        endpoint_threshold: float,
        object_speed_threshold: float,
        object_displacement_threshold: float,
        dwell_time: float,
        contact_sensor_name: str,
        contact_force_threshold: float,
        asset_cfg: SceneEntityCfg,
        object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
    ) -> torch.Tensor:
        if dwell_time <= 0.0:
            raise ValueError("dwell_time must be positive.")
        command = env.command_manager.get_term(command_name)
        robot: Articulation = env.scene[asset_cfg.name]
        target: RigidObject = env.scene[object_cfg.name]
        joint_error = torch.abs(
            math_utils.wrap_to_pi(
                robot.data.joint_pos[:, asset_cfg.joint_ids]
                - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
            )
        )
        joint_speed = torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids])
        endpoint_error = torch.linalg.norm(
            target.data.root_pos_w - command.goal_pos_w, dim=-1
        )
        object_speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
        object_displacement = torch.linalg.norm(
            target.data.root_pos_w - command.parked_object_pos_w, dim=-1
        )
        contact_mask = filtered_contact_mask(
            env, contact_sensor_name, contact_force_threshold
        )
        stable_home = (
            (command.task_phase == 1)
            & torch.all(joint_error < joint_position_threshold, dim=-1)
            & torch.all(joint_speed < joint_speed_threshold, dim=-1)
            & (endpoint_error < endpoint_threshold)
            & (object_speed < object_speed_threshold)
            & (object_displacement < object_displacement_threshold)
            & (~contact_mask)
        )
        self._dwell_elapsed[:] = torch.where(
            stable_home,
            self._dwell_elapsed + env.step_dt,
            torch.zeros_like(self._dwell_elapsed),
        )
        return self._dwell_elapsed >= dwell_time


def object_disturbed_after_sweep(
    env,
    command_name: str,
    displacement_threshold: float,
    speed_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("target_object"),
) -> torch.Tensor:
    """Fail if the parked object is struck or displaced during Home return."""
    if displacement_threshold <= 0.0 or speed_threshold <= 0.0:
        raise ValueError("Post-sweep disturbance thresholds must be positive.")
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "parked_object_pos_w"):
        raise RuntimeError(f"Command '{command_name}' has no parked object pose.")
    target: RigidObject = env.scene[object_cfg.name]
    displacement = torch.linalg.norm(
        target.data.root_pos_w - command.parked_object_pos_w, dim=-1
    )
    speed = torch.linalg.norm(target.data.root_lin_vel_w, dim=-1)
    return (command.task_phase == 1) & (
        (displacement > displacement_threshold) | (speed > speed_threshold)
    )
