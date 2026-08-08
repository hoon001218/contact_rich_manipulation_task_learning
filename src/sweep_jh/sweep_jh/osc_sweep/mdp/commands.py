"""Episode-level desired sweep command."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SweepMotionCommand(CommandTerm):
    """Sample direction, distance, desired force, and force tolerance at reset.

    The public command is five-dimensional:

    ``[direction_x, direction_y, distance_m, force_N, force_tolerance_N]``.
    """

    cfg: "SweepMotionCommandCfg"

    def __init__(self, cfg: "SweepMotionCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.robot_name]
        self._object: RigidObject = env.scene[cfg.object_name]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self.initial_pose_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["endpoint_error_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["lateral_error_m"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["normalized_lateral_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["progress_ratio"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["object_speed_mps"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        if count == 0:
            return

        angle = torch.empty(count, device=self.device).uniform_(
            *self.cfg.direction_angle_range
        )
        direction_b = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.distance_range
        )
        force = torch.empty(count, device=self.device).uniform_(*self.cfg.force_range)
        force_tolerance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.force_tolerance_range
        )
        self._command[env_ids] = torch.cat(
            (
                direction_b,
                distance.unsqueeze(-1),
                force.unsqueeze(-1),
                force_tolerance.unsqueeze(-1),
            ),
            dim=-1,
        )

        object_pos_w = self._object.data.root_pos_w[env_ids]
        object_quat_w = self._object.data.root_quat_w[env_ids]
        root_pos_w = self._robot.data.root_pos_w[env_ids]
        root_quat_w = self._robot.data.root_quat_w[env_ids]
        object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            object_pos_w,
            object_quat_w,
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(object_quat_b)
        self.initial_pose_b[env_ids] = torch.cat(
            (
                object_pos_b,
                torch.stack((roll, pitch, yaw), dim=-1),
            ),
            dim=-1,
        )

        direction_b_3d = torch.cat(
            (
                direction_b,
                torch.zeros(count, 1, device=self.device),
            ),
            dim=-1,
        )
        direction_w = math_utils.quat_apply(root_quat_w, direction_b_3d)
        self.direction_w[env_ids] = direction_w
        self.goal_pos_b[env_ids] = object_pos_b + direction_b_3d * distance.unsqueeze(
            -1
        )
        self.goal_pos_w[env_ids] = object_pos_w + direction_w * distance.unsqueeze(-1)

    def _update_metrics(self):
        self.metrics["endpoint_error_m"][:] = torch.linalg.norm(
            self._object.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        current_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._object.data.root_pos_w,
        )
        displacement_b = current_pos_b - self.initial_pose_b[:, :3]
        direction_b = self._command[:, :2]
        perpendicular_b = torch.stack(
            (-direction_b[:, 1], direction_b[:, 0]), dim=-1
        )
        lateral_error = torch.abs(
            torch.sum(displacement_b[:, :2] * perpendicular_b, dim=-1)
        )
        progress = torch.sum(displacement_b[:, :2] * direction_b, dim=-1)
        distance = torch.clamp(self._command[:, 2], min=1.0e-6)
        self.metrics["lateral_error_m"][:] = lateral_error
        self.metrics["normalized_lateral_error"][:] = lateral_error / distance
        self.metrics["progress_ratio"][:] = progress / distance
        self.metrics["object_speed_mps"][:] = torch.linalg.norm(
            self._object.data.root_lin_vel_w, dim=-1
        )

    def _update_command(self):
        pass


@configclass
class SweepMotionCommandCfg(CommandTermCfg):
    """Configuration for episode-level sweep commands."""

    class_type: type = SweepMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    direction_angle_range: tuple[float, float] = (-math.pi, math.pi)
    distance_range: tuple[float, float] = (0.10, 0.22)
    force_range: tuple[float, float] = (8.0, 25.0)
    force_tolerance_range: tuple[float, float] = (3.0, 6.0)


class ConstantVelocitySweepCommand(CommandTerm):
    """Sample a force-free planar sweep command once per episode.

    The command is ``[direction_x, direction_y, distance_m, speed_mps]``.
    Unlike :class:`SweepMotionCommand`, it contains no desired contact force or
    force tolerance.
    """

    cfg: "ConstantVelocitySweepCommandCfg"

    def __init__(self, cfg: "ConstantVelocitySweepCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene[cfg.robot_name]
        self._object: RigidObject = env.scene[cfg.object_name]
        self._command = torch.zeros(self.num_envs, 4, device=self.device)
        self.initial_pose_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.goal_pos_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.direction_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["endpoint_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["forward_speed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["progress_ratio"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        if count == 0:
            return

        angle = torch.empty(count, device=self.device).uniform_(
            *self.cfg.direction_angle_range
        )
        direction_b = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        distance = torch.empty(count, device=self.device).uniform_(
            *self.cfg.distance_range
        )
        target_speed = torch.empty(count, device=self.device).uniform_(
            *self.cfg.target_speed_range
        )
        self._command[env_ids] = torch.cat(
            (
                direction_b,
                distance.unsqueeze(-1),
                target_speed.unsqueeze(-1),
            ),
            dim=-1,
        )

        object_pos_w = self._object.data.root_pos_w[env_ids]
        object_quat_w = self._object.data.root_quat_w[env_ids]
        root_pos_w = self._robot.data.root_pos_w[env_ids]
        root_quat_w = self._robot.data.root_quat_w[env_ids]
        object_pos_b, object_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            object_pos_w,
            object_quat_w,
        )
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(object_quat_b)
        self.initial_pose_b[env_ids] = torch.cat(
            (object_pos_b, torch.stack((roll, pitch, yaw), dim=-1)),
            dim=-1,
        )

        direction_b_3d = torch.cat(
            (direction_b, torch.zeros(count, 1, device=self.device)), dim=-1
        )
        direction_w = math_utils.quat_apply(root_quat_w, direction_b_3d)
        self.direction_w[env_ids] = direction_w
        self.goal_pos_b[env_ids] = object_pos_b + direction_b_3d * distance.unsqueeze(
            -1
        )
        self.goal_pos_w[env_ids] = object_pos_w + direction_w * distance.unsqueeze(-1)

    def _update_metrics(self):
        self.metrics["endpoint_error"][:] = torch.linalg.norm(
            self._object.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        forward_speed = torch.sum(
            self._object.data.root_lin_vel_w * self.direction_w, dim=-1
        )
        current_pos_b, _ = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._object.data.root_pos_w,
        )
        displacement_b = current_pos_b - self.initial_pose_b[:, :3]
        progress = torch.sum(
            displacement_b[:, :2] * self._command[:, :2], dim=-1
        )
        self.metrics["speed_error"][:] = torch.abs(forward_speed - self._command[:, 3])
        self.metrics["forward_speed"][:] = forward_speed
        self.metrics["progress_ratio"][:] = progress / torch.clamp(
            self._command[:, 2], min=1.0e-6
        )

    def _update_command(self):
        pass


@configclass
class ConstantVelocitySweepCommandCfg(CommandTermCfg):
    """Configuration for the force-free constant-velocity sweep command."""

    class_type: type = ConstantVelocitySweepCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    robot_name: str = MISSING
    object_name: str = MISSING
    direction_angle_range: tuple[float, float] = (-math.pi, math.pi)
    distance_range: tuple[float, float] = (0.10, 0.22)
    target_speed_range: tuple[float, float] = (0.08, 0.08)


class SweepHomeConstantVelocityCommand(ConstantVelocitySweepCommand):
    """Latch from object sweeping to collision-free Home return.

    The public motion command stays four-dimensional.  ``task_phase`` is a
    separate observable state: 0 means sweep and 1 means return Home.
    """

    cfg: "SweepHomeConstantVelocityCommandCfg"

    def __init__(self, cfg: "SweepHomeConstantVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.task_phase = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._goal_dwell_elapsed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        # Snapshot the object pose at the SWEEP -> HOME transition.  The goal
        # pose alone is not enough to detect a second push because the object
        # could move inside the endpoint tolerance and still count as parked.
        self.parked_object_pos_w = torch.zeros(
            self.num_envs, 3, dtype=torch.float32, device=self.device
        )
        self.metrics["home_phase"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.metrics["parked_displacement"] = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.task_phase[env_ids] = 0
        self._goal_dwell_elapsed[env_ids] = 0.0
        self.parked_object_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]

    def _update_metrics(self):
        super()._update_metrics()
        self.metrics["home_phase"][:] = self.task_phase.float()
        displacement = torch.linalg.norm(
            self._object.data.root_pos_w - self.parked_object_pos_w, dim=-1
        )
        self.metrics["parked_displacement"][:] = torch.where(
            self.task_phase == 1,
            displacement,
            torch.zeros_like(displacement),
        )

    def _update_command(self):
        endpoint_error = torch.linalg.norm(
            self._object.data.root_pos_w - self.goal_pos_w, dim=-1
        )
        object_speed = torch.linalg.norm(
            self._object.data.root_lin_vel_w, dim=-1
        )
        stopped_at_goal = (
            (endpoint_error < self.cfg.endpoint_threshold)
            & (object_speed < self.cfg.speed_threshold)
            & (self.task_phase == 0)
        )
        self._goal_dwell_elapsed[:] = torch.where(
            stopped_at_goal,
            self._goal_dwell_elapsed + self._env.step_dt,
            torch.where(
                self.task_phase == 0,
                torch.zeros_like(self._goal_dwell_elapsed),
                self._goal_dwell_elapsed,
            ),
        )
        entering_home = (
            (self.task_phase == 0)
            & (self._goal_dwell_elapsed >= self.cfg.goal_dwell_time)
        )
        self.parked_object_pos_w[entering_home] = (
            self._object.data.root_pos_w[entering_home]
        )
        self.task_phase[:] = torch.where(
            entering_home,
            torch.ones_like(self.task_phase),
            self.task_phase,
        )


@configclass
class SweepHomeConstantVelocityCommandCfg(ConstantVelocitySweepCommandCfg):
    """Configuration for the sweep-then-Home task phase latch."""

    class_type: type = SweepHomeConstantVelocityCommand
    endpoint_threshold: float = 0.020
    speed_threshold: float = 0.020
    goal_dwell_time: float = 0.30
