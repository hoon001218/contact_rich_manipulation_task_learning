"""Custom 12-D variable-stiffness operational-space action."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import (
    OperationalSpaceControllerActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import (
    OperationalSpaceControllerAction,
)
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SweepOperationalSpaceAction(OperationalSpaceControllerAction):
    """Interpret policy actions as ``[stiffness(6), relative_pose_rpy(6)]``.

    Isaac Lab's OSC expects ``[relative_pose_axis_angle(6), stiffness(6)]``.
    This term preserves the user-facing order, maps normalized stiffness from
    ``[-1, 1]`` into the configured physical limits, converts RPY increments to
    axis-angle, and clamps the resulting UR5e joint torques.
    """

    cfg: "SweepOperationalSpaceActionCfg"

    def __init__(self, cfg: "SweepOperationalSpaceActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if self.action_dim != 12:
            raise ValueError(
                f"Sweep OSC must be 12-D, received action_dim={self.action_dim}."
            )
        self._torque_saturated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._stiffness_command = torch.zeros(self.num_envs, 6, device=self.device)
        self._pose_command = torch.zeros(self.num_envs, 6, device=self.device)

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    @property
    def torque_saturated(self) -> torch.Tensor:
        return self._torque_saturated

    @property
    def stiffness_command(self) -> torch.Tensor:
        return self._stiffness_command

    @property
    def pose_command(self) -> torch.Tensor:
        return self._pose_command

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._joint_efforts[env_ids] = 0.0
        self._torque_saturated[env_ids] = False
        self._stiffness_command[env_ids] = 0.0
        self._pose_command[env_ids] = 0.0

    def _preprocess_actions(self, actions: torch.Tensor):
        if actions.shape != (self.num_envs, 12):
            raise ValueError(
                f"Expected sweep actions with shape {(self.num_envs, 12)}, "
                f"received {tuple(actions.shape)}."
            )

        finite = torch.isfinite(actions)
        sanitized = torch.where(finite, actions, 0.0)
        normalized = torch.clamp(sanitized, -1.0, 1.0)
        self._raw_actions[:] = sanitized

        stiffness_normalized = normalized[:, :6]
        pose_normalized = normalized[:, 6:12]
        stiffness_min, stiffness_max = (
            self.cfg.controller_cfg.motion_stiffness_limits_task
        )
        stiffness = (
            0.5 * (stiffness_normalized + 1.0) * (stiffness_max - stiffness_min)
            + stiffness_min
        )

        pose_command = pose_normalized.clone()
        pose_command[:, :3] *= self._position_scale
        pose_command[:, 3:6] *= self._orientation_scale
        delta_quat = math_utils.quat_from_euler_xyz(
            pose_command[:, 3],
            pose_command[:, 4],
            pose_command[:, 5],
        )
        pose_command[:, 3:6] = math_utils.axis_angle_from_quat(delta_quat)

        # Controller-facing order: relative pose, then stiffness.
        self._processed_actions[:, :6] = pose_command
        self._processed_actions[:, 6:12] = stiffness
        self._stiffness_command[:] = stiffness
        self._pose_command[:] = pose_command

        invalid_or_clipped = (~finite.all(dim=1)) | torch.any(
            torch.abs(sanitized - normalized) > self.cfg.saturation_tolerance,
            dim=1,
        )
        self._torque_saturated[:] = invalid_or_clipped

    def apply_actions(self):
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()

        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )

        finite_efforts = torch.isfinite(self._joint_efforts).all(dim=1)
        safe_efforts = torch.where(
            finite_efforts.unsqueeze(-1), self._joint_efforts, 0.0
        )
        effort_limits = (
            self._asset.data.joint_effort_limits[:, self._joint_ids]
            * self.cfg.effort_limit_scale
        )
        effort_limits = torch.clamp(effort_limits, min=self.cfg.minimum_effort_limit)
        clamped_efforts = torch.clamp(
            safe_efforts, min=-effort_limits, max=effort_limits
        )
        effort_clipped = torch.any(
            torch.abs(clamped_efforts - safe_efforts) > self.cfg.saturation_tolerance,
            dim=1,
        )
        self._torque_saturated |= (~finite_efforts) | effort_clipped
        self._joint_efforts[:] = clamped_efforts
        self._asset.set_joint_effort_target(
            self._joint_efforts, joint_ids=self._joint_ids
        )


@configclass
class SweepOperationalSpaceActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for :class:`SweepOperationalSpaceAction`."""

    class_type: type = SweepOperationalSpaceAction
    effort_limit_scale: float = 0.9
    minimum_effort_limit: float = 1.0e-6
    saturation_tolerance: float = 1.0e-6


class OpenGripperSweepOperationalSpaceAction(SweepOperationalSpaceAction):
    """OSC arm action that continuously holds the gripper fully open."""

    cfg: "OpenGripperSweepOperationalSpaceActionCfg"

    def __init__(
        self,
        cfg: "OpenGripperSweepOperationalSpaceActionCfg",
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)
        self._gripper_joint_ids, self._gripper_joint_names = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=True
        )
        if len(self._gripper_joint_ids) == 0:
            raise ValueError("No gripper joints matched gripper_joint_names.")
        self._gripper_open_targets = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            cfg.gripper_open_position,
            device=self.device,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._asset.set_joint_position_target(
            self._gripper_open_targets[env_ids],
            joint_ids=self._gripper_joint_ids,
            env_ids=env_ids,
        )

    def apply_actions(self):
        super().apply_actions()
        # Re-assert the target at every physics step so no reset, contact
        # impulse, or stale articulation buffer can close the fingers.
        self._asset.set_joint_position_target(
            self._gripper_open_targets,
            joint_ids=self._gripper_joint_ids,
        )


@configclass
class OpenGripperSweepOperationalSpaceActionCfg(SweepOperationalSpaceActionCfg):
    """Configuration for OSC control with a fixed open gripper target."""

    class_type: type = OpenGripperSweepOperationalSpaceAction
    gripper_joint_names: list[str] = [".*(finger|knuckle).*"]
    gripper_open_position: float = 0.0

