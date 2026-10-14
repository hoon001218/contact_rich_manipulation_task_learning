"""Research action mapping around Isaac Lab's OSC and joint-position action terms."""

from __future__ import annotations

import torch


def normalized_action(actions: torch.Tensor) -> torch.Tensor:
    """Finite, bounded policy commands; callers retain a separate invalid flag."""
    return torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def jacobian_at_link_origin(jacobian_com_w, com_to_link_w, world_to_base):
    """Shift PhysX COM Jacobian to the TCP link origin and rotate into base axes."""
    angular = jacobian_com_w[:, 3:, :]
    linear = jacobian_com_w[:, :3, :] + torch.linalg.cross(
        angular.mT, com_to_link_w[:, None, :].expand_as(angular.mT), dim=-1,
    ).mT
    return torch.cat((world_to_base @ linear, world_to_base @ angular), dim=1)


def bounded_hand_target(desired, measured, maximum_step):
    """Bound PD position lead relative to measurement, without target windup."""
    return measured + (desired - measured).clamp(min=-maximum_step, max=maximum_step)


from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg, OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


class ArmOscAction(OperationalSpaceControllerAction):
    """Native OSC with normalized delta-pose/gain input and TCP link-frame data.

    The parent owns command latching and the entire OSC torque calculation.
    Only input mapping, COM/link conversion and final effort bounds differ.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if (not self._asset.is_fixed_base or self._ee_body_idx == 0
                or self._joint_names != cfg.joint_names):
            raise ValueError("OSC requires a fixed base, a non-root TCP and the configured arm joint order.")
        if (cfg.controller_cfg.target_types != ["pose_rel"]
                or cfg.controller_cfg.impedance_mode != "variable_kp"
                or cfg.body_offset is not None or cfg.task_frame_rel_path is not None):
            raise ValueError("The 12D arm adapter requires pose_rel/variable_kp at the TCP link in base axes.")
        self._gain_min = torch.tensor(cfg.stiffness_min, device=self.device)
        self._gain_span = torch.tensor(cfg.stiffness_max, device=self.device) - self._gain_min
        self._workspace_min = torch.tensor(cfg.workspace_min, device=self.device)
        self._workspace_max = torch.tensor(cfg.workspace_max, device=self.device)
        if not bool(torch.isfinite(self._gain_min).all() & torch.isfinite(self._gain_span).all()
                    & (self._gain_min > 0).all() & (self._gain_span > 0).all()
                    & torch.isfinite(self._workspace_min).all() & torch.isfinite(self._workspace_max).all()
                    & (self._workspace_min < self._workspace_max).all()):
            raise ValueError("Require positive ordered finite gains and ordered finite workspace bounds.")
        data = self._asset.data
        if bool((data.joint_stiffness[:, self._joint_ids] != 0).any()
                | (data.joint_damping[:, self._joint_ids] != 0).any()):
            raise ValueError("Disable arm actuator PD gains for torque OSC.")
        self._effort_limits = data.joint_effort_limits[:, self._joint_ids].clone()
        self.invalid_control = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.workspace_clipped = torch.zeros_like(self.invalid_control)
        self.torque_saturated = torch.zeros_like(self.invalid_control)

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def tcp_body_id(self):
        return self._ee_body_idx

    def _preprocess_actions(self, actions):
        # Parent process_actions measures the TCP before calling this hook and
        # latches the relative pose through native OSC.set_command afterwards.
        super()._preprocess_actions(normalized_action(actions))
        self._raw_actions.copy_(actions)
        self._processed_actions[:, 6:] = self._gain_min + 0.5 * (
            normalized_action(actions[:, 6:]) + 1.0
        ) * self._gain_span
        position = self._ee_pose_b[:, :3] + self._processed_actions[:, :3]
        bounded = position.clamp(min=self._workspace_min, max=self._workspace_max)
        self._processed_actions[:, :3] = bounded - self._ee_pose_b[:, :3]
        self.workspace_clipped.copy_((bounded != position).any(-1))
        self.invalid_control.copy_(~torch.isfinite(actions).all(-1) | ~torch.isfinite(self._ee_pose_b).all(-1))
        self.torque_saturated.zero_()

    def _compute_ee_jacobian(self):
        data = self._asset.data
        offset = -math_utils.quat_apply(
            data.body_link_quat_w[:, self._ee_body_idx], data.body_com_pos_b[:, self._ee_body_idx],
        )
        rotation = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w))
        self._jacobian_b.copy_(jacobian_at_link_origin(self.jacobian_w, offset, rotation))

    def _compute_ee_velocity(self):
        # Native body_vel_w is COM velocity; OSC must use the same link origin
        # as its measured pose and the corrected Jacobian above.
        velocity = self._asset.data.body_link_vel_w[:, self._ee_body_idx]
        root_q = self._asset.data.root_quat_w
        self._ee_vel_b[:, :3] = math_utils.quat_apply_inverse(root_q, velocity[:, :3])
        self._ee_vel_b[:, 3:] = math_utils.quat_apply_inverse(root_q, velocity[:, 3:])

    def apply_actions(self):
        super().apply_actions()  # Native OperationalSpaceController.compute, including damping.
        self.invalid_control.logical_or_(~torch.isfinite(self._joint_efforts).all(-1))
        bounded = self._joint_efforts.clamp(min=-self._effort_limits, max=self._effort_limits)
        self.torque_saturated.logical_or_((bounded != self._joint_efforts).any(-1))
        self._joint_efforts.copy_(torch.where(self.invalid_control[:, None], 0.0, bounded))
        # Parent sets a target buffer; physics has not advanced. Replace it with
        # bounded efforts before scene.write_data_to_sim(), as in the UR10 example.
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        self._processed_actions[ids] = 0
        self._joint_efforts[ids] = 0
        self.invalid_control[ids] = False
        self.workspace_clipped[ids] = False
        self.torque_saturated[ids] = False
        self._asset.set_joint_effort_target(self._joint_efforts[ids], joint_ids=self._joint_ids, env_ids=ids)


class HandPositionAction(JointPositionToLimitsAction):
    """Native position action with soft-limit mapping and bounded position lead."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self._joint_names != cfg.joint_names or len(self._joint_ids) != 6:
            raise ValueError("Resolve exactly the six independent hand joints in the configured order.")
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._maximum_step = self._asset.data.joint_vel_limits[:, self._joint_ids].clamp(
            max=cfg.target_rate,
        ) * env.sim.get_physics_dt()
        if not bool(torch.isfinite(limits).all() & (limits[..., 1] > limits[..., 0]).all()
                    & torch.isfinite(self._maximum_step).all() & (self._maximum_step > 0).all()):
            raise ValueError("Hand soft limits and position-lead rate must be finite and positive.")
        self._desired_position = torch.zeros_like(self._processed_actions)
        self.invalid_control = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def joint_ids(self):
        return self._joint_ids

    def process_actions(self, actions):
        super().process_actions(normalized_action(actions))
        self._raw_actions.copy_(actions)
        self._desired_position.copy_(self._processed_actions)
        self.invalid_control.copy_(~torch.isfinite(actions).all(-1))

    def apply_actions(self):
        measured = self._asset.data.joint_pos[:, self._joint_ids]
        target = bounded_hand_target(self._desired_position, measured, self._maximum_step)
        self._processed_actions.copy_(torch.where(self.invalid_control[:, None], measured, target))
        super().apply_actions()  # Native limit-scaled position action -> native implicit PD actuator.

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        measured = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._desired_position[ids] = measured
        self._processed_actions[ids] = measured
        self.invalid_control[ids] = False
        self._asset.set_joint_position_target(measured, joint_ids=self._joint_ids, env_ids=ids)


@configclass
class ArmOscActionCfg(OperationalSpaceControllerActionCfg):
    class_type: type = ArmOscAction
    asset_name: str = "robot"
    joint_names: list[str] = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    body_name: str = "right_plam_force_sensor"
    position_scale: float = 0.015
    orientation_scale: float = 0.08
    stiffness_min: tuple[float, ...] = (100.0, 100.0, 100.0, 5.0, 5.0, 5.0)
    stiffness_max: tuple[float, ...] = (800.0, 800.0, 800.0, 40.0, 40.0, 40.0)
    workspace_min: tuple[float, float, float] = (0.15, -0.55, 0.10)
    workspace_max: tuple[float, float, float] = (1.0, 0.55, 0.80)
    controller_cfg: OperationalSpaceControllerCfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"], impedance_mode="variable_kp",
        inertial_dynamics_decoupling=True, partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True, motion_damping_ratio_task=1.0,
        motion_stiffness_limits_task=(5.0, 800.0), nullspace_control="none",
    )


@configclass
class HandPositionActionCfg(JointPositionToLimitsActionCfg):
    class_type: type = HandPositionAction
    asset_name: str = "robot"
    joint_names: list[str] = [
        "right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
        "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint",
    ]
    preserve_order: bool = True
    rescale_to_limits: bool = True
    target_rate: float = 1.0
