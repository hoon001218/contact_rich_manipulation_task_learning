"""18D manipulation action: base-frame TCP motion, OSC gains, and hand position.

The simulator classes are loaded on demand after AppLauncher. Tensor helpers are
also usable on CPU without importing Kit, for checking the command contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


from ..src.catalog import FORBIDDEN_CONTACT_BODY_NAMES, FORBIDDEN_CONTACT_SENSOR_NAMES


def contact_force_exceeds_threshold(force_history: torch.Tensor, threshold: float) -> torch.Tensor:
    """Any partner's normal resultant exceeds the threshold in the recent substeps.

    Expected shape is (N, history, one sensor body, partner filters, XYZ). Do
    not sum partner vectors: contacts from opposite sides must not cancel.
    """
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Forbidden-contact force threshold must be finite and positive.")
    if force_history.ndim != 5 or force_history.shape[-1] != 3 or force_history.shape[2] != 1:
        raise ValueError("Forbidden-contact force history must have shape (N, H, 1, F, 3).")
    if not torch.isfinite(force_history).all():
        raise RuntimeError("Forbidden-contact sensor returned nonfinite forces.")
    return (torch.linalg.vector_norm(force_history, dim=-1) > threshold).any(dim=(1, 2, 3))


def forbidden_contacts(env, threshold: float = 1.0) -> torch.Tensor:
    """Actual arm/spacer contact with shelf, scene objects, or ground.

    Hand contacts, including incidental hand-Target contact, are not prohibited
    by this detector. Target movement is protected separately by the MDP. The
    filter histories retain every physics substep of the latest policy step.
    """
    detected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in FORBIDDEN_CONTACT_SENSOR_NAMES:
        sensor = env.scene[name]
        history = sensor.data.force_matrix_w_history
        if history is None or history.shape[-2] == 0:
            raise RuntimeError(f"Forbidden-contact sensor {name} has no scene contact filters.")
        detected.logical_or_(contact_force_exceeds_threshold(history, threshold))
    return detected


def map_action(
    actions: torch.Tensor,
    position_scale: torch.Tensor,
    orientation_scale: torch.Tensor,
    stiffness_min: torch.Tensor,
    stiffness_max: torch.Tensor,
    hand_limits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized motion6 / stiffness6 / hand6 into physical commands.

    Translation is metres and rotation is a base-axis rotation vector in radians.
    OSC gains follow Isaac Lab's selected inertia-compensation convention; by
    default they are physical Cartesian stiffness in N/m and N m/rad.
    Hand limits have shape (N, 6, 2), in the independent-joint name order.
    """
    if actions.ndim != 2 or actions.shape[1] != 18:
        raise ValueError(f"Expected actions with shape (N, 18), got {tuple(actions.shape)}.")
    if not torch.isfinite(actions).all():
        raise ValueError("Manipulation actions must all be finite.")
    normalized = actions.clamp(-1.0, 1.0)
    delta = normalized[:, :6] * torch.cat((position_scale, orientation_scale))
    gains = stiffness_min + (normalized[:, 6:12] + 1.0) * 0.5 * (stiffness_max - stiffness_min)
    hand = hand_limits[..., 0] + (normalized[:, 12:18] + 1.0) * 0.5 * (
        hand_limits[..., 1] - hand_limits[..., 0]
    )
    return normalized, delta, gains, hand


def pose_target_from_delta(
    measured_pose_b: torch.Tensor,
    delta_b: torch.Tensor,
    workspace_min: torch.Tensor,
    workspace_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch a measured-pose-relative target; rotate about fixed base axes.

    Quaternions are WXYZ. The rotation increment left-multiplies the measured
    orientation, matching Isaac Lab apply_delta_pose, rather than rotating about
    the moving TCP's local axes. No previous command is used here.
    """
    position = measured_pose_b[:, :3] + delta_b[:, :3]
    clamped_position = position.clamp(min=workspace_min, max=workspace_max)
    workspace_clipped = (clamped_position != position).any(dim=-1)
    rotation_vector = delta_b[:, 3:]
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    # sin(theta/2)/theta has limit 1/2; sinc avoids a zero-angle division.
    delta_quat = torch.cat((torch.cos(angle * 0.5),
                            0.5 * torch.sinc(angle / (2.0 * math.pi)) * rotation_vector), dim=-1)
    measured_quat = measured_pose_b[:, 3:7]
    scalar = delta_quat[:, :1] * measured_quat[:, :1] - (
        delta_quat[:, 1:] * measured_quat[:, 1:]
    ).sum(dim=-1, keepdim=True)
    vector = (delta_quat[:, :1] * measured_quat[:, 1:]
              + measured_quat[:, :1] * delta_quat[:, 1:]
              + torch.linalg.cross(delta_quat[:, 1:], measured_quat[:, 1:], dim=-1))
    orientation = torch.cat((scalar, vector), dim=-1)
    orientation = orientation / torch.linalg.vector_norm(orientation, dim=-1, keepdim=True)
    orientation = torch.where(orientation[:, :1] < 0, -orientation, orientation)
    return torch.cat((clamped_position, orientation), dim=-1), workspace_clipped


def jacobian_at_link_origin(
    jacobian_com_w: torch.Tensor, com_to_link_w: torch.Tensor, world_to_base: torch.Tensor
) -> torch.Tensor:
    """Shift the PhysX COM Jacobian to the actor/TCP origin and rotate its axes.

    v_link = v_COM + omega x r_COM_to_link. PhysX Jacobians map joint
    velocities to COM spatial velocities in world axes. The returned Jacobian
    uses the same link origin as the TCP pose and link-velocity observations.
    """
    angular_columns = jacobian_com_w[:, 3:, :].transpose(1, 2)
    lever = com_to_link_w[:, None, :].expand_as(angular_columns)
    linear_w = jacobian_com_w[:, :3, :] + torch.linalg.cross(
        angular_columns, lever, dim=-1
    ).transpose(1, 2)
    return torch.cat((world_to_base @ linear_w, world_to_base @ jacobian_com_w[:, 3:, :]), dim=1)


def clamp_efforts(efforts: torch.Tensor, limits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clamp to positive native effort limits; invalid controller output fails loudly."""
    if not torch.isfinite(efforts).all():
        raise RuntimeError("Operational-space controller produced nonfinite joint efforts.")
    if not torch.isfinite(limits).all() or not (limits > 0).all():
        raise RuntimeError("Arm joint effort limits must be finite and positive.")
    applied = efforts.clamp(min=-limits, max=limits)
    return applied, (applied != efforts).any(dim=-1)


def bounded_hand_target(
    desired: torch.Tensor, measured: torch.Tensor, rate_limit: torch.Tensor, dt: float,
) -> torch.Tensor:
    """Move an absolute hand target toward its goal by at most one substep.

    The position lead is bounded relative to the actual joint position. A
    policy's midpoint hand command must not apply a full-range position error
    instantaneously to the hand PD drives, especially just after reset.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Hand command time step must be finite and positive.")
    if (not torch.isfinite(desired).all() or not torch.isfinite(measured).all()
            or not torch.isfinite(rate_limit).all() or not (rate_limit > 0).all()):
        raise RuntimeError("Hand targets, measured positions, and positive rate limits must be finite.")
    step = rate_limit * dt
    return measured + (desired - measured).clamp(min=-step, max=step)


def physical_damping_matrix(
    task_inertia: torch.Tensor, stiffness: torch.Tensor, damping_ratio: float = 1.0,
) -> torch.Tensor:
    """Full coupled mass-aware Cartesian damping, in the same TCP/base frame.

    D = 2*zeta*M^.5*(M^-.5*K*M^-.5)^.5*M^.5. In one dimension this
    reduces to 2*zeta*sqrt(m*k). All matrix roots use symmetric eigensolvers;
    this respects light rotational modes instead of assuming unit task mass.
    """
    if not math.isfinite(damping_ratio) or damping_ratio <= 0:
        raise ValueError("Physical damping ratio must be finite and positive.")
    if not torch.isfinite(task_inertia).all() or not torch.isfinite(stiffness).all():
        raise RuntimeError("Task inertia and stiffness must be finite.")
    original_dtype = task_inertia.dtype
    inertia = task_inertia.to(torch.float64)
    gain = stiffness.to(torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (inertia + inertia.mT))
    if not (eigenvalues > 0).all() or not (gain > 0).all():
        raise RuntimeError("Physical damping requires positive-definite inertia and positive stiffness.")
    sqrt_mass = (eigenvectors * eigenvalues.sqrt().unsqueeze(-2)) @ eigenvectors.mT
    inverse_sqrt_mass = (eigenvectors * eigenvalues.rsqrt().unsqueeze(-2)) @ eigenvectors.mT
    modal_stiffness = inverse_sqrt_mass @ torch.diag_embed(gain) @ inverse_sqrt_mass
    frequencies_squared, modes = torch.linalg.eigh(0.5 * (modal_stiffness + modal_stiffness.mT))
    # Tiny negative values can arise only from roundoff after the SPD transforms.
    root_modal_stiffness = (modes * frequencies_squared.clamp_min(0).sqrt().unsqueeze(-2)) @ modes.mT
    damping = 2.0 * damping_ratio * sqrt_mass @ root_modal_stiffness @ sqrt_mass
    return (0.5 * (damping + damping.mT)).to(original_dtype)


def implicit_cartesian_damping(
    jacobian: torch.Tensor, mass_matrix: torch.Tensor, stiffness: torch.Tensor,
    velocity: torch.Tensor, pose_error: torch.Tensor, dt: float,
    damping_ratio: float = 1.0, regularization: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damping correction for stable sampled physical Cartesian impedance.

    The returned correction is subtracted from K*error. Solve
    (M_task + dt*D + dt^2*K)*v_next = M_task*v + dt*K*error,
    then F = K*error - (D + dt*K)*v_next. This backward-Euler spring/damper
    discretization stays stable for small wrist inertias at the 120 Hz physics
    rate. It preserves physical stiffness; stiffness itself is never divided
    by mass or weakened to hide a numerical instability.
    """
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(regularization) or regularization <= 0:
        raise ValueError("Physics time step and inertia regularization must be finite and positive.")
    if not all(torch.isfinite(value).all() for value in (jacobian, mass_matrix, stiffness, velocity, pose_error)):
        raise RuntimeError("Implicit Cartesian controller state and commands must be finite.")
    output_dtype = jacobian.dtype
    j, m, gains, v, error = [value.to(torch.float64)
                           for value in (jacobian, mass_matrix, stiffness, velocity, pose_error)]
    inverse_task_inertia = j @ torch.linalg.solve(0.5 * (m + m.mT), j.mT)
    inverse_eigenvalues, task_axes = torch.linalg.eigh(
        0.5 * (inverse_task_inertia + inverse_task_inertia.mT)
    )
    task_inertia = (task_axes * inverse_eigenvalues.clamp_min(regularization).reciprocal().unsqueeze(-2)) @ task_axes.mT
    task_inertia = 0.5 * (task_inertia + task_inertia.mT)
    damping = physical_damping_matrix(task_inertia, gains, damping_ratio)
    stiffness_matrix = torch.diag_embed(gains)
    system = task_inertia + dt * damping + dt * dt * stiffness_matrix
    rhs = (task_inertia @ v.unsqueeze(-1) + dt * (gains * error).unsqueeze(-1))
    next_velocity = torch.linalg.solve(system, rhs).squeeze(-1)
    correction = ((damping + dt * stiffness_matrix) @ next_velocity.unsqueeze(-1)).squeeze(-1)
    return (correction.to(output_dtype), next_velocity.to(output_dtype),
            task_inertia.to(output_dtype), damping.to(output_dtype))


def _axis_values(value: float | Sequence[float], count: int, label: str) -> tuple[float, ...]:
    values = (float(value),) * count if isinstance(value, (int, float)) else tuple(float(v) for v in value)
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{label} must contain {count} finite values.")
    return values


def _load_simulator_classes() -> None:
    """Create the SDK classes only after the caller has started Isaac Sim."""
    from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
    from isaaclab.managers import ActionTerm, ActionTermCfg
    from isaaclab.utils import configclass
    from isaaclab.utils import math as math_utils

    from ..src.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES, TCP_BODY_NAME

    class RearrangementAction(ActionTerm):
        """18D fixed-base TCP OSC and six independent hand position targets.

        The reference frame is the robot root, as in the Sweeping example's
        subtract_frame_transforms(root_pose_w, object_or_ee_pose_w). FrameTransformer
        in that project likewise uses base_link as its source frame. It is not
        the environment origin and is not the moving wrist or hand frame.
        """

        def __init__(self, cfg, env):
            position_scale = _axis_values(cfg.position_scale, 3, "position_scale")
            orientation_scale = _axis_values(cfg.orientation_scale, 3, "orientation_scale")
            gain_min = _axis_values(cfg.stiffness_min, 6, "stiffness_min")
            gain_max = _axis_values(cfg.stiffness_max, 6, "stiffness_max")
            workspace_min = _axis_values(cfg.workspace_min, 3, "workspace_min")
            workspace_max = _axis_values(cfg.workspace_max, 3, "workspace_max")
            if not all(v > 0 for v in (*position_scale, *orientation_scale, *gain_min)):
                raise ValueError("Motion scales and minimum OSC gains must be positive.")
            if not all(low < high for low, high in zip(gain_min, gain_max)):
                raise ValueError("Each maximum OSC gain must exceed its minimum.")
            if not all(low < high for low, high in zip(workspace_min, workspace_max)):
                raise ValueError("Each workspace maximum must exceed its minimum.")
            if not math.isfinite(cfg.effort_limit_scale) or not 0 < cfg.effort_limit_scale <= 1:
                raise ValueError("effort_limit_scale must be in (0, 1].")
            if not math.isfinite(cfg.damping_ratio) or cfg.damping_ratio <= 0:
                raise ValueError("damping_ratio must be finite and positive.")
            if not math.isfinite(cfg.hand_target_rate) or cfg.hand_target_rate <= 0:
                raise ValueError("hand_target_rate must be finite and positive.")
            if not math.isfinite(cfg.inertia_regularization) or cfg.inertia_regularization <= 0:
                raise ValueError("inertia_regularization must be finite and positive.")
            if cfg.inertial_dynamics_decoupling:
                raise ValueError("Physical Cartesian stiffness requires inertial_dynamics_decoupling=False.")
            super().__init__(cfg, env)
            if not self._asset.is_fixed_base:
                raise ValueError("Rearrangement OSC requires the fixed-base UR5e assembly.")
            self._arm_ids, arm_names = self._asset.find_joints(cfg.arm_joint_names, preserve_order=True)
            self._hand_ids, hand_names = self._asset.find_joints(cfg.hand_joint_names, preserve_order=True)
            if len(self._arm_ids) != 6 or tuple(arm_names) != tuple(cfg.arm_joint_names):
                raise ValueError("OSC requires exactly six arm joints in the configured name order.")
            if len(self._hand_ids) != 6 or tuple(hand_names) != tuple(cfg.hand_joint_names):
                raise ValueError("Hand action requires exactly six independent joints in the configured name order.")
            body_ids, _ = self._asset.find_bodies(cfg.body_name)
            if len(body_ids) != 1 or body_ids[0] == 0:
                raise ValueError("TCP must resolve to exactly one non-root articulation body.")
            self._tcp_id = body_ids[0]
            # Fixed-base PhysX omits the root body from the Jacobian body axis.
            self._jacobian_id = self._tcp_id - 1
            self._arm_columns = torch.tensor(self._arm_ids, dtype=torch.long, device=self.device)
            self._position_scale = torch.tensor(position_scale, device=self.device)
            self._orientation_scale = torch.tensor(orientation_scale, device=self.device)
            self._gain_min = torch.tensor(gain_min, device=self.device)
            self._gain_max = torch.tensor(gain_max, device=self.device)
            self._workspace_min = torch.tensor(workspace_min, device=self.device)
            self._workspace_max = torch.tensor(workspace_max, device=self.device)
            hand_limits = self._asset.data.soft_joint_pos_limits[:, self._hand_ids, :]
            if not torch.isfinite(hand_limits).all() or not (
                hand_limits[..., 1] > hand_limits[..., 0]
            ).all():
                raise ValueError("Independent hand joints require finite, nonempty soft position limits.")
            self._hand_limits = hand_limits.clone()
            self._hand_rate_limit = self._asset.data.joint_vel_limits[:, self._hand_ids].clamp(
                max=cfg.hand_target_rate
            ).clone()
            if not torch.isfinite(self._hand_rate_limit).all() or not (self._hand_rate_limit > 0).all():
                raise ValueError("Independent hand joints require positive native velocity limits.")
            self._physics_dt = env.sim.get_physics_dt()
            controller_cfg = OperationalSpaceControllerCfg(
                target_types=["pose_abs"], impedance_mode="variable_kp",
                inertial_dynamics_decoupling=cfg.inertial_dynamics_decoupling,
                gravity_compensation=True, motion_damping_ratio_task=cfg.damping_ratio,
                motion_stiffness_task=tuple(0.5 * (a + b) for a, b in zip(gain_min, gain_max)),
                motion_stiffness_limits_task=(min(gain_min), max(gain_max)),
                nullspace_control="none",
            )
            self._osc = OperationalSpaceController(controller_cfg, self.num_envs, self.device)
            self._raw_actions = torch.zeros(self.num_envs, 18, device=self.device)
            self._processed_actions = torch.zeros_like(self._raw_actions)
            self._normalized_actions = torch.zeros_like(self._raw_actions)
            self._previous_actions = torch.zeros_like(self._raw_actions)
            self._pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
            self._pose_command_b[:, 3] = 1.0
            self._stiffness_command = (0.5 * (self._gain_min + self._gain_max)).repeat(self.num_envs, 1)
            self._hand_command = hand_limits.mean(dim=-1).clone()
            self._hand_applied_command = torch.zeros_like(self._hand_command)
            self._joint_efforts = torch.zeros(self.num_envs, 6, device=self.device)
            self._task_inertia = torch.zeros(self.num_envs, 6, 6, device=self.device)
            self._damping_matrix = torch.zeros_like(self._task_inertia)
            self._torque_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._action_clipped = torch.zeros_like(self._torque_saturated)
            self._workspace_clipped = torch.zeros_like(self._torque_saturated)
            self._command_valid = torch.zeros_like(self._torque_saturated)

        @property
        def action_dim(self):
            return 18

        @property
        def raw_actions(self):
            return self._raw_actions

        @property
        def processed_actions(self):
            """Motion delta6, physical OSC gains6, hand positions6."""
            return self._processed_actions

        @property
        def normalized_actions(self):
            return self._normalized_actions

        @property
        def previous_actions(self):
            """Normalized command preceding the most recent process_actions call."""
            return self._previous_actions

        @property
        def pose_command_b(self):
            return self._pose_command_b

        @property
        def stiffness_command(self):
            return self._stiffness_command

        @property
        def hand_command(self):
            return self._hand_command

        @property
        def hand_applied_command(self):
            """Rate-bounded PD target, distinct from the desired absolute hand pose."""
            return self._hand_applied_command

        @property
        def joint_efforts(self):
            return self._joint_efforts

        @property
        def torque_saturated(self):
            return self._torque_saturated

        @property
        def action_clipped(self):
            return self._action_clipped

        @property
        def workspace_clipped(self):
            return self._workspace_clipped

        @property
        def command_valid(self):
            return self._command_valid

        @property
        def task_inertia(self):
            return self._task_inertia

        @property
        def damping_matrix(self):
            return self._damping_matrix

        def measured_tcp_pose_b(self):
            data = self._asset.data
            position, orientation = math_utils.subtract_frame_transforms(
                data.root_pos_w, data.root_quat_w,
                data.body_link_pos_w[:, self._tcp_id], data.body_link_quat_w[:, self._tcp_id],
            )
            return torch.cat((position, math_utils.quat_unique(orientation)), dim=-1)

        def process_actions(self, actions):
            if actions.shape != (self.num_envs, 18):
                raise ValueError(f"Expected actions with shape {(self.num_envs, 18)}, got {tuple(actions.shape)}.")
            normalized, delta, gains, hand = map_action(
                actions, self._position_scale, self._orientation_scale,
                self._gain_min, self._gain_max, self._hand_limits,
            )
            measured_pose = self.measured_tcp_pose_b()
            if not torch.isfinite(measured_pose).all():
                raise RuntimeError("Measured TCP pose must be finite before latching an OSC target.")
            target, workspace_clipped = pose_target_from_delta(
                measured_pose, delta, self._workspace_min, self._workspace_max,
            )
            self._previous_actions.copy_(self._normalized_actions)
            self._raw_actions.copy_(actions)
            self._normalized_actions.copy_(normalized)
            self._processed_actions.copy_(torch.cat((delta, gains, hand), dim=-1))
            self._pose_command_b.copy_(target)
            self._stiffness_command.copy_(gains)
            self._hand_command.copy_(hand)
            self._action_clipped.copy_((actions != normalized).any(dim=-1))
            self._workspace_clipped.copy_(workspace_clipped)
            self._torque_saturated.zero_()
            self._command_valid.fill_(True)
            # Absolute pose is built once from the measured pose. Physics substeps
            # recompute feedback/dynamics while retaining this stationary target.
            self._osc.set_command(torch.cat((target, gains), dim=-1))

        def apply_actions(self):
            if not self._command_valid.all():
                raise RuntimeError("An OSC action must be processed after reset before applying it.")
            data = self._asset.data
            native_jacobian = self._asset.root_physx_view.get_jacobians()[
                :, self._jacobian_id
            ].index_select(-1, self._arm_columns)
            # Native spatial Jacobian/velocity uses link COM, while our TCP is
            # the actor-frame origin. Shift BOTH the Jacobian and measured twist.
            com_to_link_w = -math_utils.quat_apply(
                data.body_link_quat_w[:, self._tcp_id], data.body_com_pos_b[:, self._tcp_id],
            )
            world_to_base = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w))
            jacobian_b = jacobian_at_link_origin(native_jacobian, com_to_link_w, world_to_base)
            tcp_velocity_w = data.body_link_vel_w[:, self._tcp_id]
            tcp_velocity_b = torch.cat((
                math_utils.quat_apply_inverse(data.root_quat_w, tcp_velocity_w[:, :3]),
                math_utils.quat_apply_inverse(data.root_quat_w, tcp_velocity_w[:, 3:]),
            ), dim=-1)
            mass_matrix = self._asset.root_physx_view.get_generalized_mass_matrices().index_select(
                1, self._arm_columns
            ).index_select(2, self._arm_columns)
            gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._arm_ids]
            measured_pose_b = self.measured_tcp_pose_b()
            efforts = self._osc.compute(
                jacobian_b=jacobian_b, current_ee_pose_b=measured_pose_b,
                # SDK's default 2*sqrt(K) damping assumes unit mass. Use its
                # public spring/gravity output and add coupled physical damping
                # below, rather than editing private SDK controller buffers.
                current_ee_vel_b=torch.zeros_like(tcp_velocity_b), mass_matrix=mass_matrix, gravity=gravity,
                current_joint_pos=data.joint_pos[:, self._arm_ids],
                current_joint_vel=data.joint_vel[:, self._arm_ids],
            )
            pose_error_b = torch.cat(math_utils.compute_pose_error(
                measured_pose_b[:, :3], measured_pose_b[:, 3:],
                self._pose_command_b[:, :3], self._pose_command_b[:, 3:],
                rot_error_type="axis_angle",
            ), dim=-1)
            damping_correction, _, task_inertia, damping = implicit_cartesian_damping(
                jacobian_b, mass_matrix, self._stiffness_command,
                tcp_velocity_b, pose_error_b, self._physics_dt,
                damping_ratio=self.cfg.damping_ratio,
                regularization=self.cfg.inertia_regularization,
            )
            self._task_inertia.copy_(task_inertia)
            self._damping_matrix.copy_(damping)
            efforts -= (jacobian_b.mT @ damping_correction.unsqueeze(-1)).squeeze(-1)
            limits = data.joint_effort_limits[:, self._arm_ids] * self.cfg.effort_limit_scale
            applied, saturated = clamp_efforts(efforts, limits)
            self._joint_efforts.copy_(applied)
            self._torque_saturated.logical_or_(saturated)
            self._asset.set_joint_effort_target(applied, joint_ids=self._arm_ids)
            # Only independent joints are commanded. The authored mimic joints
            # and existing hand PD actuators retain their own physical constraints.
            self._hand_applied_command.copy_(bounded_hand_target(
                self._hand_command, data.joint_pos[:, self._hand_ids],
                self._hand_rate_limit, self._physics_dt,
            ))
            self._asset.set_joint_position_target(self._hand_applied_command, joint_ids=self._hand_ids)

        def reset(self, env_ids=None):
            selected = slice(None) if env_ids is None else env_ids
            self._raw_actions[selected] = 0
            self._processed_actions[selected] = 0
            self._normalized_actions[selected] = 0
            self._previous_actions[selected] = 0
            self._pose_command_b[selected] = 0
            self._pose_command_b[selected, 3] = 1
            self._stiffness_command[selected] = 0.5 * (self._gain_min + self._gain_max)
            self._hand_command[selected] = self._hand_limits[selected].mean(dim=-1)
            self._hand_applied_command[selected] = 0
            self._joint_efforts[selected] = 0
            self._task_inertia[selected] = 0
            self._damping_matrix[selected] = 0
            self._torque_saturated[selected] = False
            self._action_clipped[selected] = False
            self._workspace_clipped[selected] = False
            self._command_valid[selected] = False
            self._asset.set_joint_effort_target(
                self._joint_efforts[selected], joint_ids=self._arm_ids, env_ids=selected,
            )

    @configclass
    class RearrangementActionCfg(ActionTermCfg):
        """Configurable provisional control scales for the 18D policy command.

        Stiffness is physical N/m and N m/rad; inertia decoupling is disabled.
        Damping uses coupled task inertia and backward-Euler discretization,
        rather than Isaac Lab's unit-mass explicit damping convention.
        """

        class_type: type = RearrangementAction
        asset_name: str = "robot"
        arm_joint_names: list[str] = list(ARM_JOINT_NAMES)
        hand_joint_names: list[str] = list(HAND_JOINT_NAMES)
        body_name: str = TCP_BODY_NAME
        position_scale: float | tuple[float, float, float] = 0.015
        orientation_scale: float | tuple[float, float, float] = 0.08
        stiffness_min: tuple[float, ...] = (100.0, 100.0, 100.0, 5.0, 5.0, 5.0)
        stiffness_max: tuple[float, ...] = (800.0, 800.0, 800.0, 40.0, 40.0, 40.0)
        workspace_min: tuple[float, float, float] = (0.15, -0.55, 0.10)
        workspace_max: tuple[float, float, float] = (1.0, 0.55, 0.80)
        damping_ratio: float = 1.0
        inertial_dynamics_decoupling: bool = False
        effort_limit_scale: float = 1.0
        hand_target_rate: float = 1.0
        inertia_regularization: float = 1.0e-6

    # Expose ordinary module-level class identities, including for config/pickle
    # serialization, even though their SDK imports occur on first use.
    RearrangementAction.__qualname__ = "RearrangementAction"
    RearrangementActionCfg.__qualname__ = "RearrangementActionCfg"
    globals().update(RearrangementAction=RearrangementAction, RearrangementActionCfg=RearrangementActionCfg)


def __getattr__(name: str):
    if name in ("RearrangementAction", "RearrangementActionCfg"):
        _load_simulator_classes()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
