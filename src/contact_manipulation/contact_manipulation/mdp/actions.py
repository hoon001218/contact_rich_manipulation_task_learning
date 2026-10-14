"""Native Isaac Lab action term for physical Cartesian impedance and the hand."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

def map_action(
    actions: torch.Tensor,
    motion_scale: torch.Tensor,
    stiffness_min: torch.Tensor,
    stiffness_span: torch.Tensor,
    hand_lower: torch.Tensor,
    hand_span: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map normalized delta-pose6 / stiffness6 / hand6 to physical commands.

    Translation is metres, rotation is a base-frame rotation vector in radians,
    and stiffness is N/m or N m/rad. Hand targets are absolute radians. Constant
    scale/span tensors are constructed once by the action term.
    """
    if actions.ndim != 2 or actions.shape[-1] != 18:
        raise ValueError("Actions must have shape (num_envs, 18).")
    normalized = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return (
        normalized,
        normalized[:, :6] * motion_scale,
        stiffness_min + 0.5 * (normalized[:, 6:12] + 1.0) * stiffness_span,
        hand_lower + 0.5 * (normalized[:, 12:18] + 1.0) * hand_span,
    )


def pose_target_from_delta(
    measured_pose_b: torch.Tensor,
    delta_b: torch.Tensor,
    workspace_min: torch.Tensor,
    workspace_max: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Latch measured-pose-relative WXYZ target, with rotation about base axes."""
    position = measured_pose_b[:, :3] + delta_b[:, :3]
    bounded_position = position.clamp(min=workspace_min, max=workspace_max)
    vector = delta_b[:, 3:]
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    increment = torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * math.pi)) * vector), -1)
    measured = measured_pose_b[:, 3:]
    orientation = torch.cat((
        increment[:, :1] * measured[:, :1] - (increment[:, 1:] * measured[:, 1:]).sum(-1, keepdim=True),
        increment[:, :1] * measured[:, 1:] + measured[:, :1] * increment[:, 1:]
        + torch.linalg.cross(increment[:, 1:], measured[:, 1:], dim=-1),
    ), -1)
    orientation = orientation / torch.linalg.vector_norm(orientation, dim=-1, keepdim=True).clamp_min(1e-12)
    orientation = torch.where(orientation[:, :1] < 0, -orientation, orientation)
    return torch.cat((bounded_position, orientation), -1), (position != bounded_position).any(-1)


def jacobian_at_link_origin(
    jacobian_com_w: torch.Tensor, com_to_link_w: torch.Tensor, world_to_base: torch.Tensor,
) -> torch.Tensor:
    """Shift PhysX COM Jacobian to the TCP link origin and rotate into base axes."""
    angular = jacobian_com_w[:, 3:, :]
    linear = jacobian_com_w[:, :3, :] + torch.linalg.cross(
        angular.mT, com_to_link_w[:, None, :].expand_as(angular.mT), dim=-1,
    ).mT
    return torch.cat((world_to_base @ linear, world_to_base @ angular), dim=1)


def bounded_hand_target(
    desired: torch.Tensor, measured: torch.Tensor, maximum_step: torch.Tensor,
) -> torch.Tensor:
    """Bound PD position lead relative to measurement, without target windup."""
    return measured + (desired - measured).clamp(min=-maximum_step, max=maximum_step)


def symmetric_matrix_sqrt(matrix: torch.Tensor, identity: torch.Tensor, iterations: int = 32):
    """Approximate an SPD square root with scaled Newton--Schulz iterations.

    A fixed iteration count keeps work on the device. Unlike ``linalg.eigh``,
    this does not synchronize CUDA with the host. Matrices here are only 6x6.
    The returned symmetric approximation preserves the input dtype; it does not
    promote all environments to float64 on consumer GPUs.
    """
    scale = torch.linalg.matrix_norm(matrix, ord="fro", dim=(-2, -1), keepdim=True).clamp_min(
        torch.finfo(matrix.dtype).tiny
    )
    root = matrix / scale
    inverse_root = identity
    for _ in range(iterations):
        correction = 0.5 * (3.0 * identity - inverse_root @ root)
        root = root @ correction
        inverse_root = correction @ inverse_root
    # A relative residual detects inaccurate *small* eigen-directions that a
    # Frobenius reconstruction error can hide near singular arm postures.
    residual = (identity - inverse_root @ root).abs().sum(-1).amax(-1)
    converged = torch.isfinite(residual) & (residual <= 1.0e-2)
    return 0.5 * (root + root.mT) * scale.sqrt(), converged


_sqrt_graphs = {}


def _symmetric_matrix_sqrt_cuda(matrix, identity):
    """Replay the same 32-iteration tensor operations without per-op Python launches."""
    if matrix.requires_grad or identity.requires_grad:
        return symmetric_matrix_sqrt(matrix, identity)
    key = (str(matrix.device),matrix.dtype,identity.dtype,tuple(matrix.shape),tuple(identity.shape),
           torch.backends.cuda.matmul.allow_tf32,torch.is_inference_mode_enabled())
    if key not in _sqrt_graphs:
        values, eye = matrix.clone(), identity.clone()
        with torch.cuda.device(matrix.device):
            stream = torch.cuda.Stream(device=matrix.device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    symmetric_matrix_sqrt(values, eye)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = symmetric_matrix_sqrt(values, eye)
        _sqrt_graphs[key] = values,eye,graph,outputs
    values,eye,graph,outputs = _sqrt_graphs[key]
    values.copy_(matrix)
    eye.copy_(identity)
    graph.replay()
    # Callers own their results even after another replay or environment step.
    return tuple(output.clone() for output in outputs)


def implicit_cartesian_damping(
    jacobian: torch.Tensor,
    mass_matrix: torch.Tensor,
    stiffness: torch.Tensor,
    velocity: torch.Tensor,
    pose_error: torch.Tensor,
    dt: float,
    identity: torch.Tensor,
    damping_ratio: float = 1.0,
    regularization: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward-Euler physical spring/damper correction and device error flags.

    ``M_task = (J M^-1 J.T + eps I)^-1`` regularizes singular arm postures.
    Coupled critical damping is ``D = 2*zeta*(K # M_task)``, where ``#`` is
    the matrix geometric mean. With diagonal K, only one SPD square root is
    needed, rather than three eigendecompositions. The correction is subtracted
    from native OSC's spring/gravity efforts after multiplication by ``J.T``.

    Returns correction (N,6), predicted next velocity (N,6), invalid (N,).
    Linear-solver error codes stay on the device and must be handled by the
    caller; ``check_errors=False`` avoids a host synchronization in each solve.
    """
    inverse_joint_times_j, joint_info = torch.linalg.solve_ex(
        0.5 * (mass_matrix + mass_matrix.mT), jacobian.mT, check_errors=False,
    )
    inverse_task = jacobian @ inverse_joint_times_j
    inverse_task = 0.5 * (inverse_task + inverse_task.mT) + regularization * identity
    inertia, task_info = torch.linalg.solve_ex(inverse_task, identity, check_errors=False)
    inertia = 0.5 * (inertia + inertia.mT)
    root_gain = stiffness.sqrt()
    scaled_inertia = inertia / root_gain.unsqueeze(-1) / root_gain.unsqueeze(-2)
    sqrt = _symmetric_matrix_sqrt_cuda if scaled_inertia.is_cuda else symmetric_matrix_sqrt
    root, converged = sqrt(scaled_inertia, identity)
    damping = 2.0 * damping_ratio * (root_gain.unsqueeze(-1) * root * root_gain.unsqueeze(-2))
    effective_damping = damping + dt * torch.diag_embed(stiffness)
    system = inertia + dt * effective_damping
    rhs = (inertia @ velocity.unsqueeze(-1)).squeeze(-1) + dt * stiffness * pose_error
    next_velocity, system_info = torch.linalg.solve_ex(system, rhs.unsqueeze(-1), check_errors=False)
    correction = (effective_damping @ next_velocity).squeeze(-1)
    invalid = ((joint_info != 0) | (task_info != 0) | (system_info != 0)
               | ~converged | ~torch.isfinite(correction).all(-1))
    return correction, next_velocity.squeeze(-1), invalid


from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils



def _axis_values(value: float | Sequence[float], size: int, name: str) -> tuple[float, ...]:
    values = (float(value),) * size if isinstance(value, (int, float)) else tuple(value)
    if len(values) != size or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain {size} finite values.")
    return values


class ContactManipulationAction(ActionTerm):
    """18D delta pose / physical stiffness / six independent hand positions.

    ``process_actions`` latches a target from the measured TCP once per policy
    step. Native ``ActionManager`` invokes ``apply_actions`` at every physics
    substep. Joint/body resolution and device constants are cached at startup.
    The robot needs zero arm PD gains, hand PD actuators, and native mimic joints.
    """

    def __init__(self, cfg: ContactManipulationActionCfg, env):
        super().__init__(cfg, env)
        if not self._asset.is_fixed_base:
            raise ValueError("Contact manipulation requires a fixed-base robot.")
        self._arm_ids, arm_names = self._asset.find_joints(cfg.arm_joint_names, preserve_order=True)
        self._hand_ids, hand_names = self._asset.find_joints(cfg.hand_joint_names, preserve_order=True)
        if len(arm_names) != 6 or tuple(arm_names) != tuple(cfg.arm_joint_names):
            raise ValueError("Resolve exactly six arm joints in the configured order.")
        if len(hand_names) != 6 or tuple(hand_names) != tuple(cfg.hand_joint_names):
            raise ValueError("Resolve exactly six independent hand joints in the configured order.")
        if set(self._arm_ids) & set(self._hand_ids):
            raise ValueError("Arm and hand action joints must be disjoint.")
        body_ids, _ = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1 or body_ids[0] == 0:
            raise ValueError("TCP must resolve to one non-root articulation body.")
        self._tcp_id = body_ids[0]
        self._jacobian_id = self._tcp_id - 1  # PhysX excludes fixed root from Jacobians.
        self._arm_columns = torch.tensor(self._arm_ids, dtype=torch.long, device=self.device)
        self._hand_columns = torch.tensor(self._hand_ids, dtype=torch.long, device=self.device)

        scales = (*_axis_values(cfg.position_scale, 3, "position_scale"),
                  *_axis_values(cfg.orientation_scale, 3, "orientation_scale"))
        lower = _axis_values(cfg.stiffness_min, 6, "stiffness_min")
        upper = _axis_values(cfg.stiffness_max, 6, "stiffness_max")
        workspace_min = _axis_values(cfg.workspace_min, 3, "workspace_min")
        workspace_max = _axis_values(cfg.workspace_max, 3, "workspace_max")
        if any(value <= 0 for value in (*scales, *lower)):
            raise ValueError("Action scales and minimum stiffness must be positive.")
        if any(low >= high for low, high in zip(lower, upper)):
            raise ValueError("Each stiffness maximum must exceed its minimum.")
        if any(low >= high for low, high in zip(workspace_min, workspace_max)):
            raise ValueError("Each workspace maximum must exceed its minimum.")
        for name in ("damping_ratio", "hand_target_rate", "inertia_regularization", "effort_limit_scale"):
            if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if cfg.effort_limit_scale > 1:
            raise ValueError("effort_limit_scale may not exceed native actuator limits.")
        self._motion_scale = torch.tensor(scales, device=self.device)
        self._gain_min = torch.tensor(lower, device=self.device)
        self._gain_span = torch.tensor(upper, device=self.device) - self._gain_min
        self._workspace_min = torch.tensor(workspace_min, device=self.device)
        self._workspace_max = torch.tensor(workspace_max, device=self.device)
        self._physics_dt = env.sim.get_physics_dt()
        data = self._asset.data
        limits = data.soft_joint_pos_limits.index_select(1, self._hand_columns)
        self._hand_lower = limits[..., 0].clone()
        self._hand_span = (limits[..., 1] - limits[..., 0]).clone()
        self._hand_step = data.joint_vel_limits.index_select(1, self._hand_columns).clamp(
            max=cfg.hand_target_rate,
        ) * self._physics_dt
        self._effort_limits = data.joint_effort_limits.index_select(1, self._arm_columns) * cfg.effort_limit_scale
        # Initialization checks may synchronize once; these are never in the rollout path.
        if not bool(torch.isfinite(limits).all() & (self._hand_span > 0).all()):
            raise ValueError("Hand joints require finite, nonempty soft limits.")
        if not bool(torch.isfinite(self._hand_step).all() & (self._hand_step > 0).all()):
            raise ValueError("Hand joints require positive native velocity limits.")
        if not bool(torch.isfinite(self._effort_limits).all() & (self._effort_limits > 0).all()):
            raise ValueError("Arm joints require finite positive native effort limits.")
        if bool((data.joint_stiffness[:, self._arm_ids] != 0).any()
                | (data.joint_damping[:, self._arm_ids] != 0).any()):
            raise ValueError("Disable arm PD stiffness/damping in the robot actuator configuration for torque OSC.")

        self._osc = OperationalSpaceController(OperationalSpaceControllerCfg(
            target_types=["pose_abs"], impedance_mode="variable_kp",
            inertial_dynamics_decoupling=False, gravity_compensation=True,
            motion_stiffness_task=tuple((low + high) / 2 for low, high in zip(lower, upper)),
            motion_stiffness_limits_task=(min(lower), max(upper)),
            motion_damping_ratio_task=cfg.damping_ratio, nullspace_control="none",
        ), self.num_envs, self.device)
        self._raw_actions = torch.zeros(self.num_envs, 18, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._normalized_actions = torch.zeros_like(self._raw_actions)
        self._previous_actions = torch.zeros_like(self._raw_actions)
        self._controller_command = torch.zeros(self.num_envs, 13, device=self.device)
        self._controller_command[:, 3] = 1
        self._controller_command[:, 7:] = self._gain_min + 0.5 * self._gain_span
        self._hand_command = self._hand_lower + 0.5 * self._hand_span
        self._hand_applied_command = torch.zeros_like(self._hand_command)
        self._joint_efforts = torch.zeros(self.num_envs, 6, device=self.device)
        self._zero_velocity = torch.zeros_like(self._joint_efforts)
        self._identity = torch.eye(6, device=self.device).expand(self.num_envs, -1, -1)
        self._command_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._action_clipped = torch.zeros_like(self._command_valid)
        self._workspace_clipped = torch.zeros_like(self._command_valid)
        self._torque_saturated = torch.zeros_like(self._command_valid)
        self._invalid_control = torch.zeros_like(self._command_valid)

    @property
    def action_dim(self) -> int:
        return 18

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def normalized_actions(self) -> torch.Tensor:
        return self._normalized_actions

    @property
    def previous_actions(self) -> torch.Tensor:
        return self._previous_actions

    @property
    def arm_joint_ids(self) -> list[int]:
        return self._arm_ids

    @property
    def hand_joint_ids(self) -> list[int]:
        return self._hand_ids

    @property
    def tcp_body_id(self) -> int:
        return self._tcp_id

    @property
    def invalid_control(self) -> torch.Tensor:
        """Device flags for numerical failure; termination should include these."""
        return self._invalid_control

    @property
    def torque_saturated(self) -> torch.Tensor:
        return self._torque_saturated

    @property
    def workspace_clipped(self) -> torch.Tensor:
        return self._workspace_clipped

    @property
    def action_clipped(self) -> torch.Tensor:
        return self._action_clipped

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    @property
    def hand_applied_command(self) -> torch.Tensor:
        return self._hand_applied_command

    def measured_tcp_pose_b(self) -> torch.Tensor:
        data = self._asset.data
        position, orientation = math_utils.subtract_frame_transforms(
            data.root_pos_w, data.root_quat_w,
            data.body_link_pos_w[:, self._tcp_id], data.body_link_quat_w[:, self._tcp_id],
        )
        return torch.cat((position, math_utils.quat_unique(orientation)), -1)

    def process_actions(self, actions: torch.Tensor):
        if actions.shape != self._raw_actions.shape:
            raise ValueError(f"Expected action shape {tuple(self._raw_actions.shape)}.")
        normalized, delta, stiffness, hand = map_action(
            actions, self._motion_scale, self._gain_min, self._gain_span, self._hand_lower, self._hand_span,
        )
        target, clipped = pose_target_from_delta(
            self.measured_tcp_pose_b(), delta, self._workspace_min, self._workspace_max,
        )
        self._previous_actions.copy_(self._normalized_actions)
        self._raw_actions.copy_(actions)
        self._normalized_actions.copy_(normalized)
        self._processed_actions[:, :6] = delta
        self._processed_actions[:, 6:12] = stiffness
        self._processed_actions[:, 12:] = hand
        self._controller_command[:, :7] = target
        self._controller_command[:, 7:] = stiffness
        self._hand_command.copy_(hand)
        self._action_clipped.copy_((actions != normalized).any(-1))
        self._workspace_clipped.copy_(clipped)
        self._torque_saturated.zero_()
        self._invalid_control.copy_(~torch.isfinite(actions).all(-1) | ~torch.isfinite(target).all(-1))
        self._command_valid.fill_(True)
        self._osc.set_command(self._controller_command)

    def apply_actions(self):
        data = self._asset.data
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobian_id].index_select(
            -1, self._arm_columns,
        )
        com_to_link_w = -math_utils.quat_apply(
            data.body_link_quat_w[:, self._tcp_id], data.body_com_pos_b[:, self._tcp_id],
        )
        world_to_base = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w))
        jacobian = jacobian_at_link_origin(jacobian, com_to_link_w, world_to_base)
        velocity_w = data.body_link_vel_w[:, self._tcp_id]
        velocity_b = torch.cat((
            math_utils.quat_apply_inverse(data.root_quat_w, velocity_w[:, :3]),
            math_utils.quat_apply_inverse(data.root_quat_w, velocity_w[:, 3:]),
        ), -1)
        mass_matrix = self._asset.root_physx_view.get_generalized_mass_matrices().index_select(
            1, self._arm_columns,
        ).index_select(2, self._arm_columns)
        gravity = self._asset.root_physx_view.get_gravity_compensation_forces().index_select(1, self._arm_columns)
        pose = self.measured_tcp_pose_b()
        efforts = self._osc.compute(
            jacobian_b=jacobian, current_ee_pose_b=pose, current_ee_vel_b=self._zero_velocity,
            gravity=gravity,
        )
        # Native OSC computes the physical spring and gravity term. Replace its
        # unit-mass explicit damping with coupled, implicit physical damping.
        error = torch.cat(math_utils.compute_pose_error(
            pose[:, :3], pose[:, 3:], self._controller_command[:, :3], self._controller_command[:, 3:7],
            rot_error_type="axis_angle",
        ), -1)
        correction, _, invalid = implicit_cartesian_damping(
            jacobian, mass_matrix, self._controller_command[:, 7:], velocity_b, error,
            self._physics_dt, self._identity, self.cfg.damping_ratio, self.cfg.inertia_regularization,
        )
        efforts = efforts - (jacobian.mT @ correction.unsqueeze(-1)).squeeze(-1)
        self._invalid_control.logical_or_(invalid | ~torch.isfinite(efforts).all(-1))
        bounded = torch.nan_to_num(efforts, nan=0.0, posinf=0.0, neginf=0.0).clamp(
            min=-self._effort_limits, max=self._effort_limits,
        )
        valid = self._command_valid & ~self._invalid_control
        self._joint_efforts.copy_(torch.where(valid[:, None], bounded, 0.0))
        self._torque_saturated.logical_or_((bounded != efforts).any(-1))
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._arm_ids)
        measured_hand = data.joint_pos.index_select(1, self._hand_columns)
        hand_target = bounded_hand_target(self._hand_command, measured_hand, self._hand_step)
        self._hand_applied_command.copy_(torch.where(valid[:, None], hand_target, measured_hand))
        self._asset.set_joint_position_target(self._hand_applied_command, joint_ids=self._hand_ids)

    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        for buffer in (self._raw_actions, self._processed_actions, self._normalized_actions, self._previous_actions,
                       self._joint_efforts, self._action_clipped, self._workspace_clipped, self._torque_saturated,
                       self._invalid_control, self._command_valid):
            buffer[selected] = 0
        self._controller_command[selected] = 0
        self._controller_command[selected, 3] = 1
        self._controller_command[selected, 7:] = self._gain_min + 0.5 * self._gain_span
        measured_hand = self._asset.data.joint_pos.index_select(1, self._hand_columns)[selected]
        self._hand_command[selected] = measured_hand
        self._hand_applied_command[selected] = measured_hand
        self._asset.set_joint_effort_target(self._joint_efforts[selected], joint_ids=self._arm_ids, env_ids=selected)
        self._asset.set_joint_position_target(measured_hand, joint_ids=self._hand_ids, env_ids=selected)


@configclass
class ContactManipulationActionCfg(ActionTermCfg):
    """Physical units and actor/hand joint order for the combined 18D action."""

    class_type: type = ContactManipulationAction
    asset_name: str = "robot"
    arm_joint_names: list[str] = [
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    ]
    hand_joint_names: list[str] = [
        "right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
        "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint",
    ]
    body_name: str = "right_plam_force_sensor"
    position_scale: float | tuple[float, float, float] = 0.015
    orientation_scale: float | tuple[float, float, float] = 0.08
    stiffness_min: tuple[float, ...] = (100.0, 100.0, 100.0, 5.0, 5.0, 5.0)
    stiffness_max: tuple[float, ...] = (800.0, 800.0, 800.0, 40.0, 40.0, 40.0)
    workspace_min: tuple[float, float, float] = (0.15, -0.55, 0.10)
    workspace_max: tuple[float, float, float] = (1.0, 0.55, 0.80)
    damping_ratio: float = 1.0
    effort_limit_scale: float = 1.0
    hand_target_rate: float = 1.0
    inertia_regularization: float = 1e-6
