"""Policy observations: initial geometry in EEF coordinates and two samples.

The immutable initial scene is estimated from accepted RGB-D. Live robot
proprioception changes its reference frame; current object ground truth never
updates actor geometry. Compensated F/T stays in the physical sensor frame.
The dynamic window contains the current sample and the immediately previous
sample. Each includes the action that produced it: observation at t contains
action a[t-1], and the previous observation at t-1 contains action a[t-2].
After reset the current sample has zero action and the previous sample is invalid.
Validity and role metadata are used only while assembling these measurements.
The policy receives zero-padded geometry/history and tactile readings (0/1, or
-1 when unavailable). Geometry slots 0 and 1 always belong to the target and
blocker; its caller derives masks from these positions and the observed data.
"""

from __future__ import annotations

import torch

from .state import ensure_mdp, quat_apply, quat_conjugate, quat_mul, update_mdp


def pack_policy_observation(geometry, valid, roles, history, history_valid):
    """Pack measurements with target/blocker in reserved geometry slots 0/1.

    ``valid``, ``roles`` and ``history_valid`` are internal producer bookkeeping,
    never feature channels. Observed distractors follow in their original order,
    then padding. A missing target or blocker leaves its reserved slot empty;
    distractors never take those slots. The canonical scene tensors stay intact.
    """
    geometry = torch.where(valid[..., None], geometry, 0)
    roles = torch.where(valid[..., None], roles, 0)
    target, blocker = roles[..., 0] == 1, roles[..., 1] == 1
    invalid_roles = ~((roles == 0) | (roles == 1)).all(dim=(-2, -1))
    ambiguous = (target & blocker).any(-1) | (target.sum(-1) > 1) | (blocker.sum(-1) > 1)
    if bool((invalid_roles | ambiguous).any()):
        raise ValueError("Observed object roles must be binary and identify at most one unique target and blocker.")

    capacity = geometry.shape[1]
    distractor = valid & ~(target | blocker)
    if capacity < 2 or bool((distractor.sum(-1) > capacity - 2).any()):
        raise ValueError("The object capacity must reserve two slots for target and blocker.")
    target_index, blocker_index = target.long().argmax(-1), blocker.long().argmax(-1)
    role_indices = torch.stack((target_index, blocker_index), -1)
    role_geometry = geometry.gather(1, role_indices[..., None].expand(-1, -1, geometry.shape[-1]))
    role_observed = torch.stack((target.any(-1), blocker.any(-1)), -1)
    role_geometry = torch.where(role_observed[..., None], role_geometry, 0)

    # Integer sort keys retain canonical relative order while placing padding
    # and role objects after the observed distractors.
    indices = torch.arange(capacity, device=geometry.device).expand_as(valid)
    distractor_indices = torch.where(distractor, indices, capacity + indices).argsort(-1)[:, :capacity - 2]
    distractor_geometry = geometry.gather(1, distractor_indices[..., None].expand(-1, -1, geometry.shape[-1]))
    distractor_geometry = torch.where(distractor.gather(1, distractor_indices)[..., None], distractor_geometry, 0)
    geometry = torch.cat((role_geometry, distractor_geometry), dim=1)
    history = torch.where(history_valid[..., None], history, 0)
    return torch.cat((geometry.flatten(1), history.flatten(1)), dim=-1)


def geometry_in_eef_frame(geometry, valid, tcp_pose_b):
    """Express frozen camera geometry using the *current measured* TCP frame.

    The canonical initial geometry stays in robot-base coordinates. Updating
    its reference frame uses robot proprioception, never current object GT.
    OBB dimensions remain intrinsic to each object and therefore do not rotate.
    """
    q_inv = quat_conjugate(tcp_pose_b[:, None, 3:7]).expand(*geometry.shape[:-1], 4)
    position = quat_apply(q_inv, geometry[..., :3] - tcp_pose_b[:, None, :3])
    orientation = quat_mul(q_inv, geometry[..., 3:7])
    orientation = torch.where(orientation[..., :1] < 0, -orientation, orientation)
    transformed = torch.cat((position, orientation, geometry[..., 7:10]), -1)
    return torch.where(valid[..., None], transformed, 0)


def self_motion_wrench(masses, inertias, com_position, body_rotation, com_acceleration,
                       angular_velocity, sensor_position, sensor_rotation, gravity):
    """Newton--Euler self-load of all sensor-distal bodies in the sensor frame.

    Each body's COM acceleration and angular velocity are world measurements;
    ``inertias`` are COM tensors expressed in the bodies' actor/link frames.
    Transforming angular motion into those frames is equivalent to forming
    ``I_world = R @ I_link @ R.T`` without allocating world inertia matrices.
    All body forces and moments are summed about the *current* sensor center.
    Separate link motions include articulated fingers, COM centripetal terms,
    and the angular gyroscopic term; no control-rate finite difference is used.
    """
    forces = masses[..., None] * (com_acceleration[..., :3] - gravity)
    inverse_body_rotation = quat_conjugate(body_rotation)
    alpha = quat_apply(inverse_body_rotation, com_acceleration[..., 3:6])
    omega = quat_apply(inverse_body_rotation, angular_velocity)
    angular_momentum = (inertias @ omega[..., None]).squeeze(-1)
    rotational_moment = ((inertias @ alpha[..., None]).squeeze(-1)
                         + torch.cross(omega, angular_momentum, dim=-1))
    moments = (torch.cross(com_position - sensor_position[..., None, :], forces, dim=-1)
               + quat_apply(body_rotation, rotational_moment))
    inverse_sensor_rotation = quat_conjugate(sensor_rotation)
    return torch.cat((quat_apply(inverse_sensor_rotation, forces.sum(-2)),
                      quat_apply(inverse_sensor_rotation, moments.sum(-2))), -1)


def corrected_wrench(env):
    """Remove distal gravity, translational/rotational inertia, and residual bias.

    The parent-on-child reaction includes ``sum(m*(a_COM-g))`` and its moment,
    plus ``I*alpha + omega x (I*omega)`` for every hand and distal sensor body.
    The sensor has already applied its sensing ranges and resolution to the
    total reaction. These measured values and native body kinematics belong
    to the same completed physics step. Compensation does not quantize the
    resulting external load again. Invalid/reset readings remain zero without
    evaluating stale motion. Optional MDP tare measures a residual *after*
    full self-load compensation.
    """
    from ..src.sensors import tool_wrench, tool_wrench_valid
    state = ensure_mdp(env)
    robot = env.scene["robot"]
    raw = tool_wrench(env)
    valid = tool_wrench_valid(env)
    corrected = torch.zeros_like(raw)
    selected = valid.reshape(-1).nonzero(as_tuple=False).flatten()
    if selected.numel() == 0:
        return corrected
    point = robot.data.body_link_pos_w[selected, state.sensor_id]
    rotation = robot.data.body_link_quat_w[selected, state.sensor_id]
    com = robot.data.body_com_pos_w[selected][:, state.downstream_ids]
    body_rotation = robot.data.body_link_quat_w[selected][:, state.downstream_ids]
    acceleration = robot.data.body_com_acc_w[selected][:, state.downstream_ids]
    angular_velocity = robot.data.body_com_ang_vel_w[selected][:, state.downstream_ids]
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.device, dtype=raw.dtype)
    self_load = self_motion_wrench(state.masses[selected], state.inertias[selected], com, body_rotation,
                                   acceleration, angular_velocity, point, rotation, gravity)
    corrected[selected] = raw[selected] - self_load - state.wrench_bias[selected]
    return corrected


def tare_compensated_wrench(env, env_ids=None):
    """Explicitly tare only a valid, unloaded, self-motion-compensated sensor."""
    from ..src.sensors import tool_wrench_valid
    state = ensure_mdp(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if not bool(tool_wrench_valid(env)[ids].all()):
        raise RuntimeError("Compensated F/T tare requires a completed physics step.")
    state.wrench_bias[ids] += corrected_wrench(env)[ids]


def _tcp_pose_b(env, state):
    robot = env.scene["robot"]
    root_q = quat_conjugate(robot.data.root_quat_w)
    tcp_pos = quat_apply(root_q, robot.data.body_link_pos_w[:, state.tcp_id] - robot.data.root_pos_w)
    tcp_q = quat_mul(root_q, robot.data.body_link_quat_w[:, state.tcp_id])
    tcp_q = torch.where(tcp_q[:, :1] < 0, -tcp_q, tcp_q)
    return torch.cat((tcp_pos, tcp_q), -1)


def _dynamic_observation(env, state, wrench=None, zero_action=False):
    from ..src.catalog import SI_200_8_CALIBRATION
    from ..src.sensors import tactile_contacts, tactile_valid
    cfg, robot = env.cfg.mdp, env.scene["robot"]
    tcp_pose_b = _tcp_pose_b(env, state)
    wrench = corrected_wrench(env) if wrench is None else wrench
    calibration = getattr(env.cfg, "force_torque", SI_200_8_CALIBRATION)
    scales = torch.as_tensor(calibration.ranges, device=wrench.device, dtype=wrench.dtype)
    # Compensation can produce an external load beyond the raw sensing range;
    # retain the existing policy clip independently of hardware saturation.
    normalized = (wrench / scales).clamp(-5, 5)
    valid = tactile_valid(env)
    if zero_action:
        # A reset has no completed transition in this episode. Do not retain
        # a contact measurement cached from the preceding episode.
        valid = torch.zeros_like(valid)
    contacts = tactile_contacts(env) & valid
    tactile = torch.where(valid, contacts.float(), -1.)
    action = env.action_manager.action
    if zero_action:
        action = torch.zeros((env.num_envs, 18), device=env.device)
    elif action.shape[-1] != 18:
        raise RuntimeError("The training MDP requires the 18D OSC + hand action term.")
    return torch.cat((robot.data.joint_pos[:, state.joint_ids] / cfg.joint_position_scale,
                      robot.data.joint_vel[:, state.joint_ids] / cfg.joint_velocity_scale,
                      tcp_pose_b, normalized, tactile, action.clamp(-1, 1)), -1)


def policy_observation(env):
    state = update_mdp(env)
    geometry = geometry_in_eef_frame(env.initial_object_geometry, env.initial_object_valid, _tcp_pose_b(env, state))
    return pack_policy_observation(geometry, env.initial_object_valid,
                                   env.initial_object_roles, state.history, state.history_valid)

