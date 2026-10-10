"""Policy observations: initial geometry in EEF coordinates and two samples.

The immutable initial scene is estimated from accepted RGB-D. Live robot
proprioception changes its reference frame; current object ground truth never
updates actor geometry. Compensated F/T stays in the physical sensor frame.
The dynamic window contains the current sample and the immediately previous
sample. Each includes the action that produced it: observation at t contains
action a[t-1], and the previous observation at t-1 contains action a[t-2].
After reset the current sample has zero action and the previous sample is invalid.
"""

from __future__ import annotations

import torch

from .state import ensure_mdp, quat_apply, quat_conjugate, quat_mul, update_mdp


def pack_policy_observation(geometry, valid, roles, history, history_valid):
    geometry = torch.where(valid[..., None], geometry, 0)
    roles = torch.where(valid[..., None], roles, 0)
    history = torch.where(history_valid[..., None], history, 0)
    return torch.cat((geometry.flatten(1), valid.float(), roles.flatten(1).float(), history.flatten(1),
                      history_valid.float()), dim=-1)


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


def corrected_wrench(env):
    """Gravity + residual-bias corrected reaction, at the moving sensor center.

    Joint reaction convention is parent-on-child. Static gravity support is
    ``-sum(m*g)`` and its moment is subtracted. Inertia remains observable.
    The raw sensor's optional tare is intentionally independent: an MDP bias is
    measured *after* gravity compensation, avoiding a second gravity subtraction.
    """
    from ..src.sensors import tool_wrench, tool_wrench_valid
    state = ensure_mdp(env)
    robot = env.scene["robot"]
    raw = tool_wrench(env)
    valid = tool_wrench_valid(env)
    point = robot.data.body_link_pos_w[:, state.sensor_id]
    rotation = robot.data.body_link_quat_w[:, state.sensor_id]
    com = robot.data.body_com_pos_w[:, state.downstream_ids]
    gravity = torch.tensor(env.sim.cfg.gravity, device=env.device, dtype=raw.dtype)
    support = -state.masses[..., None] * gravity
    force = support.sum(1)
    moment = torch.cross(com - point[:, None], support, dim=-1).sum(1)
    gravity_sensor = torch.cat((quat_apply(quat_conjugate(rotation), force),
                                quat_apply(quat_conjugate(rotation), moment)), -1)
    return torch.where(valid, raw - gravity_sensor - state.wrench_bias, 0)


def tare_compensated_wrench(env, env_ids=None):
    """Explicitly tare only a valid, unloaded, gravity-compensated sensor."""
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
    from ..src.sensors import tactile_contacts, tactile_valid
    cfg, robot = env.cfg.mdp, env.scene["robot"]
    tcp_pose_b = _tcp_pose_b(env, state)
    wrench = corrected_wrench(env) if wrench is None else wrench
    normalized = torch.cat((wrench[:, :3] / cfg.observation_force_scale,
                            wrench[:, 3:] / cfg.observation_torque_scale), -1).clamp(-5, 5)
    valid = tactile_valid(env)
    contacts = tactile_contacts(env) & valid
    action = env.action_manager.action
    if zero_action:
        action = torch.zeros((env.num_envs, 18), device=env.device)
    elif action.shape[-1] != 18:
        raise RuntimeError("The training MDP requires the 18D OSC + hand action term.")
    return torch.cat((robot.data.joint_pos[:, state.joint_ids] / cfg.joint_position_scale,
                      robot.data.joint_vel[:, state.joint_ids] / cfg.joint_velocity_scale,
                      tcp_pose_b, normalized, contacts.float(), valid.float(), action.clamp(-1, 1)), -1)


def policy_observation(env):
    state = update_mdp(env)
    geometry = geometry_in_eef_frame(env.initial_object_geometry, env.initial_object_valid, _tcp_pose_b(env, state))
    return pack_policy_observation(geometry, env.initial_object_valid,
                                   env.initial_object_roles, state.history, state.history_valid)

