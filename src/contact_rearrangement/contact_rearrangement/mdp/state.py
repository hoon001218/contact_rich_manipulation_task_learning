"""Shared transition state and simulator geometry for shelf rearrangement.

Only reward and termination use live object poses. Every completed physics
transition updates these episode-local accumulators once, regardless of manager
call order. Tensor geometry and configuration remain usable without Isaac Sim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MdpCfg:
    history_length: int = 2
    max_objects: int = 40
    w_clear: float = 1.0
    w_rearr: float = 1.0
    w_motion: float = 1.0
    w_time: float = 1.0
    displacement_ref: float = 0.5
    time_ref: float = 10.0
    # Provisional scales: rad/s² for acceleration, N/s and N·m/s for load increase.
    arm_joint_acceleration_ref: float = 5.0
    hand_joint_acceleration_ref: float = 10.0
    force_rate_ref: float = 200.0
    torque_rate_ref: float = 10.0
    success_reward: float = 500.0
    failure_penalty: float = 100.0
    success_overlap_area: float = 1.0e-5
    target_translation_tolerance: float = 0.02
    target_rotation_tolerance: float = 0.15
    stable_linear_velocity: float = 0.015
    stable_angular_velocity: float = 0.10
    stable_steps: int = 5
    drop_tolerance: float = 0.06
    lift_tolerance: float = 0.04
    workspace_tolerance: float = 0.015
    max_tilt: float = math.radians(50.0)
    force_hard: float = 80.0
    torque_hard: float = 4.0
    joint_position_scale: float = math.pi
    joint_velocity_scale: float = 2.0
    observation_force_scale: float = 50.0
    observation_torque_scale: float = 5.0

    def __post_init__(self):
        if self.history_length != 2 or self.max_objects != 40:
            raise ValueError("The policy interface uses current and previous samples and 40 object slots.")
        for name, value in vars(self).items():
            if not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"MDP parameter {name} must be finite.")
            if name.startswith("w_") or name.endswith("_tolerance"):
                if value < 0:
                    raise ValueError(f"MDP parameter {name} must be nonnegative.")
            elif value <= 0:
                raise ValueError(f"MDP parameter {name} must be positive.")
        if self.stable_steps != int(self.stable_steps):
            raise ValueError("stable_steps must be a positive integer.")


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def quat_apply(q, v):
    qv = q[..., 1:]
    t = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(qv, t, dim=-1)


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _compact(vertices, keep):
    """Compact a masked polygon without host synchronization or ragged tensors."""
    positions = (keep.long().cumsum(-1) - 1).clamp_min(0)
    output = torch.zeros_like(vertices)
    output.scatter_add_(-2, positions[..., None].expand_as(vertices), vertices * keep[..., None])
    return output, keep.sum(-1)


def convex_hull_xy(points):
    """Batched convex hull of the eight projected OBB corners, padded to eight.

    Angular order around the box center is a simple polygon; iteratively removing
    concave/duplicate vertices gives its convex hull, including tilted boxes.
    All operations stay on the input device; loops have fixed, small bounds.
    """
    size = points.shape[-2]
    center = points.mean(-2, keepdim=True)
    delta = points - center
    order = torch.atan2(delta[..., 1], delta[..., 0]).argsort(dim=-1, stable=True)
    polygon = points.gather(-2, order[..., None].expand_as(points))
    count = torch.full(points.shape[:-2], size, device=points.device, dtype=torch.long)
    slots = torch.arange(size, device=points.device)
    for _ in range(size):
        modulus = count.clamp_min(1)[..., None]
        previous = ((slots - 1) % modulus).long()
        following = ((slots + 1) % modulus).long()
        before = polygon.gather(-2, previous[..., None].expand_as(polygon))
        after = polygon.gather(-2, following[..., None].expand_as(polygon))
        turn = _cross2(polygon - before, after - polygon)
        duplicate = (polygon - before).square().sum(-1) < 1.0e-16
        keep = (slots < count[..., None]) & (turn >= -1.0e-10) & ~duplicate
        polygon, count = _compact(polygon, keep)
    return polygon, count


def polygon_area(vertices, count):
    slots = torch.arange(vertices.shape[-2], device=vertices.device)
    following = ((slots + 1) % count.clamp_min(1)[..., None]).long()
    after = vertices.gather(-2, following[..., None].expand_as(vertices))
    return (_cross2(vertices, after) * (slots < count[..., None])).sum(-1).abs() * 0.5


def rectangle_overlap_area(vertices, count, region):
    """Sutherland-Hodgman clipping against fixed [xmin,xmax,ymin,ymax]."""
    maximum = vertices.shape[-2] + 4
    polygon = torch.nn.functional.pad(vertices, (0, 0, 0, 4))
    slots = torch.arange(maximum, device=vertices.device)
    for axis, boundary, sign in ((0, 0, 1), (0, 1, -1), (1, 2, 1), (1, 3, -1)):
        previous = ((slots - 1) % count.clamp_min(1)[..., None]).long()
        before = polygon.gather(-2, previous[..., None].expand_as(polygon))
        limit = region[..., boundary, None]
        inside = (polygon[..., axis] - limit) * sign >= -1.0e-9
        before_inside = (before[..., axis] - limit) * sign >= -1.0e-9
        valid = slots < count[..., None]
        crosses = inside != before_inside
        denominator = polygon[..., axis] - before[..., axis]
        safe_denominator = torch.where(denominator.abs() > 1.0e-12, denominator,
                                       torch.ones_like(denominator))
        fraction = ((limit - before[..., axis]) / safe_denominator).clamp(0, 1)
        intersection = before + fraction[..., None] * (polygon - before)
        candidates = torch.stack((intersection, polygon), -2).flatten(-3, -2)
        keep = torch.stack((valid & crosses, valid & inside), -1).flatten(-2)
        compacted, count = _compact(candidates, keep)
        polygon = compacted[..., :maximum, :]
    return polygon_area(polygon, count)


def object_geometry(actor_pose, lower, upper, upright):
    """Return OBB center/corners/upright rotation from actual actor poses.

    Measured bounds are already in catalog-upright coordinates. Removing that
    catalog rotation before rotating them prevents applying YCB's correction
    twice. Off-center USD origins and roll/pitch are retained.
    """
    relative = quat_mul(actor_pose[..., 3:7], quat_conjugate(upright))
    offset = (lower + upper) * 0.5
    center = actor_pose[..., :3] + quat_apply(relative, offset)
    signs = torch.tensor(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                          (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)),
                         dtype=actor_pose.dtype, device=actor_pose.device)
    local = offset[..., None, :] + (upper - lower)[..., None, :] * 0.5 * signs
    corners = actor_pose[..., None, :3] + quat_apply(relative[..., None, :].expand(*local.shape[:-1], 4), local)
    return center, corners, relative


class MdpState:
    """Episode-local accumulators with samples ordered as current, previous.

    The two samples are one completed control transition apart. Repeated
    observation, reward, and termination reads do not advance this window.
    Scene topology is never modified here.
    """

    def __init__(self, env):
        from ..src.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES, FORCE_TORQUE_BODY_NAME, TCP_BODY_NAME
        cfg, robot = env.cfg.mdp, env.scene["robot"]
        shape, device = (env.num_envs, cfg.max_objects), env.device
        self.joint_ids = [robot.joint_names.index(name) for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
        self.tcp_id = robot.body_names.index(TCP_BODY_NAME)
        self.sensor_id = robot.body_names.index(FORCE_TORQUE_BODY_NAME)
        # PhysX disambiguates the UR and hand's two ``base_link`` names. The
        # hand base does not start with ``right_``; selecting by link name
        # would omit its 90 g mass when managers initialize before the FK model.
        link_paths = robot.root_physx_view.link_paths[0]
        self.downstream_ids = [i for i, (name, path) in enumerate(zip(robot.body_names, link_paths))
                               if name == FORCE_TORQUE_BODY_NAME or "/ROASRightHand/" in str(path)]
        if hasattr(env, "robot_pose_model"):
            parents = env.robot_pose_model.body_parents
            downstream = {FORCE_TORQUE_BODY_NAME}
            while True:
                extended = downstream | {child for child, parent in parents.items() if parent in downstream}
                if extended == downstream:
                    break
                downstream = extended
            self.downstream_ids = [i for i, name in enumerate(robot.body_names) if name in downstream]
        if len(self.downstream_ids) < 2:
            raise RuntimeError("Cannot identify the distal F/T link and its attached hand bodies.")
        self.masses = robot.root_physx_view.get_masses().to(device)[:, self.downstream_ids]
        # Native tensors store COM inertia in actor/link coordinates, column
        # major. Link orientation (not the COM principal-axis quaternion)
        # maps this tensor to the world for Newton--Euler compensation.
        inertias = robot.root_physx_view.get_inertias().to(device)[:, self.downstream_ids]
        self.inertias = inertias.reshape(env.num_envs, len(self.downstream_ids), 3, 3).transpose(-1, -2).clone()
        self.lower = self.upper = self.upright = None
        self.history = torch.zeros((*shape[:1], cfg.history_length, 72), device=device)
        # Episode bookkeeping only: this mask is not a policy feature. Invalid
        # samples are zero-padded by the observation producer.
        self.history_valid = torch.zeros((*shape[:1], cfg.history_length), device=device, dtype=torch.bool)
        self.last_time = torch.full(shape[:1], float(env.sim.current_time), device=device, dtype=torch.float64)
        self.previous_center = torch.zeros((*shape, 3), device=device)
        self.initial_center = torch.zeros_like(self.previous_center)
        self.initial_rotation = torch.zeros((*shape, 4), device=device)
        self.previous_joint_velocity = torch.zeros((env.num_envs, len(self.joint_ids)), device=device)
        self.previous_wrench = torch.zeros((env.num_envs, 6), device=device)
        self.wrench_bias = torch.zeros_like(self.previous_wrench)
        self.initial_sensor_point = torch.zeros((env.num_envs, 3), device=device)
        self.path_length = torch.zeros(env.num_envs, device=device)
        self.stable_count = torch.zeros(env.num_envs, device=device, dtype=torch.long)
        self.success = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        self.failure = torch.zeros_like(self.success)
        self.terminal_paid = torch.zeros_like(self.success)
        self.discrete_reward = torch.zeros(env.num_envs, device=device)
        self.components = torch.zeros((env.num_envs, 5), device=device)
        self.motion_costs = torch.zeros((env.num_envs, 2), device=device)
        self.blocked_ratio = torch.zeros(env.num_envs, device=device)
        self.total_overlap = torch.zeros(env.num_envs, device=device)
        self.failure_reasons = torch.zeros((env.num_envs, 6), device=device, dtype=torch.bool)


def ensure_mdp(env):
    if not hasattr(env, "mdp_state"):
        env.mdp_state = MdpState(env)
    cfg = env.cfg.mdp
    if not hasattr(env, "initial_object_geometry"):
        # ObservationManager probes dimensions before the first episode reset.
        # These placeholders must never be populated from reward ground truth.
        env.initial_object_geometry = torch.zeros((env.num_envs, cfg.max_objects, 10), device=env.device)
        env.initial_object_valid = torch.zeros((env.num_envs, cfg.max_objects), device=env.device, dtype=torch.bool)
        env.initial_object_roles = torch.zeros((env.num_envs, cfg.max_objects, 2), device=env.device)
    return env.mdp_state


def _physical_objects(env, state):
    if state.lower is None:
        from ..src.catalog import OBJECT_SPECS
        state.lower = torch.tensor([geometry.lower for geometry in env.spawn_geometries], device=env.device)
        state.upper = torch.tensor([geometry.upper for geometry in env.spawn_geometries], device=env.device)
        state.upright = torch.tensor([spec.upright_quat for spec in OBJECT_SPECS], device=env.device)
    objects = env.scene["objects"]
    ids = env.active_object_ids.clamp_min(0)
    valid = env.active_object_ids >= 0
    pose = objects.data.object_link_pose_w.gather(1, ids[..., None].expand(-1, -1, 7))
    center, corners, relative = object_geometry(pose, state.lower[ids], state.upper[ids], state.upright[ids])
    com = objects.data.object_com_pos_w.gather(1, ids[..., None].expand(-1, -1, 3))
    velocity = objects.data.object_com_vel_w.gather(1, ids[..., None].expand(-1, -1, 6)).clone()
    velocity[..., :3] += torch.cross(velocity[..., 3:], center - com, dim=-1)
    origin = env.scene.env_origins[:, None, :]
    return ids, valid, center - origin, corners - origin[..., None, :], relative, velocity


def _fixed_wrench(env, state, wrench):
    robot = env.scene["robot"]
    point = robot.data.body_link_pos_w[:, state.sensor_id]
    rotation = robot.data.body_link_quat_w[:, state.sensor_id]
    force = quat_apply(rotation, wrench[:, :3])
    torque = quat_apply(rotation, wrench[:, 3:])
    torque += torch.cross(point - state.initial_sensor_point, force, dim=-1)
    return torch.cat((force, torque), -1)


def reset_mdp(env, env_ids=None):
    """Call after camera-approved scene reset and initial-geometry construction."""
    from .observations import _dynamic_observation

    state = ensure_mdp(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    _, _, center, _, rotation, _ = _physical_objects(env, state)
    state.last_time[ids] = float(env.sim.current_time)
    state.previous_center[ids] = state.initial_center[ids] = center[ids]
    state.initial_rotation[ids] = rotation[ids]
    state.previous_joint_velocity[ids] = env.scene["robot"].data.joint_vel[ids][:, state.joint_ids]
    state.previous_wrench[ids] = state.wrench_bias[ids] = 0
    state.initial_sensor_point[ids] = env.scene["robot"].data.body_link_pos_w[ids, state.sensor_id]
    state.path_length[ids] = 0
    state.stable_count[ids] = 0
    state.success[ids] = state.failure[ids] = state.terminal_paid[ids] = False
    state.discrete_reward[ids] = 0
    state.components[ids] = state.motion_costs[ids] = 0
    state.blocked_ratio[ids] = state.total_overlap[ids] = 0
    state.failure_reasons[ids] = False
    state.history[ids] = 0
    state.history_valid[ids] = False
    state.history[ids, 0] = _dynamic_observation(env, state, zero_action=True)[ids]
    state.history_valid[ids, 0] = True


def update_mdp(env):
    """Evaluate every completed transition once, regardless of manager call order."""
    from .observations import corrected_wrench, _dynamic_observation
    from .rewards import reward_components
    from .terminations import task_conditions, terminal_transition

    state = ensure_mdp(env)
    timestamp = float(env.sim.current_time)
    advance = timestamp > state.last_time + 1.0e-9
    if not bool(advance.any()):
        return state
    cfg, dt = env.cfg.mdp, float(env.step_dt)
    ids, active, center, corners, rotation, velocity = _physical_objects(env, state)
    path = state.path_length + (torch.linalg.vector_norm(center[..., :2] - state.previous_center[..., :2], dim=-1)
                                * active).sum(-1)
    polygon, count = convex_hull_xy(corners[..., :2])
    area = polygon_area(polygon, count)
    overlap = rectangle_overlap_area(polygon, count, env.access_region[:, None].expand(-1, cfg.max_objects, -1))
    blocker = ids == env.blocker_object_id[:, None]
    target = ids == env.target_object_id[:, None]
    blocker_area = (area * blocker * active).sum(-1)
    blocked = (overlap * blocker * active).sum(-1)
    total_overlap = (overlap * ~target * active).sum(-1)
    sensor_wrench = corrected_wrench(env)
    fixed_wrench = _fixed_wrench(env, state, sensor_wrench)
    joint_velocity = env.scene["robot"].data.joint_vel[:, state.joint_ids]
    components, costs = reward_components(cfg, blocked, blocker_area, path, joint_velocity,
                                          state.previous_joint_velocity, fixed_wrench, state.previous_wrench, dt)
    success_candidate, stable_count, reasons, failure = task_conditions(
        env, state, center, corners, rotation, velocity, active, target, total_overlap, sensor_wrench,
    )
    success, terminal, paid = terminal_transition(success_candidate, failure, stable_count,
                                                   state.terminal_paid, cfg)
    weighted = components @ torch.tensor((cfg.w_clear, cfg.w_rearr, cfg.w_motion, cfg.w_time), device=env.device)
    discrete = weighted + terminal
    selected = advance.nonzero(as_tuple=False).flatten()
    dynamic = _dynamic_observation(env, state, sensor_wrench)
    state.history[selected, 1:] = state.history[selected, :-1].clone()
    state.history_valid[selected, 1:] = state.history_valid[selected, :-1].clone()
    state.history[selected, 0] = dynamic[selected]
    state.history_valid[selected, 0] = True
    state.previous_center[selected] = center[selected]
    state.previous_joint_velocity[selected] = joint_velocity[selected]
    state.previous_wrench[selected] = fixed_wrench[selected]
    state.path_length[selected] = path[selected]
    state.stable_count[selected] = stable_count[selected]
    state.success[selected] = success[selected]
    state.failure[selected] = failure[selected]
    state.terminal_paid[selected] = paid[selected]
    state.discrete_reward[selected] = discrete[selected]
    state.components[selected] = torch.cat((components, terminal[:, None]), -1)[selected]
    state.motion_costs[selected] = costs[selected]
    state.blocked_ratio[selected] = (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0, 1)[selected]
    state.total_overlap[selected] = total_overlap[selected]
    state.failure_reasons[selected] = reasons[selected]
    state.last_time[selected] = timestamp
    return state

