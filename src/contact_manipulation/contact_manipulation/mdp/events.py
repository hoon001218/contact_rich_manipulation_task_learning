"""Episode/reset bookkeeping shared by native observation/reward/done terms.

Pure tensor calculation comes first, followed by native scene-data adapters.
One batched update per common_step_counter keeps history and path integrals
independent of manager call order. Only selected rows change on partial reset.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import math
import random
from itertools import combinations
from typing import Sequence
from types import SimpleNamespace

import numpy as np

import torch
from numba import njit
import warp as wp

from .observations import (
    SI_200_8_RANGES, SI_200_8_RESOLUTION, calibrate_wrench, compensate_wrench,
    dynamic_sample, geometry_in_eef_frame, pack_policy_observation, prepare_initial_geometry,
    quat_apply, quat_conjugate, quat_mul, self_motion_wrench,
)
from .rewards import corner_path_increment, reward_components
from .terminations import task_conditions, terminal_transition


@dataclass
class MdpCfg:
    max_objects: int = 40
    clearance_decay: float = 1.0
    rearrangement_decay: float = 2.0
    motion_acceleration_decay: float = 1.0
    motion_wrench_increase_decay: float = 1.0
    time_penalty_scale: float = 0.02
    time_penalty_growth_rate: float = 0.7
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
    tactile_threshold: float = 0.01

    def __post_init__(self):
        if self.max_objects != 40:
            raise ValueError("The agreed observation contract reserves 40 object slots.")
        if not isinstance(self.stable_steps, int) or self.stable_steps < 1:
            raise ValueError("stable_steps must be a positive integer.")
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            if (name.endswith("_ref") or name.endswith("_scale") or name == "time_penalty_growth_rate"):
                if name != "time_penalty_scale" and value <= 0:
                    raise ValueError(f"{name} must be positive.")


@dataclass
class TransitionBatch:
    """Completed physics/control sample; all tensors already on the environment device."""

    center: torch.Tensor               # N,O,3; fixed environment coordinates
    corners: torch.Tensor              # N,O,8,3; fixed corner correspondence
    rotation: torch.Tensor             # N,O,4; relative to canonical upright
    velocity: torch.Tensor             # N,O,6; linear velocity at the OBB center
    joint_velocity: torch.Tensor       # N,12; six arm then six independent hand joints
    sensor_wrench: torch.Tensor        # N,6; SI-200-8 measured, self-load compensated
    sensor_position: torch.Tensor      # N,3; world
    sensor_rotation: torch.Tensor      # N,4; world, wxyz
    sensor_range_exceeded: torch.Tensor  # N,6; before clipping/compensation
    forbidden_contact: torch.Tensor    # N; partner-resolved sensor result
    state_valid: torch.Tensor          # N; robot and sensor finite/ready
    sample: torch.Tensor               # N,72; current robot/sensor/action observation
    tcp_pose_b: torch.Tensor            # N,7; measured EEF in robot base


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _compact(vertices, keep):
    indices = (keep.long().cumsum(-1) - 1).clamp_min(0)
    result = torch.zeros_like(vertices)
    result.scatter_add_(-2, indices[..., None].expand_as(vertices), torch.where(keep[..., None], vertices, 0.0))
    return result, keep.sum(-1)


def convex_hull_xy(points):
    """Exact projected OBB hull, including roll/pitch; eight fixed tensor passes.

    These loops are over eight corners, never over environments or objects.
    Padding keeps allocations rectangular and avoids GPU nonzero/host reads.
    """
    size = points.shape[-2]
    delta = points - points.mean(-2, keepdim=True)
    order = torch.atan2(delta[..., 1], delta[..., 0]).argsort(dim=-1, stable=True)
    polygon = points.gather(-2, order[..., None].expand_as(points))
    count = torch.full(points.shape[:-2], size, device=points.device, dtype=torch.long)
    slots = torch.arange(size, device=points.device)
    for _ in range(size):
        modulus = count.clamp_min(1)[..., None]
        before = polygon.gather(-2, (((slots - 1) % modulus)[..., None]).expand_as(polygon))
        after = polygon.gather(-2, (((slots + 1) % modulus)[..., None]).expand_as(polygon))
        keep = ((slots < count[..., None])
                & (_cross2(polygon - before, after - polygon) >= -1.0e-10)
                & ((polygon - before).square().sum(-1) >= 1.0e-16))
        polygon, count = _compact(polygon, keep)
    return polygon, count


def polygon_area(vertices, count):
    # Shift before the shoelace sum to avoid cancellation for translated scenes.
    vertices = vertices - vertices[..., :1, :]
    slots = torch.arange(vertices.shape[-2], device=vertices.device)
    following = (slots + 1) % count.clamp_min(1)[..., None]
    after = vertices.gather(-2, following[..., None].expand_as(vertices))
    return (_cross2(vertices, after) * (slots < count[..., None])).sum(-1).abs() * 0.5


def rectangle_overlap_area(vertices, count, region):
    """Batched convex polygon clipping against [xmin,xmax,ymin,ymax]."""
    capacity = vertices.shape[-2] + 4
    polygon = torch.nn.functional.pad(vertices, (0, 0, 0, 4))
    slots = torch.arange(capacity, device=vertices.device)
    for axis, boundary, sign in ((0, 0, 1), (0, 1, -1), (1, 2, 1), (1, 3, -1)):
        previous = (slots - 1) % count.clamp_min(1)[..., None]
        before = polygon.gather(-2, previous[..., None].expand_as(polygon))
        limit = region[..., boundary, None]
        inside = (polygon[..., axis] - limit) * sign >= -1.0e-9
        before_inside = (before[..., axis] - limit) * sign >= -1.0e-9
        denominator = polygon[..., axis] - before[..., axis]
        safe = torch.where(denominator.abs() > 1.0e-12, denominator, 1.0)
        fraction = ((limit - before[..., axis]) / safe).clamp(0, 1)
        intersection = before + fraction[..., None] * (polygon - before)
        candidates = torch.stack((intersection, polygon), -2).flatten(-3, -2)
        valid = slots < count[..., None]
        keep = torch.stack((valid & (inside != before_inside), valid & inside), -1).flatten(-2)
        compacted, count = _compact(candidates, keep)
        polygon = compacted[..., :capacity, :]
    return polygon_area(polygon, count)


def object_geometry(pose, local_center, local_corners, upright):
    """Transform cached upright catalog bounds, accounting for off-center USD origins."""
    rotation = quat_mul(pose[..., 3:7], quat_conjugate(upright))
    center = pose[..., :3] + quat_apply(rotation, local_center)
    corners = pose[..., None, :3] + quat_apply(rotation[..., None, :].expand(*local_corners.shape[:-1], 4), local_corners)
    return center, corners, rotation


class TaskState:
    """Preallocated batch buffers; geometry is vision-only, physical poses reward-only."""

    def __init__(self, num_envs, device, cfg, dt, *, dtype=torch.float32, step=0):
        cfg.__post_init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("step_dt must be finite and positive.")
        self.cfg, self.dt = cfg, dt
        self.last_step = step

        self.ready = False
        self.bindings = None
        self.observation = None
        self.empty_env_ids = torch.empty(0, dtype=torch.long, device=device)
        n, o = num_envs, cfg.max_objects
        def zeros(*shape, kind=dtype):
            return torch.zeros(shape, device=device, dtype=kind)
        self.geometry_b = zeros(n, o, 10)
        self.geometry_valid = zeros(n, o, kind=torch.bool)
        self.tcp_pose_b = zeros(n, 7)
        self.history = zeros(n, 2, 72)
        self.active = zeros(n, o, kind=torch.bool)
        self.target = zeros(n, o, kind=torch.bool)
        self.blocker = zeros(n, o, kind=torch.bool)
        self.access_region = zeros(n, 4)
        self.workspace_bounds = zeros(n, 4)
        self.shelf_top = zeros(n)
        self.ceiling = zeros(n)
        self.initial_center = zeros(n, o, 3)
        self.initial_rotation = zeros(n, o, 4)
        self.previous_corners = zeros(n, o, 8, 3)
        self.previous_joint_velocity = zeros(n, 12)
        self.previous_wrench = zeros(n, 6)
        self.wrench_bias = zeros(n, 6)
        self.initial_sensor_position = zeros(n, 3)
        self.path_length = zeros(n)
        self.elapsed_time = zeros(n, kind=torch.float64)
        self.stable_count = zeros(n, kind=torch.long)
        self.success = zeros(n, kind=torch.bool)
        self.failure = zeros(n, kind=torch.bool)
        self.terminal_paid = zeros(n, kind=torch.bool)
        self.failure_reasons = zeros(n, 7, kind=torch.bool)
        self.components = zeros(n, 5)
        self.motion_costs = zeros(n, 2)
        self.rearrangement_increment = zeros(n)
        self.total_overlap = zeros(n)

    @torch.no_grad()
    def reset(self, ids, batch, *, geometry_b, geometry_valid, active, target, blocker,
              access_region, workspace_bounds, shelf_top, ceiling, step, batch_is_selected=False):
        """Commit selected rows after physical reset and accepted initial vision.

        By default all inputs have a full-environment leading dimension. With
        batch_is_selected=True, the TransitionBatch and reset metadata contain
        only selected rows, in exactly the order of ids. This avoids computing
        or cloning unchanged environments during native partial resets.
        No physics steps, renders, or environment-wide reset occur here.
        """
        if not self.ready and ids.numel() != self.history.shape[0]:
            raise ValueError("The first reset must initialize every environment.")
        if self.ready and step != self.last_step:
            raise RuntimeError("Evaluate the completed transition before resetting episode state.")
        rows = slice(None) if batch_is_selected else ids
        for name, value in (("geometry_b", geometry_b), ("geometry_valid", geometry_valid),
                            ("active", active), ("target", target), ("blocker", blocker),
                            ("access_region", access_region), ("workspace_bounds", workspace_bounds),
                            ("shelf_top", shelf_top), ("ceiling", ceiling)):
            getattr(self, name)[ids] = value[rows]
        self.initial_center[ids] = batch.center[rows]
        self.initial_rotation[ids] = batch.rotation[rows]
        self.previous_corners[ids] = batch.corners[rows]
        self.previous_joint_velocity[ids] = batch.joint_velocity[rows]
        self.initial_sensor_position[ids] = batch.sensor_position[rows]
        self.tcp_pose_b[ids] = batch.tcp_pose_b[rows]
        self.history[ids] = 0
        self.history[ids, 0] = batch.sample[rows]
        # The reset state has no completed action/contact sample in this episode.
        self.history[ids, 0, 31:37] = 0
        self.history[ids, 0, 37:54] = -1
        self.history[ids, 0, 54:72] = 0
        for name in ("previous_wrench", "wrench_bias", "path_length", "elapsed_time", "stable_count",
                     "success", "failure", "terminal_paid", "failure_reasons", "components",
                     "motion_costs", "rearrangement_increment", "total_overlap"):
            getattr(self, name)[ids] = 0
        self.ready = True
        # Build a new tensor: extras may still reference the previous episode's
        # final observation. Unreset rows retain exactly the same measurements.
        if self.observation is None:
            geometry = geometry_in_eef_frame(self.geometry_b, self.geometry_valid, self.tcp_pose_b)
            self.observation = pack_policy_observation(geometry, self.history)
        else:
            geometry = geometry_in_eef_frame(self.geometry_b[ids], self.geometry_valid[ids], self.tcp_pose_b[ids])
            observation = self.observation.clone()
            observation[ids] = pack_policy_observation(geometry, self.history[ids])
            self.observation = observation
        self.last_step = step

    @torch.no_grad()
    def advance(self, batch, step):
        if step == self.last_step:
            return
        if not self.ready:
            raise RuntimeError("reset_task must run after scene/vision initialization before stepping.")
        if step != self.last_step + 1:
            raise RuntimeError("MDP must be evaluated once every completed control transition.")
        safe_corners = torch.nan_to_num(batch.corners, nan=0.0, posinf=0.0, neginf=0.0)
        polygon, count = convex_hull_xy(safe_corners[..., :2])
        overlap = rectangle_overlap_area(polygon, count, self.access_region[:, None, :])
        areas = polygon_area(polygon, count)
        blocked = torch.where(self.blocker & self.active, overlap, 0.0).sum(-1)
        blocker_area = torch.where(self.blocker & self.active, areas, 0.0).sum(-1)
        self.total_overlap.copy_(torch.where(self.active & ~self.target, overlap, 0.0).sum(-1))
        increment = corner_path_increment(self.previous_corners, batch.corners, self.active)
        self.rearrangement_increment.copy_(torch.nan_to_num(increment, nan=0.0, posinf=0.0, neginf=0.0))
        self.path_length.add_(self.rearrangement_increment)
        force = quat_apply(batch.sensor_rotation, batch.sensor_wrench[:, :3])
        moment = (quat_apply(batch.sensor_rotation, batch.sensor_wrench[:, 3:])
                  + torch.cross(batch.sensor_position - self.initial_sensor_position, force, dim=-1))
        fixed_wrench = torch.cat((force, moment), -1)
        components, costs = reward_components(
            self.cfg, blocked, blocker_area, self.path_length, batch.joint_velocity,
            self.previous_joint_velocity, fixed_wrench, self.previous_wrench, self.dt,
            elapsed_time=self.elapsed_time,
        )
        candidate, stable, reasons, failed = task_conditions(
            self.cfg, center=batch.center, corners=batch.corners, rotation=batch.rotation,
            velocity=batch.velocity, active=self.active, target=self.target,
            initial_center=self.initial_center, initial_rotation=self.initial_rotation,
            total_overlap=self.total_overlap, sensor_wrench=batch.sensor_wrench,
            stable_count=self.stable_count, workspace_bounds=self.workspace_bounds,
            shelf_top=self.shelf_top, ceiling=self.ceiling,
            sensor_range_exceeded=batch.sensor_range_exceeded, forbidden_contact=batch.forbidden_contact,
            state_valid=batch.state_valid,
        )
        successful, payment, paid = terminal_transition(candidate, failed, stable, self.terminal_paid, self.cfg)
        self.components[:, :4] = components
        self.components[:, 4] = payment
        self.motion_costs.copy_(costs)
        self.stable_count.copy_(stable)
        self.success.copy_(successful)
        self.failure.copy_(failed)
        self.terminal_paid.copy_(paid)
        self.failure_reasons.copy_(reasons)
        self.history[:, 1].copy_(self.history[:, 0])
        self.history[:, 0].copy_(torch.nan_to_num(batch.sample, nan=0.0, posinf=0.0, neginf=0.0))
        self.tcp_pose_b.copy_(torch.nan_to_num(batch.tcp_pose_b, nan=0.0, posinf=0.0, neginf=0.0))
        self.previous_corners.copy_(batch.corners)
        self.previous_joint_velocity.copy_(batch.joint_velocity)
        self.previous_wrench.copy_(fixed_wrench)
        self.elapsed_time.add_(self.dt)
        self.observation = None
        self.last_step = step


@dataclass
class SceneBatch:
    """Reset producer contract, filled by independent episode/initial-vision events.

    Positions/bounds use fixed environment coordinates except geometry_b (robot
    base). Object IDs index scene['object_collection']; -1 means inactive or
    unobserved. Catalog bounds are measured in canonical upright coordinates.
    Per-episode values must stay valid until reset_task commits the selected rows.
    """

    object_ids: torch.Tensor             # N,40
    target_id: torch.Tensor              # N
    blocker_id: torch.Tensor             # N
    geometry_b: torch.Tensor             # N,40,10; accepted initial vision ONLY
    geometry_valid: torch.Tensor         # N,40
    observed_ids: torch.Tensor           # N,40; association to collection IDs
    lower: torch.Tensor                  # collection_size,3
    upper: torch.Tensor                  # collection_size,3
    upright: torch.Tensor                # collection_size,4 (wxyz)
    access_region: torch.Tensor          # N,4: xmin,xmax,ymin,ymax
    workspace_bounds: torch.Tensor       # N,4
    shelf_top: torch.Tensor              # N
    ceiling: torch.Tensor                # N
    sensor_body_name: str
    distal_body_names: tuple[str, ...]    # complete hand + distal half-spacer


@dataclass
class SensorBatch:
    """Latest completed physics sample, supplied by env.task_sensor_reader().

    The reader binds the native articulation and contact sensors. Ideal
    wrench is parent-on-child joint reaction at the sensor center, sensor axes.
    Forbidden contact must be aggregated per partner, never vector-summed across
    different contact partners; see forbidden_contact_from_history below.
    """

    ideal_wrench: torch.Tensor           # N,6; before SI-200-8 measurement model
    wrench_valid: torch.Tensor           # N
    tactile_force: torch.Tensor          # N,17,3; latest net normal forces
    tactile_valid: torch.Tensor          # N,17
    forbidden_contact: torch.Tensor      # N; arm/spacer only, hand excluded


def forbidden_contact_from_history(forces, threshold=1.0):
    """Any body/partner in the recent substeps exceeds 1 N; opposing contacts cannot cancel."""
    if forces.ndim != 5 or forces.shape[-1] != 3:
        raise ValueError("Expected contact force history [N, substeps, bodies, partners, 3].")
    return ((torch.linalg.vector_norm(forces, dim=-1) > threshold).any(dim=(1, 2, 3))
            | ~torch.isfinite(forces).all(dim=(1, 2, 3, 4)))


def _ensure_state(env):
    if not hasattr(env, "_contact_mdp"):
        # ObservationManager probes output dimensions before startup/reset.
        # No scene ground truth is substituted for missing initial vision.
        env._contact_mdp = TaskState(env.num_envs, env.device, env.cfg.mdp, float(env.step_dt),
                                     step=env.common_step_counter)
    return env._contact_mdp


def _bind_scene(env, state, scene):
    """Resolve native asset IDs and constant device tensors only once."""
    from types import SimpleNamespace

    robot = env.scene["robot"]
    action = env.action_manager.get_term("manipulation")
    names = robot.body_names
    if not scene.distal_body_names or scene.sensor_body_name not in scene.distal_body_names:
        raise ValueError("Supply the complete distal hand/spacer body list, including the F/T body.")
    distal = [names.index(name) for name in scene.distal_body_names]
    if len(set(distal)) != len(distal):
        raise ValueError("F/T compensation bodies must not be duplicated.")
    joint_ids = torch.tensor(action.arm_joint_ids + action.hand_joint_ids, device=env.device)
    distal_ids = torch.tensor(distal, device=env.device)
    masses = robot.root_physx_view.get_masses().to(env.device).index_select(1, distal_ids)
    inertias = robot.root_physx_view.get_inertias().to(env.device).index_select(1, distal_ids)
    # PhysX exposes COM inertia in link axes and column-major flattened order.
    inertias = inertias.reshape(env.num_envs, len(distal), 3, 3).transpose(-1, -2).contiguous()
    n, o = env.num_envs, state.cfg.max_objects
    state.bindings = SimpleNamespace(
        robot=robot, objects=env.scene["object_collection"], action=action,
        joint_ids=joint_ids, distal_ids=distal_ids, tcp_id=action.tcp_body_id,
        sensor_id=names.index(scene.sensor_body_name), masses=masses, inertias=inertias,
        gravity=torch.tensor(env.sim.cfg.gravity, device=env.device),
        ranges=torch.tensor(SI_200_8_RANGES, device=env.device),
        resolution=torch.tensor(SI_200_8_RESOLUTION, device=env.device),
        object_ids=torch.zeros(n, o, dtype=torch.long, device=env.device),
        local_center=torch.zeros(n, o, 3, device=env.device),
        local_corners=torch.zeros(n, o, 8, 3, device=env.device),
        upright=torch.zeros(n, o, 4, device=env.device),
    )


def _validate_reset_data(env, scene, ids):
    """Validate producer contracts only at reset, never in the rollout hot path."""
    n, o = env.num_envs, env.cfg.mdp.max_objects
    device = env._contact_mdp.history.device
    for name, shape, dtype in (
        ("object_ids", (n, o), torch.long), ("target_id", (n,), torch.long),
        ("blocker_id", (n,), torch.long), ("observed_ids", (n, o), torch.long),
        ("geometry_valid", (n, o), torch.bool), ("geometry_b", (n, o, 10), None),
        ("access_region", (n, 4), None), ("workspace_bounds", (n, 4), None),
        ("shelf_top", (n,), None), ("ceiling", (n,), None),
    ):
        value = getattr(scene, name)
        if value.shape != shape or value.device != device or (dtype is not None and value.dtype != dtype):
            raise ValueError(f"SceneBatch.{name} requires shape {shape}, dtype {dtype}, device {device}.")
    for name, width in (("lower", 3), ("upper", 3), ("upright", 4)):
        value = getattr(scene, name)
        if value.ndim != 2 or value.shape[-1] != width or value.device != device:
            raise ValueError(f"SceneBatch.{name} must be a catalog tensor on {device}.")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"SceneBatch.{name} contains nonfinite values.")
    if (scene.lower.shape != scene.upper.shape or scene.upright.shape[0] != scene.lower.shape[0]
            or scene.lower.shape[0] != env.scene["object_collection"].data.object_link_pose_w.shape[1]):
        raise ValueError("Catalog bounds/upright must match the native object collection size.")
    if bool((scene.upper <= scene.lower).any() | ((scene.upright.norm(dim=-1) - 1).abs() > 1.e-3).any()):
        raise ValueError("Catalog bounds must have positive extents and upright unit quaternions.")
    for name in ("access_region", "workspace_bounds"):
        bounds = getattr(scene, name)[ids]
        if not bool(torch.isfinite(bounds).all() & (bounds[:, 0] < bounds[:, 1]).all()
                    & (bounds[:, 2] < bounds[:, 3]).all()):
            raise ValueError(f"SceneBatch.{name} must be finite and ordered.")
    if not bool((torch.isfinite(scene.shelf_top[ids]) & torch.isfinite(scene.ceiling[ids])
                 & (scene.shelf_top[ids] < scene.ceiling[ids])).all()):
        raise ValueError("Shelf/ceiling heights must be finite and ordered.")
    observed, active_ids = scene.observed_ids[ids], scene.object_ids[ids]
    valid = scene.geometry_valid[ids]
    belongs = ((observed[..., None] == active_ids[:, None, :]) & (active_ids[:, None, :] >= 0)).any(-1)
    sorted_observed = torch.where(valid, observed, -1).sort(dim=-1).values
    if bool((valid & ~belongs).any() | ((sorted_observed[:, 1:] == sorted_observed[:, :-1])
                                      & (sorted_observed[:, 1:] >= 0)).any()):
        raise ValueError("Valid initial observations must associate to distinct active objects.")
    sensors = env.task_sensor_reader()
    if not isinstance(sensors, SensorBatch):
        raise ValueError("task_sensor_reader must return SensorBatch.")
    for name, shape, dtype in (("ideal_wrench", (n, 6), None), ("wrench_valid", (n,), torch.bool),
                               ("tactile_force", (n, 17, 3), None), ("tactile_valid", (n, 17), torch.bool),
                               ("forbidden_contact", (n,), torch.bool)):
        value = getattr(sensors, name)
        if value.shape != shape or value.device != device or (dtype is not None and value.dtype != dtype):
            raise ValueError(f"SensorBatch.{name} requires shape {shape}, dtype {dtype}, device {device}.")


@torch.no_grad()
def _read_transition(env, state, env_ids=None):
    """Read all rollout rows, or only reset rows before expensive geometry/F/T work."""
    b = state.bindings
    data = b.robot.data
    rows = slice(None) if env_ids is None else env_ids
    ids = b.object_ids[rows]
    origins = env.scene.env_origins[rows, None, :]
    objects = b.objects.data
    pose = objects.object_link_pose_w[rows].gather(1, ids[..., None].expand(-1, -1, 7))
    pose[..., :3] -= origins
    center, corners, rotation = object_geometry(pose, b.local_center[rows], b.local_corners[rows], b.upright[rows])
    com = objects.object_com_pos_w[rows].gather(1, ids[..., None].expand(-1, -1, 3)) - origins
    velocity = objects.object_com_vel_w[rows].gather(1, ids[..., None].expand(-1, -1, 6))
    velocity[..., :3] += torch.cross(velocity[..., 3:], center - com, dim=-1)

    body_position, body_rotation = data.body_link_pos_w[rows], data.body_link_quat_w[rows]
    sensor_position = body_position[:, b.sensor_id]
    sensor_rotation = body_rotation[:, b.sensor_id]
    sensors = env.task_sensor_reader()
    raw_wrench, wrench_valid = sensors.ideal_wrench[rows], sensors.wrench_valid[rows]
    tactile_force = sensors.tactile_force[rows]
    measured, overrange = calibrate_wrench(raw_wrench, b.ranges, b.resolution)
    self_load = self_motion_wrench(
        b.masses[rows], b.inertias[rows], data.body_com_pos_w[rows].index_select(1, b.distal_ids),
        body_rotation.index_select(1, b.distal_ids), data.body_com_acc_w[rows].index_select(1, b.distal_ids),
        data.body_com_ang_vel_w[rows].index_select(1, b.distal_ids), sensor_position, sensor_rotation, b.gravity,
    )
    wrench = compensate_wrench(measured, self_load, state.wrench_bias[rows], wrench_valid)
    inverse_root = quat_conjugate(data.root_quat_w[rows])
    position = quat_apply(inverse_root, body_position[:, b.tcp_id] - data.root_pos_w[rows])
    orientation = quat_mul(inverse_root, body_rotation[:, b.tcp_id])
    orientation = torch.where(orientation[:, :1] < 0, -orientation, orientation)
    tcp_pose = torch.cat((position, orientation), -1)
    joint_pos = data.joint_pos[rows].index_select(1, b.joint_ids)
    joint_vel = data.joint_vel[rows].index_select(1, b.joint_ids)
    tactile_valid = sensors.tactile_valid[rows] & torch.isfinite(tactile_force).all(-1)
    sample = dynamic_sample(
        joint_pos, joint_vel, tcp_pose, wrench,
        torch.linalg.vector_norm(tactile_force, dim=-1) > state.cfg.tactile_threshold,
        tactile_valid, b.action.normalized_actions[rows], joint_position_scale=state.cfg.joint_position_scale,
        joint_velocity_scale=state.cfg.joint_velocity_scale, wrench_ranges=b.ranges,
    )
    valid = (torch.isfinite(sample).all(-1) & torch.isfinite(raw_wrench).all(-1)
             & wrench_valid & ~b.action.invalid_control[rows])
    return TransitionBatch(center, corners, rotation, velocity, joint_vel, wrench, sensor_position,
                           sensor_rotation, overrange & wrench_valid[:, None],
                           sensors.forbidden_contact[rows], valid, sample, tcp_pose)


@torch.no_grad()
def reset_task(env, env_ids=None):
    """Native reset event, ordered AFTER placement/robot reset and vision acceptance.

    The scene event provides env.task_scene_data: SceneBatch and binds
    env.task_sensor_reader. This event commits initial observations and resets
    episode accumulators; it does not invent placement, vision, or sensor values.
    """
    scene = getattr(env, "task_scene_data", None)
    if not isinstance(scene, SceneBatch) or not callable(getattr(env, "task_sensor_reader", None)):
        raise RuntimeError("Scene reset must supply SceneBatch and task_sensor_reader before reset_task.")
    state = _ensure_state(env)
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, device=env.device, dtype=torch.long))
    if ids.numel() == 0:
        return
    if ids.ndim != 1 or bool(((ids < 0) | (ids >= env.num_envs)).any()) or ids.unique().numel() != ids.numel():
        raise ValueError("Reset IDs must be distinct environment indices.")
    _validate_reset_data(env, scene, ids)
    if state.bindings is None:
        _bind_scene(env, state, scene)
    b = state.bindings
    if scene.object_ids.shape != (env.num_envs, state.cfg.max_objects) or scene.object_ids.dtype != torch.long:
        raise ValueError("Scene object_ids must be int64 [num_envs,40].")
    object_ids = scene.object_ids[ids]
    active = object_ids >= 0
    target = active & (object_ids == scene.target_id[ids, None])
    blocker = active & (object_ids == scene.blocker_id[ids, None])
    if bool(((target.sum(-1) != 1) | (blocker.sum(-1) != 1) | (target & blocker).any(-1)).any()):
        raise ValueError("Each episode requires one distinct active target and blocker.")
    sorted_ids = object_ids.sort(dim=-1).values
    if bool(((sorted_ids[:, 1:] == sorted_ids[:, :-1]) & (sorted_ids[:, 1:] >= 0)).any()):
        raise ValueError("Active collection IDs must be unique; repeated object TYPES use different bodies.")
    if bool((object_ids >= scene.lower.shape[0]).any()):
        raise ValueError("Object IDs exceed catalog bounds.")
    selected = object_ids.clamp_min(0)
    b.object_ids[ids] = selected
    lower, upper = scene.lower[selected], scene.upper[selected]
    center_local = 0.5 * (lower + upper)
    signs = lower.new_tensor(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                              (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)))
    b.local_center[ids] = center_local
    b.local_corners[ids] = center_local[..., None, :] + 0.5 * (upper - lower)[..., None, :] * signs
    b.upright[ids] = scene.upright[selected]
    observed_ids, geometry_valid = scene.observed_ids[ids], scene.geometry_valid[ids]
    observed_target = (observed_ids == scene.target_id[ids, None]) & geometry_valid
    observed_blocker = (observed_ids == scene.blocker_id[ids, None]) & geometry_valid
    role_indices = torch.stack((observed_target.long().argmax(-1), observed_blocker.long().argmax(-1)), -1)
    role_indices = torch.where(torch.stack((observed_target.any(-1), observed_blocker.any(-1)), -1), role_indices, -1)
    geometry, valid = prepare_initial_geometry(scene.geometry_b[ids], geometry_valid,
                                               role_indices[:, 0], role_indices[:, 1])
    # Only this reset batch is sorted/validated. Unreset initial vision remains frozen.
    initial = _read_transition(env, state, ids)
    if not bool(torch.isfinite(initial.sample[:, :31]).all()
                & (initial.tcp_pose_b[:, 3:7].norm(dim=-1) > 1.e-8).all()):
        raise ValueError("Reset robot joint/EEF measurements must be finite with a valid quaternion.")
    state.reset(ids, initial, geometry_b=geometry, geometry_valid=valid,
                active=active, target=target, blocker=blocker, access_region=scene.access_region[ids],
                workspace_bounds=scene.workspace_bounds[ids], shelf_top=scene.shelf_top[ids],
                ceiling=scene.ceiling[ids], step=env.common_step_counter, batch_is_selected=True)
    if "terminal_observation" in env.extras:
        # Reuse IDs already computed by native auto-reset; the learner can skip
        # timeout value inference without a second GPU nonzero()/scalar read.
        env.extras["terminal_env_ids"] = ids


@torch.no_grad()
def get_task_state(env):
    """One calculation per native control counter; no tensor scalar synchronization."""
    state = _ensure_state(env)
    if env.common_step_counter != state.last_step:
        if not state.ready:
            raise RuntimeError("reset_task must initialize the scene before stepping.")
        state.advance(_read_transition(env, state), env.common_step_counter)
        # Native env resets after reward/done. Preserve the final observation
        # here for the learner's timeout bootstrap, before any reset.
        geometry = geometry_in_eef_frame(state.geometry_b, state.geometry_valid, state.tcp_pose_b)
        state.observation = pack_policy_observation(geometry, state.history)
        env.extras["terminal_observation"] = state.observation
        env.extras["terminated"] = (state.success | state.failure).clone()
        env.extras["terminal_env_ids"] = state.empty_env_ids
    return state


# Shared layout preparation: bounded CPU geometry/IK search, independent of env count.

Point = tuple[float, float]
Point3 = tuple[float, float, float]
Polygon = tuple[Point, ...]
Region = tuple[float, float, float, float]
_EPS = 1.0e-9

# Columns are the palm TCP's local axes in world coordinates. Its +Y is the
# finger/tool forward axis: point it toward the shelf (-X). The hand's -90deg
# roll about tool +Z puts palm +Z toward world -Y and palm +X toward world +Z.
ROBOT_START_TCP_ROTATION = (
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
    (1.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class ObjectGeometry:
    name: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    asset_name: str | None = None


@dataclass
class SpawnRules:
    x_range: tuple[float, float] = (-0.89, -0.51)
    y_range: tuple[float, float] = (-0.455, 0.455)
    shelf_top: float = 1.05
    ceiling: float = 1.35
    min_objects: int = 4
    max_objects: int = 40
    min_asset_types: int = 4
    max_asset_types: int = 10
    all_asset_types: bool = False
    edge_margin: float = 0.015
    gap: float = 0.003
    hand_width: float = 0.10
    approach_margin: float = 0.01
    min_blocked_ratio: float = 0.30
    min_x_coverage: float = 0.55
    min_y_coverage: float = 0.65
    neighbor_distance: float = 0.04
    min_neighbor_fraction: float = 0.75
    min_shelf_occupancy: float = 0.35
    region_grid: tuple[int, int] = (2, 3)
    min_region_occupancy: float = 0.10
    reserve_blocker_front: bool = True
    max_attempts: int = 300


@dataclass
class RobotStartRules:
    """Bounded, collision-checked hand starts in front of the episode Blocker.

    Distances describe the palm TCP, not the closest finger surface. Full robot
    collision geometry must be checked after IK, using ``collision_clearance``.
    """

    front_distance_range: tuple[float, float] = (0.12, 0.40)
    shelf_front_clearance: float = 0.06
    lateral_half_width: float = 0.08
    height_above_shelf: tuple[float, float] = (0.20, 0.28)
    min_elbow_height: float = 0.05
    # Shelf is along world -X: wrist must be forward of the elbow, not folded back.
    min_forearm_forward: float = 0.05
    collision_clearance: float = 0.015
    max_attempts: int = 80
    ik_batch_size: int = 1024
    enabled: bool = True


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PlanarMotion:
    """One supported SE(2) motion; ``pivot`` specifies an exact fixed-pivot arc.

    Without a pivot the bounding center interpolates linearly while yaw changes.
    A pivot rotates the entire footprint about that world XY point instead.
    """

    start: PlanarPose
    end: PlanarPose
    pivot: Point | None = None


@dataclass(frozen=True)
class SceneMetrics:
    x_coverage: float
    y_coverage: float
    shelf_occupancy: float
    local_neighbor_fraction: float
    region_occupancies: tuple[float, ...]


@dataclass(frozen=True)
class SceneSample:
    active_ids: tuple[int, ...]
    target_id: int
    blocker_id: int
    positions: dict[int, tuple[float, float, float]]
    yaws: dict[int, float]
    access_region: Region
    clearance_witness: Point
    clearance_path: tuple[PlanarMotion, ...] = ()
    clearance_kind: str = "translation"
    # Keep the packer's coordinates on device for IK and physics. CPU mappings
    # remain available for independent validation/debugging and role metadata.
    planar_poses: torch.Tensor | None = field(default=None, repr=False, compare=False)
    planar_ids: tuple[int, ...] = field(default=(), repr=False, compare=False)


def rectangle_polygon(region: Region) -> Polygon:
    """Return a counterclockwise polygon for (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = region
    return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))


def _blocker_front_region(polygon: Polygon, rules: SpawnRules) -> Region:
    """Keep non-role footprints out of a conservative frontal-view corridor.

    This improves front-camera proposals; it does not measure silhouette
    visibility. The rendered acceptance step must still check both roles.
    """
    return (max(x for x, _ in polygon), rules.x_range[1],
            min(y for _, y in polygon) - rules.gap, max(y for _, y in polygon) + rules.gap)


def footprint_polygon(geometry: ObjectGeometry, center_xy: Point, yaw: float) -> Polygon:
    """Rotate the conservative upright footprint about its own bounding center."""
    hx = 0.5 * (geometry.upper[0] - geometry.lower[0])
    hy = 0.5 * (geometry.upper[1] - geometry.lower[1])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return tuple(
        (center_xy[0] + cosine * x - sine * y, center_xy[1] + sine * x + cosine * y)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    )


def robot_start_region(
    sample: SceneSample,
    geometries: Sequence[ObjectGeometry],
    spawn_rules: SpawnRules,
    robot_start: RobotStartRules,
) -> tuple[float, float, float, float, float, float]:
    """Return local TCP bounds (xmin, xmax, ymin, ymax, zmin, zmax).

    The near edge satisfies both the rotated Blocker front offset and clearance
    from the shelf front. An empty intersection fails explicitly rather than
    moving the Blocker or silently relaxing the requested distances.
    """
    values = (
        *robot_start.front_distance_range, robot_start.shelf_front_clearance,
        robot_start.lateral_half_width, *robot_start.height_above_shelf,
        robot_start.min_elbow_height, robot_start.min_forearm_forward, robot_start.collision_clearance,
        *spawn_rules.x_range, spawn_rules.shelf_top,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Robot start rules and shelf bounds must be finite.")
    if any(value < 0.0 for value in (
        *robot_start.front_distance_range, robot_start.shelf_front_clearance,
        robot_start.lateral_half_width, *robot_start.height_above_shelf,
        robot_start.min_elbow_height, robot_start.min_forearm_forward, robot_start.collision_clearance,
    )):
        raise ValueError("Robot start distances, heights and clearances must be nonnegative.")
    if robot_start.front_distance_range[0] >= robot_start.front_distance_range[1]:
        raise ValueError("Robot start front distance range must have positive width.")
    if robot_start.height_above_shelf[0] >= robot_start.height_above_shelf[1]:
        raise ValueError("Robot start height range must have positive width.")
    if robot_start.lateral_half_width <= 0.0:
        raise ValueError("Robot start lateral half width must be positive.")
    if not isinstance(robot_start.max_attempts, int) or isinstance(robot_start.max_attempts, bool) or robot_start.max_attempts <= 0:
        raise ValueError("Robot start max_attempts must be a positive integer.")
    if spawn_rules.x_range[0] >= spawn_rules.x_range[1]:
        raise ValueError("Shelf X range must have positive width.")
    if not 0 <= sample.blocker_id < len(geometries) or sample.blocker_id not in sample.active_ids:
        raise ValueError("Robot start Blocker must index an active catalog object.")
    blocker = sample.blocker_id
    if blocker not in sample.positions or blocker not in sample.yaws:
        raise ValueError("Robot start requires the Blocker pose.")
    position, yaw = sample.positions[blocker], sample.yaws[blocker]
    geometry = geometries[blocker]
    if not all(math.isfinite(value) for value in (*position, yaw, *geometry.lower, *geometry.upper)):
        raise ValueError("Robot start Blocker pose and geometry must be finite.")
    if any(high <= low for low, high in zip(geometry.lower, geometry.upper)):
        raise ValueError("Robot start Blocker geometry must have positive extent.")
    front = max(x for x, _ in footprint_polygon(geometry, position[:2], yaw))
    minimum_x = max(front + robot_start.front_distance_range[0],
                    spawn_rules.x_range[1] + robot_start.shelf_front_clearance)
    maximum_x = front + robot_start.front_distance_range[1]
    if minimum_x >= maximum_x:
        raise ValueError("Robot start front region is empty after shelf clearance.")
    return (
        minimum_x, maximum_x,
        position[1] - robot_start.lateral_half_width,
        position[1] + robot_start.lateral_half_width,
        spawn_rules.shelf_top + robot_start.height_above_shelf[0],
        spawn_rules.shelf_top + robot_start.height_above_shelf[1],
    )


def elbow_above_chord(shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]) -> float:
    """Return elbow height above its closest point on the shoulder–wrist chord.

    Positive values identify the high-elbow branch geometrically, independent
    of joint angle signs, wrapping, and the robot's base orientation.
    """
    if any(len(point) != 3 for point in (shoulder, elbow, wrist)) or not all(
        math.isfinite(value) for point in (shoulder, elbow, wrist) for value in point
    ):
        raise ValueError("Shoulder, elbow and wrist must be finite three-dimensional points.")
    chord = tuple(wrist[index] - shoulder[index] for index in range(3))
    length_squared = sum(value * value for value in chord)
    if length_squared <= 1.0e-18:
        raise ValueError("Shoulder and wrist must define a nonzero chord.")
    fraction = sum((elbow[index] - shoulder[index]) * chord[index] for index in range(3)) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    return elbow[2] - shoulder[2] - fraction * chord[2]


def oriented_box_separated(
    center_a: Sequence[float],
    rotation_a: Sequence[Sequence[float]],
    half_a: Sequence[float],
    center_b: Sequence[float],
    rotation_b: Sequence[Sequence[float]],
    half_b: Sequence[float],
    clearance: float = 0.0,
) -> bool:
    """Test 3D box separation on the complete 15-axis SAT, with an AABB shortcut.

    Rotation matrices have the local unit axes in their columns. ``clearance``
    requires that gap along a normalized separating axis; it is conservative
    for diagonal corner gaps. Touching separates when clearance is zero.
    """
    if any(len(vector) != 3 for vector in (center_a, half_a, center_b, half_b)) or any(
        len(rotation) != 3 or any(len(row) != 3 for row in rotation)
        for rotation in (rotation_a, rotation_b)
    ):
        raise ValueError("Oriented boxes require 3D centers/extents and 3×3 rotations.")
    values = (
        *center_a, *half_a, *center_b, *half_b, clearance,
        *(value for rotation in (rotation_a, rotation_b) for row in rotation for value in row),
    )
    if not all(math.isfinite(value) for value in values) or any(value < 0.0 for value in (*half_a, *half_b, clearance)):
        raise ValueError("Oriented boxes must be finite, with nonnegative half extents and clearance.")
    delta = tuple(center_b[index] - center_a[index] for index in range(3))
    # A world-axis gap also separates the full boxes, even if it is not one of
    # their face/cross axes. This cheaply rejects most distant shelf colliders.
    for index in range(3):
        radius = sum(abs(rotation_a[index][axis]) * half_a[axis] +
                     abs(rotation_b[index][axis]) * half_b[axis] for axis in range(3))
        if abs(delta[index]) + _EPS >= radius + clearance:
            return True
    axes_a = tuple(tuple(rotation_a[row][column] for row in range(3)) for column in range(3))
    axes_b = tuple(tuple(rotation_b[row][column] for row in range(3)) for column in range(3))
    axes = (*axes_a, *axes_b, *(
        (first[1] * second[2] - first[2] * second[1],
         first[2] * second[0] - first[0] * second[2],
         first[0] * second[1] - first[1] * second[0])
        for first in axes_a for second in axes_b
    ))
    for axis in axes:
        length_squared = sum(value * value for value in axis)
        if length_squared <= 1.0e-18:
            continue  # Parallel box edges do not define a separating axis.
        length = math.sqrt(length_squared)
        distance = abs(sum(delta[index] * axis[index] for index in range(3)))
        radius = sum(half_a[index] * abs(sum(axes_a[index][j] * axis[j] for j in range(3))) +
                     half_b[index] * abs(sum(axes_b[index][j] * axis[j] for j in range(3)))
                     for index in range(3))
        if distance + _EPS * length >= radius + clearance * length:
            return True
    return False


def planar_polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) * 0.5


def polygons_separated(first: Polygon, second: Polygon, gap: float = 0.0) -> bool:
    """SAT separation with at least ``gap`` on one normalized edge axis.

    Requiring a gap on an edge axis is conservative for diagonal corner gaps.
    Boundary touching counts as separated when gap is zero.
    """
    for polygon in (first, second):
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length <= _EPS:
                continue
            axis = (-dy / length, dx / length)
            projection_a = [p[0] * axis[0] + p[1] * axis[1] for p in first]
            projection_b = [p[0] * axis[0] + p[1] * axis[1] for p in second]
            if max(projection_a) + gap <= min(projection_b) + _EPS:
                return True
            if max(projection_b) + gap <= min(projection_a) + _EPS:
                return True
    return False


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def overlap_area(first: Polygon, second: Polygon) -> float:
    """Clip counterclockwise convex polygons and return their intersection area."""
    output = list(first)
    for start, end in zip(second, second[1:] + second[:1]):
        edge = (end[0] - start[0], end[1] - start[1])

        def signed_distance(point: Point) -> float:
            return _cross(edge, (point[0] - start[0], point[1] - start[1]))

        input_points, output = output, []
        if not input_points:
            break
        previous = input_points[-1]
        previous_distance = signed_distance(previous)
        for current in input_points:
            current_distance = signed_distance(current)
            previous_inside, current_inside = previous_distance >= -_EPS, current_distance >= -_EPS
            if previous_inside != current_inside:
                fraction = previous_distance / (previous_distance - current_distance)
                output.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
            if current_inside:
                output.append(current)
            previous, previous_distance = current, current_distance
    return planar_polygon_area(tuple(output))


def _convex_hull(points: Sequence[Point]) -> Polygon:
    unique = sorted(set(points))

    def turn(origin: Point, first: Point, second: Point) -> float:
        return _cross((first[0] - origin[0], first[1] - origin[1]), (second[0] - origin[0], second[1] - origin[1]))

    lower: list[Point] = []
    upper: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= _EPS:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= _EPS:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


@njit(cache=True)
def _hull_vertices(points):
    """The same monotone chain/EPS as _convex_hull, without Python containers."""
    ordered = points.copy()
    for i in range(1, len(ordered)):
        x, y = ordered[i]
        j = i - 1
        while j >= 0 and (ordered[j, 0] > x or (ordered[j, 0] == x and ordered[j, 1] > y)):
            ordered[j + 1] = ordered[j]
            j -= 1
        ordered[j + 1, 0], ordered[j + 1, 1] = x, y
    count = 0
    for i in range(len(ordered)):
        if count == 0 or ordered[i, 0] != ordered[count - 1, 0] or ordered[i, 1] != ordered[count - 1, 1]:
            ordered[count] = ordered[i]
            count += 1
    hull = np.empty((2 * count, 2))
    written = 0
    for reverse in range(2):
        begin = written
        for i in range(count):
            point = ordered[i if reverse == 0 else count - 1 - i]
            while written - begin >= 2:
                a, b = hull[written - 2], hull[written - 1]
                turn = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                if turn > _EPS:
                    break
                written -= 1
            hull[written] = point
            written += 1
        written -= 1
    return hull[:written]


@njit(cache=True)
def _rotation_envelopes(footprints, padding):
    """Compile the expensive two hulls per rotation interval together."""
    # Keep room for all 8 endpoint corners expanded by 4 offsets. Near-collinear
    # EPS decisions can retain more vertices than the ideal Minkowski bound.
    vertices = np.zeros((len(footprints) - 1, 32, 2))
    counts = np.zeros(len(vertices), dtype=np.int64)
    offsets = ((-padding, -padding), (padding, -padding), (padding, padding), (-padding, padding))
    for i in range(len(vertices)):
        hull = _hull_vertices(np.concatenate((footprints[i], footprints[i + 1])))
        expanded = np.empty((len(hull) * 4, 2))
        for j in range(len(hull)):
            for k in range(4):
                expanded[j * 4 + k, 0] = hull[j, 0] + offsets[k][0]
                expanded[j * 4 + k, 1] = hull[j, 1] + offsets[k][1]
        envelope = _hull_vertices(expanded)
        counts[i] = len(envelope)
        vertices[i, :len(envelope)] = envelope
    return vertices, counts


def _swept_polygon(polygon: Polygon, displacement: Point) -> Polygon:
    translated = tuple((x + displacement[0], y + displacement[1]) for x, y in polygon)
    return _convex_hull(polygon + translated)


def _half_extents(geometry: ObjectGeometry, yaw: float) -> Point:
    hx = 0.5 * (geometry.upper[0] - geometry.lower[0])
    hy = 0.5 * (geometry.upper[1] - geometry.lower[1])
    cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (cosine * hx + sine * hy, sine * hx + cosine * hy)


def _supported(polygon: Polygon, rules: SpawnRules) -> bool:
    return all(
        rules.x_range[0] + rules.edge_margin - _EPS <= x <= rules.x_range[1] - rules.edge_margin + _EPS
        and rules.y_range[0] + rules.edge_margin - _EPS <= y <= rules.y_range[1] - rules.edge_margin + _EPS
        for x, y in polygon
    )


def _cluster_measurements(polygons: Sequence[Polygon]) -> Point:
    points = tuple(point for polygon in polygons for point in polygon)
    x_span = max(point[0] for point in points) - min(point[0] for point in points)
    y_span = max(point[1] for point in points) - min(point[1] for point in points)
    return y_span, sum(planar_polygon_area(polygon) for polygon in polygons) / (x_span * y_span)


def cluster_metrics(sample: SceneSample, geometries: Sequence[ObjectGeometry]) -> Point:
    """Return legacy diagnostic Y span and fill of the tight bounding rectangle.

    Neither quantity is an acceptance threshold. Only active footprints count;
    the reserved future rearrangement space
    is excluded. Footprints are conservative boxes rather than occupied mesh
    surface area, so this metric is a reproducible spawn proxy for clutter.
    """
    return _cluster_measurements(tuple(
        footprint_polygon(geometries[index], sample.positions[index][:2], sample.yaws[index])
        for index in sample.active_ids
    ))


def _asset_type(geometry: ObjectGeometry) -> str:
    return geometry.asset_name or geometry.name


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy)


def polygon_distance(first: Polygon, second: Polygon) -> float:
    """Euclidean boundary distance for convex polygons (zero if intersecting)."""
    if not polygons_separated(first, second):
        return 0.0
    return min(_point_segment_distance(point, start, end)
               for points, edges in ((first, second), (second, first))
               for point in points for start, end in zip(edges, edges[1:] + edges[:1]))


def shelf_regions(rules: SpawnRules) -> tuple[Region, ...]:
    """Equal usable-support cells, X-major then Y-major, including reserved space."""
    nx, ny = rules.region_grid
    xmin, xmax = rules.x_range[0] + rules.edge_margin, rules.x_range[1] - rules.edge_margin
    ymin, ymax = rules.y_range[0] + rules.edge_margin, rules.y_range[1] - rules.edge_margin
    dx, dy = (xmax - xmin) / nx, (ymax - ymin) / ny
    return tuple((xmin + ix * dx, xmin + (ix + 1) * dx, ymin + iy * dy, ymin + (iy + 1) * dy)
                 for ix in range(nx) for iy in range(ny))


def _regional_occupancies(polygons: Sequence[Polygon], rules: SpawnRules) -> tuple[float, ...]:
    return tuple(sum(overlap_area(polygon, rectangle_polygon(region)) for polygon in polygons)
                 / ((region[1] - region[0]) * (region[3] - region[2])) for region in shelf_regions(rules))


def _scene_measurements(polygons: Sequence[Polygon], rules: SpawnRules) -> SceneMetrics:
    points = tuple(point for polygon in polygons for point in polygon)
    usable_x = rules.x_range[1] - rules.x_range[0] - 2 * rules.edge_margin
    usable_y = rules.y_range[1] - rules.y_range[0] - 2 * rules.edge_margin
    neighbors = sum(any(other != index and polygon_distance(polygon, polygons[other])
                        <= rules.neighbor_distance + _EPS for other in range(len(polygons)))
                    for index, polygon in enumerate(polygons))
    return SceneMetrics(
        (max(x for x, _ in points) - min(x for x, _ in points)) / usable_x,
        (max(y for _, y in points) - min(y for _, y in points)) / usable_y,
        sum(planar_polygon_area(polygon) for polygon in polygons) / (usable_x * usable_y),
        neighbors / len(polygons),
        _regional_occupancies(polygons, rules),
    )


def scene_metrics(sample: SceneSample, geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> SceneMetrics:
    """Whole-shelf coverage/occupancy and independent local-clutter measurements."""
    return _scene_measurements(tuple(footprint_polygon(geometries[index], sample.positions[index][:2],
                                                       sample.yaws[index]) for index in sample.active_ids), rules)


def _pose_polygon(geometry: ObjectGeometry, pose: PlanarPose) -> Polygon:
    return footprint_polygon(geometry, (pose.x, pose.y), pose.yaw)


def _motion_pose(motion: PlanarMotion, fraction: float) -> PlanarPose:
    delta = motion.end.yaw - motion.start.yaw
    yaw = motion.start.yaw + fraction * delta
    if motion.pivot is None:
        return PlanarPose(motion.start.x + fraction * (motion.end.x - motion.start.x),
                          motion.start.y + fraction * (motion.end.y - motion.start.y), yaw)
    px, py = motion.pivot
    dx, dy = motion.start.x - px, motion.start.y - py
    cosine, sine = math.cos(fraction * delta), math.sin(fraction * delta)
    return PlanarPose(px + cosine * dx - sine * dy, py + sine * dx + cosine * dy, yaw)


def motion_sweeps(geometry: ObjectGeometry, motion: PlanarMotion) -> tuple[Polygon, ...]:
    """Enclose every footprint during a motion, including rotation between poses.

    Each rotation interval is <= 5 degrees. Its endpoint convex hull is expanded
    by R*(1-cos(angle/2)), the maximum corner-arc departure from its chord. The
    square expansion contains that circular error bound. Fixed-pivot R measures
    corners from the pivot; ordinary SE(2) R measures corners from the center.
    Thus checking only clear endpoints cannot conceal a mid-rotation collision.
    """
    delta = motion.end.yaw - motion.start.yaw
    start_polygon = _pose_polygon(geometry, motion.start)
    if abs(delta) <= _EPS:
        return (_convex_hull(start_polygon + _pose_polygon(geometry, motion.end)),)
    origin = motion.pivot or (motion.start.x, motion.start.y)
    radius = max(math.hypot(x - origin[0], y - origin[1]) for x, y in start_polygon)
    divisions = max(1, math.ceil(abs(delta) / math.radians(5.0)))
    padding = radius * (1.0 - math.cos(abs(delta) / (2 * divisions)))
    footprints = np.asarray([start_polygon, *(_pose_polygon(geometry, _motion_pose(motion, step / divisions))
                                              for step in range(1, divisions + 1))])
    vertices, counts = _rotation_envelopes(footprints, padding)
    return tuple(tuple(map(tuple, polygon[:count].tolist())) for polygon, count in zip(vertices, counts))


def _path_sweeps(geometry: ObjectGeometry, path: Sequence[PlanarMotion]) -> tuple[Polygon, ...]:
    return tuple(polygon for motion in path for polygon in motion_sweeps(geometry, motion))


def _path_supported_clear(geometry: ObjectGeometry, path: Sequence[PlanarMotion], obstacles: Sequence[Polygon],
                          rules: SpawnRules, *, return_sweeps=False):
    # Endpoint rejection is necessary (never sufficient) for a clear sweep.
    # It avoids constructing dozens of convex rotation envelopes for the many
    # candidate pivots that plainly end outside the shelf or inside Target.
    for motion in path:
        polygon = _pose_polygon(geometry, motion.end)
        if not _supported(polygon, rules) or any(
            not polygons_separated(polygon, obstacle, rules.gap) for obstacle in obstacles
        ):
            return None if return_sweeps else False
    sweeps = _path_sweeps(geometry, path)
    clear = all(_supported(polygon, rules) and all(polygons_separated(polygon, obstacle, rules.gap)
                                                  for obstacle in obstacles) for polygon in sweeps)
    return (sweeps if clear else None) if return_sweeps else clear


def _check_inputs(geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> None:
    values = (*rules.x_range, *rules.y_range, rules.shelf_top, rules.ceiling, rules.edge_margin,
              rules.gap, rules.hand_width, rules.approach_margin, rules.min_blocked_ratio,
              rules.min_x_coverage, rules.min_y_coverage, rules.neighbor_distance, rules.min_neighbor_fraction,
              rules.min_shelf_occupancy, rules.min_region_occupancy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Spawn rules must have finite dimensions and thresholds.")
    if not 2 <= rules.min_objects <= rules.max_objects <= len(geometries):
        raise ValueError("Active object limits must satisfy 2 <= min <= max <= catalog size.")
    if not 2 <= rules.min_asset_types <= rules.max_asset_types <= len({_asset_type(geometry) for geometry in geometries}):
        raise ValueError("Asset type limits must satisfy 2 <= min <= max <= catalog type count.")
    if len(rules.region_grid) != 2 or any(not isinstance(value, int) or value < 1 for value in rules.region_grid):
        raise ValueError("Region grid must contain two positive integer cell counts.")
    if not isinstance(rules.reserve_blocker_front, bool):
        raise ValueError("reserve_blocker_front must be boolean.")
    if rules.max_attempts < 1 or not 0.0 < rules.min_blocked_ratio <= 1.0:
        raise ValueError("max_attempts must be positive and min_blocked_ratio must be in (0, 1].")
    if any(not 0.0 <= value <= 1.0 for value in
           (rules.min_x_coverage, rules.min_y_coverage, rules.min_neighbor_fraction,
            rules.min_shelf_occupancy, rules.min_region_occupancy)):
        raise ValueError("Coverage and neighbor fraction thresholds must be in [0, 1].")
    if min(rules.edge_margin, rules.gap, rules.approach_margin, rules.neighbor_distance) < 0 or rules.hand_width <= 0:
        raise ValueError("Margins, neighbor distance and gap must be nonnegative; hand_width must be positive.")
    if rules.x_range[1] - rules.x_range[0] <= 2.0 * rules.edge_margin:
        raise ValueError("Shelf X range has no usable support area.")
    if rules.y_range[1] - rules.y_range[0] <= 2.0 * rules.edge_margin:
        raise ValueError("Shelf Y range has no usable support area.")
    if rules.ceiling <= rules.shelf_top:
        raise ValueError("Ceiling must be above shelf support.")
    if len({geometry.name for geometry in geometries}) != len(geometries):
        raise ValueError("Catalog geometry names must be unique.")
    for geometry in geometries:
        if not all(math.isfinite(value) for value in (*geometry.lower, *geometry.upper)):
            raise ValueError(f"Nonfinite bounds for {geometry.name}.")
        if any(high <= low for low, high in zip(geometry.lower, geometry.upper)):
            raise ValueError(f"Bounds must have positive extent for {geometry.name}.")


def _validated_path(sample: SceneSample) -> tuple[PlanarMotion, ...]:
    start = PlanarPose(*sample.positions[sample.blocker_id][:2], sample.yaws[sample.blocker_id])
    # Legacy manually specified witnesses still describe a genuine translation.
    path = sample.clearance_path or (PlanarMotion(start, PlanarPose(*sample.clearance_witness, start.yaw)),)
    previous = start
    for motion in path:
        values = (motion.start.x, motion.start.y, motion.start.yaw, motion.end.x, motion.end.y, motion.end.yaw)
        if motion.pivot is not None:
            values += motion.pivot
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Clearance path poses and pivots must be finite.")
        if any(abs(a - b) > _EPS for a, b in zip((previous.x, previous.y, previous.yaw),
                                                (motion.start.x, motion.start.y, motion.start.yaw))):
            raise ValueError("Clearance path must start at the initial Blocker and remain continuous.")
        if motion.pivot is not None:
            endpoint = _motion_pose(motion, 1.0)
            if math.hypot(endpoint.x - motion.end.x, endpoint.y - motion.end.y) > _EPS:
                raise ValueError("Pivot motion endpoint must agree with its fixed-pivot arc.")
        previous = motion.end
    if any(abs(a - b) > _EPS for a, b in zip((previous.x, previous.y), sample.clearance_witness)):
        raise ValueError("Clearance witness must agree with the final path pose.")
    return path


def validate_sample(sample: SceneSample, geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> None:
    """Raise ValueError if a scene violates role, coverage, clutter or SE(2) path rules."""
    _check_inputs(geometries, rules)
    ids = sample.active_ids
    if not rules.min_objects <= len(ids) <= rules.max_objects or len(set(ids)) != len(ids):
        raise ValueError("Active IDs must be unique and respect active object limits.")
    if any(index < 0 or index >= len(geometries) for index in ids):
        raise ValueError("Active IDs must index the catalog.")
    if sample.target_id not in ids or sample.blocker_id not in ids or sample.target_id == sample.blocker_id:
        raise ValueError("Target and Blocker must be distinct active objects.")
    target_type = _asset_type(geometries[sample.target_id])
    if sum(_asset_type(geometries[index]) == target_type for index in ids) != 1:
        raise ValueError("Target asset type must occur exactly once among active objects.")
    active_types = {_asset_type(geometries[index]) for index in ids}
    if not rules.min_asset_types <= len(active_types) <= rules.max_asset_types:
        raise ValueError("Active asset type count must respect asset type limits.")
    if rules.all_asset_types and { _asset_type(geometries[index]) for index in ids } != {
            _asset_type(geometry) for geometry in geometries}:
        raise ValueError("All-asset-types scenes must contain every catalog asset type.")
    if set(sample.positions) != set(ids) or set(sample.yaws) != set(ids):
        raise ValueError("Sampled positions and yaws must contain exactly the active objects.")
    polygons: dict[int, Polygon] = {}
    for index in ids:
        position, yaw, geometry = sample.positions[index], sample.yaws[index], geometries[index]
        if not all(math.isfinite(value) for value in (*position, yaw)):
            raise ValueError("Sampled poses must be finite.")
        expected_z = rules.shelf_top - geometry.lower[2]
        if abs(position[2] - expected_z) > _EPS:
            raise ValueError("Root Z must put the canonical lower bound on the shelf.")
        if position[2] + geometry.upper[2] > rules.ceiling - rules.gap + _EPS:
            raise ValueError("An active object lacks ceiling clearance.")
        polygons[index] = footprint_polygon(geometry, position[:2], yaw)
        if not _supported(polygons[index], rules):
            raise ValueError("An active footprint is outside supported shelf margins.")
    for first, second in combinations(ids, 2):
        if not polygons_separated(polygons[first], polygons[second], rules.gap):
            raise ValueError("Active footprints overlap or lack the required gap.")
    target, blocker = sample.target_id, sample.blocker_id
    target_front = max(x for x, _ in polygons[target])
    if min(x for x, _ in polygons[blocker]) < target_front + rules.gap - _EPS:
        raise ValueError("Blocker must be on the robot side of Target with a positive planar gap.")
    half_width = rules.hand_width * 0.5 + rules.approach_margin
    target_y = sample.positions[target][1]
    expected_access = (target_front + rules.approach_margin, rules.x_range[1],
                       target_y - half_width, target_y + half_width)
    if not all(math.isfinite(value) for value in sample.access_region):
        raise ValueError("Access region must be finite.")
    if any(abs(actual - expected) > _EPS for actual, expected in zip(sample.access_region, expected_access)):
        raise ValueError("Access region must stay fixed to the initial Target and +X approach.")
    if expected_access[0] >= expected_access[1]:
        raise ValueError("Access region has no positive length.")
    access = rectangle_polygon(sample.access_region)
    if overlap_area(polygons[blocker], access) / planar_polygon_area(polygons[blocker]) < rules.min_blocked_ratio - _EPS:
        raise ValueError("Designated Blocker does not sufficiently block the initial access region.")
    for index in ids:
        if index not in (target, blocker) and not polygons_separated(polygons[index], access, rules.gap):
            raise ValueError("Surrounding objects must begin outside the initial access region.")
    if rules.reserve_blocker_front:
        blocker_front = rectangle_polygon(_blocker_front_region(polygons[blocker], rules))
        if any(index not in (target, blocker) and not polygons_separated(polygon, blocker_front)
               for index, polygon in polygons.items()):
            raise ValueError("Surrounding objects must begin outside the Blocker frontal-view corridor.")
    if not all(math.isfinite(value) for value in sample.clearance_witness):
        raise ValueError("Clearance witness must be finite.")
    path = _validated_path(sample)
    final = _pose_polygon(geometries[blocker], path[-1].end)
    if not polygons_separated(final, access, rules.gap):
        raise ValueError("Clearance witness still intersects the access region.")
    for swept in _path_sweeps(geometries[blocker], path):
        if not _supported(swept, rules):
            raise ValueError("Blocker clearance path leaves supported shelf margins.")
        if any(index != blocker and not polygons_separated(swept, polygon, rules.gap)
               for index, polygon in polygons.items()):
            raise ValueError("Blocker clearance path is obstructed by another object.")
    metrics = _scene_measurements(tuple(polygons.values()), rules)
    if metrics.x_coverage < rules.min_x_coverage - _EPS or metrics.y_coverage < rules.min_y_coverage - _EPS:
        raise ValueError("Active footprints do not meet whole-shelf coverage thresholds.")
    if metrics.local_neighbor_fraction < rules.min_neighbor_fraction - _EPS:
        raise ValueError("Too few active footprints have a nearby clutter neighbor.")
    if metrics.shelf_occupancy < rules.min_shelf_occupancy - _EPS:
        raise ValueError("Whole-shelf footprint occupancy is below the required density.")
    if min(metrics.region_occupancies) < rules.min_region_occupancy - _EPS:
        raise ValueError("A shelf region footprint occupancy is below the required density.")


def _clearance_candidates(geometry: ObjectGeometry, initial: PlanarPose, access: Region,
                          rules: SpawnRules, rng: random.Random) -> list[tuple[PlanarMotion, ...]]:
    """Finite translation, pivot, rotate/translate, and multistage proposals."""
    prefixes: list[tuple[PlanarMotion, ...]] = [()]
    for angle in (-math.pi / 2, math.pi / 2, -math.pi / 4, math.pi / 4):
        prefixes.append((PlanarMotion(initial, PlanarPose(initial.x, initial.y, initial.yaw + angle)),))
        for pivot in _pose_polygon(geometry, initial):
            provisional = PlanarMotion(initial, PlanarPose(0, 0, initial.yaw + angle), pivot)
            prefixes.append((PlanarMotion(initial, _motion_pose(provisional, 1.0), pivot),))
    # Move away from the rear Target before turning, then translate to clear A0.
    for dx in (0.015, 0.03):
        shifted = PlanarPose(initial.x + dx, initial.y, initial.yaw)
        for angle in (-math.pi / 2, math.pi / 2):
            prefixes.append((PlanarMotion(initial, shifted),
                             PlanarMotion(shifted, PlanarPose(shifted.x, shifted.y, shifted.yaw + angle))))
    poses = np.asarray([(p[-1].end.x, p[-1].end.y, p[-1].end.yaw) if p else
                        (initial.x, initial.y, initial.yaw) for p in prefixes])
    clear = _prefix_clearance(poses, (geometry.upper[0]-geometry.lower[0])/2,
                              (geometry.upper[1]-geometry.lower[1])/2, np.asarray(access), rules.gap)
    candidates = []
    for index, prefix in enumerate(prefixes):
        pose = prefix[-1].end if prefix else initial
        if prefix and clear[index]:
            candidates.append(prefix)
        _, hy = _half_extents(geometry, pose.yaw)
        for y in (access[2] - hy - rules.gap - 0.001, access[3] + hy + rules.gap + 0.001):
            candidates.append((*prefix, PlanarMotion(pose, PlanarPose(pose.x, y, pose.yaw))))
    rng.shuffle(candidates)
    return candidates


@njit(cache=True)
def _prefix_clearance(poses, hx, hy, access, gap):
    """The same four rectangle SAT axes, batched before candidate shuffling."""
    result = np.zeros(len(poses), dtype=np.bool_)
    ox, oy = (access[0]+access[1])/2, (access[2]+access[3])/2
    ohx, ohy = (access[1]-access[0])/2, (access[3]-access[2])/2
    for k in range(len(poses)):
        x, y, angle = poses[k]
        c, s = math.cos(angle), math.sin(angle)
        for ax, ay in ((c,s),(-s,c),(1.,0.),(0.,1.)):
            radius = hx*abs(ax*c+ay*s)+hy*abs(-ax*s+ay*c)
            other = ohx*abs(ax)+ohy*abs(ay)
            if abs((x-ox)*ax+(y-oy)*ay) >= radius+other+gap-_EPS:
                result[k] = True
                break
    return result


# Layout state stays on the requested device across candidate/commit rounds.
# Candidate geometry uses bounded local scratch; adjacency caches only pair flags.
_LayoutQuad = wp.types.matrix(shape=(4, 2), dtype=wp.float64)
_LayoutClip = wp.types.matrix(shape=(12, 2), dtype=wp.float64)
_PathVertices = wp.types.matrix(shape=(64, 2), dtype=wp.float64)


@wp.func
def _path_hull(points: _PathVertices, count: int):
    # Match the independent scalar monotone chain, including near-collinear EPS.
    ordered = points
    for i in range(1, count):
        x, y = ordered[i, 0], ordered[i, 1]
        j = i - 1
        while j >= 0:
            if ordered[j, 0] < x or (ordered[j, 0] == x and ordered[j, 1] <= y):
                break
            ordered[j+1, 0] = ordered[j, 0]
            ordered[j+1, 1] = ordered[j, 1]
            j -= 1
        ordered[j+1, 0] = x
        ordered[j+1, 1] = y
    unique = int(1)
    for i in range(1,count):
        if ordered[i, 0] != ordered[unique-1, 0] or ordered[i, 1] != ordered[unique-1, 1]:
            ordered[unique, 0] = ordered[i, 0]
            ordered[unique, 1] = ordered[i, 1]
            unique += 1
    hull = _PathVertices()
    written = int(0)
    for reverse in range(2):
        begin = written
        for k in range(unique):
            i = k
            if reverse == 1:
                i = unique-1-k
            x, y = ordered[i, 0], ordered[i, 1]
            while written-begin >= 2:
                ax, ay = hull[written-2, 0], hull[written-2, 1]
                bx, by = hull[written-1, 0], hull[written-1, 1]
                if (bx-ax)*(y-ay)-(by-ay)*(x-ax) > wp.float64(1.e-9):
                    break
                written -= 1
            hull[written, 0] = x
            hull[written, 1] = y
            written += 1
        written -= 1
    return hull, written


@wp.func
def _path_clear(p: _PathVertices, count: int, obstacle: _LayoutQuad, bounds: wp.vec4d, gap: wp.float64):
    for i in range(count):
        if (p[i, 0] < bounds[0]-wp.float64(1.e-9) or p[i, 0] > bounds[1]+wp.float64(1.e-9)
                or p[i, 1] < bounds[2]-wp.float64(1.e-9) or p[i, 1] > bounds[3]+wp.float64(1.e-9)):
            return False
    for edge in range(count+4):
        dx, dy = wp.float64(0.), wp.float64(0.)
        if edge < count:
            j = (edge+1) % count
            dx, dy = p[j, 0]-p[edge, 0], p[j, 1]-p[edge, 1]
        else:
            i = edge-count
            j = (i+1) % 4
            dx, dy = obstacle[j, 0]-obstacle[i, 0], obstacle[j, 1]-obstacle[i, 1]
        length = wp.sqrt(dx*dx+dy*dy)
        if length > wp.float64(1.e-9):
            ax, ay = -dy/length, dx/length
            low, high = wp.float64(wp.inf), wp.float64(-wp.inf)
            other_low, other_high = wp.float64(wp.inf), wp.float64(-wp.inf)
            for i in range(count):
                value = p[i, 0]*ax+p[i, 1]*ay
                low, high = wp.min(low,value), wp.max(high,value)
            for i in range(4):
                value = obstacle[i, 0]*ax+obstacle[i, 1]*ay
                other_low, other_high = wp.min(other_low,value), wp.max(other_high,value)
            if high+gap <= other_low+wp.float64(1.e-9) or other_high+gap <= low+wp.float64(1.e-9):
                return True
    return False


@wp.func
def _path_pose(start: wp.vec3d, end: wp.vec3d, pivot: wp.vec2d, use_pivot: bool, fraction: wp.float64):
    angle = fraction*(end[2]-start[2])
    result = start+fraction*(end-start)
    if use_pivot:
        c, s = wp.cos(angle), wp.sin(angle)
        dx, dy = start[0]-pivot[0], start[1]-pivot[1]
        result = wp.vec3d(pivot[0]+c*dx-s*dy, pivot[1]+s*dx+c*dy, start[2]+angle)
    return result


@wp.kernel
def _clearance_paths(motions: wp.array4d(dtype=wp.float64), counts: wp.array2d(dtype=wp.int32),
                     half: wp.array2d(dtype=wp.float64), targets: wp.array3d(dtype=wp.float64),
                     bounds: wp.vec4d, gap: wp.float64, valid: wp.array2d(dtype=wp.int32)):
    row, candidate = wp.tid()
    valid[row, candidate] = 0
    count = counts[row, candidate]
    if count == 0:
        return
    target = _LayoutQuad()
    for i in range(4):
        target[i, 0] = targets[row, i, 0]
        target[i, 1] = targets[row, i, 1]
    hx, hy = half[row, 0], half[row, 1]
    # Reject endpoints before constructing conservative swept envelopes.
    for k in range(count):
        p = _layout_quad(motions[row,candidate,k,3], motions[row,candidate,k,4], motions[row,candidate,k,5],hx,hy)
        points = _PathVertices()
        for i in range(4):
            points[i,0] = p[i,0]
            points[i,1] = p[i,1]
        if not _path_clear(points,4,target,bounds,gap):
            return
    for k in range(count):
        start = wp.vec3d(motions[row,candidate,k,0],motions[row,candidate,k,1],motions[row,candidate,k,2])
        end = wp.vec3d(motions[row,candidate,k,3],motions[row,candidate,k,4],motions[row,candidate,k,5])
        pivot = wp.vec2d(motions[row,candidate,k,6],motions[row,candidate,k,7])
        use_pivot = motions[row,candidate,k,8] != wp.float64(0.)
        delta = end[2]-start[2]
        previous = _layout_quad(start[0],start[1],start[2],hx,hy)
        divisions, padding = int(1), wp.float64(0.)
        rotating = wp.abs(delta) > wp.float64(1.e-9)
        if rotating:
            divisions = wp.max(1,int(wp.ceil(wp.abs(delta)/wp.float64(0.08726646259971647))))
            origin = wp.vec2d(start[0],start[1])
            if use_pivot:
                origin = pivot
            radius = wp.float64(0.)
            for i in range(4):
                radius = wp.max(radius,wp.length(wp.vec2d(previous[i,0],previous[i,1])-origin))
            padding = radius*(wp.float64(1.)-wp.cos(wp.abs(delta)/wp.float64(2*divisions)))
        for step in range(1,divisions+1):
            pose = _path_pose(start,end,pivot,use_pivot,wp.float64(step)/wp.float64(divisions))
            # The scalar non-rotation path uses authored end, including any
            # sub-EPS angular delta, rather than a pivot interpolation.
            if not rotating:
                pose = end
            current = _layout_quad(pose[0],pose[1],pose[2],hx,hy)
            points = _PathVertices()
            for i in range(4):
                points[i,0] = previous[i,0]
                points[i,1] = previous[i,1]
                points[i+4,0] = current[i,0]
                points[i+4,1] = current[i,1]
            hull, size = _path_hull(points,8)
            if rotating:
                expanded = _PathVertices()
                for i in range(size):
                    for j in range(4):
                        dx,dy = -padding,-padding
                        if j == 1 or j == 2:
                            dx = padding
                        if j >= 2:
                            dy = padding
                        expanded[i*4+j,0] = hull[i,0]+dx
                        expanded[i*4+j,1] = hull[i,1]+dy
                hull,size = _path_hull(expanded,size*4)
            if not _path_clear(hull,size,target,bounds,gap):
                return
            previous = current
    valid[row,candidate] = 1


@dataclass
class _PathRequest:
    geometry: ObjectGeometry
    candidates: list
    target: Polygon


def _choose_clearance_paths(requests, rules, device):
    """Check independent candidates together; return only the first valid index per row."""
    wp.init()
    width = max(len(r.candidates) for r in requests)
    motions = np.zeros((len(requests),width,3,9))
    counts = np.zeros((len(requests),width),np.int32)
    half, targets = [], []
    for row, request in enumerate(requests):
        g = request.geometry
        half.append(((g.upper[0]-g.lower[0])/2,(g.upper[1]-g.lower[1])/2))
        targets.append(request.target)
        for column,path in enumerate(request.candidates):
            counts[row,column] = len(path)
            for k,motion in enumerate(path):
                a,b = motion.start,motion.end
                motions[row,column,k] = (a.x,a.y,a.yaw,b.x,b.y,b.yaw,*(motion.pivot or (0.,0.)),motion.pivot is not None)
    tensors = [torch.as_tensor(a,device=device) for a in
               (motions,counts,np.asarray(half,dtype=np.float64),np.asarray(targets,dtype=np.float64))]
    valid = torch.empty(counts.shape,device=device,dtype=torch.int32)
    bounds = wp.vec4d(rules.x_range[0]+rules.edge_margin,rules.x_range[1]-rules.edge_margin,
                      rules.y_range[0]+rules.edge_margin,rules.y_range[1]-rules.edge_margin)
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if torch.device(device).type == 'cuda' else None
    wp.launch(_clearance_paths,dim=counts.shape,inputs=[*(wp.from_torch(t) for t in tensors),bounds,rules.gap,
              wp.from_torch(valid)],device=str(device),stream=stream)
    first = torch.where(valid.bool(),torch.arange(width,device=device),width).amin(-1).tolist()
    return [r.candidates[index] if index < width else None for r,index in zip(requests,first)]


@wp.func
def _layout_quad(x: wp.float64, y: wp.float64, angle: wp.float64, hx: wp.float64, hy: wp.float64):
    result = _LayoutQuad()
    c, s = wp.cos(angle), wp.sin(angle)
    for k in range(4):
        dx, dy = -hx, -hy
        if k == 1 or k == 2:
            dx = hx
        if k >= 2:
            dy = hy
        result[k, 0] = x + c * dx - s * dy
        result[k, 1] = y + s * dx + c * dy
    return result


@wp.func
def _layout_area(p: _LayoutQuad, bounds: wp.vec4d):
    vertices = _LayoutClip()
    count = int(4)
    for k in range(4):
        vertices[k, 0] = p[k, 0]
        vertices[k, 1] = p[k, 1]
    for plane in range(4):
        if count > 0:
            output = _LayoutClip()
            axis, sign = plane // 2, wp.float64(1.0)
            if plane % 2 == 1:
                sign = wp.float64(-1.0)
            written = int(0)
            previous = count - 1
            for current in range(count):
                pd = sign * (wp.float64(vertices[previous, axis]) - bounds[plane])
                cd = sign * (wp.float64(vertices[current, axis]) - bounds[plane])
                if (pd >= wp.float64(0.0)) != (cd >= wp.float64(0.0)):
                    fraction = pd / (pd - cd)
                    for component in range(2):
                        output[written, component] = wp.float64(vertices[previous, component]) + fraction * (wp.float64(vertices[current, component]) - wp.float64(vertices[previous, component]))
                    written += 1
                if cd >= wp.float64(0.0):
                    output[written, 0] = wp.float64(vertices[current, 0])
                    output[written, 1] = wp.float64(vertices[current, 1])
                    written += 1
                previous = current
            vertices, count = output, written
    area = wp.float64(0.0)
    for k in range(count):
        j = (k + 1) % count
        area += wp.float64(vertices[k, 0]) * wp.float64(vertices[j, 1]) - wp.float64(vertices[k, 1]) * wp.float64(vertices[j, 0])
    return wp.abs(area) * wp.float64(0.5)


@wp.func
def _layout_distance(a: _LayoutQuad, b: _LayoutQuad):
    distance = wp.float64(wp.inf)
    for direction in range(2):
        points, polygon = a, b
        if direction == 1:
            points, polygon = b, a
        for i in range(4):
            for j in range(4):
                k = (j + 1) % 4
                edge = wp.vec2d(polygon[k, 0] - polygon[j, 0], polygon[k, 1] - polygon[j, 1])
                delta = wp.vec2d(points[i, 0] - polygon[j, 0], points[i, 1] - polygon[j, 1])
                t = wp.clamp(wp.dot(delta, edge) / wp.dot(edge, edge), wp.float64(0.0), wp.float64(1.0))
                delta -= t * edge
                distance = wp.min(distance, wp.dot(delta, delta))
    return wp.sqrt(distance)


@wp.func
def _layout_boxes_clear(x: wp.float64, y: wp.float64, c: wp.float64, s: wp.float64,
                        hx: wp.float64, hy: wp.float64, ox: wp.float64, oy: wp.float64,
                        oc: wp.float64, os: wp.float64, ohx: wp.float64, ohy: wp.float64, gap: wp.float64):
    clear = bool(False)
    for axis in range(4):
        ax, ay = c, s
        if axis == 1:
            ax, ay = -s, c
        elif axis == 2:
            ax, ay = oc, os
        elif axis == 3:
            ax, ay = -os, oc
        radius = hx * wp.abs(ax*c + ay*s) + hy * wp.abs(-ax*s + ay*c)
        other = ohx * wp.abs(ax*oc + ay*os) + ohy * wp.abs(-ax*os + ay*oc)
        if wp.abs((x-ox)*ax + (y-oy)*ay) >= radius + other + gap - wp.float64(1.e-9):
            clear = True
            break
    return clear


@wp.kernel(enable_backward=False)
def _layout_candidates(poses: wp.array3d(dtype=wp.float64), half: wp.array3d(dtype=wp.float64),
                       placed: wp.array2d(dtype=wp.int32), state: wp.array2d(dtype=wp.int32),
                       seeds: wp.array(dtype=wp.int32), regions: wp.array2d(dtype=wp.float64),
                       areas: wp.array2d(dtype=wp.float64), vertices: wp.array4d(dtype=wp.float64),
                       axes: wp.array4d(dtype=wp.float64), boxes: wp.array3d(dtype=wp.float64),
                       counts: wp.array2d(dtype=wp.int32), gaps: wp.array2d(dtype=wp.float64),
                       bounds: wp.vec4d, gap: wp.float64, neighbor: wp.float64, region_min: wp.float64,
                       cell_area: wp.float64, candidates: wp.array3d(dtype=wp.float64),
                       scores: wp.array2d(dtype=wp.float64), collisions: wp.array2d(dtype=wp.int32)):
    row, candidate = wp.tid()
    scores[row, candidate] = wp.float64(-1.e30)
    collisions[row, candidate] = -1
    if state[row, 0] != 0:
        return
    slot = state[row, 1]
    rng = wp.rand_init(seeds[row], state[row, 2] * scores.shape[1] + candidate)
    angle = wp.float64(wp.randf(rng)) * wp.float64(2.0 * math.pi) - wp.float64(math.pi)
    c, s = wp.cos(angle), wp.sin(angle)
    hx, hy = wp.float64(half[row, slot, 0]), wp.float64(half[row, slot, 1])
    ex, ey = wp.abs(c)*hx + wp.abs(s)*hy, wp.abs(s)*hx + wp.abs(c)*hy
    xmin, xmax = bounds[0]+ex, bounds[1]-ex
    ymin, ymax = bounds[2]+ey, bounds[3]-ey
    if xmin > xmax or ymin > ymax:
        return
    # Sample directly inside the supported center interval. Deficient cells
    # receive priority; near-neighbor proposals use intersected intervals too.
    cell = int(0)
    priority = wp.float64(wp.inf)
    for j in range(regions.shape[0]):
        value = wp.float64(areas[row, j]) / cell_area + wp.float64(wp.randf(rng)) * wp.float64(0.08)
        if value < priority:
            priority, cell = value, j
    xlo, xhi = wp.max(xmin, wp.float64(regions[cell, 0])), wp.min(xmax, wp.float64(regions[cell, 1]))
    ylo, yhi = wp.max(ymin, wp.float64(regions[cell, 2])), wp.min(ymax, wp.float64(regions[cell, 3]))
    if xlo > xhi or ylo > yhi:
        xlo, xhi, ylo, yhi = xmin, xmax, ymin, ymax
    if wp.randf(rng) >= float(0.60):
        anchor = wp.randi(rng, 0, state[row, 5])
        for offset in range(state[row, 5]):
            other = (anchor + offset) % state[row, 5]
            if placed[row, other] != 0:
                ac, ass = wp.cos(wp.float64(poses[row, other, 2])), wp.sin(wp.float64(poses[row, other, 2]))
                aex = wp.abs(ac)*wp.float64(half[row, other, 0]) + wp.abs(ass)*wp.float64(half[row, other, 1])
                aey = wp.abs(ass)*wp.float64(half[row, other, 0]) + wp.abs(ac)*wp.float64(half[row, other, 1])
                direction = wp.randi(rng, 0, 4)
                clearance = gap + wp.float64(0.001) + wp.float64(wp.randf(rng)) * wp.float64(0.024)
                anchor_x, anchor_y = wp.float64(poses[row, other, 0]), wp.float64(poses[row, other, 1])
                if direction < 2:
                    sign = wp.float64(2*direction - 1)
                    x = anchor_x + sign * (aex + ex + clearance)
                    low, high = wp.max(ymin, anchor_y-aey-ey), wp.min(ymax, anchor_y+aey+ey)
                    if x >= xmin and x <= xmax and low <= high:
                        xlo, xhi, ylo, yhi = x, x, low, high
                else:
                    sign = wp.float64(2*(direction-2) - 1)
                    y = anchor_y + sign * (aey + ey + clearance)
                    low, high = wp.max(xmin, anchor_x-aex-ex), wp.min(xmax, anchor_x+aex+ex)
                    if y >= ymin and y <= ymax and low <= high:
                        xlo, xhi, ylo, yhi = low, high, y, y
                break
    x = xlo + wp.float64(wp.randf(rng))*(xhi-xlo)
    y = ylo + wp.float64(wp.randf(rng))*(yhi-ylo)
    # Static reserved polygons: AABB broad phase, cached polygon axes, then
    # the two candidate box axes. Distant sweep segments need no SAT.
    for obstacle in range(counts.shape[1]):
        count = counts[row, obstacle]
        if count == 0:
            break
        margin = wp.float64(gaps[row, obstacle])
        # sqrt(2) padding also bounds the SAT axis-gap convention at corners.
        broad_margin = wp.float64(1.414213562373096) * margin
        if (x+ex+broad_margin <= wp.float64(boxes[row, obstacle, 0]) or x-ex-broad_margin >= wp.float64(boxes[row, obstacle, 1])
                or y+ey+broad_margin <= wp.float64(boxes[row, obstacle, 2]) or y-ey-broad_margin >= wp.float64(boxes[row, obstacle, 3])):
            continue
        clear = bool(False)
        for axis in range(count):
            ax, ay = wp.float64(axes[row, obstacle, axis, 0]), wp.float64(axes[row, obstacle, axis, 1])
            radius = hx*wp.abs(ax*c + ay*s) + hy*wp.abs(-ax*s + ay*c)
            center = x*ax + y*ay
            if (center+radius+margin <= wp.float64(axes[row, obstacle, axis, 2]) + wp.float64(1.e-9)
                    or wp.float64(axes[row, obstacle, axis, 3])+margin <= center-radius + wp.float64(1.e-9)):
                clear = True
                break
        if not clear:
            for axis in range(2):
                ax, ay, radius = c, s, hx
                if axis == 1:
                    ax, ay, radius = -s, c, hy
                lo, hi = wp.float64(wp.inf), wp.float64(-wp.inf)
                for k in range(count):
                    value = wp.float64(vertices[row, obstacle, k, 0])*ax + wp.float64(vertices[row, obstacle, k, 1])*ay
                    lo, hi = wp.min(lo, value), wp.max(hi, value)
                center = x*ax + y*ay
                if center+radius+margin <= lo+wp.float64(1.e-9) or hi+margin <= center-radius+wp.float64(1.e-9):
                    clear = True
                    break
        if not clear:
            return
    near = bool(False)
    polygon = _layout_quad(x, y, angle, hx, hy)
    for other in range(state[row, 5]):
        if placed[row, other] == 0:
            continue
        ox, oy, oa = wp.float64(poses[row, other, 0]), wp.float64(poses[row, other, 1]), wp.float64(poses[row, other, 2])
        oc, os = wp.cos(oa), wp.sin(oa)
        ohx, ohy = wp.float64(half[row, other, 0]), wp.float64(half[row, other, 1])
        oex, oey = wp.abs(oc)*ohx+wp.abs(os)*ohy, wp.abs(os)*ohx+wp.abs(oc)*ohy
        dx, dy = wp.abs(x-ox)-ex-oex, wp.abs(y-oy)-ey-oey
        if dx < wp.float64(1.414213562373096)*gap and dy < wp.float64(1.414213562373096)*gap:
            if not _layout_boxes_clear(x,y,c,s,hx,hy,ox,oy,oc,os,ohx,ohy,gap):
                collisions[row, candidate] = other
                return
        if not near and dx <= neighbor and dy <= neighbor:
            near = _layout_distance(polygon, _layout_quad(ox,oy,oa,ohx,ohy)) <= neighbor
    score = wp.float64(0.0)
    for cell in range(regions.shape[0]):
        region = wp.vec4d(wp.float64(regions[cell,0]), wp.float64(regions[cell,1]), wp.float64(regions[cell,2]), wp.float64(regions[cell,3]))
        increment = _layout_area(polygon, region)
        deficit = wp.max(wp.float64(0.0), region_min*cell_area-wp.float64(areas[row,cell]))
        score += wp.float64(3.0)*wp.min(increment,deficit)/(wp.float64(4.0)*hx*hy)
        score += wp.float64(0.1)*increment/(wp.float64(areas[row,cell])+cell_area*wp.float64(0.1))
    if near:
        score += wp.float64(0.75)
    candidates[row,candidate,0] = x
    candidates[row,candidate,1] = y
    candidates[row,candidate,2] = angle
    scores[row,candidate] = score


@wp.kernel(enable_backward=False)
def _layout_commit(poses: wp.array3d(dtype=wp.float64), half: wp.array3d(dtype=wp.float64),
                   placed: wp.array2d(dtype=wp.int32), state: wp.array2d(dtype=wp.int32),
                   regions: wp.array2d(dtype=wp.float64), areas: wp.array2d(dtype=wp.float64),
                   adjacent: wp.array3d(dtype=wp.int32), candidates: wp.array3d(dtype=wp.float64),
                   scores: wp.array2d(dtype=wp.float64), collisions: wp.array2d(dtype=wp.int32),
                   bounds: wp.vec4d, neighbor: wp.float64, neighbor_min: wp.float64,
                   region_min: wp.float64, cell_area: wp.float64, x_span: wp.float64, y_span: wp.float64):
    row = wp.tid()
    if state[row,0] != 0:
        return
    slot = state[row,1]
    state[row,2] += 1
    chosen, best = int(-1), wp.float64(-1.e29)
    for k in range(scores.shape[1]):
        if wp.float64(scores[row,k]) > best:
            chosen, best = k, wp.float64(scores[row,k])
    remove = int(-1)
    if chosen >= 0:
        for k in range(3):
            poses[row,slot,k] = wp.float64(candidates[row,chosen,k])
        placed[row,slot] = 1
        state[row,3] = 0
        p = _layout_quad(wp.float64(poses[row,slot,0]),wp.float64(poses[row,slot,1]),wp.float64(poses[row,slot,2]),wp.float64(half[row,slot,0]),wp.float64(half[row,slot,1]))
        for cell in range(regions.shape[0]):
            areas[row,cell] += _layout_area(p, wp.vec4d(wp.float64(regions[cell,0]),wp.float64(regions[cell,1]),wp.float64(regions[cell,2]),wp.float64(regions[cell,3])))
        for other in range(state[row,5]):
            if other != slot and placed[row,other] != 0:
                q = _layout_quad(wp.float64(poses[row,other,0]),wp.float64(poses[row,other,1]),wp.float64(poses[row,other,2]),wp.float64(half[row,other,0]),wp.float64(half[row,other,1]))
                value = int(_layout_distance(p,q) <= neighbor + wp.float64(1.e-9))
                adjacent[row,slot,other] = value
                adjacent[row,other,slot] = value
        pending = int(-1)
        for other in range(2,state[row,5]):
            if placed[row,other] == 0:
                pending = other
                break
        if pending >= 0:
            state[row,1] = pending
        else:
            # Only cached summaries and a linear bounds scan; no pairwise
            # geometry or clipping is repeated at completion.
            valid, linked = bool(True), int(0)
            xmin, xmax, ymin, ymax = wp.float64(wp.inf),wp.float64(-wp.inf),wp.float64(wp.inf),wp.float64(-wp.inf)
            for other in range(state[row,5]):
                has_neighbor = bool(False)
                for j in range(state[row,5]):
                    if adjacent[row,other,j] != 0:
                        has_neighbor = True
                        break
                if has_neighbor:
                    linked += 1
                c,s = wp.cos(wp.float64(poses[row,other,2])),wp.sin(wp.float64(poses[row,other,2]))
                ex = wp.abs(c)*wp.float64(half[row,other,0])+wp.abs(s)*wp.float64(half[row,other,1])
                ey = wp.abs(s)*wp.float64(half[row,other,0])+wp.abs(c)*wp.float64(half[row,other,1])
                xmin,xmax = wp.min(xmin,wp.float64(poses[row,other,0])-ex),wp.max(xmax,wp.float64(poses[row,other,0])+ex)
                ymin,ymax = wp.min(ymin,wp.float64(poses[row,other,1])-ey),wp.max(ymax,wp.float64(poses[row,other,1])+ey)
            valid = wp.float64(linked) >= neighbor_min*wp.float64(state[row,5])
            valid = valid and xmax-xmin >= x_span*(bounds[1]-bounds[0]) and ymax-ymin >= y_span*(bounds[3]-bounds[2])
            for cell in range(regions.shape[0]):
                valid = valid and wp.float64(areas[row,cell]) >= region_min*cell_area - wp.float64(1.e-9)
            if valid:
                state[row,0] = 1
            else:
                state[row,0] = -1
    else:
        state[row,3] += 1
        if state[row,3] >= 2:
            # Retain the failed slot and all unrelated objects. Reinsert the
            # displaced neighbor after this slot has found room.
            most = int(0)
            for other in range(2,state[row,5]):
                hits = int(0)
                if placed[row,other] != 0:
                    for k in range(collisions.shape[1]):
                        hits += int(collisions[row,k] == other)
                if hits > most:
                    most, remove = hits, other
            if remove < 0 or state[row,4] >= 16:
                state[row,0] = -1
            state[row,3] = 0
    if remove >= 0 and state[row,0] == 0:
        state[row,4] += 1
        placed[row,remove] = 0
        p = _layout_quad(wp.float64(poses[row,remove,0]),wp.float64(poses[row,remove,1]),wp.float64(poses[row,remove,2]),wp.float64(half[row,remove,0]),wp.float64(half[row,remove,1]))
        for cell in range(regions.shape[0]):
            areas[row,cell] -= _layout_area(p,wp.vec4d(wp.float64(regions[cell,0]),wp.float64(regions[cell,1]),wp.float64(regions[cell,2]),wp.float64(regions[cell,3])))
        for other in range(state[row,5]):
            adjacent[row,remove,other] = 0
            adjacent[row,other,remove] = 0


@wp.kernel
def _layout_obstacle_geometry(vertices: wp.array4d(dtype=wp.float64), counts: wp.array2d(dtype=wp.int32),
                              axes: wp.array4d(dtype=wp.float64), boxes: wp.array3d(dtype=wp.float64)):
    row, column = wp.tid()
    count = counts[row,column]
    low_x, high_x = wp.float64(wp.inf), wp.float64(-wp.inf)
    low_y, high_y = wp.float64(wp.inf), wp.float64(-wp.inf)
    for i in range(count):
        low_x,high_x = wp.min(low_x,vertices[row,column,i,0]),wp.max(high_x,vertices[row,column,i,0])
        low_y,high_y = wp.min(low_y,vertices[row,column,i,1]),wp.max(high_y,vertices[row,column,i,1])
        j = (i+1)%count
        dx = vertices[row,column,j,0]-vertices[row,column,i,0]
        dy = vertices[row,column,j,1]-vertices[row,column,i,1]
        length = wp.sqrt(dx*dx+dy*dy)
        ax,ay = wp.float64(0.),wp.float64(0.)
        low,high = wp.float64(-wp.inf),wp.float64(wp.inf)
        if length > wp.float64(1.e-9):
            ax,ay = -dy/length,dx/length
            low,high = wp.float64(wp.inf),wp.float64(-wp.inf)
            for k in range(count):
                value = ax*vertices[row,column,k,0]+ay*vertices[row,column,k,1]
                low,high = wp.min(low,value),wp.max(high,value)
        axes[row,column,i,0] = ax
        axes[row,column,i,1] = ay
        axes[row,column,i,2] = low
        axes[row,column,i,3] = high
    boxes[row,column,0] = low_x
    boxes[row,column,1] = high_x
    boxes[row,column,2] = low_y
    boxes[row,column,3] = high_y


# Isaac Sim 5.1 bundles Warp 1.8.2; use options supported there and standalone.
wp.set_module_options({"fuse_fp": False})


def _pack_layouts(plans, geometries, rules, device, statistics=None, *, keep_device=False):
    """Fill independent role/path plans without host round-trips per object.

    A plan is (IDs, role positions/yaws, access region, path, reserved sweeps,
    seed). Role/path geometry is transferred once; only progress polls and
    final poses return to the host during packing.
    """
    wp.init()
    n, width = len(plans), max(len(plan[0]) for plan in plans)
    reserved = [[rectangle_polygon(plan[3]), *plan[5]] for plan in plans]
    for plan, polygons in zip(plans, reserved):
        if rules.reserve_blocker_front:
            blocker = footprint_polygon(geometries[plan[0][1]], plan[1][plan[0][1]][:2], plan[2][plan[0][1]])
            polygons.append(rectangle_polygon(_blocker_front_region(blocker, rules)))
    obstacles = max(map(len,reserved))
    vertices_width = max(len(p) for polygons in reserved for p in polygons)
    poses, half = np.zeros((n,width,3)), np.zeros((n,width,2))
    # State columns: status (0 pending, 1 accepted, -1 exhausted), current
    # slot, round counter, consecutive misses, repair count, active count.
    placed, state = np.zeros((n,width),np.int32), np.zeros((n,6),np.int32)
    regions = np.asarray(shelf_regions(rules))
    areas, adjacent = np.zeros((n,len(regions))), np.zeros((n,width,width),np.int32)
    vertices = np.zeros((n,obstacles,vertices_width,2))
    axes = np.zeros((n,obstacles,vertices_width,4))
    boxes, counts = np.zeros((n,obstacles,4)), np.zeros((n,obstacles),np.int32)
    gaps, seeds = np.full((n,obstacles),rules.gap), np.empty(n,np.int32)
    for row, (plan, polygons) in enumerate(zip(plans,reserved)):
        ids, positions, yaws, *_ = plan
        seeds[row] = plan[-1] & 0x7fffffff
        state[row,1], state[row,5] = 2,len(ids)
        for slot,index in enumerate(ids):
            g = geometries[index]
            half[row,slot] = ((g.upper[0]-g.lower[0])/2,(g.upper[1]-g.lower[1])/2)
            if slot < 2:
                poses[row,slot] = (*positions[index][:2],yaws[index])
        placed[row,:2] = 1
        role_polygons = np.asarray([footprint_polygon(geometries[i],positions[i][:2],yaws[i]) for i in ids[:2]])
        areas[row] = _footprint_region_areas(role_polygons,regions).sum(axis=0)
        adjacent[row,0,1] = adjacent[row,1,0] = polygon_distance(tuple(map(tuple,role_polygons[0])),tuple(map(tuple,role_polygons[1]))) <= rules.neighbor_distance + _EPS
        if len(ids) == 2:
            metrics = _scene_measurements(tuple(tuple(map(tuple,p)) for p in role_polygons), rules)
            state[row,0] = 1 if (metrics.x_coverage >= rules.min_x_coverage
                and metrics.y_coverage >= rules.min_y_coverage
                and metrics.local_neighbor_fraction >= rules.min_neighbor_fraction
                and min(metrics.region_occupancies) >= rules.min_region_occupancy - _EPS) else -1
        for column, polygon in enumerate(polygons):
            p = np.asarray(polygon)
            k = len(p)
            vertices[row,column,:k], counts[row,column] = p,k
        if rules.reserve_blocker_front:
            gaps[row,len(polygons)-1] = 0.
    device = str(device)
    buffers = [torch.empty(a.shape,dtype=torch.float64,device=device) if i in (8,9)
               else torch.as_tensor(a,device=device) for i,a in enumerate(
               (poses,half,placed,state,seeds,regions,areas,vertices,axes,boxes,counts,gaps,adjacent))]
    p,h,on,st,seed,reg,ar,v,ax,box,co,ga,adj = [wp.from_torch(t) for t in buffers]
    candidates = wp.empty((n,64,3),dtype=wp.float64,device=device)
    scores = wp.empty((n,64),dtype=wp.float64,device=device)
    collisions = wp.empty((n,64),dtype=wp.int32,device=device)
    bounds = wp.vec4d(rules.x_range[0]+rules.edge_margin,rules.x_range[1]-rules.edge_margin,
                      rules.y_range[0]+rules.edge_margin,rules.y_range[1]-rules.edge_margin)
    cell_area = ((rules.x_range[1]-rules.x_range[0]-2*rules.edge_margin)
                 * (rules.y_range[1]-rules.y_range[0]-2*rules.edge_margin)/len(regions))
    stream = wp.stream_from_torch(torch.cuda.current_stream(device)) if torch.device(device).type == 'cuda' else None
    wp.launch(_layout_obstacle_geometry,dim=(n,obstacles),inputs=[v,co,ax,box],device=device,stream=stream)
    # Poll only once per 16 resident rounds. Finished rows are masked in both
    # kernels, independent of batch ordering or how slowly another row packs.
    for begin in range(0, 4*width+32, 16):
        for _ in range(16):
            wp.launch(_layout_candidates,dim=(n,64),inputs=[p,h,on,st,seed,reg,ar,v,ax,box,co,ga,bounds,
                rules.gap,rules.neighbor_distance,rules.min_region_occupancy,cell_area,candidates,scores,collisions],device=device,stream=stream)
            wp.launch(_layout_commit,dim=n,inputs=[p,h,on,st,reg,ar,adj,candidates,scores,collisions,bounds,
                rules.neighbor_distance,rules.min_neighbor_fraction,rules.min_region_occupancy,cell_area,
                rules.min_x_coverage,rules.min_y_coverage],device=device,stream=stream)
        status = buffers[3].cpu().numpy().copy()
        if np.all(status[:,0] != 0):
            break
    if statistics is not None:
        statistics['plans'] = statistics.get('plans',0)+n
        statistics['candidate_rounds'] = statistics.get('candidate_rounds',0)+int(status[:,2].sum())
        statistics['evictions'] = statistics.get('evictions',0)+int(status[:,4].sum())
        statistics['rejected_plans'] = statistics.get('rejected_plans',0)+int((status[:,0]!=1).sum())
    result = buffers[0].cpu().numpy()
    packed = [result[row,:len(plan[0])].copy() if status[row,0] == 1 else None for row,plan in enumerate(plans)]
    if keep_device:
        return [(value,buffers[0][row,:len(plans[row][0])]) if value is not None else None
                for row,value in enumerate(packed)]
    return packed


@njit(cache=True)
def _footprint_region_areas(polygons: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Exact Sutherland–Hodgman clipping of footprint batches against shelf cells."""
    areas = np.zeros((len(polygons), len(regions)))
    vertices, output = np.empty((12, 2)), np.empty((12, 2))
    for polygon in range(len(polygons)):
        for region in range(len(regions)):
            count = len(polygons[polygon])
            vertices[:count] = polygons[polygon]
            for plane in range(4):
                if count == 0:
                    break
                axis = plane // 2
                sign = 1. if plane % 2 == 0 else -1.
                bound = regions[region, plane]
                # Match scalar clipping's edge-length-scaled epsilon exactly.
                edge_length = (regions[region, 3] - regions[region, 2] if axis == 0
                               else regions[region, 1] - regions[region, 0])
                written = 0
                previous = count - 1
                for current in range(count):
                    pd = sign * (vertices[previous, axis] - bound)
                    cd = sign * (vertices[current, axis] - bound)
                    pi, ci = pd * edge_length >= -_EPS, cd * edge_length >= -_EPS
                    if pi != ci:
                        fraction = pd / (pd - cd)
                        for component in range(2):
                            output[written, component] = (vertices[previous, component] + fraction *
                                                          (vertices[current, component] - vertices[previous, component]))
                        written += 1
                    if ci:
                        output[written] = vertices[current]
                        written += 1
                    previous = current
                vertices[:written] = output[:written]
                count = written
            value = 0.
            for point in range(count):
                other = (point + 1) % count
                value += vertices[point, 0] * vertices[other, 1] - vertices[point, 1] * vertices[other, 0]
            areas[polygon, region] = abs(value) * .5
    return areas


@njit(cache=True)
def _blocker_y_interval(polygon, access, low, high, threshold):
    """Exact piecewise-quadratic overlap interval for a vertically shifted box.

    Breakpoints occur where a vertex (including an X-clipped edge) crosses a
    horizontal access boundary. Solve each quadratic once instead of rejecting
    entire role plans for insufficient footprint blockage.
    """
    heights = list(polygon[:, 1])
    for k in range(4):
        j = (k + 1) % 4
        for x in (access[0], access[1]):
            a, b = polygon[k, 0], polygon[j, 0]
            if min(a,b) < x < max(a,b):
                heights.append(polygon[k,1] + (x-a)/(b-a)*(polygon[j,1]-polygon[k,1]))
    knots = [low, high]
    for height in heights:
        for y in (access[2]-height, access[3]-height):
            if low < y < high:
                knots.append(y)
    knots.sort()
    samples = np.empty((3*(len(knots)-1),4,2))
    for interval in range(len(knots)-1):
        for part in range(3):
            shift = knots[interval]+(knots[interval+1]-knots[interval])*part/2
            for k in range(4):
                samples[3*interval+part,k,0] = polygon[k,0]
                samples[3*interval+part,k,1] = polygon[k,1]+shift
    values = _footprint_region_areas(samples, access.reshape(1,4))[:,0]-threshold
    lower, upper = math.inf, -math.inf
    for interval in range(len(knots)-1):
        f0,fm,f1 = values[3*interval:3*interval+3]
        a = 2*(f1+f0-2*fm)
        b, c = f1-f0-a, f0
        points = [0.,1.]
        if abs(a) < 1.e-13:
            if abs(b) > 1.e-13:
                root = -c/b
                if 0 < root < 1:
                    points.append(root)
        else:
            discriminant = b*b-4*a*c
            if discriminant >= 0:
                for root in ((-b-math.sqrt(discriminant))/(2*a),(-b+math.sqrt(discriminant))/(2*a)):
                    if 0 < root < 1:
                        points.append(root)
        points.sort()
        for k in range(len(points)-1):
            middle = (points[k]+points[k+1])/2
            if (a*middle+b)*middle+c >= -1.e-14:
                length = knots[interval+1]-knots[interval]
                lower = min(lower,knots[interval]+points[k]*length)
                upper = max(upper,knots[interval]+points[k+1]*length)
    # Avoid a roundoff-only crossing of a required-area boundary.
    if upper-lower > 2.e-8:
        lower += 1.e-8
        upper -= 1.e-8
    return lower,upper


@lru_cache(maxsize=8)
def _catalog_layout_data(geometries):
    types = {_asset_type(geometry) for geometry in geometries}
    groups = {asset: tuple(i for i, geometry in enumerate(geometries) if _asset_type(geometry) == asset)
              for asset in sorted(types)}
    areas = tuple((g.upper[0] - g.lower[0]) * (g.upper[1] - g.lower[1]) for g in geometries)
    return types, groups, areas


def sample_scene(geometries: Sequence[ObjectGeometry], rules: SpawnRules, rng: random.Random,
                 target_id: int | None = None, blocker_id: int | None = None,
                 active_ids: Sequence[int] | None = None) -> SceneSample:
    """Single CPU scene using the same constructive packer as batched resets."""
    generator = _scene_proposals(geometries, rules, rng, target_id, blocker_id, active_ids)
    value = None
    while True:
        try:
            plan = generator.send(value)
        except StopIteration as done:
            return done.value
        value = _pack_layouts([plan], geometries, rules, 'cpu')[0]


def sample_scenes(geometries, rules, streams, roles, active_ids, *, device="cpu", batch_size=512,
                  statistics=None, keep_device=False):
    """Independent role/path plans with resident, batched clutter construction.

    Only failed plans are resampled. Candidate generation, selection and
    bounded neighbor repair share device state. RNG depends on each plan's seed,
    never its row or the completion order of another environment.
    """
    if not (len(streams) == len(roles) == len(active_ids)) or batch_size < 1:
        raise ValueError("Layout requests need aligned seeds/roles/compositions and a positive batch size.")
    _check_inputs(geometries, rules)
    def generate(row):
        for _ in range(32):
            try:
                return (yield from _scene_proposals(geometries, rules, streams[row], *roles[row],
                                                    active_ids[row], checked=True, defer_paths=True))
            except RuntimeError:
                if active_ids[row] is not None:
                    raise
        raise RuntimeError("No feasible composition after 32 proposals.")
    results = [None] * len(streams)
    for begin in range(0, len(streams), batch_size):
        pending = list(range(begin, min(begin + batch_size, len(streams))))
        generators = {row: generate(row) for row in pending}
        values = {row: None for row in pending}
        while pending:
            rows, plans = [], []
            for row in pending:
                try:
                    plans.append(generators[row].send(values[row]))
                    rows.append(row)
                except StopIteration as done:
                    results[row] = done.value
                except RuntimeError as error:
                    results[row] = error
            if rows:
                paths = [(row,plan) for row,plan in zip(rows,plans) if isinstance(plan,_PathRequest)]
                layouts = [(row,plan) for row,plan in zip(rows,plans) if not isinstance(plan,_PathRequest)]
                values = {}
                if paths:
                    chosen = _choose_clearance_paths([plan for _,plan in paths],rules,device)
                    values.update(zip((row for row,_ in paths),chosen))
                if layouts:
                    packed = _pack_layouts([plan for _,plan in layouts],geometries,rules,device,statistics,
                                           keep_device=keep_device)
                    values.update(zip((row for row,_ in layouts),packed))
            pending = rows
    return results


def _scene_proposals(geometries: Sequence[ObjectGeometry], rules: SpawnRules, rng: random.Random,
                 target_id: int | None = None, blocker_id: int | None = None,
                 active_ids: Sequence[int] | None = None, *, checked=False, defer_paths=False):
    """Choose episode roles/count once, then sample whole-shelf clutter around them.

    The Target species appears only once. The requested count describes asset
    types; non-Target types are duplicated as needed to meet whole-shelf density,
    even in all_asset_types mode. Types, instance IDs and count stay fixed across
    layout retries. Passing ``active_ids`` with both explicit role IDs retains
    that exact composition and ID order while proposing new poses, so image
    visibility rejection cannot redraw the episode's types or instance count.
    A conservative finite
    SE(2) witness reserves rearrangement room but is not an actor goal or proof
    of reachable physical manipulation. Camera visibility must be checked by
    initial geometric perception; planar blockage is not a visibility test.
    """
    if not checked:
        _check_inputs(geometries, rules)
    types, groups, areas = _catalog_layout_data(tuple(geometries))
    fixed_ids = tuple(active_ids) if active_ids is not None else None
    if fixed_ids is not None:
        if target_id is None or blocker_id is None:
            raise ValueError("Fixed active IDs require explicit Target and Blocker IDs.")
        if any(type(index) is not int or not 0 <= index < len(geometries) for index in fixed_ids):
            raise ValueError("Fixed active IDs must index the catalog with integer IDs.")
        if len(set(fixed_ids)) != len(fixed_ids):
            raise ValueError("Fixed active IDs must be unique.")
        if not rules.min_objects <= len(fixed_ids) <= rules.max_objects:
            raise ValueError("Fixed active IDs must respect active object limits.")
        if target_id not in fixed_ids or blocker_id not in fixed_ids:
            raise ValueError("Target and Blocker must be present in fixed active IDs.")
    for index in (target_id, blocker_id):
        if index is not None and (not isinstance(index, int) or not 0 <= index < len(geometries)):
            raise ValueError("Explicit role IDs must index the catalog.")
    if target_id is not None and target_id == blocker_id:
        raise ValueError("Target and Blocker must be distinct instance IDs.")
    if target_id is None:
        choices = tuple(index for index, geometry in enumerate(geometries)
                        if blocker_id is None or _asset_type(geometry) != _asset_type(geometries[blocker_id]))
        if not choices:
            raise ValueError("Target requires an asset type different from Blocker.")
        target_id = rng.choice(choices)
    target_type = _asset_type(geometries[target_id])
    non_target = tuple(index for index, geometry in enumerate(geometries) if _asset_type(geometry) != target_type)
    if blocker_id is None:
        if not non_target:
            raise ValueError("Target uniqueness leaves no eligible Blocker.")
        blocker_id = rng.choice(non_target)
    if _asset_type(geometries[blocker_id]) == target_type:
        raise ValueError("Target asset type cannot also be the Blocker or a surrounding object.")
    target, blocker = target_id, blocker_id
    role_types = {target_type, _asset_type(geometries[blocker])}
    if fixed_ids is not None:
        if sum(_asset_type(geometries[index]) == target_type for index in fixed_ids) != 1:
            raise ValueError("Target asset type must occur exactly once among fixed active IDs.")
        selected_types = {_asset_type(geometries[index]) for index in fixed_ids}
        if not rules.min_asset_types <= len(selected_types) <= rules.max_asset_types:
            raise ValueError("Fixed active asset type count must respect asset type limits.")
        if rules.all_asset_types and selected_types != types:
            raise ValueError("Fixed all-asset-types scenes must contain every catalog asset type.")
    elif rules.all_asset_types:
        type_count = len(types)
        if not rules.min_asset_types <= type_count <= rules.max_asset_types:
            raise ValueError("All-asset-types count must satisfy asset type limits.")
        selected_types = types
    else:
        type_count = rng.randint(rules.min_asset_types, rules.max_asset_types)
        selected_types = role_types | set(rng.sample(sorted(types - role_types), type_count - len(role_types)))
    selected_ids = [target, blocker]
    if fixed_ids is not None:
        selected_ids.extend(index for index in fixed_ids if index not in (target, blocker))
    else:
        for asset in sorted(selected_types - role_types):
            selected_ids.append(rng.choice(groups[asset]))
    # Footprint area is invariant to yaw. Pick the episode's composition once,
    # before trying positions, until its summed area meets the requested density.
    # Add copies of the least-represented non-Target types, breaking ties at
    # random. Large footprints must not dominate the episode's composition.
    area = areas.__getitem__
    remaining = {
        asset: [index for index in groups[asset] if index not in selected_ids]
        for asset in sorted(selected_types - {target_type})
    }
    counts = {asset: sum(_asset_type(geometries[index]) == asset for index in selected_ids)
              for asset in remaining}
    if fixed_ids is None:
        for copies in remaining.values():
            rng.shuffle(copies)
    usable_area = ((rules.x_range[1] - rules.x_range[0] - 2 * rules.edge_margin)
                   * (rules.y_range[1] - rules.y_range[0] - 2 * rules.edge_margin))
    # Every equal-area cell meeting its minimum also requires that minimum
    # across the whole shelf. Honor this bound before fixing the composition.
    required_occupancy = max(rules.min_shelf_occupancy, rules.min_region_occupancy)
    required_area = required_occupancy * usable_area
    active_area = sum(area(index) for index in selected_ids)
    if fixed_ids is not None and active_area < required_area - _EPS:
        raise ValueError("Fixed active IDs cannot meet whole-shelf density: "
                         f"occupancy={active_area / usable_area:.4f}, required={required_occupancy:.4f}.")
    while len(selected_ids) < rules.min_objects or active_area < required_area - _EPS:
        available = [asset for asset, copies in remaining.items() if copies]
        if not available or len(selected_ids) >= rules.max_objects:
            raise RuntimeError("Selected asset types cannot meet whole-shelf density within the fixed instance cap. "
                               f"types={sorted(selected_types)}, instances={len(selected_ids)}, "
                               f"occupancy={active_area / usable_area:.4f}, "
                               f"required={required_occupancy:.4f}.")
        minimum = min(counts[asset] for asset in available)
        asset = rng.choice([asset for asset in available if counts[asset] == minimum])
        index = remaining[asset].pop()
        selected_ids.append(index)
        counts[asset] += 1
        active_area += area(index)
    if len(selected_ids) > rules.max_objects:
        raise ValueError("Selected asset type count exceeds the active instance cap.")
    # Larger footprints are placed first, preserving the two designated roles.
    ids = (target, blocker, *sorted(selected_ids[2:], key=area, reverse=True))
    if any(rules.shelf_top + geometries[index].upper[2] - geometries[index].lower[2]
           > rules.ceiling - rules.gap for index in ids):
        raise RuntimeError("Selected composition exceeds shelf ceiling clearance.")
    for _ in range(rules.max_attempts):
        yaws = {index: rng.uniform(-math.pi, math.pi) for index in (target, blocker)}
        extents = {index: _half_extents(geometries[index], yaws[index]) for index in (target, blocker)}
        target_hx, target_hy = extents[target]
        blocker_hx, blocker_hy = extents[blocker]
        pair_gap = rng.uniform(rules.gap + 0.002, rules.gap + 0.012)
        xmin = rules.x_range[0] + rules.edge_margin + target_hx
        xmax = rules.x_range[1] - rules.edge_margin - target_hx - 2 * blocker_hx - pair_gap
        half_width = rules.hand_width / 2 + rules.approach_margin
        y_extent = max(target_hy, half_width)
        ymin, ymax = rules.y_range[0] + rules.edge_margin + y_extent, rules.y_range[1] - rules.edge_margin - y_extent
        if xmin > xmax or ymin > ymax:
            continue
        target_x, target_y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        blocker_x = target_x + target_hx + blocker_hx + pair_gap
        blocker_ymin = max(rules.y_range[0] + rules.edge_margin + blocker_hy, target_y - half_width - blocker_hy)
        blocker_ymax = min(rules.y_range[1] - rules.edge_margin - blocker_hy, target_y + half_width + blocker_hy)
        if blocker_ymin > blocker_ymax:
            continue
        access_region = (target_x + target_hx + rules.approach_margin, rules.x_range[1],
                         target_y - half_width, target_y + half_width)
        if access_region[0] >= access_region[1]:
            continue
        local = np.asarray(footprint_polygon(geometries[blocker], (blocker_x, 0.), yaws[blocker]))
        blocker_ymin, blocker_ymax = _blocker_y_interval(local, np.asarray(access_region),
            blocker_ymin, blocker_ymax, rules.min_blocked_ratio * area(blocker))
        if blocker_ymin > blocker_ymax:
            continue
        blocker_y = rng.uniform(blocker_ymin, blocker_ymax)
        positions = {target: (target_x, target_y, rules.shelf_top - geometries[target].lower[2]),
                     blocker: (blocker_x, blocker_y, rules.shelf_top - geometries[blocker].lower[2])}
        polygons = {index: footprint_polygon(geometries[index], positions[index][:2], yaws[index])
                    for index in (target, blocker)}
        initial = PlanarPose(blocker_x, blocker_y, yaws[blocker])
        path, sweeps = None, None
        candidates = _clearance_candidates(geometries[blocker], initial, access_region, rules, rng)
        if defer_paths:
            path = yield _PathRequest(geometries[blocker],candidates,polygons[target])
            if path is not None:
                sweeps = _path_sweeps(geometries[blocker],path)
        else:
            for candidate in candidates:
                sweeps = _path_supported_clear(geometries[blocker], candidate, (polygons[target],), rules,
                                               return_sweeps=True)
                if sweeps is not None:
                    path = candidate
                    break
        if path is None:
            continue
        packed = yield (ids, positions, yaws, access_region, path, sweeps, rng.getrandbits(32))
        if packed is None:
            continue
        device_poses = None
        if isinstance(packed,tuple):
            packed, device_poses = packed
        for slot, index in enumerate(ids):
            positions[index] = (float(packed[slot, 0]), float(packed[slot, 1]),
                                rules.shelf_top - geometries[index].lower[2])
            yaws[index] = float(packed[slot, 2])
        kind = "pivot" if any(motion.pivot is not None for motion in path) else (
            "rotate_translate" if any(abs(motion.end.yaw - motion.start.yaw) > _EPS for motion in path)
            else "translation")
        sample = SceneSample(fixed_ids if fixed_ids is not None else ids,
                             target, blocker, positions, yaws, access_region,
                             (path[-1].end.x, path[-1].end.y), path, kind, device_poses, ids)
        # Construction enforces support, collision margins and cached global
        # metrics. validate_sample remains an independent test/debug oracle.
        return sample
    raise RuntimeError(
        f"Could not sample a whole-shelf, supported, collision-free, initially blocked and geometrically clearable scene "
        f"after {rules.max_attempts} attempts. Check fixed roles/count, catalog geometry, shelf size and spawn thresholds."
    )


def _scene_collision_boxes(env, sample) -> list[tuple]:
    """Conservative active-object and shelf boxes in the model's env_0 frame."""
    boxes = []
    origin = env.robot_pose_model.base_origin
    for index in sample.active_ids:
        geometry = env.spawn_geometries[index]
        lower, upper = np.array(geometry.lower), np.array(geometry.upper)
        half = (upper - lower) * 0.5
        center = np.array(sample.positions[index]) + origin
        center[2] += (lower[2] + upper[2]) * 0.5
        c, s = math.cos(sample.yaws[index]), math.sin(sample.yaws[index])
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        boxes.append((center, rotation, half, geometry.name))
    boxes.extend((box.center, box.rotation, box.half_extents, box.path)
                 for box in env.robot_pose_model.shelf_boxes)
    return boxes


def robot_start_collision_reason(env, sample, transforms: dict, obstacles=None) -> str | None:
    """Check scene, hand-arm/arm-arm clearance and a non-folded forearm."""

    model = env.robot_pose_model
    clearance = env.cfg.robot_start.collision_clearance
    obstacles = _scene_collision_boxes(env, sample) if obstacles is None else obstacles
    robot_boxes = []
    for box in model.colliders:
        transform = transforms[box.body_name]
        rotation = transform[:3, :3] @ box.rotation_local
        center = transform[:3, :3] @ box.center_local + transform[:3, 3]
        half = box.half_extents
        if center[2] - np.abs(rotation[2]) @ half < clearance:
            return f"ground: {box.path}"
        for other_center, other_rotation, other_half, name in obstacles:
            if not oriented_box_separated(center, rotation, half, other_center, other_rotation, other_half, clearance):
                return f"scene: {box.path} / {name}"
        robot_boxes.append((center, rotation, half))
    for first, second in model.self_collision_pairs():
        if not oriented_box_separated(*robot_boxes[first], *robot_boxes[second], 0.002):
            return f"robot self: {model.colliders[first].body_name} / {model.colliders[second].body_name}"
    _, elbow, wrist = model.arm_landmarks(transforms)
    if elbow[0] - wrist[0] < env.cfg.robot_start.min_forearm_forward:
        return "folded forearm: wrist is not sufficiently forward of elbow toward shelf (-X)"
    return None


def oriented_boxes_separated_batch(center_a, rotation_a, half_a, center_b, rotation_b, half_b, clearance=0.0):
    """Broadcasting 15-axis OBB SAT, including the scalar oracle's world axes."""
    delta = center_b - center_a
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    absolute = relative.abs()
    translation = (rotation_a.transpose(-1, -2) @ delta[..., None]).squeeze(-1)
    extent_world_a = (rotation_a.abs() @ half_a[..., None]).squeeze(-1)
    extent_world_b = (rotation_b.abs() @ half_b[..., None]).squeeze(-1)
    separated = (delta.abs() + _EPS >= extent_world_a + extent_world_b + clearance).any(-1)
    radius_a = half_a + (absolute @ half_b[..., None]).squeeze(-1)
    radius_b = half_b + (absolute.transpose(-1, -2) @ half_a[..., None]).squeeze(-1)
    separated |= (translation.abs() + _EPS >= radius_a + clearance).any(-1)
    projection_b = (relative.transpose(-1, -2) @ translation[..., None]).squeeze(-1)
    separated |= (projection_b.abs() + _EPS >= radius_b + clearance).any(-1)
    # Cross axes A_i x B_j. Cyclic indices avoid materializing 15 world vectors
    # for every robot-collider/object pair, bounding scratch GPU memory.
    cyclic1, cyclic2 = [1, 2, 0], [2, 0, 1]
    distance = (translation[..., cyclic2, None] * relative[..., cyclic1, :]
                - translation[..., cyclic1, None] * relative[..., cyclic2, :]).abs()
    radius = (half_a[..., cyclic1, None] * absolute[..., cyclic2, :]
              + half_a[..., cyclic2, None] * absolute[..., cyclic1, :]
              + half_b[..., None, cyclic1] * absolute[..., :, cyclic2]
              + half_b[..., None, cyclic2] * absolute[..., :, cyclic1])
    # Compute the cross-product norm directly; sqrt(1-dot^2) loses precision
    # for almost parallel axes and could falsely admit a touching collider.
    axes_a = rotation_a.transpose(-1, -2)
    axes_b = rotation_b.transpose(-1, -2)
    length = torch.linalg.cross(axes_a[..., :, None, :], axes_b[..., None, :, :]).norm(dim=-1)
    separated |= ((length > 1.e-9) & (distance + _EPS * length >= radius + clearance * length)).any((-2, -1))
    return separated


def _robot_candidate_clearance(env, frames, joints, tcp, obstacle_centers, obstacle_rotations, obstacle_half,
                               obstacle_valid):
    """Evaluate all authored colliders and high-elbow geometry on the device."""
    model, rules = env.robot_pose_model, env.cfg.robot_start
    constants = model._tensor_constants(frames.device, frames.dtype)
    colliders = frames[:, constants["collider_owners"]] @ constants["collider_offsets"]
    centers, rotations = colliders[..., :3, 3], colliders[..., :3, :3]
    half = constants["collider_half"]
    bottom = centers[..., 2] - (rotations[..., 2, :].abs() * half).sum(-1)
    valid = (bottom >= rules.collision_clearance).all(-1)
    separated = oriented_boxes_separated_batch(
        centers[:, :, None], rotations[:, :, None], half[None, :, None],
        obstacle_centers[:, None], obstacle_rotations[:, None], obstacle_half[:, None], rules.collision_clearance)
    valid &= (separated | ~obstacle_valid[:, None]).all((-2, -1))
    pairs = constants["self_pairs"]
    if pairs.shape[0]:
        valid &= oriented_boxes_separated_batch(
            centers[:, pairs[:, 0]], rotations[:, pairs[:, 0]], half[pairs[:, 0]],
            centers[:, pairs[:, 1]], rotations[:, pairs[:, 1]], half[pairs[:, 1]], .002).all(-1)
    shoulder, elbow, wrist = joints[:, 1, :3, 3], joints[:, 2, :3, 3], joints[:, 3, :3, 3]
    chord = wrist - shoulder
    fraction = ((elbow - shoulder) * chord).sum(-1) / chord.square().sum(-1).clamp_min(1.e-18)
    height = elbow[:, 2] - shoulder[:, 2] - fraction.clamp(0, 1) * chord[:, 2]
    valid &= (height >= rules.min_elbow_height) & torch.isfinite(colliders).all((-3, -2, -1))
    valid &= elbow[:, 0] - wrist[:, 0] >= rules.min_forearm_forward
    return valid, height


def _scene_boxes_tensor(env, samples, proposals):
    """Compact active OBBs from device poses, with cached catalog/shelf geometry."""
    proposed, _, slots = proposals
    model = env.robot_pose_model
    constants = model._tensor_constants(proposed.device,proposed.dtype)
    if "scene_half" not in constants:
        lower = proposed.new_tensor([g.lower for g in env.spawn_geometries])
        upper = proposed.new_tensor([g.upper for g in env.spawn_geometries])
        constants["scene_half"] = (upper-lower)*.5
        constants["scene_center_z"] = (upper[:,2]+lower[:,2])*.5
        constants["shelf_centers"] = proposed.new_tensor(np.asarray([b.center for b in model.shelf_boxes])).reshape(-1,3)
        constants["shelf_rotations"] = proposed.new_tensor(np.asarray([b.rotation for b in model.shelf_boxes])).reshape(-1,3,3)
        constants["shelf_half"] = proposed.new_tensor(np.asarray([b.half_extents for b in model.shelf_boxes])).reshape(-1,3)
    ids = slots[:,:max(len(sample.active_ids) for sample in samples)]
    lookup = ids.clamp_min(0)
    selected = proposed.gather(1,lookup[...,None].expand(-1,-1,4))
    centers = selected[...,:3]+proposed.new_tensor(model.base_origin)
    centers[...,2] += constants["scene_center_z"][lookup]
    c,s = selected[...,3].cos(),selected[...,3].sin()
    zero,one = torch.zeros_like(c),torch.ones_like(c)
    rotations = torch.stack((c,-s,zero,s,c,zero,zero,zero,one),-1).reshape(*c.shape,3,3)
    n = len(samples)
    return (torch.cat((centers,constants["shelf_centers"][None].expand(n,-1,-1)),1),
            torch.cat((rotations,constants["shelf_rotations"][None].expand(n,-1,-1,-1)),1),
            torch.cat((constants["scene_half"][lookup],constants["shelf_half"][None].expand(n,-1,-1)),1),
            torch.cat((ids >= 0,torch.ones((n,len(model.shelf_boxes)),device=proposed.device,dtype=torch.bool)),1))


@torch.no_grad()
def _sample_robot_starts(env, samples, seeds, *, proposals=None):
    """Sample continuous starts with bounded batched GPU IK and OBB acceptance.

    Returns one device-tensor start tuple or ``None`` per requested scene. Failed
    rows can be resampled without discarding valid neighbors. Episode RNG and
    metadata preparation plus bounded acceptance indices use host arrays; no
    per-environment nonlinear solver or live physics trial is used.
    """
    if len(samples) != len(seeds):
        raise ValueError("Every scene requires one reproducible robot-start seed.")
    if not samples:
        return []
    model, rules = env.robot_pose_model, env.cfg.robot_start
    device, dtype = env.device, torch.float32
    count = len(samples)
    if not isinstance(rules.ik_batch_size, int) or isinstance(rules.ik_batch_size, bool) or rules.ik_batch_size <= 0:
        raise ValueError("Robot start ik_batch_size must be a positive integer.")
    regions_cpu = np.asarray([robot_start_region(sample, env.spawn_geometries, env.spawn_rules, rules)
                              for sample in samples])
    regions = torch.as_tensor(regions_cpu, device=device, dtype=dtype)
    lower, upper = regions[:, (0, 2, 4)], regions[:, (1, 3, 5)]
    origin = torch.as_tensor(model.base_origin, device=device, dtype=dtype)
    rotation = torch.tensor(ROBOT_START_TCP_ROTATION, device=device, dtype=dtype)
    constants = model._tensor_constants(device, dtype)
    # Local generators keep each episode independent of reset subset/order.
    noise = torch.as_tensor(np.stack([np.random.default_rng(int(seed) ^ 0x5D2EA91C).random(
        (rules.max_attempts, 9), dtype=np.float32) for seed in seeds]), device=device)
    if proposals is None:
        proposals = _episode_proposals(samples,len(env.spawn_geometries),device,dtype)
    centers,rotations,half,obstacle_valid = _scene_boxes_tensor(env,samples,proposals)
    result = torch.zeros((count, len(model.joint_names) + 5), device=device, dtype=dtype)
    accepted = [False] * count
    # Bound pairwise collision scratch independently of the selected count.
    # Larger batches amortize GPU launches; lowering the cap trades throughput
    # for scratch memory without changing episode candidate random streams.
    proposals = min(8, rules.max_attempts, rules.ik_batch_size)
    group_size = max(1, rules.ik_batch_size // proposals)
    for begin in range(0, count, group_size):
        pending = list(range(begin, min(count, begin + group_size)))
        for offset in range(0, rules.max_attempts, proposals):
            if not pending:
                break
            width = min(proposals, rules.max_attempts - offset)
            rows = torch.tensor(pending, device=device, dtype=torch.long)
            random_values = noise[rows, offset:offset + width]
            target = lower[rows, None] + random_values[..., :3] * (upper - lower)[rows, None] + origin
            arm = constants["default"][constants["arm_ids"]].expand(len(pending), width, -1).clone()
            branch = torch.arange(offset, offset + width, device=device) % 4
            lifts = arm.new_tensor((-2.0, -1.6, -.6, -1.0))[branch]
            elbows = arm.new_tensor((1.8, 1.2, -1.8, -1.2))[branch]
            arm[..., 1], arm[..., 2], arm[..., 3] = lifts, elbows, -lifts - elbows
            # Candidate-index strength preserves the same seeds even when a
            # small memory cap splits the usual groups of eight proposals.
            strength = (torch.arange(offset, offset + width, device=device) // 8).clamp_max(2).to(dtype)
            arm += (random_values[..., 3:] - .5) * strength[None, :, None]
            q, valid, frames, joints, tcp = model.solve_batch(
                target.reshape(-1, 3), rotation.expand(len(pending) * width, -1, -1), arm.reshape(-1, 6))
            obstacle_rows = rows.repeat_interleave(width)
            clear, heights = _robot_candidate_clearance(
                env, frames, joints, tcp, centers[obstacle_rows],rotations[obstacle_rows],
                half[obstacle_rows], obstacle_valid[obstacle_rows])
            actual = tcp[:, :3, 3] - origin
            valid &= clear & (actual >= lower[obstacle_rows]).all(-1) & (actual <= upper[obstacle_rows]).all(-1)
            valid = valid.reshape(len(pending), width)
            first = torch.where(valid, torch.arange(width, device=device), width).amin(-1)
            indices = first.cpu().tolist()  # One bounded acceptance transfer per proposal batch.
            selected = [index for index, proposal in enumerate(indices) if proposal < width]
            if selected:
                local = torch.tensor(selected, device=device)
                chosen = local * width + first[local]
                destination = rows[local]
                result[destination, :-5] = q[chosen]
                result[destination, -5:-2] = actual[chosen]
                result[destination, -2] = heights[chosen]
                result[destination, -1] = (first[local] + offset + 1).to(dtype)
                for index in selected:
                    accepted[pending[index]] = True
            pending = [row for row in pending if not accepted[row]]
    return [(result[row, :-5],regions[row],result[row, -5:-2],result[row, -2],result[row, -1])
            if accepted[row] else None for row in range(count)]


@torch.no_grad()
def _check_physical_robot_starts(env, samples, starts, poses, origins):
    """Verify every physical link against already collision-checked GPU FK.

    PhysX link quaternions use xyzw here. Checking the entire articulation
    replaces repeating CPU collider SAT after the batched state write.
    """
    if len(samples) != len(starts) or poses.shape[0] != len(starts):
        raise ValueError("Physical reset validation batch sizes differ.")
    if not starts:
        return torch.empty(0, device=poses.device, dtype=torch.bool)
    model = env.robot_pose_model
    output = []
    for begin in range(0, len(starts), 128):
        selected = starts[begin:begin + 128]
        q = torch.stack([torch.as_tensor(start[0],device=poses.device,dtype=poses.dtype) for start in selected])
        expected = model.forward_batch(q)
        actual = poses[begin:begin + len(selected)]
        positions = actual[..., :3] - origins[begin:begin + len(selected), None] + poses.new_tensor(model.base_origin)
        x, y, z, w = actual[..., 3:].unbind(-1)
        rotations = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                                 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                                 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*x.shape, 3, 3)
        match = ((positions - expected[..., :3, 3]).norm(dim=-1) <= 2.e-4).all(-1)
        match &= ((rotations - expected[..., :3, :3]).norm(dim=(-2, -1)) <= 3.e-4).all(-1)
        output.append(match & torch.isfinite(actual).all((-2, -1)))
    return torch.cat(output)


def _sample_robot_start(env, sample, seed: int) -> tuple:
    """Sample only in free space ahead of the Blocker; never step trial poses."""
    rules, model = env.cfg.robot_start, env.robot_pose_model
    region = robot_start_region(sample, env.spawn_geometries, env.spawn_rules, rules)
    lower, upper = np.array(region)[[0, 2, 4]], np.array(region)[[1, 3, 5]]
    origin = env.robot_pose_model.base_origin
    orientation = np.asarray(ROBOT_START_TCP_ROTATION)
    rng = random.Random(seed ^ 0x5D2EA91C)
    failures = {}
    obstacles = _scene_collision_boxes(env, sample)
    for attempt in range(1, rules.max_attempts + 1):
        position = np.array([rng.uniform(lo, hi) for lo, hi in zip(lower, upper)]) + origin
        reason = "unreachable elbow-up IK"

        def accept(transforms):
            nonlocal reason
            actual = transforms[model.tcp_body_name][:3, 3] - origin
            if np.any(actual < lower) or np.any(actual > upper):
                reason = "TCP outside start region"
            else:
                reason = robot_start_collision_reason(env, sample, transforms, obstacles)
            return reason is None

        q = model.solve(position, orientation, rng, min_elbow_height=rules.min_elbow_height,
                        accept_candidate=accept)
        if q is not None:
            transforms = model.forward(q)
            actual = transforms[model.tcp_body_name][:3, 3] - origin
            return q, region, actual, model.elbow_height(transforms), attempt
        failures[reason] = failures.get(reason, 0) + 1
    raise RuntimeError(
        f"No collision-free elbow-up robot start in Blocker-front region after {rules.max_attempts} attempts: "
        f"region={region}, failures={failures}."
    )



# Native reset event. Each selected row gets a newly sampled episode. USD
# inspection is shared; initial perception queries shared surface meshes.

@dataclass
class EpisodeBatch:
    object_state: torch.Tensor       # M,P,13; environment-local link pose / zero COM velocity
    object_ids: torch.Tensor         # M,40; active pool IDs, -1 padding
    active: torch.Tensor             # M,P
    target_id: torch.Tensor          # M
    blocker_id: torch.Tensor         # M
    access_region: torch.Tensor      # M,4
    joint_positions: torch.Tensor    # M,J; checked complete articulation configurations
    geometry_b: torch.Tensor         # M,40,10; independently rendered initial geometry
    geometry_valid: torch.Tensor     # M,40
    observed_ids: torch.Tensor       # M,40
    robot_regions: torch.Tensor      # M,6
    robot_tcp: torch.Tensor          # M,3; environment coordinates
    visibility: torch.Tensor         # M,2; target, blocker
    camera_pose: torch.Tensor        # M,7; world frame, wxyz
    camera_intrinsics: torch.Tensor  # M,3,3
    seeds: torch.Tensor              # M; reproducible per-environment episode seed


def pack_object_states(default_state, sample, lower, upper, upright):
    """Single-proposal convenience wrapper around the batched pose conversion."""
    return tuple(value[0] for value in pack_episode_states(default_state, [sample], lower, upper, upright))


def _episode_proposals(samples, p, device, dtype):
    """Keep XY/yaw on device; only IDs and static support heights come from metadata."""
    m = len(samples)
    active = np.zeros((m, p), dtype=bool)
    slots = np.full((m, 40), -1, dtype=np.int64)
    for row, sample in enumerate(samples):
        ids = list(sample.active_ids)
        active[row, ids], slots[row, :len(ids)] = True, ids
    if samples and all(sample.planar_poses is not None for sample in samples):
        pairs = [(row,index) for row,s in enumerate(samples) for index in s.planar_ids]
        indices = torch.tensor(pairs,device=device,dtype=torch.long)
        heights = torch.tensor([s.positions[index][2] for s in samples for index in s.planar_ids],
                               device=device,dtype=dtype)
        planar = torch.cat([s.planar_poses for s in samples]).to(device=device,dtype=dtype)
        values = torch.stack((planar[:,0],planar[:,1],heights,planar[:,2]),-1)
        proposed = torch.zeros((m,p,4),device=device,dtype=dtype)
        proposed[indices[:,0],indices[:,1]] = values
    else:
        proposed = np.zeros((m,p,4),dtype=np.float32)
        for row,sample in enumerate(samples):
            ids = list(sample.active_ids)
            proposed[row,ids,:3] = [sample.positions[index] for index in ids]
            proposed[row,ids,3] = [sample.yaws[index] for index in ids]
        proposed = torch.as_tensor(proposed,device=device,dtype=dtype)
    return proposed,torch.as_tensor(active,device=device),torch.as_tensor(slots,device=device)


def pack_episode_states(default_state, samples, lower, upper, upright, *, proposals=None):
    """Vectorize off-center/upright conversion without downloading device layouts."""
    m, p = len(samples), default_state.shape[0]
    proposed, active, slots = proposals if proposals is not None else _episode_proposals(
        samples,p,default_state.device,default_state.dtype)
    rotation = torch.zeros((m, p, 4), device=default_state.device, dtype=default_state.dtype)
    rotation[..., 0], rotation[..., 3] = torch.cos(proposed[..., 3] * .5), torch.sin(proposed[..., 3] * .5)
    center = (lower + upper) * .5
    center[:, 2] = 0  # Sampler Z is root height; only XY describes the footprint center.
    states = default_state.unsqueeze(0).expand(m, -1, -1).clone()
    states[..., :3] = torch.where(active[..., None], proposed[..., :3] - quat_apply(rotation, center[None]), states[..., :3])
    states[..., 3:7] = torch.where(active[..., None], quat_mul(rotation, upright[None]), states[..., 3:7])
    states[..., 7:] = 0
    return states, active, slots


def _episode_seed(base_seed, env_id, reset_count):
    """A row's random stream is independent of reset batching and ordering."""
    return random.Random(f"{base_seed}:{env_id}:{reset_count}").randrange(2**63 - 1)


def _initialize_episode_reset(env):
    """Inspect env_0 and bind fixed native resources once, outside the step path."""
    if hasattr(env, "_episode_reset"):
        return env._episode_reset
    from ..config.ur5e.scene import (
        OBJECT_SPECS, RobotPoseModel, inspect_geometry, FORCE_TORQUE_BODY_NAME,
    )
    from .sensors import SensorReader, InitialVisionSensor

    if not env.cfg.robot_start.enabled:
        raise ValueError("Episode starts require collision-checked elbow-up robot randomization.")
    objects, robot = env.scene["object_collection"], env.scene["robot"]
    if tuple(objects.object_names) != tuple(spec.name for spec in OBJECT_SPECS):
        raise ValueError("The native object collection order must match the static pool catalog.")
    env.spawn_geometries, env.spawn_rules = inspect_geometry(env)
    env.robot_pose_model = RobotPoseModel.from_env(env)
    env.robot_pose_model.base_origin = env.scene.env_origins[0].detach().cpu().numpy()
    lower = torch.tensor([geometry.lower for geometry in env.spawn_geometries], device=env.device)
    upper = torch.tensor([geometry.upper for geometry in env.spawn_geometries], device=env.device)
    upright = lower.new_tensor([spec.upright_quat for spec in OBJECT_SPECS])
    link_paths = robot.root_physx_view.link_paths[0]
    distal = tuple(name for name, path in zip(robot.body_names, link_paths)
                   if "/ROASRightHand/" in str(path) or name == FORCE_TORQUE_BODY_NAME)
    if FORCE_TORQUE_BODY_NAME not in distal or not any("force_sensor" in name for name in distal):
        raise RuntimeError("Cannot determine the physical hand-side F/T compensation subtree.")
    n = env.num_envs
    env.task_scene_data = SceneBatch(
        object_ids=torch.full((n, 40), -1, device=env.device, dtype=torch.long),
        target_id=torch.zeros(n, device=env.device, dtype=torch.long),
        blocker_id=torch.ones(n, device=env.device, dtype=torch.long),
        geometry_b=torch.zeros((n, 40, 10), device=env.device),
        geometry_valid=torch.zeros((n, 40), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((n, 40), -1, device=env.device, dtype=torch.long),
        lower=lower, upper=upper, upright=upright,
        access_region=torch.zeros((n, 4), device=env.device),
        workspace_bounds=lower.new_tensor((*env.spawn_rules.x_range, *env.spawn_rules.y_range)).expand(n, -1),
        shelf_top=lower.new_full((n,), env.spawn_rules.shelf_top),
        ceiling=lower.new_full((n,), env.spawn_rules.ceiling),
        sensor_body_name=FORCE_TORQUE_BODY_NAME, distal_body_names=distal,
    )
    objects.parked_mask = torch.ones((n, len(OBJECT_SPECS)), dtype=torch.bool, device=env.device)
    objects.gravity = lower.new_tensor(env.cfg.sim.gravity)
    env.task_sensor_reader = SensorReader(env)
    runtime = SimpleNamespace(
        specs=OBJECT_SPECS, vision=InitialVisionSensor(env),
        reset_counts=torch.zeros(n, dtype=torch.long, device=env.device),
        robot_regions=torch.zeros((n, 6), device=env.device),
        robot_tcp=torch.zeros((n, 3), device=env.device),
        visibility=torch.zeros((n, 2), device=env.device),
        camera_pose=torch.zeros((n, 7), device=env.device),
        camera_intrinsics=torch.zeros((n, 3, 3), device=env.device),
        episode_seeds=torch.zeros(n, dtype=torch.long, device=env.device),
    )
    env._episode_reset = runtime
    return runtime


def _write_physical_reset(env, ids, object_state, joint_positions, active):
    """One tensor write per native asset, independent of selected environment count."""
    objects, robot = env.scene["object_collection"], env.scene["robot"]
    states = object_state.clone()
    states[..., :3] += env.scene.env_origins[ids, None, :]
    objects.write_object_state_to_sim(states, env_ids=ids)
    objects.parked_mask[ids] = ~active
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root, env_ids=ids)
    robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions), env_ids=ids)
    robot.set_joint_position_target(joint_positions, env_ids=ids)
    robot.set_joint_velocity_target(torch.zeros_like(joint_positions), env_ids=ids)
    robot.set_joint_effort_target(torch.zeros_like(joint_positions), env_ids=ids)


def _check_physical_robot_start(env, sample, expected, poses, origin):
    """Compare selected PhysX links in the canonical USD model's coordinates."""
    from scipy.spatial.transform import Rotation

    model = env.robot_pose_model
    transforms = {}
    for index, name in enumerate(model.body_names):
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(poses[index, 3:]).as_matrix()
        matrix[:3, 3] = poses[index, :3] - origin + model.base_origin
        transforms[name] = matrix
    tcp = transforms[model.tcp_body_name]
    if np.linalg.norm(tcp[:3, 3] - model.base_origin - expected[2]) > 2.e-4:
        raise RuntimeError("Physical hand pose differs from the collision-checked reset kinematics.")
    if Rotation.from_matrix(np.asarray(ROBOT_START_TCP_ROTATION) @ tcp[:3, :3].T).magnitude() > .011:
        raise RuntimeError("Physical hand does not face the shelf with the agreed tool roll.")
    if model.elbow_height(transforms) < env.cfg.robot_start.min_elbow_height - 1.e-4:
        raise RuntimeError("Physical reset produced an elbow-down/low-elbow pose.")
    reason = robot_start_collision_reason(env, sample, transforms)
    if reason is not None:
        raise RuntimeError(f"Physical robot reset failed collision clearance: {reason}")


def _build_episode_batch(env, runtime, ids):
    """Prepare distinct episodes for selected rows without rendering.

    Layouts retain independent role/path seeds and device-resident clutter
    packing/repair. IK, pose conversion and PhysX writes are batched.
    Initial perception uses bounded geometric ray batches without integrating
    physics; unselected environments are never used as a staging area.
    """
    objects, robot = env.scene["object_collection"], env.scene["robot"]
    scene, specs, m = env.task_scene_data, runtime.specs, ids.numel()
    selected = ids.tolist()
    counts = runtime.reset_counts[ids].tolist()
    seeds = [_episode_seed(int(env.cfg.seed), index, count) for index, count in zip(selected, counts)]
    streams = [random.Random(seed) for seed in seeds]
    targets = [index for index, spec in enumerate(specs)
               if (env.cfg.target_asset is None or spec.asset_name == env.cfg.target_asset)
               and (env.cfg.blocker_asset is None or spec.asset_name != env.cfg.blocker_asset)]
    if not targets:
        raise ValueError("No Target type satisfies the configured episode roles.")
    roles = []
    for rng in streams:
        target = rng.choice(targets)
        blockers = [index for index, spec in enumerate(specs) if spec.asset_name != specs[target].asset_name
                    and (env.cfg.blocker_asset is None or spec.asset_name == env.cfg.blocker_asset)]
        if not blockers:
            raise ValueError("Blocker must have a different type from the unique Target.")
        roles.append((target, rng.choice(blockers)))
    saved = (objects.data.object_state_w[ids].clone(), robot.data.root_state_w[ids].clone(),
             robot.data.joint_pos[ids].clone(), robot.data.joint_vel[ids].clone(),
             robot.data.joint_pos_target[ids].clone(), robot.data.joint_vel_target[ids].clone(),
             robot.data.joint_effort_target[ids].clone(), objects.parked_mask[ids].clone())
    origins = env.scene.env_origins[ids]
    default = objects.data.default_object_state[0]
    batch = EpisodeBatch(
        object_state=default.new_zeros((m, len(specs), 13)),
        object_ids=torch.full((m, 40), -1, device=env.device, dtype=torch.long),
        active=torch.zeros((m, len(specs)), device=env.device, dtype=torch.bool),
        target_id=torch.tensor([role[0] for role in roles], device=env.device),
        blocker_id=torch.tensor([role[1] for role in roles], device=env.device),
        access_region=default.new_zeros((m, 4)), joint_positions=torch.zeros_like(saved[2]),
        geometry_b=default.new_zeros((m, 40, 10)),
        geometry_valid=torch.zeros((m, 40), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((m, 40), -1, device=env.device, dtype=torch.long),
        robot_regions=default.new_zeros((m, 6)), robot_tcp=default.new_zeros((m, 3)),
        visibility=default.new_zeros((m, 2)), camera_pose=default.new_zeros((m, 7)),
        camera_intrinsics=default.new_zeros((m, 3, 3)),
        seeds=torch.tensor(seeds, device=env.device, dtype=torch.long),
    )
    fixed_ids, errors = [None] * m, [None] * m
    pending = list(range(m))
    try:
        with runtime.vision.capture_session():
            for attempt in range(env.cfg.initial_vision.max_layout_attempts):
                candidates, samples = [], []
                # print(f"[Episode reset] sampling {len(pending)} independent episodes, proposal {attempt + 1}", flush=True)
                proposals = sample_scenes(
                    env.spawn_geometries, env.spawn_rules, [streams[row] for row in pending],
                    [roles[row] for row in pending], [fixed_ids[row] for row in pending], device=env.device,
                    keep_device=True)
                for row, sample in zip(pending, proposals):
                    if isinstance(sample, RuntimeError):
                        errors[row] = str(sample)
                        continue
                    fixed_ids[row] = sample.active_ids
                    candidates.append(row)
                    samples.append(sample)
                if not candidates:
                    continue
                proposed = _episode_proposals(samples,len(specs),env.device,default.dtype)
                starts = _sample_robot_starts(env, samples, [seeds[row] + 7_919 * (attempt + 1) for row in candidates],
                                              proposals=proposed)
                feasible = [index for index, start in enumerate(starts) if start is not None]
                for row, start in zip(candidates, starts):
                    if start is None:
                        errors[row] = "No collision-free elbow-up pose in the bounded batched IK search."
                candidates = [candidates[index] for index in feasible]
                samples = [samples[index] for index in feasible]
                starts = [starts[index] for index in feasible]
                if not candidates:
                    continue
                rows = torch.tensor(candidates, device=env.device, dtype=torch.long)
                candidate_ids = ids[rows]
                selection = torch.tensor(feasible,device=env.device,dtype=torch.long)
                states, active, slots = pack_episode_states(default, samples, scene.lower, scene.upper, scene.upright,
                                                           proposals=tuple(t[selection] for t in proposed))
                positions = torch.stack([torch.as_tensor(start[0],device=env.device,dtype=default.dtype) for start in starts])
                batch.object_state[rows], batch.active[rows], batch.object_ids[rows] = states, active, slots
                batch.joint_positions[rows] = positions
                batch.access_region[rows] = default.new_tensor([sample.access_region for sample in samples])
                batch.robot_regions[rows] = torch.stack([torch.as_tensor(start[1],device=env.device,dtype=default.dtype) for start in starts])
                batch.robot_tcp[rows] = torch.stack([torch.as_tensor(start[2],device=env.device,dtype=default.dtype) for start in starts])
                _write_physical_reset(env, candidate_ids, states, positions, active)
                env.sim.forward()  # FK only, never step or flush all environments' force buffers.
                link_poses = robot.root_physx_view.get_link_transforms()[candidate_ids]
                physical_valid = _check_physical_robot_starts(env, samples, starts, link_poses, origins[rows]).tolist()
                requests, vision_rows = [], []
                for row, sample, valid in zip(candidates, samples, physical_valid):
                    if not valid:
                        errors[row] = "Physical articulation differs from collision-checked batch FK."
                        continue
                    requests.append(dict(
                        env_id=selected[row], active_ids=sample.active_ids,
                        target_id=sample.target_id, blocker_id=sample.blocker_id,
                        workspace_bounds=(*env.spawn_rules.x_range, *env.spawn_rules.y_range),
                        shelf_top=env.spawn_rules.shelf_top, seed=seeds[row] + 7_919 * (attempt + 1)))
                    vision_rows.append(row)
                snapshots = runtime.vision.capture_batch(requests)
                accepted = set()
                accepted_rows, accepted_snapshots = [], []
                for row, snapshot in zip(vision_rows, snapshots):
                    if snapshot is None:
                        errors[row] = "Initial geometric visibility/depth requirements were not met."
                    else:
                        accepted.add(row)
                        accepted_rows.append(row)
                        accepted_snapshots.append(snapshot)
                if accepted_rows:
                    accepted_ids = torch.tensor(accepted_rows, device=env.device, dtype=torch.long)
                    for name, source in (("geometry_b", "geometry_b"), ("geometry_valid", "valid"),
                                         ("observed_ids", "observed_ids"), ("camera_pose", "camera_pose_w"),
                                         ("camera_intrinsics", "intrinsics")):
                        getattr(batch, name)[accepted_ids] = torch.stack([getattr(s, source) for s in accepted_snapshots])
                    batch.visibility[accepted_ids] = default.new_tensor([s.visible_fractions for s in accepted_snapshots])
                pending = [row for row in pending if row not in accepted]
                if not pending:
                    return batch
            failures = [(selected[row], errors[row]) for row in pending[:3]]
            raise RuntimeError(f"Independent episode preparation exhausted retries: {failures}")
    except BaseException:
        # Roll back only selected rows if preparation fails. Keep all episode
        # metadata/counters uncommitted and preserve neighboring physical state.
        objects.write_object_state_to_sim(saved[0], env_ids=ids)
        robot.write_root_state_to_sim(saved[1], env_ids=ids)
        robot.write_joint_state_to_sim(saved[2], saved[3], env_ids=ids)
        robot.set_joint_position_target(saved[4], env_ids=ids)
        robot.set_joint_velocity_target(saved[5], env_ids=ids)
        robot.set_joint_effort_target(saved[6], env_ids=ids)
        objects.parked_mask[ids] = saved[7]
        env.sim.forward()
        raise


def apply_episode_batch(env, ids, batch):
    """Commit independent episodes by selected row, without host reads or env loops."""
    runtime, scene = env._episode_reset, env.task_scene_data
    _write_physical_reset(env, ids, batch.object_state, batch.joint_positions, batch.active)
    for name in ("object_ids", "target_id", "blocker_id", "access_region", "geometry_b", "geometry_valid", "observed_ids"):
        getattr(scene, name)[ids] = getattr(batch, name)
    for name in ("robot_regions", "visibility", "camera_pose", "camera_intrinsics"):
        getattr(runtime, name)[ids] = getattr(batch, name)
    runtime.robot_tcp[ids] = batch.robot_tcp + env.scene.env_origins[ids]
    runtime.episode_seeds[ids] = batch.seeds
    runtime.reset_counts[ids] += 1
    env.task_sensor_reader.reset(ids)
    env.sim.forward()


@torch.no_grad()
def reset_scene(env, env_ids=None):
    """One native reset path: new scene, robot and initial image for each selected env."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if ids.numel() == 0:
        return
    if ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("Native reset IDs must be a one-dimensional int64 tensor.")
    runtime = _initialize_episode_reset(env)
    batch = _build_episode_batch(env, runtime, ids)
    apply_episode_batch(env, ids, batch)
