"""Native F/T/contact sensors and renderer-free, reset-only geometric perception.

The virtual viewpoint queries shared surface meshes; only visible first-hit
points enter policy geometry. No camera or reference prim is spawned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .observations import quat_apply, quat_conjugate


@dataclass
class InitialVisionCfg:
    ray_width: int = 160
    ray_height: int = 120
    ray_batch_size: int = 32  # Concurrent episode requests, refilled as they finish.
    ray_candidates_per_env: int = 4
    min_target_visible_fraction: float = 0.40
    min_blocker_visible_fraction: float = 0.95
    min_silhouette_pixels: int = 32
    min_depth_pixels: int = 16
    front_distance_range: tuple[float, float] = (0.55, 0.95)
    lateral_half_width: float = 0.35
    height_above_shelf: tuple[float, float] = (0.18, 0.48)
    aim_y_jitter: float = 0.06
    aim_z_jitter: float = 0.03
    roll_degrees: tuple[float, float] = (-3.0, 3.0)
    max_camera_attempts: int = 32
    # Large batches expose rare layouts that need more than eight proposals.
    # Only failed rows retry; role/composition and visibility thresholds stay fixed.
    max_layout_attempts: int = 32

    def __post_init__(self):
        for name in ('ray_width', 'ray_height', 'ray_batch_size', 'ray_candidates_per_env'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.ray_width % 2 or self.ray_height % 2:
            raise ValueError('Ray image dimensions must be even for aligned full-silhouette sampling.')
        for name in ("min_target_visible_fraction", "min_blocker_visible_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if min(self.min_silhouette_pixels, self.min_depth_pixels, self.max_camera_attempts,
               self.max_layout_attempts) < 1:
            raise ValueError("Camera sample counts and attempts must be positive.")
        for name in ("front_distance_range", "height_above_shelf"):
            lo, hi = getattr(self, name)
            if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
                raise ValueError(f"Invalid {name}.")
        if (not all(math.isfinite(x) for x in self.roll_degrees)
                or self.roll_degrees[0] > self.roll_degrees[1]):
            raise ValueError("Invalid camera roll range.")
        for name in ("lateral_half_width", "aim_y_jitter", "aim_z_jitter"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}.")


@dataclass
class VisionSnapshot:
    """Accepted virtual-view snapshot; optional ray depth/masks, never rendered RGB."""

    geometry_b: torch.Tensor       # [10,10], observed visible-surface OBBs
    geometry_valid: torch.Tensor  # [10]
    observed_ids: torch.Tensor     # [10], native collection indices or -1
    rgb: torch.Tensor | None       # Always None: geometric perception does not produce RGB
    depth: torch.Tensor | None     # Optional debug [H,W,1]
    role_masks: torch.Tensor | None  # Optional debug [H,W,2], target then blocker
    camera_pose_w: torch.Tensor    # [7]
    intrinsics: torch.Tensor       # [3,3]
    visible_fractions: tuple[float, float]
    pixel_counts: tuple[tuple[int, int], tuple[int, int]]
    attempts: int

    @property
    def valid(self):
        return self.geometry_valid

    @property
    def target_visible_fraction(self):
        return self.visible_fractions[0]

    @property
    def blocker_visible_fraction(self):
        return self.visible_fractions[1]


class InitialVisionRejected(RuntimeError):
    """No virtual viewpoint passed; the caller may try another episode layout."""


def fit_visible_obbs(points: torch.Tensor, object_ids: torch.Tensor, capacity: int,
                     min_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit upright OBBs from first-hit surfaces, with no per-object GPU sync.

    Points are already in the robot base frame. Unobserved bodies remain zero;
    neither their catalog extents nor their true object poses complete the fit.
    Ideal geometric rays have no depth outliers, so bounds use surface extrema.
    """
    if points.is_cuda and points.dtype == torch.float32:
        return _fit_visible_obbs_cuda(points, object_ids, capacity, min_points)
    batch, rays, _ = points.shape
    stride = capacity + 1  # Extra reduction bucket for background.
    labels = torch.where(object_ids >= 0, object_ids.long(), capacity)
    groups = (labels + torch.arange(batch, device=points.device)[:, None] * stride).flatten()
    flat = points.reshape(-1, 3)
    values = torch.cat((torch.ones_like(flat[:, :1]), flat), -1)
    moments = points.new_zeros((batch * stride, 4)).index_add_(0, groups, values)
    counts = moments[:, 0]
    mean = moments[:, 1:4] / counts[:, None].clamp_min(1)
    # Center before the covariance reduction: E[x²]-E[x]² loses float32
    # precision for small objects away from the base-frame origin.
    centered = flat - mean[groups]
    products = torch.cat((centered[:, :2].square(), (centered[:, 0] * centered[:, 1])[:, None]), -1)
    covariance_sum = points.new_zeros((batch * stride, 3)).index_add_(0, groups, products)
    variance, covariance = covariance_sum[:, :2], covariance_sum[:, 2]
    yaw = .5 * torch.atan2(2 * covariance, variance[:, 0] - variance[:, 1])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    projected = torch.stack((cosine[groups] * centered[:, 0] + sine[groups] * centered[:, 1],
                             -sine[groups] * centered[:, 0] + cosine[groups] * centered[:, 1],
                             centered[:, 2]), -1)
    indices = groups[:, None].expand(-1, 3)
    lower = points.new_full((batch * stride, 3), float('inf')).scatter_reduce_(
        0, indices, projected, reduce='amin', include_self=True)
    upper = points.new_full((batch * stride, 3), -float('inf')).scatter_reduce_(
        0, indices, projected, reduce='amax', include_self=True)
    valid = counts >= min_points
    midpoint = torch.where(valid[:, None], (lower + upper) * .5, 0.)
    center = mean + torch.stack((cosine * midpoint[:, 0] - sine * midpoint[:, 1],
                                sine * midpoint[:, 0] + cosine * midpoint[:, 1], midpoint[:, 2]), -1)
    quaternion = torch.stack((torch.cos(yaw * .5), torch.zeros_like(yaw),
                              torch.zeros_like(yaw), torch.sin(yaw * .5)), -1)
    geometry = torch.cat((center, quaternion, (upper - lower).clamp_min(.004)), -1)
    geometry = torch.where(valid[:, None], geometry, 0.)
    return geometry.reshape(batch, stride, 10)[:, :capacity], valid.reshape(batch, stride)[:, :capacity]


class SensorReader:
    """Read all environments together, once per completed physics timestamp.

    The F/T incoming joint frame is authored at the distal spacer origin with
    identity child rotation. Its parent-on-child reaction is already expressed
    in physical sensor axes. SI-200-8 calibration and gravity/inertia removal
    belong to events._read_transition, and are deliberately not repeated here.
    """

    def __init__(self, env):
        from ..config.ur5e.scene import (
            FORCE_TORQUE_BODY_NAME, TACTILE_BODY_NAMES,
        )
        from .events import SensorBatch

        self.env = env
        self.robot = env.scene["robot"]
        self.body_index = self.robot.body_names.index(FORCE_TORQUE_BODY_NAME)
        self.distal_body_names = tuple(
            name for name, path in zip(self.robot.body_names, self.robot.root_physx_view.link_paths[0])
            if name == FORCE_TORQUE_BODY_NAME or "/ROASRightHand/" in str(path)
        )
        self.tactile = env.scene["tactile_contact"]
        if (self.tactile.num_bodies != len(TACTILE_BODY_NAMES)
                or set(self.tactile.body_names) != set(TACTILE_BODY_NAMES)
                or self.tactile.num_instances != env.num_envs * len(TACTILE_BODY_NAMES)):
            raise RuntimeError("Tactile sensor must resolve all 17 links exactly once per environment.")
        indices, names = self.tactile.find_bodies(list(TACTILE_BODY_NAMES), preserve_order=True)
        if tuple(names) != TACTILE_BODY_NAMES:
            raise RuntimeError("Tactile links must map to the observation's fixed channel order.")
        self.tactile_ids = torch.tensor(indices, device=env.device, dtype=torch.long)
        n, device = env.num_envs, env.device
        self.batch = SensorBatch(
            torch.zeros((n, 6), device=device), torch.zeros(n, dtype=torch.bool, device=device),
            torch.zeros((n, 17, 3), device=device), torch.zeros((n, 17), dtype=torch.bool, device=device),
        )
        self.last_time = env.sim.current_time

    @torch.no_grad()
    def reset(self, env_ids):
        # Flush any completed sample first, so resetting some rows cannot mark
        # those stale reactions valid when another row is read at the same time.
        self()
        b = self.batch
        b.ideal_wrench[env_ids] = 0
        b.wrench_valid[env_ids] = False
        b.tactile_force[env_ids] = 0
        b.tactile_valid[env_ids] = False
        self.tactile.reset(env_ids)

    @torch.no_grad()
    def __call__(self):
        time = self.env.sim.current_time
        if time <= self.last_time:
            return self.batch
        b = self.batch
        reaction = self.robot.data.body_incoming_joint_wrench_b[:, self.body_index]
        b.ideal_wrench.copy_(reaction)
        b.wrench_valid.copy_(torch.isfinite(reaction).all(-1))
        # One native view reads all 17 links; only the body axis is reordered.
        force = self.tactile.data.net_forces_w.index_select(1, self.tactile_ids)
        b.tactile_force.copy_(force)
        b.tactile_valid.copy_(torch.isfinite(force).all(-1))
        self.last_time = time
        return b


# Native classes are imported only by the scene configuration after AppLauncher.
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.warp import convert_to_warp_mesh
from pxr import Usd, UsdGeom, UsdPhysics


class TaskContactSensor(ContactSensor):
    """Native contact buffers with a synchronous, whole-batch update fast path.

    Every physics tick makes all rows due, so discovering outdated CUDA rows
    with nonzero() is redundant. History sensors read immediately; sensors
    without history read lazily. Keep native force acquisition/history and
    fall back for other sensor configurations.
    """

    def _initialize_impl(self):
        # A stop/play recreates native views; never retain handles from an old
        # PhysX scene. Filtering can be enabled again after initialization.
        self._object_filter_base_view = None
        self._object_force_matrix_w = None
        super()._initialize_impl()

    def enable_object_filtering(self, objects, env_prim_paths):
        """Opt into one grouped, environment-local pad/object contact matrix.

        Native ContactSensor's single filter list cannot express this pairing
        for multiple sensing bodies. PhysX accepts an explicit filter list for
        each sensor pattern, retaining one native view and one matrix read.
        Build paths once; episode role changes only select a different column.
        """
        if getattr(self, "_object_filter_base_view", None) is not None:
            return
        if self.cfg.filter_prim_paths_expr or self.cfg.history_length or not self._whole_batch_updates():
            raise ValueError("Object filtering requires the grouped, history-free tactile sensor configuration.")
        if objects.num_instances != self._num_envs or len(env_prim_paths) != self._num_envs:
            raise ValueError("Tactile sensor, object collection and environment roots must have the same count.")
        sensor_paths = list(self.contact_physx_view.sensor_paths)
        object_paths = list(objects.root_physx_view.prim_paths)
        if len(sensor_paths) != self._num_envs * self._num_bodies:
            raise RuntimeError("Unexpected native tactile sensor path count.")
        if len(object_paths) != self._num_envs * objects.num_objects:
            raise RuntimeError("Unexpected native object collection path count.")
        # Collection rows are object-major, while native grouped sensor rows
        # are environment-major. Check both assumptions before creating a view.
        filters = []
        for row, sensor_path in enumerate(sensor_paths):
            env_id = row // self._num_bodies
            prefix = str(env_prim_paths[env_id]).rstrip("/") + "/"
            paths = [object_paths[obj * self._num_envs + env_id]
                     for obj in range(objects.num_objects)]
            if not sensor_path.startswith(prefix) or any(not path.startswith(prefix) for path in paths):
                raise RuntimeError("Native tactile/object row order does not match environment roots.")
            filters.append(paths)
        view = self._physics_sim_view.create_rigid_contact_view(
            sensor_paths, filter_patterns=filters, max_contact_data_count=0,
        )
        if (view.sensor_count != len(sensor_paths) or view.filter_count != objects.num_objects
                or list(view.sensor_paths) != sensor_paths):
            raise RuntimeError("Filtered contact view changed native sensor or object-column ordering.")
        self._object_filter_base_view = self._contact_physx_view
        self._contact_physx_view = view
        self._object_force_matrix_w = torch.zeros(
            (self._num_envs, self._num_bodies, objects.num_objects, 3), device=self._device,
        )
        self._task_outdated = True

    def disable_object_filtering(self):
        """Restore the original grouped net-contact backend for cost comparisons."""
        base = getattr(self, "_object_filter_base_view", None)
        if base is not None:
            self._contact_physx_view = base
            self._object_filter_base_view = None
            self._object_force_matrix_w = None
            self._task_outdated = True

    @property
    def object_force_matrix_w(self):
        """Cached [environment, native pad, collection object, xyz] pair forces."""
        if getattr(self, "_object_filter_base_view", None) is None:
            raise RuntimeError("Object-filtered tactile contact has not been enabled.")
        self._update_outdated_buffers()
        return self._object_force_matrix_w

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        if getattr(self, "_object_filter_base_view", None) is not None:
            matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            rows = slice(None) if len(env_ids) == self._num_envs else env_ids
            self._object_force_matrix_w[rows] = matrix.view_as(self._object_force_matrix_w)[rows]

    def _whole_batch_updates(self):
        return (self.cfg.update_period == 0.0 and not self.cfg.track_air_time
                and not self.cfg.track_pose)

    def update(self, dt, force_recompute=False):
        if not self._whole_batch_updates():
            return super().update(dt, force_recompute)
        self._timestamp.add_(dt)
        self._task_outdated = True
        if force_recompute or self._is_visualizing or self.cfg.history_length > 0:
            self._update_outdated_buffers()

    def _update_outdated_buffers(self):
        if not self._whole_batch_updates():
            return super()._update_outdated_buffers()
        if getattr(self, "_task_outdated", True):
            self._update_buffers_impl(range(self._num_envs))
            self._timestamp_last_update.copy_(self._timestamp)
            self._is_outdated.fill_(False)
            self._task_outdated = False

    def reset(self, env_ids=None):
        # Preserve completed samples of other rows before clearing selected
        # histories. Do not read stale PhysX contacts back into reset rows.
        self._update_outdated_buffers()
        super().reset(env_ids)
        matrix = getattr(self, "_object_force_matrix_w", None)
        if matrix is not None:
            matrix[slice(None) if env_ids is None else env_ids] = 0


class BlockerContactHold(ManagerTermBase):
    """Native reward term: same pad touches the Blocker in consecutive samples.

    RewardManager owns episode-history resets. Observation remains the original
    any-object binary contact; only this privileged reward reads pair forces.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from ..config.ur5e.scene import TACTILE_BODY_NAMES

        self.sensor = env.scene["tactile_contact"]
        ids, names = self.sensor.find_bodies(list(TACTILE_BODY_NAMES), preserve_order=True)
        if tuple(names) != TACTILE_BODY_NAMES:
            raise RuntimeError("Blocker contact reward requires all 17 canonical tactile pads.")
        with torch.inference_mode(False):
            self.pad_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
            self.previous_contact = torch.zeros((env.num_envs, 17), device=env.device, dtype=torch.bool)
            self.held_count = torch.zeros(env.num_envs, device=env.device)
        self.last_step = env.common_step_counter
        # Weight zero must also avoid registering the filtered native view.
        if cfg.weight != 0:
            self.sensor.enable_object_filtering(env.scene["object_collection"], env.scene.env_prim_paths)

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.previous_contact[rows] = False
        self.held_count[rows] = 0
        # Keep the global timestamp: resetting one row cannot advance another
        # row's contact history or re-read the previous episode's last sample.

    def __call__(self, env, threshold=0.01):
        from .events import get_task_state
        from .rewards import blocker_contact_hold

        get_task_state(env)
        if self.last_step != env.common_step_counter:
            self.sensor.enable_object_filtering(env.scene["object_collection"], env.scene.env_prim_paths)
            # Select the object before reordering pads: avoid copying all ten
            # object columns just to retain one Blocker column.
            force = self.sensor.object_force_matrix_w.gather(
                2, env.task_scene_data.blocker_id[:, None, None, None].expand(-1, 17, 1, 3),
            ).squeeze(2).index_select(1, self.pad_ids)
            held, current = blocker_contact_hold(
                force, self.previous_contact, threshold=threshold,
                valid=env.task_sensor_reader().tactile_valid,
            )
            self.held_count.copy_(held)
            self.previous_contact.copy_(current)
            self.last_step = env.common_step_counter
        return self.held_count / env.step_dt


def make_tactile_contact_cfg(body_names: tuple[str, ...]) -> ContactSensorCfg:
    """Group physical tactile links while retaining a separate force per body."""
    expression = "(" + "|".join(body_names) + ")"
    return ContactSensorCfg(
        class_type=TaskContactSensor, prim_path=f"{{ENV_REGEX_NS}}/Robot/ROASRightHand/{expression}",
        update_period=0.0, history_length=0, track_pose=False, track_air_time=False, debug_vis=False,
    )


# Geometric perception follows Isaac Lab's Warp RayCaster path. It creates no
# USD cameras, render products, annotators, or reference objects.
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _surface_moments(points: wp.array2d(dtype=wp.vec3), labels: wp.array2d(dtype=wp.int32),
                     moments: wp.array3d(dtype=wp.float32)):
    row, ray = wp.tid()
    obj = labels[row, ray]
    if obj >= 0 and obj < moments.shape[1]:
        point = points[row, ray]
        wp.atomic_add(moments, row, obj, 0, 1.0)
        for k in range(3):
            wp.atomic_add(moments, row, obj, k + 1, point[k])


@wp.kernel(enable_backward=False)
def _surface_covariance(points: wp.array2d(dtype=wp.vec3), labels: wp.array2d(dtype=wp.int32),
                        means: wp.array2d(dtype=wp.vec3), covariance: wp.array3d(dtype=wp.float32)):
    row, ray = wp.tid()
    obj = labels[row, ray]
    if obj >= 0 and obj < means.shape[1]:
        p = points[row, ray] - means[row, obj]
        wp.atomic_add(covariance, row, obj, 0, p[0] * p[0])
        wp.atomic_add(covariance, row, obj, 1, p[1] * p[1])
        wp.atomic_add(covariance, row, obj, 2, p[0] * p[1])


@wp.kernel(enable_backward=False)
def _surface_extents(points: wp.array2d(dtype=wp.vec3), labels: wp.array2d(dtype=wp.int32),
                     means: wp.array2d(dtype=wp.vec3), cosine: wp.array2d(dtype=wp.float32),
                     sine: wp.array2d(dtype=wp.float32), lower: wp.array3d(dtype=wp.float32),
                     upper: wp.array3d(dtype=wp.float32)):
    row, ray = wp.tid()
    obj = labels[row, ray]
    if obj >= 0 and obj < means.shape[1]:
        p = points[row, ray] - means[row, obj]
        c, s = cosine[row, obj], sine[row, obj]
        projected = wp.vec3(c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2])
        for k in range(3):
            wp.atomic_min(lower, row, obj, k, projected[k])
            wp.atomic_max(upper, row, obj, k, projected[k])


def _fit_visible_obbs_cuda(points, object_ids, capacity, min_points):
    """Ignore background rays before atomics; retain centered visible-point PCA."""
    wp.init()
    n, rays, _ = points.shape
    labels = object_ids.to(dtype=torch.int32).contiguous()
    values = points.contiguous()
    stream = wp.stream_from_torch(torch.cuda.current_stream(points.device))
    moments = points.new_zeros((n, capacity, 4))
    common = [wp.from_torch(values, dtype=wp.vec3), wp.from_torch(labels)]
    wp.launch(_surface_moments, dim=(n, rays), inputs=[*common, wp.from_torch(moments)], stream=stream)
    counts = moments[..., 0]
    mean = (moments[..., 1:] / counts[..., None].clamp_min(1)).contiguous()
    covariance = points.new_zeros((n, capacity, 3))
    wp.launch(_surface_covariance, dim=(n, rays), inputs=[*common, wp.from_torch(mean, dtype=wp.vec3),
              wp.from_torch(covariance)], stream=stream)
    yaw = .5 * torch.atan2(2 * covariance[..., 2], covariance[..., 0] - covariance[..., 1])
    cosine, sine = yaw.cos(), yaw.sin()
    lower = points.new_full((n, capacity, 3), float('inf'))
    upper = points.new_full((n, capacity, 3), -float('inf'))
    wp.launch(_surface_extents, dim=(n, rays), inputs=[*common, wp.from_torch(mean, dtype=wp.vec3),
              wp.from_torch(cosine), wp.from_torch(sine), wp.from_torch(lower), wp.from_torch(upper)], stream=stream)
    valid = counts >= min_points
    midpoint = torch.where(valid[..., None], (lower + upper) * .5, 0.)
    center = mean + torch.stack((cosine * midpoint[..., 0] - sine * midpoint[..., 1],
                                sine * midpoint[..., 0] + cosine * midpoint[..., 1], midpoint[..., 2]), -1)
    quaternion = torch.stack((torch.cos(yaw * .5), torch.zeros_like(yaw), torch.zeros_like(yaw),
                               torch.sin(yaw * .5)), -1)
    geometry = torch.cat((center, quaternion, (upper - lower).clamp_min(.004)), -1)
    return torch.where(valid[..., None], geometry, 0.), valid


@wp.kernel(enable_backward=False)
def _initial_viewpoints(
    seeds: wp.array(dtype=wp.int64),
    role_positions: wp.array2d(dtype=wp.vec3),
    bounds: wp.array(dtype=wp.vec4),
    tops: wp.array(dtype=wp.float32),
    front: wp.vec2, height: wp.vec2, lateral: float, aim_y: float, aim_z: float, roll: wp.vec2,
    eyes: wp.array2d(dtype=wp.vec3),
    rotations: wp.array2d(dtype=wp.quat),
):
    env, attempt = wp.tid()
    # Episode/candidate counters make sampling independent of queue/batch order.
    state = wp.rand_init(int(seeds[env] & wp.int64(4294967295)), attempt ^ int(seeds[env] >> wp.int64(32)))
    target, blocker = role_positions[env, 0], role_positions[env, 1]
    region = bounds[env]
    eye = wp.vec3(region[1] + wp.randf(state, front[0], front[1]),
                  wp.clamp(blocker[1] + wp.randf(state, -lateral, lateral), region[2], region[3]),
                  tops[env] + wp.randf(state, height[0], height[1]))
    aim = (target + blocker) * 0.5
    aim[1] += wp.randf(state, -aim_y, aim_y)
    aim[2] = tops[env] + 0.1 + wp.randf(state, -aim_z, aim_z)
    forward = wp.normalize(aim - eye)
    # ROS camera: +Z forward, +Y down. Positive front distance avoids a vertical view.
    right = wp.normalize(wp.cross(forward, wp.vec3(0.0, 0.0, 1.0)))
    down = wp.cross(forward, right)
    matrix = wp.mat33(right[0], down[0], forward[0], right[1], down[1], forward[1],
                      right[2], down[2], forward[2])
    rotation = wp.quat_from_matrix(matrix)
    eyes[env, attempt] = eye
    rotations[env, attempt] = rotation * wp.quat_from_axis_angle(
        wp.vec3(0.0, 0.0, 1.0), wp.randf(state, roll[0], roll[1]))


@wp.kernel(enable_backward=False)
def _initial_ray_frames(
    poses: wp.array2d(dtype=wp.transform),
    active_ids: wp.array2d(dtype=wp.int32),
    eyes: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    frames: wp.array2d(dtype=wp.transform),
    mesh_bounds: wp.array2d(dtype=wp.vec3),
    projected: wp.array2d(dtype=wp.vec4), width: int, height: int,
):
    view, slot = wp.tid()
    body = active_ids[view, slot]
    if body >= 0:
        # Shared by every pixel; no inverse pose or local-origin work in the ray loop.
        frame = wp.transform_multiply(wp.transform_inverse(poses[view, body]),
                                      wp.transform(eyes[view], orientations[view]))
        frames[view, slot] = frame
        camera_from_body = wp.transform_inverse(frame)
        lower, upper = mesh_bounds[body, 0], mesh_bounds[body, 1]
        bounds = wp.vec4(1.0e10, -1.0e10, 1.0e10, -1.0e10)
        crosses_eye = bool(False)
        for corner in range(8):
            point = lower
            if corner & 1:
                point[0] = upper[0]
            if corner & 2:
                point[1] = upper[1]
            if corner & 4:
                point[2] = upper[2]
            point = wp.transform_point(camera_from_body, point)
            if point[2] <= 1.0e-5:
                crosses_eye = True
            else:
                x = float(width) + 0.75 * float(width) * point[0] / point[2]
                y = float(height) + 0.75 * float(width) * point[1] / point[2]
                bounds = wp.vec4(wp.min(bounds[0], x - 1.0), wp.max(bounds[1], x + 1.0),
                                 wp.min(bounds[2], y - 1.0), wp.max(bounds[3], y + 1.0))
        if crosses_eye:
            bounds = wp.vec4(-1.0e10, 1.0e10, -1.0e10, 1.0e10)
        projected[view, slot] = bounds



@wp.kernel(enable_backward=False)
def _initial_surface_rays(
    meshes: wp.array(dtype=wp.uint64),
    frames: wp.array2d(dtype=wp.transform),
    active_ids: wp.array2d(dtype=wp.int32),
    role_slots: wp.array2d(dtype=wp.int32),
    projected: wp.array2d(dtype=wp.vec4),
    directions: wp.array(dtype=wp.vec3),
    roles: wp.array2d(dtype=wp.int32),
    width: int, height: int, far: float,
    depths: wp.array2d(dtype=wp.float32),
    labels: wp.array2d(dtype=wp.int32),
    silhouettes: wp.array3d(dtype=wp.int32),
):
    env, ray = wp.tid()
    x = ray % (2 * width)
    y = ray // (2 * width)
    in_image = x >= width // 2 and x < width // 2 + width and y >= height // 2 and y < height // 2 + height
    best = far
    owner = int(-1)
    # Each ray owns its outputs: no competing atomic depth/label writes.
    # Outside the image only the two isolated role silhouettes are needed.
    # Route directly to those meshes instead of scanning every scene body.
    body_count = active_ids.shape[1]
    if not in_image:
        body_count = 2
    for candidate in range(body_count):
        slot = candidate
        if not in_image:
            slot = role_slots[env, candidate]
        if slot < 0:
            continue
        body = active_ids[env, slot]
        if body < 0:
            break  # Compact active IDs followed by padding.
        bounds = projected[env, slot]
        if float(x) + 0.5 < bounds[0] or float(x) + 0.5 > bounds[1] or float(y) + 0.5 < bounds[2] or float(y) + 0.5 > bounds[3]:
            continue
        is_target = body == roles[env, 0]
        is_blocker = body == roles[env, 1]
        frame = frames[env, slot]
        local_direction = wp.transform_vector(frame, directions[ray])
        limit = far
        if in_image and not is_target and not is_blocker:
            limit = best
        query = wp.mesh_query_ray(meshes[body], wp.transform_get_translation(frame), local_direction, limit)
        if query.result and query.t > 0.02:
            if is_target:
                silhouettes[env, ray, 0] = 1
            if is_blocker:
                silhouettes[env, ray, 1] = 1
            if in_image and query.t < best:
                best = query.t
                owner = body
    if in_image:
        pixel = (y - height // 2) * width + x - width // 2
        depths[env, pixel] = best
        labels[env, pixel] = owner


def _visual_triangles(root, world_to_body):
    """Read authored opaque surfaces once, baking scale into metric vertices.

    Collider proxies and invisible/guide geometry are excluded. Primitive USD
    shapes are tessellated; no physics or renderer resource is created.
    """
    import trimesh
    from ..config.ur5e.scene import _gf_column
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices, faces, offset = [], [], 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        imageable = UsdGeom.Imageable(prim)
        if (not imageable or imageable.ComputeVisibility() == UsdGeom.Tokens.invisible
                or imageable.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
            holes = set(mesh.GetHoleIndicesAttr().Get() or [])
            if np.all(counts == 3) and not holes:
                triangles = indices.reshape(-1, 3)
            else:
                triangles, begin = [], 0
                for face, count in enumerate(counts):
                    if face not in holes:
                        triangles.extend((indices[begin], indices[begin + j], indices[begin + j + 1])
                                         for j in range(1, int(count) - 1))
                    begin += int(count)
                triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        elif prim.IsA(UsdGeom.Cube):
            shape = trimesh.creation.box(extents=[float(UsdGeom.Cube(prim).GetSizeAttr().Get())] * 3)
            points, triangles = shape.vertices, shape.faces
        elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            capsule = prim.IsA(UsdGeom.Capsule)
            shape_cfg = UsdGeom.Capsule(prim) if capsule else UsdGeom.Cylinder(prim)
            radius, height = float(shape_cfg.GetRadiusAttr().Get()), float(shape_cfg.GetHeightAttr().Get())
            shape = (trimesh.creation.capsule(radius=radius, height=height, count=[16, 32]) if capsule
                     else trimesh.creation.cylinder(radius=radius, height=height, sections=32))
            points, triangles = np.array(shape.vertices), shape.faces
            axis = shape_cfg.GetAxisAttr().Get()
            if axis == 'X':
                points = points[:, [2, 1, 0]]
            elif axis == 'Y':
                points = points[:, [0, 2, 1]]
        elif prim.IsA(UsdGeom.Sphere):
            shape = trimesh.creation.icosphere(subdivisions=2, radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
            points, triangles = shape.vertices, shape.faces
        else:
            continue
        if not len(points) or not len(triangles):
            continue
        transform = world_to_body @ _gf_column(xforms.GetLocalToWorldTransform(prim))
        vertices.append(points @ transform[:3, :3].T + transform[:3, 3])
        faces.append(triangles + offset)
        offset += len(points)
    if not vertices:
        return None
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)


class InitialVisionSensor:
    """Reset-only geometric visibility using shared mesh BVHs and batched rays.

    True poses position surfaces for the simulated sensor, but policy geometry
    is fitted exclusively to visible first hits. Every request retains its own
    seeded viewpoint search; no image or scene bank is shared across episodes.
    """

    def __init__(self, env):
        from ..config.ur5e.scene import OBJECT_SPECS, _gf_column, _rotation_part
        self.env, self.cfg = env, env.cfg.initial_vision
        self.cfg.__post_init__()
        self.capacity = self.cfg.ray_batch_size
        self.meshes, identifiers, bounds = [], [], []
        self.specs = OBJECT_SPECS
        stage, device = env.scene.stage, env.device
        origin_path = env.scene.env_prim_paths[0]
        cache = {}
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        def add(root, key):
            if key in cache:
                return cache[key]
            world = _gf_column(xforms.GetLocalToWorldTransform(root))
            rigid = np.eye(4)
            rigid[:3, :3], rigid[:3, 3] = _rotation_part(world), world[:3, 3]
            triangles = _visual_triangles(root, np.linalg.inv(rigid))
            if triangles is None:
                raise RuntimeError(f"No visible triangles under {root.GetPath()}.")
            points, indices = triangles
            mesh = convert_to_warp_mesh(points, indices, device=device)
            self.meshes.append(mesh)  # Own buffers for the entire simulation lifetime.
            cache[key] = (mesh.id, np.stack((points.min(0), points.max(0))))
            return cache[key]
        for spec in OBJECT_SPECS:
            root = stage.GetPrimAtPath(f'{origin_path}/{spec.name}')
            bodies = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                      if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(bodies) != 1:
                raise RuntimeError(f'Initial perception needs one physical body for {spec.name}.')
            identifier, bound = add(bodies[0], spec.asset_name)
            if not identifier:
                raise RuntimeError(f'No visible surface for object {spec.name}.')
            identifiers.append(identifier)
            bounds.append(bound)
        self.object_count = len(identifiers)
        self.mesh_ids = wp.array(identifiers, dtype=wp.uint64, device=device)
        self.mesh_bounds = wp.array(np.asarray(bounds), dtype=wp.vec3, device=device)
        width, height = self.cfg.ray_width, self.cfg.ray_height
        self.intrinsics = torch.tensor([[.75 * width, 0., width * .5],
                                        [0., .75 * width, height * .5], [0., 0., 1.]], device=device)
        rows, columns = torch.meshgrid(torch.arange(2 * height, device=device),
                                       torch.arange(2 * width, device=device), indexing='ij')
        self.directions = torch.stack(((columns + .5 - width) / (.75 * width),
                                        (rows + .5 - height) / (.75 * width), torch.ones_like(rows)), -1).float()
        self.crop_directions = self.directions[height // 2:height // 2 + height,
                                               width // 2:width // 2 + width].reshape(-1, 3).contiguous()
        self.directions = self.directions.reshape(-1, 3).contiguous()

    def capture_session(self):
        from contextlib import nullcontext
        return nullcontext()  # No renderer, visibility edits or physics stepping.

    @torch.no_grad()
    def capture(self, env_id, active_ids, target_id, blocker_id, workspace_bounds,
                shelf_top, *, seed, object_paths=None, retain_images=False):
        result = self.capture_batch([dict(env_id=env_id, active_ids=active_ids, target_id=target_id,
            blocker_id=blocker_id, workspace_bounds=workspace_bounds, shelf_top=shelf_top, seed=seed)],
            retain_images=retain_images)[0]
        if result is None:
            raise InitialVisionRejected('No geometric viewpoint passed the visibility criteria.')
        return result

    @torch.no_grad()
    def capture_batch(self, requests, *, retain_images=False, object_poses=None, root_poses=None):
        return self._capture_pool(requests, retain_images, object_poses, root_poses) if requests else []

    def _trace(self, poses, active_ids, eyes, orientations, roles):
        width, height = self.cfg.ray_width, self.cfg.ray_height
        count, device = len(eyes), eyes.device
        depths = torch.empty((count, width * height), device=device)
        labels = torch.empty((count, width * height), device=device, dtype=torch.int32)
        silhouettes = torch.zeros((count, 4 * width * height, 2), device=device, dtype=torch.int32)
        # Warp uses xyzw; policy and Isaac Lab tensors use wxyz.
        poses = poses[..., [0, 1, 2, 4, 5, 6, 3]].contiguous()
        quaternions = orientations[:, [1, 2, 3, 0]].contiguous()
        frames = torch.empty((*active_ids.shape, 7), device=device)
        projected = torch.empty((*active_ids.shape, 4), device=device)
        matches = active_ids[..., None] == roles[:, None]
        role_slots = torch.where(matches.any(1), matches.int().argmax(1), -1).int().contiguous()
        execution = (dict(stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
                     if device.type == 'cuda' else dict(device='cpu'))
        wp.launch(kernel=_initial_ray_frames, dim=active_ids.shape, inputs=[
            wp.from_torch(poses, dtype=wp.transform), wp.from_torch(active_ids),
            wp.from_torch(eyes.contiguous(), dtype=wp.vec3), wp.from_torch(quaternions, dtype=wp.quat),
            wp.from_torch(frames, dtype=wp.transform), self.mesh_bounds,
            wp.from_torch(projected, dtype=wp.vec4), width, height], **execution)
        launch = dict(kernel=_initial_surface_rays, dim=(count, 4 * width * height), inputs=[
            self.mesh_ids, wp.from_torch(frames, dtype=wp.transform), wp.from_torch(active_ids),
            wp.from_torch(role_slots), wp.from_torch(projected, dtype=wp.vec4),
            wp.from_torch(self.directions, dtype=wp.vec3), wp.from_torch(roles), width, height, 2.5,
            wp.from_torch(depths), wp.from_torch(labels), wp.from_torch(silhouettes)])
        wp.launch(**launch, **execution)
        full = silhouettes.reshape(count, 2 * height, 2 * width, 2)
        crop = full[:, height // 2:height // 2 + height, width // 2:width // 2 + width]
        denominator = full.sum((1, 2))
        border = (crop[:, 0].any(1) | crop[:, -1].any(1) | crop[:, :, 0].any(1) | crop[:, :, -1].any(1))
        framed = (denominator == crop.sum((1, 2))) & ~border
        visible = (labels[..., None] == roles[:, None]).sum(1)
        fractions = visible / denominator.clamp_min(1)
        valid = (framed.all(-1) & (denominator >= self.cfg.min_silhouette_pixels).all(-1)
                 & (visible >= self.cfg.min_depth_pixels).all(-1)
                 & (fractions[:, 0] >= self.cfg.min_target_visible_fraction)
                 & (fractions[:, 1] >= self.cfg.min_blocker_visible_fraction))
        return depths, labels, denominator, visible, fractions, valid

    def _viewpoints(self, requests, role_positions):
        """Generate all bounded candidate poses on-device, once per episode request."""
        cfg, device = self.cfg, role_positions.device
        shape = (len(requests), cfg.max_camera_attempts)
        eyes = torch.empty((*shape, 3), device=device)
        rotations = torch.empty((*shape, 4), device=device)
        seeds = torch.tensor([r['seed'] for r in requests], device=device, dtype=torch.int64)
        bounds = role_positions.new_tensor([r['workspace_bounds'] for r in requests])
        tops = role_positions.new_tensor([r['shelf_top'] for r in requests])
        execution = (dict(stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
                     if device.type == 'cuda' else dict(device='cpu'))
        wp.launch(kernel=_initial_viewpoints, dim=shape, inputs=[
            wp.from_torch(seeds), wp.from_torch(role_positions.contiguous(), dtype=wp.vec3),
            wp.from_torch(bounds, dtype=wp.vec4), wp.from_torch(tops),
            wp.vec2(*cfg.front_distance_range), wp.vec2(*cfg.height_above_shelf), cfg.lateral_half_width,
            cfg.aim_y_jitter, cfg.aim_z_jitter, wp.vec2(*(math.radians(x) for x in cfg.roll_degrees)),
            wp.from_torch(eyes, dtype=wp.vec3), wp.from_torch(rotations, dtype=wp.quat)], **execution)
        return eyes, rotations[..., [3, 0, 1, 2]].contiguous()

    def _capture_pool(self, requests, retain_images, object_poses=None, root_poses=None):
        env, cfg = self.env, self.cfg
        ids = [int(request['env_id']) for request in requests]
        objects = (env.scene['object_collection'].data.object_link_pose_w[ids]
                   if object_poses is None else object_poses)
        robot = env.scene['robot'].data
        origins = env.scene.env_origins[ids]
        # Compute in environment-local coordinates to avoid large-grid precision loss.
        objects = objects.clone()
        objects[..., :3] -= origins[:, None]
        width = max(len(request['active_ids']) for request in requests)
        active_ids = np.full((len(ids), width), -1, dtype=np.int32)
        for row, request in enumerate(requests):
            active = request['active_ids']
            if (request['target_id'] == request['blocker_id'] or request['target_id'] not in active
                    or request['blocker_id'] not in active or len(active) > env.cfg.mdp.max_objects
                    or len(set(active)) != len(active)
                    or any(index < 0 or index >= self.object_count for index in active)):
                raise ValueError('Initial perception needs distinct active Target/Blocker within capacity.')
            active_ids[row, :len(active)] = sorted(active)  # Preserve ID order for equal-depth ties.
        active_ids = torch.as_tensor(active_ids, device=env.device)
        roles = torch.tensor([[r['target_id'], r['blocker_id']] for r in requests], device=env.device, dtype=torch.int32)
        role_positions = objects.gather(1, roles.long()[..., None].expand(-1, -1, 3))
        candidate_eyes, candidate_rotations = self._viewpoints(requests, role_positions)
        roots = (robot.root_pos_w[ids] if root_poses is None else root_poses[:, :3]) - origins
        inverse_roots = quat_conjugate(robot.root_quat_w[ids] if root_poses is None else root_poses[:, 3:7])
        results, outputs = [None] * len(ids), [None] * len(ids)
        metadata = objects.new_zeros((len(ids), 6))
        next_attempt = [0] * len(ids)
        next_request = min(self.capacity, len(ids))
        pending = list(range(next_request))
        candidates_per_env = min(cfg.ray_candidates_per_env, cfg.max_camera_attempts)
        offsets = torch.arange(candidates_per_env, device=env.device)
        observed_slots = torch.arange(env.cfg.mdp.max_objects, device=env.device)
        while pending:
            rows = torch.tensor(pending, device=env.device, dtype=torch.long)
            attempts = torch.tensor([next_attempt[row] for row in pending], device=env.device)[:, None] + offsets
            view_rows = rows.repeat_interleave(candidates_per_env)
            view_attempts = attempts.clamp_max(cfg.max_camera_attempts - 1).flatten()
            eyes = candidate_eyes[view_rows, view_attempts]
            orientation = candidate_rotations[view_rows, view_attempts]
            depth, labels, full, visible, fractions, accepted = self._trace(
                objects[view_rows], active_ids[view_rows], eyes, orientation, roles[view_rows])
            accepted = accepted.reshape(len(pending), candidates_per_env) & (attempts < cfg.max_camera_attempts)
            # One bounded host transfer per proposal group, never one per candidate/pixel/object.
            first = torch.where(accepted, offsets, candidates_per_env).amin(-1).tolist()
            chosen = [slot for slot, candidate in enumerate(first) if candidate < candidates_per_env]
            if chosen:
                accepted_rows = [pending[slot] for slot in chosen]
                rows = torch.tensor(accepted_rows, device=env.device, dtype=torch.long)
                selected = torch.tensor([slot * candidates_per_env + first[slot] for slot in chosen],
                                        device=env.device, dtype=torch.long)
                depth, labels = depth[selected], labels[selected]
                eyes, orientation = eyes[selected], orientation[selected]
                # Fit once, for the first passing view only. Role point counts were already checked.
                local_points = quat_apply(orientation[:, None], depth[..., None] * self.crop_directions) + eyes[:, None]
                points_b = quat_apply(inverse_roots[rows, None], local_points - roots[rows, None])
                geometry, valid = fit_visible_obbs(points_b, labels, env.cfg.mdp.max_objects, cfg.min_depth_pixels)
                observed = torch.where(valid, observed_slots, -1)
                camera_poses = torch.cat((eyes + origins[rows], orientation), -1)
                metadata[rows] = torch.cat((fractions[selected], torch.stack((visible[selected], full[selected]), -1)
                                           .flatten(1)), -1)
                for index, (slot, row) in enumerate(zip(chosen, accepted_rows)):
                    outputs[row] = (geometry[index], valid[index], observed[index],
                        depth[index].reshape(cfg.ray_height, cfg.ray_width, 1) if retain_images else None,
                        (labels[index, :, None] == roles[row]).reshape(cfg.ray_height, cfg.ray_width, 2)
                        if retain_images else None, camera_poses[index], next_attempt[row] + first[slot] + 1)
            remaining = []
            for slot, row in enumerate(pending):
                next_attempt[row] += candidates_per_env
                if first[slot] == candidates_per_env and next_attempt[row] < cfg.max_camera_attempts:
                    remaining.append(row)
            end = min(len(ids), next_request + self.capacity - len(remaining))
            pending = remaining + list(range(next_request, end))
            next_request = end
        # Diagnostic tuples cross to the host once, after the entire request queue finishes.
        summaries = metadata.tolist()
        for row, output in enumerate(outputs):
            if output is not None:
                geometry, valid, observed, depth, masks, camera, attempt = output
                summary = summaries[row]
                results[row] = VisionSnapshot(geometry, valid, observed, None, depth, masks, camera,
                    self.intrinsics.clone(), tuple(summary[:2]),
                    ((int(summary[2]), int(summary[3])), (int(summary[4]), int(summary[5]))), attempt)
        return results
