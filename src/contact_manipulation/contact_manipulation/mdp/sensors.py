"""Native F/T/contact sensors and renderer-free, reset-only geometric perception.

The virtual viewpoint queries shared surface meshes; only visible first-hit
points enter policy geometry. No camera or reference prim is spawned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch
from isaaclab.utils.math import quat_from_matrix

from .observations import quat_apply, quat_conjugate, select_initial_geometry_indices


@dataclass
class InitialVisionCfg:
    # Preserve the startup visibility certificate independently of sensor sampling.
    ray_width: int = 160
    ray_height: int = 120
    observation_ray_width: int = 64
    observation_ray_height: int = 48
    observation_min_depth_pixels: int = 3
    ray_batch_size: int = 32  # Fixed-domain ray views per memory chunk.
    ray_candidates_per_env: int = 4
    view_domain_shape: tuple[int, int, int] = (2, 9, 2)
    min_target_visible_fraction: float = 0.40
    max_target_visible_fraction: float = 0.90
    min_blocker_visible_fraction: float = 0.95
    min_silhouette_pixels: int = 32
    min_depth_pixels: int = 16
    front_distance_range: tuple[float, float] = (0.55, 0.95)
    lateral_half_width: float = 0.35
    height_above_shelf: tuple[float, float] = (0.18, 0.48)
    aim_y_jitter: float = 0.06
    aim_z_jitter: float = 0.03
    roll_degrees: tuple[float, float] = (-3.0, 3.0)
    max_camera_attempts: int = 32  # Legacy random-view API only.
    max_layout_attempts: int = 32  # Legacy rejection-layout API only.

    def __post_init__(self):
        for name in ('ray_width', 'ray_height', 'observation_ray_width', 'observation_ray_height',
                     'observation_min_depth_pixels', 'ray_batch_size', 'ray_candidates_per_env'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if (not isinstance(self.view_domain_shape, (tuple, list)) or len(self.view_domain_shape) != 3
                or any(isinstance(value, bool) or not isinstance(value, int)
                or value < 1 for value in self.view_domain_shape)):
            raise ValueError('view_domain_shape must contain three positive integers (distance, lateral, height).')
        if any(getattr(self, name) % 2 for name in
               ('ray_width', 'ray_height', 'observation_ray_width', 'observation_ray_height')):
            raise ValueError('Ray image dimensions must be even for aligned full-silhouette sampling.')
        for name in ("min_target_visible_fraction", "max_target_visible_fraction", "min_blocker_visible_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1].")
        if self.min_target_visible_fraction > self.max_target_visible_fraction:
            raise ValueError("Target visible-fraction limits must be ordered.")
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

    geometry_b: torch.Tensor       # [capacity,10], observed visible-surface OBBs
    geometry_valid: torch.Tensor  # [capacity]
    observed_ids: torch.Tensor     # [capacity], native collection indices or -1
    rgb: torch.Tensor | None       # Always None: geometric perception does not produce RGB
    depth: torch.Tensor | None     # Optional debug [H,W,1]
    role_masks: torch.Tensor | None  # Optional debug [H,W,2], target then blocker
    camera_pose_w: torch.Tensor    # [7]
    intrinsics: torch.Tensor       # [3,3]
    visible_fractions: tuple[float, float] | None  # None for observation-only capture.
    pixel_counts: tuple[tuple[int, int], tuple[int, int]] | None
    attempts: int

    @property
    def valid(self):
        return self.geometry_valid

    @property
    def target_visible_fraction(self):
        return None if self.visible_fractions is None else self.visible_fractions[0]

    @property
    def blocker_visible_fraction(self):
        return None if self.visible_fractions is None else self.visible_fractions[1]


class InitialVisionRejected(RuntimeError):
    """The initial view or complete view domain violated the geometric sensor contract."""


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
    labels = torch.where((object_ids >= 0) & (object_ids < capacity), object_ids.long(), capacity)
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
        self._blocker_filter_base_view = None
        self._blocker_force_w = None
        self._blocker_id = None
        self._blocker_filter_paths = None
        super()._initialize_impl()

    def enable_blocker_filtering(self, objects, env_prim_paths, blocker_id: int) -> bool:
        """Track only the shared scene's Blocker in one native force column.

        Native ContactSensor's single filter list cannot express this pairing
        for multiple sensing bodies. PhysX accepts an explicit filter list for
        each sensor row. Net forces still include *all* contacting objects.
        A new shared Blocker replaces the view; partial resets reuse it.
        """
        if not 0 <= blocker_id < objects.num_objects:
            raise ValueError("Blocker ID is outside the object collection.")
        if getattr(self, "_blocker_id", None) == blocker_id:
            return False
        if self.cfg.filter_prim_paths_expr or self.cfg.history_length or not self._whole_batch_updates():
            raise ValueError("Blocker filtering requires the grouped, history-free tactile sensor configuration.")
        if objects.num_instances != self._num_envs or len(env_prim_paths) != self._num_envs:
            raise ValueError("Tactile sensor, object collection and environment roots must have the same count.")
        if getattr(self, "_blocker_filter_paths", None) is None:
            sensor_paths = list(self.contact_physx_view.sensor_paths)
            object_paths = list(objects.root_physx_view.prim_paths)
            if len(sensor_paths) != self._num_envs * self._num_bodies:
                raise RuntimeError("Unexpected native tactile sensor path count.")
            if len(object_paths) != self._num_envs * objects.num_objects:
                raise RuntimeError("Unexpected native object collection path count.")
            # Collection rows are object-major; grouped sensors are env-major.
            # Validate immutable path ordering once, outside the step path.
            roots = [str(path).rstrip("/") + "/" for path in env_prim_paths]
            if (any(not path.startswith(roots[row // self._num_bodies])
                    for row, path in enumerate(sensor_paths))
                    or any(not path.startswith(roots[row % self._num_envs])
                           for row, path in enumerate(object_paths))):
                raise RuntimeError("Native tactile/object row order does not match environment roots.")
            self._blocker_filter_paths = sensor_paths, object_paths
        sensor_paths, object_paths = self._blocker_filter_paths
        offset = blocker_id * self._num_envs
        filters = [[object_paths[offset + row // self._num_bodies]] for row in range(len(sensor_paths))]
        view = self._physics_sim_view.create_rigid_contact_view(
            sensor_paths, filter_patterns=filters, max_contact_data_count=0,
        )
        if (view.sensor_count != len(sensor_paths) or view.filter_count != 1
                or list(view.sensor_paths) != sensor_paths):
            raise RuntimeError("Blocker contact view changed native sensor or filter ordering.")
        with torch.inference_mode(False):
            force = torch.zeros((self._num_envs, self._num_bodies, 3), device=self._device)
        # Retain only the original net-force view, never past Blocker views.
        if getattr(self, "_blocker_filter_base_view", None) is None:
            self._blocker_filter_base_view = self._contact_physx_view
        self._contact_physx_view = view
        self._blocker_id = blocker_id
        self._blocker_force_w = force
        self._task_outdated = True
        return True

    def disable_blocker_filtering(self):
        """Restore the original grouped net-contact backend for cost comparisons."""
        base = getattr(self, "_blocker_filter_base_view", None)
        if base is not None:
            self._contact_physx_view = base
            self._blocker_filter_base_view = None
            self._blocker_force_w = None
            self._blocker_id = None
            self._task_outdated = True

    @property
    def blocker_force_w(self):
        """Cached [environment, native pad, xyz] forces against the Blocker."""
        if getattr(self, "_blocker_filter_base_view", None) is None:
            raise RuntimeError("Blocker-filtered tactile contact has not been enabled.")
        self._update_outdated_buffers()
        return self._blocker_force_w

    def _update_buffers_impl(self, env_ids):
        super()._update_buffers_impl(env_ids)
        if getattr(self, "_blocker_filter_base_view", None) is not None:
            matrix = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            rows = slice(None) if len(env_ids) == self._num_envs else env_ids
            self._blocker_force_w[rows] = matrix.view_as(self._blocker_force_w)[rows]

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
        matrix = getattr(self, "_blocker_force_w", None)
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
        self.blocker_id = None
        # The reset event supplies the committed shared Blocker as a host int.
        # No object-column gather or device-to-host role lookup during steps.
        env._blocker_contact_hold = self

    def configure_blocker(self, blocker_id: int):
        """Bind a committed shared Blocker; same-ID partial resets reuse it."""
        if self.cfg.weight == 0:
            return
        changed = self.sensor.enable_blocker_filtering(
            self._env.scene["object_collection"], self._env.scene.env_prim_paths, blocker_id)
        self.blocker_id = blocker_id
        if changed:
            self.reset()
            self.last_step = self._env.common_step_counter

    def reset(self, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.previous_contact[rows] = False
        self.held_count[rows] = 0
        # Keep the global timestamp: resetting one row cannot advance another
        # row's contact history or re-read the previous episode's last sample.

    def __call__(self, env, threshold=0.01):
        from .events import get_task_state
        from .rewards import blocker_contact_hold

        state = get_task_state(env)
        # A timeline PLAY recreates sensor views without necessarily resetting
        # the environment. Reconnect from the retained host ID in that case.
        if self.sensor._blocker_id is None:
            if self.blocker_id is None:
                raise RuntimeError("Blocker contact reward requires a committed episode reset.")
            self.configure_blocker(self.blocker_id)
        if self.last_step != env.common_step_counter:
            force = self.sensor.blocker_force_w.index_select(1, self.pad_ids)
            held, current = blocker_contact_hold(
                force, self.previous_contact, threshold=threshold,
                valid=env.task_sensor_reader().tactile_valid,
            )
            self.held_count.copy_(held)
            self.previous_contact.copy_(current)
            self.last_step = env.common_step_counter
        # Keep contact history current while gated so restoring the Target
        # resumes ordinary consecutive-sample contact evaluation.
        return (self.held_count * state.target_protected) / env.step_dt


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
    width: int, height: int, far: float, observation_only: int,
    depths: wp.array2d(dtype=wp.float32),
    labels: wp.array2d(dtype=wp.int32),
    silhouettes: wp.array3d(dtype=wp.int32),
):
    env, ray = wp.tid()
    x = ray % (2 * width)
    y = ray // (2 * width)
    direction_index = ray
    if observation_only == 1:
        x = ray % width + width // 2
        y = ray // width + height // 2
        direction_index = y * (2 * width) + x
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
        local_direction = wp.transform_vector(frame, directions[direction_index])
        limit = far
        if in_image and (observation_only == 1 or (not is_target and not is_blocker)):
            limit = best
        query = wp.mesh_query_ray(meshes[body], wp.transform_get_translation(frame), local_direction, limit)
        if query.result and query.t > 0.02:
            if is_target and observation_only == 0:
                silhouettes[env, ray, 0] = 1
            if is_blocker and observation_only == 0:
                silhouettes[env, ray, 1] = 1
            if in_image and query.t < best:
                best = query.t
                owner = body
    if in_image:
        pixel = (y - height // 2) * width + x - width // 2
        depths[env, pixel] = best
        labels[env, pixel] = owner


def _visual_triangles(root, world_to_body, *, reference_frame=None):
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
        relative = (xforms.GetLocalToWorldTransform(prim) if reference_frame is None
                    else xforms.ComputeRelativeTransform(prim, reference_frame)[0])
        transform = world_to_body @ _gf_column(relative)
        vertices.append(points @ transform[:3, :3].T + transform[:3, 3])
        faces.append(triangles + offset)
        offset += len(points)
    if not vertices:
        return None
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)


class InitialVisionSensor:
    """Reset-only geometric visibility using shared mesh BVHs and batched rays.

    True poses position surfaces for the simulated sensor, but policy geometry
    is fitted exclusively to visible first hits. Constructive layouts evaluate
    their complete fixed view domain once; adaptive search is legacy-only.
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
        reference_frame = stage.GetPrimAtPath(origin_path)
        cache = {}
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        def add(root, key):
            if key in cache:
                return cache[key]
            world = _gf_column(xforms.ComputeRelativeTransform(root, reference_frame)[0])
            rigid = np.eye(4)
            rigid[:3, :3], rigid[:3, 3] = _rotation_part(world), world[:3, 3]
            triangles = _visual_triangles(root, np.linalg.inv(rigid), reference_frame=reference_frame)
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
        self._mesh_id_by_object = tuple(identifiers)
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
        self._observation_grid(device)

    def _observation_grid(self, device):
        """Cache a lower-resolution ray grid with the certificate's field of view."""
        width, height = self.cfg.observation_ray_width, self.cfg.observation_ray_height
        key = (width, height, torch.device(device))
        cached = getattr(self, '_observation_grid_cache', None)
        if cached is None or cached[0] != key:
            rows, columns = torch.meshgrid(torch.arange(2 * height, device=device),
                                           torch.arange(2 * width, device=device), indexing='ij')
            directions = torch.stack(((columns + .5 - width) / (.75 * width),
                                      (rows + .5 - height) / (.75 * width), torch.ones_like(rows)), -1).float()
            crop = directions[height // 2:height // 2 + height,
                              width // 2:width // 2 + width].reshape(-1, 3).contiguous()
            intrinsics = torch.tensor([[.75 * width, 0., width * .5],
                                       [0., .75 * width, height * .5], [0., 0., 1.]], device=device)
            cached = (key, directions.reshape(-1, 3).contiguous(), crop, intrinsics)
            self._observation_grid_cache = cached
        return cached[1:]

    def capture_session(self):
        from contextlib import nullcontext
        return nullcontext()  # No renderer, visibility edits or physics stepping.

    @torch.no_grad()
    def capture(self, env_id, active_ids, target_id, blocker_id, workspace_bounds,
                shelf_top, *, seed, object_paths=None, retain_images=False, camera_pose=None):
        result = self.capture_batch([dict(env_id=env_id, active_ids=active_ids, target_id=target_id,
            blocker_id=blocker_id, workspace_bounds=workspace_bounds, shelf_top=shelf_top, seed=seed)],
            retain_images=retain_images,
            camera_poses=None if camera_pose is None else torch.as_tensor(camera_pose, device=self.env.device)[None])[0]
        if result is None:
            raise InitialVisionRejected(f'Initial geometric visibility requirements were not met: {self.last_diagnostics[0]}')
        return result

    @torch.no_grad()
    def capture_batch(self, requests, *, retain_images=False, object_poses=None, root_poses=None,
                      camera_poses=None):
        """Capture supplied world poses once, or use the legacy viewpoint search.

        A constructive layout supplies ``camera_poses`` as [request, xyz+wxyz].
        That path never samples or retries a viewpoint. Rejections are returned
        as None and their exact criteria are retained in ``last_diagnostics``.
        """
        if not requests:
            self.last_diagnostics = []
            return []
        return self._capture_pool(requests, retain_images, object_poses, root_poses, camera_poses)

    @torch.no_grad()
    def capture_at_viewpoints(self, requests, camera_poses, *, retain_images=False,
                              object_poses=None, root_poses=None):
        return self.capture_batch(requests, camera_poses=camera_poses, retain_images=retain_images,
                                  object_poses=object_poses, root_poses=root_poses)

    @torch.no_grad()
    def observe_at_viewpoint(self, requests, camera_poses, *, object_poses=None,
                             root_poses=None, retain_images=False):
        """Produce visible geometry once at a known pose, without visibility selection.

        Only actual image rays are queried. No isolated silhouettes, fractions,
        rejection, retry or candidate search are evaluated. Every request gets
        a snapshot; unobserved geometry stays invalid instead of being completed
        from privileged poses. Visibility fractions/pixel denominators are None.
        """
        if not requests:
            self.last_diagnostics = []
            return []
        ids, objects, active, roles, origins, roots, inverse = self._capture_inputs(requests, object_poses, root_poses)
        _, crop_directions, intrinsics = self._observation_grid(objects.device)
        width, height = self.cfg.observation_ray_width, self.cfg.observation_ray_height
        cameras = torch.as_tensor(camera_poses, device=objects.device, dtype=objects.dtype)
        if cameras.shape != (len(ids), 7) or not (torch.isfinite(cameras).all()
                & ((cameras[:, 3:].norm(dim=-1) - 1.).abs() <= 1.e-4).all()):
            raise ValueError('Known camera poses must be finite world xyz+wxyz unit poses [requests, 7].')
        eyes, rotations = cameras[:, :3] - origins, cameras[:, 3:]
        depths, labels = [], []
        for begin in range(0, len(ids), self.capacity):
            selection = slice(begin, begin + self.capacity)
            depth, owner = self._trace(objects[selection], active[selection], eyes[selection], rotations[selection],
                                       roles[selection], observation_only=True)
            depths.append(depth)
            labels.append(owner)
        depth, labels = torch.cat(depths), torch.cat(labels)
        points = quat_apply(rotations[:, None], depth[..., None] * crop_directions) + eyes[:, None]
        points = quat_apply(inverse[:, None], points - roots[:, None])
        geometry, valid, observed = self._fit_observation(
            points, labels, active, roles, min_depth_pixels=self.cfg.observation_min_depth_pixels)
        self.last_diagnostics = [dict(env_id=index, observation_only=True, visibility_fractions_measured=False) for index in ids]
        return [VisionSnapshot(geometry[row], valid[row], observed[row], None,
            depth[row].reshape(height, width, 1) if retain_images else None,
            (labels[row, :, None] == roles[row]).reshape(height, width, 2) if retain_images else None,
            cameras[row], intrinsics.clone(), None, None, 1) for row in range(len(ids))]

    def _compact_labels(self, labels, active_ids, capacity):
        """Map pool IDs to active slots before reducing visible points.

        Keep every active physical body, even beyond the policy's width; the
        observed geometry is ranked only after all visible surfaces are fitted.
        Padding has a separate lookup bucket so it cannot overwrite object 0.
        Snapshot IDs remain pool IDs and are mapped back after the visible fit.
        """
        count, width = active_ids.shape
        lookup = torch.full((count, self.object_count + 1), -1, device=labels.device, dtype=torch.int32)
        slots = torch.arange(width, device=labels.device, dtype=torch.int32).expand(count, -1)
        lookup.scatter_(1, torch.where(active_ids >= 0, active_ids, self.object_count).long(),
                        torch.where(active_ids >= 0, slots, -1))
        compact = lookup.gather(1, torch.where(labels >= 0, labels, self.object_count).long())
        native_ids = torch.full((count, capacity), -1, device=labels.device, dtype=torch.long)
        native_ids[:, :width] = active_ids
        return compact, native_ids

    def _fit_observation(self, points, labels, active_ids, roles, *, min_depth_pixels=None):
        """Fit every recognized physical object, then cap the policy geometry.

        All active bodies have already participated in ray occlusion. Fitting
        uses active slots instead of the potentially much larger candidate pool.
        Selection and association stay on-device and use no unobserved poses.
        """
        width = active_ids.shape[1]
        compact, native_ids = self._compact_labels(labels, active_ids, width)
        minimum = self.cfg.min_depth_pixels if min_depth_pixels is None else min_depth_pixels
        geometry, valid = fit_visible_obbs(points, compact, width, minimum)
        role_slots = (native_ids[..., None] == roles[:, None]).long().argmax(1)
        selected, selected_valid = select_initial_geometry_indices(
            geometry, valid, role_slots, self.env.cfg.mdp.max_objects)
        source = selected.clamp_min(0)
        geometry = geometry.gather(1, source[..., None].expand(-1, -1, 10))
        observed = native_ids.gather(1, source)
        return (torch.where(selected_valid[..., None], geometry, 0.), selected_valid,
                torch.where(selected_valid, observed, -1))

    def _capture_inputs(self, requests, object_poses, root_poses):
        env = self.env
        ids = [int(request['env_id']) for request in requests]
        objects = (env.scene['object_collection'].data.object_link_pose_w[ids]
                   if object_poses is None else object_poses).clone()
        origins = env.scene.env_origins[ids]
        objects[..., :3] -= origins[:, None]
        width = max(len(request['active_ids']) for request in requests)
        active_ids = np.full((len(ids), width), -1, dtype=np.int32)
        for row, request in enumerate(requests):
            active = request['active_ids']
            if (request['target_id'] == request['blocker_id'] or request['target_id'] not in active
                    or request['blocker_id'] not in active
                    or len(set(active)) != len(active)
                    or any(index < 0 or index >= self.object_count for index in active)):
                raise ValueError('Initial perception needs distinct active Target/Blocker and unique valid pool IDs.')
            active_ids[row, :len(active)] = sorted(active)
        active_ids = torch.as_tensor(active_ids, device=objects.device)
        roles = torch.tensor([[r['target_id'], r['blocker_id']] for r in requests],
                             device=objects.device, dtype=torch.int32)
        robot = env.scene['robot'].data
        roots = (robot.root_pos_w[ids] if root_poses is None else root_poses[:, :3]) - origins
        inverse_roots = quat_conjugate(robot.root_quat_w[ids] if root_poses is None else root_poses[:, 3:7])
        return ids, objects, active_ids, roles, origins, roots, inverse_roots

    def _domain_viewpoints(self, requests, role_positions):
        """Build one fixed tensor grid, with no random candidate or adaptive search."""
        cfg = self.cfg
        kwargs = dict(device=role_positions.device, dtype=role_positions.dtype)
        front, lateral, height = torch.meshgrid(
            torch.linspace(*cfg.front_distance_range, cfg.view_domain_shape[0], **kwargs),
            torch.linspace(-cfg.lateral_half_width, cfg.lateral_half_width, cfg.view_domain_shape[1], **kwargs),
            torch.linspace(*cfg.height_above_shelf, cfg.view_domain_shape[2], **kwargs), indexing='ij')
        front, lateral, height = front.flatten(), lateral.flatten(), height.flatten()
        bounds = role_positions.new_tensor([request['workspace_bounds'] for request in requests])
        tops = role_positions.new_tensor([request['shelf_top'] for request in requests])
        eyes = torch.stack((bounds[:, 1:2] + front, role_positions[:, 1, 1:2] + lateral,
                            tops[:, None] + height), -1)
        aim = role_positions.mean(1)
        aim[:, 2] = tops + .1
        forward = torch.nn.functional.normalize(aim[:, None] - eyes, dim=-1)
        up = torch.zeros_like(forward)
        up[..., 2] = 1.
        right = torch.nn.functional.normalize(torch.cross(forward, up, dim=-1), dim=-1)
        down = torch.cross(forward, right, dim=-1)
        return eyes, quat_from_matrix(torch.stack((right, down, forward), -1))

    def _visibility_failures(self, full, visible, fractions, framed):
        cfg = self.cfg
        return torch.cat((fractions[:, :1] < cfg.min_target_visible_fraction,
                          fractions[:, :1] > cfg.max_target_visible_fraction,
                          fractions[:, 1:] < cfg.min_blocker_visible_fraction,
                          full < cfg.min_silhouette_pixels, visible < cfg.min_depth_pixels, ~framed), -1)

    def _camera_mesh(self, object_id, object_pose, camera_pose):
        """Use canonical visual triangles lazily; retain only their content hash."""
        if not hasattr(self, '_mesh_id_by_object'):
            self._mesh_id_by_object = tuple(int(value) for value in self.mesh_ids.numpy())
        identifier = self._mesh_id_by_object[object_id]
        mesh = next(mesh for mesh in self.meshes if int(mesh.id) == identifier)
        vertices, indices = wp.to_torch(mesh.points), wp.to_torch(mesh.indices)
        hashes = getattr(self, '_mesh_fingerprints', {})
        if identifier not in hashes:
            digest = hashlib.sha256()
            digest.update(vertices.detach().cpu().numpy().tobytes())
            digest.update(indices.detach().cpu().numpy().tobytes())
            hashes[identifier] = digest.hexdigest()
            self._mesh_fingerprints = hashes
        points = vertices.double()
        pose, camera = object_pose.double(), camera_pose.double()
        points = quat_apply(pose[3:].expand(len(points), -1), points) + pose[:3] - camera[:3]
        points = quat_apply(quat_conjugate(camera[3:]).expand(len(points), -1), points)
        return points, points[indices.long().reshape(-1, 3)], hashes[identifier]

    def visual_footprints(self, upright_quaternions):
        """Actual visual XY hulls in each object's upright geometry frame.

        Shared meshes are copied to CPU once per call, then discarded. Returned
        tuples are immutable and shared by identical mesh/upright pairs.
        """
        from scipy.spatial import ConvexHull, QhullError
        upright = torch.as_tensor(upright_quaternions, device='cpu', dtype=torch.float64)
        if (upright.shape != (self.object_count, 4) or not torch.isfinite(upright).all()
                or not ((upright.norm(dim=-1) - 1.).abs() < 1.e-4).all()):
            raise ValueError('Supply one finite unit upright quaternion per pooled object [P,4].')
        if not hasattr(self, '_mesh_id_by_object'):
            self._mesh_id_by_object = tuple(int(value) for value in self.mesh_ids.numpy())
        native = {int(mesh.id): mesh for mesh in self.meshes}
        points, hulls, result = {}, {}, []
        for identifier, quaternion in zip(self._mesh_id_by_object, upright):
            key = (identifier, tuple(quaternion.tolist()))
            if key not in hulls:
                if identifier not in points:
                    points[identifier] = wp.to_torch(native[identifier].points).detach().cpu().double()
                xy = np.unique(quat_apply(quaternion[None], points[identifier])[:, :2].numpy(), axis=0)
                if len(xy) >= 3:
                    try:
                        xy = xy[ConvexHull(xy).vertices]
                    except QhullError:
                        pass  # A degenerate visual mesh has no 2D interior.
                hulls[key] = tuple(tuple(float(value) for value in row) for row in xy)
            result.append(hulls[key])
        return tuple(result)

    @staticmethod
    def _load_translation_cache(path, fingerprint):
        if path is not None and path.is_file():
            try:
                saved = json.loads(path.read_text())
                cached = saved['domain']
                if (cached['fingerprint'] == fingerprint
                        and hashlib.sha256(json.dumps(cached['source'], sort_keys=True).encode()).hexdigest() == fingerprint
                        and hashlib.sha256(json.dumps(cached, sort_keys=True).encode()).hexdigest() == saved['payload_sha256']):
                    return cached
            except (OSError, ValueError, KeyError, TypeError):
                pass  # A partial/corrupt cache is recomputed, never used as a certificate.
        return None

    @staticmethod
    def _save_translation_cache(path, result):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, sort_keys=True)
        envelope = dict(domain=result, payload_sha256=hashlib.sha256(payload.encode()).hexdigest())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as stream:
                temporary = stream.name
                json.dump(envelope, stream, sort_keys=True)
            os.replace(temporary, path)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _translation_hit_intervals(directions, depth_upper, triangles, translation, bounds,
                                   triangle_chunk=512, max_pairs=262144):
        """Solve affine ray/triangle hit intervals, never translated pose candidates.

        For ray e+t*d and triangle a+s*v+u*e1+w*e2, Moller-Trumbore
        t/u/w are affine in s. Linear inequality clipping gives one interval
        per ray/triangle. Fixed ray/face chunks bound intermediate GPU memory.
        Compressed intervals cross to CPU once for offline union/event sweeps.
        """
        result = []
        parallel = torch.zeros((), device=directions.device, dtype=torch.long)
        ray_chunk = max(1, max_pairs // triangle_chunk)
        for ray_begin in range(0, len(directions), ray_chunk):
            rays = directions[ray_begin:ray_begin + ray_chunk]
            upper_depth = depth_upper[ray_begin:ray_begin + ray_chunk]
            for begin in range(0, len(triangles), triangle_chunk):
                part = triangles[begin:begin + triangle_chunk]
                e1, e2 = part[:, 1] - part[:, 0], part[:, 2] - part[:, 0]
                cross = torch.cross(rays[:, None], e2[None], dim=-1)
                determinant = (e1[None] * cross).sum(-1)
                usable = determinant.abs() > 1.e-12
                parallel += (~usable).sum()
                reciprocal = torch.where(usable, determinant.reciprocal(), 0.)
                offset = -part[:, 0]
                q0 = torch.cross(offset, e1, dim=-1)
                q1 = torch.cross(-translation.expand_as(e1), e1, dim=-1)
                u0 = (offset[None] * cross).sum(-1) * reciprocal
                u1 = (-translation * cross).sum(-1) * reciprocal
                w0 = (rays[:, None] * q0[None]).sum(-1) * reciprocal
                w1 = (rays[:, None] * q1[None]).sum(-1) * reciprocal
                t0 = (e2 * q0).sum(-1)[None] * reciprocal
                t1 = (e2 * q1).sum(-1)[None] * reciprocal
                constant = torch.stack((t0 - .02, upper_depth[:, None] - t0, u0, w0, 1. - u0 - w0), -1)
                slope = torch.stack((t1, -t1, u1, w1, -u1 - w1), -1)
                positive, negative = slope > 1.e-12, slope < -1.e-12
                roots = -constant / torch.where(positive | negative, slope, 1.)
                lower = torch.where(positive, roots, -float('inf')).amax(-1).clamp_min(bounds[0])
                upper = torch.where(negative, roots, float('inf')).amin(-1).clamp_max(bounds[1])
                valid = usable & (lower < upper) & ~((~(positive | negative)) & (constant < -1.e-12)).any(-1)
                row, face = valid.nonzero(as_tuple=True)
                result.append(torch.stack(((row + ray_begin).double(), lower[row, face], upper[row, face]), -1))
        header = torch.stack((parallel.double() * 0. - 1., parallel.double(),
            parallel.double() * 0. + len(directions) * len(triangles)))[None]
        packed = torch.cat((header, *result)).cpu().numpy()
        return packed[1:], dict(near_parallel_pairs=int(packed[0, 1]), total_pairs=int(packed[0, 2]))

    @staticmethod
    def _translation_pixel_counts(intervals, ray_count, bounds):
        """Union triangle intervals per pixel before counting pixels by events."""
        merged = []
        if len(intervals):
            ordered = intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]
            groups = np.r_[0, np.flatnonzero(np.diff(ordered[:, 0])) + 1, len(ordered)]
            # One prefix maximum per hit pixel replaces a Python iteration for
            # every triangle interval. It also handles nested/touching ranges.
            for begin, end in zip(groups[:-1], groups[1:]):
                lo, hi = ordered[begin:end, 1], np.maximum.accumulate(ordered[begin:end, 2])
                starts = np.r_[True, lo[1:] > hi[:-1] + 1.e-11]
                merged.append(np.stack((lo[starts], hi[np.r_[starts[1:], True]]), -1))
        segments = np.concatenate(merged) if merged else np.empty((0, 2))
        coordinates = np.r_[bounds, segments.reshape(-1)]
        changes = np.r_[0, 0, np.tile((1, -1), len(segments))]
        coordinates, locations = np.unique(coordinates, return_inverse=True)
        delta = np.zeros(len(coordinates), dtype=np.int64)
        np.add.at(delta, locations, changes)
        return coordinates, delta.cumsum()[:-1]

    @staticmethod
    def _count_parameter_intervals(events, counts, lower, upper):
        selected = (counts >= lower) & (counts <= upper)
        result = []
        for lo, hi in np.stack((events[:-1], events[1:]), -1)[selected]:
            if result and lo <= result[-1][1] + 1.e-11:
                result[-1][1] = float(hi)
            else:
                result.append([float(lo), float(hi)])
        return result

    def _blocker_translation_domain(self, vertices, triangles, translation, bounds, mesh_hash,
                                    bbox, pixel_guard, triangle_chunk, cache_directory):
        """Target-independent silhouette curve, keyed by exact camera geometry.

        No pose rounding or approximate matching is used. Canonical identical
        Blocker inputs can share the expensive curve across Target variants.
        Internal arrays are read-only CPU data; no duplicate GPU mesh remains.
        """
        source = dict(version='blocker_translation_v1', mesh_sha256=mesh_hash,
            camera_vertices_sha256=hashlib.sha256(vertices.detach().cpu().numpy().tobytes()).hexdigest(),
            translation_camera=tuple(translation.tolist()), parameter_bounds=bounds,
            intrinsics=tuple(tuple(row) for row in self.intrinsics.tolist()),
            resolution=(self.cfg.ray_width, self.cfg.ray_height), pixel_guard=pixel_guard,
            min_silhouette_pixels=self.cfg.min_silhouette_pixels, min_depth_pixels=self.cfg.min_depth_pixels,
            min_blocker_visible_fraction=self.cfg.min_blocker_visible_fraction,
            projection_depth_bounds=(.02, 2.5), framing_margin_pixels=1.,
            numerical_tolerances=(1.e-12, 1.e-11))
        fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        memory = getattr(self, '_blocker_translation_cache', None)
        if memory is None:
            memory = self._blocker_translation_cache = {}
        path = None if cache_directory is None else Path(cache_directory) / f'blocker_{fingerprint}.json'
        if fingerprint in memory:
            if path is not None and not path.is_file():
                serializable = dict(memory[fingerprint], events=memory[fingerprint]['events'].tolist(),
                                    pixels=memory[fingerprint]['pixels'].tolist())
                self._save_translation_cache(path, serializable)
            return memory[fingerprint], 'memory'
        result = self._load_translation_cache(path, fingerprint)
        hit = 'disk' if result is not None else 'miss'
        if result is None:
            first_x, last_x, first_y, last_y = bbox
            directions = self.crop_directions.reshape(self.cfg.ray_height, self.cfg.ray_width, 3)[
                first_y:last_y + 1, first_x:last_x + 1].reshape(-1, 3).double()
            intervals, stats = self._translation_hit_intervals(directions,
                directions.new_full((len(directions),), 2.5), triangles, translation, bounds, triangle_chunk)
            events, pixels = self._translation_pixel_counts(intervals, len(directions), bounds)
            result = dict(source=source, fingerprint=fingerprint, events=tuple(map(float, events)),
                pixels=tuple(map(int, pixels)), ray_count=len(directions), triangle_intervals=len(intervals), solver=stats)
            self._save_translation_cache(path, result)
        for name, dtype in (('events', np.float64), ('pixels', np.int64)):
            result[name] = np.asarray(result[name], dtype=dtype)
            result[name].setflags(write=False)
        memory[fingerprint] = result
        return result, hit

    @torch.no_grad()
    def precompute_translation_domain(self, *, object_poses, target_id, blocker_id, camera_pose,
                                     translation, parameter_bounds, pixel_guard=2, triangle_chunk=512,
                                     parameter_margin=2.e-6, cache_directory=None):
        """Build a continuous, serializable role-pair domain at startup.

        One isolated Target query fixes its pixel silhouette and depth. Exact
        affine triangle-hit intervals give Target occlusion and Blocker pixel
        counts over s. No Blocker pose is queried or accepted during the solve.
        The certificate covers these two opaque meshes only: background must
        respect their protected view regions, and shelf/packing/IK feasibility
        must be intersected by the caller. Common rigid assembly transforms
        preserve the geometric certificate; arbitrary relative yaw does not.
        With cache_directory, a source-fingerprinted atomic JSON cache skips
        both the isolated ray query and inverse solve on later startups.
        """
        objects = torch.as_tensor(object_poses).to(dtype=torch.float32)
        camera = torch.as_tensor(camera_pose, device=objects.device, dtype=objects.dtype)
        vector = torch.as_tensor(translation, device=objects.device, dtype=torch.float64)
        bounds = tuple(float(value) for value in parameter_bounds)
        if (objects.shape != (self.object_count, 7) or camera.shape != (7,) or vector.shape != (3,)
                or target_id == blocker_id or not 0 <= target_id < self.object_count
                or not 0 <= blocker_id < self.object_count or len(bounds) != 2
                or not all(math.isfinite(value) for value in bounds) or bounds[0] >= bounds[1]
                or isinstance(pixel_guard, bool) or not isinstance(pixel_guard, int) or pixel_guard < 0
                or isinstance(triangle_chunk, bool) or not isinstance(triangle_chunk, int) or triangle_chunk < 1
                or not math.isfinite(parameter_margin) or parameter_margin < 0):
            raise ValueError('Supply distinct role IDs, pool poses [P,7], one camera [7], a vector [3] and finite ordered s bounds.')
        if not (torch.isfinite(objects).all() & torch.isfinite(camera).all() & torch.isfinite(vector).all()
                & (vector.norm() > 0) & ((camera[3:].norm() - 1.).abs() < 1.e-4)
                & ((objects[[target_id, blocker_id], 3:].norm(dim=-1) - 1.).abs() < 1.e-4).all()):
            raise ValueError('Precomputation needs finite poses, unit role/camera quaternions and a nonzero translation.')
        target_vertices, _, target_hash = self._camera_mesh(target_id, objects[target_id], camera)
        vertices, triangles, blocker_hash = self._camera_mesh(blocker_id, objects[blocker_id], camera)
        source = dict(version='inverse_translation_v2', target_mesh_sha256=target_hash, blocker_mesh_sha256=blocker_hash,
            target_id=int(target_id), blocker_id=int(blocker_id), target_pose_w=tuple(objects[target_id].tolist()),
            blocker_pose_w=tuple(objects[blocker_id].tolist()), camera_pose_w=tuple(camera.tolist()),
            translation_w=tuple(vector.tolist()), parameter_bounds_input=bounds,
            intrinsics=tuple(tuple(row) for row in self.intrinsics.tolist()),
            resolution=(self.cfg.ray_width, self.cfg.ray_height), pixel_guard=pixel_guard,
            target_fraction_bounds=(self.cfg.min_target_visible_fraction, self.cfg.max_target_visible_fraction),
            min_blocker_visible_fraction=self.cfg.min_blocker_visible_fraction,
            min_silhouette_pixels=self.cfg.min_silhouette_pixels, min_depth_pixels=self.cfg.min_depth_pixels,
            projection_depth_bounds=(.02, 2.5), framing_margin_pixels=1., structural_depth_margin=1.e-6,
            optical_assumptions=dict(projection='opaque_pinhole', camera_axes='+X right, +Y down, +Z forward',
                ray_depth='camera Z; unnormalized pixel directions', occluders='Blocker mesh only',
                target_silhouette='isolated actual mesh at the fixed camera pose'),
            parameter_margin=parameter_margin)
        fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        cache_path = None if cache_directory is None else Path(cache_directory) / f'{fingerprint}.json'
        cached = self._load_translation_cache(cache_path, fingerprint)
        if cached is not None:
            cached['diagnostics']['persistent_cache_hit'] = True
            return cached
        camera_vector = quat_apply(quat_conjugate(camera[3:].double())[None], vector[None])[0]
        fx, fy = float(self.intrinsics[0, 0]), float(self.intrinsics[1, 1])
        cx, cy = float(self.intrinsics[0, 2]), float(self.intrinsics[1, 2])
        xlo, xhi = (1. - cx) / fx, (self.cfg.ray_width - 1. - cx) / fx
        ylo, yhi = (1. - cy) / fy, (self.cfg.ray_height - 1. - cy) / fy
        x, y, z = vertices.unbind(-1)
        vx, vy, vz = camera_vector.unbind(-1)
        constant = torch.stack((z - .02, 2.5 - z, x - xlo * z, xhi * z - x,
                                y - ylo * z, yhi * z - y), -1)
        slope = torch.stack((vz, -vz, vx - xlo * vz, xhi * vz - vx, vy - ylo * vz, yhi * vz - vy))
        positive, negative = slope > 1.e-12, slope < -1.e-12
        roots = -constant / torch.where(positive | negative, slope, 1.)
        lo = max(bounds[0], float(torch.where(positive, roots, -float('inf')).amax()))
        hi = min(bounds[1], float(torch.where(negative, roots, float('inf')).amin()))
        if lo >= hi or bool(((~(positive | negative)) & (constant < 0)).any()):
            raise InitialVisionRejected('Empty precompute domain: Blocker cannot stay fully inside the image and depth range.')
        bounds = (lo, hi)
        depth_gap = float(target_vertices[:, 2].amin() - vertices[:, 2].amax()
                          - torch.maximum(camera_vector[2] * lo, camera_vector[2] * hi))
        if depth_gap <= 1.e-6:
            raise InitialVisionRejected('Blocker/Target camera-depth intervals overlap; Blocker observability is not structurally certified.')
        local_objects = objects.clone()
        local_objects[:, :3] -= camera[:3]
        roles = torch.tensor([[target_id, blocker_id]], device=objects.device, dtype=torch.int32)
        active = torch.tensor([[target_id, -1]], device=objects.device, dtype=torch.int32)
        depth, labels, full, _, _, _ = self._trace(local_objects[None], active,
            objects.new_zeros((1, 3)), camera[None, 3:], roles)
        target_mask = labels[0] == target_id
        target_pixels = int(full[0, 0])
        if not bool(self._last_trace_framed[0, 0]) or target_pixels != int(target_mask.sum()):
            raise InitialVisionRejected('Fixed Target silhouette is not completely framed.')
        if target_pixels < self.cfg.min_silhouette_pixels + pixel_guard:
            raise InitialVisionRejected('Fixed Target silhouette has too few pixels for the requested guard.')
        target_intervals, target_stats = self._translation_hit_intervals(self.crop_directions[target_mask].double(),
            depth[0, target_mask].double(), triangles, camera_vector, bounds, triangle_chunk)
        target_events, target_occluded = self._translation_pixel_counts(target_intervals, target_pixels, bounds)
        minimum_visible = max(self.cfg.min_depth_pixels + pixel_guard,
            math.ceil(self.cfg.min_target_visible_fraction * target_pixels - 1.e-9) + pixel_guard)
        maximum_visible = math.floor(self.cfg.max_target_visible_fraction * target_pixels + 1.e-9) - pixel_guard
        target_allowed = self._count_parameter_intervals(target_events, target_pixels - target_occluded,
                                                         minimum_visible, maximum_visible)
        # Positive depth makes each projected vertex monotone in s. Endpoint
        # bounds therefore cover every intermediate silhouette, with padding.
        endpoints = vertices[None] + vertices.new_tensor(bounds)[:, None, None] * camera_vector
        image = endpoints[..., :2] / endpoints[..., 2:3]
        lower, upper = image.amin((0, 1)), image.amax((0, 1))
        first_x = max(0, math.floor(float(lower[0]) * fx + cx) - 1)
        last_x = min(self.cfg.ray_width - 1, math.ceil(float(upper[0]) * fx + cx) + 1)
        first_y = max(0, math.floor(float(lower[1]) * fy + cy) - 1)
        last_y = min(self.cfg.ray_height - 1, math.ceil(float(upper[1]) * fy + cy) + 1)
        blocker_domain, blocker_cache_hit = self._blocker_translation_domain(vertices, triangles, camera_vector,
            bounds, blocker_hash, (first_x, last_x, first_y, last_y), pixel_guard, triangle_chunk, cache_directory)
        blocker_events, blocker_pixels = blocker_domain['events'], blocker_domain['pixels']
        blocker_minimum = max(self.cfg.min_silhouette_pixels, self.cfg.min_depth_pixels) + pixel_guard
        blocker_allowed = self._count_parameter_intervals(blocker_events, blocker_pixels, blocker_minimum, blocker_domain['ray_count'])
        eligible = []
        for tlo, thi in target_allowed:
            for blo, bhi in blocker_allowed:
                start, end = max(tlo, blo) + parameter_margin, min(thi, bhi) - parameter_margin
                if start < end:
                    eligible.append((start, end))
        if not eligible:
            raise InitialVisionRejected('Empty continuous domain after Target visibility and guarded Blocker silhouette constraints.')
        # These bounds come from the visual surface mesh, rather than USD
        # collision approximations. The caller can conservatively protect its
        # swept camera cone and intersect support limits before rigid assembly
        # transforms; no unseen visual geometry is added to the observation.
        visual_support = {}
        for role, local_vertices, swept in (('target', target_vertices, target_vertices[None]),
                                           ('blocker', vertices, endpoints)):
            world_vertices = quat_apply(camera[None, 3:].double(), local_vertices) + camera[None, :3].double()
            world_swept = quat_apply(camera[None, None, 3:].double(), swept) + camera[None, None, :3].double()
            projected = swept[..., :2] / swept[..., 2:3]
            visual_support[role] = dict(
                world_xyz_bounds=tuple(tuple(row) for row in torch.stack((world_vertices.amin(0), world_vertices.amax(0))).tolist()),
                swept_world_xyz_bounds=tuple(tuple(row) for row in torch.stack((world_swept.amin((0, 1)), world_swept.amax((0, 1)))).tolist()),
                camera_xyz_bounds=tuple(tuple(row) for row in torch.stack((local_vertices.amin(0), local_vertices.amax(0))).tolist()),
                swept_camera_xyz_bounds=tuple(tuple(row) for row in torch.stack((swept.amin((0, 1)), swept.amax((0, 1)))).tolist()),
                normalized_image_xy_bounds=tuple(tuple(row) for row in torch.stack((projected.amin((0, 1)), projected.amax((0, 1)))).tolist()))
        result = dict(valid_intervals=tuple(eligible), certified_parameter_bounds=bounds,
            target_visibility_events=tuple(map(float, target_events)), target_occluded_counts=tuple(map(int, target_occluded)),
            target_pixel_count=target_pixels, target_visible_pixel_bounds=(minimum_visible, maximum_visible),
            blocker_silhouette_events=tuple(map(float, blocker_events)), blocker_silhouette_counts=tuple(map(int, blocker_pixels)),
            minimum_blocker_pixels=blocker_minimum, structural_depth_gap=depth_gap,
            blocker_visible_fraction_structural=1., blocker_projected_bbox=(first_x, last_x, first_y, last_y),
            visual_support=visual_support,
            candidate_pose_queries=0, isolated_target_queries=1,
            diagnostics=dict(target_solver=target_stats, blocker_solver=blocker_domain['solver'],
                target_triangle_intervals=len(target_intervals), blocker_triangle_intervals=blocker_domain['triangle_intervals'],
                blocker_subcache_hit=blocker_cache_hit, blocker_subcache_fingerprint=blocker_domain['fingerprint'],
                persistent_cache_hit=False,
                numerical_scope='Float64 affine domains with pixel/parameter guards; not a formal floating-point certificate.'),
            source=source, fingerprint=fingerprint)
        self._save_translation_cache(cache_path, result)
        return result

    @torch.no_grad()
    def capture_view_domain(self, requests, *, object_poses, root_poses, retain_images=False,
                            preferred_camera_poses=None):
        """Evaluate a fixed camera domain once and fit only its selected view.

        Every 2×9×2 grid member (plus an optional preferred view) is evaluated,
        even after finding a valid view. Fixed chunks bound ray memory; there
        is no failed-row queue, new candidate, layout redraw or second trace.
        Selection minimizes |Target visibility - .65|, then maximizes Blocker
        visibility, then chooses the first index. All selection stays on-device.
        """
        if not requests:
            self.last_diagnostics = []
            return []
        ids, objects, active, roles, origins, roots, inverse = self._capture_inputs(requests, object_poses, root_poses)
        cfg, count = self.cfg, len(requests)
        role_positions = objects.gather(1, roles.long()[..., None].expand(-1, -1, 3))
        eyes, rotations = self._domain_viewpoints(requests, role_positions)
        if preferred_camera_poses is not None:
            preferred = torch.as_tensor(preferred_camera_poses, device=objects.device, dtype=objects.dtype)
            if preferred.shape != (count, 7) or not (torch.isfinite(preferred).all()
                    & ((preferred[:, 3:].norm(dim=-1) - 1.).abs() <= 1.e-4).all()):
                raise ValueError('Preferred camera poses must be finite world xyz+wxyz unit poses [requests, 7].')
            eyes = torch.cat(((preferred[:, :3] - origins)[:, None], eyes), 1)
            rotations = torch.cat((preferred[:, None, 3:], rotations), 1)
        views, rays = eyes.shape[1], cfg.ray_width * cfg.ray_height
        best_distance = objects.new_full((count,), float('inf'))
        best_blocker = objects.new_full((count,), -1.)
        best_index = torch.full((count,), views, device=objects.device, dtype=torch.long)
        best_depth = objects.new_zeros((count, rays))
        best_labels = torch.full((count, rays), -1, device=objects.device, dtype=torch.int32)
        statistics = objects.new_zeros((count, views, 8))
        valid_counts = torch.zeros(count, device=objects.device, dtype=torch.long)
        failure_counts = torch.zeros((count, 9), device=objects.device, dtype=torch.long)
        flat_eyes, flat_rotations = eyes.flatten(0, 1), rotations.flatten(0, 1)
        # These are fixed memory chunks of the complete domain, never retry rounds.
        for begin in range(0, count * views, self.capacity):
            indices = torch.arange(begin, min(begin + self.capacity, count * views), device=objects.device)
            rows, columns = indices // views, indices % views
            depth, labels, full, visible, fractions, accepted = self._trace(
                objects[rows], active[rows], flat_eyes[indices], flat_rotations[indices], roles[rows])
            framed = self._last_trace_framed
            statistics[rows, columns] = torch.cat((fractions,
                torch.stack((visible, full), -1).flatten(1), framed), -1)
            valid_counts.index_add_(0, rows, accepted.long())
            failure_counts.index_add_(0, rows, self._visibility_failures(full, visible, fractions, framed).long())
            distance = torch.where(accepted, (fractions[:, 0] - .65).abs(), float('inf'))
            chunk_distance = objects.new_full((count,), float('inf')).scatter_reduce_(
                0, rows, distance, reduce='amin', include_self=True)
            eligible = accepted & (distance == chunk_distance[rows])
            chunk_blocker = objects.new_full((count,), -1.).scatter_reduce_(
                0, rows, torch.where(eligible, fractions[:, 1], -1.), reduce='amax', include_self=True)
            eligible &= fractions[:, 1] == chunk_blocker[rows]
            chunk_index = torch.full_like(best_index, views).scatter_reduce_(
                0, rows, torch.where(eligible, columns, views), reduce='amin', include_self=True)
            same_distance = chunk_distance == best_distance
            improve = ((chunk_distance < best_distance)
                | (same_distance & (chunk_blocker > best_blocker))
                | (same_distance & (chunk_blocker == best_blocker) & (chunk_index < best_index)))
            improve &= chunk_index < views
            source = (torch.arange(count, device=objects.device) * views + chunk_index - begin).clamp(0, len(indices) - 1)
            best_depth.copy_(torch.where(improve[:, None], depth[source], best_depth))
            best_labels.copy_(torch.where(improve[:, None], labels[source], best_labels))
            best_distance = torch.where(improve, chunk_distance, best_distance)
            best_blocker = torch.where(improve, chunk_blocker, best_blocker)
            best_index = torch.where(improve, chunk_index, best_index)
        # Failed requests keep a closest-view diagnostic, never a fabricated snapshot.
        selected = torch.where(valid_counts > 0, best_index, (statistics[..., 0] - .65).abs().argmin(1))
        rows = torch.arange(count, device=objects.device)
        selected_eyes, selected_rotations = eyes[rows, selected], rotations[rows, selected]
        points = quat_apply(selected_rotations[:, None], best_depth[..., None] * self.crop_directions) + selected_eyes[:, None]
        points = quat_apply(inverse[:, None], points - roots[:, None])
        geometry, valid, observed = self._fit_observation(points, best_labels, active, roles)
        cameras = torch.cat((selected_eyes + origins, selected_rotations), -1)
        metadata = torch.cat((statistics[rows, selected], failure_counts,
            valid_counts[:, None], selected[:, None], statistics[..., :2].amin(1), statistics[..., :2].amax(1)), -1).tolist()
        criteria = ('target_below_min', 'target_above_max', 'blocker_below_min',
            'target_silhouette_too_small', 'blocker_silhouette_too_small',
            'target_depth_too_small', 'blocker_depth_too_small', 'target_out_of_frame', 'blocker_out_of_frame')
        results, self.last_diagnostics = [], []
        for row, detail in enumerate(metadata):
            passed = int(detail[17]) > 0
            self.last_diagnostics.append(dict(env_id=ids[row], accepted=passed, attempts=views, view_count=views,
                valid_candidate_count=int(detail[17]), selected_candidate_index=int(detail[18]) if passed else None,
                closest_candidate_index=int(detail[18]), target_visible_fraction=detail[0], blocker_visible_fraction=detail[1],
                target_pixel_counts=(int(detail[2]), int(detail[3])), blocker_pixel_counts=(int(detail[4]), int(detail[5])),
                framed=(bool(detail[6]), bool(detail[7])),
                failure_counts={name: int(value) for name, value in zip(criteria, detail[8:17]) if value},
                min_visible_fractions=tuple(detail[19:21]), max_visible_fractions=tuple(detail[21:23])))
            results.append(VisionSnapshot(geometry[row], valid[row], observed[row], None,
                best_depth[row].reshape(cfg.ray_height, cfg.ray_width, 1) if retain_images else None,
                (best_labels[row, :, None] == roles[row]).reshape(cfg.ray_height, cfg.ray_width, 2) if retain_images else None,
                cameras[row], self.intrinsics.clone(), tuple(detail[:2]),
                ((int(detail[2]), int(detail[3])), (int(detail[4]), int(detail[5]))), views) if passed else None)
        return results

    def _trace(self, poses, active_ids, eyes, orientations, roles, *, observation_only=False):
        count, device = len(eyes), eyes.device
        if observation_only:
            width, height = self.cfg.observation_ray_width, self.cfg.observation_ray_height
            directions, _, _ = self._observation_grid(device)
        else:
            width, height = self.cfg.ray_width, self.cfg.ray_height
            directions = self.directions
        depths = torch.empty((count, width * height), device=device)
        labels = torch.empty((count, width * height), device=device, dtype=torch.int32)
        silhouettes = torch.zeros((count, 1 if observation_only else 4 * width * height, 2),
                                  device=device, dtype=torch.int32)
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
        launch = dict(kernel=_initial_surface_rays, dim=(count, (1 if observation_only else 4) * width * height), inputs=[
            self.mesh_ids, wp.from_torch(frames, dtype=wp.transform), wp.from_torch(active_ids),
            wp.from_torch(role_slots), wp.from_torch(projected, dtype=wp.vec4),
            wp.from_torch(directions, dtype=wp.vec3), wp.from_torch(roles), width, height, 2.5, int(observation_only),
            wp.from_torch(depths), wp.from_torch(labels), wp.from_torch(silhouettes)])
        wp.launch(**launch, **execution)
        if observation_only:
            return depths, labels
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
                 & (fractions[:, 0] <= self.cfg.max_target_visible_fraction)
                 & (fractions[:, 1] >= self.cfg.min_blocker_visible_fraction))
        self._last_trace_framed = framed
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

    def _capture_pool(self, requests, retain_images, object_poses=None, root_poses=None, camera_poses=None):
        env, cfg = self.env, self.cfg
        ids, objects, active_ids, roles, origins, roots, inverse_roots = self._capture_inputs(requests, object_poses, root_poses)
        if camera_poses is None:
            role_positions = objects.gather(1, roles.long()[..., None].expand(-1, -1, 3))
            candidate_eyes, candidate_rotations = self._viewpoints(requests, role_positions)
            attempt_limit = cfg.max_camera_attempts
        else:
            camera_poses = torch.as_tensor(camera_poses, device=objects.device, dtype=objects.dtype)
            if camera_poses.shape != (len(ids), 7):
                raise ValueError('Explicit camera poses must have shape [requests, 7] in world xyz+wxyz.')
            if not (torch.isfinite(camera_poses).all()
                    & ((camera_poses[:, 3:].norm(dim=-1) - 1.).abs() <= 1.e-4).all()):
                raise ValueError('Explicit camera poses must be finite with unit wxyz quaternions.')
            candidate_eyes = (camera_poses[:, :3] - origins)[:, None]
            candidate_rotations = camera_poses[:, None, 3:]
            attempt_limit = 1
        results, outputs = [None] * len(ids), [None] * len(ids)
        metadata = objects.new_zeros((len(ids), 6))
        diagnostic_values = objects.new_zeros((len(ids), 9))
        failure_counts = torch.zeros((len(ids), 9), device=objects.device, dtype=torch.long)
        maxima = objects.new_zeros((len(ids), 2))
        next_attempt = [0] * len(ids)
        next_request = min(self.capacity, len(ids))
        pending = list(range(next_request))
        candidates_per_env = min(cfg.ray_candidates_per_env, attempt_limit)
        offsets = torch.arange(candidates_per_env, device=env.device)
        while pending:
            rows = torch.tensor(pending, device=env.device, dtype=torch.long)
            attempts = torch.tensor([next_attempt[row] for row in pending], device=env.device)[:, None] + offsets
            view_rows = rows.repeat_interleave(candidates_per_env)
            view_attempts = attempts.clamp_max(attempt_limit - 1).flatten()
            eyes = candidate_eyes[view_rows, view_attempts]
            orientation = candidate_rotations[view_rows, view_attempts]
            depth, labels, full, visible, fractions, accepted = self._trace(
                objects[view_rows], active_ids[view_rows], eyes, orientation, roles[view_rows])
            framed = getattr(self, '_last_trace_framed', torch.ones_like(full, dtype=torch.bool))
            failures = self._visibility_failures(full, visible, fractions, framed)
            in_budget = attempts < attempt_limit
            failure_counts[rows] += (failures.reshape(len(pending), candidates_per_env, 9)
                                     & in_budget[..., None]).sum(1)
            maxima[rows] = torch.maximum(maxima[rows], fractions.reshape(len(pending), candidates_per_env, 2).amax(1))
            accepted = accepted.reshape(len(pending), candidates_per_env) & in_budget
            # One bounded host transfer per proposal group, never one per candidate/pixel/object.
            first = torch.where(accepted, offsets, candidates_per_env).amin(-1).tolist()
            final_slots = torch.tensor([candidate if candidate < candidates_per_env else
                                        min(candidates_per_env, attempt_limit - next_attempt[row]) - 1
                                        for row, candidate in zip(pending, first)], device=objects.device)
            final_views = torch.arange(len(pending), device=objects.device) * candidates_per_env + final_slots
            diagnostic_values[rows] = torch.cat((fractions[final_views],
                torch.stack((visible[final_views], full[final_views]), -1).flatten(1),
                framed[final_views], (attempts.gather(1, final_slots[:, None]) + 1)), -1)
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
                geometry, valid, observed = self._fit_observation(points_b, labels, active_ids[rows], roles[rows])
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
                if first[slot] == candidates_per_env and next_attempt[row] < attempt_limit:
                    remaining.append(row)
            end = min(len(ids), next_request + self.capacity - len(remaining))
            pending = remaining + list(range(next_request, end))
            next_request = end
        # Diagnostic tuples cross to the host once, after the entire request queue finishes.
        summaries = metadata.tolist()
        diagnostics = torch.cat((diagnostic_values, failure_counts, maxima), -1).tolist()
        criteria = ('target_below_min', 'target_above_max', 'blocker_below_min',
                    'target_silhouette_too_small', 'blocker_silhouette_too_small',
                    'target_depth_too_small', 'blocker_depth_too_small',
                    'target_out_of_frame', 'blocker_out_of_frame')
        self.last_diagnostics = []
        for row, output in enumerate(outputs):
            detail = diagnostics[row]
            counts = {name: int(count) for name, count in zip(criteria, detail[9:18]) if count}
            self.last_diagnostics.append(dict(env_id=ids[row], accepted=output is not None,
                attempts=int(detail[8]), target_visible_fraction=detail[0], blocker_visible_fraction=detail[1],
                target_pixel_counts=(int(detail[2]), int(detail[3])),
                blocker_pixel_counts=(int(detail[4]), int(detail[5])),
                framed=(bool(detail[6]), bool(detail[7])), failure_counts=counts,
                max_visible_fractions=tuple(detail[18:20])))
            if output is not None:
                geometry, valid, observed, depth, masks, camera, attempt = output
                summary = summaries[row]
                results[row] = VisionSnapshot(geometry, valid, observed, None, depth, masks, camera,
                    self.intrinsics.clone(), tuple(summary[:2]),
                    ((int(summary[2]), int(summary[3])), (int(summary[4]), int(summary[5]))), attempt)
        return results
