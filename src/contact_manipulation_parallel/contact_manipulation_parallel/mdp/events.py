"""Episode/reset bookkeeping shared by native observation/reward/done terms.

Pure tensor calculation comes first, followed by native scene-data adapters.
One batched update per common_step_counter keeps history and task transitions
independent of manager call order. Only selected rows change on partial reset.
"""

from dataclasses import dataclass, field, fields, replace
from collections.abc import Mapping
from contextlib import contextmanager
from functools import lru_cache
import math
import random
import json
import os
import hashlib
import shutil
import tarfile
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace

import numpy as np

import torch
from numba import njit
import warp as wp
from isaaclab.utils.math import convert_quat, matrix_from_quat

from .observations import (
    SI_200_8_RANGES, SI_200_8_RESOLUTION, calibrate_wrench,
    adjacent_geometry_permutation, dynamic_sample, geometry_in_eef_frame,
    pack_policy_observation, prepare_initial_geometry,
    quat_apply, quat_conjugate, quat_mul,
)
from .rewards import clearance_quality, object_motion_cost, singularity_margin
from .terminations import (
    FAILURE_REASON_NAMES, configured_terminations, object_support_validity, task_conditions, terminal_transition,
)

_DEFAULT_MAX_OBJECTS = 10  # Policy slots; physical scene count is configured separately.


@dataclass
class MdpCfg:
    max_objects: int = _DEFAULT_MAX_OBJECTS
    # Geometric warning/hard margins; initial tuning, not certified safety limits.
    # L matches the existing IK angular weighting; s is inverse-Frobenius margin.
    singularity_characteristic_length: float = 0.25
    motion_singularity_threshold: float = 0.07
    motion_singularity_decay: float = math.log(2.0) / 0.03**2
    singularity_termination_threshold: float = 0.005
    # One-time terminal payments; failure_penalty is a positive cost magnitude.
    success_reward: float = 250.0
    failure_penalty: float = 3.0
    success_overlap_area: float = 1.0e-5
    target_translation_tolerance: float = 0.02
    target_rotation_tolerance: float = 0.15
    stable_linear_velocity: float = 0.05
    stable_angular_velocity: float = 0.30
    success_duration: float = 0.50
    # Geometric proxies, not measured contact forces. Validate with resting
    # collider geometry; progress permits motion and small recovering pivots.
    progress_support_gap: float = 0.010
    success_support_gap: float = 0.005
    support_penetration_tolerance: float = 0.005
    success_xy_tolerance: float = 0.005
    support_warning_distance: float = 0.010
    support_stability_decay: float = math.log(2.0) / 0.010**2
    # User-selected linear tilt-angle exponent: isolated 45-degree tilt has half quality.
    stability_tilt_decay: float = math.log(2.0) / (math.pi / 4.0)
    support_loss_distance: float = 0.030
    support_loss_duration: float = 0.100
    max_tilt: float = math.radians(80.0)
    tilt_duration: float = 0.20
    joint_position_scale: float = math.pi
    joint_velocity_scale: float = 2.0
    tactile_threshold: float = 0.01
    shelf_translation_tolerance: float = 0.01
    shelf_rotation_tolerance: float = math.radians(2.0)

    def __post_init__(self):
        if self.max_objects < 2:
            raise ValueError("Observation capacity must include distinct Target and Blocker slots.")
        if min(self.success_duration, self.support_loss_duration) <= 0:
            raise ValueError("Success and physical-failure durations must be positive.")
        if not (self.success_support_gap <= self.progress_support_gap < self.support_loss_distance
                and 0 <= self.support_warning_distance < self.support_loss_distance
                and self.support_stability_decay > 0):
            raise ValueError("Support warning/completion ranges must precede the loss threshold.")
        if not 0 < self.max_tilt <= math.pi or self.tilt_duration <= 0:
            raise ValueError("Fall angle must be in (0, pi] and tilt_duration must be positive.")
        if not (self.singularity_characteristic_length > 0
                and 0 < self.singularity_termination_threshold < self.motion_singularity_threshold):
            raise ValueError("Singularity requires a positive length and hard margin below the warning margin.")
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            if (name.endswith("_ref") or name.endswith("_scale")) and value <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass
class TransitionBatch:
    """Completed physics/control sample; all tensors already on the environment device."""

    center: torch.Tensor               # N,O,3; fixed environment coordinates
    corners: torch.Tensor              # N,O,8,3; fixed corner correspondence
    rotation: torch.Tensor             # N,O,4; relative to canonical upright
    velocity: torch.Tensor             # N,O,6; linear velocity at the OBB center
    sensor_wrench: torch.Tensor        # N,6; SI-200-8 measured, sensor axes, includes self-load
    sensor_position: torch.Tensor      # N,3; world
    sensor_rotation: torch.Tensor      # N,4; world, wxyz
    sensor_range_exceeded: torch.Tensor  # N,6; before clipping
    shelf_pose: torch.Tensor           # N,7; world link pose, wxyz
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


_ProjectedPolygon = wp.types.matrix(shape=(12, 2), dtype=wp.float32)


@wp.kernel(enable_backward=False)
def _projected_areas(corners: wp.array4d(dtype=wp.float32), active: wp.array2d(dtype=wp.bool),
                     regions: wp.array2d(dtype=wp.float32), areas: wp.array2d(dtype=wp.float32),
                     overlaps: wp.array2d(dtype=wp.float32)):
    row, obj = wp.tid()
    areas[row, obj] = 0.0
    overlaps[row, obj] = 0.0
    if not active[row, obj]:
        return
    polygon = _ProjectedPolygon()
    center = wp.vec2(0.0)
    for i in range(8):
        point = wp.vec2(corners[row, obj, i, 0], corners[row, obj, i, 1])
        center += point * 0.125
        polygon[i, 0] = point[0]
        polygon[i, 1] = point[1]
    # Stable insertion sort matches the tensor oracle's angular ordering.
    for i in range(1, 8):
        x, y = polygon[i, 0], polygon[i, 1]
        angle = wp.atan2(y - center[1], x - center[0])
        j = i - 1
        while j >= 0:
            previous_angle = wp.atan2(polygon[j, 1] - center[1], polygon[j, 0] - center[0])
            if previous_angle <= angle:
                break
            polygon[j + 1, 0] = polygon[j, 0]
            polygon[j + 1, 1] = polygon[j, 1]
            j -= 1
        polygon[j + 1, 0] = x
        polygon[j + 1, 1] = y
    count = int(8)
    for iteration in range(8):
        compact = _ProjectedPolygon()
        written = int(0)
        for i in range(count):
            before, after = (i + count - 1) % count, (i + 1) % count
            dx, dy = polygon[i, 0] - polygon[before, 0], polygon[i, 1] - polygon[before, 1]
            ex, ey = polygon[after, 0] - polygon[i, 0], polygon[after, 1] - polygon[i, 1]
            if dx * ey - dy * ex >= -1.0e-10 and dx * dx + dy * dy >= 1.0e-16:
                compact[written, 0] = polygon[i, 0]
                compact[written, 1] = polygon[i, 1]
                written += 1
        polygon, count = compact, written
    area = float(0.0)
    for i in range(count):
        j = (i + 1) % count
        ax, ay = polygon[i, 0] - polygon[0, 0], polygon[i, 1] - polygon[0, 1]
        bx, by = polygon[j, 0] - polygon[0, 0], polygon[j, 1] - polygon[0, 1]
        area += ax * by - ay * bx
    areas[row, obj] = wp.abs(area) * 0.5
    for edge in range(4):
        clipped = _ProjectedPolygon()
        written = int(0)
        axis, sign = edge // 2, float(1.0)
        if edge % 2 == 1:
            sign = -1.0
        limit = regions[row, edge]
        for i in range(count):
            j = (i + count - 1) % count
            inside = (polygon[i, axis] - limit) * sign >= -1.0e-9
            before_inside = (polygon[j, axis] - limit) * sign >= -1.0e-9
            if inside != before_inside:
                denominator = polygon[i, axis] - polygon[j, axis]
                if wp.abs(denominator) <= 1.0e-12:
                    denominator = 1.0
                fraction = wp.clamp((limit - polygon[j, axis]) / denominator, 0.0, 1.0)
                for a in range(2):
                    clipped[written, a] = polygon[j, a] + fraction * (polygon[i, a] - polygon[j, a])
                written += 1
            if inside:
                clipped[written, 0] = polygon[i, 0]
                clipped[written, 1] = polygon[i, 1]
                written += 1
        polygon, count = clipped, written
    area = float(0.0)
    for i in range(count):
        j = (i + 1) % count
        ax, ay = polygon[i, 0] - polygon[0, 0], polygon[i, 1] - polygon[0, 1]
        bx, by = polygon[j, 0] - polygon[0, 0], polygon[j, 1] - polygon[0, 1]
        area += ax * by - ay * bx
    overlaps[row, obj] = wp.abs(area) * 0.5


def projected_areas(corners, active, region):
    if corners.is_cuda and corners.dtype == torch.float32:
        wp.init()
        areas = corners.new_empty(active.shape)
        overlaps = torch.empty_like(areas)
        wp.launch(_projected_areas, dim=active.shape, inputs=[wp.from_torch(corners), wp.from_torch(active),
                  wp.from_torch(region), wp.from_torch(areas), wp.from_torch(overlaps)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(corners.device)))
        return areas, overlaps
    polygon, count = convex_hull_xy(corners[..., :2])
    areas = polygon_area(polygon, count)
    overlaps = rectangle_overlap_area(polygon, count, region[:, None, :])
    return torch.where(active, areas, 0.0), torch.where(active, overlaps, 0.0)


def object_geometry(pose, local_center, local_corners, upright):
    """Transform cached upright catalog bounds, accounting for off-center USD origins."""
    rotation = quat_mul(pose[..., 3:7], quat_conjugate(upright))
    center = pose[..., :3] + quat_apply(rotation, local_center)
    corners = pose[..., None, :3] + quat_apply(rotation[..., None, :].expand(*local_corners.shape[:-1], 4), local_corners)
    return center, corners, rotation


@wp.kernel(enable_backward=False)
def _collision_support_minimum(vertices: wp.array3d(dtype=wp.float32), counts: wp.array(dtype=wp.int64),
                               object_ids: wp.array2d(dtype=wp.int64), normals: wp.array3d(dtype=wp.float32),
                               centers: wp.array3d(dtype=wp.float32), output: wp.array2d(dtype=wp.float32)):
    row, obj, lane = wp.tid()
    asset = int(object_ids[row, obj])
    minimum = float(wp.inf)
    if asset >= 0 and asset < counts.shape[0]:
        normal = wp.vec3(normals[row, obj, 0], normals[row, obj, 1], normals[row, obj, 2])
        center = wp.vec3(centers[row, obj, 0], centers[row, obj, 1], centers[row, obj, 2])
        for vertex in range(lane, int(counts[asset]), 128):
            point = wp.vec3(vertices[asset, vertex, 0], vertices[asset, vertex, 1], vertices[asset, vertex, 2])
            minimum = wp.min(minimum, wp.dot(point - center, normal))
    # The last launch dimension is exactly one CUDA block. Lanes read adjacent
    # vertices and reduce locally; neither an expanded N*O*V tensor nor atomics
    # are needed, even for the largest catalog collision mesh.
    tile = wp.tile(minimum)
    reduced = wp.tile_min(tile)
    wp.tile_store(output[row], reduced, offset=obj)


def collision_support_minimum(vertices_catalog, vertex_counts, object_ids, normals, local_center):
    """Exact directional collision minima from a shared, unexpanded catalog.

    The CUDA path launches one block per environment/object and skips padded
    catalog vertices using persistent counts. Only its (N,O) result is allocated
    per sample. The CPU implementation is an oracle for small standalone tests.
    """
    if (vertices_catalog.ndim != 3 or vertices_catalog.shape[-1] != 3
            or vertex_counts.shape != vertices_catalog.shape[:1]
            or vertex_counts.dtype != torch.long or object_ids.ndim != 2 or object_ids.dtype != torch.long
            or normals.shape != (*object_ids.shape, 3) or local_center.shape != normals.shape):
        raise ValueError("Supply catalog vertices/counts, int64 object IDs, and matching normals/centers.")
    if any(value.device != normals.device for value in (vertices_catalog, vertex_counts, object_ids, local_center)):
        raise ValueError("Collision support catalog, IDs, normals, and centers must share a device.")
    if normals.is_cuda:
        if (normals.dtype != torch.float32 or vertices_catalog.dtype != torch.float32
                or local_center.dtype != torch.float32):
            raise ValueError("The CUDA collision support kernel requires float32 geometry.")
        wp.init()
        output = normals.new_empty(object_ids.shape)
        wp.launch(_collision_support_minimum, dim=(*object_ids.shape, 128), block_dim=128,
                  inputs=[wp.from_torch(vertices_catalog), wp.from_torch(vertex_counts), wp.from_torch(object_ids),
                          wp.from_torch(normals), wp.from_torch(local_center), wp.from_torch(output)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(normals.device)))
        return output
    output = normals.new_full(object_ids.shape, math.inf)
    for asset in range(vertices_catalog.shape[0]):
        selected = object_ids == asset
        count = int(vertex_counts[asset])
        if count and selected.any():
            directions = normals[selected]
            output[selected] = ((vertices_catalog[asset, :count] @ directions.mT).amin(0)
                                - (local_center[selected] * directions).sum(-1))
    return output


class TaskState:
    """Cached task state with frozen actor vision and tracked-object critic geometry."""

    def __init__(self, num_envs, device, cfg, dt, *, dtype=torch.float32, step=0, termination_cfg=None,
                 active_capacity=None):
        cfg.__post_init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("step_dt must be finite and positive.")
        self.cfg, self.dt = cfg, dt
        self.last_step = step

        self.ready = False
        self.bindings = None
        self.observation = None
        self.critic_observation = None
        self.empty_env_ids = torch.empty(0, dtype=torch.long, device=device)
        n, observed = num_envs, cfg.max_objects
        o = observed if active_capacity is None else active_capacity
        if not isinstance(o, int) or isinstance(o, bool) or o < 2:
            raise ValueError("Physical object capacity must include distinct Target and Blocker.")
        self.active_capacity = o
        def zeros(*shape, kind=dtype):
            return torch.zeros(shape, device=device, dtype=kind)
        self.geometry_b = zeros(n, observed, 10)
        self.geometry_valid = zeros(n, observed, kind=torch.bool)
        self.critic_geometry_b = zeros(n, observed, 10)
        self.critic_source_indices = torch.full((n, observed), -1, dtype=torch.long, device=device)
        self.tcp_pose_b = zeros(n, 7)
        self.history = zeros(n, 2, 72)
        self.active = zeros(n, o, kind=torch.bool)
        self.target = zeros(n, o, kind=torch.bool)
        self.blocker = zeros(n, o, kind=torch.bool)
        self.object_motion_cost = zeros(n)
        self.access_region = zeros(n, 4)
        self.workspace_bounds = zeros(n, 4)
        self.shelf_top = zeros(n)
        self.ceiling = zeros(n)
        self.initial_shelf_pose = zeros(n, 7)
        self.env_origins = zeros(n, 3)
        self.support_bounds = zeros(n, 4)
        self.support_height = zeros(n)
        self.support_vertices = None  # One immutable collision catalog, never cloned per environment.
        self.support_vertex_counts = None
        self.support_error = None
        self.initial_center = zeros(n, o, 3)
        self.initial_rotation = zeros(n, o, 4)
        self.initial_blocker_overlap = zeros(n)
        self.previous_wrench = zeros(n, 6)
        self.elapsed_time = zeros(n, kind=torch.float64)
        self.stable_count = zeros(n, kind=torch.long)
        self.tilt_count = zeros(n, o, kind=torch.long)
        self.support_loss_count = zeros(n, o, kind=torch.long)
        self.support_distance = zeros(n, o)
        self.stability_distance = zeros(n, o)
        self.tilt_angle = zeros(n, o)
        self.clearance_progress_valid = zeros(n, kind=torch.bool)
        self.target_protected = zeros(n, kind=torch.bool)
        self.completion_valid = zeros(n, kind=torch.bool)
        self.raw_goal = zeros(n, kind=torch.bool)
        self.raw_goal_seen = zeros(n, kind=torch.bool)
        self.unsupported_goal_seen = zeros(n, kind=torch.bool)
        self.goal_failure_seen = zeros(n, kind=torch.bool)
        # Cumulative fresh native episode counts, never last-episode snapshots.
        self.completed_totals = zeros(5, kind=torch.long)
        self.success = zeros(n, kind=torch.bool)
        self.failure = zeros(n, kind=torch.bool)
        self.success_paid = zeros(n, kind=torch.bool)
        self.failure_paid = zeros(n, kind=torch.bool)
        self.failure_reasons = zeros(n, len(FAILURE_REASON_NAMES), kind=torch.bool)
        self.failure_mask, self.success_enabled = configured_terminations(termination_cfg, device=device)
        self.invalid_reason_index = FAILURE_REASON_NAMES.index("invalid_state")
        # Named transition measurements; each native reward term owns its formula.
        self.clearance_quality = zeros(n)
        self.force_growth = zeros(n)
        self.torque_growth = zeros(n)
        self.terminal_reward = zeros(n)
        self.blocker_overlap = zeros(n)
        self.singularity_margin = zeros(n)

    @torch.no_grad()
    def reset(self, ids, batch, *, geometry_b, geometry_valid, active, target, blocker,
              access_region, workspace_bounds, shelf_top, ceiling, shelf_local_corners,
              step, batch_is_selected=False, critic_source_indices=None):
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
        self.initial_shelf_pose[ids] = batch.shelf_pose[rows]
        # Rules describe the measured panel in environment coordinates. Store
        # its rectangle in the shelf link frame, then follow the CURRENT shelf
        # pose at every transition. Whole-shelf bounds include posts/other tiers.
        bounds, height = workspace_bounds[rows], shelf_top[rows]
        panel = torch.stack((
            torch.stack((bounds[:, 0], bounds[:, 2], height), -1),
            torch.stack((bounds[:, 0], bounds[:, 3], height), -1),
            torch.stack((bounds[:, 1], bounds[:, 2], height), -1),
            torch.stack((bounds[:, 1], bounds[:, 3], height), -1)), -2)
        panel = quat_apply(quat_conjugate(batch.shelf_pose[rows, None, 3:]),
                           panel - (batch.shelf_pose[rows, :3] - self.env_origins[ids])[:, None])
        lo, hi = panel.amin(-2), panel.amax(-2)
        self.support_bounds[ids] = torch.stack((lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1]), -1)
        self.support_height[ids] = panel[..., 2].mean(-1)
        self.initial_center[ids] = batch.center[rows]
        self.initial_rotation[ids] = batch.rotation[rows]
        self.object_motion_cost[ids] = object_motion_cost(
            batch.center[rows], batch.corners[rows], batch.velocity[rows], self.active[ids], self.blocker[ids],
        )
        self.tcp_pose_b[ids] = batch.tcp_pose_b[rows]
        if critic_source_indices is None:
            # Numerical callers without an ID association never invent GT slots.
            self.critic_source_indices[ids] = -1
        else:
            self.critic_source_indices[ids] = critic_source_indices[rows]
        selected = self.critic_source_indices[ids].clamp_min(0)
        corners = batch.corners[rows].gather(1, selected[..., None, None].expand(-1, -1, 8, 3))
        # The rigid canonical OBB size is fixed during an episode. Corner order
        # is ---, --+, -+-, -++, +--, +-+, ++-, +++.
        self.critic_geometry_b[ids, :, 7:] = torch.stack((
            torch.linalg.vector_norm(corners[:, :, 4] - corners[:, :, 0], dim=-1),
            torch.linalg.vector_norm(corners[:, :, 2] - corners[:, :, 0], dim=-1),
            torch.linalg.vector_norm(corners[:, :, 1] - corners[:, :, 0], dim=-1),
        ), dim=-1)
        self._update_critic_geometry(batch.center[rows], batch.rotation[rows], ids)
        self.history[ids] = 0
        self.history[ids, 0] = batch.sample[rows]
        # The reset state has no completed action/contact sample in this episode.
        self.history[ids, 0, 31:37] = 0
        self.history[ids, 0, 37:54] = -1
        self.history[ids, 0, 54:72] = 0
        for name in ("previous_wrench", "elapsed_time", "stable_count", "tilt_count",
                     "support_loss_count", "support_distance", "stability_distance", "tilt_angle",
                     "clearance_progress_valid", "completion_valid", "raw_goal",
                     "raw_goal_seen", "unsupported_goal_seen", "goal_failure_seen",
                     "success", "failure", "success_paid", "failure_paid", "failure_reasons", "clearance_quality",
                     "force_growth", "torque_growth", "terminal_reward", "blocker_overlap"):
            getattr(self, name)[ids] = 0
        # One batched geometry query for reset rows only. The initial reference
        # stays fixed until those environments start another episode.
        _, overlap = projected_areas(batch.corners[rows], blocker[rows] & active[rows], access_region[rows])
        initial_overlap = overlap.sum(-1)
        self.initial_blocker_overlap[ids] = initial_overlap
        self.target_protected[ids] = True
        self.clearance_quality[ids] = clearance_quality(initial_overlap, initial_overlap)
        # No completed physics sample yet; reset rows start with zero warning cost.
        self.singularity_margin[ids] = self.cfg.motion_singularity_threshold
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

    def _update_critic_geometry(self, center, rotation, ids=None):
        """Gather the fixed recognized slots from the already-read physics batch."""
        rows = slice(None) if ids is None else ids
        source = self.critic_source_indices[rows]
        selected = source.clamp_min(0)
        position = center.gather(1, selected[..., None].expand(-1, -1, 3))
        orientation = rotation.gather(1, selected[..., None].expand(-1, -1, 4))
        if self.bindings is not None:
            # Root properties were acquired by _read_transition/reset_snapshot
            # at this physics timestamp; these public reads reuse native caches.
            data = self.bindings.robot.data
            inverse_root = quat_conjugate(data.root_quat_w[rows, None])
            root_position = data.root_pos_w[rows] - self.env_origins[rows]
            position = quat_apply(inverse_root, position - root_position[:, None])
            orientation = quat_mul(inverse_root, orientation)
        geometry = torch.cat((position, orientation, self.critic_geometry_b[rows, :, 7:]), dim=-1)
        valid = self.geometry_valid[rows] & (source >= 0)
        self.critic_geometry_b[rows] = torch.where(valid[..., None], torch.nan_to_num(
            geometry, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        self.critic_observation = None

    @torch.no_grad()
    def advance(self, batch, step, *, singularity=None, hand_velocity=None):
        if step == self.last_step:
            return
        if not self.ready:
            raise RuntimeError("reset_task must run after scene/vision initialization before stepping.")
        if step != self.last_step + 1:
            raise RuntimeError("MDP must be evaluated once every completed control transition.")
        if singularity is None:
            self.singularity_margin.fill_(self.cfg.motion_singularity_threshold)
        else:
            if singularity.shape != self.singularity_margin.shape:
                raise ValueError("Singularity margin must contain one value per environment.")
            self.singularity_margin.copy_(singularity)
        safe_corners = torch.nan_to_num(batch.corners, nan=0.0, posinf=0.0, neginf=0.0)
        # Clearance and its success gate concern only the designated Blocker.
        # Skip polygon clipping for other objects; their safety checks remain.
        _, overlap = projected_areas(safe_corners, self.blocker & self.active, self.access_region)
        blocked = overlap.sum(-1)
        self.blocker_overlap.copy_(blocked)
        self.clearance_quality.copy_(torch.nan_to_num(
            clearance_quality(blocked, self.initial_blocker_overlap),
            nan=0.0, posinf=0.0, neginf=0.0,
        ))
        # Force/torque norms are rotation-invariant. Keep the current sensor
        # center: adding r x F at the initial origin would charge pure travel
        # under a constant load as a torque increase.
        growth = (torch.linalg.vector_norm(batch.sensor_wrench.reshape(-1, 2, 3), dim=-1)
                  - torch.linalg.vector_norm(self.previous_wrench.reshape(-1, 2, 3), dim=-1)).clamp_min(0.0)
        growth = torch.nan_to_num(growth, nan=0.0, posinf=0.0, neginf=0.0)
        self.force_growth.copy_(growth[:, 0])
        self.torque_growth.copy_(growth[:, 1])
        self.object_motion_cost.copy_(object_motion_cost(
            batch.center, batch.corners, batch.velocity, self.active, self.blocker,
        ))
        support = self._support_validity(batch)
        candidate, stable, tilted, reasons, failed = task_conditions(
            self.cfg, center=batch.center, corners=batch.corners, rotation=batch.rotation,
            velocity=batch.velocity, active=self.active, target=self.target,
            initial_center=self.initial_center, initial_rotation=self.initial_rotation,
            blocker_overlap=self.blocker_overlap, sensor_wrench=batch.sensor_wrench,
            stable_count=self.stable_count, tilt_count=self.tilt_count, step_dt=self.dt,
            workspace_bounds=self.workspace_bounds,
            shelf_top=self.shelf_top, ceiling=self.ceiling,
            shelf_pose=batch.shelf_pose, initial_shelf_pose=self.initial_shelf_pose,
            state_valid=batch.state_valid,
            failure_mask=self.failure_mask,
            singularity=singularity,
            support_validity=support,
            raw_goal_out=self.raw_goal,
            target_protected_out=self.target_protected,
        )
        # The official velocity term is logged by the native manager. Include
        # its result here too, so success/payment never wins an overspeed step.
        if hand_velocity is not None:
            if hand_velocity.shape != failed.shape or hand_velocity.dtype != torch.bool:
                raise ValueError("Hand velocity failure must contain one boolean per environment.")
            failed = failed | hand_velocity
            candidate = candidate & ~hand_velocity
            stable = torch.where(hand_velocity, 0, stable)
        successful, payment, success_paid, failure_paid = terminal_transition(
            candidate, failed, stable, self.success_paid, self.failure_paid, self.cfg,
            step_dt=self.dt,
        )
        self.clearance_progress_valid.copy_(support.progress_valid & self.target_protected
                                           & ~failed & ~reasons[:, self.invalid_reason_index])
        self.completion_valid.copy_(support.completion_valid)
        self.raw_goal_seen.logical_or_(self.raw_goal)
        self.unsupported_goal_seen.logical_or_(self.raw_goal & ~support.progress_valid)
        self.goal_failure_seen.logical_or_(self.raw_goal & failed)
        self.support_loss_count.copy_(support.support_loss_count)
        self.support_distance.copy_(support.support_distance)
        self.stability_distance.copy_(support.stability_distance)
        self.tilt_angle.copy_(support.tilt_angle)
        self.terminal_reward.copy_(payment)
        self.stable_count.copy_(stable)
        self.tilt_count.copy_(tilted)
        self.success.copy_(successful)
        self.failure.copy_(failed)
        self.success_paid.copy_(success_paid)
        self.failure_paid.copy_(failure_paid)
        self.failure_reasons.copy_(reasons)
        self.history[:, 1].copy_(self.history[:, 0])
        self.history[:, 0].copy_(torch.nan_to_num(batch.sample, nan=0.0, posinf=0.0, neginf=0.0))
        self.tcp_pose_b.copy_(torch.nan_to_num(batch.tcp_pose_b, nan=0.0, posinf=0.0, neginf=0.0))
        self._update_critic_geometry(batch.center, batch.rotation)
        self.previous_wrench.copy_(batch.sensor_wrench)
        self.elapsed_time.add_(self.dt)
        self.observation = None
        self.last_step = step

    def _support_validity(self, batch):
        """Replay the fixed geometric task check without small-op CPU dispatch."""
        if not batch.center.is_cuda or torch.cuda.is_current_stream_capturing():
            return self._support_tensors(batch)
        key = (torch.is_inference_mode_enabled(), torch.backends.cuda.matmul.allow_tf32)
        cache = getattr(self, "_support_graphs", {})
        values = (batch.center, batch.corners, batch.rotation, batch.velocity, batch.shelf_pose)
        if key not in cache:
            with torch.inference_mode(False), torch.cuda.device(batch.center.device):
                static = tuple(value.clone() for value in values)
                sample = SimpleNamespace(**dict(zip(("center", "corners", "rotation", "velocity", "shelf_pose"), static)))
                stream = torch.cuda.Stream(device=batch.center.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._support_tensors(sample)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = self._support_tensors(sample)
            cache[key] = (static, graph, output)
            self._support_graphs = cache
        static, graph, output = cache[key]
        for destination, source in zip(static, values):
            destination.copy_(source)
        graph.replay()
        return output

    def _support_tensors(self, batch):
        """Current shelf-relative geometry; collision plane minima avoid OBB pivot artifacts."""
        inverse = quat_conjugate(batch.shelf_pose[:, None, 3:])
        shelf_origin = batch.shelf_pose[:, :3] - self.env_origins
        center = quat_apply(inverse, batch.center - shelf_origin[:, None])
        corners = quat_apply(inverse[:, :, None], batch.corners - shelf_origin[:, None, None])
        relative = quat_mul(inverse, batch.rotation)
        normal = relative.new_zeros((*relative.shape[:-1], 3))
        normal[..., 2] = 1
        normal = quat_apply(quat_conjugate(relative), normal)
        bottom = None
        if self.support_vertices is not None:
            # Dot cached canonical vertices with the shelf normal expressed in
            # object axes. No mesh transforms, USD calls or CPU queries per step.
            minimum = collision_support_minimum(
                self.support_vertices, self.support_vertex_counts, self.bindings.object_ids,
                normal, self.bindings.local_center)
            bottom = center[..., 2] + minimum - self.support_error[self.bindings.object_ids]
        return object_support_validity(
            self.cfg, center_shelf=center, corners_shelf=corners, velocity=batch.velocity,
            active=self.active, upright_z=normal[..., 2], support_bounds=self.support_bounds,
            support_height=self.support_height, support_bottom=bottom,
            support_loss_count=self.support_loss_count, step_dt=self.dt,
        )


@dataclass
class SceneBatch:
    """Reset producer contract, filled by independent episode/initial-vision events.

    Positions/bounds use fixed environment coordinates except geometry_b (robot
    base). Object IDs index scene['object_collection']; -1 means inactive or
    unobserved. Catalog bounds are measured in canonical upright coordinates.
    Per-episode values must stay valid until reset_task commits the selected rows.
    """

    object_ids: torch.Tensor             # N,A; all active physical objects, -1 padding
    target_id: torch.Tensor              # N
    blocker_id: torch.Tensor             # N
    geometry_b: torch.Tensor             # N,O,10; selected initial vision ONLY (O=10)
    geometry_valid: torch.Tensor         # N,O
    observed_ids: torch.Tensor           # N,O; association to collection IDs
    lower: torch.Tensor                  # collection_size,3
    upper: torch.Tensor                  # collection_size,3
    upright: torch.Tensor                # collection_size,4 (wxyz)
    access_region: torch.Tensor          # N,4: xmin,xmax,ymin,ymax
    workspace_bounds: torch.Tensor       # N,4
    shelf_top: torch.Tensor              # N
    ceiling: torch.Tensor                # N
    shelf_local_corners: torch.Tensor    # N,8,3; full physical shelf in its link frame
    sensor_body_name: str
    support_vertices: torch.Tensor | None = None  # catalog,V,3 canonical collision hulls
    support_vertex_mask: torch.Tensor | None = None  # catalog,V, bool
    support_error: torch.Tensor | None = None  # catalog; primitive approximation bound in meters


@dataclass
class SensorBatch:
    """Latest completed physics sample, supplied by env.task_sensor_reader().

    The reader binds the native articulation and contact sensors. Ideal
    wrench is parent-on-child joint reaction at the sensor center, sensor axes.
    """

    ideal_wrench: torch.Tensor           # N,6; before SI-200-8 measurement model
    wrench_valid: torch.Tensor           # N
    tactile_force: torch.Tensor          # N,17,3; latest net normal forces
    tactile_valid: torch.Tensor          # N,17


class RoleObbVisualizer:
    """Shared viewport drawing for OBBs and the optional sensor monitor.

    Physical role bounds follow current body-link poses. Observation bounds
    remain the initial visible-surface snapshot used by the policy. Both layers
    share one draw callback because DebugDraw's buffers are global. No USD
    prims, rigid bodies, collision shapes or markers are made.
    """

    has_debug_vis_implementation = True
    callback_name = "contact_manipulation_parallel_role_obbs"

    def __init__(self, env):
        import weakref

        self._env = weakref.proxy(env)
        self.cfg = env.cfg
        self.debug_vis = False
        self.observation_debug_vis = bool(env.cfg.observation_obb_debug_vis)
        self._draw = None
        self.set_debug_vis(env.cfg.role_obb_debug_vis)
        # Native env.close() deletes the manager before clearing callbacks.
        weakref.finalize(env.event_manager, self.close)

    def set_debug_vis(self, debug_vis):
        """Called directly by Isaac Lab's standard visualization checkbox."""
        self.debug_vis = self.cfg.role_obb_debug_vis = bool(debug_vis)
        self._refresh_debug_vis()
        return True

    def set_observation_debug_vis(self, debug_vis):
        self.observation_debug_vis = self.cfg.observation_obb_debug_vis = bool(debug_vis)
        self._refresh_debug_vis()
        return True

    def _refresh_debug_vis(self):
        import weakref

        env = self._env
        sensor_monitor = getattr(env, "_sensor_monitor", None)
        if not (self.debug_vis or self.observation_debug_vis
                or (sensor_monitor is not None and sensor_monitor.debug_vis)):
            if env.sim.render_callback_exists(self.callback_name):
                env.sim.remove_render_callback(self.callback_name)
            if self._draw is not None:
                self._draw.clear_lines()
                self._draw.clear_points()
            return
        if self._draw is None:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._rows = torch.arange(env.num_envs, device=env.device)[:, None]
            self._bits = torch.tensor([(i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(8)], device=env.device)
            self._edges = torch.tensor([(i, i ^ bit) for i in range(8) for bit in (1, 2, 4)
                                        if i < (i ^ bit)], device=env.device)
            self._colors = [color for _ in range(env.num_envs)
                            for color in ((0., 1., 0., 1.), (1., 0., 0., 1.)) for _ in range(12)]
            self._widths = [3.] * (24*env.num_envs)
        if not env.sim.render_callback_exists(self.callback_name):
            env.sim.add_render_callback(self.callback_name,
                lambda event, obj=weakref.proxy(self): obj.update(event))
        self.update()

    def _observation_corners(self, state):
        """Render exactly the valid policy slots, never privileged body bounds."""
        valid = state.geometry_valid
        geometry = state.geometry_b[valid]
        # OBB fields are center, wxyz quaternion and full side lengths in base
        # coordinates. Re-expressing this fixed snapshot in world coordinates
        # is equivalent to undoing the policy's current EEF-frame transform.
        local = (self._bits - .5) * geometry[:, None, 7:10]
        corners_b = geometry[:, None, :3] + quat_apply(geometry[:, None, 3:7], local)
        robot = self._env.scene["robot"]
        capacity = valid.shape[1]
        position = robot.data.root_pos_w[:, None].expand(-1, capacity, -1)[valid]
        orientation = robot.data.root_quat_w[:, None].expand(-1, capacity, -1)[valid]
        return position[:, None] + quat_apply(orientation[:, None], corners_b)

    @torch.no_grad()
    def update(self, _event=None):
        env = self._env
        # Native live visualizers also skip STOP/PAUSE: PhysX views may have
        # been invalidated by the timeline even while the window stays open.
        if not env.sim.is_playing():
            return
        scene, state = getattr(env, "task_scene_data", None), getattr(env, "_contact_mdp", None)
        sensor_monitor = getattr(env, "_sensor_monitor", None)
        sensors_enabled = sensor_monitor is not None and sensor_monitor.debug_vis
        # Startup rendering can precede the first committed scene/MDP reset.
        if (not (self.debug_vis or self.observation_debug_vis or sensors_enabled) or scene is None
                or state is None or state.bindings is None or env._is_closed):
            return
        line_batches, colors, widths = [], [], []
        points = None
        if self.observation_debug_vis:
            corners = self._observation_corners(state)
            lines = corners[:, self._edges].reshape(-1, 2, 3)
            line_batches.append(lines)
            colors.extend([(0., .8, 1., 1.)] * lines.shape[0])
            widths.extend([1.5] * lines.shape[0])
        if self.debug_vis:
            roles = torch.stack((scene.target_id, scene.blocker_id), -1)
            pose = env.scene["object_collection"].data.object_link_pose_w[self._rows, roles]
            lower, upper = scene.lower[roles], scene.upper[roles]
            local = lower[..., None, :] + (upper-lower)[..., None, :]*self._bits
            _, corners, _ = object_geometry(pose, .5*(lower+upper), local, scene.upright[roles])
            line_batches.append(corners[:, :, self._edges].reshape(-1, 2, 3))
            colors.extend(self._colors)
            widths.extend(self._widths)
        if sensors_enabled:
            primitives = sensor_monitor.render_geometry()
            if primitives is not None:
                sensor_lines, sensor_colors, sensor_widths, points, point_colors, point_sizes = primitives
                line_batches.append(sensor_lines)
                colors.extend(sensor_colors)
                widths.extend(sensor_widths)
        # Transfers belong only to GUI drawing, never the physics-step loop.
        lines = (torch.cat(line_batches).transpose(0, 1).contiguous().cpu().tolist()
                 if line_batches else [[], []])
        self._draw.clear_lines()
        self._draw.draw_lines(lines[0], lines[1], colors, widths)
        self._draw.clear_points()
        if points is not None:
            self._draw.draw_points(points.cpu().tolist(), point_colors, point_sizes)

    def close(self):
        self.debug_vis = self.observation_debug_vis = False
        sensor_monitor = getattr(self._env, "_sensor_monitor", None)
        if sensor_monitor is not None:
            sensor_monitor.close()
        self._refresh_debug_vis()


class ObservationObbVisualizer:
    """Independent native checkbox backed by the shared viewport renderer."""

    has_debug_vis_implementation = True

    def __init__(self, visualizer):
        self._visualizer = visualizer
        self.cfg = visualizer.cfg

    @property
    def debug_vis(self):
        return self._visualizer.observation_debug_vis

    def set_debug_vis(self, debug_vis):
        return self._visualizer.set_observation_debug_vis(debug_vis)


def initialize_role_obb_visualization(env, env_ids=None):
    """Prepare GUI toggles; headless training creates no visualizer resources."""
    if env.sim.has_gui():
        if not hasattr(env, "_role_obb_visualizer"):
            env._role_obb_visualizer = RoleObbVisualizer(env)
            env._observation_obb_visualizer = ObservationObbVisualizer(env._role_obb_visualizer)
        if not hasattr(env, "_sensor_monitor"):
            from .sensors import SensorMonitor

            env._sensor_monitor = SensorMonitor(env)
            env._role_obb_visualizer._refresh_debug_vis()


def _ensure_state(env):
    if not hasattr(env, "_contact_mdp"):
        # ObservationManager probes output dimensions before startup/reset.
        # No scene ground truth is substituted for missing initial vision.
        env._contact_mdp = TaskState(env.num_envs, env.device, env.cfg.mdp, float(env.step_dt),
                                     step=env.common_step_counter,
                                     termination_cfg=getattr(env.cfg, "terminations", None),
                                     active_capacity=_active_object_capacity(env))
    return env._contact_mdp


def _active_object_capacity(env):
    """Physical slots include unobserved objects used by rewards and failures."""
    rules = getattr(env, 'spawn_rules', getattr(env.cfg, 'spawn_rules', None))
    return rules.max_objects if rules is not None else env.cfg.mdp.max_objects


def _bind_scene(env, state, scene):
    """Resolve native asset IDs and constant device tensors only once."""
    from types import SimpleNamespace

    robot = env.scene["robot"]
    # Resolve asset IDs through the public API; stock action terms do not expose
    # the task-specific attributes previously supplied by our subclasses.
    arm_ids, _ = robot.find_joints(env.cfg.actions.arm_action.joint_names, preserve_order=True)
    hand_ids, _ = robot.find_joints(env.cfg.actions.hand_action.joint_names, preserve_order=True)
    tcp_ids, _ = robot.find_bodies(env.cfg.actions.arm_action.body_name)
    if len(arm_ids) != 6 or len(hand_ids) != 6 or len(tcp_ids) != 1:
        raise ValueError("Resolve six arm joints, six hand joints, and one TCP body.")
    names = robot.body_names
    joint_ids = torch.tensor(arm_ids + hand_ids, device=env.device)
    n, o = env.num_envs, state.active_capacity
    state.env_origins.copy_(env.scene.env_origins)
    if scene.support_vertices is not None:
        state.support_vertices = scene.support_vertices
        state.support_vertex_counts = scene.support_vertex_mask.sum(-1)
        state.support_error = scene.support_error
    # Object COM offsets are fixed in this task. Native COM properties
    # otherwise fetch CPU buffers again at every physics timestamp/reset.
    object_com_local = env.scene["object_collection"].data.object_com_pos_b.clone()
    state.bindings = SimpleNamespace(
        robot=robot, objects=env.scene["object_collection"], shelf=env.scene["shelf"],
        joint_ids=joint_ids, tcp_id=tcp_ids[0],
        sensor_id=names.index(scene.sensor_body_name), object_com_local=object_com_local,
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
    physical = _active_object_capacity(env)
    device = env._contact_mdp.history.device
    for name, shape, dtype in (
        ("object_ids", (n, physical), torch.long), ("target_id", (n,), torch.long),
        ("blocker_id", (n,), torch.long), ("observed_ids", (n, o), torch.long),
        ("geometry_valid", (n, o), torch.bool), ("geometry_b", (n, o, 10), None),
        ("access_region", (n, 4), None), ("workspace_bounds", (n, 4), None),
        ("shelf_top", (n,), None), ("ceiling", (n,), None),
        ("shelf_local_corners", (n, 8, 3), None),
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
    shelf_corners = scene.shelf_local_corners[ids]
    if not bool(torch.isfinite(shelf_corners).all()
                & (shelf_corners.amax(-2) > shelf_corners.amin(-2)).all()):
        raise ValueError("Shelf local OBB corners must be finite with positive extents.")
    if scene.support_vertices is not None:
        vertices, mask, error = scene.support_vertices, scene.support_vertex_mask, scene.support_error
        if (vertices.ndim != 3 or vertices.shape[0] != scene.lower.shape[0] or vertices.shape[-1] != 3
                or vertices.device != device or mask is None or mask.shape != vertices.shape[:-1]
                or mask.dtype != torch.bool or mask.device != device or error is None
                or error.shape != scene.lower.shape[:1] or error.device != device):
            raise ValueError("Collision support catalog requires matching vertices, masks, and error bounds.")
        if not bool((torch.isfinite(vertices).all(-1) | ~mask).all() & mask.any(-1).all()
                    & torch.isfinite(error).all() & (error >= 0).all()):
            raise ValueError("Collision support vertices/errors must be finite with a nonempty hull per asset.")
        prefix = torch.arange(vertices.shape[1], device=device)[None] < mask.sum(-1)[:, None]
        if not torch.equal(prefix, mask):
            raise ValueError("Collision support catalog must pack valid vertices before padding.")
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
                               ("tactile_force", (n, 17, 3), None), ("tactile_valid", (n, 17), torch.bool)):
        value = getattr(sensors, name)
        if value.shape != shape or value.device != device or (dtype is not None and value.dtype != dtype):
            raise ValueError(f"SensorBatch.{name} requires shape {shape}, dtype {dtype}, device {device}.")


@torch.no_grad()
def _transition_tensors(env, state, rows, values):
    """Pure tensor transform after native physics/sensor acquisition."""
    b = state.bindings
    (object_pose, object_velocity, body_position, body_rotation,
     root_position, root_rotation, joint_position,
     joint_velocity, ideal_wrench, wrench_valid_all, tactile_force_all, tactile_valid_all,
     shelf_pose, actions, joint_effort) = values
    ids = b.object_ids[rows]
    origins = env.scene.env_origins[rows, None, :]
    pose = object_pose[rows].gather(1, ids[..., None].expand(-1, -1, 7))
    pose[..., :3] -= origins
    center, corners, rotation = object_geometry(pose, b.local_center[rows], b.local_corners[rows], b.upright[rows])
    local_com = b.object_com_local[rows].gather(1, ids[..., None].expand(-1, -1, 3))
    com = pose[..., :3] + quat_apply(pose[..., 3:7], local_com)
    velocity = object_velocity[rows].gather(1, ids[..., None].expand(-1, -1, 6))
    velocity[..., :3] += torch.cross(velocity[..., 3:], center - com, dim=-1)

    body_position, body_rotation = body_position[rows], body_rotation[rows]
    sensor_position = body_position[:, b.sensor_id]
    sensor_rotation = body_rotation[:, b.sensor_id]
    raw_wrench, wrench_valid = ideal_wrench[rows], wrench_valid_all[rows]
    tactile_force = tactile_force_all[rows]
    measured, overrange = calibrate_wrench(raw_wrench, b.ranges, b.resolution)
    # Use the measured reaction directly; hand weight and inertia remain in
    # both observation and reward. Invalid/reset sensor samples stay zero.
    wrench = torch.where(wrench_valid[:, None], measured, 0.0)
    inverse_root = quat_conjugate(root_rotation[rows])
    position = quat_apply(inverse_root, body_position[:, b.tcp_id] - root_position[rows])
    orientation = quat_mul(inverse_root, body_rotation[:, b.tcp_id])
    orientation = torch.where(orientation[:, :1] < 0, -orientation, orientation)
    tcp_pose = torch.cat((position, orientation), -1)
    joint_pos = joint_position[rows].index_select(1, b.joint_ids)
    joint_vel = joint_velocity[rows].index_select(1, b.joint_ids)
    tactile_valid = tactile_valid_all[rows] & torch.isfinite(tactile_force).all(-1)
    sample = dynamic_sample(
        joint_pos, joint_vel, tcp_pose, wrench,
        torch.linalg.vector_norm(tactile_force, dim=-1) > state.cfg.tactile_threshold,
        tactile_valid, torch.nan_to_num(actions[rows], nan=0.0, posinf=1.0, neginf=-1.0), joint_position_scale=state.cfg.joint_position_scale,
        joint_velocity_scale=state.cfg.joint_velocity_scale, wrench_ranges=b.ranges,
    )
    valid = (torch.isfinite(sample).all(-1) & torch.isfinite(raw_wrench).all(-1)
             & wrench_valid & torch.isfinite(actions[rows]).all(-1)
             & torch.isfinite(joint_effort[rows]).all(-1))
    return TransitionBatch(center, corners, rotation, velocity, wrench, sensor_position,
                           sensor_rotation, overrange & wrench_valid[:, None],
                           shelf_pose[rows], valid, sample, tcp_pose)


@torch.no_grad()
def _read_transition(env, state, env_ids=None):
    """Acquire native data once, then replay the complete rollout tensor transform."""
    b = state.bindings
    data, objects, sensors = b.robot.data, b.objects.data, env.task_sensor_reader()
    values = (objects.object_link_pose_w, objects.object_com_vel_w,
              data.body_link_pos_w, data.body_link_quat_w,
              data.root_pos_w, data.root_quat_w, data.joint_pos, data.joint_vel,
              sensors.ideal_wrench, sensors.wrench_valid, sensors.tactile_force, sensors.tactile_valid,
              b.shelf.data.root_link_pose_w, env.action_manager.action, data.applied_torque)
    rows = slice(None) if env_ids is None else env_ids
    if env_ids is not None or not values[0].is_cuda:
        return _transition_tensors(env, state, rows, values)
    key = (torch.is_inference_mode_enabled(), torch.backends.cuda.matmul.allow_tf32)
    cache = getattr(state, "_transition_graphs", {})
    if key not in cache:
        # Graph-owned buffers and Philox bookkeeping must be normal tensors:
        # native training alternates inference rollouts and non-inference work.
        with torch.inference_mode(False), torch.cuda.device(values[0].device):
            static = tuple(t.clone() for t in values)
            stream = torch.cuda.Stream(device=values[0].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    _transition_tensors(env, state, rows, static)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _transition_tensors(env, state, rows, static)
        cache[key] = (static, graph, output)
        state._transition_graphs = cache
    static, graph, output = cache[key]
    for destination, source in zip(static, values):
        destination.copy_(source)
    graph.replay()
    return TransitionBatch(**{name: value.clone() for name, value in vars(output).items()})


def _read_reset_snapshot(env, state, ids):
    """Read only measurements consumed at reset; no force/COM/dynamics fetch."""
    b = state.bindings
    data = b.robot.data
    selected = b.object_ids[ids]
    pose = b.objects.data.object_link_pose_w[ids].gather(1, selected[..., None].expand(-1, -1, 7))
    pose[..., :3] -= env.scene.env_origins[ids, None]
    center, corners, rotation = object_geometry(
        pose, b.local_center[ids], b.local_corners[ids], b.upright[ids])
    position, orientation = data.body_link_pos_w[ids], data.body_link_quat_w[ids]
    inverse_root = quat_conjugate(data.root_quat_w[ids])
    tcp_position = quat_apply(inverse_root, position[:, b.tcp_id] - data.root_pos_w[ids])
    tcp_rotation = quat_mul(inverse_root, orientation[:, b.tcp_id])
    tcp_rotation = torch.where(tcp_rotation[:, :1] < 0, -tcp_rotation, tcp_rotation)
    tcp_pose = torch.cat((tcp_position, tcp_rotation), -1)
    joints = data.joint_pos[ids].index_select(1, b.joint_ids) / state.cfg.joint_position_scale
    velocity = data.joint_vel[ids].index_select(1, b.joint_ids) / state.cfg.joint_velocity_scale
    sample = torch.cat((joints, velocity, tcp_pose, joints.new_zeros((ids.numel(), 41))), -1)
    # reset_scene writes zero object velocities before this snapshot. Reuse that
    # known reset state instead of requesting an extra PhysX velocity buffer.
    return SimpleNamespace(center=center, corners=corners, rotation=rotation,
                           velocity=center.new_zeros((*center.shape[:-1], 6)),
                           shelf_pose=b.shelf.data.root_link_pose_w[ids], sensor_position=position[:, b.sensor_id],
                           tcp_pose_b=tcp_pose, sample=sample)


def _prepare_reset_geometry(scene, ids):
    """Validate and prepare immutable scene fields, separately from robot state."""
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
    lower, upper = scene.lower[selected], scene.upper[selected]
    center = .5 * (lower + upper)
    signs = lower.new_tensor(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                             (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)))
    corners = center[..., None, :] + .5 * (upper - lower)[..., None, :] * signs
    observed_ids, valid = scene.observed_ids[ids], scene.geometry_valid[ids]
    observed_target = (observed_ids == scene.target_id[ids, None]) & valid
    observed_blocker = (observed_ids == scene.blocker_id[ids, None]) & valid
    roles = torch.stack((observed_target.long().argmax(-1), observed_blocker.long().argmax(-1)), -1)
    roles = torch.where(torch.stack((observed_target.any(-1), observed_blocker.any(-1)), -1), roles, -1)
    geometry, valid, observed_indices = prepare_initial_geometry(
        scene.geometry_b[ids], valid, roles[:, 0], roles[:, 1], return_indices=True,
        preserve_distractor_order=True)
    observed = observed_ids.gather(1, observed_indices.clamp_min(0))
    matches = ((observed[:, :, None] == object_ids[:, None]) & active[:, None] & valid[:, :, None])
    if bool((valid & ~matches.any(-1)).any()):
        raise ValueError("Recognized objects must refer to active physical objects.")
    critic_sources = torch.where(valid, matches.long().argmax(-1), -1)
    return SimpleNamespace(object_ids=selected, local_center=center, local_corners=corners,
                           upright=scene.upright[selected], active=active, target=target, blocker=blocker,
                           geometry=geometry, valid=valid, critic_source_indices=critic_sources)


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
    runtime = getattr(env, "_episode_reset", None)
    # Only the native producer can authorize reuse, after a successful commit.
    # Consume this marker: direct/external SceneBatch resets keep full validation.
    shared = (runtime is not None and getattr(runtime, "mdp_reset_scene", None) is scene
              and getattr(runtime, "mdp_reset_ids", None) is env_ids)
    if runtime is not None:
        runtime.mdp_reset_scene = None
    key = ((id(scene), id(runtime.shared_template), id(scene.lower), id(scene.upper), id(scene.upright))
           if shared else None)
    cached = getattr(runtime, "mdp_geometry", None) if shared else None
    if cached is None or cached[0] != key:
        # Native shared metadata is an immutable singleton broadcast by reset_scene.
        rows = ids[:1] if shared else ids
        _validate_reset_data(env, scene, rows)
        prepared = _prepare_reset_geometry(scene, rows)
        if shared:
            runtime.mdp_geometry = (key, prepared)
    else:
        prepared = cached[1]
    if state.bindings is None:
        _bind_scene(env, state, scene)
    b = state.bindings
    for name in ("object_ids", "local_center", "local_corners", "upright"):
        getattr(b, name)[ids] = getattr(prepared, name)
    initial = _read_reset_snapshot(env, state, ids)
    if not bool(torch.isfinite(initial.shelf_pose).all()
                & (initial.shelf_pose[:, 3:].norm(dim=-1) > 1.e-8).all()):
        raise ValueError("Reset shelf pose must be finite with a valid quaternion.")
    if not bool(torch.isfinite(initial.sample[:, :31]).all()
                & (initial.tcp_pose_b[:, 3:7].norm(dim=-1) > 1.e-8).all()):
        raise ValueError("Reset robot joint/EEF measurements must be finite with a valid quaternion.")
    expand = lambda value: value.expand(ids.numel(), *value.shape[1:])
    manager = getattr(env, "termination_manager", None)
    if state.ready and runtime is not None and manager is not None:
        # Native termination was evaluated before reset events. Count these
        # selected completed episodes before their state is cleared. Export GPU
        # snapshots via the existing pending-metrics/native logger path.
        ended = manager.dones[ids] & (state.elapsed_time[ids] > 0)
        state.completed_totals.add_(torch.stack((
            ended.sum(), (ended & state.success[ids]).sum(),
            (ended & state.raw_goal_seen[ids]).sum(),
            (ended & state.unsupported_goal_seen[ids]).sum(),
            (ended & state.goal_failure_seen[ids]).sum())))
        names = ("completed", "valid_success", "raw_goal_seen", "unsupported_goal_seen", "goal_and_failure")
        metrics = dict(getattr(runtime, "pending_reset_metrics", None) or {})
        metrics.update({f"Task_Count/{name}": value.clone() for name, value in zip(names, state.completed_totals)})
        metrics["Task_Rate/valid_success"] = state.completed_totals[1] / state.completed_totals[0].clamp_min(1)
        runtime.pending_reset_metrics = metrics
    # Preserve the ray image's sorted snapshot. Apply one per-episode neighboring
    # permutation to BOTH frozen actor geometry and the critic's tracked IDs.
    # Shared layout caches remain immutable; partial resets touch selected rows.
    geometry, valid = expand(prepared.geometry), expand(prepared.valid)
    order = adjacent_geometry_permutation(
        valid, getattr(env.cfg, "observation_adjacent_swap_probability", 0.0))
    geometry = geometry.gather(1, order[..., None].expand(-1, -1, 10))
    valid = valid.gather(1, order)
    critic_sources = expand(prepared.critic_source_indices).gather(1, order)
    state.reset(ids, initial, geometry_b=geometry, geometry_valid=valid,
                critic_source_indices=critic_sources,
                active=expand(prepared.active), target=expand(prepared.target), blocker=expand(prepared.blocker),
                access_region=scene.access_region[ids],
                workspace_bounds=scene.workspace_bounds[ids], shelf_top=scene.shelf_top[ids],
                ceiling=scene.ceiling[ids], shelf_local_corners=scene.shelf_local_corners[ids],
                step=env.common_step_counter, batch_is_selected=True)
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
        # The public native property reads the CURRENT post-physics wrist/arm
        # Jacobian. OSC's private cached buffer belongs to the preceding step.
        # Shared once here for reward and termination; no matrix is read on CPU.
        arm = env.action_manager.get_term("arm_action")
        margin = singularity_margin(arm.jacobian_w, characteristic_length=state.cfg.singularity_characteristic_length)
        # Read the manager's resolved SceneEntityCfg, not env.cfg's unresolved
        # copy. The official function reuses the native joint-velocity buffer.
        velocity_failure = None
        manager = getattr(env, "termination_manager", None)
        if manager is not None and "hand_velocity" in manager.active_terms:
            term = manager.get_term_cfg("hand_velocity")
            if not term.time_out:
                velocity_failure = term.func(env, **term.params)
        state.advance(_read_transition(env, state), env.common_step_counter,
                      singularity=margin, hand_velocity=velocity_failure)
        # Native env resets after reward/done. Preserve the final observation
        # here for the learner's timeout bootstrap, before any reset.
        geometry = geometry_in_eef_frame(state.geometry_b, state.geometry_valid, state.tcp_pose_b)
        state.observation = pack_policy_observation(geometry, state.history)
        env.extras["terminal_observation"] = state.observation
        env.extras["terminated"] = ((state.success & state.success_enabled) | state.failure).clone()
        env.extras["terminal_env_ids"] = state.empty_env_ids
    # Native _reset_idx clears extras['log'] after calling reset events. Merge
    # their small diagnostics here, during the subsequent observation read.
    runtime = getattr(env, "_episode_reset", None)
    metrics = getattr(runtime, "pending_reset_metrics", None)
    if metrics is not None:
        env.extras.setdefault("log", {}).update(metrics)
        runtime.pending_reset_metrics = None
    return state


# Shared layout preparation: CPU metadata and bounded device batches for geometry/IK.

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
    support_vertices: tuple[tuple[float, float, float], ...] = field(default=(), compare=False, repr=False)
    support_error: float = field(default=0.0, compare=False)
    collision_footprint_vertices: tuple[Point, ...] = field(default=(), compare=False, repr=False)
    visual_footprint_vertices: tuple[Point, ...] = field(default=(), compare=False, repr=False)


@dataclass
class SpawnRules:
    x_range: tuple[float, float] = (-0.89, -0.51)
    y_range: tuple[float, float] = (-0.455, 0.455)
    shelf_top: float = 1.05
    ceiling: float = 1.35
    min_objects: int = _DEFAULT_MAX_OBJECTS
    max_objects: int = _DEFAULT_MAX_OBJECTS
    min_asset_types: int = 4
    max_asset_types: int = 10
    all_asset_types: bool = False
    generation_mode: str = "constructive"
    parameter_cache_directory: str = ""
    edge_margin: float = 0.015
    gap: float = 0.003
    max_neighbor_gap: float = 0.010
    hand_width: float = 0.10
    approach_margin: float = 0.01
    min_blocked_ratio: float = 0.30
    min_x_coverage: float = 0.55
    min_y_coverage: float = 0.65
    neighbor_distance: float = 0.04
    min_neighbor_fraction: float = 0.75
    min_shelf_occupancy: float = 0.0
    region_grid: tuple[int, int] = (2, 3)
    min_region_occupancy: float = 0.0
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
    # Continuous reset budget after an anchor exists. Initial template
    # preparation covers fixed position/orientation strata and IK branches.
    max_attempts: int = 80
    ik_batch_size: int = 1024
    orientation_range: tuple[float, float, float] = (math.radians(10.0),) * 3
    reset_attempts: int = 3
    recovery_attempts: int = 2
    recovery_position_half_width: tuple[float, float, float] = (0.02, 0.02, 0.015)
    recovery_rotation_scale: float = 0.25
    # Legacy sampler admission only; constructive resets validate an actual
    # anchor rather than interpreting random single-seed IK results as safety.
    domain_probe_count: int = 8
    domain_probe_min_success: int = 4
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


class _PoseMapping(Mapping):
    """Download non-role poses only for explicit debug/validation access."""

    def __init__(self, roles, ids, poses, heights, *, yaw=False):
        self._values = dict(roles)
        self.ids, self.poses, self.support_heights, self.yaw = ids, poses, heights, yaw

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, key):
        if key not in self.ids:
            raise KeyError(key)
        if key not in self._values:
            host = self.poses.detach().cpu().numpy()
            for slot, index in enumerate(self.ids):
                self._values[index] = (float(host[slot, 2]) if self.yaw else
                                      (float(host[slot, 0]), float(host[slot, 1]), self.support_heights[index]))
        return self._values[key]


@dataclass(frozen=True)
class SceneSample:
    active_ids: tuple[int, ...]
    target_id: int
    blocker_id: int
    positions: Mapping[int, tuple[float, float, float]]
    yaws: Mapping[int, float]
    access_region: Region
    clearance_witness: Point
    clearance_path: tuple[PlanarMotion, ...] = ()
    clearance_kind: str = "translation"
    # Keep the packer's coordinates on device for IK and physics. CPU mappings
    # remain available for independent validation/debugging and role metadata.
    planar_poses: torch.Tensor | None = field(default=None, repr=False, compare=False)
    planar_ids: tuple[int, ...] = field(default=(), repr=False, compare=False)
    neighbor_pairs: tuple[tuple[int, int], ...] = ()
    reserved_regions: tuple[Region, ...] = ()
    camera_pose: tuple[float, ...] | None = None
    # A fixed finite construction domain, not new random scene proposals. The
    # sensor evaluates every layout/view pair once before choosing parameters.
    visibility_variants: tuple["SceneSample", ...] = field(default=(), repr=False, compare=False)
    # Offline parameter-domain provenance; this is not a live visibility
    # measurement and must never be inserted into the policy observation.
    parameter_certificate: Mapping | None = field(default=None, repr=False, compare=False)


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


def _validate_robot_start_rules(spawn_rules, robot_start):
    for name in ("orientation_range", "recovery_position_half_width"):
        values = getattr(robot_start, name)
        if len(values) != 3 or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(f"Robot start {name} must contain three finite nonnegative values.")
    for name in ("reset_attempts", "recovery_attempts", "domain_probe_count", "domain_probe_min_success"):
        value = getattr(robot_start, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"Robot start {name} must be a positive integer.")
    if robot_start.domain_probe_min_success > robot_start.domain_probe_count:
        raise ValueError("Robot start domain probe successes cannot exceed its candidate count.")
    if not math.isfinite(robot_start.recovery_rotation_scale) or not 0 < robot_start.recovery_rotation_scale <= 1:
        raise ValueError("Robot start recovery_rotation_scale must be in (0, 1].")
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
    _validate_robot_start_rules(spawn_rules, robot_start)
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


@njit(cache=True)
def _motion_envelopes(hx, hy, start, end, pivot, has_pivot):
    """Generate a selected motion's complete sweep without Python corner objects."""
    delta = end[2] - start[2]
    divisions = max(1, math.ceil(abs(delta) / math.radians(5.0))) if abs(delta) > _EPS else 1
    footprints = np.empty((divisions + 1, 4, 2))
    signs = ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))
    for step in range(divisions + 1):
        fraction = step / divisions
        x, y, yaw = start
        if step:
            yaw += fraction * delta
            if has_pivot and abs(delta) > _EPS:
                c, s = math.cos(fraction * delta), math.sin(fraction * delta)
                dx, dy = start[0] - pivot[0], start[1] - pivot[1]
                x, y = pivot[0] + c*dx - s*dy, pivot[1] + s*dx + c*dy
            else:
                x += fraction * (end[0] - start[0])
                y += fraction * (end[1] - start[1])
        # Translation follows the scalar endpoint semantics, including tiny yaw.
        if abs(delta) <= _EPS and step:
            x, y, yaw = end
        c, s = math.cos(yaw), math.sin(yaw)
        for k in range(4):
            dx, dy = hx * signs[k][0], hy * signs[k][1]
            footprints[step, k, 0] = x + c*dx - s*dy
            footprints[step, k, 1] = y + s*dx + c*dy
    if abs(delta) <= _EPS:
        hull = _hull_vertices(footprints.reshape(8, 2))
        vertices = np.zeros((1, 32, 2))
        vertices[0, :len(hull)] = hull
        return vertices, np.array([len(hull)], dtype=np.int64)
    ox, oy = (pivot[0], pivot[1]) if has_pivot else (start[0], start[1])
    radius = 0.
    for k in range(4):
        radius = max(radius, math.hypot(footprints[0, k, 0] - ox, footprints[0, k, 1] - oy))
    return _rotation_envelopes(footprints, radius * (1. - math.cos(abs(delta) / (2 * divisions))))


@dataclass
class _SweepPolygons:
    vertices: np.ndarray           # interval, padded vertex, XY; host preparation only
    counts: np.ndarray             # interval; no per-polygon tuple conversion

    def __len__(self):
        return len(self.counts)


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


def _compact_constructive_positions(polygons, centers, lower_y, gaps, fixed_polygons=()):
    """Compact one already-safe shared layout downward using analytic gap limits.

    ``polygons`` contain the already-rotated local collision footprints and
    ``centers`` their placement origins. Fixed polygons are in world XY and can
    reserve the complete Target/Blocker task module. Every object is visited
    once in stable initial-Y order. Its final Y is the highest relevant closed-
    form collision boundary below its initial position, or the support floor.
    There is no proposal acceptance loop, distance bisection, or random redraw.

    Bodies not yet processed remain obstacles at their initial poses. A center
    order alone cannot order the vertical cross-sections of rotated footprints.
    Previously processed bodies are obstacles at their final poses instead.
    The caller must supply an initially supported, nonintersecting arrangement.
    The returned pairs record the body that determined each placement at that
    moment; later moves can change that pair's final distance. Static reserves
    never appear as a body pair.
    """
    local = tuple(np.asarray(polygon, dtype=np.float64) for polygon in polygons)
    initial = np.asarray(centers, dtype=np.float64)
    count = len(local)
    if initial.shape != (count, 2) or not np.isfinite(initial).all() or not math.isfinite(lower_y):
        raise ValueError("Constructive compaction needs finite XY centers and a finite support floor.")
    distances = np.asarray(gaps, dtype=np.float64)
    if distances.shape != (count,) or not np.isfinite(distances).all() or (distances < 0.0).any():
        raise ValueError("Constructive compaction needs one finite nonnegative gap per body.")
    for polygon in local:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError("Constructive compaction needs finite convex XY footprints.")
    fixed = tuple(np.asarray(polygon, dtype=np.float64) for polygon in fixed_polygons)
    if any(polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
           or not np.isfinite(polygon).all() for polygon in fixed):
        raise ValueError("Reserved geometry must contain finite convex XY polygons.")

    result = initial.copy()
    neighbors = []
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(lower_y), float(np.abs(initial).max()) if count else 1.0)
    for index in np.argsort(initial[:, 1], kind="stable"):
        polygon, x = local[index], float(initial[index, 0])
        initial_y, gap = float(initial[index, 1]), float(distances[index])
        floor_y = float(lower_y - polygon[:, 1].min())
        if floor_y > initial_y + tolerance:
            raise ValueError("An initial constructive body lies below the support floor.")
        best_y, parent = floor_y, None
        for obstacle in fixed:
            limit = footprint_line_limit(polygon, obstacle, x, gap)
            if limit is not None and best_y < limit <= initial_y + tolerance:
                best_y, parent = limit, None
        for other in range(count):
            if other == index:
                continue
            # Unprocessed rows still equal their original safe arrangement;
            # processed rows contain the already accepted analytic placement.
            obstacle = local[other] + result[other]
            limit = footprint_line_limit(polygon, obstacle, x, gap)
            if limit is not None and best_y - tolerance <= limit <= initial_y + tolerance:
                best_y, parent = max(best_y, limit), int(other)
        result[index, 1] = min(best_y, initial_y)
        if parent is not None:
            neighbors.append((parent, int(index)))
    return result, tuple(neighbors)


@njit(cache=True)
def _convex_polygon_distance(first, second):
    """Shared-scene CPU geometry kernel; arrays avoid vertex-count JIT variants."""
    separated = False
    for which in range(2):
        polygon = first if which == 0 else second
        for edge in range(len(polygon)):
            following = (edge + 1) % len(polygon)
            dx = polygon[following, 0] - polygon[edge, 0]
            dy = polygon[following, 1] - polygon[edge, 1]
            length = math.sqrt(dx * dx + dy * dy)
            if length <= _EPS:
                continue
            nx, ny = -dy / length, dx / length
            minimum_a, maximum_a, minimum_b, maximum_b = math.inf, -math.inf, math.inf, -math.inf
            for point in first:
                projection = point[0] * nx + point[1] * ny
                minimum_a, maximum_a = min(minimum_a, projection), max(maximum_a, projection)
            for point in second:
                projection = point[0] * nx + point[1] * ny
                minimum_b, maximum_b = min(minimum_b, projection), max(maximum_b, projection)
            if maximum_a <= minimum_b + _EPS or maximum_b <= minimum_a + _EPS:
                separated = True
                break
        if separated:
            break
    if not separated:
        return 0.0
    squared_distance = math.inf
    for which in range(2):
        points, polygon = (first, second) if which == 0 else (second, first)
        for point in points:
            for edge in range(len(polygon)):
                following = (edge + 1) % len(polygon)
                dx = polygon[following, 0] - polygon[edge, 0]
                dy = polygon[following, 1] - polygon[edge, 1]
                wx, wy = point[0] - polygon[edge, 0], point[1] - polygon[edge, 1]
                length_squared = dx * dx + dy * dy
                fraction = (max(0.0, min(1.0, (wx * dx + wy * dy) / length_squared))
                            if length_squared > 0.0 else 0.0)
                ex, ey = wx - fraction * dx, wy - fraction * dy
                squared_distance = min(squared_distance, ex * ex + ey * ey)
    return math.sqrt(squared_distance)


def polygon_distance(first: Polygon, second: Polygon) -> float:
    """Euclidean convex boundary distance (zero if intersecting), without Python vertex loops."""
    return _convex_polygon_distance(np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64))


def footprint_line_limit(moving_polygon, obstacle_polygon, x: float, gap: float) -> float | None:
    """Highest forbidden placement Y at a fixed X, without an iterative search.

    ``moving_polygon`` is a convex footprint in its already-rotated local
    coordinates, retaining the chosen placement origin. ``obstacle_polygon``
    is convex and in world XY. The returned Y is the upper boundary of
    ``obstacle + (-moving) + disk(gap)`` at ``x``. A center above that boundary
    is separated by at least ``gap``; ``None`` means the vertical line misses
    the forbidden region altogether. Inputs are never silently recentered.

    The finite Minkowski hull and closed-form segment/circle intersections run
    only for the shared scene's host-side construction, never per physics step
    or per cloned environment. In particular, there is no gap bisection,
    placement rejection, or conditional regeneration here.
    """
    from scipy.spatial import ConvexHull, QhullError

    moving = np.asarray(moving_polygon, dtype=np.float64)
    obstacle = np.asarray(obstacle_polygon, dtype=np.float64)
    if (moving.ndim != 2 or moving.shape[1] != 2 or len(moving) < 3
            or obstacle.ndim != 2 or obstacle.shape[1] != 2 or len(obstacle) < 3
            or not np.isfinite(moving).all() or not np.isfinite(obstacle).all()
            or not math.isfinite(x) or not math.isfinite(gap) or gap < 0.0):
        raise ValueError("Supply finite convex XY polygons, a finite X and a nonnegative gap.")
    try:
        # A convex Minkowski boundary consists of the angle-ordered edge union.
        # This uses at most Va+Vb points rather than allocating Va*Vb sums for
        # detailed authored outlines (the current cup has nearly 500 vertices).
        # ConvexHull also normalizes clockwise inputs and duplicate/collinear
        # vertices before the edge merge. Every hull is linear in storage.
        operands = (obstacle[ConvexHull(obstacle).vertices], -moving[ConvexHull(moving).vertices])
        start, edge_sets = np.zeros(2), []
        for polygon in operands:
            bottom = np.lexsort((polygon[:, 0], polygon[:, 1]))[0]
            polygon = np.roll(polygon, -bottom, axis=0)
            start += polygon[0]
            edge_sets.append(np.roll(polygon, -1, axis=0) - polygon)
        merged = np.concatenate(edge_sets)
        angles = np.mod(np.arctan2(merged[:, 1], merged[:, 0]), 2.0 * math.pi)
        merged = merged[np.argsort(angles, kind="stable")]
        vertices = np.vstack((start, start + np.cumsum(merged, axis=0)[:-1]))
        boundary = vertices[ConvexHull(vertices).vertices]
    except QhullError as error:
        raise ValueError("Footprints must define a nonzero-area configuration obstacle.") from error

    tolerance = 32.0 * np.finfo(np.float64).eps * max(1.0, abs(x), float(np.abs(boundary).max()), gap)
    delta_x = x - boundary[:, 0]
    caps = np.abs(delta_x) <= gap + tolerance
    if not caps.any() and (x < boundary[:, 0].min() - gap or x > boundary[:, 0].max() + gap):
        return None
    values = [boundary[caps, 1] + np.sqrt(np.maximum(0.0, gap * gap - delta_x[caps] ** 2))]

    edges = np.roll(boundary, -1, axis=0) - boundary
    # Each straight boundary segment translates along its outward unit normal.
    normals = np.stack((edges[:, 1], -edges[:, 0]), axis=-1) / np.linalg.norm(edges, axis=-1)[:, None]
    starts = boundary + gap * normals
    ends = starts + edges
    eligible = ((x >= np.minimum(starts[:, 0], ends[:, 0]) - tolerance)
                & (x <= np.maximum(starts[:, 0], ends[:, 0]) + tolerance))
    nonvertical = eligible & (np.abs(edges[:, 0]) > tolerance)
    fraction = np.clip((x - starts[nonvertical, 0]) / edges[nonvertical, 0], 0.0, 1.0)
    values.append(starts[nonvertical, 1] + fraction * edges[nonvertical, 1])
    vertical = eligible & ~nonvertical
    values.append(np.maximum(starts[vertical, 1], ends[vertical, 1]))
    candidates = np.concatenate(values)
    return float(candidates.max()) if len(candidates) else None


@njit(cache=True)
def _clutter_pack_boundaries(positive, negative, counts, gap):
    """Merge each configuration polygon once, without Python pair allocation."""
    count, obstacles = len(negative), len(positive)
    total = 0
    for current in range(count):
        for other in range(current + 1, obstacles):
            total += (counts[current] + counts[other]) * (2 if other < count else 1)
    boundaries = np.empty((total, 6))
    offsets = np.zeros((count, obstacles, 2), dtype=np.int64)
    bounds = np.zeros((count, obstacles, 4))
    written = 0
    for current in range(count):
        for other in range(current + 1, obstacles):
            width = counts[current] + counts[other]
            begin, reverse_begin = written, written + width
            offsets[current, other, 0], offsets[current, other, 1] = begin, begin + width
            if other < count:
                offsets[other, current, 0], offsets[other, current, 1] = reverse_begin, reverse_begin + width
            x = positive[other, 0, 0] + negative[current, 0, 0]
            y = positive[other, 0, 1] + negative[current, 0, 1]
            lo_x, hi_x, lo_y, hi_y = math.inf, -math.inf, math.inf, -math.inf
            first, second = 0, 0
            for point in range(width):
                if first < counts[other] and (second == counts[current]
                        or positive[other, first, 4] <= negative[current, second, 4]):
                    dx, dy = positive[other, first, 2], positive[other, first, 3]
                    first += 1
                else:
                    dx, dy = negative[current, second, 2], negative[current, second, 3]
                    second += 1
                length = math.hypot(dx, dy)
                normal_x, normal_y = dy / length, -dx / length
                row = begin + point
                boundaries[row, 0], boundaries[row, 1] = x, y
                boundaries[row, 2], boundaries[row, 3] = x + gap * normal_x, y + gap * normal_y
                boundaries[row, 4], boundaries[row, 5] = dx, dy
                if other < count:
                    for column in range(6):
                        boundaries[reverse_begin + point, column] = -boundaries[row, column]
                lo_x, hi_x, lo_y, hi_y = min(lo_x, x), max(hi_x, x), min(lo_y, y), max(hi_y, y)
                x, y = x + dx, y + dy
            bounds[current, other, 0], bounds[current, other, 1] = lo_x, hi_x
            bounds[current, other, 2], bounds[current, other, 3] = lo_y, hi_y
            if other < count:
                bounds[other, current, 0], bounds[other, current, 1] = -hi_x, -lo_x
                bounds[other, current, 2], bounds[other, current, 3] = -hi_y, -lo_y
            written += width * (2 if other < count else 1)
    return boundaries, offsets, bounds


@njit(cache=True)
def _clutter_line_interval(boundary, begin, end, bounds, axis, coordinate, gap):
    """Section of one convex Minkowski sum with a Euclidean gap disk."""
    fixed = 1 - axis
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(coordinate), gap)
    # At an outer tangent line every contact has exactly the requested gap;
    # its closed section must not be mistaken for a forbidden open interval.
    if coordinate <= bounds[2 * fixed] - gap + tolerance or coordinate >= bounds[2 * fixed + 1] + gap - tolerance:
        return math.inf, -math.inf
    lower, upper = math.inf, -math.inf
    for index in range(begin, end):
        point = boundary[index]
        difference = coordinate - point[fixed]
        if abs(difference) <= gap + tolerance:
            radius = math.sqrt(max(0.0, gap * gap - difference * difference))
            lower = min(lower, point[axis] - radius)
            upper = max(upper, point[axis] + radius)
        start, delta = point[2 + fixed], point[4 + fixed]
        if min(start, start + delta) - tolerance <= coordinate <= max(start, start + delta) + tolerance:
            if abs(delta) > tolerance:
                fraction = min(1.0, max(0.0, (coordinate - start) / delta))
                value = point[2 + axis] + fraction * point[4 + axis]
                lower, upper = min(lower, value), max(upper, value)
            else:
                lower = min(lower, point[2 + axis], point[2 + axis] + point[4 + axis])
                upper = max(upper, point[2 + axis], point[2 + axis] + point[4 + axis])
    return lower, upper


@njit(cache=True)
def _clutter_position_sweeps(centers, local_bounds, support, boundaries, offsets, bounds, orders, axes, draws, gap):
    """Fixed-work coordinate Gibbs updates; every draw is in the full free union."""
    count, obstacles = offsets.shape[:2]
    forbidden = np.empty((obstacles, 2))
    free = np.empty((obstacles + 1, 2))
    moves, split_updates, maximum_intervals, total_distance = 0, 0, 0, 0.0
    for sweep in range(len(orders)):
        for position in range(count):
            current = orders[sweep, position]
            for part in range(2):
                axis = axes[sweep, position, part]
                fixed = 1 - axis
                lower = support[2 * axis] - local_bounds[current, 2 * axis]
                upper = support[2 * axis + 1] - local_bounds[current, 2 * axis + 1]
                tolerance = 1.e-10 * max(1.0, abs(lower), abs(upper))
                if lower > upper + tolerance:
                    return centers, -1, current, axis, moves, split_updates, maximum_intervals, total_distance
                if not lower - tolerance <= centers[current, axis] <= upper + tolerance:
                    return centers, -2, current, axis, moves, split_updates, maximum_intervals, total_distance
                used = 0
                for other in range(obstacles):
                    if other == current:
                        continue
                    shift_fixed = centers[other, fixed] if other < count else 0.0
                    lo, hi = _clutter_line_interval(
                        boundaries, offsets[current, other, 0], offsets[current, other, 1],
                        bounds[current, other], axis, centers[current, fixed] - shift_fixed, gap)
                    if lo > hi:
                        continue
                    shift_axis = centers[other, axis] if other < count else 0.0
                    lo, hi = lo + shift_axis, hi + shift_axis
                    if lo + tolerance < centers[current, axis] < hi - tolerance:
                        return centers, -3, current, other, moves, split_updates, maximum_intervals, total_distance
                    lo, hi = max(lower, lo), min(upper, hi)
                    if lo < hi:
                        forbidden[used, 0], forbidden[used, 1] = lo, hi
                        used += 1
                # Finite interval union; insertion sort avoids an allocation for
                # each of the 2 * count * sweeps coordinate updates.
                for index in range(1, used):
                    lo, hi = forbidden[index, 0], forbidden[index, 1]
                    previous = index - 1
                    while previous >= 0 and forbidden[previous, 0] > lo:
                        forbidden[previous + 1] = forbidden[previous]
                        previous -= 1
                    forbidden[previous + 1, 0], forbidden[previous + 1, 1] = lo, hi
                cursor, available = lower, 0
                for index in range(used):
                    lo, hi = forbidden[index, 0], forbidden[index, 1]
                    if lo > cursor:
                        free[available, 0], free[available, 1] = cursor, lo
                        available += 1
                    cursor = max(cursor, hi)
                if cursor < upper:
                    free[available, 0], free[available, 1] = cursor, upper
                    available += 1
                if available == 0:
                    # A supported exact contact configuration can have only an
                    # isolated valid center. Preserve it, but never repair an
                    # overlapping input by silently drawing another pose.
                    # Shelf and every open forbidden interval were checked
                    # against the current center immediately above.
                    continue
                maximum_intervals = max(maximum_intervals, available)
                split_updates += available > 1
                length = 0.0
                for interval in range(available):
                    length += free[interval, 1] - free[interval, 0]
                close_width = max(0.0, 0.010 - max(0.003, gap))
                close = draws[sweep, position, part, 0] < 0.65 and close_width > 0.0
                measure = length
                if close:
                    measure = 0.0
                    for interval in range(available):
                        measure += 2.0 * min(close_width, 0.5 * (free[interval, 1] - free[interval, 0]))
                selected = draws[sweep, position, part, 1] * measure
                value = centers[current, axis]
                for interval in range(available):
                    lo, hi = free[interval, 0], free[interval, 1]
                    width = min(close_width, 0.5 * (hi - lo)) if close else hi - lo
                    weight = 2.0 * width if close else width
                    if selected <= weight or interval == available - 1:
                        if close and selected > width:
                            value = hi - min(width, selected - width)
                        else:
                            value = lo + min(width, selected)
                        # Keep roundoff away from both forbidden boundaries;
                        # this does not remove any open free interval support.
                        margin = min(1.e-12, 0.25 * (hi - lo))
                        value = min(hi - margin, max(lo + margin, value))
                        break
                    selected -= weight
                total_distance += abs(value - centers[current, axis])
                moves += abs(value - centers[current, axis]) > tolerance
                centers[current, axis] = value
    return centers, 0, -1, -1, moves, split_updates, maximum_intervals, total_distance


def _mix_clutter_positions(local_polygons, centers, support_bounds, fixed_obstacles, gap, rng, sweeps=3,
                           *, center_bounds=None):
    """Mix a valid seed over exact collision-free XY coordinate intervals.

    The already-rotated convex footprints retain their original placement
    origins. ``support_bounds`` is ``(xmin, xmax, ymin, ymax)``; fixed obstacles
    are world-XY convex pieces, including role/path/view reserves. All movable
    objects are obstacles for every other object. Yaw and asset composition do
    not change. This is a fixed number of direct conditional draws, without
    pose proposals, approval retries, compaction, or a convergence gate.

    Minkowski boundaries are prepared once per polygon pair and packed for
    compiled line/disk intersections. A uniform component gives every free
    interval positive probability; the other component favors 3–10 mm gaps.
    The input must already be supported and separated by at least ``gap``.
    """
    from scipy.spatial import ConvexHull, QhullError

    polygons = tuple(np.asarray(polygon, dtype=np.float64) for polygon in local_polygons)
    fixed = tuple(np.asarray(polygon, dtype=np.float64) for polygon in fixed_obstacles)
    result, support = np.asarray(centers, dtype=np.float64), np.asarray(support_bounds, dtype=np.float64)
    if (result.shape != (len(polygons), 2) or support.shape != (4,) or not np.isfinite(result).all()
            or not np.isfinite(support).all() or support[0] >= support[1] or support[2] >= support[3]
            or not math.isfinite(gap) or gap < 0.0 or not isinstance(sweeps, int) or sweeps < 0):
        raise ValueError("Clutter mixing needs finite XY centers/support, a nonnegative gap and integer sweeps.")
    normalized = []
    for polygon in polygons + fixed:
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError("Clutter mixing needs finite nonzero-area convex XY polygons.")
        try:
            normalized.append(polygon[ConvexHull(polygon).vertices])
        except QhullError as error:
            raise ValueError("Clutter mixing needs nonzero-area configuration obstacles.") from error
    count, obstacles = len(polygons), len(normalized)
    local_bounds = np.empty((count, 4))
    for index, polygon in enumerate(normalized[:count]):
        local_bounds[index] = polygon[:, 0].min(), polygon[:, 0].max(), polygon[:, 1].min(), polygon[:, 1].max()
    if center_bounds is not None:
        limits = np.asarray(center_bounds, dtype=np.float64)
        if limits.shape != (count, 4) or not np.isfinite(limits).all():
            raise ValueError('Clutter center constraints must be finite [objects, xmin/xmax/ymin/ymax].')
        lower = np.maximum(support[0::2]-local_bounds[:, 0::2], limits[:, 0::2])
        upper = np.minimum(support[1::2]-local_bounds[:, 1::2], limits[:, 1::2])
        # The compiled sampler consumes wall offsets, not these vertices, so
        # encode the additional OBB/panel constraints without changing the
        # actual collision polygons used by every Minkowski boundary.
        local_bounds[:, 0::2] = support[0::2]-lower
        local_bounds[:, 1::2] = support[1::2]-upper
    counts = np.array([len(polygon) for polygon in normalized], dtype=np.int64)
    width = max(counts, default=0)
    positive, negative = np.zeros((obstacles, width, 5)), np.zeros((count, width, 5))
    # Normalize each input once. The compiled pair merge reuses its ordered
    # edges rather than allocating/sorting a polygon for every object pair.
    for sign, destination, selected in ((1.0, positive, normalized), (-1.0, negative, normalized[:count])):
        for index, original in enumerate(selected):
            polygon = sign * original
            bottom = np.lexsort((polygon[:, 0], polygon[:, 1]))[0]
            polygon = np.roll(polygon, -bottom, axis=0)
            edges = np.roll(polygon, -1, axis=0) - polygon
            destination[index, :len(polygon), :2] = polygon
            destination[index, :len(polygon), 2:4] = edges
            destination[index, :len(polygon), 4] = np.mod(np.arctan2(edges[:, 1], edges[:, 0]), 2.0 * math.pi)
    boundaries, offsets, bounds = _clutter_pack_boundaries(positive, negative, counts, float(gap))
    orders = np.empty((sweeps, count), dtype=np.int64)
    axes = np.empty((sweeps, count, 2), dtype=np.int64)
    draws = np.empty((sweeps, count, 2, 2))
    for sweep in range(sweeps):
        order = list(range(count))
        rng.shuffle(order)
        orders[sweep] = order
        for position in range(count):
            first = int(rng.random() < 0.5)
            axes[sweep, position] = first, 1 - first
            for part in range(2):
                draws[sweep, position, part] = rng.random(), rng.random()
    result, status, current, other, moves, split_updates, maximum_intervals, distance = _clutter_position_sweeps(
        result.copy(), local_bounds, support, boundaries, offsets, bounds, orders, axes, draws, float(gap))
    if status:
        reason = {-1: "empty shelf interval", -2: "unsupported initial center", -3: "overlapping initial/current geometry",
                  -4: "empty free coordinate interval"}[status]
        raise ValueError(f"Clutter mixing {reason}: body={current}, axis/obstacle={other}.")
    return result, {
        "method": "fixed_coordinate_free_intervals", "sweeps": sweeps, "coordinate_draws": 2 * count * sweeps,
        "pose_proposals": 0, "approval_retries": 0, "minkowski_pairs": count * (obstacles - 1),
        "changed_coordinates": moves, "split_interval_updates": split_updates,
        "maximum_free_intervals": maximum_intervals, "coordinate_travel": distance,
    }


def _constructive_slicing_plans(sizes, gap: float, max_depth: float, max_width: float):
    """Compute exact Pareto slicing layouts for every ordered leaf interval.

    Each leaf size is ``(width_Y, depth_X)``; plans are ``(width, depth, tree)``.
    Axis joins keep a fixed ``gap`` between their two bounding rectangles.
    This finite dynamic program runs once for the shared scene construction,
    with no random rejection, environment loop, or physics-step work. Its
    Pareto frontier is exact: no plan-count cap or approximate size bins are
    used, and leaves keep the supplied order rather than permuting objects.
    """
    dimensions = np.asarray(sizes, dtype=np.float64)
    if dimensions.shape == (0,):
        dimensions = dimensions.reshape(0, 2)
    if (dimensions.ndim != 2 or dimensions.shape[1] != 2
            or not np.isfinite(dimensions).all() or (dimensions <= 0.0).any()
            or not all(math.isfinite(value) for value in (gap, max_depth, max_width))
            or gap < 0.0 or max_depth <= 0.0 or max_width <= 0.0):
        raise ValueError("Supply positive finite (width_Y, depth_X) sizes and bounds, and a nonnegative gap.")
    count = len(dimensions)
    plans = {(index, index): [(0.0, 0.0, None)] for index in range(count + 1)}
    for index, (width, depth) in enumerate(dimensions):
        plans[index, index + 1] = (
            [(float(width), float(depth), ('leaf', index))]
            if width <= max_width and depth <= max_depth else []
        )
    for length in range(2, count + 1):
        for begin in range(count - length + 1):
            end = begin + length
            candidates = []
            for split in range(begin + 1, end):
                for left_width, left_depth, left_tree in plans[begin, split]:
                    for right_width, right_depth, right_tree in plans[split, end]:
                        width, depth = left_width + gap + right_width, max(left_depth, right_depth)
                        if width <= max_width and depth <= max_depth:
                            candidates.append((width, depth, ('y', left_tree, right_tree, left_width + gap)))
                        width, depth = max(left_width, right_width), left_depth + gap + right_depth
                        if width <= max_width and depth <= max_depth:
                            candidates.append((width, depth, ('x', left_tree, right_tree, left_depth + gap)))
            # At increasing width, only a strictly smaller depth can be
            # nondominated. Stable sorting preserves deterministic tree choice
            # for identical dimensions without comparing nested tree objects.
            candidates.sort(key=lambda plan: (plan[0], plan[1]))
            frontier, smallest_depth = [], math.inf
            for plan in candidates:
                if plan[1] < smallest_depth:
                    frontier.append(plan)
                    smallest_depth = plan[1]
            plans[begin, end] = frontier
    return plans


def _constructive_unpack_plan(tree, origin_xy, sizes, gap: float):
    """Unpack a slicing tree into leaf centers from a lower-left XY origin.

    Join offsets already include ``gap`` as authored by the slicing planner;
    the parameter is retained explicitly to make the placement contract clear.
    This is one finite tree traversal for a shared layout, not an environment
    loop or a repeated placement/condition search.
    """
    if (len(origin_xy) != 2 or not all(math.isfinite(value) for value in origin_xy)
            or not math.isfinite(gap) or gap < 0.0):
        raise ValueError("Supply a finite lower-left XY origin and a nonnegative gap.")
    centers, pending = {}, [(tree, float(origin_xy[0]), float(origin_xy[1]))]
    while pending:
        node, x, y = pending.pop()
        if node is None:
            continue
        if node[0] == 'leaf':
            index = node[1]
            width, depth = sizes[index]
            centers[index] = (x + .5 * depth, y + .5 * width)
        elif node[0] == 'x':
            pending.append((node[2], x + node[3], y))
            pending.append((node[1], x, y))
        elif node[0] == 'y':
            pending.append((node[2], x, y + node[3]))
            pending.append((node[1], x, y))
        else:
            raise ValueError(f"Unknown slicing-plan node: {node[0]!r}.")
    return centers


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


def _path_sweeps(geometry: ObjectGeometry, path: Sequence[PlanarMotion], *, packed=False):
    if packed:
        half = ((geometry.upper[0] - geometry.lower[0]) * .5, (geometry.upper[1] - geometry.lower[1]) * .5)
        arrays = [_motion_envelopes(*half, (m.start.x, m.start.y, m.start.yaw),
                  (m.end.x, m.end.y, m.end.yaw), m.pivot or (0., 0.), m.pivot is not None) for m in path]
        return _SweepPolygons(np.concatenate([a[0] for a in arrays]), np.concatenate([a[1] for a in arrays]))
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


@njit(cache=True)
def _candidate_path_arrays(x, y, yaw, hx, hy, access, gap):
    """Pack the fixed path family without constructing per-motion Python objects."""
    prefixes = np.zeros((25, 2, 9))
    lengths = np.zeros(25, np.int32)
    poses = np.empty((25, 3))
    poses[0] = (x, y, yaw)
    cursor = 1
    c, s = math.cos(yaw), math.sin(yaw)
    for angle in (-math.pi/2, math.pi/2, -math.pi/4, math.pi/4):
        end_yaw = yaw + angle
        prefixes[cursor, 0, :6] = (x, y, yaw, x, y, end_yaw)
        lengths[cursor], poses[cursor] = 1, (x, y, end_yaw)
        cursor += 1
        delta = end_yaw - yaw
        ca, sa = math.cos(delta), math.sin(delta)
        for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            pivot_x, pivot_y = x + c*px - s*py, y + s*px + c*py
            dx, dy = x - pivot_x, y - pivot_y
            ex, ey = pivot_x + ca*dx - sa*dy, pivot_y + sa*dx + ca*dy
            eyaw = yaw + delta
            prefixes[cursor, 0] = (x, y, yaw, ex, ey, eyaw, pivot_x, pivot_y, 1.)
            lengths[cursor], poses[cursor] = 1, (ex, ey, eyaw)
            cursor += 1
    for dx in (.015, .03):
        for angle in (-math.pi/2, math.pi/2):
            prefixes[cursor, 0, :6] = (x, y, yaw, x+dx, y, yaw)
            prefixes[cursor, 1, :6] = (x+dx, y, yaw, x+dx, y, yaw+angle)
            lengths[cursor], poses[cursor] = 2, (x+dx, y, yaw+angle)
            cursor += 1
    clear = _prefix_clearance(poses, hx, hy, access, gap)
    motions, counts = np.zeros((74, 3, 9)), np.zeros(74, np.int32)
    written = 0
    for index in range(25):
        count = lengths[index]
        if count and clear[index]:
            motions[written, :count] = prefixes[index, :count]
            counts[written] = count
            written += 1
        px, py, angle = poses[index]
        extent = abs(math.sin(angle))*hx + abs(math.cos(angle))*hy
        for end_y in (access[2]-extent-gap-.001, access[3]+extent+gap+.001):
            motions[written, :count] = prefixes[index, :count]
            motions[written, count, :6] = (px, py, angle, px, end_y, angle)
            counts[written] = count + 1
            written += 1
    return motions[:written], counts[:written]


@dataclass
class _PackedPaths:
    motions: np.ndarray
    counts: np.ndarray

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, index):
        values = self.motions[index, :self.counts[index]]
        return tuple(PlanarMotion(PlanarPose(*map(float, row[:3])), PlanarPose(*map(float, row[3:6])),
                                  tuple(map(float, row[6:8])) if row[8] else None) for row in values)


def _clearance_candidates(geometry: ObjectGeometry, initial: PlanarPose, access: Region,
                          rules: SpawnRules, rng: random.Random, *, packed=False):
    """Finite translation, pivot, rotate/translate, and multistage proposals."""
    if packed:
        motions, counts = _candidate_path_arrays(initial.x, initial.y, initial.yaw,
            (geometry.upper[0]-geometry.lower[0])/2, (geometry.upper[1]-geometry.lower[1])/2,
            np.asarray(access), rules.gap)
        order = list(range(len(counts)))
        rng.shuffle(order)
        return _PackedPaths(motions[order], counts[order])
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
    candidates: list[tuple[PlanarMotion, ...]] | _PackedPaths
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
        if isinstance(request.candidates, _PackedPaths):
            size = len(request.candidates)
            motions[row, :size] = request.candidates.motions
            counts[row, :size] = request.candidates.counts
            continue
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
    obstacles = max(len(plan[5]) for plan in plans) + 1 + int(rules.reserve_blocker_front)
    vertices_width = max(4, max(int(plan[5].counts.max()) if isinstance(plan[5], _SweepPolygons)
                              else max(map(len, plan[5])) for plan in plans))
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
    catalog_half = (np.asarray([g.upper for g in geometries]) - np.asarray([g.lower for g in geometries]))[:, :2] * .5
    for row, plan in enumerate(plans):
        ids, positions, yaws, *_ = plan
        seeds[row] = plan[-1] & 0x7fffffff
        state[row,1], state[row,5] = 2,len(ids)
        half[row, :len(ids)] = catalog_half[list(ids)]
        poses[row, :2] = [(*positions[index][:2], yaws[index]) for index in ids[:2]]
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
        vertices[row, 0, :4], counts[row, 0] = rectangle_polygon(plan[3]), 4
        sweeps = plan[5]
        if isinstance(sweeps, _SweepPolygons):
            copy_width = min(vertices_width, sweeps.vertices.shape[1])
            vertices[row, 1:1+len(sweeps), :copy_width] = sweeps.vertices[:, :copy_width]
            counts[row, 1:1+len(sweeps)] = sweeps.counts
        else:  # Scalar/debug callers may supply arbitrary polygons, including degenerate edges.
            for column, polygon in enumerate(sweeps, 1):
                vertices[row, column, :len(polygon)], counts[row, column] = polygon, len(polygon)
        if rules.reserve_blocker_front:
            column = len(sweeps) + 1
            vertices[row, column, :4] = rectangle_polygon(_blocker_front_region(role_polygons[1], rules))
            counts[row, column], gaps[row, column] = 4, 0.
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
    # Arguments and addresses stay fixed throughout packing. Record them once
    # instead of rebuilding ~40 ctypes arguments for every kernel/round.
    propose = wp.launch(_layout_candidates, dim=(n,64), inputs=[p,h,on,st,seed,reg,ar,v,ax,box,co,ga,bounds,
        rules.gap,rules.neighbor_distance,rules.min_region_occupancy,cell_area,candidates,scores,collisions],
        device=device, stream=stream, record_cmd=True)
    commit = wp.launch(_layout_commit, dim=n, inputs=[p,h,on,st,reg,ar,adj,candidates,scores,collisions,bounds,
        rules.neighbor_distance,rules.min_neighbor_fraction,rules.min_region_occupancy,cell_area,
        rules.min_x_coverage,rules.min_y_coverage], device=device, stream=stream, record_cmd=True)
    # Poll only once per 16 resident rounds; completed rows remain masked.
    for begin in range(0, 4*width+32, 16):
        for _ in range(16):
            propose.launch(stream=stream)
            commit.launch(stream=stream)
        status = buffers[3].cpu().numpy().copy()
        if np.all(status[:,0] != 0):
            break
    if statistics is not None:
        statistics['plans'] = statistics.get('plans',0)+n
        statistics['candidate_rounds'] = statistics.get('candidate_rounds',0)+int(status[:,2].sum())
        statistics['evictions'] = statistics.get('evictions',0)+int(status[:,4].sum())
        statistics['rejected_plans'] = statistics.get('rejected_plans',0)+int((status[:,0]!=1).sum())
    if keep_device:
        return [(None, buffers[0][row, :len(plan[0])]) if status[row, 0] == 1 else None
                for row, plan in enumerate(plans)]
    result = buffers[0].cpu().numpy()
    return [result[row, :len(plan[0])].copy() if status[row, 0] == 1 else None
            for row, plan in enumerate(plans)]


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


def _collision_polygon(geometry, yaw):
    """Canonical collision envelope, rotated about the existing OBB-center convention."""
    if geometry.collision_footprint_vertices:
        points = np.asarray(geometry.collision_footprint_vertices, dtype=np.float64)
        points = points - (np.asarray(geometry.lower[:2]) + geometry.upper[:2]) * .5
    else:
        half = (np.asarray(geometry.upper[:2]) - geometry.lower[:2]) * .5
        points = np.asarray(((-1, -1), (1, -1), (1, 1), (-1, 1))) * half
    c, s = math.cos(yaw), math.sin(yaw)
    return points @ np.asarray(((c, s), (-s, c)))


def _constructive_pivot(polygon, center):
    """Rotate around a rear support vertex while every point stays in its front half-plane."""
    origin = polygon[int(np.argmin(polygon[:, 0]))]
    vectors = polygon - origin
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])[np.linalg.norm(vectors, axis=-1) > 1.e-12]
    limits = (-math.pi / 2 - float(angles.min()), math.pi / 2 - float(angles.max()))
    candidates = []
    for angle in limits:
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((c, -s), (s, c)))
        final = vectors @ rotation.T + origin
        candidates.append((float(np.ptp(final[:, 0])), -abs(angle), angle, rotation, final))
    _, _, angle, rotation, final = min(candidates, key=lambda entry: entry[:2])
    finish = (np.asarray(center) - origin) @ rotation.T + origin
    return origin, angle, finish, final


def _pivot_bounds(polygon, pivot, angle):
    """Exact coordinate extrema of vertex arcs; no sampled motion acceptance loop."""
    relative = np.asarray(polygon) - pivot
    phases = np.arctan2(relative[:, 1], relative[:, 0])
    radii = np.linalg.norm(relative, axis=-1)
    low, high = phases + min(0., angle), phases + max(0., angle)
    endpoint_angles = np.stack((phases, phases + angle), -1)
    x = radii[:, None] * np.cos(endpoint_angles) + pivot[0]
    y = radii[:, None] * np.sin(endpoint_angles) + pivot[1]
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    for k in range(-3, 4):
        critical = k * math.pi / 2
        mask = (low <= critical) & (critical <= high)
        if mask.any():
            xx = pivot[0] + radii[mask] * math.cos(critical)
            yy = pivot[1] + radii[mask] * math.sin(critical)
            xmin, xmax = min(xmin, xx.min()), max(xmax, xx.max())
            ymin, ymax = min(ymin, yy.min()), max(ymax, yy.max())
    return float(xmin), float(xmax), float(ymin), float(ymax)


def _constructive_pair_view(geometries, target, blocker, yaws, bx, side, rules, rng):
    """Invert perspective center projection once; no viewpoint candidates or redraws.

    This couples a lateral role offset to one front view. Projected body bounds
    provide the geometric construction; the sensor still measures true mesh
    silhouette fractions and explicitly diagnoses any violated invariant.
    """
    from scipy.spatial.transform import Rotation
    eye = np.array((bx + rng.uniform(.95, 1.0), side * rng.uniform(.25, .35),
                    rules.shelf_top + rng.uniform(.18, .20)))
    aim = np.array((bx * .5, 0., rules.shelf_top + .10))
    forward = aim - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0., 0., 1.))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    frame = np.stack((right, down, forward), -1)
    projected, centers = [], []
    for index, x in ((target, 0.), (blocker, bx)):
        geometry = geometries[index]
        z = geometry.upper[2] - geometry.lower[2]
        xy = np.asarray(footprint_polygon(geometry, (x, 0.), yaws[index]))
        corners = np.concatenate((np.c_[xy, np.full(4, rules.shelf_top)],
                                  np.c_[xy, np.full(4, rules.shelf_top + z)]))
        rays = (corners - eye) @ frame
        image = rays[:, :2] / rays[:, 2:3]
        projected.append((image.min(0), image.max(0)))
        centers.append((np.array((x, 0., rules.shelf_top + z * .5)) - eye) @ frame)
    (tlo, thi), (blo, bhi) = projected
    vertical_overlap = max(0., min(thi[1], bhi[1]) - max(tlo[1], blo[1])) / max(1.e-12, thi[1] - tlo[1])
    horizontal_visible = 1. - min(.30, .70 * vertical_overlap) / max(vertical_overlap, 1.e-12)
    ht, hb = (thi[0] - tlo[0]) * .5, (bhi[0] - blo[0]) * .5
    desired_b = centers[0][0] / centers[0][2] - side * (hb + (2 * horizontal_visible - 1) * ht)
    by = (desired_b * centers[1][2] - centers[1][0]) / (right[1] - desired_b * forward[1])
    half_access = rules.hand_width * .5 + rules.approach_margin
    blocker_box = np.asarray(footprint_polygon(geometries[blocker], (bx, 0.), yaws[blocker]))
    target_front = _half_extents(geometries[target], yaws[target])[0]
    area = (geometries[blocker].upper[0] - geometries[blocker].lower[0]) * (
        geometries[blocker].upper[1] - geometries[blocker].lower[1])
    limit = half_access + float(np.abs(blocker_box[:, 1]).max())
    lo, hi = _blocker_y_interval(blocker_box,
        np.asarray((target_front + rules.approach_margin, bx + 1., -half_access, half_access)),
        -limit, limit, rules.min_blocked_ratio * area)
    if lo > hi:
        raise RuntimeError("The requested role geometry has no constructive initial blockage interval.")
    return float(np.clip(by, lo, hi)), eye, Rotation.from_matrix(frame).as_quat()


def _constructive_task_module(geometries, target, blocker, yaws, local, bx, side, rules, rng, gap):
    """A discrete direction's exact pivot/slide envelope, before choosing its domain."""
    by, eye, camera_rotation = _constructive_pair_view(
        geometries, target, blocker, yaws, bx, side, rules, rng)
    center = np.array((bx, by))
    polygon = local[blocker] + center
    pivot, angle, turned_center, turned_polygon = _constructive_pivot(polygon, center)
    offsets = turned_polygon - turned_center
    half_access = rules.hand_width * .5 + rules.approach_margin
    # X separation from Target holds for the entire motion. Only the access
    # boundary, rather than Target's possibly wider footprint, sets the Y goal.
    goal_y = (half_access - offsets[:, 1].min() + gap if side > 0
              else -half_access - offsets[:, 1].max() - gap)
    goal = np.array((turned_center[0], goal_y))
    goal_polygon = offsets + goal
    sweep = _pivot_bounds(polygon, pivot, angle)
    target_polygon = local[target]
    bounds = (min(float(target_polygon[:, 0].min()), sweep[0], float(goal_polygon[:, 0].min())),
              max(float(target_polygon[:, 0].max()), sweep[1], float(goal_polygon[:, 0].max())),
              min(-half_access, float(target_polygon[:, 1].min()), sweep[2], float(goal_polygon[:, 1].min())),
              max(half_access, float(target_polygon[:, 1].max()), sweep[3], float(goal_polygon[:, 1].max())))
    return center, pivot, angle, turned_center, goal, bounds, eye, camera_rotation


def _constructive_neighbors(local, positions, ids, rules):
    world = {index: local[index] + positions[index][:2] for index in ids}
    pairs = []
    for first, second in combinations(ids, 2):
        a, b = world[first], world[second]
        bound_gap = np.maximum(0., np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)))
        if float(bound_gap @ bound_gap) > (rules.max_neighbor_gap + 1.e-8) ** 2:
            continue
        distance = polygon_distance(tuple(map(tuple, a)), tuple(map(tuple, b)))
        if rules.gap - 1.e-8 <= distance <= rules.max_neighbor_gap + 1.e-8:
            pairs.append((first, second))
    return tuple(pairs)


@lru_cache(maxsize=128)
def _certified_footprint_profile(lower, upper, vertices):
    """Invariant convex width data, shared by identical copies of an asset."""
    from scipy.spatial import ConvexHull, QhullError
    from scipy.spatial.distance import pdist

    if vertices:
        points = np.asarray(vertices, dtype=np.float64)
    else:
        half = (np.asarray(upper[:2]) - lower[:2]) * .5
        points = np.asarray(((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))) * half
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("A certified background footprint must contain finite convex XY vertices.")
    try:
        points = points[ConvexHull(points).vertices]
    except QhullError as error:
        raise ValueError("A certified background footprint must have positive area.") from error
    diameter = float(pdist(points).max())
    edges = np.roll(points, -1, axis=0) - points
    normals = np.stack((-edges[:, 1], edges[:, 0]), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    widths = np.ptp(points @ normals.T, axis=0)
    index = int(np.argmin(widths))
    minimum = min(float(widths[index]), diameter)
    # R(yaw).T @ world_Y = (sin(yaw), cos(yaw)).
    angle = math.atan2(float(normals[index, 0]), float(normals[index, 1]))
    return diameter, minimum, angle


@lru_cache(maxsize=128)
def _certified_visual_padding(lower, upper, vertices, visual_vertices=()):
    """Yaw-invariant radius enclosing the visual hull around its collision hull.

    Distance to a convex set is convex, so its maximum over the visual hull
    occurs at a vertex (or a fallback OBB corner). Rotation preserves this enclosure. It is
    used only for the protected view strip; physical X support stays exact.
    """
    if not vertices:
        return 0.
    from scipy.spatial import ConvexHull

    points = np.asarray(vertices, dtype=np.float64)
    polygon = points[ConvexHull(points).vertices]
    corners = np.asarray(visual_vertices if visual_vertices else
                         ((lower[0], lower[1]), (upper[0], lower[1]),
                          (upper[0], upper[1]), (lower[0], upper[1])), dtype=np.float64)
    if corners.ndim != 2 or corners.shape[1] != 2 or not np.isfinite(corners).all():
        raise ValueError("A certified visual footprint needs finite XY vertices.")
    edges = np.roll(polygon, -1, axis=0)-polygon
    delta = corners[:, None, :]-polygon[None, :, :]
    cross = edges[None, :, 0]*delta[..., 1]-edges[None, :, 1]*delta[..., 0]
    inside = (cross >= -1.e-12).all(axis=1)
    parameter = np.clip((delta*edges).sum(axis=-1)/(edges*edges).sum(axis=-1), 0., 1.)
    distances = np.linalg.norm(delta-parameter[..., None]*edges, axis=-1).min(axis=1)
    return float(np.where(inside, 0., distances).max())


def _certified_background_capacity(geometries, background_ids, rules):
    """Certify one fixed two-row topology before learning, without pose search.

    Invariant diameter sorting followed by the fixed 0,1,1,0 row assignment is
    structural asset assignment, not an alternate-layout search. Each row's
    cached prefix bound states the width needed when any p of its bodies have
    unrestricted yaw. At least one free body per nonempty row gives every
    background slot full-circle marginal support, with correlated other yaws.
    A false depth certificate or nonpositive module allowance must be handled
    by startup configuration; it is not permission to redraw a reset layout.
    """
    ids = tuple(background_ids)
    if (len(set(ids)) != len(ids) or any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
            or not 0 <= index < len(geometries) for index in ids)):
        raise ValueError("Certified background IDs must be distinct spawned pool indices.")
    if (not math.isfinite(rules.gap) or not math.isfinite(rules.max_neighbor_gap)
            or not 0. <= rules.gap <= rules.max_neighbor_gap):
        raise ValueError("Certified background gaps must be finite, nonnegative and ordered.")
    xmin, xmax = rules.x_range[0] + rules.edge_margin, rules.x_range[1] - rules.edge_margin
    ymin, ymax = rules.y_range[0] + rules.edge_margin, rules.y_range[1] - rules.edge_margin
    if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax)) or xmin >= xmax or ymin >= ymax:
        raise ValueError("Certified background support bounds must have finite positive extent.")
    profiles, visual_padding = {}, {}
    for index in ids:
        geometry = geometries[index]
        if (not all(math.isfinite(value) for value in (*geometry.lower, *geometry.upper))
                or any(high <= low for low, high in zip(geometry.lower, geometry.upper))):
            raise ValueError("Certified background geometry needs finite, positive 3D bounds.")
        if geometry.upper[2] - geometry.lower[2] > rules.ceiling - rules.shelf_top - rules.gap + 1.e-12:
            raise ValueError(f"Background body {index} lacks the configured ceiling clearance.")
        profiles[index] = _certified_footprint_profile(tuple(geometry.lower), tuple(geometry.upper),
                                                      tuple(geometry.collision_footprint_vertices))
        visual_padding[index] = _certified_visual_padding(tuple(geometry.lower), tuple(geometry.upper),
                                                         tuple(geometry.collision_footprint_vertices),
                                                         tuple(geometry.visual_footprint_vertices))
    ordered = sorted(ids, key=lambda index: (-profiles[index][0], _asset_type(geometries[index]), index))
    rows = [[], []]
    for rank, index in enumerate(ordered):
        rows[((rank + 1) // 2) % 2].append(index)
    row_data = []
    for row in rows:
        base = sum(profiles[index][1] for index in row)
        padding = max((visual_padding[index] for index in row), default=0.)
        increases = sorted((profiles[index][0] - profiles[index][1] for index in row), reverse=True)
        prefix = np.cumsum(increases)
        spans = tuple(float(base + value + 2*padding + rules.max_neighbor_gap * max(0, len(row)-1))
                      for value in prefix)
        row_data.append(dict(ids=tuple(row), minimum_width_sum=base, span_upper_by_free=spans,
                             visual_padding=padding,
                             depth_upper=max((profiles[index][0] for index in row), default=0.)))
    nonempty = [row for row in row_data if row['ids']]
    depth = sum(row['depth_upper'] for row in nonempty) + rules.max_neighbor_gap * max(0, len(nonempty)-1)
    minimum_span = max((row['span_upper_by_free'][0] for row in nonempty), default=0.)
    independent_span = max((row['span_upper_by_free'][-1] for row in nonempty), default=0.)
    outside_gap = rules.max_neighbor_gap if ids else 0.
    return dict(background_ids=ids, rows=tuple(row_data), profiles=profiles, visual_padding=visual_padding,
                support_bounds=(float(xmin), float(xmax), float(ymin), float(ymax)),
                minimum_gap=float(rules.gap), maximum_gap=float(rules.max_neighbor_gap),
                depth_upper=float(depth), depth_valid=depth <= xmax-xmin + 1.e-12,
                minimum_slab_width=float(minimum_span), independent_slab_width=float(independent_span),
                marginal_module_width_limit=float(ymax-ymin-outside_gap-minimum_span),
                independent_module_width_limit=float(ymax-ymin-outside_gap-independent_span))


def _sample_certified_background(geometries, background_ids, rules, module_bounds, rng, *,
                                 side=1, capacity=None, spread_gaps=False):
    """Sample one inherently valid background from a startup-certified domain.

    ``module_bounds`` is a world-XY (xmin,xmax,ymin,ymax) region containing the
    roles, protected visual cones and maneuver reserve. Background occupies
    only its +Y or -Y side. Each row is built once at exact Minkowski gap
    boundaries, then translated as a whole. There is no generated-pose
    acceptance test, alternative topology, guard selection or regeneration.

    With spread_gaps=True, each row mixes the dense base gaps with a
    close-biased continuous distribution over its remaining width budget.
    Thus max_neighbor_gap bounds the dense base, not every possible gap.
    Returned neighbor pairs name the active predecessor constraint; their
    actual requested distances are recorded in certificate['neighbor_gaps'].

    Returns (OBB-center xyz positions, physical yaws, neighbor ID pairs,
    parameter certificate). Static domain errors must be caught during startup
    before this sampler is used by the reset path.
    """
    ids = tuple(background_ids)
    capacity = _certified_background_capacity(geometries, ids, rules) if capacity is None else capacity
    expected_bounds = (rules.x_range[0] + rules.edge_margin, rules.x_range[1] - rules.edge_margin,
                       rules.y_range[0] + rules.edge_margin, rules.y_range[1] - rules.edge_margin)
    if (capacity['background_ids'] != ids or capacity['support_bounds'] != expected_bounds
            or capacity['minimum_gap'] != rules.gap or capacity['maximum_gap'] != rules.max_neighbor_gap):
        raise ValueError("The background certificate does not match its IDs, support bounds or gap rules.")
    if not capacity['depth_valid']:
        raise ValueError(f"Fixed two-row background needs depth {capacity['depth_upper']:.6f}m, "
                         f"but measured usable depth is {expected_bounds[1]-expected_bounds[0]:.6f}m.")
    if side not in (-1, 1) or len(module_bounds) != 4 or not all(math.isfinite(value) for value in module_bounds):
        raise ValueError("A certified module needs finite XY bounds and side +1 or -1.")
    if module_bounds[0] >= module_bounds[1] or module_bounds[2] >= module_bounds[3]:
        raise ValueError("The certified module must have positive XY extent.")
    certificate = dict(capacity, side=int(side), spread_gaps=bool(spread_gaps))
    if not ids:
        return {}, {}, (), dict(certificate, free_ids=(), full_yaw_ids=(), neighbor_gaps=())
    xmin, xmax, ymin, ymax = expected_bounds
    # Validate the worst outside gap before drawing random parameters. An
    # interval certified here remains valid for every sampled outside gap.
    module_end = module_bounds[3] if side > 0 else -module_bounds[2]
    support_low, support_high = (ymin, ymax) if side > 0 else (-ymax, -ymin)
    certified_low = max(support_low, module_end + rules.max_neighbor_gap)
    if support_high-certified_low <= capacity['minimum_slab_width'] + 1.e-10:
        raise ValueError("The fixed background topology has no positive continuous yaw domain on this module side.")
    outside_gap = rng.uniform(rules.gap, rules.max_neighbor_gap)
    slab_low = max(support_low, module_end + outside_gap)
    slab_width = support_high-slab_low
    yaws, free_ids, full_yaw_ids, half_widths = {}, [], [], {}
    for row in capacity['rows']:
        members = row['ids']
        if not members:
            continue
        # Prefix widths are monotone. This selects an integer freedom budget,
        # not a candidate pose or a tested alternate layout.
        free_count = int(np.searchsorted(row['span_upper_by_free'], slab_width-1.e-10, side='left'))
        chosen = set(rng.sample(members, free_count))
        free_ids.extend(sorted(chosen))
        base = row['minimum_width_sum'] + 2*row['visual_padding'] + sum(
            capacity['profiles'][index][0]-capacity['profiles'][index][1] for index in chosen)
        slack = slab_width-base-rules.max_neighbor_gap*max(0, len(members)-1)
        constrained = len(members)-len(chosen)
        epsilon = slack/(2*constrained) if constrained else 0.
        for index in members:
            diameter, minimum, angle = capacity['profiles'][index]
            if index in chosen or diameter-minimum <= epsilon:
                yaws[index], half_widths[index] = rng.uniform(-math.pi, math.pi), math.pi
                full_yaw_ids.append(index)
            else:
                delta = 2*math.asin(min(1., epsilon/(2*diameter)))
                angle += rng.randrange(2)*math.pi + rng.uniform(-delta, delta)
                yaws[index], half_widths[index] = (angle+math.pi) % (2*math.pi)-math.pi, delta
    local = {index: _collision_polygon(geometries[index], yaws[index]) * (1., float(side)) for index in ids}
    visual = {}
    for index in ids:
        geometry = geometries[index]
        if geometry.visual_footprint_vertices:
            points = np.asarray(geometry.visual_footprint_vertices)-(
                np.asarray(geometry.lower[:2])+geometry.upper[:2])*.5
            c, s = math.cos(yaws[index]), math.sin(yaws[index])
            visual[index] = (points @ np.asarray(((c, s), (-s, c))))*(1., float(side))
        else:
            visual[index] = np.asarray(footprint_polygon(geometry, (0., 0.), yaws[index]))*(1., float(side))
    rows = [row['ids'] for row in capacity['rows'] if row['ids']]
    depths = [max(float(np.ptp(local[index][:, 0])) for index in row) for row in rows]
    row_gap = rng.uniform(rules.gap, rules.max_neighbor_gap) if len(rows) > 1 else 0.
    used_depth = sum(depths) + row_gap*max(0, len(rows)-1)
    if spread_gaps and len(rows) > 1 and rng.random() >= .5:
        row_gap += rng.random()**2*max(0., xmax-xmin-used_depth)
        used_depth = sum(depths)+row_gap
    plane = xmin + rng.random()*max(0., xmax-xmin-used_depth)
    centers, neighbors, neighbor_gaps, row_budgets = {}, [], [], []
    for row, depth in zip(rows, depths):
        drawn_gaps = None
        if spread_gaps and len(row) > 1:
            drawn_gaps = np.asarray([rng.uniform(rules.gap, rules.max_neighbor_gap) for _ in row[1:]])
            width = sum(float(np.ptp(local[index][:, 1])) for index in row)
            padding = max(capacity['visual_padding'][index] for index in row)
            surplus = max(0., slab_width-width-2*padding-float(drawn_gaps.sum()))
            extra = rng.random()**2*surplus if rng.random() >= .5 else 0.
            if extra:
                # Squared standard normals normalized by their sum have
                # Dirichlet(alpha=.5) law, without scene proposal rejection.
                weights = np.asarray([rng.gauss(0., 1.)**2 for _ in drawn_gaps])
                drawn_gaps += extra*weights/max(float(weights.sum()), 1.e-300)
            row_budgets.append(dict(ids=row, width_sum=width, visual_padding=padding,
                                    initial_surplus=surplus, extra_gap_total=extra,
                                    gap_sum=float(drawn_gaps.sum())))
        placed = []
        for rank, index in enumerate(row):
            x = plane-float(local[index][:, 0].min())
            if not placed:
                y = 0.
            else:
                gap = (float(drawn_gaps[rank-1]) if drawn_gaps is not None else
                       rng.uniform(rules.gap, rules.max_neighbor_gap))
                y, parent = max((footprint_line_limit(local[index], local[other]+centers[other], x, gap), other)
                                for other in placed)
                neighbors.append((parent, index))
                neighbor_gaps.append(float(gap))
            centers[index] = np.array((x, y))
            placed.append(index)
        row_low = min(float((polygon+centers[index])[:, 1].min())
                      for index in row for polygon in (local[index], visual[index]))
        row_high = max(float((polygon+centers[index])[:, 1].max())
                       for index in row for polygon in (local[index], visual[index]))
        shift = slab_low-row_low + rng.random()*max(0., slab_width-(row_high-row_low))
        for index in row:
            centers[index][1] += shift
        plane += depth+row_gap
    positions = {index: (float(centers[index][0]), float(side*centers[index][1]),
                         rules.shelf_top-geometries[index].lower[2]) for index in ids}
    certificate.update(free_ids=tuple(free_ids), full_yaw_ids=tuple(full_yaw_ids),
                       angle_half_widths=half_widths, outside_gap=float(outside_gap),
                       available_slab_width=float(slab_width), neighbor_gaps=tuple(neighbor_gaps),
                       row_x_gap=float(row_gap), row_gap_budgets=tuple(row_budgets))
    return positions, yaws, tuple(neighbors), certificate


def _certified_rotated_bounds(points, yaw_half_width):
    """Exact XY bounds for a finite point set over a continuous symmetric arc."""
    points = np.asarray(points, dtype=np.float64)
    half = float(yaw_half_width)
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.linalg.norm(points, axis=-1)
    endpoint_angles = angles[:, None] + np.asarray((-half, half))
    xs, ys = radii[:, None]*np.cos(endpoint_angles), radii[:, None]*np.sin(endpoint_angles)
    bounds = [float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())]
    for output, angle, sign in ((0, math.pi, -1.), (1, 0., 1.),
                                (2, -math.pi/2, -1.), (3, math.pi/2, 1.)):
        contains = np.ceil((angles-half-angle)/(2*math.pi)) <= np.floor((angles+half-angle)/(2*math.pi))
        if contains.any():
            extremum = sign*float(radii[contains].max())
            bounds[output] = min(bounds[output], extremum) if sign < 0 else max(bounds[output], extremum)
    return tuple(bounds)


def _certify_role_translation_cells(geometries, target, blocker, target_yaw, blocker_yaw, bx,
                                  translation_intervals, rules, *, yaw_half_width,
                                  cell_width=.004, camera_eye=None):
    """Certify continuous role parameters once, without sampled-pose acceptance.

    The Target OBB center is (0,0), Blocker center is (bx,s), and one common
    Rz(psi) rotates both bodies and the optional camera, with psi in the
    returned yaw_bounds. Each returned parameter_bounds is a closed s cell.
    Its eroded Blocker rectangle lies inside every initial Blocker OBB in
    that cell. Clipping it against a conservative access half-strip proves
    blockage_lower_bound for the entire continuous domain.

    goal_y certifies a supported local-Y translation on the chosen side;
    it clears the world-Y access strip for every psi. Physical module X
    bounds enclose the actual convex collision footprints throughout this
    motion. Module Y bounds also enclose the visual OBBs and camera, so
    backgrounds outside this strip cannot obscure their intervening rays.
    These are sufficient geometric certificates, not visual mesh-silhouette
    ratios; the separate startup vision domain must certify those ratios.

    All cells are serializable dictionaries with parameter_bounds,
    yaw_bounds, module_bounds, goal_y, side, blockage_lower_bound and
    camera_x_bounds_over_yaw. The last key is None when camera_eye is None.
    No global shelf translation or runtime domain selection is done here.
    """
    half = float(yaw_half_width)
    if (not all(math.isfinite(value) for value in (target_yaw, blocker_yaw, bx, half, cell_width))
            or not 0. <= half < math.pi/2 or cell_width <= 0.):
        raise ValueError("Role certification needs finite poses, positive cell width and yaw half-width below pi/2.")
    if target == blocker or not 0 <= target < len(geometries) or not 0 <= blocker < len(geometries):
        raise ValueError("Role certification needs distinct valid Target and Blocker IDs.")
    if not 0. <= rules.min_blocked_ratio <= 1.:
        raise ValueError("Role blockage ratio must lie in [0,1].")
    intervals = tuple(tuple(map(float, interval)) for interval in translation_intervals)
    if any(len(interval) != 2 or not all(math.isfinite(v) for v in interval) or interval[0] >= interval[1]
           for interval in intervals):
        raise ValueError("Role translation domains must be finite intervals of positive width.")
    t_collision, b_collision = _collision_polygon(geometries[target], target_yaw), _collision_polygon(geometries[blocker], blocker_yaw)
    # This separating local-X support plane remains valid throughout all
    # local-Y witness translations and every common world rotation.
    if bx+float(b_collision[:, 0].min())-float(t_collision[:, 0].max()) < rules.gap-1.e-12:
        return []
    t_visual = np.asarray(footprint_polygon(geometries[target], (0., 0.), target_yaw))
    b_visual = np.asarray(footprint_polygon(geometries[blocker], (bx, 0.), blocker_yaw))
    target_front = _certified_rotated_bounds(t_visual, half)[1]+rules.approach_margin
    half_access = rules.hand_width*.5+rules.approach_margin
    b_zero_bounds = _certified_rotated_bounds(b_visual, half)
    b_half = (np.asarray(geometries[blocker].upper[:2])-geometries[blocker].lower[:2])*.5
    area = float(4*b_half[0]*b_half[1])
    if area <= 0.:
        raise ValueError("Role certification needs positive Blocker OBB area.")
    cosine, sine = math.cos(blocker_yaw), math.sin(blocker_yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    camera_bounds = None
    if camera_eye is not None:
        eye = np.asarray(camera_eye, dtype=np.float64)
        if eye.ndim != 1 or len(eye) not in (2, 3) or not np.isfinite(eye).all():
            raise ValueError("A role camera must be a finite XY or XYZ point.")
        camera_bounds = _certified_rotated_bounds(eye[None, :2], half)
    cells = []
    for begin, end in intervals:
        edges = np.linspace(begin, end, max(1, math.ceil((end-begin)/cell_width))+1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            middle = (lo+hi)*.5
            initial_vertices = np.concatenate((b_visual+(0., lo), b_visual+(0., hi)))
            radius = float(np.linalg.norm(initial_vertices, axis=-1).max())
            erosion = (hi-lo)*.5+2*radius*math.sin(half*.5)
            inner_half = b_half-erosion
            max_blocker_x = _certified_rotated_bounds(initial_vertices, half)[1]
            if max_blocker_x <= target_front:
                continue
            access = rectangle_polygon((target_front, max_blocker_x+1.e-9, -half_access, half_access))
            bound = 0.
            if (inner_half > 0.).all():
                inner = np.asarray(((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)))*inner_half
                inner = inner @ rotation.T+(bx, middle)
                bound = overlap_area(tuple(map(tuple, inner)), access)/area
            # A second sufficient certificate keeps the baseline rectangle
            # intact and insets the access half-planes instead. A fixed point
            # subset of this clipped rectangle stays in A0 under every cell
            # transform, preserving its area. Thin Blockers need not lose
            # their entire X width to isotropic rectangle erosion.
            baseline = b_visual+(0., middle)
            maximum_abs = np.abs(baseline).max(axis=0)
            dx = maximum_abs[0]*(1-math.cos(half))+(maximum_abs[1]+(hi-lo)*.5)*math.sin(half)
            dy = maximum_abs[0]*math.sin(half)+maximum_abs[1]*(1-math.cos(half))+(hi-lo)*.5
            inset_front = target_front+dx
            if dy < half_access and inset_front < float(baseline[:, 0].max()):
                inset = rectangle_polygon((inset_front, float(baseline[:, 0].max())+1.e-9,
                                           -half_access+dy, half_access-dy))
                bound = max(bound, overlap_area(tuple(map(tuple, baseline)), inset)/area)
            if bound < rules.min_blocked_ratio:
                continue
            side = 1 if middle >= 0. else -1
            clearance = half_access+rules.gap
            goal = max(hi, (clearance-b_zero_bounds[2])/math.cos(half)) if side > 0 else min(
                lo, (-clearance-b_zero_bounds[3])/math.cos(half))
            low_slide, high_slide = min(lo, goal), max(hi, goal)
            collision_vertices = np.concatenate((t_collision, b_collision+(bx, low_slide),
                                                  b_collision+(bx, high_slide)))
            visual_vertices = np.concatenate((t_visual, b_visual+(0., low_slide), b_visual+(0., high_slide)))
            physical_bounds = _certified_rotated_bounds(collision_vertices, half)
            visual_bounds = _certified_rotated_bounds(visual_vertices, half)
            y_min, y_max = visual_bounds[2:]
            if camera_bounds is not None:
                y_min, y_max = min(y_min, camera_bounds[2]), max(y_max, camera_bounds[3])
            cells.append(dict(parameter_bounds=(float(lo), float(hi)), yaw_bounds=(-half, half),
                              module_bounds=(*physical_bounds[:2], float(y_min), float(y_max)),
                              goal_y=float(goal), side=side, blockage_lower_bound=float(bound),
                              camera_x_bounds_over_yaw=None if camera_bounds is None else camera_bounds[:2],
                              collision_module_bounds=physical_bounds, visual_module_bounds=visual_bounds))
    return cells


def _certified_pivot_rotated_bounds(initial, pivot, angle, parameter_bounds, yaw_half_width):
    """Arc bounds over continuous pivot angle, local-Y shift and common yaw.

    For R(psi)[p+(0,s)+R(theta)v], linear s reaches its extrema at its
    endpoints. Angular extrema lie on a boundary or where both angular
    derivatives vanish. These finite algebraic events cover the full motion;
    no sampled poses or outer polygon approximation define this certificate.
    """
    initial, pivot = np.asarray(initial, dtype=np.float64), np.asarray(pivot, dtype=np.float64)
    lo, hi = map(float, parameter_bounds)
    half = float(yaw_half_width)
    if (initial.ndim != 2 or initial.shape[1] != 2 or not len(initial) or pivot.shape != (2,)
            or not np.isfinite(initial).all() or not np.isfinite(pivot).all()
            or not all(math.isfinite(value) for value in (angle, lo, hi, half))
            or lo > hi or not 0. <= half < math.pi/2):
        raise ValueError("Pivot bounds need finite XY vertices, a pivot, ordered parameters and a common yaw below pi/2.")

    def rotate(point, value):
        c, s = math.cos(value), math.sin(value)
        return np.array((c*point[0]-s*point[1], s*point[0]+c*point[1]))

    def stationary(low, high, base):
        return (base+k*math.pi for k in range(math.ceil((low-base-1.e-13)/math.pi),
                                             math.floor((high-base+1.e-13)/math.pi)+1))

    lower, upper = min(0., angle), max(0., angle)
    values = []
    for parameter in (lo, hi):
        p = pivot+(0., parameter)
        phase_p = math.atan2(p[1], p[0])
        for vector in initial-pivot:
            phase_v = math.atan2(vector[1], vector[0])
            pairs = [(psi, theta) for psi in (-half, half) for theta in (lower, upper)]
            for axis in (0., math.pi/2):
                for psi in (-half, half):
                    pairs.extend((psi, theta) for theta in stationary(lower, upper, axis-psi-phase_v))
                for theta in (lower, upper):
                    endpoint = p+rotate(vector, theta)
                    phase = math.atan2(endpoint[1], endpoint[0])
                    pairs.extend((psi, theta) for psi in stationary(-half, half, axis-phase))
                for psi in stationary(-half, half, axis-phase_p):
                    pairs.extend((psi, theta) for theta in stationary(lower, upper, axis-psi-phase_v))
            values.extend(rotate(p+rotate(vector, theta), psi) for psi, theta in pairs)
    values = np.asarray(values)
    # Outward numerical padding is negligible beside the physical gap but
    # prevents a rounded trigonometric extremum from understating its bound.
    return (float(values[:, 0].min()-1.e-12), float(values[:, 0].max()+1.e-12),
            float(values[:, 1].min()-1.e-12), float(values[:, 1].max()+1.e-12))


def _precompute_normal_pivot_family(base_family, geometries, rules):
    """Transform unchanged initial-A0 cells into a normal pivot+slide family.

    Rear support vertices and edge-normal angular events determine the
    minimum final Y width within the front-half-plane rotation limits. One
    fixed pivot/turn recipe is retained for the whole returned family. Exact
    arc/common-yaw bounds and the final local-Y endpoint reserve its motion.
    Background composition, shelf and camera bounds are intersected later;
    this helper neither regenerates layouts nor changes the vision/A0 limits.
    """
    target, blocker = base_family['target_id'], base_family['blocker_id']
    if target == blocker or not 0 <= target < len(geometries) or not 0 <= blocker < len(geometries):
        raise ValueError("A normal pivot family needs distinct valid role IDs.")
    if base_family.get('fixed_objects'):
        raise ValueError("Normal pivot compilation expects an unguarded role family.")
    ty, by, bx = base_family['target_yaw'], base_family['blocker_yaw'], base_family['blocker_x']
    if not all(math.isfinite(value) for value in (ty, by, bx)):
        raise ValueError("Normal pivot role poses must be finite.")
    initial = _collision_polygon(geometries[blocker], by)+(bx, 0.)
    target_collision = _collision_polygon(geometries[target], ty)
    if float(initial[:, 0].min()-target_collision[:, 0].max()) < rules.gap-1.e-12:
        return None

    def rotate(points, angle):
        c, s = math.cos(angle), math.sin(angle)
        return np.asarray(points) @ np.array(((c, s), (-s, c)))

    def visual(index, yaw, center=(0., 0.)):
        geometry = geometries[index]
        if not geometry.visual_footprint_vertices:
            return np.asarray(footprint_polygon(geometry, center, yaw))
        points = np.asarray(geometry.visual_footprint_vertices)-(
            np.asarray(geometry.lower[:2])+geometry.upper[:2])*.5
        return rotate(points, yaw)+center

    def union(*bounds):
        values = np.asarray(bounds)
        return (float(values[:, 0].min()), float(values[:, 1].max()),
                float(values[:, 2].min()), float(values[:, 3].max()))

    t_visual, b_visual = visual(target, ty), visual(blocker, by, (bx, 0.))
    box = np.asarray(footprint_polygon(geometries[blocker], (bx, 0.), by))
    recipes = []
    for pivot in initial[initial[:, 0] <= initial[:, 0].min()+1.e-10]:
        vectors = initial-pivot
        phases = np.arctan2(vectors[:, 1], vectors[:, 0])[np.linalg.norm(vectors, axis=-1) > 1.e-12]
        low, high = -math.pi/2-float(phases.min()), math.pi/2-float(phases.max())
        if low > high+1.e-12:
            continue
        angles = [low, high]
        for edge in np.roll(initial, -1, axis=0)-initial:
            base = -math.atan2(edge[1], edge[0])
            angles.extend(base+k*math.pi for k in range(math.ceil((low-base-1.e-13)/math.pi),
                                                       math.floor((high-base+1.e-13)/math.pi)+1))
        angle = min(angles, key=lambda value: (float(np.ptp(rotate(vectors, value)[:, 1])), abs(value)))
        if abs(angle) < 1.e-7:
            continue
        turned = rotate(initial-pivot, angle)+pivot
        center = rotate(np.array(((bx, 0.),))-pivot, angle)[0]+pivot
        zero_box = rotate(box-pivot, angle)+pivot-(0., center[1])
        cells = []
        for source in base_family['role_cells']:
            lo, hi = source['parameter_bounds']
            half = max(abs(value) for value in source['yaw_bounds'])
            if (lo >= hi or not 0. < half < math.pi/2 or source['side'] not in (-1, 1)
                    or source['blockage_lower_bound'] < rules.min_blocked_ratio-1.e-12):
                continue
            if not any(vlo <= lo+1.e-12 and hi <= vhi+1.e-12
                       for vlo, vhi in base_family['visibility_domain']['valid_intervals']):
                raise ValueError("A normal pivot A0 cell exceeds the certified visibility domain.")
            box_bounds = _certified_rotated_bounds(zero_box, half)
            clearance = rules.hand_width*.5+rules.approach_margin+rules.gap
            goal = max(center[1]+hi, (clearance-box_bounds[2])/math.cos(half)) if source['side'] > 0 else min(
                center[1]+lo, (-clearance-box_bounds[3])/math.cos(half))
            final = turned+(0., goal-center[1])
            final_visual = rotate(b_visual-pivot, angle)+pivot+(0., goal-center[1])
            physical = union(_certified_rotated_bounds(target_collision, half),
                _certified_pivot_rotated_bounds(initial, pivot, angle, (lo, hi), half),
                _certified_rotated_bounds(final, half))
            visible = union(_certified_rotated_bounds(t_visual, half),
                _certified_pivot_rotated_bounds(b_visual, pivot, angle, (lo, hi), half),
                _certified_rotated_bounds(final_visual, half))
            camera = source['camera_x_bounds_over_yaw']
            if camera is not None:
                camera_bounds = _certified_rotated_bounds(np.asarray(base_family['camera_pose'])[None, :2], half)
                visible = union(visible, camera_bounds)
            bounds = (*physical[:2], min(physical[2], visible[2]), max(physical[3], visible[3]))
            cells.append(dict(source, module_bounds=bounds, collision_module_bounds=physical,
                visual_module_bounds=visible, goal_x=float(center[0]), goal_y=float(goal)))
        if cells:
            metadata = dict(pivot_offset_from_blocker=tuple(map(float, pivot-(bx, 0.))),
                turn_offset_from_blocker=tuple(map(float, center-(bx, 0.))), turn_angle=float(angle))
            recipes.append((min(cell['module_bounds'][3]-cell['module_bounds'][2] for cell in cells), cells, metadata))
    if not recipes:
        return None
    _, cells, metadata = min(recipes, key=lambda item: item[0])
    return dict(base_family, role_cells=cells, pivot=metadata, kind='normal_pivot')


def _certified_pivot_envelope(initial, pivot, angle, final, normals_count=256):
    """Outer convex motion envelope from exact arc supports at fixed normals."""
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    phi = np.arange(normals_count)*2*math.pi/normals_count
    normals = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    vectors = initial-pivot
    radii = np.linalg.norm(vectors, axis=-1)
    phases = np.arctan2(vectors[:, 1], vectors[:, 0])
    low, high = min(0., angle), max(0., angle)
    critical = (phi[None]-phases[:, None]-low) % (2*math.pi)+low
    endpoints = np.maximum(np.cos(phases[:, None]-phi), np.cos(phases[:, None]+angle-phi))
    support = (radii[:, None]*np.where(critical <= high+1.e-13, 1., endpoints)).max(0)+normals@pivot
    support = np.maximum(support, (final@normals.T).max(0))+1.e-11
    vertices = HalfspaceIntersection(np.c_[normals, -support], initial.mean(0)).intersections
    return vertices[ConvexHull(vertices).vertices]


def _precompute_guarded_role_family(base_family, geometries, rules, guard_id, *, role_cell=None,
                                    guard_yaw=0., parameter_half_width=.001,
                                    yaw_half_width=math.radians(.25), edge_slack_bounds=(.0005, .0015),
                                    camera_front_range=(.55, .95)):
    """Certify one pivot-required continuous family during startup only.

    Inputs describe a normal inverse-mesh role family, not a saved scene.
    The selected positive-s cell is narrowed once. One guard support-X lies
    between the initial and pivoted Blocker fronts; its Y is determined by
    the exact motion-envelope boundary and initial visual cone. There are
    no guard pose candidates or reset calls here. Return None when these
    sufficient geometric inequalities cannot certify this family.

    Positive world-Y clearance is blocked by a strict guard configuration
    obstacle before A0 clears. Negative world-Y clearance violates full
    footprint support at the actual shelf edge before A0 clears. Arbitrary
    XY translations are not ruled out. The supported pivot+slide witness,
    fixed companion, camera protection and continuous s/common-yaw domain
    are certified together; the caller still intersects background capacity
    and composition in prepare_certified_scene_domain.
    """
    from scipy.spatial import ConvexHull, QhullError

    target, blocker = base_family['target_id'], base_family['blocker_id']
    if (guard_id in (target, blocker) or not 0 <= guard_id < len(geometries)
            or _asset_type(geometries[guard_id]) == _asset_type(geometries[target])):
        raise ValueError("A guarded family needs a distinct non-Target companion from the spawned pool.")
    if (not all(math.isfinite(value) for value in (guard_yaw, parameter_half_width, yaw_half_width,
                                                  *edge_slack_bounds, *camera_front_range))
            or parameter_half_width <= 0. or not 0. < yaw_half_width < math.pi/2
            or not 0. <= edge_slack_bounds[0] < edge_slack_bounds[1]
            or not 0. < camera_front_range[0] < camera_front_range[1]):
        raise ValueError("Guarded family parameters need finite positive continuous ranges.")
    if role_cell is None:
        role_cell = next((cell for cell in base_family['role_cells'] if cell['parameter_bounds'][0] >= 0.), None)
    if role_cell is None or role_cell['parameter_bounds'][0] < 0.:
        return None
    lo, hi = role_cell['parameter_bounds']
    s0, s_half = (lo+hi)*.5, min(parameter_half_width, (hi-lo)*.5)
    yaw_low, yaw_high = role_cell['yaw_bounds']
    half = min(yaw_half_width, -yaw_low, yaw_high)
    if s_half <= 0. or half <= 0.:
        return None
    s_bounds = (float(s0-s_half), float(s0+s_half))
    if not any(vlo <= s_bounds[0]+1.e-12 and s_bounds[1] <= vhi+1.e-12
               for vlo, vhi in base_family['visibility_domain']['valid_intervals']):
        return None
    ty, by, bx = base_family['target_yaw'], base_family['blocker_yaw'], base_family['blocker_x']
    camera = np.asarray(base_family['camera_pose'], dtype=np.float64)
    certified = _certify_role_translation_cells(geometries, target, blocker, ty, by, bx,
        (s_bounds,), rules, yaw_half_width=half, camera_eye=camera[:3], cell_width=2*s_half+1.e-12)
    if not certified or any(geometries[i].upper[2]-geometries[i].lower[2] >
                            rules.ceiling-rules.shelf_top-rules.gap for i in (target, blocker, guard_id)):
        return None
    initial = _collision_polygon(geometries[blocker], by)+(bx, s0)
    t_shape = _collision_polygon(geometries[target], ty)
    b_box = np.asarray(footprint_polygon(geometries[blocker], (bx, s0), by))
    t_box = np.asarray(footprint_polygon(geometries[target], (0., 0.), ty))
    pivot, turn_angle, turned_center, turned_shape = _constructive_pivot(initial, np.array((bx, s0)))
    guard_gap = (rules.gap+rules.max_neighbor_gap)*.5
    if float(initial[:, 0].max()-turned_shape[:, 0].max()) <= 2*guard_gap:
        return None
    final_yaw = by+turn_angle
    zero_final_box = np.asarray(footprint_polygon(geometries[blocker], (float(turned_center[0]), 0.), final_yaw))
    half_access = rules.hand_width*.5+rules.approach_margin
    goal_y = max(float(turned_center[1]), (half_access+rules.gap-
        _certified_rotated_bounds(zero_final_box, half)[2])/math.cos(half)+s_half)
    final_center = np.array((turned_center[0], goal_y))
    final_shape = _collision_polygon(geometries[blocker], final_yaw)+final_center
    try:
        envelope = _certified_pivot_envelope(initial, pivot, turn_angle, final_shape)
    except QhullError:
        return None

    def visual_polygon(index, yaw, center=(0., 0.)):
        geometry = geometries[index]
        if not geometry.visual_footprint_vertices:
            return np.asarray(footprint_polygon(geometry, center, yaw))
        points = np.asarray(geometry.visual_footprint_vertices)-(
            np.asarray(geometry.lower[:2])+geometry.upper[:2])*.5
        c, s = math.cos(yaw), math.sin(yaw)
        return points@np.array(((c, s), (-s, c)))+center

    t_visual, b_visual = visual_polygon(target, ty), visual_polygon(blocker, by, (bx, s0))
    guard_local, guard_visual = _collision_polygon(geometries[guard_id], guard_yaw), visual_polygon(guard_id, guard_yaw)
    left = float(initial[:, 0].max()+turned_shape[:, 0].max()+guard_gap)*.5
    gx = left-float(guard_local[:, 0].min())
    boundary = footprint_line_limit(guard_local, envelope, gx, guard_gap)
    if boundary is None:
        return None
    initial_strip = _certified_rotated_bounds(np.concatenate((t_visual, b_visual+(0., -s_half),
        b_visual+(0., s_half), camera[None, :2])), half)
    strip_padding = s_half+2*float(np.linalg.norm(guard_visual, axis=-1).max())*math.sin(half*.5)
    gy = max(float(boundary), initial_strip[3]-float(guard_visual[:, 1].min())+rules.gap+strip_padding)
    guard_center = np.array((gx, gy))
    guard_shape, guard_visible = guard_local+guard_center, guard_visual+guard_center
    visual_separation = _certified_rotated_bounds(guard_visible, half)[2]-initial_strip[3]
    path_gap = polygon_distance(envelope, guard_shape)-s_half
    if visual_separation <= 0. or path_gap < rules.gap:
        return None

    def access_lower(delta_y):
        maximum = np.abs(b_box).max(0)
        dx = maximum[0]*(1-math.cos(half))+(maximum[1]+s_half)*math.sin(half)
        dy = maximum[0]*math.sin(half)+maximum[1]*(1-math.cos(half))+s_half
        front = _certified_rotated_bounds(t_box, half)[1]+rules.approach_margin+dx
        width_y = half_access-dy
        if width_y <= 0. or front >= float(b_box[:, 0].max()):
            return 0.
        region = rectangle_polygon((front, float(b_box[:, 0].max())+1.e-9, -width_y, width_y))
        return overlap_area(tuple(map(tuple, b_box+(0., delta_y))), region)

    upper = footprint_line_limit(initial, guard_shape, 0., 0.)
    lower_mirrored = footprint_line_limit(initial*(1., -1.), guard_shape*(1., -1.), 0., 0.)
    access = np.asarray(rectangle_polygon((float(t_box[:, 0].max())+rules.approach_margin,
        float(b_box[:, 0].max())+.01, -half_access, half_access)))
    access_upper = footprint_line_limit(b_box, access, 0., 0.)
    if upper is None or lower_mirrored is None or access_upper is None:
        return None
    low_probe, high_probe = max(0., -lower_mirrored), min(upper, access_upper)
    if low_probe >= high_probe:
        return None
    up_probe = (low_probe+high_probe)*.5
    configuration = (guard_shape[:, None]-initial[None]).reshape(-1, 2)
    try:
        faces = ConvexHull(configuration).equations
    except QhullError:
        return None
    up_margin = min(float(-offset-_certified_rotated_bounds(np.array(((nx*up_probe, ny*up_probe),)), half)[3]
                          -abs(ny)*s_half) for nx, ny, offset in faces)
    up_area = access_lower(up_probe)
    if up_margin <= 0. or up_area <= 1.e-12:
        return None
    physical_points = np.concatenate((t_shape, envelope+(0., -s_half), envelope+(0., s_half), guard_shape))
    physical_bounds = _certified_rotated_bounds(physical_points, half)
    bottom = initial[int(initial[:, 1].argmin())]
    bottom_upper = _certified_rotated_bounds(np.array((bottom+(0., -s_half), bottom+(0., s_half))), half)[3]
    down_probe = physical_bounds[2]-bottom_upper-edge_slack_bounds[1]-rules.edge_margin-.001
    down_area = access_lower(down_probe)
    if down_area <= 1.e-12:
        return None
    final_box = np.asarray(footprint_polygon(geometries[blocker], tuple(final_center), final_yaw))
    try:
        visual_envelope = _certified_pivot_envelope(b_box, pivot, turn_angle, final_box)
    except QhullError:
        return None
    visual_bounds = _certified_rotated_bounds(np.concatenate((t_visual, visual_envelope+(0., -s_half),
        visual_envelope+(0., s_half), guard_visible, camera[None, :2])), half)
    bounds = (*physical_bounds[:2], *visual_bounds[2:])
    camera_bounds = _certified_rotated_bounds(camera[None, :2], half)
    xmin, xmax = rules.x_range[0]+rules.edge_margin, rules.x_range[1]-rules.edge_margin
    ymin = rules.y_range[0]+rules.edge_margin
    x_bounds = (max(xmin-physical_bounds[0], rules.x_range[1]+camera_front_range[0]-camera_bounds[0]),
                min(xmax-physical_bounds[1], rules.x_range[1]+camera_front_range[1]-camera_bounds[1]))
    if x_bounds[0] >= x_bounds[1]:
        return None
    cell = dict(certified[0], module_bounds=bounds, collision_module_bounds=physical_bounds,
        visual_module_bounds=visual_bounds, goal_x=float(final_center[0]), goal_y=float(goal_y), side=1,
        camera_x_bounds_over_yaw=camera_bounds[:2], translation_x_bounds=x_bounds,
        translation_y_bounds=(ymin+edge_slack_bounds[0]-physical_bounds[2],
                              ymin+edge_slack_bounds[1]-physical_bounds[2]))
    metadata = dict(pivot_offset_from_blocker=tuple(map(float, pivot-(bx, s0))),
        turn_offset_from_blocker=tuple(map(float, turned_center-(bx, s0))), turn_angle=float(turn_angle))
    return dict(base_family, role_cells=[cell], fixed_objects={int(guard_id):(float(gx), float(gy), float(guard_yaw))},
        pivot=metadata, yaw_half_width=float(half), kind='pivot_guarded_partial', guard_certificate=dict(
            guard_id=int(guard_id), initial_visual_separation=float(visual_separation),
            witness_gap_lower_bound=float(path_gap), pure_up_probe=float(up_probe),
            pure_up_configuration_margin=float(up_margin), pure_up_remaining_access_area=float(up_area),
            pure_down_probe=float(down_probe), pure_down_shelf_exit_lower_bound=.001,
            pure_down_remaining_access_area=float(down_area),
            support_convention='Full collision footprint within actual shelf y_range; no general XY impossibility claim.'))


def _constructive_lateral_guards(sample, geometries, rules):
    """Place up to two existing bodies around a certified pivot exit.

    Each non-role body contributes one analytic upper and lower placement;
    those finite domains are evaluated once. There are no pose samples, random
    redraws or until-valid loops. Return a copied positions mapping and the
    moved pool IDs. An empty ID tuple means that no guarded domain exists.

    A guard's inner X face stays ahead of the rotated Blocker's translation
    strip, while overlapping its initial X extent. Its Y comes from the exact
    configuration-space boundary of an outer pivot/translation envelope. The
    envelope uses exact circular-arc support at 128 fixed normals, not sampled
    physics poses. Both pure +/-Y exits at the initial X/yaw must be blocked
    even with zero clearance margin. General XY translations are not excluded.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    positions = dict(sample.positions)
    path = sample.clearance_path
    if len(path) != 2 or path[0].pivot is None or abs(path[1].end.yaw - path[1].start.yaw) > 1.e-10:
        return positions, ()
    blocker = sample.blocker_id
    moving = _collision_polygon(geometries[blocker], sample.yaws[blocker])
    center = np.asarray(positions[blocker][:2])
    world = {index: _collision_polygon(geometries[index], sample.yaws[index]) + positions[index][:2]
             for index in sample.active_ids}
    turn = path[0].end
    rotated = _collision_polygon(geometries[blocker], turn.yaw) + (turn.x, turn.y)
    rotated_front, initial_front = float(rotated[:, 0].max()), float(world[blocker][:, 0].max())
    if initial_front - rotated_front <= rules.gap * 4.0 / 3.0:
        return positions, ()

    phi = np.arange(128) * (2.0 * math.pi / 128)
    normals = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    pivot = np.asarray(path[0].pivot)
    vectors = world[blocker] - pivot
    radii = np.linalg.norm(vectors, axis=-1)
    phases = np.arctan2(vectors[:, 1], vectors[:, 0])
    angle = turn.yaw - path[0].start.yaw
    low_angle, high_angle = min(0., angle), max(0., angle)
    critical = (phi[None, :] - phases[:, None] - low_angle) % (2.0 * math.pi) + low_angle
    endpoint_cos = np.maximum(np.cos(phases[:, None] - phi),
                              np.cos(phases[:, None] + angle - phi))
    support = (radii[:, None] * np.where(critical <= high_angle + 1.e-13, 1., endpoint_cos)).max(0)
    support += normals @ pivot
    finish = path[-1].end
    final = _collision_polygon(geometries[blocker], finish.yaw) + (finish.x, finish.y)
    support = np.maximum(support, (final @ normals.T).max(0)) + 1.e-12
    intersection = HalfspaceIntersection(np.column_stack((normals, -support)), world[blocker].mean(0))
    envelope = intersection.intersections[ConvexHull(intersection.intersections).vertices]

    def forbidden(polygon):
        upper = footprint_line_limit(moving, polygon, float(center[0]), 0.)
        if upper is None:
            return None
        lower = -footprint_line_limit(moving * (1., -1.), polygon * (1., -1.), float(center[0]), 0.)
        return lower, upper

    def distance(first, second):
        delta = np.maximum(0., np.maximum(first.min(0) - second.max(0), second.min(0) - first.max(0)))
        bound = float(np.linalg.norm(delta))
        if bound > rules.max_neighbor_gap + 1.e-8:
            return bound
        return polygon_distance(tuple(map(tuple, first)), tuple(map(tuple, second)))

    intervals = {index: forbidden(polygon) for index, polygon in world.items() if index != blocker}
    access = forbidden(np.asarray(rectangle_polygon(sample.access_region)))
    if access is None:
        return positions, ()
    shelf_low = rules.y_range[0] + rules.edge_margin - moving[:, 1].min()
    shelf_high = rules.y_range[1] - rules.edge_margin - moving[:, 1].max()

    def blocked(replacements):
        lower, upper = shelf_low, shelf_high
        current = dict(intervals)
        current.update({item[0]: item[3] for item in replacements})
        for interval in current.values():
            if interval is None:
                continue
            if interval[1] <= center[1] + 1.e-9:
                lower = max(lower, interval[1])
            elif interval[0] >= center[1] - 1.e-9:
                upper = min(upper, interval[0])
            else:
                return False
        return lower > access[0] + 1.e-8 and upper < access[1] - 1.e-8

    candidates = []
    for index in sample.active_ids:
        if index in (sample.target_id, blocker):
            continue
        local = _collision_polygon(geometries[index], sample.yaws[index])
        left = min(initial_front - rules.gap / 3.0,
                   rules.x_range[1] - rules.edge_margin - float(np.ptp(local[:, 0])))
        if left <= rotated_front + rules.gap:
            continue
        x = left - float(local[:, 0].min())
        for side in (-1., 1.):
            limit = footprint_line_limit(local * (1., side), envelope * (1., side), x, rules.gap + 1.e-10)
            if limit is None:
                continue
            xy = np.array((x, side * limit))
            polygon = local + xy
            if (polygon[:, 1].min() < rules.y_range[0] + rules.edge_margin
                    or polygon[:, 1].max() > rules.y_range[1] - rules.edge_margin):
                continue
            invalid = {other for other, obstacle in world.items() if other != index
                       and distance(polygon, obstacle) < rules.gap - 1.e-8}
            candidates.append((index, polygon, xy, forbidden(polygon), invalid,
                               distance(polygon, world[blocker]), side))
    domains = []
    for first in candidates:
        if not first[4] and blocked((first,)):
            domains.append((first,))
        for second in candidates:
            if first[6] >= second[6] or first[0] == second[0]:
                continue
            if first[4] - {second[0]} or second[4] - {first[0]}:
                continue
            if blocked((first, second)) and distance(first[1], second[1]) >= rules.gap - 1.e-8:
                domains.append((first, second))
    if not domains:
        return positions, ()
    selected = min(domains, key=lambda domain: (max(item[5] for item in domain) > rules.max_neighbor_gap + 1.e-8,
                                               len(domain), max(item[5] for item in domain)))
    for item in selected:
        positions[item[0]] = (float(item[2][0]), float(item[2][1]), positions[item[0]][2])
    return positions, tuple(item[0] for item in selected)


def _constructive_composition(geometries, rules, rng, target_id=None, blocker_id=None, active_ids=None):
    """Choose roles and balanced species counts without drawing any poses.

    This is shared by the original constructor and the startup-certified
    catalog consumer. Draw order and explicit-composition errors remain the
    same as the original constructor, including spawned-copy selection.
    """
    _check_inputs(geometries, rules)
    if not rules.gap <= rules.max_neighbor_gap:
        raise ValueError("The minimum footprint gap must not exceed its neighbor upper bound.")
    types, groups, _ = _catalog_layout_data(tuple(geometries))
    target_id = rng.choice(tuple(range(len(geometries)))) if target_id is None else target_id
    if not 0 <= target_id < len(geometries):
        raise ValueError("Target ID is outside the spawned pool.")
    target_type = _asset_type(geometries[target_id])
    choices = sorted(types - {target_type})
    if blocker_id is None:
        blocker_id = rng.choice(groups[rng.choice(choices)])
    if not 0 <= blocker_id < len(geometries) or _asset_type(geometries[blocker_id]) == target_type:
        raise ValueError("Blocker must be a spawned instance of a different asset type.")
    if active_ids is not None:
        ids = tuple(active_ids)
        if len(set(ids)) != len(ids) or target_id not in ids or blocker_id not in ids:
            raise ValueError("Fixed composition needs unique pool IDs and both episode roles.")
    else:
        count = rng.randint(rules.min_objects, rules.max_objects)
        kind_count = len(types) if rules.all_asset_types else rng.randint(
            rules.min_asset_types, min(rules.max_asset_types, count, len(types)))
        blocker_type = _asset_type(geometries[blocker_id])
        selected = [blocker_type, *rng.sample([t for t in choices if t != blocker_type], kind_count - 2)]
        base, remainder = divmod(count - 1, kind_count - 1)
        extra = set(rng.sample(selected, remainder))
        selected_ids = [target_id, blocker_id]
        for asset in selected:
            needed = base + int(asset in extra) - int(asset == blocker_type)
            available = [index for index in groups[asset] if index != blocker_id]
            if needed > len(available):
                raise ValueError(f"Pool lacks {needed} additional copies of {asset}; increase pool capacity.")
            selected_ids.extend(rng.sample(available, needed))
        ids = tuple(selected_ids)
    if not rules.min_objects <= len(ids) <= rules.max_objects:
        raise ValueError("Fixed composition does not match the configured active count.")
    counts = {asset: sum(_asset_type(geometries[index]) == asset for index in ids)
              for asset in {_asset_type(geometries[index]) for index in ids}}
    other_counts = [value for asset, value in counts.items() if asset != target_type]
    if counts.get(target_type) != 1 or max(other_counts) - min(other_counts) > 1:
        raise ValueError("Composition must contain one Target species and balanced non-Target counts.")
    return ids, target_id, blocker_id


def construct_scene(geometries, rules, rng, target_id=None, blocker_id=None, active_ids=None, *, device="cpu"):
    """Construct one scene from geometry and random parameters, never redraw a rejected layout.

    Finite slicing arithmetic chooses a domain before any pose is committed.
    Convex collision footprints are compacted at analytic configuration-space
    boundaries. This host-side work runs for one shared scene, not per clone.
    Exact mesh visibility remains an independently measured sensor invariant.
    """
    ids, target_id, blocker_id = _constructive_composition(
        geometries, rules, rng, target_id, blocker_id, active_ids)
    yaws = {index: rng.uniform(-math.pi, math.pi) for index in ids}
    local = {index: _collision_polygon(geometries[index], yaws[index]) for index in ids}
    target, blocker = target_id, blocker_id
    target_polygon, blocker_polygon = local[target], local[blocker]
    neighbor_gap = rng.uniform(rules.gap, rules.max_neighbor_gap)
    half_access = rules.hand_width * .5 + rules.approach_margin
    # The rear support plane makes the pivot clear of Target for the whole arc.
    bx = float(target_polygon[:, 0].max() - blocker_polygon[:, 0].min() + neighbor_gap)
    modules = [_constructive_task_module(geometries, target, blocker, yaws, local,
                                       bx, side, rules, rng, neighbor_gap) for side in (-1., 1.)]
    xmin, xmax = rules.x_range[0] + rules.edge_margin, rules.x_range[1] - rules.edge_margin
    ymin, ymax = rules.y_range[0] + rules.edge_margin, rules.y_range[1] - rules.edge_margin
    depth, width = xmax - xmin, ymax - ymin
    background = [index for index in ids if index not in (target, blocker)]
    rng.shuffle(background)
    sizes = [(float(np.ptp(local[index][:, 1])), float(np.ptp(local[index][:, 0]))) for index in background]
    packing_gap = rules.max_neighbor_gap
    plans = _constructive_slicing_plans(sizes, packing_gap, depth, width)
    domains = []
    # Both directions are members of one finite parameter domain. No scene is
    # committed, checked and redrawn when one direction cannot fit.
    for module_index, entry in enumerate(modules):
        module = entry[5]
        module_depth, module_width = module[1] - module[0], module[3] - module[2]
        if module_depth <= depth:
            for split in range(len(background) + 1):
                for left in plans[0, split]:
                    for right in plans[split, len(background)]:
                        occupied_width = left[0] + module_width + right[0] + packing_gap * (bool(split) + bool(len(background) - split))
                        if occupied_width <= width:
                            domains.append((module_index, split, left, right, occupied_width))
    if not domains:
        raise RuntimeError(f"Constructive layout domain is empty for N={len(ids)}, roles={target}/{blocker}: "
                           f"task domains={[(round(m[5][1]-m[5][0], 4), round(m[5][3]-m[5][2], 4)) for m in modules]}, "
                           f"shelf={depth:.4f}x{width:.4f}m. "
                           "No random redraw was performed.")
    module_index, split, left, right, occupied_width = rng.choice(domains)
    initial_center, pivot, angle, rotation_center, goal_center, module, camera_eye, camera_xyzw = modules[module_index]
    module_depth, module_width = module[1] - module[0], module[3] - module[2]
    base_y = rng.uniform(ymin, ymax - occupied_width)
    module_y = base_y + left[0] + (packing_gap if split else 0.)
    offset = np.array((rng.uniform(xmin, xmax - module_depth) - module[0], module_y - module[2]))
    centers = {target: offset.copy(), blocker: initial_center + offset}
    reserve = (xmin, xmax, module_y, module_y + module_width)
    for first, end, plan, y0 in ((0, split, left, base_y),
                                  (split, len(background), right, module_y + module_width + (packing_gap if split < len(background) else 0.))):
        if first == end:
            continue
        origin = (rng.uniform(xmin, xmax - plan[1]), y0)
        packed = _constructive_unpack_plan(plan[2], origin, sizes, packing_gap)
        indices = background[first:end]
        locations = np.asarray([np.asarray(packed[k]) - (local[index].min(0) + local[index].max(0)) * .5
                                for k, index in zip(range(first, end), indices)])
        polygons = [local[index] for index in indices]
        # Both directional passes use the same separation metric, so the
        # second pass starts outside every expanded configuration obstacle.
        wing_gap = rng.uniform(rules.gap, rules.max_neighbor_gap)
        gaps = [wing_gap] * len(indices)
        # Two fixed directional passes, not an until-valid relaxation loop.
        swapped, _ = _compact_constructive_positions([p[:, ::-1] for p in polygons], locations[:, ::-1],
                                                       origin[0], gaps)
        locations, _ = _compact_constructive_positions(polygons, swapped[:, ::-1], y0, gaps)
        centers.update(zip(indices, locations))
    positions = {index: (float(centers[index][0]), float(centers[index][1]),
                         rules.shelf_top - geometries[index].lower[2]) for index in ids}
    start = PlanarPose(float(centers[blocker][0]), float(centers[blocker][1]), yaws[blocker])
    turn = PlanarPose(float(rotation_center[0] + offset[0]), float(rotation_center[1] + offset[1]), yaws[blocker] + angle)
    finish = PlanarPose(float(goal_center[0] + offset[0]), float(goal_center[1] + offset[1]), turn.yaw)
    path = (PlanarMotion(start, turn, tuple(map(float, pivot + offset))), PlanarMotion(turn, finish))
    target_half = _half_extents(geometries[target], yaws[target])
    access = (positions[target][0] + target_half[0] + rules.approach_margin, rules.x_range[1],
              positions[target][1] - half_access, positions[target][1] + half_access)
    eye = camera_eye + np.r_[offset, 0.]
    camera = tuple(map(float, np.r_[eye, camera_xyzw[3], camera_xyzw[:3]]))
    sample = SceneSample(ids, target, blocker, positions, yaws, access, (finish.x, finish.y), path,
                         "pivot_translate", reserved_regions=(reserve,), camera_pose=camera)
    positions, guards = _constructive_lateral_guards(sample, geometries, rules)
    base = replace(sample, neighbor_pairs=_constructive_neighbors(local, sample.positions, ids, rules))
    return replace(base, positions=positions,
                   neighbor_pairs=(_constructive_neighbors(local, positions, ids, rules) if guards
                                   else base.neighbor_pairs),
                   clearance_kind="guarded_pivot_translate" if guards else "pivot_translate",
                   # Guards use the exact motion envelope; their occupied space
                   # need not remain outside the earlier rectangular packing strip.
                   reserved_regions=() if guards else (reserve,),
                   visibility_variants=(base,) if guards else ())


def sample_scene(geometries: Sequence[ObjectGeometry], rules: SpawnRules, rng: random.Random,
                 target_id: int | None = None, blocker_id: int | None = None,
                 active_ids: Sequence[int] | None = None) -> SceneSample:
    """Single CPU scene using the same constructive packer as batched resets."""
    if rules.generation_mode in ('parameterized', 'clutter'):
        raise ValueError("Parameterized scenes require a startup-certified catalog; use sample_certified_episode.")
    if rules.generation_mode == "constructive":
        return construct_scene(geometries, rules, rng, target_id, blocker_id, active_ids)
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
    """Construct each seeded layout once, or explicitly run the legacy sampler.

    Constructive requests never redraw a layout. Legacy mode alone retains
    failed-plan resampling and batched candidate/neighbor repair. RNG streams
    remain independent of request order and another environment's completion.
    """
    if not (len(streams) == len(roles) == len(active_ids)) or batch_size < 1:
        raise ValueError("Layout requests need aligned seeds/roles/compositions and a positive batch size.")
    _check_inputs(geometries, rules)
    if rules.generation_mode in ('parameterized', 'clutter'):
        raise ValueError("Parameterized scenes require a startup-certified catalog; use sample_certified_episode.")
    if rules.generation_mode == "constructive":
        return [construct_scene(geometries, rules, rng, *role, active, device=device)
                for rng, role, active in zip(streams, roles, active_ids)]
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
        candidates = _clearance_candidates(geometries[blocker], initial, access_region, rules, rng, packed=defer_paths)
        if defer_paths:
            path = yield _PathRequest(geometries[blocker],candidates,polygons[target])
            if path is not None:
                sweeps = _path_sweeps(geometries[blocker],path,packed=True)
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
        if device_poses is not None:
            heights = {index: rules.shelf_top - geometries[index].lower[2] for index in ids}
            positions = _PoseMapping(positions, ids, device_poses, heights)
            yaws = _PoseMapping(yaws, ids, device_poses, heights, yaw=True)
        else:
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
    cross_separated = (length > 1.e-9) & (distance + _EPS * length >= radius + clearance * length)
    # Relative projections lose significance for almost parallel axes. The
    # eager CPU/dtype oracle uses direct FP64 projections on those axes only.
    near = torch.where(length < 1.e-4)
    if near[0].numel():
        leading = length.shape[:-2]
        rows = near[:-2]
        aa = axes_a.expand(*leading, 3, 3)
        bb = axes_b.expand(*leading, 3, 3)
        cross = torch.linalg.cross(aa[(*rows, near[-2])].double(), bb[(*rows, near[-1])].double())
        norm = cross.norm(dim=-1)
        delta64 = center_b.expand(*leading, 3)[rows].double() - center_a.expand(*leading, 3)[rows].double()
        ha = half_a.expand(*leading, 3)[rows].double()
        hb = half_b.expand(*leading, 3)[rows].double()
        radius64 = (ha * (aa[rows].double() * cross[..., None, :]).sum(-1).abs()).sum(-1)
        radius64 += (hb * (bb[rows].double() * cross[..., None, :]).sum(-1).abs()).sum(-1)
        cross_separated[near] = (norm > 1.e-9) & (
            (delta64 * cross).sum(-1).abs() + _EPS * norm >= radius64 + clearance * norm)
    separated |= cross_separated.any((-2, -1))
    return separated


@wp.func
def _collider_transform(frames: wp.array4d(dtype=wp.float32), owners: wp.array(dtype=wp.int64),
                        offsets: wp.array3d(dtype=wp.float32), row: int, collider: int):
    frame, offset = wp.mat44(0.0), wp.mat44(0.0)
    owner = int(owners[collider])
    for i in range(4):
        for j in range(4):
            frame[i, j] = frames[row, owner, i, j]
            offset[i, j] = offsets[collider, i, j]
    return frame * offset


@wp.func
def _parallel_cross_separated(ca: wp.vec3, ra: wp.mat33, ha: wp.vec3,
                              cb: wp.vec3, rb: wp.mat33, hb: wp.vec3,
                              i: int, j: int, gap: float):
    # The relative-rotation shortcut divides cancellation error by a tiny axis
    # length. Recompute only these axes from the original FP32 geometry in FP64.
    a = wp.vec3d(wp.float64(ra[0, i]), wp.float64(ra[1, i]), wp.float64(ra[2, i]))
    b = wp.vec3d(wp.float64(rb[0, j]), wp.float64(rb[1, j]), wp.float64(rb[2, j]))
    axis = wp.cross(a, b)
    length = wp.length(axis)
    if length <= wp.float64(1.e-9):
        return False
    delta = wp.vec3d(wp.float64(cb[0]) - wp.float64(ca[0]),
                     wp.float64(cb[1]) - wp.float64(ca[1]),
                     wp.float64(cb[2]) - wp.float64(ca[2]))
    radius = wp.float64(0.0)
    for k in range(3):
        aa = wp.vec3d(wp.float64(ra[0, k]), wp.float64(ra[1, k]), wp.float64(ra[2, k]))
        bb = wp.vec3d(wp.float64(rb[0, k]), wp.float64(rb[1, k]), wp.float64(rb[2, k]))
        radius += wp.float64(ha[k]) * wp.abs(wp.dot(aa, axis)) + wp.float64(hb[k]) * wp.abs(wp.dot(bb, axis))
    return wp.abs(wp.dot(delta, axis)) + wp.float64(1.e-9) * length >= radius + wp.float64(gap) * length


@wp.func
def _sat_separated(ca: wp.vec3, ra: wp.mat33, ha: wp.vec3,
                   cb: wp.vec3, rb: wp.mat33, hb: wp.vec3, gap: float):
    # Keep the eager oracle's EPS, world-axis shortcut and cross-axis length.
    delta = cb - ca
    for axis in range(3):
        a, b = float(0.0), float(0.0)
        for k in range(3):
            a += wp.abs(ra[axis, k]) * ha[k]
            b += wp.abs(rb[axis, k]) * hb[k]
        if wp.abs(delta[axis]) + 1.e-9 >= a + b + gap:
            return True
    relative = wp.transpose(ra) * rb
    translation = wp.transpose(ra) * delta
    for axis in range(3):
        a, b = float(0.0), float(0.0)
        for k in range(3):
            a += wp.abs(relative[axis, k]) * hb[k]
            b += wp.abs(relative[k, axis]) * ha[k]
        if wp.abs(translation[axis]) + 1.e-9 >= ha[axis] + a + gap:
            return True
        projection = float(0.0)
        for k in range(3):
            projection += relative[k, axis] * translation[k]
        if wp.abs(projection) + 1.e-9 >= hb[axis] + b + gap:
            return True
    for i in range(3):
        i1, i2 = (i + 1) % 3, (i + 2) % 3
        aa = wp.vec3(ra[0, i], ra[1, i], ra[2, i])
        for j in range(3):
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            bb = wp.vec3(rb[0, j], rb[1, j], rb[2, j])
            length = wp.length(wp.cross(aa, bb))
            if length < 1.e-4:
                if _parallel_cross_separated(ca, ra, ha, cb, rb, hb, i, j, gap):
                    return True
            else:
                distance = wp.abs(translation[i2] * relative[i1, j] - translation[i1] * relative[i2, j])
                radius = (ha[i1] * wp.abs(relative[i2, j]) + ha[i2] * wp.abs(relative[i1, j])
                          + hb[j1] * wp.abs(relative[i, j2]) + hb[j2] * wp.abs(relative[i, j1]))
                if distance + 1.e-9 * length >= radius + gap * length:
                    return True
    return False


@wp.kernel(enable_backward=False)
def _robot_clearance_kernel(
    frames: wp.array4d(dtype=wp.float32), joints: wp.array4d(dtype=wp.float32),
    owners: wp.array(dtype=wp.int64), offsets: wp.array3d(dtype=wp.float32),
    half: wp.array2d(dtype=wp.float32), pairs: wp.array2d(dtype=wp.int32),
    pair_counts: wp.array(dtype=wp.int32), centers: wp.array3d(dtype=wp.float32),
    rotations: wp.array4d(dtype=wp.float32), obstacle_half: wp.array3d(dtype=wp.float32),
    obstacle_valid: wp.array2d(dtype=wp.bool), candidate_valid: wp.array(dtype=wp.bool),
    clearance: float, min_height: float, min_forward: float,
    valid: wp.array2d(dtype=wp.bool), heights: wp.array(dtype=wp.float32),
):
    row, collider = wp.tid()
    shoulder = wp.vec3(joints[row, 1, 0, 3], joints[row, 1, 1, 3], joints[row, 1, 2, 3])
    elbow = wp.vec3(joints[row, 2, 0, 3], joints[row, 2, 1, 3], joints[row, 2, 2, 3])
    wrist = wp.vec3(joints[row, 3, 0, 3], joints[row, 3, 1, 3], joints[row, 3, 2, 3])
    chord = wrist - shoulder
    fraction = wp.dot(elbow - shoulder, chord) / wp.max(wp.dot(chord, chord), 1.e-18)
    height = elbow[2] - shoulder[2] - wp.clamp(fraction, 0.0, 1.0) * chord[2]
    if collider == 0:
        heights[row] = height
    valid[row, collider] = False
    if not candidate_valid[row] or not (height >= min_height and elbow[0] - wrist[0] >= min_forward):
        return
    transform = _collider_transform(frames, owners, offsets, row, collider)
    for i in range(4):
        for j in range(4):
            if not wp.isfinite(transform[i, j]):
                return
    center = wp.vec3(transform[0, 3], transform[1, 3], transform[2, 3])
    rotation = wp.mat33(transform[0, 0], transform[0, 1], transform[0, 2],
                        transform[1, 0], transform[1, 1], transform[1, 2],
                        transform[2, 0], transform[2, 1], transform[2, 2])
    extent = wp.vec3(half[collider, 0], half[collider, 1], half[collider, 2])
    vertical_radius = float(0.0)
    for axis in range(3):
        vertical_radius += wp.abs(rotation[2, axis]) * extent[axis]
    bottom = center[2] - vertical_radius
    if not (bottom >= clearance):
        return
    for obstacle in range(centers.shape[1]):
        if obstacle_valid[row, obstacle]:
            other_center = wp.vec3(centers[row, obstacle, 0], centers[row, obstacle, 1], centers[row, obstacle, 2])
            other_rotation = wp.mat33(0.0)
            for i in range(3):
                for j in range(3):
                    other_rotation[i, j] = rotations[row, obstacle, i, j]
            other_half = wp.vec3(obstacle_half[row, obstacle, 0], obstacle_half[row, obstacle, 1],
                                 obstacle_half[row, obstacle, 2])
            if not _sat_separated(center, rotation, extent, other_center, other_rotation, other_half, clearance):
                return
    # Each self-pair belongs to its first collider, without a global pair scan.
    for pair in range(pair_counts[collider]):
        other = pairs[collider, pair]
        other_transform = _collider_transform(frames, owners, offsets, row, other)
        other_center = wp.vec3(other_transform[0, 3], other_transform[1, 3], other_transform[2, 3])
        other_rotation = wp.mat33(other_transform[0, 0], other_transform[0, 1], other_transform[0, 2],
                                  other_transform[1, 0], other_transform[1, 1], other_transform[1, 2],
                                  other_transform[2, 0], other_transform[2, 1], other_transform[2, 2])
        other_half = wp.vec3(half[other, 0], half[other, 1], half[other, 2])
        if not _sat_separated(center, rotation, extent, other_center, other_rotation, other_half, 0.002):
            return
    valid[row, collider] = True


def _robot_candidate_clearance(env, frames, joints, tcp, obstacle_centers, obstacle_rotations, obstacle_half,
                               obstacle_valid, candidate_valid=None):
    """Fused FP32 CUDA clearance; eager tensors remain the CPU/dtype oracle."""
    if frames.is_cuda and frames.dtype == torch.float32:
        model, rules = env.robot_pose_model, env.cfg.robot_start
        constants = model._tensor_constants(frames.device, frames.dtype)
        if "clearance_pairs" not in constants:
            owned = [[] for _ in model.colliders]
            for first, second in model.self_collision_pairs():
                owned[first].append(second)
            width = max((len(row) for row in owned), default=0)
            pairs = np.full((len(owned), max(1, width)), -1, dtype=np.int32)
            for index, row in enumerate(owned):
                pairs[index, :len(row)] = row
            constants["clearance_pairs"] = torch.as_tensor(pairs, device=frames.device)
            constants["clearance_pair_counts"] = torch.tensor([len(row) for row in owned],
                                                              device=frames.device, dtype=torch.int32)
        if candidate_valid is None:
            candidate_valid = torch.ones(frames.shape[0], device=frames.device, dtype=torch.bool)
        output = torch.empty((frames.shape[0], len(model.colliders)), device=frames.device, dtype=torch.bool)
        height = frames.new_empty(frames.shape[0])
        wp.launch(_robot_clearance_kernel, dim=output.shape, inputs=[
            wp.from_torch(frames), wp.from_torch(joints), wp.from_torch(constants["collider_owners"]),
            wp.from_torch(constants["collider_offsets"]), wp.from_torch(constants["collider_half"]),
            wp.from_torch(constants["clearance_pairs"]), wp.from_torch(constants["clearance_pair_counts"]),
            wp.from_torch(obstacle_centers), wp.from_torch(obstacle_rotations), wp.from_torch(obstacle_half),
            wp.from_torch(obstacle_valid), wp.from_torch(candidate_valid), rules.collision_clearance,
            rules.min_elbow_height, rules.min_forearm_forward, wp.from_torch(output), wp.from_torch(height)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(frames.device)))
        return output.all(-1), height
    valid, height = _robot_candidate_clearance_eager(
        env, frames, joints, tcp, obstacle_centers, obstacle_rotations, obstacle_half, obstacle_valid)
    return (valid if candidate_valid is None else valid & candidate_valid), height


def _robot_candidate_clearance_eager(env, frames, joints, tcp, obstacle_centers, obstacle_rotations, obstacle_half,
                                     obstacle_valid, *, details=None):
    """Evaluate all authored colliders and high-elbow geometry on the device."""
    model, rules = env.robot_pose_model, env.cfg.robot_start
    constants = model._tensor_constants(frames.device, frames.dtype)
    colliders = frames[:, constants["collider_owners"]] @ constants["collider_offsets"]
    centers, rotations = colliders[..., :3, 3], colliders[..., :3, :3]
    half = constants["collider_half"]
    bottom = centers[..., 2] - (rotations[..., 2, :].abs() * half).sum(-1)
    ground_clear = (bottom >= rules.collision_clearance).all(-1)
    separated = oriented_boxes_separated_batch(
        centers[:, :, None], rotations[:, :, None], half[None, :, None],
        obstacle_centers[:, None], obstacle_rotations[:, None], obstacle_half[:, None], rules.collision_clearance)
    scene_clear = (separated | ~obstacle_valid[:, None]).all((-2, -1))
    valid = ground_clear & scene_clear
    pairs = constants["self_pairs"]
    if pairs.shape[0]:
        self_clear = oriented_boxes_separated_batch(
            centers[:, pairs[:, 0]], rotations[:, pairs[:, 0]], half[pairs[:, 0]],
            centers[:, pairs[:, 1]], rotations[:, pairs[:, 1]], half[pairs[:, 1]], .002).all(-1)
        valid &= self_clear
    elif details is not None:
        self_clear = torch.ones_like(valid)
    shoulder, elbow, wrist = joints[:, 1, :3, 3], joints[:, 2, :3, 3], joints[:, 3, :3, 3]
    chord = wrist - shoulder
    fraction = ((elbow - shoulder) * chord).sum(-1) / chord.square().sum(-1).clamp_min(1.e-18)
    height = elbow[:, 2] - shoulder[:, 2] - fraction.clamp(0, 1) * chord[:, 2]
    finite = torch.isfinite(colliders).all((-3, -2, -1))
    elbow_up = height >= rules.min_elbow_height
    forearm_forward = elbow[:, 0] - wrist[:, 0] >= rules.min_forearm_forward
    valid &= elbow_up & finite & forearm_forward
    if details is not None:
        details.update(finite=finite, elbow_up=elbow_up, forearm_forward=forearm_forward,
                       ground_clear=ground_clear, scene_clear=scene_clear, self_clear=self_clear)
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


def _robot_start_regions(blocker_pose, half, spawn_rules, rules):
    """Analytic rotated-box front bound for the entire reset batch on device."""
    front = blocker_pose[:, 0] + blocker_pose[:, 3].cos().abs() * half[:, 0] + blocker_pose[:, 3].sin().abs() * half[:, 1]
    near = (front + rules.front_distance_range[0]).clamp_min(spawn_rules.x_range[1] + rules.shelf_front_clearance)
    return torch.stack((near, front + rules.front_distance_range[1],
        blocker_pose[:, 1] - rules.lateral_half_width, blocker_pose[:, 1] + rules.lateral_half_width,
        torch.full_like(front, spawn_rules.shelf_top + rules.height_above_shelf[0]),
        torch.full_like(front, spawn_rules.shelf_top + rules.height_above_shelf[1])), -1)


@wp.kernel(enable_backward=False)
def _robot_start_noise(seeds: wp.array(dtype=wp.int64), noise: wp.array3d(dtype=wp.float32)):
    row, attempt = wp.tid()
    state = wp.rand_init(int(seeds[row] & wp.int64(4294967295)),
                         attempt ^ int(seeds[row] >> wp.int64(32)) ^ 1563339036)
    for component in range(12):
        noise[row, attempt, component] = wp.randf(state)


def _robot_start_singularity_settings(env):
    """Bind the same enabled native failure and numeric parameters as the MDP.

    Configuration is host metadata, read once before candidate tensor work or
    CUDA capture. Standalone IK callers without native termination config keep
    their original behavior. The resulting tuple also fingerprints both reset
    caches, whose captured Python constants must not survive a config change.
    """
    mdp_cfg = getattr(env.cfg, 'mdp', None)
    termination_cfg = getattr(env.cfg, 'terminations', None)
    if mdp_cfg is None or termination_cfg is None:
        return None
    enabled, _ = configured_terminations(termination_cfg, device='cpu')
    if not bool(enabled[FAILURE_REASON_NAMES.index('singularity')]):
        return None
    body_name = env.cfg.actions.arm_action.body_name
    try:
        body_index = env.robot_pose_model.body_names.index(body_name)
    except ValueError as error:
        raise ValueError(f"Controlled arm body {body_name!r} is absent from reset kinematics.") from error
    model = env.robot_pose_model
    if not hasattr(model, '_native_body_com_offsets'):
        robot = env.scene['robot']
        # Authored COMs are fixed in this task. Read native metadata once,
        # before CUDA capture, rather than querying its CPU property at reset.
        local = torch.as_tensor(robot.root_physx_view.get_coms())[0, :, :3].detach().cpu().tolist()
        model._native_body_com_offsets = {name: tuple(map(float, value))
                                          for name, value in zip(robot.body_names, local)}
    com = model._native_body_com_offsets[body_name]
    if not all(math.isfinite(value) for value in com):
        raise ValueError(f"Controlled arm body {body_name!r} has a non-finite native COM.")
    return (body_name, body_index, float(mdp_cfg.singularity_characteristic_length),
            float(mdp_cfg.singularity_termination_threshold), com)


def _robot_start_nonsingular(model, frames, joints, settings):
    """Evaluate the native controlled-link Jacobian at candidate FK poses.

    PhysX's public Jacobian maps velocities at the controlled link COM. Match
    that point, rather than the link origin or the palm used by reset IK.
    Static native COM metadata is cached before CUDA capture; candidate FK
    transforms it using the same authored body ownership/offsets and axes.
    """
    _, body_index, length, threshold, com = settings
    constants = model._tensor_constants(frames.device, frames.dtype)
    owner = constants['body_owners'][body_index:body_index+1]
    tool = frames.index_select(1, owner).squeeze(1) @ constants['body_offsets'][body_index]
    offsets = constants.setdefault('singularity_com_offsets', {})
    if com not in offsets:
        offsets[com] = frames.new_tensor(com)
    point = tool[:, :3, 3] + (tool[:, :3, :3] @ offsets[com][:, None]).squeeze(-1)
    axes = (joints[:, :, :3, :3] @ constants['axes'][None, :, :, None]).squeeze(-1)
    linear = torch.linalg.cross(axes, point[:, None]-joints[:, :, :3, 3])
    jacobian = torch.cat((linear, axes), -1).transpose(1, 2)
    margin = singularity_margin(jacobian, characteristic_length=length)
    return torch.isfinite(margin) & (margin > threshold)


@contextmanager
def _robot_kinematics_precision():
    """Use FP32 matmul for geometric tolerances, then restore policy settings."""
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@torch.no_grad()
@_robot_kinematics_precision()
def _sample_robot_starts(env, samples, seeds, *, proposals=None, scene_cache=None, return_tensors=False,
                         recovery=False, max_attempts=None, diagnostics=None):
    """Find a scene anchor, or sample continuous starts around its valid pose.

    A new scene uses a fixed 3x3x3 TCP grid at 5/50/95% of its existing bounds,
    the nominal orientation plus the eight 90%-range rotation-vector corners,
    and the four existing IK branches. This temporary anchor is not a clone's
    final initial pose. Anchored clone resets retain continuous randomized
    targets, failure-only recovery and the configured random candidate budget.
    Explicit max_attempts can truncate the fixed domain for diagnostic calls.
    CUDA keeps acceptance masks on-device across fixed candidate chunks. No
    layout redraw or live physics trial is used in either path.
    A diagnostics dictionary requests a failure-only eager replay of these same
    candidates, bypassing the CUDA graph and fused IK/clearance implementations.
    """
    if len(samples) != len(seeds):
        raise ValueError("Every scene requires one reproducible robot-start seed.")
    if not samples:
        return []
    model, rules = env.robot_pose_model, env.cfg.robot_start
    device, dtype = env.device, torch.float32
    count = len(samples)
    attempts_limit = rules.max_attempts if max_attempts is None else max_attempts
    if not isinstance(attempts_limit, int) or isinstance(attempts_limit, bool) or attempts_limit <= 0:
        raise ValueError("Robot start candidate budget must be a positive integer.")
    if not isinstance(rules.ik_batch_size, int) or isinstance(rules.ik_batch_size, bool) or rules.ik_batch_size <= 0:
        raise ValueError("Robot start ik_batch_size must be a positive integer.")
    # The shared producer owns immutable singleton proposals. Cache only scene
    # inputs, never candidates or acceptance results for the new robot pose.
    singularity_settings = _robot_start_singularity_settings(env)
    key = (id(model), tuple(vars(rules).items()), tuple(vars(env.spawn_rules).items()), singularity_settings)
    cached = getattr(scene_cache, "robot", None)
    if cached is None or cached[0] != key:
        _validate_robot_start_rules(env.spawn_rules, rules)
        source_samples = [scene_cache.sample] if scene_cache is not None else samples
        source = scene_cache.proposals if scene_cache is not None else proposals
        if source is None:
            source = _episode_proposals(source_samples,len(env.spawn_geometries),device,dtype)
        boxes = _scene_boxes_tensor(env,source_samples,source)
        constants = model._tensor_constants(device, dtype)
        blockers = torch.tensor([sample.blocker_id for sample in source_samples], device=device)
        blocker_pose = source[0][torch.arange(len(source_samples), device=device), blockers]
        regions = _robot_start_regions(blocker_pose, constants['scene_half'][blockers], env.spawn_rules, rules)
        if not bool(torch.isfinite(regions).all() & (regions[:, 0] < regions[:, 1]).all()):
            raise ValueError("Robot start front region is empty or non-finite after shelf clearance.")
        cached = (key, (*boxes, regions))
        if scene_cache is not None:
            scene_cache.robot = cached
    centers, rotations, half, obstacle_valid, regions = (
        value.expand(count, *value.shape[1:]) for value in cached[1])
    constants = model._tensor_constants(device, dtype)
    if singularity_settings is not None:
        com = singularity_settings[4]
        offsets = constants.setdefault('singularity_com_offsets', {})
        if com not in offsets:
            offsets[com] = torch.tensor(com, device=device, dtype=dtype)
    if "start_origin" not in constants:
        constants["start_origin"] = torch.as_tensor(model.base_origin, device=device, dtype=dtype)
        constants["start_rotation"] = torch.tensor(ROBOT_START_TCP_ROTATION, device=device, dtype=dtype)
    origin, rotation = constants["start_origin"], constants["start_rotation"]
    template = getattr(scene_cache, "template", None)
    anchored = template is not None
    if anchored:
        anchor = getattr(scene_cache, "start_anchor", None)
        if anchor is None or anchor[0] is not template:
            arm = template.joint_positions[0, constants["arm_ids"]].to(device=device, dtype=dtype)
            tcp = model.arm_forward_batch(arm[None])[2][0]
            anchor = (template, arm, tcp[:3, :3].clone(), tcp[:3, 3] - origin)
            scene_cache.start_anchor = anchor
        anchor_arm = anchor[1].expand(count, -1)
        nominal_rotation = anchor[2].expand(count, -1, -1)
        if recovery:
            half_width = regions.new_tensor(rules.recovery_position_half_width)
            low = torch.maximum(regions[:, ::2], anchor[3] - half_width)
            high = torch.minimum(regions[:, 1::2], anchor[3] + half_width)
            regions = torch.stack((low, high), -1).flatten(-2)
    else:
        if recovery:
            raise ValueError("Robot start recovery requires a validated scene template.")
        anchor_arm = constants["default"][constants["arm_ids"]].expand(count, -1)
        nominal_rotation = rotation.expand(count, -1, -1)
    lower, upper = regions[:, ::2], regions[:, 1::2]
    orientation_range = regions.new_tensor(rules.orientation_range) * (
        rules.recovery_rotation_scale if recovery else 1.0)
    if "start_lifts" not in constants:
        constants["start_lifts"] = regions.new_tensor((-2.0, -1.6, -.6, -1.0))
        constants["start_elbows"] = regions.new_tensor((1.8, 1.2, -1.8, -1.2))

    def candidate_inputs(random, low, high, reference_rotation, reference_arm, offset, width):
        absolute = torch.arange(offset, offset + width, device=device)
        # Four joint seeds solve the SAME sampled pose. A change of memory cap
        # therefore never changes either the target or its random stream.
        selected = random[:, absolute // 4]
        target = low[:, None] + selected[..., :3] * (high - low)[:, None] + origin
        rotvec = (2 * selected[..., 3:6] - 1) * orientation_range
        angle = rotvec.norm(dim=-1, keepdim=True)
        quaternion = torch.cat(((angle * .5).cos(), .5 * torch.sinc(angle / (2 * math.pi)) * rotvec), -1)
        target_rotation = reference_rotation[:, None] @ matrix_from_quat(quaternion)
        branch = absolute % 4
        lifts = constants["start_lifts"][branch]
        elbows = constants["start_elbows"][branch]
        arm = constants["default"][constants["arm_ids"]].expand(len(low), width, -1).clone()
        arm[..., 1], arm[..., 2], arm[..., 3] = lifts, elbows, -lifts - elbows
        strength = (absolute // 8).clamp_max(2).to(dtype)
        arm += (selected[..., 6:] - .5) * strength[None, :, None]
        if anchored:
            arm = torch.where((branch == 0)[None, :, None], reference_arm[:, None], arm)
        return target, target_rotation, arm
    if not anchored:
        # Static strata cover the allowed position/orientation domain before
        # choosing one template. Zero-width orientation axes do not duplicate
        # candidates. Absolute candidate IDs preserve pose/branch order across
        # CPU/GPU chunk sizes just as in the continuous reset path.
        strata = regions.new_tensor((.05, .5, .95))
        position_fractions = torch.cartesian_prod(strata, strata, strata)
        rotation_axes = [regions.new_tensor((.05, .95) if value else (.5,))
                         for value in rules.orientation_range]
        rotation_fractions = torch.cartesian_prod(*rotation_axes)
        if any(rules.orientation_range):
            rotation_fractions = torch.cat((regions.new_full((1, 3), .5), rotation_fractions))
        pose_count = len(position_fractions)*len(rotation_fractions)
        parameters = regions.new_full((pose_count, 12), .5)
        parameters[:, :3] = position_fractions[:, None].expand(-1, len(rotation_fractions), -1).reshape(-1, 3)
        parameters[:, 3:6] = rotation_fractions[None].expand(len(position_fractions), -1, -1).reshape(-1, 3)
        attempts_limit = min(max_attempts, pose_count*4) if max_attempts is not None else pose_count*4
        noise = parameters[None].expand(count, -1, -1).contiguous()
    # Local generators keep each anchored episode independent of reset subset/order.
    elif torch.device(device).type == "cuda":
        noise = torch.empty((count, attempts_limit, 12), device=device, dtype=dtype)
        seed_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
        wp.launch(_robot_start_noise, dim=noise.shape[:2], inputs=[wp.from_torch(seed_tensor), wp.from_torch(noise)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
    else:
        noise = torch.as_tensor(np.stack([np.random.default_rng(int(seed) ^ 0x5D2EA91C).random(
            (attempts_limit, 12), dtype=np.float32) for seed in seeds]), device=device)
    result = torch.zeros((count, len(model.joint_names) + 5), device=device, dtype=dtype)
    if torch.device(device).type == "cuda" and diagnostics is None:
        # Keep capture buffers and PyTorch generator bookkeeping ordinary
        # tensors even when an RSL-RL rollout calls reset in InferenceMode.
        # They can then be replayed safely from either execution mode.
        with torch.inference_mode(False):
            result = result.clone()
            # A small first chunk handles easy scenes without running unnecessary
            # nonlinear solves. Later chunks retain fixed rows and skip accepted
            # rows inside the IK/clearance kernels; there is no host readback or
            # variable-size GPU nonzero between chunks. Capture the tensor dispatch
            # as one graph so inactive tail chunks have little Python overhead.
            maximum = min(32 if count <= 32 else 8, attempts_limit, rules.ik_batch_size)
            group_size = 1 << max(0, (rules.ik_batch_size // maximum).bit_length() - 1)
            cache = getattr(model, "_robot_start_graphs", {})
            model._robot_start_graphs = cache
            for begin in range(0, count, group_size):
                rows = min(group_size, count - begin)
                bucket = 1 << (rows - 1).bit_length()
                key = (str(noise.device), dtype, bucket, centers.shape[1], maximum, attempts_limit, anchored, recovery,
                       tuple(vars(rules).items()), singularity_settings, torch.backends.cuda.matmul.allow_tf32)
                if key not in cache:
                    # These are independent input buffers. Scene changes update
                    # them on replay instead of retaining stale obstacle geometry.
                    inputs = [value[begin:begin + 1].expand(bucket, *value.shape[1:]).clone()
                              for value in (noise, centers, rotations, half, obstacle_valid, regions,
                                            nominal_rotation, anchor_arm)]
                    enabled = torch.zeros(bucket, device=device, dtype=torch.bool)
                    inputs.append(enabled)

                    def candidates(values):
                        (random, scene_centers, scene_rotations, scene_half, scene_valid, bounds,
                         reference_rotation, reference_arm, enabled_rows) = values
                        low, high = bounds[:, ::2], bounds[:, 1::2]
                        accepted_rows = ~enabled_rows
                        output = torch.zeros((bucket, result.shape[1]), device=device, dtype=dtype)
                        offset = 0
                        while offset < attempts_limit:
                            width = min(8 if offset == 0 else maximum, maximum, attempts_limit - offset)
                            target, target_rotation, arm = candidate_inputs(
                                random, low, high, reference_rotation, reference_arm, offset, width)
                            active_candidates = (~accepted_rows[:, None]).expand(-1, width).reshape(-1).contiguous()
                            q, valid, frames, joints, tcp = model.solve_batch(
                                target.reshape(-1, 3), target_rotation.reshape(-1, 3, 3),
                                arm.reshape(-1, 6), active_mask=active_candidates)
                            if singularity_settings is not None:
                                valid &= _robot_start_nonsingular(model, frames, joints, singularity_settings)
                            clear, heights = _robot_candidate_clearance(
                                env, frames, joints, tcp,
                                scene_centers[:, None].expand(-1, width, -1, -1).reshape(-1, *scene_centers.shape[1:]),
                                scene_rotations[:, None].expand(-1, width, -1, -1, -1).reshape(-1, *scene_rotations.shape[1:]),
                                scene_half[:, None].expand(-1, width, -1, -1).reshape(-1, *scene_half.shape[1:]),
                                scene_valid[:, None].expand(-1, width, -1).reshape(-1, scene_valid.shape[1]),
                                candidate_valid=valid & active_candidates)
                            actual = tcp[:, :3, 3].reshape(bucket, width, 3) - origin
                            valid = valid.reshape(bucket, width) & clear.reshape(bucket, width)
                            valid &= (actual >= low[:, None]).all(-1) & (actual <= high[:, None]).all(-1)
                            first = torch.where(valid, torch.arange(width, device=device), width).amin(-1)
                            found = first < width
                            chosen = torch.arange(bucket, device=device) * width + first.clamp_max(width - 1)
                            selected = torch.cat((q[chosen], actual.reshape(-1, 3)[chosen],
                                                  heights[chosen, None], (first + offset + 1).to(dtype)[:, None]), -1)
                            output = torch.where(found[:, None], selected, output)
                            accepted_rows |= found
                            offset += width
                        return output

                    with torch.cuda.device(device):
                        stream = torch.cuda.Stream(device=device)
                        stream.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(stream):
                            candidates(inputs)
                        torch.cuda.current_stream(device).wait_stream(stream)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = candidates(inputs)
                    # At most a small set of count buckets is needed. Bound graph
                    # memory if object counts or runtime rules change repeatedly.
                    if len(cache) >= 8:
                        cache.pop(next(iter(cache)))
                    # CUDA graph capture owns intermediate buffers, but these two
                    # tensors were created outside capture and must stay alive.
                    cache[key] = (inputs, graph, output, origin, orientation_range)
                inputs, graph, output, _, _ = cache[key]
                for buffer, value in zip(inputs[:-1], (noise, centers, rotations, half, obstacle_valid, regions,
                                                       nominal_rotation, anchor_arm)):
                    buffer[:rows].copy_(value[begin:begin + rows])
                inputs[-1].zero_()
                inputs[-1][:rows] = True
                graph.replay()
                result[begin:begin + rows].copy_(output[:rows])
            if return_tensors:
                return result, regions
            attempts = result[:, -1].to(torch.long).cpu().tolist()
            return [(result[row, :-5], regions[row], result[row, -5:-2], result[row, -2], result[row, -1])
                    if attempt else None for row, attempt in enumerate(attempts)]
    accepted = [False] * count
    if diagnostics is not None:
        diagnostic_names = ("finite", "ik", "nonsingular", "elbow_up", "forearm_forward",
                            "ground_clear", "scene_clear", "self_clear", "tcp_bounds")
        diagnostic_counts = torch.zeros((count, len(diagnostic_names), 2), device=device, dtype=torch.long)
        evaluated = torch.zeros(count, device=device, dtype=torch.long)
        minimum_errors = torch.full((count, 2), float("inf"), device=device)
    # Small asynchronous resets benefit from processing more candidates per
    # launch; full resets already have enough rows to fill the GPU. Keep the
    # same total scratch cap and the absolute candidate random streams.
    proposals = min(32 if count <= 32 else 8, attempts_limit, rules.ik_batch_size)
    group_size = max(1, rules.ik_batch_size // proposals)
    for begin in range(0, count, group_size):
        pending = list(range(begin, min(count, begin + group_size)))
        for offset in range(0, attempts_limit, proposals):
            if not pending:
                break
            width = min(proposals, attempts_limit - offset)
            rows = torch.tensor(pending, device=device, dtype=torch.long)
            target, target_rotation, arm = candidate_inputs(
                noise[rows], lower[rows], upper[rows], nominal_rotation[rows], anchor_arm[rows], offset, width)
            if diagnostics is None:
                q, valid, frames, joints, tcp = model.solve_batch(
                    target.reshape(-1, 3), target_rotation.reshape(-1, 3, 3), arm.reshape(-1, 6))
            else:
                q, valid, frames, joints, tcp = model._solve_batch_eager(
                    target.reshape(-1, 3), target_rotation.reshape(-1, 3, 3), arm.reshape(-1, 6))
                ik_valid = valid.clone()
            nonsingular = None
            if singularity_settings is not None:
                nonsingular = _robot_start_nonsingular(model, frames, joints, singularity_settings)
                valid &= nonsingular
            obstacle_rows = rows.repeat_interleave(width)
            if diagnostics is None:
                clear, heights = _robot_candidate_clearance(
                    env, frames, joints, tcp, centers[obstacle_rows],rotations[obstacle_rows],
                    half[obstacle_rows], obstacle_valid[obstacle_rows])
            else:
                predicates = {}
                clear, heights = _robot_candidate_clearance_eager(
                    env, frames, joints, tcp, centers[obstacle_rows], rotations[obstacle_rows],
                    half[obstacle_rows], obstacle_valid[obstacle_rows], details=predicates)
            actual = tcp[:, :3, 3] - origin
            in_bounds = (actual >= lower[obstacle_rows]).all(-1) & (actual <= upper[obstacle_rows]).all(-1)
            valid &= clear & in_bounds
            if diagnostics is not None:
                predicates["finite"] &= torch.isfinite(q).all(-1)
                predicates.update(ik=ik_valid, nonsingular=(torch.ones_like(valid) if nonsingular is None
                                                          else nonsingular), tcp_bounds=in_bounds)
                masks = torch.stack([predicates[name] for name in diagnostic_names], -1)
                cumulative = masks.long().cumprod(-1).bool()
                diagnostic_counts[rows] += torch.stack((masks, cumulative), -1).reshape(
                    len(rows), width, len(diagnostic_names), 2).sum(1)
                evaluated[rows] += width
                errors = torch.stack(((target.reshape(-1, 3) - tcp[:, :3, 3]).norm(dim=-1),
                    model._rotation_error_batch(target_rotation.reshape(-1, 3, 3), tcp[:, :3, :3]).norm(dim=-1)), -1)
                best = torch.nan_to_num(errors, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
                minimum_errors[rows] = torch.minimum(minimum_errors[rows], best.reshape(len(rows), width, 2).amin(1))
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
    if diagnostics is not None:
        counts_host, evaluated_host = diagnostic_counts.cpu().tolist(), evaluated.cpu().tolist()
        errors_host = minimum_errors.cpu().tolist()
        regions_host = regions.cpu().tolist()
        diagnostics.update(backend="torch_eager_fp32", ik_iterations=48, candidate_budget=attempts_limit,
            rows=[dict(evaluated=evaluated_host[row], tcp_region_xyz=regions_host[row],
                independent_pass=dict(zip(diagnostic_names, [item[0] for item in counts_host[row]])),
                cumulative_pass=dict(zip(diagnostic_names, [item[1] for item in counts_host[row]])),
                minimum_position_error_m=errors_host[row][0] if math.isfinite(errors_host[row][0]) else None,
                minimum_rotation_error_rad=errors_host[row][1] if math.isfinite(errors_host[row][1]) else None,
                recovered=accepted[row], selected_candidate=int(result[row, -1])) for row in range(count)])
    if return_tensors:
        return result, regions
    return [(result[row, :-5],regions[row],result[row, -5:-2],result[row, -2],result[row, -1])
            if accepted[row] else None for row in range(count)]


def _recover_initial_robot_starts(env, samples, seeds, starts, proposals):
    """Replay failed anchors once, with identical poses and strict predicates.

    This runs only after the normal initial search fails. Eager tensor IK and
    clearance bypass CUDA graph/fused implementations, with TF32 disabled for
    the replay and restored even on failure. Successful anchors still undergo
    the existing physical-articulation FK check in the episode builder.
    """
    failed = [index for index, start in enumerate(starts) if start is None]
    if not failed:
        return starts, {}
    rows = torch.tensor(failed, device=env.device, dtype=torch.long)
    selected_proposals = tuple(value[rows] for value in proposals)
    diagnostics, recovered = {}, [None] * len(failed)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        recovered = _sample_robot_starts(
            env, [samples[index] for index in failed], [seeds[index] for index in failed],
            proposals=selected_proposals, diagnostics=diagnostics)
    except Exception as error:
        # Retain the original failed-anchor error if diagnostics cannot run.
        # Never accept an unchecked/default pose to hide an initialization error.
        diagnostics["replay_error"] = f"{type(error).__name__}: {error}"
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    from importlib.metadata import PackageNotFoundError, version
    versions = {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda}
    for package in ("warp-lang", "isaacsim", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    model = env.robot_pose_model
    model_data = {name: np.asarray(getattr(model, name, ())).tolist() for name in
                  ("base_origin", "root_transform", "arm_limits", "default_joint_positions")}
    model_data.update(joint_names=list(getattr(model, "joint_names", ())),
        body_names=list(getattr(model, "body_names", ())), tcp_body_name=getattr(model, "tcp_body_name", None),
        native_body_com=getattr(model, "_native_body_com_offsets", {}),
        joints=[dict(name=joint.name, parent=joint.parent, child=joint.child, axis=np.asarray(joint.axis).tolist(),
                     frame0=np.asarray(joint.frame0).tolist(), frame1_inverse=np.asarray(joint.frame1_inverse).tolist(),
                     index=joint.index, reverse=joint.reverse) for joint in getattr(model, "joints", ())],
        colliders=[dict(body=box.body_name, center=np.asarray(box.center_local).tolist(),
                        half=np.asarray(box.half_extents).tolist(),
                        rotation=np.asarray(box.rotation_local).tolist()) for box in getattr(model, "colliders", ())],
        shelf_boxes=[dict(center=np.asarray(box.center).tolist(), half=np.asarray(box.half_extents).tolist(),
                          rotation=np.asarray(box.rotation).tolist()) for box in getattr(model, "shelf_boxes", ())])
    model_hash = hashlib.sha256(json.dumps(model_data, sort_keys=True).encode()).hexdigest()
    source_files = (Path(__file__), Path(__file__).parents[1] / "config/ur5e/scene.py")
    source_hashes = {str(path.relative_to(Path(__file__).parents[1])):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    poses = selected_proposals[0].detach().cpu().tolist()
    log_dir = getattr(env.cfg, "log_dir", None)
    log_path = None if not log_dir else Path(log_dir) / "initial_robot_start_diagnostics.jsonl"
    reports, updated = {}, list(starts)
    for slot, index in enumerate(failed):
        sample = samples[index]
        scene = dict(active_ids=list(sample.active_ids), target_id=sample.target_id, blocker_id=sample.blocker_id,
                     poses={str(obj): poses[slot][obj] for obj in sample.active_ids})
        report = dict(schema=1, scene=scene,
            scene_sha256=hashlib.sha256(json.dumps(scene, sort_keys=True).encode()).hexdigest(),
            model_sha256=model_hash, model=model_data, source_sha256=source_hashes,
            versions=versions, device=str(env.device), env_seed=getattr(env.cfg, "seed", None),
            ik_seed=seeds[index], robot_start_rules=vars(env.cfg.robot_start),
            spawn_rules=vars(env.spawn_rules), tf32_before=previous_tf32, tf32_replay=False,
            backend=diagnostics.get("backend"), ik_iterations=diagnostics.get("ik_iterations"),
            candidate_budget=diagnostics.get("candidate_budget"),
            replay=diagnostics.get("rows", [{}] * len(failed))[slot],
            replay_error=diagnostics.get("replay_error"), recovered_candidate=recovered[slot] is not None)
        if torch.device(env.device).type == "cuda":
            report["gpu"] = torch.cuda.get_device_name(env.device)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
        summary = {name: report[name] for name in ("backend", "candidate_budget", "replay",
                                                  "replay_error", "recovered_candidate")}
        if log_path is not None:
            summary["diagnostic_file"] = str(log_path)
        print("[Initial IK replay] " + json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        reports[index] = summary
        if recovered[slot] is not None:
            updated[index] = recovered[slot]
    return updated, reports


@torch.no_grad()
@_robot_kinematics_precision()
def _check_physical_robot_starts(env, samples, starts, poses, origins, *, details=None):
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
    if details is not None:
        details['rows'] = [None] * len(starts)
    for begin in range(0, len(starts), 128):
        selected = starts[begin:begin + 128]
        # This startup-only check is an independent reference for the FP32 IK
        # result. TF32 rounding must not decide a 0.2 mm collision invariant.
        q = torch.stack([torch.as_tensor(start[0],device=poses.device,dtype=torch.float64) for start in selected])
        expected = model.forward_batch(q)
        actual = poses[begin:begin + len(selected)].to(torch.float64)
        positions = (actual[..., :3] - origins[begin:begin + len(selected), None].to(torch.float64)
                     + actual.new_tensor(model.base_origin))
        rotations = matrix_from_quat(convert_quat(actual[..., 3:], to="wxyz"))
        position_error = (positions - expected[..., :3, 3]).norm(dim=-1)
        rotation_error = (rotations - expected[..., :3, :3]).norm(dim=(-2, -1))
        match = ((position_error <= 2.e-4).all(-1) & (rotation_error <= 3.e-4).all(-1)
                 & torch.isfinite(actual).all((-2, -1)))
        output.append(match)
        if details is not None:
            for row in (~match).nonzero().flatten().cpu().tolist():
                score = torch.nan_to_num(torch.maximum(position_error[row] / 2.e-4,
                                                       rotation_error[row] / 3.e-4), nan=float('inf'))
                link = int(score.argmax())
                details['rows'][begin + row] = dict(
                    link=model.body_names[link], reference_dtype='float64',
                    maximum_position_error_m=float(position_error[row].amax()),
                    maximum_rotation_matrix_error=float(rotation_error[row].amax()),
                    position_error_m=float(position_error[row, link]),
                    rotation_matrix_error=float(rotation_error[row, link]),
                    actual_position=positions[row, link].cpu().tolist(),
                    expected_position=expected[row, link, :3, 3].cpu().tolist())
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



# Native reset event. A scene round shares one layout/vision; robot starts
# are continuously sampled and collision-checked on every selected-row reset.

@dataclass
class EpisodeBatch:
    object_state: torch.Tensor       # M,P,13; environment-local link pose / zero COM velocity
    object_ids: torch.Tensor         # M,O; active pool IDs, -1 padding
    active: torch.Tensor             # M,P
    target_id: torch.Tensor          # M
    blocker_id: torch.Tensor         # M
    access_region: torch.Tensor      # M,4
    joint_positions: torch.Tensor    # M,J; checked complete articulation configurations
    geometry_b: torch.Tensor         # M,O,10; initial geometry in robot base frame
    geometry_valid: torch.Tensor     # M,O
    observed_ids: torch.Tensor       # M,O
    robot_regions: torch.Tensor      # M,6
    robot_tcp: torch.Tensor          # M,3; environment coordinates
    visibility: torch.Tensor         # M,2; target, blocker
    camera_pose: torch.Tensor        # M,7; world frame (shared template: env-local), wxyz
    camera_intrinsics: torch.Tensor  # M,3,3
    seeds: torch.Tensor              # M; reproducible per-environment episode seed


def pack_object_states(default_state, sample, lower, upper, upright):
    """Single-proposal convenience wrapper around the batched pose conversion."""
    return tuple(value[0] for value in pack_episode_states(default_state, [sample], lower, upper, upright))


def _episode_proposals(samples, p, device, dtype, *, support_heights=None, max_objects=None):
    """Keep XY/yaw on device; only IDs and static support heights come from metadata."""
    m = len(samples)
    active = np.zeros((m, p), dtype=bool)
    capacity = _DEFAULT_MAX_OBJECTS if max_objects is None else max_objects
    if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 2:
        raise ValueError("Episode proposals require an integer object slot capacity of at least two.")
    slots = np.full((m, capacity), -1, dtype=np.int64)
    lengths = np.fromiter((len(s.active_ids) for s in samples), dtype=np.int64, count=m)
    if (lengths > capacity).any():
        raise ValueError("An episode has more active objects than the configured physical capacity.")
    rows = np.repeat(np.arange(m), lengths)
    columns = np.arange(len(rows)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    ids = np.concatenate([s.active_ids for s in samples]).astype(np.int64, copy=False) if m else np.empty(0, np.int64)
    active[rows, ids], slots[rows, columns] = True, ids
    if samples and all(sample.planar_poses is not None for sample in samples):
        planar_lengths = np.fromiter((len(s.planar_ids) for s in samples), dtype=np.int64, count=m)
        indices = torch.as_tensor(np.stack((np.repeat(np.arange(m), planar_lengths),
            np.concatenate([s.planar_ids for s in samples]))), device=device, dtype=torch.long)
        if support_heights is not None:
            heights = support_heights.to(device=device, dtype=dtype)[indices[1]]
        else:  # Standalone callers without the cached catalog retain the metadata interface.
            heights = torch.tensor([(s.positions.support_heights[index] if isinstance(s.positions, _PoseMapping)
                                     else s.positions[index][2]) for s in samples for index in s.planar_ids],
                                   device=device,dtype=dtype)
        planar = torch.cat([s.planar_poses for s in samples]).to(device=device,dtype=dtype)
        values = torch.stack((planar[:,0],planar[:,1],heights,planar[:,2]),-1)
        proposed = torch.zeros((m,p,4),device=device,dtype=dtype)
        proposed[indices[0],indices[1]] = values
    else:
        proposed = np.zeros((m,p,4),dtype=np.float32)
        for row,sample in enumerate(samples):
            ids = list(sample.active_ids)
            proposed[row,ids,:3] = [sample.positions[index] for index in ids]
            proposed[row,ids,3] = [sample.yaws[index] for index in ids]
        proposed = torch.as_tensor(proposed,device=device,dtype=dtype)
    return proposed,torch.as_tensor(active,device=device),torch.as_tensor(slots,device=device)


def pack_episode_states(default_state, samples, lower, upper, upright, *, proposals=None, max_objects=None):
    """Vectorize off-center/upright conversion without downloading device layouts."""
    m, p = len(samples), default_state.shape[0]
    proposed, active, slots = proposals if proposals is not None else _episode_proposals(
        samples,p,default_state.device,default_state.dtype,max_objects=max_objects)
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
        OBJECT_SPECS, RobotPoseModel, inspect_geometry, inspect_shelf_bounds, FORCE_TORQUE_BODY_NAME,
    )
    from .sensors import SensorReader, InitialVisionSensor

    if not env.cfg.robot_start.enabled:
        raise ValueError("Episode starts require collision-checked elbow-up robot randomization.")
    objects = env.scene["object_collection"]
    if tuple(objects.object_names) != tuple(spec.name for spec in OBJECT_SPECS):
        raise ValueError("The native object collection order must match the static pool catalog.")
    env.spawn_geometries, env.spawn_rules = inspect_geometry(env)
    env.robot_pose_model = RobotPoseModel.from_env(env)
    env.robot_pose_model.base_origin = env.scene.env_origins[0].detach().cpu().numpy()
    lower = torch.tensor([geometry.lower for geometry in env.spawn_geometries], device=env.device)
    upper = torch.tensor([geometry.upper for geometry in env.spawn_geometries], device=env.device)
    upright = lower.new_tensor([spec.upright_quat for spec in OBJECT_SPECS])
    vertex_count = max(len(geometry.support_vertices) for geometry in env.spawn_geometries)
    if vertex_count == 0:
        raise RuntimeError("Native objects require measured collision support vertices.")
    support_vertices = lower.new_zeros((len(env.spawn_geometries), vertex_count, 3))
    support_mask = torch.zeros(support_vertices.shape[:-1], device=env.device, dtype=torch.bool)
    for index, geometry in enumerate(env.spawn_geometries):
        count = len(geometry.support_vertices)
        support_vertices[index, :count] = lower.new_tensor(geometry.support_vertices)
        support_mask[index, :count] = True
    shelf_lower, shelf_upper = lower.new_tensor(inspect_shelf_bounds(env))
    signs = lower.new_tensor(((-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                             (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)))
    shelf_corners = (shelf_lower + shelf_upper) * .5 + (shelf_upper - shelf_lower) * .5 * signs
    n, o = env.num_envs, env.cfg.mdp.max_objects
    physical = _active_object_capacity(env)
    env.task_scene_data = SceneBatch(
        object_ids=torch.full((n, physical), -1, device=env.device, dtype=torch.long),
        target_id=torch.zeros(n, device=env.device, dtype=torch.long),
        blocker_id=torch.ones(n, device=env.device, dtype=torch.long),
        geometry_b=torch.zeros((n, o, 10), device=env.device),
        geometry_valid=torch.zeros((n, o), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((n, o), -1, device=env.device, dtype=torch.long),
        lower=lower, upper=upper, upright=upright,
        access_region=torch.zeros((n, 4), device=env.device),
        workspace_bounds=lower.new_tensor((*env.spawn_rules.x_range, *env.spawn_rules.y_range)).expand(n, -1),
        shelf_top=lower.new_full((n,), env.spawn_rules.shelf_top),
        ceiling=lower.new_full((n,), env.spawn_rules.ceiling),
        shelf_local_corners=shelf_corners.expand(n, -1, -1),
        sensor_body_name=FORCE_TORQUE_BODY_NAME,
        support_vertices=support_vertices, support_vertex_mask=support_mask,
        support_error=lower.new_tensor([geometry.support_error for geometry in env.spawn_geometries]),
    )
    # Keep native actors stable; shared-scene commits select which pool bodies
    # participate in dynamics and collision detection.
    objects.parked_mask = torch.zeros((n, len(OBJECT_SPECS)), dtype=torch.bool, device=env.device)
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
    # Reuse the perception meshes instead of treating empty visual OBB corners
    # as possible occluders. This is one CPU hull extraction per asset at
    # startup; shared copies and every subsequent reset reuse these vertices.
    visual_footprints = runtime.vision.visual_footprints(upright)
    env.spawn_geometries = tuple(replace(geometry, visual_footprint_vertices=vertices)
        for geometry, vertices in zip(env.spawn_geometries, visual_footprints))
    if env.spawn_rules.generation_mode in ('parameterized', 'clutter'):
        directory = (Path(env.spawn_rules.parameter_cache_directory) if env.spawn_rules.parameter_cache_directory
                     else Path(__file__).resolve().parents[4] / 'outputs/contact_manipulation_parallel/parameter_domains')
        runtime.parameter_catalog = initialize_certified_catalog(
            env, runtime, directory,
            reference_count=10 if env.spawn_rules.generation_mode == 'clutter' else None)
    env._episode_reset = runtime
    return runtime


def _write_robot_reset(env, ids, joint_positions):
    """Write selected robot rows without changing object state or activation."""
    robot = env.scene["robot"]
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root, env_ids=ids)
    zero_joints = torch.zeros_like(joint_positions)
    robot.write_joint_state_to_sim(joint_positions, zero_joints, env_ids=ids)
    robot.set_joint_position_target(joint_positions, env_ids=ids)
    robot.set_joint_velocity_target(zero_joints, env_ids=ids)
    robot.set_joint_effort_target(zero_joints, env_ids=ids)


def _write_physical_reset(env, ids, object_state, joint_positions, active):
    """Commit flags before GPU poses; unchanged shared masks require no flag work."""
    objects = env.scene["object_collection"]
    # USD flag changes must reach PhysX before writing the new dynamic poses.
    # A partial reset can only reuse the scene's existing shared active mask.
    objects.set_active_objects(active, env_ids=ids)
    states = object_state.clone()
    states[..., :3] += env.scene.env_origins[ids, None, :]
    objects.write_object_state_to_sim(states, env_ids=ids)
    _write_robot_reset(env, ids, joint_positions)


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


def _robot_start_domain_successes(env, sample, start, proposals, seed):
    """Probe independent 6D targets for the legacy sampler's distribution gate.

    A valid anchor alone does not show that its configured position/orientation
    domain is usable. Each row receives one target and the anchor IK seed, so
    retries cannot turn a tiny feasible pocket into an apparent broad domain.
    These are analytical GPU IK/clearance checks, without live physics writes.
    """
    count = env.cfg.robot_start.domain_probe_count
    anchor = SimpleNamespace(
        joint_positions=torch.as_tensor(start[0], device=env.device).unsqueeze(0),
        robot_tcp=torch.as_tensor(start[2], device=env.device).unsqueeze(0),
    )
    cache = SimpleNamespace(template=anchor, sample=sample, proposals=proposals, robot=None)
    tensor_starts = torch.device(env.device).type == "cuda"
    starts = _sample_robot_starts(
        env, [sample] * count, [_episode_seed(seed, index, 0) for index in range(count)],
        scene_cache=cache, return_tensors=tensor_starts, max_attempts=1,
    )
    if tensor_starts:
        return int((starts[0][:, -1] > 0).sum().item())
    return sum(start is not None for start in starts)


def sample_certified_episode(catalog, geometries, rules, rng, target_id, blocker_id):
    """Bind one categorical composition to a precompiled scene family.

    All cell support, camera and visibility constraints belong to the startup
    catalog. Reset draws the balanced composition, selects one family whose
    typed fixed companions are available, then samples its stored CDF once.
    It does not prepare/intersect cells, query visibility, approve generated
    poses, or retry another family. Catalog-wide coverage and the worst-case
    background budget must already have been established by its compiler.
    """
    ids, target, blocker = _constructive_composition(
        geometries, rules, rng, target_id, blocker_id)
    pair = (_asset_type(geometries[target]), _asset_type(geometries[blocker]))
    families = getattr(catalog, 'families', catalog).get(pair, ())
    available = {}
    for index in ids:
        if index not in (target, blocker):
            available.setdefault(_asset_type(geometries[index]), []).append(index)
    eligible = [family for family in families
                if all(asset in available for asset in (family.get('fixed_objects') or {}))]
    if not eligible:
        raise RuntimeError(f"Certified catalog has no family for ordered pair {pair} and its sampled composition; "
                           "startup categorical coverage is incomplete. No redraw was performed.")
    family = rng.choice(eligible)
    companions = {available[asset][0]: pose for asset, pose in (family.get('fixed_objects') or {}).items()}
    background = tuple(index for index in ids if index not in (target, blocker) and index not in companions)
    capacity = _certified_background_capacity(geometries, background, rules)
    loader = getattr(catalog, 'load_visibility', None)
    visibility = loader(family) if callable(loader) else family.get('visibility_domain')
    if visibility is None:
        raise ValueError("A certified catalog needs its visibility loader or an inline visibility domain.")
    expected = family.get('visibility_fingerprint')
    if expected is not None and visibility.get('fingerprint') != expected:
        raise ValueError("Loaded visibility fingerprint does not match the selected certified family.")
    domain = SimpleNamespace(geometries=tuple(geometries), rules=rules, active_ids=ids,
        target_id=target, blocker_id=blocker, target_yaw=family['target_yaw'],
        blocker_yaw=family['blocker_yaw'], blocker_x=family['blocker_x'],
        camera_pose=np.asarray(family['camera_pose'], dtype=np.float64), visibility=visibility,
        cells=family['cells'], cumulative=family['cumulative'], background_ids=background,
        capacity=capacity, fixed_objects=companions, pivot=family.get('pivot'))
    return sample_certified_scene(domain, rng)


@njit(cache=True)
def _clutter_grid_mask(xs, ys, normals, bounds):
    """Conservative convex cell coverage, without a vertex-by-image tensor."""
    mask = np.empty((len(xs), len(ys)), dtype=np.bool_)
    for x in range(len(xs)):
        for y in range(len(ys)):
            inside = True
            for plane in range(len(bounds)):
                if xs[x] * normals[plane, 0] + ys[y] * normals[plane, 1] > bounds[plane] + 1.e-12:
                    inside = False
                    break
            mask[x, y] = inside
    return mask


def _clutter_gpu_seed(polygon_variants, support_variants, fixed_obstacles, support_bounds, gap, *,
                      device, obb_support_bounds=None, resolution=.0005, minimum_count=0):
    """Direct GPU free-grid construction in the caller's balanced body order.

    Each body supplies already-rotated convex collider alternatives and their
    matching full OBB polygons. Collider support and OBB support use separate
    bounds. Inflated masks may extend past the shelf; only the uninflated
    geometry is constrained there, avoiding an artificial half-gap at walls.

    Every body uses one joint variant/position argmax over its complete free
    grid. The mandatory head may be processed largest first, retaining original
    row IDs; missing core rows stay unplaced for the joint constraint system.
    Optional rows retain their input order and stop at the first failure. No
    pocket or complete trajectory is reserved here. This is a partial graph
    seed, not an accepted active scene or permission to drop mandatory types.
    FFT kernels stay cached only until their last use in this call; unique
    rotated shapes must not accumulate full-grid spectra across all bodies.
    """
    from scipy.fft import next_fast_len

    device = torch.device(device)
    support = np.asarray(support_bounds, dtype=np.float64)
    obb_support = support if obb_support_bounds is None else np.asarray(obb_support_bounds, dtype=np.float64)
    if (device.type != 'cuda' or support.shape != (4,) or obb_support.shape != (4,)
            or not np.isfinite(support).all() or not np.isfinite(obb_support).all()
            or support[0] >= support[1] or support[2] >= support[3]
            or obb_support[0] >= obb_support[1] or obb_support[2] >= obb_support[3]
            or not math.isfinite(gap) or gap < 0 or not math.isfinite(resolution) or resolution <= 0):
        raise ValueError('GPU clutter construction needs CUDA, finite support bounds and a nonnegative gap.')
    polygons = tuple(tuple(np.asarray(poly, dtype=np.float64) for poly in row) for row in polygon_variants)
    obbs = tuple(tuple(np.asarray(poly, dtype=np.float64) for poly in row) for row in support_variants)
    fixed = tuple(np.asarray(poly, dtype=np.float64) for poly in fixed_obstacles)
    if len(polygons) != len(obbs) or any(not row or len(row) != len(obb) for row, obb in zip(polygons, obbs)):
        raise ValueError('Every clutter body needs matching nonempty collider/OBB variants.')
    if (not isinstance(minimum_count, int) or isinstance(minimum_count, bool)
            or not 0 <= minimum_count <= len(polygons)):
        raise ValueError('The mandatory clutter head must be an integer within the input rows.')
    for poly in (*fixed, *(poly for row in polygons for poly in row), *(poly for row in obbs for poly in row)):
        if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
            raise ValueError('Clutter colliders, OBBs and fixed guards must be finite convex XY polygons.')
    centers = np.full((len(polygons), 2), np.nan)
    variant_ids = np.full(len(polygons), -1, dtype=np.int64)
    if not polygons:
        return centers, variant_ids, dict(method='gpu_fft_free_grid', maximum_greedy_prefix=0,
                                         input_order_preserved=True, pose_proposals=0, approval_retries=0)

    # A fixed finite set of support planes encloses the actual convex polygon.
    # The cell support term makes raster overlap conservative. Geometry is
    # never replaced in PhysX, nor expanded for the shelf support constraints.
    angles = np.arange(64) * (2.0 * math.pi / 64)
    normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    half_gap = .5 * gap
    cell_support = resolution * .5 * np.abs(normals).sum(axis=1)
    margin = (math.ceil(half_gap / resolution) + 2) * resolution
    origin = support[[0, 2]] - margin
    shape = np.ceil((support[[1, 3]] - support[[0, 2]] + 2 * margin) / resolution).astype(int)
    fft_shape = tuple(next_fast_len(int(value) + 2) for value in shape)
    occupied = np.zeros(tuple(shape), dtype=np.bool_)

    def plane_bounds(poly):
        return (poly @ normals.T).max(axis=0) + half_gap + cell_support

    for poly in fixed:
        low = np.maximum(0, np.floor((poly.min(axis=0) - half_gap - origin) / resolution).astype(int) - 1)
        high = np.minimum(shape, np.ceil((poly.max(axis=0) + half_gap - origin) / resolution).astype(int) + 1)
        if np.any(low >= high):
            continue
        occupied[low[0]:high[0], low[1]:high[1]] |= _clutter_grid_mask(
            origin[0] + (np.arange(low[0], high[0]) + .5) * resolution,
            origin[1] + (np.arange(low[1], high[1]) + .5) * resolution, normals, plane_bounds(poly))
    grid = torch.as_tensor(occupied, device=device)
    padded = torch.nn.functional.pad

    def transform(values):
        return torch.fft.rfft2(values, s=fft_shape)

    xs = origin[0] + (np.arange(-1, shape[0] + 1) + .5) * resolution
    ys = origin[1] + (np.arange(-1, shape[1] + 1) + .5) * resolution
    walls = ((xs[:, None] < support[0]) | (xs[:, None] > support[1])
             | (ys[None, :] < support[2]) | (ys[None, :] > support[3]))
    wall_fft = transform(torch.as_tensor(walls, dtype=torch.float64, device=device))
    cache, error = {}, torch.zeros((), dtype=torch.float64, device=device)
    def size_order(index):
        row = polygons[index]
        extent = max(float(np.ptp(poly, axis=0).max()) for poly in row)
        area = max(abs(float(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
                    - poly[:, 1] * np.roll(poly[:, 0], -1)))) for poly in row)
        return -extent, -area, index
    order = sorted(range(minimum_count), key=size_order) + list(range(minimum_count, len(polygons)))
    keys = tuple(tuple((poly.shape, poly.tobytes(), obb.shape, obb.tobytes())
                       for poly, obb in zip(row, obbs[index])) for index, row in enumerate(polygons))
    last_use = {key: step for step, index in enumerate(order) for key in keys[index]}
    unique_kernels, peak_cached_kernels = 0, 0
    queries = 0
    for step, index in enumerate(order):
        if index >= minimum_count and minimum_count and (variant_ids[:minimum_count] < 0).any():
            break
        variants = polygons[index]
        scene_fft = transform(padded(grid.to(torch.float64), (1, 1, 1, 1)))
        choices = []
        for variant, (poly, obb, key) in enumerate(zip(variants, obbs[index], keys[index])):
            if key not in cache:
                low = np.floor((poly.min(axis=0) - half_gap) / resolution) * resolution
                high = np.ceil((poly.max(axis=0) + half_gap) / resolution) * resolution
                extent = np.maximum(1, np.rint((high - low) / resolution).astype(int))
                mask = _clutter_grid_mask(low[0] + (np.arange(extent[0]) + .5) * resolution,
                    low[1] + (np.arange(extent[1]) + .5) * resolution, normals, plane_bounds(poly))
                cells = np.argwhere(mask)
                if len(cells) == 0:
                    raise ValueError('A clutter footprint has an empty raster; nonzero-area geometry is required.')
                begin, end = cells.min(axis=0), cells.max(axis=0) + 1
                mask = torch.as_tensor(mask[begin[0]:end[0], begin[1]:end[1]], device=device)
                mask_origin = low + begin * resolution
                body = padded(mask, (1, 1, 1, 1))
                ring = (torch.nn.functional.max_pool2d(body[None, None].to(torch.float64), 3, 1, 1)[0, 0] > 0) & ~body
                collider_low = support[[0, 2]] - poly.min(axis=0)
                collider_high = support[[1, 3]] - poly.max(axis=0)
                allowed_low = np.maximum(collider_low, obb_support[[0, 2]] - obb.min(axis=0))
                allowed_high = np.minimum(collider_high, obb_support[[1, 3]] - obb.max(axis=0))
                lower = np.ceil((allowed_low - origin + mask_origin) / resolution - 1.e-9).astype(int)
                upper = np.floor((allowed_high - origin + mask_origin) / resolution + 1.e-9).astype(int)
                cache[key] = (mask, mask_origin, transform(body.to(torch.float64)).conj(),
                              transform(ring.to(torch.float64)).conj(), lower, upper)
                unique_kernels += 1
                peak_cached_kernels = max(peak_cached_kernels, len(cache))
            mask, mask_origin, kernel, contact_kernel, lower, upper = cache[key]
            height, width = mask.shape
            if height > shape[0] or width > shape[1] or np.any(lower > upper):
                del kernel, contact_kernel
                continue
            size = (int(shape[0]) - height + 1, int(shape[1]) - width + 1)
            counts = torch.fft.irfft2(scene_fft * kernel, s=fft_shape)[:size[0], :size[1]]
            rounded = counts.round()
            error = torch.maximum(error, (counts - rounded).abs().amax())
            free = rounded < .5
            rows, columns = torch.arange(size[0], device=device), torch.arange(size[1], device=device)
            free &= (rows[:, None] >= int(lower[0])) & (rows[:, None] <= int(upper[0]))
            free &= (columns[None, :] >= int(lower[1])) & (columns[None, :] <= int(upper[1]))
            contact = torch.fft.irfft2((scene_fft + wall_fft) * contact_kernel, s=fft_shape)[:size[0], :size[1]]
            error = torch.maximum(error, (contact - contact.round()).abs().amax())
            score = torch.where(free, contact.round(), torch.full_like(contact, -1.e15))
            value, position = score.flatten().max(0)
            choices.append((value, position, variant, size[1], mask, mask_origin))
            del counts, rounded, free, rows, columns, contact, score, kernel, contact_kernel
        del scene_fft
        for key in keys[index]:
            if last_use[key] == step:
                cache.pop(key, None)
        queries += 1
        if not choices:
            if index < minimum_count:
                continue
            break
        selected = int(torch.stack([choice[0] for choice in choices]).argmax().item())
        value, position, variant, width, mask, mask_origin = choices[selected]
        value, position = float(value.item()), int(position.item())
        if value < -1.e14:
            if index < minimum_count:
                continue
            break
        x, y = divmod(position, width)
        height, width = mask.shape
        grid[x:x + height, y:y + width] |= mask
        centers[index] = origin + np.asarray((x, y)) * resolution - mask_origin
        variant_ids[index] = variant
    numerical_error = float(error.item())
    if numerical_error >= .1:
        raise RuntimeError(f'Clutter FFT integer separation lost its numerical margin: {numerical_error}.')
    missing = np.flatnonzero(variant_ids < 0)
    prefix = int(missing[0]) if len(missing) else len(polygons)
    return centers, variant_ids, dict(method='gpu_fft_free_grid', maximum_greedy_prefix=prefix,
        input_order_preserved=minimum_count == 0, row_ids_preserved=True, processing_order=order,
        mandatory_count=minimum_count, mandatory_seeded_count=int((variant_ids[:minimum_count] >= 0).sum()),
        seeded_count=int((variant_ids >= 0).sum()), pose_proposals=0, approval_retries=0, placement_queries=queries,
        resolution=float(resolution), grid_shape=shape.tolist(), fft_shape=fft_shape,
        fft_roundtrip_max_error=numerical_error, unique_shape_kernels=unique_kernels,
        peak_cached_shape_kernels=peak_cached_kernels, virtual_reservation=False)


def _clutter_composition_sequence(geometries, rules, rng, target_id, blocker_id):
    """Choose types first; every prefix has one Target and balanced other types.

    The final active count is selected from the geometric capacity of this
    same sequence. Running out of space never substitutes a smaller species.
    """
    _check_inputs(geometries, rules)
    _, groups, _ = _catalog_layout_data(tuple(geometries))
    target_type, blocker_type = (_asset_type(geometries[index]) for index in (target_id, blocker_id))
    if target_type == blocker_type:
        raise ValueError('Clutter roles must use distinct asset types.')
    kinds = len(groups) if rules.all_asset_types else rng.randint(
        rules.min_asset_types, min(rules.max_asset_types, rules.min_objects, len(groups)))
    others = [blocker_type, *rng.sample(sorted(set(groups)-{target_type, blocker_type}), kinds-2)]
    available = {name: list(groups[name]) for name in others}
    for values in available.values():
        rng.shuffle(values)
    available[blocker_type].remove(blocker_id)
    sequence, round_types = [target_id, blocker_id], list(others)
    round_types.remove(blocker_type)
    rng.shuffle(round_types)
    while len(sequence) < rules.max_objects:
        for name in round_types:
            if len(sequence) == rules.max_objects:
                break
            if not available[name]:
                raise ValueError(f'The spawned pool cannot supply the configured count of {name}.')
            sequence.append(available[name].pop())
        round_types = list(others)
        rng.shuffle(round_types)
    return tuple(sequence), (target_type, *others)


def _clutter_protection_polygons(sample, geometries):
    """Protect actual role footprints and their initial ray cones only.

    The XY cone from the camera to each role silhouette is conservative for
    every background height. It preserves the precomputed mutual visibility
    without reserving the old full-depth, full-module rectangle. Areas behind
    the role and on either side remain usable. A relocation pocket is added
    separately after packing; no preselected pushing/pivot route is reserved.
    """
    if sample.camera_pose is None:
        raise ValueError('Clutter construction requires its certified initial viewpoint.')
    pieces = [_collision_polygon(geometries[index], sample.yaws[index])
              + np.asarray(sample.positions[index][:2]) for index in sample.active_ids]
    for index in (sample.target_id, sample.blocker_id):
        geometry, angle = geometries[index], sample.yaws[index]
        if geometry.visual_footprint_vertices:
            points = np.asarray(geometry.visual_footprint_vertices, dtype=np.float64)
            points = points-(np.asarray(geometry.lower[:2])+geometry.upper[:2])*.5
            c, s = math.cos(angle), math.sin(angle)
            points = points @ np.asarray(((c, s), (-s, c))) + sample.positions[index][:2]
        else:
            points = np.asarray(footprint_polygon(geometry, sample.positions[index][:2], angle))
        pieces.append(np.asarray(_convex_hull(tuple(map(tuple, points)) + (tuple(sample.camera_pose[:2]),))))
    return tuple(pieces)


def _solve_clutter_joint_domain(polygon_variants, support_variants, seed_positions, seed_variants,
                                fixed_obstacles, physical_obstacles, pocket_variants, pocket_obb_variants,
                                support_bounds, access_region, gap, *, minimum_count=18,
                                obb_support_bounds=None, role_translation_bounds=None):
    """Generate one shared packing domain with a free virtual relocation pocket.

    A GPU seed supplies separation axes, never fixed XY positions. Two yaw
    variants are supplied for every background. Only a balanced tail prefix
    may be inactive; the first ``minimum_count`` backgrounds are mandatory.
    Two continuous variables may translate roles, camera and visual cones
    together within their pre-certified support/viewpoint domain. Pocket
    placement may reuse the initial visual void, while its final OBB
    must clear A0. Actual convex shapes govern collisions; canonical OBBs
    separately enforce the existing completion predicate's shelf bounds.

    This is one constraint-system solve, without pose proposals, visibility
    approval, scene redraws or retries. A longer prefix is preferred, but the
    first feasible incumbent is sufficient; exact capacity is not claimed.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    from scipy.spatial import ConvexHull
    import time

    started = time.perf_counter()

    def polygon(value):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2 or len(value) < 3 or not np.isfinite(value).all():
            raise ValueError('Joint clutter domains require finite convex XY polygons.')
        return value[ConvexHull(value).vertices]

    variants = tuple(tuple(polygon(part) for part in item) for item in polygon_variants)
    walls = tuple(tuple(polygon(part) for part in item) for item in support_variants)
    pocket = tuple(polygon(part) for part in pocket_variants)
    pocket_walls = tuple(polygon(part) for part in pocket_obb_variants)
    count = len(variants)
    seed, seed_ids = np.asarray(seed_positions, dtype=np.float64), np.asarray(seed_variants, dtype=np.int64)
    support = np.asarray(support_bounds, dtype=np.float64)
    obb_support = support if obb_support_bounds is None else np.asarray(obb_support_bounds, dtype=np.float64)
    access = np.asarray(access_region, dtype=np.float64)
    translation = (np.zeros((2, 2)) if role_translation_bounds is None else
                   np.asarray(role_translation_bounds, dtype=np.float64))
    if (count < minimum_count or minimum_count < 0 or len(walls) != count
            or any(len(parts) != 2 for parts in variants+walls+(pocket, pocket_walls))
            or seed.shape != (count, 2) or seed_ids.shape != (count,)
            or ((seed_ids < -1) | (seed_ids > 1)).any()
            or not np.isfinite(seed[seed_ids >= 0]).all()
            or translation.shape != (2, 2) or not np.isfinite(translation).all()
            or (translation[:, 0] > translation[:, 1]).any()
            or any(bounds.shape != (4,) or not np.isfinite(bounds).all()
                   or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
                   for bounds in (support, obb_support, access))
            or not math.isfinite(gap) or gap < 0.):
        raise ValueError('Invalid joint clutter variants, seed, count or support/A0 bounds.')
    fixed = tuple(polygon(part) for part in fixed_obstacles)
    physical = tuple(polygon(part) for part in physical_obstacles)
    access_polygon = np.asarray(rectangle_polygon(tuple(access)))
    all_variants, all_walls = variants+(pocket,), walls+(pocket_walls,)
    bodies, virtual = count+1, count
    normals = np.stack((np.cos(np.arange(8)*math.pi/4), np.sin(np.arange(8)*math.pi/4)), axis=1)
    projections = np.asarray([[(part @ normals.T).max(axis=0) for part in parts] for parts in all_variants])
    base, delta = projections[:, 0], projections[:, 1]-projections[:, 0]
    yaw_offset, active_offset = 2*bodies, 3*bodies
    role_offset = active_offset+count
    variables = role_offset+2
    rows, columns, values, upper_rows, lower_rows = [], [], [], [], []
    guard, effective_gap = 1.e-5, float(gap)+1.e-5
    # Preserve a positive wall/viewpoint margin despite HiGHS feasibility
    # tolerances. Zero-width/fixed role coordinates remain exactly fixed.
    translation = translation.copy()
    translation_inset = np.where(translation[:, 1]-translation[:, 0] >= 2*guard, guard, 0.)
    translation[:, 0] += translation_inset
    translation[:, 1] -= translation_inset
    scale = max(1., float(np.abs(support).max()), float(np.abs(obb_support).max()), float(np.abs(translation).max()),
                *(float(np.abs(part).max()) for part in fixed+physical+(access_polygon,)),
                *(float(np.abs(part).max()) for parts in all_variants for part in parts))
    big_m = 8.*scale

    def constraint(coefficients, upper, lower=-np.inf):
        index = len(upper_rows)
        for key, value in coefficients.items():
            if value:
                rows.append(index)
                columns.append(key)
                values.append(value)
        upper_rows.append(upper)
        lower_rows.append(lower)

    def safe_axis(first, second):
        direction = first.mean(axis=0)-second.mean(axis=0)
        axes = normals
        if np.linalg.norm(direction) > 1.e-12:
            direction /= np.linalg.norm(direction)
            axes = np.concatenate((axes, direction[None, :], -direction[None, :]))
        margins = (first @ axes.T).min(axis=0)-(second @ axes.T).max(axis=0)
        if float(margins.max()) < gap-2.e-7:
            edges = np.concatenate((np.roll(first, -1, axis=0)-first,
                                    np.roll(second, -1, axis=0)-second))
            lengths = np.linalg.norm(edges, axis=1)
            edges, lengths = edges[lengths > 1.e-12], lengths[lengths > 1.e-12]
            extra = np.stack((-edges[:, 1], edges[:, 0]), axis=1)/lengths[:, None]
            distance = ((first[:, None, :]-second[None, :, :])**2).sum(axis=-1)
            a, b = np.unravel_index(int(distance.argmin()), distance.shape)
            direction = first[a]-second[b]
            if np.linalg.norm(direction) > 1.e-12:
                direction /= np.linalg.norm(direction)
                extra = np.concatenate((extra, direction[None, :], -direction[None, :]))
            axes = np.concatenate((axes, extra, -extra))
            margins = (first @ axes.T).min(axis=0)-(second @ axes.T).max(axis=0)
        index = int(margins.argmax())
        if margins[index] < gap-2.e-7:
            raise ValueError('The clutter graph seed overlaps an object or protected initial view.')
        return axes[index]

    def support_pair(index, normal, parts=None):
        parts = all_variants[index] if parts is None else parts
        h = [float((part @ normal).max()) for part in parts]
        return h[0], h[1]-h[0]

    for index in range(bodies):
        for parts, bounds in ((all_variants[index], support), (all_walls[index], obb_support)):
            for axis, limit in ((0, bounds[1]), (2, bounds[3]), (4, -bounds[0]), (6, -bounds[2])):
                normal = normals[axis]
                h, change = support_pair(index, normal, parts)
                constraint({2*index: normal[0], 2*index+1: normal[1], yaw_offset+index: change}, limit-h-guard)
        for other in range(index+1, bodies):
            if other < count and seed_ids[index] >= 0 and seed_ids[other] >= 0:
                normal = safe_axis(variants[index][seed_ids[index]]+seed[index],
                                   variants[other][seed_ids[other]]+seed[other])
                hi, di = support_pair(index, -normal)
                hj, dj = support_pair(other, normal)
                constraint({2*index: -normal[0], 2*index+1: -normal[1],
                            2*other: normal[0], 2*other+1: normal[1],
                            yaw_offset+index: di, yaw_offset+other: dj,
                            active_offset+index: big_m, active_offset+other: big_m},
                           2*big_m-effective_gap-hi-hj)
            else:
                choices = []
                for axis, normal in enumerate(normals):
                    choice, variables = variables, variables+1
                    choices.append(choice)
                    negative = (axis+4) % 8
                    constraint({2*index: -normal[0], 2*index+1: -normal[1],
                                2*other: normal[0], 2*other+1: normal[1],
                                yaw_offset+index: delta[index, negative], yaw_offset+other: delta[other, axis],
                                choice: big_m}, big_m-effective_gap-base[index, negative]-base[other, axis])
                coefficients = {choice: 1. for choice in choices}
                coefficients[active_offset+index] = -1.
                if other < count:
                    coefficients[active_offset+other] = -1.
                constraint(coefficients, np.inf, -1. if other < count else 0.)
        obstacles = (tuple((part, None) for part in fixed) if index < count else
                     tuple((part, None) for part in physical)+((access_polygon, pocket_walls),))
        for obstacle, override in obstacles:
            if index < count and seed_ids[index] >= 0:
                normal = safe_axis(variants[index][seed_ids[index]]+seed[index], obstacle)
                h, change = support_pair(index, -normal)
                constraint({2*index: -normal[0], 2*index+1: -normal[1], yaw_offset+index: change,
                            active_offset+index: big_m, role_offset: normal[0], role_offset+1: normal[1]},
                           big_m-effective_gap-h-float((obstacle @ normal).max()))
            else:
                choices = []
                for normal in normals:
                    choice, variables = variables, variables+1
                    choices.append(choice)
                    h, change = support_pair(index, -normal, override)
                    constraint({2*index: -normal[0], 2*index+1: -normal[1], yaw_offset+index: change,
                                choice: big_m, role_offset: min(normal[0], 0.) if override is not None else normal[0],
                                role_offset+1: normal[1]}, big_m-effective_gap-h-float((obstacle @ normal).max()))
                coefficients = {choice: 1. for choice in choices}
                if index < count:
                    coefficients[active_offset+index] = -1.
                constraint(coefficients, np.inf, 0. if index < count else 1.)
    for index in range(count-1):
        constraint({active_offset+index: -1., active_offset+index+1: 1.}, 0.)
    lower, upper = np.zeros(variables), np.ones(variables)
    for index in range(bodies):
        lower[2*index:2*index+2] = support[[0, 2]]
        upper[2*index:2*index+2] = support[[1, 3]]
    integral = np.zeros(variables, dtype=np.int32)
    integral[yaw_offset:] = 1
    integral[role_offset:role_offset+2] = 0
    lower[role_offset:role_offset+2] = translation[:, 0]
    upper[role_offset:role_offset+2] = translation[:, 1]
    for index in np.flatnonzero(seed_ids >= 0):
        lower[yaw_offset+index] = upper[yaw_offset+index] = seed_ids[index]
    lower[active_offset:active_offset+minimum_count] = 1.
    objective = np.zeros(variables)
    objective[active_offset:active_offset+count] = -1.
    matrix = coo_matrix((values, (rows, columns)), shape=(len(upper_rows), variables)).tocsc()
    row_lower, row_upper = np.asarray(lower_rows), np.asarray(upper_rows)
    prepared = time.perf_counter()-started
    result = milp(objective, integrality=integral, bounds=Bounds(lower, upper),
                  constraints=LinearConstraint(matrix, row_lower, row_upper),
                  options={'presolve': True, 'mip_rel_gap': 2.0})
    if result.x is None:
        raise RuntimeError(f'Joint clutter domain is infeasible: minimum={minimum_count}, requested={count}, '
                           f'seeded={int((seed_ids >= 0).sum())}, status={result.status}: {result.message}')
    solution = result.x.copy()
    solution[integral != 0] = np.rint(solution[integral != 0])
    residual = matrix @ solution
    violation = max(0., float(np.max(residual-row_upper)), float(np.max(row_lower-residual)))
    if violation > .9*guard:
        raise RuntimeError(f'Joint clutter solver exceeded its numerical gap guard: {violation:.3g}m.')
    active = solution[active_offset:active_offset+count].astype(bool)
    selected = int(active.sum())
    if selected < minimum_count or active[:selected].sum() != selected:
        raise RuntimeError('Joint clutter solver returned an invalid mandatory/balanced prefix.')
    centers = solution[:2*bodies].reshape(bodies, 2)
    variant_ids = solution[yaw_offset:yaw_offset+bodies].astype(np.int64)
    provenance = dict(method='joint_separation_domain', minimum_background=minimum_count,
        requested_background=count, active_background=selected, seeded_background=int((seed_ids >= 0).sum()),
        variables=variables, unfixed_binary_variables=int(((integral != 0) & (upper > lower)).sum()),
        constraints=len(upper_rows), preparation_seconds=prepared,
        solve_seconds=time.perf_counter()-started-prepared, numerical_gap_guard=guard,
        maximum_constraint_violation=violation, scene_redraws=0, approval_retries=0,
        optimal_capacity_claimed=False, solver_status=int(result.status),
        role_translation=tuple(map(float, solution[role_offset:role_offset+2])),
        role_translation_inset=tuple(map(float, translation_inset)), numerical_wall_guard=guard)
    return (centers[:selected].copy(), variant_ids[:selected].copy(), centers[virtual].copy(),
            int(variant_ids[virtual]), provenance)


def _clutter_initial_cells(family, geometries, rules, target, blocker, front_range, obb_tolerance=.005,
                           lateral_half_width=.35):
    """Initial support/camera domains; a previous motion reserve is irrelevant.

    The old role catalogue supplies only mutual visibility and initial A0
    certificates. Its background slab and complete motion bounds are dropped
    here, rather than treating them as a capacity certificate for larger clutter.
    """
    support = (rules.x_range[0]+rules.edge_margin, rules.x_range[1]-rules.edge_margin,
               rules.y_range[0]+rules.edge_margin, rules.y_range[1]-rules.edge_margin)
    local_t = _collision_polygon(geometries[target], family['target_yaw'])
    local_b = _collision_polygon(geometries[blocker], family['blocker_yaw'])
    cells, weights, seen = [], [], set()
    for source in family['cells']:
        lo, hi = source['parameter_bounds']
        low_angle, high_angle = source['yaw_bounds']
        key = (lo, hi, low_angle, high_angle)
        if key in seen:
            continue
        seen.add(key)
        half = max(abs(low_angle), abs(high_angle))
        points = np.concatenate((local_t, local_b+(family['blocker_x'], lo),
                                 local_b+(family['blocker_x'], hi)))
        bounds = _certified_rotated_bounds(points, half)
        obb = _certified_rotated_bounds(np.asarray((
            *footprint_polygon(geometries[target], (0., 0.), family['target_yaw']),
            *footprint_polygon(geometries[blocker], (family['blocker_x'], lo), family['blocker_yaw']),
            *footprint_polygon(geometries[blocker], (family['blocker_x'], hi), family['blocker_yaw']))), half)
        camera = _certified_rotated_bounds(np.asarray(family['camera_pose'])[None, :2], half)
        x = (max(support[0]-bounds[0], rules.x_range[0]-obb_tolerance-obb[0],
                 rules.x_range[1]+front_range[0]-camera[0]),
             min(support[1]-bounds[1], rules.x_range[1]+obb_tolerance-obb[1],
                 rules.x_range[1]+front_range[1]-camera[1]))
        y = (max(support[2]-bounds[2], rules.y_range[0]-obb_tolerance-obb[2],
                 -lateral_half_width-camera[2]),
             min(support[3]-bounds[3], rules.y_range[1]+obb_tolerance-obb[3],
                 lateral_half_width-camera[3]))
        if x[0] < x[1] and y[0] < y[1]:
            cells.append(dict(parameter_bounds=(lo, hi), yaw_bounds=(low_angle, high_angle),
                              x_translation_bounds=x, y_translation_bounds=y,
                              initial_bounds=bounds, blockage_lower_bound=source['blockage_lower_bound']))
            weights.append((hi-lo)*(high_angle-low_angle)*(x[1]-x[0])*(y[1]-y[0]))
    if not cells:
        raise ValueError('Role/view certificate has no initial shelf/camera translation domain.')
    return tuple(cells), np.cumsum(weights)


def _sample_clutter_roles(catalog, geometries, rules, rng, target, blocker, front_range=(.55, .95),
                          obb_tolerance=.005, lateral_half_width=.35):
    """Draw one role/view relation, with no reserved route or separate group."""
    pair = (_asset_type(geometries[target]), _asset_type(geometries[blocker]))
    families = tuple(family for family in catalog.families[pair] if not family.get('fixed_objects'))
    family = rng.choice(families)
    cells, cumulative = _clutter_initial_cells(family, geometries, rules, target, blocker,
                                              front_range, obb_tolerance, lateral_half_width)
    cell = cells[int(np.searchsorted(cumulative, rng.random()*cumulative[-1], side='right'))]
    parameter, angle = rng.uniform(*cell['parameter_bounds']), rng.uniform(*cell['yaw_bounds'])
    # A front/side anchor consolidates free space before the shared clutter is
    # mixed. These are correlated continuous parameters, not sampled scenes
    # that are approved/rejected by a later capacity/visibility test.
    xlo, xhi = cell['x_translation_bounds']
    ylo, yhi = cell['y_translation_bounds']
    x = rng.uniform(max(xlo, xhi-.02), xhi)
    side = -1 if rng.random() < .5 else 1
    y = (rng.uniform(ylo, min(ylo+.02, yhi)) if side < 0
         else rng.uniform(max(ylo, yhi-.02), yhi))
    shift = np.array((x, y))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s), (s, c)))
    centers = {target: shift, blocker: rotation @ np.array((family['blocker_x'], parameter))+shift}
    positions = {index: (*map(float, center), rules.shelf_top-geometries[index].lower[2])
                 for index, center in centers.items()}
    yaws = {target: family['target_yaw']+angle, blocker: family['blocker_yaw']+angle}
    camera = np.asarray(family['camera_pose'], dtype=np.float64).copy()
    camera[:2] = rotation @ camera[:2]+shift
    w, qx, qy, qz = camera[3:]
    ca, sa = math.cos(angle*.5), math.sin(angle*.5)
    camera[3:] = (ca*w-sa*qz, ca*qx-sa*qy, ca*qy+sa*qx, ca*qz+sa*w)
    visibility = catalog.load_visibility(family)
    segment = int(np.searchsorted(visibility['target_visibility_events'], parameter, side='right'))-1
    visible = 1.-visibility['target_occluded_counts'][segment]/visibility['target_pixel_count']
    access_half = rules.hand_width*.5+rules.approach_margin
    front = positions[target][0]+_half_extents(geometries[target], yaws[target])[0]+rules.approach_margin
    access = (front, rules.x_range[1], y-access_half, y+access_half)
    provenance = dict(visibility_fingerprint=visibility['fingerprint'], parameter=parameter,
        common_yaw=angle, predicted_role_visibility=(visible, visibility['blocker_visible_fraction_structural']),
        live_visibility_measured=False, full_route_reserved=False, initial_role_bounds=cell['initial_bounds'],
        role_translation_bounds=((xlo-x, xhi-x), (ylo-y, yhi-y)))
    return SceneSample((target, blocker), target, blocker, positions, yaws, access,
        tuple(map(float, centers[blocker])), clearance_kind='clutter_roles',
        camera_pose=tuple(map(float, camera)), parameter_certificate=provenance)


def sample_clutter_episode(catalog, geometries, rules, rng, target, blocker, *, device,
                           front_range=(.55, .95), obb_tolerance=.005, lateral_half_width=.35):
    """Construct one connected clutter domain; solve positions before committing.

    All physical objects share one packing problem. The GPU footprint seed
    supplies separating relations, then one joint constraint system positions
    every background and an empty Blocker destination. No whole movement path,
    role-only slab, camera search or rejected-scene redraw belongs to this rule.
    """
    sequence, types = _clutter_composition_sequence(geometries, rules, rng, target, blocker)
    desired = rng.randint(rules.min_objects, rules.max_objects)
    background = sequence[2:desired]
    roles = _sample_clutter_roles(catalog, geometries, rules, rng, target, blocker,
                                  front_range, obb_tolerance, lateral_half_width)
    physical = tuple(_collision_polygon(geometries[index], roles.yaws[index]) + roles.positions[index][:2]
                     for index in roles.active_ids)
    fixed = _clutter_protection_polygons(roles, geometries)
    support = (rules.x_range[0]+rules.edge_margin, rules.x_range[1]-rules.edge_margin,
               rules.y_range[0]+rules.edge_margin, rules.y_range[1]-rules.edge_margin)
    obb_support = (rules.x_range[0]-obb_tolerance, rules.x_range[1]+obb_tolerance,
                   rules.y_range[0]-obb_tolerance, rules.y_range[1]+obb_tolerance)
    base_yaws = []
    for index in background:
        geometry = geometries[index]
        _, _, angle = _certified_footprint_profile(tuple(geometry.lower), tuple(geometry.upper),
                                                   tuple(geometry.collision_footprint_vertices))
        base_yaws.append(angle+math.pi*.5+rng.randrange(2)*math.pi+rng.uniform(-.001, .001))
    # Every type appears in this balanced head. Choosing its copy uniformly
    # gives each surrounding type continuous full-circle orientation support;
    # other yaws remain coupled to the shallow shelf's packing capacity.
    free = rng.randrange(min(rules.min_objects-2, len(background)))
    base_yaws[free] = rng.uniform(-math.pi, math.pi)
    turns = tuple((0, 0) if slot == free else (0, 1) for slot in range(len(background)))
    polygons = tuple(tuple(_collision_polygon(geometries[index], angle+turn*math.pi*.5) for turn in allowed)
                     for index, angle, allowed in zip(background, base_yaws, turns))
    obbs = tuple(tuple(np.asarray(footprint_polygon(geometries[index], (0., 0.), angle+turn*math.pi*.5))
                       for turn in allowed) for index, angle, allowed in zip(background, base_yaws, turns))
    b_geometry = geometries[blocker]
    _, _, pocket_angle = _certified_footprint_profile(tuple(b_geometry.lower), tuple(b_geometry.upper),
                                                    tuple(b_geometry.collision_footprint_vertices))
    pocket_angle += math.pi*.5
    pocket_polygons = tuple(_collision_polygon(b_geometry, pocket_angle+turn*math.pi*.5) for turn in (0, 1))
    pocket_obbs = tuple(np.asarray(footprint_polygon(b_geometry, (0., 0.), pocket_angle+turn*math.pi*.5))
                        for turn in (0, 1))
    seed_centers, seed_variants, seed_stats = _clutter_gpu_seed(
        polygons, obbs, fixed, support, rules.gap, device=device, obb_support_bounds=obb_support,
        minimum_count=rules.min_objects-2)
    minimum_bg = rules.min_objects-2
    missing_tail = np.flatnonzero(np.asarray(seed_variants[minimum_bg:]) < 0)
    # Optional counts are bounded by the seed's known separating domain. Do
    # not add hundreds of unnecessary branch variables for an unpacked tail;
    # mandatory objects remain in the single joint solve even if unseeded.
    limit = minimum_bg+int(missing_tail[0]) if len(missing_tail) else len(background)
    centers, variants, pocket_center, pocket_variant, joint_stats = _solve_clutter_joint_domain(
        polygons[:limit], obbs[:limit], seed_centers[:limit], seed_variants[:limit],
        fixed, physical, pocket_polygons, pocket_obbs,
        support, roles.access_region, rules.gap, minimum_count=minimum_bg,
        obb_support_bounds=obb_support,
        role_translation_bounds=roles.parameter_certificate['role_translation_bounds'])
    translation = np.asarray(joint_stats['role_translation'])
    # Move the complete role/view assembly together. Its silhouette ratios
    # are unchanged, while the shelf's front edge stays fixed in world space.
    role_positions = {index: (position[0]+translation[0], position[1]+translation[1], position[2])
                      for index, position in roles.positions.items()}
    camera = np.asarray(roles.camera_pose).copy()
    camera[:2] += translation
    a0 = roles.access_region
    roles = replace(roles, positions=role_positions, camera_pose=tuple(map(float, camera)),
        access_region=(a0[0]+translation[0], a0[1], a0[2]+translation[1], a0[3]+translation[1]))
    fixed = tuple(polygon+translation for polygon in fixed)
    active_bg = background[:len(centers)]
    actual_polygons = tuple(polygons[index][int(variant)] for index, variant in enumerate(variants))
    actual_obbs = tuple(obbs[index][int(variant)] for index, variant in enumerate(variants))
    center_limits = np.asarray([(obb_support[0]-polygon[:, 0].min(), obb_support[1]-polygon[:, 0].max(),
                                 obb_support[2]-polygon[:, 1].min(), obb_support[3]-polygon[:, 1].max())
                                for polygon in actual_obbs])
    pocket_polygon = pocket_polygons[int(pocket_variant)]+np.asarray(pocket_center)
    centers, mix_stats = _mix_clutter_positions(actual_polygons, centers, support,
        (*fixed, pocket_polygon), rules.gap, rng, center_bounds=center_limits)
    positions, yaws = dict(roles.positions), dict(roles.yaws)
    for slot, index in enumerate(active_bg):
        positions[index] = (*map(float, centers[slot]), rules.shelf_top-geometries[index].lower[2])
        yaws[index] = base_yaws[slot]+turns[slot][int(variants[slot])]*math.pi*.5
    lo, hi = pocket_polygon.min(0), pocket_polygon.max(0)
    provenance = dict(roles.parameter_certificate, method='joint_clutter', type_composition=types,
        desired_count=desired, active_count=2+len(active_bg), gpu_seed=seed_stats,
        joint_packing=joint_stats, mixing=mix_stats,
        relocation_pocket=dict(center=tuple(map(float, pocket_center)),
            yaw=float(pocket_angle+int(pocket_variant)*math.pi*.5), route_reserved=False))
    return replace(roles, active_ids=(target, blocker, *active_bg), positions=positions, yaws=yaws,
        clearance_witness=tuple(map(float, pocket_center)), clearance_kind='joint_clutter_free_pocket',
        clearance_path=(), reserved_regions=((float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])),),
        parameter_certificate=provenance)


def precompute_certified_role_family(geometries, upright_quats, sensor, target_id, blocker_id,
                                    target_yaw, blocker_yaw, rules, *,
                                    yaw_half_width=math.radians(2.5), cache_directory=None,
                                    camera_lateral_offset=0.):
    """Compile one continuous role family before reset, without pose proposals.

    Target and Blocker OBB centers start at (0,0) and (bx,0). Their actual
    convex footprints have a separating X support gap; the scalar parameter
    translates Blocker along local +Y. A common rigid assembly transform then
    preserves the inverse mesh certificate. This compiles only the requested
    relative yaws: it does not certify their full-circle marginal coverage.

    The inverse solve uses an identity camera at the origin. Blocker root pose
    is constructed directly in that frame, using Blocker geometry/yaw only.
    In particular, no subtraction of Target-dependent world coordinates can
    perturb its float32 bytes or prevent exact Blocker subcache reuse. Native
    scene poses are subsequently packed as usual; the solver's pixel and
    parameter guards remain necessary across float32 world-frame conversion.
    Empty visibility or A0/maneuver domains are startup errors, never redraws.
    """
    count = len(geometries)
    if (target_id == blocker_id or any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
            or not 0 <= index < count for index in (target_id, blocker_id))
            or sensor.object_count != count
            or not all(math.isfinite(value) for value in (target_yaw, blocker_yaw, yaw_half_width, camera_lateral_offset))
            or not 0. < yaw_half_width < math.pi/2):
        raise ValueError("A certified role family needs distinct pool roles, matching sensor capacity and finite yaws.")
    upright = torch.as_tensor(upright_quats).detach().to(device='cpu', dtype=torch.float64)
    if (upright.shape != (count, 4) or not bool(torch.isfinite(upright).all())
            or not bool(((upright.norm(dim=-1)-1.).abs() <= 1.e-5).all())):
        raise ValueError("Certified role upright quaternions must be finite unit wxyz values [pool,4].")
    if (not math.isfinite(rules.gap) or not math.isfinite(rules.max_neighbor_gap)
            or not 0. <= rules.gap <= rules.max_neighbor_gap):
        raise ValueError("Certified role gaps must be finite, nonnegative and ordered.")
    height = float(sensor.cfg.height_above_shelf[0])
    if not math.isfinite(height) or height <= 0.:
        raise ValueError("A certified front camera needs a finite positive height above the shelf.")
    target, blocker = geometries[target_id], geometries[blocker_id]
    t_polygon = _collision_polygon(target, target_yaw)
    b_polygon = _collision_polygon(blocker, blocker_yaw)
    gap = (rules.gap+rules.max_neighbor_gap)*.5
    bx = float(t_polygon[:, 0].max()-b_polygon[:, 0].min()+gap)
    b_visual_front = float(np.asarray(footprint_polygon(blocker, (0., 0.), blocker_yaw))[:, 0].max())
    # Camera's local axes are +worldY, -worldZ, -worldX. Its distance from
    # Blocker's front OBB support is fixed, independent of either Target.
    camera_from_blocker = b_visual_front+.765
    camera_x = bx+camera_from_blocker
    camera_pose = (camera_x, float(camera_lateral_offset), rules.shelf_top+height, .5, -.5, -.5, .5)
    centers = []
    for geometry, angle in ((target, target_yaw), (blocker, blocker_yaw)):
        center = (np.asarray(geometry.lower[:2])+geometry.upper[:2])*.5
        c, s = math.cos(angle), math.sin(angle)
        centers.append(np.array((c*center[0]-s*center[1], s*center[0]+c*center[1])))
    t_center, b_center = centers
    device = sensor.env.device
    poses = torch.zeros((count, 7), device=device, dtype=torch.float32)
    poses[:, 3] = 1.
    # Compute the two canonical root positions independently in float64;
    # Blocker depth never contains bx, which depends on Target geometry/yaw.
    role_positions = ((-t_center[1]-camera_lateral_offset, height+target.lower[2], camera_x+t_center[0]),
                      (-b_center[1]-camera_lateral_offset, height+blocker.lower[2], camera_from_blocker+b_center[0]))
    local_yaws = torch.tensor(((math.cos(target_yaw*.5), 0., 0., math.sin(target_yaw*.5)),
                              (math.cos(blocker_yaw*.5), 0., 0., math.sin(blocker_yaw*.5))),
                             dtype=torch.float64)
    camera_inverse = upright.new_tensor((.5, .5, .5, -.5)).expand(2, -1)
    role_quats = quat_mul(camera_inverse, quat_mul(local_yaws, upright[[target_id, blocker_id]]))
    poses[[target_id, blocker_id], :3] = poses.new_tensor(role_positions)
    poses[[target_id, blocker_id], 3:] = role_quats.to(device=device, dtype=poses.dtype)
    identity_camera = poses.new_tensor((0., 0., 0., 1., 0., 0., 0.))
    visibility = sensor.precompute_translation_domain(
        object_poses=poses, target_id=target_id, blocker_id=blocker_id,
        camera_pose=identity_camera, translation=(1., 0., 0.),
        parameter_bounds=(-.16, .16), cache_directory=cache_directory)
    cells = _certify_role_translation_cells(
        geometries, target_id, blocker_id, target_yaw, blocker_yaw, bx,
        visibility['valid_intervals'], rules, yaw_half_width=yaw_half_width,
        camera_eye=camera_pose[:3])
    if not cells:
        raise ValueError("The requested role family has no certified continuous A0/maneuver domain at startup.")
    return dict(target_id=int(target_id), blocker_id=int(blocker_id), target_yaw=float(target_yaw),
                blocker_yaw=float(blocker_yaw), blocker_x=bx, camera_pose=camera_pose,
                visibility_domain=visibility, role_cells=cells,
                canonical_role_poses=tuple(tuple(map(float, row)) for row in torch.cat((
                    torch.tensor(role_positions, dtype=torch.float64), role_quats), -1).tolist()),
                yaw_half_width=float(yaw_half_width), role_support_gap=float(gap))


def _catalog_scene_cells(family, rules, capacity, *, camera_front_range=(.55, .95)):
    """Compile shared parameter bounds once, valid for every composition."""
    xmin, xmax, ymin, ymax = capacity['support_bounds']
    padding = rules.max_neighbor_gap+capacity['minimum_background_slab_width']
    cells, weights = [], []
    for source in family['role_cells']:
        bounds = source['module_bounds']
        camera_low, camera_high = source['camera_x_bounds_over_yaw']
        x = (max(xmin-bounds[0], rules.x_range[1]+camera_front_range[0]-camera_low),
             min(xmax-bounds[1], rules.x_range[1]+camera_front_range[1]-camera_high))
        if 'translation_x_bounds' in source:
            x = (max(x[0], source['translation_x_bounds'][0]), min(x[1], source['translation_x_bounds'][1]))
        if bounds[3]-bounds[2] >= capacity['maximum_module_width']-1.e-10:
            continue
        for side in (-1, 1):
            y = ((ymin+padding-bounds[2], ymax-bounds[3]) if side < 0
                 else (ymin-bounds[2], ymax-padding-bounds[3]))
            if 'translation_y_bounds' in source:
                y = (max(y[0], source['translation_y_bounds'][0]), min(y[1], source['translation_y_bounds'][1]))
            if x[0] < x[1] and y[0] < y[1]:
                cells.append(dict(source, x_translation_bounds=x, y_translation_bounds=y, background_side=side))
                lo, hi = source['parameter_bounds']
                a, b = source['yaw_bounds']
                weights.append((hi-lo)*(b-a)*(x[1]-x[0])*(y[1]-y[0]))
    return cells, np.cumsum(weights).tolist()


class CertifiedSceneCatalog:
    """Startup-checked parameter grammar; no completed scenes are stored.

    Only the selected visibility curve is read at a scene change. Per-reset
    work selects categorical metadata and draws continuous parameters; all
    mesh/A0/support/coverage approval belongs to compilation before training.
    """

    def __init__(self, manifest, sensor, cache_directory):
        self.sensor, self.directory = sensor, Path(cache_directory)
        self.families = {}
        for pair in manifest['pairs']:
            key = (pair['target_asset'], pair['blocker_asset'])
            values = pair['families']
            normal = [dict(target_yaw=value['target_yaw'], blocker_yaw=value['blocker_yaw'],
                           role_cells=value['cells']) for value in values if not value.get('fixed_objects')]
            if any(certified_role_yaw_gaps(normal).values()):
                raise ValueError(f"Certified catalog has incomplete role yaw support for {key}.")
            for value in values:
                value['cumulative'] = np.asarray(value['cumulative'], dtype=np.float64)
                if (len(value['cells']) != len(value['cumulative']) or not len(value['cells'])
                        or not np.isfinite(value['cumulative']).all()
                        or not (np.diff(np.r_[0., value['cumulative']]) > 0.).all()):
                    raise ValueError(f"Certified catalog has invalid cell volumes for {key}.")
                path = self.directory / f"{value['visibility_fingerprint']}.json"
                if not path.is_file():
                    raise ValueError(f"Certified visibility cache is missing: {path}")
                value['visibility_cache_file'] = str(path)
            self.families[key] = tuple(values)
        source = manifest['source']
        names = tuple(asset['asset'] for asset in source['assets'])
        expected = {(target, blocker) for target in names for blocker in names if target != blocker
            and (source['target_asset'] is None or source['target_asset'] == target)
            and (source['blocker_asset'] is None or source['blocker_asset'] == blocker)}
        if set(self.families) != expected:
            raise ValueError("Certified catalog is missing configured ordered asset pairs.")
        if not manifest['guarded_family_count'] or not any(
                family.get('fixed_objects') and family.get('pivot') for values in self.families.values() for family in values):
            raise ValueError("Certified catalog must include a pivot-required parameter domain.")

    def load_visibility(self, family):
        domain = self.sensor._load_translation_cache(Path(family['visibility_cache_file']),
                                                     family['visibility_fingerprint'])
        if domain is None:
            raise RuntimeError("Certified visibility cache changed or became unreadable after startup.")
        return domain


def _certified_catalog_source(env, runtime, *, reference_count=None):
    """Content identity of geometry, sensor settings and the generation rule."""
    sensor, geoms = runtime.vision, env.spawn_geometries
    _, groups, _ = _catalog_layout_data(tuple(geoms))
    identity = torch.tensor((0., 0., 0., 1., 0., 0., 0.), device=env.device)
    assets = []
    for name, ids in sorted(groups.items()):
        index = ids[0]
        _, _, visual_hash = sensor._camera_mesh(index, identity, identity)
        geometry = geoms[index]
        # Clutter reuses only the role/view/A0 relations of the reference
        # grammar. Its canonical supply is independent of the live body pool;
        # this is not a capacity certificate for the new physical scene.
        copies = len(ids) if reference_count is None else min(len(ids), math.ceil(
            (reference_count-1)/(env.spawn_rules.min_asset_types-1)))
        assets.append(dict(asset=name, copies=copies, lower=geometry.lower, upper=geometry.upper,
            collision=geometry.collision_footprint_vertices, visual_mesh_sha256=visual_hash,
            upright=env.task_scene_data.upright[index].tolist()))
    rules = {item.name: getattr(env.spawn_rules, item.name) for item in fields(env.spawn_rules)
             if item.name not in ('generation_mode', 'parameter_cache_directory', 'max_attempts')}
    if reference_count is not None:
        rules.update(min_objects=reference_count, max_objects=reference_count)
    return dict(version='continuous_scene_grammar_v4', assets=assets, rules=rules,
                # Observation sampling changes neither the certified domain
                # nor its full-resolution visibility/pixel guards. Reuse the
                # existing role catalog when only observation resolution varies.
                vision={item.name: getattr(sensor.cfg, item.name) for item in fields(sensor.cfg)
                        if item.name not in ('observation_ray_width', 'observation_ray_height',
                                             'observation_min_depth_pixels')},
                target_asset=env.cfg.target_asset, blocker_asset=env.cfg.blocker_asset)


def _bundled_catalog_directory(fingerprint, cache_directory, *, bundle_directory=None):
    """Restore the Git-tracked catalog once, without replacing generated caches."""
    bundle = (Path(bundle_directory) if bundle_directory is not None
              else Path(__file__).resolve().parents[1] / "data/initial_scene_catalog")
    try:
        manifest = json.loads((bundle / "manifest.json").read_text())
        if manifest['schema'] != 1:
            raise ValueError("Unsupported bundle schema.")
        if manifest['fingerprint'] != fingerprint:
            return None  # Changed research settings retain the normal compiler path.
        checksum = manifest['archive_sha256']
        if len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
            raise ValueError("Invalid bundle checksum.")
        directory = Path(cache_directory) / f"bundled_{checksum}"
        marker = directory / '.bundle.sha256'
        if marker.is_file() and marker.read_text().strip() == checksum:
            return directory
        print("[Scene grammar] restoring Git-bundled initial catalog (first use only)", flush=True)
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.restore-', dir=directory.parent) as temporary:
            temporary = Path(temporary)
            archive, digest = temporary / 'catalog.tar.gz', hashlib.sha256()
            with archive.open('wb') as output:
                for part in manifest['parts']:
                    name, part_digest, size = part['name'], hashlib.sha256(), 0
                    if Path(name).name != name:
                        raise ValueError("Bundle part must be a basename.")
                    with (bundle / name).open('rb') as source:
                        while block := source.read(1024 * 1024):
                            part_digest.update(block)
                            digest.update(block)
                            size += len(block)
                            output.write(block)
                    if size != part['size'] or part_digest.hexdigest() != part['sha256']:
                        raise ValueError(f"Bundle part checksum mismatch: {name}")
            if digest.hexdigest() != checksum:
                raise ValueError("Bundle archive checksum mismatch.")
            staged = temporary / 'parameter_domains'
            staged.mkdir()
            names, unpacked = set(), 0
            catalog_name = f'catalog_{fingerprint}.json'
            with tarfile.open(archive, mode='r|gz') as stream:
                for member in stream:
                    name = member.name
                    is_domain = (len(name) == 69 and name.endswith('.json')
                                 and all(c in '0123456789abcdef' for c in name[:-5]))
                    if (not member.isfile() or Path(name).name != name or name in names
                            or not (name == catalog_name or is_domain)):
                        raise ValueError(f"Invalid bundled cache member: {name}")
                    names.add(name)
                    unpacked += member.size
                    if len(names) > manifest['file_count'] or unpacked > manifest['unpacked_bytes']:
                        raise ValueError("Bundle exceeds its declared file count or size.")
                    with stream.extractfile(member) as source, (staged / name).open('wb') as output:
                        shutil.copyfileobj(source, output)
            if (catalog_name not in names or len(names) != manifest['file_count']
                    or unpacked != manifest['unpacked_bytes']):
                raise ValueError("Bundle is missing declared cache files.")
            (staged / marker.name).write_text(checksum + '\n')
            try:
                staged.rename(directory)
            except OSError:
                if not marker.is_file() or marker.read_text().strip() != checksum:
                    raise  # Another process may have finished the same restore.
        return directory
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        raise RuntimeError(f"Git-bundled initial catalog could not be restored: {error}. "
                           "Update the complete parallel project with git pull.") from error


def initialize_certified_catalog(env, runtime, cache_directory, *, reference_count=None):
    """Build/load the continuous grammar before the first episode is sampled.

    Fixed relative-yaw/view relations are compiled offline. Failed relations
    are not sampled at reset, and missing marginal coverage is a startup
    error. The cache contains scalar domains and partial role relations,
    never a bank of complete object arrangements.
    """
    directory = Path(cache_directory)
    source = _certified_catalog_source(env, runtime, reference_count=reference_count)
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    expected = getattr(env.cfg, "initial_catalog_fingerprint", None)
    if expected is not None and fingerprint != expected:
        raise RuntimeError(
            f"Initial catalog inputs differ between PCs: expected {expected}, computed {fingerprint}. "
            "Object geometry, asset composition or scene/vision settings differ; no catalog was rebuilt.")
    bundled = None
    if not getattr(getattr(env, 'spawn_rules', None), 'parameter_cache_directory', ''):
        bundled = _bundled_catalog_directory(fingerprint, directory)
        if bundled is not None:
            directory = bundled
            print(f"[Scene grammar] using Git-bundled catalog {fingerprint}", flush=True)
    path = directory / f'catalog_{fingerprint}.json'
    manifest = runtime.vision._load_translation_cache(path, fingerprint)
    if (expected is not None or bundled is not None) and manifest is None:
        raise RuntimeError(f"Required transferred initial catalog is missing or invalid: {path}. "
                           "No catalog was rebuilt; check the bundled or explicitly selected cache.")
    if manifest is not None:
        if expected is not None:
            print(f"[Scene grammar] matched transferred catalog {fingerprint} at {path}", flush=True)
        return CertifiedSceneCatalog(manifest, runtime.vision, directory)
    geoms, rules, sensor = env.spawn_geometries, env.spawn_rules, runtime.vision
    if reference_count is not None:
        rules = replace(rules, min_objects=reference_count, max_objects=reference_count)
    capacity = certified_catalog_capacity(geoms, rules)
    if not capacity['depth_valid'] or capacity['maximum_module_width'] <= 0.:
        raise ValueError("The requested object count has no certified background capacity: "
            f"count={rules.min_objects}..{rules.max_objects}, "
            f"background depth={capacity['maximum_background_depth']:.6f} m, "
            f"available role width={capacity['maximum_module_width']:.6f} m.")
    _, groups, _ = _catalog_layout_data(tuple(geoms))
    names = tuple(sorted(groups))
    pairs, guarded_count = [], 0
    from .sensors import InitialVisionRejected

    def coverage(values):
        return certified_role_yaw_gaps([dict(target_yaw=value['target_yaw'],
            blocker_yaw=value['blocker_yaw'], role_cells=value['cells'])
            for value in values if not value.get('fixed_objects')])

    def intersects_gap(angle, half, gaps):
        # Startup angular subdivision concerns only uncovered continuous
        # intervals. It is not a scene acceptance loop in reset.
        return any(max(low, angle-half+turn) < min(high, angle+half+turn)-1.e-12
            for low, high in gaps for turn in (-2*math.pi, 0., 2*math.pi))

    for target_name in names:
        if env.cfg.target_asset is not None and target_name != env.cfg.target_asset:
            continue
        for blocker_name in names:
            if target_name == blocker_name or (env.cfg.blocker_asset is not None and blocker_name != env.cfg.blocker_asset):
                continue
            target, blocker = groups[target_name][0], groups[blocker_name][0]
            minimum_angle = _certified_footprint_profile(tuple(geoms[blocker].lower), tuple(geoms[blocker].upper),
                                                        tuple(geoms[blocker].collision_footprint_vertices))[2]
            # This is the worst budget over ALL permitted surrounding type/
            # count allocations of this ordered pair. No composition is
            # changed or excluded to make an angle fit.
            module_width = capacity['ordered_pair_budgets'][f'{target_name}|{blocker_name}']['minimum_module_width']
            support = capacity['support_bounds']
            pair_capacity = dict(capacity, maximum_module_width=module_width,
                minimum_background_slab_width=support[3]-support[2]-rules.max_neighbor_gap-module_width)
            pair_source = dict(catalog_fingerprint=fingerprint, target_asset=target_name, blocker_asset=blocker_name)
            pair_fingerprint = hashlib.sha256(json.dumps(pair_source, sort_keys=True).encode()).hexdigest()
            pair_path = directory / f'pair_{pair_fingerprint}.json'
            checkpoint = sensor._load_translation_cache(pair_path, pair_fingerprint)
            if checkpoint is not None:
                if any(coverage(checkpoint['families']).values()):
                    raise RuntimeError(f'Incomplete certified pair checkpoint: {pair_path}')
                pairs.append(checkpoint)
                guarded_count += sum(bool(value.get('fixed_objects')) for value in checkpoint['families'])
                continue
            compiled = []
            for step in (10., 5., 2.5, 1.25):
                half = math.radians(step*.5)
                for free_role in ('target', 'blocker'):
                    for degrees in np.arange(0., 360., step):
                        angle = math.radians(float(degrees))
                        gaps = coverage(compiled)
                        if not intersects_gap(angle, half, gaps[free_role]):
                            continue
                        tyaw = angle if free_role == 'target' else 0.
                        byaw = minimum_angle if free_role == 'target' else angle
                        for camera_y in (0., -.08, .08, -.12, .12, -.16, .16):
                            try:
                                family = precompute_certified_role_family(geoms, env.task_scene_data.upright, sensor,
                                    target, blocker, tyaw, byaw, rules, yaw_half_width=half,
                                    camera_lateral_offset=camera_y, cache_directory=directory)
                            except (ValueError, InitialVisionRejected):
                                continue
                            selected = family
                            cells, cumulative = _catalog_scene_cells(selected, rules, pair_capacity,
                                camera_front_range=sensor.cfg.front_distance_range)
                            if not cells:
                                selected = _precompute_normal_pivot_family(family, geoms, rules)
                                if selected is None:
                                    continue
                                cells, cumulative = _catalog_scene_cells(selected, rules, pair_capacity,
                                    camera_front_range=sensor.cfg.front_distance_range)
                            if not cells:
                                continue
                            descriptor = dict(target_yaw=tyaw, blocker_yaw=byaw, blocker_x=family['blocker_x'],
                                camera_pose=family['camera_pose'], cells=cells, cumulative=cumulative,
                                visibility_fingerprint=family['visibility_domain']['fingerprint'])
                            if selected.get('pivot'):
                                descriptor['pivot'] = selected['pivot']
                            compiled.append(descriptor)
                            if not guarded_count:
                                for guard_name in names:
                                    if guard_name not in capacity['ordered_pair_budgets'][f'{target_name}|{blocker_name}']['guard_assets']:
                                        continue
                                    available = [index for index in groups[guard_name] if index != blocker]
                                    if not available:
                                        continue
                                    guarded = _precompute_guarded_role_family(family, geoms, rules, available[0])
                                    if guarded is None:
                                        continue
                                    guard_budget = capacity['ordered_pair_budgets'][f'{target_name}|{blocker_name}']['guard_budgets'][guard_name]
                                    if guard_budget['maximum_background_depth'] > support[1]-support[0]+1.e-12:
                                        continue
                                    guard_capacity = dict(pair_capacity, maximum_module_width=guard_budget['minimum_module_width'],
                                        minimum_background_slab_width=guard_budget['minimum_background_slab_width'])
                                    guard_cells, guard_cumulative = _catalog_scene_cells(guarded, rules, guard_capacity,
                                        camera_front_range=sensor.cfg.front_distance_range)
                                    if guard_cells:
                                        compiled.append(dict(descriptor, cells=guard_cells, cumulative=guard_cumulative,
                                            fixed_objects={_asset_type(geoms[index]): pose
                                                for index, pose in guarded['fixed_objects'].items()},
                                            pivot=guarded['pivot'], guard_certificate=guarded['guard_certificate']))
                                        guarded_count += 1
                                        break
                            break
                if not any(coverage(compiled).values()):
                    break
            gaps = coverage(compiled)
            if any(gaps.values()):
                raise RuntimeError(f"Continuous role-yaw coverage is incomplete for {target_name}/{blocker_name}: {gaps}")
            checkpoint = dict(fingerprint=pair_fingerprint, source=pair_source, target_asset=target_name,
                blocker_asset=blocker_name, families=compiled)
            sensor._save_translation_cache(pair_path, checkpoint)
            pairs.append(checkpoint)
            print(f'[Scene grammar] compiled {target_name}/{blocker_name}: {len(compiled)} continuous families', flush=True)
    manifest = dict(fingerprint=fingerprint, source=source, capacity=capacity, pairs=pairs,
                    guarded_family_count=guarded_count)
    # Do not activate a grammar that silently drops the pivot-required subset.
    if not guarded_count:
        raise RuntimeError("Continuous normal-role domains compiled, but pivot-required domains are not yet certified.")
    sensor._save_translation_cache(path, manifest)
    return CertifiedSceneCatalog(manifest, sensor, directory)


def certified_catalog_capacity(geometries, rules):
    """Measure one topology's worst budget over all allowed type allocations.

    Enumerate categorical counts once at startup, not poses or layout
    candidates. Identical background multisets share their calculation. The
    resulting width/depth bound applies before reset chooses any composition.
    Increased active counts are measured here rather than assumed to fit.
    """
    types, groups, _ = _catalog_layout_data(tuple(geometries))
    names = tuple(sorted(types))
    seen, guard_seen, maximum_depth, maximum_width, count_cases = {}, {}, 0., 0., 0
    pair_budgets = {}
    worst = ()
    for count in range(rules.min_objects, rules.max_objects+1):
        kind_counts = (len(names),) if rules.all_asset_types else range(
            rules.min_asset_types, min(rules.max_asset_types, count, len(names))+1)
        for kind_count in kind_counts:
            if not 2 <= kind_count <= min(count, len(names)):
                raise ValueError("Certified catalog counts require at least two available types within the active count.")
            base, remainder = divmod(count-1, kind_count-1)
            for target in names:
                for others in combinations((name for name in names if name != target), kind_count-1):
                    for extras in combinations(others, remainder):
                        counts = {name: base+int(name in extras) for name in others}
                        for blocker in others:
                            count_cases += 1
                            signature = tuple((name, amount-int(name == blocker)) for name, amount in counts.items())
                            signature = tuple((name, amount) for name, amount in signature if amount)
                            if signature not in seen:
                                ids = []
                                for name, amount in signature:
                                    if amount > len(groups[name]):
                                        raise ValueError(f"The spawned pool cannot supply {amount} background copies of {name}.")
                                    ids.extend(groups[name][:amount])
                                seen[signature] = _certified_background_capacity(geometries, tuple(ids), rules)
                            capacity = seen[signature]
                            maximum_depth = max(maximum_depth, capacity['depth_upper'])
                            if capacity['minimum_slab_width'] > maximum_width:
                                maximum_width, worst = capacity['minimum_slab_width'], capacity['background_ids']
                            support = capacity['support_bounds']
                            available = support[3]-support[2]-rules.max_neighbor_gap-capacity['minimum_slab_width']
                            key = f'{target}|{blocker}'
                            budget = pair_budgets.setdefault(key, dict(minimum_module_width=available,
                                maximum_module_width=available, narrowest_background_ids=capacity['background_ids'],
                                guard_assets=set(), guard_budgets={}))
                            budget['guard_assets'].update(name for name, amount in signature if amount)
                            for guard_name, _ in signature:
                                residual = tuple((name, amount-int(name == guard_name))
                                    for name, amount in signature if amount-int(name == guard_name))
                                if residual not in guard_seen:
                                    guard_ids = tuple(index for name, amount in residual for index in groups[name][:amount])
                                    guard_seen[residual] = _certified_background_capacity(geometries, guard_ids, rules)
                                reduced = guard_seen[residual]
                                item = budget['guard_budgets'].setdefault(guard_name,
                                    dict(minimum_background_slab_width=0., maximum_background_depth=0.))
                                item['minimum_background_slab_width'] = max(item['minimum_background_slab_width'],
                                    reduced['minimum_slab_width'])
                                item['maximum_background_depth'] = max(item['maximum_background_depth'], reduced['depth_upper'])
                                item['minimum_module_width'] = support[3]-support[2]-rules.max_neighbor_gap-item['minimum_background_slab_width']
                            budget['minimum_module_width'] = min(budget['minimum_module_width'], available)
                            if available > budget['maximum_module_width']:
                                budget.update(maximum_module_width=available, narrowest_background_ids=capacity['background_ids'])
    if not seen:
        raise ValueError("The categorical scene configuration has no valid type allocation.")
    for budget in pair_budgets.values():
        budget['guard_assets'] = tuple(sorted(budget['guard_assets']))
    support = (rules.x_range[0]+rules.edge_margin, rules.x_range[1]-rules.edge_margin,
               rules.y_range[0]+rules.edge_margin, rules.y_range[1]-rules.edge_margin)
    return dict(count_allocations=count_cases, unique_background_multisets=len(seen),
        maximum_background_depth=maximum_depth, minimum_background_slab_width=maximum_width,
        depth_valid=maximum_depth <= support[1]-support[0]+1.e-12,
        maximum_module_width=support[3]-support[2]-rules.max_neighbor_gap-maximum_width,
        worst_width_background_ids=worst, support_bounds=support, ordered_pair_budgets=pair_budgets)


def certified_role_yaw_gaps(families):
    """Find uncovered angles for ONE ordered asset pair before training.

    A valid family contributes continuous common-yaw cells to both roles.
    Sampling angles on a grid cannot establish full-circle support; this
    interval union can. Callers must group by the ordered Target/Blocker asset
    pair first. Empty results for both roles are required for every pair.
    """
    result = {}
    period = 2*math.pi
    for role in ('target', 'blocker'):
        intervals = []
        for family in families:
            for cell in family['role_cells']:
                low, high = (family[f'{role}_yaw']+value for value in cell['yaw_bounds'])
                if not all(math.isfinite(value) for value in (low, high)) or low >= high:
                    raise ValueError("Yaw coverage requires finite positive continuous intervals.")
                if high-low >= period:
                    intervals = [(0., period)]
                    break
                start = low % period
                end = start+high-low
                intervals.append((start, min(end, period)))
                if end > period:
                    intervals.append((0., end-period))
        finish, gaps = 0., []
        for low, high in sorted(intervals):
            if low > finish+1.e-12:
                gaps.append((finish, low))
            finish = max(finish, high)
        if finish < period-1.e-12:
            gaps.append((finish, period))
        result[role] = tuple(gaps)
    return result


def prepare_certified_scene_domain(geometries, rules, active_ids, target_id, blocker_id,
                                   target_yaw, blocker_yaw, blocker_x, camera_pose,
                                   visibility_domain, role_cells, *, camera_front_range=(.55, .95),
                                   fixed_objects=None, pivot=None):
    """Intersect continuous role certificates with one fixed background topology.

    This is startup work, not a reset proposal/acceptance stage. The caller
    supplies the inverse-mesh visibility domain and analytic role/motion cells.
    Empty configurations fail here, before any RNG is consumed. It is the
    catalog compiler's responsibility to certify every required role and yaw
    range; compiling a single domain does not establish catalog-wide coverage.
    """
    ids = tuple(active_ids)
    if (len(set(ids)) != len(ids) or target_id == blocker_id or target_id not in ids or blocker_id not in ids
            or not rules.min_objects <= len(ids) <= rules.max_objects
            or any(not 0 <= index < len(geometries) for index in ids)):
        raise ValueError("Certified scenes require distinct active pool IDs and two distinct roles within capacity.")
    target_type = _asset_type(geometries[target_id])
    counts = {asset: sum(_asset_type(geometries[index]) == asset for index in ids)
              for asset in {_asset_type(geometries[index]) for index in ids}}
    others = [count for asset, count in counts.items() if asset != target_type]
    if counts[target_type] != 1 or not others or max(others)-min(others) > 1:
        raise ValueError("A certified composition needs one Target species and balanced non-Target counts.")
    camera = np.asarray(camera_pose, dtype=np.float64)
    if (camera.shape != (7,) or not np.isfinite(camera).all() or abs(np.linalg.norm(camera[3:])-1.) > 1.e-5
            or not all(math.isfinite(value) for value in (target_yaw, blocker_yaw, blocker_x, *camera_front_range))
            or not 0 < camera_front_range[0] <= camera_front_range[1]):
        raise ValueError("Certified role parameters need finite yaws, a unit camera pose and positive front bounds.")
    companions = {} if fixed_objects is None else dict(fixed_objects)
    if any(index not in ids or index in (target_id, blocker_id) or len(pose) != 3
           or not all(math.isfinite(value) for value in pose) for index, pose in companions.items()):
        raise ValueError("Fixed companions need distinct non-role pool IDs and finite local x,y,yaw poses.")
    if pivot is not None:
        if (len(pivot['pivot_offset_from_blocker']) != 2 or len(pivot['turn_offset_from_blocker']) != 2
                or not all(math.isfinite(value) for value in (*pivot['pivot_offset_from_blocker'],
                    *pivot['turn_offset_from_blocker'], pivot['turn_angle']))):
            raise ValueError("A certified pivot needs finite pivot/center offsets and an angle.")
    background = tuple(index for index in ids if index not in (target_id, blocker_id) and index not in companions)
    capacity = _certified_background_capacity(geometries, background, rules)
    if not capacity['depth_valid']:
        raise ValueError("The fixed background topology has insufficient shelf depth.")
    xmin, xmax, ymin, ymax = capacity['support_bounds']
    prepared, weights = [], []
    for cell in role_cells:
        low, high = cell['parameter_bounds']
        yaw_low, yaw_high = cell['yaw_bounds']
        bounds = tuple(cell['module_bounds'])
        if (not low < high or not yaw_low < yaw_high or len(bounds) != 4
                or not all(math.isfinite(value) for value in (*bounds, low, high, yaw_low, yaw_high))
                or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]
                or cell['blockage_lower_bound'] < rules.min_blocked_ratio-1.e-12):
            raise ValueError("Role cells must certify positive continuous bounds and initial access blockage.")
        if not any(vlow <= low+1.e-12 and high <= vhigh+1.e-12
                   for vlow, vhigh in visibility_domain['valid_intervals']):
            raise ValueError("A role cell exceeds the certified inverse-visibility intervals.")
        if cell['camera_x_bounds_over_yaw'] is None:
            raise ValueError("Scene role cells must certify the front camera's continuous X range.")
        camera_low, camera_high = cell['camera_x_bounds_over_yaw']
        x_bounds = (max(xmin-bounds[0], rules.x_range[1]+camera_front_range[0]-camera_low),
                    min(xmax-bounds[1], rules.x_range[1]+camera_front_range[1]-camera_high))
        if 'translation_x_bounds' in cell:
            x_bounds = (max(x_bounds[0], cell['translation_x_bounds'][0]),
                        min(x_bounds[1], cell['translation_x_bounds'][1]))
        padding = (rules.max_neighbor_gap + capacity['minimum_slab_width']) if background else 0.
        # One of these deterministic parameter domains is chosen at startup.
        # Reset never tests the constructed bodies or switches their layout.
        for background_side in (-1, 1):
            y_bounds = ((ymin+padding-bounds[2], ymax-bounds[3]) if background_side < 0
                        else (ymin-bounds[2], ymax-padding-bounds[3]))
            if 'translation_y_bounds' in cell:
                y_bounds = (max(y_bounds[0], cell['translation_y_bounds'][0]),
                            min(y_bounds[1], cell['translation_y_bounds'][1]))
            if x_bounds[0] < x_bounds[1] and y_bounds[0] < y_bounds[1]:
                prepared.append(dict(cell, x_translation_bounds=x_bounds, y_translation_bounds=y_bounds,
                                     background_side=background_side))
                weights.append((high-low)*(yaw_high-yaw_low)*(x_bounds[1]-x_bounds[0])
                               *(y_bounds[1]-y_bounds[0]))
    if not prepared:
        raise ValueError("No positive continuous scene domain fits roles, camera and the fixed background topology.")
    return SimpleNamespace(geometries=tuple(geometries), rules=rules, active_ids=ids,
        target_id=target_id, blocker_id=blocker_id, target_yaw=float(target_yaw), blocker_yaw=float(blocker_yaw),
        blocker_x=float(blocker_x), camera_pose=camera, visibility=visibility_domain,
        cells=tuple(prepared), cumulative=np.cumsum(weights), background_ids=background, capacity=capacity,
        fixed_objects=companions, pivot=pivot)


def sample_certified_scene(domain, rng):
    """Draw one parameter tuple and construct its scene without approval tests.

    Only a shared scene uses this small host-side construction; environment
    clones receive the packed tensors. There is no camera/layout candidate,
    visibility trace, collision acceptance test or failed-row resampling here.
    """
    cell = domain.cells[int(np.searchsorted(domain.cumulative, rng.random()*domain.cumulative[-1], side='right'))]
    parameter = rng.uniform(*cell['parameter_bounds'])
    angle = rng.uniform(*cell['yaw_bounds'])
    shift = np.array((rng.uniform(*cell['x_translation_bounds']), rng.uniform(*cell['y_translation_bounds'])))
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s), (s, c)))
    target, blocker = domain.target_id, domain.blocker_id
    centers = {target: shift, blocker: rotation @ np.array((domain.blocker_x, parameter))+shift}
    positions = {index: (float(center[0]), float(center[1]), domain.rules.shelf_top-domain.geometries[index].lower[2])
                 for index, center in centers.items()}
    yaws = {target: domain.target_yaw+angle, blocker: domain.blocker_yaw+angle}
    for index, (x, y, yaw) in domain.fixed_objects.items():
        center = rotation @ np.array((x, y))+shift
        positions[index] = (float(center[0]), float(center[1]), domain.rules.shelf_top-domain.geometries[index].lower[2])
        yaws[index] = yaw+angle
    bounds = np.asarray(cell['module_bounds']) + shift[[0, 0, 1, 1]]
    background_positions, background_yaws, neighbors, background_certificate = _sample_certified_background(
        domain.geometries, domain.background_ids, domain.rules, tuple(bounds), rng,
        side=cell['background_side'], capacity=domain.capacity, spread_gaps=True)
    positions.update(background_positions)
    yaws.update(background_yaws)
    destination = rotation @ np.array((cell.get('goal_x', domain.blocker_x), cell['goal_y'])) + shift
    start = PlanarPose(*map(float, centers[blocker]), yaws[blocker])
    if domain.pivot is None:
        finish = PlanarPose(*map(float, destination), yaws[blocker])
        path, kind = (PlanarMotion(start, finish),), 'certified_translation'
    else:
        initial = np.array((domain.blocker_x, parameter))
        local_pivot = initial+domain.pivot['pivot_offset_from_blocker']
        turned = rotation @ (initial+domain.pivot['turn_offset_from_blocker'])+shift
        turn = PlanarPose(*map(float, turned), yaws[blocker]+domain.pivot['turn_angle'])
        finish = PlanarPose(*map(float, destination), turn.yaw)
        path = (PlanarMotion(start, turn, tuple(map(float, rotation @ local_pivot+shift))), PlanarMotion(turn, finish))
        kind = 'certified_guarded_pivot_translate' if domain.fixed_objects else 'certified_pivot_translate'
    half_access = domain.rules.hand_width*.5+domain.rules.approach_margin
    target_front = _half_extents(domain.geometries[target], yaws[target])[0]
    access = (positions[target][0]+target_front+domain.rules.approach_margin, domain.rules.x_range[1],
              positions[target][1]-half_access, positions[target][1]+half_access)
    camera = domain.camera_pose.copy()
    camera[:2] = rotation @ camera[:2]+shift
    w, x, y, z = camera[3:]
    ca, sa = math.cos(angle*.5), math.sin(angle*.5)
    camera[3:] = (ca*w-sa*z, ca*x-sa*y, ca*y+sa*x, ca*z+sa*w)
    visibility = domain.visibility
    segment = int(np.searchsorted(visibility['target_visibility_events'], parameter, side='right'))-1
    target_fraction = 1.-visibility['target_occluded_counts'][segment]/visibility['target_pixel_count']
    provenance = dict(visibility_fingerprint=visibility['fingerprint'], parameter=parameter,
        common_yaw=angle, predicted_role_visibility=(target_fraction, visibility['blocker_visible_fraction_structural']),
        live_visibility_measured=False, background=background_certificate)
    return SceneSample(domain.active_ids, target, blocker, positions, yaws, access, tuple(map(float, destination)),
        path, kind, neighbor_pairs=neighbors,
        reserved_regions=(tuple(bounds),), camera_pose=tuple(map(float, camera)), parameter_certificate=provenance)


def _capture_constructive_views(env, runtime, samples, requests, default, origins, roots):
    """Select from one finite joint layout/view domain without regenerating it.

    The constructor already specifies every guarded/unguarded position variant.
    All members keep their episode's composition and yaw, and all are queried
    together exactly once. Selection prefers the first valid member; physics,
    initial observations and IK then receive that same member's packed state.
    """
    families = [(sample, *sample.visibility_variants) for sample in samples]
    variants = [sample for family in families for sample in family]
    parent_ids = [row for row, family in enumerate(families) for _ in family]
    parents = torch.tensor(parent_ids, device=default.device, dtype=torch.long)
    variant_origins = origins[parents]
    proposed = _episode_proposals(variants, default.shape[0], default.device, default.dtype,
        support_heights=getattr(runtime, "support_heights", None), max_objects=_active_object_capacity(env))
    scene = env.task_scene_data
    states, active, slots = pack_episode_states(default, variants, scene.lower, scene.upper, scene.upright,
                                                proposals=proposed)
    poses = states[..., :7].clone()
    poses[..., :3] += variant_origins[:, None]
    if any(sample.camera_pose is None for sample in variants):
        raise RuntimeError("Constructive layout domain omitted a parameterized initial viewpoint.")
    cameras = default.new_tensor([sample.camera_pose for sample in variants])
    cameras[:, :3] += variant_origins
    variant_requests = [dict(requests[parent], active_ids=sample.active_ids,
        target_id=sample.target_id, blocker_id=sample.blocker_id) for sample, parent in zip(variants, parent_ids)]
    snapshots = runtime.vision.capture_view_domain(variant_requests, object_poses=poses,
        root_poses=roots[parents], preferred_camera_poses=cameras)
    diagnostics = getattr(runtime.vision, "last_diagnostics", ())
    if not isinstance(diagnostics, (list, tuple)) or len(diagnostics) != len(variants):
        diagnostics = [{} for _ in variants]
    chosen, summaries, begin = [], [], 0
    for family in families:
        members = range(begin, begin + len(family))
        valid = [index for index in members if snapshots[index] is not None]
        index = valid[0] if valid else begin
        chosen.append(index)
        detail = dict(diagnostics[index])
        domain = diagnostics[begin:begin + len(family)]
        failures = {}
        for member in domain:
            for name, count in member.get("failure_counts", {}).items():
                failures[name] = failures.get(name, 0) + count
        view_count = sum(member.get("view_count", member.get("attempts", 0)) for member in domain)
        detail.update(accepted=bool(valid), variant_count=len(family),
            selected_variant_index=index - begin if valid else None,
            selected_variant_valid_candidate_count=detail.get("valid_candidate_count", 0),
            valid_candidate_count=sum(member.get("valid_candidate_count", 0) for member in domain),
            view_count=view_count, attempts=view_count, failure_counts=failures,
            variant_diagnostics=list(domain))
        summaries.append(detail)
        begin += len(family)
    runtime.vision.last_diagnostics = summaries
    selection = torch.tensor(chosen, device=default.device, dtype=torch.long)
    return ([variants[index] for index in chosen], [snapshots[index] for index in chosen],
        tuple(value[selection] for value in proposed), states[selection], active[selection], slots[selection])


def _build_episode_batch(env, runtime, ids, *, scene_generation=None):
    """Prepare checked episodes, or a singleton template for a shared scene round.

    Constructive layouts and their viewpoint are generated once. Geometry and
    IK checks diagnose invalid rules instead of silently resampling layouts.
    Legacy rejection sampling remains explicitly available for regression.
    Preparation is transactional and never integrates physics.
    """
    objects, robot = env.scene["object_collection"], env.scene["robot"]
    scene, specs, m = env.task_scene_data, runtime.specs, ids.numel()
    capacity, observation_capacity = _active_object_capacity(env), env.cfg.mdp.max_objects
    clutter = env.spawn_rules.generation_mode == 'clutter'
    parameterized = env.spawn_rules.generation_mode in ('parameterized', 'clutter')
    constructive = parameterized or env.spawn_rules.generation_mode == "constructive"
    layout_budget = 1 if constructive else env.cfg.initial_vision.max_layout_attempts
    selected = ids.tolist()
    counts = runtime.reset_counts[ids].tolist()
    if scene_generation is not None:
        if m != 1:
            raise ValueError("A shared scene template must be prepared in one environment.")
        counts = [scene_generation]
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
    saved = (robot.data.root_state_w[ids].clone(),
             robot.data.joint_pos[ids].clone(), robot.data.joint_vel[ids].clone(),
             robot.data.joint_pos_target[ids].clone(), robot.data.joint_vel_target[ids].clone(),
             robot.data.joint_effort_target[ids].clone())
    origins = env.scene.env_origins[ids]
    default = objects.data.default_object_state[0]
    if not hasattr(runtime, "support_heights"):
        runtime.support_heights = default.new_tensor([env.spawn_rules.shelf_top - g.lower[2]
                                                     for g in env.spawn_geometries])
    batch = EpisodeBatch(
        object_state=default.new_zeros((m, len(specs), 13)),
        object_ids=torch.full((m, capacity), -1, device=env.device, dtype=torch.long),
        active=torch.zeros((m, len(specs)), device=env.device, dtype=torch.bool),
        target_id=torch.tensor([role[0] for role in roles], device=env.device),
        blocker_id=torch.tensor([role[1] for role in roles], device=env.device),
        access_region=default.new_zeros((m, 4)), joint_positions=torch.zeros_like(saved[1]),
        geometry_b=default.new_zeros((m, observation_capacity, 10)),
        geometry_valid=torch.zeros((m, observation_capacity), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((m, observation_capacity), -1, device=env.device, dtype=torch.long),
        robot_regions=default.new_zeros((m, 6)), robot_tcp=default.new_zeros((m, 3)),
        visibility=default.new_zeros((m, 2)), camera_pose=default.new_zeros((m, 7)),
        camera_intrinsics=default.new_zeros((m, 3, 3)),
        seeds=torch.tensor(seeds, device=env.device, dtype=torch.long),
    )
    fixed_ids, errors = [None] * m, [None] * m
    pending = list(range(m))
    robot_staged = False
    try:
        with runtime.vision.capture_session():
            for attempt in range(layout_budget):
                candidates, samples = [], []
                # print(f"[Episode reset] sampling {len(pending)} independent episodes, proposal {attempt + 1}", flush=True)
                if clutter:
                    proposals = [sample_clutter_episode(runtime.parameter_catalog, env.spawn_geometries,
                        env.spawn_rules, streams[row], *roles[row], device=env.device,
                        front_range=env.cfg.initial_vision.front_distance_range,
                        lateral_half_width=env.cfg.initial_vision.lateral_half_width,
                        obb_tolerance=env.cfg.mdp.success_xy_tolerance) for row in pending]
                elif parameterized:
                    proposals = [sample_certified_episode(runtime.parameter_catalog, env.spawn_geometries,
                        env.spawn_rules, streams[row], *roles[row]) for row in pending]
                else:
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
                rows = torch.tensor(candidates, device=env.device, dtype=torch.long)
                candidate_ids = ids[rows]
                # Object-only vision needs no robot IK or temporary PhysX writes.
                # Its selected variant also supplies eventual writer/IK poses.
                roots = robot.data.default_root_state[candidate_ids, :7].clone()
                roots[:, :3] += origins[rows]
                requests = [dict(
                    env_id=selected[row], active_ids=sample.active_ids,
                    target_id=sample.target_id, blocker_id=sample.blocker_id,
                    workspace_bounds=(*env.spawn_rules.x_range, *env.spawn_rules.y_range),
                    shelf_top=env.spawn_rules.shelf_top, seed=seeds[row] + 7_919 * (attempt + 1))
                    for row, sample in zip(candidates, samples)]
                if constructive and not parameterized:
                    samples, snapshots, proposed, states, active, slots = _capture_constructive_views(
                        env, runtime, samples, requests, default, origins[rows], roots)
                else:
                    proposed = _episode_proposals(samples, len(specs), env.device, default.dtype,
                        support_heights=runtime.support_heights, max_objects=capacity)
                    states, active, slots = pack_episode_states(default, samples, scene.lower, scene.upper,
                                                               scene.upright, proposals=proposed)
                    vision_poses = states[..., :7].clone()
                    vision_poses[..., :3] += origins[rows, None]
                    if parameterized:
                        cameras = default.new_tensor([sample.camera_pose for sample in samples])
                        cameras[:, :3] += origins[rows]
                        snapshots = runtime.vision.observe_at_viewpoint(requests, cameras,
                            object_poses=vision_poses, root_poses=roots)
                    else:
                        snapshots = runtime.vision.capture_batch(requests, object_poses=vision_poses, root_poses=roots)
                visible = [index for index, snapshot in enumerate(snapshots) if snapshot is not None]
                diagnostics = getattr(runtime.vision, "last_diagnostics", ())
                for index, (row, snapshot) in enumerate(zip(candidates, snapshots)):
                    if snapshot is None:
                        detail = diagnostics[index] if isinstance(diagnostics, (list, tuple)) and index < len(diagnostics) else None
                        errors[row] = (f"Initial geometric visibility/depth invariant rejected: {detail}"
                                       if detail is not None else "Initial geometric visibility/depth requirements were not met.")
                if not visible:
                    continue
                selection = torch.tensor(visible, device=env.device, dtype=torch.long)
                proposed = tuple(t[selection] for t in proposed)
                states, active, slots = states[selection], active[selection], slots[selection]
                candidates = [candidates[index] for index in visible]
                samples = [samples[index] for index in visible]
                snapshots = [snapshots[index] for index in visible]
                starts = _sample_robot_starts(env, samples, [seeds[row] + 7_919 * (attempt + 1) for row in candidates],
                                              proposals=proposed)
                replay_reports = {}
                if any(start is None for start in starts):
                    starts, replay_reports = _recover_initial_robot_starts(
                        env, samples, [seeds[row] + 7_919 * (attempt + 1) for row in candidates], starts, proposed)
                feasible = [index for index, start in enumerate(starts) if start is not None]
                for index, (row, start) in enumerate(zip(candidates, starts)):
                    if start is None:
                        errors[row] = ("No collision-free elbow-up pose in the bounded batched IK search. "
                                       f"Initial replay diagnostics: {replay_reports.get(index)}")
                if not feasible:
                    continue
                if scene_generation is not None and not constructive:
                    # Legacy distribution test only. A finite random probe
                    # count is not a safety invariant and must not reject a
                    # constructive scene that has a physically validated
                    # anchor. Native FK/collision checks below remain required;
                    # shared resets retain bounded search and anchor recovery.
                    index = feasible[0]
                    successes = _robot_start_domain_successes(
                        env, samples[index], starts[index], tuple(t[index:index + 1] for t in proposed),
                        seeds[candidates[index]] + 7_919 * (attempt + 1),
                    )
                    rules = env.cfg.robot_start
                    if successes < rules.domain_probe_min_success:
                        errors[candidates[index]] = (
                            f"Robot start domain has only {successes}/{rules.domain_probe_count} "
                            f"valid single-target probes; requires {rules.domain_probe_min_success}."
                        )
                        continue
                candidates = [candidates[index] for index in feasible]
                samples = [samples[index] for index in feasible]
                starts = [starts[index] for index in feasible]
                snapshots = [snapshots[index] for index in feasible]
                rows = torch.tensor(candidates, device=env.device, dtype=torch.long)
                candidate_ids = ids[rows]
                selection = torch.tensor(feasible, device=env.device, dtype=torch.long)
                states, active, slots = states[selection], active[selection], slots[selection]
                positions = torch.stack([torch.as_tensor(start[0], device=env.device, dtype=default.dtype) for start in starts])
                batch.object_state[rows], batch.active[rows], batch.object_ids[rows] = states, active, slots
                batch.joint_positions[rows] = positions
                batch.access_region[rows] = default.new_tensor([sample.access_region for sample in samples])
                batch.robot_regions[rows] = torch.stack([torch.as_tensor(start[1],device=env.device,dtype=default.dtype) for start in starts])
                batch.robot_tcp[rows] = torch.stack([torch.as_tensor(start[2],device=env.device,dtype=default.dtype) for start in starts])
                # Candidate validation needs articulation FK only. Objects and
                # their shared flags stay untouched until the final commit.
                robot_staged = True
                _write_robot_reset(env, candidate_ids, positions)
                env.sim.forward()  # FK only; no physics integration or neighboring state changes.
                # The public getter refreshes articulation FK even without
                # Fabric; a raw PhysX view read may still contain the old pose.
                link_poses = robot.data.body_link_pose_w[candidate_ids].clone()
                link_poses[..., 3:] = convert_quat(link_poses[..., 3:], to="xyzw")
                physical_details = {}
                physical_valid = _check_physical_robot_starts(
                    env, samples, starts, link_poses, origins[rows], details=physical_details).tolist()
                accepted_rows, accepted_snapshots = [], []
                for index, (row, snapshot, valid) in enumerate(zip(candidates, snapshots, physical_valid)):
                    if valid:
                        accepted_rows.append(row)
                        accepted_snapshots.append(snapshot)
                    else:
                        detail = physical_details.get('rows', [None] * len(starts))[index]
                        errors[row] = f"Physical articulation differs from collision-checked batch FK: {detail}"
                accepted = set(accepted_rows)
                if accepted_rows:
                    accepted_ids = torch.tensor(accepted_rows, device=env.device, dtype=torch.long)
                    for name, source in (("geometry_b", "geometry_b"), ("geometry_valid", "valid"),
                                         ("observed_ids", "observed_ids"), ("camera_pose", "camera_pose_w"),
                                         ("camera_intrinsics", "intrinsics")):
                        getattr(batch, name)[accepted_ids] = torch.stack([getattr(s, source) for s in accepted_snapshots])
                    predicted = {row: sample.parameter_certificate['predicted_role_visibility']
                        for row, sample in zip(candidates, samples)} if parameterized else {}
                    batch.visibility[accepted_ids] = default.new_tensor([
                        snapshot.visible_fractions if snapshot.visible_fractions is not None else predicted[row]
                        for row, snapshot in zip(accepted_rows, accepted_snapshots)])
                pending = [row for row in pending if row not in accepted]
                if not pending:
                    if scene_generation is not None:
                        # Cached camera pose is local; metadata/OBBs already use
                        # robot-base coordinates and never depend on reset EEF.
                        batch.camera_pose[:, :3] -= origins[0]
                        return batch, samples[0]
                    return batch
            failures = [(selected[row], errors[row]) for row in pending[:3]]
            if constructive:
                raise RuntimeError(f"Constructive preparation rejected invariants (no layout resampling): {failures}")
            raise RuntimeError(f"Independent episode preparation exhausted retries: {failures}")
    finally:
        # Preparation is read-only for objects, including on failure. Restore
        # selected robot rows even after success so a later preparation error
        # cannot leave env_0 at a staged pose before the batch is committed.
        if robot_staged:
            robot.write_root_state_to_sim(saved[0], env_ids=ids)
            robot.write_joint_state_to_sim(saved[1], saved[2], env_ids=ids)
            robot.set_joint_position_target(saved[3], env_ids=ids)
            robot.set_joint_velocity_target(saved[4], env_ids=ids)
            robot.set_joint_effort_target(saved[5], env_ids=ids)
            env.sim.forward()


def apply_episode_batch(env, ids, batch):
    """Commit selected rows after applying any shared pool-flag transition."""
    runtime, scene = env._episode_reset, env.task_scene_data
    shelf = env.scene["shelf"]
    shelf_state = shelf.data.default_root_state[ids].clone()
    shelf_state[:, :3] += env.scene.env_origins[ids]
    shelf_state[:, 7:] = 0
    # Validate/commit the shared activity mask before changing any other asset.
    _write_physical_reset(env, ids, batch.object_state, batch.joint_positions, batch.active)
    shelf.write_root_state_to_sim(shelf_state, env_ids=ids)
    runtime.mdp_reset_scene = None
    for name in ("object_ids", "target_id", "blocker_id", "access_region", "geometry_b", "geometry_valid", "observed_ids"):
        getattr(scene, name)[ids] = getattr(batch, name)
    for name in ("robot_regions", "visibility", "camera_pose", "camera_intrinsics"):
        getattr(runtime, name)[ids] = getattr(batch, name)
    runtime.robot_tcp[ids] = batch.robot_tcp + env.scene.env_origins[ids]
    runtime.episode_seeds[ids] = batch.seeds
    runtime.reset_counts[ids] += 1
    env.task_sensor_reader.reset(ids)
    env.sim.forward()


def _shared_reset_batch(env, runtime, ids, template, sample):
    """Broadcast frozen vision/layout and sample fresh continuous EEF starts.

    Only IK failures are retried. No layout generation or visibility query is
    performed here; accepted starts are retained while failed rows resample.
    """
    count = ids.numel()
    # Layout/vision are read-only inputs to the native writers. Materialize only
    # fields changed below; those writers copy the remaining broadcast views.
    batch = EpisodeBatch(**{f.name: getattr(template, f.name).expand(
        count, *getattr(template, f.name).shape[1:]) for f in fields(EpisodeBatch)})
    for name in ("camera_pose", "joint_positions", "robot_regions", "robot_tcp"):
        setattr(batch, name, getattr(batch, name).clone())
    batch.camera_pose[:, :3] += env.scene.env_origins[ids]
    reset_keys = torch.stack((ids, runtime.reset_counts[ids]), -1).cpu().tolist()
    seeds = [_episode_seed(int(env.cfg.seed), index, reset_count)
             for index, reset_count in reset_keys]
    batch.seeds = torch.tensor(seeds, device=env.device, dtype=torch.long)
    cache = getattr(runtime, "shared_inputs", None)
    if cache is None or cache.template is not template or cache.sample is not sample:
        cache = SimpleNamespace(template=template, sample=sample, proposals=_episode_proposals(
            [sample], len(runtime.specs), env.device, batch.object_state.dtype,
            support_heights=runtime.support_heights, max_objects=_active_object_capacity(env)), robot=None)
    rules = env.cfg.robot_start
    pending = list(range(count))
    recovery_count, searches = 0, 0
    for attempt in range(rules.reset_attempts + rules.recovery_attempts):
        if not pending:
            break
        recovering = attempt >= rules.reset_attempts
        before = len(pending)
        searches += 1
        tensor_starts = torch.device(env.device).type == "cuda"
        starts = _sample_robot_starts(
            env, [sample] * len(pending), [seeds[row] + 7_919 * (attempt + 1) for row in pending],
            scene_cache=cache, return_tensors=tensor_starts, recovery=recovering)
        if tensor_starts:
            result, regions = starts
            rows = torch.tensor(pending, device=env.device)
            accepted = result[:, -1] > 0
            for name, value in (("joint_positions", result[:, :-5]), ("robot_regions", regions),
                                ("robot_tcp", result[:, -5:-2])):
                destination = getattr(batch, name)
                destination[rows] = torch.where(accepted[:, None], value, destination[rows])
            # One final transfer per bounded search, after all proposal chunks;
            # only rare failures enter the host retry list. Accepted tensor
            # rows never travel through per-environment Python tuples/stacks.
            attempts = result[:, -1].to(torch.long).cpu().tolist()
            pending = [row for row, chosen in zip(pending, attempts) if chosen == 0]
        else:
            accepted = [index for index, start in enumerate(starts) if start is not None]
            if accepted:
                rows = torch.tensor([pending[index] for index in accepted], device=env.device)
                for name, column in (("joint_positions", 0), ("robot_regions", 1), ("robot_tcp", 2)):
                    getattr(batch, name)[rows] = torch.stack([
                        torch.as_tensor(starts[index][column], device=env.device, dtype=batch.object_state.dtype)
                        for index in accepted])
            pending = [row for row, start in zip(pending, starts) if start is None]
        if recovering:
            recovery_count += before - len(pending)
    # A bounded search must not kill a long run when just one environment has
    # unlucky proposals. This exception restores ONLY the failed row from the
    # same scene's collision/FK-validated template. It deliberately forfeits a
    # new random pose for that episode, and is recorded separately from search
    # success so frequent recovery cannot silently collapse the start domain.
    if pending:
        rows = torch.tensor(pending, device=env.device, dtype=torch.long)
        for name in ("joint_positions", "robot_regions", "robot_tcp"):
            getattr(batch, name)[rows] = getattr(template, name).expand(len(pending), -1)
        runtime.last_robot_start_fallback = dict(
            env_ids=ids[rows].cpu().tolist(), seeds=[seeds[row] for row in pending],
            reset_counts=runtime.reset_counts[ids[rows]].cpu().tolist(),
            scene_generation=getattr(runtime, "scene_generation", -1)
                + int(getattr(runtime, "shared_template", None) is not template),
            candidate_budget=rules.max_attempts,
            reset_attempts=rules.reset_attempts, recovery_attempts=rules.recovery_attempts,
            reason="bounded_robot_start_search_exhausted_verified_template_restored",
        )
    runtime.robot_start_fallback_total = getattr(runtime, "robot_start_fallback_total", 0) + len(pending)
    runtime.pending_reset_metrics = {
        "Reset/robot_start_fallback_count": float(len(pending)),
        "Reset/robot_start_fallback_fraction": len(pending) / count,
        "Reset/robot_start_fallback_total": float(runtime.robot_start_fallback_total),
        "Reset/robot_start_recovery_count": float(recovery_count),
        "Reset/robot_start_search_rounds": float(searches),
    }
    runtime.shared_inputs = cache
    return batch


@torch.no_grad()
def reset_scene(env, env_ids=None):
    """Restore a shared scene, with freshly randomized robot starts on every reset."""
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if ids.numel() == 0:
        return
    if ids.ndim != 1 or ids.dtype != torch.long:
        raise ValueError("Native reset IDs must be a one-dimensional int64 tensor.")
    runtime = _initialize_episode_reset(env)
    refresh = (not hasattr(runtime, "shared_template")
               or env.common_step_counter >= runtime.next_scene_step)
    if refresh:
        if ids.numel() != env.num_envs:
            raise RuntimeError("A shared scene can only change when all environments reset together.")
        generation = getattr(runtime, "scene_generation", -1) + 1
        # Only env_0 prepares the new layout/vision. All environments reset at
        # this boundary; the temporary robot-only check validates native FK.
        template, sample = _build_episode_batch(
            env, runtime, torch.zeros(1, device=env.device, dtype=torch.long), scene_generation=generation)
    else:
        template, sample = runtime.shared_template, runtime.shared_sample
    batch = _shared_reset_batch(env, runtime, ids, template, sample)
    apply_episode_batch(env, ids, batch)
    if refresh:
        runtime.shared_template, runtime.shared_sample = template, sample
        runtime.scene_generation = generation
        runtime.scene_start_step = env.common_step_counter
        runtime.next_scene_step = env.common_step_counter + env.max_episode_length
    # Same ID is a no-op on partial resets. Also reconnect if a timeline
    # stop/play recreated native sensor views while retaining the layout.
    contact_hold = getattr(env, "_blocker_contact_hold", None)
    if contact_hold is not None:
        contact_hold.configure_blocker(sample.blocker_id)
    runtime.mdp_reset_scene = env.task_scene_data
    runtime.mdp_reset_ids = env_ids
    fallback = getattr(runtime, "last_robot_start_fallback", None)
    if fallback is not None:
        # Only exceptional anchor recovery reaches the host/file path. Normal
        # reset diagnostics stay in the native runner's TensorBoard logs.
        log_dir = getattr(env.cfg, "log_dir", None)
        if log_dir:
            path = Path(log_dir) / "robot_start_fallbacks.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            record = dict(fallback, step=int(env.common_step_counter),
                          scene_generation=int(getattr(runtime, "scene_generation", 0)))
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        runtime.last_robot_start_fallback = None
