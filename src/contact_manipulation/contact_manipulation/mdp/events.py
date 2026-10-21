"""Episode/reset bookkeeping shared by native observation/reward/done terms.

Pure tensor calculation comes first, followed by native scene-data adapters.
One batched update per common_step_counter keeps history and path integrals
independent of manager call order. Only selected rows change on partial reset.
"""

from dataclasses import dataclass, field, fields
from collections.abc import Mapping
from functools import lru_cache
import math
import random
import json
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
    SI_200_8_RANGES, SI_200_8_RESOLUTION, calibrate_wrench, compensate_wrench,
    dynamic_sample, geometry_in_eef_frame, pack_policy_observation, prepare_initial_geometry,
    quat_apply, quat_conjugate, quat_mul, self_motion_wrench,
)
from .rewards import clearance_quality, corner_path_increment, reward_components, singularity_margin
from .terminations import (
    FAILURE_REASON_NAMES, configured_terminations, object_support_validity, task_conditions, terminal_transition,
)


@dataclass
class MdpCfg:
    max_objects: int = 10
    clearance_decay: float = 1.0
    rearrangement_decay: float = 20.0
    # The original four Motion scores each have weight 0.15. Near zero cost,
    # 0.15 * 0.004 matches min's native 0.03 penalty weight * 0.02 s dt.
    motion_action_rate_decay: float = 0.004
    motion_joint_velocity_decay: float = 0.004
    # Geometric warning/hard margins; initial tuning, not certified safety limits.
    # L matches the existing IK angular weighting; s is inverse-Frobenius margin.
    singularity_characteristic_length: float = 0.25
    motion_singularity_threshold: float = 0.07
    motion_singularity_decay: float = math.log(2.0) / 0.03**2
    singularity_termination_threshold: float = 0.005
    # Initial design: half quality at +100 N/s or +5 N*m/s, respectively.
    # Dead bands cover small sensor changes, not validated task impact limits.
    motion_force_rate_decay: float = math.log(2.0) / 90.0**2
    motion_torque_rate_decay: float = math.log(2.0) / 4.5**2
    force_rate_deadband: float = 10.0
    torque_rate_deadband: float = 0.5
    # One-time bonus for the enabled success termination.
    # With default a_C=1 and distance/contact support, the baseline-corrected
    # dense upper bound at clearance weight 4 is about 3.778/step;
    # the previous 180 discounted bound must not justify these scales again.
    success_reward: float = 100.0
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
        if self.max_objects != 10:
            raise ValueError("The agreed observation contract reserves 10 object slots.")
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
    sensor_wrench: torch.Tensor        # N,6; SI-200-8 measured, self-load compensated
    sensor_position: torch.Tensor      # N,3; world
    sensor_rotation: torch.Tensor      # N,4; world, wxyz
    sensor_range_exceeded: torch.Tensor  # N,6; before clipping/compensation
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
    """Preallocated batch buffers; geometry is vision-only, physical poses reward-only."""

    def __init__(self, num_envs, device, cfg, dt, *, dtype=torch.float32, step=0, termination_cfg=None):
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
        self.initial_shelf_pose = zeros(n, 7)
        self.env_origins = zeros(n, 3)
        self.support_bounds = zeros(n, 4)
        self.support_height = zeros(n)
        self.support_vertices = None  # One immutable collision catalog, never cloned per environment.
        self.support_vertex_counts = None
        self.support_error = None
        self.initial_center = zeros(n, o, 3)
        self.initial_rotation = zeros(n, o, 4)
        self.initial_clearance_quality = zeros(n)
        self.previous_corners = zeros(n, o, 8, 3)
        self.shelf_local_corners = zeros(n, 8, 3)
        self.previous_shelf_corners = zeros(n, 8, 3)
        self.previous_wrench = zeros(n, 6)
        self.wrench_bias = zeros(n, 6)
        self.path_length = zeros(n)
        self.elapsed_time = zeros(n, kind=torch.float64)
        self.stable_count = zeros(n, kind=torch.long)
        self.tilt_count = zeros(n, o, kind=torch.long)
        self.support_loss_count = zeros(n, o, kind=torch.long)
        self.support_distance = zeros(n, o)
        self.clearance_progress_valid = zeros(n, kind=torch.bool)
        self.completion_valid = zeros(n, kind=torch.bool)
        self.raw_goal = zeros(n, kind=torch.bool)
        self.raw_goal_seen = zeros(n, kind=torch.bool)
        self.unsupported_goal_seen = zeros(n, kind=torch.bool)
        self.goal_failure_seen = zeros(n, kind=torch.bool)
        # Cumulative fresh native episode counts, never last-episode snapshots.
        self.completed_totals = zeros(5, kind=torch.long)
        self.success = zeros(n, kind=torch.bool)
        self.failure = zeros(n, kind=torch.bool)
        self.terminal_paid = zeros(n, kind=torch.bool)
        self.failure_reasons = zeros(n, len(FAILURE_REASON_NAMES), kind=torch.bool)
        self.failure_mask, self.success_enabled = configured_terminations(termination_cfg, device=device)
        self.invalid_reason_index = FAILURE_REASON_NAMES.index("invalid_state")
        # Clearance, rearrangement, coupled F/T growth, terminal.
        self.components = zeros(n, 4)
        self.wrench_rate_cost = zeros(n, 2)
        self.rearrangement_increment = zeros(n)
        self.blocker_overlap = zeros(n)
        self.singularity_margin = zeros(n)

    @torch.no_grad()
    def reset(self, ids, batch, *, geometry_b, geometry_valid, active, target, blocker,
              access_region, workspace_bounds, shelf_top, ceiling, shelf_local_corners,
              step, batch_is_selected=False):
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
        self.shelf_local_corners[ids] = shelf_local_corners[rows]
        # Fixed episode origin at the initial shelf position, world-oriented
        # axes: retain rotation displacement without adding large clone offsets.
        self.previous_shelf_corners[ids] = quat_apply(
            batch.shelf_pose[rows, None, 3:], self.shelf_local_corners[ids],
        )
        self.initial_center[ids] = batch.center[rows]
        self.initial_rotation[ids] = batch.rotation[rows]
        self.previous_corners[ids] = batch.corners[rows]
        self.tcp_pose_b[ids] = batch.tcp_pose_b[rows]
        self.history[ids] = 0
        self.history[ids, 0] = batch.sample[rows]
        # The reset state has no completed action/contact sample in this episode.
        self.history[ids, 0, 31:37] = 0
        self.history[ids, 0, 37:54] = -1
        self.history[ids, 0, 54:72] = 0
        for name in ("previous_wrench", "wrench_bias", "path_length", "elapsed_time", "stable_count", "tilt_count",
                     "support_loss_count", "support_distance", "clearance_progress_valid", "completion_valid", "raw_goal",
                     "raw_goal_seen", "unsupported_goal_seen", "goal_failure_seen",
                     "success", "failure", "terminal_paid", "failure_reasons", "components",
                     "wrench_rate_cost", "rearrangement_increment", "blocker_overlap"):
            getattr(self, name)[ids] = 0
        # One batched geometry query for reset rows only. The initial reference
        # stays fixed until those environments start another episode.
        areas, overlap = projected_areas(batch.corners[rows], blocker[rows] & active[rows], access_region[rows])
        initial_quality = torch.nan_to_num(
            clearance_quality(self.cfg, overlap.sum(-1), areas.sum(-1)), nan=0.0, posinf=0.0, neginf=0.0,
        )
        self.initial_clearance_quality[ids] = initial_quality
        self.components[ids, 0] = initial_quality
        self.components[ids, 1] = 1.0  # Reset/idle rearrangement has zero penalty.
        self.components[ids, 2] = 1.0  # No completed load-growth transition at reset.
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

    @torch.no_grad()
    def advance(self, batch, step, *, singularity=None):
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
        # Skip polygon clipping for other objects; their path/safety checks remain.
        areas, overlap = projected_areas(safe_corners, self.blocker & self.active, self.access_region)
        blocked = overlap.sum(-1)
        blocker_area = areas.sum(-1)
        self.blocker_overlap.copy_(blocked)
        shelf_corners = ((batch.shelf_pose[:, :3] - self.initial_shelf_pose[:, :3])[:, None]
                         + quat_apply(batch.shelf_pose[:, None, 3:], self.shelf_local_corners))
        increment = (corner_path_increment(self.previous_corners, batch.corners, self.active & ~self.blocker)
                     + torch.linalg.vector_norm(shelf_corners - self.previous_shelf_corners, dim=-1).mean(-1))
        self.rearrangement_increment.copy_(torch.nan_to_num(increment, nan=0.0, posinf=0.0, neginf=0.0))
        self.path_length.add_(self.rearrangement_increment)
        # Force/torque norms are rotation-invariant. Keep the current sensor
        # center: adding r x F at the initial origin would charge pure travel
        # under a constant load as a torque increase.
        components, costs = reward_components(
            self.cfg, blocked, blocker_area, self.rearrangement_increment,
            batch.sensor_wrench, self.previous_wrench, self.dt,
        )
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
        )
        successful, payment, paid = terminal_transition(
            candidate, failed, stable, self.terminal_paid, self.cfg, step_dt=self.dt,
            success_enabled=self.success_enabled,
        )
        self.clearance_progress_valid.copy_(support.progress_valid & ~failed & ~reasons[:, self.invalid_reason_index])
        self.completion_valid.copy_(support.completion_valid)
        self.raw_goal_seen.logical_or_(self.raw_goal)
        self.unsupported_goal_seen.logical_or_(self.raw_goal & ~support.progress_valid)
        self.goal_failure_seen.logical_or_(self.raw_goal & failed)
        self.support_loss_count.copy_(support.support_loss_count)
        self.support_distance.copy_(support.support_distance)
        self.components[:, :3] = components
        self.components[:, 3] = payment
        self.wrench_rate_cost.copy_(costs)
        self.stable_count.copy_(stable)
        self.tilt_count.copy_(tilted)
        self.success.copy_(successful)
        self.failure.copy_(failed)
        self.terminal_paid.copy_(paid)
        self.failure_reasons.copy_(reasons)
        self.history[:, 1].copy_(self.history[:, 0])
        self.history[:, 0].copy_(torch.nan_to_num(batch.sample, nan=0.0, posinf=0.0, neginf=0.0))
        self.tcp_pose_b.copy_(torch.nan_to_num(batch.tcp_pose_b, nan=0.0, posinf=0.0, neginf=0.0))
        self.previous_corners.copy_(batch.corners)
        self.previous_shelf_corners.copy_(shelf_corners)
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

    object_ids: torch.Tensor             # N,10
    target_id: torch.Tensor              # N
    blocker_id: torch.Tensor             # N
    geometry_b: torch.Tensor             # N,10,10; accepted initial vision ONLY
    geometry_valid: torch.Tensor         # N,10
    observed_ids: torch.Tensor           # N,10; association to collection IDs
    lower: torch.Tensor                  # collection_size,3
    upper: torch.Tensor                  # collection_size,3
    upright: torch.Tensor                # collection_size,4 (wxyz)
    access_region: torch.Tensor          # N,4: xmin,xmax,ymin,ymax
    workspace_bounds: torch.Tensor       # N,4
    shelf_top: torch.Tensor              # N
    ceiling: torch.Tensor                # N
    shelf_local_corners: torch.Tensor    # N,8,3; full physical shelf in its link frame
    sensor_body_name: str
    distal_body_names: tuple[str, ...]    # complete hand + distal half-spacer
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


def _ensure_state(env):
    if not hasattr(env, "_contact_mdp"):
        # ObservationManager probes output dimensions before startup/reset.
        # No scene ground truth is substituted for missing initial vision.
        env._contact_mdp = TaskState(env.num_envs, env.device, env.cfg.mdp, float(env.step_dt),
                                     step=env.common_step_counter,
                                     termination_cfg=getattr(env.cfg, "terminations", None))
    return env._contact_mdp


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
    if not scene.distal_body_names or scene.sensor_body_name not in scene.distal_body_names:
        raise ValueError("Supply the complete distal hand/spacer body list, including the F/T body.")
    distal = [names.index(name) for name in scene.distal_body_names]
    if len(set(distal)) != len(distal):
        raise ValueError("F/T compensation bodies must not be duplicated.")
    joint_ids = torch.tensor(arm_ids + hand_ids, device=env.device)
    distal_ids = torch.tensor(distal, device=env.device)
    masses = robot.root_physx_view.get_masses().to(env.device).index_select(1, distal_ids)
    inertias = robot.root_physx_view.get_inertias().to(env.device).index_select(1, distal_ids)
    # PhysX exposes COM inertia in link axes and column-major flattened order.
    inertias = inertias.reshape(env.num_envs, len(distal), 3, 3).transpose(-1, -2).contiguous()
    n, o = env.num_envs, state.cfg.max_objects
    state.env_origins.copy_(env.scene.env_origins)
    if scene.support_vertices is not None:
        state.support_vertices = scene.support_vertices
        state.support_vertex_counts = scene.support_vertex_mask.sum(-1)
        state.support_error = scene.support_error
    # These asset properties are fixed in this task. Native COM properties
    # otherwise fetch CPU buffers again at every physics timestamp/reset.
    body_com_local = robot.data.body_com_pos_b.index_select(1, distal_ids).clone()
    object_com_local = env.scene["object_collection"].data.object_com_pos_b.clone()
    state.bindings = SimpleNamespace(
        robot=robot, objects=env.scene["object_collection"], shelf=env.scene["shelf"],
        joint_ids=joint_ids, distal_ids=distal_ids, tcp_id=tcp_ids[0],
        sensor_id=names.index(scene.sensor_body_name), masses=masses, inertias=inertias,
        body_com_local=body_com_local, object_com_local=object_com_local,
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
     body_acceleration, body_angular_velocity, root_position, root_rotation, joint_position,
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
    distal_rotation = body_rotation.index_select(1, b.distal_ids)
    distal_com = (body_position.index_select(1, b.distal_ids)
                  + quat_apply(distal_rotation, b.body_com_local[rows]))
    self_load = self_motion_wrench(
        b.masses[rows], b.inertias[rows], distal_com,
        distal_rotation, body_acceleration[rows].index_select(1, b.distal_ids),
        body_angular_velocity[rows].index_select(1, b.distal_ids), sensor_position, sensor_rotation, b.gravity,
    )
    wrench = compensate_wrench(measured, self_load, state.wrench_bias[rows], wrench_valid)
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
              data.body_link_pos_w, data.body_link_quat_w, data.body_com_acc_w,
              data.body_com_ang_vel_w, data.root_pos_w, data.root_quat_w, data.joint_pos, data.joint_vel,
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
    return SimpleNamespace(center=center, corners=corners, rotation=rotation,
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
    geometry, valid = prepare_initial_geometry(scene.geometry_b[ids], valid, roles[:, 0], roles[:, 1])
    return SimpleNamespace(object_ids=selected, local_center=center, local_corners=corners,
                           upright=scene.upright[selected], active=active, target=target, blocker=blocker,
                           geometry=geometry, valid=valid)


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
    state.reset(ids, initial, geometry_b=expand(prepared.geometry), geometry_valid=expand(prepared.valid),
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
        state.advance(_read_transition(env, state), env.common_step_counter, singularity=margin)
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


@dataclass
class SpawnRules:
    x_range: tuple[float, float] = (-0.89, -0.51)
    y_range: tuple[float, float] = (-0.455, 0.455)
    shelf_top: float = 1.05
    ceiling: float = 1.35
    min_objects: int = 10
    max_objects: int = 10
    min_asset_types: int = 4
    max_asset_types: int = 10
    all_asset_types: bool = True
    edge_margin: float = 0.015
    gap: float = 0.003
    hand_width: float = 0.10
    approach_margin: float = 0.01
    min_blocked_ratio: float = 0.30
    min_x_coverage: float = 0.55
    min_y_coverage: float = 0.65
    neighbor_distance: float = 0.04
    min_neighbor_fraction: float = 0.75
    min_shelf_occupancy: float = 0.20
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
    orientation_range: tuple[float, float, float] = (math.radians(10.0),) * 3
    reset_attempts: int = 3
    recovery_attempts: int = 2
    recovery_position_half_width: tuple[float, float, float] = (0.02, 0.02, 0.015)
    recovery_rotation_scale: float = 0.25
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


@torch.no_grad()
def _sample_robot_starts(env, samples, seeds, *, proposals=None, scene_cache=None, return_tensors=False,
                         recovery=False, max_attempts=None):
    """Sample continuous starts with bounded batched GPU IK and OBB acceptance.

    Returns one device-tensor start tuple or ``None`` per requested scene. Failed
    rows can be resampled without discarding valid neighbors. Episode RNG and
    metadata preparation use host arrays. CUDA keeps acceptance and pending
    masks on device across candidate chunks; its native caller can consume the
    batched result directly. No live physics trial is used.
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
    key = (id(model), tuple(vars(rules).items()), tuple(vars(env.spawn_rules).items()))
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
    if not anchored:
        # The scene producer first establishes the nominal shelf-facing anchor.
        # Its local-domain probe and every subsequent reset add pose variation.
        orientation_range = torch.zeros_like(orientation_range)
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
    # Local generators keep each episode independent of reset subset/order.
    if torch.device(device).type == "cuda":
        noise = torch.empty((count, attempts_limit, 12), device=device, dtype=dtype)
        seed_tensor = torch.tensor(seeds, device=device, dtype=torch.int64)
        wp.launch(_robot_start_noise, dim=noise.shape[:2], inputs=[wp.from_torch(seed_tensor), wp.from_torch(noise)],
                  stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
    else:
        noise = torch.as_tensor(np.stack([np.random.default_rng(int(seed) ^ 0x5D2EA91C).random(
            (attempts_limit, 12), dtype=np.float32) for seed in seeds]), device=device)
    result = torch.zeros((count, len(model.joint_names) + 5), device=device, dtype=dtype)
    if torch.device(device).type == "cuda":
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
                       tuple(vars(rules).items()), torch.backends.cuda.matmul.allow_tf32)
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
            q, valid, frames, joints, tcp = model.solve_batch(
                target.reshape(-1, 3), target_rotation.reshape(-1, 3, 3), arm.reshape(-1, 6))
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
    if return_tensors:
        return result, regions
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
        rotations = matrix_from_quat(convert_quat(actual[..., 3:], to="wxyz"))
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



# Native reset event. A scene round shares one layout/vision; robot starts
# are continuously sampled and collision-checked on every selected-row reset.

@dataclass
class EpisodeBatch:
    object_state: torch.Tensor       # M,P,13; environment-local link pose / zero COM velocity
    object_ids: torch.Tensor         # M,10; active pool IDs, -1 padding
    active: torch.Tensor             # M,P
    target_id: torch.Tensor          # M
    blocker_id: torch.Tensor         # M
    access_region: torch.Tensor      # M,4
    joint_positions: torch.Tensor    # M,J; checked complete articulation configurations
    geometry_b: torch.Tensor         # M,10,10; initial geometry in robot base frame
    geometry_valid: torch.Tensor     # M,10
    observed_ids: torch.Tensor       # M,10
    robot_regions: torch.Tensor      # M,6
    robot_tcp: torch.Tensor          # M,3; environment coordinates
    visibility: torch.Tensor         # M,2; target, blocker
    camera_pose: torch.Tensor        # M,7; world frame (shared template: env-local), wxyz
    camera_intrinsics: torch.Tensor  # M,3,3
    seeds: torch.Tensor              # M; reproducible per-environment episode seed


def pack_object_states(default_state, sample, lower, upper, upright):
    """Single-proposal convenience wrapper around the batched pose conversion."""
    return tuple(value[0] for value in pack_episode_states(default_state, [sample], lower, upper, upright))


def _episode_proposals(samples, p, device, dtype, *, support_heights=None):
    """Keep XY/yaw on device; only IDs and static support heights come from metadata."""
    m = len(samples)
    active = np.zeros((m, p), dtype=bool)
    slots = np.full((m, 10), -1, dtype=np.int64)
    lengths = np.fromiter((len(s.active_ids) for s in samples), dtype=np.int64, count=m)
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
        OBJECT_SPECS, RobotPoseModel, inspect_geometry, inspect_shelf_bounds, FORCE_TORQUE_BODY_NAME,
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
    link_paths = robot.root_physx_view.link_paths[0]
    distal = tuple(name for name, path in zip(robot.body_names, link_paths)
                   if "/ROASRightHand/" in str(path) or name == FORCE_TORQUE_BODY_NAME)
    if FORCE_TORQUE_BODY_NAME not in distal or not any("force_sensor" in name for name in distal):
        raise RuntimeError("Cannot determine the physical hand-side F/T compensation subtree.")
    n = env.num_envs
    env.task_scene_data = SceneBatch(
        object_ids=torch.full((n, 10), -1, device=env.device, dtype=torch.long),
        target_id=torch.zeros(n, device=env.device, dtype=torch.long),
        blocker_id=torch.ones(n, device=env.device, dtype=torch.long),
        geometry_b=torch.zeros((n, 10, 10), device=env.device),
        geometry_valid=torch.zeros((n, 10), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((n, 10), -1, device=env.device, dtype=torch.long),
        lower=lower, upper=upper, upright=upright,
        access_region=torch.zeros((n, 4), device=env.device),
        workspace_bounds=lower.new_tensor((*env.spawn_rules.x_range, *env.spawn_rules.y_range)).expand(n, -1),
        shelf_top=lower.new_full((n,), env.spawn_rules.shelf_top),
        ceiling=lower.new_full((n,), env.spawn_rules.ceiling),
        shelf_local_corners=shelf_corners.expand(n, -1, -1),
        sensor_body_name=FORCE_TORQUE_BODY_NAME, distal_body_names=distal,
        support_vertices=support_vertices, support_vertex_mask=support_mask,
        support_error=lower.new_tensor([geometry.support_error for geometry in env.spawn_geometries]),
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
    objects.refresh_parking()
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root, env_ids=ids)
    zero_joints = torch.zeros_like(joint_positions)
    robot.write_joint_state_to_sim(joint_positions, zero_joints, env_ids=ids)
    robot.set_joint_position_target(joint_positions, env_ids=ids)
    robot.set_joint_velocity_target(zero_joints, env_ids=ids)
    robot.set_joint_effort_target(zero_joints, env_ids=ids)


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
    """Probe independent 6D targets once before admitting a shared layout.

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


def _build_episode_batch(env, runtime, ids, *, scene_generation=None):
    """Prepare checked episodes, or a singleton template for a shared scene round.

    Layouts retain independent role/path seeds and device-resident clutter
    packing/repair. IK, pose conversion and PhysX writes are batched.
    Initial perception uses bounded geometric ray batches without integrating
    physics; unselected environments are never used as a staging area.
    """
    objects, robot = env.scene["object_collection"], env.scene["robot"]
    scene, specs, m = env.task_scene_data, runtime.specs, ids.numel()
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
    saved = (objects.data.object_state_w[ids].clone(), robot.data.root_state_w[ids].clone(),
             robot.data.joint_pos[ids].clone(), robot.data.joint_vel[ids].clone(),
             robot.data.joint_pos_target[ids].clone(), robot.data.joint_vel_target[ids].clone(),
             robot.data.joint_effort_target[ids].clone(), objects.parked_mask[ids].clone())
    origins = env.scene.env_origins[ids]
    default = objects.data.default_object_state[0]
    if not hasattr(runtime, "support_heights"):
        runtime.support_heights = default.new_tensor([env.spawn_rules.shelf_top - g.lower[2]
                                                     for g in env.spawn_geometries])
    batch = EpisodeBatch(
        object_state=default.new_zeros((m, len(specs), 13)),
        object_ids=torch.full((m, 10), -1, device=env.device, dtype=torch.long),
        active=torch.zeros((m, len(specs)), device=env.device, dtype=torch.bool),
        target_id=torch.tensor([role[0] for role in roles], device=env.device),
        blocker_id=torch.tensor([role[1] for role in roles], device=env.device),
        access_region=default.new_zeros((m, 4)), joint_positions=torch.zeros_like(saved[2]),
        geometry_b=default.new_zeros((m, 10, 10)),
        geometry_valid=torch.zeros((m, 10), device=env.device, dtype=torch.bool),
        observed_ids=torch.full((m, 10), -1, device=env.device, dtype=torch.long),
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
                proposed = _episode_proposals(samples, len(specs), env.device, default.dtype,
                    support_heights=runtime.support_heights)
                states, active, slots = pack_episode_states(default, samples, scene.lower, scene.upper, scene.upright,
                                                           proposals=proposed)
                rows = torch.tensor(candidates, device=env.device, dtype=torch.long)
                candidate_ids = ids[rows]
                # Object-only vision needs no robot IK or temporary PhysX writes.
                # Use exactly the world link poses that the eventual writer uses.
                vision_poses = states[..., :7].clone()
                vision_poses[..., :3] += origins[rows, None]
                roots = robot.data.default_root_state[candidate_ids, :7].clone()
                roots[:, :3] += origins[rows]
                requests = [dict(
                    env_id=selected[row], active_ids=sample.active_ids,
                    target_id=sample.target_id, blocker_id=sample.blocker_id,
                    workspace_bounds=(*env.spawn_rules.x_range, *env.spawn_rules.y_range),
                    shelf_top=env.spawn_rules.shelf_top, seed=seeds[row] + 7_919 * (attempt + 1))
                    for row, sample in zip(candidates, samples)]
                snapshots = runtime.vision.capture_batch(requests, object_poses=vision_poses, root_poses=roots)
                visible = [index for index, snapshot in enumerate(snapshots) if snapshot is not None]
                for row, snapshot in zip(candidates, snapshots):
                    if snapshot is None:
                        errors[row] = "Initial geometric visibility/depth requirements were not met."
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
                feasible = [index for index, start in enumerate(starts) if start is not None]
                for row, start in zip(candidates, starts):
                    if start is None:
                        errors[row] = "No collision-free elbow-up pose in the bounded batched IK search."
                if not feasible:
                    continue
                if scene_generation is not None:
                    # A shared template needs a usable randomized domain, not
                    # just one lucky start. The singleton stage never touches
                    # other environments or integrates simulation time.
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
                _write_physical_reset(env, candidate_ids, states, positions, active)
                env.sim.forward()  # FK only; no physics integration or neighboring state changes.
                link_poses = robot.root_physx_view.get_link_transforms()[candidate_ids]
                physical_valid = _check_physical_robot_starts(env, samples, starts, link_poses, origins[rows]).tolist()
                accepted_rows, accepted_snapshots = [], []
                for row, snapshot, valid in zip(candidates, snapshots, physical_valid):
                    if valid:
                        accepted_rows.append(row)
                        accepted_snapshots.append(snapshot)
                    else:
                        errors[row] = "Physical articulation differs from collision-checked batch FK."
                accepted = set(accepted_rows)
                if accepted_rows:
                    accepted_ids = torch.tensor(accepted_rows, device=env.device, dtype=torch.long)
                    for name, source in (("geometry_b", "geometry_b"), ("geometry_valid", "valid"),
                                         ("observed_ids", "observed_ids"), ("camera_pose", "camera_pose_w"),
                                         ("camera_intrinsics", "intrinsics")):
                        getattr(batch, name)[accepted_ids] = torch.stack([getattr(s, source) for s in accepted_snapshots])
                    batch.visibility[accepted_ids] = default.new_tensor([s.visible_fractions for s in accepted_snapshots])
                pending = [row for row in pending if row not in accepted]
                if not pending:
                    if scene_generation is not None:
                        # Cached camera pose is local; metadata/OBBs already use
                        # robot-base coordinates and never depend on reset EEF.
                        batch.camera_pose[:, :3] -= origins[0]
                        return batch, samples[0]
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
        objects.refresh_parking()
        env.sim.forward()
        raise


def apply_episode_batch(env, ids, batch):
    """Commit episodes by selected row, without host reads or env loops."""
    runtime, scene = env._episode_reset, env.task_scene_data
    runtime.mdp_reset_scene = None
    shelf = env.scene["shelf"]
    shelf_state = shelf.data.default_root_state[ids].clone()
    shelf_state[:, :3] += env.scene.env_origins[ids]
    shelf_state[:, 7:] = 0
    shelf.write_root_state_to_sim(shelf_state, env_ids=ids)
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
            support_heights=runtime.support_heights), robot=None)
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
        # Only env_0 stages the new layout/vision. All environments are resetting
        # at this boundary. A singleton physical check validates model/PhysX FK.
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
