"""Native F/T/contact sensors and renderer-free, reset-only geometric perception.

The virtual viewpoint queries shared surface meshes; only visible first-hit
points enter policy geometry. No camera or reference prim is spawned.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch

from .observations import quat_apply, quat_conjugate


@dataclass
class InitialVisionCfg:
    ray_width: int = 320
    ray_height: int = 240
    ray_batch_size: int = 8
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
        for name in ('ray_width', 'ray_height', 'ray_batch_size'):
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

    geometry_b: torch.Tensor       # [40,10], observed visible-surface OBBs
    geometry_valid: torch.Tensor  # [40]
    observed_ids: torch.Tensor     # [40], native collection indices or -1
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
    batch, rays, _ = points.shape
    stride = capacity + 1  # Extra reduction bucket for background/robot/shelf.
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
            FORCE_TORQUE_BODY_NAME, TACTILE_BODY_NAMES, FORBIDDEN_CONTACT_BODY_NAMES,
        )
        from .events import SensorBatch

        self.env = env
        self.robot = env.scene["robot"]
        self.body_index = self.robot.body_names.index(FORCE_TORQUE_BODY_NAME)
        self.distal_body_names = tuple(
            name for name, path in zip(self.robot.body_names, self.robot.root_physx_view.link_paths[0])
            if name == FORCE_TORQUE_BODY_NAME or "/ROASRightHand/" in str(path)
        )
        self.tactile = tuple(env.scene[f"contact_{name}"] for name in TACTILE_BODY_NAMES)
        self.forbidden = tuple(env.scene[f"forbidden_{name}"] for name in FORBIDDEN_CONTACT_BODY_NAMES)
        for name, sensor in zip(TACTILE_BODY_NAMES, self.tactile):
            if (sensor.num_bodies != 1 or sensor.body_names != [name]
                    or sensor.num_instances != env.num_envs):
                raise RuntimeError(f"Tactile body {name!r} must resolve once per environment.")
        for name, sensor in zip(FORBIDDEN_CONTACT_BODY_NAMES, self.forbidden):
            if (sensor.num_bodies != 1 or sensor.num_instances != env.num_envs
                    or sensor.cfg.history_length < env.cfg.decimation):
                raise RuntimeError(f"Forbidden sensor {name!r} must cover every substep and replica.")
        n, device = env.num_envs, env.device
        self.batch = SensorBatch(
            torch.zeros((n, 6), device=device), torch.zeros(n, dtype=torch.bool, device=device),
            torch.zeros((n, 17, 3), device=device), torch.zeros((n, 17), dtype=torch.bool, device=device),
            torch.zeros(n, dtype=torch.bool, device=device),
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
        b.forbidden_contact[env_ids] = False
        for sensor in (*self.tactile, *self.forbidden):
            sensor.reset(env_ids)

    @torch.no_grad()
    def __call__(self):
        time = self.env.sim.current_time
        if time <= self.last_time:
            return self.batch
        b = self.batch
        reaction = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.body_index]
        b.ideal_wrench.copy_(reaction)
        b.wrench_valid.copy_(torch.isfinite(reaction).all(-1))
        # Fixed sensor loops (17 + 9), never loops over environment indices.
        for channel, sensor in enumerate(self.tactile):
            force = sensor.data.net_forces_w[:, 0]
            b.tactile_force[:, channel].copy_(force)
            b.tactile_valid[:, channel].copy_(torch.isfinite(force).all(-1))
        b.forbidden_contact.zero_()
        for sensor in self.forbidden:
            forces = sensor.data.force_matrix_w_history
            # Keep partners separate: opposing contacts must not cancel.
            bad = (forces.square().sum(-1) > 1.0) | ~torch.isfinite(forces).all(-1)
            b.forbidden_contact.logical_or_(bad.flatten(1).any(-1))
        self.last_time = time
        return b


# Native classes are imported only by the scene configuration after AppLauncher.
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, SensorBase
from isaaclab.utils import configclass
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics


class _ReplicaSensorBase(SensorBase):
    """Use actual PhysX counts when Fabric cloning has only one USD template."""

    def _initialize_impl(self):
        super()._initialize_impl()
        from isaacsim.core.simulation_manager import SimulationManager
        count = SimulationManager.get_physics_sim_view().create_rigid_body_view(
            self.cfg.prim_path.replace(".*", "*")
        ).count
        if count < 1:
            raise RuntimeError(f"No physical sensor body: {self.cfg.prim_path}")
        if count != self._num_envs:
            self._num_envs = count
            self._is_outdated = torch.ones(count, dtype=torch.bool, device=self._device)
            self._timestamp = torch.zeros(count, device=self._device)
            self._timestamp_last_update = torch.zeros_like(self._timestamp)


class TaskContactSensor(ContactSensor, _ReplicaSensorBase):
    """Native force/history buffers with correct physical replica counts."""


class TaskFrameTransformer(FrameTransformer, _ReplicaSensorBase):
    """Native EEF transforms with correct physical replica counts."""


class ForbiddenContactSensor(TaskContactSensor):
    """Resolve one monitored body and exact partner bodies before physics starts."""

    def __init__(self, cfg):
        from ..config.ur5e.scene import OBJECT_SPECS
        stage = sim_utils.get_current_stage()
        expression = cfg.prim_path.split("/Robot/", 1)[0]
        environments = sim_utils.find_matching_prims(expression)
        template = min((str(prim.GetPath()) for prim in environments), key=lambda p: (len(p), p))
        robot = stage.GetPrimAtPath(f"{template}/Robot")
        bodies = [prim for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
                  if prim.GetName() == cfg.observed_body_name and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                  and "/ROASRightHand/" not in str(prim.GetPath())]
        if len(bodies) != 1:
            raise RuntimeError(f"Cannot resolve monitored arm/spacer body {cfg.observed_body_name}.")
        cfg = cfg.copy()
        cfg.prim_path = expression + str(bodies[0].GetPath())[len(template):]
        for prim in sim_utils.find_matching_prims(cfg.prim_path):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        filters = []
        roots = [f"{template}/Shelf", *(f"{template}/{spec.name}" for spec in OBJECT_SPECS), "/World/Ground"]
        for path in roots:
            root = stage.GetPrimAtPath(path)
            if not root.IsValid():
                raise RuntimeError(f"Missing forbidden-contact partner: {path}")
            prims = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            partners = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if not partners:
                partners = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)
                            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
            if not partners:
                raise RuntimeError(f"No partner rigid body or collider under {path}")
            for prim in partners:
                partner = str(prim.GetPath())
                filters.append(expression + partner[len(template):] if partner.startswith(template + "/") else partner)
        cfg.filter_prim_paths_expr = sorted(set(filters))
        super().__init__(cfg)

    def _initialize_impl(self):
        super()._initialize_impl()
        if (self.num_bodies != 1
                or self.contact_physx_view.filter_count != len(self.cfg.filter_prim_paths_expr)):
            raise RuntimeError("Forbidden-contact filters must resolve one body to its exact partners.")


@configclass
class ForbiddenContactSensorCfg(ContactSensorCfg):
    class_type: type = ForbiddenContactSensor
    observed_body_name: str = ""


def make_tactile_contact_cfg(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        class_type=TaskContactSensor, prim_path=f"{{ENV_REGEX_NS}}/Robot/ROASRightHand/{body_name}",
        update_period=0.0, history_length=0, track_pose=False, track_air_time=False, debug_vis=False,
    )


def make_forbidden_contact_cfg(body_name: str, history_length: int = 4) -> ContactSensorCfg:
    return ForbiddenContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}", observed_body_name=body_name,
        update_period=0.0, history_length=history_length, track_pose=False, track_air_time=False,
        max_contact_data_count_per_prim=128, debug_vis=False,
    )


# Geometric perception follows Isaac Lab's Warp RayCaster path. It creates no
# USD cameras, render products, annotators, or reference objects.
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _initial_surface_rays(
    meshes: wp.array(dtype=wp.uint64),
    poses: wp.array2d(dtype=wp.transform),
    enabled: wp.array2d(dtype=wp.int32),
    eyes: wp.array(dtype=wp.vec3),
    orientations: wp.array(dtype=wp.quat),
    directions: wp.array(dtype=wp.vec3),
    roles: wp.array2d(dtype=wp.int32),
    width: int, height: int, far: float,
    depths: wp.array2d(dtype=wp.float32),
    labels: wp.array2d(dtype=wp.int32),
    silhouettes: wp.array3d(dtype=wp.int32),
):
    env, ray = wp.tid()
    origin = eyes[env]
    direction = wp.quat_rotate(orientations[env], directions[ray])
    x = ray % (2 * width)
    y = ray // (2 * width)
    in_image = x >= width // 2 and x < width // 2 + width and y >= height // 2 and y < height // 2 + height
    best = far
    owner = int(-1)
    # Each ray owns its outputs: no competing atomic depth/label writes.
    # Outside the image only the two isolated role silhouettes are needed.
    # Route directly to those meshes instead of scanning every scene body.
    body_count = meshes.shape[0]
    if not in_image:
        body_count = 2
    for slot in range(body_count):
        body = slot
        if not in_image:
            body = roles[env, slot]
        is_target = body == roles[env, 0]
        is_blocker = body == roles[env, 1]
        if enabled[env, body] != 0 and (in_image or is_target or is_blocker):
            inverse = wp.transform_inverse(poses[env, body])
            start = wp.transform_point(inverse, origin)
            local_direction = wp.transform_vector(inverse, direction)
            query = wp.mesh_query_ray(meshes[body], start, local_direction, far)
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
        self.meshes, identifiers = [], []
        self.body_indices = []
        self.specs = OBJECT_SPECS
        stage, device = env.scene.stage, env.device
        origin_path = env.scene.env_prim_paths[0]
        cache = {}
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        def add(root, key, *, local_origin=None):
            if key in cache:
                return cache[key]
            world = _gf_column(xforms.GetLocalToWorldTransform(root))
            rigid = np.eye(4)
            if local_origin is None:
                rigid[:3, :3], rigid[:3, 3] = _rotation_part(world), world[:3, 3]
            else:
                rigid[:3, 3] = local_origin
            triangles = _visual_triangles(root, np.linalg.inv(rigid))
            if triangles is None:
                return 0
            points, indices = triangles
            mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                           indices=wp.array(indices.flatten(), dtype=wp.int32, device=device))
            self.meshes.append(mesh)  # Own buffers for the entire simulation lifetime.
            cache[key] = mesh.id
            return mesh.id
        for spec in OBJECT_SPECS:
            root = stage.GetPrimAtPath(f'{origin_path}/{spec.name}')
            bodies = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                      if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(bodies) != 1:
                raise RuntimeError(f'Initial perception needs one physical body for {spec.name}.')
            identifier = add(bodies[0], spec.asset_name)
            if not identifier:
                raise RuntimeError(f'No visible surface for object {spec.name}.')
            identifiers.append(identifier)
        self.object_count = len(identifiers)
        for index, path in enumerate(env.scene['robot'].root_physx_view.link_paths[0]):
            identifier = add(stage.GetPrimAtPath(str(path)), str(path))
            if identifier:
                identifiers.append(identifier)
                self.body_indices.append(index)
        self.shelf_index = len(identifiers)
        identifier = add(stage.GetPrimAtPath(f'{origin_path}/Shelf'), 'shelf',
                         local_origin=env.scene.env_origins[0].cpu().numpy())
        if not identifier:
            raise RuntimeError('No visible shelf surface for geometric occlusion.')
        identifiers.append(identifier)
        self.mesh_ids = wp.array(identifiers, dtype=wp.uint64, device=device)
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
    def capture_batch(self, requests, *, retain_images=False):
        results = []
        for begin in range(0, len(requests), self.capacity):
            results.extend(self._capture_pool(requests[begin:begin + self.capacity], retain_images))
        return results

    def _trace(self, poses, enabled, eyes, orientations, roles):
        width, height = self.cfg.ray_width, self.cfg.ray_height
        count, device = len(eyes), eyes.device
        depths = torch.empty((count, width * height), device=device)
        labels = torch.empty((count, width * height), device=device, dtype=torch.int32)
        silhouettes = torch.zeros((count, 4 * width * height, 2), device=device, dtype=torch.int32)
        # Warp uses xyzw; policy and Isaac Lab tensors use wxyz.
        poses = poses[..., [0, 1, 2, 4, 5, 6, 3]].contiguous()
        quaternions = orientations[:, [1, 2, 3, 0]].contiguous()
        launch = dict(kernel=_initial_surface_rays, dim=(count, 4 * width * height), inputs=[
            self.mesh_ids, wp.from_torch(poses, dtype=wp.transform), wp.from_torch(enabled),
            wp.from_torch(eyes.contiguous(), dtype=wp.vec3), wp.from_torch(quaternions, dtype=wp.quat),
            wp.from_torch(self.directions, dtype=wp.vec3), wp.from_torch(roles), width, height, 2.5,
            wp.from_torch(depths), wp.from_torch(labels), wp.from_torch(silhouettes)])
        if device.type == 'cuda':
            wp.launch(**launch, stream=wp.stream_from_torch(torch.cuda.current_stream(device)))
        else:
            wp.launch(**launch, device='cpu')
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

    def _capture_pool(self, requests, retain_images):
        from isaaclab.utils.math import (quat_from_matrix, create_rotation_matrix_from_view,
            convert_camera_frame_orientation_convention, quat_mul)
        env, cfg = self.env, self.cfg
        ids = [int(request['env_id']) for request in requests]
        objects = env.scene['object_collection'].data.object_link_pose_w[ids]
        robot = env.scene['robot'].data
        origins = env.scene.env_origins[ids]
        # Compute in environment-local coordinates to avoid large-grid precision loss.
        objects = objects.clone()
        objects[..., :3] -= origins[:, None]
        robot_poses = robot.body_link_state_w[ids][:, self.body_indices, :7].clone()
        robot_poses[..., :3] -= origins[:, None]
        shelf_pose = torch.zeros((len(ids), 1, 7), device=env.device)
        shelf_pose[..., 3] = 1
        poses = torch.cat((objects, robot_poses, shelf_pose), 1)
        enabled = torch.ones(poses.shape[:2], device=env.device, dtype=torch.int32)
        enabled[:, :self.object_count] = 0
        for row, request in enumerate(requests):
            active = request['active_ids']
            if (request['target_id'] == request['blocker_id'] or request['target_id'] not in active
                    or request['blocker_id'] not in active or len(active) > env.cfg.mdp.max_objects):
                raise ValueError('Initial perception needs distinct active Target/Blocker within capacity.')
            enabled[row, list(active)] = 1
        roles = torch.tensor([[r['target_id'], r['blocker_id']] for r in requests], device=env.device, dtype=torch.int32)
        role_poses = objects.gather(1, roles.long()[..., None].expand(-1, -1, 7)).tolist()
        generators = [random.Random(request['seed']) for request in requests]
        results, pending = [None] * len(ids), list(range(len(ids)))
        tensor = objects.new_tensor
        for attempt in range(1, cfg.max_camera_attempts + 1):
            if not pending:
                break
            eyes, aims, rolls = [], [], []
            for row in pending:
                rng, request = generators[row], requests[row]
                target, blocker = role_poses[row]
                bounds, top = request['workspace_bounds'], request['shelf_top']
                midpoint = [(a + b) * .5 for a, b in zip(target[:3], blocker[:3])]
                eyes.append((bounds[1] + rng.uniform(*cfg.front_distance_range),
                             max(bounds[2], min(bounds[3], blocker[1] + rng.uniform(-cfg.lateral_half_width, cfg.lateral_half_width))),
                             top + rng.uniform(*cfg.height_above_shelf)))
                aims.append((midpoint[0], midpoint[1] + rng.uniform(-cfg.aim_y_jitter, cfg.aim_y_jitter),
                             top + .10 + rng.uniform(-cfg.aim_z_jitter, cfg.aim_z_jitter)))
                half_roll = math.radians(rng.uniform(*cfg.roll_degrees)) * .5
                rolls.append((math.cos(half_roll), 0., 0., math.sin(half_roll)))
            eyes = tensor(eyes)
            orientation = quat_from_matrix(create_rotation_matrix_from_view(eyes, tensor(aims), 'Z', device=env.device))
            orientation = convert_camera_frame_orientation_convention(orientation, origin='opengl', target='ros')
            orientation = quat_mul(orientation, tensor(rolls))
            depth, labels, full, visible, fractions, accepted = self._trace(
                poses[pending], enabled[pending], eyes, orientation, roles[pending])
            candidates = [index for index, okay in enumerate(accepted.tolist()) if okay]
            if not candidates:
                continue
            rows = [pending[index] for index in candidates]
            depth, labels = depth[candidates], labels[candidates]
            eyes, orientation = eyes[candidates], orientation[candidates]
            full, visible, fractions = full[candidates], visible[candidates], fractions[candidates]
            # Fit observed geometry only after the viewpoint passes visibility.
            local_points = quat_apply(orientation[:, None].expand(-1, depth.shape[1], -1),
                                      depth[..., None] * self.crop_directions) + eyes[:, None]
            roots = robot.root_pos_w[ids][rows] - origins[rows]
            points_b = quat_apply(quat_conjugate(robot.root_quat_w[ids][rows])[:, None].expand(-1, depth.shape[1], -1),
                                  local_points - roots[:, None])
            labels = torch.where(labels < self.object_count, labels, -1)
            geometry, valid = fit_visible_obbs(points_b, labels, env.cfg.mdp.max_objects, cfg.min_depth_pixels)
            decisions = valid.gather(1, roles[rows].long()).all(-1).tolist()
            counts = torch.stack((visible, full), -1).tolist()
            ratios = fractions.tolist()
            for index, row in enumerate(rows):
                if not decisions[index]:
                    continue
                observed = torch.arange(env.cfg.mdp.max_objects, device=env.device)
                observed = torch.where(valid[index], observed, -1)
                height, width = cfg.ray_height, cfg.ray_width
                results[row] = VisionSnapshot(
                    geometry[index].clone(), valid[index].clone(), observed,
                    None, depth[index].reshape(height, width, 1).clone() if retain_images else None,
                    (labels[index, :, None] == roles[row]).reshape(height, width, 2).clone() if retain_images else None,
                    torch.cat((eyes[index] + origins[row], orientation[index])), self.intrinsics.clone(),
                    tuple(ratios[index]), tuple(tuple(pair) for pair in counts[index]), attempt)
            pending = [row for row in pending if results[row] is None]
        return results
