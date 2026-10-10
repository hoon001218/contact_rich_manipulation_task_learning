"""Joint F/T, per-link binary tactile, and accepted initial-vision observations."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from contextlib import contextmanager
from itertools import product

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_mul, quat_from_matrix, create_rotation_matrix_from_view, convert_camera_frame_orientation_convention,
)

from .assets import (
    TACTILE_BODY_NAMES, ToolForceTorqueSensor, _set_world_transform_on_reference_root, _world_transform,
)
from .catalog import ASSET_SPECS, OBJECT_SPECS, FORBIDDEN_CONTACT_BODY_NAMES


def _load_contact_sensor_class() -> None:
    from pxr import PhysxSchema, Usd, UsdPhysics
    import omni.timeline

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.utils import configclass

    class ForbiddenContactSensor(ContactSensor):
        """One-to-many contact sensor whose exact collider filters come from USD.

        Build reporter APIs and paths before simulation starts; never toggle
        rigid-body flags during an episode. A wildcard matching multiple
        partners per environment is not used because PhysX's batched filter
        patterns must match the sensor count (or one global static collider).
        """

        def __init__(self, cfg):
            if omni.timeline.get_timeline_interface().is_playing():
                raise RuntimeError("Forbidden-contact sensors must be created before simulation starts.")
            stage = sim_utils.get_current_stage()
            env_expression = cfg.prim_path.split("/Robot/", 1)[0]
            environments = sim_utils.find_matching_prims(env_expression)
            if not environments:
                raise RuntimeError("Forbidden-contact sensor cannot find its environment template.")
            template = min((str(prim.GetPath()) for prim in environments), key=lambda p: (len(p), p))
            robot = stage.GetPrimAtPath(f"{template}/Robot")
            candidates = [prim for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies())
                          if prim.GetName() == cfg.observed_body_name
                          and prim.HasAPI(UsdPhysics.RigidBodyAPI)
                          and "/ROASRightHand/" not in str(prim.GetPath())]
            if len(candidates) != 1:
                raise RuntimeError(f"Cannot uniquely resolve monitored body {cfg.observed_body_name}.")
            body_path = str(candidates[0].GetPath())
            cfg = cfg.copy()
            cfg.prim_path = env_expression + body_path[len(template):]
            for prim in sim_utils.find_matching_prims(cfg.prim_path):
                reporter = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                reporter.CreateThresholdAttr().Set(0.0)

            from .catalog import OBJECT_NAMES

            filters = []
            # On GPU, dynamic/kinematic partner filters must name rigid bodies,
            # not their child collision shapes. Static ground has no rigid body
            # and is the one supported collision-shape exception.
            roots = [f"{template}/Shelf", *(f"{template}/{name}" for name in OBJECT_NAMES), "/World/Ground"]
            for root_path in roots:
                root = stage.GetPrimAtPath(root_path)
                if not root.IsValid():
                    raise RuntimeError(f"Missing forbidden-contact partner subtree {root_path}.")
                count_before = len(filters)
                rigid_bodies = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                                if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                partners = rigid_bodies
                if not partners:
                    partners = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                                if prim.HasAPI(UsdPhysics.CollisionAPI)
                                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
                for prim in partners:
                    path = str(prim.GetPath())
                    filters.append(env_expression + path[len(template):]
                                   if path.startswith(template + "/") else path)
                if len(filters) == count_before:
                    raise RuntimeError(f"No physics body or static collider under forbidden-contact partner {root_path}.")
            cfg.filter_prim_paths_expr = sorted(set(filters))
            super().__init__(cfg)

        def _initialize_impl(self):
            super()._initialize_impl()
            if self._num_bodies != 1 or self.contact_physx_view.filter_count != len(self.cfg.filter_prim_paths_expr):
                raise RuntimeError("Forbidden-contact sensor partner filters did not resolve one-to-many correctly.")

    @configclass
    class ForbiddenContactSensorCfg(ContactSensorCfg):
        class_type: type = ForbiddenContactSensor
        observed_body_name: str = ""

    ForbiddenContactSensor.__qualname__ = "ForbiddenContactSensor"
    ForbiddenContactSensorCfg.__qualname__ = "ForbiddenContactSensorCfg"
    globals().update(ForbiddenContactSensor=ForbiddenContactSensor,
                     ForbiddenContactSensorCfg=ForbiddenContactSensorCfg)


def make_forbidden_contact_cfg(body_name: str, history_length: int = 4):
    """Create one RL-only sensor, retaining all substeps of a policy transition."""
    if body_name not in FORBIDDEN_CONTACT_BODY_NAMES:
        raise ValueError(f"Unknown forbidden-contact body {body_name}.")
    if not isinstance(history_length, int) or history_length < 1:
        raise ValueError("Forbidden-contact history_length must be a positive integer.")
    if "ForbiddenContactSensorCfg" not in globals():
        _load_contact_sensor_class()
    return globals()["ForbiddenContactSensorCfg"](
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}", observed_body_name=body_name,
        update_period=0.0, history_length=history_length, track_pose=False,
        track_air_time=False, max_contact_data_count_per_prim=128, debug_vis=False,
    )



TACTILE_SENSOR_NAMES = tuple(f"contact_{body_name}" for body_name in TACTILE_BODY_NAMES)


@configclass
class TactileCfg:
    """Binary contact threshold applied to each link's net normal force, in newtons."""

    force_threshold: float = 0.01


def make_tactile_contact_cfg(body_name: str) -> ContactSensorCfg:
    """Attach a separate reporter to an existing URDF sensor body's collider."""
    if body_name not in TACTILE_BODY_NAMES:
        raise ValueError(f"Unknown tactile sensor body: {body_name}")
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/ROASRightHand/{body_name}",
        update_period=0.0,
        history_length=0,
        track_pose=False,
        track_air_time=False,
        debug_vis=False,
    )


class BinaryTactileSensor:
    """Cache 17 independent ContactSensors in the canonical URDF-link order.

    ContactSensor reports world-frame net normal contact forces. The scalar
    norm is compared with TactileCfg.force_threshold here; the SDK sensor's
    force_threshold option only controls its unused air/contact-time tracking.
    The caller updates this cache only after a completed physics step.
    """

    def __init__(self, env):
        self.force_threshold = env.cfg.tactile.force_threshold
        if not math.isfinite(self.force_threshold) or self.force_threshold <= 0.0:
            raise ValueError("Tactile force_threshold must be finite and positive, in newtons.")
        self.sensors: tuple[ContactSensor, ...] = tuple(env.scene[name] for name in TACTILE_SENSOR_NAMES)
        for name, body_name, sensor in zip(TACTILE_SENSOR_NAMES, TACTILE_BODY_NAMES, self.sensors):
            if sensor.num_bodies != 1 or sensor.body_names != [body_name]:
                raise RuntimeError(
                    f"Tactile sensor {name!r} must resolve exactly {body_name!r}; got {sensor.body_names}."
                )
        shape = (env.num_envs, len(TACTILE_BODY_NAMES))
        self.net_forces_w = torch.zeros((*shape, 3), device=env.device)
        self.contacts = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.valid = torch.zeros_like(self.contacts)

    def update(self) -> None:
        forces = []
        for name, sensor in zip(TACTILE_SENSOR_NAMES, self.sensors):
            force = sensor.data.net_forces_w
            if force.shape != (self.contacts.shape[0], 1, 3):
                raise RuntimeError(f"Tactile sensor {name!r} has unexpected force shape {tuple(force.shape)}.")
            forces.append(force[:, 0])
        measured = torch.stack(forces, dim=1)
        if not torch.isfinite(measured).all():
            raise RuntimeError("Tactile ContactSensor produced nonfinite contact forces.")
        self.net_forces_w.copy_(measured)
        self.contacts.copy_(torch.linalg.vector_norm(measured, dim=-1) > self.force_threshold)
        self.valid.fill_(True)

    def reset(self, env_ids=None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self.net_forces_w[selected] = 0.0
        self.contacts[selected] = False
        self.valid[selected] = False

    def read(self) -> torch.Tensor:
        return self.contacts.clone()


@configclass
class InitialVisionCfg:
    """Acceptance of a single RGB-D snapshot, with actual rendered silhouettes."""

    min_target_visible_fraction: float = 0.40
    min_blocker_visible_fraction: float = 1.0
    blocker_pixel_tolerance: int = 2
    min_silhouette_pixels: int = 32
    camera_side: str = "+X"
    silhouette_reference: str = "full_unoccluded_target_at_same_pose"
    require_full_target_in_frame: bool = True
    front_distance_range: tuple[float, float] = (0.55, 0.95)
    lateral_half_width: float = 0.35
    height_above_shelf: tuple[float, float] = (0.18, 0.48)
    aim_y_jitter: float = 0.06
    aim_z_jitter: float = 0.03
    roll_degrees: tuple[float, float] = (-3.0, 3.0)
    max_camera_attempts: int = 32
    max_layout_attempts: int = 8


def _reference_offset(asset_index: int, env_id: int = 0) -> np.ndarray:
    # Each type gets an isolated rendering lane outside every working camera.
    return np.array((0.0, 5.0 * asset_index, 20.0 + 5.0 * env_id))


@clone
def spawn_vision_references(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Build static render-only references before PhysX creates its views."""
    del cfg, translation, orientation, kwargs
    from omni.physx.scripts import utils as physx_utils

    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    sim_utils.standardize_xform_ops(root)
    for index, spec in enumerate(ASSET_SPECS):
        path = f"{prim_path}/{spec.name}"
        usd_cfg = sim_utils.UsdFileCfg(usd_path=spec.usd_path, semantic_tags=[("class", spec.name)])
        usd_cfg.func(path, usd_cfg, translation=tuple(_reference_offset(index)))
        asset = stage.GetPrimAtPath(path)
        for prim in Usd.PrimRange(asset):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        bodies = [prim for prim in Usd.PrimRange(asset) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise RuntimeError(f"Reference asset needs exactly one original body: {path}")
        asset.SetCustomDataByKey("reference_body", str(bodies[0].GetPath().MakeRelativePath(asset.GetPath())))
        # This follows the SDK visualization-marker stripping approach, applied
        # exclusively to new references. Live task bodies are never hidden.
        for prim in Usd.PrimRange(asset):
            for api in (UsdPhysics.ArticulationRootAPI, PhysxSchema.PhysxArticulationAPI,
                        UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxRigidBodyAPI):
                if prim.HasAPI(api):
                    prim.RemoveAPI(api)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        physx_utils.removeRigidBodySubtree(asset)
    return root


@configclass
class VisionReferencesCfg(SpawnerCfg):
    func: Callable = spawn_vision_references


@contextmanager
def _without_physics(sim):
    setting = "/app/player/playSimulations"
    previous = sim.get_setting(setting)
    try:
        sim.set_setting(setting, False)
        yield
    finally:
        sim.set_setting(setting, previous)


class InitialCamera(Camera):
    """Guard implicit SDK stage refreshes during camera pose queries/writes.

    IsaacLab 2.3.2's first Fabric pose read calls update_stage(), which updates
    Kit even from Camera.reset(). These cameras must not integrate physics.
    """

    def _update_poses(self, env_ids):
        with _without_physics(sim_utils.SimulationContext.instance()):
            super()._update_poses(env_ids)

    def set_world_poses(self, positions=None, orientations=None, env_ids=None, convention="ros"):
        with _without_physics(sim_utils.SimulationContext.instance()):
            super().set_world_poses(positions, orientations, env_ids, convention)


def make_initial_camera_cfg(*, reference: bool = False) -> CameraCfg:
    return CameraCfg(
        class_type=InitialCamera,
        prim_path="{ENV_REGEX_NS}/" + ("SilhouetteCamera" if reference else "InitialCamera"),
        update_period=1.0e9,
        width=640,
        height=480,
        data_types=["instance_segmentation_fast"] if reference else [
            "rgb", "distance_to_image_plane", "instance_segmentation_fast"
        ],
        colorize_instance_segmentation=False,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, horizontal_aperture=24.0, clipping_range=(0.02, 2.5)
        ),
    )


def _instance_mask(camera, env_id: int, root_path: str) -> torch.Tensor:
    info = camera.data.info[env_id].get("instance_segmentation_fast", {})
    labels = info.get("idToLabels")
    if labels is None:
        raise RuntimeError("Camera produced no instance-label mapping.")
    pixels = camera.data.output["instance_segmentation_fast"][env_id, ..., 0]
    mask = torch.zeros_like(pixels, dtype=torch.bool)
    for key, label in labels.items():
        if isinstance(label, str) and (label == root_path or label.startswith(root_path + "/")):
            mask |= pixels == int(key)
    return mask


def fit_visible_obb(points_b: torch.Tensor) -> torch.Tensor:
    """Fit an upright 10D OBB to observed depth points in the fixed robot base.

    This estimates the visible surface only. No catalog dimensions or simulator
    object pose are used to complete occluded geometry.
    """
    xy = points_b[:, :2]
    mean = xy.mean(dim=0)
    centered = xy - mean
    _, axes = torch.linalg.eigh(centered.T @ centered / max(len(points_b) - 1, 1))
    first = axes[:, -1]
    first = torch.where(first[0] < 0, -first, first)
    second = torch.stack((-first[1], first[0]))
    rotation = torch.stack((first, second), dim=-1)
    projected = centered @ rotation
    # Percentiles reduce isolated depth artifacts without using hidden surfaces.
    lower, upper = torch.quantile(projected, 0.01, dim=0), torch.quantile(projected, 0.99, dim=0)
    xy_center = mean + rotation @ ((lower + upper) * 0.5)
    z_lower, z_upper = torch.quantile(points_b[:, 2], 0.01), torch.quantile(points_b[:, 2], 0.99)
    yaw = torch.atan2(first[1], first[0])
    quaternion = torch.stack((torch.cos(yaw * 0.5), yaw * 0, yaw * 0, torch.sin(yaw * 0.5)))
    extents = torch.cat((upper - lower, (z_upper - z_lower)[None])).clamp(min=0.004)
    return torch.cat((xy_center, ((z_lower + z_upper) * 0.5)[None], quaternion, extents))


def estimate_initial_scene(env, env_id: int) -> dict[str, torch.Tensor]:
    """RGB-D and simulator instance masks form a perception prototype at reset.

    Instance segmentation/role association is supplied by simulation. The
    geometry is reconstructed from camera depth; live object transforms,
    velocities and catalog bounds never become actor inputs.
    """
    from isaaclab.utils.math import quat_apply, quat_apply_inverse

    camera = env.scene["initial_camera"]
    depth = camera.data.output["distance_to_image_plane"][env_id, ..., 0]
    h, w = depth.shape
    rows, columns = torch.meshgrid(torch.arange(h, device=env.device), torch.arange(w, device=env.device), indexing="ij")
    intrinsics = camera.data.intrinsic_matrices[env_id]
    points_c = torch.stack(((columns + 0.5 - intrinsics[0, 2]) * depth / intrinsics[0, 0],
                            (rows + 0.5 - intrinsics[1, 2]) * depth / intrinsics[1, 1], depth), dim=-1)
    shape = points_c.shape
    points_c = points_c.reshape(-1, 3)
    camera_q = camera.data.quat_w_ros[env_id].expand(len(points_c), -1)
    points_w = quat_apply(camera_q, points_c) + camera.data.pos_w[env_id]
    robot = env.scene["robot"]
    root_pos, root_q = robot.data.root_pos_w[env_id], robot.data.root_quat_w[env_id]
    points_b = quat_apply_inverse(root_q.expand(len(points_c), -1), points_w - root_pos).reshape(shape)
    finite = torch.isfinite(points_b).all(dim=-1) & torch.isfinite(depth) & (depth > 0) & (depth < 2.5)
    capacity = env.cfg.mdp.max_objects
    geometry = torch.zeros((capacity, 10), device=env.device)
    valid = torch.zeros(capacity, dtype=torch.bool, device=env.device)
    roles = torch.zeros((capacity, 2), device=env.device)
    object_ids = torch.full((capacity,), -1, dtype=torch.long, device=env.device)
    sample = env.spawn_samples[env_id]
    for slot, object_id in enumerate(sample.active_ids):
        if slot >= capacity:
            raise ValueError("Policy object slots cannot hold the active scene.")
        path = f"{env.scene.env_prim_paths[env_id]}/{OBJECT_SPECS[object_id].name}"
        mask = _instance_mask(camera, env_id, path) & finite
        points = points_b[mask]
        if len(points) < 16:
            continue
        # Fitting at most 4096 visible points bounds reset work deterministically.
        if len(points) > 4096:
            indices = torch.linspace(0, len(points) - 1, 4096, device=env.device).long()
            points = points[indices]
        geometry[slot] = fit_visible_obb(points)
        valid[slot] = True
        object_ids[slot] = object_id
        roles[slot, 0] = float(object_id == sample.target_id)
        roles[slot, 1] = float(object_id == sample.blocker_id)
    if not bool((roles.sum(dim=0) == 1).all()):
        raise InitialVisionRejected("Initial RGB-D has too few valid depth pixels for a role object.")
    return {"geometry": geometry, "valid": valid, "roles": roles, "object_ids": object_ids}


class InitialVisionRejected(RuntimeError):
    """No camera passed; caller may retry poses with the same episode IDs."""


class InitialVisionSensor:
    """Use isolated reference renders without changing any live body visibility."""

    def __init__(self, env):
        self.env = env
        cfg = env.cfg.initial_vision
        camera_values = (*cfg.front_distance_range, *cfg.height_above_shelf, *cfg.roll_degrees,
                         cfg.lateral_half_width, cfg.aim_y_jitter, cfg.aim_z_jitter)
        if (not all(math.isfinite(value) for value in camera_values)
                or cfg.max_layout_attempts < 1 or cfg.max_camera_attempts < 1
                or cfg.blocker_pixel_tolerance < 0 or cfg.min_silhouette_pixels < 1
                or not 0 <= cfg.min_blocker_visible_fraction <= 1
                or not 0 <= cfg.min_target_visible_fraction <= 1
                or not 0 < cfg.front_distance_range[0] <= cfg.front_distance_range[1]
                or not 0 < cfg.height_above_shelf[0] <= cfg.height_above_shelf[1]
                or cfg.roll_degrees[0] > cfg.roll_degrees[1]
                or min(cfg.lateral_half_width, cfg.aim_y_jitter, cfg.aim_z_jitter) < 0):
            raise ValueError("Invalid initial-camera ranges, attempts, pixel tolerances or visible fractions.")
        if (cfg.camera_side != "+X" or not cfg.require_full_target_in_frame
                or cfg.silhouette_reference != "full_unoccluded_target_at_same_pose"):
            raise ValueError("Initial vision requires a +X camera and complete same-pose reference silhouettes.")
        self.camera = env.scene["initial_camera"]
        self.reference_camera = env.scene["silhouette_camera"]
        self._warm = False
        self._references = {}
        for env_id, env_path in enumerate(env.scene.env_prim_paths):
            for asset_index, spec in enumerate(ASSET_SPECS):
                root_path = f"{env_path}/VisionReferences/{spec.name}"
                root = env.scene.stage.GetPrimAtPath(root_path)
                relative = root.GetCustomDataByKey("reference_body")
                body = root if relative == "." else env.scene.stage.GetPrimAtPath(root.GetPath().AppendPath(relative))
                world = _world_transform(body)
                if env_id:
                    world.SetTranslateOnly(world.ExtractTranslation() + Gf.Vec3d(0., 0., 5. * env_id))
                    _set_world_transform_on_reference_root(body, world)
                rigid_rotation = Gf.Matrix4d(1.0).SetRotate(Gf.Transform(world).GetRotation())
                stretch = world * rigid_rotation.GetInverse()
                stretch.SetTranslateOnly(Gf.Vec3d(0.0))
                self._references[env_id, spec.name] = (root_path, body, stretch, _reference_offset(asset_index, env_id))
        n, h, w = env.num_envs, self.camera.cfg.height, self.camera.cfg.width
        env.initial_vision_rgb = torch.zeros((n, h, w, 3), dtype=torch.uint8, device=env.device)
        env.initial_vision_depth = torch.zeros((n, h, w, 1), device=env.device)
        env.initial_vision_role_masks = torch.zeros((n, h, w, 2), dtype=torch.bool, device=env.device)
        env.initial_camera_pose_w = torch.zeros((n, 7), device=env.device)
        env.initial_camera_intrinsics = torch.zeros((n, 3, 3), device=env.device)
        env.blocker_visibility_validated = torch.zeros(n, dtype=torch.bool, device=env.device)
        env.blocker_visible_fraction = torch.full((n,), torch.nan, device=env.device)
        env.initial_vision_pixel_counts = torch.zeros((n, 2, 2), dtype=torch.long, device=env.device)
        env.initial_camera_attempts = torch.zeros(n, dtype=torch.long, device=env.device)
        if hasattr(env.cfg, "mdp"):
            capacity = env.cfg.mdp.max_objects
            env.initial_object_geometry = torch.zeros((n, capacity, 10), device=env.device)
            env.initial_object_valid = torch.zeros((n, capacity), dtype=torch.bool, device=env.device)
            env.initial_object_roles = torch.zeros((n, capacity, 2), device=env.device)
            env.initial_observed_object_ids = torch.full((n, capacity), -1, dtype=torch.long, device=env.device)

    def _render(self, camera):
        # SDK texture warmup; render() forwards Fabric but does not integrate.
        for _ in range(5 if not self._warm else 2):
            self.env.sim.render()
        self._warm = True
        with _without_physics(self.env.sim):
            camera.update(0.0, force_recompute=True)

    def prepare(self):
        """Warm lazy annotator buffers before any camera.data read at reset."""
        if not self._warm:
            self._render(self.camera)
            self._render(self.reference_camera)

    def _reference_mask(self, env_id, object_id, position, quaternion):
        spec = OBJECT_SPECS[object_id]
        root_path, body, stretch, offset = self._references[env_id, spec.asset_name]
        q = quaternion.detach().cpu().tolist()
        matrix = Gf.Matrix4d(1.0).SetRotate(Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
        matrix.SetTranslateOnly(Gf.Vec3d(*(position.detach().cpu().numpy() + offset)))
        _set_world_transform_on_reference_root(body, stretch * matrix)
        ids = torch.tensor([env_id], device=self.env.device)
        shift = torch.as_tensor(offset, device=self.env.device, dtype=torch.float32)[None]
        self.reference_camera.set_world_poses(
            self.camera.data.pos_w[env_id:env_id + 1] + shift,
            self.camera.data.quat_w_ros[env_id:env_id + 1], ids, convention="ros",
        )
        self.reference_camera.reset(ids)
        self._render(self.reference_camera)
        return _instance_mask(self.reference_camera, env_id, root_path).clone()

    def _full_in_frame(self, env_id, object_id) -> bool:
        from scipy.spatial.transform import Rotation

        sample, geometry = self.env.spawn_samples[env_id], self.env.spawn_geometries[object_id]
        half = (np.array(geometry.upper) - geometry.lower) * 0.5
        points = np.array(list(product((-half[0], half[0]), (-half[1], half[1]),
                                       (geometry.lower[2], geometry.upper[2]))))
        yaw = sample.yaws[object_id]
        rotation = np.array(((math.cos(yaw), -math.sin(yaw), 0), (math.sin(yaw), math.cos(yaw), 0), (0, 0, 1)))
        points = points @ rotation.T + np.array(sample.positions[object_id])
        points += self.env.scene.env_origins[env_id].cpu().numpy()
        pose = self.camera.data
        q = pose.quat_w_ros[env_id].cpu().numpy()
        camera_rotation = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        local = (points - pose.pos_w[env_id].cpu().numpy()) @ camera_rotation
        near, far = self.camera.cfg.spawn.clipping_range
        if np.any(local[:, 2] <= near) or np.any(local[:, 2] >= far):
            return False
        projected = local @ pose.intrinsic_matrices[env_id].cpu().numpy().T
        uv = projected[:, :2] / projected[:, 2:3]
        return bool(np.all(uv > 1) and np.all(uv < (self.camera.cfg.width - 2, self.camera.cfg.height - 2)))

    def capture(self, env_ids: torch.Tensor, *, layout_attempt: int = 0) -> dict[int, dict]:
        cfg, env = self.env.cfg.initial_vision, self.env
        if cfg.camera_side != "+X" or not 0 <= cfg.min_target_visible_fraction <= 1:
            raise ValueError("Initial camera requires +X and a valid target fraction.")
        results = {}
        states = env.scene["objects"].data.object_link_state_w
        for env_id in env_ids.cpu().tolist():
            sample = env.spawn_samples[env_id]
            rng = random.Random(int(env.spawn_seeds[env_id]) ^ 0x29AD93 ^ (layout_attempt * 0xC47213))
            origin = env.scene.env_origins[env_id].cpu().numpy()
            blocker = np.array(sample.positions[sample.blocker_id])
            target = np.array(sample.positions[sample.target_id])
            ids = torch.tensor([env_id], device=env.device)
            best = (0.0, 0.0)
            for attempt in range(1, cfg.max_camera_attempts + 1):
                eye = origin + np.array((env.spawn_rules.x_range[1] + rng.uniform(*cfg.front_distance_range),
                    np.clip(blocker[1] + rng.uniform(-cfg.lateral_half_width, cfg.lateral_half_width), -.50, .50),
                    env.spawn_rules.shelf_top + rng.uniform(*cfg.height_above_shelf)))
                aim = origin + (target + blocker) * 0.5
                aim[1] += rng.uniform(-cfg.aim_y_jitter, cfg.aim_y_jitter)
                aim[2] = origin[2] + env.spawn_rules.shelf_top + 0.10 + rng.uniform(-cfg.aim_z_jitter, cfg.aim_z_jitter)
                eyes = torch.tensor(eye[None], dtype=torch.float32, device=env.device)
                aims = torch.tensor(aim[None], dtype=torch.float32, device=env.device)
                q_gl = quat_from_matrix(create_rotation_matrix_from_view(eyes, aims, "Z", device=env.device))
                q_ros = convert_camera_frame_orientation_convention(q_gl, origin="opengl", target="ros")
                # Local optical-axis roll also randomizes angle independently.
                half_roll = math.radians(rng.uniform(*cfg.roll_degrees)) * 0.5
                roll = torch.tensor([[math.cos(half_roll), 0., 0., math.sin(half_roll)]], device=env.device)
                self.camera.set_world_poses(eyes, quat_mul(q_ros, roll), ids, convention="ros")
                self.camera.reset(ids)
                self._render(self.camera)
                actual_masks, counts = [], []
                for object_id in (sample.target_id, sample.blocker_id):
                    actual = _instance_mask(self.camera, env_id, f"{env.scene.env_prim_paths[env_id]}/{OBJECT_SPECS[object_id].name}").clone()
                    full = self._reference_mask(env_id, object_id, states[env_id, object_id, :3],
                                                states[env_id, object_id, 3:7])
                    actual_masks.append(actual)
                    counts.append((int(actual.sum()), int(full.sum())))
                fractions = tuple(min(1., visible / full) if full else 0. for visible, full in counts)
                best = max(best, fractions)
                in_frame = all(self._full_in_frame(env_id, index) for index in (sample.target_id, sample.blocker_id))
                enough_pixels = all(full >= cfg.min_silhouette_pixels for _, full in counts)
                blocker_ok = counts[1][0] + cfg.blocker_pixel_tolerance >= cfg.min_blocker_visible_fraction * counts[1][1]
                if enough_pixels and in_frame and fractions[0] >= cfg.min_target_visible_fraction and blocker_ok:
                    if hasattr(env.cfg, "mdp"):
                        try:
                            perceived_scene = estimate_initial_scene(env, env_id)
                        except InitialVisionRejected:
                            continue
                    else:
                        perceived_scene = None
                    results[env_id] = {
                        "rgb": self.camera.data.output["rgb"][env_id, ..., :3].clone(),
                        "depth": self.camera.data.output["distance_to_image_plane"][env_id].clone(),
                        "masks": torch.stack(actual_masks, dim=-1),
                        "pose": torch.cat((self.camera.data.pos_w[env_id], self.camera.data.quat_w_ros[env_id])).clone(),
                        "intrinsics": self.camera.data.intrinsic_matrices[env_id].clone(),
                        "fractions": fractions, "counts": counts, "attempts": attempt,
                        "perceived_scene": perceived_scene,
                    }
                    break
            else:
                raise InitialVisionRejected(f"env={env_id}: no front camera passed after {cfg.max_camera_attempts} attempts; best target/blocker={best}.")
        return results

    def commit(self, results: dict[int, dict]) -> None:
        env = self.env
        for env_id, result in results.items():
            env.initial_vision_rgb[env_id] = result["rgb"]
            env.initial_vision_depth[env_id] = result["depth"]
            env.initial_vision_role_masks[env_id] = result["masks"]
            env.initial_camera_pose_w[env_id] = result["pose"]
            env.initial_camera_intrinsics[env_id] = result["intrinsics"]
            env.target_visible_fraction[env_id], env.blocker_visible_fraction[env_id] = result["fractions"]
            env.initial_vision_pixel_counts[env_id] = torch.tensor(result["counts"], device=env.device)
            env.initial_camera_attempts[env_id] = result["attempts"]
            env.target_visibility_validated[env_id] = env.blocker_visibility_validated[env_id] = True
            perceived = result.get("perceived_scene")
            if perceived is not None:
                env.initial_object_geometry[env_id] = perceived["geometry"]
                env.initial_object_valid[env_id] = perceived["valid"]
                env.initial_object_roles[env_id] = perceived["roles"]
                env.initial_observed_object_ids[env_id] = perceived["object_ids"]


def initialize_sensors(env):
    if not hasattr(env, "tool_force_torque_sensor"):
        env.tool_force_torque_sensor = ToolForceTorqueSensor(env.scene["robot"])
        env.ft_last_update_time = env.sim.current_time
    if not hasattr(env, "binary_tactile_sensor"):
        env.binary_tactile_sensor = BinaryTactileSensor(env)
        env.tactile_last_update_time = env.sim.current_time
    if not hasattr(env, "initial_vision_sensor"):
        env.initial_vision_sensor = InitialVisionSensor(env)


def tool_wrench(env) -> torch.Tensor:
    # Rendering and reset do not advance current_time: stale joint reactions
    # remain invalid until the first subsequent completed physics step.
    initialize_sensors(env)
    timestamp = env.sim.current_time
    if timestamp > env.ft_last_update_time:
        env.tool_force_torque_sensor.update()
        env.ft_last_update_time = timestamp
    return env.tool_force_torque_sensor.read()


def tool_wrench_valid(env) -> torch.Tensor:
    tool_wrench(env)
    return env.tool_force_torque_sensor.valid[:, None].clone()


def tactile_contacts(env) -> torch.Tensor:
    """Return bool [num_envs, 17], retaining reset-invalid zeros during rendering."""
    initialize_sensors(env)
    timestamp = env.sim.current_time
    if timestamp > env.tactile_last_update_time:
        env.binary_tactile_sensor.update()
        env.tactile_last_update_time = timestamp
    return env.binary_tactile_sensor.read()


def tactile_valid(env) -> torch.Tensor:
    tactile_contacts(env)
    return env.binary_tactile_sensor.valid.clone()


def initial_rgb(env):
    initialize_sensors(env)
    return env.initial_vision_rgb.clone()


def initial_depth(env):
    initialize_sensors(env)
    return env.initial_vision_depth.clone()


def initial_role_masks(env):
    initialize_sensors(env)
    return env.initial_vision_role_masks.clone()


def initial_camera_pose(env):
    initialize_sensors(env)
    return env.initial_camera_pose_w.clone()
