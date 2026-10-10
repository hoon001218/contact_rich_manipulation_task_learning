"""Inspect spawned USD geometry and apply seeded episode resets to the scene.

Sampling stays simulator-independent in src/sampling.py. This module bridges those
rules to USD measurements, current role metadata, and PhysX object state.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING

import torch
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as mdp

from .state import reset_mdp
from ..src.catalog import OBJECT_SPECS
from ..src.assets import RobotPoseModel, FORCE_TORQUE_BODY_NAME, FORCE_TORQUE_PROXIMAL_BODY_NAME
from ..src.sensors import initialize_sensors, InitialVisionRejected
from ..src.sampling import (
    ROBOT_START_TCP_ROTATION, ObjectGeometry, SpawnRules,
    oriented_box_separated, robot_start_region, sample_scene,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

Point3 = tuple[float, float, float]
Bounds = tuple[Point3, Point3]


# USD geometry and support measurements


def _range_bounds(bound) -> Bounds:
    """Convert a USD bounding box to finite, nonempty aligned limits."""
    extent = bound.ComputeAlignedRange()
    if extent.IsEmpty():
        raise ValueError("USD bounding box is empty.")
    lower = tuple(float(value) for value in extent.GetMin())
    upper = tuple(float(value) for value in extent.GetMax())
    if not all(math.isfinite(value) for value in (*lower, *upper)):
        raise ValueError("USD bounding box has nonfinite limits.")
    if any(high <= low for low, high in zip(lower, upper)):
        raise ValueError("USD bounding box must have positive extent on each axis.")
    return lower, upper


def _rotate_point(point: Point3, quaternion: tuple[float, float, float, float]) -> Point3:
    """Apply a wxyz quaternion, normalizing catalog rotations defensively."""
    norm = math.sqrt(sum(value * value for value in quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Object upright quaternion must be finite and nonzero.")
    w, x, y, z = (value / norm for value in quaternion)
    px, py, pz = point
    tx, ty, tz = 2.0 * (y * pz - z * py), 2.0 * (z * px - x * pz), 2.0 * (x * py - y * px)
    return (
        px + w * tx + y * tz - z * ty,
        py + w * ty + z * tx - x * tz,
        pz + w * tz + x * ty - y * tx,
    )


def _body_stretch(body: Usd.Prim) -> Gf.Matrix4d:
    """Keep composed body scaling while stripping the rigid world pose.

    PhysX pose tensors contain translation/rotation but omit USD scaling. A
    body's own scaling is also omitted by ComputeUntransformedBound, although it
    still changes its simulated collision dimensions. Include ancestor scaling
    as well, since it is baked into the physics shapes during scene creation.
    """
    world_matrix = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rotation_matrix = Gf.Matrix4d(1.0).SetRotate(Gf.Transform(world_matrix).GetRotation())
    stretch = world_matrix * rotation_matrix.GetInverse()
    stretch.SetTranslateOnly(Gf.Vec3d(0.0))
    return stretch


def _upright_bounds(
    bounds: Bounds, quaternion: tuple[float, float, float, float], stretch: Gf.Matrix4d | None = None
) -> Bounds:
    """Scale and rotate all eight local AABB corners into the upright frame."""
    local_corners = product(*zip(*bounds))
    if stretch is not None:
        local_corners = (tuple(stretch.TransformDir(Gf.Vec3d(*corner))) for corner in local_corners)
    corners = tuple(_rotate_point(corner, quaternion) for corner in local_corners)
    return (
        tuple(min(corner[axis] for corner in corners) for axis in range(3)),
        tuple(max(corner[axis] for corner in corners) for axis in range(3)),
    )


def _environment_zero_path(env: ManagerBasedEnv, path: str) -> str:
    """Resolve the actual config prim expression to its env_0 instance."""
    env_path = env.scene.env_prim_paths[0]
    return path.format(ENV_REGEX_NS=env_path).replace(env.scene.env_regex_ns, env_path, 1)


def _shelf_rules(cache: UsdGeom.BBoxCache, shelf: Usd.Prim, origin: Point3, rules: SpawnRules) -> SpawnRules:
    """Discover support/ceiling panels and clip the support inside vertical posts."""
    collisions: list[Bounds] = []
    panels: list[Bounds] = []
    for prim in Usd.PrimRange(shelf, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        try:
            lower_w, upper_w = _range_bounds(cache.ComputeWorldBound(prim))
        except ValueError:
            # Flat or empty collision metadata does not describe a shelf slab.
            continue
        lower = tuple(value - offset for value, offset in zip(lower_w, origin))
        upper = tuple(value - offset for value, offset in zip(upper_w, origin))
        bounds = lower, upper
        collisions.append(bounds)
        width_x, width_y, height = (high - low for low, high in zip(lower, upper))
        if width_x > 0.25 and width_y > 0.5 and height < 0.06:
            panels.append(bounds)

    if not panels:
        raise RuntimeError(f"No horizontal shelf collision panels found below {shelf.GetPath()}.")
    lower, upper = min(panels, key=lambda bounds: abs(bounds[1][2] - rules.shelf_top))
    shelf_top = upper[2]
    ceiling_panels = [
        panel
        for panel in panels
        if panel[0][2] > shelf_top + 1.0e-4
        and panel[0][0] < upper[0]
        and panel[1][0] > lower[0]
        and panel[0][1] < upper[1]
        and panel[1][1] > lower[1]
    ]
    if not ceiling_panels:
        raise RuntimeError(f"No shelf collision panel above the support at z={shelf_top:.4f}.")
    ceiling = min(panel[0][2] for panel in ceiling_panels)

    midpoint_y = 0.5 * (lower[1] + upper[1])
    negative_inside_edges: list[float] = []
    positive_inside_edges: list[float] = []
    for post_lower, post_upper in collisions:
        width_x, width_y, height = (high - low for low, high in zip(post_lower, post_upper))
        if not (width_x < 0.015 and width_y > 0.015 and height > 0.5):
            continue
        if post_lower[2] > shelf_top or post_upper[2] < shelf_top:
            continue
        if post_lower[0] > upper[0] or post_upper[0] < lower[0]:
            continue
        if lower[1] <= post_upper[1] < midpoint_y:
            negative_inside_edges.append(post_upper[1])
        elif midpoint_y < post_lower[1] <= upper[1]:
            positive_inside_edges.append(post_lower[1])

    # If a custom shelf lacks the slender authored posts, keep a conservative
    # five-centimeter inset on each undiscovered side of the slab.
    minimum_y = max(negative_inside_edges) if negative_inside_edges else lower[1] + 0.05
    maximum_y = min(positive_inside_edges) if positive_inside_edges else upper[1] - 0.05
    if minimum_y >= maximum_y:
        raise RuntimeError("Shelf posts leave no usable support region.")
    return replace(
        rules,
        x_range=(lower[0], upper[0]),
        y_range=(minimum_y, maximum_y),
        shelf_top=shelf_top,
        ceiling=ceiling,
    )


def inspect_spawn_geometry(env: ManagerBasedEnv) -> tuple[tuple[ObjectGeometry, ...], SpawnRules]:
    """Inspect env_0 body bounds and shelf collisions after the scene has spawned.

    Object bounds include descendants and composed body scale, exclude its rigid
    world pose, and are rotated by its catalog upright quaternion. Shelf bounds
    are expressed in environment coordinates and retain the configured sampling
    constraints except for the measured support region and vertical clearance.
    """
    stage = env.scene.stage
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    objects = env.scene["objects"]
    geometries: list[ObjectGeometry] = []
    for spec in OBJECT_SPECS:
        path = _environment_zero_path(env, objects.cfg.rigid_objects[spec.name].prim_path)
        root = stage.GetPrimAtPath(path)
        if not root.IsValid():
            raise RuntimeError(f"Spawned object prim is missing: {path}")
        bodies = [
            prim
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(bodies) != 1:
            raise RuntimeError(f"Object {spec.name} must contain exactly one rigid body; found {len(bodies)}.")
        body = bodies[0]
        rigid_api = UsdPhysics.RigidBodyAPI(body)
        if rigid_api.GetRigidBodyEnabledAttr().Get() is not True:
            raise RuntimeError(f"Object {spec.name} has a disabled rigid body: {body.GetPath()}")
        if rigid_api.GetKinematicEnabledAttr().Get() is True:
            raise RuntimeError(f"Object {spec.name} must be dynamic, but its body is kinematic.")
        colliders = [
            prim
            for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
        ]
        if not colliders:
            raise RuntimeError(f"Object {spec.name} has no enabled collision shape below {body.GetPath()}.")
        try:
            bounds = _range_bounds(cache.ComputeUntransformedBound(body))
            lower, upper = _upright_bounds(bounds, spec.upright_quat, _body_stretch(body))
        except ValueError as error:
            raise RuntimeError(f"Cannot inspect object geometry for {spec.name}: {error}") from error
        geometries.append(ObjectGeometry(spec.name, lower, upper, asset_name=spec.asset_name))

    shelf_path = _environment_zero_path(env, env.scene.cfg.shelf.prim_path)
    shelf = stage.GetPrimAtPath(shelf_path)
    if not shelf.IsValid():
        raise RuntimeError(f"Spawned shelf prim is missing: {shelf_path}")
    origin = tuple(float(value) for value in env.scene.env_origins[0].tolist())
    rules = _shelf_rules(cache, shelf, origin, env.cfg.spawn_rules)
    return tuple(geometries), rules


# Episode roles, metadata, and state reset


def _initialize_spawn_metadata(env: ManagerBasedEnv) -> None:
    if hasattr(env, "spawn_geometries"):
        return
    env.spawn_geometries, env.spawn_rules = inspect_spawn_geometry(env)
    count = len(OBJECT_SPECS)
    env.active_object_mask = torch.zeros((env.num_envs, count), device=env.device, dtype=torch.bool)
    env.active_object_ids = torch.full(
        (env.num_envs, env.spawn_rules.max_objects), -1, device=env.device, dtype=torch.long
    )
    env.target_object_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    env.blocker_object_id = torch.full_like(env.target_object_id, -1)
    env.access_region = torch.zeros((env.num_envs, 4), device=env.device)
    # These become valid only after actual RGB/instance-reference acceptance.
    env.target_visibility_validated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.target_visible_fraction = torch.full((env.num_envs,), torch.nan, device=env.device)
    env.spawn_initial_state = torch.zeros((env.num_envs, count, 13), device=env.device)
    env.spawn_reset_counts = torch.zeros(env.num_envs, dtype=torch.long)
    env.spawn_seeds = torch.zeros_like(env.spawn_reset_counts)
    env.spawn_samples = [None] * env.num_envs
    robot = env.scene["robot"]
    env.robot_initial_joint_pos = robot.data.default_joint_pos.clone()
    env.robot_start_tcp_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    env.robot_start_region = torch.zeros((env.num_envs, 6), device=env.device)
    env.robot_start_elbow_height = torch.zeros(env.num_envs, device=env.device)
    env.robot_start_attempts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    initialize_sensors(env)


def _scene_collision_boxes(env: ManagerBasedEnv, sample) -> list[tuple]:
    """Conservative active-object and shelf boxes in the model's env_0 frame."""
    boxes = []
    origin = env.scene.env_origins[0].cpu().numpy()
    for index in sample.active_ids:
        geometry = env.spawn_geometries[index]
        lower, upper = np.array(geometry.lower), np.array(geometry.upper)
        half = (upper - lower) * 0.5
        center = np.array(sample.positions[index]) + origin
        center[2] += (lower[2] + upper[2]) * 0.5
        c, s = math.cos(sample.yaws[index]), math.sin(sample.yaws[index])
        rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        boxes.append((center, rotation, half, OBJECT_SPECS[index].name))
    boxes.extend((box.center, box.rotation, box.half_extents, box.path)
                 for box in env.robot_pose_model.shelf_boxes)
    return boxes


def robot_start_collision_reason(env: ManagerBasedEnv, sample, transforms: dict) -> str | None:
    """Check every arm/hand collider against the scene, plus nonadjacent arm links."""
    model = env.robot_pose_model
    clearance = env.cfg.robot_start.collision_clearance
    obstacles = _scene_collision_boxes(env, sample)
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
        # Fixed sensor halves belong to the wrist housing for the existing
        # mechanical-neighbor exclusions; their scene bounds remain separate.
        arm_owner = "wrist_3_link" if box.body_name in (FORCE_TORQUE_PROXIMAL_BODY_NAME, FORCE_TORQUE_BODY_NAME) else box.body_name
        robot_boxes.append((arm_owner, center, rotation, half))
    arm_bodies = {"base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                  "wrist_1_link", "wrist_2_link", "wrist_3_link",
                  FORCE_TORQUE_PROXIMAL_BODY_NAME, FORCE_TORQUE_BODY_NAME}
    for index, (name, center, rotation, half) in enumerate(robot_boxes):
        if name not in arm_bodies:
            continue
        for other_name, other_center, other_rotation, other_half in robot_boxes[index + 1:]:
            if other_name not in arm_bodies or name == other_name:
                continue
            # Successive UR housings share a mechanical joint assembly. Their
            # enclosing boxes can overlap through an intervening shoulder or
            # wrist housing, so exclude links within two joints of each other.
            first_parent, other_parent = model.body_parents.get(name), model.body_parents.get(other_name)
            if other_name in (first_parent, model.body_parents.get(first_parent)) or name in (
                other_parent, model.body_parents.get(other_parent)
            ):
                continue
            if not oriented_box_separated(center, rotation, half, other_center, other_rotation, other_half, 0.002):
                return f"arm self: {name} / {other_name}"
    return None


def _sample_robot_start(env: ManagerBasedEnv, sample, seed: int) -> tuple:
    """Sample only in free space ahead of the Blocker; never step trial poses."""
    rules, model = env.cfg.robot_start, env.robot_pose_model
    region = robot_start_region(sample, env.spawn_geometries, env.spawn_rules, rules)
    lower, upper = np.array(region)[[0, 2, 4]], np.array(region)[[1, 3, 5]]
    origin = env.scene.env_origins[0].cpu().numpy()
    orientation = np.asarray(ROBOT_START_TCP_ROTATION)
    rng = random.Random(seed ^ 0x5D2EA91C)
    failures = {}
    for attempt in range(1, rules.max_attempts + 1):
        position = np.array([rng.uniform(lo, hi) for lo, hi in zip(lower, upper)]) + origin
        reason = "unreachable elbow-up IK"

        def accept(transforms):
            nonlocal reason
            actual = transforms[model.tcp_body_name][:3, 3] - origin
            if np.any(actual < lower) or np.any(actual > upper):
                reason = "TCP outside start region"
            else:
                reason = robot_start_collision_reason(env, sample, transforms)
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


def validate_robot_start_state(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Cross-check accepted FK/collision conditions against actual PhysX link poses."""
    if not env.cfg.robot_start.enabled:
        return
    from scipy.spatial.transform import Rotation

    robot, model = env.scene["robot"], env.robot_pose_model
    physical = robot.root_physx_view.get_link_transforms().cpu().numpy()
    origin_zero = env.scene.env_origins[0].cpu().numpy()
    for env_id in env_ids.cpu().tolist():
        origin = env.scene.env_origins[env_id].cpu().numpy()
        transforms = {}
        for body_id, name in enumerate(model.body_names):
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(physical[env_id, body_id, 3:7]).as_matrix()
            transform[:3, 3] = physical[env_id, body_id, :3] - origin + origin_zero
            transforms[name] = transform
        actual_tcp = transforms[model.tcp_body_name][:3, 3] - origin_zero
        actual_orientation = transforms[model.tcp_body_name][:3, :3]
        orientation_error = Rotation.from_matrix(
            np.asarray(ROBOT_START_TCP_ROTATION) @ actual_orientation.T
        ).magnitude()
        if orientation_error > 0.01 + 1.0e-5:
            raise RuntimeError(f"Actual tool does not face the shelf: TCP rotation error={orientation_error}rad.")
        # The assembled UR wrist/tool +Z coincides with palm TCP +Y. Check the
        # native wrist pose too, so a frame/mount mismatch cannot pass silently.
        tool_forward = transforms["wrist_3_link"][:3, 2]
        if np.linalg.norm(tool_forward - np.array((-1.0, 0.0, 0.0))) > 0.01 + 1.0e-5:
            raise RuntimeError(f"Actual UR tool forward axis does not face the shelf: {tool_forward}.")
        region = env.robot_start_region[env_id].cpu().numpy()
        if np.any(actual_tcp < region[[0, 2, 4]] - 1.0e-5) or np.any(actual_tcp > region[[1, 3, 5]] + 1.0e-5):
            raise RuntimeError("Actual hand TCP is outside its Blocker-front starting region.")
        elbow_height = model.elbow_height(transforms)
        if elbow_height < env.cfg.robot_start.min_elbow_height - 1.0e-5:
            raise RuntimeError(f"Robot reset selected an elbow-down/low-elbow pose: {elbow_height}.")
        reason = robot_start_collision_reason(env, env.spawn_samples[env_id], transforms)
        if reason is not None:
            raise RuntimeError(f"Actual robot starting pose has insufficient collision clearance: {reason}")
        expected = env.robot_start_tcp_pos_w[env_id].cpu().numpy()
        np.testing.assert_allclose(actual_tcp + origin, expected, atol=1.0e-5, rtol=0.0)


def reset_spawn_scene(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Sample active identities/roles/poses without altering unselected environments.

    The fixed access region and clearance witness belong to task/sampler metadata.
    The witness is never supplied as an actor goal or a Blocker final pose.
    """
    if env_ids.numel() == 0:
        return
    _initialize_spawn_metadata(env)
    env.initial_vision_sensor.prepare()
    if env.cfg.robot_start.enabled and not hasattr(env, "robot_pose_model"):
        env.robot_pose_model = RobotPoseModel.from_env(env)
    objects = env.scene["objects"]
    # Stage all samples first, so an impossible distribution fails before pool mutation.
    selected_ids = env_ids.cpu().tolist()
    samples = []
    seeds = []
    robot_starts = []
    for env_id in selected_ids:
        seed = int(env.cfg.seed) + 1_000_003 * env_id + 97_409 * int(env.spawn_reset_counts[env_id])
        seeds.append(seed)
        rng = random.Random(seed)
        # Fix externally assigned episode roles before generating any poses.
        targets = [
            index for index, spec in enumerate(OBJECT_SPECS)
            if env.cfg.target_asset is None or spec.asset_name == env.cfg.target_asset
            if env.cfg.blocker_asset is None or spec.asset_name != env.cfg.blocker_asset
        ]
        if not targets:
            raise ValueError(
                f"No Target asset distinct from Blocker: target={env.cfg.target_asset}, blocker={env.cfg.blocker_asset}"
            )
        target_id = rng.choice(targets)
        target_asset = OBJECT_SPECS[target_id].asset_name
        blockers = [
            index for index, spec in enumerate(OBJECT_SPECS)
            if spec.asset_name != target_asset
            and (env.cfg.blocker_asset is None or spec.asset_name == env.cfg.blocker_asset)
        ]
        if not blockers:
            raise ValueError(
                f"Blocker asset must differ from the unique Target asset {target_asset}: {env.cfg.blocker_asset}"
            )
        blocker_id = rng.choice(blockers)
        sample = sample_scene(
            env.spawn_geometries, env.spawn_rules, rng, target_id=target_id, blocker_id=blocker_id
        )
        samples.append(sample)
        robot_starts.append(_sample_robot_start(env, sample, seed) if env.cfg.robot_start.enabled else None)

    # Rendering requires native candidate poses. Preserve the selected state
    # and metadata so exhausted vision retries are transactional as well.
    robot = env.scene["robot"]
    # This writer expects actor pose with COM velocity, including moving bodies.
    previous_states = objects.data.object_state_w[env_ids].clone()
    previous_q, previous_v = robot.data.joint_pos[env_ids].clone(), robot.data.joint_vel[env_ids].clone()
    previous_targets = robot.data.joint_pos_target[env_ids].clone()
    previous_root = robot.data.root_state_w[env_ids].clone()
    previous_velocity_targets = robot.data.joint_vel_target[env_ids].clone()
    names = ("active_object_mask", "active_object_ids", "target_object_id", "blocker_object_id", "access_region",
             "target_visibility_validated", "target_visible_fraction", "blocker_visibility_validated",
             "blocker_visible_fraction", "initial_vision_rgb", "initial_vision_depth", "initial_vision_role_masks",
             "initial_camera_pose_w", "initial_camera_intrinsics", "initial_vision_pixel_counts",
             "initial_camera_attempts", "spawn_initial_state", "spawn_seeds",
             "spawn_reset_counts", "robot_initial_joint_pos", "robot_start_tcp_pos_w", "robot_start_region",
             "robot_start_elbow_height", "robot_start_attempts")
    if hasattr(env.cfg, "mdp"):
        names += ("initial_object_geometry", "initial_object_valid", "initial_object_roles", "initial_observed_object_ids")
    saved = {name: getattr(env, name)[selected_ids].clone() for name in names}
    previous_samples = [env.spawn_samples[index] for index in selected_ids]
    camera = env.scene["initial_camera"]
    previous_camera = (camera.data.pos_w[env_ids].clone(), camera.data.quat_w_ros[env_ids].clone())
    fixed_ids = [sample.active_ids for sample in samples]
    try:
        for layout_attempt in range(env.cfg.initial_vision.max_layout_attempts):
            _apply_spawn_candidates(env, env_ids, samples, seeds, robot_starts)
            env.spawn_reset_counts[selected_ids] = saved["spawn_reset_counts"] + 1
            env.scene.write_data_to_sim()
            env.sim.forward()
            try:
                images = env.initial_vision_sensor.capture(env_ids, layout_attempt=layout_attempt)
            except InitialVisionRejected as error:
                if layout_attempt + 1 >= env.cfg.initial_vision.max_layout_attempts:
                    raise
                print(f"[VISION RETRY]: layout={layout_attempt + 1}: {error}", flush=True)
                samples, robot_starts = [], []
                for env_id, seed, active_ids in zip(selected_ids, seeds, fixed_ids):
                    retry_seed = seed ^ ((layout_attempt + 1) * 0x71D5AF)
                    old = env.spawn_samples[env_id]
                    sample = sample_scene(env.spawn_geometries, env.spawn_rules, random.Random(retry_seed),
                                          target_id=old.target_id, blocker_id=old.blocker_id, active_ids=active_ids)
                    samples.append(sample)
                    robot_starts.append(_sample_robot_start(env, sample, retry_seed) if env.cfg.robot_start.enabled else None)
                continue
            env.initial_vision_sensor.commit(images)
            env.tool_force_torque_sensor.reset(env_ids)
            env.binary_tactile_sensor.reset(env_ids)
            # Wrenches stay invalid at this unchanged physics timestamp.
            env.ft_last_update_time = env.sim.current_time
            env.tactile_last_update_time = env.sim.current_time
            return
    except Exception:
        objects.write_object_state_to_sim(previous_states, env_ids=env_ids)
        robot.write_root_state_to_sim(previous_root, env_ids=env_ids)
        robot.write_joint_state_to_sim(previous_q, previous_v, env_ids=env_ids)
        robot.set_joint_position_target(previous_targets, env_ids=env_ids)
        robot.set_joint_velocity_target(previous_velocity_targets, env_ids=env_ids)
        for name, value in saved.items():
            getattr(env, name)[selected_ids] = value
        for index, sample in zip(selected_ids, previous_samples):
            env.spawn_samples[index] = sample
        objects.parked_mask = ~env.active_object_mask
        camera.set_world_poses(*previous_camera, env_ids, convention="ros")
        camera.reset(env_ids)
        env.scene.write_data_to_sim()
        env.sim.forward()
        raise


def _apply_spawn_candidates(env, env_ids, samples, seeds, robot_starts):
    selected_ids = env_ids.cpu().tolist()
    objects = env.scene["objects"]
    mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    states = objects.data.default_object_state[env_ids].clone()
    states[..., 7:] = 0.0
    for row, (env_id, sample, seed) in enumerate(zip(selected_ids, samples, seeds)):
        env.active_object_mask[env_id] = False
        env.active_object_ids[env_id] = -1
        env.active_object_ids[env_id, :len(sample.active_ids)] = torch.tensor(sample.active_ids, device=env.device)
        env.active_object_mask[env_id, list(sample.active_ids)] = True
        env.target_object_id[env_id] = sample.target_id
        env.blocker_object_id[env_id] = sample.blocker_id
        env.access_region[env_id] = torch.tensor(sample.access_region, device=env.device)
        env.target_visibility_validated[env_id] = False
        env.target_visible_fraction[env_id] = torch.nan
        env.spawn_samples[env_id] = sample
        env.spawn_seeds[env_id] = seed
        env.spawn_reset_counts[env_id] += 1
        for object_id in sample.active_ids:
            geometry, yaw = env.spawn_geometries[object_id], sample.yaws[object_id]
            cx = (geometry.lower[0] + geometry.upper[0]) * 0.5
            cy = (geometry.lower[1] + geometry.upper[1]) * 0.5
            cosine, sine = math.cos(yaw), math.sin(yaw)
            px, py, pz = sample.positions[object_id]
            states[row, object_id, :3] = torch.tensor(
                (px - cosine * cx + sine * cy, py - sine * cx - cosine * cy, pz), device=env.device
            )
            # Quaternion product: world-Z yaw * canonical upright rotation.
            qw, qx, qy, qz = OBJECT_SPECS[object_id].upright_quat
            cw, sw = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            states[row, object_id, 3:7] = torch.tensor(
                (cw * qw - sw * qz, cw * qx - sw * qy, cw * qy + sw * qx, cw * qz + sw * qw),
                device=env.device,
            )

    states[..., :3] += env.scene.env_origins[env_ids, None, :]
    objects.write_object_state_to_sim(states, env_ids=env_ids)
    env.spawn_initial_state[env_ids] = states
    robot = env.scene["robot"]
    joint_positions = robot.data.default_joint_pos[env_ids].clone()
    for row, (env_id, start) in enumerate(zip(selected_ids, robot_starts)):
        if start is not None:
            q, region, tcp_local, elbow_height, attempts = start
            joint_positions[row] = torch.as_tensor(q, dtype=joint_positions.dtype, device=env.device)
            env.robot_start_region[env_id] = torch.tensor(region, device=env.device)
            env.robot_start_tcp_pos_w[env_id] = torch.as_tensor(tcp_local, device=env.device) + env.scene.env_origins[env_id]
            env.robot_start_elbow_height[env_id] = elbow_height
            env.robot_start_attempts[env_id] = attempts
    robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions), env_ids=env_ids)
    robot.set_joint_position_target(joint_positions, env_ids=env_ids)
    env.robot_initial_joint_pos[env_ids] = joint_positions
    # Countergravity is applied in world coordinates at COM by TaskObjectPool
    # each physics substep. Active task objects keep their native dynamics.
    objects.parked_mask = ~env.active_object_mask
    objects.gravity = torch.tensor(env.cfg.sim.gravity, device=env.device)
