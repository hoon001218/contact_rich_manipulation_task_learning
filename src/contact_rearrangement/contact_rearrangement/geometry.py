"""Read the spawned USD geometry used to constrain randomized shelf scenes."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from typing import TYPE_CHECKING

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .layout import OBJECT_SPECS
from .sampling import ObjectGeometry, SpawnRules

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

Point3 = tuple[float, float, float]
Bounds = tuple[Point3, Point3]


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
