"""Native scene assets and once-only USD geometry/kinematics inspection.

The simulator clones one authored robot, shelf and fixed 10-body object pool.
Episode roles and poses are selected by MDP reset events; no prim topology is
changed during reset. This module has no dependency on the previous project.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
import warp as wp
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

NUCLEUS_ROOT = os.environ.get("CONTACT_MANIPULATION_NUCLEUS_ROOT", "omniverse://192.168.0.13").rstrip("/")
DEFAULT_SHELF_USD_PATH = f"{NUCLEUS_ROOT}/Library/Shelf/Arena/Collected_speedrack_shape/speedrack_shape.usd"
SHELF_POSITION = (-0.7, 0.0, 0.0)


@clone
def spawn_shelf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Use native USD spawning and explicitly define mass on the shelf body.

    The source USD has RigidBodyAPI but no MassAPI; UsdFileCfg.mass_props only
    modifies an existing MassAPI, so applying the schema is necessary here.
    """
    spawn_cfg = cfg.copy()
    spawn_cfg.mass_props = None
    prim = sim_utils.spawn_from_usd(prim_path, spawn_cfg, translation, orientation, **kwargs)
    bodies = sim_utils.get_all_matching_child_prims(
        prim_path, predicate=lambda p: p.HasAPI(UsdPhysics.RigidBodyAPI))
    if len(bodies) != 1:
        raise ValueError("Shelf must contain exactly one rigid body.")
    sim_utils.define_mass_properties(str(bodies[0].GetPath()), cfg.mass_props)
    return prim


@dataclass(frozen=True)
class ObjectSpec:
    """An authored physics asset and its canonical upright orientation (wxyz)."""

    name: str
    relative_usd_path: str
    upright_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    asset_name: str = ""

    @property
    def usd_path(self) -> str:
        return os.environ.get(
            f"CONTACT_MANIPULATION_{(self.asset_name or self.name).upper()}_USD_PATH",
            f"{NUCLEUS_ROOT}/{self.relative_usd_path}",
        )


# Axis_Aligned_Physics currently contains these four authored physics assets.
# Rendered labels and mustard's cap show that local -Y is the top of all four.
# Rotate -Y to world +Z before applying a sampled world-Z yaw.
_YCB_UPRIGHT = (math.sqrt(0.5), -math.sqrt(0.5), 0.0, 0.0)
_YCB_FOLDER = "NVIDIA/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics"
ASSET_SPECS = (
    ObjectSpec("bottle_1", "Library/Shelf/Objects/Bottle_6/Bottle_6.usd"),
    ObjectSpec("cup_1", "Library/Shelf/Objects/Collected_Cup_1/Cup_1.usd"),
    ObjectSpec("cup_2", "Library/Shelf/Objects/Collected_Cup_4/Cup_4.usd"),
    ObjectSpec("mug_1", "Library/Shelf/Objects/Collected_Mug_2/Mug_2.usd"),
    ObjectSpec("mug_2", "Library/Shelf/Objects/Collected_Mug_3/Mug_3.usd"),
    ObjectSpec("can_1", "Library/Shelf/Objects/Can_6/Can_6.usd"),
    ObjectSpec("ycb_003_cracker_box", f"{_YCB_FOLDER}/003_cracker_box.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_004_sugar_box", f"{_YCB_FOLDER}/004_sugar_box.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_005_tomato_soup_can", f"{_YCB_FOLDER}/005_tomato_soup_can.usd", _YCB_UPRIGHT),
    ObjectSpec("ycb_006_mustard_bottle", f"{_YCB_FOLDER}/006_mustard_bottle.usd", _YCB_UPRIGHT),
)
# Load a reusable pool so non-Target assets can occupy every surrounding slot,
# identities can change on reset without rebuilding USD or PhysX topology.
MAX_ACTIVE_OBJECTS = 10
OBJECT_COPIES_PER_ASSET = 1
OBJECT_SPECS = tuple(
    ObjectSpec(
        name=spec.name if copy_index == 0 else f"{spec.name}_copy_{copy_index + 1:02d}",
        relative_usd_path=spec.relative_usd_path,
        upright_quat=spec.upright_quat,
        asset_name=spec.name,
    )
    for spec in ASSET_SPECS
    for copy_index in range(OBJECT_COPIES_PER_ASSET)
)
OBJECT_NAMES = tuple(spec.name for spec in OBJECT_SPECS)


def parking_position(index: int) -> tuple[float, float, float]:
    """Isolate unused bodies above the scene, clear of floor/shelf and each other."""
    return (-0.64 + 0.32 * (index % 5), -0.64 + 0.32 * ((index // 5) % 5), 4.0 + 0.4 * (index // 25))


# Robot assembly.

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_index_1_joint",
    "right_middle_1_joint",
    "right_ring_1_joint",
    "right_little_1_joint",
)
FINGERTIP_BODY_NAMES = (
    "right_thumb_force_sensor_4",
    "right_index_force_sensor_3",
    "right_middle_force_sensor_3",
    "right_ring_force_sensor_3",
    "right_little_force_sensor_3",
)
# Hand placement/IK reference and tactile pad; the OSC/observation EEF is wrist_3_link.
TCP_BODY_NAME = "right_plam_force_sensor"
FORCE_TORQUE_PROXIMAL_BODY_NAME = "SensorMountProximal"
FORCE_TORQUE_BODY_NAME = "ToolForceTorqueLink"
FORCE_TORQUE_JOINT_NAME = "ToolForceTorqueJoint"
TOOL_SENSOR_MOUNT_JOINT_NAME = "ToolSensorMountJoint"
TACTILE_BODY_NAMES = (
    TCP_BODY_NAME,
    *(f"right_thumb_force_sensor_{index}" for index in range(1, 5)),
    *(
        f"right_{finger}_force_sensor_{index}"
        for finger in ("index", "middle", "ring", "little")
        for index in range(1, 4)
    ),
)

DEFAULT_UR5E_USD_PATH = (
    f"{NUCLEUS_ROOT}/NVIDIA/Assets/Isaac/5.0/"
    "Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_HAND_URDF_PATH = (
    REPOSITORY_ROOT
    / "assets/robots/Roas_provided_inspire_urdf/RH56E2 URDF/"
    "RH56E2_R_2025_9_11/urdf/RH56E2_R_2025_9_11_relative.urdf"
)

_UR_TOOL_FRAME_CANDIDATES = ("tool0", "tool_frame", "flange", "wrist_3_link")
_HAND_BASE_BODY_CANDIDATES = ("base_link",)
_PREEXISTING_MOUNT_JOINT_NAMES = ("robot_gripper_joint",)


def validate_hand_urdf_assets(urdf_path: str) -> None:
    """Fail early when the URDF or one of its meshes is missing or still an LFS pointer."""
    source = Path(urdf_path)
    if not source.is_file():
        raise FileNotFoundError(f"Hand URDF not found: {source}")

    missing: list[Path] = []
    lfs_pointers: list[Path] = []
    urdf = ElementTree.parse(source)
    tactile_links = {link.attrib["name"]: link for link in urdf.findall("link") if "force_sensor" in link.attrib["name"]}
    if set(tactile_links) != set(TACTILE_BODY_NAMES):
        raise ValueError(
            f"Hand URDF must contain the 17 canonical tactile links; "
            f"missing={sorted(set(TACTILE_BODY_NAMES) - set(tactile_links))}, "
            f"unexpected={sorted(set(tactile_links) - set(TACTILE_BODY_NAMES))}."
        )
    for name, link in tactile_links.items():
        if link.find("collision/geometry/mesh") is None:
            raise ValueError(f"Tactile link {name!r} needs its own URDF collision mesh.")
    for mesh in urdf.findall(".//mesh"):
        mesh_path = source.parent / mesh.attrib["filename"]
        if not mesh_path.is_file():
            missing.append(mesh_path)
            continue
        with mesh_path.open("rb") as stream:
            is_pointer = stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")
        if is_pointer:
            lfs_pointers.append(mesh_path)

    if missing:
        preview = ", ".join(str(path) for path in missing[:3])
        raise FileNotFoundError(f"Hand URDF references missing mesh files: {preview}")
    if lfs_pointers:
        preview = ", ".join(str(path) for path in lfs_pointers[:3])
        raise RuntimeError(
            f"Hand mesh files are Git LFS pointers, not mesh binaries ({len(lfs_pointers)} references). "
            f"Fetch the LFS assets before training. Examples: {preview}"
        )


def _world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def _find_named_prim(
    stage: Usd.Stage,
    subtree_path: str,
    candidates: tuple[str, ...],
    *,
    rigid_body_only: bool = False,
) -> Usd.Prim:
    root = stage.GetPrimAtPath(subtree_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid asset subtree: {subtree_path}")
    prims = list(Usd.PrimRange(root))
    for candidate in candidates:
        for prim in prims:
            if prim.GetName() == candidate and (not rigid_body_only or prim.HasAPI(UsdPhysics.RigidBodyAPI)):
                return prim
    available = [prim.GetName() for prim in prims if not rigid_body_only or prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    raise RuntimeError(f"Could not find any of {candidates} below {subtree_path}. Available names: {available}")


def _nearest_rigid_body_ancestor(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    raise RuntimeError(f"No rigid-body ancestor found for {prim.GetPath()}")


def _make_offset_matrix(
    translation: tuple[float, float, float], rotation_deg: tuple[float, float, float]
) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    rotation = (
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rotation_deg[0])
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), rotation_deg[1])
        * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_deg[2])
    )
    matrix.SetRotate(rotation)
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def _matrix_to_pose(
    matrix: Gf.Matrix4d,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat().GetNormalized()
    imaginary = quaternion.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (float(quaternion.GetReal()), float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _set_world_transform_on_reference_root(root_prim: Usd.Prim, world_matrix: Gf.Matrix4d) -> None:
    local_matrix = world_matrix * _world_transform(root_prim.GetParent()).GetInverse()
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(local_matrix)
    xformable.SetResetXformStack(False)


def _create_fixed_joint(
    stage: Usd.Stage,
    joint_path: str,
    parent_body: Usd.Prim,
    child_body: Usd.Prim,
    joint_frame_world: Gf.Matrix4d,
) -> None:
    parent_pos, parent_rot = _matrix_to_pose(joint_frame_world * _world_transform(parent_body).GetInverse())
    child_pos, child_rot = _matrix_to_pose(joint_frame_world * _world_transform(child_body).GetInverse())
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([parent_body.GetPath()])
    joint.CreateBody1Rel().SetTargets([child_body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*parent_pos))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(parent_rot[0], Gf.Vec3f(*parent_rot[1:])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pos))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(child_rot[0], Gf.Vec3f(*child_rot[1:])))
    joint.CreateCollisionEnabledAttr().Set(False)


def _create_sensor_mount(
    robot_root: Usd.Prim,
    tool_body: Usd.Prim,
    start_world: Gf.Vec3d,
    length: float,
    radius: float,
    mass: float,
) -> tuple[Gf.Vec3d, Usd.Prim]:
    """Split the spacer at a fixed articulation load-cell joint at its center.

    Both halves are separate links so their mass/inertia are explicit without
    modifying the supplied wrist mass. The distal link's actor frame and its
    incoming joint's child frame both originate at the sensor center and align
    with physical wrist/tool XYZ. All hand loads cross this single fixed joint.
    """
    stage = robot_root.GetStage()
    sensor_world = Gf.Matrix4d(1.0)
    sensor_world.SetRotate(Gf.Transform(_world_transform(tool_body)).GetRotation())
    displacement = sensor_world.TransformDir(Gf.Vec3d(0.0, 0.0, length))
    sensor_world.SetTranslateOnly(start_world + displacement * 0.5)
    half_length, half_mass = length * 0.5, mass * 0.5
    inertia_radial = half_mass * (3.0 * radius**2 + half_length**2) / 12.0
    inertia_axial = half_mass * radius**2 / 2.0

    def half_link(name: str, world: Gf.Matrix4d, center_z: float, color: tuple[float, float, float]):
        body = UsdGeom.Xform.Define(stage, robot_root.GetPath().AppendChild(name)).GetPrim()
        _set_world_transform_on_reference_root(body, world)
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateDisableGravityAttr(False)
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(half_mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, center_z))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia_radial, inertia_radial, inertia_axial))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        cylinder = UsdGeom.Cylinder.Define(stage, body.GetPath().AppendChild("Cylinder"))
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateHeightAttr(half_length)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateExtentAttr([
            Gf.Vec3f(-radius, -radius, -half_length * 0.5),
            Gf.Vec3f(radius, radius, half_length * 0.5),
        ])
        cylinder.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, center_z))
        cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(cylinder.GetPrim())
        return body

    proximal_world = Gf.Matrix4d(sensor_world)
    proximal_world.SetTranslateOnly(start_world + displacement * 0.25)
    proximal = half_link(FORCE_TORQUE_PROXIMAL_BODY_NAME, proximal_world, 0.0, (0.18, 0.20, 0.23))
    distal = half_link(FORCE_TORQUE_BODY_NAME, sensor_world, length * 0.25, (0.24, 0.27, 0.30))
    proximal_joint_world = Gf.Matrix4d(sensor_world)
    proximal_joint_world.SetTranslateOnly(start_world)
    _create_fixed_joint(stage, str(robot_root.GetPath().AppendChild(TOOL_SENSOR_MOUNT_JOINT_NAME)),
                        tool_body, proximal, proximal_joint_world)
    _create_fixed_joint(stage, str(robot_root.GetPath().AppendChild(FORCE_TORQUE_JOINT_NAME)),
                        proximal, distal, sensor_world)
    return start_world + displacement, distal


def _remove_nested_articulation_roots(stage: Usd.Stage, subtree_path: str) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(subtree_path)):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _deactivate_preexisting_mount_joints(stage: Usd.Stage, robot_path: str) -> None:
    """Disable mount joints authored in the UR5e USD before attaching this hand."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.GetName() not in _PREEXISTING_MOUNT_JOINT_NAMES:
            continue
        joint = UsdPhysics.Joint(prim)
        if joint:
            joint.GetJointEnabledAttr().Set(False)
            prim.SetActive(False)


def _deinstance_subtree(stage: Usd.Stage, subtree_path: str) -> None:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(subtree_path)):
        if prim.IsInstance():
            prim.SetInstanceable(False)


def _tactile_parent_proxies(points: np.ndarray, pads: list[np.ndarray]):
    """Two low-vertex convexes, with the pad-side piece recessed behind the pads.

    This is template authoring, never an episode/physics-step operation. Each
    pad's thin PCA axis defines its face; the sign points away from the parent.
    Try the four tangent cuts around the first pad and retain the cut with the
    smallest worst surface half-space error. Mass/inertia are not derived here.
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection

    points = np.asarray(points, dtype=np.float64)
    directions = np.array([d for d in product((-1., 0., 1.), repeat=3) if any(d)])
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    envelope = np.column_stack((directions, -(points @ directions.T).max(axis=0)))
    faces, tangents = [], None
    for pad in pads:
        centered = pad - pad.mean(axis=0)
        _, axes = np.linalg.eigh(centered.T @ centered / len(pad))
        normal = axes[:, 0]
        if normal @ (pad.mean(axis=0) - points.mean(axis=0)) < 0:
            normal = -normal
        # Keep the core behind the pad, including its rounded perimeter.
        faces.append(np.r_[normal, -(pad @ normal).max() + .0012])
        if tangents is None:
            tangents = axes[:, 1:].T
    faces = np.asarray(faces)

    def polyhedron(planes):
        inside = points[(points @ planes[:, :3].T + planes[:, 3] < -1.e-8).all(axis=1)]
        if len(inside) < 4:
            return None
        vertices = HalfspaceIntersection(planes, inside.mean(axis=0)).intersections
        hull = ConvexHull(vertices)
        if len(vertices) > 64:
            raise RuntimeError('Hand proxy exceeds the native convex vertex budget.')
        # Consistent outward winding for the authored mesh.
        triangles = hull.simplices.copy()
        a, b, c = vertices[triangles[:, 0]], vertices[triangles[:, 1]], vertices[triangles[:, 2]]
        reverse = (np.cross(b-a, c-a) * hull.equations[:, :3]).sum(axis=1) < 0
        triangles[reverse] = triangles[reverse][:, [0, 2, 1]]
        return vertices, triangles

    candidates = []
    for tangent in tangents:
        for sign in (-1., 1.):
            normal = sign * tangent
            boundary = (pads[0] @ normal).min() - .001
            cut = np.r_[normal, -boundary]
            regions = (np.vstack((envelope, cut)), np.vstack((envelope, -cut, faces)))
            pieces = [polyhedron(region) for region in regions]
            if any(piece is None for piece in pieces):
                continue
            error = np.minimum.reduce([
                (points @ region[:, :3].T + region[:, 3]).max(axis=1) for region in regions
            ]).clip(min=0)
            candidates.append((float(error.max()), pieces))
    if not candidates:
        raise RuntimeError('Cannot construct sensor-clear hand collision proxies.')
    return min(candidates, key=lambda candidate: candidate[0])[1]


def _preserve_tactile_parent_slots(stage: Usd.Stage, hand_path: str) -> None:
    """Replace only five hand parent colliders with two convex pieces each.

    Keep the 17 tactile bodies, all robot joints, and authored inertial properties.
    Existing visual meshes are untouched. Native PhysX cooks the small convexes.
    """
    parents = ("right_plam_1", *(f"right_{finger}_2" for finger in ("index", "middle", "ring", "little")))
    bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(hand_path))
              if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def collision_points(body, reference):
        colliders = [p for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
        meshes = {str(p.GetPath()): p for collider in colliders for p in Usd.PrimRange(collider)
                  if p.IsA(UsdGeom.Mesh)}
        clouds = []
        for mesh in meshes.values():
            transform = np.asarray(cache.GetLocalToWorldTransform(mesh) * reference)
            local = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
            clouds.append(local @ transform[:3, :3] + transform[3, :3])
        if not clouds:
            raise RuntimeError(f'No collision mesh for hand body {body.GetPath()}.')
        return np.unique(np.concatenate(clouds), axis=0), colliders

    for name in parents:
        body = bodies[name]
        reference = cache.GetLocalToWorldTransform(body).GetInverse()
        points, originals = collision_points(body, reference)
        pad_names = ('right_plam_force_sensor',) if name == 'right_plam_1' else (
            name.removesuffix('_2') + '_force_sensor_2', name.removesuffix('_2') + '_force_sensor_3')
        pads = [collision_points(bodies[pad], reference)[0] for pad in pad_names]
        pieces = _tactile_parent_proxies(points, pads)
        for collider in originals:
            if '/collisions/' not in str(collider.GetPath()):
                raise RuntimeError('Hand proxy replacement requires separate collision and visual subtrees.')
            collider.SetActive(False)
        for index, (vertices, triangles) in enumerate(pieces):
            mesh = UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild(f'TactileParentProxy_{index}'))
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
            mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
            mesh.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr('none')
            mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            sim_utils.define_mesh_collision_properties(str(mesh.GetPath()),
                sim_utils.ConvexHullPropertiesCfg(hull_vertex_limit=64), stage=stage)
            sim_utils.modify_collision_properties(str(mesh.GetPath()),
                sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.), stage=stage)
        # Ensure final hand-only collision overrides survive instance sharing.
        body.CreateAttribute('contactManipulation:handProxy', Sdf.ValueTypeNames.Bool, custom=True).Set(True)


@clone
def spawn_ur5e_hand(
    prim_path: str,
    cfg: "Ur5eHandSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn UR5e, import the RH56E2 right-hand URDF, and join them as one articulation."""
    del kwargs
    if not all(math.isfinite(value) and value > 0.0 for value in (
        cfg.sensor_mount_length, cfg.sensor_mount_radius, cfg.sensor_mount_mass
    )):
        raise ValueError("Sensor mount length, radius and mass must be finite and positive.")
    validate_hand_urdf_assets(cfg.hand_urdf_path)

    stage = get_current_stage()
    ur_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.ur5e_usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=False,  # Only the hand's tactile links need reporting.
    )
    ur_cfg.func(prim_path, ur_cfg, translation=translation, orientation=orientation)

    hand_path = f"{prim_path}/ROASRightHand"
    hand_cfg = sim_utils.UrdfFileCfg(
        asset_path=cfg.hand_urdf_path,
        fix_base=False,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        make_instanceable=False,
        self_collision=False,
        collider_type="convex_hull",
        activate_contact_sensors=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=20.0, damping=1.0),
        ),
    )
    hand_cfg.func(hand_path, hand_cfg)

    robot_root = stage.GetPrimAtPath(prim_path)
    hand_root = stage.GetPrimAtPath(hand_path)
    if not robot_root.IsValid() or not hand_root.IsValid():
        raise RuntimeError(
            "UR5e or hand did not compose into the stage. Check CONTACT_MANIPULATION_UR5E_USD_PATH and "
            "CONTACT_MANIPULATION_HAND_URDF_PATH."
        )

    _deinstance_subtree(stage, hand_path)
    _preserve_tactile_parent_slots(stage, hand_path)
    tool_frame = _find_named_prim(stage, prim_path, cfg.tool_frame_candidates)
    tool_body = _nearest_rigid_body_ancestor(tool_frame)
    if tool_body.IsInstance():
        tool_body.SetInstanceable(False)
    hand_base = _find_named_prim(stage, hand_path, cfg.hand_base_candidates, rigid_body_only=True)

    _deactivate_preexisting_mount_joints(stage, prim_path)
    hand_root_world = _world_transform(hand_root)
    hand_base_relative_to_root = _world_transform(hand_base) * hand_root_world.GetInverse()
    mount_offset = _make_offset_matrix(cfg.mount_translation, cfg.mount_rotation_deg)
    # The authored tool0 permutes the physical wrist/tool axes. Compose the
    # hand in its tool body frame before applying the roll about forward +Z.
    tool_body_world = _world_transform(tool_body)
    attachment_in_tool_body = _world_transform(tool_frame) * tool_body_world.GetInverse()
    tool_roll = _make_offset_matrix((0.0, 0.0, 0.0), (0.0, 0.0, cfg.mount_tool_roll_deg))
    # USD row vectors: vendor alignment, attachment frame, tool +Z roll, world.
    mount_world = mount_offset * attachment_in_tool_body * tool_roll * tool_body_world
    hand_anchor, sensor_body = _create_sensor_mount(
        robot_root, tool_body, mount_world.ExtractTranslation(), cfg.sensor_mount_length,
        cfg.sensor_mount_radius, cfg.sensor_mount_mass,
    )
    mount_world.SetTranslateOnly(hand_anchor)
    desired_hand_root_world = hand_base_relative_to_root.GetInverse() * mount_world
    _set_world_transform_on_reference_root(hand_root, desired_hand_root_world)

    _remove_nested_articulation_roots(stage, hand_path)
    _create_fixed_joint(stage, f"{prim_path}/UR5e_hand_mount_joint", sensor_body, hand_base, mount_world)
    return robot_root


@configclass
class Ur5eHandSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for the assembled robot."""

    func: Callable = spawn_ur5e_hand
    ur5e_usd_path: str = DEFAULT_UR5E_USD_PATH
    hand_urdf_path: str = str(DEFAULT_HAND_URDF_PATH)
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=1.0,
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=2,
    )
    tool_frame_candidates: tuple[str, ...] = _UR_TOOL_FRAME_CANDIDATES
    hand_base_candidates: tuple[str, ...] = _HAND_BASE_BODY_CANDIDATES
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # The right-hand vendor frame is mirrored relative to the former left-hand
    # asset. Rotate it by -90 degrees so the fingers extend away from the robot
    # along the flange's forward direction rather than back into the arm.
    mount_rotation_deg: tuple[float, float, float] = (0.0, -90.0, 0.0)
    mount_tool_roll_deg: float = -90.0
    # Sensor and adapter proxy; 50 mm diameter matches the hand's base flange.
    sensor_mount_length: float = 0.05
    sensor_mount_radius: float = 0.025
    # Nominal sensor/adapter mass pending hardware measurement; split equally.
    sensor_mount_mass: float = 0.20


def make_ur5e_hand_cfg() -> ArticulationCfg:
    """Create the assembled articulation with environment-overridable asset paths."""
    hand_urdf_path = os.environ.get(
        "CONTACT_MANIPULATION_HAND_URDF_PATH", str(DEFAULT_HAND_URDF_PATH)
    )
    spawn_cfg = Ur5eHandSpawnerCfg(
        ur5e_usd_path=os.environ.get("CONTACT_MANIPULATION_UR5E_USD_PATH", DEFAULT_UR5E_USD_PATH),
        hand_urdf_path=hand_urdf_path,
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            # The examples place the shelf on -x and the robot base at this height.
            pos=(0.0, 0.0, 0.79505),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": -0.2496,
                "shoulder_lift_joint": -1.7995,
                "elbow_joint": 1.1994,
                "wrist_1_joint": 0.6010,
                "wrist_2_joint": 1.5712,
                # Retain hand_rl's open-hand posture for spawn inspection.
                "wrist_3_joint": -1.5700,
                ".*(thumb|index|middle|ring|little).*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Native explicit actuator clips OSC feed-forward efforts before
            # submitting them to PhysX; zero gains leave OSC as the arm controller.
            "arm": IdealPDActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
                effort_limit={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                effort_limit_sim={
                    "shoulder_pan_joint": 150.0,
                    "shoulder_lift_joint": 150.0,
                    "elbow_joint": 150.0,
                    "wrist_1_joint": 28.0,
                    "wrist_2_joint": 28.0,
                    "wrist_3_joint": 28.0,
                },
                velocity_limit_sim={
                    "shoulder_pan_joint": 3.14,
                    "shoulder_lift_joint": 3.14,
                    "elbow_joint": 3.14,
                    "wrist_1_joint": 6.28,
                    "wrist_2_joint": 6.28,
                    "wrist_3_joint": 6.28,
                },
                stiffness=0.0,
                damping=0.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[".*(thumb|index|middle|ring|little).*"],
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
                stiffness={
                    "(?!(?:" + "|".join(HAND_JOINT_NAMES) + ")$).*(thumb|index|middle|ring|little).*": 0.0,
                    **{name: 20.0 for name in HAND_JOINT_NAMES},
                },
                damping={
                    "(?!(?:" + "|".join(HAND_JOINT_NAMES) + ")$).*(thumb|index|middle|ring|little).*": 0.0,
                    **{name: 1.0 for name in HAND_JOINT_NAMES},
                },
            ),
        },
    )


# Robot reset kinematics. Batched tensor candidates never write articulation
# state or advance the simulation; scalar FK remains an independent oracle.


@dataclass(frozen=True)
class RobotColliderBox:
    """Conservative bounds of an authored collider in its physical link frame."""

    body_name: str
    center_local: np.ndarray
    rotation_local: np.ndarray
    half_extents: np.ndarray
    path: str


@dataclass(frozen=True)
class ShelfColliderBox:
    """Authored shelf collider bounds in the env_0 world frame."""

    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray
    path: str


@dataclass(frozen=True)
class _RobotJoint:
    name: str
    parent: str
    child: str
    frame0: np.ndarray
    frame1_inverse: np.ndarray
    axis: np.ndarray
    index: int | None
    reverse: bool = False


def _rotation_wxyz(quaternion) -> np.ndarray:
    """Convert a scalar-first quaternion to a column-vector rotation matrix."""
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(quaternion, dtype=np.float64)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def _rigid_matrix(position, quaternion) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation_wxyz(quaternion)
    matrix[:3, 3] = position
    return matrix


def _gf_column(matrix: Gf.Matrix4d) -> np.ndarray:
    """USD uses row-vector matrices; the kinematic model uses column vectors."""
    return np.asarray(matrix, dtype=np.float64).T.copy()


def _rotation_part(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix[:3, :3])
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def _joint_frame(position, quaternion, stretch: np.ndarray) -> np.ndarray:
    matrix = _rigid_matrix(position, quaternion)
    matrix[:3, 3] = stretch @ matrix[:3, 3]
    # Articulation links have uniform asset scale. Keeping this conversion also
    # handles orthogonal scale imported in a fixed hand link's local frame.
    matrix[:3, :3] = _rotation_part(stretch @ matrix[:3, :3])
    return matrix


def _box_from_usd(prim: Usd.Prim, matrix: np.ndarray, cache: UsdGeom.BBoxCache):
    """Bake collider and ancestor scale into a conservative oriented box."""
    if prim.IsA(UsdGeom.Mesh):
        # Collision meshes can inherit purpose=guide and have no authored
        # extent. Read the actual vertices so visual purpose/visibility cannot
        # silently remove collision geometry from the reset acceptance check.
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"Collision mesh has no vertices: {prim.GetPath()}")
        if not np.isfinite(points).all():
            raise ValueError(f"Collision mesh has nonfinite vertices: {prim.GetPath()}")
        lower, upper = points.min(axis=0), points.max(axis=0)
    else:
        bounds = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            raise ValueError(f"Empty collision bounds: {prim.GetPath()}")
        lower, upper = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
    if not np.isfinite([lower, upper]).all() or np.any(upper < lower):
        raise ValueError(f"Invalid collision bounds: {prim.GetPath()}")
    rotation = _rotation_part(matrix)
    corners = np.asarray(list(product(*zip(lower, upper))))
    transformed = corners @ matrix[:3, :3].T + matrix[:3, 3]
    aligned = transformed @ rotation
    aligned_lower, aligned_upper = aligned.min(axis=0), aligned.max(axis=0)
    center = rotation @ ((aligned_lower + aligned_upper) * 0.5)
    half_extents = np.maximum((aligned_upper - aligned_lower) * 0.5, 1.0e-6)
    return center, rotation, half_extents


_ArmVector = wp.types.vector(length=6, dtype=wp.float32)
_ArmMatrix = wp.types.matrix(shape=(6, 6), dtype=wp.float32)
_ArmPoints = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.func
def _ik_orientation_error(target: wp.mat33, current: wp.mat33):
    relative = target * wp.transpose(current)
    vector = 0.5 * wp.vec3(relative[2, 1] - relative[1, 2], relative[0, 2] - relative[2, 0],
                            relative[1, 0] - relative[0, 1])
    sine = wp.length(vector)
    cosine = wp.clamp(0.5 * (relative[0, 0] + relative[1, 1] + relative[2, 2] - 1.0), -1.0, 1.0)
    angle = wp.atan2(sine, cosine)
    if cosine < -0.9999 and sine < 1.0e-4:
        index = int(0)
        for k in range(1, 3):
            if relative[k, k] > relative[index, index]:
                index = k
        axis = wp.vec3(relative[0, index], relative[1, index], relative[2, index])
        axis[index] += 1.0
        return axis * (angle / wp.max(wp.length(axis), 1.0e-8))
    return vector * (angle / wp.max(sine, 1.0e-8))


@wp.kernel(enable_backward=False)
def _arm_ik_kernel(targets: wp.array(dtype=wp.vec3), orientations: wp.array(dtype=wp.mat33),
                   seeds: wp.array(dtype=_ArmVector), before: wp.array(dtype=wp.mat44),
                   after: wp.array(dtype=wp.mat44), axes: wp.array(dtype=wp.vec3),
                   skew: wp.array(dtype=wp.mat33), skew2: wp.array(dtype=wp.mat33),
                   limits: wp.array(dtype=wp.vec2), root: wp.mat44, tcp_offset: wp.mat44,
                   tcp_owner: int, iterations: int, active: wp.array(dtype=wp.bool),
                   result: wp.array(dtype=_ArmVector)):
    row = wp.tid()
    arm = seeds[row]
    for j in range(6):
        arm[j] = wp.clamp(arm[j], limits[j][0] + 1.0e-6, limits[j][1] - 1.0e-6)
    if not active[row]:
        result[row] = arm
        return
    for iteration in range(iterations):
        frame = root
        tcp = root * tcp_offset
        origins, directions = _ArmPoints(), _ArmPoints()
        for j in range(6):
            joint = frame * before[j]
            axis = wp.transform_vector(joint, axes[j])
            for k in range(3):
                origins[j, k] = joint[k, 3]
                directions[j, k] = axis[k]
            rotation = wp.identity(n=3, dtype=float) + wp.sin(arm[j]) * skew[j] + (1.0 - wp.cos(arm[j])) * skew2[j]
            motion = wp.mat44(rotation[0, 0], rotation[0, 1], rotation[0, 2], 0.0,
                              rotation[1, 0], rotation[1, 1], rotation[1, 2], 0.0,
                              rotation[2, 0], rotation[2, 1], rotation[2, 2], 0.0,
                              0.0, 0.0, 0.0, 1.0)
            frame = joint * motion * after[j]
            if j + 1 == tcp_owner:
                tcp = frame * tcp_offset
        position = wp.vec3(tcp[0, 3], tcp[1, 3], tcp[2, 3])
        rotation = wp.mat33(tcp[0, 0], tcp[0, 1], tcp[0, 2], tcp[1, 0], tcp[1, 1], tcp[1, 2],
                            tcp[2, 0], tcp[2, 1], tcp[2, 2])
        translation = targets[row] - position
        angular = _ik_orientation_error(orientations[row], rotation)
        # Only stop well inside the original acceptance tolerances.
        if wp.length_sq(translation) <= 1.0e-12 and wp.length_sq(angular) <= 1.0e-10:
            break
        error = _ArmVector(translation[0], translation[1], translation[2],
                           0.25 * angular[0], 0.25 * angular[1], 0.25 * angular[2])
        jacobian = _ArmMatrix()
        for j in range(6):
            axis = wp.vec3(directions[j, 0], directions[j, 1], directions[j, 2])
            origin = wp.vec3(origins[j, 0], origins[j, 1], origins[j, 2])
            linear = wp.cross(axis, position - origin)
            for k in range(3):
                jacobian[k, j] = linear[k]
                jacobian[k + 3, j] = 0.25 * axis[k]
        system = jacobian * wp.transpose(jacobian)
        for j in range(6):
            system[j, j] += 0.0001
        # The damped system is positive definite. Cholesky needs no pivot or
        # host/library dispatch; all six-joint work stays in this CUDA thread.
        lower = _ArmMatrix()
        for i in range(6):
            for j in range(i + 1):
                value = system[i, j]
                for k in range(j):
                    value -= lower[i, k] * lower[j, k]
                if i == j:
                    lower[i, j] = wp.sqrt(wp.max(value, 1.0e-12))
                else:
                    lower[i, j] = value / lower[j, j]
        forward, correction = _ArmVector(), _ArmVector()
        for i in range(6):
            value = error[i]
            for j in range(i):
                value -= lower[i, j] * forward[j]
            forward[i] = value / lower[i, i]
        for backward in range(6):
            i = 5 - backward
            value = forward[i]
            for j in range(i + 1, 6):
                value -= lower[j, i] * correction[j]
            correction[i] = value / lower[i, i]
        delta = wp.transpose(jacobian) * correction
        for j in range(6):
            arm[j] = wp.clamp(arm[j] + wp.clamp(delta[j], -0.35, 0.35),
                              limits[j][0] + 1.0e-6, limits[j][1] - 1.0e-6)
    result[row] = arm


class RobotPoseModel:
    """Exact USD joint-frame FK and bounded IK for the mounted open hand.

    Body transforms and collider bounds use meters and column-vector matrices.
    Environment origins are applied by the reset caller. This model is built
    once from env_0; checking candidates never changes the simulation.
    """

    @classmethod
    def from_env(cls, env) -> "RobotPoseModel":
        model = cls()
        robot = env.scene["robot"]
        stage = env.scene.stage
        origin_path = env.scene.env_prim_paths[0]
        robot_path = robot.cfg.prim_path.format(ENV_REGEX_NS=origin_path).replace(
            env.scene.env_regex_ns, origin_path, 1
        )
        root = stage.GetPrimAtPath(robot_path)
        if not root.IsValid():
            raise RuntimeError(f"Robot USD prim is missing: {robot_path}")
        model.joint_names = tuple(robot.joint_names)
        model.body_names = tuple(robot.body_names)
        model.arm_ids = np.asarray([model.joint_names.index(name) for name in ARM_JOINT_NAMES])
        model.default_joint_positions = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)
        model.joint_limits = robot.data.joint_pos_limits[0].detach().cpu().numpy().astype(np.float64)
        model.arm_limits = model.joint_limits[model.arm_ids]
        model.tcp_body_name = TCP_BODY_NAME
        # PhysX appends an index to duplicated link names (both assets have a
        # base_link). Use full paths to retain the correct hand/base association.
        link_paths = robot.root_physx_view.link_paths[0]
        path_to_name = dict(zip(map(str, link_paths), model.body_names))
        xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide],
            useExtentsHint=False,
            ignoreVisibility=True,
        )
        rigid_authored, stretches = {}, {}
        for path, name in path_to_name.items():
            prim = stage.GetPrimAtPath(path)
            world = _gf_column(xforms.GetLocalToWorldTransform(prim))
            rigid = np.eye(4)
            rigid[:3, :3], rigid[:3, 3] = _rotation_part(world), world[:3, 3]
            rigid_authored[name] = rigid
            stretches[name] = rigid[:3, :3].T @ world[:3, :3]
        raw_joints = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not body0 or not body1:
                continue  # The fixed world/base joint is represented by root_pose.
            name0, name1 = path_to_name.get(str(body0[0])), path_to_name.get(str(body1[0]))
            if name0 is None or name1 is None:
                raise RuntimeError(f"Joint {prim.GetPath()} references a body outside the articulation.")
            def frame(position, quaternion, stretch):
                q = quaternion.Get()
                return _joint_frame(
                    position.Get(), [q.GetReal(), *q.GetImaginary()], stretch
                )
            frame0 = frame(joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr(), stretches[name0])
            frame1 = frame(joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr(), stretches[name1])
            axis = np.zeros(3)
            joint_id = None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                token = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "X"
                axis["XYZ".index(token)] = 1.0
                if prim.GetName() not in model.joint_names:
                    raise RuntimeError(f"Revolute joint {prim.GetName()} is missing from PhysX DOFs.")
                joint_id = model.joint_names.index(prim.GetName())
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise RuntimeError(f"Unsupported robot joint type: {prim.GetTypeName()} {prim.GetPath()}")
            raw_joints.append((prim.GetName(), name0, name1, frame0, frame1, axis, joint_id))
        model.root_body_name = model.body_names[0]
        root_pose = robot.data.body_link_state_w[0, 0, :7].detach().cpu().numpy()
        model.root_transform = _rigid_matrix(root_pose[:3], root_pose[3:])
        ordered, reached = [], {model.root_body_name}
        pending = list(raw_joints)
        while pending:
            for row in pending:
                name, body0, body1, frame0, frame1, axis, index = row
                if (body0 in reached) == (body1 in reached):
                    continue
                reverse = body1 in reached
                parent, child = (body1, body0) if reverse else (body0, body1)
                ordered.append(_RobotJoint(name, parent, child, frame0, np.linalg.inv(frame1), axis, index, reverse))
                reached.add(child)
                pending.remove(row)
                break
            else:
                raise RuntimeError("Robot joints must form a connected tree without closed loops.")
        if reached != set(model.body_names):
            raise RuntimeError(f"Robot bodies are not connected to the root: {set(model.body_names) - reached}")
        model.joints = tuple(ordered)
        model.body_parents = {joint.child: joint.parent for joint in ordered}
        model._joint_by_name = {joint.name: joint for joint in ordered}
        model._joint_for_child = {joint.child: joint for joint in ordered}
        chain, body = [], model.tcp_body_name
        while body != model.root_body_name:
            joint = model._joint_for_child[body]
            chain.append(joint)
            body = joint.parent
        model._tcp_chain = tuple(reversed(chain))
        model.colliders = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            body = prim
            while body.IsValid() and str(body.GetPath()) not in path_to_name:
                body = body.GetParent()
            if not body.IsValid():
                raise RuntimeError(f"Robot collider has no physical link: {prim.GetPath()}")
            body_name = path_to_name[str(body.GetPath())]
            relative = np.linalg.inv(rigid_authored[body_name]) @ _gf_column(xforms.GetLocalToWorldTransform(prim))
            center, rotation, half_extents = _box_from_usd(prim, relative, cache)
            model.colliders.append(RobotColliderBox(body_name, center, rotation, half_extents, str(prim.GetPath())))
        shelf_path = env.scene.cfg.shelf.prim_path.format(ENV_REGEX_NS=origin_path).replace(
            env.scene.env_regex_ns, origin_path, 1
        )
        model.shelf_boxes = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(shelf_path), Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            center, rotation, half_extents = _box_from_usd(
                prim, _gf_column(xforms.GetLocalToWorldTransform(prim)), cache
            )
            model.shelf_boxes.append(ShelfColliderBox(center, rotation, half_extents, str(prim.GetPath())))
        current_q = robot.data.joint_pos[0].detach().cpu().numpy()
        predicted = model.forward(current_q)
        physical_positions = robot.data.body_link_pos_w[0].detach().cpu().numpy()
        physical_quaternions = robot.data.body_link_quat_w[0].detach().cpu().numpy()
        position_error, rotation_error = 0.0, 0.0
        for index, name in enumerate(model.body_names):
            physical_rotation = _rotation_wxyz(physical_quaternions[index])
            position_error = max(position_error, float(np.linalg.norm(predicted[name][:3, 3] - physical_positions[index])))
            rotation_error = max(rotation_error, float(np.linalg.norm(predicted[name][:3, :3] - physical_rotation)))
        model.fk_position_error, model.fk_rotation_error = position_error, rotation_error
        if position_error > 1.0e-5 or rotation_error > 1.0e-5:
            raise RuntimeError(
                f"USD robot FK differs from PhysX links: position={position_error:.6g}m rotation={rotation_error:.6g}."
            )
        return model

    @staticmethod
    def _motion(joint: _RobotJoint, joint_positions: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        if joint.index is not None:
            angle = float(joint_positions[joint.index])
            x, y, z = joint.axis
            cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            transform[:3, :3] = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
        transform = joint.frame0 @ transform @ joint.frame1_inverse
        return np.linalg.inv(transform) if joint.reverse else transform

    def forward(self, joint_positions: np.ndarray) -> dict[str, np.ndarray]:
        positions = np.asarray(joint_positions, dtype=np.float64)
        if positions.shape != self.default_joint_positions.shape:
            raise ValueError(f"Expected {len(self.joint_names)} joint positions, got {positions.shape}.")
        transforms = {self.root_body_name: self.root_transform.copy()}
        for joint in self.joints:
            transforms[joint.child] = transforms[joint.parent] @ self._motion(joint, positions)
        return transforms

    def _tcp_transform(self, joint_positions: np.ndarray) -> np.ndarray:
        matrix = self.root_transform.copy()
        for joint in self._tcp_chain:
            matrix = matrix @ self._motion(joint, joint_positions)
        return matrix

    def arm_landmarks(self, transforms):
        """Shoulder, elbow and wrist joint centers in canonical world coordinates."""
        centers = []
        for name in ("shoulder_lift_joint", "elbow_joint", "wrist_1_joint"):
            joint = self._joint_by_name[name]
            frame = np.linalg.inv(joint.frame1_inverse) if joint.reverse else joint.frame0
            centers.append((transforms[joint.parent] @ frame)[:3, 3])
        return np.stack(centers)

    def elbow_height(self, joint_positions) -> float:
        """Elbow height above the shoulder-to-wrist chord, in world meters."""
        transforms = joint_positions if isinstance(joint_positions, dict) else self.forward(joint_positions)
        from ...mdp.events import elbow_above_chord
        return elbow_above_chord(*self.arm_landmarks(transforms))

    def self_collision_pairs(self):
        """Reset collision pairs, shared by scalar validation and GPU acceptance.

        Check every hand collider against the arm, except wrist_3 and the fixed
        sensor mount: their relative poses cannot change in open-hand reset IK.
        Hand internal joint housings are likewise fixed during this search.
        Only arm-arm pairs retain the two-joint mechanical-housing exclusion.
        """
        if hasattr(self, '_self_pair_cache'):
            return self._self_pair_cache
        aliases = {FORCE_TORQUE_PROXIMAL_BODY_NAME: "wrist_3_link", FORCE_TORQUE_BODY_NAME: "wrist_3_link"}
        names = [aliases.get(box.body_name, box.body_name) for box in self.colliders]
        arm_names = {"base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                     "wrist_1_link", "wrist_2_link", "wrist_3_link"}
        pairs = []
        for first, name in enumerate(names):
            for second in range(first + 1, len(names)):
                other = names[second]
                first_arm, second_arm = name in arm_names, other in arm_names
                if not first_arm and not second_arm or name == other:
                    continue
                if first_arm and second_arm:
                    parent, other_parent = self.body_parents.get(name), self.body_parents.get(other)
                    if other in (parent, self.body_parents.get(parent)) or name in (
                            other_parent, self.body_parents.get(other_parent)):
                        continue
                elif (name if first_arm else other) == "wrist_3_link":
                    continue
                pairs.append((first, second))
        self._self_pair_cache = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        return self._self_pair_cache

    def _tensor_constants(self, device, dtype):
        """Collapse fixed/default hand joints once into the six moving arm frames."""
        key = (str(device), dtype)
        cache = getattr(self, "_tensor_cache", {})
        if key in cache:
            return cache[key]
        arm_ids = list(map(int, self.arm_ids))
        owners, offsets = {self.root_body_name: 0}, {self.root_body_name: np.eye(4)}
        before, after, axes = [None] * 6, [None] * 6, [None] * 6
        for joint in self.joints:
            if joint.index in arm_ids:
                index = arm_ids.index(joint.index)
                if owners[joint.parent] != index:
                    raise RuntimeError("Tensor reset IK requires the configured serial six-joint arm.")
                before[index] = offsets[joint.parent] @ (
                    np.linalg.inv(joint.frame1_inverse) if joint.reverse else joint.frame0)
                after[index] = np.linalg.inv(joint.frame0) if joint.reverse else joint.frame1_inverse
                axes[index] = joint.axis * (-1.0 if joint.reverse else 1.0)
                owners[joint.child], offsets[joint.child] = index + 1, np.eye(4)
            else:
                owners[joint.child] = owners[joint.parent]
                offsets[joint.child] = offsets[joint.parent] @ self._motion(joint, self.default_joint_positions)
        def tensor(value):
            return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)
        axes = np.asarray(axes)
        skew = np.zeros((6, 3, 3))
        skew[:, 0, 1], skew[:, 0, 2] = -axes[:, 2], axes[:, 1]
        skew[:, 1, 0], skew[:, 1, 2] = axes[:, 2], -axes[:, 0]
        skew[:, 2, 0], skew[:, 2, 1] = -axes[:, 1], axes[:, 0]
        collider_offsets = []
        for box in self.colliders:
            local = np.eye(4)
            local[:3, :3], local[:3, 3] = box.rotation_local, box.center_local
            collider_offsets.append(offsets[box.body_name] @ local)
        constants = {
            "before": tensor(before), "after": tensor(after), "axes": tensor(axes),
            "skew": tensor(skew), "skew2": tensor(skew @ skew), "root": tensor(self.root_transform),
            "arm_ids": torch.tensor(arm_ids, device=device),
            "default": tensor(self.default_joint_positions),
            "limits": tensor(np.stack((np.maximum(self.arm_limits[:, 0], -2 * math.pi),
                                        np.minimum(self.arm_limits[:, 1], 2 * math.pi)), -1)),
            "body_owners": torch.tensor([owners[name] for name in self.body_names], device=device),
            "body_offsets": tensor([offsets[name] for name in self.body_names]),
            "tcp_owner": owners[self.tcp_body_name], "tcp_offset": tensor(offsets[self.tcp_body_name]),
            "collider_owners": torch.tensor([owners[box.body_name] for box in self.colliders], device=device),
            "collider_offsets": tensor(collider_offsets),
            "collider_half": tensor([box.half_extents for box in self.colliders]),
            "self_pairs": torch.as_tensor(self.self_collision_pairs(), device=device, dtype=torch.long),
        }
        cache[key] = constants
        self._tensor_cache = cache
        return constants

    def arm_forward_batch(self, arm_positions, *, jacobian=False):
        """Exact USD FK for all rows; only the fixed six-joint chain uses a loop."""
        constants = self._tensor_constants(arm_positions.device, arm_positions.dtype)
        batch = arm_positions.shape[0]
        rotation = torch.eye(3, device=arm_positions.device, dtype=arm_positions.dtype)
        angles = arm_positions[..., None, None]
        rotations = rotation + angles.sin() * constants["skew"] + (1 - angles.cos()) * constants["skew2"]
        motions = torch.zeros((batch, 6, 4, 4), device=arm_positions.device, dtype=arm_positions.dtype)
        motions[..., :3, :3], motions[..., 3, 3] = rotations, 1.0
        frames = [constants["root"].expand(batch, -1, -1)]
        joint_frames, world_axes = [], []
        for index in range(6):
            joint = frames[-1] @ constants["before"][index]
            joint_frames.append(joint)
            if jacobian:
                world_axes.append(joint[:, :3, :3] @ constants["axes"][index])
            frames.append(joint @ motions[:, index] @ constants["after"][index])
        frames = torch.stack(frames, 1)
        joint_frames = torch.stack(joint_frames, 1)
        tcp = frames[:, constants["tcp_owner"]] @ constants["tcp_offset"]
        if not jacobian:
            return frames, joint_frames, tcp
        axes = torch.stack(world_axes, 1)
        linear = torch.linalg.cross(axes, tcp[:, None, :3, 3] - joint_frames[:, :, :3, 3])
        return frames, joint_frames, tcp, torch.cat((linear, axes), -1).transpose(1, 2)

    def forward_batch(self, joint_positions):
        """Physical-link transforms in ``body_names`` order, for open-hand reset poses."""
        constants = self._tensor_constants(joint_positions.device, joint_positions.dtype)
        frames, _, _ = self.arm_forward_batch(joint_positions[:, constants["arm_ids"]])
        return frames[:, constants["body_owners"]] @ constants["body_offsets"]

    @staticmethod
    def _rotation_error_batch(target, current):
        relative = target @ current.transpose(-1, -2)
        vector = .5 * torch.stack((relative[:, 2, 1] - relative[:, 1, 2],
                                  relative[:, 0, 2] - relative[:, 2, 0],
                                  relative[:, 1, 0] - relative[:, 0, 1]), -1)
        sine = vector.norm(dim=-1)
        cosine = (.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)).clamp(-1, 1)
        angle = torch.atan2(sine, cosine)
        # Near pi the skew part vanishes; recover the axis from R+I using its
        # largest diagonal. This also handles exactly opposite initial poses.
        diagonal = ((relative.diagonal(dim1=-2, dim2=-1) + 1) * .5).clamp_min(0).sqrt()
        index = diagonal.argmax(-1)
        column = (relative + torch.eye(3, device=current.device, dtype=current.dtype)).gather(
            -1, index[:, None, None].expand(-1, 3, 1)).squeeze(-1)
        axis = column / column.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
        ordinary = vector * (angle / sine.clamp_min(1.e-8))[:, None]
        return torch.where(((cosine < -.9999) & (sine < 1.e-4))[:, None], axis * angle[:, None], ordinary)

    @torch.no_grad()
    def solve_batch(self, target_positions, target_rotations, arm_seeds, *, iterations=48, active_mask=None):
        """Fuse FP32 CUDA IK per candidate; retain tensor solvers as the oracle.

        The kernel uses the same damped Jacobian solve and joint limits, with
        early exit strictly inside the pose tolerances. Other CUDA dtypes keep
        the fixed-iteration graph path. CPU follows the tensor equations.
        """
        if target_positions.is_cuda and target_positions.dtype == torch.float32:
            return self._solve_batch_fused(
                target_positions, target_rotations, arm_seeds, iterations=iterations, active_mask=active_mask)
        if target_positions.device.type != "cuda":
            output = self._solve_batch_eager(target_positions, target_rotations, arm_seeds, iterations=iterations)
            if active_mask is not None:
                output[1].logical_and_(active_mask)
            return output
        count = target_positions.shape[0]
        if count == 0:
            raise ValueError("An IK batch must contain at least one target.")
        bucket = 1 << (count - 1).bit_length()
        key = (str(target_positions.device), target_positions.dtype, bucket, iterations)
        cache = getattr(self, "_ik_graph_cache", {})
        if key not in cache:
            positions = target_positions[:1].expand(bucket, -1).clone()
            rotations = target_rotations[:1].expand(bucket, -1, -1).clone()
            seeds = arm_seeds[:1].expand(bucket, -1).clone()
            # Allocator/library initialization happens outside capture. A
            # dedicated warmup stream follows PyTorch's graph capture contract.
            with torch.cuda.device(target_positions.device):
                stream = torch.cuda.Stream(device=target_positions.device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._solve_batch_eager(positions, rotations, seeds, iterations=iterations)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = self._solve_batch_eager(positions, rotations, seeds, iterations=iterations)
            cache[key] = (positions, rotations, seeds, graph, outputs)
            self._ik_graph_cache = cache
        positions, rotations, seeds, graph, outputs = cache[key]
        positions[:count].copy_(target_positions)
        rotations[:count].copy_(target_rotations)
        seeds[:count].copy_(arm_seeds)
        graph.replay()
        selected = tuple(output[:count] for output in outputs)
        if active_mask is not None:
            selected[1].logical_and_(active_mask)
        return selected

    def _solve_batch_fused(self, target_positions, target_rotations, arm_seeds, *, iterations, active_mask=None):
        wp.init()
        constants = self._tensor_constants(target_positions.device, target_positions.dtype)
        cache = constants.get("warp_ik")
        if cache is None:
            # Small static transforms cross to host only once, during setup.
            cache = [wp.from_torch(constants[name].contiguous(), dtype=dtype) for name, dtype in (
                ("before", wp.mat44), ("after", wp.mat44), ("axes", wp.vec3),
                ("skew", wp.mat33), ("skew2", wp.mat33), ("limits", wp.vec2))]
            cache += [wp.mat44(*constants[name].cpu().flatten().tolist()) for name in ("root", "tcp_offset")]
            constants["warp_ik"] = cache
        arm = torch.empty_like(arm_seeds, memory_format=torch.contiguous_format)
        if active_mask is None:
            active_mask = torch.ones(len(arm), device=arm.device, dtype=torch.bool)
        wp.launch(_arm_ik_kernel, dim=len(arm), inputs=[
            wp.from_torch(target_positions.contiguous(), dtype=wp.vec3),
            wp.from_torch(target_rotations.contiguous(), dtype=wp.mat33),
            wp.from_torch(arm_seeds.contiguous(), dtype=_ArmVector), *cache,
            constants["tcp_owner"], iterations, wp.from_torch(active_mask.contiguous()),
            wp.from_torch(arm, dtype=_ArmVector)],
            stream=wp.stream_from_torch(torch.cuda.current_stream(arm.device)))
        frames, joints, tcp = self.arm_forward_batch(arm)
        valid = ((target_positions - tcp[:, :3, 3]).norm(dim=-1) <= 5.e-4)
        valid &= self._rotation_error_batch(target_rotations, tcp[:, :3, :3]).norm(dim=-1) <= .01
        valid &= torch.isfinite(arm).all(-1)
        valid &= active_mask
        positions = constants["default"].expand(len(arm), -1).clone()
        positions[:, constants["arm_ids"]] = arm
        return positions, valid, frames, joints, tcp

    @torch.no_grad()
    def _solve_batch_eager(self, target_positions, target_rotations, arm_seeds, *, iterations=48):
        """Fixed-iteration damped IK, without per-candidate SciPy/host synchronization.

        Every row is a distinct continuous target/branch. Caller batches branch
        proposals and applies geometric elbow/collision acceptance afterwards.
        """
        constants = self._tensor_constants(target_positions.device, target_positions.dtype)
        lower, upper = constants["limits"].unbind(-1)
        arm = arm_seeds.clamp(lower + 1.e-6, upper - 1.e-6)
        eye = torch.eye(6, device=arm.device, dtype=arm.dtype)
        for _ in range(iterations):
            _, _, tcp, jacobian = self.arm_forward_batch(arm, jacobian=True)
            error = torch.cat((target_positions - tcp[:, :3, 3],
                               .25 * self._rotation_error_batch(target_rotations, tcp[:, :3, :3])), -1)
            jacobian[:, 3:] *= .25
            system = jacobian @ jacobian.transpose(-1, -2) + .0001 * eye
            correction, info = torch.linalg.solve_ex(system, error[..., None], check_errors=False)
            delta = (jacobian.transpose(-1, -2) @ correction).squeeze(-1)
            delta = torch.where((info == 0)[:, None], delta, torch.zeros_like(delta))
            arm = (arm + delta.clamp(-.35, .35)).clamp(lower + 1.e-6, upper - 1.e-6)
        frames, joints, tcp = self.arm_forward_batch(arm)
        error_position = (target_positions - tcp[:, :3, 3]).norm(dim=-1)
        error_rotation = self._rotation_error_batch(target_rotations, tcp[:, :3, :3]).norm(dim=-1)
        valid = (error_position <= 5.e-4) & (error_rotation <= .01) & torch.isfinite(arm).all(-1)
        positions = constants["default"].expand(arm.shape[0], -1).clone()
        positions[:, constants["arm_ids"]] = arm
        return positions, valid, frames, joints, tcp

    def solve(self, tcp_position, tcp_rotation, rng, initial_seed=None, *, min_elbow_height=0.05,
              accept_candidate: Callable[[dict[str, np.ndarray]], bool] | None = None) -> np.ndarray | None:
        """Find a bounded elbow-up solution, trying other IK branches after rejection."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        target_position = np.asarray(tcp_position, dtype=np.float64)
        target_rotation = np.asarray(tcp_rotation, dtype=np.float64)
        if target_position.shape != (3,) or target_rotation.shape != (3, 3):
            raise ValueError("TCP target must be a 3-vector position and a 3x3 rotation.")
        if not np.isfinite(target_position).all() or not np.isfinite(target_rotation).all():
            raise ValueError("TCP target must be finite.")
        full = self.default_joint_positions.copy()
        lower, upper = self.arm_limits[:, 0], self.arm_limits[:, 1]
        # Infinite continuous limits still need a finite optimization interval;
        # +/- one complete turn includes all UR5e IK branches.
        lower, upper = np.maximum(lower, -2.0 * math.pi), np.minimum(upper, 2.0 * math.pi)
        if np.any(lower >= upper):
            raise RuntimeError("Robot arm joint limits are empty.")
        def residual(arm):
            full[self.arm_ids] = arm
            current = self._tcp_transform(full)
            orientation = Rotation.from_matrix(target_rotation @ current[:3, :3].T).as_rotvec()
            return np.concatenate((current[:3, 3] - target_position, 0.25 * orientation))
        seeds = []
        if initial_seed is not None:
            initial = np.asarray(initial_seed, dtype=np.float64)
            seeds.append(initial[self.arm_ids] if initial.shape == full.shape else initial)
        default = self.default_joint_positions[self.arm_ids]
        # Try the measured upper branch first. Acceptance still uses geometric
        # elbow height rather than assuming that a joint-angle sign is enough.
        for lift, elbow in ((-2.0, 1.8), (-1.6, 1.2), (-0.6, -1.8), (-1.0, -1.2)):
            seed = default.copy()
            seed[1:3] = lift, elbow
            seed[3] = -lift - elbow
            seeds.append(seed)
        seeds.append(default.copy())
        for _ in range(2):
            seeds.append(np.asarray([rng.uniform(lo, hi) for lo, hi in zip(lower, upper)]))
        for seed in seeds:
            solution = least_squares(
                residual, np.clip(seed, lower + 1.0e-8, upper - 1.0e-8),
                bounds=(lower, upper), max_nfev=90, ftol=1.0e-7, xtol=1.0e-7, gtol=1.0e-7,
            )
            error = residual(solution.x)
            if np.linalg.norm(error[:3]) > 5.0e-4 or np.linalg.norm(error[3:]) > 2.5e-3:
                continue
            result = self.default_joint_positions.copy()
            result[self.arm_ids] = solution.x
            transforms = self.forward(result)
            if self.elbow_height(transforms) >= min_elbow_height and (
                accept_candidate is None or accept_candidate(transforms)
            ):
                return result
        return None


# Shelf object pool.

YCB_BOX_ASSETS = frozenset({"ycb_003_cracker_box", "ycb_004_sugar_box"})


@clone
def spawn_flat_box_from_usd(
    prim_path: str,
    cfg: "FlatBoxUsdCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Replace the scanned mesh collider with a body-axis-aligned box at spawn.

    The outer clone decorator copies the completed collision override to every
    environment before PhysX builds its views. No source USD file is modified.
    """
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise RuntimeError(f"Flat-box asset {prim_path} must contain exactly one rigid body.")
    body = bodies[0]
    colliders = [
        prim for prim in Usd.PrimRange(body)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    ]
    if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Flat-box asset {prim_path} must contain one enabled scanned mesh collider.")
    original = colliders[0]
    # Fit in the unscaled body frame. The proxy inherits body/ancestor scale;
    # geometry inspection applies that scale and the upright rotation once.
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False, ignoreVisibility=True,
    )
    bounds = cache.ComputeUntransformedBound(body).ComputeAlignedRange()
    lower, upper = bounds.GetMin(), bounds.GetMax()
    if bounds.IsEmpty() or any(
        not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo
        for lo, hi in zip(lower, upper)
    ):
        raise RuntimeError(f"Flat-box asset {prim_path} has invalid local bounds.")
    proxy = UsdGeom.Cube.Define(root.GetStage(), body.GetPath().AppendChild("FlatBoxCollision"))
    proxy.CreateSizeAttr(1.0)
    proxy.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    proxy.AddTranslateOp().Set((lower + upper) * 0.5)
    proxy.AddScaleOp().Set(Gf.Vec3f(*(upper - lower)))
    proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    proxy.MakeInvisible()
    proxy_prim = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(proxy_prim).CreateCollisionEnabledAttr(True)
    PhysxSchema.PhysxCollisionAPI.Apply(proxy_prim)
    for attribute in original.GetAttributes():
        if attribute.GetName().startswith("physxCollision:") and attribute.HasAuthoredValueOpinion():
            proxy_prim.CreateAttribute(attribute.GetName(), attribute.GetTypeName()).Set(attribute.Get())
    material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(materialPurpose="physics")
    if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        UsdShade.MaterialBindingAPI.Apply(proxy_prim).Bind(material, materialPurpose="physics")
    UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
    return root


@configclass
class FlatBoxUsdCfg(sim_utils.UsdFileCfg):
    """Keep authored rigid-body properties and visuals, replacing only collision geometry."""

    func: Callable = spawn_flat_box_from_usd


class TaskObjectPool(RigidObjectCollection):
    """Native collection with persistent countergravity for isolated parked objects.

    The permanent wrench composer is reserved for parking. Transient task forces
    can use the native instantaneous composer and are added by the parent class.
    Active objects retain gravity and receive zero parking force.
    """

    def __init__(self, cfg):
        self.parked_mask: torch.Tensor | None = None
        self.gravity: torch.Tensor | None = None
        self._countergravity: torch.Tensor | None = None
        super().__init__(cfg)

    def write_object_link_pose_to_sim(self, object_pose, env_ids=None, object_ids=None) -> None:
        super().write_object_link_pose_to_sim(object_pose, env_ids=env_ids, object_ids=object_ids)
        # Installed SDK does not invalidate the separate COM-pose cache on writes.
        self._data._object_com_pose_w.timestamp = -1.0

    def refresh_parking(self):
        if self.parked_mask is None:
            return
        composer = self.permanent_wrench_composer
        # A zero wrench still activates native per-step force submission. Check
        # only at reset, and leave the native composer inactive when all objects
        # are on the shelf.
        composer.reset()
        if not self.parked_mask.any():
            return
        if self._countergravity is None:
            self._countergravity = -self.data.default_mass.to(self.device) * self.gravity
        # Refresh conversion frames after repeated pose writes at one timestamp.
        composer.set_forces_and_torques(
            forces=self._countergravity * self.parked_mask[..., None], is_global=True,
        )
        # No position offset: native collection applies the force at the COM.
        # No custom write_data_to_sim or direct PhysX force submission is needed.


def make_object_collection_cfg() -> RigidObjectCollectionCfg:
    """Load the catalog once; reset activates a sampled subset on the working shelf."""
    return RigidObjectCollectionCfg(
        class_type=TaskObjectPool,
        rigid_objects={
            spec.name: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                spawn=(FlatBoxUsdCfg if spec.asset_name in YCB_BOX_ASSETS else sim_utils.UsdFileCfg)(
                    usd_path=spec.usd_path,
                    collision_props=(
                        sim_utils.CollisionPropertiesCfg(contact_offset=0.005)
                        if spec.asset_name.startswith("ycb_") else None
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        kinematic_enabled=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        max_depenetration_velocity=1.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=parking_position(index),
                    rot=spec.upright_quat,
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )
            for index, spec in enumerate(OBJECT_SPECS)
        }
    )


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


def _environment_zero_path(env: object, path: str) -> str:
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


def inspect_geometry(env: object) -> tuple[tuple[ObjectGeometry, ...], SpawnRules]:
    """Inspect env_0 body bounds and shelf collisions after the scene has spawned.

    Object bounds include descendants and composed body scale, exclude its rigid
    world pose, and are rotated by its catalog upright quaternion. Shelf bounds
    are expressed in environment coordinates and retain the configured sampling
    constraints except for the measured support region and vertical clearance.
    """
    from ...mdp.events import ObjectGeometry

    stage = env.scene.stage
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=True,
    )
    objects = env.scene["object_collection"]
    # Copies have identical authored geometry. Inspect each asset type once.
    measured = {}
    for spec in OBJECT_SPECS:
        if spec.asset_name in measured:
            continue
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
        measured[spec.asset_name] = (lower, upper)
    geometries = tuple(ObjectGeometry(spec.name, *measured[spec.asset_name], asset_name=spec.asset_name)
                       for spec in OBJECT_SPECS)

    shelf_path = _environment_zero_path(env, env.scene.cfg.shelf.prim_path)
    shelf = stage.GetPrimAtPath(shelf_path)
    if not shelf.IsValid():
        raise RuntimeError(f"Spawned shelf prim is missing: {shelf_path}")
    origin = tuple(float(value) for value in env.scene.env_origins[0].tolist())
    rules = _shelf_rules(cache, shelf, origin, env.cfg.spawn_rules)
    return tuple(geometries), rules


def share_rigid_body_geometry(stage, template_path: str, output_path: str) -> int:
    """Bake completed body subtrees once, then share their geometry via USD instances.

    Body roots remain individually addressable and movable. Descendant bodies or
    joints are never instanced. Baking happens after collision/mount edits and
    before native cloning/PhysX initialization, so modified colliders survive.
    """
    # Stage.Flatten preserves instances as references to generated prototype
    # roots. Expand existing geometry instances before copying subtrees, or
    # those references would point outside the exported body library.
    _deinstance_subtree(stage, template_path)
    root = stage.GetPrimAtPath(template_path)
    bodies = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.IsA(UsdGeom.Mesh):
            continue
        descendants = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))[1:]
        if any(child.HasAPI(UsdPhysics.RigidBodyAPI) or child.IsA(UsdPhysics.Joint) for child in descendants):
            continue
        if any(child.IsA(UsdGeom.Mesh) for child in descendants):
            bodies.append(str(prim.GetPath()))
    # Flatten only once: resolved references, scale, collision overrides and
    # absolute texture asset paths are retained without rewriting source assets.
    flattened = stage.Flatten()
    layer = Sdf.Layer.CreateNew(output_path)
    library = Usd.Stage.Open(layer)
    for index, path in enumerate(bodies):
        expected_meshes = sum(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(stage.GetPrimAtPath(path)))
        destination = Sdf.Path(f'/Body_{index}')
        Sdf.CopySpec(flattened, Sdf.Path(path), layer, destination)
        prototype = library.GetPrimAtPath(destination)
        # References cannot target materials outside their referenced subtree.
        # Copy those materials into this prototype and retarget bindings.
        bindings = [(rel, rel.GetTargets()) for prim in Usd.PrimRange(prototype)
                    for rel in prim.GetRelationships() if rel.GetName().startswith('material:binding')]
        materials = {}
        for relationship, targets in bindings:
            remapped = []
            for target in targets:
                if target.HasPrefix(destination) or not flattened.GetPrimAtPath(target):
                    remapped.append(target)
                    continue
                if target not in materials:
                    material = destination.AppendPath(f'_SharedMaterials/Material_{len(materials)}')
                    Sdf.CreatePrimInLayer(layer, material.GetParentPath())
                    Sdf.CopySpec(flattened, target, layer, material)
                    materials[target] = material
                remapped.append(materials[target])
            relationship.SetTargets(remapped)
        # Root bindings/poses remain editable; only descendants form prototypes.
        live = stage.GetPrimAtPath(path)
        # The baked subtree contains the resolved asset and our collision edits.
        # An older, stronger reference can restore its original offsets or
        # colliders once local descendant edits become instance proxies.
        live.GetReferences().SetReferences([Sdf.Reference(output_path, destination)])
        live.SetInstanceable(True)
        if not live.IsInstance():
            raise RuntimeError(f'Failed to share geometry below {path}.')
        actual_meshes = sum(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(live, Usd.TraverseInstanceProxies()))
        if actual_meshes != expected_meshes:
            raise RuntimeError(f'Geometry sharing changed mesh count below {path}: {expected_meshes} -> {actual_meshes}.')
    layer.Save()
    return len(bodies)


def spawn_shared_geometry(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Finalize template geometry through a native last-in-scene asset spawner."""
    import atexit
    import shutil
    import tempfile
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    sim_utils.standardize_xform_ops(root, translation=translation, orientation=orientation)
    directory = tempfile.mkdtemp(prefix='contact_geometry_')
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    output = str(Path(directory) / 'geometry.usdc')
    count = share_rigid_body_geometry(stage, '/World/envs/env_0', output)
    print(f'[INFO] Shared geometry for {count} rigid bodies before environment cloning.', flush=True)
    return root


@configclass
class SharedGeometryCfg(sim_utils.SpawnerCfg):
    func: Callable = spawn_shared_geometry


