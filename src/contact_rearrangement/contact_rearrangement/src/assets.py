"""Robot assembly and shelf-object pool construction for the task scene."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass

from .catalog import NUCLEUS_ROOT, OBJECT_SPECS, parking_position
from .sampling import elbow_above_chord


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
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
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
        if mesh_path.read_bytes()[:80].startswith(b"version https://git-lfs.github.com/spec/v1"):
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


def _preserve_tactile_parent_slots(stage: Usd.Stage, hand_path: str) -> None:
    """Keep five parent CAD recesses from being filled by a single convex hull.

    Their original collision meshes contain the palm/pad-2 slots, but a single
    convex hull encloses those tactile bodies and intercepts object contact.
    Only these existing parent colliders are decomposed; all 17 tactile-link
    colliders, rigid bodies, joints, inertias, and visuals remain unchanged.
    """
    parent_names = ("right_plam_1", *(f"right_{finger}_2" for finger in ("index", "middle", "ring", "little")))
    bodies = [prim for prim in Usd.PrimRange(stage.GetPrimAtPath(hand_path)) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = []
    for name in parent_names:
        matching = [body for body in bodies if body.GetName() == name]
        if len(matching) != 1:
            raise RuntimeError(f"Tactile parent {name!r} must resolve exactly one rigid body, got {len(matching)}.")
        body_colliders = [
            prim for prim in Usd.PrimRange(matching[0])
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        ]
        if not body_colliders:
            raise RuntimeError(f"Tactile parent {name!r} has no enabled collision geometry.")
        for collider in body_colliders:
            meshes = [UsdGeom.Mesh(prim) for prim in Usd.PrimRange(collider) if prim.IsA(UsdGeom.Mesh)]
            if not any(len(mesh.GetPointsAttr().Get() or []) >= 4 and mesh.GetFaceVertexCountsAttr().Get() for mesh in meshes):
                raise RuntimeError(f"Tactile parent collider {collider.GetPath()} has no usable triangle mesh.")
            colliders.append(collider)
    # Tight surface projection and finer decomposition preserve millimetre
    # slots; the SDK defaults (10% error, 32 hulls) can bridge those recesses.
    cfg = sim_utils.ConvexDecompositionPropertiesCfg(
        max_convex_hulls=128, hull_vertex_limit=64, voxel_resolution=1_000_000,
        error_percentage=0.5, shrink_wrap=True, min_thickness=0.0001,
    )
    for collider in colliders:
        sim_utils.define_mesh_collision_properties(str(collider.GetPath()), cfg, stage=stage)


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
        activate_contact_sensors=True,
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
            "UR5e or hand did not compose into the stage. Check CONTACT_REARRANGEMENT_UR5E_USD_PATH and "
            "CONTACT_REARRANGEMENT_HAND_URDF_PATH."
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
        "CONTACT_REARRANGEMENT_HAND_URDF_PATH", str(DEFAULT_HAND_URDF_PATH)
    )
    spawn_cfg = Ur5eHandSpawnerCfg(
        ur5e_usd_path=os.environ.get("CONTACT_REARRANGEMENT_UR5E_USD_PATH", DEFAULT_UR5E_USD_PATH),
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
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(ARM_JOINT_NAMES),
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
                stiffness=400.0,
                damping=40.0,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[".*(thumb|index|middle|ring|little).*"],
                effort_limit_sim=10.0,
                velocity_limit_sim=1.0,
                stiffness=20.0,
                damping=1.0,
            ),
        },
    )


class ToolForceTorqueSensor:
    """Six-axis fixed-joint load cell at the spacer center, in SI units.

    ``[Fx, Fy, Fz, Tx, Ty, Tz]`` is the wrench transmitted from the arm-side
    parent to the hand-side child. Its origin is the spacer center and XYZ
    follows the physical wrist/tool frame (+Z along the cylinder toward the
    hand), before the hand's additional mount roll. Raw readings include hand
    gravity, inertia and contacts; they are not external-contact-only forces.

    The installed PhysX Tensor API reports the incoming wrench in the *child
    joint frame*. We author that frame as identity at the distal link origin,
    so the backend's vector already has the requested frame and moment point:
    https://nvidia-omniverse.github.io/PhysX/physx/5.5.1/docs/Articulations.html#link-incoming-joint-force

    Call ``update`` after a completed physics step. ``reset`` invalidates only
    selected environments without inventing a zero physical measurement. The
    zero cache after reset is accompanied by ``valid=False`` until updated.
    """

    def __init__(self, robot):
        self.robot = robot
        try:
            self.body_index = robot.body_names.index(FORCE_TORQUE_BODY_NAME)
        except ValueError as exc:
            raise RuntimeError("Articulation has no distal F/T sensor link.") from exc
        shape = (robot.num_instances, 6)
        self.data_raw_wrench = torch.zeros(shape, dtype=torch.float32, device=robot.device)
        self.bias = torch.zeros_like(self.data_raw_wrench)
        self.valid = torch.zeros(robot.num_instances, dtype=torch.bool, device=robot.device)
        self.metadata = {
            "body_name": FORCE_TORQUE_BODY_NAME,
            "joint_name": FORCE_TORQUE_JOINT_NAME,
            "frame": "sensor_center_physical_tool_xyz",
            "axis_z": "physical tool forward, from arm toward hand",
            "direction": "parent_to_child",
            "components": ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
            "units": ("N", "N", "N", "N*m", "N*m", "N*m"),
            "sample": "mean reaction over the last completed physics step",
            "includes": ("gravity", "inertia", "contact", "external_force"),
            "nominal_mount_mass_kg": robot.cfg.spawn.sensor_mount_mass,
        }

    def reset(self, env_ids=None) -> None:
        """Clear the cache/tare for selected episodes and mark readings invalid."""
        selected = slice(None) if env_ids is None else env_ids
        self.data_raw_wrench[selected] = 0.0
        self.bias[selected] = 0.0
        self.valid[selected] = False

    def update(self, dt: float | None = None) -> None:
        """Read the actual link reaction after simulation; never apply a force."""
        del dt
        # Read directly: the IsaacLab 2.3.2 wrapper has a contradictory frame
        # docstring and a time_stamp/timestamp cache typo. Backend source and
        # PhysX documentation both explicitly specify the child joint frame.
        wrench = self.robot.root_physx_view.get_link_incoming_joint_force()[:, self.body_index, :]
        if not torch.isfinite(wrench).all():
            raise RuntimeError("PhysX returned a nonfinite F/T sensor joint reaction.")
        self.data_raw_wrench.copy_(wrench)
        self.valid[:] = True

    def read(self, env_ids=None, *, tared: bool = False) -> torch.Tensor:
        """Return cached wrench; callers must also honor the ``valid`` mask."""
        values = self.data_raw_wrench - self.bias if tared else self.data_raw_wrench
        return values.clone() if env_ids is None else values[env_ids].clone()

    def tare(self, env_ids=None) -> None:
        """Record an explicit instantaneous bias, without gravity compensation."""
        selected = slice(None) if env_ids is None else env_ids
        if not self.valid[selected].all():
            raise RuntimeError("Cannot tare before a completed physics-step F/T sample.")
        self.bias[selected] = self.data_raw_wrench[selected]


# Robot reset kinematics. Candidate poses are calculated on the CPU without
# writing articulation state or advancing the simulation.


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

    def elbow_height(self, joint_positions) -> float:
        """Elbow height above the shoulder-to-wrist chord, in world meters."""
        transforms = joint_positions if isinstance(joint_positions, dict) else self.forward(joint_positions)
        centers = []
        for name in ("shoulder_lift_joint", "elbow_joint", "wrist_1_joint"):
            joint = self._joint_by_name[name]
            frame = np.linalg.inv(joint.frame1_inverse) if joint.reverse else joint.frame0
            centers.append((transforms[joint.parent] @ frame)[:3, 3])
        return elbow_above_chord(*centers)

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
    """Apply countergravity at parked bodies' COM in every physics substep.

    Task objects retain native gravity and get zero countergravity. A live GPU
    simulation/switch is avoided, and world COM application prevents torque on
    assets with off-center pivots or COMs. Other external wrenches stay separate.
    """

    def __init__(self, cfg):
        self.parked_mask: torch.Tensor | None = None
        self.gravity: torch.Tensor | None = None
        super().__init__(cfg)

    def write_object_link_pose_to_sim(self, object_pose, env_ids=None, object_ids=None) -> None:
        super().write_object_link_pose_to_sim(object_pose, env_ids=env_ids, object_ids=object_ids)
        # Isaac Lab 2.3 updates link poses and any combined COM-state cache,
        # but not the separate COM-pose cache. Repeated resets at one physics
        # timestamp would otherwise apply countergravity at a stale point and
        # spin parked bodies whose mass center differs from their link origin.
        self._data._object_com_pose_w.timestamp = -1.0

    def write_data_to_sim(self) -> None:
        super().write_data_to_sim()
        if self.parked_mask is None:
            return
        forces = -self.data.default_mass.to(self.device) * self.gravity * self.parked_mask[..., None]
        self.root_physx_view.apply_forces_and_torques_at_position(
            force_data=self.reshape_data_to_view(forces).contiguous(),
            torque_data=None,
            position_data=self.reshape_data_to_view(self.data.object_com_pos_w).contiguous(),
            indices=torch.arange(self.root_physx_view.count, device=self.device, dtype=torch.int32),
            is_global=True,
        )


def make_object_collection_cfg() -> RigidObjectCollectionCfg:
    """Load the catalog once; reset activates a sampled subset on the working shelf."""
    return RigidObjectCollectionCfg(
        class_type=TaskObjectPool,
        rigid_objects={
            spec.name: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                spawn=(FlatBoxUsdCfg if spec.asset_name in YCB_BOX_ASSETS else sim_utils.UsdFileCfg)(
                    usd_path=spec.usd_path,
                    semantic_tags=[("class", spec.asset_name)],
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
