"""UR5e and ROAS-provided left force-sensor hand assembly."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
HAND_JOINT_NAMES = (
    "left_thumb_1_joint",
    "left_thumb_2_joint",
    "left_index_1_joint",
    "left_middle_1_joint",
    "left_ring_1_joint",
    "left_little_1_joint",
)
FINGERTIP_BODY_NAMES = (
    "thumb_force_sensor_4",
    "index_force_sensor_3",
    "middle_force_sensor_3",
    "ring_force_sensor_3",
    "little_force_sensor_3",
)
TCP_BODY_NAME = "palm_force_sensor"

DEFAULT_UR5E_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.0/"
    "Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HAND_URDF_PATH = (
    REPOSITORY_ROOT / "assets/robots/Roas_provided_urdf/urdf/urdf_left_with_force_sensor.urdf"
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
    for mesh in ElementTree.parse(source).findall(".//mesh"):
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


@clone
def spawn_ur5e_hand(
    prim_path: str,
    cfg: "Ur5eHandSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn UR5e, import the left-hand URDF, and join both into one articulation."""
    del kwargs
    validate_hand_urdf_assets(cfg.hand_urdf_path)

    stage = get_current_stage()
    ur_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.ur5e_usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=True,
    )
    ur_cfg.func(prim_path, ur_cfg, translation=translation, orientation=orientation)

    hand_path = f"{prim_path}/ROASLeftHand"
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
            "UR5e or hand did not compose into the stage. Check HAND_RL_UR5E_USD_PATH and "
            "HAND_RL_HAND_URDF_PATH."
        )

    _deinstance_subtree(stage, hand_path)
    tool_frame = _find_named_prim(stage, prim_path, cfg.tool_frame_candidates)
    tool_body = _nearest_rigid_body_ancestor(tool_frame)
    hand_base = _find_named_prim(stage, hand_path, cfg.hand_base_candidates, rigid_body_only=True)

    _deactivate_preexisting_mount_joints(stage, prim_path)
    hand_root_world = _world_transform(hand_root)
    hand_base_relative_to_root = _world_transform(hand_base) * hand_root_world.GetInverse()
    mount_world = _make_offset_matrix(cfg.mount_translation, cfg.mount_rotation_deg) * _world_transform(tool_frame)
    desired_hand_root_world = hand_base_relative_to_root.GetInverse() * mount_world
    _set_world_transform_on_reference_root(hand_root, desired_hand_root_world)

    _remove_nested_articulation_roots(stage, hand_path)
    _create_fixed_joint(stage, f"{prim_path}/UR5e_hand_mount_joint", tool_body, hand_base, mount_world)
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
    # The extra 180-degree pitch makes the fingers extend from the wrist toward
    # the shelf (+X in the environment) instead of back toward the robot.
    mount_rotation_deg: tuple[float, float, float] = (0.0, 180.0, -90.0)


def make_ur5e_hand_cfg() -> ArticulationCfg:
    """Create the assembled articulation with environment-overridable asset paths."""
    hand_urdf_path = os.environ.get(
        "HAND_RL_HAND_URDF_PATH",
        os.environ.get("HAND_RL_RH56_URDF_PATH", str(DEFAULT_HAND_URDF_PATH)),
    )
    spawn_cfg = Ur5eHandSpawnerCfg(
        ur5e_usd_path=os.environ.get("HAND_RL_UR5E_USD_PATH", DEFAULT_UR5E_USD_PATH),
        hand_urdf_path=hand_urdf_path,
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": -0.25,
                "shoulder_lift_joint": -1.80,
                "elbow_joint": 1.20,
                "wrist_1_joint": -0.97,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
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
