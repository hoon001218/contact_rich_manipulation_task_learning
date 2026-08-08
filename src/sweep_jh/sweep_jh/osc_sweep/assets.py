"""UR5e, virtual F/T sensor, and Robotiq 2F-85 assembly for Sweep JH.

The native gripper collision geometry is used directly. No synthetic contact-pad
bodies, contact-pad meshes, or contact sensors are created by this asset.
"""

from __future__ import annotations

import os
from collections.abc import Callable

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

FT_SENSOR_BODY_NAME = "VirtualFTSensor"
EEF_CENTER_BODY_NAME = "SweepToolCenter"
GRIPPER_SIDE_AXIS_LOCAL = (1.0, 0.0, 0.0)

DEFAULT_UR5E_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.0/"
    "Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
)
DEFAULT_ROBOTIQ_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.1/"
    "Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
)

_UR_TOOL_FRAME_CANDIDATES = ("tool0", "tool_frame", "flange", "wrist_3_link")
_GRIPPER_BASE_BODY_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_2f_85_base_link",
    "robotiq_base_link",
    "base_link",
)
_PREEXISTING_MOUNT_JOINT_NAMES = ("robot_gripper_joint",)


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
            if prim.GetName() != candidate:
                continue
            if rigid_body_only and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            return prim

    rigid_names = [
        prim.GetName()
        for prim in prims
        if not rigid_body_only or prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    raise RuntimeError(
        f"Could not find any of {candidates} below {subtree_path}. "
        f"Available names: {rigid_names}"
    )


def _nearest_rigid_body_ancestor(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    raise RuntimeError(f"No rigid-body ancestor found for {prim.GetPath()}")


def _make_offset_matrix(
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
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
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _set_world_transform_on_reference_root(
    root_prim: Usd.Prim, world_matrix: Gf.Matrix4d
) -> None:
    parent = root_prim.GetParent()
    local_matrix = world_matrix * _world_transform(parent).GetInverse()
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
    parent_local = joint_frame_world * _world_transform(parent_body).GetInverse()
    child_local = joint_frame_world * _world_transform(child_body).GetInverse()
    parent_pos, parent_rot = _matrix_to_pose(parent_local)
    child_pos, child_rot = _matrix_to_pose(child_local)

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([parent_body.GetPath()])
    joint.CreateBody1Rel().SetTargets([child_body.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*parent_pos))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(parent_rot[0], Gf.Vec3f(*parent_rot[1:]))
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*child_pos))
    joint.CreateLocalRot1Attr().Set(
        Gf.Quatf(child_rot[0], Gf.Vec3f(*child_rot[1:]))
    )
    joint.CreateCollisionEnabledAttr().Set(False)


def _spawn_fixed_body(
    stage: Usd.Stage,
    robot_path: str,
    body_name: str,
    world_matrix: Gf.Matrix4d,
    *,
    size: tuple[float, float, float],
    mass: float,
    collision_enabled: bool,
    color: tuple[float, float, float],
) -> Usd.Prim:
    robot_prim = stage.GetPrimAtPath(robot_path)
    local_matrix = world_matrix * _world_transform(robot_prim).GetInverse()
    translation, orientation = _matrix_to_pose(local_matrix)
    body_path = f"{robot_path}/{body_name}"

    body_cfg = sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=collision_enabled,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    body_cfg.func(
        body_path,
        body_cfg,
        translation=translation,
        orientation=orientation,
    )
    body = stage.GetPrimAtPath(body_path)
    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Failed to create rigid body: {body_path}")
    return body


def _deactivate_old_mount_joints(stage: Usd.Stage, robot_path: str) -> None:
    root = stage.GetPrimAtPath(robot_path)
    for prim in Usd.PrimRange(root):
        if prim.GetName() not in _PREEXISTING_MOUNT_JOINT_NAMES:
            continue
        joint = UsdPhysics.Joint(prim)
        if joint:
            joint.GetJointEnabledAttr().Set(False)
            prim.SetActive(False)


def _remove_nested_articulation_roots(stage: Usd.Stage, subtree_path: str) -> None:
    root = stage.GetPrimAtPath(subtree_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)


def _deinstance_geometry(stage: Usd.Stage, subtree_path: str) -> None:
    root = stage.GetPrimAtPath(subtree_path)
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)


def _sanitize_collision_apis(stage: Usd.Stage, subtree_path: str) -> None:
    valid_geometry_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )
    root = stage.GetPrimAtPath(subtree_path)
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if any(prim.IsA(geometry_type) for geometry_type in valid_geometry_types):
            continue
        prim.RemoveAPI(UsdPhysics.CollisionAPI)


@clone
def spawn_ur5e_robotiq_ft(
    prim_path: str,
    cfg: "Ur5eRobotiqFtSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn and assemble one source robot before Isaac Lab clones it."""
    del kwargs
    stage = get_current_stage()

    ur_cfg = sim_utils.UsdFileCfg(
        usd_path=cfg.ur5e_usd_path,
        rigid_props=cfg.rigid_props,
        articulation_props=cfg.articulation_props,
        activate_contact_sensors=False,
    )
    ur_cfg.func(
        prim_path,
        ur_cfg,
        translation=translation,
        orientation=orientation,
    )

    gripper_path = f"{prim_path}/Robotiq2F85"
    gripper_cfg = sim_utils.UsdFileCfg(usd_path=cfg.robotiq_usd_path)
    gripper_cfg.func(gripper_path, gripper_cfg)

    robot_root = stage.GetPrimAtPath(prim_path)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    if not robot_root.IsValid() or not gripper_root.IsValid():
        raise RuntimeError(
            "UR5e or Robotiq USD did not compose into the stage. "
            "Check SWEEP_UR5E_USD_PATH and SWEEP_ROBOTIQ_USD_PATH."
        )

    _deinstance_geometry(stage, gripper_path)
    tool_frame = _find_named_prim(stage, prim_path, cfg.tool_frame_candidates)
    tool_body = _nearest_rigid_body_ancestor(tool_frame)
    gripper_base = _find_named_prim(
        stage,
        gripper_path,
        cfg.gripper_base_candidates,
        rigid_body_only=True,
    )

    _deactivate_old_mount_joints(stage, prim_path)
    root_world = _world_transform(gripper_root)
    base_world = _world_transform(gripper_base)
    base_relative_to_root = base_world * root_world.GetInverse()
    sensor_world = _world_transform(tool_frame)
    desired_gripper_base_world = (
        _make_offset_matrix(cfg.mount_translation, cfg.mount_rotation_deg)
        * sensor_world
    )
    desired_gripper_root_world = (
        base_relative_to_root.GetInverse() * desired_gripper_base_world
    )
    _set_world_transform_on_reference_root(
        gripper_root, desired_gripper_root_world
    )

    _remove_nested_articulation_roots(stage, gripper_path)
    _sanitize_collision_apis(stage, prim_path)
    _sanitize_collision_apis(stage, gripper_path)

    ft_body = _spawn_fixed_body(
        stage,
        prim_path,
        FT_SENSOR_BODY_NAME,
        sensor_world,
        size=cfg.ft_sensor_size,
        mass=cfg.ft_sensor_mass,
        collision_enabled=False,
        color=(0.95, 0.55, 0.10),
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/UR5e_virtual_ft_parent_joint",
        tool_body,
        ft_body,
        sensor_world,
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/VirtualFTSensor_gripper_child_joint",
        ft_body,
        gripper_base,
        sensor_world,
    )

    center_world = (
        _make_offset_matrix(cfg.eef_center_offset, (0.0, 0.0, 0.0))
        * desired_gripper_base_world
    )
    center_body = _spawn_fixed_body(
        stage,
        prim_path,
        EEF_CENTER_BODY_NAME,
        center_world,
        size=(0.008, 0.008, 0.008),
        mass=cfg.virtual_link_mass,
        collision_enabled=False,
        color=(0.20, 0.80, 0.95),
    )
    _create_fixed_joint(
        stage,
        f"{prim_path}/gripper_center_joint",
        gripper_base,
        center_body,
        center_world,
    )

    return robot_root


@configclass
class Ur5eRobotiqFtSpawnerCfg(RigidObjectSpawnerCfg):
    """Configuration for the assembled sweep robot."""

    func: Callable = spawn_ur5e_robotiq_ft

    ur5e_usd_path: str = DEFAULT_UR5E_USD_PATH
    robotiq_usd_path: str = DEFAULT_ROBOTIQ_USD_PATH
    rigid_props: sim_utils.RigidBodyPropertiesCfg = (
        sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        )
    )
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = (
        sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        )
    )
    tool_frame_candidates: tuple[str, ...] = _UR_TOOL_FRAME_CANDIDATES
    gripper_base_candidates: tuple[str, ...] = _GRIPPER_BASE_BODY_CANDIDATES
    mount_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount_rotation_deg: tuple[float, float, float] = (0.0, 90.0, 0.0)

    ft_sensor_size: tuple[float, float, float] = (0.025, 0.025, 0.025)
    ft_sensor_mass: float = 1.0e-3
    virtual_link_mass: float = 1.0e-3
    eef_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.16)


def make_ur5e_robotiq_ft_cfg() -> ArticulationCfg:
    """Create the articulation configuration using environment-overridable USD paths."""
    spawn_cfg = Ur5eRobotiqFtSpawnerCfg(
        ur5e_usd_path=os.environ.get(
            "SWEEP_UR5E_USD_PATH", DEFAULT_UR5E_USD_PATH
        ),
        robotiq_usd_path=os.environ.get(
            "SWEEP_ROBOTIQ_USD_PATH", DEFAULT_ROBOTIQ_USD_PATH
        ),
    )
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=spawn_cfg,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.775),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.75,
                "elbow_joint": 1.90,
                "wrist_1_joint": -1.72,
                "wrist_2_joint": -1.57,
                "wrist_3_joint": 0.0,
                ".*(finger|knuckle).*": 0.0,
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
                stiffness=0.0,
                damping=0.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[".*(finger|knuckle).*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=2000.0,
                damping=100.0,
            ),
        },
    )


def validate_asset_constants() -> None:
    """Cheap import-time validation for values that affect every cloned environment."""
    if len(ARM_JOINT_NAMES) != 6:
        raise ValueError("OSC sweep requires exactly six UR5e arm joints.")
