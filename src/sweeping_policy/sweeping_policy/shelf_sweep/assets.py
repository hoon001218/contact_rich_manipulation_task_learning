"""Nucleus asset paths and the combined UR5e/Robotiq articulation."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
GRIPPER_JOINT_EXPR = ".*(finger|knuckle).*"
GRIPPER_JOINT_NAMES = (
    "finger_joint",
    "right_outer_knuckle_joint",
    "left_outer_finger_joint",
    "left_inner_finger_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_finger_joint",
    "right_inner_finger_joint",
    "right_inner_finger_knuckle_joint",
)

NUCLEUS_ROOT = "omniverse://192.168.0.13"
DEFAULT_ROBOT_USD_PATH = (
    f"{NUCLEUS_ROOT}/Library/Shelf/Robots/UR5e/Collected_UR5e_v4/"
    "UR5e_v4.usd"
)
DEFAULT_SHELF_USD_PATH = (
    f"{NUCLEUS_ROOT}/Library/Shelf/Arena/Collected_speedrack_shape/"
    "speedrack_shape.usd"
)
DEFAULT_OBJECT_USD_PATHS = {
    "bottle_1": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Bottle_6/Bottle_6.usd",
    "cup_1": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Collected_Cup_1/Cup_1.usd",
    "cup_2": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Collected_Cup_4/Cup_4.usd",
    "mug_1": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Collected_Mug_2/Mug_2.usd",
    "mug_2": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Collected_Mug_3/Mug_3.usd",
    "can_1": f"{NUCLEUS_ROOT}/Library/Shelf/Objects/Can_6/Can_6.usd",
}


def make_ur5e_robotiq_cfg() -> ArticulationCfg:
    """Load the pre-assembled UR5e/Robotiq USD as one articulation."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEPING_POLICY_ROBOT_USD_PATH", DEFAULT_ROBOT_USD_PATH
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.79505),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -2.2,
                "elbow_joint": 2.2,
                "wrist_1_joint": 0.0,
                "wrist_2_joint": 1.57,
                "wrist_3_joint": 0.785,
                GRIPPER_JOINT_EXPR: 0.0,
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
                stiffness={
                    "shoulder_pan_joint": 3328.7,
                    "shoulder_lift_joint": 4989.4,
                    "elbow_joint": 2394.2,
                    "wrist_1_joint": 3500.2,
                    "wrist_2_joint": 3174.8,
                    "wrist_3_joint": 3032.4,
                },
                damping={
                    "shoulder_pan_joint": 368.8,
                    "shoulder_lift_joint": 561.2,
                    "elbow_joint": 274.1,
                    "wrist_1_joint": 396.2,
                    "wrist_2_joint": 363.1,
                    "wrist_3_joint": 345.0,
                },
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=list(GRIPPER_JOINT_NAMES),
                effort_limit_sim=200.0,
                velocity_limit_sim=2.0,
                stiffness=2000.0,
                damping=1000.0,
            ),
        },
    )
