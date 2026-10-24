"""UR5e specialization following the native example's scene/config composition."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp import JointPositionToLimitsActionCfg, OperationalSpaceControllerActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from ...shelf_manipulation_env_cfg import ShelfManipulationEnvCfg, ShelfManipulationSceneCfg
from ...mdp.sensors import make_tactile_contact_cfg
from .scene import (
    DEFAULT_SHELF_USD_PATH, SHELF_POSITION, TACTILE_BODY_NAMES,
    make_object_collection_cfg, make_ur5e_hand_cfg, spawn_shelf, SharedGeometryCfg,
)


@configclass
class UR5eContactSceneCfg(ShelfManipulationSceneCfg):
    """One authored template and fixed object pool, without rendered perception."""

    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            func=spawn_shelf,
            usd_path=os.environ.get("CONTACT_MANIPULATION_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False, disable_gravity=False),
            # Match examples/Sweeping-Policy; PhysX derives inertia from the colliders.
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=SHELF_POSITION),
    )
    robot = make_ur5e_hand_cfg()
    object_collection = make_object_collection_cfg()
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link", name="end_effector",
        )],
    )
    tactile_contact = make_tactile_contact_cfg(TACTILE_BODY_NAMES)
    # Run last, after all collider/contact edits and before native cloning.
    shared_geometry = AssetBaseCfg(prim_path="/World/SharedGeometry", spawn=SharedGeometryCfg())


@configclass
class UR5eContactManipulationEnvCfg(ShelfManipulationEnvCfg):
    """Torque OSC arm, independent hand PD and geometric initial perception."""

    scene: UR5eContactSceneCfg = UR5eContactSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False,
    )

    def __post_init__(self):
        super().__post_init__()
        # The native OSC supplies feed-forward efforts. Zero actuator gains
        # preserve torque control; IdealPD applies the configured effort limits.
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0
        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            # Use the UR5e terminal link, before the sensor/hand, for action and observation.
            task_frame_rel_path="Robot/wrist_3_link",
            position_scale=0.5,
            orientation_scale=0.5,
            stiffness_scale=100.0,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="variable_kp",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="none",
            ),
            nullspace_joint_pos_target="none",
        )
        self.actions.hand_action = JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=[
                "right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
                "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint",
            ],
            scale=0.5,
            preserve_order=True,
            rescale_to_limits=True,
        )
        # Velocity cost and termination use the same twelve independent joints.
        independent_joint_names = self.actions.arm_action.joint_names + self.actions.hand_action.joint_names
        self.rewards.motion_joint_velocity.params["asset_cfg"].joint_names = independent_joint_names
        if self.terminations.hand_velocity is not None:
            self.terminations.hand_velocity.params["asset_cfg"].joint_names = independent_joint_names
