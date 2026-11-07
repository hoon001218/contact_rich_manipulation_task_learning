"""UR5e specialization following the native example's scene/config composition."""

import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.envs.mdp import JointPositionToLimitsActionCfg, OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers import EventTermCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from ...shelf_manipulation_env_cfg import ShelfManipulationEnvCfg, ShelfManipulationSceneCfg
from ...mdp.sensors import make_tactile_contact_cfg
from ...mdp.events import configure_hand_action_limits
from ...mdp.observations import wrench_action_limits
from .scene import (
    DEFAULT_SHELF_USD_PATH, SHELF_POSITION, TACTILE_BODY_NAMES, FORCE_TORQUE_BODY_NAME,
    make_object_collection_cfg, make_ur5e_hand_cfg, spawn_shelf, SharedGeometryCfg,
)


class SensorFrameWrenchAction(OperationalSpaceControllerAction):
    """Native OSC with sensor-origin wrench inputs, without replacing its controller.

    Native OSC transforms a task wrench's moment to the root origin before
    multiplying the COM geometric Jacobian. Adapt only that reference point:
    after its native transform the moment must instead be about the sensor
    body's COM. Scaling, gravity compensation, effort submission and actuator
    limits remain native. Sensor-range clipping precedes native scaling and
    moment conversion. The authored COM and raw bounds are cached once.
    """

    def __init__(self, cfg, env):
        if (list(cfg.controller_cfg.target_types) != ["wrench_abs"]
                or cfg.controller_cfg.impedance_mode != "fixed"
                or cfg.body_offset is not None
                or cfg.task_frame_rel_path != f"Robot/{cfg.body_name}"):
            raise ValueError("Sensor wrench actions require fixed pure wrench control at the sensor body/frame.")
        raw_limits = wrench_action_limits(cfg.wrench_scale)
        super().__init__(cfg, env)
        self._raw_wrench_limits = self.raw_actions.new_tensor(raw_limits)
        self._sensor_com_offset = self._asset.data.body_com_pos_b[:, self._ee_body_idx].clone()

    def _preprocess_actions(self, actions):
        # Keep the policy sample and ActionManager history intact for PPO.
        # Native OSC does not implement ActionCfg.clip for wrench commands.
        super()._preprocess_actions(actions.clamp(-self._raw_wrench_limits, self._raw_wrench_limits))
        # In sensor axes, p_COM/root = R^T p_sensor/root + c_COM/sensor.
        # Native OSC subsequently adds p_sensor/root x F_root. This subtraction
        # leaves tau_COM = R tau_sensor + (p_sensor - p_COM) x F_root.
        lever = (quat_apply_inverse(self._ee_pose_b[:, 3:7], self._ee_pose_b[:, :3])
                 + self._sensor_com_offset)
        self.processed_actions[:, 3:6] -= torch.cross(lever, self.processed_actions[:, :3], dim=-1)


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
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{FORCE_TORQUE_BODY_NAME}", name="end_effector",
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
            class_type=SensorFrameWrenchAction,
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            ],
            body_name=FORCE_TORQUE_BODY_NAME,
            # The load-cell center defines control axes and all EEF observations.
            task_frame_rel_path=f"Robot/{FORCE_TORQUE_BODY_NAME}",
            wrench_scale=(10.0, 10.0, 10.0, 1.0, 1.0, 1.0),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["wrench_abs"],
                impedance_mode="fixed",
                motion_control_axes_task=(0, 0, 0, 0, 0, 0),
                contact_wrench_control_axes_task=(1, 1, 1, 1, 1, 1),
                inertial_dynamics_decoupling=False,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=True,
                contact_wrench_stiffness_task=None,
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
        self.events.hand_action_limits = EventTermCfg(func=configure_hand_action_limits, mode="startup")
        # If enabled, velocity cost covers twelve independent joints; overspeed is arm-only.
        velocity_reward = getattr(self.rewards, "motion_joint_velocity", None)
        if velocity_reward is not None:
            velocity_reward.params["asset_cfg"].joint_names = (
                self.actions.arm_action.joint_names + self.actions.hand_action.joint_names
            )
        if self.terminations.hand_velocity is not None:
            self.terminations.hand_velocity.params["asset_cfg"].joint_names = self.actions.arm_action.joint_names
