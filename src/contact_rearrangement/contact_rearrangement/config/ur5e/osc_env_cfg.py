"""UR5e + hand assets, contact reporters, and operational-space control settings."""

from isaaclab.utils import configclass

from ...shelf_rearrangement_env_cfg import (
    LearningActionsCfg, ShelfRearrangementEnvCfg, ShelfRearrangementSceneCfg,
    ShelfRearrangementSpawnEnvCfg,
)
from ...mdp.actions import RearrangementActionCfg
from ...src.assets import HAND_JOINT_NAMES, make_object_collection_cfg, make_ur5e_hand_cfg
from ...src.sensors import make_forbidden_contact_cfg, make_tactile_contact_cfg


@configclass
class UR5eHandSceneCfg(ShelfRearrangementSceneCfg):
    robot = make_ur5e_hand_cfg()
    objects = make_object_collection_cfg()

    # Keep all 17 URDF sensor links separate: palm, thumb 1-4, then
    # index/middle/ring/little 1-3. Each attribute creates its own ContactSensor.
    contact_right_plam_force_sensor = make_tactile_contact_cfg("right_plam_force_sensor")
    contact_right_thumb_force_sensor_1 = make_tactile_contact_cfg("right_thumb_force_sensor_1")
    contact_right_thumb_force_sensor_2 = make_tactile_contact_cfg("right_thumb_force_sensor_2")
    contact_right_thumb_force_sensor_3 = make_tactile_contact_cfg("right_thumb_force_sensor_3")
    contact_right_thumb_force_sensor_4 = make_tactile_contact_cfg("right_thumb_force_sensor_4")
    contact_right_index_force_sensor_1 = make_tactile_contact_cfg("right_index_force_sensor_1")
    contact_right_index_force_sensor_2 = make_tactile_contact_cfg("right_index_force_sensor_2")
    contact_right_index_force_sensor_3 = make_tactile_contact_cfg("right_index_force_sensor_3")
    contact_right_middle_force_sensor_1 = make_tactile_contact_cfg("right_middle_force_sensor_1")
    contact_right_middle_force_sensor_2 = make_tactile_contact_cfg("right_middle_force_sensor_2")
    contact_right_middle_force_sensor_3 = make_tactile_contact_cfg("right_middle_force_sensor_3")
    contact_right_ring_force_sensor_1 = make_tactile_contact_cfg("right_ring_force_sensor_1")
    contact_right_ring_force_sensor_2 = make_tactile_contact_cfg("right_ring_force_sensor_2")
    contact_right_ring_force_sensor_3 = make_tactile_contact_cfg("right_ring_force_sensor_3")
    contact_right_little_force_sensor_1 = make_tactile_contact_cfg("right_little_force_sensor_1")
    contact_right_little_force_sensor_2 = make_tactile_contact_cfg("right_little_force_sensor_2")
    contact_right_little_force_sensor_3 = make_tactile_contact_cfg("right_little_force_sensor_3")


@configclass
class UR5eHandLearningSceneCfg(UR5eHandSceneCfg):
    """Keep allowed hand contact separate from arm/spacer scene collisions."""

    forbidden_base_link = make_forbidden_contact_cfg("base_link")
    forbidden_shoulder_link = make_forbidden_contact_cfg("shoulder_link")
    forbidden_upper_arm_link = make_forbidden_contact_cfg("upper_arm_link")
    forbidden_forearm_link = make_forbidden_contact_cfg("forearm_link")
    forbidden_wrist_1_link = make_forbidden_contact_cfg("wrist_1_link")
    forbidden_wrist_2_link = make_forbidden_contact_cfg("wrist_2_link")
    forbidden_wrist_3_link = make_forbidden_contact_cfg("wrist_3_link")
    forbidden_SensorMountProximal = make_forbidden_contact_cfg("SensorMountProximal")
    forbidden_ToolForceTorqueLink = make_forbidden_contact_cfg("ToolForceTorqueLink")


@configclass
class UR5eHandActionsCfg(LearningActionsCfg):
    manipulate = RearrangementActionCfg(asset_name="robot")


@configclass
class ContactRearrangementEnvCfg(ShelfRearrangementSpawnEnvCfg):
    """Spawn and sensing preview with the robot holding its initial pose."""

    scene: UR5eHandSceneCfg = UR5eHandSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)


@configclass
class ContactRearrangementRLEnvCfg(ShelfRearrangementEnvCfg):
    """18D variable-stiffness OSC and independent hand-joint position actions."""

    scene: UR5eHandLearningSceneCfg = UR5eHandLearningSceneCfg(
        num_envs=4, env_spacing=2.5, replicate_physics=True
    )
    actions: UR5eHandActionsCfg = UR5eHandActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0
        # Only independent hand joints receive posture drives. Authored mimic
        # constraints couple the remaining joints without competing PD targets.
        mimic_joints = "(?!(?:" + "|".join(HAND_JOINT_NAMES) + ")$).*(thumb|index|middle|ring|little).*"
        self.scene.robot.actuators["hand"].stiffness = {
            mimic_joints: 0.0, **{name: 20.0 for name in HAND_JOINT_NAMES}
        }
        self.scene.robot.actuators["hand"].damping = {
            mimic_joints: 0.0, **{name: 1.0 for name in HAND_JOINT_NAMES}
        }
