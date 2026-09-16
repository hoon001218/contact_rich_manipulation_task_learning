"""Manager-based shelf-sweeping scene with a UR5e and ROAS tactile hand."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import (
    ARM_JOINT_NAMES,
    CONTACT_FORCE_THRESHOLD,
    DEFAULT_OBJECT_USD_PATHS,
    DEFAULT_SHELF_USD_PATH,
    EEF_BODY_NAME,
    FORCE_SENSOR_BODY_NAMES,
    HAND_JOINT_NAMES,
    make_force_contact_sensor_cfg,
    make_ur5e_roas_hand_cfg,
)


OBJECT_INITIAL_POSES = {
    "bottle_1": (-0.75, -0.20, 1.05),
    "cup_1": (-0.75, 0.00, 1.05),
    "cup_2": (-0.75, 0.20, 1.05),
    "mug_1": (-0.60, -0.20, 1.05),
    "mug_2": (-0.60, 0.00, 1.05),
    "can_1": (-0.60, 0.20, 1.05),
}

WORKING_X_RANGE = (-0.78, -0.57)
WORKING_Y_RANGE = (-0.22, 0.22)
WORKING_HEIGHT = 1.05
WAITING_HEIGHT = 1.80
WAITING_XY = tuple((pose[0], pose[1]) for pose in OBJECT_INITIAL_POSES.values())
OBJECT_WIDTHS = (0.06, 0.06, 0.05, 0.09, 0.09, 0.06)
PUSH_DISTANCE = 0.18

ARM_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(ARM_JOINT_NAMES),
    preserve_order=True,
)
HAND_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(HAND_JOINT_NAMES),
    preserve_order=True,
)
EEF_CFG = SceneEntityCfg("robot", body_names=[EEF_BODY_NAME])
FORCE_SENSOR_SCENE_NAMES = tuple(
    f"contact_{body_name}" for body_name in FORCE_SENSOR_BODY_NAMES
)


def _asset_override(name: str, default: str) -> str:
    """Resolve this project's asset override, then the sweeping-policy override."""
    new_name = f"ROAS_SWEEPING_{name.upper()}_USD_PATH"
    legacy_name = f"SWEEPING_POLICY_{name.upper()}_USD_PATH"
    return os.environ.get(new_name, os.environ.get(legacy_name, default))


def _make_object_collection() -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for name, default_path in DEFAULT_OBJECT_USD_PATHS.items():
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=OBJECT_INITIAL_POSES[name],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=_asset_override(name, default_path),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                ),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


@configclass
class RoasShelfSweepSceneCfg(InteractiveSceneCfg):
    """The reference shelf/objects plus the assembled tactile-hand robot."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_asset_override("shelf", DEFAULT_SHELF_USD_PATH),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    robot = make_ur5e_roas_hand_cfg()
    objects = _make_object_collection()

    # The URDF importer keeps fixed joints so every force-sensor mesh remains a
    # separate rigid body. Each body gets an independent ContactSensor instance.
    contact_palm_force_sensor = make_force_contact_sensor_cfg("palm_force_sensor")
    contact_thumb_force_sensor_1 = make_force_contact_sensor_cfg("thumb_force_sensor_1")
    contact_thumb_force_sensor_2 = make_force_contact_sensor_cfg("thumb_force_sensor_2")
    contact_thumb_force_sensor_3 = make_force_contact_sensor_cfg("thumb_force_sensor_3")
    contact_thumb_force_sensor_4 = make_force_contact_sensor_cfg("thumb_force_sensor_4")
    contact_index_force_sensor_1 = make_force_contact_sensor_cfg("index_force_sensor_1")
    contact_index_force_sensor_2 = make_force_contact_sensor_cfg("index_force_sensor_2")
    contact_index_force_sensor_3 = make_force_contact_sensor_cfg("index_force_sensor_3")
    contact_middle_force_sensor_1 = make_force_contact_sensor_cfg("middle_force_sensor_1")
    contact_middle_force_sensor_2 = make_force_contact_sensor_cfg("middle_force_sensor_2")
    contact_middle_force_sensor_3 = make_force_contact_sensor_cfg("middle_force_sensor_3")
    contact_ring_force_sensor_1 = make_force_contact_sensor_cfg("ring_force_sensor_1")
    contact_ring_force_sensor_2 = make_force_contact_sensor_cfg("ring_force_sensor_2")
    contact_ring_force_sensor_3 = make_force_contact_sensor_cfg("ring_force_sensor_3")
    contact_little_force_sensor_1 = make_force_contact_sensor_cfg("little_force_sensor_1")
    contact_little_force_sensor_2 = make_force_contact_sensor_cfg("little_force_sensor_2")
    contact_little_force_sensor_3 = make_force_contact_sensor_cfg("little_force_sensor_3")

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=2500.0,
        ),
    )


@configclass
class ActionsCfg:
    """Relative palm pose control and six independent hand-joint commands."""

    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=EEF_BODY_NAME,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
    )
    hand = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=list(HAND_JOINT_NAMES),
        scale=1.0,
        rescale_to_limits=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Robot, target, and all 17 binary tactile-contact observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": ARM_CFG},
        )
        arm_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": ARM_CFG},
            scale=0.1,
        )
        hand_joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            params={"asset_cfg": HAND_CFG},
        )
        hand_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": HAND_CFG},
            scale=0.2,
        )
        target_position = ObsTerm(
            func=mdp.target_position_in_eef_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "eef_cfg": EEF_CFG,
                "object_collection_cfg": SceneEntityCfg("objects"),
            },
        )
        target_width = ObsTerm(func=mdp.target_object_width)
        contact_states = ObsTerm(
            func=mdp.force_sensor_contacts,
            params={
                "sensor_names": FORCE_SENSOR_SCENE_NAMES,
                "force_threshold": CONTACT_FORCE_THRESHOLD,
            },
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    randomize_objects = EventTerm(
        func=mdp.randomize_working_and_waiting_objects,
        mode="reset",
        params={
            "working_x_range": WORKING_X_RANGE,
            "working_y_range": WORKING_Y_RANGE,
            "working_height": WORKING_HEIGHT,
            "waiting_height": WAITING_HEIGHT,
            "waiting_xy": WAITING_XY,
            "object_widths": OBJECT_WIDTHS,
            "push_distance": PUSH_DISTANCE,
            "asset_cfg": SceneEntityCfg("objects"),
        },
    )


@configclass
class RewardsCfg:
    """Minimal reward terms retained only for environment smoke training."""

    target_goal = RewTerm(
        func=mdp.target_goal_proximity,
        weight=1.0,
        params={
            "std": PUSH_DISTANCE,
            "object_collection_cfg": SceneEntityCfg("objects"),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    arm_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": ARM_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.any_object_below_height,
        params={
            "minimum_height": 0.90,
            "object_collection_cfg": SceneEntityCfg("objects"),
        },
    )


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class RoasShelfSweepEnvCfg(ManagerBasedRLEnvCfg):
    """Initial vectorized environment for robot/sensor validation."""

    scene: RoasShelfSweepSceneCfg = RoasShelfSweepSceneCfg(
        num_envs=1024,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**28
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.viewer.eye = (2.2, 2.2, 1.8)
        self.viewer.lookat = (-0.45, 0.0, 0.85)
