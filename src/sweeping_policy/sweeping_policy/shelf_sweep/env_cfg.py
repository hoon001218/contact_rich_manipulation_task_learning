"""Manager-based scene baseline for the UR5e shelf-sweeping task."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise


from isaaclab.sensors import FrameTransformerCfg
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from . import mdp
from .assets import (
    ARM_JOINT_NAMES,
    DEFAULT_OBJECT_USD_PATHS,
    DEFAULT_SHELF_USD_PATH,
    GRIPPER_JOINT_NAMES,
    make_ur5e_robotiq_cfg,
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
EEF_BODY_NAME = "robotiq_base_link"
EEF_OFFSET = (0.13, 0.0, 0.0)

ARM_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True
)
EEF_CFG = SceneEntityCfg("robot", body_names=[EEF_BODY_NAME])


def _make_object_collection() -> RigidObjectCollectionCfg:
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for name, default_path in DEFAULT_OBJECT_USD_PATHS.items():
        env_name = f"SWEEPING_POLICY_{name.upper()}_USD_PATH"
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=OBJECT_INITIAL_POSES[name],
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.environ.get(env_name, default_path),
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
class ShelfSweepSceneCfg(InteractiveSceneCfg):
    """Ground, shelf, assembled robot, and the six reference objects."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    shelf = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get(
                "SWEEPING_POLICY_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.7, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    robot = make_ur5e_robotiq_cfg()
    objects = _make_object_collection()
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=2500.0
        ),
    )


@configclass
class ActionsCfg:
    """Six-dimensional relative EEF pose plus one binary gripper action."""

    arm = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=EEF_BODY_NAME,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=EEF_OFFSET
        ),
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        # Normalized policy action -> [metres, axis-angle radians].
        # scale=(0.02, 0.02, 0.02, 0.10, 0.10, 0.10),
        scale=0.5
    )
    gripper = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(GRIPPER_JOINT_NAMES),
        open_command_expr={".*": 0.0},
        close_command_expr={
            "finger_joint": 0.5,
            "right_outer_knuckle_joint": 0.5,
            "left_outer_finger_joint": 0.0,
            "left_inner_finger_knuckle_joint": -0.5,
            "left_inner_finger_joint": -0.5,
            "right_outer_finger_joint": 0.0,
            "right_inner_finger_joint": 0.5,
            "right_inner_finger_knuckle_joint": -0.5,
        },
    )


@configclass
class ObservationsCfg:
    """Reference observation order with target and goal expressed in EEF."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.sweep_joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_vel = ObsTerm(
            func=mdp.sweep_joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)
        target_obs_state = ObsTerm(
            func=mdp.target_position_in_eef_frame,
            params={
                "eef_cfg": EEF_CFG,
                "object_collection_cfg": SceneEntityCfg("objects"),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        target_obj_width = ObsTerm(
            func=mdp.target_object_width,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        ee_pose = ObsTerm(
            func=mdp.eef_pose_in_robot_root_frame,
            params={"eef_cfg": EEF_CFG},
        )
        goal_pos = ObsTerm(
            func=mdp.goal_position_in_eef_frame,
            params={"eef_cfg": EEF_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = True
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
    initialize_reaching_pose = EventTerm(
        func=mdp.initialize_robot_at_reaching_pose,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "shelf_cfg": SceneEntityCfg("shelf"),
            "arm_joint_names": ARM_JOINT_NAMES,
            "eef_body_name": EEF_BODY_NAME,
            "eef_offset": EEF_OFFSET,
            "reaching_x_offset": -0.02,
            "reaching_z_offset": 0.09,
            "position_noise": 0.01,
            "max_iterations": 40,
            "damping": 0.05,
            "step_size": 0.5,
            "position_tolerance": 0.003,
            "orientation_tolerance": 0.001,
            # The object pose, push direction, and EEF position noise stay fixed.
            # Failed environments retry only from these bounded arm joint seeds.
            "joint_seed_offsets": (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.15, -0.10, 0.10, 0.0, 0.0, 0.0),
                (-0.15, 0.10, -0.10, 0.0, 0.0, 0.0),
            ),
        },
    )


@configclass
class RewardsCfg:
    """Reward formulation from the reference random sweep environment."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    shelf_collision = RewTerm(func=mdp.shelf_Collision, weight=-0.5)
    object_collision = RewTerm(func=mdp.object_collision, weight=-0.5)
    reaching = RewTerm(func=mdp.reward_for_hand_reaching, weight=0.0)
    orientation = RewTerm(func=mdp.align_ee_target, weight=2.0)
    sweeping_object = RewTerm(func=mdp.pushing_target, weight=6.0)
    homing_after_sweep = RewTerm(func=mdp.homing_reward, weight=9.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_drop = DoneTerm(
        func=mdp.drop_object_termination,
        time_out=False,
        params={
            "height_condition": 1.04,
            "rotation_condition": 0.9,
            "object_collection_cfg": SceneEntityCfg("objects"),
        },
    )
    push_fast = DoneTerm(
        func=mdp.push_fast_termination,
        time_out=False,
        params={
            "speed_condition": 0.3,
            "object_collection_cfg": SceneEntityCfg("objects"),
        },
    )
    shelf_collision = DoneTerm(
        func=mdp.shelf_collision_termination,
        time_out=False,
        params={"threshold": 0.1},
    )
    hand_velocity = DoneTerm(
        func=mdp.hand_velocity_termination,
        time_out=False,
        params={
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    obj_collision = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "object_collision",
            "weight": -0.5,
            "num_steps": 250_000,
        },
    )


@configclass
class ShelfSweepEnvCfg(ManagerBasedRLEnvCfg):
    """Initial vectorized training configuration for scene validation."""

    scene: ShelfSweepSceneCfg = ShelfSweepSceneCfg(
        num_envs=4096,
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
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = (
            1024 * 1024 * 16 * 16
        )
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024 * 16
        self.viewer.eye = (2.2, 2.2, 1.8)
        self.viewer.lookat = (-0.45, 0.0, 0.85)


        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.13, 0.0, 0.0],
                    ),
                ),
            ],
        )


