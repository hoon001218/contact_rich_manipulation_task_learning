"""Manager-based UR5e/RH56 primitive shelf-grasp environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp
from .assets import ARM_JOINT_NAMES, FINGERTIP_BODY_NAMES, HAND_JOINT_NAMES, TCP_BODY_NAME, make_ur5e_rh56_cfg


SHELF_TOP_HEIGHT = 0.495
OBJECT_SIDE_LENGTH = 0.05

ARM_CFG = SceneEntityCfg("robot", joint_names=list(ARM_JOINT_NAMES), preserve_order=True)
HAND_CFG = SceneEntityCfg("robot", joint_names=list(HAND_JOINT_NAMES), preserve_order=True)
CONTROLLED_JOINT_CFG = SceneEntityCfg(
    "robot", joint_names=list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES), preserve_order=True
)
TCP_CFG = SceneEntityCfg("robot", body_names=[TCP_BODY_NAME])
FINGERTIP_CFG = SceneEntityCfg("robot", body_names=list(FINGERTIP_BODY_NAMES), preserve_order=True)
OBJECT_CFG = SceneEntityCfg("object")


def _shelf_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0)


@configclass
class ShelfGraspSceneCfg(InteractiveSceneCfg):
    """Assembled robot, primitive shelf, and grasp object."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    shelf_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShelfBoard",
        spawn=sim_utils.CuboidCfg(
            size=(0.58, 0.72, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_shelf_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.30, 0.20)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.62, 0.0, SHELF_TOP_HEIGHT - 0.025)),
    )
    shelf_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShelfBack",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.72, 0.72),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_shelf_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.27, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.93, 0.0, 0.83)),
    )
    shelf_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShelfLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.58, 0.04, 0.72),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_shelf_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.27, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.62, 0.38, 0.83)),
    )
    shelf_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ShelfRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.58, 0.04, 0.72),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=_shelf_material(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.38, 0.27, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.62, -0.38, 0.83)),
    )

    robot = make_ur5e_rh56_cfg()

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(OBJECT_SIDE_LENGTH, OBJECT_SIDE_LENGTH, OBJECT_SIDE_LENGTH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 0.90)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.58, 0.0, SHELF_TOP_HEIGHT + 0.5 * OBJECT_SIDE_LENGTH + 0.002),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )


@configclass
class ActionsCfg:
    """Six arm actions followed by six independent RH56 actions."""

    arm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        scale=0.35,
        use_default_offset=True,
        preserve_order=True,
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
    @configclass
    class PolicyCfg(ObsGroup):
        arm_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ARM_CFG})
        arm_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ARM_CFG}, scale=0.1)
        hand_joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, params={"asset_cfg": HAND_CFG})
        hand_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": HAND_CFG}, scale=0.2)
        tcp_pose = ObsTerm(func=mdp.tcp_pose_in_robot_root_frame, params={"tcp_cfg": TCP_CFG})
        object_pose = ObsTerm(
            func=mdp.object_pose_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": OBJECT_CFG},
        )
        object_to_tcp = ObsTerm(func=mdp.object_to_tcp_vector, params={"tcp_cfg": TCP_CFG, "object_cfg": OBJECT_CFG})
        object_velocity = ObsTerm(
            func=mdp.object_velocity_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": OBJECT_CFG},
            scale=0.2,
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.04, 0.04),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": CONTROLLED_JOINT_CFG,
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.07, 0.07), "y": (-0.18, 0.18), "yaw": (-math.pi, math.pi)},
            "velocity_range": {},
            "asset_cfg": OBJECT_CFG,
        },
    )


@configclass
class RewardsCfg:
    reach_object = RewTerm(
        func=mdp.tcp_object_proximity,
        weight=2.0,
        params={"std": 0.12, "tcp_cfg": TCP_CFG, "object_cfg": OBJECT_CFG},
    )
    enclose_object = RewTerm(
        func=mdp.fingertip_enclosure,
        weight=3.0,
        params={"std": 0.10, "fingertip_cfg": FINGERTIP_CFG, "object_cfg": OBJECT_CFG},
    )
    lift_progress = RewTerm(
        func=mdp.lift_progress,
        weight=12.0,
        params={
            "shelf_top_height": SHELF_TOP_HEIGHT,
            "object_half_height": 0.5 * OBJECT_SIDE_LENGTH,
            "target_lift_height": 0.12,
            "object_cfg": OBJECT_CFG,
        },
    )
    held_grasp = RewTerm(
        func=mdp.lifted_and_held,
        weight=6.0,
        params={
            "shelf_top_height": SHELF_TOP_HEIGHT,
            "object_half_height": 0.5 * OBJECT_SIDE_LENGTH,
            "minimum_lift": 0.03,
            "maximum_tcp_distance": 0.18,
            "tcp_cfg": TCP_CFG,
            "object_cfg": OBJECT_CFG,
        },
    )
    success = RewTerm(
        func=mdp.success_bonus,
        weight=25.0,
        params={
            "shelf_top_height": SHELF_TOP_HEIGHT,
            "object_half_height": 0.5 * OBJECT_SIDE_LENGTH,
            "success_lift_height": 0.12,
            "maximum_tcp_distance": 0.18,
            "maximum_tip_distance": 0.09,
            "minimum_close_fingertips": 2,
            "tcp_cfg": TCP_CFG,
            "fingertip_cfg": FINGERTIP_CFG,
            "object_cfg": OBJECT_CFG,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.object_lifted_to_goal,
        params={
            "shelf_top_height": SHELF_TOP_HEIGHT,
            "object_half_height": 0.5 * OBJECT_SIDE_LENGTH,
            "success_lift_height": 0.12,
            "maximum_tcp_distance": 0.18,
            "maximum_tip_distance": 0.09,
            "minimum_close_fingertips": 2,
            "tcp_cfg": TCP_CFG,
            "fingertip_cfg": FINGERTIP_CFG,
            "object_cfg": OBJECT_CFG,
        },
    )
    object_dropped = DoneTerm(
        func=mdp.object_fell_from_shelf,
        params={"minimum_height": 0.25, "object_cfg": OBJECT_CFG},
    )


@configclass
class CommandsCfg:
    pass


@configclass
class CurriculumCfg:
    pass


@configclass
class HandGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Vectorized joint-position shelf-grasp task."""

    scene: ShelfGraspSceneCfg = ShelfGraspSceneCfg(num_envs=1024, env_spacing=2.2, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.viewer.eye = (2.0, 2.0, 1.7)
        self.viewer.lookat = (0.55, 0.0, 0.60)
