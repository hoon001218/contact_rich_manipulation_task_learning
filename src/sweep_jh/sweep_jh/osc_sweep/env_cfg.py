"""Independent manager-based Sweep JH environment."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import OperationalSpaceControllerCfg
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
from .assets import (
    ARM_JOINT_NAMES,
    EEF_CENTER_BODY_NAME,
    FT_SENSOR_BODY_NAME,
    GRIPPER_SIDE_AXIS_LOCAL,
    make_ur5e_robotiq_ft_cfg,
)

ARM_ENTITY_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(ARM_JOINT_NAMES),
    preserve_order=True,
)
EEF_ENTITY_CFG = SceneEntityCfg(
    "robot",
    body_names=[EEF_CENTER_BODY_NAME],
)
FT_ENTITY_CFG = SceneEntityCfg(
    "robot",
    body_names=[FT_SENSOR_BODY_NAME],
)


@configclass
class OscSweepSceneCfg(InteractiveSceneCfg):
    """Open table, assembled robot, and target cube."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/OpenTable",
        spawn=sim_utils.CuboidCfg(
            size=(1.20, 0.90, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.48, 0.36, 0.24)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.75)),
    )

    robot = make_ur5e_robotiq_ft_cfg()

    target_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.65,
                dynamic_friction=0.45,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.45, 0.90)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.50, 0.0, 0.805),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=2500.0
        ),
    )


@configclass
class CommandsCfg:
    desired_motion = mdp.SweepMotionCommandCfg(
        robot_name="robot",
        object_name="target_object",
        resampling_time_range=(1.0e9, 1.0e9),
        direction_angle_range=(-math.pi, math.pi),
        distance_range=(0.10, 0.22),
        force_range=(8.0, 25.0),
        force_tolerance_range=(3.0, 6.0),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """The policy controls only six stiffness and six temporal pose values."""

    arm_action = mdp.SweepOperationalSpaceActionCfg(
        asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES),
        body_name=EEF_CENTER_BODY_NAME,
        body_offset=None,
        controller_cfg=OperationalSpaceControllerCfg(
            target_types=["pose_rel"],
            impedance_mode="variable_kp",
            motion_control_axes_task=(1, 1, 1, 1, 1, 1),
            contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
            inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True,
            motion_stiffness_task=(
                120.0,
                120.0,
                120.0,
                35.0,
                35.0,
                35.0,
            ),
            motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            motion_stiffness_limits_task=(20.0, 300.0),
            nullspace_control="none",
        ),
        position_scale=0.025,
        orientation_scale=0.12,
        stiffness_scale=1.0,
        nullspace_joint_pos_target="none",
        effort_limit_scale=0.9,
    )


@configclass
class ObservationsCfg:
    """41-D task-space observation without joint, target, or contact-point state."""

    @configclass
    class PolicyCfg(ObsGroup):
        eef_pose = ObsTerm(
            func=mdp.end_effector_pose_b,
            params={"asset_cfg": EEF_ENTITY_CFG},
        )
        eef_twist = ObsTerm(
            func=mdp.end_effector_twist_b,
            params={"asset_cfg": EEF_ENTITY_CFG},
        )
        ft_sensor = ObsTerm(
            func=mdp.virtual_ft_wrench_b,
            params={"asset_cfg": FT_ENTITY_CFG},
        )
        initial_target_pose = ObsTerm(
            func=mdp.initial_target_pose_b,
            params={"command_name": "desired_motion"},
        )
        desired_motion = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "desired_motion"},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    reset_arm = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.04, 0.04),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": ARM_ENTITY_CFG,
        },
    )
    reset_target = EventTerm(
        func=mdp.reset_target_object,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.06, 0.06),
                "y": (-0.18, 0.18),
                "z": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_object"),
        },
    )


@configclass
class RewardsCfg:
    """Normalized Cartesian shaping for approach, push, and goal stopping."""

    push_pose = RewTerm(
        func=mdp.push_pose_error,
        weight=-1.0,
        params={
            "command_name": "desired_motion",
            "stand_off": 0.065,
            "position_scale": 0.10,
            "near_distance": 0.08,
            "far_distance": 0.14,
            "force_low": 1.0,
            "force_high": 4.0,
            "eef_cfg": EEF_ENTITY_CFG,
            "ft_cfg": FT_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    push_axis_alignment = RewTerm(
        func=mdp.push_axis_alignment_error,
        weight=-1.0,
        params={
            "command_name": "desired_motion",
            "side_axis_local": GRIPPER_SIDE_AXIS_LOCAL,
            "near_distance": 0.08,
            "far_distance": 0.18,
            "eef_cfg": EEF_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    normal_force_tracking = RewTerm(
        func=mdp.normal_force_tracking,
        weight=2.0,
        params={
            "command_name": "desired_motion",
            "near_distance": 0.08,
            "far_distance": 0.14,
            "force_low": 1.0,
            "force_high": 4.0,
            "eef_cfg": EEF_ENTITY_CFG,
            "ft_cfg": FT_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    tangential_force = RewTerm(
        func=mdp.tangential_force_ratio,
        weight=-0.75,
        params={
            "command_name": "desired_motion",
            "near_distance": 0.08,
            "far_distance": 0.14,
            "force_scale": 25.0,
            "eef_cfg": EEF_ENTITY_CFG,
            "ft_cfg": FT_ENTITY_CFG,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    delta_progress = RewTerm(
        func=mdp.CartesianDeltaProgress,
        weight=6.0,
        params={"command_name": "desired_motion", "rate_scale": 1.0},
    )
    endpoint_error = RewTerm(
        func=mdp.normalized_endpoint_error,
        weight=-2.0,
        params={
            "command_name": "desired_motion",
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    lateral_error = RewTerm(
        func=mdp.normalized_lateral_error,
        weight=-2.0,
        params={"command_name": "desired_motion"},
    )
    overshoot = RewTerm(
        func=mdp.normalized_overshoot,
        weight=-4.0,
        params={"command_name": "desired_motion"},
    )
    goal_speed = RewTerm(
        func=mdp.near_goal_speed,
        weight=-1.0,
        params={
            "command_name": "desired_motion",
            "goal_region_scale": 0.06,
            "speed_scale": 0.05,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    stopped_at_goal = RewTerm(
        func=mdp.stopped_at_goal,
        weight=5.0,
        params={
            "command_name": "desired_motion",
            "position_tolerance": 0.025,
            "speed_tolerance": 0.02,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )

    pose_action_rate = RewTerm(func=mdp.pose_action_rate, weight=-0.5)
    stiffness_action_rate = RewTerm(
        func=mdp.stiffness_action_rate, weight=-0.25
    )
    force_excess = RewTerm(
        func=mdp.force_limit_excess,
        weight=-0.5,
        params={
            "soft_limit": 75.0,
            "hard_limit": 100.0,
            "ft_cfg": FT_ENTITY_CFG,
        },
    )
    ft_torque = RewTerm(
        func=mdp.ft_torque_excess,
        weight=-0.5,
        params={
            "deadband": 1.5,
            "hard_limit": 15.0,
            "asset_cfg": FT_ENTITY_CFG,
        },
    )
    torque_saturation = RewTerm(
        func=mdp.torque_saturation,
        weight=-0.5,
        params={"action_name": "arm_action"},
    )
    success = RewTerm(
        func=mdp.is_terminated_term,
        weight=600.0,
        params={"term_keys": ["success"]},
    )
    failure = RewTerm(
        func=mdp.is_terminated_term,
        weight=-600.0,
        params={
            "term_keys": [
                "target_invalid_pose",
                "excessive_wrench",
                "arm_speed",
            ]
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.TargetStoppedAtGoal,
        time_out=False,
        params={
            "command_name": "desired_motion",
            "endpoint_threshold": 0.025,
            "lateral_threshold": 0.12,
            "speed_threshold": 0.02,
            "dwell_time": 0.25,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    target_invalid_pose = DoneTerm(
        func=mdp.target_invalid_pose,
        time_out=False,
        params={
            "minimum_height": 0.72,
            "maximum_tilt": 0.80,
            "object_cfg": SceneEntityCfg("target_object"),
        },
    )
    excessive_wrench = DoneTerm(
        func=mdp.excessive_ft_wrench,
        time_out=False,
        params={
            "force_limit": 100.0,
            "torque_limit": 15.0,
            "asset_cfg": FT_ENTITY_CFG,
        },
    )
    arm_speed = DoneTerm(
        func=mdp.arm_joint_speed_limit,
        time_out=False,
        params={"maximum_speed": 6.5, "asset_cfg": ARM_ENTITY_CFG},
    )


@configclass
class CurriculumCfg:
    """No curriculum is required for the first clean baseline."""

    pass


@configclass
class JHSweepEnvCfg(ManagerBasedRLEnvCfg):
    """Vectorized variable-stiffness OSC sweep training environment."""

    scene: OscSweepSceneCfg = OscSweepSceneCfg(
        num_envs=2048,
        env_spacing=2.0,
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
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**17
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**25
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.viewer.eye = (2.2, 2.2, 1.8)
        self.viewer.lookat = (0.45, 0.0, 0.80)
