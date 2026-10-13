"""Shared shelf scene and MDP manager configuration, independent of robot choice."""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import (
    ActionTermCfg, EventTermCfg as EventTerm, ObservationGroupCfg, ObservationTermCfg,
    RewardTermCfg, TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp as task_mdp
from .mdp.events import reset_mdp, reset_spawn_scene
from .src.catalog import DEFAULT_SHELF_USD_PATH, SHELF_POSITION, ForceTorqueSensorCfg
from .src.sampling import RobotStartRules, SpawnRules
from .src.sensors import (
    InitialVisionCfg, TactileCfg, VisionReferencesCfg, make_initial_camera_cfg,
    tool_wrench, tool_wrench_valid, tool_wrench_range_exceeded,
    initial_rgb, initial_depth, initial_role_masks, initial_camera_pose,
    tactile_contacts, tactile_valid,
)


@configclass
class ShelfRearrangementSceneCfg(InteractiveSceneCfg):
    """Shared asset pool; reset provides seeded poses, active masks, and roles."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("CONTACT_REARRANGEMENT_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SHELF_POSITION),
    )
    robot: ArticulationCfg = MISSING
    objects: RigidObjectCollectionCfg = MISSING
    # Initial vision is captured once per episode. Reuse two render products
    # and a render-only reference pool instead of cloning RTX cameras per env.
    vision_references = AssetBaseCfg(prim_path="/World/VisionReferences", spawn=VisionReferencesCfg())
    initial_camera = make_initial_camera_cfg()
    silhouette_camera = make_initial_camera_cfg(reference=True)


@configclass
class ActionsCfg:
    """No external actions at the spawn stage; the robot holds its initial joint targets."""


@configclass
class ObservationsCfg:
    """Raw F/T, binary tactile contact, and the accepted initial RGB-D snapshot."""

    @configclass
    class SensorsCfg(ObservationGroupCfg):
        wrench = ObservationTermCfg(func=tool_wrench)
        wrench_valid = ObservationTermCfg(func=tool_wrench_valid)
        wrench_range_exceeded = ObservationTermCfg(func=tool_wrench_range_exceeded)
        tactile = ObservationTermCfg(func=tactile_contacts)
        tactile_valid = ObservationTermCfg(func=tactile_valid)
        rgb = ObservationTermCfg(func=initial_rgb)
        depth = ObservationTermCfg(func=initial_depth)
        role_masks = ObservationTermCfg(func=initial_role_masks)
        camera_pose = ObservationTermCfg(func=initial_camera_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    sensors = SensorsCfg()


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=reset_spawn_scene, mode="reset")


@configclass
class ShelfRearrangementSpawnEnvCfg(ManagerBasedEnvCfg):
    """Reusable manager-based scene with deterministic full and partial resets."""

    scene: ShelfRearrangementSceneCfg = ShelfRearrangementSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 0
    spawn_rules: SpawnRules = SpawnRules()
    robot_start: RobotStartRules = RobotStartRules()
    target_asset: str | None = None
    blocker_asset: str | None = None
    initial_vision: InitialVisionCfg = InitialVisionCfg()
    shared_initial_scene: bool = False
    tactile: TactileCfg = TactileCfg()
    force_torque: ForceTorqueSensorCfg = ForceTorqueSensorCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)


@configclass
class LearningActionsCfg:
    """Robot-specific control is supplied by config/<robot>/."""

    manipulate: ActionTermCfg = MISSING


@configclass
class LearningObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=task_mdp.policy_observation)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class LearningEventsCfg:
    # Ordered: accepted scene/vision first, then episode accumulators/history.
    reset_scene = EventTerm(func=reset_spawn_scene, mode="reset")
    reset_history = EventTerm(func=reset_mdp, mode="reset")


@configclass
class LearningRewardsCfg:
    # Each function returns the transition's component / step_dt. Isaac Lab
    # applies the configured weight and dt once, and logs each named term.
    clearance = RewardTermCfg(func=task_mdp.clearance, weight=1.0)
    rearrangement = RewardTermCfg(func=task_mdp.rearrangement, weight=1.0)
    motion = RewardTermCfg(func=task_mdp.motion, weight=1.0)
    time = RewardTermCfg(func=task_mdp.time_penalty, weight=1.0)
    terminal = RewardTermCfg(func=task_mdp.terminal, weight=1.0)


@configclass
class LearningTerminationsCfg:
    success = TerminationTermCfg(func=task_mdp.success)
    failure = TerminationTermCfg(func=task_mdp.failure)
    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)


@configclass
class ShelfRearrangementEnvCfg(ManagerBasedRLEnvCfg):
    """Initial vision, current/previous sensing, and 18D variable-stiffness manipulation."""

    scene: ShelfRearrangementSceneCfg = ShelfRearrangementSceneCfg(
        num_envs=4, env_spacing=2.5, replicate_physics=True
    )
    actions: LearningActionsCfg = LearningActionsCfg()
    observations: LearningObservationsCfg = LearningObservationsCfg()
    events: LearningEventsCfg = LearningEventsCfg()
    rewards: LearningRewardsCfg = LearningRewardsCfg()
    terminations: LearningTerminationsCfg = LearningTerminationsCfg()
    mdp: task_mdp.MdpCfg = task_mdp.MdpCfg()
    seed: int = 0
    spawn_rules: SpawnRules = SpawnRules()
    robot_start: RobotStartRules = RobotStartRules()
    target_asset: str | None = None
    blocker_asset: str | None = None
    initial_vision: InitialVisionCfg = InitialVisionCfg()
    shared_initial_scene: bool = True
    tactile: TactileCfg = TactileCfg()
    force_torque: ForceTorqueSensorCfg = ForceTorqueSensorCfg()
    forbidden_contact_threshold: float = 1.0

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.episode_length_s = 8.0
        self.num_rerenders_on_reset = 0  # reset_scene already captures accepted RGB-D
        # Shared reset/perception inspect only the USD template. Fabric keeps
        # independent physics replicas without composing the full USD tree N times.
        self.scene.clone_in_fabric = self.shared_initial_scene
        self.scene.initial_camera.render_shared_template_only = self.shared_initial_scene
        self.scene.silhouette_camera.render_shared_template_only = self.shared_initial_scene
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
