"""Common scene, MDP terms and simulation settings, following Sweeping-Policy/min."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.events import MdpCfg, SpawnRules, RobotStartRules
from .mdp.sensors import InitialVisionCfg


@configclass
class ShelfManipulationSceneCfg(InteractiveSceneCfg):
    """Common world and required assets supplied by the robot-specific config."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    robot: ArticulationCfg = MISSING
    shelf: RigidObjectCfg = MISSING
    object_collection: RigidObjectCollectionCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING


@configclass
class ActionsCfg:
    """Robot-specific action terms, supplied by the UR5e config like the examples."""

    arm_action: ActionTermCfg = MISSING
    hand_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    """Initial geometry and current/previous sensor samples: 244D, without masks/role IDs."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Ordered terms: initial scene 100D, current sample 72D, previous sample 72D."""

        initial_scene = ObservationTermCfg(func=mdp.initial_scene_geometry)
        current = ObservationTermCfg(func=mdp.current_robot_sample)
        previous = ObservationTermCfg(func=mdp.previous_robot_sample)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Scene reset must finish before the MDP snapshot/accumulators are reset."""

    reset_scene = EventTermCfg(func=mdp.reset_scene, mode="reset")
    reset_mdp = EventTermCfg(func=mdp.reset_task, mode="reset")


@configclass
class RewardsCfg:
    """Exponential research quality scores and unchanged Time/Terminal terms."""

    clearance = RewardTermCfg(func=mdp.clearance, weight=1.0)
    rearrangement = RewardTermCfg(func=mdp.rearrangement, weight=1.0)
    motion = RewardTermCfg(
        func=mdp.motion, weight=1.0,
        params={
            "arm_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "hand_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )
    time = RewardTermCfg(func=mdp.time_penalty, weight=1.0)
    terminal = RewardTermCfg(func=mdp.terminal, weight=1.0)


@configclass
class TerminationsCfg:
    """Shared-scene timeout and research success/failure conditions."""

    time_out = TerminationTermCfg(func=mdp.scene_time_out, time_out=True)
    success = TerminationTermCfg(func=mdp.success)
    failure = TerminationTermCfg(func=mdp.failure)


@configclass
class ShelfManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Compose native managers without overriding the environment's step/reset loop."""

    seed: int = 0
    scene: ShelfManipulationSceneCfg = ShelfManipulationSceneCfg(num_envs=1, env_spacing=MISSING)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    mdp: MdpCfg = MdpCfg()
    spawn_rules: SpawnRules = SpawnRules()
    robot_start: RobotStartRules = RobotStartRules()
    initial_vision: InitialVisionCfg = InitialVisionCfg()
    target_asset: str | None = None
    blocker_asset: str | None = None

    decimation: int = 2
    episode_length_s: float = 8.0

    def __post_init__(self):
        # Common timing matches Sweeping-Policy: 100 Hz physics / 50 Hz policy.
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        # Native solver setting measured with 1024 environments; retain dt and
        # solver iterations while reducing partition/launch overhead.
        self.sim.physx.gpu_max_num_partitions = 1
        self.num_rerenders_on_reset = 0  # Initial perception does not render.
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
