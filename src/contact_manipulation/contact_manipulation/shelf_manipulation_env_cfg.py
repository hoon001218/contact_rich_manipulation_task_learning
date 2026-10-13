"""Native manager configuration; UR5e specialization supplies the static scene assets."""

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.events import MdpCfg, SpawnRules, RobotStartRules
from .mdp.sensors import InitialVisionCfg


@configclass
class ShelfManipulationSceneCfg(InteractiveSceneCfg):
    """Required scene assets, populated by config/ur5e/ur5e_env_cfg.py."""

    robot: ArticulationCfg = MISSING
    shelf: AssetBaseCfg = MISSING
    object_collection: RigidObjectCollectionCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING


@configclass
class ActionsCfg:
    """One 18D action: TCP delta6, stiffness6, independent hand joints6."""

    manipulation = mdp.ContactManipulationActionCfg()


@configclass
class ObservationsCfg:
    """Initial geometry and current/previous sensor samples: 544D, without masks/role IDs."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Internal packing fixes the agreed slot/channel order."""

        measurements = ObservationTermCfg(func=mdp.policy_observation)

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
    """Terms return rates so native weight*dt preserves the discrete formulation."""

    clearance = RewardTermCfg(func=mdp.clearance, weight=1.0)
    rearrangement = RewardTermCfg(func=mdp.rearrangement, weight=1.0)
    motion = RewardTermCfg(func=mdp.motion, weight=1.0)
    time = RewardTermCfg(func=mdp.time_penalty, weight=1.0)
    terminal = RewardTermCfg(func=mdp.terminal, weight=1.0)


@configclass
class TerminationsCfg:
    """Native timeout and research success/failure conditions."""

    time_out = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
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

    decimation: int = MISSING
    episode_length_s: float = MISSING
