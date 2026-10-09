"""Base environment for inspecting the scene before specifying the learning MDP."""

from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from .mdp import reset_spawn_scene
from .scene_cfg import ContactRearrangementSceneCfg
from .sampling import SpawnRules


@configclass
class ActionsCfg:
    """No external actions at the spawn stage; the robot holds its initial joint targets."""


@configclass
class ObservationsCfg:
    """The policy observation specification will be added in the next implementation stage."""


@configclass
class EventsCfg:
    reset_scene = EventTerm(func=reset_spawn_scene, mode="reset")


@configclass
class InitialVisionCfg:
    """Requirements for the future front-camera acceptance step; not yet measured."""

    min_target_visible_fraction: float = 0.40
    camera_side: str = "+X"
    silhouette_reference: str = "full_unoccluded_target_at_same_pose"
    require_full_target_in_frame: bool = True


@configclass
class ContactRearrangementEnvCfg(ManagerBasedEnvCfg):
    """Reusable manager-based scene with deterministic full and partial resets."""

    scene: ContactRearrangementSceneCfg = ContactRearrangementSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    seed: int = 0
    spawn_rules: SpawnRules = SpawnRules()
    target_asset: str | None = None
    blocker_asset: str | None = None
    initial_vision: InitialVisionCfg = InitialVisionCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
