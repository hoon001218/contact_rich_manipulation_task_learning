"""Common scene, MDP terms and simulation settings, following Sweeping-Policy/min."""

from dataclasses import MISSING
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import is_alive, joint_vel_out_of_manual_limit
from isaaclab.envs.ui import ManagerBasedRLEnvWindow
from isaaclab.managers import ActionTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp
from .mdp.events import MdpCfg, SpawnRules, RobotStartRules
from .mdp.sensors import BlockerContactHold, InitialVisionCfg


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
    """Separate 244D actor/critic groups with identical history and tracked IDs."""

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
    class CriticCfg(ObservationGroupCfg):
        """Current true OBBs for initial vision slots, followed by the same history."""

        current_scene = ObservationTermCfg(func=mdp.critic_scene_geometry)
        current = ObservationTermCfg(func=mdp.current_robot_sample)
        previous = ObservationTermCfg(func=mdp.previous_robot_sample)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Scene reset must finish before the MDP snapshot/accumulators are reset."""

    role_obb_visualization = EventTermCfg(func=mdp.initialize_role_obb_visualization, mode="startup")
    reset_scene = EventTermCfg(func=mdp.reset_scene, mode="reset")
    reset_mdp = EventTermCfg(func=mdp.reset_task, mode="reset")


@configclass
class RewardsCfg:
    """Raw task scores/costs, integrated by the native RewardManager."""

    clearance = RewardTermCfg(func=mdp.clearance, weight=12.0)
    blocker_distance = RewardTermCfg(
        func=mdp.blocker_distance, weight=3.0, params={"decay": 10.0},
    )
    blocker_contact_hold = RewardTermCfg(func=BlockerContactHold, weight=0.2)
    shelf_collision = RewardTermCfg(
        func=mdp.shelf_collision, weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("shelf"),
            "translation_threshold": 0.005,
            "rotation_threshold": math.pi / 180.0,
            "linear_velocity_threshold": 0.005,
            "angular_velocity_threshold": 0.05,
        },
    )
    object_collision = RewardTermCfg(func=mdp.object_collision, weight=-0.5)
    # Command variation and actual speed are different control objectives.
    motion_action_rate = RewardTermCfg(func=mdp.motion_action_rate, weight=-0.03)
    motion_joint_velocity = RewardTermCfg(
        func=mdp.motion_joint_velocity, weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )
    # Separate SI costs: +1 N costs 0.03 reward/s; +0.05 N*m costs 0.000075 reward/s.
    # Per-step norm increases are squared directly, without rate conversion or exp.
    motion_force_growth = RewardTermCfg(func=mdp.motion_force_growth, weight=-0.03)
    motion_torque_growth = RewardTermCfg(func=mdp.motion_torque_growth, weight=-0.03)
    motion_singularity = RewardTermCfg(func=mdp.motion_singularity, weight=0.30)
    object_stability = RewardTermCfg(func=mdp.object_stability, weight=1.0)
    # Native alive cost; terminal success/failure steps are exempt.
    time_penalty = RewardTermCfg(func=is_alive, weight=-1.0)
    terminal = RewardTermCfg(func=mdp.terminal, weight=1.0)


@configclass
class TerminationsCfg:
    """Native cause-specific termination logs over the cached task predicates."""

    time_out = TerminationTermCfg(func=mdp.scene_time_out, time_out=True)
    success = TerminationTermCfg(func=mdp.success)
    object_support_lost = TerminationTermCfg(func=mdp.failure, params={"reason": "object_support_lost"})
    object_tilt = TerminationTermCfg(func=mdp.failure, params={"reason": "object_tilt"})
    shelf_displacement = TerminationTermCfg(func=mdp.failure, params={"reason": "shelf_displacement"})
    invalid_state = TerminationTermCfg(func=mdp.failure, params={"reason": "invalid_state"})
    singularity = TerminationTermCfg(func=mdp.failure, params={"reason": "singularity"})
    # Match min's Inspire threshold; the UR5e config selects only the six arm joints.
    hand_velocity = TerminationTermCfg(
        func=joint_vel_out_of_manual_limit,
        params={"max_velocity": 3.0, "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


class ShelfManipulationEnvWindow(ManagerBasedRLEnvWindow):
    """Extend the native debug panel with OBB toggles and sensor monitoring."""

    def __init__(self, env, window_name="IsaacLab"):
        import omni.ui as ui

        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("Target / Blocker OBB", env._role_obb_visualizer)
                    self._create_debug_vis_ui_element("Observed objects OBB", env._observation_obb_visualizer)
                    self._create_debug_vis_ui_element("Tactile / F/T monitor", env._sensor_monitor)
                    frame = ui.Frame(width=ui.Fraction(1), height=0, visible=env._sensor_monitor.debug_vis)
                    self.ui_window_elements["sensor_monitor_panel"] = frame
                    env._sensor_monitor.set_vis_frame(frame)
                    env._sensor_monitor.set_env_selection(env.cfg.viewer.env_index)
                    self._ui_listeners.append(env._sensor_monitor)


@configclass
class ShelfManipulationEnvCfg(ManagerBasedRLEnvCfg):
    """Compose native managers without overriding the environment's step/reset loop."""

    seed: int = 0
    ui_window_class_type: type | None = ShelfManipulationEnvWindow
    scene: ShelfManipulationSceneCfg = ShelfManipulationSceneCfg(num_envs=1, env_spacing=MISSING)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    mdp: MdpCfg = MdpCfg()
    spawn_rules: SpawnRules = SpawnRules(
        generation_mode="clutter", min_objects=15,
        max_objects=int(os.environ.get("CONTACT_MANIPULATION_PARALLEL_NUM_OBJECTS", "25")),
    )
    robot_start: RobotStartRules = RobotStartRules()
    initial_vision: InitialVisionCfg = InitialVisionCfg()
    # Optional cross-PC comparison: require the transferred catalog instead of rebuilding it.
    initial_catalog_fingerprint: str | None = None
    # Reset-only swaps of neighboring image-ordered distractors; roles stay 0/1.
    observation_adjacent_swap_probability: float = 0.2
    target_asset: str | None = None
    blocker_asset: str | None = None
    role_obb_debug_vis: bool = True
    observation_obb_debug_vis: bool = True
    sensor_monitor_debug_vis: bool = False

    decimation: int = 1
    episode_length_s: float = 8.0

    def __post_init__(self):
        # Policy, physics and OSC all run at 100 Hz.
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        # Native solver setting measured with 1024 environments; retain dt and
        # solver iterations while reducing partition/launch overhead.
        self.sim.physx.gpu_max_num_partitions = 1
        # Whole-scene contact patches exceeded the native 163840 default.
        # Keep margin above the observed 184896-patch peak.
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        self.num_rerenders_on_reset = 0  # Initial perception does not render.
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
