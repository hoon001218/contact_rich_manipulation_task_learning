"""Scene and manager configuration for the shelf rearrangement environment."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm, ObservationGroupCfg, ObservationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import make_object_collection_cfg, make_ur5e_hand_cfg
from .catalog import DEFAULT_SHELF_USD_PATH, SHELF_POSITION
from .sampling import RobotStartRules, SpawnRules
from .spawn import reset_spawn_scene
from .sensors import (
    InitialVisionCfg, TactileCfg, VisionReferencesCfg, make_initial_camera_cfg, make_tactile_contact_cfg,
    tool_wrench, tool_wrench_valid, initial_rgb, initial_depth, initial_role_masks, initial_camera_pose,
    tactile_contacts, tactile_valid,
)


@configclass
class ContactRearrangementSceneCfg(InteractiveSceneCfg):
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
    robot = make_ur5e_hand_cfg()
    objects = make_object_collection_cfg()
    vision_references = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/VisionReferences", spawn=VisionReferencesCfg())
    initial_camera = make_initial_camera_cfg()
    silhouette_camera = make_initial_camera_cfg(reference=True)

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
class ActionsCfg:
    """No external actions at the spawn stage; the robot holds its initial joint targets."""


@configclass
class ObservationsCfg:
    """Raw F/T, binary tactile contact, and the accepted initial RGB-D snapshot."""

    @configclass
    class SensorsCfg(ObservationGroupCfg):
        wrench = ObservationTermCfg(func=tool_wrench)
        wrench_valid = ObservationTermCfg(func=tool_wrench_valid)
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
    robot_start: RobotStartRules = RobotStartRules()
    target_asset: str | None = None
    blocker_asset: str | None = None
    initial_vision: InitialVisionCfg = InitialVisionCfg()
    tactile: TactileCfg = TactileCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
