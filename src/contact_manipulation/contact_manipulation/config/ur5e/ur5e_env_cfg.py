"""UR5e specialization following the native example's scene/config composition."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from ...shelf_manipulation_env_cfg import ShelfManipulationEnvCfg, ShelfManipulationSceneCfg
from ...mdp.sensors import (
    TaskFrameTransformer, make_forbidden_contact_cfg, make_tactile_contact_cfg,
)
from .scene import (
    DEFAULT_SHELF_USD_PATH, SHELF_POSITION, TACTILE_BODY_NAMES, FORBIDDEN_CONTACT_BODY_NAMES,
    make_object_collection_cfg, make_ur5e_hand_cfg, SharedGeometryCfg,
)


@configclass
class UR5eContactSceneCfg(ShelfManipulationSceneCfg):
    """One authored template and fixed object pool, without rendered perception."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(), collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    shelf = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.environ.get("CONTACT_MANIPULATION_SHELF_USD_PATH", DEFAULT_SHELF_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=SHELF_POSITION),
    )
    robot = make_ur5e_hand_cfg()
    object_collection = make_object_collection_cfg()
    ee_frame = FrameTransformerCfg(
        class_type=TaskFrameTransformer,
        prim_path="{ENV_REGEX_NS}/Robot/base_link", debug_vis=False,
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ROASRightHand/right_plam_force_sensor", name="end_effector",
        )],
    )
    # Assign one native sensor per physical tactile link; loops construct only
    # configuration attributes, never per-environment tensors or simulation data.
    for _name in TACTILE_BODY_NAMES:
        locals()[f"contact_{_name}"] = make_tactile_contact_cfg(_name)
    for _name in FORBIDDEN_CONTACT_BODY_NAMES:
        locals()[f"forbidden_{_name}"] = make_forbidden_contact_cfg(_name, history_length=4)
    del _name
    # Run last, after all collider/contact edits and before native cloning.
    shared_geometry = AssetBaseCfg(prim_path="/World/SharedGeometry", spawn=SharedGeometryCfg())


@configclass
class UR5eContactManipulationEnvCfg(ShelfManipulationEnvCfg):
    """Torque OSC arm, independent hand PD and geometric initial perception."""

    scene: UR5eContactSceneCfg = UR5eContactSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False,
    )

    def __post_init__(self):
        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.05
        self.episode_length_s = 8.0
        self.num_rerenders_on_reset = 0  # Initial perception does not render.
        self.viewer.eye = (1.6, -2.0, 2.0)
        self.viewer.lookat = (-0.5, 0.0, 1.05)
