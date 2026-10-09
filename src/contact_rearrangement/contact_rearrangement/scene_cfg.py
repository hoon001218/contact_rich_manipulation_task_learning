"""Robot, fixed shelf, and a reusable pool of authored physics objects."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .assets import make_ur5e_hand_cfg
from .layout import DEFAULT_SHELF_USD_PATH, OBJECT_SPECS, SHELF_POSITION, parking_position
from .object_pool import TaskObjectPool
from .object_spawner import FlatBoxUsdCfg, YCB_BOX_ASSETS


def make_object_collection_cfg() -> RigidObjectCollectionCfg:
    """Load the catalog once; reset activates a sampled subset on the working shelf."""
    return RigidObjectCollectionCfg(
        class_type=TaskObjectPool,
        rigid_objects={
            spec.name: RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                spawn=(FlatBoxUsdCfg if spec.asset_name in YCB_BOX_ASSETS else sim_utils.UsdFileCfg)(
                    usd_path=spec.usd_path,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=False,
                        kinematic_enabled=False,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=2,
                        max_depenetration_velocity=1.0,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=parking_position(index),
                    rot=spec.upright_quat,
                    lin_vel=(0.0, 0.0, 0.0),
                    ang_vel=(0.0, 0.0, 0.0),
                ),
            )
            for index, spec in enumerate(OBJECT_SPECS)
        }
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
