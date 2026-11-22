"""Learning 2: extend evaluation access region to the shelf rear wall."""

from isaaclab.utils import configclass

from ... import mdp
from ...config.ur5e.agents.rsl_rl_ppo_cfg import ContactManipulationPPORunnerCfg
from ..common import ExperimentEnvCfg


@configclass
class AccessBackWallEnvCfg(ExperimentEnvCfg):
    # Preserve the 2026-11-15 comparison's relocation while the main task evolves.
    relocation_include_front_corridors: bool = False
    access_region_to_back_wall: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.rewards.blocker_relocation.func = mdp.blocker_relocation_contact_progress
        self.rewards.blocker_relocation.params = {"decay": 10.0, "stop_on_success": True}


@configclass
class ExperimentPPORunnerCfg(ContactManipulationPPORunnerCfg):
    experiment_name = "contact_manipulation_symmetric_access_back_wall"
