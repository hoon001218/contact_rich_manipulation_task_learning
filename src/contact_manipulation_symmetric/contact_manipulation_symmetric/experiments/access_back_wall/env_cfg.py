"""Learning 2: extend evaluation access region to the shelf rear wall."""

from isaaclab.utils import configclass

from ...config.ur5e.agents.rsl_rl_ppo_cfg import ContactManipulationPPORunnerCfg
from ..common import ExperimentEnvCfg


@configclass
class AccessBackWallEnvCfg(ExperimentEnvCfg):
    # Spawn rules, relocation scoring and action scales stay at the baseline.
    access_region_to_back_wall: bool = True


@configclass
class ExperimentPPORunnerCfg(ContactManipulationPPORunnerCfg):
    experiment_name = "contact_manipulation_symmetric_access_back_wall"
