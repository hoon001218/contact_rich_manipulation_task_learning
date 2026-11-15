"""Learning 3: smaller OSC scales, shared within force and torque groups."""

from isaaclab.utils import configclass

from ...config.ur5e.agents.rsl_rl_ppo_cfg import ContactManipulationPPORunnerCfg
from ..common import ExperimentEnvCfg


@configclass
class SmallWrenchEnvCfg(ExperimentEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.wrench_scale = (0.1, 0.1, 0.1, 0.001, 0.001, 0.001)


@configclass
class ExperimentPPORunnerCfg(ContactManipulationPPORunnerCfg):
    experiment_name = "contact_manipulation_symmetric_small_wrench"
