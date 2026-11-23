"""Learning 3: smaller OSC scales, shared within force and torque groups."""

from isaaclab.utils import configclass

from ... import mdp
from ...config.ur5e.agents.rsl_rl_ppo_cfg import ContactManipulationPPORunnerCfg
from ..common import ExperimentEnvCfg


@configclass
class SmallWrenchEnvCfg(ExperimentEnvCfg):
    # Preserve the 2026-11-15 comparison's relocation while the main task evolves.
    relocation_include_front_corridors: bool = False
    access_region_to_back_wall: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.rewards.blocker_relocation.func = mdp.blocker_relocation_contact_progress
        self.rewards.blocker_relocation.params = {"decay": 10.0}
        self.actions.arm_action.wrench_scale = (0.1, 0.1, 0.1, 0.001, 0.001, 0.001)


@configclass
class ExperimentPPORunnerCfg(ContactManipulationPPORunnerCfg):
    experiment_name = "contact_manipulation_symmetric_small_wrench"
