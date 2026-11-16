"""Learning 1: contact-weighted metric relocation progress, including front corridors."""

from isaaclab.utils import configclass

from ... import mdp
from ...config.ur5e.agents.rsl_rl_ppo_cfg import ContactManipulationPPORunnerCfg
from ..common import ExperimentEnvCfg


@configclass
class RelocationDistanceEnvCfg(ExperimentEnvCfg):
    relocation_include_front_corridors: bool = True
    # Preserve the 2026-11-15 comparison's original access depth.
    access_region_to_back_wall: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.rewards.blocker_relocation.func = mdp.blocker_relocation_distance_progress
        # Reference distance = official TCP speed * actual policy step duration.
        # At the unchanged 50 Hz this is 4.0 m/s * 0.02 s = 0.08 m.
        self.rewards.blocker_relocation.params = {"reference_speed": 4.0, "threshold": 0.01}


@configclass
class ExperimentPPORunnerCfg(ContactManipulationPPORunnerCfg):
    experiment_name = "contact_manipulation_symmetric_relocation_distance"
