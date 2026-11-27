"""Shared configuration for the three earlier reward/control experiments."""

from isaaclab.utils import configclass

from ..config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg


@configclass
class ExperimentEnvCfg(UR5eContactManipulationEnvCfg):
    """Keep the baseline managers, but omit the terminal reward contribution."""

    def __post_init__(self):
        super().__post_init__()
        # Success/failure/time-out termination conditions remain inherited.
        self.rewards.terminal = None
        # Match these experiments' historical linear contact weighting.
        self.rewards.clearance.params["contact_exponent"] = 1.0
