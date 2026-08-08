"""Register the single JH UR5e OSC sweep environment."""

import gymnasium as gym

from .env_cfg import JHSweepEnvCfg
from .rsl_rl_ppo_cfg import JHSweepPPORunnerCfg

TASK_ID = "Isaac-Sweep-JH-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": JHSweepEnvCfg,
            "rsl_rl_cfg_entry_point": JHSweepPPORunnerCfg,
        },
    )
