"""Register the ROAS-hand shelf-sweeping environment."""

import gymnasium as gym

from .env_cfg import RoasShelfSweepEnvCfg
from .rsl_rl_ppo_cfg import RoasShelfSweepPPORunnerCfg


TASK_ID = "Isaac-Shelf-Sweep-UR5e-ROAS-Hand-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": RoasShelfSweepEnvCfg,
            "rsl_rl_cfg_entry_point": RoasShelfSweepPPORunnerCfg,
        },
    )
