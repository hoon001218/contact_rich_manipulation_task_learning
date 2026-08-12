"""Register the Nucleus-backed shelf-sweep scene baseline."""

import gymnasium as gym

from .env_cfg import ShelfSweepEnvCfg
from .rsl_rl_ppo_cfg import ShelfSweepPPORunnerCfg


TASK_ID = "Isaac-Shelf-Sweep-UR5e-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": ShelfSweepEnvCfg,
            "rsl_rl_cfg_entry_point": ShelfSweepPPORunnerCfg,
        },
    )
