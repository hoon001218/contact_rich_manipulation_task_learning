"""Register the RH56 shelf-grasp environment."""

import gymnasium as gym

from .env_cfg import HandGraspEnvCfg
from .rsl_rl_ppo_cfg import HandGraspPPORunnerCfg


TASK_ID = "Isaac-Hand-RH56-Grasp-v0"

if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": HandGraspEnvCfg,
            "rsl_rl_cfg_entry_point": HandGraspPPORunnerCfg,
        },
    )

