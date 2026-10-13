"""Register the UR5e task using Isaac Lab's native manager-based environment."""

import gymnasium as gym

TASK_ID = "Isaac-Contact-Manipulation-UR5e-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_env_cfg:UR5eContactManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ContactManipulationPPORunnerCfg",
    },
)
