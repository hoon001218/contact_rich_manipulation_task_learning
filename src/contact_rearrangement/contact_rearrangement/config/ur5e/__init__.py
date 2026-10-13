"""Register the UR5e task without importing simulator-dependent configurations."""

import gymnasium as gym

TASK_ID = "Isaac-Contact-Rearrangement-UR5e-v0"

gym.register(
    id=TASK_ID,
    entry_point="contact_rearrangement.env:ContactRearrangementRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.osc_env_cfg:ContactRearrangementRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ContactRearrangementPPORunnerCfg",
    },
)
