"""Register the UR5e task using Isaac Lab's native manager-based environment."""

import gymnasium as gym

TASK_ID = "Isaac-Contact-Manipulation-Symmetric-UR5e-v0"

gym.register(
    id=TASK_ID,
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur5e_env_cfg:UR5eContactManipulationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:ContactManipulationPPORunnerCfg",
    },
)

# String entry points preserve AppLauncher-before-simulation import ordering.
EXPERIMENT_TASKS = {
    "relocation_distance": (
        "Isaac-Contact-Manipulation-Symmetric-UR5e-RelocationDistance-v0", "RelocationDistanceEnvCfg",
    ),
    "access_back_wall": (
        "Isaac-Contact-Manipulation-Symmetric-UR5e-AccessBackWall-v0", "AccessBackWallEnvCfg",
    ),
    "small_wrench": (
        "Isaac-Contact-Manipulation-Symmetric-UR5e-SmallWrench-v0", "SmallWrenchEnvCfg",
    ),
}
for _variant, (_task_id, _env_cfg_name) in EXPERIMENT_TASKS.items():
    _module = f"contact_manipulation_symmetric.experiments.{_variant}.env_cfg"
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_module}:{_env_cfg_name}",
            "rsl_rl_cfg_entry_point": f"{_module}:ExperimentPPORunnerCfg",
        },
    )
