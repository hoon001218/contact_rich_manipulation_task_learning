#!/usr/bin/env python3
"""Train the JH UR5e variable-stiffness OSC sweep policy with RSL-RL."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import gymnasium as gym


TASK_ID = "Isaac-Sweep-JH-v0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Keep the launcher usable directly from a source checkout, even when the
# package has not yet been installed into the Isaac Lab Python environment.
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _has_task_argument(arguments: list[str]) -> bool:
    return any(argument == "--task" or argument.startswith("--task=") for argument in arguments)


# Register without importing Isaac Lab configuration modules before AppLauncher.
if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "sweep_jh.osc_sweep.env_cfg:JHSweepEnvCfg",
            "rsl_rl_cfg_entry_point": "sweep_jh.osc_sweep.rsl_rl_ppo_cfg:JHSweepPPORunnerCfg",
        },
    )

if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

repository_root = Path(__file__).resolve().parents[3]
stock_trainer = repository_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
if not stock_trainer.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL trainer not found: {stock_trainer}")

sys.path.insert(0, str(stock_trainer.parent))
runpy.run_path(str(stock_trainer), run_name="__main__")
