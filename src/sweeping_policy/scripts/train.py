#!/usr/bin/env python3
"""Train the shelf-sweep baseline with Isaac Lab's stock RSL-RL runner."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import gymnasium as gym

TASK_ID = "Isaac-Shelf-Sweep-UR5e-v0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _has_task_argument(arguments: list[str]) -> bool:
    return any(
        argument == "--task" or argument.startswith("--task=") for argument in arguments
    )


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "sweeping_policy.shelf_sweep.env_cfg:ShelfSweepEnvCfg",
            "rsl_rl_cfg_entry_point": (
                "sweeping_policy.shelf_sweep.rsl_rl_ppo_cfg:" "ShelfSweepPPORunnerCfg"
            ),
        },
    )

if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

repository_root = Path(__file__).resolve().parents[3]
trainer = (
    repository_root
    / "IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py"
)

if not trainer.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL trainer not found: {trainer}")

sys.path.insert(0, str(trainer.parent))
runpy.run_path(str(trainer), run_name="__main__")
