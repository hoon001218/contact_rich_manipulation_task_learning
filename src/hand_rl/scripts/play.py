#!/usr/bin/env python3
"""Play a trained UR5e/force-sensor hand shelf-grasp policy."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import gymnasium as gym


TASK_ID = "Isaac-Hand-RH56-Grasp-v0"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def _has_task_argument(arguments: list[str]) -> bool:
    return any(argument == "--task" or argument.startswith("--task=") for argument in arguments)


if TASK_ID not in gym.registry:
    gym.register(
        id=TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "hand_rl.grasp.env_cfg:HandGraspEnvCfg",
            "rsl_rl_cfg_entry_point": "hand_rl.grasp.rsl_rl_ppo_cfg:HandGraspPPORunnerCfg",
        },
    )

if not _has_task_argument(sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))

repository_root = Path(__file__).resolve().parents[3]
stock_player = repository_root / "IsaacLab" / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
if not stock_player.is_file():
    raise FileNotFoundError(f"Isaac Lab RSL-RL player not found: {stock_player}")

sys.path.insert(0, str(stock_player.parent))
runpy.run_path(str(stock_player), run_name="__main__")
