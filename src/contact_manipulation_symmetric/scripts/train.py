"""Follow the example's AppLauncher/Hydra/env/wrapper/native runner flow.

Use its randomized initial episode counters; scene reset still initializes the
research task's vision and sensor history.
"""

import argparse
from datetime import datetime
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PACKAGE_ROOT.parents[1] / "IsaacLab")).expanduser().resolve()
# Use one selected checkout for AppLauncher, task utilities, wrappers, and CLI.
sys.path[:0] = [str(PACKAGE_ROOT), *(str(ISAACLAB_ROOT / "source" / name)
                  for name in ("isaaclab", "isaaclab_tasks", "isaaclab_rl")),
               str(ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl")]
if not (ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl/cli_args.py").is_file():
    raise FileNotFoundError(f"Isaac Lab checkout not found: {ISAACLAB_ROOT}. Set ISAACLAB_PATH.")

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Contact-Manipulation-Symmetric-UR5e-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--export_io_descriptors", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if any(value is not None and value <= 0 for value in
       (args_cli.num_envs, args_cli.max_iterations, args_cli.video_length, args_cli.video_interval)):
    parser.error("Environment, iteration, and video step counts must be positive.")
installed_version = version("rsl-rl-lib")
if int(installed_version.split(".")[0]) < 5:
    parser.error("This project's separate graph actor/critic requires RSL-RL >= 5.0.")
if args_cli.video:
    args_cli.enable_cameras = True  # Recording only; policy perception does not render.
sys.argv = [sys.argv[0], *hydra_args]
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab.envs.mdp import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import contact_manipulation_symmetric  # Register the local task after AppLauncher.
from contact_manipulation_symmetric.config.ur5e.agents.ppo import load_checkpoint
from contact_manipulation_symmetric.mdp.events import SensorMonitoringLogger, TensorboardSensorMonitor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def prepare_time_penalty_curriculum(env_cfg, agent_cfg):
    """Arm the official reward schedule; return its absolute starting iteration."""
    fraction, reward = env_cfg.time_penalty_start_fraction, env_cfg.rewards.time_penalty
    env_cfg.curriculum.time_penalty = None
    if fraction is None or reward is None:
        return None
    if (isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0):
        raise ValueError("time_penalty_start_fraction must be in [0, 1] or None.")
    for name in ("max_iterations", "num_steps_per_env"):
        value = getattr(agent_cfg, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not math.isfinite(reward.weight):
        raise ValueError("Time penalty weight must be finite.")
    iteration = math.ceil(agent_cfg.max_iterations * fraction)
    env_cfg.curriculum.time_penalty = CurriculumTermCfg(
        func=modify_reward_weight,
        params={"term_name": "time_penalty", "weight": reward.weight,
                "num_steps": iteration * agent_cfg.num_steps_per_env - 1},
    )
    reward.weight = 0.0
    return iteration


def sync_time_penalty_curriculum(env, env_cfg, *, completed_iterations=0, num_steps_per_env):
    """Apply the launch/checkpoint phase once without changing environment clocks."""
    term = env_cfg.curriculum.time_penalty
    if term is None:
        return
    term.params["num_steps"] -= completed_iterations * num_steps_per_env
    manager = env.unwrapped.curriculum_manager
    manager.cfg.time_penalty.params.update(term.params)
    manager.compute()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Configure, create, wrap, and train with the standard managers and runner."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = agent_cfg.device = args_cli.device
    env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    if args_cli.headless and not args_cli.video:
        # Reset perception uses geometry queries and needs no renderer updates.
        env_cfg.sim.render_interval = 2**31 - 1
    time_penalty_start_iteration = prepare_time_penalty_curriculum(env_cfg, agent_cfg)

    log_root = Path("logs/rsl_rl", agent_cfg.experiment_name).resolve()
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Logging experiment in directory: {log_root}", flush=True)
    print(f"Exact experiment name requested from command line: {run_name}", flush=True)
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    resume_path = (get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
                   if agent_cfg.resume else None)
    env_cfg.log_dir = str(log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    runner = None
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(log_dir / "videos/train"),
                step_trigger=lambda step: step % args_cli.video_interval == 0,
                video_length=args_cli.video_length, disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if resume_path is not None:
            print(f"[INFO]: Loading model checkpoint from: {resume_path}", flush=True)
            load_checkpoint(runner, resume_path)
        # Native checkpoint labels are zero-based: iter=0 follows one rollout.
        completed = int(runner.current_learning_iteration) + 1 if resume_path is not None else 0
        sync_time_penalty_curriculum(
            env, env_cfg, completed_iterations=completed, num_steps_per_env=agent_cfg.num_steps_per_env,
        )
        if env_cfg.tensorboard_sensor_monitor and not runner.logger.disable_logs:
            monitor = TensorboardSensorMonitor(env.unwrapped, agent_cfg.num_steps_per_env)
            env.unwrapped._tensorboard_sensor_monitor = monitor
            runner.logger = SensorMonitoringLogger(runner.logger, monitor)
        if time_penalty_start_iteration is not None:
            target_weight = env_cfg.curriculum.time_penalty.params["weight"]
            print(f"[INFO] Time penalty: 0 -> {target_weight} at iteration {time_penalty_start_iteration}"
                  f" (completed rollouts: {completed}; applied at reset).", flush=True)
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        started = time.time()
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        print(f"Training time: {time.time() - started:.2f} seconds", flush=True)
    finally:
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        simulation_app.close()
