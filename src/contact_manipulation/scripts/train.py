"""Train with Isaac Lab's native RSL-RL flow and the research episode clock."""

import argparse
from datetime import datetime
from importlib.metadata import version
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
parser.add_argument("--task", default="Isaac-Contact-Manipulation-UR5e-v0")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import contact_manipulation  # Register the local task after AppLauncher.

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


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
            runner.load(resume_path, map_location=agent_cfg.device)
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        started = time.time()
        # Initial vision, previous-sample padding, and exponential time penalty
        # all begin at reset. Randomizing only native timeout counters breaks this.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
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
