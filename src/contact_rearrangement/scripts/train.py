#!/usr/bin/env python3
"""Train the research task using Isaac Lab's native RSL-RL training flow."""

import argparse
from datetime import datetime
from pathlib import Path
import math
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
for package in ("isaaclab", "isaaclab_tasks", "isaaclab_rl"):
    sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/source" / package))
sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/scripts/reinforcement_learning/rsl_rl"))

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-Contact-Rearrangement-UR5e-v0")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", "--iterations", type=int, default=None)
parser.add_argument("--num_steps_per_env", "--rollout_steps", type=int, default=None)
parser.add_argument("--save_interval", "--checkpoint_interval", type=int, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--log_dir", type=Path, default=None)
parser.add_argument("--resume_checkpoint", type=Path, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--validate", action="store_true", help="Run separate research MDP and checkpoint checks.")
parser.add_argument("--disable_role_obb", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if any(value is not None and value <= 0 for value in
       (args_cli.num_envs, args_cli.max_iterations, args_cli.num_steps_per_env, args_cli.save_interval)):
    parser.error("Environment, iteration, rollout and checkpoint counts must be positive.")
if args_cli.episode_length_s is not None and (
    not math.isfinite(args_cli.episode_length_s) or args_cli.episode_length_s <= 0
):
    parser.error("episode_length_s must be finite and positive.")
args_cli.enable_cameras = True  # Initial RGB-D is part of this research task.
sys.argv = [sys.argv[0], *hydra_args]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import contact_rearrangement  # Register the task, as in the sweeping example.
from contact_rearrangement.src.visualization import RoleObbVisualizer


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Configure, create, wrap, and train with the native runner."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    for value, name in ((args_cli.max_iterations, "max_iterations"),
                        (args_cli.num_steps_per_env, "num_steps_per_env"),
                        (args_cli.save_interval, "save_interval")):
        if value is not None:
            setattr(agent_cfg, name, value)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device
    agent_cfg.device = args_cli.device
    if args_cli.headless and not args_cli.video:
        env_cfg.sim.render_interval = 2**30  # Initial snapshots render explicitly at reset.

    log_root = REPOSITORY_ROOT / "logs/rsl_rl" / agent_cfg.experiment_name
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"[INFO] Logging experiment in directory: {log_root}", flush=True)
    print(f"Exact experiment name requested from command line: {run_name}", flush=True)
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = args_cli.log_dir or log_root / run_name
    resume_path = args_cli.resume_checkpoint
    if resume_path is None and agent_cfg.resume:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = str(log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    runner = None
    visualizer = None
    try:
        validation = None
        if args_cli.validate:
            sys.path.insert(0, str(PACKAGE_ROOT / "tests"))
            from check_learning import LearningValidation
            validation = LearningValidation(env.unwrapped)
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
            runner.load(str(resume_path), map_location=agent_cfg.device)
        if env.unwrapped.sim.has_gui() and not args_cli.disable_role_obb:
            visualizer = RoleObbVisualizer(env.unwrapped)
            env.unwrapped.role_obb_visualizer = visualizer
            visualizer.update()
        dump_yaml(str(log_dir / "params/env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent_cfg)
        if validation is not None:
            validation.before_training(runner)
        started = time.time()
        # Keep initial vision, sensor history, and episode clock synchronized.
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
        print(f"Training time: {round(time.time() - started, 2)} seconds", flush=True)
        if validation is not None:
            validation.after_training(runner, agent_cfg.to_dict())
    finally:
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.flush()
            runner.logger.writer.close()
        if visualizer is not None:
            env.unwrapped.role_obb_visualizer = None
            visualizer.close()
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
