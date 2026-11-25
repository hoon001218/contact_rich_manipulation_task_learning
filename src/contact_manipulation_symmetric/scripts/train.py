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
from contact_manipulation_symmetric.mdp.curriculums import (
    object_count_range, scene_layout_probabilities, scene_object_count_curriculum,
)
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


def checkpoint_completed_iterations(path):
    """Read the native clock before the initial reset, without allocating on GPU."""
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    iteration = checkpoint.get("iter")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("Resume checkpoint must contain a nonnegative native 'iter' integer.")
    return iteration + 1


def prepare_scene_object_curriculum(env_cfg, agent_cfg, *, completed_iterations=0):
    """Configure the native curriculum before any episode is generated."""
    env_cfg.curriculum.scene_object_count = None
    env_cfg.spawn_object_count_range = None
    start = env_cfg.scene_object_curriculum_start_iteration
    if start is None:
        return None
    end = env_cfg.scene_object_curriculum_end_iteration
    rollout = agent_cfg.num_steps_per_env
    for name, value, minimum in (("scene_object_curriculum_start_iteration", start, 0),
                                 ("scene_object_curriculum_end_iteration", end, 1),
                                 ("num_steps_per_env", rollout, 1),
                                 ("completed_iterations", completed_iterations, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if end <= start:
        raise ValueError("scene_object_curriculum_end_iteration must exceed its start iteration.")
    if env_cfg.spawn_rules.generation_mode != "sequential":
        raise ValueError("The scene object-count curriculum requires sequential generation.")
    params = dict(start_step=start * rollout, end_step=end * rollout,
                  final_min=env_cfg.spawn_rules.min_objects, final_max=env_cfg.spawn_rules.max_objects,
                  resume_steps=completed_iterations * rollout)
    env_cfg.spawn_object_count_range = object_count_range(
        params["resume_steps"], start_step=params["start_step"], end_step=params["end_step"],
        final_min=params["final_min"], final_max=params["final_max"],
    )
    env_cfg.curriculum.scene_object_count = CurriculumTermCfg(
        func=scene_object_count_curriculum, params=params,
    )
    return start, end


def prepare_scene_layout_curriculum(env_cfg, agent_cfg, *, completed_iterations=0):
    """Initialize the shared-scene schedule before the first native reset."""
    env_cfg.layout_curriculum_start_step = None
    env_cfg.layout_curriculum_end_step = None
    env_cfg.layout_curriculum_resume_steps = 0
    start = env_cfg.layout_curriculum_start_iteration
    if start is None or env_cfg.layout_scene_override is not None:
        return None
    end = env_cfg.layout_curriculum_end_iteration
    steps_per_iteration = agent_cfg.num_steps_per_env
    for name, value, minimum in (("layout_curriculum_start_iteration", start, 0),
                                 ("layout_curriculum_end_iteration", end, 1),
                                 ("num_steps_per_env", steps_per_iteration, 1),
                                 ("completed_iterations", completed_iterations, 0)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if env_cfg.spawn_rules.generation_mode != "sequential":
        raise ValueError("The layout curriculum requires the sequential scene generator.")
    start_step, end_step = start * steps_per_iteration, end * steps_per_iteration
    resume_steps = completed_iterations * steps_per_iteration
    scene_layout_probabilities(
        resume_steps, start_step=start_step, end_step=end_step,
        start_weights=env_cfg.layout_curriculum_start_weights, end_weights=env_cfg.layout_curriculum_end_weights,
    )
    env_cfg.layout_curriculum_start_step = start_step
    env_cfg.layout_curriculum_end_step = end_step
    env_cfg.layout_curriculum_resume_steps = resume_steps
    return start, end


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
    resume_completed = checkpoint_completed_iterations(resume_path)
    layout_phase = prepare_scene_layout_curriculum(
        env_cfg, agent_cfg, completed_iterations=resume_completed,
    )
    object_count_phase = None
    if layout_phase is None and env_cfg.layout_scene_override is None:
        object_count_phase = prepare_scene_object_curriculum(
            env_cfg, agent_cfg, completed_iterations=resume_completed,
        )
    else:
        env_cfg.curriculum.scene_object_count = None
        env_cfg.spawn_object_count_range = None
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
        if completed != resume_completed:
            raise RuntimeError("Loaded checkpoint clock differs from the scene curriculum's initial clock.")
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
        if object_count_phase is not None:
            start, end = object_count_phase
            print(f"[INFO] Scene object curriculum: Target + Blocker only through iteration {start}; "
                  f"grow to {env_cfg.spawn_rules.min_objects}..{env_cfg.spawn_rules.max_objects} "
                  f"by iteration {end} (completed rollouts: {completed}; new shared scenes only).", flush=True)
        if layout_phase is not None:
            start, end = layout_phase
            print(f"[INFO] Scene layout curriculum (easy, medium, hard): "
                  f"{env_cfg.layout_curriculum_start_weights} at iteration {start} -> "
                  f"{env_cfg.layout_curriculum_end_weights} at iteration {end} "
                  f"(completed rollouts: {completed}; new shared scenes only).", flush=True)
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
