"""Follow Sweeping-Policy/min's native RSL-RL inference flow for the graph policy.

Use the example's checkpoint/runner/inference sequence. Keep this entry point
because stock play unconditionally exports JIT/ONNX, which the graph actor
adapter does not implement. Playback itself uses the native inference policy.
"""

import argparse
from importlib.metadata import version
import os
from pathlib import Path
import sys
import time
import traceback

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", PACKAGE_ROOT.parents[1] / "IsaacLab")).expanduser().resolve()
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
parser.add_argument("--num_envs", type=int, default=None, help="Override the task's environment count (default: 1).")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_steps", type=int, default=None, help="Exit after this many control steps; otherwise play until closed.")
parser.add_argument("--real-time", action="store_true", help="Pace playback to simulation time when possible.")
parser.add_argument("--video", action="store_true", help="Record one video and exit after --video_length steps.")
parser.add_argument("--video_length", type=int, default=300)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if any(value is not None and value <= 0 for value in
       (args_cli.num_envs, args_cli.max_steps, args_cli.video_length)):
    parser.error("Environment, playback, and video step counts must be positive.")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import contact_manipulation  # Register the local task after AppLauncher.
from contact_manipulation.config.ur5e.agents.ppo import load_checkpoint, prepare_inference_checkpoint

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    """Load trained weights, infer deterministic actions, and let the env auto-reset."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    agent_cfg.device = env_cfg.sim.device
    if args_cli.headless and not args_cli.video:
        env_cfg.sim.render_interval = 2**31 - 1

    log_root = Path("logs/rsl_rl", agent_cfg.experiment_name).resolve()
    if args_cli.checkpoint:
        checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    else:
        checkpoint = Path(get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    saved_checkpoint = prepare_inference_checkpoint(agent_cfg, checkpoint)
    print(f"[INFO] Loading model checkpoint from: {checkpoint}", flush=True)
    env_cfg.log_dir = str(checkpoint.parent)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    try:
        if args_cli.video:
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(checkpoint.parent / "videos/play"),
                name_prefix=f"{checkpoint.stem}-{time.time_ns()}",
                step_trigger=lambda step: step == 0,
                video_length=args_cli.video_length, disable_logger=True,
            )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # Preserve strict model/F/T checks, but inference needs no optimizer state.
        load_checkpoint(runner, checkpoint, inference=True, checkpoint=saved_checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        if args_cli.video:
            # Initialize the RGB annotator before RecordVideo captures its first
            # frame. Render only: do not advance physics or the episode clock.
            for _ in range(3):
                env.unwrapped.render()
        dt = env.unwrapped.step_dt
        limit = args_cli.max_steps
        if args_cli.video:
            limit = args_cli.video_length if limit is None else min(limit, args_cli.video_length)
        steps = 0
        print(f"[INFO] Playing iteration {runner.current_learning_iteration} in {env.num_envs} environments.", flush=True)
        while simulation_app.is_running() and (limit is None or steps < limit):
            started = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
            steps += 1
            if args_cli.real_time:
                time.sleep(max(0.0, dt - (time.perf_counter() - started)))
        print(f"[INFO] Played {steps} control steps.", flush=True)
    finally:
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
