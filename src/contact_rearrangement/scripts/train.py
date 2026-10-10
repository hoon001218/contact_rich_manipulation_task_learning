#!/usr/bin/env python3
"""Train the initial-scene graph with current/previous-step sensing and PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/source/isaaclab"))
sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/source/isaaclab_tasks"))

from isaaclab.app import AppLauncher
from contact_rearrangement.config.ur5e import TASK_ID

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default=TASK_ID, help="Registered Gym task providing environment and PPO defaults.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--iterations", type=int, default=None)
parser.add_argument("--rollout_steps", type=int, default=None)
parser.add_argument("--episode_length_s", type=float, default=None, help="Provisional training time limit.")
parser.add_argument("--log_dir", type=Path, default=None)
parser.add_argument("--resume", type=Path, default=None)
parser.add_argument("--checkpoint_interval", type=int, default=None)
parser.add_argument("--validate", action="store_true", help="Check physics, frames, episode isolation, and optimizer updates.")
parser.add_argument("--disable_role_obb", action="store_true", help="Hide viewport Target/Blocker outlines.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if any(value is not None and value <= 0 for value in
       (args.num_envs, args.iterations, args.rollout_steps, args.checkpoint_interval)):
    parser.error("Environment, iteration, rollout, and checkpoint counts must be positive.")
if args.episode_length_s is not None and not 0 < args.episode_length_s < float("inf"):
    parser.error("episode_length_s must be finite and positive.")
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab.utils.math import combine_frame_transforms

from contact_rearrangement.src.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from contact_rearrangement.config.ur5e.agents.ppo import PPOTrainer
from contact_rearrangement.config.ur5e.agents.model import ContactRearrangementPolicy
from contact_rearrangement.src.visualization import RoleObbVisualizer


def validate_transition(env, before_geometry, observation, reward, terminated, truncated, extra, action,
                        before_history):
    """Check completed transitions before/after the SDK's selective auto-reset."""
    state = env.mdp_state
    object_width = env.cfg.mdp.max_objects * 13
    history_steps = env.cfg.mdp.history_length
    expected_width = object_width + history_steps * 90
    if observation.shape != (env.num_envs, expected_width) or not torch.isfinite(observation).all():
        raise AssertionError(f"Policy observation must be finite [N,{expected_width}].")
    if history_steps != 2:
        raise AssertionError("Policy sensing must contain exactly current and previous samples.")
    done = terminated | truncated
    continuing = ~done
    torch.testing.assert_close(env.initial_object_geometry[continuing], before_geometry[continuing], rtol=0, atol=0)
    torch.testing.assert_close(extra["terminal_observation"][continuing], observation[continuing], rtol=0, atol=0)
    weights = reward.new_tensor((env.cfg.mdp.w_clear, env.cfg.mdp.w_rearr,
                                env.cfg.mdp.w_motion, env.cfg.mdp.w_time))
    expected = extra["reward_components"][:, :4] @ weights + extra["reward_components"][:, 4]
    torch.testing.assert_close(reward, expected, rtol=1e-5, atol=1e-5)
    if done.any():
        if not (state.history_valid[done].sum(-1) == 1).all():
            raise AssertionError("Reset must leave exactly one fresh history sample.")
        torch.testing.assert_close(state.history[done, 0, -18:], torch.zeros_like(action[done]), rtol=0, atol=0)
        torch.testing.assert_close(state.history[done, 1], torch.zeros_like(state.history[done, 1]), rtol=0, atol=0)
        if state.history_valid[done, 1].any():
            raise AssertionError("Reset previous sample must be invalid.")
        if not (extra["terminal_observation"][done, -history_steps:].sum(-1) == 2).all():
            raise AssertionError("PPO requires the completed episode's final history, before reset.")
    if continuing.any():
        torch.testing.assert_close(state.history[continuing, 0, -18:], action[continuing], rtol=0, atol=0)
    final_history = extra["terminal_observation"][:, object_width:-history_steps].reshape(env.num_envs, 2, 89)
    torch.testing.assert_close(final_history[:, 0, -18:], action, rtol=0, atol=0)
    torch.testing.assert_close(final_history[:, 1], before_history[:, 0], rtol=0, atol=0)
    robot = env.scene["robot"]
    arm = [robot.joint_names.index(name) for name in ARM_JOINT_NAMES]
    hand = [robot.joint_names.index(name) for name in HAND_JOINT_NAMES]
    mimic = [i for i, name in enumerate(robot.joint_names) if i not in arm + hand]
    for indices in (arm, mimic):
        torch.testing.assert_close(robot.data.joint_stiffness[:, indices], torch.zeros_like(robot.data.joint_stiffness[:, indices]))
        torch.testing.assert_close(robot.data.joint_damping[:, indices], torch.zeros_like(robot.data.joint_damping[:, indices]))
    if not torch.isfinite(robot.data.joint_pos).all():
        raise AssertionError("Native articulation state is nonfinite.")
    term = env.action_manager.get_term("manipulate")
    if not torch.isfinite(term.joint_efforts).all():
        raise AssertionError("OSC effort is nonfinite.")
    # Independently reconstruct the base-frame initial scene from actor EEF
    # geometry using the SDK transform utility used in examples.
    geometry = observation[:, :400].reshape(env.num_envs, 40, 10)
    tcp = term.measured_tcp_pose_b()[:, None].expand(-1, 40, -1)
    position, rotation = combine_frame_transforms(tcp[..., :3].reshape(-1, 3), tcp[..., 3:].reshape(-1, 4),
                                                 geometry[..., :3].reshape(-1, 3), geometry[..., 3:7].reshape(-1, 4))
    valid = env.initial_object_valid
    expected = env.initial_object_geometry
    torch.testing.assert_close(position.reshape(env.num_envs, 40, 3)[valid], expected[..., :3][valid], atol=1e-6, rtol=0)
    alignment = (rotation.reshape(env.num_envs, 40, 4)[valid] * expected[..., 3:7][valid]).sum(-1).abs()
    torch.testing.assert_close(alignment, torch.ones_like(alignment), atol=1e-6, rtol=0)
    torch.testing.assert_close(geometry[..., 7:][valid], expected[..., 7:][valid], atol=0, rtol=0)


def validate_forbidden_contact(env):
    """Place a real pool object into the proximal spacer and verify its reporter.

    This physics fixture uses simulator geometry only to set up a collision.
    The scene's frozen initial perception is left untouched. The normal SDK
    termination/reset path must retain the physical contact failure in extras.
    """
    from contact_rearrangement.src.assets import FORCE_TORQUE_PROXIMAL_BODY_NAME
    from contact_rearrangement.mdp import quat_apply, quat_mul, quat_conjugate
    from contact_rearrangement.src.catalog import OBJECT_SPECS

    robot, objects = env.scene["robot"], env.scene["objects"]
    body = robot.body_names.index(FORCE_TORQUE_PROXIMAL_BODY_NAME)
    object_id = int(env.blocker_object_id[0])
    geometry = env.spawn_geometries[object_id]
    pose = objects.data.object_link_pose_w[0, object_id].clone()
    relative = quat_mul(pose[3:7], quat_conjugate(pose.new_tensor(OBJECT_SPECS[object_id].upright_quat)))
    center_offset = pose.new_tensor(geometry.lower) + pose.new_tensor(geometry.upper)
    pose[:3] = robot.data.body_com_pos_w[0, body] - quat_apply(relative, center_offset * 0.5)
    objects.write_object_link_pose_to_sim(pose[None, None], object_ids=[object_id],
                                         env_ids=torch.tensor([0], device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    term = env.action_manager.get_term("manipulate")
    action = torch.zeros((env.num_envs, 18), device=env.device)
    limits = robot.data.soft_joint_pos_limits[:, term._hand_ids]
    action[:, 12:] = (2 * robot.data.joint_pos[:, term._hand_ids] - limits[..., 0] - limits[..., 1]) / (
        limits[..., 1] - limits[..., 0])
    with torch.no_grad():
        _, _, terminated, _, extra = env.step(action)
    if not terminated[0] or not extra["failure_reasons"][0, 5]:
        raise AssertionError("Physical spacer/object collision did not trigger the filtered contact failure.")
    print("FORBIDDEN_CONTACT_VALIDATED " + json.dumps(dict(reasons=extra["failure_reasons"][0].tolist())), flush=True)
    # Start fresh after the deliberately invalid fixture for the PPO smoke.
    return env.reset()[0]


def validate_control_hold(env):
    """An unloaded open-hand hold must stay stable without hard-load failures."""
    term, robot = env.action_manager.get_term("manipulate"), env.scene["robot"]
    initial = term.measured_tcp_pose_b().clone()
    limits = robot.data.soft_joint_pos_limits[:, term._hand_ids]
    action = torch.zeros((env.num_envs, 18), device=env.device)
    action[:, 12:] = (2 * robot.data.joint_pos[:, term._hand_ids] - limits[..., 0] - limits[..., 1]) / (
        limits[..., 1] - limits[..., 0])
    steps = min(16, env.max_episode_length - 1)
    if steps < 1:
        raise ValueError("Control validation needs an episode longer than one policy step.")
    force_torque_peak = torch.zeros(2, device=env.device)
    for _ in range(steps):
        before = env.initial_object_geometry.clone()
        before_history = env.mdp_state.history.clone()
        with torch.no_grad():
            observations, reward, terminated, truncated, extra = env.step(action)
        validate_transition(env, before, observations["policy"], reward, terminated, truncated, extra, action,
                            before_history)
        if terminated.any() or truncated.any():
            raise AssertionError(f"Unloaded hold terminated: {extra['failure_reasons'].tolist()}")
        wrench = extra["compensated_wrench"]
        force_torque_peak = torch.maximum(force_torque_peak, torch.stack((wrench[:, :3].norm(dim=-1).max(),
                                                                        wrench[:, 3:].norm(dim=-1).max())))
    final = term.measured_tcp_pose_b()
    position_drift = (final[:, :3] - initial[:, :3]).norm(dim=-1)
    angle_drift = 2 * torch.acos((final[:, 3:] * initial[:, 3:]).sum(-1).abs().clamp(0, 1))
    if position_drift.max() > 0.01 or angle_drift.max() > 0.05:
        raise AssertionError("Unloaded OSC hold drift exceeded 1 cm / 0.05 rad.")
    print("CONTROL_HOLD_VALIDATED " + json.dumps(dict(steps=steps, position_drift=position_drift.tolist(),
                                                     angle_drift=angle_drift.tolist(),
                                                     force_torque_peak=force_torque_peak.tolist())), flush=True)
    return env.reset()[0]


def main():
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    runner_cfg = load_cfg_from_registry(args.task, "ppo_cfg_entry_point")
    cfg.seed = args.seed
    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s
    if args.iterations is not None:
        runner_cfg.max_iterations = args.iterations
    if args.checkpoint_interval is not None:
        runner_cfg.save_interval = args.checkpoint_interval
    if args.rollout_steps is not None:
        runner_cfg.algorithm.rollout_steps = args.rollout_steps
    log_dir = args.log_dir or (REPOSITORY_ROOT / "outputs" / runner_cfg.experiment_name / "learning" /
                               datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    env = gym.make(args.task, cfg=cfg).unwrapped
    visualizer = None
    try:
        observations, _ = env.reset()
        if args.validate:
            observations = validate_control_hold(env)
            observations = validate_forbidden_contact(env)
        if env.sim.has_gui() and not args.disable_role_obb:
            visualizer = RoleObbVisualizer(env)
            visualizer.update()
        obs = observations["policy"].detach().clone()
        model_cfg = runner_cfg.policy
        if (model_cfg.max_objects, model_cfg.history_steps) != (cfg.mdp.max_objects, cfg.mdp.history_length):
            raise ValueError("Registered policy object/history capacities do not match the environment.")
        if args.resume is not None:
            trainer = PPOTrainer.load(args.resume, device=env.device)
            if trainer.model.cfg != model_cfg:
                raise ValueError(
                    f"Checkpoint model settings do not match this environment: "
                    f"history_steps={trainer.model.cfg.history_steps}, expected={model_cfg.history_steps}. "
                    "Start a new training run for the current/previous-step policy."
                )
            if args.rollout_steps is not None:
                trainer.cfg.rollout_steps = args.rollout_steps
        else:
            trainer = PPOTrainer(ContactRearrangementPolicy(model_cfg).to(env.device), runner_cfg.algorithm)
        model = trainer.model
        log_dir.mkdir(parents=True, exist_ok=True)
        configuration = dict(task=args.task, seed=args.seed, num_envs=env.num_envs, step_dt=env.step_dt,
                             max_iterations=runner_cfg.max_iterations, save_interval=runner_cfg.save_interval,
                             episode_length_s=cfg.episode_length_s, mdp=asdict(cfg.mdp),
                             model=asdict(model.cfg), ppo=asdict(trainer.cfg),
                             object_frame="current EEF", wrench_frame="physical sensor",
                             perception="initial depth OBB, simulation instance masks and role association")
        (log_dir / "config.json").write_text(json.dumps(configuration, indent=2) + "\n")
        print("LEARNING_READY " + json.dumps(dict(observation_dim=obs.shape[-1], action_dim=18,
                                                     parameters=sum(p.numel() for p in model.parameters()),
                                                     observed_objects=env.initial_object_valid.sum(-1).tolist())), flush=True)
        total_terminated = total_truncated = transitions = 0
        diagnostic_failures = 0
        started = time.monotonic()
        with (log_dir / "metrics.jsonl").open("a" if args.resume else "w", buffering=1) as metrics_file:
            for _ in range(runner_cfg.max_iterations):
                storage = {key: [] for key in ("observations", "actions", "old_log_probs", "rewards",
                                               "old_values", "next_values", "terminated", "truncated")}
                previous_parameters = [p.detach().clone() for p in model.parameters()] if args.validate else None
                failure_counts = torch.zeros(6, device=env.device, dtype=torch.long)
                wrench_peak = torch.zeros(2, device=env.device)
                for _ in range(trainer.cfg.rollout_steps):
                    before_geometry = env.initial_object_geometry.clone() if args.validate else None
                    before_history = env.mdp_state.history.clone() if args.validate else None
                    with torch.no_grad():
                        action, log_prob, value = model.act(obs)
                        following, reward, terminated, truncated, extra = env.step(action)
                        next_obs = following["policy"].detach().clone()
                        next_value = model.value(extra["terminal_observation"])
                    if visualizer is not None:
                        visualizer.update()
                    if args.validate:
                        validate_transition(env, before_geometry, next_obs, reward, terminated, truncated, extra, action,
                                            before_history)
                        if terminated.any() and diagnostic_failures < 3:
                            print("TERMINAL_DIAGNOSTIC " + json.dumps(dict(
                                reasons=extra["failure_reasons"][terminated].tolist(),
                                compensated_wrench=extra["compensated_wrench"][terminated].tolist(),
                                action=action[terminated].tolist())), flush=True)
                            diagnostic_failures += 1
                    wrench = extra["compensated_wrench"]
                    wrench_peak = torch.maximum(wrench_peak, torch.stack((wrench[:, :3].norm(dim=-1).max(),
                                                                          wrench[:, 3:].norm(dim=-1).max())))
                    values = (obs, action, log_prob, reward, value, next_value, terminated, truncated)
                    for key, field in zip(storage, values):
                        storage[key].append(field.detach().clone())
                    obs = next_obs
                    failure_counts += extra["failure_reasons"].sum(0)
                    total_terminated += int(terminated.sum())
                    total_truncated += int(truncated.sum())
                    transitions += env.num_envs
                metrics = trainer.update(**{key: torch.stack(fields) for key, fields in storage.items()})
                if args.validate and not any(not torch.equal(old, current) for old, current in zip(previous_parameters, model.parameters())):
                    raise AssertionError("PPO did not update model parameters.")
                metrics.update(transitions=transitions, terminated=total_terminated, truncated=total_truncated,
                               mean_reward=float(torch.stack(storage["rewards"]).mean()),
                               history_samples=env.mdp_state.history_valid.sum(-1).tolist(),
                               failure_counts=failure_counts.tolist(), elapsed_s=time.monotonic() - started)
                metrics["force_torque_peak"] = wrench_peak.tolist()
                metrics_file.write(json.dumps(metrics) + "\n")
                print("PPO_UPDATE " + json.dumps(metrics), flush=True)
                if trainer.update_count % runner_cfg.save_interval == 0:
                    trainer.save(log_dir / f"model_{trainer.update_count}.pt", environment=configuration)
            final_checkpoint = log_dir / "model_latest.pt"
            trainer.save(final_checkpoint, environment=configuration)
            if args.validate:
                restored = PPOTrainer.load(final_checkpoint, device=env.device)
                with torch.no_grad():
                    torch.testing.assert_close(restored.model.value(obs), model.value(obs), rtol=0, atol=0)
                if restored.update_count != trainer.update_count:
                    raise AssertionError("Checkpoint did not preserve PPO update count.")
                print("LEARNING_VALIDATED " + json.dumps(dict(transitions=transitions, terminated=total_terminated,
                                                               truncated=total_truncated, checkpoint=str(final_checkpoint))), flush=True)
    finally:
        if visualizer is not None:
            visualizer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
