"""Check one comparison task's native reset/step path without starting PPO.

Run tasks in separate processes. This probe uses the registered configuration,
native public reset API and real assets, including partial and shared-scene resets.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
LAB = Path(os.environ.get("ISAACLAB_PATH", ROOT / "IsaacLab")).expanduser().resolve()
sys.path[:0] = [str(ROOT / "src/contact_manipulation_symmetric"),
               *(str(LAB / "source" / name) for name in ("isaaclab", "isaaclab_tasks", "isaaclab_rl"))]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--scenes", type=int, default=2)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 2 or args.scenes < 1 or args.steps < 1:
    parser.error("Use at least two environments, one scene and one control step.")
app = AppLauncher(args).app


def run(report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry

    import contact_manipulation_symmetric  # noqa: F401
    from contact_manipulation_symmetric.mdp.events import get_task_state

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.sim.render_interval = 2**31 - 1
    assert cfg.rewards.terminal is None
    assert cfg.terminations.success is not None
    report.update(task=args.task, seed=args.seed, num_envs=args.num_envs,
                  physics_dt=cfg.sim.dt, policy_dt=cfg.sim.dt * cfg.decimation,
                  wrench_scale=list(cfg.actions.arm_action.wrench_scale),
                  relocation_front_corridors=cfg.relocation_include_front_corridors,
                  access_to_back_wall=cfg.access_region_to_back_wall,
                  scenes=[])
    wrapped = gym.make(args.task, cfg=cfg)
    env = wrapped.unwrapped
    try:
        assert "terminal" not in env.reward_manager.active_terms
        assert "success" in env.termination_manager.active_terms
        for generation in range(args.scenes):
            if generation:
                # Exercise the production full-reset path at a shared-scene boundary.
                env._episode_reset.next_scene_step = env.common_step_counter
            observations, _ = env.reset()
            state = get_task_state(env)
            runtime = env._episode_reset
            sample = runtime.shared_sample
            assert runtime.scene_generation == generation
            torch.testing.assert_close(observations["policy"], observations["critic"], rtol=0, atol=0)
            assert observations["policy"].shape == (args.num_envs, 232)
            original = state.access_region.new_tensor(sample.access_region)
            if cfg.access_region_to_back_wall:
                torch.testing.assert_close(state.access_region[:, 0], state.workspace_bounds[:, 0])
                torch.testing.assert_close(state.access_region[:, 1:], original[None, 1:].expand(args.num_envs, -1))
            else:
                torch.testing.assert_close(state.access_region, original[None].expand(args.num_envs, -1))
            term = cfg.rewards.blocker_relocation
            initial_reward = term.func(env, **term.params)
            torch.testing.assert_close(initial_reward, torch.zeros_like(initial_reward), atol=1e-7, rtol=0)
            item = dict(generation=generation, target_id=sample.target_id, blocker_id=sample.blocker_id,
                        active_ids=list(sample.active_ids),
                        positions={str(k): list(map(float, v)) for k, v in sample.positions.items()},
                        yaws={str(k): float(v) for k, v in sample.yaws.items()},
                        generated_access=list(sample.access_region),
                        evaluated_access=state.access_region[0].cpu().tolist(),
                        workspace=state.workspace_bounds[0].cpu().tolist(),
                        reserved_regions=[list(r) for r in sample.reserved_regions],
                        goal_polygons=state.relocation_regions[0].cpu().tolist(),
                        goal_counts=state.relocation_vertex_counts[0].cpu().tolist(),
                        reward_terms=list(env.reward_manager.active_terms),
                        termination_terms=list(env.termination_manager.active_terms))
            # A partial native reset must neither regenerate the shared layout
            # nor modify the unselected robot/history rows.
            reset_ids = torch.arange(1, args.num_envs, 2, device=env.device)
            untouched = torch.arange(0, args.num_envs, 2, device=env.device)
            robot_before = env.scene["robot"].data.joint_pos[untouched].clone()
            center_before = state.previous_blocker_center[untouched].clone()
            elapsed_before = state.elapsed_time[untouched].clone()
            env.reset(env_ids=reset_ids)
            assert runtime.shared_sample is sample
            assert runtime.scene_generation == generation
            torch.testing.assert_close(env.scene["robot"].data.joint_pos[untouched], robot_before,
                                       rtol=0, atol=1e-6)
            torch.testing.assert_close(state.previous_blocker_center[untouched], center_before, rtol=0, atol=0)
            torch.testing.assert_close(state.elapsed_time[untouched], elapsed_before, rtol=0, atol=0)
            item["partial_reset_preserved_other_rows"] = True
            actions = torch.zeros((args.num_envs, env.action_manager.total_action_dim), device=env.device)
            # Keep the real hand open, rather than mapping raw zero to its middle angle.
            actions[:, 6:] = -1.0 / cfg.actions.hand_action.scale
            done_count = 0
            for _ in range(args.steps):
                observations, reward, terminated, truncated, _ = env.step(actions)
                assert bool(torch.isfinite(reward).all())
                assert bool(torch.isfinite(observations["policy"]).all())
                torch.testing.assert_close(observations["policy"], observations["critic"], rtol=0, atol=0)
                done_count += int((terminated | truncated).sum())
            item.update(control_steps=args.steps, completed_episodes=done_count,
                        current_raw_relocation=term.func(env, **term.params).cpu().tolist())
            report["scenes"].append(item)
            print(f"[Experiments] {args.task}: scene {generation}, partial reset and {args.steps} steps passed.",
                  flush=True)
        report["status"] = "complete"
    finally:
        wrapped.close()


if __name__ == "__main__":
    result = {"status": "running", "task": args.task, "PPO_started": False}
    try:
        run(result)
    except Exception:
        result.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        app.close(wait_for_replicator=False, skip_cleanup=True)
