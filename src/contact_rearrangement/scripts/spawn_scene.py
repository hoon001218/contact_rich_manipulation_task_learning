#!/usr/bin/env python3
"""Launch the new shelf environment and hold the robot at its initial posture."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
# Use this checkout even when the Python environment has another Isaac Lab
# checkout installed in editable mode. No editable installation is required.
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/source/isaaclab"))

from isaaclab.app import AppLauncher
from contact_rearrangement.layout import ASSET_SPECS, MAX_ACTIVE_OBJECTS, OBJECT_NAMES, OBJECT_SPECS


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of independently sampled scenes.")
parser.add_argument("--seed", type=int, default=0, help="Base seed for per-environment reset streams.")
parser.add_argument("--min_objects", type=int, default=4, help="Minimum number of active objects.")
parser.add_argument("--max_objects", type=int, default=MAX_ACTIVE_OBJECTS, help="Maximum active objects (up to 10).")
parser.add_argument("--all_asset_types", action="store_true", help="Use one of every catalog type; fixes active count to 10.")
parser.add_argument("--target_asset", type=str, default=None, help="Fix the episode Target asset type.")
parser.add_argument("--blocker_asset", type=str, default=None, help="Fix the episode Blocker asset type.")
parser.add_argument(
    "--max_steps", type=int, default=0, help="Stop after this many environment steps; 0 runs until closed."
)
parser.add_argument(
    "--reset_interval", type=int, default=None,
    help="Reset all scenes every N environment steps; default is 5 simulated seconds; 0 disables periodic reset."
)
parser.add_argument("--validate", action="store_true", help="Check robot/objects, settling, and full/partial resets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.max_steps < 0 or (
    args_cli.reset_interval is not None and args_cli.reset_interval < 0
):
    parser.error("num_envs must be positive; max_steps and reset_interval must be nonnegative.")
if not 4 <= args_cli.min_objects <= args_cli.max_objects <= MAX_ACTIVE_OBJECTS:
    parser.error("Active object counts must satisfy 4 <= min_objects <= max_objects <= 10.")
if args_cli.all_asset_types:
    if args_cli.max_objects < len(ASSET_SPECS):
        parser.error("--all_asset_types requires --max_objects 10.")
    args_cli.min_objects = args_cli.max_objects = len(ASSET_SPECS)
asset_names = {spec.name for spec in ASSET_SPECS}
for role, asset in (("Target", args_cli.target_asset), ("Blocker", args_cli.blocker_asset)):
    if asset is not None and asset not in asset_names:
        parser.error(f"Unknown {role} asset: {asset}")
if args_cli.target_asset is not None and args_cli.target_asset == args_cli.blocker_asset:
    parser.error("Target asset type must occur only once, so Blocker must use a different type.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Simulator-dependent imports must follow AppLauncher.
import torch
from dataclasses import replace

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.math import quat_conjugate, quat_mul

from contact_rearrangement.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES, TACTILE_BODY_NAMES
from contact_rearrangement.env_cfg import ContactRearrangementEnvCfg
from contact_rearrangement.sampling import scene_metrics, validate_sample


def validate_reset_state(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Check all object poses/velocities and robot targets immediately after reset."""
    objects = env.scene["objects"]
    expected = env.spawn_initial_state[env_ids]
    torch.testing.assert_close(objects.data.object_link_pos_w[env_ids], expected[..., :3], atol=1e-5, rtol=0.0)
    # Read the backend directly: the pose writer also updates Isaac Lab's cache,
    # so matching only that cache cannot detect stale GPU body mappings.
    physical_pose = objects.reshape_view_to_data(objects.root_physx_view.get_transforms())[env_ids]
    torch.testing.assert_close(physical_pose[..., :3], expected[..., :3], atol=1e-5, rtol=0.0)
    physical_quat = torch.roll(physical_pose[..., 3:7], shifts=1, dims=-1)
    physical_similarity = (physical_quat * expected[..., 3:7]).sum(dim=-1).abs()
    torch.testing.assert_close(physical_similarity, torch.ones_like(physical_similarity), atol=1e-5, rtol=0.0)
    # Compare orientation without depending on the quaternion's sign.
    quaternions = objects.data.object_link_quat_w[env_ids]
    similarity = (quaternions * expected[..., 3:7]).sum(dim=-1).abs()
    torch.testing.assert_close(similarity, torch.ones_like(similarity), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(objects.data.object_com_vel_w[env_ids], expected[..., 7:], atol=1e-5, rtol=0.0)
    for env_id in env_ids.cpu().tolist():
        validate_sample(env.spawn_samples[env_id], env.spawn_geometries, env.spawn_rules)
        sample = env.spawn_samples[env_id]
        actual_ids = env.active_object_mask[env_id].nonzero().flatten().cpu().tolist()
        if set(actual_ids) != set(sample.active_ids):
            raise RuntimeError("Active simulation mask disagrees with the sampled scene.")
        roles = int(env.target_object_id[env_id]), int(env.blocker_object_id[env_id])
        if roles != (sample.target_id, sample.blocker_id):
            raise RuntimeError("Role metadata disagrees with the sampled scene.")
        target_asset = OBJECT_SPECS[sample.target_id].asset_name
        if sum(OBJECT_SPECS[index].asset_name == target_asset for index in sample.active_ids) != 1:
            raise RuntimeError("Target asset type must appear exactly once in each scene.")
        if env.target_visibility_validated[env_id] or not torch.isnan(env.target_visible_fraction[env_id]):
            raise RuntimeError("Image visibility must remain unvalidated until a camera measurement is available.")
    validate_pool_flags(env)
    robot = env.scene["robot"]
    torch.testing.assert_close(
        robot.data.joint_pos[env_ids], robot.data.default_joint_pos[env_ids], atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        robot.data.joint_pos_target[env_ids], robot.data.default_joint_pos[env_ids], atol=1e-5, rtol=0.0
    )


def validate_scene(env: ManagerBasedEnv) -> None:
    """Fail on missing asset links or invalid physical state."""
    robot, objects = env.scene["robot"], env.scene["objects"]
    if tuple(objects.object_names) != OBJECT_NAMES:
        raise RuntimeError(f"Unexpected object identities/order: {objects.object_names}")
    missing_joints = set(ARM_JOINT_NAMES + HAND_JOINT_NAMES) - set(robot.joint_names)
    missing_links = set(TACTILE_BODY_NAMES) - set(robot.body_names)
    if missing_joints or missing_links:
        raise RuntimeError(f"Missing robot joints: {sorted(missing_joints)}; hand links: {sorted(missing_links)}")
    if not robot.is_fixed_base:
        raise RuntimeError("The UR5e base must be fixed.")
    if not torch.isfinite(robot.data.joint_pos).all() or not torch.isfinite(objects.data.object_state_w).all():
        raise RuntimeError("Non-finite robot or object states.")


def validate_pool_flags(env: ManagerBasedEnv) -> None:
    objects = env.scene["objects"]
    if objects.root_physx_view.get_disable_gravities().any():
        raise RuntimeError("Pool bodies must retain native gravity; only parked bodies receive countergravity.")
    if objects.root_physx_view.get_disable_simulations().any():
        raise RuntimeError("Pool body simulation must remain enabled to preserve GPU body indexing.")


def validate_settling(env: ManagerBasedEnv) -> None:
    objects = env.scene["objects"]
    validate_pool_flags(env)
    active = env.active_object_mask
    delta = objects.data.object_link_pos_w - env.spawn_initial_state[..., :3]
    fallen = active & (delta[..., 2] < -0.10)
    if fallen.any():
        failures = [
            (env_id, OBJECT_NAMES[object_id],
             env.spawn_initial_state[env_id, object_id, :3].cpu().tolist(),
             objects.data.object_link_pos_w[env_id, object_id].cpu().tolist())
            for env_id, object_id in fallen.nonzero().cpu().tolist()
        ]
        raise RuntimeError(f"Active objects fell below their shelf (env, name, initial, current): {failures}")
    drifted = active & (delta[..., :2].norm(dim=-1) > 0.05)
    if drifted.any():
        failures = [
            (env_id, OBJECT_NAMES[object_id],
             env.spawn_initial_state[env_id, object_id, :3].cpu().tolist(),
             objects.data.object_link_pos_w[env_id, object_id].cpu().tolist(),
             objects.data.object_com_vel_w[env_id, object_id].cpu().tolist())
            for env_id, object_id in drifted.nonzero().cpu().tolist()
        ]
        raise RuntimeError(f"Active objects drifted more than 5 cm (env, name, initial, current, velocity): {failures}")
    rotation = quat_mul(objects.data.object_link_quat_w, quat_conjugate(env.spawn_initial_state[..., 3:7]))
    vertical_alignment = 1.0 - 2.0 * rotation[..., 1:3].square().sum(dim=-1)
    if (vertical_alignment[active] < 0.966).any():
        raise RuntimeError("An active object tilted more than 15 degrees during PD hold.")
    inactive = ~active
    moved_inactive = inactive & (delta.abs().amax(dim=-1) > 1e-5)
    if moved_inactive.any():
        failures = [
            (env_id, OBJECT_NAMES[object_id], delta[env_id, object_id].cpu().tolist(),
             objects.data.object_com_vel_w[env_id, object_id].cpu().tolist())
            for env_id, object_id in moved_inactive.nonzero().cpu().tolist()[:8]
        ]
        raise RuntimeError(f"Parked objects moved (env, name, delta, velocity): {failures}")
    torch.testing.assert_close(
        objects.data.object_link_pos_w[inactive], env.spawn_initial_state[..., :3][inactive], atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        objects.data.object_com_vel_w[inactive], torch.zeros_like(objects.data.object_com_vel_w[inactive]),
        atol=1e-5, rtol=0.0,
    )
    validate_pool_flags(env)


def describe_samples(env: ManagerBasedEnv) -> None:
    for env_id, sample in enumerate(env.spawn_samples):
        metrics = scene_metrics(sample, env.spawn_geometries, env.spawn_rules)
        assets = [OBJECT_SPECS[index].asset_name for index in sample.active_ids]
        print(
            f"[SPAWN]: env={env_id} seed={int(env.spawn_seeds[env_id])} active={len(assets)} "
            f"target={OBJECT_NAMES[sample.target_id]} blocker={OBJECT_NAMES[sample.blocker_id]} "
            f"x_coverage={metrics.x_coverage:.1%} y_coverage={metrics.y_coverage:.1%} "
            f"shelf_occupancy={metrics.shelf_occupancy:.1%} neighbors={metrics.local_neighbor_fraction:.1%} "
            f"clearance={sample.clearance_kind} visibility=unvalidated assets={assets}", flush=True,
        )


def validate_settling_and_reset(env: ManagerBasedEnv) -> None:
    """Check shelf support and perturb/reset one environment while preserving the others."""
    objects = env.scene["objects"]
    validate_settling(env)

    selected = torch.tensor([0], device=env.device, dtype=torch.long)
    untouched = objects.data.object_state_w[1:].clone()
    untouched_mask = env.active_object_mask[1:].clone()
    untouched_roles = torch.stack((env.target_object_id[1:], env.blocker_object_id[1:])).clone()
    untouched_access = env.access_region[1:].clone()
    untouched_counts = env.spawn_reset_counts[1:].clone()
    untouched_visibility = env.target_visibility_validated[1:].clone()
    untouched_visible_fraction = env.target_visible_fraction[1:].clone()
    perturbed = objects.data.object_state_w[selected].clone()
    perturbed[..., 0] += 0.05
    perturbed[..., 7] = 0.10
    objects.write_object_state_to_sim(perturbed, env_ids=selected)
    env.reset(env_ids=selected)
    validate_reset_state(env, selected)
    if env.num_envs > 1:
        torch.testing.assert_close(objects.data.object_state_w[1:], untouched, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(env.active_object_mask[1:], untouched_mask)
        torch.testing.assert_close(torch.stack((env.target_object_id[1:], env.blocker_object_id[1:])), untouched_roles)
        torch.testing.assert_close(env.access_region[1:], untouched_access)
        torch.testing.assert_close(env.spawn_reset_counts[1:], untouched_counts)
        torch.testing.assert_close(env.target_visibility_validated[1:], untouched_visibility)
        torch.testing.assert_close(env.target_visible_fraction[1:], untouched_visible_fraction, equal_nan=True)
    env.reset()
    validate_reset_state(env, torch.arange(env.num_envs, device=env.device))
    print("[INFO]: Scene validation passed: clutter rules, pool flags, support, and full/partial reset.", flush=True)


def main() -> None:
    cfg = ContactRearrangementEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.target_asset = args_cli.target_asset
    cfg.blocker_asset = args_cli.blocker_asset
    cfg.spawn_rules = replace(
        cfg.spawn_rules, min_objects=args_cli.min_objects, max_objects=args_cli.max_objects,
        all_asset_types=args_cli.all_asset_types,
    )
    env = ManagerBasedEnv(cfg=cfg)
    try:
        reset_interval = args_cli.reset_interval
        if reset_interval is None:
            reset_interval = math.ceil(5.0 / env.step_dt)
        env.reset()
        print(
            f"[INFO]: Spawned UR5e + RH56E2, shelf, {len(ASSET_SPECS)} asset types, "
            f"{len(OBJECT_NAMES)} pool bodies, {args_cli.min_objects}–{args_cli.max_objects} active per scene."
        )
        if reset_interval:
            print(
                f"[INFO]: Automatic reset every {reset_interval} environment steps "
                f"({reset_interval * env.step_dt:.2f} simulated seconds)."
            )
        else:
            print("[INFO]: Automatic reset disabled.")
        print(f"[INFO]: Measured shelf/rules: {env.spawn_rules}")
        describe_samples(env)
        if args_cli.validate:
            validate_scene(env)
            validate_reset_state(env, torch.arange(env.num_envs, device=env.device))

        steps = 0
        with torch.inference_mode():
            while simulation_app.is_running() and (args_cli.max_steps == 0 or steps < args_cli.max_steps):
                # The empty action tensor advances physics while reset's PD
                # targets hold the robot. This is scene inspection, not training.
                env.step(env.action_manager.action)
                steps += 1
                if args_cli.validate:
                    validate_scene(env)
                    if steps == 1:
                        validate_settling(env)
                if reset_interval and steps % reset_interval == 0:
                    if args_cli.validate:
                        validate_settling(env)
                    env.reset()
                    print(f"[INFO]: Reset at step {steps} ({steps * env.step_dt:.2f} simulated seconds).")
                    if args_cli.validate:
                        validate_reset_state(env, torch.arange(env.num_envs, device=env.device))
                    describe_samples(env)
            if args_cli.validate:
                validate_settling_and_reset(env)
        print(f"[INFO]: Completed {steps} environment steps ({steps * env.step_dt:.2f} simulated seconds).")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Kit shutdown may terminate Python before an uncaught exception is
        # printed. Keep the original failure visible before closing the app.
        traceback.print_exc()
        # Kit's fast shutdown otherwise masks Python exceptions with exit 0.
        import omni.kit.app
        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        simulation_app.close()
