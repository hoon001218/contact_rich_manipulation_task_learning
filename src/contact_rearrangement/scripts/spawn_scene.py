#!/usr/bin/env python3
"""Spawn shelf clutter and hold a randomized, collision-free elbow-up robot start."""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from collections import Counter
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
# Use this checkout even when the Python environment has another Isaac Lab
# checkout installed in editable mode. No editable installation is required.
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "IsaacLab/source/isaaclab"))

from isaaclab.app import AppLauncher
from contact_rearrangement.catalog import ASSET_SPECS, MAX_ACTIVE_OBJECTS, OBJECT_NAMES, OBJECT_SPECS


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1, help="Number of independently sampled scenes.")
parser.add_argument("--seed", type=int, default=0, help="Base seed for per-environment reset streams.")
parser.add_argument("--min_asset_types", type=int, default=4, help="Minimum number of distinct types in an episode.")
parser.add_argument("--max_asset_types", type=int, default=len(ASSET_SPECS), help="Maximum distinct types (up to 10).")
parser.add_argument("--min_objects", type=int, default=4, help="Minimum active instances; density may require more.")
parser.add_argument("--max_objects", type=int, default=MAX_ACTIVE_OBJECTS, help="Safety cap on active instances (up to 40).")
parser.add_argument("--all_asset_types", action="store_true", help="Select all 10 types; non-Target copies may repeat to meet density.")
parser.add_argument("--min_shelf_occupancy", type=float, default=None, help="Minimum occupied footprint fraction of usable shelf area.")
parser.add_argument("--min_region_occupancy", type=float, default=None, help="Minimum occupied footprint fraction in each shelf region.")
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
parser.add_argument("--disable_role_obb", action="store_true", help="Hide the green Target / red Blocker viewport outlines.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1 or args_cli.max_steps < 0 or (
    args_cli.reset_interval is not None and args_cli.reset_interval < 0
):
    parser.error("num_envs must be positive; max_steps and reset_interval must be nonnegative.")
if not 2 <= args_cli.min_objects <= args_cli.max_objects <= MAX_ACTIVE_OBJECTS:
    parser.error("Instance limits must satisfy 2 <= min_objects <= max_objects <= 40.")
if not 2 <= args_cli.min_asset_types <= args_cli.max_asset_types <= len(ASSET_SPECS):
    parser.error("Type limits must satisfy 2 <= min_asset_types <= max_asset_types <= 10.")
if args_cli.all_asset_types:
    if args_cli.max_objects < len(ASSET_SPECS):
        parser.error("--all_asset_types requires an instance cap of at least 10.")
    args_cli.min_asset_types = args_cli.max_asset_types = len(ASSET_SPECS)
if args_cli.max_objects < args_cli.min_asset_types:
    parser.error("The instance cap must allow at least one instance of each selected type.")
for option in ("min_shelf_occupancy", "min_region_occupancy"):
    value = getattr(args_cli, option)
    if value is not None and not 0.0 <= value < 1.0:
        parser.error(f"--{option} must be a finite fraction in [0, 1).")
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
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from contact_rearrangement.assets import ARM_JOINT_NAMES, HAND_JOINT_NAMES, TACTILE_BODY_NAMES
from contact_rearrangement.env_cfg import ContactRearrangementEnvCfg
from contact_rearrangement.sampling import scene_metrics, validate_sample
from contact_rearrangement.spawn import validate_robot_start_state
from contact_rearrangement.visualization import RoleObbVisualizer


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
        robot.data.joint_pos[env_ids], env.robot_initial_joint_pos[env_ids], atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        robot.data.joint_pos_target[env_ids], env.robot_initial_joint_pos[env_ids], atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(robot.data.joint_vel[env_ids], torch.zeros_like(robot.data.joint_vel[env_ids]))
    # Refresh FK without integrating physics, then check native link transforms.
    _ = robot.data.body_link_pos_w
    validate_robot_start_state(env, env_ids)


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
            f"types={len(set(assets))} "
            f"target={OBJECT_NAMES[sample.target_id]} blocker={OBJECT_NAMES[sample.blocker_id]} "
            f"x_coverage={metrics.x_coverage:.1%} y_coverage={metrics.y_coverage:.1%} "
            f"shelf_occupancy={metrics.shelf_occupancy:.1%} neighbors={metrics.local_neighbor_fraction:.1%} "
            f"min_region_occupancy={min(metrics.region_occupancies):.1%} "
            f"clearance={sample.clearance_kind} visibility=unvalidated "
            f"asset_counts={dict(sorted(Counter(assets).items()))}", flush=True,
        )
        if env.cfg.robot_start.enabled:
            tcp = (env.robot_start_tcp_pos_w[env_id] - env.scene.env_origins[env_id]).cpu().tolist()
            print(f"[ROBOT START]: env={env_id} tcp_local={tcp} "
                  f"elbow_height={float(env.robot_start_elbow_height[env_id]):.3f}m "
                  f"attempts={int(env.robot_start_attempts[env_id])} collision_clearance=validated", flush=True)


def validate_role_obb(env: ManagerBasedEnv, visualizer: RoleObbVisualizer) -> None:
    """Check line count and that drawing leaves the USD stage and physics intact."""
    objects = env.scene["objects"]
    view = objects.root_physx_view
    transforms = view.get_transforms().clone()
    velocities = view.get_velocities().clone()
    gravities = view.get_disable_gravities().clone()
    simulations = view.get_disable_simulations().clone()
    prims = tuple((str(prim.GetPath()), tuple(prim.GetAppliedSchemas())) for prim in env.scene.stage.Traverse())
    visualizer.update()
    if visualizer.line_count != 24 * env.num_envs:
        raise RuntimeError("Role outlines must draw exactly 12 edges for each Target and Blocker.")
    if tuple((str(prim.GetPath()), tuple(prim.GetAppliedSchemas())) for prim in env.scene.stage.Traverse()) != prims:
        raise RuntimeError("Role outline drawing must not modify USD prims or physics schemas.")
    torch.testing.assert_close(view.get_transforms(), transforms, atol=0.0, rtol=0.0)
    torch.testing.assert_close(view.get_velocities(), velocities, atol=0.0, rtol=0.0)
    torch.testing.assert_close(view.get_disable_gravities(), gravities)
    torch.testing.assert_close(view.get_disable_simulations(), simulations)


def validate_settling_and_reset(env: ManagerBasedEnv, visualizer: RoleObbVisualizer | None = None) -> None:
    """Check shelf support and perturb/reset one environment while preserving the others."""
    objects = env.scene["objects"]
    validate_settling(env)

    selected = torch.tensor([0], device=env.device, dtype=torch.long)
    untouched = objects.data.object_state_w[1:].clone()
    robot = env.scene["robot"]
    untouched_robot_q = robot.data.joint_pos[1:].clone()
    untouched_robot_vel = robot.data.joint_vel[1:].clone()
    untouched_robot_targets = robot.data.joint_pos_target[1:].clone()
    untouched_robot_initial = env.robot_initial_joint_pos[1:].clone()
    untouched_robot_tcp = env.robot_start_tcp_pos_w[1:].clone()
    untouched_robot_region = env.robot_start_region[1:].clone()
    untouched_robot_elbow = env.robot_start_elbow_height[1:].clone()
    untouched_robot_attempts = env.robot_start_attempts[1:].clone()
    untouched_mask = env.active_object_mask[1:].clone()
    untouched_roles = torch.stack((env.target_object_id[1:], env.blocker_object_id[1:])).clone()
    untouched_access = env.access_region[1:].clone()
    untouched_counts = env.spawn_reset_counts[1:].clone()
    untouched_visibility = env.target_visibility_validated[1:].clone()
    untouched_visible_fraction = env.target_visible_fraction[1:].clone()
    perturbed = objects.data.object_state_w[selected].clone()
    perturbed[..., 0] += 0.05
    perturbed[..., 7] = 0.10
    # Prime the separate COM-pose cache before a same-timestamp write. The
    # collection must refresh it without a physics step before applying forces.
    _ = objects.data.object_com_pos_w
    objects.write_object_state_to_sim(perturbed, env_ids=selected)
    expected_com = objects.data.object_link_pos_w + quat_apply(
        objects.data.object_link_quat_w, objects.data.object_com_pos_b
    )
    torch.testing.assert_close(objects.data.object_com_pos_w, expected_com, atol=1e-6, rtol=0.0)
    perturbed_robot = robot.data.joint_pos[selected].clone()
    perturbed_robot[:, robot.joint_names.index(ARM_JOINT_NAMES[0])] += 0.05
    robot.write_joint_state_to_sim(perturbed_robot, torch.zeros_like(perturbed_robot), env_ids=selected)
    robot.set_joint_position_target(perturbed_robot, env_ids=selected)
    previous_tcp = env.robot_start_tcp_pos_w[selected].clone()
    env.reset(env_ids=selected)
    validate_reset_state(env, selected)
    if env.cfg.robot_start.enabled and torch.equal(env.robot_start_tcp_pos_w[selected], previous_tcp):
        raise RuntimeError("Robot start must be resampled when its environment is reset.")
    if visualizer is not None:
        validate_role_obb(env, visualizer)
    if env.num_envs > 1:
        torch.testing.assert_close(robot.data.joint_pos[1:], untouched_robot_q, atol=0.0, rtol=0.0)
        torch.testing.assert_close(robot.data.joint_vel[1:], untouched_robot_vel, atol=0.0, rtol=0.0)
        torch.testing.assert_close(robot.data.joint_pos_target[1:], untouched_robot_targets, atol=0.0, rtol=0.0)
        torch.testing.assert_close(env.robot_initial_joint_pos[1:], untouched_robot_initial, atol=0.0, rtol=0.0)
        torch.testing.assert_close(env.robot_start_tcp_pos_w[1:], untouched_robot_tcp, atol=0.0, rtol=0.0)
        torch.testing.assert_close(env.robot_start_region[1:], untouched_robot_region, atol=0.0, rtol=0.0)
        torch.testing.assert_close(env.robot_start_elbow_height[1:], untouched_robot_elbow, atol=0.0, rtol=0.0)
        torch.testing.assert_close(env.robot_start_attempts[1:], untouched_robot_attempts)
        torch.testing.assert_close(objects.data.object_state_w[1:], untouched, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(env.active_object_mask[1:], untouched_mask)
        torch.testing.assert_close(torch.stack((env.target_object_id[1:], env.blocker_object_id[1:])), untouched_roles)
        torch.testing.assert_close(env.access_region[1:], untouched_access)
        torch.testing.assert_close(env.spawn_reset_counts[1:], untouched_counts)
        torch.testing.assert_close(env.target_visibility_validated[1:], untouched_visibility)
        torch.testing.assert_close(env.target_visible_fraction[1:], untouched_visible_fraction, equal_nan=True)
    env.reset()
    validate_reset_state(env, torch.arange(env.num_envs, device=env.device))
    if visualizer is not None:
        validate_role_obb(env, visualizer)
    print("[INFO]: Scene validation passed: clutter rules, pool flags, support, elbow-up robot starts, and full/partial reset.", flush=True)


def main() -> None:
    cfg = ContactRearrangementEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.target_asset = args_cli.target_asset
    cfg.blocker_asset = args_cli.blocker_asset
    cfg.spawn_rules = replace(
        cfg.spawn_rules, min_objects=args_cli.min_objects, max_objects=args_cli.max_objects,
        min_asset_types=args_cli.min_asset_types, max_asset_types=args_cli.max_asset_types,
        all_asset_types=args_cli.all_asset_types,
    )
    for option in ("min_shelf_occupancy", "min_region_occupancy"):
        value = getattr(args_cli, option)
        if value is not None:
            cfg.spawn_rules = replace(cfg.spawn_rules, **{option: value})
    env = ManagerBasedEnv(cfg=cfg)
    visualizer = None
    try:
        reset_interval = args_cli.reset_interval
        if reset_interval is None:
            reset_interval = math.ceil(5.0 / env.step_dt)
        env.reset()
        if not args_cli.disable_role_obb and (env.sim.has_gui() or args_cli.validate):
            visualizer = RoleObbVisualizer(env)
            visualizer.update()
            print("[INFO]: Role OBB outlines: Target green, Blocker red (viewport debug draw only).")
        print(
            f"[INFO]: Spawned UR5e + RH56E2, shelf, {len(ASSET_SPECS)} asset types, "
            f"{len(OBJECT_NAMES)} pool bodies, {args_cli.min_asset_types}–{args_cli.max_asset_types} types per scene, "
            f"instances selected for >= {cfg.spawn_rules.min_shelf_occupancy:.0%} shelf occupancy "
            f"(instance cap {args_cli.max_objects})."
        )
        if reset_interval:
            print(
                f"[INFO]: Automatic reset every {reset_interval} environment steps "
                f"({reset_interval * env.step_dt:.2f} simulated seconds)."
            )
        else:
            print("[INFO]: Automatic reset disabled.")
        print(f"[INFO]: Measured shelf/rules: {env.spawn_rules}")
        if cfg.robot_start.enabled:
            print(f"[INFO]: Robot starting region/rules: {cfg.robot_start}")
            print(f"[INFO]: Robot FK checked against PhysX: {env.robot_pose_model.fk_position_error:.3g}m "
                  f"position error; {len(env.robot_pose_model.colliders)} robot colliders; "
                  f"{len(env.robot_pose_model.shelf_boxes)} shelf colliders.")
        describe_samples(env)
        if args_cli.validate:
            validate_scene(env)
            validate_reset_state(env, torch.arange(env.num_envs, device=env.device))
            if visualizer is not None:
                validate_role_obb(env, visualizer)

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
                    if args_cli.validate and visualizer is not None:
                        validate_role_obb(env, visualizer)
                if visualizer is not None:
                    visualizer.update()
            if args_cli.validate:
                validate_settling_and_reset(env, visualizer)
                if visualizer is not None:
                    print("[INFO]: Role OBB validation passed: 12 edges per role; USD and PhysX unchanged.", flush=True)
        print(f"[INFO]: Completed {steps} environment steps ({steps * env.step_dt:.2f} simulated seconds).", flush=True)
    finally:
        if visualizer is not None:
            visualizer.close()
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
