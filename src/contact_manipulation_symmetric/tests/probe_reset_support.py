"""Measure native shelf support after production full and partial resets.

No checkpoint, PPO, rendered camera, reward or automatic episode termination is
used. The registered production environment performs every reset. Physics is
then stepped directly with zero sensor wrench and raw-zero (midpoint) hand
commands. By default the reset robot is translated above both the shelf and
the unused-object parking grid to isolate object support from robot contact.

Run height comparisons in separate processes with the SAME seed, for example:
  ./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_reset_support.py \
      --headless --num-envs 16 --scenes 3 --physics-steps 40 --seed 42 \
      --spawn-height-offset 0.0001 --output logs/diagnostics/support_0p1mm.json

Reported bottom gaps use authored collision support geometry, not the cooked
PhysX convex shapes. Optional pair forces are actual native normal-force sums;
enabling them adds contact-report APIs to the diagnostic's object bodies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
LAB = Path(os.environ.get("ISAACLAB_PATH", ROOT / "IsaacLab")).expanduser().resolve()
sys.path[:0] = [str(ROOT / "src/contact_manipulation_symmetric"),
               *(str(LAB / "source" / name) for name in ("isaaclab", "isaaclab_tasks", "isaaclab_rl"))]

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Isaac-Contact-Manipulation-Symmetric-UR5e-v0")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=16)
    parser.add_argument("--scenes", type=int, default=3)
    parser.add_argument("--physics-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spawn-height-offset", type=float, default=0.0001)
    parser.add_argument("--robot-mode", choices=("parked", "zero-wrench"), default="parked")
    parser.add_argument("--robot-height-shift", type=float, default=8.0)
    parser.add_argument("--skip-partial", action="store_true")
    parser.add_argument("--contact-pairs", action="store_true")
    parser.add_argument("--contact-envs", type=int, nargs="+", default=[0])
    parser.add_argument("--max-traces", type=int, default=8)
    parser.add_argument("--drop-depth", type=float, default=0.03)
    parser.add_argument("--settle-start", type=float, default=0.25,
                        help="Start of the post-settling movement measurement window, seconds.")
    parser.add_argument("--settle-end", type=float, default=0.8,
                        help="End of the post-settling movement measurement window, seconds.")
    parser.add_argument("--output", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.scenes, args.physics_steps, args.max_traces) < 1:
        parser.error("Environment, scene, step and trace counts must be positive.")
    if not math.isfinite(args.spawn_height_offset) or args.spawn_height_offset < 0:
        parser.error("spawn-height-offset must be finite and nonnegative.")
    if not math.isfinite(args.drop_depth) or args.drop_depth <= 0:
        parser.error("drop-depth must be finite and positive.")
    if (not math.isfinite(args.settle_start) or not math.isfinite(args.settle_end)
            or not 0 <= args.settle_start < args.settle_end):
        parser.error("Use finite 0 <= settle-start < settle-end.")
    if not math.isfinite(args.robot_height_shift) or args.robot_height_shift < 6:
        parser.error("robot-height-shift must be at least 6 m, above the unused-object grid.")
    if any(index < 0 or index >= args.num_envs for index in args.contact_envs):
        parser.error("contact-envs must be valid environment IDs.")
    args.contact_envs = sorted(set(args.contact_envs))
    return args


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(report), ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def movement_statistics(values, reset_ids, active_ids, asset_names, *, keep_env_values=False):
    """Describe per-object values and per-environment maxima on CPU.

    Percentiles across objects and across environment maxima are deliberately
    separate. Partial-reset statistics include only the rows actually reset.
    """
    import torch

    values = values[reset_ids].double()
    flat = values.flatten()
    indices = torch.arange(len(flat))[torch.isfinite(flat)]
    finite = flat[indices]
    result = dict(object_sample_count=int(len(flat)), nonfinite_count=int(len(flat) - len(finite)))
    if not len(finite):
        return result
    quantiles = torch.quantile(finite, finite.new_tensor([.5, .95, .99]))
    result.update(p50=float(quantiles[0]), p95=float(quantiles[1]), p99=float(quantiles[2]),
                  max=float(finite.max()))

    def sample(index):
        row, column = divmod(int(index), len(active_ids))
        return dict(env_id=int(reset_ids[row]), pool_id=active_ids[column],
                    asset=asset_names[active_ids[column]], value=float(flat[index]))

    result["max_sample"] = sample(indices[finite.argmax()])
    result["p50_sample"] = sample(indices[(finite - quantiles[0]).abs().argmin()])
    result["p95_sample"] = sample(indices[(finite - quantiles[1]).abs().argmin()])
    env_max = values.nan_to_num(nan=-math.inf).amax(-1)
    finite_env = env_max[torch.isfinite(env_max)]
    q_env = torch.quantile(finite_env, finite_env.new_tensor([.5, .95, .99]))
    result["per_env_peak"] = dict(p50=float(q_env[0]), p95=float(q_env[1]), p99=float(q_env[2]),
                                  max=float(finite_env.max()), environment_count=int(len(finite_env)))
    if keep_env_values:
        result["per_env_peak"]["env_ids"] = reset_ids.tolist()
        result["per_env_peak"]["values"] = env_max.tolist()
    return result


def run(args, report):
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_conjugate, quat_mul

    import contact_manipulation_symmetric  # noqa: F401
    from contact_manipulation_symmetric.config.ur5e.scene import ARM_JOINT_NAMES, OBJECT_SPECS

    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.seed, cfg.scene.num_envs, cfg.sim.device = args.seed, args.num_envs, args.device
    cfg.spawn_rules.spawn_height_offset = args.spawn_height_offset
    cfg.sim.render_interval = 2**31 - 1
    cfg.curriculum.scene_object_count = None
    cfg.spawn_object_count_range = None
    cfg.scene_object_curriculum_start_iteration = None
    for name in ("role_obb_debug_vis", "observation_obb_debug_vis", "access_region_debug_vis",
                 "relocation_region_debug_vis", "sensor_monitor_debug_vis", "tensorboard_sensor_monitor"):
        setattr(cfg, name, False)
    if args.contact_pairs:
        for asset in cfg.scene.object_collection.rigid_objects.values():
            asset.spawn.activate_contact_sensors = True
    report.update(task=args.task, seed=args.seed, num_envs=args.num_envs,
        spawn_height_offset_m=args.spawn_height_offset, physics_dt_s=cfg.sim.dt,
        policy_decimation=cfg.decimation, robot_mode=args.robot_mode,
        robot_height_shift_m=args.robot_height_shift if args.robot_mode == "parked" else 0.0,
        contact_pairs=args.contact_pairs, contact_env_ids=args.contact_envs,
        drop_depth_m=args.drop_depth, phases=[],
        movement_windows_s={"early": [0., args.settle_start],
                            "settled": [args.settle_start, args.settle_end]},
        interpretation={
            "support_gap": "Minimum authored collision-support vertex height in the CURRENT shelf link frame, minus its support plane. Not a cooked-shape measurement.",
            "conservative_gap": "support_gap minus the authored primitive approximation error bound.",
            "z_velocity": "Native rigid-body COM world-Z velocity, m/s.",
            "contacts": "Optional native object-to-shelf / object-to-robot normal-force vector sums, N. Step zero is invalid after reset. Tactile alone does not exclude non-pad robot contacts.",
            "control": "Native production actions/controllers at configured decimation; no policy, reward, termination or automatic reset. Parked mode changes only reset robot root translation.",
            "object_count": "Object-count curriculum disabled, using the configured full clutter range.",
            "movement_windows": "Statistics include only reset-selected environment/object pairs. Each sample is a window peak or accumulated path, as its metric name states. Per-object percentiles differ from percentiles of each environment's maximum.",
            "tilt": "Angle of canonical upright +Z, excluding yaw. Shelf-relative tilt follows the current shelf plane. Initial/window tilt changes compare upright directions, not Euler roll/pitch. Angles use atan2(cross norm, dot) to resolve small motion.",
            "rebound": "Maximum increase in root world Z after an earlier running minimum within the measurement window; distinct from the intentional initial downward height offset.",
        })
    wrapped = gym.make(args.task, cfg=cfg)
    env = wrapped.unwrapped
    pair_view, pair_rows = None, []
    try:
        all_ids = torch.arange(env.num_envs, device=env.device)
        partial_ids = all_ids[1::2] if env.num_envs > 1 else all_ids
        objects, shelf, robot = (env.scene[name] for name in ("object_collection", "shelf", "robot"))
        arm_ids = torch.tensor([robot.joint_names.index(name) for name in ARM_JOINT_NAMES], device=env.device)
        actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

        def native_pose(asset):
            raw = asset.root_physx_view.get_transforms()
            if asset is objects:
                raw = objects.reshape_view_to_data(raw)
            return torch.cat((raw[..., :3], raw[..., 6:7], raw[..., 3:6]), -1).clone()

        def make_pair_view():
            # Only selected environments, all pool slots; the view survives mask changes.
            paths, filters, rows = [], [], []
            for pool_id in range(objects.num_objects):
                for env_id in args.contact_envs:
                    path = str(objects.root_physx_view.prim_paths[pool_id * env.num_envs + env_id])
                    prefix = env.scene.env_prim_paths[env_id]
                    paths.append(path)
                    filters.append([str(shelf.root_physx_view.prim_paths[env_id]), prefix + "/Robot/*"])
                    rows.append((env_id, pool_id))
            view = robot._physics_sim_view.create_rigid_contact_view(paths, filter_patterns=filters,
                                                                    max_contact_data_count=0)
            if list(map(str, view.sensor_paths)) != paths or view.filter_count != 2:
                raise RuntimeError("Native contact-pair row/filter mapping differs from the requested mapping.")
            return view, rows

        previous_active = None
        for scene_index in range(args.scenes):
            if scene_index:
                env._episode_reset.next_scene_step = env.common_step_counter
            before_full = native_pose(objects) if scene_index else None
            env.reset()
            if not scene_index:
                geometries = env.spawn_geometries
                report["geometry"] = [dict(pool_id=i, name=spec.name, asset=spec.asset_name,
                    lower=list(geometries[i].lower), upper=list(geometries[i].upper),
                    support_vertex_count=len(geometries[i].support_vertices),
                    support_error_m=geometries[i].support_error)
                    for i, spec in enumerate(OBJECT_SPECS)]
                if args.contact_pairs:
                    pair_view, pair_rows = make_pair_view()
            sample = env._episode_reset.shared_sample
            active_ids = list(sample.active_ids)
            newly_dynamic = sorted(set(active_ids) - set(previous_active)) if previous_active is not None else []
            for kind, reset_ids in (("full", all_ids), ("partial", partial_ids)):
                if kind == "partial" and args.skip_partial:
                    continue
                before_partial = native_pose(objects) if kind == "partial" else before_full
                if kind == "partial":
                    env.reset(env_ids=reset_ids)
                    if env._episode_reset.shared_sample is not sample:
                        raise RuntimeError("Partial reset unexpectedly changed the shared layout.")
                if args.robot_mode == "parked":
                    pose = robot.data.root_link_pose_w[reset_ids].clone()
                    pose[:, 2] += args.robot_height_shift
                    robot.write_root_pose_to_sim(pose, env_ids=reset_ids)
                    env.sim.forward()
                phase = dict(kind=kind, scene_index=scene_index,
                    scene_generation=int(env._episode_reset.scene_generation),
                    reset_env_ids=reset_ids.cpu().tolist(), active_ids=active_ids,
                    newly_dynamic_ids=newly_dynamic if kind == "full" else [],
                    target_id=int(sample.target_id), blocker_id=int(sample.blocker_id),
                    activation_flags=[bool(v) for v in objects._active_flags], series=[], traces=[])
                report["phases"].append(phase)
                active = torch.tensor(active_ids, device=env.device)
                pose = native_pose(objects)
                expected = env._episode_reset.shared_template.object_state[0, :, :7].clone()
                expected = expected[None].expand(len(reset_ids), -1, -1).clone()
                expected[..., :3] += env.scene.env_origins[reset_ids, None]
                position_error = (pose[reset_ids, :, :3] - expected[..., :3]).norm(dim=-1)
                phase["initial_pose_error_m"] = float(position_error.max())
                phase["initial_active_pose_error_m"] = float(position_error[:, active].max())
                native_q, expected_q = pose[reset_ids, :, 3:7], expected[..., 3:7]
                phase["initial_quaternion_component_error"] = float(torch.minimum(
                    (native_q - expected_q).norm(dim=-1), (native_q + expected_q).norm(dim=-1)).max())
                phase["initial_cached_vs_native_pose_error_m"] = float(
                    (objects.data.object_link_pos_w - pose[..., :3]).norm(dim=-1).max())
                if kind == "partial" and len(reset_ids) < env.num_envs:
                    untouched = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                    untouched[reset_ids] = False
                    phase["unselected_pose_error_m"] = float(
                        (pose[untouched, :, :3] - before_partial[untouched, :, :3]).norm(dim=-1).max())
                selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                selected[reset_ids] = True
                vertices = env.task_scene_data.support_vertices[active]
                vertex_mask = env.task_scene_data.support_vertex_mask[active]
                errors = env.task_scene_data.support_error[active]
                upright = env.task_scene_data.upright[active]
                # Convert the original measured panel point to the shelf body frame.
                initial_shelf = shelf.data.default_root_state[:, :7].clone()
                initial_shelf[:, :3] += env.scene.env_origins
                panel = pose.new_tensor([sum(env.spawn_rules.x_range) / 2,
                                         sum(env.spawn_rules.y_range) / 2, env.spawn_rules.shelf_top])
                panel = panel[None] + env.scene.env_origins
                panel_z = quat_apply_inverse(initial_shelf[:, 3:7], panel - initial_shelf[:, :3])[:, 2]
                shelf_start = native_pose(shelf)
                object_start = pose[:, active].clone()
                initial_canonical = quat_mul(object_start[..., 3:7],
                    quat_conjugate(upright)[None].expand_as(object_start[..., 3:7]))
                up_vector = object_start.new_tensor([0., 0., 1.]).expand(env.num_envs, len(active_ids), -1)
                initial_up = quat_apply(initial_canonical, up_vector)
                movement = {}
                history, contact_history = [], []
                asset_groups = {}
                for column, pool_id in enumerate(active_ids):
                    asset_groups.setdefault(OBJECT_SPECS[pool_id].asset_name, []).append(column)

                for step in range(args.physics_steps + 1):
                    if step:
                        if (step - 1) % cfg.decimation == 0:
                            env.action_manager.process_action(actions)
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(dt=cfg.sim.dt)
                    poses = native_pose(objects)[:, active]
                    velocities = objects.reshape_view_to_data(objects.root_physx_view.get_velocities())[:, active]
                    shelf_pose = native_pose(shelf)
                    normal = quat_apply(shelf_pose[:, 3:7], shelf_pose.new_tensor([0., 0., 1.]).expand(env.num_envs, -1))
                    canonical_rotation = quat_mul(poses[..., 3:7],
                        quat_conjugate(upright)[None].expand_as(poses[..., 3:7]))
                    axis = quat_apply_inverse(canonical_rotation, normal[:, None].expand(-1, len(active_ids), -1))
                    projections = torch.einsum("nad,avd->nav", axis, vertices)
                    minimum = projections.masked_fill(~vertex_mask[None], math.inf).amin(-1)
                    root_height = quat_apply_inverse(
                        shelf_pose[:, None, 3:7].expand(-1, len(active_ids), -1),
                        poses[..., :3] - shelf_pose[:, None, :3])[..., 2]
                    gap = root_height + minimum - panel_z[:, None]
                    conservative_gap = gap - errors[None]
                    up = quat_apply(canonical_rotation, up_vector)
                    shelf_up = normal[:, None].expand_as(up)

                    def up_angle(first, second):
                        return torch.atan2(torch.linalg.cross(first, second).norm(dim=-1),
                                           (first * second).sum(-1))

                    shelf_tilt = up_angle(up, shelf_up)
                    initial_tilt = up_angle(up, initial_up)
                    linear_speed = velocities[..., :3].norm(dim=-1)
                    angular_speed = velocities[..., 3:6].norm(dim=-1)
                    time_s = step * cfg.sim.dt
                    for window_name, (start_s, end_s) in report["movement_windows_s"].items():
                        if not start_s - 1.e-9 <= time_s <= end_s + 1.e-9:
                            continue
                        if window_name not in movement:
                            movement[window_name] = dict(start_step=step, end_step=step, frame_count=0,
                                anchor_position=poses[..., :3].clone(), anchor_up=up.clone(),
                                previous_position=poses[..., :3].clone(), minimum_z=poses[..., 2].clone(),
                                metrics={name: torch.zeros_like(gap) for name in (
                                    "peak_COM_linear_speed_m_s", "peak_angular_speed_rad_s",
                                    "peak_tilt_from_initial_deg", "peak_tilt_from_window_start_deg",
                                    "peak_shelf_relative_tilt_deg", "max_root_displacement_m",
                                    "max_root_XY_displacement_m", "root_path_length_m", "root_Z_rebound_m")})
                        window = movement[window_name]
                        metrics = window["metrics"]
                        displacement = poses[..., :3] - window["anchor_position"]
                        peaks = dict(peak_COM_linear_speed_m_s=linear_speed,
                            peak_angular_speed_rad_s=angular_speed,
                            peak_tilt_from_initial_deg=torch.rad2deg(initial_tilt),
                            peak_tilt_from_window_start_deg=torch.rad2deg(up_angle(up, window["anchor_up"])),
                            peak_shelf_relative_tilt_deg=torch.rad2deg(shelf_tilt),
                            max_root_displacement_m=displacement.norm(dim=-1),
                            max_root_XY_displacement_m=displacement[..., :2].norm(dim=-1))
                        for name, value in peaks.items():
                            torch.maximum(metrics[name], value, out=metrics[name])
                        metrics["root_path_length_m"] += (poses[..., :3] - window["previous_position"]).norm(dim=-1)
                        torch.minimum(window["minimum_z"], poses[..., 2], out=window["minimum_z"])
                        torch.maximum(metrics["root_Z_rebound_m"], poses[..., 2] - window["minimum_z"],
                                      out=metrics["root_Z_rebound_m"])
                        window["previous_position"].copy_(poses[..., :3])
                        window["end_step"], window["frame_count"] = step, window["frame_count"] + 1
                    tactile = env.task_sensor_reader()
                    tactile_max = torch.linalg.vector_norm(tactile.tactile_force, dim=-1).amax(-1)
                    arm_speed = robot.data.joint_vel[:, arm_ids].abs().amax(-1)
                    links = robot.root_physx_view.get_link_transforms()
                    robot_bottom = links[..., 2].amin(-1)
                    active_top_z = poses[..., 2].amax(-1)
                    # Origins-only isolation witness: not a collision-distance query.
                    isolation = robot_bottom - active_top_z
                    tilt = torch.acos(normal[:, 2].clamp(-1., 1.))
                    shelf_displacement = (shelf_pose[:, :3] - shelf_start[:, :3]).norm(dim=-1)
                    finite = torch.isfinite(poses).all(-1) & torch.isfinite(velocities).all(-1) & torch.isfinite(gap)
                    row = dict(physics_step=step, time_s=step * cfg.sim.dt,
                        support_gap_min_m=float(gap.min()), conservative_gap_min_m=float(conservative_gap.min()),
                        selected_support_gap_min_m=float(gap[selected].min()),
                        below_support_3mm_count=int((gap < -.003).sum()),
                        drop_count=int((gap < -args.drop_depth).sum()),
                        selected_drop_count=int((gap[selected] < -args.drop_depth).sum()),
                        object_vz_min_m_s=float(velocities[..., 2].min()),
                        object_xy_displacement_max_m=float((poses[..., :2] - object_start[..., :2]).norm(dim=-1).max()),
                        COM_linear_speed_max_m_s=float(linear_speed[selected].max()),
                        angular_speed_max_rad_s=float(angular_speed[selected].max()),
                        initial_up_tilt_change_max_deg=float(torch.rad2deg(initial_tilt[selected]).max()),
                        shelf_relative_tilt_max_deg=float(torch.rad2deg(shelf_tilt[selected]).max()),
                        shelf_displacement_max_m=float(shelf_displacement.max()),
                        shelf_tilt_max_rad=float(tilt.max()),
                        tactile_force_max_N=float(tactile_max.max()), tactile_valid=bool(step),
                        arm_speed_max_rad_s=float(arm_speed.max()),
                        robot_min_body_origin_z_minus_object_max_origin_z_m=float(isolation.min()),
                        nonfinite_object_count=int((~finite).sum()),
                        asset_min_gap_m={name: float(gap[:, columns].min()) for name, columns in asset_groups.items()})
                    contact = None
                    if pair_view is not None and step:
                        contact = pair_view.get_contact_force_matrix(dt=cfg.sim.dt).norm(dim=-1).cpu()
                        enabled = torch.tensor([pool_id in active_ids for _, pool_id in pair_rows])
                        row["object_shelf_contact_max_N"] = float(contact[enabled, 0].max())
                        row["object_robot_contact_max_N"] = float(contact[enabled, 1].max())
                        row["object_shelf_contact_rows"] = int((contact[enabled, 0] > .001).sum())
                    phase["series"].append(row)
                    # Bound diagnostic memory to one phase; only a few traces enter JSON.
                    history.append(torch.cat((poses, velocities[..., 2:3], gap[..., None],
                        conservative_gap[..., None], linear_speed[..., None], angular_speed[..., None],
                        shelf_tilt[..., None], initial_tilt[..., None]), -1).cpu())
                    contact_history.append(contact)
                    if not bool(finite.all()):
                        break
                history = torch.stack(history)
                worst_gap, worst_step = history[..., 8].min(0)
                phase["movement_windows"] = {}
                movement_trace_candidates = []
                reset_ids_cpu = reset_ids.cpu()
                for window_name, window in movement.items():
                    start_s, end_s = report["movement_windows_s"][window_name]
                    summary = dict(requested_start_s=start_s, requested_end_s=end_s,
                        actual_start_s=window["start_step"] * cfg.sim.dt,
                        actual_end_s=window["end_step"] * cfg.sim.dt,
                        complete=window["end_step"] * cfg.sim.dt >= end_s - 1.e-9,
                        frame_count=window["frame_count"], reset_env_count=len(reset_ids),
                        active_object_count=len(active_ids), metrics={})
                    for name, values in window["metrics"].items():
                        summary["metrics"][name] = movement_statistics(values.cpu(), reset_ids_cpu,
                            active_ids, [spec.asset_name for spec in OBJECT_SPECS],
                            keep_env_values=name in ("peak_angular_speed_rad_s",
                                                    "peak_tilt_from_window_start_deg", "root_Z_rebound_m"))
                    phase["movement_windows"][window_name] = summary
                    for name in ("peak_angular_speed_rad_s", "peak_tilt_from_window_start_deg", "root_Z_rebound_m"):
                        sample_row = summary["metrics"][name].get("max_sample")
                        if sample_row:
                            movement_trace_candidates.append(sample_row["env_id"] * len(active_ids)
                                + active_ids.index(sample_row["pool_id"]))
                ordered = worst_gap.flatten().argsort().tolist()
                # Cover different failed pool slots before duplicating one slot
                # across many clones. Keep the very worst row as the first trace.
                chosen = list(dict.fromkeys([ordered[0], *movement_trace_candidates]))[:args.max_traces]
                covered = {flat % len(active_ids) for flat in chosen}
                for flat in ordered:
                    pool_column = flat % len(active_ids)
                    if pool_column not in covered:
                        chosen.append(flat)
                        covered.add(pool_column)
                    if len(chosen) == args.max_traces:
                        break
                if len(chosen) < args.max_traces:
                    chosen += [flat for flat in ordered if flat not in chosen][:args.max_traces - len(chosen)]
                pair_lookup = {item: index for index, item in enumerate(pair_rows)}
                for flat in chosen:
                    env_id, column = divmod(flat, len(active_ids))
                    pool_id = active_ids[column]
                    trace = dict(env_id=env_id, pool_id=pool_id, name=OBJECT_SPECS[pool_id].name,
                        asset=OBJECT_SPECS[pool_id].asset_name,
                        role="target" if pool_id == sample.target_id else "blocker" if pool_id == sample.blocker_id else "other",
                        reset_selected=bool(selected[env_id]), newly_dynamic=pool_id in phase["newly_dynamic_ids"],
                        minimum_gap_m=float(worst_gap[env_id, column]),
                        minimum_gap_step=int(worst_step[env_id, column]),
                        columns=["x_m", "y_m", "z_m", "qw", "qx", "qy", "qz", "COM_vz_m_s",
                                 "support_gap_m", "conservative_gap_m", "COM_linear_speed_m_s",
                                 "angular_speed_rad_s", "shelf_relative_tilt_rad", "initial_up_tilt_change_rad"],
                        frames=history[:, env_id, column].tolist())
                    pair_row = pair_lookup.get((env_id, pool_id))
                    if pair_row is not None:
                        trace["contact_columns"] = ["object_shelf_normal_sum_N", "object_robot_normal_sum_N"]
                        trace["contacts"] = [None if item is None else item[pair_row].tolist()
                                             for item in contact_history]
                    phase["traces"].append(trace)
                phase["summary"] = dict(physics_steps=len(history) - 1,
                    minimum_gap_m=float(worst_gap.min()),
                    dropped_environment_count=int((worst_gap < -args.drop_depth).any(-1).sum()),
                    dropped_env_object_count=int((worst_gap < -args.drop_depth).sum()),
                    drop_counts_by_env=(worst_gap < -args.drop_depth).sum(-1).tolist(),
                    drop_counts_by_active_pool_id={str(pool_id): int((worst_gap[:, column] < -args.drop_depth).sum())
                        for column, pool_id in enumerate(active_ids)},
                    max_robot_contact_N=max((item.get("object_robot_contact_max_N", 0.) for item in phase["series"]), default=0.) if args.contact_pairs else None,
                    max_tactile_force_N=max(item["tactile_force_max_N"] for item in phase["series"]))
                save_report(args.output, report)
                print(f"[Support probe] scene={scene_index} {kind} gap={float(worst_gap.min()):+.6f}m "
                      f"dropped={phase['summary']['dropped_env_object_count']} / {env.num_envs * len(active_ids)}", flush=True)
            previous_active = active_ids
    finally:
        wrapped.close()


if __name__ == "__main__":
    arguments = parse_args()
    report = dict(status="running", command_line=sys.argv)
    app = AppLauncher(arguments).app
    try:
        run(arguments, report)
        report["status"] = "complete"
    except BaseException:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        save_report(arguments.output, report)
        app.close(wait_for_replicator=False, skip_cleanup=True)
