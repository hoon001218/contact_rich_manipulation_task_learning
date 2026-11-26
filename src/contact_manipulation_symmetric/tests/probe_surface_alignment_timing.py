"""Time a contact-independent nearest-Blocker-face reward in a real Isaac Lab batch."""

import argparse
import json
import math
from pathlib import Path
import time

from probe_ft_compensation import AppLauncher, REPOSITORY_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--step-block", type=int, default=20)
    parser.add_argument("--exclude-fingertips", action="store_true", help="Use the original 12-pad benchmark.")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT /
                        "logs/diagnostics/surface_alignment_timing_2026-11-26/production_17_pads.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.step_block < 1:
        parser.error("Environment and block counts must be positive.")
    return args


def run(args, report):
    import gymnasium as gym
    import numpy as np
    import torch
    import trimesh
    import warp as wp
    from pxr import Usd, UsdPhysics

    import contact_manipulation_symmetric  # noqa: F401, register after AppLauncher
    from contact_manipulation_symmetric.config.ur5e.scene import FINGERTIP_BODY_NAMES, OBJECT_SPECS, TACTILE_BODY_NAMES
    from contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg
    from contact_manipulation_symmetric.mdp import rewards
    from contact_manipulation_symmetric.mdp.collision_surfaces import collision_mesh
    from surface_alignment_kernel import SurfaceAlignmentKernel

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if not args.device.startswith("cuda"):
        raise ValueError("Use a CUDA simulation device to measure GPU overhead.")
    device = torch.device(args.device)
    wp.init()
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))

    def gpu_mesh(points, triangles):
        return wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=str(device)),
                       indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=str(device)))

    def mesh_ids(mesh, count):
        return wp.array(np.full(count, mesh.id, dtype=np.uint64), dtype=wp.uint64, device=str(device))

    def timing_stats(values):
        return dict(median_ms=float(np.median(values)), mean_ms=float(np.mean(values)),
                    min_ms=float(min(values)), max_ms=float(max(values)), samples_ms=values)

    def time_calls(cases, *, loops=192, rounds=7, graphed=False):
        for fn in cases.values():
            for _ in range(8):
                fn()
        torch.cuda.synchronize(device)
        graphs = {}
        if graphed:
            for name, fn in cases.items():
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_stream = wp.stream_from_torch(torch.cuda.current_stream(device))
                    for _ in range(loops):
                        fn(graph_stream)
                graphs[name] = graph
        gpu_samples, wall_samples = ({name: [] for name in cases} for _ in range(2))
        names = list(cases)
        for i in range(rounds):
            order = names[i % len(names):] + names[:i % len(names)]
            for name in order:
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                wall = time.perf_counter()
                start.record()
                if graphed:
                    graphs[name].replay()
                else:
                    for _ in range(loops):
                        cases[name]()
                end.record()
                end.synchronize()
                wall_samples[name].append((time.perf_counter() - wall) * 1000. / loops)
                gpu_samples[name].append(start.elapsed_time(end) / loops)
        return {name: {"cuda": timing_stats(gpu_samples[name]), "wall": timing_stats(wall_samples[name])}
                for name in cases}

    cfg = UR5eContactManipulationEnvCfg()
    term = cfg.rewards.blocker_normal_alignment
    decay = term.params["decay"]
    selected = [i for i, name in enumerate(TACTILE_BODY_NAMES)
                if not args.exclude_fingertips or name not in FINGERTIP_BODY_NAMES]
    num_pads = len(selected)
    pad_ids = torch.tensor(selected, dtype=torch.long, device=device)
    report.update(num_envs=args.num_envs, num_pads=num_pads, queries_per_policy_step=args.num_envs*num_pads,
                  decay=decay, distance_weight="constant 1; no distance cutoff within shelf workspace",
                  gpu=torch.cuda.get_device_name(device), warp_version=wp.__version__, status="running")

    # One outward plane isolates the angle, lack of distance weighting, rigid transform, and denominator.
    cube = trimesh.creation.box(extents=[.1, .1, .1])
    cube_mesh = gpu_mesh(cube.vertices, cube.faces)
    pts = torch.zeros((6, num_pads, 3), device=device)
    pts[..., 0] = pts.new_tensor([.051, .5, .051, .051, .051, .051])[:, None]
    normals = torch.zeros_like(pts)
    normals[..., 0] = -1.
    normals[2, :, :2] = normals.new_tensor([-math.sqrt(.5), math.sqrt(.5)])
    normals[3, :, :2] = normals.new_tensor([0., 1.])
    normals[4, :, :2] = normals.new_tensor([1., 0.])
    normals[5, 0] = 0.
    poses = torch.zeros((6, 7), device=device)
    # Check the correct normals after a 90-degree world rotation and a nonzero origin.
    poses[:, :3] = poses.new_tensor([2., 3., 1.])
    poses[:, 3] = poses[:, 6] = math.sqrt(.5)
    pts = torch.stack((-pts[..., 1], pts[..., 0], pts[..., 2]), dim=-1) + poses[:, None, :3]
    normals = torch.stack((-normals[..., 1], normals[..., 0], normals[..., 2]), dim=-1)
    expected = pts.new_tensor([1., 1., .5, math.exp(-decay), math.exp(-2.*decay), (num_pads-1)/num_pads])
    for variant in ("parallel", "serial"):
        score = SurfaceAlignmentKernel(pts, normals, poses, mesh_ids(cube_mesh, len(pts)), decay, variant=variant)
        torch.testing.assert_close(score.launch(stream), expected, atol=1.e-5, rtol=1.e-5)
    report["analytic_plane_check"] = expected.cpu().tolist()
    save()

    cfg.seed, cfg.scene.num_envs, cfg.sim.device = 42, args.num_envs, args.device
    cfg.layout_scene_override = (-1, "direct", "direct")
    cfg.sim.render_interval = 2**31 - 1
    env_wrapper = gym.make("Isaac-Contact-Manipulation-Symmetric-UR5e-v0", cfg=cfg)
    try:
        env = env_wrapper.unwrapped
        env.reset()
        print(f"[Surface] Reset {env.num_envs} production environments", flush=True)
        objects = env.scene["object_collection"]
        paths = list(objects.root_physx_view.prim_paths)
        actual = int(env.task_scene_data.blocker_id[0])
        chosen = OBJECT_SPECS[actual].asset_name
        report["live_blocker"] = chosen
        stage = env.scene.stage
        meshes, bounds, mesh_info = {}, {}, {}
        for i, spec in enumerate(OBJECT_SPECS):
            if spec.asset_name in meshes:
                continue
            root = stage.GetPrimAtPath(str(paths[i * env.num_envs]))
            if not root.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                          if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                (root,) = bodies
            vertices, triangles, info = collision_mesh(stage, root)
            meshes[spec.asset_name] = gpu_mesh(vertices, triangles)
            bounds[spec.asset_name] = np.stack((vertices.min(0), vertices.max(0)))
            mesh_info[spec.asset_name] = info
            print(f"[Surface] {spec.asset_name}: {len(triangles)} collision triangles", flush=True)
        report["geometry"] = mesh_info
        save()

        points_buf = torch.empty((env.num_envs, num_pads, 3), device=device)
        normals_buf = torch.empty_like(points_buf)
        poses_buf = torch.empty((env.num_envs, 7), device=device)

        def read_live_geometry():
            points, normals = rewards.tactile_pad_frames(env)
            torch.index_select(points, 1, pad_ids, out=points_buf)
            torch.index_select(normals, 1, pad_ids, out=normals_buf)
            indices = env.task_scene_data.blocker_id[:, None, None].expand(-1, 1, 7)
            torch.gather(objects.data.object_link_pose_w, 1, indices, out=poses_buf[:, None, :])

        read_live_geometry()
        live_kernels = {variant: SurfaceAlignmentKernel(points_buf, normals_buf, poses_buf,
                        mesh_ids(meshes[chosen], env.num_envs), decay, variant=variant) for variant in ("parallel", "serial")}
        live_cases = {name: (lambda launch_stream=None, kernel=kernel: kernel.launch(launch_stream or stream))
                      for name, kernel in live_kernels.items()}
        torch.testing.assert_close(live_cases["parallel"](), live_cases["serial"](), atol=1.e-5, rtol=1.e-5)
        report["live_geometry_kernel"] = time_calls(live_cases, graphed=True)
        live_fn = {}
        for name, launch in live_cases.items():
            def full(launch=launch):
                read_live_geometry()
                return launch()
            live_fn[name] = full
        baseline = lambda: rewards.blocker_normal_alignment(env, decay=decay, exclude_fingertips=args.exclude_fingertips)
        term = env.reward_manager.get_term_cfg("blocker_normal_alignment")
        if not args.exclude_fingertips:
            if term.func is not rewards.blocker_surface_normal_alignment:
                raise RuntimeError("The native reward manager is not using the surface reward.")
            live_fn["production"] = lambda: term.func(env, **term.params)
            torch.testing.assert_close(live_fn["production"](), live_cases["parallel"](), atol=1.e-5, rtol=1.e-5)
            velocity = env.reward_manager.get_term_cfg("motion_joint_velocity")
            asset_cfg = velocity.params["asset_cfg"]
            joint_names = env.scene[asset_cfg.name].joint_names
            report["velocity_reward_joints"] = [joint_names[index] for index in asset_cfg.joint_ids]
            if set(report["velocity_reward_joints"]) != set(cfg.actions.arm_action.joint_names + cfg.actions.hand_action.joint_names):
                raise RuntimeError("The native velocity penalty must select all 12 policy-controlled joints.")
        report["full_reward_term"] = time_calls({"center": baseline, **live_fn})
        print("[Surface] Full term timings complete", flush=True)
        save()

        # Exercise each real Blocker type at multiple distances with identical query counts.
        # Shelf simulation itself always shares one asset type across the batch.
        local_box_rng = np.random.default_rng(42)
        pad_positions = local_box_rng.uniform(-1., 1., size=(env.num_envs, num_pads, 3))
        face_axes = local_box_rng.integers(0, 3, size=pad_positions.shape[:2])
        axis_mask = np.eye(3)[face_axes]
        pad_gaps = local_box_rng.uniform(.002, .25, size=pad_positions.shape[:2] + (1,))
        native_normals = np.where(pad_positions >= 0., -1., 1.) * axis_mask
        workload = {}
        for name, mesh in meshes.items():
            mid, half = bounds[name].mean(0), (bounds[name][1] - bounds[name][0]) * .5
            edge_positions = pad_positions * (1. - axis_mask) - native_normals
            points_buf.copy_(points_buf.new_tensor(mid + half * edge_positions - native_normals * pad_gaps))
            normals_buf.copy_(normals_buf.new_tensor(native_normals))
            poses_buf.zero_()
            poses_buf[:, 3] = 1.
            ids = mesh_ids(mesh, env.num_envs)
            kernels = {
                variant: SurfaceAlignmentKernel(points_buf, normals_buf, poses_buf, ids, decay, variant=variant)
                for variant in live_kernels
            }
            asset_cases = {
                variant: (lambda launch_stream=None, kernel=kernel: kernel.launch(launch_stream or stream))
                for variant, kernel in kernels.items()
            }
            scores = asset_cases["parallel"]()
            torch.testing.assert_close(scores, asset_cases["serial"](), atol=1.e-5, rtol=1.e-5)
            if not bool(torch.isfinite(scores).all() and (scores > 0.).all() and (scores <= 1.).all()):
                raise RuntimeError(f"Invalid nearest surface score for {name}.")
            workload[name] = {variant: result["cuda"] for variant, result in
                              time_calls(asset_cases, graphed=True, loops=128, rounds=5).items()}
        report["all_asset_kernels_cuda"] = workload
        save()

        # Interleave baseline and prototype on the same continuous fixed-action rollout.
        # The reward is diagnostic only and does not drive these actions.
        mgr = env.reward_manager
        original = mgr.get_term_cfg("blocker_normal_alignment")
        substitutes = {"center": original.replace(
            func=lambda current_env: baseline(), params={},
        )}
        for name, fn in live_fn.items():
            substitutes[name] = original if name == "production" else original.replace(
                func=lambda current_env, fn=fn: fn(), params={},
            )
        action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=device)
        step_samples = {name: [] for name in substitutes}
        for _ in range(6):
            env.step(action)
        torch.cuda.synchronize(device)
        names = list(substitutes)
        rounds = 6
        steps = min(args.step_block, (env.max_episode_length - 7) // (len(names) * rounds))
        if steps < 1:
            raise RuntimeError("A shared scene is too short for this timing comparison.")
        for i in range(rounds):
            ordered = names if i < len(names) else names[::-1]
            offset = i % len(names)
            order = ordered[offset:] + ordered[:offset]
            for mode in order:
                mgr.set_term_cfg("blocker_normal_alignment", substitutes[mode])
                started = time.perf_counter()
                for _ in range(steps):
                    env.step(action)
                torch.cuda.synchronize(device)
                step_samples[mode].append((time.perf_counter() - started) * 1000. / steps)
        mgr.set_term_cfg("blocker_normal_alignment", original)
        if not bool((env.task_scene_data.blocker_id == actual).all()):
            raise RuntimeError("The scene changed in a timed block.")
        report["environment_step"] = {name: timing_stats(values) for name, values in step_samples.items()}
        report["steps_per_block"] = steps
        report["episode_length_range_after_timing"] = [
            int(env.episode_length_buf.min()), int(env.episode_length_buf.max()),
        ]
        report["policy_step_dt_s"] = env.step_dt
        report["status"] = "complete"
        save()
    finally:
        env_wrapper.close()


if __name__ == "__main__":
    import traceback

    args = parse_args()
    app = AppLauncher(args).app
    try:
        run(args, {})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)
