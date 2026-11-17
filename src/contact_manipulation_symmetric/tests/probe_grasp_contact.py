"""Record real tactile coverage while the unchanged assembled hand closes.

The production robot, implicit hand drives, mapped position actions, wrench
OSC, gravity, contacts and physics step are reused. A 300 g sphere rests on a
diagnostic support. Only initial states are written between independent trials.
The arm receives zero sensor-frame wrench with its normal gravity compensation.
Object-filtered contact is recorded separately from the observation's net-force
binary so self-contact and support contact cannot inflate the grasp pad count.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

from probe_ft_compensation import AppLauncher, REPOSITORY_ROOT, smooth_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "logs/diagnostics/grasp_contact_2026-11-16")
    parser.add_argument("--video-output", type=Path, default=REPOSITORY_ROOT / "docs/videos/grasp_contact_2026-11-16/grasp.mp4")
    parser.add_argument("--diameters", type=float, nargs="+", default=[.06])
    parser.add_argument("--mass", type=float, default=.3)
    parser.add_argument("--fixed-object", action="store_true", help="Optional fixed-object diagnostic control.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--along-fingers", type=float, default=.06, help="Sphere center's palm-tangent offset in m.")
    parser.add_argument("--surface-gap", type=float, default=.003, help="Initial palm/sphere surface clearance in m.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if any(not math.isfinite(x) or not .02 <= x <= .14 for x in args.diameters):
        parser.error("Diameters must lie in [0.02,0.14] m.")
    if args.mass <= 0 or args.video_fps <= 0:
        parser.error("Mass and FPS must be positive.")
    if not args.no_video:
        args.enable_cameras = True
    return args


def summarize(samples, metadata):
    import numpy as np

    trials = []
    for trial_id, obj in enumerate(metadata["objects"]):
        rows = [s for s in samples if s["trial_index"] == trial_id]
        if not rows:
            continue
        bits = np.asarray([s["object_binary"] for s in rows], dtype=bool)
        net = np.asarray([s["binary"] for s in rows], dtype=bool)
        counts = bits.sum(-1)
        hold = np.asarray([s["phase"] == "hold" for s in rows])
        policy = np.asarray([s["policy_sample"] for s in rows])
        speeds = np.asarray([s["joint_velocity_rad_s"] for s in rows])
        limits = np.asarray(metadata["joint_velocity_limits_rad_s"])
        trials.append({"trial_index": trial_id, "object": obj, "samples": len(rows),
            "simultaneous_object_pad_peak": int(counts.max()),
            "simultaneous_object_pad_mean": float(counts.mean()),
            "simultaneous_net_pad_peak": int(net.sum(-1).max()),
            "hold_object_pad_min": int(counts[hold].min()) if hold.any() else None,
            "hold_object_pad_mean": float(counts[hold].mean()) if hold.any() else None,
            "hold_object_pad_max": int(counts[hold].max()) if hold.any() else None,
            "policy_object_pad_peak": int(counts[policy].max()) if policy.any() else None,
            "ever_object_pad_ids": np.flatnonzero(bits.any(0)).tolist(),
            "ever_object_pad_names": [name for name, on in zip(metadata["pad_names"], bits.any(0)) if on],
            "per_pad_object_contact_fraction": bits.mean(0).tolist(),
            "max_object_pad_force_N": np.asarray([s["object_pad_force_N"] for s in rows]).max(0).tolist(),
            "hand_joint_velocity_min_rad_s": speeds.min(0).tolist(),
            "hand_joint_velocity_max_rad_s": speeds.max(0).tolist(),
            "hand_joint_speed_max_rad_s": np.abs(speeds).max(0).tolist(),
            "joint_samples_over_limit_plus_1e_3": (np.abs(speeds) > limits + .001).sum(0).tolist(),
            "whole_hand_object_force_peak_N": max(sum(s["hand_body_object_force_N"].values()) for s in rows),
            "object_displacement_m": float(np.linalg.norm(np.asarray(rows[-1]["object_position_w_m"])
                                                            - np.asarray(rows[0]["object_position_w_m"])))})
    return trials


def run(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from pxr import Usd, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors import Camera, CameraCfg

    scene = importlib.import_module("contact_manipulation_symmetric.config.ur5e.scene")
    env_module = importlib.import_module("contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module("contact_manipulation_symmetric.mdp.events")
    sensors = importlib.import_module("contact_manipulation_symmetric.mdp.sensors")
    obs = importlib.import_module("contact_manipulation_symmetric.mdp.observations")
    rewards = importlib.import_module("contact_manipulation_symmetric.mdp.rewards")
    cfg = env_module.UR5eContactManipulationEnvCfg()
    robot_cfg = cfg.scene.robot.copy()
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    sim_cfg = cfg.sim.copy()
    sim_cfg.device, sim_cfg.render_interval = args.device, 2**31 - 1
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    robot = Articulation(robot_cfg)
    tactile_cfg = sensors.make_tactile_contact_cfg(scene.TACTILE_BODY_NAMES)
    tactile_cfg.prim_path = tactile_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    tactile = tactile_cfg.class_type(tactile_cfg)
    # Supports and objects are authored once, then native initial poses are
    # written between independent trials. Inactive trials are parked remotely.
    objects, supports = [], []
    for index, diameter in enumerate(args.diameters):
        obj = RigidObject(RigidObjectCfg(prim_path=f"/World/GraspObject_{index}",
            spawn=sim_utils.SphereCfg(radius=diameter / 2, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=args.fixed_object,
                    disable_gravity=args.fixed_object, max_depenetration_velocity=1.,
                    solver_position_iteration_count=16, solver_velocity_iteration_count=1),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
                mass_props=sim_utils.MassPropertiesCfg(mass=args.mass),
                physics_material=sim_cfg.physics_material.copy(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .49, .13), roughness=.7)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(index * 2., 0., -10.))))
        support = RigidObject(RigidObjectCfg(prim_path=f"/World/GraspSupport_{index}",
            spawn=sim_utils.CuboidCfg(size=(.32, .32, .025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
                physics_material=sim_cfg.physics_material.copy(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.19, .24, .29), roughness=.8)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(index * 2., 0., -12.))))
        objects.append(obj)
        supports.append(support)
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.9, .94, 1.))
    light.func("/World/GraspLight", light)
    camera = marker = frames = None
    if not args.no_video:
        for index in range(2):
            sim_utils.create_prim(f"/World/GraspViews/View_{index}", "Xform")
        camera = Camera(CameraCfg(prim_path="/World/GraspViews/View_.*/Camera", width=640, height=480,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=24.,
                                                               clipping_range=(.001, 15.))))
        marker = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/GraspPadMarkers", markers={
            key: sim_utils.SphereCfg(radius=.0015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            for key, color in {"object": (.2, .95, .4), "other": (1., .65, .12), "off": (.2, .25, .32)}.items()}))
        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/World/GraspSensorFrame"
        frame_cfg.markers = {"frame": frame_cfg.markers["frame"]}
        frame_cfg.markers["frame"].scale = (.035, .035, .035)
        frames = VisualizationMarkers(frame_cfg)
        for path in ("/World/GraspPadMarkers", "/World/GraspSensorFrame"):
            for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError("Visual grasp markers cannot contain physics schemas.")
    sim.reset()
    initial_q = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
    robot.set_joint_position_target(initial_q)
    robot.reset()
    sim.forward()
    tactile.reset()
    events.configure_hand_action_limits(SimpleNamespace(scene={"robot": robot}))

    class ProbeScene(dict):
        env_prim_paths = ["/World/envs/env_0"]

    env = SimpleNamespace(scene=ProbeScene(robot=robot, tactile_contact=tactile), num_envs=1,
                          device=sim.device, sim=sim, cfg=cfg)
    reader = sensors.SensorReader(env)
    arm_action = cfg.actions.arm_action.class_type(cfg.actions.arm_action.copy(), env)
    hand_action = cfg.actions.hand_action.class_type(cfg.actions.hand_action.copy(), env)
    pad_ids, pad_names = robot.find_bodies(list(scene.TACTILE_BODY_NAMES), preserve_order=True)
    if tuple(pad_names) != scene.TACTILE_BODY_NAMES:
        raise RuntimeError("Canonical tactile order mismatch.")
    pad_ids = torch.tensor(pad_ids, device=sim.device, dtype=torch.long)
    driven_ids = list(hand_action._driven_joint_ids)
    sensor_id = robot.body_names.index(scene.FORCE_TORQUE_BODY_NAME)
    arm_ids, _ = robot.find_joints(list(scene.ARM_JOINT_NAMES), preserve_order=True)
    body_paths = robot.root_physx_view.link_paths[0]
    hand_paths = [str(p) for p in body_paths if "/ROASRightHand/" in str(p)]
    views, body_views = [], []
    sensor_paths = list(tactile.contact_physx_view.sensor_paths)
    for obj in objects:
        view = robot._physics_sim_view.create_rigid_contact_view(sensor_paths,
            filter_patterns=[[obj.cfg.prim_path]] * 17, max_contact_data_count=0)
        if list(view.sensor_paths) != sensor_paths or view.sensor_count != 17 or view.filter_count != 1:
            raise RuntimeError("Filtered tactile witness row mapping changed.")
        views.append(view)
        body_view = robot._physics_sim_view.create_rigid_contact_view(hand_paths,
            filter_patterns=[[obj.cfg.prim_path]] * len(hand_paths), max_contact_data_count=0)
        if list(body_view.sensor_paths) != hand_paths or body_view.filter_count != 1:
            raise RuntimeError("Whole-hand contact witness row mapping changed.")
        body_views.append(body_view)
    dt = sim.get_physics_dt()
    stride = round(1 / (dt * args.video_fps))
    if stride < 1 or not math.isclose(stride * dt * args.video_fps, 1.):
        raise ValueError("FPS must divide the physics frequency.")
    phases = [("open", .6), ("thumb_opposition", 1.2), ("close", 2.), ("hold", 2.), ("release", 2.), ("open_after", .6)]
    threshold = cfg.mdp.tactile_threshold
    metadata = {"status": "running", "dt": dt, "policy_decimation": cfg.decimation,
        "command_line": sys.argv, "video_fps": args.video_fps,
        "initial_joint_names": list(robot.joint_names), "initial_joint_position_rad": initial_q[0].cpu().tolist(),
        "placement": {"along_fingers_m": args.along_fingers, "surface_gap_m": args.surface_gap},
        "threshold_N": threshold, "pad_names": list(pad_names),
        "joint_names": [robot.joint_names[i] for i in driven_ids],
        "joint_velocity_limits_rad_s": robot.root_physx_view.get_dof_max_velocities()[0, driven_ids].cpu().tolist(),
        "joint_effort_limits_Nm": robot.data.joint_effort_limits[0, driven_ids].cpu().tolist(),
        "joint_armature": robot.data.joint_armature[0, driven_ids].cpu().tolist(),
        "hand_setting": {key: getattr(robot_cfg.actuators["hand"], key) for key in
                         ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim")},
        "physics": {"dt": dt, "gravity": list(sim_cfg.gravity),
            "robot_position_iterations": robot_cfg.spawn.articulation_props.solver_position_iteration_count,
            "robot_velocity_iterations": robot_cfg.spawn.articulation_props.solver_velocity_iteration_count},
        "objects": [{"shape": "sphere", "size_m": d, "mass_kg": args.mass, "fixed": args.fixed_object}
                    for d in args.diameters],
        "object": {"shape": "sphere", "size_m": args.diameters[0], "mass_kg": args.mass, "fixed": args.fixed_object},
        "phases": phases, "arm_command": "Zero sensor wrench; unchanged native OSC gravity compensation.",
        "state_writes": "Initial pose/reset between trials only. No state writes during closing/contact.",
        "support": "Dynamic sphere rests on kinematic support; support contacts count only in net binary.",
        "contact_definition": "Strict normal resultant magnitude > production tactile_threshold; same 17 canonical pads.",
        "filtered_count": "Native rigid contact view filtered to current object only; excludes self/support contacts.",
        "whole_hand_object_force_peak_N_definition": "Sum of per-link normal-force magnitudes; not the net force-vector norm.",
        "friction": {key: getattr(sim_cfg.physics_material, key) for key in ("static_friction", "dynamic_friction")}}
    samples, frame_indices = [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = args.output_dir / "scene.mp4"
    writer = imageio.get_writer(str(raw_video), fps=args.video_fps, codec="libx264", quality=8) if camera else None
    zero_wrench = initial_q.new_zeros((1, 6))
    base_root = robot.data.default_root_state.clone()

    def save():
        data = {"metadata": metadata, "summary": summarize(samples, metadata),
                "frame_indices": frame_indices, "samples": samples}
        (args.output_dir / "report.json").write_text(json.dumps(data, indent=2) + "\n")

    try:
        for trial, (obj, support, diameter) in enumerate(zip(objects, supports, args.diameters)):
            for index, (other, table) in enumerate(zip(objects, supports)):
                for asset, z in ((other, -10.), (table, -12.)):
                    state = asset.data.default_root_state.clone()
                    state[:, :3] = state.new_tensor([index * 2., 0., z])
                    state[:, 7:] = 0.
                    asset.write_root_state_to_sim(state)
                    asset.reset()
            robot.write_root_state_to_sim(base_root)
            robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
            robot.set_joint_position_target(initial_q)
            robot.set_joint_velocity_target(torch.zeros_like(initial_q))
            robot.set_joint_effort_target(torch.zeros_like(initial_q))
            robot.reset()
            sim.forward()
            tactile.reset()
            reader.reset(torch.tensor([0], device=sim.device))
            arm_action.reset()
            hand_action.reset()
            points = rewards.tactile_pad_points(env)[0]
            palm_rotation = robot.data.body_link_quat_w[:, pad_ids[0]]
            normal = obs.quat_apply(palm_rotation, points.new_tensor([[0., 0., 1.]]))[0]
            tangent = points[5:].mean(0) - points[0]
            tangent -= normal * (tangent @ normal)
            tangent /= tangent.norm()
            side = torch.cross(normal, tangent, dim=-1)
            center = points[0] + tangent * args.along_fingers + normal * (diameter / 2 + args.surface_gap)
            object_state = obj.data.default_root_state.clone()
            object_state[:, :3] = center
            object_state[:, 7:] = 0.
            obj.write_root_state_to_sim(object_state)
            obj.reset()
            if not args.fixed_object:
                support_state = support.data.default_root_state.clone()
                support_state[:, :3] = center
                support_state[:, 2] -= diameter / 2 + .0125 + .0005
                support_state[:, 7:] = 0.
                support.write_root_state_to_sim(support_state)
                support.reset()
            sim.forward()
            reader.reset(torch.tensor([0], device=sim.device))
            metadata["objects"][trial]["initial_center_w_m"] = center.cpu().tolist()
            metadata["objects"][trial]["initial_pad_points_w_m"] = points.cpu().tolist()
            metadata["objects"][trial]["initial_palm_normal_w"] = normal.cpu().tolist()
            print(f"[Grasp] Trial {trial}: diameter={diameter:g}, center={center.cpu().tolist()}, normal={normal.cpu().tolist()}", flush=True)
            if camera:
                target = .5 * (center + robot.data.body_link_pos_w[0, sensor_id])
                # Side views remain above the support; the downward palm
                # normal would otherwise put both cameras beneath the table.
                eyes = torch.stack((target - .08 * normal - .34 * side - .08 * tangent,
                                    center - .05 * normal + .26 * side + .07 * tangent))
                camera.set_world_poses_from_view(eyes, torch.stack((target, center)))
                for _ in range(4):
                    sim.render()
            trial_step = 0
            for phase, duration in phases:
                steps = round(duration / dt)
                for step in range(steps):
                    amount = smooth_step((step + 1) / steps)
                    flexion = 0. if phase in ("open", "thumb_opposition", "open_after") else (
                        amount if phase == "close" else 1. if phase == "hold" else 1. - amount)
                    opposition = (0. if phase in ("open", "open_after") else amount if phase == "thumb_opposition"
                                  else 1. - amount if phase == "release" else 1.)
                    fraction = zero_wrench.new_full((1, 6), flexion)
                    fraction[:, 0] = opposition
                    if trial_step % cfg.decimation == 0:
                        arm_action.process_actions(zero_wrench)
                        hand_action.process_actions((2 * fraction - 1) / cfg.actions.hand_action.scale)
                        accepted_fraction = fraction.clone()
                    arm_action.apply_actions()
                    hand_action.apply_actions()
                    robot.write_data_to_sim()
                    sim.step(render=False)
                    robot.update(dt)
                    obj.update(dt)
                    tactile.update(dt)
                    batch = reader()
                    binary, valid, _, _ = sensors.sensor_monitor_snapshot(batch.tactile_force, batch.tactile_valid,
                        batch.ideal_wrench, batch.wrench_valid, threshold)
                    forces = batch.tactile_force[0].norm(dim=-1)
                    object_forces = views[trial].get_contact_force_matrix(dt=dt).reshape(1, 17, 3).index_select(
                        1, reader.tactile_ids)[0].norm(dim=-1)
                    body_forces = body_views[trial].get_contact_force_matrix(dt=dt).reshape(-1, 3).norm(dim=-1)
                    object_binary = valid[0] & (object_forces > threshold)
                    if not bool(torch.isfinite(robot.data.joint_vel).all() & torch.isfinite(object_forces).all() & valid.all()):
                        raise RuntimeError("Nonfinite state or invalid tactile channel.")
                    trial_step += 1
                    samples.append({"time_s": (len(samples) + 1) * dt, "trial_time_s": trial_step * dt,
                        "trial_index": trial, "trial_name": f"{diameter * 1000:.0f} mm sphere", "phase": phase,
                        "hand_fraction": accepted_fraction[0].cpu().tolist(), "policy_sample": trial_step % cfg.decimation == 0,
                        "binary": binary[0].cpu().tolist(), "object_binary": object_binary.cpu().tolist(),
                        "pad_force_N": forces.cpu().tolist(), "object_pad_force_N": object_forces.cpu().tolist(),
                        "hand_body_object_force_N": {path.rsplit("/", 1)[-1]: value
                            for path, value in zip(hand_paths, body_forces.cpu().tolist())},
                        "joint_velocity_rad_s": robot.data.joint_vel[0, driven_ids].cpu().tolist(),
                        "joint_position_rad": robot.data.joint_pos[0, driven_ids].cpu().tolist(),
                        "joint_position_target_rad": hand_action._driven_targets[0].cpu().tolist(),
                        "object_position_w_m": obj.data.root_pos_w[0].cpu().tolist(),
                        "arm_max_speed_rad_s": float(robot.data.joint_vel[:, arm_ids].abs().max())})
                    if camera and len(samples) % stride == 0:
                        current_points = rewards.tactile_pad_points(env)[0]
                        rotations = robot.data.body_link_quat_w[0].index_select(0, pad_ids)
                        normals = obs.quat_apply(rotations, points.new_tensor([[0., 0., 1.]]).expand(17, -1))
                        colors = torch.where(object_binary, 0, torch.where(binary[0], 1, 2))
                        marker.visualize(translations=current_points + normals * .0008, marker_indices=colors)
                        frames.visualize(translations=robot.data.body_link_pos_w[:, sensor_id],
                                         orientations=robot.data.body_link_quat_w[:, sensor_id])
                        sim.render()
                        camera.update(dt * stride, force_recompute=True)
                        rgb = camera.data.output["rgb"][..., :3].cpu().numpy()
                        writer.append_data(np.concatenate((rgb[0], rgb[1]), axis=1))
                        frame_indices.append(len(samples) - 1)
                    if args.max_steps and len(samples) >= args.max_steps:
                        break
                print(f"[Grasp] {trial}/{phase}: current object pads={int(object_binary.sum())}, net={int(binary.sum())}", flush=True)
                save()
                if args.max_steps and len(samples) >= args.max_steps:
                    break
            if args.max_steps and len(samples) >= args.max_steps:
                break
        metadata["status"] = "smoke_complete" if args.max_steps else "complete"
    except BaseException as error:
        metadata["status"], metadata["error"] = "failed", repr(error)
        raise
    finally:
        if writer:
            writer.close()
        save()
    if camera:
        from grasp_contact_video import render_overlay

        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        metadata["video"] = render_overlay(raw_video, args.video_output, samples, frame_indices, metadata)
        save()
    print("GRASP_CONTACT_SUMMARY", json.dumps(summarize(samples, metadata)), flush=True)


if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException as error:
        import traceback

        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "error.json").write_text(json.dumps({"error": repr(error),
            "traceback": traceback.format_exc()}, indent=2) + "\n")
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)
