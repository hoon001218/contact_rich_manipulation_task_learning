"""Compare identical scripted hand closure at two speeds with native contact evidence.

Default trials use the production symmetric Hand/OSC and a 300 g sphere.
--training-assets NAME [...] or all instead spawns the actual training USD
assets and runs a scripted upright side push with fixed Hand angles. Native
object mass, collision approximation/offset and upright orientation are reused.
Hand/object/support use production baseline contact-material configs. The
material randomization, crowded shelf and policy are deliberately absent.
Only initial states are written between trials. Native whole-Hand/object contact
points and signed separation distinguish actual cooked-shape penetration from
an apparent visual overlap or pad-only contact omission. Absence of a reported
contact cannot prove absence of tunneling. USD mesh metadata describes authored
source geometry, not an exported cooked collider or a full signed distance test.
Physics can be 100 or 200 Hz; command updates stay exactly 50 Hz. The one-env
CPU copies and detailed traces are diagnostic only, not training hot-path code.
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
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "logs/diagnostics/hand_penetration_2026-11-20")
    parser.add_argument("--video-output", type=Path, default=REPOSITORY_ROOT / "docs/videos/hand_penetration_2026-11-20/penetration.mp4")
    parser.add_argument("--diameters", type=float, nargs="+", default=[.06])
    parser.add_argument("--training-assets", nargs="+", metavar="NAME",
                        help="Use actual ASSET_SPECS names, or 'all'; scripted side-push instead of sphere closure.")
    parser.add_argument("--grasp", action="store_true", help="Wrap actual training objects; no scripted side push.")
    parser.add_argument("--grasp-fixture", choices=("palm-up", "supported"), default="palm-up",
                        help="Start above the upwards palm or beside an upright object on a shelf-like plate.")
    parser.add_argument("--grasp-close-fractions", type=float, nargs="+", default=[.5, .8],
                        help="Independent trials at these fractions, each with half-closure then full-closure holds.")
    parser.add_argument("--hand-effort-limit", type=float, default=None, help="Diagnostic implicit-drive ceiling, N m; omitted retains production.")
    parser.add_argument("--hand-stiffness", type=float, default=None)
    parser.add_argument("--hand-damping", type=float, default=None)
    parser.add_argument("--asset-push-duration", type=float, default=1.6)
    parser.add_argument("--asset-push-stroke", type=float, default=.06, help="Total diagnostic tool travel in m.")
    parser.add_argument("--push-force", type=float, default=1.8, help="Diagnostic commanded force cap in N.")
    parser.add_argument("--close-durations", type=float, nargs="+", default=[4., 1.],
                        help="Seconds for the same smooth close/reopen path, slow then regular.")
    parser.add_argument("--physics-hz", type=int, choices=(100, 200), default=100)
    parser.add_argument("--solve-articulation-contact-last", action="store_true",
                        help="Use the PhysX scene's alternate articulation contact solve order.")
    parser.add_argument("--hand-max-depenetration-velocity", type=float,
                        help="Override the maximum depenetration velocity of Hand links in m/s.")
    parser.add_argument("--initial-hand-fraction", type=float, default=0.,
                        help="Common initial flexion in the URDF-coupled range; independent of production reset changes.")
    parser.add_argument("--mass", type=float, default=.3)
    parser.add_argument("--fixed-object", action="store_true", help="Optional fixed-object diagnostic control.")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--along-fingers", type=float, default=.06, help="Object center's palm-tangent offset in m.")
    parser.add_argument("--surface-gap", type=float, default=.003, help="Initial hand/object surface clearance in m.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if any(not math.isfinite(x) or not .02 <= x <= .14 for x in args.diameters):
        parser.error("Diameters must lie in [0.02,0.14] m.")
    if args.grasp and not args.training_assets:
        parser.error("--grasp requires --training-assets NAME [...] or all.")
    if any(not math.isfinite(x) or not 0 < x <= 1 for x in args.grasp_close_fractions):
        parser.error("Grasp closure fractions must be in (0,1].")
    if any(x is not None and (not math.isfinite(x) or x <= 0) for x in
           (args.hand_effort_limit, args.hand_stiffness, args.hand_damping)):
        parser.error("Diagnostic Hand effort/stiffness/damping overrides must be positive when supplied.")
    hand_depenetration_velocity = args.hand_max_depenetration_velocity
    if hand_depenetration_velocity is not None and not (
            math.isfinite(hand_depenetration_velocity) and hand_depenetration_velocity > 0):
        parser.error("--hand-max-depenetration-velocity must be finite and positive.")
    if any(not math.isfinite(x) or x <= 0 for x in args.close_durations):
        parser.error("Closure durations must be finite and positive.")
    if not math.isfinite(args.initial_hand_fraction) or not 0 <= args.initial_hand_fraction < 1:
        parser.error("Initial fraction must lie in [0,1).")
    if any(not math.isfinite(x) or x <= 0 for x in (args.asset_push_duration, args.push_force)):
        parser.error("Asset push duration and force must be finite and positive.")
    if not math.isfinite(args.asset_push_stroke) or not 0.015 <= args.asset_push_stroke <= 0.12:
        parser.error("Asset push stroke must be in [0.015, 0.12] m.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive.")
    if not math.isfinite(args.mass) or args.mass <= 0 or args.video_fps <= 0:
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



def detailed_contacts(view, dt, hand_paths):
    """Read native active contact patches, preserving their actual body mapping."""
    import torch

    force, point, normal, separation, count, start = view.get_contact_data(dt=dt)
    counts, starts = count.reshape(-1).cpu().tolist(), start.reshape(-1).cpu().tolist()
    owners, indices = [], []
    for body, (size, first) in enumerate(zip(counts, starts)):
        if size:
            if first < 0 or first + size > view.max_contact_data_count:
                raise RuntimeError("Native contact buffer capacity/order overflow.")
            owners.extend([body] * size)
            indices.extend(range(first, first + size))
    if not indices:
        return []
    ids = torch.tensor(indices, device=force.device, dtype=torch.long)
    values = [tensor.index_select(0, ids).cpu().tolist() for tensor in (force, point, normal, separation)]
    return [{"body": hand_paths[body].rsplit("/", 1)[-1], "normal_force_N": f[0],
             "point_w_m": p, "normal_w": n, "separation_m": gap[0]}
            for body, f, p, n, gap in zip(owners, *values)]


def penetration_summary(samples, metadata):
    result = summarize(samples, metadata)
    for trial in result:
        rows = [row for row in samples if row["trial_index"] == trial["trial_index"]]
        gaps = [row["minimum_native_contact_separation_m"] for row in rows
                if row["minimum_native_contact_separation_m"] is not None]
        trial["minimum_native_contact_separation_m"] = min(gaps, default=None)
        trial["maximum_reported_patch_penetration_m"] = max(0., -min(gaps)) if gaps else None
        trial["native_contact_samples"] = sum(bool(row["whole_hand_native_contacts"]) for row in rows)
        trial["non_pad_contact_samples"] = sum(bool(row["object_contacting_non_pad_bodies"]) for row in rows)
        baseline = [row for row in rows if row["phase"] in ("open", "baseline")]
        trial["initial_baseline_contact_samples"] = sum(bool(row["whole_hand_native_contacts"]) for row in baseline)
        initial_quat = rows[0]["object_pose_w"][3:]
        final_quat = rows[-1]["object_pose_w"][3:]
        dot = min(1., abs(sum(a * b for a, b in zip(initial_quat, final_quat))))
        trial["object_rotation_angle_rad"] = 2. * math.acos(dot)
        trial["object_rotation_peak_rad"] = max(2. * math.acos(min(1., abs(sum(
            a * b for a, b in zip(initial_quat, row["object_pose_w"][3:]))))) for row in rows)
        def world_up_in_body(quat):
            w, x, y, z = quat
            return (2. * (x*z-w*y), 2. * (y*z+w*x), 1. - 2. * (x*x+y*y))
        initial_up = world_up_in_body(initial_quat)
        trial["object_upright_tilt_peak_rad"] = max(math.acos(max(-1., min(1., sum(
            a*b for a,b in zip(initial_up, world_up_in_body(row["object_pose_w"][3:])))))) for row in rows)
        trial["object_final_pose_w"] = rows[-1]["object_pose_w"]
        trial["object_displacement_peak_m"] = max(math.sqrt(sum((a-b)**2 for a,b in zip(
            rows[0]["object_position_w_m"], row["object_position_w_m"]))) for row in rows)
        trial["maximum_hand_native_contact_penetration_body"] = min(
            (contact for row in rows for contact in row["whole_hand_native_contacts"]),
            key=lambda contact: contact["separation_m"], default=None)
        stage_metrics = {}
        for stage in ("hold_mid", "hold"):
            stage_rows = [row for row in rows if row["phase"] == stage]
            if not stage_rows:
                continue
            opposed = [bool(row.get("object_opposed_contact", False)) for row in stage_rows]
            longest = run_length = 0
            for value in opposed:
                run_length = run_length + 1 if value else 0
                longest = max(longest, run_length)
            loads = [sum(row["hand_body_object_force_N"].values()) for row in stage_rows]
            separation = [row["minimum_native_contact_separation_m"] for row in stage_rows
                          if row["minimum_native_contact_separation_m"] is not None]
            stage_metrics[stage] = {"samples": len(stage_rows), "opposed_contact_fraction": sum(opposed)/len(opposed),
                "longest_continuous_opposed_contact_s": longest*metadata["dt"],
                "object_pad_peak": max(sum(row["object_binary"]) for row in stage_rows),
                "whole_hand_normal_force_mean_N": sum(loads)/len(loads),
                "whole_hand_normal_force_peak_N": max(loads),
                "native_separation_min_m": min(separation, default=None)}
        trial["grasp_stages"] = stage_metrics
        trial["warning"] = ("Initial baseline has contacts: compare as loaded fixture, not a clean first-touch test."
                            if trial["initial_baseline_contact_samples"] else None)
    return result


def run(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.math import compute_pose_error

    scene = importlib.import_module("contact_manipulation_symmetric.config.ur5e.scene")
    env_module = importlib.import_module("contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module("contact_manipulation_symmetric.mdp.events")
    sensors = importlib.import_module("contact_manipulation_symmetric.mdp.sensors")
    obs = importlib.import_module("contact_manipulation_symmetric.mdp.observations")
    rewards = importlib.import_module("contact_manipulation_symmetric.mdp.rewards")
    cfg = env_module.UR5eContactManipulationEnvCfg()
    robot_cfg = cfg.scene.robot.copy()
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    for argument, field in (("hand_effort_limit", "effort_limit_sim"),
                            ("hand_stiffness", "stiffness"), ("hand_damping", "damping")):
        value = getattr(args, argument)
        if value is not None:
            setattr(robot_cfg.actuators["hand"], field, value)
    sim_cfg = cfg.sim.copy()
    sim_cfg.device, sim_cfg.render_interval = args.device, 2**31 - 1
    sim_cfg.dt = 1. / args.physics_hz
    if args.solve_articulation_contact_last:
        sim_cfg.physx.solve_articulation_contact_last = True
    command_decimation = args.physics_hz // 50
    asset_mode = bool(args.training_assets)
    palm_up_grasp = args.grasp and args.grasp_fixture == "palm-up"
    asset_by_name = {spec.name: spec for spec in scene.ASSET_SPECS}
    selected_assets = list(asset_by_name) if args.training_assets == ["all"] else args.training_assets
    if asset_mode and (len(set(selected_assets)) != len(selected_assets) or any(name not in asset_by_name for name in selected_assets)):
        raise ValueError(f"Use distinct training asset names from {list(asset_by_name)}, or 'all'.")
    cases = ([(name, fraction) for name in selected_assets for fraction in args.grasp_close_fractions] if args.grasp else
             [(name, args.asset_push_duration) for name in selected_assets] if asset_mode else
             [(diameter, duration) for diameter in args.diameters for duration in args.close_durations])
    source_objects = scene.make_object_collection_cfg().rigid_objects if asset_mode else None
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    robot = Articulation(robot_cfg)
    stage = sim_utils.get_current_stage()
    hand_path = f"{robot_cfg.prim_path}/ROASRightHand"
    hand_depenetration_velocity = args.hand_max_depenetration_velocity
    if hand_depenetration_velocity is not None:
        sim_utils.modify_rigid_body_properties(
            hand_path, sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=hand_depenetration_velocity),
            stage=stage,
        )
    hand_link_depenetration_velocities = {
        str(prim.GetPath()): PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxDepenetrationVelocityAttr().Get()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(hand_path))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    if hand_depenetration_velocity is not None and (
            not hand_link_depenetration_velocities
            or any(value is None or not math.isclose(value, hand_depenetration_velocity, rel_tol=1.e-6)
                   for value in hand_link_depenetration_velocities.values())):
        raise RuntimeError(f"Failed to set max depenetration velocity on every Hand link: {hand_link_depenetration_velocities}")
    contact_order = stage.GetPrimAtPath(sim_cfg.physics_prim_path).GetAttribute(
        "physxScene:solveArticulationContactLast"
    ).Get()
    if contact_order != sim_cfg.physx.solve_articulation_contact_last:
        raise RuntimeError(f"Unexpected USD articulation contact order: {contact_order!r}")
    tactile_cfg = sensors.make_tactile_contact_cfg(scene.TACTILE_BODY_NAMES)
    tactile_cfg.prim_path = tactile_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    tactile = tactile_cfg.class_type(tactile_cfg)
    # Supports and objects are authored once, then native initial poses are
    # written between independent trials. Inactive trials are parked remotely.
    objects, supports = [], []
    for index, (diameter, _) in enumerate(cases):
        if asset_mode:
            object_cfg = source_objects[diameter].copy()
            object_cfg.prim_path = f"/World/GraspObject_{index}"
            object_cfg.init_state.pos = (index * 2., 0., -10.)
            object_cfg.spawn.activate_contact_sensors = True
            object_cfg.spawn.rigid_props.kinematic_enabled = args.fixed_object
            object_cfg.spawn.rigid_props.disable_gravity = args.fixed_object
            obj = RigidObject(object_cfg)
        else:
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
            spawn=sim_utils.CuboidCfg(size=(.40, .40, .025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.005, rest_offset=0.),
                physics_material=sim_cfg.physics_material.copy(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.19, .24, .29), roughness=.8)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(index * 2., 0., -12.))))
        objects.append(obj)
        supports.append(support)
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.9, .94, 1.))
    light.func("/World/GraspLight", light)
    camera = marker = frames = contact_markers = None
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
        contact_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/World/PenetrationContactMarkers", markers={
                "negative_separation": sim_utils.SphereCfg(radius=.0008,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1., .1, .15))),
                "contact_skin": sim_utils.SphereCfg(radius=.0008,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.1, .85, 1.))),
            }))
        for path in ("/World/GraspPadMarkers", "/World/GraspSensorFrame", "/World/PenetrationContactMarkers"):
            for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError("Visual grasp markers cannot contain physics schemas.")
    if asset_mode:
        materials = cfg.scene.shared_geometry.spawn
        for category in ("hand", "object", "shelf"):
            material_cfg = getattr(materials, f"{category}_contact_material")
            material_cfg.func(f"/World/DiagnosticMaterials/{category}", material_cfg)
        stage = sim_utils.get_current_stage()
        bindings = [("hand", robot_cfg.prim_path + "/ROASRightHand")]
        bindings += [("object", obj.cfg.prim_path) for obj in objects]
        bindings += [("shelf", support.cfg.prim_path) for support in supports]
        for category, path in bindings:
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(f"/World/DiagnosticMaterials/{category}")),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
    sim.reset()
    initial_q = robot.data.default_joint_pos.clone()
    if asset_mode:
        initial_q[:, robot.joint_names.index("wrist_3_joint")] += math.pi if palm_up_grasp else math.pi / 2.
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
    # Both assets have a base_link. Use the native row for each full path.
    hand_body_ids = [index for index, path in enumerate(body_paths) if "/ROASRightHand/" in str(path)]
    hand_paths = [str(body_paths[index]) for index in hand_body_ids]
    hand_names = [path.rsplit("/", 1)[-1] for path in hand_paths]
    thumb_indices = [i for i, name in enumerate(hand_names) if name.startswith("right_thumb")]
    finger_prefixes = ("right_index", "right_middle", "right_ring", "right_little")
    other_indices = [i for i, name in enumerate(hand_names) if name.startswith(finger_prefixes)]
    views, body_views = [], []
    sensor_paths = list(tactile.contact_physx_view.sensor_paths)
    for obj in objects:
        view = robot._physics_sim_view.create_rigid_contact_view(sensor_paths,
            filter_patterns=[[str(obj.root_physx_view.prim_paths[0])]] * 17, max_contact_data_count=0)
        if list(view.sensor_paths) != sensor_paths or view.sensor_count != 17 or view.filter_count != 1:
            raise RuntimeError("Filtered tactile witness row mapping changed.")
        views.append(view)
        body_view = robot._physics_sim_view.create_rigid_contact_view(hand_paths,
            filter_patterns=[[str(obj.root_physx_view.prim_paths[0])]] * len(hand_paths), max_contact_data_count=8192)
        if list(body_view.sensor_paths) != hand_paths or body_view.filter_count != 1:
            raise RuntimeError("Whole-hand contact witness row mapping changed.")
        body_views.append(body_view)
    # Every speed case starts with exactly the same physical Hand angles/targets.
    initial_raw = initial_q.new_full((1, 6), (2 * args.initial_hand_fraction - 1) / cfg.actions.hand_action.scale)
    hand_action.process_actions(initial_raw)
    initial_q[:, driven_ids] = hand_action._driven_targets
    authored_colliders = []
    stage = sim_utils.get_current_stage()
    for path in hand_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            physx = PhysxSchema.PhysxCollisionAPI(prim)
            authored_colliders.append({"path": str(prim.GetPath()), "body": path.rsplit("/", 1)[-1],
                "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                "approximation": UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else None,
                "contact_offset_authored": physx.GetContactOffsetAttr().Get() if physx else None,
                "rest_offset_authored": physx.GetRestOffsetAttr().Get() if physx else None,
                "source_meshes": [{"path": str(child.GetPath()),
                                   "vertex_count": len(UsdGeom.Mesh(child).GetPointsAttr().Get() or [])}
                    for child in Usd.PrimRange(prim) if child.IsA(UsdGeom.Mesh)]})
    object_info, object_supports = [], []
    reference = stage.GetPrimAtPath("/World/envs/env_0")
    hand_supports = []
    if asset_mode:
        for body_id in hand_body_ids:
            body = stage.GetPrimAtPath(str(body_paths[body_id]))
            colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                         if prim.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
            if colliders:
                vertices, _ = scene._object_support_vertices(body, colliders, (1., 0., 0., 0.), reference_frame=reference)
                hand_supports.append((body_id, initial_q.new_tensor(vertices)))
    for (name, duration), obj in zip(cases, objects):
        mass = float(obj.root_physx_view.get_masses()[0, 0])
        body = stage.GetPrimAtPath(str(obj.root_physx_view.prim_paths[0]))
        info = {"shape": "training USD" if asset_mode else "sphere", "mass_kg": mass,
                "fixed": args.fixed_object, "close_duration_s": 1.5 if args.grasp else duration,
                "solve_articulation_contact_last": contact_order,
                "object_max_depenetration_velocity_m_s": (
                    PhysxSchema.PhysxRigidBodyAPI(body).GetMaxDepenetrationVelocityAttr().Get()),
                "hand_max_depenetration_velocity_m_s": sorted(
                    {value for value in hand_link_depenetration_velocities.values() if value is not None})}
        if asset_mode:
            colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                         if prim.HasAPI(UsdPhysics.CollisionAPI)
                         and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
            vertices, error = scene._object_support_vertices(body, colliders,
                                asset_by_name[name].upright_quat, reference_frame=reference)
            vertices = initial_q.new_tensor(vertices)
            lower, upper = vertices.amin(0), vertices.amax(0)
            object_supports.append((vertices, lower, upper))
            info.update(asset_name=name, usd_path=asset_by_name[name].usd_path,
                grasp_close_fraction=duration if args.grasp else None,
                requested_hand_effort_limit_Nm=args.hand_effort_limit,
                native_body_path=str(obj.root_physx_view.prim_paths[0]),
                upright_quat=list(asset_by_name[name].upright_quat),
                size_m=(upper - lower).cpu().tolist(), support_error_bound_m=error,
                enabled_collider_paths=[str(prim.GetPath()) for prim in colliders])
        else:
            object_supports.append(None)
            info["size_m"] = name
        object_info.append(info)
    dt = sim.get_physics_dt()
    stride = round(1 / (dt * args.video_fps))
    if stride < 1 or not math.isclose(stride * dt * args.video_fps, 1.):
        raise ValueError("FPS must divide the physics frequency.")
    phases = []  # Each speed trial stores its own close/reopen durations below.
    threshold = cfg.mdp.tactile_threshold
    metadata = {"status": "running", "experiment": "training_asset_grasp" if args.grasp else "training_asset_side_push" if asset_mode else "hand_penetration_speed_pair", "dt": dt,
        "policy_decimation": command_decimation, "command_hz": 50,
        "authored_hand_colliders": authored_colliders,
        "hand_body_names": hand_names,
        "native_contact_separation_definition": "PhysX contact patch signed separation: negative means local penetration; positive is contact skin. No contact does not prove no tunneling.",
        "geometry_scope": "Source USD collider metadata and actual native contact data; no cooked-mesh export or full triangle/SDF intersection test.",
        "contact_point_capacity_per_view": 8192,
        "video_contact_markers": "Visual-only actual native contact points: red=negative separation, cyan=nonnegative skin contact.",
        "command_schedule": "Command at each 50Hz interval requests the path value at that interval end; identical targets at 100/200Hz.",
        "initial_hand_fraction": args.initial_hand_fraction,
        "command_line": sys.argv, "video_fps": args.video_fps,
        "initial_joint_names": list(robot.joint_names), "initial_joint_position_rad": initial_q[0].cpu().tolist(),
        "placement": {"along_fingers_m": args.along_fingers, "surface_gap_m": args.surface_gap},
        "threshold_N": threshold, "pad_names": list(pad_names),
        "joint_names": [robot.joint_names[i] for i in driven_ids],
        "joint_velocity_limits_rad_s": robot.root_physx_view.get_dof_max_velocities()[0, driven_ids].cpu().tolist(),
        "joint_effort_limits_Nm": robot.data.joint_effort_limits[0, driven_ids].cpu().tolist(),
        "native_hand_effort_limits_Nm": robot.root_physx_view.get_dof_max_forces()[0, driven_ids].cpu().tolist(),
        "hand_force_interpretation": "Raising effort ceiling does not prove stronger grip: compare measured native object forces at matched closure. Implicit actuator reported torque is an estimate, not measured drive torque.",
        "opposition_definition": "Whole-Hand object-filtered normal resultant: thumb and non-thumb finger groups both exceed tactile threshold and their vector dot product is negative. This is opposed contact evidence, not proof of force closure or stable grasp.",
        "grasp_position_hold": ("Diagnostic wrench generator at initial sensor pose: translation Kp300N/m, Kd40Ns/m, force cap20N; angular Kp2Nm/rad Kd.1Nms/rad, cap2Nm. Commands pass through unchanged production wrench OSC/IdealPD." if args.grasp else None),
        "joint_armature": robot.data.joint_armature[0, driven_ids].cpu().tolist(),
        "hand_setting": {key: getattr(robot_cfg.actuators["hand"], key) for key in
                         ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim")},
        "physics": {"dt": dt, "gravity": list(sim_cfg.gravity),
            "robot_position_iterations": robot_cfg.spawn.articulation_props.solver_position_iteration_count,
            "robot_velocity_iterations": robot_cfg.spawn.articulation_props.solver_velocity_iteration_count,
            "solve_articulation_contact_last": contact_order,
            "requested_hand_max_depenetration_velocity_m_s": hand_depenetration_velocity,
            "hand_link_max_depenetration_velocities_m_s": hand_link_depenetration_velocities},
        "objects": object_info, "object": object_info[0],
        "grasp_fixture": args.grasp_fixture if args.grasp else None,
        "asset_fixture": (("Actual task spawners/colliders/native masses and baseline task materials, upright plus recorded yaw; scripted diagnostic grasp." if args.grasp else
                           "Actual task spawners/colliders/upright/native masses and baseline task contact materials; dynamic body on diagnostic support, no scene-friction randomization. Scripted push, not policy evaluation.") if asset_mode else None),
        "push_force_N": args.push_force if asset_mode and not args.grasp else None,
        "tool_stroke_m": args.asset_push_stroke if asset_mode and not args.grasp else None,
        "phases": phases,
        "arm_command": ("Diagnostic initial sensor pose hold using production wrench OSC." if args.grasp else
                        "Smooth, bounded horizontal tool travel tracked by diagnostic translation PD 80/12 and angular PD .5/.02. Sends only the resulting sensor wrench through production OSC. Contact r-cross-F." if asset_mode else
                        "Zero sensor wrench; unchanged native OSC gravity compensation."),
        "state_writes": "Initial pose/reset between trials only. No state writes during closing/contact.",
        "support": ("Object starts above the palm; the diagnostic plate is parked away." if palm_up_grasp else
                    "Actual upright training object on a 400x400x25mm diagnostic support; no scene layout recreation." if asset_mode else
                    "Dynamic sphere rests on kinematic support; support contacts count only in net binary."),
        "contact_definition": "Strict normal resultant magnitude > production tactile_threshold; same 17 canonical pads.",
        "filtered_count": "Native rigid contact view filtered to current object only; excludes self/support contacts.",
        "whole_hand_object_force_peak_N_definition": "Sum of per-link normal-force magnitudes; not the net force-vector norm.",
        "friction": ({category: {key: getattr(getattr(materials, f"{category}_contact_material"), key)
                         for key in ("static_friction", "dynamic_friction", "friction_combine_mode")}
                      for category in ("hand", "object", "shelf")} if asset_mode else
                     {key: getattr(sim_cfg.physics_material, key) for key in ("static_friction", "dynamic_friction")})}
    samples, frame_indices = [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = args.output_dir / "scene.mp4"
    writer = imageio.get_writer(str(raw_video), fps=args.video_fps, codec="libx264", quality=8) if camera else None
    zero_wrench = initial_q.new_zeros((1, 6))
    base_root = robot.data.default_root_state.clone()

    def save():
        data = {"metadata": metadata, "summary": penetration_summary(samples, metadata),
                "frame_indices": frame_indices, "samples": samples}
        temporary = args.output_dir / "report.json.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(args.output_dir / "report.json")

    try:
        for trial, (obj, support, (diameter, close_duration)) in enumerate(zip(objects, supports, cases)):
            phases = ([("baseline", 1.), ("thumb_opposition", 1.), ("close_mid", 1.5), ("hold_mid", 1.),
                       ("close", 1.5), ("hold", 2.), ("release", 1.5), ("open_after", .8)] if args.grasp else
                      [("baseline", .6), ("approach", 1.6), ("push", close_duration),
                       ("brake", .8), ("hold", 1.)] if asset_mode else
                      [("open", .6), ("thumb_opposition", 1.2), ("close", close_duration),
                       ("hold", 1.), ("release", close_duration), ("open_after", .6)])
            metadata["objects"][trial]["phases"] = phases
            for _, duration in phases:
                if not math.isclose(duration * 50., round(duration * 50.), abs_tol=1.e-6):
                    raise ValueError("Every phase duration must be an integer number of 50Hz command intervals.")
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
            if asset_mode:
                if palm_up_grasp and float(normal[2]) < .95:
                    raise RuntimeError("Grasp fixture requires palm up.")
                if not palm_up_grasp and abs(float(normal[2])) > .05:
                    raise RuntimeError("The plate fixture requires a horizontal palm normal.")
                vertices, lower, upper = object_supports[trial]
                contact_center = points[5:].mean(0)
                hand_vertices = torch.cat([robot.data.body_link_pos_w[0, body_id] + obs.quat_apply(
                    robot.data.body_link_quat_w[:, body_id].expand(len(local), -1), local)
                    for body_id, local in hand_supports])
                if args.grasp:
                    # Align the asset's narrower horizontal extent across the
                    # thumb/finger side direction; preserve its upright axis.
                    narrow_direction = (side if palm_up_grasp else normal).clone()
                    narrow_direction[2] = 0.
                    narrow_direction /= narrow_direction.norm()
                    narrow_axis = 0 if float(upper[0]-lower[0]) <= float(upper[1]-lower[1]) else 1
                    yaw = math.atan2(float(narrow_direction[1]), float(narrow_direction[0])) - narrow_axis * math.pi / 2.
                    yaw_q = initial_q.new_tensor([[math.cos(yaw/2), 0., 0., math.sin(yaw/2)]])
                    vertices = obs.quat_apply(yaw_q.expand(len(vertices), -1), vertices)
                    lower, upper = vertices.amin(0), vertices.amax(0)
                    asset_rotation = obs.quat_mul(yaw_q, initial_q.new_tensor([asset_by_name[diameter].upright_quat]))[0]
                    contact_center = points[0] + tangent * args.along_fingers
                    center = contact_center - (lower + upper) * .5
                    if palm_up_grasp:
                        footprint = ((hand_vertices[:, :2] >= center[:2] + lower[:2] - .003)
                                     & (hand_vertices[:, :2] <= center[:2] + upper[:2] + .003)).all(-1)
                        candidates = hand_vertices[footprint] if bool(footprint.any()) else hand_vertices
                        center[2] = candidates[:, 2].max() + args.surface_gap - lower[2]
                        support_z = None
                    else:
                        support_z = torch.minimum(contact_center[2] - (upper[2] - lower[2]) * .5,
                                                  hand_vertices[:, 2].min() - .015)
                        center[2] = support_z - lower[2] + .001
                        fingerward = ((hand_vertices - contact_center) @ tangent).abs()
                        half_span = ((vertices - (lower + upper) * .5) @ tangent).abs().max()
                        in_span = (fingerward <= half_span + .005) & (hand_vertices[:, 2] >= support_z)
                        in_span &= hand_vertices[:, 2] <= center[2] + upper[2]
                        candidates = hand_vertices[in_span] if bool(in_span.any()) else hand_vertices
                        front = (candidates @ narrow_direction).max()
                        center += narrow_direction * (front + args.surface_gap - (center @ narrow_direction +
                                                                                  (vertices @ narrow_direction).min()))
                    metadata["objects"][trial]["grasp_yaw_rad"] = yaw
                    metadata["objects"][trial]["grasp_placement"] = (
                        "Upright object above overlapping source hand collider vertices. Dynamic object can settle on the hand."
                        if palm_up_grasp else "Upright object beside the palm on a fixed plate, with its narrow side facing the palm.")
                else:
                    hand_lowest = hand_vertices[:, 2].min()
                    support_z = torch.minimum(contact_center[2] - (upper[2] - lower[2]) * .5,
                                              hand_lowest - .015)
                    center = contact_center - (lower + upper) * .5
                    center[2] = support_z - lower[2] + .0005
                    lateral = torch.cross(initial_q.new_tensor([0., 0., 1.]), normal, dim=-1)
                    half_width = ((vertices - (lower + upper) * .5) @ lateral).abs().max()
                    in_span = ((hand_vertices - contact_center) @ lateral).abs() <= half_width + .005
                    in_span &= (hand_vertices[:, 2] >= support_z) & (hand_vertices[:, 2] <= center[2] + upper[2])
                    candidates = hand_vertices[in_span] if bool(in_span.any()) else hand_vertices
                    front = (candidates @ normal).max()
                    center += normal * (front + args.surface_gap - (center @ normal + (vertices @ normal).min()))
                    asset_rotation = initial_q.new_tensor(asset_by_name[diameter].upright_quat)
                object_state = obj.data.default_root_state.clone()
                object_state[:, :3] = center
                object_state[:, 3:7] = asset_rotation
                object_state[:, 7:] = 0.
                obj.write_root_state_to_sim(object_state)
                obj.reset()
                support_state = support.data.default_root_state.clone()
                support_state[:, :3] = center + (lower + upper) * .5
                support_state[:, 2] = support_z - .0125 if support_z is not None else -12.
                support_state[:, 7:] = 0.
                support.write_root_state_to_sim(support_state)
                support.reset()
                initial_sensor_rotation = robot.data.body_link_quat_w[:, sensor_id].clone()
                initial_sensor_position = robot.data.body_link_pos_w[0, sensor_id].clone()
                metadata["objects"][trial]["support_top_z_m"] = None if support_z is None else float(support_z)
            else:
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
            print(f"[Penetration] Trial {trial}: asset={diameter}, center={center.cpu().tolist()}, normal={normal.cpu().tolist()}", flush=True)
            if camera:
                camera_center = center + (lower + upper) * .5 if asset_mode else center
                target = .5 * (camera_center + robot.data.body_link_pos_w[0, sensor_id])
                if palm_up_grasp:
                    eyes = torch.stack((target + .10 * normal - .34 * side + .08 * tangent,
                                        camera_center + .14 * normal + .24 * side + .15 * tangent))
                elif asset_mode:
                    up = points.new_tensor([0., 0., 1.])
                    eyes = torch.stack((target + .30 * up - .14 * normal - .10 * tangent,
                                        camera_center + .15 * up + .20 * normal + .17 * tangent))
                else:
                    eyes = torch.stack((target - .08 * normal - .34 * side - .08 * tangent,
                                        camera_center - .05 * normal + .26 * side + .07 * tangent))
                camera.set_world_poses_from_view(eyes, torch.stack((target, camera_center)))
                for _ in range(4):
                    sim.render()
            trial_step = 0
            for phase, duration in phases:
                steps = round(duration / dt)
                for step in range(steps):
                    amount = smooth_step((step + command_decimation) / steps)
                    if args.grasp:
                        flexion = (0. if phase in ("baseline", "thumb_opposition", "open_after") else
                                   .5 * close_duration * amount if phase == "close_mid" else
                                   .5 * close_duration if phase == "hold_mid" else
                                   close_duration * (.5 + .5 * amount) if phase == "close" else
                                   close_duration if phase == "hold" else close_duration * (1.-amount))
                        opposition = (0. if phase in ("baseline", "open_after") else amount if phase == "thumb_opposition"
                                      else 1.-amount if phase == "release" else 1.)
                    else:
                        flexion = 0. if phase in ("open", "thumb_opposition", "open_after") else (
                            amount if phase == "close" else 1. if phase == "hold" else 1. - amount)
                        opposition = (0. if phase in ("open", "open_after") else amount if phase == "thumb_opposition"
                                      else 1. - amount if phase == "release" else 1.)
                    fraction = zero_wrench.new_full((1, 6), args.initial_hand_fraction + (1 - args.initial_hand_fraction) * flexion)
                    fraction[:, 0] = args.initial_hand_fraction + (1 - args.initial_hand_fraction) * opposition
                    if trial_step % command_decimation == 0:
                        if args.grasp:
                            sensor_pos = robot.data.body_link_pos_w[0, sensor_id]
                            velocity_w = robot.data.body_link_lin_vel_w[0, sensor_id]
                            force_w = 300. * (initial_sensor_position - sensor_pos) - 40. * velocity_w
                            force_w *= 20. / force_w.norm().clamp_min(20.)
                            _, error = compute_pose_error(sensor_pos[None], robot.data.body_link_quat_w[:, sensor_id],
                                                           sensor_pos[None], initial_sensor_rotation)
                            torque_w = 2. * error[0] - .1 * robot.data.body_com_ang_vel_w[0, sensor_id]
                            torque_w *= 2. / torque_w.norm().clamp_min(2.)
                            inverse = obs.quat_conjugate(robot.data.body_link_quat_w[:, sensor_id])
                            accepted_wrench = torch.cat((obs.quat_apply(inverse, force_w[None]),
                                                         obs.quat_apply(inverse, torque_w[None])), -1)
                            arm_action.process_actions(accepted_wrench / initial_q.new_tensor(cfg.actions.arm_action.wrench_scale))
                        elif asset_mode:
                            approach = .012
                            advance = (0. if phase == "baseline" else approach * amount if phase == "approach" else
                                       approach + (args.asset_push_stroke - approach) * amount if phase == "push" else
                                       args.asset_push_stroke)
                            sensor_pos = robot.data.body_link_pos_w[0, sensor_id]
                            velocity_w = robot.data.body_link_lin_vel_w[0, sensor_id]
                            target_position = initial_sensor_position + normal * advance
                            force_w = 80. * (target_position - sensor_pos) - 12. * velocity_w
                            force_w *= args.push_force / force_w.norm().clamp_min(args.push_force)
                            contact = rewards.tactile_pad_points(env)[0, 5:].mean(0)
                            touching = bool(samples and samples[-1]["trial_index"] == trial
                                            and samples[-1]["whole_hand_native_contacts"])
                            torque_w = torch.cross(contact - sensor_pos, force_w, dim=-1) if touching else torch.zeros_like(force_w)
                            _, error = compute_pose_error(sensor_pos[None], robot.data.body_link_quat_w[:, sensor_id],
                                                           sensor_pos[None], initial_sensor_rotation)
                            correction = .5 * error[0] - .02 * robot.data.body_com_ang_vel_w[0, sensor_id]
                            correction *= 1.5 / correction.norm().clamp_min(1.5)
                            torque_w += correction
                            inverse = obs.quat_conjugate(robot.data.body_link_quat_w[:, sensor_id])
                            accepted_wrench = torch.cat((obs.quat_apply(inverse, force_w[None]),
                                                         obs.quat_apply(inverse, torque_w[None])), -1)
                            arm_action.process_actions(accepted_wrench / initial_q.new_tensor(cfg.actions.arm_action.wrench_scale))
                            fraction.fill_(args.initial_hand_fraction)
                        else:
                            accepted_wrench = zero_wrench
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
                    body_force_vectors = body_views[trial].get_contact_force_matrix(dt=dt).reshape(-1, 3)
                    body_forces = body_force_vectors.norm(dim=-1)
                    object_binary = valid[0] & (object_forces > threshold)
                    hand_contact_forces = dict(zip(hand_names, body_forces.cpu().tolist()))
                    thumb_contact = any(hand_contact_forces[hand_names[i]] > threshold for i in thumb_indices)
                    other_contact = any(hand_contact_forces[hand_names[i]] > threshold for i in other_indices)
                    thumb_vector = body_force_vectors[thumb_indices].sum(0)
                    other_vector = body_force_vectors[other_indices].sum(0)
                    denominator = float(thumb_vector.norm() * other_vector.norm())
                    opposition_cosine = float(thumb_vector @ other_vector) / denominator if denominator > 1.e-12 else None
                    thumb_force_sum = float(body_forces[thumb_indices].sum())
                    other_force_sum = float(body_forces[other_indices].sum())
                    thumb_pad_contact = bool(object_binary[1:5].any())
                    other_pad_contact = bool(object_binary[5:].any())
                    if not bool(torch.isfinite(robot.data.joint_vel).all() & torch.isfinite(object_forces).all() & valid.all()):
                        raise RuntimeError("Nonfinite state or invalid tactile channel.")
                    contact_data = detailed_contacts(body_views[trial], dt, hand_paths)
                    trial_step += 1
                    samples.append({"time_s": (len(samples) + 1) * dt, "trial_time_s": trial_step * dt,
                        "trial_index": trial, "trial_name": f"{diameter}/grasp {close_duration:g}" if args.grasp else str(diameter) if asset_mode else f"{diameter * 1000:.0f} mm / close {close_duration:g}s", "phase": phase,
                        "hand_fraction": accepted_fraction[0].cpu().tolist(),
                        "grasp_stage_fraction": float(accepted_fraction[0, 1]) if args.grasp else None,
                        "object_thumb_contact": thumb_contact, "object_other_finger_contact": other_contact,
                        "object_opposed_contact": thumb_contact and other_contact and opposition_cosine is not None and opposition_cosine < 0.,
                        "object_thumb_other_force_cosine": opposition_cosine,
                        "object_thumb_normal_force_sum_N": thumb_force_sum,
                        "object_other_finger_normal_force_sum_N": other_force_sum,
                        "object_thumb_pad_contact": thumb_pad_contact, "object_other_finger_pad_contact": other_pad_contact,
                        "object_opposed_pad_contact": thumb_pad_contact and other_pad_contact, "policy_sample": trial_step % command_decimation == 0,
                        "binary": binary[0].cpu().tolist(), "object_binary": object_binary.cpu().tolist(),
                        "pad_force_N": forces.cpu().tolist(), "object_pad_force_N": object_forces.cpu().tolist(),
                        "hand_body_object_force_N": hand_contact_forces,
                        "joint_velocity_rad_s": robot.data.joint_vel[0, driven_ids].cpu().tolist(),
                        "joint_position_rad": robot.data.joint_pos[0, driven_ids].cpu().tolist(),
                        "joint_position_target_rad": hand_action._driven_targets[0].cpu().tolist(),
                        "object_position_w_m": obj.data.root_pos_w[0].cpu().tolist(),
                        "object_pose_w": obj.data.root_link_pose_w[0].cpu().tolist(),
                        "hand_body_pose_w": robot.data.body_link_pose_w[0, hand_body_ids].cpu().tolist(),
                        "arm_joint_position_rad": robot.data.joint_pos[0, arm_ids].cpu().tolist(),
                        "arm_joint_velocity_rad_s": robot.data.joint_vel[0, arm_ids].cpu().tolist(),
                        "ideal_sensor_wrench": batch.ideal_wrench[0].cpu().tolist(),
                        "commanded_sensor_wrench": accepted_wrench[0].cpu().tolist(),
                        "sensor_pose_w": robot.data.body_link_pose_w[0, sensor_id].cpu().tolist(),
                        "whole_hand_native_contacts": contact_data,
                        "minimum_native_contact_separation_m": min((c["separation_m"] for c in contact_data), default=None),
                        "object_contacting_non_pad_bodies": [name for name, force in hand_contact_forces.items()
                            if name not in pad_names and force > threshold],
                        "arm_max_speed_rad_s": float(robot.data.joint_vel[:, arm_ids].abs().max())})
                    if camera and len(samples) % stride == 0:
                        current_points = rewards.tactile_pad_points(env)[0]
                        rotations = robot.data.body_link_quat_w[0].index_select(0, pad_ids)
                        normals = obs.quat_apply(rotations, points.new_tensor([[0., 0., 1.]]).expand(17, -1))
                        colors = torch.where(object_binary, 0, torch.where(binary[0], 1, 2))
                        marker.visualize(translations=current_points + normals * .0008, marker_indices=colors)
                        contact_markers.set_visibility(bool(contact_data))
                        if contact_data:
                            contact_markers.visualize(
                                translations=points.new_tensor([c["point_w_m"] for c in contact_data]),
                                marker_indices=torch.tensor([int(c["separation_m"] >= 0.) for c in contact_data],
                                                            device=sim.device))
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
                if args.max_steps and len(samples) >= args.max_steps:
                    break
            save()
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
    print("HAND_PENETRATION_SUMMARY", json.dumps(penetration_summary(samples, metadata)), flush=True)


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
