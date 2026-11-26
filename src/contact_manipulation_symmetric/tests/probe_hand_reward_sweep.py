"""Compare the production pad rewards at fixed poses, then measure real Blocker contact.

Run with Isaac Lab and --headless. Uses training assets in an isolated fixture,
without changing the environment configuration used by train.py or play.py.
"""

import argparse
import json
import math
from pathlib import Path

from probe_ft_compensation import AppLauncher, REPOSITORY_ROOT, smooth_step


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--assets", nargs="+", default=["can_1", "cup_2", "ycb_003_cracker_box"])
    parser.add_argument("--no-video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--geometry-only", action="store_true")
    mode.add_argument("--contact-only", action="store_true")
    mode.add_argument("--approach", action="store_true", help="Record one continuous approach and closure per asset.")
    mode.add_argument("--grasp-search", action="store_true", help="Compare PhysX-verified wrist and finger grasp candidates.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output_dir is None:
        experiment = ("hand_grasp_alignment" if args.grasp_search else
                      "hand_reward_approach" if args.approach else "hand_reward_sweep")
        args.output_dir = REPOSITORY_ROOT / f"logs/diagnostics/{experiment}_2026-11-26"
    if not args.no_video:
        args.enable_cameras = True
    return args


def run(args):
    from types import SimpleNamespace

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.math import compute_pose_error

    from contact_manipulation_symmetric.config.ur5e import scene
    from contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg
    from contact_manipulation_symmetric.mdp import events, observations as obs, rewards, sensors

    cfg = UR5eContactManipulationEnvCfg()
    robot_cfg = cfg.scene.robot.copy()
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    sim_cfg = cfg.sim.copy()
    sim_cfg.device, sim_cfg.render_interval = args.device, 2**31 - 1
    sim = sim_utils.SimulationContext(sim_cfg)
    stage = sim_utils.get_current_stage()
    env_path = "/World/envs/env_0"
    sim_utils.create_prim(env_path, "Xform")
    robot = Articulation(robot_cfg)
    tactile_cfg = sensors.make_tactile_contact_cfg(scene.TACTILE_BODY_NAMES)
    tactile_cfg.prim_path = tactile_cfg.prim_path.replace("{ENV_REGEX_NS}", env_path)
    tactile = tactile_cfg.class_type(tactile_cfg)

    specs = {spec.name: spec for spec in scene.ASSET_SPECS}
    source_objects = scene.make_object_collection_cfg().rigid_objects
    objects = {}
    for index, name in enumerate(args.assets):
        object_cfg = source_objects[name].copy()
        object_cfg.prim_path = f"{env_path}/TestBlocker_{index}"
        object_cfg.init_state.pos = (index * 2., 0., -10.)
        object_cfg.spawn.activate_contact_sensors = True
        object_cfg.spawn.rigid_props.kinematic_enabled = True
        object_cfg.spawn.rigid_props.disable_gravity = True
        objects[name] = RigidObject(object_cfg)

    material_cfg = cfg.scene.shared_geometry.spawn
    categories = {"hand": [f"{robot_cfg.prim_path}/ROASRightHand"],
                  "object": [obj.cfg.prim_path for obj in objects.values()]}
    for category, paths in categories.items():
        material = getattr(material_cfg, f"{category}_contact_material")
        material_path = f"/World/HandRewardMaterials/{category}"
        material.func(material_path, material)
        for path in paths:
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
                UsdShade.Material(stage.GetPrimAtPath(material_path)),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")

    writer, camera, markers = None, None, None
    video_fps = 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        light = sim_utils.DomeLightCfg(intensity=1700., color=(.92, .96, 1.))
        light.func("/World/Light", light)
        for index in range(2):
            sim_utils.create_prim(f"/World/HandRewardViews/View_{index}", "Xform")
        camera = Camera(CameraCfg(prim_path="/World/HandRewardViews/View_.*/Camera", width=640, height=480,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=20., horizontal_aperture=24.,
                                                               clipping_range=(.001, 15.))))
        markers = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/HandRewardPads", markers={
            key: sim_utils.SphereCfg(radius=.0018, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            for key, color in {"off": (.23, .36, .55), "swept": (.96, .53, .12), "contact": (.16, .96, .38)}.items()}))

    sim.reset()
    tactile.reset()

    class ProbeScene(dict):
        env_prim_paths = [env_path]

    env = SimpleNamespace(scene=ProbeScene(robot=robot, tactile_contact=tactile), num_envs=1,
                          device=sim.device, sim=sim, cfg=cfg)
    events.configure_hand_action_limits(env)
    hand = cfg.actions.hand_action.class_type(cfg.actions.hand_action.copy(), env)
    arm = cfg.actions.arm_action.class_type(cfg.actions.arm_action.copy(), env)
    reader = sensors.SensorReader(env)
    q = robot.data.default_joint_pos.clone()
    q[:, robot.joint_names.index("wrist_3_joint")] += math.pi / 2.
    zero = torch.zeros_like(q)
    raw_range = 1. / cfg.actions.hand_action.scale
    open_command = q.new_full((1, 6), -raw_range)
    hand.process_actions(open_command)
    q[:, hand._driven_joint_ids] = hand._driven_targets
    arm_ids, _ = robot.find_joints(list(scene.ARM_JOINT_NAMES), preserve_order=True)
    original_arm_limits = robot.data.joint_pos_limits[:, arm_ids].clone()
    sensor_idx = robot.body_names.index(scene.FORCE_TORQUE_BODY_NAME)
    pad_names = list(scene.TACTILE_BODY_NAMES)
    exclude_tips = cfg.rewards.blocker_normal_alignment.params["exclude_fingertips"]
    alignment_mask = [not exclude_tips or name not in scene.FINGERTIP_BODY_NAMES for name in pad_names]
    align_ids = torch.tensor(alignment_mask, device=sim.device, dtype=torch.bool)
    names = ("Thumb opposition", "Thumb flexion", "Index", "Middle", "Ring", "Little", "All fingers")
    finger_names = ("thumb", "thumb", "index", "middle", "ring", "little")
    threshold = cfg.mdp.tactile_threshold
    dt = sim.get_physics_dt()
    distance_weight, align_weight = cfg.rewards.blocker_distance.weight, cfg.rewards.blocker_normal_alignment.weight
    distance_decay = cfg.rewards.blocker_distance.params["decay"]
    align_decay = cfg.rewards.blocker_normal_alignment.params["decay"]
    wrench_scale = q.new_tensor(cfg.actions.arm_action.wrench_scale)
    pad_forces_zero = q.new_zeros(17)
    report = {"status": "running", "pad_names": pad_names, "alignment_mask": alignment_mask, "video_fps": video_fps,
              "dt": dt, "policy_decimation": cfg.decimation, "threshold_N": threshold,
              "weights": [distance_weight, align_weight, cfg.rewards.blocker_contact_hold.weight],
              "decays": [distance_decay, align_decay], "assets": {}, "geometry": [], "contact": [], "frames": [],
              "hand_material": [material_cfg.hand_contact_material.static_friction,
                                material_cfg.hand_contact_material.dynamic_friction],
              "object_material": [material_cfg.object_contact_material.static_friction,
                                  material_cfg.object_contact_material.dynamic_friction],
              "contact_fixture": "Kinematic Blocker; six arm joints constrained to ±0.0001 rad of the initial angles."}
    if args.approach:
        report["contact_fixture"] = ("Kinematic Blocker; open-hand OSC approach; six arm joints constrained "
                                     "to ±0.0001 rad of their reached angles during Close and Hold.")
        report["approach_gap_mm"] = "Start clearance proxy: 3 mm + sensor retreat projected on the approach axis."

    def set_joint_state(pose):
        robot.write_joint_state_to_sim(pose, zero)
        robot.set_joint_position_target(pose)
        robot.set_joint_velocity_target(zero)
        robot.set_joint_effort_target(zero)
        sim.forward()

    def command_at(channel, raw):
        command = open_command.clone()
        command[:, channel if channel < 6 else slice(None)] = raw
        return command

    def arm_limits(pose):
        bounds = original_arm_limits if pose is None else torch.stack((pose - .0001, pose + .0001), dim=-1)
        robot.write_joint_position_limit_to_sim(bounds, joint_ids=arm_ids, warn_limit_violation=False)

    def set_command(hand_command, position, rotation):
        current = robot.data.body_link_pose_w[:, sensor_idx]
        force_w = 300. * (position - current[0, :3]) - 40. * robot.data.body_link_lin_vel_w[0, sensor_idx]
        force_w *= 20. / force_w.norm().clamp_min(20.)
        _, error = compute_pose_error(current[:, :3], current[:, 3:7], current[:, :3], rotation[None])
        torque_w = 2. * error[0] - .1 * robot.data.body_com_ang_vel_w[0, sensor_idx]
        torque_w *= 2. / torque_w.norm().clamp_min(2.)
        inverse = obs.quat_conjugate(current[:, 3:7])
        wrench = torch.cat((obs.quat_apply(inverse, force_w[None]), obs.quat_apply(inverse, torque_w[None])), -1)
        arm.process_actions(wrench / wrench_scale)
        hand.process_actions(hand_command)

    def step_simulation(obj):
        arm.apply_actions()
        hand.apply_actions()
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        obj.update(dt)
        tactile.update(dt)

    set_joint_state(q)
    base_points, base_normals = rewards.tactile_pad_frames(env)
    normal = base_normals[0, 0]
    up = q.new_tensor([0., 0., 1.])
    # Keep the asset upright, with its narrower horizontal dimension across the palm normal.
    front = normal.clone()
    front[2] = 0.
    front /= front.norm()
    along = torch.cross(up, front, dim=-1)
    hand_width_axis = base_points[0, 5:].mean(0) - base_points[0, 0]
    if float(along @ hand_width_axis) < 0.:
        along = -along
    focus = base_points[0, 5:].mean(0)
    reference = stage.GetPrimAtPath(env_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False, ignoreVisibility=True)
    bounds_stage = Usd.Stage.CreateInMemory()

    def colliders(body):
        return [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True]

    hand_vertices = []
    for idx, path in enumerate(robot.root_physx_view.link_paths[0]):
        if "/ROASRightHand/" in str(path):
            body = stage.GetPrimAtPath(str(path))
            shapes = colliders(body)
            if shapes:
                vertices, _ = scene._object_support_vertices(body, shapes, (1., 0., 0., 0.), reference_frame=reference)
                local = q.new_tensor(vertices)
                hand_vertices.append(robot.data.body_link_pos_w[0, idx] + obs.quat_apply(
                    robot.data.body_link_quat_w[:, idx].expand(len(local), -1), local))
    hand_vertices = torch.cat(hand_vertices)
    asset_frames = {}
    sensor_paths = list(tactile.contact_physx_view.sensor_paths)
    for index, (name, obj) in enumerate(objects.items()):
        body = stage.GetPrimAtPath(str(obj.root_physx_view.prim_paths[0]))
        bounds_body = bounds_stage.OverridePrim(f"/Object_{index}")
        bounds_body.GetReferences().AddReference(stage.GetRootLayer().identifier, body.GetPath())
        upright = specs[name].upright_quat
        bounds = scene._range_bounds(bbox_cache.ComputeUntransformedBound(bounds_body))
        lower, upper = scene._upright_bounds(bounds, upright, scene._body_stretch(bounds_body))
        lower, upper = q.new_tensor(lower), q.new_tensor(upper)
        vertices, _ = scene._object_support_vertices(body, colliders(body), upright, reference_frame=reference)
        size = upper - lower
        narrow_axis = int(size[:2].argmin())
        yaw = math.atan2(float(front[1]), float(front[0])) - narrow_axis * math.pi / 2.
        yaw_q = q.new_tensor([[math.cos(yaw/2.), 0., 0., math.sin(yaw/2.)]])
        rotation = obs.quat_mul(yaw_q, q.new_tensor([upright]))
        offset = obs.quat_apply(yaw_q, ((upper + lower) * .5)[None])[0]
        vertices = q.new_tensor(vertices)
        rotated = obs.quat_apply(yaw_q.expand(len(vertices), -1), vertices) - offset
        # Use the actual source collision support for the placement gap and
        # visual/proxy USD bounds for the reward, as in the task.
        relative_hand = hand_vertices - focus
        side = rotated @ along
        side_hand = relative_hand @ along
        in_view = (side_hand >= side.min() - .004) & (side_hand <= side.max() + .004)
        in_view &= ((relative_hand[:, 2] >= rotated[:, 2].min() - .004)
                    & (relative_hand[:, 2] <= rotated[:, 2].max() + .004))
        palm_front = ((hand_vertices[in_view] if bool(in_view.any()) else hand_vertices) @ front).max()
        center = focus + (palm_front - (rotated @ front).min() - (focus @ front)) * front
        view = robot._physics_sim_view.create_rigid_contact_view(sensor_paths,
            filter_patterns=[[str(body.GetPath())]] * 17, max_contact_data_count=0)
        if list(view.sensor_paths) != sensor_paths or view.sensor_count != 17 or view.filter_count != 1:
            raise RuntimeError("Object-filtered contact view must match all native tactile rows.")
        asset_frames[name] = (center, rotation, yaw_q, offset, size * .5, view)
        report["assets"][name] = {"obb_size_m": size.cpu().tolist(), "yaw_rad": yaw,
                                   "touch_plane_center_w_m": center.cpu().tolist(), "source": specs[name].usd_path}

    def place(name, gap):
        for obj in objects.values():
            obj.write_root_state_to_sim(obj.data.default_root_state.clone())
            obj.reset()
        center, rotation, yaw_q, offset, half, _ = asset_frames[name]
        center = center + gap * front
        obj = objects[name]
        state = obj.data.default_root_state.clone()
        state[:, :3], state[:, 3:7], state[:, 7:] = (center - offset)[None], rotation, 0.
        obj.write_root_state_to_sim(state)
        obj.reset()
        sim.forward()
        return center, yaw_q, half

    def set_views(center):
        if camera is None:
            return
        target = .55 * focus + .20 * base_points[0, 0] + .25 * center
        eyes = torch.stack((target + .29 * up - .21 * front - .085 * along,
                            target + .095 * up + .16 * front + .28 * along))
        camera.set_world_poses_from_view(eyes, torch.stack((target, target)))
        for _ in range(3):
            sim.render()

    def measure(name, mode, channel, gap, raw, box, forces, baseline=None):
        points, normals = rewards.tactile_pad_frames(env)
        center, rotation, half = box
        ds = rewards._blocker_pad_distances(points, center[None], rotation, half[None])[0]
        alignment = rewards.blocker_normal_alignment_quality(points[:, align_ids], normals[:, align_ids],
                                                              center[None], decay=align_decay)[0]
        angle_cos = torch.nn.functional.cosine_similarity(center[None] - points[0], normals[0], dim=-1)
        distance_score = torch.exp(-distance_decay * ds)
        align_score = torch.exp(-align_decay * (1. - angle_cos.clamp(-1., 1.)))
        if not torch.isclose(align_score[align_ids].mean(), alignment):
            raise RuntimeError("Logged per-pad alignment does not reconstruct the production reward.")
        distance, align = float(distance_score.mean()), float(alignment)
        contacts = forces > threshold
        row = dict(asset=name, mode=mode, case=f"{name}/{gap*1000:.0f}mm/{names[channel]}",
                   joint=names[channel], gap_mm=gap*1000, raw=raw,
                   distance=distance, alignment=align, force_N=forces.cpu().tolist(), contacts=contacts.cpu().tolist(),
                   distance_mm=(1000.*ds).cpu().tolist(), cosine=angle_cos.cpu().tolist(),
                   d_delta=0. if baseline is None else distance_weight * (distance - baseline["distance"]),
                   a_delta=0. if baseline is None else align_weight * (align - baseline["alignment"]))
        return row, points[0], normals[0], contacts

    def capture(row, points, normals, contacts, channel):
        if writer is None:
            return
        color = torch.tensor([int(channel == 6 or finger_names[channel] in name) for name in pad_names],
                             device=sim.device, dtype=torch.long)
        color = torch.where(contacts, 2, color)
        markers.visualize(translations=points + normals * .001, marker_indices=color)
        sim.render()
        camera.update(1./video_fps, force_recompute=True)
        rgb = camera.data.output["rgb"][..., :3].cpu().numpy()
        writer.append_data(np.concatenate(tuple(rgb), axis=1))
        report["frames"].append(row)

    def save():
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def grasp_trials():
        from itertools import product
        from scipy.spatial.transform import Rotation
        from probe_hand_penetration import detailed_contacts

        env.scene.stage = stage
        env.scene.env_regex_ns = env_path
        env.scene.cfg = SimpleNamespace(shelf=SimpleNamespace(prim_path=f"{env_path}/ProbeShelf"))
        sim_utils.create_prim(f"{env_path}/ProbeShelf", "Xform")
        set_joint_state(q)
        model = scene.RobotPoseModel.from_env(env)
        palm = model.forward(q[0].cpu().numpy())[scene.TCP_BODY_NAME]
        tangent = focus - base_points[0, 0]
        tangent = tangent - (tangent @ front) * front
        tangent /= tangent.norm()
        thumb_axis = torch.cross(front, tangent, dim=-1)
        if float(thumb_axis @ (base_points[0, 1] - base_points[0, 0])) < 0.:
            thumb_axis = -thumb_axis
        axes = torch.stack((front, tangent, thumb_axis)).cpu().numpy()
        local_poses = {
            "nominal": (0, 0, 0, 0, 0, 0),
            **{f"proximal {s} mm": (0, s, 0, 0, 0, 0) for s in (12, 24, 36)},
            "toward thumb": (0, 12, -12, 0, 0, 0),
            "toward thumb deep": (5, 24, -18, 0, 0, 0),
            "toward little": (0, 12, 12, 0, 0, 0),
            "inward": (5, 0, 0, 0, 0, 0),
            "outward": (-5, 0, 0, 0, 0, 0),
        }
        for axis, label, values in ((0, "roll", (-40, -20, 20, 40)), (1, "pitch", (-18, 18)), (2, "yaw", (-25, 25))):
            for angle in values:
                angles = [0, 0, 0]
                angles[axis] = angle
                local_poses[f"{label} {angle:+d}°"] = (0, 12, -6, *angles)
        hand_paths = [str(path) for path in robot.root_physx_view.link_paths[0] if "/ROASRightHand/" in str(path)]
        thumb_ids = [index for index, path in enumerate(hand_paths) if path.rsplit("/", 1)[-1].startswith("right_thumb")]
        other_ids = [index for index, path in enumerate(hand_paths) if path.rsplit("/", 1)[-1].startswith(
            ("right_index", "right_middle", "right_ring", "right_little"))]
        full_close = (2.,) * 6
        report["search"] = {}
        report["search_poses"] = local_poses
        report["search_axes_world"] = axes.tolist()
        report["search_gates"] = {"min_consecutive_hold_s": .4, "min_held_pads": 2,
                                  "max_native_penetration_mm": 5.0,
                                  "opposition_cosine_below": -.25}

        for name, obj in objects.items():
            arm_limits(None)
            set_joint_state(q)
            center, yaw, half = place(name, .003)
            _, _, _, offset, _, view = asset_frames[name]
            body_view = robot._physics_sim_view.create_rigid_contact_view(hand_paths,
                filter_patterns=[[str(obj.root_physx_view.prim_paths[0])]] * len(hand_paths), max_contact_data_count=4096)
            if list(body_view.sensor_paths) != hand_paths or body_view.filter_count != 1:
                raise RuntimeError("Inconsistent whole-hand contact view in grasp search.")
            pivot = center.cpu().numpy()
            pose_values = np.asarray(list(local_poses.values()), dtype=float)
            rotation = Rotation.from_rotvec(np.deg2rad(pose_values[:, 3:]) @ axes).as_matrix()
            targets = pivot + (rotation @ (palm[:3, 3] - pivot)) + (pose_values[:, :3] * .001) @ axes
            poses, valid, *_ = model.solve_batch(q.new_tensor(targets), q.new_tensor(rotation @ palm[:3, :3]),
                                                  q[:, arm_ids].expand(len(targets), -1))
            poses[:, hand._driven_joint_ids] = q[:, hand._driven_joint_ids]
            poses[0] = q[0]
            valid[0] = True
            pose_lookup = dict(zip(local_poses, poses.split(1)))
            feasible = {key for key, ok in zip(local_poses, valid.cpu().tolist()) if ok}
            trials = []
            tested = set()
            comparison = {"trials": trials, "ik_rejected": sorted(set(local_poses) - feasible)}
            report["search"][name] = comparison

            def trial(pose_key, command, *, record_label=None):
                pose = pose_lookup[pose_key]
                arm_limits(None)
                set_joint_state(pose)
                arm_limits(pose[:, arm_ids])
                robot.reset()
                tactile.reset()
                hand.reset()
                arm.reset()
                arm_anchor = robot.data.body_link_pose_w[0, sensor_idx].clone()
                target = q.new_tensor([command])
                initial_force = 0.
                samples, baseline = [], None
                previous = torch.zeros_like(pad_forces_zero, dtype=torch.bool)
                samples_last = 20
                hold_start = 170 - cfg.decimation * samples_last
                for step in range(170):
                    amount = smooth_step((step - 16.) / 100.)
                    if step % cfg.decimation == 0:
                        set_command(open_command + amount * (target - open_command), arm_anchor[:3], arm_anchor[3:7])
                    step_simulation(obj)
                    if step % cfg.decimation != cfg.decimation - 1:
                        continue
                    # Read one step before the hold window to preserve contact continuity.
                    if step != 15 and step < hold_start - cfg.decimation and record_label is None:
                        continue
                    force = view.get_contact_force_matrix(dt=dt).reshape(1, 17, 3).index_select(1, reader.tactile_ids)
                    body = body_view.get_contact_force_matrix(dt=dt).reshape(-1, 3)
                    thumb_force, other_force = body[thumb_ids].sum(0), body[other_ids].sum(0)
                    opposition = float(torch.nn.functional.cosine_similarity(thumb_force, other_force, dim=0))
                    opposed = bool(thumb_force.norm() > threshold and other_force.norm() > threshold and opposition < -.25)
                    sample = measure(name, "contact", 6, .003, -raw_range + 2 * raw_range * amount,
                                     (obj.data.root_link_pose_w[0, :3] + offset, yaw, half), force[0].norm(dim=-1), baseline)
                    row = sample[0]
                    held, new, previous = rewards.blocker_contact_hold(force, previous[None], threshold=threshold)
                    previous = previous[0]
                    row.update(case=f"{record_label or pose_key}", joint="Closure fraction", raw=amount,
                               contact_hold=float(held + .5 * new), opposed=opposed, opposition_cosine=opposition,
                               time_s=(step+1)*dt, hand_raw=(open_command + amount * (target - open_command))[0].cpu().tolist())
                    if step == 15:
                        initial_force = float(body.norm(dim=-1).max())
                    if baseline is None:
                        baseline = row
                    if step >= hold_start:
                        samples.append(row)
                    if record_label is not None:
                        report["contact"].append(row)
                        if step % 10 == 9:
                            capture(*sample, 6)
                contact_data = detailed_contacts(body_view, dt, hand_paths)
                penetration = max((max(0., -point["separation_m"]) for point in contact_data), default=0.) * 1000.
                held_bits = np.asarray([row["contacts"] for row in samples]).all(0)
                alignment = float(np.mean([row["alignment"] for row in samples]))
                distance = float(np.mean([row["distance"] for row in samples]))
                hold = float(np.mean([row["contact_hold"] for row in samples]))
                stable = int(held_bits.sum()) >= 2 and penetration <= 5. and initial_force <= threshold
                return dict(pose=pose_key, command=list(command), alignment=alignment, distance=distance,
                            hold=hold, total=distance_weight*distance + align_weight*alignment + report["weights"][2]*hold,
                            held_pad_names=[pad for pad, held in zip(pad_names, held_bits) if held],
                            opposing=bool(all(row["opposed"] for row in samples)), eligible=stable,
                            baseline_open_force_N=initial_force, max_final_penetration_mm=penetration,
                            contact_anchors=contact_data)

            def evaluate(pose_key, command):
                command = tuple(command)
                if pose_key in feasible and (pose_key, command) not in tested:
                    tested.add((pose_key, command))
                    trials.append(trial(pose_key, command))

            evaluate("nominal", full_close)
            for thumb in product((-2., 0., 2.), repeat=2):
                evaluate("nominal", (*thumb, *full_close[2:]))
            save()
            print(f"GRASP SEARCH {name}: finished {len(trials)} nominal thumb settings", flush=True)
            rank = lambda row: (row["eligible"], row["alignment"])
            best_nominal = max(trials, key=rank)
            for pose_key in local_poses:
                evaluate(pose_key, best_nominal["command"])
            # Evaluate opposing-grip probes and refine fingers around the best physically supported candidates.
            for pose_key in ("proximal 24 mm", "toward thumb", "toward thumb deep", "roll -20°", "roll +20°"):
                evaluate(pose_key, full_close)
                evaluate(pose_key, (0., 1., 2., 2., 2., 1.))
            seed_rows = {"highest_alignment": max(trials, key=rank)}
            opposed = [row for row in trials if row["eligible"] and row["opposing"]]
            if opposed:
                seed_rows["opposed"] = max(opposed, key=rank)
            for seed in seed_rows.values():
                for axis, delta in product(range(6), (-1., 1.)):
                    command = seed["command"].copy()
                    command[axis] = min(2., max(-2., command[axis] + delta))
                    evaluate(seed["pose"], command)
            eligible = [row for row in trials if row["eligible"]]
            opponents = [row for row in eligible if row["opposing"]]
            selected = {"baseline": trials[0]}
            if eligible:
                selected["best_alignment"] = max(eligible, key=lambda row: row["alignment"])
            if opponents:
                selected["best_opposed"] = max(opponents, key=lambda row: row["alignment"])
            comparison["selected"] = selected
            save()
            print(f"GRASP SEARCH {name}: {len(trials)} candidates, {len(eligible)} stable, {len(opponents)} opposed", flush=True)
            if camera is not None:
                set_views(center)
                comparison["replay"] = {label: trial(row["pose"], row["command"], record_label=label)
                                        for label, row in selected.items()}
            save()

    if camera is not None:
        writer = imageio.get_writer(args.output_dir / "raw.mp4", fps=video_fps, codec="libx264", quality=8)
    try:
        if args.grasp_search:
            grasp_trials()
            report["status"] = "complete"
            return
        for name in (() if args.contact_only or args.approach else objects):
            for gap in (.060, .003):
                box = place(name, gap)
                if name in (args.assets[0], args.assets[-1]):
                    set_views(box[0])
                for channel in range(len(names)):
                    baseline = None
                    for raw in np.linspace(-raw_range, raw_range, 17):
                        hand.process_actions(command_at(channel, raw))
                        pose = q.clone()
                        pose[:, hand._driven_joint_ids] = hand._driven_targets
                        set_joint_state(pose)
                        if not torch.equal(robot.data.joint_pos[:, arm_ids], q[:, arm_ids]):
                            raise RuntimeError("Arm coordinates changed in the frozen geometry sweep.")
                        sample = measure(name, "geometry", channel, gap, float(raw), box, pad_forces_zero, baseline)
                        row = sample[0]
                        baseline = row if baseline is None else baseline
                        report["geometry"].append(row)
                        if name in (args.assets[0], args.assets[-1]) and channel in (2, 6):
                            capture(*sample, channel)
                    print(f"GEOMETRY {name} {gap*1000:.0f}mm {names[channel]} done", flush=True)
            save()
        if not args.geometry_only:
            set_joint_state(q)
            if not args.approach:
                arm_limits(q[:, arm_ids])
            for name, obj in objects.items():
                _, _, _, offset, _, view = asset_frames[name]
                gap = .003
                box = place(name, gap)
                set_views(box[0])
                channels = (6,) if args.approach else (1, 2, 6)
                for channel in channels:
                    if args.approach:
                        arm_limits(None)
                    set_joint_state(q)
                    robot.reset()
                    tactile.reset()
                    hand.reset()
                    arm.reset()
                    reader.reset(torch.tensor([0], device=sim.device))
                    sensor_pose = robot.data.body_link_pose_w[0, sensor_idx].clone()
                    anchor, rotation = sensor_pose[:3], sensor_pose[3:7]
                    if args.approach:
                        # Retreat before the recording, then approach the same world-fixed Blocker.
                        start_offset = .057 * front
                        for step in range(220):
                            if step % cfg.decimation == 0:
                                set_command(open_command, anchor - smooth_step(step / 150.) * start_offset, rotation)
                            step_simulation(obj)
                        trial_steps, close_start, close_steps = 520, 260, 185
                    else:
                        trial_steps, close_start, close_steps = 300, 45, 185
                    initial_object_pose = obj.data.root_link_pose_w[0].clone()
                    baseline = None
                    previous_contact = torch.zeros_like(pad_forces_zero, dtype=torch.bool)
                    raw = -raw_range
                    for step in range(trial_steps):
                        target = anchor
                        if args.approach:
                            target = anchor - (1. - smooth_step((step - 30.) / 160.)) * start_offset
                            if step == close_start:
                                # Constrain the arm at its reached pose without writing its state.
                                arm_limits(robot.data.joint_pos[:, arm_ids])
                                sensor_pose = robot.data.body_link_pose_w[0, sensor_idx].clone()
                            if step >= close_start:
                                target, rotation = sensor_pose[:3], sensor_pose[3:7]
                        if step % cfg.decimation == 0:
                            raw = raw_range * (2. * smooth_step((step - close_start) / close_steps) - 1.)
                            set_command(command_at(channel, raw), target, rotation)
                        step_simulation(obj)
                        if step % cfg.decimation != cfg.decimation - 1:
                            continue
                        force_vectors = view.get_contact_force_matrix(dt=dt).reshape(1, 17, 3).index_select(
                            1, reader.tactile_ids)
                        forces = force_vectors[0].norm(dim=-1)
                        object_pose = obj.data.root_link_pose_w[0]
                        current_box = (object_pose[:3] + offset, box[1], box[2])
                        sample = measure(name, "contact", channel, gap, raw, current_box, forces, baseline)
                        row = sample[0]
                        held, new, previous_contact = rewards.blocker_contact_hold(
                            force_vectors, previous_contact[None], threshold=threshold)
                        previous_contact = previous_contact[0]
                        wrist = robot.data.body_link_pose_w[0, sensor_idx]
                        row.update(time_s=(step+1)*dt, wrist_drift_mm=1000.*float((wrist[:3]-sensor_pose[:3]).norm()),
                                   wrist_drift_deg=math.degrees(2.*math.acos(min(1., abs(float(wrist[3:7] @ sensor_pose[3:7]))))),
                                   object_drift_mm=1000.*float((object_pose[:3]-initial_object_pose[:3]).norm()),
                                   hand_position_rad=robot.data.joint_pos[0, hand._driven_joint_ids].cpu().tolist(),
                                   contact_hold=float(held + .5 * new))
                        if args.approach:
                            phase = ("Start" if step < 30 else "Approach" if step < 190 else
                                     "Open near" if step < close_start else "Close" if step < close_start + close_steps else "Hold")
                            row.update(phase=phase, approach_gap_mm=1000. * (gap + float((anchor - wrist[:3]) @ front)),
                                       tracking_error_mm=1000. * float((target - wrist[:3]).norm()),
                                       arm_max_speed_rad_s=float(robot.data.joint_vel[:, arm_ids].abs().max()),
                                       wrist_position_w=wrist[:3].cpu().tolist())
                        if step < (30 if args.approach else close_start):
                            baseline = row
                        report["contact"].append(row)
                        if channel in (2, 6) and step % 10 == 9:
                            capture(*sample, channel)
                    print(f"CONTACT {name} {names[channel]} done", flush=True)
                save()
        report["status"] = "complete"
    finally:
        if writer is not None:
            writer.close()
        save()


if __name__ == "__main__":
    import traceback

    launch_args = parse_args()
    app = AppLauncher(launch_args).app
    try:
        run(launch_args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)
