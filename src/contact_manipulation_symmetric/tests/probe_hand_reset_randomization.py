"""Audit randomized raw Hand reset actions in production Isaac Sim scenes."""

import argparse
import itertools
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/contact_manipulation_symmetric"))

from isaaclab.app import AppLauncher


LAYOUTS = (
    (-1, "direct", "direct"), (1, "direct", "front"), (-1, "direct", "blocked"),
    (1, "front", "direct"), (-1, "front", "front"), (1, "front", "blocked"),
    (-1, "blocked", "direct"), (1, "blocked", "front"), (-1, "blocked", "front"),
)


def load_raw_actions(path):
    values = json.loads(path.read_text())
    if isinstance(values, dict):
        values = values.get("raw_actions")
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a non-empty array of six-channel raw actions.")
    for index, row in enumerate(values):
        if not isinstance(row, list) or len(row) != 6 or any(
            type(value) not in (int, float) or not -2 <= value <= 2 for value in row
        ):
            raise ValueError(f"Row {index + 1} must contain six numbers in [-2, 2].")
    return values


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--seed", type=int, default=2511)
    parser.add_argument("--num-scenes", type=int)
    parser.add_argument("--physics-steps", type=int, default=8)
    parser.add_argument("--raw-actions-file", type=Path, help="JSON array of six-channel raw actions in [-2, 2].")
    parser.add_argument("--export-hand-hulls", type=Path, help="Write the native cooked Hand hulls as JSON.")
    parser.add_argument("--hand-envelope-certificate", type=Path,
                        help="NPZ table of certified Hand envelopes for matching arm starts.")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "outputs/contact_manipulation_symmetric/hand_reset_randomization_2026-11-25/native.json")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.raw_actions = None
    if args.raw_actions_file is not None:
        try:
            args.raw_actions = load_raw_actions(args.raw_actions_file)
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read {args.raw_actions_file}: {error}")
        if args.num_envs is not None and args.num_envs != len(args.raw_actions):
            parser.error("The number of environments must match the number of input actions.")
        args.num_envs = len(args.raw_actions)
        args.num_scenes = 1 if args.num_scenes is None else args.num_scenes
    elif args.export_hand_hulls is not None:
        args.num_envs = 1 if args.num_envs is None else args.num_envs
        args.num_scenes = 1 if args.num_scenes is None else args.num_scenes
    else:
        args.num_envs = 64 if args.num_envs is None else args.num_envs
        args.num_scenes = len(LAYOUTS) if args.num_scenes is None else args.num_scenes
        if args.num_envs < 64 or args.num_scenes < 2:
            parser.error("The default sweep requires at least 64 environments and two scenes.")
    if args.num_envs < 1 or args.num_scenes < 1 or args.physics_steps < 1:
        parser.error("Use at least one environment, one scene, and one physics step.")
    if args.hand_envelope_certificate is not None and args.raw_actions is None:
        parser.error("The envelope guard requires --raw-actions-file from the certified box table.")
    return args


def export_hand_hulls(env, output):
    """Cook each enabled Hand collision mesh with its production USD settings."""
    import numpy as np
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

    from contact_manipulation_symmetric.config.ur5e.scene import _gf_column, _rotation_part

    model = env.robot_pose_model
    robot = env.scene["robot"]
    stage = env.scene.stage
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        raise RuntimeError("Expected the production stage to use meters.")
    poses = robot.data.joint_pos[0].detach().cpu().numpy()
    body_paths = {name: str(path) for name, path in zip(model.body_names, robot.root_physx_view.link_paths[0])
                  if "/ROASRightHand/" in str(path)}
    names = {name: path.rsplit("/", 1)[-1] for name, path in body_paths.items()}
    transforms = model.forward(poses)
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    cooking = get_physx_cooking_interface()
    bodies, colliders = {}, []
    for box in model.colliders:
        if box.body_name not in body_paths:
            continue
        body = stage.GetPrimAtPath(body_paths[box.body_name])
        collider = stage.GetPrimAtPath(box.path)
        name = names[box.body_name]
        body_world = _gf_column(xforms.GetLocalToWorldTransform(body))
        body_world[:3, :3] = _rotation_part(body_world)
        meshes = [prim for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise RuntimeError(f"No source collision mesh found under {collider.GetPath()}.")
        for source in meshes:
            # URDF collision schemas live on Xform containers, but the cooking API
            # only accepts UsdGeomMesh. Reproduce the input on a temporary stage.
            scratch = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(scratch, UsdGeom.GetStageMetersPerUnit(stage))
            UsdGeom.SetStageUpAxis(scratch, UsdGeom.GetStageUpAxis(stage))
            mesh = UsdGeom.Mesh.Define(scratch, "/CollisionMesh")
            source_mesh = UsdGeom.Mesh(source)
            mesh.CreatePointsAttr(source_mesh.GetPointsAttr().Get())
            mesh.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
            mesh.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())
            mesh.CreateOrientationAttr(source_mesh.GetOrientationAttr().Get())
            for schema in collider.GetAppliedSchemas():
                mesh.GetPrim().AddAppliedSchema(schema)
            for source_attr in collider.GetAuthoredAttributes():
                attr_name = source_attr.GetName()
                if attr_name.startswith(("physics:", "physx")) and source_attr.Get() is not None:
                    mesh.GetPrim().CreateAttribute(attr_name, source_attr.GetTypeName()).Set(source_attr.Get())

            result = []

            def capture(status, convexes):
                if status == PhysxCollisionRepresentationResult.RESULT_VALID:
                    result.extend([[(point.x, point.y, point.z) for point in hull.vertices] for hull in convexes])

            cache = UsdUtils.StageCache.Get()
            stage_id = cache.Insert(scratch)
            try:
                cooking.request_convex_collision_representation(
                    stage_id=stage_id.ToLongInt(), collision_prim_id=PhysicsSchemaTools.sdfPathToInt(mesh.GetPath()),
                    run_asynchronously=False, on_result=capture,
                )
            finally:
                cache.Erase(stage_id)
            if not result:
                raise RuntimeError(f"No cooked PhysX hulls returned for {source.GetPath()}.")
            mesh_body = np.linalg.inv(body_world) @ _gf_column(xforms.GetLocalToWorldTransform(source))
            hulls = [(np.asarray(hull) @ mesh_body[:3, :3].T + mesh_body[:3, 3]).tolist() for hull in result]
            bodies.setdefault(name, []).extend(hulls)
            colliders.append(dict(body=name, prim=str(collider.GetPath()), mesh=str(source.GetPath()),
                                  hull_count=len(hulls), mesh_stretch=np.linalg.svd(mesh_body[:3, :3], compute_uv=False).tolist(),
                                  approximation=UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()))
    if not bodies:
        raise RuntimeError("No Hand collision shapes found.")

    def sample_transforms(q):
        world = model.forward(q)
        return dict(joint_positions=dict(zip(model.joint_names, q.tolist())),
                    body_transforms={names[name]: world[name].tolist() for name in body_paths})

    # Check coordinate remapping independently of the reference posture. The
    # policy action term gives the coupled joint commands without stepping PhysX.
    hand = env.action_manager.get_term("hand_action")
    sample_q = robot.data.joint_pos[0].clone()
    checks = []
    raw = (-1.8, 1.7, -0.3, 0.7, 1.5, -0.9)
    for command in (raw, raw[::-1]):
        hand.process_actions(sample_q.new_tensor(command).expand(env.num_envs, -1))
        sample_q[hand._driven_joint_ids] = hand._driven_targets[0]
        checks.append(dict(raw=command, **sample_transforms(sample_q.detach().cpu().numpy())))

    result = dict(frame="usd_rigid_body_local_m", cooking_source="production mesh and schema on scratch USD prims",
                  base_body="base_link", bodies=bodies, colliders=colliders,
                  reference_pose=dict(zip(model.joint_names, poses.tolist())),
                  reference_body_transforms={names[name]: transforms[name].tolist() for name in body_paths},
                  fk_checks=checks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"HAND_COOKED_HULLS {len(bodies)} bodies, {sum(map(len, bodies.values()))} hulls: {output}", flush=True)


def geometry_masks(env, sample, joint_positions):
    """Re-evaluate the production clearance bounds with actual varying Hand FK."""
    import numpy as np
    import torch
    from contact_manipulation_symmetric.mdp.events import _scene_collision_boxes, oriented_boxes_separated_batch

    model = env.robot_pose_model
    hand = [box for box in model.colliders if "/ROASRightHand/" in box.path]
    names = tuple(dict.fromkeys(box.body_name for box in hand))
    pose = [model.forward(q) for q in joint_positions.detach().cpu().numpy()]
    transforms = np.stack([[row[name] for name in names] for row in pose])
    selected = transforms[:, [names.index(box.body_name) for box in hand]]
    rotation = selected[..., :3, :3]
    world_centers = (
        np.einsum("nbij,bj->nbi", rotation, np.stack([box.center_local for box in hand]))
        + selected[..., :3, 3]
    )
    world_rotations = rotation @ np.stack([box.rotation_local for box in hand])
    obstacles = _scene_collision_boxes(env, sample)
    to_gpu = lambda array: torch.as_tensor(array, device=env.device, dtype=torch.float64)
    center, rotation, half = map(to_gpu, (world_centers, world_rotations, np.stack([b.half_extents for b in hand])))
    scene_center, scene_rotation, scene_half = (
        to_gpu(np.stack([box[column] for box in obstacles])) for column in range(3)
    )
    split = len(sample.active_ids)

    def failures(clearance):
        separate = oriented_boxes_separated_batch(
            center[:, :, None], rotation[:, :, None], half[None, :, None],
            scene_center[None, None], scene_rotation[None, None], scene_half[None, None], clearance,
        )
        return torch.stack(((~separate[..., :split]).any((-2, -1)),
                            (~separate[..., split:]).any((-2, -1))), -1).cpu().tolist()

    return failures(0.0), failures(env.cfg.robot_start.collision_clearance)


def compatible_hand_arms(env, sample, lower, upper, covering_boxes):
    """Match swept Hand boxes to existing valid arm anchors, in one batch."""
    import numpy as np
    import torch
    from isaaclab.utils.math import matrix_from_quat
    from contact_manipulation_symmetric.mdp.events import _scene_collision_boxes, oriented_boxes_separated_batch

    robot = env.scene["robot"]
    tcp_id = robot.body_names.index(env.robot_pose_model.tcp_body_name)
    obstacles = _scene_collision_boxes(env, sample)
    obstacle_center, obstacle_rotation, obstacle_half = (
        lower.new_tensor(np.stack([box[index] for box in obstacles])) for index in range(3))
    origin_offset = env.scene.env_origins - env.scene.env_origins[0]
    tcp = robot.data.body_link_pos_w[:, tcp_id] - origin_offset
    rotation = matrix_from_quat(robot.data.body_link_quat_w[:, tcp_id])
    center, half = (lower + upper) * 0.5, (upper - lower) * 0.5
    world_center = (rotation @ center.T).transpose(-1, -2) + tcp[:, None, :]
    world_half = (rotation.abs() @ half.T).transpose(-1, -2)
    obstacle_world_half = (obstacle_rotation.abs() @ obstacle_half[..., None]).squeeze(-1)
    delta = (world_center[:, :, None] - obstacle_center).abs()
    gap = (delta - world_half[:, :, None] - obstacle_world_half).amax(-1).amin(-1)
    required = env.cfg.robot_start.collision_clearance + 1.0e-6
    compatible = gap >= required
    # A world-axis test is cheap. Use the existing OBB SAT once for boxes
    # that still lack an arm, since a tilted separator may be available.
    unresolved = (~compatible.any(dim=0)).nonzero().flatten()
    if unresolved.numel():
        compatible[:, unresolved] = oriented_boxes_separated_batch(
            world_center[:, unresolved, None], rotation[:, None, None], half[unresolved, None],
            obstacle_center, obstacle_rotation, obstacle_half, required,
        ).all(-1)
    matching = compatible.float() @ covering_boxes.T > 0
    if not matching.any(dim=0).all():
        missing = int((~matching.any(dim=0)).sum())
        raise RuntimeError(f"No existing arm anchor has the requested clearance for {missing} Hand poses.")
    original = torch.arange(len(covering_boxes), device=covering_boxes.device)
    fallback = matching.int().argmax(dim=0)
    selected = torch.where(matching.diagonal(), original, fallback)
    return selected, compatible.any(dim=0), len(unresolved)


def hand_reports(view, hand_paths, filters, categories, row_size, dt):
    """Decode the native GPU pair-filtered contact patches and separations."""
    import torch

    _, _, _, separation, counts, starts = view.get_contact_data(dt=dt)
    active = (counts > 0).nonzero(as_tuple=False)
    if not active.numel():
        return
    rows, columns = active.unbind(1)
    pairs = active.cpu().tolist()
    sizes = counts[rows, columns].cpu().tolist()
    offsets = starts[rows, columns].cpu().tolist()
    patch_ids = []
    for count, first in zip(sizes, offsets):
        if first < 0 or first+count > len(separation):
            raise RuntimeError("Native Hand contact buffer exhausted.")
        patch_ids.extend(range(first, first+count))
    values = separation.flatten()[torch.tensor(patch_ids, device=separation.device)].cpu().tolist()
    cursor = 0
    for (row, column), count in zip(pairs, sizes):
        worst = min(values[cursor:cursor+count])
        cursor += count
        yield (row//row_size, categories[column], worst, hand_paths[row].rsplit("/", 1)[-1],
               filters[row][column].rsplit("/", 1)[-1])


def run(args):
    import gymnasium as gym
    import numpy as np
    import torch

    import contact_manipulation_symmetric
    from contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg
    from contact_manipulation_symmetric.mdp.events import _write_robot_reset

    cfg = UR5eContactManipulationEnvCfg()
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.render_interval = 2**31-1
    cfg.layout_scene_override = LAYOUTS[0]
    envelopes = None
    if args.hand_envelope_certificate is not None:
        from probe_hand_reset_sampler import gaps_by_axis

        with np.load(args.hand_envelope_certificate.with_suffix(".npz")) as boxes:
            action_lo, action_hi = boxes["lower"], boxes["upper"]
            raw = np.asarray(args.raw_actions)[:, None, :]
            covered = ((action_lo <= raw) & (raw <= action_hi)).all(axis=-1)
            if not covered.any(axis=-1).all():
                raise ValueError("A diagnostic pose is outside all certified action boxes.")
            envelopes = (boxes["tcp_envelope_lower_m"], boxes["tcp_envelope_upper_m"], covered)
    if args.device is not None:
        cfg.sim.device = args.device
    env = gym.make("Isaac-Contact-Manipulation-Symmetric-UR5e-v0", cfg=cfg).unwrapped
    if args.export_hand_hulls is not None:
        try:
            env.reset()
            export_hand_hulls(env, args.export_hand_hulls)
        except Exception:
            env.close()
            raise
        if args.raw_actions_file is None:
            env.close()
            return
    robot, shelf, objects = env.scene["robot"], env.scene["shelf"], env.scene["object_collection"]
    hand = env.action_manager.get_term("hand_action")
    n = env.num_envs
    ids = torch.arange(n, device=env.device)
    if envelopes is not None:
        envelope_lo = torch.as_tensor(envelopes[0], dtype=torch.float32, device=env.device)
        envelope_hi = torch.as_tensor(envelopes[1], dtype=torch.float32, device=env.device)
        covering_boxes = torch.as_tensor(envelopes[2], dtype=torch.float32, device=env.device)
    rng = torch.Generator().manual_seed(args.seed ^ 0x317)
    hand_paths, filters = [], []
    shelf_paths = list(map(str, shelf.root_physx_view.prim_paths))
    object_paths = list(map(str, objects.root_physx_view.prim_paths))
    hand_num, arm_num = None, None
    for env_id, links in enumerate(robot.root_physx_view.link_paths):
        links = list(map(str, links))
        own = [path for path in links if "/ROASRightHand/" in path]
        arm = [path for path in links if path not in own]
        hand_num, arm_num = len(own), len(arm)
        obstacles = arm + [shelf_paths[env_id]] + object_paths[env_id::n]
        for path in own:
            hand_paths.append(path)
            filters.append([other for other in own if other != path] + obstacles)
    categories = ["self"]*(hand_num-1) + ["arm"]*arm_num + ["shelf"] + ["object"]*objects.num_objects
    contact_view = robot._physics_sim_view.create_rigid_contact_view(
        hand_paths, filter_patterns=filters, max_contact_data_count=32768)
    if (list(map(str, contact_view.sensor_paths)) != hand_paths or len(hand_paths) != hand_num*n
            or contact_view.filter_count != len(categories)):
        raise RuntimeError("Unexpected Hand contact view ordering.")
    report = dict(seed=args.seed, scale=cfg.actions.hand_action.scale, num_envs=n, layout_count=args.num_scenes,
                  physics_dt=cfg.sim.dt, physics_steps=args.physics_steps, robot_start_clearance_m=cfg.robot_start.collision_clearance,
                  hand_body_count=len(hand_paths)//n, scenes=[],
                  raw_actions_file=str(args.raw_actions_file.resolve()) if args.raw_actions_file else None,
                  envelope_guard=envelopes is not None)

    def trial(sample, initial_q, object_state, shelf_state, raw):
        actions = initial_q.new_zeros((n, env.action_manager.total_action_dim))
        actions[:, -6:] = raw
        env.action_manager.process_action(actions)
        q = initial_q.clone()
        q[:, hand._driven_joint_ids] = hand._driven_targets
        B, C = geometry_masks(env, sample, q)
        objects.write_object_state_to_sim(object_state)
        shelf.write_root_state_to_sim(shelf_state)
        _write_robot_reset(env, ids, q)
        env.sim.forward()
        pairs = [{} for _ in range(n)]
        peak_force, peak_speed, peak_error = (q.new_zeros(n) for _ in range(3))
        for _ in range(args.physics_steps):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=cfg.sim.dt)
            force = contact_view.get_net_contact_forces(dt=cfg.sim.dt).reshape(n, -1, 3)
            peak_force = torch.maximum(peak_force, torch.linalg.vector_norm(force, dim=-1).amax(-1))
            speed = robot.data.joint_vel[:, hand._driven_joint_ids].abs().amax(-1)
            error = (robot.data.joint_pos[:, hand._driven_joint_ids] - q[:, hand._driven_joint_ids]).abs().amax(-1)
            peak_speed = torch.maximum(peak_speed, speed)
            peak_error = torch.maximum(peak_error, error)
            for row, cat, sep, left, right in hand_reports(
                    contact_view, hand_paths, filters, categories, hand_num, cfg.sim.dt):
                prior = pairs[row].get(cat)
                if prior is None or sep < prior["separation_m"]:
                    pairs[row][cat] = dict(separation_m=sep, hand=left, other=right)
        return [dict(raw=command, scene_box_overlap=overlap, scene_clearance_violation=margin,
                     worst_contact=contact, peak_hand_link_net_force_N=force, peak_hand_speed_rad_s=speed,
                     peak_target_error_rad=error)
                for command, overlap, margin, contact, force, speed, error in zip(
                    raw.cpu().tolist(), B, C, pairs, peak_force.cpu().tolist(), peak_speed.cpu().tolist(),
                    peak_error.cpu().tolist())]

    try:
        for scene_index in range(args.num_scenes):
            layout = LAYOUTS[scene_index % len(LAYOUTS)]
            env.cfg.layout_scene_override = layout
            if scene_index:
                env._episode_reset.next_scene_step = env.common_step_counter
            if scene_index or args.export_hand_hulls is None:
                env.reset()
            sample = env._episode_reset.shared_sample
            initial_q = robot.data.joint_pos.clone()
            guard = None
            if envelopes is not None:
                started = time.perf_counter()
                selected, admissible, sat_boxes = compatible_hand_arms(env, sample, envelope_lo, envelope_hi, covering_boxes)
                initial_q = initial_q[selected]
                changed = int((selected != ids).sum())
                usable = admissible.cpu().numpy()
                guard = dict(eligible_boxes=int(usable.sum()), arm_reassignments=changed,
                             sat_boxes=sat_boxes, elapsed_ms=(time.perf_counter() - started) * 1000,
                             raw_support_gaps=gaps_by_axis(action_lo[usable], action_hi[usable]))
            object_state = objects.data.object_state_w.clone()
            shelf_state = shelf.data.root_state_w.clone()
            corner_count = 0
            if args.raw_actions is not None:
                raw = torch.tensor(args.raw_actions, dtype=initial_q.dtype)
            else:
                raw = torch.rand((n, 6), generator=rng, dtype=initial_q.dtype).mul_(2).sub_(1)
                if scene_index == args.num_scenes - 1:
                    corners = torch.tensor(list(itertools.product((-1., 1.), repeat=6)), dtype=raw.dtype)
                    raw[:len(corners)] = corners
                    corner_count = len(corners)
            controls = trial(sample, initial_q, object_state, shelf_state, torch.zeros_like(raw).to(env.device))
            randomized = trial(sample, initial_q, object_state, shelf_state, raw.to(env.device))
            row = dict(index=scene_index, layout=layout, assets=len(sample.active_ids),
                       blocker=env.spawn_geometries[sample.blocker_id].asset_name, corner_count=corner_count,
                       guard=guard,
                       controls=controls, randomized=randomized)
            report["scenes"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"HAND_RESET_SCENE {scene_index + 1}/{args.num_scenes} {layout} "
                  f"baseline_reports={sum(bool(r['worst_contact']) for r in controls)} "
                  f"sample_reports={sum(bool(r['worst_contact']) for r in randomized)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    app = AppLauncher(args).app
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()
