"""Measure RH56E2 motion under a controlled native object contact load.

Run through Isaac Lab, for example::

    ./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
        --headless --device cuda:0 --prismatic-guide --output /tmp/hand_contact.json

The selected package supplies the URDF, joint/collision setup, hand actuator,
simulation timestep and articulation solver settings. A fixed hand base and
zero gravity isolate the hand from arm control, rewards and the learned policy.
A dynamic sphere loads a distal pad in the negative flexion direction. The
optional native prismatic guide prevents lateral escape. No joint states are
overwritten during the test, and no physics parameters are tuned here.
Master targets use the same startup action-limit helper as training; the
physical limits on all twelve joints remain independent of that action range.

For a contact-free close/open cycle, use --target-flexion 1
--return-to-open-steps 300 --load-sequence "" --release-steps 0. The guide is
restricted to an initially open hand because it is authored before simulation.

The JSON reports measurements, without assigning numeric pass/fail criteria.
Check status == "complete" and each phase's contact counts before interpreting
its extrema. Completion alone does not establish that contact was sustained.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
from xml.etree import ElementTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", REPOSITORY_ROOT / "IsaacLab")).expanduser().resolve()
sys.path[:0] = [*(str(REPOSITORY_ROOT / "src" / package) for package in
                  ("contact_manipulation", "contact_manipulation_symmetric")),
               *(str(ISAACLAB_ROOT / "source" / package) for package in
                 ("isaaclab", "isaaclab_tasks", "isaaclab_rl"))]

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=("contact_manipulation", "contact_manipulation_symmetric"),
                        default="contact_manipulation")
    parser.add_argument("--finger", choices=("index", "middle", "ring", "little", "thumb"), default="index")
    parser.add_argument("--prismatic-guide", action="store_true")
    parser.add_argument("--load-sequence", default="1,3", help="Comma-separated loads in N; empty disables loading.")
    parser.add_argument("--settle-steps", type=int, default=200)
    parser.add_argument("--plateau-steps", type=int, default=200)
    parser.add_argument("--ramp-steps", type=int, default=10)
    parser.add_argument("--release-steps", type=int, default=100)
    parser.add_argument("--return-to-open-steps", type=int, default=0)
    parser.add_argument("--target-flexion", type=float, default=0., help="Master target fraction between configured feasible action bounds.")
    parser.add_argument("--radius", type=float, default=.012, help="Sphere radius in m.")
    parser.add_argument("--mass", type=float, default=.2, help="Sphere mass in kg.")
    parser.add_argument("--contact-threshold", type=float, default=.1, help="Force threshold in N used only to count contact frames.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/hand_contact.json"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    try:
        args.loads = [float(value) for value in args.load_sequence.split(",")] if args.load_sequence else []
    except ValueError:
        parser.error("--load-sequence must contain comma-separated numbers.")
    if any(not math.isfinite(value) or value < 0. for value in args.loads):
        parser.error("Loads must be finite and nonnegative.")
    if any(getattr(args, name) < 0 for name in
           ("settle_steps", "plateau_steps", "ramp_steps", "release_steps", "return_to_open_steps")):
        parser.error("Step counts must be nonnegative.")
    if not 0. <= args.target_flexion <= 1.:
        parser.error("--target-flexion must be in [0,1].")
    if args.prismatic_guide and args.target_flexion != 0.:
        parser.error("The pre-authored guide requires --target-flexion 0.")
    if any(not math.isfinite(value) or value <= 0. for value in (args.radius, args.mass)):
        parser.error("Sphere radius and mass must be finite and positive.")
    if not math.isfinite(args.contact_threshold) or args.contact_threshold < 0.:
        parser.error("--contact-threshold must be finite and nonnegative.")
    if args.settle_steps + args.return_to_open_steps + len(args.loads)*args.plateau_steps + args.release_steps == 0:
        parser.error("At least one simulation step is required.")
    return args


def run_probe(args):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    scene = importlib.import_module(f"{args.package}.config.ur5e.scene")
    env_module = importlib.import_module(f"{args.package}.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module(f"{args.package}.mdp.events")
    production = scene.make_ur5e_hand_cfg()
    urdf_path = production.spawn.hand_urdf_path
    scene.validate_hand_urdf_assets(urdf_path)
    source_joints = {joint.attrib["name"]: joint for joint in ElementTree.parse(urdf_path).findall("joint")
                     if joint.attrib["type"] == "revolute"}
    mimic_specs = {name: joint.find("mimic").attrib for name, joint in source_joints.items()
                   if joint.find("mimic") is not None}
    urdf_limits = {name: [float(joint.find("limit").attrib[key]) for key in ("lower", "upper")]
                   for name, joint in source_joints.items()}
    source_expected = {name: {"position_limits_rad": urdf_limits[name],
                              "velocity_limit_rad_s": float(joint.find("limit").attrib["velocity"]),
                              "effort_limit_Nm": float(joint.find("limit").attrib["effort"])}
                       for name, joint in source_joints.items()}
    master_names, pad_names, mimic_names = list(scene.HAND_JOINT_NAMES), list(scene.TACTILE_BODY_NAMES), list(mimic_specs)
    conversion_directory = tempfile.mkdtemp(prefix="contact_hand_probe_")

    def spawn_hand(prim_path, cfg, translation=None, orientation=None, **kwargs):
        original = cfg.copy()
        original.func = sim_utils.spawn_from_urdf
        prim = sim_utils.spawn_from_urdf(prim_path, original, translation, orientation, **kwargs)
        stage = sim_utils.get_current_stage()
        scene._deinstance_subtree(stage, prim_path)
        scene._configure_hand_joints(stage, prim_path, urdf_path)
        scene._configure_hand_collisions(stage, prim_path)
        return prim

    sim_cfg = env_module.UR5eContactManipulationEnvCfg().sim.copy()
    sim_cfg.device, sim_cfg.gravity, sim_cfg.render_interval = args.device, (0., 0., 0.), 2**31-1
    sim = sim_utils.SimulationContext(sim_cfg)
    hand = Articulation(ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UrdfFileCfg(
            func=spawn_hand, asset_path=urdf_path, fix_base=True, merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=True, make_instanceable=False,
            self_collision=False, collider_type="convex_hull", joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=.1),
            articulation_props=production.spawn.articulation_props.copy(), activate_contact_sensors=True,
            usd_dir=conversion_directory, force_usd_conversion=True),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.}, joint_vel={".*": 0.}),
        actuators={"hand": production.actuators["hand"].copy()}))
    sphere = RigidObject(RigidObjectCfg(
        prim_path="/World/LoadSphere",
        spawn=sim_utils.SphereCfg(
            radius=args.radius, activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=.1,
                                                        linear_damping=.05, angular_damping=.05),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.4, dynamic_friction=.3)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2., 2., 2.))))
    stage = sim_utils.get_current_stage()
    bodies = {prim.GetName(): prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Hand"))
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim)
              for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Hand")) if prim.IsA(UsdPhysics.RevoluteJoint)}
    pad_name = f"right_{args.finger}_force_sensor_{4 if args.finger == 'thumb' else 3}"
    child_name = f"right_{args.finger}_{4 if args.finger == 'thumb' else 2}_joint"
    loaded_joint = joints[child_name]
    cache = UsdGeom.XformCache()

    def quaternion_matrix(quaternion):
        return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]]).as_matrix()

    def authored_pose(body_name):
        matrix = cache.GetLocalToWorldTransform(bodies[body_name])
        return np.asarray(matrix.ExtractTranslation()), np.asarray(matrix)[:3, :3].T

    def live_pose(body_name):
        index = hand.body_names.index(body_name)
        return (hand.data.body_link_pos_w[0, index].cpu().numpy().copy(),
                quaternion_matrix(hand.data.body_link_quat_w[0, index].cpu().numpy()))

    def collision_points_local(body_name):
        body = bodies[body_name]
        inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for prim in Usd.PrimRange(body):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            ancestor = prim
            while ancestor and ancestor != body and not ancestor.HasAPI(UsdPhysics.CollisionAPI):
                ancestor = ancestor.GetParent()
            if ancestor == body or not ancestor:
                continue
            matrix = cache.GetLocalToWorldTransform(prim)*inverse
            points.extend(np.asarray(matrix.Transform(Gf.Vec3d(*point)))
                          for point in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        if not points:
            raise RuntimeError(f"No collision mesh points found for {body_name}.")
        return np.asarray(points)

    pad_points = collision_points_local(pad_name)

    def fixture_geometry(pose):
        parent_name = stage.GetPrimAtPath(loaded_joint.GetBody0Rel().GetTargets()[0]).GetName()
        parent_position, parent_rotation = pose(parent_name)
        quaternion = loaded_joint.GetLocalRot0Attr().Get()
        joint_rotation = quaternion_matrix([quaternion.GetReal(), *quaternion.GetImaginary()])
        local_axis = np.zeros(3)
        local_axis["XYZ".index(loaded_joint.GetAxisAttr().Get())] = 1.
        axis = parent_rotation@joint_rotation@local_axis
        origin = parent_position+parent_rotation@np.asarray(loaded_joint.GetLocalPos0Attr().Get())
        pad_position, pad_rotation = pose(pad_name)
        points = pad_points@pad_rotation.T+pad_position
        center = (points.min(0)+points.max(0))*.5
        tangent = np.cross(axis, center-origin)
        tangent /= np.linalg.norm(tangent)
        support = float(np.max((points-center)@tangent))
        sphere_center = center+tangent*(support+args.radius+.001)
        return sphere_center, -tangent, origin, axis, center

    authored_geometry = fixture_geometry(authored_pose)
    guide_metadata = None
    if args.prismatic_guide:
        guide_center, guide_direction, _, _, _ = authored_geometry
        sphere.cfg.init_state.pos = tuple(map(float, guide_center))
        sphere_prim = stage.GetPrimAtPath("/World/LoadSphere")
        translate_ops = [op for op in UsdGeom.Xformable(sphere_prim).GetOrderedXformOps()
                         if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if len(translate_ops) != 1:
            raise RuntimeError("Sphere must have exactly one authored translate XformOp.")
        vector_type = {UsdGeom.XformOp.PrecisionDouble: Gf.Vec3d,
                       UsdGeom.XformOp.PrecisionFloat: Gf.Vec3f,
                       UsdGeom.XformOp.PrecisionHalf: Gf.Vec3h}[translate_ops[0].GetPrecision()]
        translate_ops[0].Set(vector_type(*guide_center))
        placed_center = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(sphere_prim).ExtractTranslation())
        if not np.allclose(placed_center, guide_center, rtol=0., atol=1.e-7):
            raise RuntimeError(f"Prephysics sphere placement differs from guide center: {placed_center-guide_center}.")
        guide = UsdPhysics.PrismaticJoint.Define(stage, "/World/LoadGuide")
        guide.CreateBody1Rel().SetTargets([Sdf.Path("/World/LoadSphere")])
        guide.CreateAxisAttr("X")
        guide.CreateLocalPos0Attr(Gf.Vec3f(*guide_center))
        guide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        quaternion = Gf.Rotation(Gf.Vec3d(1., 0., 0.), Gf.Vec3d(*guide_direction)).GetQuat()
        rotation = Gf.Quatf(quaternion.GetReal(), Gf.Vec3f(*quaternion.GetImaginary()))
        guide.CreateLocalRot0Attr(rotation)
        guide.CreateLocalRot1Attr(rotation)
        guide.CreateLowerLimitAttr(-float("inf"))
        guide.CreateUpperLimitAttr(float("inf"))
        guide_metadata = {"joint_path": "/World/LoadGuide", "axis_world": guide_direction.tolist(),
                          "anchor_world_m": guide_center.tolist(), "free_dof": "translation_along_force_axis",
                          "prephysics_sphere_center_m": placed_center.tolist()}

    sensors, pad_validity = {}, {}
    for name in pad_names:
        body = bodies[name]
        colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(body)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
        reporter = body.HasAPI(PhysxSchema.PhysxContactReportAPI)
        if not colliders or not reporter:
            raise RuntimeError(f"Pad {name} has no enabled collider or contact reporter.")
        pad_validity[name] = {"body_path": str(body.GetPath()), "enabled_colliders": colliders, "contact_reporter": reporter}
        sensors[name] = ContactSensor(ContactSensorCfg(
            prim_path=str(body.GetPath()), update_period=0., filter_prim_paths_expr=["/World/LoadSphere"],
            max_contact_data_count_per_prim=32))
    sphere_sensor = ContactSensor(ContactSensorCfg(prim_path="/World/LoadSphere", update_period=0.))
    sim.reset()
    hand.reset()
    sphere.reset()
    dt, names = sim.get_physics_dt(), list(hand.joint_names)
    if len(names) != 12 or set(names) != set(source_joints):
        raise RuntimeError(f"Expected all 12 URDF hand joints, received {names}.")
    master_ids = [names.index(name) for name in master_names]
    native_limits = hand.root_physx_view.get_dof_limits()[0].cpu().numpy().copy()
    soft_limits_before = hand.data.soft_joint_pos_limits[0].cpu().numpy().copy()
    # The hand-only asset uses UrdfFileCfg.asset_path. Expose the production
    # assembly's path field to the unchanged startup helper, forwarding the
    # actual asset's data and joint lookup rather than duplicating its logic.
    action_robot = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(hand_urdf_path=urdf_path)),
                                   find_joints=hand.find_joints, data=hand.data)
    events.configure_hand_action_limits(SimpleNamespace(scene={"robot": action_robot}))
    soft_limits = hand.data.soft_joint_pos_limits[0].cpu().numpy().copy()
    native_limits_after_action_setup = hand.root_physx_view.get_dof_limits()[0].cpu().numpy().copy()
    bounds = np.asarray([urdf_limits[name] for name in names])
    targets = hand.data.default_joint_pos.clone()
    for index in master_ids:
        lower, upper = soft_limits[index]
        targets[:, index] = lower+args.target_flexion*(upper-lower)
    native_properties = {key: dict(zip(names, getattr(hand.root_physx_view, getter)()[0].cpu().tolist()))
                         for key, getter in (("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings"),
                                             ("effort_limit_Nm", "get_dof_max_forces"),
                                             ("velocity_limit_rad_s", "get_dof_max_velocities"))}
    joint_metadata = {name: {"schemas": joint.GetPrim().GetAppliedSchemas(),
                            "mimic_attributes": {attr.GetName(): attr.Get() for attr in joint.GetPrim().GetAuthoredAttributes()
                                                 if attr.GetName().startswith("physxMimicJoint:")}}
                      for name, joint in joints.items()}
    comparison_atol, comparison_rtol = 1.e-6, 1.e-6
    compliance, mimic_alignment = {}, {}
    for name in mimic_names:
        prim = joints[name].GetPrim()
        apis = PhysxSchema.PhysxMimicJointAPI.GetAll(prim)
        if len(apis) != 1:
            raise RuntimeError(f"Expected one native mimic on {prim.GetPath()}.")
        namespace = apis[0].GetGearingAttr().GetNamespace()
        values = {key: prim.GetAttribute(f"{namespace}:{key}").Get()
                  for key in ("naturalFrequency", "dampingRatio")}
        compliance[name] = {"namespace": namespace, **values,
                            "positive_finite": all(value is not None and math.isfinite(float(value)) and float(value) > 0.
                                                   for value in values.values())}
        api, source = apis[0], mimic_specs[name]
        reference_paths = api.GetReferenceJointRel().GetTargets()
        reference_names = [stage.GetPrimAtPath(path).GetName() if stage.GetPrimAtPath(path).IsValid() else None
                           for path in reference_paths]
        gearing, offset = api.GetGearingAttr().Get(), api.GetOffsetAttr().Get()
        source_multiplier, source_offset = float(source.get("multiplier", 1.)), float(source.get("offset", 0.))
        # Zero is invariant under angular unit/sign conversions. Do not infer
        # the native offset convention for a future nonzero source offset.
        zero_offset_comparable = source_offset == 0.
        mimic_alignment[name] = {
            "source_reference_joint": source["joint"], "native_reference_paths": list(map(str, reference_paths)),
            "native_reference_joint_names": reference_names,
            "reference_matches": reference_names == [source["joint"]],
            "source_multiplier": source_multiplier, "expected_native_gearing": -source_multiplier,
            "native_gearing": gearing,
            "ratio_matches": gearing is not None and math.isclose(float(gearing), -source_multiplier,
                                                                   rel_tol=comparison_rtol, abs_tol=comparison_atol),
            "source_offset_rad": source_offset, "native_offset_raw": offset,
            "offset_comparison": "zero_source_offset_only", "offset_comparison_supported": zero_offset_comparable,
            "offset_matches": (offset is not None and math.isclose(float(offset), 0., rel_tol=0., abs_tol=comparison_atol))
                              if zero_offset_comparable else None}
    alignment_rows = {}
    for index, name in enumerate(names):
        expected = source_expected[name]
        actual_velocity = native_properties["velocity_limit_rad_s"][name]
        actual_effort = native_properties["effort_limit_Nm"][name]
        alignment_rows[name] = {
            "position_limits_match": bool(np.allclose(native_limits[index], expected["position_limits_rad"],
                                                       rtol=comparison_rtol, atol=comparison_atol)),
            "velocity_limit_matches": math.isclose(actual_velocity, expected["velocity_limit_rad_s"],
                                                     rel_tol=comparison_rtol, abs_tol=comparison_atol),
            "effort_limit_matches": math.isclose(actual_effort, expected["effort_limit_Nm"],
                                                   rel_tol=comparison_rtol, abs_tol=comparison_atol)}
    feasible_positions, _ = scene.hand_joint_limits(urdf_path)
    dependent_ids = [index for index, name in enumerate(names) if name not in master_names]
    alignment_summary = {
        "comparison_absolute_tolerance": comparison_atol, "comparison_relative_tolerance": comparison_rtol,
        "joint_count": len(names), "position_limit_matches": sum(row["position_limits_match"] for row in alignment_rows.values()),
        "velocity_limit_matches": sum(row["velocity_limit_matches"] for row in alignment_rows.values()),
        "effort_limit_matches": sum(row["effort_limit_matches"] for row in alignment_rows.values()),
        "all_source_joint_fields_match": all(all(row.values()) for row in alignment_rows.values()),
        "mimic_count": len(compliance), "positive_compliance_count": sum(row["positive_finite"] for row in compliance.values()),
        "mimic_reference_matches": sum(row["reference_matches"] for row in mimic_alignment.values()),
        "mimic_ratio_matches": sum(row["ratio_matches"] for row in mimic_alignment.values()),
        "mimic_offset_comparable_count": sum(row["offset_comparison_supported"] for row in mimic_alignment.values()),
        "mimic_offset_matches": sum(row["offset_matches"] is True for row in mimic_alignment.values()),
        "all_source_mimic_fields_match": all(row["reference_matches"] and row["ratio_matches"] and row["offset_matches"] is True
                                             for row in mimic_alignment.values()),
        "action_helper": f"{args.package}.mdp.events.configure_hand_action_limits",
        "physical_limits_unchanged_by_action_setup": bool(np.array_equal(native_limits, native_limits_after_action_setup)),
        "dependent_soft_limits_unchanged": bool(np.array_equal(soft_limits_before[dependent_ids], soft_limits[dependent_ids])),
        "master_soft_limits_match_feasible_range": all(np.allclose(soft_limits[names.index(name)], feasible_positions[name],
                                                                   rtol=comparison_rtol, atol=comparison_atol)
                                                       for name in master_names),
        "per_joint": alignment_rows, "per_mimic": mimic_alignment}
    samples, phases = [], {}
    stop_reason = None
    qmin, qmax, qdotmax = np.full(12, np.inf), np.full(12, -np.inf), np.zeros(12)
    mimicmax, padmax = np.zeros(len(mimic_names)), np.zeros(len(pad_names))
    force_tensor = torch.zeros((1, 1, 3), device=sim.device)
    force_direction = np.zeros(3)

    def advance():
        hand.set_joint_position_target(targets[:, master_ids], joint_ids=master_ids)
        hand.write_data_to_sim()
        sphere.permanent_wrench_composer.set_forces_and_torques(forces=force_tensor, is_global=True)
        sphere.write_data_to_sim()
        sim.step(render=False)
        hand.update(dt)
        sphere.update(dt)
        for sensor in sensors.values():
            sensor.update(dt)
        sphere_sensor.update(dt)

    def capture(phase, step, load):
        nonlocal stop_reason
        q = hand.data.joint_pos[0].cpu().numpy().copy()
        qdot = hand.data.joint_vel[0].cpu().numpy().copy()
        errors = np.asarray([q[names.index(name)]-float(mimic_specs[name].get("multiplier", 1.))*q[names.index(mimic_specs[name]["joint"])]
                             -float(mimic_specs[name].get("offset", 0.)) for name in mimic_names])
        pair = np.asarray([float(torch.linalg.vector_norm(sensors[name].data.force_matrix_w.reshape(-1, 3), dim=-1).max())
                           for name in pad_names])
        contact = float(torch.linalg.vector_norm(sphere_sensor.data.net_forces_w.reshape(-1, 3), dim=-1).max())
        samples.append({"phase": phase, "step": step, "load_N": load, "q_rad": q.tolist(), "qdot_rad_s": qdot.tolist(),
                        "mimic_error_rad": errors.tolist(), "pad_pair_force_N": pair.tolist(),
                        "sphere_contact_force_N": contact, "sphere_position_m": sphere.data.root_pos_w[0].cpu().tolist()})
        if not all(np.isfinite(value).all() for value in (q, qdot, errors, pair, contact)):
            stop_reason = "nonfinite_physics_measurement"
            return
        np.minimum(qmin, q, out=qmin)
        np.maximum(qmax, q, out=qmax)
        np.maximum(qdotmax, np.abs(qdot), out=qdotmax)
        np.maximum(mimicmax, np.abs(errors), out=mimicmax)
        np.maximum(padmax, pair, out=padmax)
        summary = phases.setdefault(phase, {"steps": 0, "sphere_contact_frames": 0, "selected_pad_contact_frames": 0,
            "min_urdf_lower_error_rad": 0., "max_urdf_upper_error_rad": 0., "max_abs_qdot_rad_s": 0.,
            "max_mimic_error_rad": 0., "max_sphere_contact_force_N": 0., "max_selected_pad_pair_force_N": 0.,
            "master_targets_rad": dict(zip(master_names, targets[0, master_ids].cpu().tolist()))})
        summary["steps"] += 1
        summary["sphere_contact_frames"] += int(contact > args.contact_threshold)
        summary["selected_pad_contact_frames"] += int(pair[pad_names.index(pad_name)] > args.contact_threshold)
        summary["min_urdf_lower_error_rad"] = min(summary["min_urdf_lower_error_rad"], float(np.minimum(q-bounds[:, 0], 0.).min()))
        summary["max_urdf_upper_error_rad"] = max(summary["max_urdf_upper_error_rad"], float(np.maximum(q-bounds[:, 1], 0.).max()))
        summary["max_abs_qdot_rad_s"] = max(summary["max_abs_qdot_rad_s"], float(np.abs(qdot).max()))
        summary["max_mimic_error_rad"] = max(summary["max_mimic_error_rad"], float(np.abs(errors).max()))
        summary["max_sphere_contact_force_N"] = max(summary["max_sphere_contact_force_N"], contact)
        summary["max_selected_pad_pair_force_N"] = max(summary["max_selected_pad_pair_force_N"], float(pair[pad_names.index(pad_name)]))

    def run_phase(name, steps, peak_load=0., previous_load=0.):
        for step in range(steps):
            if stop_reason:
                break
            load = previous_load+(peak_load-previous_load)*min(1., (step+1)/max(args.ramp_steps, 1))
            force_tensor[0, 0] = torch.as_tensor(force_direction*load, device=sim.device)
            advance()
            capture(name, step, load)

    run_phase("settle", args.settle_steps)
    if args.return_to_open_steps and not stop_reason:
        targets[:, master_ids] = 0.
        run_phase("reopen", args.return_to_open_steps)
    if args.prismatic_guide and not stop_reason:
        offset = sphere.data.root_pos_w[0].cpu().numpy()-guide_center
        line_distance = float(np.linalg.norm(offset-np.dot(offset, guide_direction)*guide_direction))
        guide_metadata["post_settle_distance_from_guide_line_m"] = line_distance
        if not math.isfinite(line_distance) or line_distance >= 1.e-5:
            raise RuntimeError(f"Sphere is {line_distance} m off its native prismatic guide line after settling.")
    initial_center, force_direction, origin, axis, center = authored_geometry if stop_reason else fixture_geometry(live_pose)
    if args.prismatic_guide:
        guide_metadata["settled_anchor_difference_m"] = (initial_center-guide_center).tolist()
        guide_metadata["settled_axis_difference"] = float(np.linalg.norm(force_direction-guide_direction))
        initial_center, force_direction = guide_center, guide_direction
    fixture_placed = args.prismatic_guide
    if args.loads and not stop_reason:
        state = sphere.data.default_root_state.clone()
        state[:, :3] = torch.as_tensor(initial_center, device=sim.device)
        sphere.write_root_state_to_sim(state)
        sphere.reset()
        fixture_placed = True
    displacement_origin = initial_center if fixture_placed else sphere.data.root_pos_w[0].cpu().numpy().copy()
    previous_load = 0.
    for index, load in enumerate(args.loads):
        run_phase(f"load_{index+1}_{load:g}N", args.plateau_steps, load, previous_load)
        previous_load = load
    run_phase("release", args.release_steps)
    displacement = sphere.data.root_pos_w[0].cpu().numpy()-displacement_origin
    articulation = production.spawn.articulation_props
    requested_steps = args.settle_steps+args.return_to_open_steps+len(args.loads)*args.plateau_steps+args.release_steps
    return {"status": "stopped" if stop_reason else "complete", "stop_reason": stop_reason,
            "package": args.package, "finger": args.finger, "urdf_path": urdf_path, "conversion_directory": conversion_directory,
            "dt_s": dt, "solver_position_iterations": articulation.solver_position_iteration_count,
            "solver_velocity_iterations": articulation.solver_velocity_iteration_count,
            "contact_count_threshold_N": args.contact_threshold, "requested_load_sequence_N": args.loads,
            "protocol": {"settle_steps": args.settle_steps, "return_to_open_steps": args.return_to_open_steps,
                         "plateau_steps": args.plateau_steps, "ramp_steps": args.ramp_steps, "release_steps": args.release_steps,
                         "target_flexion_fraction": args.target_flexion,
                         "requested_steps": requested_steps, "recorded_steps": len(samples)},
            "fixture": {"pad": pad_name, "child_joint": child_name, "prismatic_guide": guide_metadata,
                        "placed_for_contact": fixture_placed,
                        "sphere_radius_m": args.radius, "sphere_mass_kg": args.mass,
                        "initial_sphere_center_m": initial_center.tolist(), "force_direction_world": force_direction.tolist(),
                        "joint_origin_world": origin.tolist(), "joint_axis_world": axis.tolist(), "pad_center_world": center.tolist(),
                        "unit_force_child_torque_Nm": float(np.dot(np.cross(center-origin, force_direction), axis)),
                        "sphere_displacement_m": displacement.tolist(),
                        "sphere_displacement_along_force_m": float(np.dot(displacement, force_direction))},
            "joint_names": names, "mimic_joint_names": mimic_names, "pad_names": pad_names,
            "urdf_limits_rad": urdf_limits, "native_limits_rad": dict(zip(names, native_limits.tolist())),
            "source_expected_joint_properties": source_expected,
            "soft_joint_pos_limits_rad": dict(zip(names, soft_limits.tolist())),
            "nominal_config_alignment": alignment_summary, "active_axis_mimic_compliance": compliance,
            "native_joint_properties": native_properties, "native_joint_metadata": joint_metadata, "pad_validity": pad_validity,
            "q_min_rad": dict(zip(names, qmin.tolist())), "q_max_rad": dict(zip(names, qmax.tolist())),
            "max_abs_qdot_rad_s": dict(zip(names, qdotmax.tolist())),
            "max_mimic_error_rad": dict(zip(mimic_names, mimicmax.tolist())),
            "pad_pair_force_max_N": dict(zip(pad_names, padmax.tolist())), "phase_summary": phases, "samples": samples}


def json_safe(value):
    """Keep stopped-run reports valid JSON when PhysX returns NaN or infinity."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main():
    args = parse_args()
    app = AppLauncher(args).app
    try:
        try:
            result = run_probe(args)
        except BaseException as error:
            traceback.print_exc()
            result = {"status": "error", "error_type": type(error).__name__, "error": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
        print("HAND_CONTACT_PROBE_RESULT", json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                                                       "phase_summary": result.get("phase_summary", {})}), flush=True)
    finally:
        # This standalone diagnostic uses no replicator work. The native immediate
        # close avoids the stage-cleanup hang observed with these mimic/contact views.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
