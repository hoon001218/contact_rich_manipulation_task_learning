"""Measure RH56E2 motion under a controlled native object contact load.

Run through Isaac Lab, for example::

    ./IsaacLab/isaaclab.sh -p src/contact_manipulation/tests/probe_hand_contact.py \
        --headless --device cuda:0 --prismatic-guide --output /tmp/hand_contact.json

The selected package supplies the URDF, joint/collision setup, hand actuator,
simulation timestep and articulation solver settings. A fixed hand base and
zero gravity isolate the hand from arm control, rewards and the learned policy.
A dynamic sphere loads a distal pad in the negative flexion direction. The
optional native prismatic guide prevents lateral escape. No joint states are
overwritten during the test. By default no physics parameters are tuned here;
--hand-stiffness, --hand-damping and --hand-effort-limit override only this
diagnostic's copied hand actuator settings.
--hand-armature, --hand-velocity-limit and --hand-self-collision isolate native
inertia/speed/contact settings without changing the production configuration.
Reports include actual backend drive types, armatures and position targets;
reading gains alone does not prove that the intended drive is active.
Master targets use the same startup action-limit helper as training; the
physical limits on all twelve joints remain independent of that action range.
Packages with mapped joint drives expand those master commands with their
production URDF mapping; native-mimic packages retain master-only commands.

For a contact-free close/open cycle, use --target-flexion 1
--return-to-open-steps 300 --load-sequence "" --release-steps 0. The guide is
restricted to an initially open hand because it is authored before simulation.

For pressing a fixed obstacle, use --fixed-obstacle --target-flexion 1
--ramp-steps 200 --plateau-steps 400 --return-to-open-steps 200. Only the selected
finger closes; a kinematic box is placed before physics starts and never moved.
Add --video-output /tmp/hand_contact.mp4 to record overview and contact close-up
views at 25 fps. Recording adds only cameras, lights, visual materials and an
image overlay; physics settings and the sequence of physics steps are unchanged.

For pushing a supported dynamic box, replace --fixed-obstacle with --push-object
and set --mass .3. The native default friction material is used on the box and
table. The hand base remains fixed, with its initial orientation chosen so the
selected finger's closing tangent is horizontal. Only the box feels gravity.
Add --translate-hand --target-flexion 0 to hold all finger targets at zero and
move the whole hand with a native external prismatic drive instead of flexion.
Use --hand-stiffness 20, 50 or 100 in separate runs for a controlled gain
comparison. Damping, effort limits and the production configuration stay unchanged.
For an effort comparison, use --hand-stiffness 1.2 --hand-damping .06 and
--hand-effort-limit 1 or 10. The original URDF effort remains recorded separately;
source-limit alignment is deliberately false when the requested effort differs.

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
    parser.add_argument("--fixed-obstacle", action="store_true",
                        help="Close the selected finger against a stationary kinematic box; no external load.")
    parser.add_argument("--push-object", action="store_true",
                        help="Push a gravity-enabled dynamic box resting on an immovable table, using default friction.")
    parser.add_argument("--translate-hand", action="store_true",
                        help="Move the whole hand on a native prismatic guide; keep all finger targets at zero.")
    parser.add_argument("--translation-distance", type=float, default=.1, help="Whole-hand forward stroke in m.")
    parser.add_argument("--hand-stiffness", type=float,
                        help="Diagnostic-only Hand Kp override for all driven joints; inherit production if omitted.")
    parser.add_argument("--hand-damping", type=float,
                        help="Diagnostic-only Hand Kd override; inherit production if omitted.")
    parser.add_argument("--hand-effort-limit", type=float,
                        help="Diagnostic-only Hand effort_limit_sim override in N m; inherit URDF if omitted.")
    parser.add_argument("--hand-armature", type=float,
                        help="Diagnostic-only joint armature override; inherit production if omitted.")
    parser.add_argument("--hand-velocity-limit", type=float,
                        help="Diagnostic-only native joint speed limit in rad/s; inherit URDF if omitted.")
    parser.add_argument("--hand-self-collision", choices=("inherit", "on", "off"), default="inherit",
                        help="Diagnostic-only articulation self-collision override.")
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
    parser.add_argument("--video-output", type=Path, help="Optional MP4 with overview, close-up and measured angles.")
    parser.add_argument("--video-fps", type=int, default=25)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive.")
    if args.video_output:
        args.enable_cameras = True
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
    if args.fixed_obstacle and args.push_object:
        parser.error("Choose either --fixed-obstacle or --push-object.")
    if args.translate_hand and (not args.push_object or args.target_flexion != 0.):
        parser.error("--translate-hand requires --push-object and --target-flexion 0.")
    if not math.isfinite(args.translation_distance) or args.translation_distance <= 0.:
        parser.error("--translation-distance must be finite and positive.")
    if args.hand_stiffness is not None and (not math.isfinite(args.hand_stiffness) or args.hand_stiffness <= 0.):
        parser.error("--hand-stiffness must be finite and positive.")
    if args.hand_damping is not None and (not math.isfinite(args.hand_damping) or args.hand_damping < 0.):
        parser.error("--hand-damping must be finite and nonnegative.")
    if args.hand_effort_limit is not None and (not math.isfinite(args.hand_effort_limit) or args.hand_effort_limit <= 0.):
        parser.error("--hand-effort-limit must be finite and positive.")
    if args.hand_armature is not None and (not math.isfinite(args.hand_armature) or args.hand_armature < 0.):
        parser.error("--hand-armature must be finite and nonnegative.")
    if args.hand_velocity_limit is not None and (not math.isfinite(args.hand_velocity_limit) or args.hand_velocity_limit <= 0.):
        parser.error("--hand-velocity-limit must be finite and positive.")
    if args.fixed_obstacle or args.push_object:
        if args.prismatic_guide or (args.target_flexion <= 0. and not args.translate_hand):
            parser.error("Box contact tests require a positive --target-flexion and no --prismatic-guide.")
        args.loads = []
    if any(not math.isfinite(value) or value <= 0. for value in (args.radius, args.mass)):
        parser.error("Sphere radius and mass must be finite and positive.")
    if not math.isfinite(args.contact_threshold) or args.contact_threshold < 0.:
        parser.error("--contact-threshold must be finite and nonnegative.")
    if (args.settle_steps + args.return_to_open_steps
            + (1 if args.fixed_obstacle or args.push_object else len(args.loads))*args.plateau_steps + args.release_steps == 0):
        parser.error("At least one simulation step is required.")
    return args


def annotate_contact_forces(image, sample):
    """Append measured normal resultants; these are not individual point forces.

    The box value is the norm of its net normal-force vector. Each link value
    is the norm of its filtered link/box normal resultant. Magnitudes from
    different links must not be added to reconstruct the box's resultant.
    """
    from PIL import Image, ImageDraw, ImageFont

    width, height = image.size
    annotated = Image.new("RGB", (width, height+160), (20, 25, 34))
    annotated.paste(image, (0, 0))
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    force = float(sample.get("hand_box_normal_force_N", sample["sphere_contact_force_N"]))
    label = "Hand / box normal resultant" if "hand_box_normal_force_N" in sample else "Box normal contact resultant"
    draw.text((20, height+8), f"{label}: {force:.3f} N", font=font, fill=(255, 215, 110))
    draw.text((width-220, height+12), "Bar scale: 0 - 12 N", font=small, fill=(190, 210, 225))
    draw.rectangle((20, height+42, width-20, height+54), fill=(55, 64, 78))
    if force > 0.:
        draw.rectangle((20, height+42, 20+(width-40)*min(force/12., 1.), height+54), fill=(245, 170, 65))
    contacts = sorted(((name.removeprefix("right_"), float(value))
                       for name, value in sample.get("hand_body_contact_force_N", {}).items() if value > .001),
                      key=lambda item: -item[1])
    if not contacts:
        draw.text((20, height+66), "Link / box contacts: none above 0.001 N", font=small, fill=(190, 210, 225))
    for index, (name, value) in enumerate(contacts[:6]):
        draw.text((20+(width//3)*(index % 3), height+66+28*(index//3)),
                  f"{name}: {value:.3f} N", font=small, fill=(130, 225, 235))
    draw.text((20, height+126), "Normal forces only; tangential friction is excluded. Values are resultants per link, not per contact point.",
              font=small, fill=(175, 185, 200))
    return annotated


def run_probe(args):
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    push_object = getattr(args, "push_object", False)
    translate_hand = getattr(args, "translate_hand", False)
    scene = importlib.import_module(f"{args.package}.config.ur5e.scene")
    env_module = importlib.import_module(f"{args.package}.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module(f"{args.package}.mdp.events")
    mapped_drives = hasattr(env_module, "UrdfCoupledHandAction")
    production = scene.make_ur5e_hand_cfg()
    hand_actuator = production.actuators["hand"].copy()
    if getattr(args, "hand_stiffness", None) is not None:
        hand_actuator.stiffness = args.hand_stiffness
    if getattr(args, "hand_damping", None) is not None:
        hand_actuator.damping = args.hand_damping
    if getattr(args, "hand_effort_limit", None) is not None:
        hand_actuator.effort_limit_sim = args.hand_effort_limit
    if getattr(args, "hand_armature", None) is not None:
        hand_actuator.armature = args.hand_armature
    if getattr(args, "hand_velocity_limit", None) is not None:
        hand_actuator.velocity_limit_sim = args.hand_velocity_limit
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
    sim_cfg.device, sim_cfg.gravity, sim_cfg.render_interval = args.device, (0., 0., -9.81 if push_object else 0.), 2**31-1
    if push_object:
        sim_cfg.physics_material = sim_utils.RigidBodyMaterialCfg()
    sim = sim_utils.SimulationContext(sim_cfg)
    articulation_props = production.spawn.articulation_props.copy()
    if getattr(args, "hand_self_collision", "inherit") != "inherit":
        articulation_props.enabled_self_collisions = args.hand_self_collision == "on"
    if translate_hand:
        articulation_props.fix_root_link = False
    hand = Articulation(ArticulationCfg(
        prim_path="/World/Hand",
        spawn=sim_utils.UrdfFileCfg(
            func=spawn_hand, asset_path=urdf_path, fix_base=not translate_hand, merge_fixed_joints=False,
            convert_mimic_joints_to_normal_joints=not mapped_drives, make_instanceable=False,
            self_collision=False, collider_type="convex_hull", joint_drive=None,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=.1),
            articulation_props=articulation_props, activate_contact_sensors=True,
            usd_dir=conversion_directory, force_usd_conversion=True),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={".*": 0.}, joint_vel={".*": 0.}),
        actuators={"hand": hand_actuator}))
    stage = sim_utils.get_current_stage()
    bodies = {prim.GetName(): prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Hand"))
              if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = {prim.GetName(): UsdPhysics.RevoluteJoint(prim)
              for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Hand")) if prim.IsA(UsdPhysics.RevoluteJoint)}
    pad_name = f"right_{args.finger}_force_sensor_{4 if args.finger == 'thumb' else 3}"
    child_name = f"right_{args.finger}_{4 if args.finger == 'thumb' else 2}_joint"
    selected_motor = f"right_{args.finger}_{2 if args.finger == 'thumb' else 1}_joint"
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
        if args.fixed_obstacle or getattr(args, "push_object", False):
            # The master and all its descendants close together. Use the full
            # chain's pad velocity, not just the distal joint's tangent.
            tangent = np.zeros(3)
            for name, source in source_joints.items():
                multiplier, current = 1., source
                while (mimic := current.find("mimic")) is not None:
                    multiplier *= float(mimic.attrib.get("multiplier", 1.))
                    current = source_joints[mimic.attrib["joint"]]
                if current.attrib["name"] != selected_motor:
                    continue
                joint = joints[name]
                parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0]).GetName()
                position, rotation = pose(parent)
                q = joint.GetLocalRot0Attr().Get()
                joint_frame = rotation@quaternion_matrix([q.GetReal(), *q.GetImaginary()])
                joint_axis = joint_frame[:, "XYZ".index(joint.GetAxisAttr().Get())]
                joint_origin = position+rotation@np.asarray(joint.GetLocalPos0Attr().Get())
                tangent += multiplier*np.cross(joint_axis, center-joint_origin)
        if np.linalg.norm(tangent) < 1.e-9:
            raise RuntimeError("Selected finger has no closing direction at the fixture pad.")
        tangent /= np.linalg.norm(tangent)
        if args.fixed_obstacle or getattr(args, "push_object", False):
            # Parent link colliders can protrude beyond a tactile mesh. Clear
            # the whole selected finger initially, rather than embedding its
            # housing in a box placed using only the pad's support surface.
            points = np.concatenate([
                collision_points_local(name)@pose(name)[1].T+pose(name)[0]
                for name in bodies if name.startswith(f"right_{args.finger}_")])
        support = float(np.max((points-center)@tangent))
        sphere_center = center+tangent*(support+args.radius+.001)
        return sphere_center, -tangent, origin, axis, center

    authored_geometry = fixture_geometry(authored_pose)
    support_metadata = None
    if push_object:
        # Author one fixed base pose before physics starts. Align the selected
        # closing tangent with +X and the palm-to-tip direction with gravity.
        points = np.concatenate([collision_points_local(name)@authored_pose(name)[1].T+authored_pose(name)[0]
                                 for name in bodies])
        normal = -authored_geometry[1]
        up = points.mean(0)-authored_geometry[4]
        up -= np.dot(up, normal)*normal
        up /= np.linalg.norm(up)
        lateral = np.cross(up, normal)
        rotation = np.stack((normal, lateral, up))
        quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Hand"))
        ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient]
        orient = ops[0] if ops else xform.AddOrientOp()
        quat_type, vec_type = ((Gf.Quatd, Gf.Vec3d) if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
                               else (Gf.Quatf, Gf.Vec3f))
        orient.Set(quat_type(float(quaternion[0]), vec_type(*map(float, quaternion[1:]))))
        hand.cfg.init_state.rot = tuple(map(float, quaternion))
        cache.Clear()
        authored_geometry = fixture_geometry(authored_pose)
    translation_drive, translation_target = None, 0.
    if translate_hand:
        urdf = ElementTree.parse(urdf_path)
        child_links = {joint.find("child").attrib["link"] for joint in urdf.findall("joint")}
        root_names = [link.attrib["name"] for link in urdf.findall("link") if link.attrib["name"] not in child_links]
        if len(root_names) != 1 or root_names[0] not in bodies:
            raise RuntimeError(f"Expected one physical hand root, got {root_names}.")
        root_name = root_names[0]
        root_position, root_rotation = authored_pose(root_name)
        # An excluded, maximal-coordinate joint constrains a FLOATING hand.
        # A reduced-coordinate root joint to world would instead fix its base.
        guide = UsdPhysics.PrismaticJoint.Define(stage, "/World/HandTranslation")
        guide.CreateExcludeFromArticulationAttr(True)
        guide.CreateBody1Rel().SetTargets([bodies[root_name].GetPath()])
        guide.CreateAxisAttr("X")
        guide.CreateLocalPos0Attr(Gf.Vec3f(*root_position))
        guide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
        inverse = Rotation.from_matrix(root_rotation.T).as_quat()
        guide.CreateLocalRot0Attr(Gf.Quatf(1., Gf.Vec3f(0., 0., 0.)))
        guide.CreateLocalRot1Attr(Gf.Quatf(float(inverse[3]), Gf.Vec3f(*map(float, inverse[:3]))))
        guide.CreateLowerLimitAttr(-.01)
        guide.CreateUpperLimitAttr(args.translation_distance+.01)
        translation_drive = UsdPhysics.DriveAPI.Apply(guide.GetPrim(), "linear")
        translation_drive.CreateTypeAttr("force")
        translation_drive.CreateStiffnessAttr(20000.)
        translation_drive.CreateDampingAttr(500.)
        translation_drive.CreateMaxForceAttr(200.)
        translation_drive.CreateTargetPositionAttr(0.)
        translation_drive.CreateTargetVelocityAttr(0.)
    fixture_orientation = (1., 0., 0., 0.)
    fixed_box_size = (2.*args.radius, 2.*args.radius, 8.*args.radius)
    if args.fixed_obstacle:
        normal, axis = -authored_geometry[1], authored_geometry[3]
        lateral = axis-np.dot(axis, normal)*normal
        lateral /= np.linalg.norm(lateral)
        quaternion = Rotation.from_matrix(np.column_stack((normal, lateral, np.cross(normal, lateral)))).as_quat()
        fixture_orientation = tuple(map(float, quaternion[[3, 0, 1, 2]]))
    if push_object:
        fixed_box_size = (.080, .024, .060)
        points = np.concatenate([collision_points_local(name)@authored_pose(name)[1].T+authored_pose(name)[0]
                                 for name in bodies])
        floor_z = min(authored_geometry[4][2]-fixed_box_size[2]/2., points[:, 2].min()-.01)
        box_center = authored_geometry[0].copy()
        box_center[0] += fixed_box_size[0]/2.-args.radius
        box_center[2] = floor_z+fixed_box_size[2]/2.+.0005
        authored_geometry = (box_center, *authored_geometry[1:])
        support_metadata = {"top_z_m": float(floor_z), "size_m": [.6, .6, .02],
                            "body_type": "kinematic (never moved)",
                            "initial_hand_quaternion_wxyz": hand.cfg.init_state.rot,
                            "min_initial_hand_clearance_m": float(points[:, 2].min()-floor_z)}
        support = sim_utils.CuboidCfg(
            size=tuple(support_metadata["size_m"]),
            # GPU contact filtering supports rigid actors, but not a collision-
            # only static collider. A never-moved kinematic actor is immovable.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.28, .32, .38), roughness=.8))
        support.func("/World/Support", support, translation=(float(box_center[0]+.12), float(box_center[1]), float(floor_z-.01)))
    obstacle_spawn = dict(
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=args.fixed_obstacle, disable_gravity=not push_object, max_depenetration_velocity=.1,
            linear_damping=.05, angular_damping=.05),
        mass_props=sim_utils.MassPropertiesCfg(mass=args.mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
        physics_material=(sim_utils.RigidBodyMaterialCfg() if push_object else
                          sim_utils.RigidBodyMaterialCfg(static_friction=.4, dynamic_friction=.3)))
    sphere = RigidObject(RigidObjectCfg(
        prim_path="/World/LoadSphere",  # Preserve the existing contact-filter path.
        spawn=(sim_utils.CuboidCfg(size=fixed_box_size, **obstacle_spawn) if args.fixed_obstacle or push_object else
               sim_utils.SphereCfg(radius=args.radius, **obstacle_spawn)),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(map(float, authored_geometry[0])) if args.fixed_obstacle or push_object else (2., 2., 2.),
            rot=fixture_orientation)))
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
    hand_contact = (ContactSensor(ContactSensorCfg(
        prim_path="/World/Hand/.*", update_period=0.,
        filter_prim_paths_expr=["/World/LoadSphere"]+(["/World/Support"] if push_object else []),
        max_contact_data_count_per_prim=64)) if args.fixed_obstacle or push_object else None)
    video_camera, video_frames = None, 0
    if args.video_output:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw, ImageFont
        from isaaclab.sensors import Camera, CameraCfg

        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.10, .62, .85), roughness=.6)
        material.func("/World/ProbeMaterials/SelectedFinger", material)
        for name, body in bodies.items():
            if name.startswith(f"right_{args.finger}_"):
                sim_utils.bind_visual_material(body.GetPath(), "/World/ProbeMaterials/SelectedFinger")
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .49, .13), roughness=.7)
        material.func("/World/ProbeMaterials/Obstacle", material)
        sim_utils.bind_visual_material("/World/LoadSphere", "/World/ProbeMaterials/Obstacle")
        light = sim_utils.DomeLightCfg(intensity=1800., color=(.90, .94, 1.))
        light.func("/World/ProbeLight", light)
        for index in range(2):
            sim_utils.create_prim(f"/World/ProbeViews/View_{index}", "Xform")
        video_camera = Camera(CameraCfg(
            prim_path="/World/ProbeViews/View_.*/Camera", width=640, height=640, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24., horizontal_aperture=24., clipping_range=(.001, 10.))))
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    sim.reset()
    hand.reset()
    sphere.reset()
    dt, names = sim.get_physics_dt(), list(hand.joint_names)
    if translate_hand:
        if hand.is_fixed_base:
            raise RuntimeError("Whole-hand translation requires a floating articulation with an external guide.")
        root_id = hand.body_names.index(root_name)
        translation_origin = hand.data.body_link_pos_w[0, root_id].cpu().numpy().copy()
        translation_orientation = quaternion_matrix(hand.data.body_link_quat_w[0, root_id].cpu().numpy())
    if video_camera is not None:
        video_stride = round(1./(dt*args.video_fps))
        if video_stride < 1 or not math.isclose(video_stride*dt*args.video_fps, 1., abs_tol=1.e-6):
            raise ValueError("--video-fps must exactly divide the physics frequency.")
        center, axis, normal = authored_geometry[4], authored_geometry[3], -authored_geometry[1]
        close_target = .5*(center+authored_geometry[2])
        # A more frontal view separates the inner fingers from their neighbors.
        close_offset = (.12*axis-.22*normal if args.finger in ("middle", "ring", "little")
                        else .25*axis-.14*normal)
        eyes = np.asarray([[-.20, -.34, -.04], close_target+close_offset])
        view_targets = np.asarray([[.01, -.035, -.15], close_target])
        if push_object:
            center = (points.min(0)+points.max(0))*.5
            close_target = .5*(authored_geometry[4]+authored_geometry[2])+np.array([.03, 0., 0.])
            eyes = np.asarray([center+[-.22, -.38, .15], close_target+[-.20, -.22, .12]])
            view_targets = np.asarray([center+[.02, 0., -.02], close_target])
        video_camera.set_world_poses_from_view(
            torch.as_tensor(eyes, device=sim.device, dtype=torch.float32),
            torch.as_tensor(view_targets, device=sim.device, dtype=torch.float32))
        for _ in range(8):
            sim.render()
        args.video_output.parent.mkdir(parents=True, exist_ok=True)
        args._video_writer = imageio.get_writer(
            str(args.video_output), fps=args.video_fps, codec="libx264", quality=8,
            ffmpeg_params=["-movflags", "+faststart"])
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
        targets[:, index] = lower+(0. if args.fixed_obstacle or push_object else args.target_flexion)*(upper-lower)
    driven_ids = master_ids
    if mapped_drives:
        mapped_names, mapped_indices, multipliers, offsets = scene.hand_joint_mapping(urdf_path)
        if set(mapped_names) != set(names):
            raise RuntimeError("Production hand mapping must cover all twelve imported joints.")
        driven_ids = [names.index(name) for name in mapped_names]
        mapped_indices = torch.tensor(mapped_indices, dtype=torch.long, device=sim.device)
        mapped_multipliers = targets.new_tensor(multipliers)
        mapped_offsets = targets.new_tensor(offsets)
        driven_targets = targets.new_empty((1, len(driven_ids)))
    native_properties = {key: dict(zip(names, getattr(hand.root_physx_view, getter)()[0].cpu().tolist()))
                         for key, getter in (("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings"),
                                             ("effort_limit_Nm", "get_dof_max_forces"),
                                             ("velocity_limit_rad_s", "get_dof_max_velocities"),
                                             ("drive_type", "get_drive_types"),
                                             ("armature", "get_dof_armatures"))}
    if push_object:
        actual_mass = float(sphere.root_physx_view.get_masses()[0].item())
        if not math.isclose(actual_mass, args.mass, rel_tol=1.e-6):
            raise RuntimeError(f"Dynamic box mass differs from requested mass: {actual_mass}.")
        support_metadata.update({"box_mass_kg": actual_mass,
            "static_friction": obstacle_spawn["physics_material"].static_friction,
            "dynamic_friction": obstacle_spawn["physics_material"].dynamic_friction,
            "restitution": obstacle_spawn["physics_material"].restitution,
            "friction_combine_mode": obstacle_spawn["physics_material"].friction_combine_mode})
    joint_metadata = {name: {"schemas": joint.GetPrim().GetAppliedSchemas(),
                            "joint_enabled": joint.GetJointEnabledAttr().Get(),
                            "authored_drive_type": UsdPhysics.DriveAPI.Get(joint.GetPrim(), "angular").GetTypeAttr().Get(),
                            "mimic_attributes": {attr.GetName(): attr.Get() for attr in joint.GetPrim().GetAuthoredAttributes()
                                                 if attr.GetName().startswith("physxMimicJoint:")}}
                      for name, joint in joints.items()}
    comparison_atol, comparison_rtol = 1.e-6, 1.e-6
    compliance, mimic_alignment = {}, {}
    if mapped_drives:
        for name in names:
            if PhysxSchema.PhysxMimicJointAPI.GetAll(joints[name].GetPrim()):
                raise RuntimeError(f"Mapped-drive hand must not have a native mimic on {name}.")
    for name in (() if mapped_drives else mimic_names):
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
        "all_source_mimic_fields_match": (None if mapped_drives else
            all(row["reference_matches"] and row["ratio_matches"] and row["offset_matches"] is True
                for row in mimic_alignment.values())),
        "action_helper": f"{args.package}.mdp.events.configure_hand_action_limits",
        "physical_limits_unchanged_by_action_setup": bool(np.array_equal(native_limits, native_limits_after_action_setup)),
        "dependent_soft_limits_unchanged": bool(np.array_equal(soft_limits_before[dependent_ids], soft_limits[dependent_ids])),
        "master_soft_limits_match_feasible_range": all(np.allclose(soft_limits[names.index(name)], feasible_positions[name],
                                                                   rtol=comparison_rtol, atol=comparison_atol)
                                                       for name in master_names),
        "per_joint": alignment_rows, "per_mimic": mimic_alignment}
    if mapped_drives:
        alignment_summary.update({"expected_native_mimic_count": 0, "native_mimic_absence_verified": True,
                                  "source_mimic_comparison_applicable": False})
    samples, phases = [], {}
    startup = {"diagnostic_overrides": {key: getattr(args, key, None) for key in
               ("hand_stiffness", "hand_damping", "hand_effort_limit", "hand_armature", "hand_self_collision", "hand_velocity_limit")},
               "joint_names": names, "native_joint_properties": native_properties,
               "native_joint_metadata": joint_metadata,
               "enabled_self_collisions": articulation_props.enabled_self_collisions,
               "source_expected_joint_properties": source_expected,
               "nominal_config_alignment": alignment_summary, "dt_s": dt,
               "hand_is_fixed_base": bool(hand.is_fixed_base),
               "initial_joint_pos_rad": hand.data.joint_pos[0].cpu().tolist(),
               "initial_joint_vel_rad_s": hand.data.joint_vel[0].cpu().tolist(),
               "initial_joint_targets_rad": targets[0].cpu().tolist(),
               "support": support_metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".startup.json").write_text(json.dumps(json_safe(startup), indent=2, allow_nan=False))
    args._probe_progress = {"startup": startup, "samples": samples, "phase_summary": phases}
    stop_reason = None
    qmin, qmax, qdotmax = np.full(12, np.inf), np.full(12, -np.inf), np.zeros(12)
    mimicmax, padmax = np.zeros(len(mimic_names)), np.zeros(len(pad_names))
    force_tensor = torch.zeros((1, 1, 3), device=sim.device)
    force_direction = np.zeros(3)

    def advance():
        if getattr(args, "translate_hand", False):
            translation_drive.GetTargetPositionAttr().Set(float(translation_target))
        if mapped_drives:
            # Match production command expansion, including reopening. Do not
            # follow measured master angles displaced by the contact load.
            torch.index_select(targets[:, master_ids], 1, mapped_indices, out=driven_targets)
            driven_targets.mul_(mapped_multipliers).add_(mapped_offsets)
            targets[:, driven_ids] = driven_targets
        hand.set_joint_position_target(targets[:, driven_ids], joint_ids=driven_ids)
        hand.write_data_to_sim()
        if not (args.fixed_obstacle or getattr(args, "push_object", False)):
            sphere.permanent_wrench_composer.set_forces_and_torques(forces=force_tensor, is_global=True)
            sphere.write_data_to_sim()
        sim.step(render=False)
        hand.update(dt)
        sphere.update(dt)
        for sensor in sensors.values():
            sensor.update(dt)
        sphere_sensor.update(dt)
        if hand_contact is not None:
            hand_contact.update(dt)

    def capture(phase, step, load):
        nonlocal stop_reason, video_frames
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
        if mapped_drives:
            commanded = targets[0].cpu().numpy()
            samples[-1].update({"joint_targets_rad": commanded.tolist(), "joint_tracking_error_rad": (q-commanded).tolist(),
                "native_joint_targets_rad": hand.root_physx_view.get_dof_position_targets()[0].cpu().tolist(),
                "commanded_urdf_relation_error_rad": [float(commanded[names.index(name)]
                    -float(mimic_specs[name].get("multiplier", 1.))*commanded[names.index(mimic_specs[name]["joint"])]
                    -float(mimic_specs[name].get("offset", 0.))) for name in mimic_names]})
        if translate_hand:
            position = hand.data.body_link_pos_w[0, root_id].cpu().numpy()
            rotation = quaternion_matrix(hand.data.body_link_quat_w[0, root_id].cpu().numpy())
            base_offset = position-translation_origin
            samples[-1].update({"base_target_x_m": float(translation_target), "base_offset_m": base_offset.tolist(),
                "base_orientation_error_rad": float(Rotation.from_matrix(rotation@translation_orientation.T).magnitude()),
                "max_finger_target_error_rad": float(np.max(np.abs(q-targets[0].cpu().numpy())))})
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
        if args.fixed_obstacle or push_object:
            body_vectors = hand_contact.data.force_matrix_w[0, :, 0]
            body_forces = torch.linalg.vector_norm(body_vectors, dim=-1).cpu().tolist()
            samples[-1]["hand_body_contact_force_N"] = dict(zip(hand_contact.body_names, body_forces))
            summary["body_contact_force_max_N"] = {
                name: max(summary.get("body_contact_force_max_N", {}).get(name, 0.), value)
                for name, value in zip(hand_contact.body_names, body_forces)}
            obstacle_error = float(np.linalg.norm(
                np.asarray(samples[-1]["sphere_position_m"])-authored_geometry[0]))
            summary["max_obstacle_position_error_m"] = max(summary.get("max_obstacle_position_error_m", 0.), obstacle_error)
            summary["final_master_targets_rad"] = dict(zip(master_names, targets[0, master_ids].cpu().tolist()))
            if args.fixed_obstacle and obstacle_error > 1.e-5:
                stop_reason = "fixed_obstacle_moved"
            if push_object:
                hand_force = float(torch.linalg.vector_norm(body_vectors.sum(dim=0)))
                table_force = float(torch.linalg.vector_norm(hand_contact.data.force_matrix_w[0, :, 1].sum(dim=0)))
                displacement = sphere.data.root_pos_w[0].cpu().numpy()-authored_geometry[0]
                quaternion = sphere.data.root_quat_w[0].cpu().numpy()
                tilt = math.acos(float(np.clip(quaternion_matrix(quaternion)[2, 2], -1., 1.)))
                samples[-1].update({"hand_box_normal_force_N": hand_force, "hand_table_normal_force_N": table_force,
                    "box_displacement_m": displacement.tolist(), "box_xy_displacement_m": float(np.linalg.norm(displacement[:2])),
                    "box_tilt_rad": tilt, "box_linear_velocity_m_s": sphere.data.root_lin_vel_w[0].cpu().tolist()})
                summary["hand_box_contact_frames"] = summary.get("hand_box_contact_frames", 0)+int(hand_force > args.contact_threshold)
                summary["hand_table_contact_frames"] = summary.get("hand_table_contact_frames", 0)+int(table_force > args.contact_threshold)
                summary["max_hand_box_normal_force_N"] = max(summary.get("max_hand_box_normal_force_N", 0.), hand_force)
                summary["max_box_xy_displacement_m"] = max(summary.get("max_box_xy_displacement_m", 0.), samples[-1]["box_xy_displacement_m"])
                summary["max_box_tilt_rad"] = max(summary.get("max_box_tilt_rad", 0.), tilt)
        if video_camera is not None and len(samples) % video_stride == 0:
            # Native render() does not advance physics. Keep all annotations in
            # the encoded image, without adding collision or simulation prims.
            sim.render()
            video_camera.update(dt*video_stride, force_recompute=True)
            rgb = video_camera.data.output["rgb"][..., :3].cpu().numpy()
            image = Image.new("RGB", (1280, 800), (20, 25, 34))
            for index in range(2):
                image.paste(Image.fromarray(rgb[index]), (640*index, 80))
            draw = ImageDraw.Draw(image)
            elapsed = len(samples)*dt
            phase_text = ("CLOSE" if phase == "press" and step < args.ramp_steps else
                          "HOLD" if phase == "press" else phase.upper())
            if translate_hand:
                phase_text = ("PUSH" if phase == "press" and step < args.ramp_steps else
                              "HOLD" if phase == "press" else "RETRACT" if phase == "reopen" else phase.upper())
            draw.text((20, 12), f"{args.finger.upper()} | {phase_text} | t = {elapsed:.2f} s | real time", font=font, fill="white")
            legend = (f"Dynamic box: {args.mass*1000:.0f} g | default friction 0.5 / 0.5" if push_object else
                      "Overview: cyan = tested finger; orange = fixed obstacle")
            draw.text((20, 48), legend, font=small_font, fill=(190, 210, 225))
            gain_label = (f"Kp {native_properties['stiffness'][selected_motor]:g} | "
                          f"Kd {native_properties['damping'][selected_motor]:g} | "
                          f"limit {native_properties['effort_limit_Nm'][selected_motor]:g} Nm | "
                          f"armature {native_properties['armature'][selected_motor]:g} kg m2")
            draw.text((660, 48), gain_label, font=small_font, fill=(190, 210, 225))
            selected = [i for i, name in enumerate(names) if name.startswith(f"right_{args.finger}_")]
            angles = "  ".join(f"q{name.split('_')[-2]}={math.degrees(q[i]):+.1f} deg" for i, name in
                                ((i, names[i]) for i in selected))
            violation = max(0., float((bounds[selected, 0]-q[selected]).max()))
            draw.text((20, 730), f"Measured: {angles}", font=small_font, fill="white")
            visible_force = samples[-1].get("hand_box_normal_force_N", contact)
            draw.text((20, 762), f"Below URDF lower limit: {math.degrees(violation):.1f} deg | contact: {visible_force > args.contact_threshold}",
                      font=small_font, fill=(255, 105, 100) if violation > math.radians(1.) else (150, 225, 175))
            if push_object:
                draw.rectangle((740, 754, 1279, 799), fill=(20, 25, 34))
                draw.text((750, 762), f"Box travel: {samples[-1]['box_xy_displacement_m']*1000:.1f} mm | tilt: {math.degrees(tilt):.1f} deg",
                          font=small_font, fill=(170, 225, 185))
            image = annotate_contact_forces(image, samples[-1])
            if translate_hand:
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 921, 1279, 959), fill=(20, 25, 34))
                draw.text((20, 930), f"Finger targets: all 0 rad | base target / actual: {translation_target*1000:.1f} / {base_offset[0]*1000:.1f} mm | max finger error: {math.degrees(samples[-1]['max_finger_target_error_rad']):.1f} deg",
                          font=small_font, fill=(185, 210, 245))
            args._video_writer.append_data(np.asarray(image))
            video_frames += 1

    def run_phase(name, steps, peak_load=0., previous_load=0.):
        for step in range(steps):
            if stop_reason:
                break
            load = previous_load+(peak_load-previous_load)*min(1., (step+1)/max(args.ramp_steps, 1))
            force_tensor[0, 0] = torch.as_tensor(force_direction*load, device=sim.device)
            args._probe_progress.update(attempted_phase=name, attempted_phase_step=step,
                                        attempted_physics_step=len(samples))
            advance()
            capture(name, step, load)

    run_phase("settle", args.settle_steps)
    if args.fixed_obstacle or push_object:
        selected_id = names.index(selected_motor)
        lower, upper = soft_limits[selected_id]
        peak_angle = lower+args.target_flexion*(upper-lower)
        for phase, steps, goal in (
            ("press", args.plateau_steps, peak_angle),
            ("reopen", args.return_to_open_steps, lower),
        ):
            initial = targets[:, selected_id].clone()
            for step in range(steps):
                if stop_reason:
                    break
                if getattr(args, "translate_hand", False):
                    u = min(1., (step+1)/max(args.ramp_steps, 1))
                    progress = u*u*(3.-2.*u)
                    translation_target = args.translation_distance*(progress if phase == "press" else 1.-progress)
                else:
                    targets[:, selected_id] = initial+(goal-initial)*min(1., (step+1)/max(args.ramp_steps, 1))
                if hasattr(args, "_probe_progress"):
                    args._probe_progress.update(attempted_phase=phase, attempted_phase_step=step,
                                                attempted_physics_step=len(samples))
                advance()
                capture(phase, step, 0.)
    elif args.return_to_open_steps and not stop_reason:
        targets[:, master_ids] = 0.
        run_phase("reopen", args.return_to_open_steps)
    if args.prismatic_guide and not stop_reason:
        offset = sphere.data.root_pos_w[0].cpu().numpy()-guide_center
        line_distance = float(np.linalg.norm(offset-np.dot(offset, guide_direction)*guide_direction))
        guide_metadata["post_settle_distance_from_guide_line_m"] = line_distance
        if not math.isfinite(line_distance) or line_distance >= 1.e-5:
            raise RuntimeError(f"Sphere is {line_distance} m off its native prismatic guide line after settling.")
    initial_center, force_direction, origin, axis, center = (
        authored_geometry if stop_reason or args.fixed_obstacle or push_object else fixture_geometry(live_pose))
    if args.prismatic_guide:
        guide_metadata["settled_anchor_difference_m"] = (initial_center-guide_center).tolist()
        guide_metadata["settled_axis_difference"] = float(np.linalg.norm(force_direction-guide_direction))
        initial_center, force_direction = guide_center, guide_direction
    fixture_placed = args.prismatic_guide or args.fixed_obstacle or push_object
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
    requested_steps = (args.settle_steps+args.return_to_open_steps+args.release_steps
                       + (1 if args.fixed_obstacle or push_object else len(args.loads))*args.plateau_steps)
    return {"status": "stopped" if stop_reason else "complete", "stop_reason": stop_reason,
            "package": args.package, "finger": args.finger, "urdf_path": urdf_path, "conversion_directory": conversion_directory,
            "diagnostic_overrides": {key: getattr(args, key, None) for key in
                                     ("hand_stiffness", "hand_damping", "hand_effort_limit", "hand_armature", "hand_self_collision", "hand_velocity_limit")},
            "enabled_self_collisions": articulation_props.enabled_self_collisions,
            "hand_control": {"mode": "urdf_mapped_joint_drives" if mapped_drives else "native_mimic",
                             "parse_mimic": not mapped_drives, "expected_native_mimic_count": 0 if mapped_drives else len(mimic_names),
                             "driven_joint_names": [names[index] for index in driven_ids],
                             "mimic_error_definition": ("Measured URDF angle-relation residual; no native mimic constraint."
                                                        if mapped_drives else "Measured native mimic angle-relation residual.")},
            "dt_s": dt, "solver_position_iterations": articulation.solver_position_iteration_count,
            "video": ({"path": str(args.video_output.resolve()), "fps": args.video_fps,
                       "frames": video_frames, "playback_speed": 1., "physics_steps_per_frame": video_stride}
                      if video_camera is not None else None),
            "solver_velocity_iterations": articulation.solver_velocity_iteration_count,
            "contact_count_threshold_N": args.contact_threshold, "requested_load_sequence_N": args.loads,
            "protocol": {"settle_steps": args.settle_steps, "return_to_open_steps": args.return_to_open_steps,
                         "translate_hand": translate_hand,
                         "translation_distance_m": args.translation_distance if translate_hand else None,
                         "plateau_steps": args.plateau_steps, "ramp_steps": args.ramp_steps, "release_steps": args.release_steps,
                         "target_flexion_fraction": args.target_flexion,
                         "requested_steps": requested_steps, "recorded_steps": len(samples)},
            "fixture": {"pad": pad_name, "child_joint": child_name, "prismatic_guide": guide_metadata,
                        "push_object": push_object, "support": support_metadata,
                        "fixed_obstacle": args.fixed_obstacle, "selected_motor": selected_motor,
                        "shape": "box" if args.fixed_obstacle or push_object else "sphere",
                        "box_size_m": fixed_box_size if args.fixed_obstacle or push_object else None,
                        "initial_gap_m": .001, "orientation_wxyz": fixture_orientation,
                        "initial_gap_reference": "selected finger collision meshes" if args.fixed_obstacle or push_object else "selected pad mesh",
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
            result = {"status": "error", "error_type": type(error).__name__, "error": str(error),
                      "partial": getattr(args, "_probe_progress", None)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False))
        print("HAND_CONTACT_PROBE_RESULT", json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                                                       "phase_summary": result.get("phase_summary", {})}), flush=True)
    finally:
        if (writer := getattr(args, "_video_writer", None)) is not None:
            writer.close()
        # Frames have been copied and the MP4 writer has flushed synchronously.
        # Immediate close avoids the cleanup hang seen with these contact views.
        app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
