"""Task reward terms following the Sweeping-Policy ManagerBased examples.

Dense terms return an environment-sized raw score or positive cost. The native
RewardManager applies the configured weight and control dt. Only the one-time
terminal bonus is converted to a rate to preserve its fixed episode payment.
Shared physical snapshots avoid duplicate geometry, sensor reads and histories.
"""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_error_magnitude

__all__ = [
    "clearance_quality", "clearance", "shelf_collision", "object_collision",
    "object_motion_cost", "shelf_motion_cost",
    "tactile_pad_points", "blocker_distance_quality", "blocker_contact_hold", "blocker_distance",
    "motion_action_rate", "motion_joint_velocity", "motion_force_growth", "motion_torque_growth",
    "object_stability", "time_penalty",
    "singularity_margin", "singularity_penalty", "motion_singularity",
    "terminal",
]


def tactile_pad_points(env):
    """Return 17 current tactile contact-surface points in world coordinates.

    The first call inspects env_0's enabled collision meshes, not the link
    origins. All supplied hand pads expose their local +Z side; their parent
    bodies sit on the -Z side. Select a triangle near the area centroid of that
    exposed surface and retain its centroid as an actual surface point. Geometry,
    scale and fixed channel ordering are cached once. Later calls only transform
    those 17 offsets using native batched rigid-body poses on the scene device.
    """
    from .observations import quat_apply

    binding = getattr(env, "_tactile_reward_geometry", None)
    if binding is None:
        import numpy as np
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaaclab.sim.utils import get_current_stage
        from ..config.ur5e.scene import TACTILE_BODY_NAMES

        robot = env.scene["robot"]
        indices, names = robot.find_bodies(list(TACTILE_BODY_NAMES), preserve_order=True)
        if tuple(names) != TACTILE_BODY_NAMES:
            raise RuntimeError("Reward surface points require all 17 canonical tactile bodies.")
        stage = get_current_stage()
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        offsets = []
        for index in indices:
            body = stage.GetPrimAtPath(robot.root_physx_view.link_paths[0][index])
            transform = np.asarray(cache.GetLocalToWorldTransform(body), dtype=np.float64)
            # Strip only rotation/translation from the body transform: inherited
            # mesh/body scale must remain in the physical offset, in meters.
            left, _, right = np.linalg.svd(transform[:3, :3])
            body_rotation = left @ right
            meshes = {}
            for collider in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
                if (not collider.HasAPI(UsdPhysics.CollisionAPI)
                        or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False):
                    continue
                for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
                    if prim.IsA(UsdGeom.Mesh):
                        meshes[str(prim.GetPath())] = prim
            triangles = []
            for prim in meshes.values():
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
                matrix = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
                points = (points @ matrix[:3, :3] + matrix[3, :3] - transform[3, :3]) @ body_rotation.T
                vertex_ids = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
                start = 0
                for count in mesh.GetFaceVertexCountsAttr().Get():
                    face = vertex_ids[start:start + count]
                    for corner in range(1, count - 1):
                        triangles.append(points[face[[0, corner, corner + 1]]])
                    start += count
            if not triangles:
                raise RuntimeError(f"Tactile body has no enabled collision triangles: {body.GetPath()}")
            triangles = np.asarray(triangles)
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            double_area = np.linalg.norm(normals, axis=-1)
            front = normals[:, 2] > 0.5 * double_area
            if not np.any(front):
                raise RuntimeError(f"Tactile collision mesh has no exposed +Z surface: {body.GetPath()}")
            centers = triangles[front].mean(axis=1)
            surface_center = np.average(centers, weights=double_area[front], axis=0)
            offsets.append(centers[np.square(centers - surface_center).sum(-1).argmin()])
        position = robot.data.body_link_pos_w
        binding = (robot, torch.tensor(indices, device=env.device, dtype=torch.long),
                   torch.as_tensor(np.asarray(offsets), device=env.device, dtype=position.dtype))
        env._tactile_reward_geometry = binding

    robot, indices, offsets = binding
    position = robot.data.body_link_pos_w.index_select(1, indices)
    rotation = robot.data.body_link_quat_w.index_select(1, indices)
    return position + quat_apply(rotation, offsets)


def blocker_distance_quality(pad_points, blocker_center, blocker_rotation, blocker_half_extent,
                             *, decay=math.log(2.0) / 0.10):
    """Exp quality of the nearest pad point to the current solid Blocker OBB.

    Points are [N, 17, 3]; center, wxyz rotation and half extents are [N, 3/4/3]
    in the same frame. Points inside or on the OBB have zero distance. Nonfinite
    pads are ignored, while an invalid Blocker or no valid pad gives zero
    quality and infinite distance. No simulator access or device scalar reads.
    """
    from .observations import quat_apply, quat_conjugate

    if pad_points.ndim != 3 or pad_points.shape[1:] != (17, 3):
        raise ValueError("Expected 17 tactile points per environment, shaped [N, 17, 3].")
    batch = pad_points.shape[0]
    if (blocker_center.shape != (batch, 3) or blocker_rotation.shape != (batch, 4)
            or blocker_half_extent.shape != (batch, 3)):
        raise ValueError("Blocker center, wxyz rotation and half extents must have shapes [N, 3/4/3].")
    if not math.isfinite(decay) or decay <= 0:
        raise ValueError("Distance decay must be finite and positive.")
    relative = quat_apply(quat_conjugate(blocker_rotation)[:, None],
                          pad_points - blocker_center[:, None])
    outside = (relative.abs() - blocker_half_extent[:, None]).clamp_min(0.0)
    distance = torch.linalg.vector_norm(outside, dim=-1)
    valid_blocker = (torch.isfinite(blocker_center).all(-1) & torch.isfinite(blocker_rotation).all(-1)
                     & torch.isfinite(blocker_half_extent).all(-1) & (blocker_half_extent > 0).all(-1)
                     & (blocker_rotation.square().sum(-1) > 0))
    valid = torch.isfinite(pad_points).all(-1) & valid_blocker[:, None]
    distance = torch.where(valid, distance, math.inf).amin(-1)
    return torch.exp(-decay * distance), distance


def blocker_contact_hold(filtered_force, previous_contact, *, threshold=0.01, valid=None):
    """Count pads touching the Blocker in both policy samples; return new flags.

    ``filtered_force`` is the current Blocker-only normal force [N, 17, 3], never
    the any-object force used by observations. ``previous_contact`` is [N, 17]
    bool, cleared only for reset rows by the caller. Invalid readings become
    false so they also break continuity when the caller caches the new flags.
    This pure helper does not update history or imply contact between samples.
    """
    if (filtered_force.ndim != 3 or filtered_force.shape[1:] != (17, 3)
            or previous_contact.shape != filtered_force.shape[:2] or previous_contact.dtype != torch.bool):
        raise ValueError("Expected Blocker force [N, 17, 3] and previous boolean contacts [N, 17].")
    if valid is not None and (valid.shape != previous_contact.shape or valid.dtype != torch.bool):
        raise ValueError("Optional contact validity must have shape [N, 17] and boolean dtype.")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Contact threshold must be finite and nonnegative.")
    current = torch.isfinite(filtered_force).all(-1) & (
        torch.linalg.vector_norm(filtered_force, dim=-1) > threshold)
    if valid is not None:
        current = current & valid
    return (current & previous_contact).sum(-1).to(filtered_force.dtype), current


def clearance_quality(cfg, blocked, blocker_area):
    """Absolute Blocker overlap quality, shared by reset and completed steps."""
    ratio = (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    return torch.exp(-cfg.clearance_decay * ratio)


def clearance(env):
    """Initial-relative quality; invalid placement suppresses only positive progress."""
    from .events import get_task_state

    state = get_task_state(env)
    delta = state.clearance_quality - state.initial_clearance_quality
    return torch.where(state.clearance_progress_valid, delta, delta.clamp_max(0.0))


def blocker_distance(env, decay=math.log(2.0) / 0.10):
    """Nearest tactile surface distance; pay only while Target remains protected."""
    from .events import get_task_state
    from .observations import quat_apply, quat_conjugate, quat_mul

    state = get_task_state(env)
    scene = env.task_scene_data
    blocker = scene.blocker_id
    pose = env.scene["object_collection"].data.object_link_pose_w.gather(
        1, blocker[:, None, None].expand(-1, 1, 7)).squeeze(1)
    rotation = quat_mul(pose[:, 3:], quat_conjugate(scene.upright[blocker]))
    lower, upper = scene.lower[blocker], scene.upper[blocker]
    center = pose[:, :3] + quat_apply(rotation, (lower + upper) * .5)
    quality, _ = blocker_distance_quality(
        tactile_pad_points(env), center, rotation, (upper - lower) * .5, decay=decay)
    return torch.where(state.target_protected, quality, 0.0)


def object_motion_cost(center, corners, velocity, active, blocker):
    """Sum active non-Blocker mean OBB corner speeds, in meters per second.

    The linear velocity is measured at the OBB center, in the same fixed axes
    as the corners and angular velocity. Averaging the eight corners preserves
    the physical speed under pure translation; objects themselves are summed.
    No saturation is applied to valid costs. Nonfinite object measurements are
    unavailable costs; the existing physical-state termination handles them.
    """
    selected = active & ~blocker
    twist = torch.where(selected[..., None], velocity, 0.0)
    origin = torch.where(selected[..., None], center, 0.0)
    offset = torch.where(selected[..., None, None], corners, 0.0) - origin.unsqueeze(-2)
    corner_velocity = twist[..., None, :3] + torch.linalg.cross(twist[..., None, 3:], offset, dim=-1)
    speed = torch.linalg.vector_norm(corner_velocity, dim=-1).mean(-1)
    speed = torch.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(speed.sum(-1), nan=0.0, posinf=0.0, neginf=0.0)


def object_collision(env):
    """Current non-Blocker corner motion, including translation and rotation."""
    from .events import get_task_state

    state = get_task_state(env)
    return state.object_motion_cost


def shelf_motion_cost(pose, velocity, initial_pose, *, translation_threshold=0.005,
                      rotation_threshold=math.pi / 180.0, linear_velocity_threshold=0.005,
                      angular_velocity_threshold=0.05):
    """Example-style moved flag with separate, unit-consistent thresholds.

    Pose uses the shelf link frame; velocity is its COM twist. No old gripper
    link names, fixed shelf height or added contact sensor is required.
    """
    displacement = torch.linalg.vector_norm(pose[:, :3] - initial_pose[:, :3], dim=-1)
    rotation = quat_error_magnitude(pose[:, 3:], initial_pose[:, 3:])
    linear_speed = torch.linalg.vector_norm(velocity[:, :3], dim=-1)
    angular_speed = torch.linalg.vector_norm(velocity[:, 3:], dim=-1)
    finite = torch.isfinite(pose).all(-1) & torch.isfinite(velocity).all(-1)
    moved = ((displacement > translation_threshold) | (rotation > rotation_threshold)
             | (linear_speed > linear_velocity_threshold) | (angular_speed > angular_velocity_threshold))
    return (moved | ~finite).to(pose.dtype)


def shelf_collision(env, asset_cfg, *, translation_threshold=0.005,
                    rotation_threshold=math.pi / 180.0, linear_velocity_threshold=0.005,
                    angular_velocity_threshold=0.05):
    """Shelf pose/speed disturbance; the native manager assigns its negative weight."""
    from .events import get_task_state

    state = get_task_state(env)
    shelf = env.scene[asset_cfg.name]
    return shelf_motion_cost(
        shelf.data.root_link_pose_w, shelf.data.root_com_vel_w, state.initial_shelf_pose,
        translation_threshold=translation_threshold, rotation_threshold=rotation_threshold,
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
    )


def motion_force_growth(env):
    """Squared positive force-norm increase per control step, in N squared."""
    from .events import get_task_state

    return get_task_state(env).force_growth.square()


def motion_torque_growth(env):
    """Squared positive torque-norm increase per control step, in (N*m) squared."""
    from .events import get_task_state

    return get_task_state(env).torque_growth.square()


def motion_action_rate(env):
    """Native L2 cost; invalid-state rows must not send NaN/Inf to PPO."""
    from isaaclab.envs.mdp.rewards import action_rate_l2

    return torch.nan_to_num(action_rate_l2(env), nan=0.0, posinf=0.0, neginf=0.0)


def motion_joint_velocity(env, asset_cfg):
    """Native selected-joint L2 cost with the same invalid-state protection."""
    from isaaclab.envs.mdp.rewards import joint_vel_l2

    return torch.nan_to_num(joint_vel_l2(env, asset_cfg), nan=0.0, posinf=0.0, neginf=0.0)


def time_penalty(env):
    """Mean exponential cost rate over the completed control interval.

    RewardManager multiplies this by dt, producing exactly
    -integral[t-dt,t] a*(exp(b*s)-1) ds. The task clock advances once per
    completed transition and resets only for selected environments.
    """
    from .events import get_task_state

    state = get_task_state(env)
    scale, growth = state.cfg.time_penalty_scale, state.cfg.time_penalty_growth_rate
    window = growth * env.step_dt
    # expm1(window)/window - 1, without cancellation for tiny intervals.
    offset = (window * (.5 + window * (1 / 6 + window * (1 / 24 + window / 120)))
              if abs(window) < 1.0e-3 else math.expm1(window) / window - 1.0)
    start = (state.elapsed_time - env.step_dt).clamp_min(0.0)
    cost_rate = scale * (torch.expm1(growth * start) * (1.0 + offset) + offset)
    return torch.where(state.elapsed_time > 0.0, -cost_rate, 0.0).to(state.clearance_quality.dtype)


def object_stability(env):
    """Worst active object's tilt and OBB-XY/collision-bottom support risk.

    Distance has a 10 mm slack; tilt uses the angle itself, excluding yaw.
    Independent inverse-unit coefficients make their joint risk dimensionless.
    Existing center-based support-loss and tilt terminations remain separate.
    """
    from .events import get_task_state

    state = get_task_state(env)
    distance = torch.where(state.active, state.stability_distance, 0.0)
    tilt = torch.where(state.active, state.tilt_angle, 0.0)
    excess = (distance - state.cfg.support_warning_distance).clamp_min(0.0)
    risk = state.cfg.support_stability_decay * excess.square() + state.cfg.stability_tilt_decay * tilt
    risk = torch.nan_to_num(risk, nan=math.inf, posinf=math.inf, neginf=math.inf)
    return torch.expm1(-risk.amax(-1))


def singularity_margin(jacobian, *, characteristic_length=0.25):
    """Conservative weakest-direction margin for a batch of six-axis arm Jacobians.

    Scale linear rows by 1/L so linear/angular entries are dimensionless.
    L=0.25 m follows the existing IK position/orientation weighting convention.
    For invertible J_scaled, s=1/||inverse(J_scaled)||_F lies between
    sigma_min/sqrt(6) and sigma_min; it is NOT the exact minimum singular value.
    Native inv_ex(check_errors=False) keeps CUDA errors/results on the device,
    unlike svdvals' host error check. Failed/nonfinite rows conservatively get 0.
    This diagnostic never changes the OSC Jacobian, inverse, or effort targets.
    """
    if jacobian.ndim != 3 or jacobian.shape[-2:] != (6, 6):
        raise ValueError("Singularity requires six arm joints and a Jacobian shaped [N, 6, 6].")
    if not math.isfinite(characteristic_length) or characteristic_length <= 0:
        raise ValueError("Singularity characteristic length must be finite and positive.")
    scaled = torch.cat((jacobian[:, :3] / characteristic_length, jacobian[:, 3:]), dim=1)
    valid = torch.isfinite(scaled).all(dim=(-2, -1))
    identity = torch.eye(6, dtype=jacobian.dtype, device=jacobian.device)
    inverse, info = torch.linalg.inv_ex(torch.where(valid[:, None, None], scaled, identity), check_errors=False)
    norm = torch.linalg.vector_norm(inverse, dim=(-2, -1))
    valid = valid & (info == 0) & torch.isfinite(inverse).all(dim=(-2, -1)) & torch.isfinite(norm) & (norm > 0)
    return torch.where(valid, norm.reciprocal(), 0.0)


def singularity_penalty(margin, *, threshold=0.07, decay=math.log(2.0) / 0.03**2):
    """Zero above the warning margin; bounded exp-minus-one cost below it."""
    safe = torch.where(torch.isfinite(margin), margin.clamp_min(0.0), 0.0)
    cost = (threshold - safe).clamp_min(0.0).square()
    return torch.exp(-decay * cost) - 1.0


def motion_singularity(env):
    """Read the same post-physics arm margin used by the singularity termination."""
    from .events import get_task_state

    state = get_task_state(env)
    return singularity_penalty(
        state.singularity_margin, threshold=state.cfg.motion_singularity_threshold,
        decay=state.cfg.motion_singularity_decay,
    )


def terminal(env):
    """One-time success bonus converted to a native reward rate."""
    from .events import get_task_state

    return get_task_state(env).terminal_reward / env.step_dt
