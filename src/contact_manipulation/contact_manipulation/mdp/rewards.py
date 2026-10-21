"""Batched research rewards and thin Isaac Lab RewardManager adapters.

The numerical kernels use only PyTorch and do not read simulator state. Every
argument carries an environment leading dimension; there are no per-environment
Python loops or device-to-host reads. Native RewardTermCfg weights remain the
only reward weights. Clearance pays its exp(-a*x) quality relative to episode
start; rearrangement and motion use exp(-a*x) - 1 penalties. Cached components
retain the original qualities for task-state diagnostics. The adapters divide discrete payments by
dt to cancel the native RewardManager's dt factor.
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "corner_path_increment", "clearance_quality", "reward_components", "clearance", "rearrangement",
    "tactile_pad_points", "blocker_distance_quality", "blocker_contact_hold", "blocker_distance",
    "motion_action_rate", "motion_joint_velocity", "motion_wrench_growth", "object_stability",
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


def corner_path_increment(previous_corners, corners, active):
    """Sum selected objects' mean eight-corner displacement, in meters.

    Corners have fixed object-local correspondence and shape (N, O, 8, 3),
    expressed in the fixed environment frame. Translation, rotation, pivoting,
    and out-and-back movement all contribute. Averaging the eight corners makes
    a pure translation of one object contribute exactly its translation length;
    objects themselves are summed, including objects absent from initial vision.
    Inactive padded slots are removed before arithmetic, even if they hold NaN.
    """
    if corners.shape != previous_corners.shape or corners.shape[-2:] != (8, 3):
        raise ValueError("Current and previous corners must have matching (..., objects, 8, 3) shapes.")
    if active.shape != corners.shape[:-2]:
        raise ValueError("Active mask must contain one flag per object.")
    delta = torch.where(active[..., None, None], corners - previous_corners, 0.0)
    return torch.linalg.vector_norm(delta, dim=-1).mean(-1).sum(-1)


def clearance_quality(cfg, blocked, blocker_area):
    """Absolute Blocker overlap quality, shared by reset and completed steps."""
    ratio = (blocked / blocker_area.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    return torch.exp(-cfg.clearance_decay * ratio)


def reward_components(cfg, blocked, blocker_area, path_increment, wrench, previous_wrench, dt):
    """Return unweighted clearance, rearrangement and coupled F/T growth quality.

    Clearance evaluates only the designated Blocker; its manager adapter
    subtracts the episode's initial quality. The caller supplies non-Blocker
    objects' and the shelf's corner path for this transition, including rotation
    and return motion, to Rearrangement's exp(-a*x).
    Its quality returns to one when objects stop; the manager adapter subtracts
    one so stopped objects have zero penalty and past motion is not charged again.

    F/T uses one calibrated norm of positive load growth, followed by one
    exponential. The factor 1/2 and native weight .30 preserve the previous
    .15-per-channel small-cost slopes and total -.30 bound; the finite-cost
    curve intentionally differs from adding two separate rewards. Keep both
    excess-rate squared values for diagnostics. Constant load, release and direction-only changes have
    quality one, which maps to zero penalty in the manager adapters. Wrenches
    are self-load compensated at the current sensor center:
    rotations of their axes preserve norms, but translating the torque origin
    would change the measured load. Return the two excess-rate squared costs
    separately for diagnostics, in (N/s)^2 and (N*m/s)^2.

    Nonfinite numerical results are replaced by zero; termination separately marks invalid
    active physical state as failure so a failed simulation cannot poison PPO.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("Control dt must be finite and positive.")
    batch_shape = blocked.shape
    if blocker_area.shape != batch_shape or path_increment.shape != batch_shape:
        raise ValueError("Areas and path length must contain one value per environment.")
    if wrench.shape != (*batch_shape, 6) or previous_wrench.shape != wrench.shape:
        raise ValueError("Current and previous wrench must contain six components per environment.")

    clear = clearance_quality(cfg, blocked, blocker_area)
    rearr = torch.exp(-cfg.rearrangement_decay * path_increment.clamp_min(0.0))

    force_increase = (torch.linalg.vector_norm(wrench[..., :3], dim=-1)
                      - torch.linalg.vector_norm(previous_wrench[..., :3], dim=-1)).clamp_min(0.0)
    torque_increase = (torch.linalg.vector_norm(wrench[..., 3:], dim=-1)
                       - torch.linalg.vector_norm(previous_wrench[..., 3:], dim=-1)).clamp_min(0.0)
    force_cost = (force_increase / dt - cfg.force_rate_deadband).clamp_min(0.0).square()
    torque_cost = (torque_increase / dt - cfg.torque_rate_deadband).clamp_min(0.0).square()
    rate_cost = torch.stack((force_cost, torque_cost), -1)
    wrench_cost = .5 * (cfg.motion_force_rate_decay * force_cost + cfg.motion_torque_rate_decay * torque_cost)
    wrench_score = torch.exp(-wrench_cost)

    components = torch.stack((clear, rearr, wrench_score), -1)
    return (torch.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(rate_cost, nan=0.0, posinf=0.0, neginf=0.0))


def _reward_rate(env, index, *, baseline=0.0):
    # Lazy import keeps the numerical kernels independent of scene integration.
    from .events import get_task_state

    value = get_task_state(env).components[:, index]
    if baseline:
        value = value - baseline
    return value / env.step_dt


def clearance(env):
    """Initial-relative quality; invalid placement suppresses only positive progress."""
    from .events import get_task_state

    state = get_task_state(env)
    delta = state.components[:, 0] - state.initial_clearance_quality
    return torch.where(state.clearance_progress_valid, delta, delta.clamp_max(0.0)) / env.step_dt


def blocker_distance(env, decay=math.log(2.0) / 0.10):
    """Nearest tactile surface point to the current privileged Blocker OBB."""
    from .events import get_task_state
    from .observations import quat_apply, quat_conjugate, quat_mul

    get_task_state(env)
    scene = env.task_scene_data
    blocker = scene.blocker_id
    pose = env.scene["object_collection"].data.object_link_pose_w.gather(
        1, blocker[:, None, None].expand(-1, 1, 7)).squeeze(1)
    rotation = quat_mul(pose[:, 3:], quat_conjugate(scene.upright[blocker]))
    lower, upper = scene.lower[blocker], scene.upper[blocker]
    center = pose[:, :3] + quat_apply(rotation, (lower + upper) * .5)
    quality, _ = blocker_distance_quality(
        tactile_pad_points(env), center, rotation, (upper - lower) * .5, decay=decay)
    return quality / env.step_dt


def rearrangement(env):
    """This step's non-Blocker object and shelf corner-displacement penalty."""
    return _reward_rate(env, 1, baseline=1.0)


def motion_action_rate(env):
    """Exp-minus-one penalty of native raw action differences, without /dt in x."""
    from isaaclab.envs.mdp.rewards import action_rate_l2
    from .events import get_task_state

    cfg = get_task_state(env).cfg
    score = torch.exp(-cfg.motion_action_rate_decay * action_rate_l2(env))
    # Invalid quality becomes the maximum bounded penalty, never an idle bonus.
    return (torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) - 1.0) / env.step_dt


def motion_joint_velocity(env, arm_cfg, hand_cfg):
    """Exp-minus-one penalty of native joint-velocity sums at the final substep."""
    from isaaclab.envs.mdp.rewards import joint_vel_l2
    from .events import get_task_state

    cfg = get_task_state(env).cfg
    velocity_cost = joint_vel_l2(env, arm_cfg) + joint_vel_l2(env, hand_cfg)
    score = torch.exp(-cfg.motion_joint_velocity_decay * velocity_cost)
    return (torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0) - 1.0) / env.step_dt


def motion_wrench_growth(env):
    """One calibrated positive F/T growth norm; unloading does not cancel loading."""
    return _reward_rate(env, 2, baseline=1.0)


def object_stability(env):
    """Worst active object's distance to the finite support surface, beyond slack.

    This is a state-risk cost, separate from non-Blocker path disturbance.
    It uses the same geometric distance as object_support_lost, not a sum of
    the previous lift/drop/ejection penalties. No per-object loop or state query.
    """
    from .events import get_task_state

    state = get_task_state(env)
    distance = torch.where(state.active, state.support_distance, 0.0)
    distance = torch.nan_to_num(distance, nan=math.inf, posinf=math.inf, neginf=math.inf)
    excess = (distance - state.cfg.support_warning_distance).clamp_min(0.0)
    worst = excess.square().amax(-1)
    return torch.expm1(-state.cfg.support_stability_decay * worst) / env.step_dt


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
    ) / env.step_dt


def terminal(env):
    """One-time success bonus converted to a native reward rate."""
    return _reward_rate(env, 3)
