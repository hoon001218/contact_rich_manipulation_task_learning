"""Reset randomization for selecting and placing shelf objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_task_state_buffers(env: ManagerBasedRLEnv) -> None:
    """Allocate reset-owned task state before other managers inspect its shape.

    Isaac Lab constructs the Observation Manager before it performs the first
    environment reset. Consequently, observation terms may be evaluated before
    object randomization has populated these buffers. The placeholder values
    exist only during manager construction and are overwritten by the first
    reset event.
    """
    if not hasattr(env, "target_id"):
        # Object zero is a valid placeholder for observation shape inference.
        env.target_id = torch.zeros(
            (env.num_envs, 1), dtype=torch.long, device=env.device
        )
    if not hasattr(env, "target_width"):
        env.target_width = torch.zeros(
            (env.num_envs, 1), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "sweep_dir"):
        env.sweep_dir = torch.zeros(
            (env.num_envs, 3), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "target_work_pos_w"):
        env.target_work_pos_w = torch.zeros(
            (env.num_envs, 3), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "target_goal_pos_w"):
        env.target_goal_pos_w = torch.zeros(
            (env.num_envs, 3), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "desired_reaching_pose_w"):
        env.desired_reaching_pose_w = torch.zeros(
            (env.num_envs, 7), dtype=torch.float32, device=env.device
        )
    if not hasattr(env, "reaching_ik_success"):
        env.reaching_ik_success = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
    if not hasattr(env, "reaching_ik_attempt_count"):
        env.reaching_ik_attempt_count = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )


def randomize_working_and_waiting_objects(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    working_x_range: tuple[float, float],
    working_y_range: tuple[float, float],
    working_height: float,
    waiting_height: float,
    waiting_xy: tuple[tuple[float, float], ...],
    object_widths: tuple[float, ...],
    push_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> None:
    """Place one random object on the work shelf and all others on the top shelf.

    The target is sampled independently for each vectorized environment. Its x/y
    coordinates are sampled continuously, so no work-shelf grid is involved.
    The fixed x/y slots are used only to keep the five waiting objects separated.
    """
    objects: RigidObjectCollection = env.scene[asset_cfg.name]
    if len(waiting_xy) != objects.num_objects:
        raise ValueError(
            f"waiting_xy must contain one slot per object: "
            f"{len(waiting_xy)} != {objects.num_objects}"
        )
    if len(object_widths) != objects.num_objects:
        raise ValueError(
            f"object_widths must contain one width per object: "
            f"{len(object_widths)} != {objects.num_objects}"
        )
    if push_distance > working_y_range[1] - working_y_range[0]:
        raise ValueError("push_distance is larger than the working y range.")

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    _ensure_task_state_buffers(env)
    target_ids = torch.randint(
        low=0,
        high=objects.num_objects,
        size=(num_resets,),
        device=env.device,
    )

    # Store the selected object index for future target observations/rewards.
    env.target_id[env_ids, 0] = target_ids
    width_tensor = torch.tensor(object_widths, device=env.device)
    env.target_width[env_ids, 0] = width_tensor[target_ids]

    states = torch.zeros(
        (num_resets, objects.num_objects, 13),
        dtype=torch.float32,
        device=env.device,
    )
    waiting_xy_tensor = torch.tensor(
        waiting_xy,
        dtype=torch.float32,
        device=env.device,
    )
    states[..., :2] = waiting_xy_tensor.unsqueeze(0)
    states[..., 2] = waiting_height
    states[..., :3] += env.scene.env_origins[env_ids].unsqueeze(1)
    states[..., 3:7] = math_utils.random_yaw_orientation(
        num=num_resets * objects.num_objects,
        device=env.device,
    ).reshape(num_resets, objects.num_objects, 4)

    reset_rows = torch.arange(num_resets, device=env.device)
    random_xy = torch.rand((num_resets, 2), device=env.device)
    random_xy[:, 0] = (
        working_x_range[0]
        + random_xy[:, 0] * (working_x_range[1] - working_x_range[0])
    )
    random_xy[:, 1] = (
        working_y_range[0]
        + random_xy[:, 1] * (working_y_range[1] - working_y_range[0])
    )
    states[reset_rows, target_ids, :2] = (
        random_xy + env.scene.env_origins[env_ids, :2]
    )
    states[reset_rows, target_ids, 2] = (
        working_height + env.scene.env_origins[env_ids, 2]
    )

    # A positive direction means pushing toward +y. The EEF will be placed on
    # the opposite (-y) side of the object. Use only directions that leave the
    # full requested push distance inside the working spawn range.
    can_push_positive = random_xy[:, 1] + push_distance <= working_y_range[1]
    can_push_negative = random_xy[:, 1] - push_distance >= working_y_range[0]
    both_directions = can_push_positive & can_push_negative
    random_direction = torch.where(
        torch.rand(num_resets, device=env.device) < 0.5,
        -torch.ones(num_resets, device=env.device),
        torch.ones(num_resets, device=env.device),
    )
    direction = torch.where(
        both_directions,
        random_direction,
        torch.where(
            can_push_positive,
            torch.ones_like(random_direction),
            -torch.ones_like(random_direction),
        ),
    )
    if not torch.all(can_push_positive | can_push_negative):
        raise RuntimeError("Sampled an object position with no feasible push direction.")

    env.sweep_dir[env_ids] = 0.0
    env.sweep_dir[env_ids, 1] = direction * push_distance
    env.target_work_pos_w[env_ids] = states[reset_rows, target_ids, :3]
    env.target_goal_pos_w[env_ids] = (
        env.target_work_pos_w[env_ids] + env.sweep_dir[env_ids]
    )

    objects.write_object_link_state_to_sim(states, env_ids=env_ids)


class initialize_robot_at_reaching_pose(ManagerTermBase):
    """Solve a reset-time IK problem for the pre-contact reaching pose."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Event Manager construction precedes Observation Manager construction,
        # while the first reset occurs later. Allocate observation-facing task
        # state now so its dimensions can be inferred safely.
        _ensure_task_state_buffers(env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["robot_cfg"].name]
        self.shelf: RigidObject = env.scene[params["shelf_cfg"].name]
        self.arm_joint_ids, _ = self.robot.find_joints(
            params["arm_joint_names"], preserve_order=True
        )
        body_ids, body_names = self.robot.find_bodies(params["eef_body_name"])
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one EEF body named {params['eef_body_name']!r}; "
                f"found {body_names}."
            )
        self.eef_body_id = int(body_ids[0])
        self.jacobian_body_id = (
            self.eef_body_id - 1 if self.robot.is_fixed_base else self.eef_body_id
        )
        self.jacobian_joint_ids = (
            self.arm_joint_ids
            if self.robot.is_fixed_base
            else [joint_id + 6 for joint_id in self.arm_joint_ids]
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        shelf_cfg: SceneEntityCfg,
        arm_joint_names: tuple[str, ...],
        eef_body_name: str,
        eef_offset: tuple[float, float, float],
        reaching_x_offset: float,
        reaching_z_offset: float,
        position_noise: float,
        max_iterations: int,
        damping: float,
        step_size: float,
        position_tolerance: float,
        orientation_tolerance: float,
        joint_seed_offsets: tuple[tuple[float, ...], ...],
    ) -> None:
        del robot_cfg, shelf_cfg, arm_joint_names, eef_body_name
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
        if len(env_ids) == 0:
            return
        if not all(
            hasattr(env, name)
            for name in ("target_work_pos_w", "target_width", "sweep_dir")
        ):
            raise RuntimeError(
                "Object randomization must run before reaching-pose IK initialization."
            )

        direction = torch.sign(env.sweep_dir[env_ids, 1])
        desired_eef_pos = env.target_work_pos_w[env_ids].clone()
        desired_eef_pos[:, 0] += reaching_x_offset
        desired_eef_pos[:, 1] -= env.target_width[env_ids, 0] * direction
        desired_eef_pos[:, 2] += reaching_z_offset
        desired_eef_pos += math_utils.sample_uniform(
            -position_noise,
            position_noise,
            (len(env_ids), 3),
            device=env.device,
        )

        # Construct a fixed orientation that maximizes the configured alignment
        # reward. align_ee_target compares row 1 of the EEF rotation matrix with
        # row 2 of the shelf rotation matrix. This fixes one axis but leaves a
        # free twist. Preserve the nominal EEF x-axis as much as possible so the
        # reset target does not introduce an unnecessary large wrist rotation.
        # No orientation noise is applied.
        shelf_rotation = math_utils.matrix_from_quat(
            self.shelf.data.default_root_state[env_ids, 3:7]
        )
        nominal_eef_rotation = math_utils.matrix_from_quat(
            self.robot.data.body_quat_w[env_ids, self.eef_body_id]
        )
        desired_y_axis = shelf_rotation[:, 2]
        desired_x_axis = nominal_eef_rotation[:, 0]
        desired_x_axis = desired_x_axis - torch.sum(
            desired_x_axis * desired_y_axis, dim=-1, keepdim=True
        ) * desired_y_axis
        x_axis_norm = torch.linalg.norm(desired_x_axis, dim=-1, keepdim=True)
        desired_x_axis = torch.where(
            x_axis_norm > 1.0e-6,
            desired_x_axis / torch.clamp_min(x_axis_norm, 1.0e-6),
            shelf_rotation[:, 0],
        )
        desired_z_axis = torch.linalg.cross(
            desired_x_axis, desired_y_axis, dim=-1
        )
        desired_z_axis = torch.nn.functional.normalize(desired_z_axis, dim=-1)
        desired_x_axis = torch.linalg.cross(
            desired_y_axis, desired_z_axis, dim=-1
        )
        desired_eef_rotation = torch.stack(
            (desired_x_axis, desired_y_axis, desired_z_axis),
            dim=1,
        )
        desired_eef_quat = math_utils.quat_unique(
            math_utils.quat_from_matrix(desired_eef_rotation)
        )
        offset = torch.tensor(
            eef_offset, dtype=torch.float32, device=env.device
        ).repeat(len(env_ids), 1)
        desired_body_pos = desired_eef_pos - math_utils.quat_apply(
            desired_eef_quat, offset
        )
        env.desired_reaching_pose_w[env_ids, :3] = desired_eef_pos
        env.desired_reaching_pose_w[env_ids, 3:7] = desired_eef_quat

        if len(joint_seed_offsets) == 0:
            raise ValueError("joint_seed_offsets must contain at least one seed.")
        seed_offsets = torch.tensor(
            joint_seed_offsets, dtype=torch.float32, device=env.device
        )
        if seed_offsets.ndim != 2 or seed_offsets.shape[1] != len(self.arm_joint_ids):
            raise ValueError(
                "Each joint seed offset must contain one value per arm joint: "
                f"expected {len(self.arm_joint_ids)}, got shape {tuple(seed_offsets.shape)}."
            )

        # The sampled object pose, push direction, target pose, and position noise
        # remain fixed across all attempts. Only the arm joint seed changes, which
        # avoids filtering difficult object positions out of the training set.
        success = torch.zeros(len(env_ids), dtype=torch.bool, device=env.device)
        attempt_count = torch.zeros(len(env_ids), dtype=torch.long, device=env.device)
        final_pos_error = torch.full(
            (len(env_ids),), torch.inf, dtype=torch.float32, device=env.device
        )
        final_rot_error_norm = torch.full_like(final_pos_error, torch.inf)
        zero_joint_vel = torch.zeros_like(self.robot.data.joint_vel[env_ids])
        identity = torch.eye(6, device=env.device).unsqueeze(0)

        for seed_offset in seed_offsets:
            pending_rows = torch.where(~success)[0]
            if len(pending_rows) == 0:
                break
            pending_env_ids = env_ids[pending_rows]
            attempt_count[pending_rows] += 1

            seed_joint_pos = self.robot.data.default_joint_pos[pending_env_ids].clone()
            seed_joint_pos[:, self.arm_joint_ids] += seed_offset.unsqueeze(0)
            seed_limits = self.robot.data.soft_joint_pos_limits[
                pending_env_ids[:, None], self.arm_joint_ids
            ]
            seed_joint_pos[:, self.arm_joint_ids] = torch.clamp(
                seed_joint_pos[:, self.arm_joint_ids],
                seed_limits[..., 0],
                seed_limits[..., 1],
            )
            seed_joint_vel = torch.zeros_like(
                self.robot.data.joint_vel[pending_env_ids]
            )
            self.robot.set_joint_position_target(
                seed_joint_pos, env_ids=pending_env_ids
            )
            self.robot.set_joint_velocity_target(
                seed_joint_vel, env_ids=pending_env_ids
            )
            self.robot.write_joint_state_to_sim(
                seed_joint_pos, seed_joint_vel, env_ids=pending_env_ids
            )

            for _ in range(max_iterations):
                # Do not continue this seed for environments that have already
                # converged during this attempt.
                active_rows = pending_rows[~success[pending_rows]]
                if len(active_rows) == 0:
                    break
                active_env_ids = env_ids[active_rows]

                current_pos = self.robot.data.body_pos_w[
                    active_env_ids, self.eef_body_id
                ]
                current_quat = self.robot.data.body_quat_w[
                    active_env_ids, self.eef_body_id
                ]
                pos_error, rot_error = math_utils.compute_pose_error(
                    current_pos,
                    current_quat,
                    desired_body_pos[active_rows],
                    desired_eef_quat[active_rows],
                    rot_error_type="axis_angle",
                )
                converged = (
                    torch.linalg.norm(pos_error, dim=-1) < position_tolerance
                ) & (
                    torch.linalg.norm(rot_error, dim=-1) < orientation_tolerance
                )
                success[active_rows[converged]] = True
                solve_rows = active_rows[~converged]
                if len(solve_rows) == 0:
                    break

                solve_env_ids = env_ids[solve_rows]
                error = torch.cat(
                    (pos_error[~converged], rot_error[~converged]), dim=-1
                )
                jacobian = self.robot.root_physx_view.get_jacobians()[
                    solve_env_ids, self.jacobian_body_id, :, :
                ]
                jacobian = jacobian[..., self.jacobian_joint_ids]
                dls_inverse = torch.linalg.solve(
                    jacobian @ jacobian.transpose(1, 2)
                    + damping**2 * identity,
                    error.unsqueeze(-1),
                )
                delta_joint = (
                    jacobian.transpose(1, 2) @ dls_inverse
                ).squeeze(-1)
                delta_joint = torch.clamp(step_size * delta_joint, -0.2, 0.2)

                joint_pos = self.robot.data.joint_pos[solve_env_ids].clone()
                joint_pos[:, self.arm_joint_ids] += delta_joint
                limits = self.robot.data.soft_joint_pos_limits[
                    solve_env_ids[:, None], self.arm_joint_ids
                ]
                joint_pos[:, self.arm_joint_ids] = torch.clamp(
                    joint_pos[:, self.arm_joint_ids],
                    limits[..., 0],
                    limits[..., 1],
                )
                joint_vel = torch.zeros_like(
                    self.robot.data.joint_vel[solve_env_ids]
                )
                self.robot.set_joint_position_target(
                    joint_pos, env_ids=solve_env_ids
                )
                self.robot.set_joint_velocity_target(
                    joint_vel, env_ids=solve_env_ids
                )
                self.robot.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=solve_env_ids
                )

            # Re-evaluate every environment assigned to this seed after the last
            # update, including the result of the final IK iteration.
            current_pos = self.robot.data.body_pos_w[
                pending_env_ids, self.eef_body_id
            ]
            current_quat = self.robot.data.body_quat_w[
                pending_env_ids, self.eef_body_id
            ]
            current_eef_pos = current_pos + math_utils.quat_apply(
                current_quat, offset[pending_rows]
            )
            pos_error_norm = torch.linalg.norm(
                desired_eef_pos[pending_rows] - current_eef_pos, dim=-1
            )
            _, rot_error = math_utils.compute_pose_error(
                current_pos,
                current_quat,
                current_pos,
                desired_eef_quat[pending_rows],
                rot_error_type="axis_angle",
            )
            rot_error_norm = torch.linalg.norm(rot_error, dim=-1)
            finite = torch.isfinite(pos_error_norm) & torch.isfinite(rot_error_norm)
            seed_success = (
                finite
                & (pos_error_norm < position_tolerance)
                & (rot_error_norm < orientation_tolerance)
            )
            success[pending_rows] = seed_success
            final_pos_error[pending_rows] = pos_error_norm
            final_rot_error_norm[pending_rows] = rot_error_norm

        # Preserve solved poses as command targets at the first policy step.
        final_joint_pos = self.robot.data.joint_pos[env_ids].clone()
        self.robot.set_joint_position_target(final_joint_pos, env_ids=env_ids)
        self.robot.set_joint_velocity_target(zero_joint_vel, env_ids=env_ids)

        env.reaching_ik_success[env_ids] = success
        env.reaching_ik_attempt_count[env_ids] = attempt_count
        if not torch.all(success):
            failed_rows = torch.where(~success)[0]
            failed_env_ids = env_ids[failed_rows]
            preview_rows = failed_rows[:8]
            details = [
                {
                    "env_id": int(env_ids[row].item()),
                    "target_pos_w": env.target_work_pos_w[env_ids[row]].tolist(),
                    "sweep_dir": env.sweep_dir[env_ids[row]].tolist(),
                    "desired_eef_pos_w": desired_eef_pos[row].tolist(),
                    "position_error": float(final_pos_error[row].item()),
                    "orientation_error": float(final_rot_error_norm[row].item()),
                }
                for row in preview_rows
            ]
            raise RuntimeError(
                "Reaching-pose IK reset validation failed for "
                f"{len(failed_env_ids)}/{len(env_ids)} environments after "
                f"{len(seed_offsets)} joint seeds. No episode was started and "
                "the object pose was not resampled. First failures: "
                f"{details}"
            )
