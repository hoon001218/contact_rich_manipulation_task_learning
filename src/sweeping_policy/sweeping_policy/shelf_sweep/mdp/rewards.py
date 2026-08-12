"""Reward terms adapted from the reference random shelf-sweep task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


EEF_OFFSET = (0.13, 0.0, 0.0)
LEFT_FINGER_OFFSET = (0.13, 0.07, 0.0)
RIGHT_FINGER_OFFSET = (0.13, -0.07, 0.0)
WRIST_OFFSET = (-0.14, 0.0, 0.0)


def _target_state(
    env: ManagerBasedRLEnv,
    object_collection_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    rows = torch.arange(env.num_envs, device=env.device)
    target_ids = env.target_id.squeeze(-1).long()
    return (
        objects.data.object_pos_w[rows, target_ids],
        objects.data.object_lin_vel_w[rows, target_ids],
    )


def _offset_body_position(
    robot: Articulation,
    body_id: int,
    offset: tuple[float, float, float],
) -> torch.Tensor:
    offset_tensor = torch.tensor(
        offset,
        dtype=torch.float32,
        device=robot.device,
    ).repeat(robot.num_instances, 1)
    return robot.data.body_pos_w[:, body_id] + math_utils.quat_apply(
        robot.data.body_quat_w[:, body_id], offset_tensor
    )


def reward_for_hand_reaching(
    env: ManagerBasedRLEnv,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Reference reaching reward; retained for logging with zero configured weight."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(eef_body_name)
    body_id = body_ids[0]
    target_pos_w, _ = _target_state(env, object_collection_cfg)
    sweep_direction = torch.sign(env.sweep_dir[:, 1])

    eef_pos_w = _offset_body_position(robot, body_id, EEF_OFFSET)
    wrist_pos_w = _offset_body_position(robot, body_id, WRIST_OFFSET)
    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] -= 0.02
    offset_pos[:, 1] -= env.target_width[:, 0] * sweep_direction
    offset_pos[:, 2] += 0.09

    distance_ee = torch.linalg.norm(offset_pos - eef_pos_w, dim=-1)
    # Kept to preserve the reference reaching-pose calculation.
    _ = torch.linalg.norm(offset_pos[:, 1:3] - wrist_pos_w[:, 1:3], dim=-1)
    return torch.exp(-10.0 * distance_ee)


def align_ee_target(
    env: ManagerBasedRLEnv,
    shelf_cfg: SceneEntityCfg = SceneEntityCfg("shelf"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Reference signed squared alignment between EEF y and shelf z axes."""
    robot: Articulation = env.scene[robot_cfg.name]
    shelf: RigidObject = env.scene[shelf_cfg.name]
    body_ids, _ = robot.find_bodies(eef_body_name)
    ee_rotation = math_utils.matrix_from_quat(
        robot.data.body_quat_w[:, body_ids[0]]
    )
    shelf_rotation = math_utils.matrix_from_quat(
        shelf.data.default_root_state[:, 3:7]
    )
    ee_y = ee_rotation[..., 1]
    shelf_z = shelf_rotation[..., 2]
    alignment = torch.bmm(
        ee_y.unsqueeze(1), shelf_z.unsqueeze(-1)
    ).squeeze(-1).squeeze(-1)
    return torch.sign(alignment) * alignment.square()


def pushing_target(
    env: ManagerBasedRLEnv,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    eef_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Reference goal-distance, contact-pose, and object-speed shaping."""
    robot: Articulation = env.scene[robot_cfg.name]
    body_ids, _ = robot.find_bodies(eef_body_name)
    body_id = body_ids[0]
    target_pos_w, target_lin_vel_w = _target_state(env, object_collection_cfg)
    sweep_direction = torch.sign(env.sweep_dir[:, 1])
    eef_pos_w = _offset_body_position(robot, body_id, EEF_OFFSET)
    wrist_pos_w = _offset_body_position(robot, body_id, WRIST_OFFSET)

    offset_pos = target_pos_w.clone()
    offset_pos[:, 0] -= 0.02
    offset_pos[:, 1] -= env.target_width[:, 0] * sweep_direction
    offset_pos[:, 2] += 0.09
    distance = torch.linalg.norm(env.target_goal_pos_w - target_pos_w, dim=-1)
    zeta_m = torch.where(
        torch.linalg.norm(offset_pos - eef_pos_w, dim=-1) < 0.04,
        torch.where(torch.abs(offset_pos[:, 1] - wrist_pos_w[:, 1]) < 0.04, 1.0, 0.0),
        0.0,
    )
    object_velocity_reward = torch.where(
        torch.abs(target_lin_vel_w[:, 1]) > 0.05,
        torch.where(torch.abs(target_lin_vel_w[:, 1]) < 0.1, 0.5, -0.5),
        0.0,
    )
    return torch.where(
        distance < 0.03,
        2.0 * torch.exp(-5.0 * distance),
        zeta_m * ((1.0 - distance / 0.18) + object_velocity_reward),
    )


def homing_reward(
    env: ManagerBasedRLEnv,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference reward for returning toward the nominal arm pose after success."""
    robot: Articulation = env.scene[robot_cfg.name]
    target_pos_w, _ = _target_state(env, object_collection_cfg)
    distance = torch.linalg.norm(
        env.target_goal_pos_w[:, 1:] - target_pos_w[:, 1:], dim=-1
    )
    joint_error = torch.sum(
        torch.abs(
            robot.data.joint_pos[:, :5]
            - robot.data.default_joint_pos[:, :5]
        ),
        dim=1,
    )
    home_pose_reward = torch.exp(-0.5 * joint_error)
    gate = 0.5 * (1.0 - torch.tanh(100.0 * (distance - 0.03)))
    return home_pose_reward * gate


def object_collision(
    env: ManagerBasedRLEnv,
    object_collection_cfg: SceneEntityCfg = SceneEntityCfg("objects"),
) -> torch.Tensor:
    """Penalize motion of every non-target object, independently per environment."""
    objects: RigidObjectCollection = env.scene[object_collection_cfg.name]
    target_ids = env.target_id.squeeze(-1).long()
    rows = torch.arange(env.num_envs, device=env.device)
    velocities = torch.round(objects.data.object_lin_vel_w.clone(), decimals=2)
    velocities[rows, target_ids] = 0.0
    return torch.tanh(torch.sum(torch.abs(velocities)))


class shelf_Collision(ManagerTermBase):
    """Reference shelf-motion and robot/shelf proximity penalty."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene["robot"]
        body_ids, _ = self.robot.find_bodies("robotiq_base_link")
        self.body_id = body_ids[0]
        self.shelf: RigidObject = env.scene["shelf"]
        self.initial_shelf_pos = (
            self.shelf.data.default_root_state[:, :3] + env.scene.env_origins
        )

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        del env
        shelf_velocity = self.shelf.data.root_vel_w
        shelf_delta = self.shelf.data.root_pos_w - self.initial_shelf_pos
        moved = torch.where(
            torch.linalg.norm(shelf_delta, dim=-1)
            + torch.linalg.norm(shelf_velocity, dim=-1)
            > 0.005,
            1.0,
            0.0,
        )

        shelf_reference = self.shelf.data.root_pos_w.clone()
        shelf_reference[:, 2] += 1.06
        eef_pos = _offset_body_position(self.robot, self.body_id, EEF_OFFSET)
        left_finger = _offset_body_position(
            self.robot, self.body_id, LEFT_FINGER_OFFSET
        )
        right_finger = _offset_body_position(
            self.robot, self.body_id, RIGHT_FINGER_OFFSET
        )
        wrist = _offset_body_position(self.robot, self.body_id, WRIST_OFFSET)
        zeta = torch.where(
            torch.linalg.norm(shelf_reference - eef_pos, dim=-1) < 0.2,
            1.0,
            0.0,
        )
        left_reward = torch.clamp(
            1.0 - (left_finger[:, 2] - shelf_reference[:, 2]) / 0.02,
            0.0,
            1.0,
        )
        right_reward = torch.clamp(
            1.0 - (right_finger[:, 2] - shelf_reference[:, 2]) / 0.02,
            0.0,
            1.0,
        )
        wrist_reward = torch.clamp(
            1.0 - (wrist[:, 2] - shelf_reference[:, 2]) / 0.08,
            0.0,
            1.0,
        )
        return moved + zeta * (left_reward + right_reward + wrist_reward)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reference squared velocity penalty over the six UR5e arm joints."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.joint_vel[:, :6]), dim=1)
