"""Seeded clutter resets and lifecycle management for the reusable object pool."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as mdp

from ..geometry import inspect_spawn_geometry
from ..layout import OBJECT_SPECS
from ..sampling import sample_scene

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _initialize_spawn_metadata(env: ManagerBasedEnv) -> None:
    if hasattr(env, "spawn_geometries"):
        return
    env.spawn_geometries, env.spawn_rules = inspect_spawn_geometry(env)
    count = len(OBJECT_SPECS)
    env.active_object_mask = torch.zeros((env.num_envs, count), device=env.device, dtype=torch.bool)
    env.active_object_ids = torch.full(
        (env.num_envs, env.spawn_rules.max_objects), -1, device=env.device, dtype=torch.long
    )
    env.target_object_id = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    env.blocker_object_id = torch.full_like(env.target_object_id, -1)
    env.access_region = torch.zeros((env.num_envs, 4), device=env.device)
    # The initial-vision camera is not implemented at the spawn stage. Never
    # interpret a geometric overlap test as measured image visibility.
    env.target_visibility_validated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env.target_visible_fraction = torch.full((env.num_envs,), torch.nan, device=env.device)
    env.spawn_initial_state = torch.zeros((env.num_envs, count, 13), device=env.device)
    env.spawn_reset_counts = torch.zeros(env.num_envs, dtype=torch.long)
    env.spawn_seeds = torch.zeros_like(env.spawn_reset_counts)
    env.spawn_samples = [None] * env.num_envs


def reset_spawn_scene(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Sample active identities/roles/poses without altering unselected environments.

    The fixed access region and clearance witness belong to task/sampler metadata.
    The witness is never supplied as an actor goal or a Blocker final pose.
    """
    if env_ids.numel() == 0:
        return
    _initialize_spawn_metadata(env)
    mdp.reset_scene_to_default(env, env_ids, reset_joint_targets=True)
    objects = env.scene["objects"]
    # Stage all samples first, so an impossible distribution fails before pool mutation.
    selected_ids = env_ids.cpu().tolist()
    samples = []
    seeds = []
    for env_id in selected_ids:
        seed = int(env.cfg.seed) + 1_000_003 * env_id + 97_409 * int(env.spawn_reset_counts[env_id])
        seeds.append(seed)
        rng = random.Random(seed)
        # Fix externally assigned episode roles before generating any poses.
        targets = [
            index for index, spec in enumerate(OBJECT_SPECS)
            if env.cfg.target_asset is None or spec.asset_name == env.cfg.target_asset
            if env.cfg.blocker_asset is None or spec.asset_name != env.cfg.blocker_asset
        ]
        if not targets:
            raise ValueError(
                f"No Target asset distinct from Blocker: target={env.cfg.target_asset}, blocker={env.cfg.blocker_asset}"
            )
        target_id = rng.choice(targets)
        target_asset = OBJECT_SPECS[target_id].asset_name
        blockers = [
            index for index, spec in enumerate(OBJECT_SPECS)
            if spec.asset_name != target_asset
            and (env.cfg.blocker_asset is None or spec.asset_name == env.cfg.blocker_asset)
        ]
        if not blockers:
            raise ValueError(
                f"Blocker asset must differ from the unique Target asset {target_asset}: {env.cfg.blocker_asset}"
            )
        blocker_id = rng.choice(blockers)
        samples.append(sample_scene(
            env.spawn_geometries, env.spawn_rules, rng, target_id=target_id, blocker_id=blocker_id
        ))

    states = objects.data.default_object_state[env_ids].clone()
    states[..., 7:] = 0.0
    for row, (env_id, sample, seed) in enumerate(zip(selected_ids, samples, seeds)):
        env.active_object_mask[env_id] = False
        env.active_object_ids[env_id] = -1
        env.active_object_ids[env_id, :len(sample.active_ids)] = torch.tensor(sample.active_ids, device=env.device)
        env.active_object_mask[env_id, list(sample.active_ids)] = True
        env.target_object_id[env_id] = sample.target_id
        env.blocker_object_id[env_id] = sample.blocker_id
        env.access_region[env_id] = torch.tensor(sample.access_region, device=env.device)
        env.target_visibility_validated[env_id] = False
        env.target_visible_fraction[env_id] = torch.nan
        env.spawn_samples[env_id] = sample
        env.spawn_seeds[env_id] = seed
        env.spawn_reset_counts[env_id] += 1
        for object_id in sample.active_ids:
            geometry, yaw = env.spawn_geometries[object_id], sample.yaws[object_id]
            cx = (geometry.lower[0] + geometry.upper[0]) * 0.5
            cy = (geometry.lower[1] + geometry.upper[1]) * 0.5
            cosine, sine = math.cos(yaw), math.sin(yaw)
            px, py, pz = sample.positions[object_id]
            states[row, object_id, :3] = torch.tensor(
                (px - cosine * cx + sine * cy, py - sine * cx - cosine * cy, pz), device=env.device
            )
            # Quaternion product: world-Z yaw * canonical upright rotation.
            qw, qx, qy, qz = OBJECT_SPECS[object_id].upright_quat
            cw, sw = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            states[row, object_id, 3:7] = torch.tensor(
                (cw * qw - sw * qz, cw * qx - sw * qy, cw * qy + sw * qx, cw * qz + sw * qw),
                device=env.device,
            )

    states[..., :3] += env.scene.env_origins[env_ids, None, :]
    objects.write_object_state_to_sim(states, env_ids=env_ids)
    env.spawn_initial_state[env_ids] = states
    # Countergravity is applied in world coordinates at COM by TaskObjectPool
    # each physics substep. Active task objects keep their native dynamics.
    objects.parked_mask = ~env.active_object_mask
    objects.gravity = torch.tensor(env.cfg.sim.gravity, device=env.device)
