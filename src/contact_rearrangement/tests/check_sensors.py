"""Check rendered initial vision and the articulation F/T sensor on the GPU.

Run with the Isaac Lab Python environment, for example:
  python src/contact_rearrangement/tests/check_sensors.py --num_envs 4

The public scene script runs 151 environment steps with its default five-second
reset and full/partial-reset validation. Additional checks preserve native
physics state through render warmup/capture, freeze the accepted RGB-D snapshot
between resets, preserve untouched sensor observations, and verify six-axis
joint reactions against applied loads and measured rigid-body dynamics.

The load bench adds two environment steps and seven physics steps, then resets
to a clean accepted episode. Camera rendering never advances physics time.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
public_script = REPOSITORY_ROOT / "src/contact_rearrangement/scripts/spawn_scene.py"
fixture_parser = argparse.ArgumentParser(description=__doc__)
fixture_parser.add_argument("--num_envs", type=int, default=4, choices=(1, 2, 4))
fixture_parser.add_argument("--seed", type=int, default=0)
fixture_args = fixture_parser.parse_args()
sys.argv = [str(public_script), "--num_envs", str(fixture_args.num_envs), "--seed", str(fixture_args.seed),
            "--headless", "--device", "cuda:0", "--max_steps", "151", "--validate"]
scope = runpy.run_path(str(public_script), run_name="initial_sensors_validation_public_script")
public_globals = scope["main"].__globals__
simulation_app = scope["simulation_app"]

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from contact_rearrangement.src.assets import FORCE_TORQUE_BODY_NAME, FORCE_TORQUE_PROXIMAL_BODY_NAME
from contact_rearrangement.src.sensors import InitialVisionCfg, InitialVisionSensor, tool_wrench, tool_wrench_valid


def validate_invalid_ranges():
    """Reject invalid ranges before scene/camera handles are accessed."""
    invalid_ranges = (
        {"min_target_visible_fraction": 1.01},
        {"height_above_shelf": (0.16, float("inf"))},
        {"aim_y_jitter": -0.01},
    )
    for overrides in invalid_ranges:
        try:
            InitialVisionSensor(SimpleNamespace(cfg=SimpleNamespace(initial_vision=InitialVisionCfg(**overrides))))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Initial-camera range accepted before rendering: {overrides}")


SNAPSHOT_NAMES = (
    "initial_vision_rgb", "initial_vision_depth", "initial_vision_role_masks", "initial_camera_pose_w",
    "initial_camera_intrinsics", "initial_vision_pixel_counts", "initial_camera_attempts",
    "target_visibility_validated", "target_visible_fraction", "blocker_visibility_validated", "blocker_visible_fraction",
)
audit = {"resets": 0, "partial_resets": 0, "steps": 0, "renders": 0, "captures": 0}


def snapshot(env, selected=slice(None)):
    return {name: getattr(env, name)[selected].clone() for name in SNAPSHOT_NAMES}


def assert_snapshot_equal(env, saved, selected=slice(None)):
    for name, expected in saved.items():
        torch.testing.assert_close(getattr(env, name)[selected], expected, atol=0.0, rtol=0.0, equal_nan=True,
                                   msg=f"Accepted snapshot changed outside reset: {name}")


def physics_state(env):
    robot, objects = env.scene["robot"], env.scene["objects"]
    return {
        "time": env.sim.current_time,
        "step": env.sim.current_time_step_index,
        "object_pose": objects.root_physx_view.get_transforms().clone(),
        "object_velocity": objects.root_physx_view.get_velocities().clone(),
        "robot_pose": robot.root_physx_view.get_link_transforms().clone(),
        "robot_velocity": robot.root_physx_view.get_link_velocities().clone(),
        "joint_pose": robot.data.joint_pos.clone(),
        "joint_velocity": robot.data.joint_vel.clone(),
        "joint_targets": robot.data.joint_pos_target.clone(),
    }


def assert_physics_preserved(env, saved, context):
    actual = physics_state(env)
    assert actual["time"] == saved["time"], f"{context} advanced physics time"
    assert actual["step"] == saved["step"], f"{context} advanced the native physics-step index"
    for name in set(saved) - {"time", "step"}:
        torch.testing.assert_close(actual[name], saved[name], atol=0.0, rtol=0.0,
                                   msg=f"{context} changed native physical state: {name}")


def validate_snapshot(env, selected):
    cfg = env.cfg.initial_vision
    masks = env.initial_vision_role_masks[selected]
    counts = env.initial_vision_pixel_counts[selected]
    assert masks.dtype == torch.bool and masks.shape[-1] == 2
    torch.testing.assert_close(masks.sum(dim=(1, 2)), counts[..., 0], atol=0.0, rtol=0.0)
    assert bool((counts[..., 1] >= cfg.min_silhouette_pixels).all())
    fractions = counts[..., 0].to(torch.float64) / counts[..., 1]
    fractions = fractions.clamp(max=1.0)
    torch.testing.assert_close(env.target_visible_fraction[selected], fractions[:, 0].float(), atol=1e-7, rtol=0)
    torch.testing.assert_close(env.blocker_visible_fraction[selected], fractions[:, 1].float(), atol=1e-7, rtol=0)
    assert bool((fractions[:, 0] >= cfg.min_target_visible_fraction).all())
    assert bool((counts[:, 1, 0] + cfg.blocker_pixel_tolerance
                 >= cfg.min_blocker_visible_fraction * counts[:, 1, 1]).all())
    assert bool(env.target_visibility_validated[selected].all() and env.blocker_visibility_validated[selected].all())
    depth = env.initial_vision_depth[selected, ..., 0]
    role_depth = depth[masks.any(dim=-1)]
    assert bool(torch.isfinite(role_depth).all() and (role_depth > 0).all()), "Role pixels lack measured RGB-D depth"
    rgb = env.initial_vision_rgb[selected]
    assert rgb.dtype == torch.uint8 and bool((rgb > 0).any()) and rgb.shape[-1] == 3
    pose = env.initial_camera_pose_w[selected]
    torch.testing.assert_close(pose[:, 3:].norm(dim=-1), torch.ones(len(selected), device=env.device),
                               atol=1e-5, rtol=0)
    local_x = pose[:, 0] - env.scene.env_origins[selected, 0]
    distances = local_x - env.spawn_rules.x_range[1]
    assert bool((distances >= cfg.front_distance_range[0] - 1e-5).all()
                and (distances <= cfg.front_distance_range[1] + 1e-5).all())


def verify_force_torque_load_response(env, sensor):
    robot = env.scene["robot"]
    view = robot.root_physx_view
    model = env.robot_pose_model
    body_names = tuple(robot.body_names)
    selected_names = {FORCE_TORQUE_BODY_NAME}
    while True:
        descendants = {child for child, parent in model.body_parents.items() if parent in selected_names}
        if descendants <= selected_names:
            break
        selected_names |= descendants
    ids = np.asarray([body_names.index(name) for name in selected_names])
    masses = view.get_masses()[0].cpu().numpy()[ids]
    # The API defines column-major inertia at COM in the actor/body frame.
    inertias = view.get_inertias()[0].cpu().numpy()[ids].reshape(-1, 3, 3).transpose(0, 2, 1)
    gravity = np.asarray(env.sim.cfg.gravity, dtype=np.float64)
    gravity_enabled = view.get_disable_gravities()[0].cpu().numpy()[ids] == 0
    assert gravity_enabled.all(), "This bench expects gravity enabled on every hand-side link."
    print(f"[F/T BENCH]: {len(ids)} downstream links; total mass={masses.sum():.6f}kg", flush=True)
    dt = env.physics_dt
    errors = []

    def sample(force_s, torque_s):
        pose_before = view.get_link_transforms()[0, sensor.body_index].cpu().numpy()
        rotation_before = Rotation.from_quat(pose_before[3:]).as_matrix()
        force_w = rotation_before @ force_s
        torque_w = rotation_before @ torque_s
        force = torch.as_tensor(force_w, dtype=torch.float32, device=env.device).reshape(1, 1, 3)
        torque = torch.as_tensor(torque_w, dtype=torch.float32, device=env.device).reshape(1, 1, 3)
        robot.set_external_force_and_torque(force, torque, body_ids=[sensor.body_index], env_ids=[0], is_global=True)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)
        sensor.update(dt)
        poses = view.get_link_transforms()[0].cpu().numpy()
        all_rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
        com_local = view.get_coms()[0, :, :3].cpu().numpy()
        com_w = poses[:, :3] + np.einsum("nij,nj->ni", all_rotations, com_local)
        acc_w = view.get_link_accelerations()[0].cpu().numpy()[ids]
        vel_w = view.get_link_velocities()[0].cpu().numpy()[ids]
        rotations = all_rotations[ids]
        inertia_w = rotations @ inertias @ rotations.transpose(0, 2, 1)
        effective_forces = masses[:, None] * (acc_w[:, :3] - gravity)
        rotational_moments = np.einsum("nij,nj->ni", inertia_w, acc_w[:, 3:])
        rotational_moments += np.cross(vel_w[:, 3:], np.einsum("nij,nj->ni", inertia_w, vel_w[:, 3:]))
        point = poses[sensor.body_index, :3]
        net_force_w = effective_forces.sum(axis=0) - force_w
        net_torque_w = (np.cross(com_w[ids] - point, effective_forces) + rotational_moments).sum(axis=0)
        net_torque_w -= np.cross(com_w[sensor.body_index] - point, force_w) + torque_w
        rotation = all_rotations[sensor.body_index]
        expected = np.r_[rotation.T @ net_force_w, rotation.T @ net_torque_w]
        observed = sensor.read([0])[0].cpu().numpy()
        errors.append(observed - expected)

    try:
        sample(np.zeros(3), np.zeros(3))
        for axis in range(3):
            force = np.zeros(3)
            force[axis] = 3.0
            sample(force, np.zeros(3))
        for axis in range(3):
            torque = np.zeros(3)
            torque[axis] = 0.10
            sample(np.zeros(3), torque)
    finally:
        zero = torch.zeros((1, 1, 3), device=env.device)
        robot.set_external_force_and_torque(zero, zero, body_ids=[sensor.body_index], env_ids=[0], is_global=True)
    errors = np.asarray(errors)
    # Pose/velocity are end-of-step values while reactions are step averages;
    # the tolerance includes this small discretization and native damping.
    force_error = np.max(np.linalg.norm(errors[:, :3], axis=1))
    torque_error = np.max(np.linalg.norm(errors[:, 3:], axis=1))
    assert force_error < 0.10, f"Force balance mismatch: {force_error}N"
    assert torque_error < 0.015, f"Sensor-center torque balance mismatch: {torque_error}Nm"
    print(f"[F/T BENCH]: six-axis applied-load balance passed; "
          f"max_force_error={force_error:.6g}N max_torque_error={torque_error:.6g}Nm", flush=True)


original_render = InitialVisionSensor._render
original_capture = InitialVisionSensor.capture


def audited_render(self, camera):
    saved = physics_state(self.env)
    result = original_render(self, camera)
    assert_physics_preserved(self.env, saved, "Initial/reference camera render")
    audit["renders"] += 1
    return result


def audited_capture(self, env_ids, **kwargs):
    saved = physics_state(self.env)
    result = original_capture(self, env_ids, **kwargs)
    assert_physics_preserved(self.env, saved, "Complete initial snapshot capture")
    audit["captures"] += 1
    return result


InitialVisionSensor._render = audited_render
InitialVisionSensor.capture = audited_capture
original_env_class = scope["ManagerBasedEnv"]


class AuditedEnv(original_env_class):
    def reset(self, seed=None, env_ids=None, options=None):
        selected = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        untouched_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        untouched_mask[selected] = False
        untouched = untouched_mask.nonzero().flatten()
        saved_snap = snapshot(self, untouched) if untouched.numel() else {}
        sensor = self.tool_force_torque_sensor
        saved_wrench = sensor.read(untouched)
        saved_valid = sensor.valid[untouched].clone()
        time, step = self.sim.current_time, self.sim.current_time_step_index
        result = super().reset(seed=seed, env_ids=selected, options=options)
        assert self.sim.current_time == time and self.sim.current_time_step_index == step, (
            f"Reset capture advanced physics: time {time}->{self.sim.current_time}; "
            f"step {step}->{self.sim.current_time_step_index}"
        )
        validate_snapshot(self, selected)
        assert not bool(sensor.valid[selected].any()), "F/T reset values must remain invalid until physics advances"
        torch.testing.assert_close(sensor.read(selected), torch.zeros((len(selected), 6), device=self.device), atol=0, rtol=0)
        torch.testing.assert_close(tool_wrench(self)[selected], sensor.read(selected), atol=0, rtol=0)
        assert not bool(tool_wrench_valid(self)[selected].any()), "Observation reread made stale reset wrench valid"
        if untouched.numel():
            assert_snapshot_equal(self, saved_snap, untouched)
            torch.testing.assert_close(sensor.read(untouched), saved_wrench, atol=0, rtol=0)
            torch.testing.assert_close(sensor.valid[untouched], saved_valid, atol=0, rtol=0)
            audit["partial_resets"] += 1
        self._audited_accepted_snapshot = snapshot(self)
        audit["resets"] += 1
        print(f"[SENSOR AUDIT]: reset#{audit['resets']} selected={selected.tolist()} "
              f"camera_attempts={self.initial_camera_attempts[selected].tolist()} "
              f"target_visible={self.target_visible_fraction[selected].tolist()} "
              f"blocker_visible={self.blocker_visible_fraction[selected].tolist()} "
              f"F/T invalid; capture physics-step delta=0", flush=True)
        return result

    def step(self, action):
        result = super().step(action)
        assert_snapshot_equal(self, self._audited_accepted_snapshot)
        assert bool(self.tool_force_torque_sensor.valid.all()), "Completed step did not produce fresh F/T measurements"
        assert bool(torch.isfinite(self.tool_force_torque_sensor.read()).all())
        assert self.ft_last_update_time == self.sim.current_time, "F/T observation timestamp lags completed physics"
        audit["steps"] += 1
        return result


public_globals["ManagerBasedEnv"] = AuditedEnv
original_validate = scope["validate_settling_and_reset"]


def validate_with_load_bench(env, visualizer=None):
    original_validate(env, visualizer)
    robot = env.scene["robot"]
    assert robot.num_bodies == 41 and robot.num_joints == 18
    assert len(env.robot_pose_model.colliders) == 41
    assert {FORCE_TORQUE_BODY_NAME, FORCE_TORQUE_PROXIMAL_BODY_NAME} <= set(robot.body_names)
    # The public validator ends in a full reset. One ordinary environment step
    # proves invalid→valid and makes untouched environments' F/T valid before
    # the following partial-reset preservation check.
    env.step(env.action_manager.action)
    selected = torch.tensor([0], device=env.device, dtype=torch.long)
    env.reset(env_ids=selected)
    env.step(env.action_manager.action)
    frozen = snapshot(env)
    before_bench = env.sim.current_time_step_index
    verify_force_torque_load_response(env, env.tool_force_torque_sensor)
    assert env.sim.current_time_step_index - before_bench == 7
    # The bench explicitly updates the raw sensor while directly stepping
    # physics. Bring the observation callback's time marker to that same step.
    tool_wrench(env)
    assert env.ft_last_update_time == env.sim.current_time
    assert_snapshot_equal(env, frozen)
    assert bool(tool_wrench_valid(env).all())
    env.reset()
    all_ids = torch.arange(env.num_envs, device=env.device)
    scope["validate_reset_state"](env, all_ids)
    validate_snapshot(env, all_ids)
    assert not bool(env.tool_force_torque_sensor.valid.any())
    print(f"[SENSOR AUDIT]: PASS {env.num_envs} envs; "
          f"public151steps+2envsteps+7benchphysicssteps; resets={audit['resets']} "
          f"partial_resets={audit['partial_resets']} captures={audit['captures']} renders={audit['renders']}; "
          "native geometry41body/18DOF/41collider; snapshots frozen; selected reset isolated; six-axis load verified",
          flush=True)


public_globals["validate_settling_and_reset"] = validate_with_load_bench

try:
    validate_invalid_ranges()
    scope["main"]()
except Exception:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    simulation_app.close()
