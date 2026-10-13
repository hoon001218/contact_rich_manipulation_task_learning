"""Check rendered initial vision and the articulation F/T sensor on the GPU.

Run with the Isaac Lab Python environment, for example:
  python src/contact_rearrangement/tests/check_sensors.py --num_envs 4

The public scene script runs 151 environment steps with its default five-second
reset and full/partial-reset validation. Additional checks preserve native
physics state through render warmup/capture, freeze the accepted RGB-D snapshot
between resets, preserve untouched sensor observations, and verify six-axis
joint reactions against applied loads and measured rigid-body dynamics. The
runtime compensated observation is also checked during unloaded wrist/finger
motion and all six directions of known external force/torque.

The load bench adds two environment steps and 19 physics steps, then resets
to a clean accepted episode. Camera rendering never advances physics time.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


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

from contact_rearrangement.mdp.observations import corrected_wrench
from contact_rearrangement.mdp.state import MdpCfg
from contact_rearrangement.mdp.events import reset_spawn_scene
from contact_rearrangement.src.assets import (
    ARM_JOINT_NAMES, HAND_JOINT_NAMES, FORCE_TORQUE_BODY_NAME, FORCE_TORQUE_PROXIMAL_BODY_NAME,
    ToolForceTorqueSensor,
)
from contact_rearrangement.src.sensors import (
    InitialVisionCfg, InitialVisionRejected, InitialVisionSensor, tool_wrench, tool_wrench_valid,
)


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


def verify_shared_cameras(env):
    cameras = (env.scene["initial_camera"], env.scene["silhouette_camera"])
    assert all(camera.num_instances == 1 for camera in cameras), "Reset cameras must not scale with env count"
    paths = [path for camera in cameras for path in camera.render_product_paths]
    assert len(paths) == len(set(paths)) == 2, "Initial vision must reuse exactly two render products"
    for camera in cameras:
        assert camera.data.pos_w.shape == (1, 3)
    references = env.scene.stage.GetPrimAtPath("/World/VisionReferences")
    assert len(list(references.GetChildren())) == 10, "Silhouette references must be a shared catalog pool"


def verify_independent_vision_retries(env):
    """Force one environment to retry; retain another's accepted physical scene."""
    if env.num_envs < 2:
        return
    capture = InitialVisionSensor.capture
    protected = {}
    injected = False
    accepted_zero = 0
    before_counts = env.spawn_reset_counts.clone()

    def protected_state():
        robot, objects = env.scene["robot"], env.scene["objects"]
        return {"objects": objects.data.object_state_w[:1].clone(),
                "root": robot.data.root_state_w[:1].clone(),
                "q": robot.data.joint_pos[:1].clone(),
                "v": robot.data.joint_vel[:1].clone(),
                "targets": robot.data.joint_pos_target[:1].clone()}

    def retry_once(sensor, env_ids, **kwargs):
        nonlocal injected, accepted_zero
        assert len(env_ids) == 1, "Layout rejection must be isolated to one environment"
        index = int(env_ids[0])
        if index == 1:
            assert protected, "First environment should already have an accepted snapshot"
            assert env.spawn_samples[0] is protected["sample"], "Retry resampled an accepted environment"
            for key, value in protected_state().items():
                torch.testing.assert_close(value, protected["physical"][key], rtol=0, atol=1e-6)
            if not injected:
                injected = True
                raise InitialVisionRejected("fixture: retry env1 while preserving env0")
        result = capture(sensor, env_ids, **kwargs)
        if index == 0:
            accepted_zero += 1
            protected.update(sample=env.spawn_samples[0], physical=protected_state(), result=result[0])
        return result

    selected = torch.tensor([0, 1], device=env.device)
    with patch.object(InitialVisionSensor, "capture", retry_once):
        env.reset(env_ids=selected)
    assert injected and accepted_zero == 1
    torch.testing.assert_close(env.spawn_reset_counts[selected], before_counts[selected] + 1)
    for name, key in (("initial_vision_rgb", "rgb"), ("initial_vision_depth", "depth"),
                      ("initial_vision_role_masks", "masks"), ("initial_camera_pose_w", "pose")):
        torch.testing.assert_close(getattr(env, name)[0], protected["result"][key], rtol=0, atol=0)
    verify_shared_cameras(env)
    print("[VISION RETRY FIXTURE]: PASS; env1 retries independently; env0 scene/snapshot retained", flush=True)


def verify_vision_retry_rollback(env):
    """Exhaust a later environment's retry and restore the entire selected batch."""
    if env.num_envs < 2:
        return
    capture = InitialVisionSensor.capture
    before_physics, before_snapshot = physics_state(env), snapshot(env)
    metadata_names = ("active_object_mask", "active_object_ids", "target_object_id", "blocker_object_id",
                      "access_region", "spawn_initial_state", "spawn_reset_counts", "spawn_seeds",
                      "robot_initial_joint_pos", "robot_start_tcp_pos_w", "robot_start_region",
                      "robot_start_elbow_height", "robot_start_attempts")
    metadata = {name: getattr(env, name).clone() for name in metadata_names}
    before_samples = list(env.spawn_samples)
    before_raw = env.tool_force_torque_sensor.read()
    before_valid = env.tool_force_torque_sensor.valid.clone()
    camera = env.scene["initial_camera"]
    before_camera = (camera.data.pos_w.clone(), camera.data.quat_w_ros.clone())
    layout_attempts = env.cfg.initial_vision.max_layout_attempts
    accepted_zero = False

    def exhaust_later(sensor, env_ids, **kwargs):
        nonlocal accepted_zero
        assert len(env_ids) == 1
        if int(env_ids[0]) == 1:
            assert accepted_zero
            raise InitialVisionRejected("fixture: exhaust env1 retries")
        result = capture(sensor, env_ids, **kwargs)
        accepted_zero = True
        # Let env0 use its full sampling budget; then exhaust env1 immediately.
        env.cfg.initial_vision.max_layout_attempts = 1
        return result

    selected = torch.tensor([0, 1], device=env.device)
    try:
        with patch.object(InitialVisionSensor, "capture", exhaust_later):
            try:
                reset_spawn_scene(env, selected)
            except InitialVisionRejected as error:
                assert str(error) == "fixture: exhaust env1 retries"
            else:
                raise AssertionError("Exhausted camera retries should abort the selected reset transaction")
    finally:
        env.cfg.initial_vision.max_layout_attempts = layout_attempts
    assert accepted_zero
    after_physics = physics_state(env)
    assert after_physics["time"] == before_physics["time"]
    assert after_physics["step"] == before_physics["step"]
    for key in set(before_physics) - {"time", "step"}:
        torch.testing.assert_close(after_physics[key], before_physics[key], rtol=0, atol=1e-6,
                                   msg=f"Exhausted retry failed to restore native state: {key}")
    assert_snapshot_equal(env, before_snapshot)
    for name, value in metadata.items():
        torch.testing.assert_close(getattr(env, name), value, rtol=0, atol=0)
    assert all(actual is before for actual, before in zip(env.spawn_samples, before_samples))
    torch.testing.assert_close(env.tool_force_torque_sensor.read(), before_raw, rtol=0, atol=0)
    torch.testing.assert_close(env.tool_force_torque_sensor.valid, before_valid)
    torch.testing.assert_close(camera.data.pos_w, before_camera[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(camera.data.quat_w_ros, before_camera[1], rtol=0, atol=1e-6)
    verify_shared_cameras(env)
    print("[VISION ROLLBACK FIXTURE]: PASS; exhausted env1 restores selected native state/metadata/snapshots", flush=True)


def verify_shared_scene_template(env):
    """Shared layouts retain independent native bodies and need no partial-reset render."""
    if env.num_envs < 2:
        return
    original_mode = env.cfg.shared_initial_scene
    capture = InitialVisionSensor.capture
    captures, accepted = [], []

    def count_capture(sensor, env_ids, **kwargs):
        captures.extend(env_ids.cpu().tolist())
        result = capture(sensor, env_ids, **kwargs)
        accepted.extend(result)
        return result

    def assert_common_scene():
        objects, robot = env.scene["objects"], env.scene["robot"]
        native = objects.reshape_view_to_data(objects.root_physx_view.get_transforms()).clone()
        native[..., :3] -= env.scene.env_origins[:, None, :]
        torch.testing.assert_close(native, native[:1].expand_as(native), rtol=0, atol=1e-5)
        torch.testing.assert_close(robot.data.joint_pos, robot.data.joint_pos[:1].expand_as(robot.data.joint_pos),
                                   rtol=0, atol=1e-6)
        for name in ("initial_vision_rgb", "initial_vision_depth", "initial_vision_role_masks",
                     "active_object_ids", "target_object_id", "blocker_object_id", "spawn_seeds"):
            value = getattr(env, name)
            torch.testing.assert_close(value, value[:1].expand_as(value), rtol=0, atol=0)
        camera_positions = env.initial_camera_pose_w[:, :3] - env.scene.env_origins
        torch.testing.assert_close(camera_positions, camera_positions[:1].expand_as(camera_positions), rtol=0, atol=1e-6)
        assert all(sample is env.shared_initial_scene_template["sample"] for sample in env.spawn_samples)

    env.cfg.shared_initial_scene = True
    try:
        if not hasattr(env, "shared_initial_scene_template"):
            before_initialization = physics_state(env)
            try:
                reset_spawn_scene(env, torch.tensor([1], device=env.device))
            except ValueError as error:
                assert "full environment reset" in str(error)
            else:
                raise AssertionError("Shared mode needs an initial full reset before restoring a partial clone")
            assert_physics_preserved(env, before_initialization, "Rejected initial partial shared reset")
        # Reverse the complete batch to check that template capture still uses
        # env0 and that template counters map to selected rows correctly.
        all_ids = torch.arange(env.num_envs, device=env.device).flip(0)
        before_counts = env.spawn_reset_counts.clone()
        with patch.object(InitialVisionSensor, "capture", count_capture):
            env.reset(env_ids=all_ids)
        assert captures and set(captures) == {0} and accepted == [0]
        torch.testing.assert_close(env.spawn_reset_counts, before_counts + 1)
        assert_common_scene()
        first_template = env.shared_initial_scene_template
        first_generation = env.shared_initial_scene_generation
        native_before, snapshot_before = physics_state(env), snapshot(env)
        renders_before, captures_before = audit["renders"], len(captures)
        with patch.object(InitialVisionSensor, "capture", count_capture):
            env.reset(env_ids=torch.tensor([1], device=env.device))
        assert audit["renders"] == renders_before and len(captures) == captures_before
        assert env.shared_initial_scene_template is first_template
        assert env.shared_initial_scene_generation == first_generation
        assert_physics_preserved(env, native_before, "Shared partial reset of an already-initial clone")
        assert_snapshot_equal(env, snapshot_before)
        assert_common_scene()

        captures.clear()
        accepted.clear()
        with patch.object(InitialVisionSensor, "capture", count_capture):
            env.reset()
        assert captures and set(captures) == {0} and accepted == [0]
        template = env.shared_initial_scene_template
        assert template is not first_template and template["seed"] != first_template["seed"]
        assert env.shared_initial_scene_generation == first_generation + 1
        assert_common_scene()

        before_physics, before_snapshot = physics_state(env), snapshot(env)
        before_counts = env.spawn_reset_counts.clone()
        attempts = env.cfg.initial_vision.max_layout_attempts

        def reject_template(sensor, env_ids, **kwargs):
            assert env_ids.cpu().tolist() == [0]
            raise InitialVisionRejected("fixture: reject shared replacement")

        try:
            env.cfg.initial_vision.max_layout_attempts = 1
            with patch.object(InitialVisionSensor, "capture", reject_template):
                try:
                    reset_spawn_scene(env, all_ids)
                except InitialVisionRejected as error:
                    assert str(error) == "fixture: reject shared replacement"
                else:
                    raise AssertionError("Shared template rejection should abort replacement")
        finally:
            env.cfg.initial_vision.max_layout_attempts = attempts
        assert env.shared_initial_scene_template is template
        assert env.shared_initial_scene_generation == first_generation + 1
        torch.testing.assert_close(env.spawn_reset_counts, before_counts)
        assert_snapshot_equal(env, before_snapshot)
        after_physics = physics_state(env)
        assert after_physics["time"] == before_physics["time"] and after_physics["step"] == before_physics["step"]
        for key in set(before_physics) - {"time", "step"}:
            torch.testing.assert_close(after_physics[key], before_physics[key], rtol=0, atol=1e-6)
        verify_shared_cameras(env)
        print("[SHARED SCENE FIXTURE]: PASS; common native layouts/joints, env0-only initial capture, "
              "partial reset without render, transactional template replacement", flush=True)
    finally:
        env.cfg.shared_initial_scene = original_mode
        env.reset()


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
    all_masses = view.get_masses().to(env.device)[:, ids]
    masses = all_masses[0].cpu().numpy()
    # The API defines column-major inertia at COM in the actor/body frame.
    all_inertias = view.get_inertias().to(env.device)[:, ids].reshape(env.num_envs, -1, 3, 3).transpose(-1, -2)
    inertias = all_inertias[0].cpu().numpy()
    gravity = np.asarray(env.sim.cfg.gravity, dtype=np.float64)
    gravity_enabled = view.get_disable_gravities()[0].cpu().numpy()[ids] == 0
    assert gravity_enabled.all(), "This bench expects gravity enabled on every hand-side link."
    # The hand's base has a disambiguated name and must be included along with
    # the fingers and distal spacer, rather than selecting only right_* links.
    hand_base_ids = [index for index in ids if "/ROASRightHand/" in str(view.link_paths[0][index])
                     and not body_names[index].startswith("right_")]
    assert hand_base_ids, "The compensation bench omitted the attached hand base."
    print(f"[F/T BENCH]: {len(ids)} downstream links; total mass={masses.sum():.6f}kg; hand base included", flush=True)
    dt = env.physics_dt
    errors, compensated_errors, ideal_compensated_errors = [], [], []
    quantization_errors, unloaded_gravity_only, unloaded_compensated = [], [], []
    half_resolution = np.asarray(sensor.calibration.resolution) * .5
    original_targets = robot.data.joint_pos_target.clone()
    missing = object()
    original_mdp_cfg = getattr(env.cfg, "mdp", missing)
    original_attributes = {
        name: getattr(env, name, missing)
        for name in ("mdp_state", "initial_object_geometry", "initial_object_valid", "initial_object_roles")
    }
    original_raw_wrench = sensor.data_raw_wrench.clone()
    original_ideal_wrench = sensor.data_ideal_wrench.clone()
    original_range_exceeded = sensor.range_exceeded.clone()
    original_sensor_bias, original_sensor_valid = sensor.bias.clone(), sensor.valid.clone()
    original_ft_timestamp = env.ft_last_update_time
    # Exercise the production observation with actual tensor API values. The
    # spawn preview has no training MDP, so provide only its physical caches.
    env.mdp_state = SimpleNamespace(
        sensor_id=sensor.body_index, downstream_ids=ids.tolist(), masses=all_masses,
        inertias=all_inertias, wrench_bias=torch.zeros((env.num_envs, 6), device=env.device),
    )
    if original_mdp_cfg is missing:
        env.cfg.mdp = MdpCfg()

    def sample(force_s, torque_s, *, unloaded_motion=False):
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
        ideal = sensor.read_ideal([0])[0].cpu().numpy()
        errors.append(ideal - expected)
        quantization_error = observed - ideal
        quantization_errors.append(quantization_error)
        assert not bool(sensor.range_exceeded[0].any()), "Load-response fixture exceeded the selected calibration."
        assert np.all(np.abs(quantization_error) <= half_resolution + 2e-6), (
            f"Measured load exceeded half-resolution quantization error: {quantization_error}"
        )
        external_moment = np.cross(com_w[sensor.body_index] - point, force_w) + torque_w
        expected_contact_reaction = -np.r_[rotation.T @ force_w, rotation.T @ external_moment]
        compensated = corrected_wrench(env)[0].cpu().numpy()
        assert np.isfinite(compensated).all(), "Compensated observation contains nonfinite values."
        compensated_errors.append(compensated - expected_contact_reaction)
        ideal_compensated_errors.append(compensated - expected_contact_reaction - quantization_error)
        # Calls during a single completed physics step must be repeatable and
        # may not advance dynamics, reset sensor validity, or rewrite raw data.
        saved = physics_state(env)
        torch.testing.assert_close(corrected_wrench(env)[0], torch.as_tensor(compensated, device=env.device),
                                   atol=0, rtol=0)
        torch.testing.assert_close(sensor.read([0])[0], torch.as_tensor(observed, device=env.device), atol=0, rtol=0)
        torch.testing.assert_close(sensor.read_ideal([0])[0], torch.as_tensor(ideal, device=env.device), atol=0, rtol=0)
        assert_physics_preserved(env, saved, "Repeated compensated F/T read")
        if unloaded_motion:
            # The gravity-only baseline uses the independently computed SciPy
            # transform, rather than the production compensation helper.
            support = -masses[:, None] * gravity
            gravity_force = support.sum(axis=0)
            gravity_moment = np.cross(com_w[ids] - point, support).sum(axis=0)
            gravity_sensor = np.r_[rotation.T @ gravity_force, rotation.T @ gravity_moment]
            unloaded_gravity_only.append(observed - gravity_sensor)
            unloaded_compensated.append(compensated)
            for name in env.binary_tactile_sensor.sensors:
                assert not bool((name.data.net_forces_w[0].norm(dim=-1) > env.cfg.tactile.force_threshold).any()), (
                    "Unloaded inertia-compensation motion touched another collider."
                )

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
        # Small opposing wrist targets create actual angular/COM acceleration;
        # a finger target also moves articulated downstream links independently.
        # Their millimetre-scale swept volume stays inside the accepted start.
        wrist_id = robot.joint_names.index(ARM_JOINT_NAMES[-1])
        finger_id = robot.joint_names.index(HAND_JOINT_NAMES[2])
        for direction in (1.0, -1.0):
            targets = original_targets[0:1].clone()
            targets[:, wrist_id] += direction * 0.02
            targets[:, finger_id] += 0.03 if direction > 0 else 0.0
            robot.set_joint_position_target(targets, env_ids=[0])
            for _ in range(6):
                sample(np.zeros(3), np.zeros(3), unloaded_motion=True)
    finally:
        zero = torch.zeros((1, 1, 3), device=env.device)
        robot.set_external_force_and_torque(zero, zero, body_ids=[sensor.body_index], env_ids=[0], is_global=True)
        robot.set_joint_position_target(original_targets)
        sensor.data_raw_wrench.copy_(original_raw_wrench)
        sensor.data_ideal_wrench.copy_(original_ideal_wrench)
        sensor.range_exceeded.copy_(original_range_exceeded)
        sensor.bias.copy_(original_sensor_bias)
        sensor.valid.copy_(original_sensor_valid)
        # Physics has advanced; restoring this pre-bench timestamp allows the
        # next ordinary raw observation to refresh from the completed step.
        env.ft_last_update_time = original_ft_timestamp
        if original_mdp_cfg is missing:
            del env.cfg.mdp
        else:
            env.cfg.mdp = original_mdp_cfg
        for name, previous in original_attributes.items():
            if previous is missing:
                if hasattr(env, name):
                    delattr(env, name)
            else:
                setattr(env, name, previous)
    errors = np.asarray(errors)
    # Pose/velocity are end-of-step values while reactions are step averages;
    # the tolerance includes this small discretization and native damping.
    force_error = np.max(np.linalg.norm(errors[:, :3], axis=1))
    torque_error = np.max(np.linalg.norm(errors[:, 3:], axis=1))
    assert force_error < 0.10, f"Force balance mismatch: {force_error}N"
    assert torque_error < 0.015, f"Sensor-center torque balance mismatch: {torque_error}Nm"
    print(f"[F/T BENCH]: six-axis applied-load balance passed; "
          f"max_force_error={force_error:.6g}N max_torque_error={torque_error:.6g}Nm", flush=True)
    compensated_errors = np.asarray(compensated_errors)
    ideal_compensated_errors = np.asarray(ideal_compensated_errors)
    force_residual_error = np.linalg.norm(compensated_errors[:, :3], axis=-1).max()
    torque_residual_error = np.linalg.norm(compensated_errors[:, 3:], axis=-1).max()
    ideal_force_error = np.linalg.norm(ideal_compensated_errors[:, :3], axis=-1).max()
    ideal_torque_error = np.linalg.norm(ideal_compensated_errors[:, 3:], axis=-1).max()
    assert ideal_force_error < 0.10, f"Ideal compensated external-force mismatch: {ideal_force_error}N"
    assert ideal_torque_error < 0.015, f"Ideal compensated external-torque mismatch: {ideal_torque_error}Nm"
    assert force_residual_error < 0.10 + np.linalg.norm(half_resolution[:3])
    assert torque_residual_error < 0.015 + np.linalg.norm(half_resolution[3:])
    gravity_only, compensated = np.asarray(unloaded_gravity_only), np.asarray(unloaded_compensated)
    gravity_force_peak = np.linalg.norm(gravity_only[:, :3], axis=-1).max()
    gravity_torque_peak = np.linalg.norm(gravity_only[:, 3:], axis=-1).max()
    compensated_force_peak = np.linalg.norm(compensated[:, :3], axis=-1).max()
    compensated_torque_peak = np.linalg.norm(compensated[:, 3:], axis=-1).max()
    assert gravity_force_peak > 0.10 and gravity_torque_peak > 0.015, (
        f"The unloaded motion did not exercise measurable inertia: {gravity_force_peak}N/{gravity_torque_peak}Nm"
    )
    assert compensated_force_peak < 0.30 * gravity_force_peak, "Force inertia was not removed from the observation."
    assert compensated_torque_peak < 0.30 * gravity_torque_peak, "Torque inertia was not removed from the observation."
    print(f"[F/T COMPENSATION]: PASS; external-load errors={force_residual_error:.6g}N/"
          f"{torque_residual_error:.6g}Nm; unloaded gravity-only peaks={gravity_force_peak:.6g}N/"
          f"{gravity_torque_peak:.6g}Nm -> full-compensation peaks={compensated_force_peak:.6g}N/"
          f"{compensated_torque_peak:.6g}Nm", flush=True)
    print(f"[F/T SI-200-8]: PASS; ideal external-load errors={ideal_force_error:.6g}N/"
          f"{ideal_torque_error:.6g}Nm; max axis quantization error="
          f"{np.abs(np.asarray(quantization_errors)).max(axis=0).tolist()}", flush=True)


def verify_calibration_cache(env):
    """Exercise the production sensor's calibration/cache with known GPU loads.

    A controlled tensor backend tests overrange and reset without applying an
    out-of-range physical load to the live articulation.
    """
    saved = physics_state(env)
    loads = torch.tensor([[200.04, -199.96, 359.96, 8.004, -7.999, .0074],
                          [0.04, -.06, 200.04, .0024, -.0026, -.0074]], device=env.device)
    backend = SimpleNamespace(get_link_incoming_joint_force=lambda: loads[:, None, :])
    robot = SimpleNamespace(body_names=[FORCE_TORQUE_BODY_NAME], num_instances=2, device=env.device,
                            root_physx_view=backend,
                            cfg=SimpleNamespace(spawn=SimpleNamespace(sensor_mount_mass=.2)))
    sensor = ToolForceTorqueSensor(robot, env.cfg.force_torque)
    sensor.update()
    expected = loads.new_tensor([[200, -200, 360, 8, -8, .005],
                                [0, -.1, 200, 0, -.005, -.005]])
    flags = torch.tensor([[True, False, False, True, False, False],
                          [False, False, False, False, False, False]], device=env.device)
    torch.testing.assert_close(sensor.read(), expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(sensor.read_ideal(), loads, rtol=0, atol=0)
    torch.testing.assert_close(sensor.range_exceeded, flags)
    before_raw, before_ideal = sensor.read([0]), sensor.read_ideal([0])
    sensor.tare([1])
    torch.testing.assert_close(sensor.read([1], tared=True), torch.zeros((1, 6), device=env.device), rtol=0, atol=0)
    sensor.reset([1])
    torch.testing.assert_close(sensor.read([0]), before_raw, rtol=0, atol=0)
    torch.testing.assert_close(sensor.read_ideal([0]), before_ideal, rtol=0, atol=0)
    torch.testing.assert_close(sensor.range_exceeded[0], flags[0])
    assert bool(sensor.valid[0]) and not bool(sensor.valid[1])
    assert not bool(sensor.range_exceeded[1].any())
    torch.testing.assert_close(sensor.read_ideal([1]), torch.zeros((1, 6), device=env.device), rtol=0, atol=0)
    assert_physics_preserved(env, saved, "Calibration/cache fixture")
    print("[F/T CALIBRATION CACHE]: PASS; per-axis range, resolution, overload, tare, partial reset", flush=True)


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
        saved_ideal = sensor.read_ideal(untouched)
        saved_overrange = sensor.range_exceeded[untouched].clone()
        saved_valid = sensor.valid[untouched].clone()
        time, step = self.sim.current_time, self.sim.current_time_step_index
        result = super().reset(seed=seed, env_ids=selected, options=options)
        assert self.sim.current_time == time and self.sim.current_time_step_index == step, (
            f"Reset capture advanced physics: time {time}->{self.sim.current_time}; "
            f"step {step}->{self.sim.current_time_step_index}"
        )
        validate_snapshot(self, selected)
        verify_shared_cameras(self)
        assert not bool(sensor.valid[selected].any()), "F/T reset values must remain invalid until physics advances"
        torch.testing.assert_close(sensor.read(selected), torch.zeros((len(selected), 6), device=self.device), atol=0, rtol=0)
        torch.testing.assert_close(sensor.read_ideal(selected), torch.zeros((len(selected), 6), device=self.device), atol=0, rtol=0)
        assert not bool(sensor.range_exceeded[selected].any())
        torch.testing.assert_close(tool_wrench(self)[selected], sensor.read(selected), atol=0, rtol=0)
        assert not bool(tool_wrench_valid(self)[selected].any()), "Observation reread made stale reset wrench valid"
        if untouched.numel():
            assert_snapshot_equal(self, saved_snap, untouched)
            torch.testing.assert_close(sensor.read(untouched), saved_wrench, atol=0, rtol=0)
            torch.testing.assert_close(sensor.read_ideal(untouched), saved_ideal, atol=0, rtol=0)
            torch.testing.assert_close(sensor.range_exceeded[untouched], saved_overrange)
            torch.testing.assert_close(sensor.valid[untouched], saved_valid, atol=0, rtol=0)
            audit["partial_resets"] += 1
        self._audited_accepted_snapshot = snapshot(self)
        self._audited_camera_frames = [self.scene[name]._frame.clone()
                                       for name in ("initial_camera", "silhouette_camera")]
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
        for name, frame in zip(("initial_camera", "silhouette_camera"), self._audited_camera_frames):
            torch.testing.assert_close(self.scene[name]._frame, frame,
                                       msg="Initial camera buffers must not be updated by ordinary environment steps")
        assert bool(self.tool_force_torque_sensor.valid.all()), "Completed step did not produce fresh F/T measurements"
        assert bool(torch.isfinite(self.tool_force_torque_sensor.read()).all())
        assert self.ft_last_update_time == self.sim.current_time, "F/T observation timestamp lags completed physics"
        audit["steps"] += 1
        return result


public_globals["ManagerBasedEnv"] = AuditedEnv
original_validate = scope["validate_settling_and_reset"]


def validate_with_load_bench(env, visualizer=None):
    original_validate(env, visualizer)
    verify_independent_vision_retries(env)
    verify_vision_retry_rollback(env)
    verify_shared_scene_template(env)
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
    verify_calibration_cache(env)
    verify_force_torque_load_response(env, env.tool_force_torque_sensor)
    assert env.sim.current_time_step_index - before_bench == 19
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
          f"public151steps+2envsteps+19benchphysicssteps; resets={audit['resets']} "
          f"partial_resets={audit['partial_resets']} captures={audit['captures']} renders={audit['renders']}; "
          "native geometry41body/18DOF/41collider; snapshots frozen; selected reset isolated; "
          "six-axis load and dynamic compensation verified",
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
