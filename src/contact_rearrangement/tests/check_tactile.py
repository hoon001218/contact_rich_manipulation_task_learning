"""Verify all 17 tactile links with real PhysX contacts and selected resets.

Run with Isaac Lab Python:
  python src/contact_rearrangement/tests/check_tactile.py --num_envs 2

The public scene runs for 151 steps, including its five-second reset and
full/partial-reset checks. A test-only kinematic sphere then touches every pad
in an isolated area. It is absent from the normal task environment.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=2, choices=(1, 2, 4))
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
public_script = ROOT / "src/contact_rearrangement/scripts/spawn_scene.py"
sys.argv = [str(public_script), "--num_envs", str(args.num_envs), "--seed", str(args.seed),
            "--headless", "--device", "cuda:0", "--max_steps", "151", "--validate"]
scope = runpy.run_path(str(public_script), run_name="tactile_validation_public_script")
public = scope["main"].__globals__
app = scope["simulation_app"]

import numpy as np
import torch
from pxr import PhysxSchema, UsdPhysics
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensor

from contact_rearrangement.assets import TACTILE_BODY_NAMES
from contact_rearrangement.sensors import (
    BinaryTactileSensor, InitialVisionSensor, TACTILE_SENSOR_NAMES, TactileCfg,
    tactile_contacts, tactile_valid,
)


audit = {"steps": 0, "resets": 0, "partial_resets": 0, "renders": 0, "pads": []}


def tactile_snapshot(env, selected=slice(None)):
    sensor = env.binary_tactile_sensor
    return tuple(value[selected].clone() for value in (sensor.contacts, sensor.valid, sensor.net_forces_w))


def assert_snapshot(env, expected, selected=slice(None)):
    for actual, saved in zip(tactile_snapshot(env, selected), expected):
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)


def check_mapping(env):
    assert len(TACTILE_SENSOR_NAMES) == len(set(TACTILE_SENSOR_NAMES)) == 17
    assert len(env.binary_tactile_sensor.sensors) == 17
    robot = env.scene["robot"]
    assert robot.num_bodies == 41 and robot.num_joints == 18
    assert len(env.robot_pose_model.colliders) == 41
    for scene_name, body_name in zip(TACTILE_SENSOR_NAMES, TACTILE_BODY_NAMES):
        sensor = env.scene[scene_name]
        assert isinstance(sensor, ContactSensor) and sensor.num_bodies == 1
        assert sensor.body_names == [body_name]
        expected = [f"{path}/Robot/ROASRightHand/{body_name}" for path in env.scene.env_prim_paths]
        assert list(sensor.body_physx_view.prim_paths) == expected
        for path in expected:
            prim = env.scene.stage.GetPrimAtPath(path)
            assert prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(PhysxSchema.PhysxContactReportAPI)


original_render = InitialVisionSensor._render


def audited_render(self, camera):
    env = self.env
    saved = tactile_snapshot(env)
    time, index = env.sim.current_time, env.sim.current_time_step_index
    result = original_render(self, camera)
    assert env.sim.current_time == time and env.sim.current_time_step_index == index
    tactile_contacts(env)
    tactile_valid(env)
    assert_snapshot(env, saved)
    audit["renders"] += 1
    return result


InitialVisionSensor._render = audited_render
original_env_class = scope["ManagerBasedEnv"]


class AuditedEnv(original_env_class):
    def __init__(self, cfg, *positional, **kwargs):
        cfg.scene.tactile_probe = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TactileProbe",
            spawn=sim_utils.SphereCfg(
                radius=0.003,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 10.0)),
        )
        super().__init__(cfg, *positional, **kwargs)

    def reset(self, seed=None, env_ids=None, options=None):
        selected = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                    else torch.as_tensor(env_ids, dtype=torch.long, device=self.device))
        untouched_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        untouched_mask[selected] = False
        untouched = untouched_mask.nonzero().flatten()
        saved = tactile_snapshot(self, untouched)
        time, index = self.sim.current_time, self.sim.current_time_step_index
        result = super().reset(seed=seed, env_ids=selected, options=options)
        assert self.sim.current_time == time and self.sim.current_time_step_index == index
        assert not bool(tactile_contacts(self)[selected].any())
        assert not bool(tactile_valid(self)[selected].any())
        assert not bool(self.binary_tactile_sensor.net_forces_w[selected].any())
        if untouched.numel():
            assert_snapshot(self, saved, untouched)
            audit["partial_resets"] += 1
        audit["resets"] += 1
        return result

    def step(self, action):
        result = super().step(action)
        obs = result[0]["sensors"]
        contacts, valid = tactile_contacts(self), tactile_valid(self)
        assert contacts.dtype == valid.dtype == torch.bool and contacts.shape == (self.num_envs, 17)
        assert bool(valid.all()) and not bool(contacts.any()), "Collision-free start produced a tactile contact."
        torch.testing.assert_close(obs["tactile"], contacts, atol=0, rtol=0)
        torch.testing.assert_close(obs["tactile_valid"], valid, atol=0, rtol=0)
        assert self.tactile_last_update_time == self.sim.current_time
        audit["steps"] += 1
        return result


public["ManagerBasedEnv"] = AuditedEnv


def verify_all_pads(env):
    robot, probe = env.scene["robot"], env.scene["tactile_probe"]
    sensor = env.binary_tactile_sensor
    ids = torch.arange(env.num_envs, device=env.device)
    body_indices = [robot.body_names.index(name) for name in TACTILE_BODY_NAMES]
    pad_boxes = {box.body_name: box for box in env.robot_pose_model.colliders if box.body_name in TACTILE_BODY_NAMES}
    assert len(pad_boxes) == 17
    joint_positions = robot.data.joint_pos.clone()
    isolated_root = robot.data.root_state_w.clone()
    isolated_root[:, 2] += 4.0
    isolated_root[:, 7:] = 0.0
    parked_probe = probe.data.default_root_state.clone()
    parked_probe[:, :3] += env.scene.env_origins

    def park():
        probe.write_root_state_to_sim(parked_probe)

    def restore_hand():
        robot.write_root_state_to_sim(isolated_root)
        robot.write_joint_state_to_sim(joint_positions, torch.zeros_like(joint_positions))
        robot.set_joint_position_target(joint_positions)
        robot.set_joint_velocity_target(torch.zeros_like(joint_positions))
        env.scene.write_data_to_sim()
        env.sim.forward()

    def advance(count=1):
        for _ in range(count):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
            tactile_contacts(env)

    def touch(channel, selected, penetration):
        body_name = TACTILE_BODY_NAMES[channel]
        box = pad_boxes[body_name]
        poses = robot.root_physx_view.get_link_transforms()[:, body_indices[channel]].cpu().numpy()
        rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
        # The mesh's exposed pad face lies along the thin positive axis.
        thin_axis = int(np.argmin(box.half_extents))
        local_point = box.center_local + box.rotation_local[:, thin_axis] * (
            box.half_extents[thin_axis] + probe.cfg.spawn.radius - penetration
        )
        positions = poses[:, :3] + np.einsum("nij,j->ni", rotations, local_point)
        state = parked_probe[selected].clone()
        state[:, :3] = torch.as_tensor(positions[selected.cpu().numpy()], dtype=torch.float32, device=env.device)
        probe.write_root_state_to_sim(state, env_ids=selected)

    park()
    restore_hand()
    advance(4)
    assert not bool(sensor.contacts.any()), "Gravity/holding torque alone caused contact."
    env.initial_vision_sensor._render(env.scene["initial_camera"])
    for channel, body_name in enumerate(TACTILE_BODY_NAMES):
        activated = False
        for penetration in (0.001, 0.002, 0.003):
            park()
            restore_hand()
            advance(2)
            assert not bool(sensor.contacts.any()), f"Probe separation left a stale contact before {body_name}."
            touch(channel, ids[:1], penetration)
            for _ in range(6):
                advance()
                if sensor.contacts[0, channel]:
                    activated = True
                    force = float(sensor.net_forces_w[0, channel].norm())
                    break
            if activated:
                break
        assert activated, f"Real external probe never activated {body_name}."
        assert bool(sensor.valid.all())
        if env.num_envs > 1:
            assert not bool(sensor.contacts[1:].any()), f"Contact leaked to a different environment: {body_name}."
        actual = torch.stack([s.data.net_forces_w[:, 0] for s in sensor.sensors], dim=1)
        torch.testing.assert_close(sensor.net_forces_w, actual, atol=0, rtol=0)
        torch.testing.assert_close(sensor.contacts, actual.norm(dim=-1) > env.cfg.tactile.force_threshold, atol=0, rtol=0)
        saved = tactile_snapshot(env)
        env.initial_vision_sensor._render(env.scene["initial_camera"])
        assert_snapshot(env, saved)
        print(f"[TACTILE BENCH]: channel={channel:02d} body={body_name} on force={force:.6g}N "
              f"penetration={penetration:.4f}m", flush=True)
        audit["pads"].append(body_name)
        park()
        advance(4)
        assert not bool(sensor.contacts.any()), f"Detached pad did not return off: {body_name}."

    if env.num_envs > 1:
        # Keep a measured ON channel in an untouched environment while resetting
        # env_0; a zero-only comparison would not detect lost contact state.
        park()
        restore_hand()
        advance(2)
        touch(0, ids[1:2], 0.002)
        for _ in range(6):
            advance()
            if sensor.contacts[1, 0]:
                break
        assert sensor.contacts[1, 0], "Untouched-env reset fixture did not establish contact."
        env.reset(env_ids=ids[:1])
        assert sensor.contacts[1, 0] and sensor.valid[1, 0]
        park()
    env.reset()
    scope["validate_reset_state"](env, ids)


original_validate = scope["validate_settling_and_reset"]


def validate_with_tactile_bench(env, visualizer=None):
    original_validate(env, visualizer)
    check_mapping(env)
    verify_all_pads(env)
    assert tuple(audit["pads"]) == TACTILE_BODY_NAMES
    print(f"[TACTILE AUDIT]: PASS {env.num_envs} envs; {audit['steps']} env steps; "
          f"{audit['resets']} resets, {audit['partial_resets']} partial resets, {audit['renders']} renders; "
          "17/17 pads ON with real contacts and OFF after separation; independent body mapping; "
          "reset invalidation/untouched contact preservation; native robot 41 bodies/18 DOFs/41 collider prims",
          flush=True)


public["validate_settling_and_reset"] = validate_with_tactile_bench

try:
    for threshold in (0.0, -0.01, float("nan"), float("inf")):
        try:
            BinaryTactileSensor(SimpleNamespace(cfg=SimpleNamespace(tactile=TactileCfg(force_threshold=threshold))))
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid tactile threshold accepted: {threshold}")
    scope["main"]()
except Exception:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    app.close()
