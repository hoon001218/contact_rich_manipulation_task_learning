"""MDP invariants independent of Kit, remote assets, and GPU availability."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement import mdp
from contact_rearrangement.mdp import observations as mdp_observations
from contact_rearrangement.src.sampling import overlap_area, rectangle_polygon


class GeometryTests(unittest.TestCase):
    def test_tilted_projected_box_matches_independent_convex_hull_and_clipping(self):
        random = np.random.default_rng(71)
        q = Rotation.random(80, random_state=random).as_quat()[:, [3, 0, 1, 2]]
        pose = torch.zeros((80, 7), dtype=torch.float64)
        pose[:, :3] = torch.tensor(random.uniform(-0.1, 0.1, (80, 3)))
        pose[:, 3:] = torch.tensor(q)
        lower = torch.tensor([-.12, -.04, -.15], dtype=torch.float64).expand(80, 3)
        upper = -lower
        upright = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(80, 4)
        _, corners, _ = mdp.object_geometry(pose, lower, upper, upright)
        polygon, count = mdp.convex_hull_xy(corners[..., :2])
        area = mdp.polygon_area(polygon, count)
        region = (-.08, .09, -.07, .06)
        clipped = mdp.rectangle_overlap_area(polygon, count, torch.tensor(region).expand(80, 4))
        for i in range(80):
            reference = ConvexHull(corners[i, :, :2].numpy())
            reference_polygon = tuple(tuple(value) for value in reference.points[reference.vertices])
            self.assertAlmostEqual(float(area[i]), reference.volume, places=9)
            self.assertAlmostEqual(float(clipped[i]), overlap_area(reference_polygon, rectangle_polygon(region)), places=8)

    def test_catalog_upright_rotation_and_off_center_origin_are_applied_once(self):
        upright = torch.tensor([[math.sqrt(.5), -math.sqrt(.5), 0., 0.]])
        pose = torch.cat((torch.tensor([[.4, -.2, 1.05]]), upright), -1)
        lower, upper = torch.tensor([[.01, -.06, 0.]]), torch.tensor([[.11, .04, .20]])
        center, corners, relative = mdp.object_geometry(pose, lower, upper, upright)
        torch.testing.assert_close(center, torch.tensor([[.46, -.21, 1.15]]))
        torch.testing.assert_close(relative, torch.tensor([[1., 0., 0., 0.]]))
        polygon, count = mdp.convex_hull_xy(corners[..., :2])
        self.assertEqual(int(count[0]), 4)
        self.assertAlmostEqual(float(mdp.polygon_area(polygon, count)), .01, places=6)
        self.assertAlmostEqual(float(mdp.rectangle_overlap_area(polygon, count,
                               torch.tensor([[.45, .55, -.5, .5]]))), .006, places=6)

    def test_empty_and_boundary_only_intersections_have_zero_area(self):
        points = torch.tensor([[[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]])
        count = torch.tensor([4])
        for region in ((.5, 1., -.5, .5), (.6, 1., -.5, .5), (2., 3., 2., 3.)):
            result = mdp.rectangle_overlap_area(points, count, torch.tensor([region]))
            self.assertEqual(float(result), 0.)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = mdp.MdpCfg()
        self.active = torch.tensor([[True, True, False]])
        self.velocity = torch.zeros((1, 3, 6))
        self.wrench = torch.zeros((1, 6))

    def components(self, **changes):
        arguments = dict(cfg=self.cfg, blocked=torch.tensor([.01]), blocker_area=torch.tensor([.02]),
                         path_length=torch.zeros(1), velocity=self.velocity, previous_velocity=self.velocity,
                         wrench=self.wrench, previous_wrench=self.wrench, active=self.active, dt=1 / 30)
        arguments.update(changes)
        return mdp.reward_components(**arguments)

    def test_clear_denominator_is_blocker_area_and_zero_clearance_reaches_one(self):
        components, _ = self.components()
        self.assertAlmostEqual(float(components[0, 0]), math.exp(-.5), places=6)
        clear, _ = self.components(blocked=torch.zeros(1))
        self.assertEqual(float(clear[0, 0]), 1.)
        full, _ = self.components(blocked=torch.tensor([.02]))
        self.assertAlmostEqual(float(full[0, 0]), math.exp(-1), places=6)

    def test_velocity_cost_sums_instances_and_padding_cannot_change_reward(self):
        moving = self.velocity.clone()
        moving[:, :2, 0] = self.cfg.linear_velocity_soft + self.cfg.linear_velocity_ref
        _, cost = self.components(velocity=moving, previous_velocity=moving)
        self.assertAlmostEqual(float(cost[0, 0]), 2., places=6)
        moving[:, 2] = 1.0e5
        _, padded = self.components(velocity=moving, previous_velocity=moving)
        torch.testing.assert_close(cost, padded)

    def test_load_is_unsquared_and_four_motion_groups_have_equal_weight(self):
        wrench = self.wrench.clone()
        wrench[0, 0] = self.cfg.force_soft + 2 * self.cfg.force_ref
        components, cost = self.components(wrench=wrench, previous_wrench=wrench)
        self.assertEqual(float(cost[0, 2]), 2.)
        self.assertAlmostEqual(float(components[0, 2]), (3 + math.exp(-2)) / 4, places=6)

    def test_finite_difference_scales_and_time_reference_are_applied_once(self):
        previous = self.wrench.clone()
        current = previous.clone()
        current[0, 0] = 1.
        _, cost30 = self.components(wrench=current, previous_wrench=previous, dt=1 / 30)
        comp60, cost60 = self.components(wrench=current, previous_wrench=previous, dt=1 / 60)
        self.assertAlmostEqual(float(cost60[0, 3] / cost30[0, 3]), 4., places=6)
        self.assertAlmostEqual(float(comp60[0, 3]), -(1 / 60) / self.cfg.time_ref, places=8)

    def test_failure_wins_and_terminal_reward_is_one_shot(self):
        yes, no = torch.tensor([True]), torch.tensor([False])
        count = torch.tensor([self.cfg.stable_steps])
        success, terminal, paid = mdp.terminal_transition(yes, yes, count, no, self.cfg)
        self.assertFalse(bool(success))
        self.assertEqual(float(terminal), -self.cfg.failure_penalty)
        _, repeated, _ = mdp.terminal_transition(yes, yes, count, paid, self.cfg)
        self.assertEqual(float(repeated), 0.)

    def test_policy_padding_is_removed_before_normalization_and_flat_shape_is_fixed(self):
        geometry = torch.zeros((2, 40, 10))
        valid = torch.zeros((2, 40), dtype=torch.bool)
        roles = torch.zeros((2, 40, 2))
        history = torch.zeros((2, 2, 89))
        history_valid = torch.zeros((2, 2), dtype=torch.bool)
        valid[:, 0] = True
        geometry[:, 0] = 1.
        geometry[:, 1:] = float("nan")
        roles[:, 1:] = 9.
        history[:, 1:] = float("nan")
        history_valid[:, 0] = True
        packed = mdp.pack_policy_observation(geometry, valid, roles, history, history_valid)
        self.assertEqual(tuple(packed.shape), (2, 700))
        self.assertTrue(bool(torch.isfinite(packed).all()))
        self.assertEqual(float(packed[:, 10:400].abs().sum()), 0.)

    def test_invalid_scales_and_thresholds_fail_early(self):
        for changed in (dict(displacement_ref=0.), dict(w_clear=-1.), dict(time_ref=float("nan")),
                        dict(force_hard=5.), dict(stable_steps=1.5), dict(history_length=8)):
            with self.assertRaises(ValueError):
                mdp.MdpCfg(**changed)


def make_tensor_env(num_envs=1):
    """Minimal physical-state fixture for testing the actual transition cache."""
    cfg = mdp.MdpCfg(stable_steps=2)
    state = SimpleNamespace()
    state.lower = torch.tensor([[-.04, -.04, 0.]] * 3)
    state.upper = torch.tensor([[.04, .04, .15]] * 3)
    state.upright = torch.tensor([[1., 0., 0., 0.]] * 3)
    state.sensor_id = 0
    state.tcp_id = 0
    state.joint_ids = list(range(12))
    state.last_time = torch.zeros(1, dtype=torch.float64)
    state.history = torch.zeros((1, 2, 89))
    state.history_valid = torch.ones((1, 2), dtype=torch.bool)
    state.previous_center = torch.zeros((1, 40, 3))
    state.initial_center = torch.zeros_like(state.previous_center)
    state.initial_rotation = torch.tensor([[[1., 0., 0., 0.]] * 40])
    state.previous_velocity = torch.zeros((1, 40, 6))
    state.previous_wrench = torch.zeros((1, 6))
    state.wrench_bias = torch.zeros((1, 6))
    state.initial_sensor_point = torch.zeros((1, 3))
    state.path_length = torch.zeros(1)
    state.stable_count = torch.zeros(1, dtype=torch.long)
    state.success = torch.zeros(1, dtype=torch.bool)
    state.failure = torch.zeros_like(state.success)
    state.terminal_paid = torch.zeros_like(state.success)
    state.discrete_reward = torch.zeros(1)
    state.components = torch.zeros((1, 5))
    state.motion_costs = torch.zeros((1, 4))
    state.blocked_ratio = torch.zeros(1)
    state.total_overlap = torch.zeros(1)
    state.failure_reasons = torch.zeros((1, 6), dtype=torch.bool)
    pose = torch.tensor([[[-.75, 0., 1.05, 1., 0., 0., 0.],
                          [-.6, 0., 1.05, 1., 0., 0., 0.],
                          [-.6, .25, 1.05, 1., 0., 0., 0.]]])
    object_data = SimpleNamespace(object_link_pose_w=pose, object_com_pos_w=pose[..., :3].clone(),
                                  object_com_vel_w=torch.zeros((1, 3, 6)))
    objects = SimpleNamespace(data=object_data)
    robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.zeros((1, 1, 3)),
                                                 body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
                                                 root_pos_w=torch.zeros((1, 3)),
                                                 root_quat_w=torch.tensor([[1., 0., 0., 0.]]),
                                                 joint_pos=torch.zeros((1, 12)),
                                                 joint_vel=torch.zeros((1, 12))))

    class Scene(dict):
        env_origins = torch.zeros((1, 3))

    env = SimpleNamespace(num_envs=1, device="cpu", cfg=SimpleNamespace(mdp=cfg), mdp_state=state,
                          sim=SimpleNamespace(current_time=0.), step_dt=1 / 30,
                          scene=Scene(objects=objects, robot=robot),
                          active_object_ids=torch.tensor([[0, 1, 2] + [-1] * 37]),
                          target_object_id=torch.tensor([0]), blocker_object_id=torch.tensor([1]),
                          access_region=torch.tensor([[-.70, -.50, -.06, .06]]),
                          initial_object_geometry=torch.zeros((1, 40, 10)),
                          initial_object_valid=torch.zeros((1, 40), dtype=torch.bool),
                          initial_object_roles=torch.zeros((1, 40, 2)),
                          spawn_rules=SimpleNamespace(x_range=(-.9, -.5), y_range=(-.45, .45),
                                                      shelf_top=1.05, ceiling=1.48),
                          action_manager=SimpleNamespace(action=torch.zeros((1, 18))))
    if num_envs != 1:
        # Duplicate independent scenes while retaining catalog geometry, which
        # has an object axis rather than an environment axis.
        for owner in (state, object_data, robot.data, env, env.action_manager):
            for name, value in list(vars(owner).items()):
                if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                    setattr(owner, name, value.repeat(num_envs, *([1] * (value.ndim - 1))))
        env.scene.env_origins = torch.zeros((num_envs, 3))
        env.num_envs = num_envs
    _, _, center, _, _, _ = mdp._physical_objects(env, state)
    state.previous_center[:] = state.initial_center[:] = center
    return env


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.env = make_tensor_env()
        self.patches = (patch.object(mdp_observations, "corrected_wrench", lambda env: torch.zeros((1, 6))),
                        patch.object(mdp_observations, "_dynamic_observation", lambda env, state, *args: torch.zeros((1, 89))))
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def move(self, object_id, delta):
        self.env.scene["objects"].data.object_link_pose_w[0, object_id, :3] += torch.tensor(delta)
        self.env.sim.current_time += self.env.step_dt

    def test_out_and_back_remains_cumulative_and_duplicate_manager_reads_do_not_add_distance(self):
        self.move(2, (0., .05, 0.))
        reward = mdp.reward_rate(self.env)
        state = mdp.update_mdp(self.env)
        self.assertAlmostEqual(float(state.path_length), .05, places=6)
        torch.testing.assert_close(reward * self.env.step_dt, state.discrete_reward)
        mdp.policy_observation(self.env)
        mdp.success(self.env)
        mdp.failure(self.env)
        self.assertAlmostEqual(float(state.path_length), .05, places=6)
        self.move(2, (0., -.05, 0.))
        self.assertAlmostEqual(float(mdp.update_mdp(self.env).path_length), .10, places=6)

    def test_blocker_clearance_is_insufficient_if_a_distractor_enters_fixed_access_region(self):
        self.move(1, (0., -.20, 0.))
        self.env.scene["objects"].data.object_link_pose_w[0, 2, 1] = 0.
        state = mdp.update_mdp(self.env)
        self.assertEqual(float(state.blocked_ratio), 0.)
        self.assertGreater(float(state.total_overlap), 0.)
        self.assertFalse(bool(state.success))

    def test_target_motion_does_not_move_access_region_or_qualify_as_success(self):
        original = self.env.access_region.clone()
        self.move(1, (0., -.20, 0.))
        self.env.scene["objects"].data.object_link_pose_w[0, 0, 1] += .05
        mdp.update_mdp(self.env)
        self.env.sim.current_time += self.env.step_dt
        state = mdp.update_mdp(self.env)
        self.assertEqual(float(state.total_overlap), 0.)
        self.assertFalse(bool(state.success))
        torch.testing.assert_close(self.env.access_region, original)

    def test_success_requires_stable_steps_then_bonus_is_paid_only_once(self):
        self.move(1, (0., -.20, 0.))
        first = mdp.update_mdp(self.env)
        self.assertFalse(bool(first.success))
        self.env.sim.current_time += self.env.step_dt
        second = mdp.update_mdp(self.env)
        self.assertTrue(bool(second.success))
        self.assertEqual(float(second.components[0, 4]), self.env.cfg.mdp.success_reward)
        self.env.sim.current_time += self.env.step_dt
        third = mdp.update_mdp(self.env)
        self.assertEqual(float(third.components[0, 4]), 0.)

    def test_live_object_ground_truth_cannot_enter_actor_observation(self):
        before = mdp.policy_observation(self.env)
        self.move(1, (0., -.20, 0.))
        after = mdp.policy_observation(self.env)
        torch.testing.assert_close(before, after, atol=0., rtol=0.)
        self.assertGreater(float(self.env.mdp_state.path_length), 0.)

    def test_dropped_active_object_is_a_failure_without_planar_displacement_cost(self):
        self.move(2, (0., 0., -.1))
        state = mdp.update_mdp(self.env)
        self.assertTrue(bool(state.failure))
        self.assertTrue(bool(state.failure_reasons[0, 0]))
        self.assertEqual(float(state.path_length), 0.)

    def test_unused_pool_body_is_excluded_from_geometry_motion_and_failure(self):
        env, state = self.env, self.env.mdp_state
        data = env.scene["objects"].data
        data.object_link_pose_w = torch.cat((data.object_link_pose_w,
                                           torch.tensor([[[20., 20., -5., 1., 0., 0., 0.]]])), 1)
        data.object_com_pos_w = data.object_link_pose_w[..., :3]
        data.object_com_vel_w = torch.cat((data.object_com_vel_w, torch.full((1, 1, 6), 1.0e5)), 1)
        state.lower = torch.cat((state.lower, state.lower[:1]), 0)
        state.upper = torch.cat((state.upper, state.upper[:1]), 0)
        state.upright = torch.cat((state.upright, state.upright[:1]), 0)
        env.sim.current_time += env.step_dt
        mdp.update_mdp(env)
        self.assertFalse(bool(state.failure))
        self.assertEqual(float(state.path_length), 0.)
        self.assertEqual(float(state.motion_costs[:, :2].sum()), 0.)

    def test_initial_geometry_is_expressed_in_current_eef_with_nonidentity_root_and_tcp(self):
        env, state = self.env, self.env.mdp_state
        root_rotation = Rotation.from_euler("xyz", [.4, -.1, .8])
        tcp_rotation_b = Rotation.from_euler("xyz", [.1, .6, -.7])
        object_rotation_b = Rotation.from_euler("xyz", [-.2, .3, 1.1])
        root_position = np.array([2., 3., .4])
        tcp_position_b = np.array([.2, -.15, .8])
        object_position_b = np.array([.5, .1, .95])
        robot = env.scene["robot"].data
        robot.root_pos_w = torch.tensor(root_position[None])
        robot.root_quat_w = torch.tensor(root_rotation.as_quat()[[3, 0, 1, 2]][None])
        robot.body_link_pos_w = torch.tensor((root_position + root_rotation.apply(tcp_position_b))[None, None])
        robot.body_link_quat_w = torch.tensor((root_rotation * tcp_rotation_b).as_quat()[[3, 0, 1, 2]][None, None])
        env.initial_object_geometry = torch.zeros((1, 40, 10), dtype=torch.float64)
        env.initial_object_geometry[0, 0] = torch.tensor(np.r_[object_position_b,
                                                       object_rotation_b.as_quat()[[3, 0, 1, 2]], [.07, .06, .2]])
        env.initial_object_geometry[0, 1:] = float("nan")
        env.initial_object_valid[0, 0] = True
        original = env.initial_object_geometry.clone()
        observed = mdp.policy_observation(env)[0, :400].reshape(40, 10)
        expected_position = tcp_rotation_b.inv().apply(object_position_b - tcp_position_b)
        expected_orientation = (tcp_rotation_b.inv() * object_rotation_b).as_quat()[[3, 0, 1, 2]]
        if expected_orientation[0] < 0:
            expected_orientation *= -1
        np.testing.assert_allclose(observed[0, :3], expected_position, atol=1.0e-10)
        np.testing.assert_allclose(observed[0, 3:7], expected_orientation, atol=1.0e-10)
        torch.testing.assert_close(observed[0, 7:], original[0, 0, 7:])
        self.assertEqual(float(observed[1:].abs().sum()), 0.)
        torch.testing.assert_close(env.initial_object_geometry, original, equal_nan=True)
        robot.body_link_pos_w += torch.tensor([[[.03, 0., 0.]]], dtype=torch.float64)
        moved = mdp.policy_observation(env)[0, :10]
        self.assertGreater(float(torch.linalg.vector_norm(moved[:3] - observed[0, :3])), .029)
        torch.testing.assert_close(env.initial_object_geometry, original, equal_nan=True)


class PreviousStepTests(unittest.TestCase):
    def setUp(self):
        self.env = make_tensor_env(num_envs=2)
        contacts = torch.zeros((2, 17), dtype=torch.bool)
        sensors = SimpleNamespace(tactile_contacts=lambda env: contacts,
                                  tactile_valid=lambda env: torch.ones_like(contacts))
        self.patches = (patch.dict(sys.modules, {"contact_rearrangement.src.sensors": sensors}),
                        patch.object(mdp_observations, "corrected_wrench", lambda env: torch.zeros((2, 6))))
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        mdp.reset_mdp(self.env)

    def advance(self, joint_position, previous_action):
        self.env.scene["robot"].data.joint_pos[:, 0] = joint_position
        self.env.action_manager.action[:, 0] = previous_action
        self.env.sim.current_time += self.env.step_dt
        return mdp.update_mdp(self.env)

    def test_only_current_and_previous_samples_survive_and_actions_match_their_transitions(self):
        self.advance(.1, .2)
        self.advance(.3, .4)
        state = self.advance(.5, .6)
        expected_q = torch.tensor([.5, .3]) / self.env.cfg.mdp.joint_position_scale
        torch.testing.assert_close(state.history[:, :, 0], expected_q.expand(2, -1))
        # At decision t the newest sample carries a[t-1]; the previous
        # sample carries a[t-2]. The t-2 state itself has been discarded.
        torch.testing.assert_close(state.history[:, :, -18], torch.tensor([[.6, .4], [.6, .4]]))
        self.assertEqual(tuple(state.history.shape), (2, 2, 89))
        self.assertTrue(bool(state.history_valid.all()))
        self.assertEqual(tuple(mdp.policy_observation(self.env).shape), (2, 700))

    def test_repeated_manager_reads_do_not_shift_previous_sample(self):
        state = self.advance(.1, .2)
        original_history, original_valid = state.history.clone(), state.history_valid.clone()
        for _ in range(3):
            mdp.policy_observation(self.env)
            mdp.reward_rate(self.env)
            mdp.success(self.env)
            mdp.failure(self.env)
        torch.testing.assert_close(state.history, original_history)
        torch.testing.assert_close(state.history_valid, original_valid)

    def test_full_reset_invalidates_previous_sample_and_zeroes_stale_action(self):
        self.advance(.1, .2)
        state = self.advance(.3, .4)
        self.env.action_manager.action[:] = .9
        mdp.reset_mdp(self.env)
        torch.testing.assert_close(state.history_valid, torch.tensor([[True, False], [True, False]]))
        torch.testing.assert_close(state.history[:, 1], torch.zeros((2, 89)))
        torch.testing.assert_close(state.history[:, 0, -18:], torch.zeros((2, 18)))
        original_history = state.history.clone()
        mdp.policy_observation(self.env)
        torch.testing.assert_close(state.history, original_history)
        self.advance(.5, .6)
        torch.testing.assert_close(state.history[:, 1, -18:], torch.zeros((2, 18)))
        torch.testing.assert_close(state.history[:, 1, 0], torch.full((2,), .3 / math.pi))

    def test_partial_reset_clears_only_selected_episode_and_retains_other_previous_sample(self):
        self.advance(.1, .2)
        state = self.advance(.3, .4)
        continuing_history = state.history[1].clone()
        continuing_valid = state.history_valid[1].clone()
        self.env.action_manager.action[:] = .9
        mdp.reset_mdp(self.env, torch.tensor([0]))
        torch.testing.assert_close(state.history_valid[0], torch.tensor([True, False]))
        torch.testing.assert_close(state.history[0, 1], torch.zeros(89))
        torch.testing.assert_close(state.history[0, 0, -18:], torch.zeros(18))
        torch.testing.assert_close(state.history[1], continuing_history)
        torch.testing.assert_close(state.history_valid[1], continuing_valid)
        self.advance(.5, .6)
        torch.testing.assert_close(state.history[0, 1, -18:], torch.zeros(18))
        torch.testing.assert_close(state.history[1, 1], continuing_history[0])


class WrenchTests(unittest.TestCase):
    def test_pre_reset_sensor_mass_includes_disambiguated_hand_base(self):
        env = make_tensor_env()
        del env.mdp_state
        ur_names = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link",
                    "wrist_1_link", "wrist_2_link", "wrist_3_link"]
        hand_names = ["base_link_1", "right_plam_force_sensor"] + [f"right_link_{i}" for i in range(30)]
        names = ur_names + ["SensorMountProximal", "ToolForceTorqueLink"] + hand_names
        paths = ([f"/World/Robot/{name}" for name in ur_names]
                 + ["/World/Robot/SensorMountProximal", "/World/Robot/ToolForceTorqueLink"]
                 + ["/World/Robot/ROASRightHand/base_link"]
                 + [f"/World/Robot/ROASRightHand/{name}" for name in hand_names[1:]])
        masses = torch.ones((1, len(names)))
        masses[0, names.index("base_link_1")] = .09
        robot = env.scene["robot"]
        robot.body_names = names
        robot.joint_names = [f"joint_{i}" for i in range(12)]
        robot.root_physx_view = SimpleNamespace(link_paths=[paths], get_masses=lambda: masses)
        assets = SimpleNamespace(ARM_JOINT_NAMES=robot.joint_names[:6], HAND_JOINT_NAMES=robot.joint_names[6:],
                                 FORCE_TORQUE_BODY_NAME="ToolForceTorqueLink", TCP_BODY_NAME="right_plam_force_sensor")
        with patch.dict(sys.modules, {"contact_rearrangement.src.assets": assets}):
            state = mdp.MdpState(env)
        self.assertEqual(len(state.downstream_ids), 33)
        self.assertIn(names.index("base_link_1"), state.downstream_ids)
        self.assertNotIn(names.index("base_link"), state.downstream_ids)
        self.assertNotIn(names.index("SensorMountProximal"), state.downstream_ids)
        self.assertAlmostEqual(float(state.masses.sum()), 32.09, places=5)

    def test_invalid_raw_wrench_stays_zero_and_static_gravity_is_compensated_once(self):
        env = make_tensor_env()
        state, robot = env.mdp_state, env.scene["robot"].data
        state.downstream_ids, state.masses = [0], torch.tensor([[1.]])
        state.wrench_bias = torch.zeros((1, 6))
        robot.body_com_pos_w = torch.tensor([[[.1, 0., 0.]]])
        env.sim.cfg = SimpleNamespace(gravity=(0., 0., -9.81))
        raw = torch.zeros((1, 6))
        valid = torch.zeros((1, 1), dtype=torch.bool)
        sensors = SimpleNamespace(tool_wrench=lambda env: raw, tool_wrench_valid=lambda env: valid)
        with patch.dict(sys.modules, {"contact_rearrangement.src.sensors": sensors}):
            torch.testing.assert_close(mdp.corrected_wrench(env), torch.zeros_like(raw))
            valid[:] = True
            raw[:] = torch.tensor([[0., 0., 9.81, 0., -.981, 0.]])
            torch.testing.assert_close(mdp.corrected_wrench(env), torch.zeros_like(raw), atol=1.0e-6, rtol=0.)
            state.wrench_bias[0, 0] = .3
            raw[0, 0] += .3
            torch.testing.assert_close(mdp.corrected_wrench(env), torch.zeros_like(raw), atol=1.0e-6, rtol=0.)

    def test_wrench_rate_uses_same_world_axes_and_initial_sensor_moment_point(self):
        env = make_tensor_env()
        state, robot = env.mdp_state, env.scene["robot"].data
        robot.body_link_pos_w[0, 0] = torch.tensor([.2, 0., 0.])
        robot.body_link_quat_w[0, 0] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        sensor = torch.tensor([[1., 0., 0., 0., 0., 0.]])
        fixed = mdp._fixed_wrench(env, state, sensor)
        torch.testing.assert_close(fixed, torch.tensor([[0., 1., 0., 0., 0., .2]]), atol=1.0e-6, rtol=0.)
        # Same physical force, expressed after rotating the sensor axes again.
        robot.body_link_quat_w[0, 0] = torch.tensor([1., 0., 0., 0.])
        sensor[:] = torch.tensor([[0., 1., 0., 0., 0., 0.]])
        torch.testing.assert_close(mdp._fixed_wrench(env, state, sensor), fixed, atol=1.0e-6, rtol=0.)


if __name__ == "__main__":
    unittest.main()
