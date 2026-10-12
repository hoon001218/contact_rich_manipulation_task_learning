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
        self.joint_velocity = torch.zeros((1, 12))
        self.wrench = torch.zeros((1, 6))

    def components(self, **changes):
        arguments = dict(cfg=self.cfg, blocked=torch.tensor([.01]), blocker_area=torch.tensor([.02]),
                         path_length=torch.zeros(1), joint_velocity=self.joint_velocity,
                         previous_joint_velocity=self.joint_velocity,
                         wrench=self.wrench, previous_wrench=self.wrench, dt=1 / 30)
        arguments.update(changes)
        return mdp.reward_components(**arguments)

    def test_clear_denominator_is_blocker_area_and_zero_clearance_reaches_one(self):
        components, _ = self.components()
        self.assertAlmostEqual(float(components[0, 0]), math.exp(-.5), places=6)
        clear, _ = self.components(blocked=torch.zeros(1))
        self.assertEqual(float(clear[0, 0]), 1.)
        full, _ = self.components(blocked=torch.tensor([.02]))
        self.assertAlmostEqual(float(full[0, 0]), math.exp(-1), places=6)

    def test_large_constant_joint_speed_and_load_do_not_reduce_motion_reward(self):
        stationary, stationary_costs = self.components()
        moving = torch.arange(1, 13, dtype=torch.float32)[None] * 100
        wrench = torch.tensor([[1.e5, -2.e5, 3.e5, 4.e3, -5.e3, 6.e3]])
        for velocity in (moving, -moving):
            components, costs = self.components(joint_velocity=velocity, previous_joint_velocity=velocity,
                                                wrench=wrench, previous_wrench=wrench)
            torch.testing.assert_close(costs, stationary_costs)
            torch.testing.assert_close(components[:, 2], stationary[:, 2])
            self.assertEqual(float(components[0, 2]), 1.)

    def test_acceleration_is_independent_of_constant_joint_velocity_offset(self):
        current = self.joint_velocity.clone()
        current[:, 0] = .5
        current[:, 6] = -.25
        _, initial_cost = self.components(joint_velocity=current)
        offset = torch.arange(1, 13, dtype=torch.float32)[None]
        _, shifted_cost = self.components(joint_velocity=current + offset, previous_joint_velocity=offset)
        _, opposite_cost = self.components(joint_velocity=-current)
        torch.testing.assert_close(initial_cost, shifted_cost)
        torch.testing.assert_close(initial_cost, opposite_cost)

    def test_joint_acceleration_uses_group_refs_and_control_dt(self):
        moving = self.joint_velocity.clone()
        moving[:, 0] = self.cfg.arm_joint_acceleration_ref / 30
        moving[:, 6] = -2 * self.cfg.hand_joint_acceleration_ref / 30
        _, cost30 = self.components(joint_velocity=moving, dt=1 / 30)
        _, cost60 = self.components(joint_velocity=moving, dt=1 / 60)
        self.assertAlmostEqual(float(cost30[0, 0]), 5., places=5)
        self.assertAlmostEqual(float(cost60[0, 0] / cost30[0, 0]), 4., places=6)

    def test_motion_rejects_mimic_joint_duplicates_or_mismatched_previous_sample(self):
        with self.assertRaisesRegex(ValueError, "12 independent joints"):
            self.components(joint_velocity=torch.zeros((1, 18)), previous_joint_velocity=torch.zeros((1, 18)))
        with self.assertRaisesRegex(ValueError, "12 independent joints"):
            self.components(previous_joint_velocity=torch.zeros((1, 6)))

    def test_acceleration_and_wrench_rate_have_equal_weight_in_motion_reward(self):
        moving = self.joint_velocity.clone()
        moving[:, 0] = self.cfg.arm_joint_acceleration_ref / 30
        wrench = self.wrench.clone()
        wrench[:, 0] = 2 * self.cfg.force_rate_ref / 30
        components, costs = self.components(joint_velocity=moving, wrench=wrench)
        torch.testing.assert_close(costs, torch.tensor([[1., 4.]]))
        self.assertAlmostEqual(float(components[0, 2]), (math.exp(-1) + math.exp(-4)) / 2, places=6)

    def test_finite_difference_scales_and_time_reference_are_applied_once(self):
        previous = self.wrench.clone()
        current = previous.clone()
        current[0, 0] = 1.
        _, cost30 = self.components(wrench=current, previous_wrench=previous, dt=1 / 30)
        comp60, cost60 = self.components(wrench=current, previous_wrench=previous, dt=1 / 60)
        self.assertAlmostEqual(float(cost60[0, 1] / cost30[0, 1]), 4., places=6)
        self.assertAlmostEqual(float(comp60[0, 3]), -(1 / 60) / self.cfg.time_ref, places=8)

    def test_force_and_torque_increase_use_norms_and_independent_rate_refs(self):
        previous = torch.tensor([[-3., 0., 0., 0., -1., 0.]])
        current = previous.clone()
        current[:, 0] -= self.cfg.force_rate_ref / 30
        current[:, 4] -= 2 * self.cfg.torque_rate_ref / 30
        _, costs30 = self.components(wrench=current, previous_wrench=previous)
        _, costs60 = self.components(wrench=current, previous_wrench=previous, dt=1 / 60)
        self.assertAlmostEqual(float(costs30[0, 1]), 5., places=5)
        self.assertAlmostEqual(float(costs60[0, 1] / costs30[0, 1]), 4., places=6)
        # Increasing a negative reaction force means increasing its magnitude.
        _, positive_costs = self.components(wrench=-current, previous_wrench=-previous)
        torch.testing.assert_close(costs30, positive_costs)

    def test_decreasing_load_and_equal_magnitude_direction_changes_have_no_motion_cost(self):
        previous = torch.tensor([[20., 0., 0., 0., 2., 0.]])
        for current in (previous * .2, -previous, torch.tensor([[0., 20., 0., 2., 0., 0.]])):
            components, costs = self.components(wrench=current, previous_wrench=previous)
            self.assertEqual(float(costs.sum()), 0.)
            self.assertEqual(float(components[0, 2]), 1.)

    def test_load_increase_cost_is_independent_of_constant_baseline_magnitude(self):
        previous = torch.tensor([[2., 0., 0., 1., 0., 0.]])
        increase = torch.tensor([[1., 0., 0., .25, 0., 0.]])
        _, initial_costs = self.components(wrench=previous + increase, previous_wrench=previous)
        offset = torch.tensor([[40., 0., 0., 2., 0., 0.]])
        _, shifted_costs = self.components(wrench=previous + offset + increase, previous_wrench=previous + offset)
        torch.testing.assert_close(initial_costs, shifted_costs)

    def test_failure_wins_and_terminal_reward_is_one_shot(self):
        yes, no = torch.tensor([True]), torch.tensor([False])
        count = torch.tensor([self.cfg.stable_steps])
        success, terminal, paid = mdp.terminal_transition(yes, yes, count, no, self.cfg)
        self.assertFalse(bool(success))
        self.assertEqual(float(terminal), -self.cfg.failure_penalty)
        _, repeated, _ = mdp.terminal_transition(yes, yes, count, paid, self.cfg)
        self.assertEqual(float(repeated), 0.)

    def test_policy_packing_contains_only_geometry_and_sensor_history(self):
        geometry = torch.zeros((2, 40, 10))
        valid = torch.zeros((2, 40), dtype=torch.bool)
        roles = torch.zeros((2, 40, 2))
        history = torch.arange(144, dtype=torch.float32).reshape(1, 2, 72).repeat(2, 1, 1)
        history_valid = torch.zeros((2, 2), dtype=torch.bool)
        valid[:, :3] = True
        geometry[:, :3] = 1.
        geometry[:, 3:] = float("nan")
        roles[:, 0, 0] = 1.
        roles[:, 1, 1] = 1.
        roles[:, 3:] = 9.
        history[:, 1:] = float("nan")
        history_valid[:, 0] = True
        packed = mdp.pack_policy_observation(geometry, valid, roles, history, history_valid)
        self.assertEqual(tuple(packed.shape), (2, 544))
        self.assertTrue(bool(torch.isfinite(packed).all()))
        self.assertEqual(float(packed[:, 30:400].abs().sum()), 0.)
        torch.testing.assert_close(packed[:, 400:472], history[:, 0])
        torch.testing.assert_close(packed[:, 472:], torch.zeros((2, 72)))

    def test_invalid_scales_and_thresholds_fail_early(self):
        for changed in (dict(displacement_ref=0.), dict(w_clear=-1.), dict(time_ref=float("nan")),
                        dict(force_hard=0.), dict(torque_hard=-1.), dict(stable_steps=1.5), dict(history_length=8),
                        dict(arm_joint_acceleration_ref=0.), dict(hand_joint_acceleration_ref=-1.),
                        dict(force_rate_ref=0.), dict(torque_rate_ref=0.)):
            with self.assertRaises(ValueError):
                mdp.MdpCfg(**changed)

    def test_hard_force_and_torque_limits_have_no_reward_soft_threshold_dependency(self):
        cfg = mdp.MdpCfg(force_hard=5., torque_hard=.2)
        self.assertEqual(cfg.force_hard, 5.)
        self.assertEqual(cfg.torque_hard, .2)


class ObservationPackingTests(unittest.TestCase):
    def setUp(self):
        self.geometry = torch.zeros((2, 40, 10))
        self.geometry[..., 0] = torch.arange(40) + 10
        self.geometry[..., 3] = 1.
        self.geometry[..., 7:] = .1
        self.valid = torch.zeros((2, 40), dtype=torch.bool)
        self.valid[0, [1, 3, 8, 12, 17]] = True
        self.valid[1, [0, 4, 7, 11, 16]] = True
        self.roles = torch.zeros((2, 40, 2))
        self.roles[0, 8, 0] = self.roles[0, 3, 1] = 1.
        self.roles[1, 4, 0] = self.roles[1, 16, 1] = 1.
        self.history = torch.randn((2, 2, 72))
        self.history_valid = torch.ones((2, 2), dtype=torch.bool)

    def pack(self):
        return mdp.pack_policy_observation(self.geometry, self.valid, self.roles,
                                           self.history, self.history_valid)

    def test_roles_move_to_reserved_slots_and_distractors_retain_relative_order(self):
        packed = self.pack()
        objects = packed[:, :400].reshape(2, 40, 10)
        for env_id, canonical_indices in enumerate(([8, 3, 1, 12, 17], [4, 16, 0, 7, 11])):
            torch.testing.assert_close(objects[env_id, :5], self.geometry[env_id, canonical_indices])
        torch.testing.assert_close(objects[:, 5:], torch.zeros((2, 35, 10)))
        torch.testing.assert_close(packed[:, 400:], self.history.flatten(1))
        self.assertEqual(tuple(packed.shape), (2, 544))

    def test_unobserved_target_or_blocker_keeps_its_reserved_slot_empty(self):
        self.valid[0, 8] = False
        self.valid[1, 16] = False
        self.geometry[0, 8] = self.geometry[1, 16] = float("nan")
        objects = self.pack()[:, :400].reshape(2, 40, 10)
        torch.testing.assert_close(objects[0, 0], torch.zeros(10))
        torch.testing.assert_close(objects[1, 1], torch.zeros(10))
        torch.testing.assert_close(objects[0, 1], self.geometry[0, 3])
        torch.testing.assert_close(objects[1, 0], self.geometry[1, 4])
        torch.testing.assert_close(objects[0, 2:5], self.geometry[0, [1, 12, 17]])
        torch.testing.assert_close(objects[1, 2:5], self.geometry[1, [0, 7, 11]])
        self.assertTrue(bool(torch.isfinite(objects).all()))

    def test_no_role_association_does_not_promote_distractors(self):
        self.roles.zero_()
        objects = self.pack()[:, :400].reshape(2, 40, 10)
        torch.testing.assert_close(objects[:, :2], torch.zeros((2, 2, 10)))
        torch.testing.assert_close(objects[0, 2:7], self.geometry[0, self.valid[0]])
        torch.testing.assert_close(objects[1, 2:7], self.geometry[1, self.valid[1]])
        torch.testing.assert_close(objects[:, 7:], torch.zeros((2, 33, 10)))

    def test_packing_leaves_initial_geometry_roles_and_history_unchanged(self):
        inputs = (self.geometry, self.valid, self.roles, self.history, self.history_valid)
        originals = tuple(tensor.clone() for tensor in inputs)
        self.pack()
        for tensor, original in zip(inputs, originals):
            torch.testing.assert_close(tensor, original)

    def test_ambiguous_observed_roles_raise_instead_of_silently_reassigning_objects(self):
        for changed in ((0, 1, 0, 1.), (0, 8, 1, 1.), (1, 7, 0, .5), (1, 7, 1, float("nan"))):
            with self.subTest(changed=changed):
                original = self.roles.clone()
                env_id, slot, role, value = changed
                self.roles[env_id, slot, role] = value
                with self.assertRaisesRegex(ValueError, "unique target and blocker"):
                    self.pack()
                self.roles.copy_(original)

    def test_capacity_cannot_silently_drop_observed_distractors(self):
        self.valid[:] = True
        self.roles.zero_()
        with self.assertRaisesRegex(ValueError, "reserve two slots"):
            self.pack()


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
    state.history = torch.zeros((1, 2, 72))
    state.history_valid = torch.ones((1, 2), dtype=torch.bool)
    state.previous_center = torch.zeros((1, 40, 3))
    state.initial_center = torch.zeros_like(state.previous_center)
    state.initial_rotation = torch.tensor([[[1., 0., 0., 0.]] * 40])
    state.previous_joint_velocity = torch.zeros((1, 12))
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
    state.motion_costs = torch.zeros((1, 2))
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
                        patch.object(mdp_observations, "_dynamic_observation", lambda env, state, *args: torch.zeros((1, 72))))
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

    def test_static_objects_do_not_hide_robot_joint_motion_or_repeat_derivative_update(self):
        env = self.env
        env.scene["robot"].data.joint_vel[0, 0] = 1.5
        env.sim.current_time += env.step_dt
        state = mdp.update_mdp(env)
        self.assertEqual(float(state.path_length), 0.)
        self.assertAlmostEqual(float(state.motion_costs[0, 0]), 81., delta=1.0e-4)
        self.assertEqual(float(state.motion_costs[0, 1]), 0.)
        costs = state.motion_costs.clone()
        previous_joint_velocity = state.previous_joint_velocity.clone()
        for _ in range(3):
            mdp.policy_observation(env)
            mdp.reward_rate(env)
            mdp.success(env)
            mdp.failure(env)
        torch.testing.assert_close(state.motion_costs, costs)
        torch.testing.assert_close(state.previous_joint_velocity, previous_joint_velocity)
        env.sim.current_time += env.step_dt
        state = mdp.update_mdp(env)
        self.assertEqual(float(state.motion_costs.sum()), 0.)
        self.assertEqual(float(state.components[0, 2]), 1.)

    def test_constant_large_load_still_triggers_hard_force_or_torque_termination(self):
        for channel, limit in ((0, self.env.cfg.mdp.force_hard), (3, self.env.cfg.mdp.torque_hard)):
            with self.subTest(channel=channel):
                env = make_tensor_env()
                wrench = torch.zeros((1, 6))
                wrench[0, channel] = limit + 1
                env.mdp_state.previous_wrench[:] = wrench
                env.sim.current_time += env.step_dt
                with patch.object(mdp_observations, "corrected_wrench", lambda env: wrench):
                    state = mdp.update_mdp(env)
                self.assertEqual(float(state.motion_costs.sum()), 0.)
                self.assertEqual(float(state.components[0, 2]), 1.)
                self.assertTrue(bool(state.failure))
                self.assertTrue(bool(state.failure_reasons[0, 4]))
                self.assertEqual(float(state.components[0, 4]), -env.cfg.mdp.failure_penalty)

    def test_object_motion_with_robot_still_changes_rearrangement_but_not_robot_motion_costs(self):
        initial_motion = torch.ones(1)
        self.env.scene["objects"].data.object_com_vel_w[0, 2] = torch.tensor([2., 0., 0., 0., 0., 3.])
        self.move(2, (0., .05, 0.))
        state = mdp.update_mdp(self.env)
        self.assertLess(float(state.components[0, 1]), 1.)
        torch.testing.assert_close(state.components[:, 2], initial_motion)
        self.assertEqual(float(state.motion_costs[:, :2].sum()), 0.)
        # Moving physical objects still cannot satisfy the stability criterion.
        self.assertEqual(int(state.stable_count), 0)

    def test_mimic_joint_velocities_are_not_counted_as_independent_robot_motion(self):
        robot = self.env.scene["robot"].data
        robot.joint_vel = torch.cat((robot.joint_vel, torch.full((1, 6), 1.0e5)), -1)
        self.env.sim.current_time += self.env.step_dt
        state = mdp.update_mdp(self.env)
        self.assertEqual(float(state.motion_costs[:, :2].sum()), 0.)
        torch.testing.assert_close(state.previous_joint_velocity, torch.zeros((1, 12)))

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
        env.initial_object_roles[0, 0, 0] = 1.
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
        self.contacts = torch.zeros((2, 17), dtype=torch.bool)
        self.sensor_valid = torch.ones_like(self.contacts)
        sensors = SimpleNamespace(tactile_contacts=lambda env: self.contacts,
                                  tactile_valid=lambda env: self.sensor_valid)
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
        self.assertEqual(tuple(state.history.shape), (2, 2, 72))
        self.assertTrue(bool(state.history_valid.all()))
        self.assertEqual(tuple(mdp.policy_observation(self.env).shape), (2, 544))

    def test_tactile_readings_distinguish_contact_absence_from_unavailable_sensor(self):
        self.contacts[:, 0] = True
        self.contacts[:, 2] = True  # A stale ON reading must be ignored.
        self.sensor_valid[:, 2] = False
        state = self.advance(.1, .2)
        tactile = state.history[:, 0, 37:54]
        torch.testing.assert_close(tactile[:, :3], torch.tensor([[1., 0., -1.], [1., 0., -1.]]))
        torch.testing.assert_close(tactile[:, 3:], torch.zeros((2, 14)))
        packed = mdp.policy_observation(self.env)
        torch.testing.assert_close(packed[:, 437:454], tactile)
        self.assertEqual(tuple(packed.shape), (2, 544))

    def test_reset_padding_and_missing_tactile_are_identifiable_from_observation_data(self):
        state = self.env.mdp_state
        torch.testing.assert_close(state.history[:, 0, 27:31], torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
        torch.testing.assert_close(state.history[:, 0, 37:54], torch.full((2, 17), -1.))
        torch.testing.assert_close(state.history[:, 1], torch.zeros((2, 72)))
        packed = mdp.policy_observation(self.env)
        samples = packed[:, 400:].reshape(2, 2, 72)
        observed_valid = torch.linalg.vector_norm(samples[:, :, 27:31], dim=-1) > 0
        torch.testing.assert_close(observed_valid, torch.tensor([[True, False], [True, False]]))
        self.assertFalse(bool(((samples[:, 0, 37:54] == 0) | (samples[:, 0, 37:54] == 1)).any()))

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
        torch.testing.assert_close(state.history[:, 1], torch.zeros((2, 72)))
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
        torch.testing.assert_close(state.history[0, 1], torch.zeros(72))
        torch.testing.assert_close(state.history[0, 0, -18:], torch.zeros(18))
        torch.testing.assert_close(state.history[1], continuing_history)
        torch.testing.assert_close(state.history_valid[1], continuing_valid)
        self.advance(.5, .6)
        torch.testing.assert_close(state.history[0, 1, -18:], torch.zeros(18))
        torch.testing.assert_close(state.history[1, 1], continuing_history[0])

    def test_partial_reset_initializes_measured_joint_velocity_without_cross_episode_acceleration(self):
        robot = self.env.scene["robot"].data
        robot.joint_vel[0] = 1.
        robot.joint_vel[1] = 2.
        state = self.advance(.1, .2)
        robot.joint_vel[0] = 3.
        robot.joint_vel[1] = 4.
        mdp.reset_mdp(self.env, torch.tensor([0]))
        torch.testing.assert_close(state.previous_joint_velocity[0], torch.full((12,), 3.))
        torch.testing.assert_close(state.previous_joint_velocity[1], torch.full((12,), 2.))
        state = self.advance(.3, .4)
        self.assertEqual(float(state.motion_costs[0, 0]), 0.)
        expected_continuing = 6 * (60 / self.env.cfg.mdp.arm_joint_acceleration_ref) ** 2
        expected_continuing += 6 * (60 / self.env.cfg.mdp.hand_joint_acceleration_ref) ** 2
        self.assertAlmostEqual(float(state.motion_costs[1, 0]), expected_continuing, delta=expected_continuing * 1.0e-6)
        torch.testing.assert_close(state.previous_joint_velocity[0], torch.full((12,), 3.))
        torch.testing.assert_close(state.previous_joint_velocity[1], torch.full((12,), 4.))

    def test_partial_reset_preserves_continuing_roles_and_reorders_new_episode_roles(self):
        env = self.env
        env.initial_object_geometry[..., 0] = torch.arange(40) + 1
        env.initial_object_geometry[..., 3] = 1.
        env.initial_object_geometry[..., 7:] = .1
        env.initial_object_valid[:, [2, 7, 10]] = True
        env.initial_object_roles[:, 7, 0] = 1.
        env.initial_object_roles[:, 2, 1] = 1.
        self.advance(.1, .2)
        self.advance(.3, .4)
        continuing = mdp.policy_observation(env)[1].clone()
        # A new initial camera estimate may associate other canonical slots.
        env.initial_object_roles[0].zero_()
        env.initial_object_roles[0, 10, 0] = 1.
        env.initial_object_roles[0, 7, 1] = 1.
        mdp.reset_mdp(env, torch.tensor([0]))
        packed = mdp.policy_observation(env)
        torch.testing.assert_close(packed[0, :30].reshape(3, 10), env.initial_object_geometry[0, [10, 7, 2]])
        torch.testing.assert_close(packed[1], continuing)


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
        inertias = torch.diag(torch.tensor([.2, .3, .5])).reshape(1, 1, 9).repeat(1, len(names), 1)
        robot = env.scene["robot"]
        robot.body_names = names
        robot.joint_names = [f"joint_{i}" for i in range(12)]
        robot.root_physx_view = SimpleNamespace(link_paths=[paths], get_masses=lambda: masses,
                                               get_inertias=lambda: inertias)
        assets = SimpleNamespace(ARM_JOINT_NAMES=robot.joint_names[:6], HAND_JOINT_NAMES=robot.joint_names[6:],
                                 FORCE_TORQUE_BODY_NAME="ToolForceTorqueLink", TCP_BODY_NAME="right_plam_force_sensor")
        with patch.dict(sys.modules, {"contact_rearrangement.src.assets": assets}):
            state = mdp.MdpState(env)
        self.assertEqual(len(state.downstream_ids), 33)
        self.assertIn(names.index("base_link_1"), state.downstream_ids)
        self.assertNotIn(names.index("base_link"), state.downstream_ids)
        self.assertNotIn(names.index("SensorMountProximal"), state.downstream_ids)
        self.assertAlmostEqual(float(state.masses.sum()), 32.09, places=5)
        torch.testing.assert_close(state.inertias,
                                   torch.diag(torch.tensor([.2, .3, .5])).expand(1, 33, 3, 3))

    def test_invalid_raw_wrench_stays_zero_and_static_gravity_is_compensated_once(self):
        env = make_tensor_env()
        state, robot = env.mdp_state, env.scene["robot"].data
        state.downstream_ids, state.masses = [0], torch.tensor([[1.]])
        state.inertias = torch.eye(3).reshape(1, 1, 3, 3)
        state.wrench_bias = torch.zeros((1, 6))
        robot.body_com_pos_w = torch.tensor([[[.1, 0., 0.]]])
        robot.body_com_acc_w = torch.zeros((1, 1, 6))
        robot.body_com_ang_vel_w = torch.zeros((1, 1, 3))
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

    def wrench_env(self, num_envs=1, bodies=1):
        env = make_tensor_env(num_envs)
        state, robot = env.mdp_state, env.scene["robot"].data
        for owner in (state, robot, env.scene["objects"].data, env, env.action_manager):
            for name, value in list(vars(owner).items()):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    setattr(owner, name, value.double())
        state.downstream_ids = list(range(bodies))
        state.masses = torch.ones((num_envs, bodies), dtype=torch.float64)
        state.inertias = torch.eye(3, dtype=torch.float64).expand(num_envs, bodies, 3, 3).clone()
        state.wrench_bias = torch.zeros((num_envs, 6), dtype=torch.float64)
        robot.body_link_pos_w = torch.zeros((num_envs, bodies, 3), dtype=torch.float64)
        robot.body_link_quat_w = torch.zeros((num_envs, bodies, 4), dtype=torch.float64)
        robot.body_link_quat_w[..., 0] = 1.
        robot.body_com_pos_w = torch.zeros((num_envs, bodies, 3), dtype=torch.float64)
        robot.body_com_acc_w = torch.zeros((num_envs, bodies, 6), dtype=torch.float64)
        robot.body_com_ang_vel_w = torch.zeros((num_envs, bodies, 3), dtype=torch.float64)
        env.sim.cfg = SimpleNamespace(gravity=(0., 0., 0.))
        return env

    def read_corrected(self, env, raw, valid=None):
        if valid is None:
            valid = torch.ones((env.num_envs, 1), dtype=torch.bool)
        sensors = SimpleNamespace(tool_wrench=lambda env: raw, tool_wrench_valid=lambda env: valid)
        with patch.dict(sys.modules, {"contact_rearrangement.src.sensors": sensors}):
            return mdp.corrected_wrench(env)

    def test_translation_removes_ma_and_its_sensor_center_moment_without_removing_external_load(self):
        env = self.wrench_env()
        robot, state = env.scene["robot"].data, env.mdp_state
        state.masses[:] = 2.
        robot.body_com_pos_w[:] = torch.tensor([.1, .2, -.3])
        robot.body_com_acc_w[..., :3] = torch.tensor([3., -2., 1.])
        expected_self_load = torch.tensor([[6., -4., 2., -.8, -2., -1.6]], dtype=torch.float64)
        external_residual = torch.tensor([[1.1, -.6, .4, .3, -.2, .1]], dtype=torch.float64)
        raw = expected_self_load + external_residual
        before = raw.clone()
        torch.testing.assert_close(self.read_corrected(env, raw), external_residual, atol=2.0e-7, rtol=0.)
        torch.testing.assert_close(raw, before)  # Compensation is only a read-side operation.

    def test_anisotropic_angular_acceleration_is_removed(self):
        env = self.wrench_env()
        robot, state = env.scene["robot"].data, env.mdp_state
        state.inertias[0, 0] = torch.diag(torch.tensor([2., 3., 5.], dtype=torch.float64))
        robot.body_com_acc_w[..., 3:] = torch.tensor([.3, -.4, .2], dtype=torch.float64)
        raw = torch.tensor([[0., 0., 0., .6, -1.2, 1.]], dtype=torch.float64)
        torch.testing.assert_close(self.read_corrected(env, raw), torch.zeros_like(raw), atol=1.0e-12, rtol=0.)

    def test_gyroscopic_moment_is_removed_even_at_zero_angular_acceleration(self):
        env = self.wrench_env()
        robot, state = env.scene["robot"].data, env.mdp_state
        state.inertias[0, 0] = torch.diag(torch.tensor([2., 3., 5.], dtype=torch.float64))
        robot.body_com_ang_vel_w[:] = torch.tensor([1., 2., 3.], dtype=torch.float64)
        raw = torch.tensor([[0., 0., 0., 12., -9., 2.]], dtype=torch.float64)
        torch.testing.assert_close(self.read_corrected(env, raw), torch.zeros_like(raw), atol=1.0e-12, rtol=0.)

    def test_articulating_links_use_individual_world_motion_and_link_inertia_frames(self):
        env = self.wrench_env(bodies=3)
        robot, state = env.scene["robot"].data, env.mdp_state
        env.sim.cfg.gravity = (0., 0., -9.81)
        rotations = Rotation.from_euler("xyz", [[.4, -.2, .7], [-.6, .5, -.1], [.8, .3, -.4]])
        robot.body_link_quat_w[0] = torch.tensor(rotations.as_quat()[:, [3, 0, 1, 2]])
        # Principal inertia orientation is deliberately different. Native
        # get_inertias() already returns the tensor in the actor/link frame.
        robot.body_com_quat_w = torch.tensor([[[1., 0., 0., 0.]] * 3], dtype=torch.float64)
        robot.body_link_pos_w[0, 0] = torch.tensor([.6, -.3, 1.2])
        robot.body_com_pos_w[0] = torch.tensor([[.63, -.28, 1.21], [.7, -.2, 1.4], [.8, -.4, 1.5]])
        state.masses[:] = torch.tensor([[.2, .5, .1]])
        state.inertias[0] = torch.tensor([[[.03, .002, .001], [.002, .05, -.003], [.001, -.003, .07]],
                                         [[.02, -.001, .004], [-.001, .04, .002], [.004, .002, .06]],
                                         [[.01, .001, -.002], [.001, .03, .001], [-.002, .001, .05]]])
        robot.body_com_acc_w[0] = torch.tensor([[1., -.5, 2., .3, .1, -.4],
                                               [-2., 1., .5, -.6, .8, .2],
                                               [.4, 2., -1., .2, -.9, .7]])
        robot.body_com_ang_vel_w[0] = torch.tensor([[.4, -.2, .8], [-.5, 1., .3], [.9, .6, -.7]])
        matrices = rotations.as_matrix()
        inertias_world = matrices @ state.inertias[0].numpy() @ matrices.transpose(0, 2, 1)
        accelerations, omega = robot.body_com_acc_w[0].numpy(), robot.body_com_ang_vel_w[0].numpy()
        forces = state.masses[0].numpy()[:, None] * (accelerations[:, :3] - np.array(env.sim.cfg.gravity))
        lever = robot.body_com_pos_w[0].numpy() - robot.body_link_pos_w[0, 0].numpy()
        moments = (np.cross(lever, forces)
                   + (inertias_world @ accelerations[:, 3:, None]).squeeze(-1)
                   + np.cross(omega, (inertias_world @ omega[..., None]).squeeze(-1)))
        expected = torch.tensor(np.concatenate((matrices[0].T @ forces.sum(0),
                                                matrices[0].T @ moments.sum(0)))[None])
        residual = torch.tensor([[.8, -.4, .2, -.1, .15, .05]], dtype=torch.float64)
        torch.testing.assert_close(self.read_corrected(env, expected + residual), residual, atol=1.0e-12, rtol=0.)
        # Approximating moving fingers with the sensor's acceleration fails.
        robot.body_com_acc_w[:, 1:] = robot.body_com_acc_w[:, :1]
        wrong_rigid_proxy = self.read_corrected(env, expected + residual)
        self.assertGreater(float(torch.linalg.vector_norm(wrong_rigid_proxy - residual)), .5)

    def test_invalid_reset_measurement_does_not_read_stale_motion_or_nan_bias(self):
        env = self.wrench_env()
        robot = env.scene["robot"].data
        del robot.body_com_acc_w
        del robot.body_com_ang_vel_w
        env.mdp_state.wrench_bias[:] = float("nan")
        raw = torch.full((1, 6), float("nan"), dtype=torch.float64)
        valid = torch.zeros((1, 1), dtype=torch.bool)
        torch.testing.assert_close(self.read_corrected(env, raw, valid), torch.zeros_like(raw))

    def test_partial_reset_invalidity_does_not_contaminate_a_continuing_environment(self):
        env = self.wrench_env(num_envs=2)
        robot, state = env.scene["robot"].data, env.mdp_state
        robot.body_com_acc_w[0] = float("nan")
        robot.body_com_ang_vel_w[0] = float("nan")
        robot.body_com_pos_w[0] = float("nan")
        robot.body_com_acc_w[1, 0, 0] = 3.
        state.wrench_bias[:] = torch.tensor([[.4, 0., 0., 0., 0., 0.], [.7, 0., 0., 0., 0., 0.]],
                                            dtype=torch.float64)
        valid = torch.tensor([[False], [True]])
        raw = torch.tensor([[float("nan")] * 6, [4., .2, 0., 0., 0., 0.]], dtype=torch.float64)
        expected = torch.tensor([[0., 0., 0., 0., 0., 0.], [.3, .2, 0., 0., 0., 0.]], dtype=torch.float64)
        sensors = SimpleNamespace(tool_wrench=lambda env: raw, tool_wrench_valid=lambda env: valid,
                                  tactile_contacts=lambda env: torch.zeros((2, 17), dtype=torch.bool),
                                  tactile_valid=lambda env: torch.zeros((2, 17), dtype=torch.bool))
        with patch.dict(sys.modules, {"contact_rearrangement.src.sensors": sensors}):
            torch.testing.assert_close(mdp.corrected_wrench(env), expected, atol=1.0e-8, rtol=0.)
            inertias_before = state.inertias.clone()
            mdp.reset_mdp(env, torch.tensor([0]))
            torch.testing.assert_close(mdp.corrected_wrench(env), expected, atol=1.0e-8, rtol=0.)
            torch.testing.assert_close(state.inertias, inertias_before)
            self.assertEqual(float(state.wrench_bias[0].abs().sum()), 0.)
            self.assertAlmostEqual(float(state.wrench_bias[1, 0]), .7, places=6)

    def test_tare_removes_only_residual_after_dynamic_compensation_and_does_not_accumulate_it_twice(self):
        env = self.wrench_env()
        env.scene["robot"].data.body_com_acc_w[0, 0, 0] = 2.
        raw = torch.tensor([[2.3, 0., 0., 0., 0., 0.]], dtype=torch.float64)
        valid = torch.ones((1, 1), dtype=torch.bool)
        sensors = SimpleNamespace(tool_wrench=lambda env: raw, tool_wrench_valid=lambda env: valid)
        with patch.dict(sys.modules, {"contact_rearrangement.src.sensors": sensors}):
            mdp.tare_compensated_wrench(env)
            torch.testing.assert_close(env.mdp_state.wrench_bias,
                                       torch.tensor([[.3, 0., 0., 0., 0., 0.]], dtype=torch.float64))
            torch.testing.assert_close(mdp.corrected_wrench(env), torch.zeros_like(raw))
            mdp.tare_compensated_wrench(env)
            self.assertAlmostEqual(float(env.mdp_state.wrench_bias[0, 0]), .3, places=12)
            valid[:] = False
            with self.assertRaisesRegex(RuntimeError, "completed physics step"):
                mdp.tare_compensated_wrench(env)

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
