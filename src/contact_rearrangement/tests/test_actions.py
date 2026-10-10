"""CPU checks for the 18D action contract and spatial frame mathematics."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_rearrangement.mdp.actions import (
    _axis_values,
    FORBIDDEN_CONTACT_SENSOR_NAMES,
    bounded_hand_target,
    clamp_efforts,
    contact_force_exceeds_threshold,
    forbidden_contacts,
    implicit_cartesian_damping,
    jacobian_at_link_origin,
    map_action,
    pose_target_from_delta,
    physical_damping_matrix,
)


class ActionMappingTests(unittest.TestCase):
    def setUp(self):
        self.translation_scale = torch.tensor([0.01, 0.02, 0.03])
        self.rotation_scale = torch.tensor([0.04, 0.05, 0.06])
        self.minimum = torch.tensor([100.0] * 3 + [5.0] * 3)
        self.maximum = torch.tensor([800.0] * 3 + [40.0] * 3)
        self.hand_limits = torch.tensor([[[-0.2, 0.8], [0.0, 1.0], [-0.4, 0.0],
                                          [0.2, 1.4], [-1.0, 1.0], [0.0, 0.6]]])

    def map(self, actions):
        return map_action(actions, self.translation_scale, self.rotation_scale,
                          self.minimum, self.maximum, self.hand_limits)

    def test_three_segments_have_independent_meanings_and_units(self):
        action = torch.tensor([[1, -1, 0, -1, 0, 1,
                                -1, 0, 1, -1, 0, 1,
                                -1, 0, 1, -1, 0, 1]], dtype=torch.float32)
        normalized, delta, gains, hand = self.map(action)
        torch.testing.assert_close(normalized, action)
        torch.testing.assert_close(delta, torch.tensor([[0.01, -0.02, 0, -0.04, 0, 0.06]]))
        torch.testing.assert_close(gains, torch.tensor([[100, 450, 800, 5, 22.5, 40]], dtype=torch.float32))
        torch.testing.assert_close(hand, torch.tensor([[-0.2, 0.5, 0, 0.2, 0, 0.6]]))
        # Command construction must not modify the actor's source tensor.
        torch.testing.assert_close(action[:, :3], torch.tensor([[1, -1, 0]], dtype=torch.float32))

    def test_out_of_range_commands_clip_and_stiffness_stays_positive(self):
        action = torch.full((1, 18), 2.0)
        normalized, delta, gains, hand = self.map(action)
        torch.testing.assert_close(normalized, torch.ones_like(action))
        torch.testing.assert_close(gains, self.maximum[None])
        torch.testing.assert_close(hand, self.hand_limits[..., 1])
        torch.testing.assert_close(delta, torch.tensor([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06]]))

    def test_nonfinite_policy_output_is_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid):
                action = torch.zeros(1, 18)
                action[0, 10] = invalid
                with self.assertRaisesRegex(ValueError, "finite"):
                    self.map(action)

    def test_wrong_action_dimension_is_rejected(self):
        for shape in ((1, 12), (18,), (2, 1, 18)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.map(torch.zeros(shape))


class BaseFrameTargetTests(unittest.TestCase):
    LOWER = torch.tensor([-2.0, -2.0, -2.0], dtype=torch.float64)
    UPPER = -LOWER

    def target(self, pose, delta):
        return pose_target_from_delta(torch.tensor([pose], dtype=torch.float64),
                                      torch.tensor([delta], dtype=torch.float64), self.LOWER, self.UPPER)

    def test_rotation_vector_uses_fixed_base_axes_and_left_composition(self):
        # Measured TCP is rotated 90 degrees about base Y. The command rotates
        # about BASE X, whose axis differs from this TCP's local X axis.
        half = math.sqrt(0.5)
        pose = [0.2, -0.1, 0.3, half, 0, half, 0]
        target, clipped = self.target(pose, [0.01, 0.02, -0.03, math.pi / 2, 0, 0])
        torch.testing.assert_close(target, torch.tensor([[0.21, -0.08, 0.27, 0.5, 0.5, 0.5, 0.5]],
                                                        dtype=torch.float64))
        self.assertFalse(clipped.item())

    def test_each_command_uses_measured_pose_without_previous_target_accumulation(self):
        measured = [0.2, 0.1, 0.3, 1, 0, 0, 0]
        increment = [0.01, 0, 0, 0, 0, 0]
        first, _ = self.target(measured, increment)
        next_command, _ = self.target(measured, increment)
        torch.testing.assert_close(first, next_command)
        self.assertAlmostEqual(next_command[0, 0].item(), 0.21)
        updated_measurement = measured.copy()
        updated_measurement[0] += 0.003
        next_command, _ = self.target(updated_measurement, increment)
        self.assertAlmostEqual(next_command[0, 0].item(), 0.213)

    def test_zero_and_tiny_rotations_have_finite_unit_quaternions(self):
        for rotation in (0.0, 1e-12, 0.3):
            with self.subTest(rotation=rotation):
                target, _ = self.target([0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, rotation, 0])
                self.assertTrue(torch.isfinite(target).all())
                self.assertAlmostEqual(torch.linalg.vector_norm(target[0, 3:]).item(), 1.0)
                self.assertGreater(target[0, 3].item(), 0)

    def test_workspace_clamps_translation_in_base_coordinates(self):
        target, clipped = self.target([1.99, -1.99, 0, 1, 0, 0, 0], [0.02, -0.03, 0.04, 0, 0, 0])
        torch.testing.assert_close(target[0, :3], torch.tensor([2, -2, 0.04], dtype=torch.float64))
        self.assertTrue(clipped.item())


class SpatialControlTests(unittest.TestCase):
    def test_absolute_hand_goal_is_applied_with_measured_position_lead_and_native_rate_caps(self):
        measured = torch.tensor([[0.0, 0.2, 0.4]])
        desired = torch.tensor([[0.8, 0.0, 0.404]])
        applied = bounded_hand_target(desired, measured, torch.tensor([[1.0, 0.5, 1.0]]), 1 / 120)
        torch.testing.assert_close(applied, torch.tensor([[1 / 120, 0.2 - 0.5 / 120, 0.404]]))
        # A blocked finger cannot build up an accumulating reference error.
        repeated = bounded_hand_target(desired, measured, torch.tensor([[1.0, 0.5, 1.0]]), 1 / 120)
        torch.testing.assert_close(applied, repeated)

    def test_com_to_tcp_point_shift_and_base_rotation_match_rigid_velocity(self):
        # One joint produces COM linear +X and angular +Z in world axes.
        jacobian = torch.tensor([[[1.0], [0], [0], [0], [0], [2.0]]])
        # The TCP origin is 0.25 m along world +X from the COM.
        lever = torch.tensor([[0.25, 0, 0]])
        # The robot root has yaw pi, as in the shelf examples.
        world_to_base = torch.diag(torch.tensor([-1.0, -1.0, 1.0]))[None]
        shifted = jacobian_at_link_origin(jacobian, lever, world_to_base)
        torch.testing.assert_close(shifted[0, :, 0], torch.tensor([-1.0, -0.5, 0, 0, 0, 2.0]))
        velocity = (shifted @ torch.tensor([[[0.3]]])).squeeze(-1)
        torch.testing.assert_close(velocity, torch.tensor([[-0.3, -0.15, 0, 0, 0, 0.6]]))
        # The native query tensor is shared by other terms and must stay intact.
        torch.testing.assert_close(jacobian[0, :, 0], torch.tensor([1.0, 0, 0, 0, 0, 2.0]))

    def test_effort_limits_are_native_per_joint_and_saturation_is_per_environment(self):
        effort = torch.tensor([[200.0, -30, 20, 0, 28, -28], [1.0, 2, 3, 4, 5, 6]])
        limits = torch.tensor([[150.0, 150, 150, 28, 28, 28]]).expand_as(effort)
        clamped, saturated = clamp_efforts(effort, limits)
        torch.testing.assert_close(clamped[0], torch.tensor([150.0, -30, 20, 0, 28, -28]))
        self.assertEqual(saturated.tolist(), [True, False])

    def test_nonfinite_controller_output_and_invalid_limits_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            clamp_efforts(torch.tensor([[math.nan]]), torch.ones(1, 1))
        for limit in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(limit=limit), self.assertRaisesRegex(RuntimeError, "limits"):
                clamp_efforts(torch.zeros(1, 1), torch.tensor([[limit]]))

    def test_axis_config_rejects_wrong_length_and_nonfinite_values(self):
        self.assertEqual(_axis_values(0.015, 3, "scale"), (0.015, 0.015, 0.015))
        for value in ((1, 2), (1, 2, math.nan), (math.inf, 1, 2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _axis_values(value, 3, "scale")


class ForbiddenContactTests(unittest.TestCase):
    def test_separate_partners_do_not_cancel_opposing_normal_forces(self):
        history = torch.zeros(2, 4, 1, 3, 3)
        history[0, -1, 0, 0, 0] = 1.2
        history[0, -1, 0, 1, 0] = -1.2
        history[1, -1, 0, 0, 0] = 0.99
        self.assertEqual(contact_force_exceeds_threshold(history, 1.0).tolist(), [True, False])

    def test_earlier_physics_substep_contact_survives_until_policy_evaluation(self):
        history = torch.zeros(1, 4, 1, 1, 3)
        history[0, 0, 0, 0, 2] = 1.01
        self.assertEqual(contact_force_exceeds_threshold(history, 1.0).tolist(), [True])
        history.zero_()
        self.assertEqual(contact_force_exceeds_threshold(history, 1.0).tolist(), [False])

    def test_all_nine_nonhand_sensors_are_required_and_unioned_per_environment(self):
        self.assertEqual(len(FORBIDDEN_CONTACT_SENSOR_NAMES), 9)
        scene = {name: SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=torch.zeros(2, 4, 1, 1, 3)))
                 for name in FORBIDDEN_CONTACT_SENSOR_NAMES}
        scene[FORBIDDEN_CONTACT_SENSOR_NAMES[4]].data.force_matrix_w_history[1, -1, 0, 0, 0] = 2
        env = SimpleNamespace(num_envs=2, device="cpu", scene=scene)
        self.assertEqual(forbidden_contacts(env).tolist(), [False, True])
        del scene[FORBIDDEN_CONTACT_SENSOR_NAMES[0]]
        with self.assertRaises(KeyError):
            forbidden_contacts(env)

    def test_invalid_force_data_and_threshold_fail_instead_of_silent_no_contact(self):
        for threshold in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                contact_force_exceeds_threshold(torch.zeros(1, 4, 1, 1, 3), threshold)
        with self.assertRaises(ValueError):
            contact_force_exceeds_threshold(torch.zeros(1, 4, 2, 1, 3), 1.0)
        history = torch.zeros(1, 4, 1, 1, 3)
        history[0, 0, 0, 0, 0] = math.nan
        with self.assertRaises(RuntimeError):
            contact_force_exceeds_threshold(history, 1.0)


class PhysicalImpedanceTests(unittest.TestCase):
    def test_diagonal_inertia_recovers_physical_critical_damping(self):
        masses = torch.tensor([[0.01, 0.0014, 0.02, 1, 2, 0.03]], dtype=torch.float64)
        gains = torch.tensor([[800, 800, 800, 40, 40, 40]], dtype=torch.float64)
        damping = physical_damping_matrix(torch.diag_embed(masses), gains)
        torch.testing.assert_close(damping, torch.diag_embed(2 * torch.sqrt(masses * gains)))

    def test_coupled_inertia_uses_full_matrix_damping(self):
        inertia = torch.diag(torch.tensor([0.01, 0.02, 0.1, 1, 2, 0.004], dtype=torch.float64))[None]
        inertia[0, 0, 1] = inertia[0, 1, 0] = 0.003
        gains = torch.tensor([[300, 600, 800, 5, 20, 40]], dtype=torch.float64)
        damping = physical_damping_matrix(inertia, gains)
        self.assertGreater(abs(damping[0, 0, 1]).item(), 0.01)
        torch.testing.assert_close(damping, damping.mT)
        # Critical geometric-mean damping satisfies D M^-1 D = 4K.
        torch.testing.assert_close(damping @ torch.linalg.solve(inertia, damping),
                                   4 * torch.diag_embed(gains), atol=1e-9, rtol=1e-9)
        self.assertTrue((torch.linalg.eigvalsh(damping) > 0).all())

    def test_backward_euler_energy_decays_at_light_wrist_inertia_and_large_dt_frequency(self):
        j = torch.eye(6, dtype=torch.float64)[None]
        m = torch.diag(torch.tensor([1, 1, 1, 0.0014, 0.002, 0.003], dtype=torch.float64))[None]
        gains = torch.tensor([[800, 800, 800, 40, 40, 40]], dtype=torch.float64)
        error = torch.tensor([[0.01, 0, 0, 0.05, 0.02, 0]], dtype=torch.float64)
        velocity = torch.zeros_like(error)
        dt = 1 / 120
        energy = 0.5 * (gains * error.square()).sum()
        for _ in range(60):
            correction, expected_velocity, _, _ = implicit_cartesian_damping(j, m, gains, velocity, error, dt)
            force = gains * error - correction
            velocity = velocity + dt * torch.linalg.solve(m, force.unsqueeze(-1)).squeeze(-1)
            torch.testing.assert_close(velocity, expected_velocity, atol=1e-12, rtol=1e-10)
            error = error - dt * velocity
            next_energy = 0.5 * ((gains * error.square()).sum()
                                 + (velocity.unsqueeze(-2) @ m @ velocity.unsqueeze(-1)).sum())
            self.assertLessEqual(next_energy.item(), energy.item() + 1e-12)
            energy = next_energy
        self.assertLess(energy.item(), 1e-8)

    def test_rank_deficient_jacobian_has_finite_regularized_solution(self):
        j = torch.eye(6)[None]
        j[0, 5] = 0
        results = implicit_cartesian_damping(j, torch.eye(6)[None], torch.full((1, 6), 100.0),
                                             torch.zeros(1, 6), torch.ones(1, 6) * 0.01, 1 / 120)
        self.assertTrue(all(torch.isfinite(value).all() for value in results))
        self.assertGreaterEqual(results[2][0, 5, 5].item(), 999999)

    def test_invalid_dynamics_are_rejected(self):
        j, m, gains, v, error = torch.eye(6)[None], torch.eye(6)[None], torch.ones(1, 6), torch.zeros(1, 6), torch.zeros(1, 6)
        for dt in (0, -1, math.nan, math.inf):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                implicit_cartesian_damping(j, m, gains, v, error, dt)
        with self.assertRaises(RuntimeError):
            physical_damping_matrix(-m, gains)
        gains[0, 0] = math.nan
        with self.assertRaises(RuntimeError):
            implicit_cartesian_damping(j, m, gains, v, error, 1 / 120)


if __name__ == "__main__":
    unittest.main()
