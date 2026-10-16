"""CPU MDP invariants and numerical controller checks; no Isaac Sim scene required."""

import ast
from contextlib import contextmanager
import dataclasses
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
import warp as wp

# Use the actual checkout math implementation without starting Kit for CPU tests.
# Production imports the public Isaac Lab module normally after AppLauncher.
if "isaaclab.utils.math" not in sys.modules:
    import importlib.util
    import os
    lab_root = Path(os.environ.get("ISAACLAB_PATH", Path(__file__).resolve().parents[3] / "IsaacLab"))
    math_path = lab_root / "source/isaaclab/isaaclab/utils/math.py"
    spec = importlib.util.spec_from_file_location("isaaclab.utils.math", math_path)
    lab_math = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lab_math
    spec.loader.exec_module(lab_math)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_manipulation.mdp import events as task, observations as obs, rewards, terminations

def scene(batch=2, objects=5):
    geometry = torch.zeros((batch, objects, 10), dtype=torch.float64)
    geometry[..., 0] = torch.arange(objects)
    geometry[..., 3] = 1.0
    geometry[..., 7:] = 0.1
    return geometry


def identity(batch, bodies=None):
    shape = (batch, 4) if bodies is None else (batch, bodies, 4)
    quaternion = torch.zeros(shape, dtype=torch.float64)
    quaternion[..., 0] = 1.0
    return quaternion


def rotation_matrix(q):
    """Independent quaternion-to-matrix formula used only by the force oracle."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class InitialGeometryTests(unittest.TestCase):
    def test_roles_reorder_once_distractors_preserve_order_and_padding_is_zero(self):
        source = scene()
        original = source.clone()
        valid = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]])
        geometry, ordered_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([3, 2]), torch.tensor([0, 4]), capacity=7
        )
        torch.testing.assert_close(geometry[0, :4, 0], torch.tensor([3., 0., 1., 4.], dtype=source.dtype))
        torch.testing.assert_close(geometry[1, :4, 0], torch.tensor([2., 4., 0., 3.], dtype=source.dtype))
        self.assertTrue(ordered_valid[:, :4].all())
        self.assertFalse(ordered_valid[:, 4:].any())
        self.assertEqual(int(geometry[:, 4:].count_nonzero()), 0)
        torch.testing.assert_close(source, original)

    def test_unobserved_target_and_unassociated_blocker_leave_reserved_slots_empty(self):
        source = scene(batch=1)
        valid = torch.tensor([[True, True, False, True, False]])
        geometry, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([2]), torch.tensor([-1]), capacity=5
        )
        self.assertEqual(int(geometry[:, :2].count_nonzero()), 0)
        self.assertFalse(result_valid[:, :2].any())
        torch.testing.assert_close(geometry[0, 2:, 0], source[0, [0, 1, 3], 0])

    def test_source_can_be_larger_than_output_when_only_visible_objects_fit(self):
        source = scene(batch=1, objects=100)
        valid = torch.zeros((1, 100), dtype=torch.bool)
        valid[:, [6, 30, 90]] = True
        geometry, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([90]), torch.tensor([30]), capacity=4
        )
        torch.testing.assert_close(geometry[0, :3, 0], source[0, [90, 30, 6], 0])
        self.assertEqual(int(result_valid.sum()), 3)

    def test_invalid_role_assignments_or_capacity_fail_at_snapshot_acceptance(self):
        source, valid = scene(batch=1), torch.ones((1, 5), dtype=torch.bool)
        for target, blocker in ((0, 0), (-2, 1), (5, 0)):
            with self.subTest(target=target, blocker=blocker), self.assertRaises(ValueError):
                obs.prepare_initial_geometry(source, valid, torch.tensor([target]), torch.tensor([blocker]))
        with self.assertRaisesRegex(ValueError, "capacity"):
            obs.prepare_initial_geometry(source, valid, torch.tensor([0]), torch.tensor([1]), capacity=4)

    def test_valid_geometry_rejects_nonfinite_nonpositive_extent_and_nonunit_quaternion(self):
        valid = torch.ones((1, 5), dtype=torch.bool)
        for component, value in ((0, float("nan")), (8, float("inf")), (7, 0.0), (9, -.2),
                                 (3, 0.0), (3, 2.0)):
            source = scene(batch=1)
            source[0, 2, component] = value
            with self.subTest(component=component, value=value), self.assertRaisesRegex(
                ValueError, "finite.*positive extents.*unit quaternions"
            ):
                obs.prepare_initial_geometry(source, valid, torch.tensor([0]), torch.tensor([1]))

    def test_slight_quaternion_drift_is_normalized_without_mutating_source_and_invalid_nan_is_padding(self):
        source = scene(batch=1)
        source[0, 0, 3] = 1.0005
        source[0, 2] = float("nan")
        valid = torch.tensor([[True, True, False, True, True]])
        result, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([0]), torch.tensor([1]), capacity=6
        )
        self.assertEqual(float(result[0, 0, 3]), 1.0)
        self.assertEqual(float(source[0, 0, 3]), 1.0005)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(int(result[:, 4:].count_nonzero()), 0)
        self.assertFalse(result_valid[:, 4:].any())

    def test_eef_coordinates_follow_robot_measurement_without_mutating_frozen_geometry(self):
        source = scene(batch=1, objects=2)
        source[0, 0, :3] = torch.tensor([2., 2., 3.])
        original = source.clone()
        tcp = torch.tensor([[1., 2., 3., math.sqrt(.5), 0., 0., math.sqrt(.5)]], dtype=source.dtype)
        result = obs.geometry_in_eef_frame(source, torch.tensor([[True, False]]), tcp)
        torch.testing.assert_close(result[0, 0, :3], torch.tensor([0., -1., 0.], dtype=source.dtype))
        torch.testing.assert_close(result[0, 0, 3:7], torch.tensor([math.sqrt(.5), 0., 0., -math.sqrt(.5)], dtype=source.dtype))
        torch.testing.assert_close(result[0, 0, 7:], source[0, 0, 7:])
        self.assertEqual(int(result[0, 1].count_nonzero()), 0)
        torch.testing.assert_close(source, original)

    def test_quaternion_sign_equivalent_geometry_has_same_policy_encoding(self):
        source = scene(batch=1, objects=2)
        source[0, 1, 0] = 0.0
        source[0, 1, 3] = -1.0
        tcp = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=source.dtype)
        result = obs.geometry_in_eef_frame(source, torch.ones((1, 2), dtype=torch.bool), tcp)
        torch.testing.assert_close(result[:, 0], result[:, 1])


class DynamicObservationTests(unittest.TestCase):
    def values(self, batch=3):
        tcp = torch.zeros((batch, 7))
        tcp[:, 3] = 1.0
        return dict(joint_pos=torch.zeros((batch, 12)), joint_vel=torch.zeros((batch, 12)),
                    tcp_pose_b=tcp, compensated_wrench=torch.zeros((batch, 6)),
                    tactile_contacts=torch.zeros((batch, 17), dtype=torch.bool),
                    tactile_valid=torch.ones((batch, 17), dtype=torch.bool),
                    previous_action=torch.zeros((batch, 18)))

    def test_72_channels_have_agreed_units_sensor_axes_and_missing_tactile(self):
        values = self.values(batch=1)
        values["joint_pos"][:] = math.pi
        values["joint_vel"][:] = 2.0
        values["compensated_wrench"][:] = torch.tensor([200., -200., 360., -8., 8., 80.])
        values["tactile_contacts"][:, [0, 5]] = True
        values["tactile_valid"][:, [5, 16]] = False
        values["previous_action"][:, 0] = 2.0
        sample = obs.dynamic_sample(**values)
        self.assertEqual(sample.shape, (1, 72))
        torch.testing.assert_close(sample[:, :24], torch.ones((1, 24)))
        torch.testing.assert_close(sample[:, 24:31], values["tcp_pose_b"])
        torch.testing.assert_close(sample[0, 31:37], torch.tensor([1., -1., 1., -1., 1., 5.]))
        torch.testing.assert_close(sample[0, [37, 42, 53]], torch.tensor([1., -1., -1.]))
        self.assertEqual(float(sample[0, 54]), 1.0)

    def test_pack_has_exactly_geometry_current_previous_with_no_role_or_mask_channels(self):
        geometry = scene(batch=3, objects=10)
        current = obs.dynamic_sample(**self.values())
        previous = current.clone()
        previous[:, 0] = -0.5
        history = torch.stack((current, previous), dim=1).double()
        packed = obs.pack_policy_observation(geometry, history)
        self.assertEqual(packed.shape, (3, 244))
        torch.testing.assert_close(packed[:, :100], geometry.flatten(1))
        torch.testing.assert_close(packed[:, 100:172], current.double())
        torch.testing.assert_close(packed[:, 172:], previous.double())

    def test_invalid_mimic_joint_or_long_history_dimensions_are_rejected(self):
        values = self.values()
        values["joint_pos"] = torch.zeros((3, 18))
        with self.assertRaisesRegex(ValueError, "12 independent"):
            obs.dynamic_sample(**values)
        with self.assertRaisesRegex(ValueError, "current and previous"):
            obs.pack_policy_observation(scene(), torch.zeros((2, 16, 72)))

    def test_hot_path_batches_1024_environments_without_tensor_scalar_reads(self):
        batch = 1024
        values = self.values(batch)
        geometry = scene(batch, 10).float()
        valid = torch.ones((batch, 10), dtype=torch.bool)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("GPU scalar synchronization")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("GPU condition synchronization")):
            current = obs.dynamic_sample(**values)
            transformed = obs.geometry_in_eef_frame(geometry, valid, values["tcp_pose_b"])
            packed = obs.pack_policy_observation(transformed, torch.stack((current, current), dim=1))
            raw, _ = obs.calibrate_wrench(values["compensated_wrench"])
            corrected = obs.compensate_wrench(raw, raw, raw, torch.ones(batch, dtype=torch.bool))
        self.assertEqual(packed.shape, (batch, 244))
        torch.testing.assert_close(corrected, torch.zeros_like(raw))


class WrenchTests(unittest.TestCase):
    def test_saturation_resolution_and_overload_apply_to_raw_before_compensation(self):
        ideal = torch.tensor([[200.2, -.14, 360.0, 8.2, .011, -.014]], dtype=torch.float64)
        measured, exceeded = obs.calibrate_wrench(ideal)
        torch.testing.assert_close(measured, torch.tensor([[200., -.1, 360., 8., .01, -.015]], dtype=ideal.dtype))
        torch.testing.assert_close(exceeded, torch.tensor([[True, False, False, True, False, False]]))
        load = measured - .003
        corrected = obs.compensate_wrench(measured, load, torch.zeros_like(load), torch.tensor([True]))
        torch.testing.assert_close(corrected, torch.full_like(corrected, .003))

    def test_static_gravity_and_lever_arm_remove_distal_weight(self):
        masses = torch.tensor([[2.]], dtype=torch.float64)
        com = torch.tensor([[[1., 0., 0.]]], dtype=masses.dtype)
        wrench = obs.self_motion_wrench(
            masses, torch.eye(3, dtype=masses.dtype)[None, None], com, identity(1, 1),
            torch.zeros((1, 1, 6), dtype=masses.dtype), torch.zeros((1, 1, 3), dtype=masses.dtype),
            torch.zeros((1, 3), dtype=masses.dtype), identity(1), torch.tensor([0., 0., -9.81]),
        )
        torch.testing.assert_close(wrench, torch.tensor([[0., 0., 19.62, 0., -19.62, 0.]], dtype=masses.dtype))

    def test_articulated_rotational_inertia_matches_independent_world_matrix_oracle(self):
        generator = torch.Generator().manual_seed(47)
        def random(*shape):
            return torch.randn(shape, generator=generator, dtype=torch.float64)
        batch, bodies = 7, 5
        masses = random(batch, bodies).abs() + .1
        basis = random(batch, bodies, 3, 3)
        inertias = basis @ basis.transpose(-1, -2)
        body_q, sensor_q = random(batch, bodies, 4), random(batch, 4)
        body_q = body_q / body_q.norm(dim=-1, keepdim=True)
        sensor_q = sensor_q / sensor_q.norm(dim=-1, keepdim=True)
        com, acceleration, omega = random(batch, bodies, 3), random(batch, bodies, 6), random(batch, bodies, 3)
        sensor_pos, gravity = random(batch, 3), torch.tensor([0., 0., -9.81], dtype=torch.float64)
        actual = obs.self_motion_wrench(masses, inertias, com, body_q, acceleration, omega,
                                        sensor_pos, sensor_q, gravity)
        body_r, sensor_r = rotation_matrix(body_q), rotation_matrix(sensor_q)
        inertia_world = body_r @ inertias @ body_r.transpose(-1, -2)
        forces = masses[..., None] * (acceleration[..., :3] - gravity)
        momentum = (inertia_world @ omega[..., None]).squeeze(-1)
        moment = (inertia_world @ acceleration[..., 3:, None]).squeeze(-1)
        moment += torch.linalg.cross(omega, momentum, dim=-1)
        moment += torch.linalg.cross(com-sensor_pos[:, None], forces, dim=-1)
        inverse_sensor = sensor_r.transpose(-1, -2)
        expected = torch.cat(((inverse_sensor @ forces.sum(1)[..., None]).squeeze(-1),
                              (inverse_sensor @ moment.sum(1)[..., None]).squeeze(-1)), dim=-1)
        torch.testing.assert_close(actual, expected)

    def test_compensation_cancels_self_load_and_bias_preserves_external_load_and_masks_invalid(self):
        self_load = torch.arange(18, dtype=torch.float64).reshape(3, 6)
        bias = torch.full_like(self_load, .025)
        external = torch.tensor([[1., -2., 3., .1, -.2, .3]], dtype=self_load.dtype).expand(3, -1)
        raw = self_load + bias + external
        raw[1] = float("nan")
        result = obs.compensate_wrench(raw, self_load, bias, torch.tensor([True, False, True]))
        torch.testing.assert_close(result[[0, 2]], external[[0, 2]])
        torch.testing.assert_close(result[1], torch.zeros(6, dtype=result.dtype))

def settings():
    return SimpleNamespace(
        clearance_decay=1.0, time_penalty_scale=0.02, time_penalty_growth_rate=0.7,
        rearrangement_decay=2.0, motion_acceleration_decay=1.0, motion_wrench_increase_decay=1.0,
        arm_joint_acceleration_ref=5.0, hand_joint_acceleration_ref=10.0,
        force_rate_ref=200.0, torque_rate_ref=10.0, stable_steps=5,
        success_reward=500.0, failure_penalty=100.0, target_translation_tolerance=0.02,
        target_rotation_tolerance=0.15, stable_linear_velocity=0.015,
        stable_angular_velocity=0.10, success_overlap_area=1.e-5,
        workspace_tolerance=0.015, drop_tolerance=0.06, lift_tolerance=0.04,
        max_tilt=math.radians(50), force_hard=80.0, torque_hard=4.0,
        shelf_translation_tolerance=.01, shelf_rotation_tolerance=math.radians(2),
    )


def reward_inputs(n=1):
    return dict(
        blocked=torch.zeros(n, dtype=torch.float64), blocker_area=torch.ones(n, dtype=torch.float64),
        path_increment=torch.zeros(n, dtype=torch.float64),
        wrench=torch.zeros(n, 6, dtype=torch.float64), previous_wrench=torch.zeros(n, 6, dtype=torch.float64),
        elapsed_time=torch.zeros(n, dtype=torch.float64), dt=0.1,
    )


@contextmanager
def native_joint_acc_reward():
    """Execute Isaac Lab's actual reward arithmetic without importing Kit."""
    import os
    lab = Path(os.environ.get("ISAACLAB_PATH", Path(__file__).resolve().parents[3] / "IsaacLab"))
    source = lab / "source/isaaclab/isaaclab/envs/mdp/rewards.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'joint_acc_l2')
    syntax = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              function], type_ignores=[])
    namespace = {'torch': torch, 'SceneEntityCfg': lambda name: SimpleNamespace(name=name, joint_ids=slice(None))}
    exec(compile(ast.fix_missing_locations(syntax), str(source), 'exec'), namespace)
    native_module = types.ModuleType('isaaclab.envs.mdp.rewards')
    native_module.joint_acc_l2 = namespace['joint_acc_l2']
    with patch.dict(sys.modules, {'isaaclab.envs.mdp.rewards': native_module}):
        yield native_module


def physical_inputs(n=1, objects=3):
    center = torch.zeros(n, objects, 3, dtype=torch.float64)
    center[..., 2] = 0.1
    signs = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=center.dtype)
    corners = center[..., None, :] + 0.1 * signs
    rotation = torch.zeros(n, objects, 4, dtype=center.dtype)
    rotation[..., 0] = 1.0
    active = torch.ones(n, objects, dtype=torch.bool)
    target = torch.zeros_like(active)
    target[:, 0] = True
    return dict(center=center, corners=corners, rotation=rotation,
                velocity=torch.zeros(n, objects, 6, dtype=center.dtype), active=active, target=target,
                initial_center=center.clone(), initial_rotation=rotation.clone(),
                blocker_overlap=torch.zeros(n, dtype=center.dtype),
                sensor_wrench=torch.zeros(n, 6, dtype=center.dtype),
                stable_count=torch.zeros(n, dtype=torch.long),
                workspace_bounds=torch.tensor([-1., 1., -1., 1.], dtype=center.dtype),
                shelf_top=0.0, ceiling=0.8)


class RewardTests(unittest.TestCase):
    def test_clearance_uses_blocker_fraction_and_quality_terms_use_exponential_inputs(self):
        data = reward_inputs(3)
        data["blocked"][:] = torch.tensor([0.0, 0.1, 0.5])
        data["blocker_area"][:] = torch.tensor([0.2, 0.2, 0.5], dtype=torch.float64)
        data["path_increment"][:] = torch.tensor([0.0, 0.25, 0.5])
        components, costs = rewards.reward_components(settings(), **data)
        expected = torch.exp(torch.tensor([0.0, -0.5, -1.0], dtype=torch.float64))
        torch.testing.assert_close(components[:, 0], expected)
        torch.testing.assert_close(components[:, 1], torch.exp(-2 * data["path_increment"]))
        torch.testing.assert_close(components[:, 2], torch.ones(3, dtype=torch.float64))
        self.assertTrue(((components[:, :3] > 0) & (components[:, :3] <= 1)).all())
        torch.testing.assert_close(costs, torch.zeros_like(costs))

    def test_rearrangement_uses_current_increment_without_repeated_penalty_after_motion(self):
        early, late = reward_inputs(5), reward_inputs(5)
        early["path_increment"][0] = .1
        late["path_increment"][-1] = .1
        early_scores, _ = rewards.reward_components(settings(), **early)
        late_scores, _ = rewards.reward_components(settings(), **late)
        torch.testing.assert_close(early_scores[:, 1].sum(), late_scores[:, 1].sum())
        torch.testing.assert_close(early_scores[1:, 1], torch.ones(4, dtype=torch.float64))
        self.assertAlmostEqual(early_scores[0, 1].item(), math.exp(-.2))

    def test_force_and_torque_only_penalize_magnitude_increase(self):
        data = reward_inputs(4)
        data["previous_wrench"][:, 0] = 20.0
        data["previous_wrench"][:, 3] = 1.0
        data["wrench"][0] = torch.tensor([40., 0., 0., 2., 0., 0.])  # Each +one rate reference.
        data["wrench"][1] = data["previous_wrench"][1]
        data["wrench"][2] = 0.0  # Release contributes no cost.
        data["wrench"][3] = torch.tensor([0., -20., 0., 0., -1., 0.])  # Direction only.
        scores, costs = rewards.reward_components(settings(), **data)
        torch.testing.assert_close(costs, torch.tensor([2., 0., 0., 0.], dtype=torch.float64))
        expected_quality = torch.tensor([math.exp(-1), 1., 1., 1.], dtype=torch.float64)
        torch.testing.assert_close(scores[:, 2], expected_quality)

    def test_wrench_growth_score_keeps_raw_diagnostics_and_normalizes_rms_rate(self):
        data = reward_inputs(3)
        data["wrench"][:, 0] = torch.tensor([20., 40., 200.])
        scores, costs = rewards.reward_components(settings(), **data)
        torch.testing.assert_close(costs, torch.tensor([1., 4., 100.], dtype=torch.float64))
        expected_quality = torch.tensor([math.exp(-math.sqrt(x / 2)) for x in (1., 4., 100.)],
                                        dtype=torch.float64)
        torch.testing.assert_close(scores[:, 2], expected_quality)
        # A constant physical ramp has the same per-step quality at any sample rate.
        for steps in (2, 4, 20):
            ramp = reward_inputs(steps)
            ramp["dt"] = .2 / steps
            ramp["previous_wrench"][:, 0] = torch.arange(steps) * (200 * ramp["dt"])
            ramp["wrench"][:, 0] = ramp["previous_wrench"][:, 0] + 200 * ramp["dt"]
            values, _ = rewards.reward_components(settings(), **ramp)
            torch.testing.assert_close(values[:, 2], torch.full((steps,), math.exp(-math.sqrt(.5)),
                                                               dtype=torch.float64))

    def test_wrench_score_decreases_with_rate_and_decay_and_stays_bounded(self):
        data = reward_inputs(5)
        normalized_rates = torch.tensor([0., .1, 1., 2., 100.], dtype=torch.float64)
        data['wrench'][:, 0] = normalized_rates * data['dt'] * settings().force_rate_ref
        scores, costs = rewards.reward_components(settings(), **data)
        torch.testing.assert_close(costs, normalized_rates.square())
        torch.testing.assert_close(scores[:, 2], torch.exp(-normalized_rates / math.sqrt(2)))
        self.assertTrue((scores[1:, 2] < scores[:-1, 2]).all())
        cfg = settings()
        cfg.motion_wrench_increase_decay = 2.
        stronger, _ = rewards.reward_components(cfg, **data)
        torch.testing.assert_close(stronger[:, 2], scores[:, 2].square())

    def test_corner_path_translation_rotation_return_and_inactive_nan(self):
        data = physical_inputs(objects=2)
        before = data["corners"]
        translated = before + torch.tensor([0.3, 0.4, 0.0], dtype=before.dtype)
        active = torch.tensor([[True, False]])
        translated[:, 1] = math.nan
        increment = rewards.corner_path_increment(before, translated, active)
        torch.testing.assert_close(increment, torch.tensor([0.5], dtype=before.dtype))
        returned = rewards.corner_path_increment(translated, before, active)
        torch.testing.assert_close(increment + returned, torch.tensor([1.0], dtype=before.dtype))
        # Rotate the cube 90 degrees about its center: every XY corner moves 0.2 m.
        rotated = before.clone()
        rotated[..., 0], rotated[..., 1] = -before[..., 1], before[..., 0]
        angular_increment = rewards.corner_path_increment(before, rotated, active)
        torch.testing.assert_close(angular_increment, torch.tensor([0.2], dtype=before.dtype))

    def test_exact_time_integral_independent_of_step_partition(self):
        cfg = settings()
        total_time = 8.0
        reference = -cfg.time_penalty_scale * (math.expm1(cfg.time_penalty_growth_rate * total_time)
                                              / cfg.time_penalty_growth_rate - total_time)
        for steps in (1, 80, 240, 800):
            data = reward_inputs(steps)
            data["dt"] = total_time / steps
            data["elapsed_time"] = torch.arange(steps, dtype=torch.float64) * data["dt"]
            components, _ = rewards.reward_components(cfg, **data)
            self.assertAlmostEqual(components[:, 3].sum().item(), reference, places=12)
            self.assertTrue(torch.all(components[1:, 3] < components[:-1, 3]))

    def test_small_first_time_step_retains_nonzero_cost(self):
        data = reward_inputs()
        data["dt"] = 1.e-8
        scores, _ = rewards.reward_components(settings(), **data)
        expected = -0.02 * 0.7 * data["dt"] ** 2 / 2
        self.assertAlmostEqual(scores[0, 3].item() / expected, 1.0, places=8)

    def test_nonfinite_state_does_not_produce_nonfinite_reward(self):
        data = reward_inputs(3)
        data["wrench"][0, 0] = math.nan
        data["wrench"][1, 1] = math.inf
        data["path_increment"][2] = math.nan
        scores, costs = rewards.reward_components(settings(), **data)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(costs).all())
        self.assertEqual(scores[0, 2], 0.0)
        self.assertEqual(scores[2, 1], 0.0)

    def test_large_parallel_reward_batch_equals_independent_chunks(self):
        torch.manual_seed(11)
        data = reward_inputs(4096)
        for key in ("blocked", "path_increment", "wrench", "previous_wrench"):
            data[key].uniform_(0.0, 1.0)
        data["elapsed_time"].uniform_(0.0, 8.0)
        all_scores, all_costs = rewards.reward_components(settings(), **data)
        split_scores, split_costs = [], []
        for start in range(0, 4096, 257):
            chunk = {k: v[start:start + 257] if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            scores, costs = rewards.reward_components(settings(), **chunk)
            split_scores.append(scores)
            split_costs.append(costs)
        torch.testing.assert_close(all_scores, torch.cat(split_scores))
        torch.testing.assert_close(all_costs, torch.cat(split_costs))

    def test_reward_batch_preserves_dtype_without_device_scalar_or_host_reads(self):
        def reject(*args, **kwargs):
            raise AssertionError("Reward kernels must not read device scalars or copy tensors to the CPU")
        for dtype in (torch.float32, torch.float64):
            data = {key: value.to(dtype) if isinstance(value, torch.Tensor) else value
                    for key, value in reward_inputs(1024).items()}
            with patch.object(torch.Tensor, '__bool__', reject), \
                 patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'cpu', reject):
                components, costs = rewards.reward_components(settings(), **data)
            self.assertEqual(components.shape, (1024, 4))
            self.assertEqual(costs.shape, (1024,))
            self.assertEqual(components.dtype, dtype)
            self.assertEqual(costs.dtype, dtype)

    def test_motion_uses_native_selected_joint_acceleration_rms_and_exponential_quality(self):
        acceleration = torch.zeros(4, 18, dtype=torch.float64)
        acceleration[0, :6] = 5.
        acceleration[1, :6] = 10.
        acceleration[0, 6:12] = 10.
        acceleration[1, 6:12] = 20.
        acceleration[:, 12:] = 10000.  # Passive/mimic joints must not enter either group.
        acceleration[3, 0] = math.nan
        asset = SimpleNamespace(num_joints=18, data=SimpleNamespace(joint_acc=acceleration,
                                                    joint_vel=torch.full_like(acceleration, 100.)))
        state = SimpleNamespace(cfg=settings(), components=torch.ones(4, 5, dtype=torch.float64))
        env = SimpleNamespace(scene={'robot': asset}, step_dt=.02)
        for arm_ids, hand_ids in ((list(range(6)), list(range(6, 12))), (slice(0, 6), slice(6, 12))):
            with self.subTest(arm_ids=arm_ids), native_joint_acc_reward() as native, \
                 patch.object(task, 'get_task_state', return_value=state), \
                 patch.object(native, 'joint_acc_l2', wraps=native.joint_acc_l2) as called:
                actual = rewards.motion(env, SimpleNamespace(name='robot', joint_ids=arm_ids),
                                        SimpleNamespace(name='robot', joint_ids=hand_ids)) * env.step_dt
            self.assertEqual(called.call_count, 2)
            expected = torch.tensor([.5 * (math.exp(-1) + 1), .5 * (math.exp(-2) + 1), 1.],
                                    dtype=acceleration.dtype)
            torch.testing.assert_close(actual[:3], expected)
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        # Both the acceleration RMS and the wrench RMS at their references yield exp(-1).
        state.components[:, 2] = math.exp(-1)
        with native_joint_acc_reward(), patch.object(task, 'get_task_state', return_value=state):
            actual = rewards.motion(env, SimpleNamespace(name='robot', joint_ids=slice(0, 6)),
                                    SimpleNamespace(name='robot', joint_ids=slice(6, 12))) * env.step_dt
        self.assertAlmostEqual(actual[0].item(), math.exp(-1))

    def test_motion_quality_uses_joint_count_metadata_without_tensor_host_reads(self):
        acceleration = torch.full((1024, 12), 5.)
        acceleration[:, 6:] = 10.
        cfg = settings()
        state = SimpleNamespace(cfg=cfg, components=torch.ones(1024, 5))
        asset = SimpleNamespace(num_joints=12, data=SimpleNamespace(joint_acc=acceleration))
        env = SimpleNamespace(scene={'robot': asset}, step_dt=.02)
        arm_cfg = SimpleNamespace(name='robot', joint_ids=slice(None, 6))
        hand_cfg = SimpleNamespace(name='robot', joint_ids=slice(6, None))
        def reject(*args, **kwargs):
            raise AssertionError('Motion must not read tensor values on the host')
        with native_joint_acc_reward(), patch.object(task, 'get_task_state', return_value=state), \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'cpu', reject):
            actual = rewards.motion(env, arm_cfg, hand_cfg) * env.step_dt
        torch.testing.assert_close(actual, torch.full((1024,), .5 * (math.exp(-1) + 1)))
        # Manager integration cancels the adapter's rate conversion for a fixed snapshot.
        env.step_dt = .01
        with native_joint_acc_reward(), patch.object(task, 'get_task_state', return_value=state):
            faster = rewards.motion(env, arm_cfg, hand_cfg) * env.step_dt
        torch.testing.assert_close(faster, actual)


class TerminationTests(unittest.TestCase):
    def test_stable_success_failure_priority_and_single_terminal_payment(self):
        cfg = settings()
        candidate = torch.tensor([True, True, True, False])
        failed = torch.tensor([False, False, True, False])
        stable = torch.tensor([4, 5, 5, 0])
        paid = torch.zeros(4, dtype=torch.bool)
        success, terminal, paid = terminations.terminal_transition(candidate, failed, stable, paid, cfg)
        self.assertEqual(success.tolist(), [False, True, False, False])
        self.assertEqual(terminal.tolist(), [0., 500., -100., 0.])
        _, repeated, _ = terminations.terminal_transition(candidate, failed, stable, paid, cfg)
        torch.testing.assert_close(repeated, torch.zeros_like(repeated))

    def test_target_protection_overlap_and_all_objects_settled(self):
        data = physical_inputs(5)
        data["center"][1, 0, 0] += 0.021
        data["rotation"][2, 0] = torch.tensor([math.cos(0.08), 0., 0., math.sin(0.08)])
        data["blocker_overlap"][3] = 1.e-4
        data["velocity"][4, 2, 0] = 0.016
        data["stable_count"][:] = 2
        candidate, stable, _, failed = terminations.task_conditions(settings(), **data)
        self.assertEqual(candidate.tolist(), [True, False, False, False, False])
        self.assertEqual(stable.tolist(), [3, 0, 0, 0, 0])
        self.assertFalse(failed.any())

    def test_each_failure_reason_and_invalid_state_is_independent(self):
        data = physical_inputs(7)
        data["corners"][0, 1, :, 2] -= 0.061
        data["rotation"][1, 1] = torch.tensor([math.cos(math.pi / 6), math.sin(math.pi / 6), 0., 0.])
        data["corners"][2, 1, :, 0] += 1.1
        data["corners"][3, 1, :, 2] += 0.041
        data["sensor_wrench"][4, 0] = 80.1
        data["initial_shelf_pose"] = torch.zeros(7, 7, dtype=torch.float64)
        data["initial_shelf_pose"][:, 3] = 1.
        data["shelf_pose"] = data["initial_shelf_pose"].clone()
        data["shelf_pose"][5, 0] = .011
        data["center"][6, 1, 1] = math.nan
        candidate, stable, reasons, failed = terminations.task_conditions(settings(), **data)
        torch.testing.assert_close(reasons, torch.eye(7, dtype=torch.bool))
        self.assertTrue(failed.all())
        self.assertFalse(candidate.any())
        torch.testing.assert_close(stable, torch.zeros(7, dtype=torch.long))

    def test_nonfinite_initial_geometry_wrench_velocity_and_robot_fail(self):
        data = physical_inputs(5)
        data["initial_rotation"][0, 1] = math.nan
        data["sensor_wrench"][1, 0] = math.nan
        data["velocity"][2, 1, 0] = math.inf
        data["state_valid"] = torch.tensor([True, True, True, False, True])
        data["rotation"][4, 1] = 0.0  # Invalid quaternion must not count as upright.
        _, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(reasons[:, 6].all())
        self.assertTrue(failed.all())

    def test_inactive_nan_slots_do_not_poison_conditions(self):
        data = physical_inputs(2)
        data["active"][:, 2] = False
        for key in ("center", "corners", "rotation", "velocity", "initial_center", "initial_rotation"):
            data[key][:, 2] = math.nan
        candidate, stable, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(candidate.all())
        self.assertFalse(reasons.any())
        self.assertFalse(failed.any())
        torch.testing.assert_close(stable, torch.ones_like(stable))

    def test_shelf_displacement_rotation_sign_and_invalid_pose(self):
        data = physical_inputs(8)
        initial = torch.zeros(8, 7, dtype=torch.float64)
        initial[:, :3] = torch.arange(8, dtype=torch.float64)[:, None] * 2.5
        initial[:, 3] = 1.
        pose = initial.clone()
        pose[0, 0] += .009  # Below translation tolerance.
        pose[1, 0] += .011
        for row, angle in ((2, 1.9), (3, 2.1)):
            half = math.radians(angle) / 2
            pose[row, 3:] = torch.tensor([math.cos(half), 0., 0., math.sin(half)], dtype=pose.dtype)
        pose[4, 3:] *= -1  # Quaternion sign must not trigger a rotation.
        pose[5, 3:] *= 2   # Normalize before comparing.
        pose[6, 0] = math.nan
        pose[7, 3:] = 0
        data.update(shelf_pose=pose, initial_shelf_pose=initial)
        _, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertEqual(failed.tolist(), [False, True, False, True, False, False, True, True])
        self.assertEqual(reasons[:, 5].tolist(), [False, True, False, True, False, False, False, True])
        self.assertTrue(reasons[6:, 6].all())

    def test_total_sensor_overrange_fails_even_with_zero_compensated_load(self):
        data = physical_inputs(2)
        data["sensor_range_exceeded"] = torch.zeros(2, 6, dtype=torch.bool)
        data["sensor_range_exceeded"][1, 2] = True
        _, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertEqual(failed.tolist(), [False, True])
        self.assertTrue(reasons[1, 4])

    def test_per_environment_workspace_and_shelf_offsets(self):
        data = physical_inputs(2)
        data["corners"][1, :, :, 2] += 2.0
        data["shelf_top"] = torch.tensor([0.0, 2.0], dtype=torch.float64)
        data["ceiling"] = torch.tensor([0.8, 2.8], dtype=torch.float64)
        data["workspace_bounds"] = data["workspace_bounds"].repeat(2, 1)
        candidate, _, _, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(candidate.all())
        self.assertFalse(failed.any())

    def test_large_parallel_conditions_equal_independent_chunks(self):
        torch.manual_seed(19)
        data = physical_inputs(2048, objects=10)
        data["velocity"].normal_(mean=0.0, std=0.006)
        data["blocker_overlap"].uniform_(0.0, 2.e-5)
        data["stable_count"].random_(0, 8)
        data["sensor_wrench"].normal_(mean=0.0, std=1.0)
        full = terminations.task_conditions(settings(), **data)
        chunks = [[], [], [], []]
        for start in range(0, 2048, 257):
            chunk = {k: v[start:start + 257] if isinstance(v, torch.Tensor) and k != "workspace_bounds" else v
                     for k, v in data.items()}
            partial = terminations.task_conditions(settings(), **chunk)
            for output, value in zip(chunks, partial):
                output.append(value)
        for expected, parts in zip(full, chunks):
            torch.testing.assert_close(expected, torch.cat(parts))

@unittest.skipUnless("isaaclab.controllers" in sys.modules, "Requires Isaac Sim AppLauncher")
class NativeControllerTests(unittest.TestCase):
    def test_native_hand_soft_limit_mapping_and_selected_reset(self):
        from unittest.mock import Mock
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
        from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction

        names = [f"finger_{i}" for i in range(6)]
        cfg = JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=names, preserve_order=True, rescale_to_limits=True,
        )
        limits = torch.tensor([[-.5, 1.5]]).expand(3, 6, 2).clone()
        data = SimpleNamespace(soft_joint_pos_limits=limits, joint_pos=torch.zeros(3, 6))
        asset = SimpleNamespace(data=data, num_joints=6,
            find_joints=lambda names, preserve_order: (list(range(6)), names), set_joint_position_target=Mock())
        env = SimpleNamespace(scene={'robot': asset}, num_envs=3, device='cpu')
        hand = cfg.class_type(cfg, env)
        self.assertIs(type(hand), JointPositionToLimitsAction)
        hand.process_actions(torch.tensor([[-1.]*6, [0.]*6, [1.]*6]))
        expected = torch.tensor([[-.5]*6, [.5]*6, [1.5]*6])
        torch.testing.assert_close(hand.processed_actions, expected)
        hand.apply_actions()
        torch.testing.assert_close(asset.set_joint_position_target.call_args.args[0], expected)
        # Native mapping uses the live limits and passes the full position target.
        data.soft_joint_pos_limits[1, :, 1] = .5
        hand.process_actions(torch.full((3, 6), 2.))
        torch.testing.assert_close(hand.processed_actions[1], torch.full((6,), .5))
        before = hand.processed_actions.clone()
        hand.reset(torch.tensor([1]))
        torch.testing.assert_close(hand.raw_actions[[0, 2]], torch.full((2, 6), 2.))
        torch.testing.assert_close(hand.raw_actions[1], torch.zeros(6))
        torch.testing.assert_close(hand.processed_actions, before)

    def test_native_osc_action_scaling_and_gain_clamp(self):
        from isaaclab.controllers import OperationalSpaceControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
        from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction

        names = [f"arm_{i}" for i in range(6)]
        cfg = OperationalSpaceControllerActionCfg(
            asset_name="robot", joint_names=names, body_name="tcp",
            position_scale=.015, orientation_scale=.08, stiffness_scale=100.,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"], impedance_mode="variable_kp",
                motion_stiffness_limits_task=(5., 800.), nullspace_control="none",
            ),
        )
        root_pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).expand(4, -1).clone()
        body_pose = root_pose[:, None].expand(-1, 2, -1).clone()
        body_pose[:, 1, 0] = .3
        data = SimpleNamespace(root_pos_w=root_pose[:, :3], root_quat_w=root_pose[:, 3:],
            body_pos_w=body_pose[:, :, :3], body_quat_w=body_pose[:, :, 3:])
        asset = SimpleNamespace(data=data, num_joints=6, is_fixed_base=True,
            find_joints=lambda names: (list(range(6)), names),
            find_bodies=lambda name: ([1], [name]))
        env = SimpleNamespace(scene={'robot': asset}, num_envs=4, device='cpu',
            sim=SimpleNamespace(get_physics_dt=lambda: .01))
        arm = cfg.class_type(cfg, env)
        self.assertIs(type(arm), OperationalSpaceControllerAction)
        actions = torch.ones(4, 12)
        actions[:, 6:] = torch.tensor([-1., 0., 1., 10.])[:, None]
        arm.process_actions(actions)
        torch.testing.assert_close(arm.processed_actions[:, :3], torch.full((4, 3), .015))
        torch.testing.assert_close(arm.processed_actions[:, 3:6], torch.full((4, 3), .08))
        torch.testing.assert_close(arm.processed_actions[:, 6:],
                                   torch.tensor([5., 5., 100., 800.])[:, None].expand(-1, 6))
        arm.reset(torch.tensor([1]))
        torch.testing.assert_close(arm.raw_actions[1], torch.zeros(12))
        torch.testing.assert_close(arm.raw_actions[[0, 2, 3]], actions[[0, 2, 3]])

    def test_native_osc_relative_latch_and_inertia_scaled_pd(self):
        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

        cfg = OperationalSpaceControllerCfg(
            target_types=["pose_rel"], impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True, partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True, motion_damping_ratio_task=1.0,
            motion_stiffness_limits_task=(5., 800.), nullspace_control="none",
        )
        controller = OperationalSpaceController(cfg, 1024, "cpu")
        half = math.sqrt(.5)
        pose = torch.tensor([[.3, 0, .4, half, 0, half, 0]]).expand(1024, -1).clone()
        gains = torch.tensor([[450., 450, 450, 22.5, 22.5, 22.5]]).expand(1024, -1)
        delta = torch.tensor([[.01, -.01, .005, .03, 0, 0]]).expand(1024, -1)
        command = torch.cat((delta, gains), -1)
        controller.set_command(command, current_ee_pose_b=pose)
        target = controller.desired_ee_pose_b.clone()
        controller.set_command(command, current_ee_pose_b=pose)
        torch.testing.assert_close(controller.desired_ee_pose_b, target)
        torch.testing.assert_close(target[:, :3], pose[:, :3] + delta[:, :3])
        torch.testing.assert_close(target[0, 3:], half * torch.tensor(
            [math.cos(.015), math.sin(.015), math.cos(.015), math.sin(.015)]))
        diagonal = torch.tensor([[1., 2., 3., .01, .02, .03]]).expand(1024, -1)
        mass = torch.diag_embed(diagonal)
        velocity = torch.full((1024, 6), .02)
        gravity = torch.full_like(velocity, .1)
        efforts = controller.compute(jacobian_b=torch.eye(6).expand(1024, -1, -1),
            current_ee_pose_b=pose, current_ee_vel_b=velocity, mass_matrix=mass, gravity=gravity)
        expected = diagonal * (gains * delta - 2 * gains.sqrt() * velocity) + gravity
        torch.testing.assert_close(efforts, expected, atol=2e-5, rtol=2e-5)


def _clip_oracle(vertices, bounds):
    """Independent scalar half-plane oracle operating on SciPy's convex hull."""
    current = [np.asarray(vertex) for vertex in vertices]
    for axis, bound, keep_lower in ((0, bounds[0], False), (0, bounds[1], True),
                                     (1, bounds[2], False), (1, bounds[3], True)):
        output = []
        if not current:
            return 0.0
        for previous, following in zip(current[-1:] + current[:-1], current):
            previous_inside = previous[axis] <= bound if keep_lower else previous[axis] >= bound
            next_inside = following[axis] <= bound if keep_lower else following[axis] >= bound
            if previous_inside != next_inside:
                ratio = (bound - previous[axis]) / (following[axis] - previous[axis])
                output.append(previous + ratio * (following - previous))
            if next_inside:
                output.append(following)
        current = output
    if len(current) < 3:
        return 0.0
    polygon = np.asarray(current)
    return abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
               - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2


def fixture(n=3, *, dtype=torch.float64):
    cfg = task.MdpCfg()
    o = cfg.max_objects
    center = torch.zeros(n, o, 3, dtype=dtype)
    center[..., 0] = torch.arange(o, dtype=dtype) * 0.12
    center[..., 2] = 0.05
    signs = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=dtype)
    corners = center[..., None, :] + 0.05 * signs
    rotation = torch.zeros(n, o, 4, dtype=dtype)
    rotation[..., 0] = 1.0
    sensor_rotation = torch.zeros(n, 4, dtype=dtype)
    sensor_rotation[:, 0] = 1.0
    tcp_pose = torch.zeros(n, 7, dtype=dtype)
    tcp_pose[:, 3] = 1.0
    batch = task.TransitionBatch(
        center=center, corners=corners, rotation=rotation,
        velocity=torch.zeros(n, o, 6, dtype=dtype),
        sensor_wrench=torch.zeros(n, 6, dtype=dtype), sensor_position=torch.zeros(n, 3, dtype=dtype),
        sensor_rotation=sensor_rotation, sensor_range_exceeded=torch.zeros(n, 6, dtype=torch.bool),
        shelf_pose=torch.tensor([0., 0., 0., 1., 0., 0., 0.], dtype=dtype).expand(n, 7).clone(), state_valid=torch.ones(n, dtype=torch.bool),
        sample=torch.arange(72, dtype=dtype).expand(n, -1).clone(), tcp_pose_b=tcp_pose,
    )
    active = torch.ones(n, o, dtype=torch.bool)
    target = torch.zeros_like(active)
    target[:, 0] = True
    blocker = torch.zeros_like(active)
    blocker[:, 1] = True
    geometry = torch.zeros(n, o, 10, dtype=dtype)
    geometry[..., :3] = center
    geometry[..., 3] = 1.0
    geometry[..., 7:] = 0.1
    reset = dict(geometry_b=geometry, geometry_valid=active.clone(), active=active, target=target,
                 blocker=blocker, access_region=torch.tensor([-0.1, 0.2, -0.2, 0.2], dtype=dtype).expand(n, -1),
                 workspace_bounds=torch.tensor([-10., 10., -2., 2.], dtype=dtype).expand(n, -1),
                 shelf_top=torch.zeros(n, dtype=dtype), ceiling=torch.ones(n, dtype=dtype))
    state = task.TaskState(n, "cpu", cfg, 0.1, dtype=dtype)
    state.reset(torch.arange(n), batch, **reset, step=0)
    return state, batch, reset


def clone_batch(batch, **updates):
    values = {field.name: getattr(batch, field.name).clone() for field in dataclasses.fields(batch)}
    values.update(updates)
    return task.TransitionBatch(**values)


def native_fixture(n=3):
    """Native tensor layout with deliberately independent camera and GT geometry."""
    class Scene(dict):
        pass

    origins = torch.zeros(n, 3)
    origins[:, 1] = torch.arange(n) * 3.0
    root_q = torch.tensor([1., 0., 0., 0.]).expand(n, 4).clone()
    body_q = root_q[:, None].expand(n, 4, 4).clone()
    body_pos = origins[:, None].expand(n, 4, 3).clone()
    body_pos[:, 1:, 2] += .5
    body_com = body_pos.clone()
    body_com[:, 3, 0] += .1
    data = types.SimpleNamespace(
        body_link_pos_w=body_pos, body_link_quat_w=body_q,
        body_com_pos_w=body_com, body_com_pos_b=body_com-body_pos, body_com_acc_w=torch.zeros(n, 4, 6),
        body_com_ang_vel_w=torch.zeros(n, 4, 3), root_pos_w=origins.clone(), root_quat_w=root_q,
        joint_pos=torch.zeros(n, 12), joint_vel=torch.zeros(n, 12), joint_acc=torch.zeros(n, 12),
        applied_torque=torch.zeros(n, 12),
    )
    inertia = torch.diag(torch.tensor([.01, .02, .03])).flatten().expand(n, 4, 9).clone()
    arm_names = [f"arm_{i}" for i in range(6)]
    hand_names = [f"hand_{i}" for i in range(6)]
    joint_names = arm_names + hand_names
    body_names = ["base", "tcp", "ft", "hand"]
    robot = types.SimpleNamespace(
        body_names=body_names, num_joints=12, data=data,
        find_joints=lambda names, preserve_order=False: ([joint_names.index(name) for name in names], names),
        find_bodies=lambda name: ([body_names.index(name)], [name]),
        root_physx_view=types.SimpleNamespace(get_masses=lambda: torch.ones(n, 4), get_inertias=lambda: inertia),
    )
    physical_pose = torch.zeros(n, 4, 7)
    physical_pose[..., :3] = origins[:, None]
    physical_pose[..., 0] += torch.tensor([-.1, .1, .3, .5])
    physical_pose[..., 2] += .05
    physical_pose[..., 3] = 1.
    objects = types.SimpleNamespace(data=types.SimpleNamespace(
        object_link_pose_w=physical_pose, object_com_pos_w=physical_pose[..., :3].clone(),
        object_com_pos_b=torch.zeros(n, 4, 3),
        object_com_vel_w=torch.zeros(n, 4, 6),
    ))
    shelf_pose = torch.cat((origins.clone(), root_q.clone()), -1)
    scene = Scene(robot=robot, object_collection=objects,
                  shelf=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=shelf_pose)))
    scene.env_origins = origins
    action = types.SimpleNamespace(normalized_actions=torch.zeros(n, 18), applied_torque=data.applied_torque)
    object_ids = torch.full((n, 10), -1, dtype=torch.long)
    object_ids[:, :3] = torch.tensor([2, 0, 3])
    observed_ids = torch.full((n, 10), -1, dtype=torch.long)
    observed_ids[:, :3] = torch.tensor([3, 2, 0])
    geometry = torch.zeros(n, 10, 10)
    geometry[:, :3, 0] = torch.tensor([.51, .31, -.11])
    geometry[:, :3, 2] = .12
    geometry[:, :3, 3] = 1.
    geometry[:, :3, 7:] = .08
    geometry_valid = observed_ids >= 0
    source = task.SceneBatch(
        object_ids=object_ids, target_id=torch.full((n,), 2), blocker_id=torch.zeros(n, dtype=torch.long),
        geometry_b=geometry, geometry_valid=geometry_valid, observed_ids=observed_ids,
        lower=torch.full((4, 3), -.05), upper=torch.full((4, 3), .05),
        upright=torch.tensor([1., 0., 0., 0.]).expand(4, 4).clone(),
        access_region=torch.tensor([-.2, .29, -.2, .2]).expand(n, 4).clone(),
        workspace_bounds=torch.tensor([-1., 1., -1., 1.]).expand(n, 4).clone(),
        shelf_top=torch.zeros(n), ceiling=torch.ones(n), sensor_body_name="ft", distal_body_names=("ft", "hand"),
    )
    external = torch.tensor([1., 2., 3., .1, .2, .3]).expand(n, 6).clone()
    self_load = torch.tensor([0., 0., 20., 0., -1., 0.])
    sensors = task.SensorBatch(
        ideal_wrench=external+self_load, wrench_valid=torch.ones(n, dtype=torch.bool),
        tactile_force=torch.zeros(n, 17, 3), tactile_valid=torch.ones(n, 17, dtype=torch.bool),
    )
    class ActionManager:
        @property
        def action(self):
            return action.normalized_actions

    env = types.SimpleNamespace(
        num_envs=n, device="cpu", cfg=types.SimpleNamespace(mdp=task.MdpCfg(), actions=SimpleNamespace(
            arm_action=SimpleNamespace(joint_names=arm_names, body_name="tcp"),
            hand_action=SimpleNamespace(joint_names=hand_names))), step_dt=.1,
        common_step_counter=0, scene=scene, sim=types.SimpleNamespace(cfg=types.SimpleNamespace(gravity=(0., 0., -10.))),
        action_manager=ActionManager(),
        task_scene_data=source, sensors=sensors, extras={}, sensor_read_count=0,
    )
    def read_sensors():
        env.sensor_read_count += 1
        return env.sensors
    env.task_sensor_reader = read_sensors
    return env, action, external


class GeometryTests(unittest.TestCase):
    def test_random_roll_pitch_yaw_obb_hull_and_overlap_match_scipy(self):
        rng = np.random.default_rng(33)
        rotations = Rotation.random(200, random_state=rng)
        sizes = rng.uniform(0.01, 0.5, (200, 3))
        centers = rng.uniform(-0.3, 0.3, (200, 3))
        signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        corners = np.einsum("nij,nkj->nki", rotations.as_matrix(), sizes[:, None, :] * signs) + centers[:, None, :]
        points = torch.tensor(corners[..., :2], dtype=torch.float64)
        polygons, counts = task.convex_hull_xy(points)
        areas = task.polygon_area(polygons, counts).numpy()
        bounds = np.array([-0.15, 0.25, -0.2, 0.3])
        overlaps = task.rectangle_overlap_area(polygons, counts, torch.tensor(bounds)).numpy()
        expected_areas, expected_overlaps = [], []
        for xy in corners[..., :2]:
            hull = ConvexHull(xy)
            expected_areas.append(hull.volume)
            expected_overlaps.append(_clip_oracle(xy[hull.vertices], bounds))
        np.testing.assert_allclose(areas, expected_areas, rtol=1.e-11, atol=1.e-12)
        np.testing.assert_allclose(overlaps, expected_overlaps, rtol=1.e-10, atol=1.e-12)

    def test_axis_aligned_duplicate_corners_and_disjoint_clipping(self):
        _, batch, _ = fixture(1)
        points = batch.corners[:, 0, :, :2]
        polygons, count = task.convex_hull_xy(points)
        self.assertEqual(count.item(), 4)
        torch.testing.assert_close(task.polygon_area(polygons, count), torch.tensor([0.01], dtype=points.dtype))
        overlap = task.rectangle_overlap_area(polygons, count, torch.tensor([2., 3., 2., 3.]))
        torch.testing.assert_close(overlap, torch.zeros_like(overlap))

    def test_degenerate_zero_boxes_have_zero_area_and_overlap(self):
        points = torch.zeros(10, 8, 2, dtype=torch.float64)
        polygon, count = task.convex_hull_xy(points)
        torch.testing.assert_close(task.polygon_area(polygon, count), torch.zeros(10, dtype=points.dtype))
        overlap = task.rectangle_overlap_area(polygon, count, torch.tensor([-1., 1., -1., 1.]))
        torch.testing.assert_close(overlap, torch.zeros_like(overlap))

    def test_object_geometry_removes_catalog_upright_once_with_off_center_origin(self):
        q = torch.tensor([[math.sqrt(0.5), 0., 0., math.sqrt(0.5)]], dtype=torch.float64)
        pose = torch.cat((torch.tensor([[2., 3., 0.]], dtype=q.dtype), q), -1)
        local_center = torch.tensor([[0.2, 0., 0.1]], dtype=q.dtype)
        local_corners = local_center[:, None, :].expand(-1, 8, -1).clone()
        center, corners, relative = task.object_geometry(pose, local_center, local_corners, q)
        torch.testing.assert_close(center, pose[:, :3] + local_center)
        torch.testing.assert_close(corners, (pose[:, :3] + local_center)[:, None, :].expand(-1, 8, -1))
        torch.testing.assert_close(relative, torch.tensor([[1., 0., 0., 0.]], dtype=q.dtype))


class StateTests(unittest.TestCase):
    def test_history_current_previous_and_once_per_global_step(self):
        state, batch, _ = fixture()
        initial = state.history.clone()
        batch.sample.fill_(5.0)
        state.advance(batch, 1)
        torch.testing.assert_close(state.history[:, 1], initial[:, 0])
        torch.testing.assert_close(state.history[:, 0], batch.sample)
        snapshot = {k: v.clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
        batch.sample.fill_(8.0)
        state.advance(batch, 1)
        for key, value in snapshot.items():
            torch.testing.assert_close(getattr(state, key), value)
        state.advance(batch, 2)
        torch.testing.assert_close(state.history[:, 1], torch.full_like(batch.sample, 5.0))
        torch.testing.assert_close(state.history[:, 0], torch.full_like(batch.sample, 8.0))
        torch.testing.assert_close(state.elapsed_time, torch.full_like(state.elapsed_time, 0.2))

    def test_partial_reset_only_changes_selected_episode_rows(self):
        state, batch, reset = fixture(4)
        batch.corners.add_(torch.tensor([0.001, 0.0, 0.0], dtype=batch.corners.dtype))
        batch.center[..., 0].add_(0.001)
        state.advance(batch, 1)
        untouched = torch.tensor([0, 2])
        snapshot = {k: v[untouched].clone() for k, v in vars(state).items()
                    if isinstance(v, torch.Tensor) and v.shape[:1] == state.history.shape[:1]}
        ids = torch.tensor([1, 3])
        batch.sample.fill_(6.0)
        state.reset(ids, batch, **reset, step=1)
        for key, value in snapshot.items():
            torch.testing.assert_close(getattr(state, key)[untouched], value)
        torch.testing.assert_close(state.path_length[ids], torch.zeros(2, dtype=state.path_length.dtype))
        torch.testing.assert_close(state.elapsed_time[ids], torch.zeros(2, dtype=state.elapsed_time.dtype))
        torch.testing.assert_close(state.history[ids, 1], torch.zeros_like(state.history[ids, 1]))
        torch.testing.assert_close(state.history[ids, 0, 31:37], torch.zeros(2, 6, dtype=state.history.dtype))
        torch.testing.assert_close(state.history[ids, 0, 37:54], -torch.ones(2, 17, dtype=state.history.dtype))
        # Late settling/render physics without a completed control step cannot advance history.
        batch.sample.fill_(12.0)
        state.advance(batch, 1)
        torch.testing.assert_close(state.history[ids, 0, :31], torch.full((2, 31), 6.0, dtype=state.history.dtype))
        state.advance(batch, 2)
        torch.testing.assert_close(state.elapsed_time, torch.tensor([.2, .1, .2, .1], dtype=state.elapsed_time.dtype))

    def test_cumulative_corner_path_keeps_out_and_back_movement(self):
        state, batch, _ = fixture(1)
        before = clone_batch(batch)
        batch.corners[..., 0].add_(0.01)
        batch.center[..., 0].add_(0.01)
        state.advance(batch, 1)
        torch.testing.assert_close(state.path_length, torch.tensor([0.01 * state.cfg.max_objects], dtype=state.path_length.dtype))
        outward_score = state.components[:, 1].clone()
        state.advance(before, 2)
        torch.testing.assert_close(state.path_length, torch.tensor([0.02 * state.cfg.max_objects], dtype=state.path_length.dtype))
        torch.testing.assert_close(state.components[:, 1], outward_score)
        torch.testing.assert_close(outward_score,
                                   torch.exp(-state.cfg.rearrangement_decay * state.path_length / 2))
        # Remaining still restores maximal quality despite the nonzero cumulative path.
        state.advance(before, 3)
        torch.testing.assert_close(state.components[:, 1], torch.ones(1, dtype=state.path_length.dtype))

    def test_only_designated_blocker_overlap_affects_clearance_and_success(self):
        state, batch, values = fixture(2)
        # Clear blocker, but leave target and another active object in the approach area.
        delta = torch.tensor([.4, 0., 0.], dtype=batch.center.dtype)
        batch.center[:, 1] += delta
        batch.corners[:, 1] += delta
        batch.center[:, 2, 0] = .1
        batch.corners[:, 2, :, 0] -= .14
        # Only env 1's blocker remains halfway in the access region.
        batch.center[1, 1, 0] = .2
        batch.corners[1, 1, :, 0] -= .32
        state.reset(torch.arange(2), batch, **values, step=0)
        for step in range(1, state.cfg.stable_steps + 1):
            state.advance(batch, step)
        torch.testing.assert_close(state.blocker_overlap,
                                   torch.tensor([0., .005], dtype=batch.center.dtype))
        torch.testing.assert_close(state.components[:, 0],
                                   torch.tensor([1., math.exp(-.5)], dtype=batch.center.dtype))
        torch.testing.assert_close(state.success, torch.tensor([True, False]))

    def test_rearrangement_includes_target_blocker_and_unobserved_active_distractor(self):
        state, batch, values = fixture(3)
        values['geometry_valid'][:, 2] = False
        state.reset(torch.arange(3), batch, **values, step=0)
        # Each row moves exactly one role; vision availability cannot hide privileged cost.
        rows = torch.arange(3)
        batch.center[rows, rows, 0] += .01
        batch.corners[rows, rows, :, 0] += .01
        state.advance(batch, 1)
        torch.testing.assert_close(state.components[:, 1],
                                   torch.full((3,), math.exp(-.01 * state.cfg.rearrangement_decay),
                                              dtype=batch.center.dtype))

    def test_wrench_transforms_to_fixed_world_axes_and_initial_sensor_origin(self):
        state, batch, _ = fixture(1)
        batch.sensor_rotation[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        batch.sensor_position[0, 0] = 1.0
        batch.sensor_wrench[0, 0] = 1.0
        state.advance(batch, 1)
        expected = torch.tensor([[0., 1., 0., 0., 0., 1.]], dtype=state.previous_wrench.dtype)
        torch.testing.assert_close(state.previous_wrench, expected, rtol=1.e-6, atol=1.e-7)
        # Same resultant wrench at the fixed origin despite moving the measurement origin.
        batch.sensor_position[0, 0] = 2.0
        batch.sensor_wrench[0, 5] = -1.0
        state.advance(batch, 2)
        torch.testing.assert_close(state.previous_wrench, expected, rtol=1.e-6, atol=1.e-7)
        torch.testing.assert_close(state.wrench_rate_cost, torch.zeros(1, dtype=expected.dtype), atol=1.e-12, rtol=0)

    def test_global_step_gaps_fail_explicitly(self):
        state, batch, _ = fixture(1)
        with self.assertRaisesRegex(RuntimeError, "once every"):
            state.advance(batch, 2)

    def test_invalid_state_failure_has_finite_learning_outputs(self):
        state, batch, _ = fixture(2)
        batch.corners[0, 1] = math.nan
        batch.sample[0, 0] = math.nan
        batch.state_valid[0] = False
        state.advance(batch, 1)
        self.assertTrue(state.failure[0])
        self.assertEqual(state.components[0, 4].item(), -100.)
        self.assertTrue(torch.isfinite(state.components).all())
        self.assertTrue(torch.isfinite(state.history).all())

    def test_1024_env_advance_without_tensor_scalar_reads(self):
        state, batch, _ = fixture(1024, dtype=torch.float32)
        batch.corners[..., 0].add_(0.001)
        batch.center[..., 0].add_(0.001)
        def reject_scalar(*args, **kwargs):
            raise AssertionError("Per-step Tensor scalar read / host synchronization")
        with patch.object(torch.Tensor, "__bool__", reject_scalar), patch.object(torch.Tensor, "item", reject_scalar):
            state.advance(batch, 1)
        self.assertEqual(state.components.shape, (1024, 5))
        torch.testing.assert_close(state.path_length, torch.full((1024,), 0.001 * state.cfg.max_objects), atol=1.e-5, rtol=1.e-5)
        self.assertTrue(torch.isfinite(state.components).all())


class NativeAdapterTests(unittest.TestCase):
    def test_shared_geometry_cache_matches_generic_reset_and_keeps_neighbors(self):
        cached, _, _ = native_fixture(5)
        generic, _, _ = native_fixture(5)
        runtime = cached._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = cached.task_scene_data, None
        with patch.object(task, '_prepare_reset_geometry', wraps=task._prepare_reset_geometry) as prepare:
            task.reset_task(cached)
            self.assertEqual(prepare.call_args.args[1].numel(), 1)
            task.reset_task(generic)
            cached.common_step_counter = generic.common_step_counter = 1
            task.get_task_state(cached)
            task.get_task_state(generic)
            ids = torch.tensor([4, 1])
            cached.scene['robot'].data.joint_pos[ids] += .02
            generic.scene['robot'].data.joint_pos[ids] += .02
            runtime.mdp_reset_scene, runtime.mdp_reset_ids = cached.task_scene_data, ids
            with patch.object(task, '_validate_reset_data', side_effect=AssertionError('repeated shared validation')):
                task.reset_task(cached, ids)
            self.assertEqual(prepare.call_count, 2)  # One shared singleton and the generic reset.
        task.reset_task(generic, ids)
        for name, value in vars(generic._contact_mdp).items():
            if isinstance(value, torch.Tensor):
                with self.subTest(field=name):
                    torch.testing.assert_close(getattr(cached._contact_mdp, name), value)
        for name in ('object_ids', 'local_center', 'local_corners', 'upright'):
            torch.testing.assert_close(getattr(cached._contact_mdp.bindings, name),
                                       getattr(generic._contact_mdp.bindings, name))
        self.assertIsNone(runtime.mdp_reset_scene)

    def test_shared_geometry_refresh_invalidates_cache_and_external_resets_validate(self):
        env, _, _ = native_fixture()
        runtime = env._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        task.reset_task(env)
        previous = env._contact_mdp.geometry_b.clone()
        env.task_scene_data.geometry_b[:, :3, 0] += .03
        runtime.shared_template = object()
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        with patch.object(task, '_prepare_reset_geometry', wraps=task._prepare_reset_geometry) as prepare:
            task.reset_task(env)
            self.assertEqual(prepare.call_count, 1)
        self.assertFalse(torch.equal(env._contact_mdp.geometry_b, previous))
        env.task_scene_data.blocker_id[:] = env.task_scene_data.target_id
        # The producer's one-use marker was consumed. External mutations cannot
        # silently reuse the native shared-scene cache.
        with self.assertRaisesRegex(ValueError, 'distinct active target and blocker'):
            task.reset_task(env)

    def test_shared_geometry_marker_requires_the_same_reset_subset(self):
        env, _, _ = native_fixture()
        runtime = env._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        task.reset_task(env)
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, torch.tensor([0])
        env.task_scene_data.blocker_id[1] = env.task_scene_data.target_id[1]
        with self.assertRaisesRegex(ValueError, 'distinct active target and blocker'):
            task.reset_task(env, torch.tensor([1]))

    def test_native_reward_adapters_preserve_quality_and_do_not_repenalize_past_motion(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        env.scene['object_collection'].data.object_link_pose_w[:, 0, 0] += .01
        env.scene['object_collection'].data.object_com_pos_w[:, 0, 0] += .01
        env.sensors.ideal_wrench[:, 0] += 4.
        env.common_step_counter = 1
        state = task.get_task_state(env)
        # Native RewardManager's dt integrates the adapter back to the chosen step score.
        actual_path = rewards.rearrangement(env) * env.step_dt
        torch.testing.assert_close(actual_path, torch.full((3,), math.exp(-.02)), atol=1.e-6, rtol=1.e-5)
        arm_cfg = SimpleNamespace(name='robot', joint_ids=list(range(6)))
        hand_cfg = SimpleNamespace(name='robot', joint_ids=list(range(6, 12)))
        with native_joint_acc_reward():
            actual_motion = rewards.motion(env, arm_cfg, hand_cfg) * env.step_dt
        expected_wrench_quality = torch.exp(-state.cfg.motion_wrench_increase_decay
                                           * torch.sqrt(state.wrench_rate_cost / 2))
        torch.testing.assert_close(actual_motion, .5 * (1 + expected_wrench_quality))
        env.common_step_counter = 2
        torch.testing.assert_close(rewards.rearrangement(env) * env.step_dt, torch.ones(3))
        with native_joint_acc_reward():
            torch.testing.assert_close(rewards.motion(env, arm_cfg, hand_cfg) * env.step_dt, torch.ones(3))

    def test_named_observation_terms_preserve_contract_and_advance_once_in_any_order(self):
        env, action, _ = native_fixture()
        terms = (obs.initial_scene_geometry, obs.current_robot_sample, obs.previous_robot_sample)
        probed = tuple(term(env) for term in terms)
        self.assertEqual(tuple(value.shape for value in probed), ((3, 100), (3, 72), (3, 72)))
        self.assertEqual(int(torch.cat(probed, dim=-1).count_nonzero()), 0)
        self.assertEqual(env.sensor_read_count, 0)

        task.reset_task(env)
        initial_current = obs.current_robot_sample(env).clone()
        action.normalized_actions[:, 0] = .4
        env.common_step_counter = 1
        reads_before = env.sensor_read_count
        # Term evaluation order must not shift history or acquire sensors twice.
        previous = obs.previous_robot_sample(env)
        current = obs.current_robot_sample(env)
        geometry = obs.initial_scene_geometry(env)
        packed = torch.cat((geometry, current, previous), dim=-1)
        torch.testing.assert_close(packed, obs.policy_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(previous, initial_current, rtol=0, atol=0)
        torch.testing.assert_close(current[:, 54], torch.full((3,), .4))
        self.assertEqual(env.sensor_read_count, reads_before + 1)
        torch.testing.assert_close(geometry.reshape(3, 10, 10), task.geometry_in_eef_frame(
            env._contact_mdp.geometry_b, env._contact_mdp.geometry_valid, env._contact_mdp.tcp_pose_b
        ))

    def test_named_observation_terms_preserve_terminal_snapshot_during_partial_reset(self):
        env, action, _ = native_fixture()
        task.reset_task(env)
        action.normalized_actions[:, 0] = .6
        env.common_step_counter = 1
        terms = (obs.initial_scene_geometry, obs.current_robot_sample, obs.previous_robot_sample)
        completed = torch.cat(tuple(term(env) for term in terms), dim=-1)
        terminal_snapshot = env.extras["terminal_observation"]
        torch.testing.assert_close(terminal_snapshot, completed, rtol=0, atol=0)

        task.reset_task(env, torch.tensor([1]))
        reads_after_reset = env.sensor_read_count
        restarted = torch.cat(tuple(term(env) for term in terms), dim=-1)
        torch.testing.assert_close(restarted, obs.policy_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(restarted[[0, 2]], completed[[0, 2]], rtol=0, atol=0)
        self.assertEqual(int(restarted[1, 172:].count_nonzero()), 0)
        self.assertEqual(int(restarted[1, 154:172].count_nonzero()), 0)
        torch.testing.assert_close(env.extras["terminal_observation"], completed, rtol=0, atol=0)
        self.assertIs(env.extras["terminal_observation"], terminal_snapshot)
        self.assertEqual(env.sensor_read_count, reads_after_reset)

    def test_native_actions_and_applied_effort_have_independent_finite_checks(self):
        env, action, _ = native_fixture(4)
        task.reset_task(env)
        action.normalized_actions[:] = torch.linspace(-2, 2, 18)
        action.normalized_actions[0, 0] = math.nan
        action.applied_torque[1, 0] = math.nan
        batch = task._read_transition(env, env._contact_mdp)
        torch.testing.assert_close(batch.sample[:, -18:], torch.nan_to_num(action.normalized_actions).clamp(-1, 1))
        torch.testing.assert_close(batch.state_valid, torch.tensor([False, False, True, True]))

    def test_missing_scene_and_sensor_provider_fail_before_native_binding(self):
        env, _, _ = native_fixture()
        del env.task_sensor_reader
        with self.assertRaisesRegex(RuntimeError, "SceneBatch.*task_sensor_reader"):
            task.reset_task(env)

    def test_sensor_provider_shape_and_boolean_contract_is_checked_on_reset(self):
        invalid_values = (("ideal_wrench", torch.zeros(3, 5)), ("wrench_valid", torch.ones(3, 1, dtype=torch.bool)),
                          ("wrench_valid", torch.ones(3)), ("tactile_force", torch.zeros(3, 16, 3)),
                          ("tactile_valid", torch.ones(3, 17)))
        for name, value in invalid_values:
            env, _, _ = native_fixture()
            setattr(env.sensors, name, value)
            with self.subTest(name=name, shape=value.shape), self.assertRaisesRegex(ValueError, f"SensorBatch.{name}"):
                task.reset_task(env)

    def test_valid_initial_observations_cannot_refer_to_inactive_or_duplicated_objects(self):
        for invalid_id in (-1, 1, 2):
            env, _, _ = native_fixture()
            env.task_scene_data.observed_ids[:, 0] = invalid_id
            with self.subTest(invalid_id=invalid_id), self.assertRaisesRegex(ValueError, "distinct active objects"):
                task.reset_task(env)

    def test_invalid_duplicate_association_cannot_hide_the_valid_target_slot(self):
        env, _, _ = native_fixture()
        source = env.task_scene_data
        source.geometry_valid[:, 0] = False
        source.observed_ids[:, 0] = source.target_id
        task.reset_task(env)
        state = task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b[:, 0], source.geometry_b[:, 1])
        self.assertTrue(state.geometry_valid[:, 0].all())

    def test_observation_dimension_probe_does_not_invent_geometry_and_first_step_requires_reset(self):
        env, _, _ = native_fixture()
        initial = task.get_task_state(env)
        self.assertFalse(initial.ready)
        self.assertEqual(int(initial.geometry_b.count_nonzero()), 0)
        self.assertEqual(env.sensor_read_count, 0)
        env.common_step_counter = 1
        with self.assertRaisesRegex(RuntimeError, "reset_task"):
            task.get_task_state(env)

    def test_initial_reset_keeps_role_slots_and_base_coordinates_independent_of_env_origin(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        initial_reads = env.sensor_read_count
        state = task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b[:, :3], env.task_scene_data.geometry_b[:, [1, 2, 0]])
        torch.testing.assert_close(state.tcp_pose_b[:, :3], torch.tensor([0., 0., .5]).expand(3, 3))
        torch.testing.assert_close(state.initial_center[:, :3, 0], torch.tensor([.3, -.1, .5]).expand(3, 3))
        self.assertEqual(env.sensor_read_count, initial_reads)
        self.assertEqual(int(state.history[:, 1].count_nonzero()), 0)
        torch.testing.assert_close(state.history[:, 0, 37:54], -torch.ones(3, 17))
        self.assertEqual(int(state.history[:, 0, 54:].count_nonzero()), 0)

    def test_one_native_step_compensates_gravity_and_inertia_and_updates_history_once(self):
        env, action, external = native_fixture()
        task.reset_task(env)
        initial_reads = env.sensor_read_count
        initial = env._contact_mdp.history[:, 0].clone()
        data = env.scene["robot"].data
        # Two 1 kg distal links: +2 m/s² in Z adds 4 N; both Iyy=.02.
        # Lever moment changes by -.2 Nm; angular acceleration adds +.04 Nm.
        data.body_com_acc_w[:, 2:, 2] = 2.
        data.body_com_acc_w[:, 2:, 4] = 1.
        env.sensors.ideal_wrench[:, 2] += 4.
        env.sensors.ideal_wrench[:, 4] -= .16
        action.normalized_actions[:, 0] = .4
        env.sensors.tactile_force[:, 0, 2] = .011
        env.sensors.tactile_force[:, 1, 2] = .009
        env.sensors.tactile_valid[:, 2] = False
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.history[:, 1], initial)
        torch.testing.assert_close(state.history[:, 0, 31:37], external / torch.tensor(task.SI_200_8_RANGES))
        torch.testing.assert_close(state.history[:, 0, 37:40], torch.tensor([1., 0., -1.]).expand(3, 3))
        torch.testing.assert_close(state.history[:, 0, 54], torch.full((3,), .4))
        self.assertEqual(env.sensor_read_count, initial_reads + 1)
        saved = state.history.clone()
        env.sensors.ideal_wrench[:, 0] += 10.
        self.assertIs(task.get_task_state(env), state)
        torch.testing.assert_close(state.history, saved)
        self.assertEqual(env.sensor_read_count, initial_reads + 1)

    def test_partial_reset_preserves_other_environment_state_and_completed_terminal_observation(self):
        env, action, _ = native_fixture()
        task.reset_task(env)
        self.assertNotIn("terminal_env_ids", env.extras)
        action.normalized_actions[:, 0] = .6
        action.applied_torque[1, 0] = math.nan
        env.common_step_counter = 1
        state = task.get_task_state(env)
        final_observation = env.extras["terminal_observation"].clone()
        self.assertTrue(env.extras["terminated"][1])
        self.assertEqual(final_observation.shape, (3, 244))
        self.assertEqual(env.extras["terminal_env_ids"].numel(), 0)
        self.assertEqual(float(state.components[1, 4]), -100.)
        untouched = torch.tensor([0, 2])
        snapshot = {key: value[untouched].clone() for key, value in vars(state).items()
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (env.num_envs,)}
        action.applied_torque[1, 0] = 0.
        task.reset_task(env, torch.tensor([1]))
        for key, expected in snapshot.items():
            torch.testing.assert_close(getattr(state, key)[untouched], expected)
        torch.testing.assert_close(env.extras["terminal_observation"], final_observation)
        torch.testing.assert_close(env.extras["terminal_env_ids"], torch.tensor([1]))
        self.assertEqual(int(state.history[1, 1].count_nonzero()), 0)
        self.assertEqual(int(state.history[1, 0, 54:].count_nonzero()), 0)
        self.assertFalse(state.failure[1])
        env.common_step_counter = 2
        task.get_task_state(env)
        self.assertEqual(env.extras["terminal_env_ids"].numel(), 0)
        torch.testing.assert_close(state.elapsed_time, torch.tensor([.2, .1, .2], dtype=state.elapsed_time.dtype))

    def test_current_object_ground_truth_changes_rewards_but_not_frozen_actor_geometry(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        state = task.get_task_state(env)
        initial_geometry = state.geometry_b.clone()
        object_data = env.scene["object_collection"].data
        object_data.object_link_pose_w[:, 0, 0] += .01
        object_data.object_com_pos_w[:, 0, 0] += .01
        # Subsequent edits to the producer are not accepted until the next reset.
        env.task_scene_data.geometry_b[:, :3, 0] += 100.
        env.common_step_counter = 1
        task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b, initial_geometry)
        torch.testing.assert_close(state.path_length, torch.full((3,), .01), atol=1.e-6, rtol=1.e-5)
        actor_geometry = env.extras["terminal_observation"][:, :100].reshape(3, 10, 10)
        torch.testing.assert_close(actor_geometry, task.geometry_in_eef_frame(
            initial_geometry, state.geometry_valid, state.tcp_pose_b
        ))

    def test_sensor_overrange_nonfinite_effort_and_shelf_motion_each_fail_only_affected_env(self):
        env, action, _ = native_fixture(4)
        task.reset_task(env)
        env.sensors.ideal_wrench[0, 0] = 201.
        action.applied_torque[1, 0] = math.nan
        env.scene["shelf"].data.root_link_pose_w[2, 0] += .02
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([True, True, True, False]))
        torch.testing.assert_close(state.components[:, 4], torch.tensor([-100., -100., -100., 0.]))
        self.assertTrue(torch.isfinite(env.extras["terminal_observation"]).all())

    def test_shelf_reference_changes_only_on_selected_reset(self):
        env, _, _ = native_fixture(4)
        task.reset_task(env)
        state = env._contact_mdp
        initial = state.initial_shelf_pose.clone()
        env.scene['shelf'].data.root_link_pose_w[1, 0] += .02
        env.common_step_counter = 1
        task.get_task_state(env)
        self.assertTrue(state.failure[1])
        task.reset_task(env, torch.tensor([1]))
        torch.testing.assert_close(state.initial_shelf_pose[[0,2,3]], initial[[0,2,3]])
        torch.testing.assert_close(state.initial_shelf_pose[1], env.scene['shelf'].data.root_link_pose_w[1])
        self.assertFalse(state.failure[1])

    def test_nonfinite_robot_measurement_fails_and_terminal_observation_remains_finite(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        env.scene["robot"].data.joint_pos[1, 3] = math.nan
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([False, True, False]))
        self.assertTrue(torch.isfinite(env.extras["terminal_observation"]).all())
        self.assertTrue(torch.isfinite(state.components).all())

    def test_native_adapter_hot_path_does_not_read_tensor_scalars(self):
        env, _, _ = native_fixture(32)
        task.reset_task(env)
        env.common_step_counter = 1
        def reject_scalar(*args, **kwargs):
            raise AssertionError("Per-step Tensor scalar read / host synchronization")
        with patch.object(torch.Tensor, "__bool__", reject_scalar), patch.object(torch.Tensor, "item", reject_scalar):
            state = task.get_task_state(env)
        self.assertEqual(state.history.shape, (32, 2, 72))
        self.assertEqual(env.extras["terminal_observation"].shape, (32, 244))

@wp.kernel(enable_backward=False)
def _layout_math_probe(data: wp.array2d(dtype=wp.float64), regions: wp.array2d(dtype=wp.float64),
                       gap: wp.float64, result: wp.array2d(dtype=wp.float64)):
    row = wp.tid()
    x,y,a,hx,hy = data[row,0],data[row,1],data[row,2],data[row,3],data[row,4]
    ox,oy,oa,ohx,ohy = data[row,5],data[row,6],data[row,7],data[row,8],data[row,9]
    first = task._layout_quad(x,y,a,hx,hy)
    second = task._layout_quad(ox,oy,oa,ohx,ohy)
    result[row,0] = wp.float64(task._layout_boxes_clear(x,y,wp.cos(a),wp.sin(a),hx,hy,
        ox,oy,wp.cos(oa),wp.sin(oa),ohx,ohy,gap))
    result[row,1] = task._layout_distance(first,second)
    result[row,2] = task._layout_area(first,wp.vec4d(regions[row,0],regions[row,1],regions[row,2],regions[row,3]))


class EpisodeLayoutTests(unittest.TestCase):
    def test_packed_path_family_preserves_motion_order_pivots_and_random_stream(self):
        import random
        geometry = task.ObjectGeometry("box", (-.08, -.04, 0.), (.08, .04, .12))
        rules = task.SpawnRules()
        for seed in range(30):
            pose = task.PlanarPose(-.7 + seed*.003, .015*math.sin(seed), seed*.57)
            access = (-.77, -.5, -.1, .1)
            first, second = random.Random(seed), random.Random(seed)
            oracle = task._clearance_candidates(geometry, pose, access, rules, first)
            packed = task._clearance_candidates(geometry, pose, access, rules, second, packed=True)
            self.assertEqual(len(packed), len(oracle))
            self.assertEqual(first.random(), second.random())
            for index, expected in enumerate(oracle):
                self.assertEqual(packed[index], expected)


    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.cache = tempfile.TemporaryDirectory(prefix='contact_layout_tests_')
        task.wp.init()
        cls.previous_cache = task.wp.config.kernel_cache_dir
        task.wp.config.kernel_cache_dir = cls.cache.name

    @classmethod
    def tearDownClass(cls):
        task.wp.config.kernel_cache_dir = cls.previous_cache
        cls.cache.cleanup()

    @staticmethod
    def catalog(copies=3, width_scale=1.1):
        sizes = ((.05, .05, .14), (.085, .06, .19), (.075, .10, .12),
                 (.065, .065, .11), (.10, .08, .17), (.055, .055, .15),
                 (.10, .10, .14), (.06, .16, .21), (.045, .095, .18), (.065, .065, .10))
        # Ten synthetic objects must cover the new 20% minimum without copies.
        return tuple(task.ObjectGeometry(f"type_{kind}_{copy}",
                                        (.013-width_scale*x/2, -.02-y/2, -.007),
                                        (.013+width_scale*x/2, -.02+y/2, z-.007), f"type_{kind}")
                     for kind, (x, y, z) in enumerate(sizes) for copy in range(copies))

    def test_fixed_ten_body_pool_meets_whole_shelf_density_without_copies(self):
        import random
        from collections import Counter
        catalog, rules = self.catalog(copies=1), task.SpawnRules()
        sample = task.sample_scene(catalog, rules, random.Random(1))
        task.validate_sample(sample, catalog, rules)
        metrics = task.scene_metrics(sample, catalog, rules)
        counts = Counter(catalog[index].asset_name for index in sample.active_ids)
        self.assertGreaterEqual(metrics.shelf_occupancy, .20)
        self.assertGreaterEqual(min(metrics.region_occupancies), .10)
        self.assertEqual(counts[catalog[sample.target_id].asset_name], 1)
        self.assertNotEqual(catalog[sample.target_id].asset_name, catalog[sample.blocker_id].asset_name)
        self.assertEqual(max(counts.values()), 1)
        self.assertEqual(len(sample.active_ids), 10)
        self.assertEqual(len(counts), 10)
        self.assertGreaterEqual(min(counts.values()), 1)
        for index in sample.active_ids:
            self.assertAlmostEqual(sample.positions[index][2] + catalog[index].lower[2], rules.shelf_top)

    def test_device_layout_feeds_physics_and_collision_boxes_without_host_pose_reads(self):
        import random
        catalog, rules = self.catalog(), task.SpawnRules()
        sample = task.sample_scene(catalog,rules,random.Random(1))
        # Catalog storage, active slots, and packing order need not be identical.
        order = sample.active_ids[::-1]
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            lower = torch.tensor([g.lower for g in catalog],device=device)
            upper = torch.tensor([g.upper for g in catalog],device=device)
            upright = torch.zeros((len(catalog),4),device=device)
            upright[:,0] = 1
            default = torch.zeros((len(catalog),13),device=device)
            default[:,0],default[:,3] = 5.,1.
            expected = task.pack_episode_states(default,[sample],lower,upper,upright)
            resident = dataclasses.replace(sample,planar_ids=order,planar_poses=torch.tensor(
                [(*sample.positions[i][:2],sample.yaws[i]) for i in order],device=device,dtype=torch.float64))
            cache = {}
            model = SimpleNamespace(base_origin=np.array([.1,.2,0.]),
                shelf_boxes=[SimpleNamespace(center=np.zeros(3),rotation=np.eye(3),half_extents=np.ones(3),path='shelf')],
                _tensor_constants=lambda *args: cache)
            env = SimpleNamespace(robot_pose_model=model,spawn_geometries=catalog)
            old_boxes = task._scene_collision_boxes(env,sample)
            with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('host pose transfer')), \
                 patch.object(torch.Tensor,'numpy',side_effect=AssertionError('host pose transfer')):
                proposed = task._episode_proposals([resident],len(catalog),device,torch.float32)
                actual = task.pack_episode_states(default,[resident],lower,upper,upright,proposals=proposed)
                centers,rotations,half,valid = task._scene_boxes_tensor(env,[resident],proposed)
                # The runtime catalog supplies heights without reading per-object metadata.
                resident_without_positions = dataclasses.replace(resident, positions={})
                cached = task._episode_proposals([resident_without_positions],len(catalog),device,torch.float32,
                    support_heights=lower.new_tensor([rules.shelf_top-g.lower[2] for g in catalog]))
                for a,b in zip(cached,proposed):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for tensor,column in ((centers,0),(rotations,1),(half,2)):
                torch.testing.assert_close(tensor[0],torch.tensor(np.array([b[column] for b in old_boxes]),
                    device=device,dtype=tensor.dtype),rtol=2e-6,atol=2e-7)
            self.assertTrue(valid.all())

    def test_too_small_selected_species_capacity_is_rejected_before_layout(self):
        import random
        with self.assertRaisesRegex(RuntimeError, "cannot meet whole-shelf density"):
            task.sample_scene(self.catalog(copies=4, width_scale=1.), task.SpawnRules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35), random.Random(0))

    def test_layout_rejection_can_retry_composition_while_preserving_episode_roles(self):
        import random
        catalog, rules, rng = self.catalog(copies=4, width_scale=1.), task.SpawnRules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35), random.Random(0)
        target = rng.randrange(len(catalog))
        blocker = rng.choice([i for i, geometry in enumerate(catalog)
                              if geometry.asset_name != catalog[target].asset_name])
        with self.assertRaisesRegex(RuntimeError, "cannot meet whole-shelf density"):
            task.sample_scene(catalog, rules, rng, target_id=target, blocker_id=blocker)
        sample = task.sample_scene(catalog, rules, rng, target_id=target, blocker_id=blocker)
        self.assertEqual((sample.target_id, sample.blocker_id), (target, blocker))
        task.validate_sample(sample, catalog, rules)

    def test_front_start_region_respects_rotated_blocker_and_elbow_geometry(self):
        geometry = (task.ObjectGeometry("target", (-.025, -.025, 0), (.025, .025, .1)),
                    task.ObjectGeometry("blocker", (-.08, -.06, 0), (.12, .04, .1)))
        sample = task.SceneSample((0, 1), 0, 1,
                                  {0: (-.85, .1, 1.05), 1: (-.75, .1, 1.05)},
                                  {0: 0., 1: math.pi/2}, (-.8, -.5, .04, .16), (-.75, -.1))
        bounds = task.robot_start_region(sample, geometry,
                                          dataclasses.replace(task.SpawnRules(), x_range=(-.9, -.5)),
                                          task.RobotStartRules())
        np.testing.assert_allclose(bounds, (-.44, -.30, .02, .18, 1.25, 1.33))
        self.assertGreater(task.elbow_above_chord((0,0,1), (.3,0,1.4), (.6,0,1)), 0)
        self.assertLess(task.elbow_above_chord((0,0,1), (.3,0,.6), (.6,0,1)), 0)

    def test_batched_path_selection_matches_scalar_sweeps(self):
        import random
        catalog = self.catalog()
        rules = dataclasses.replace(task.SpawnRules(), x_range=(-.4,.4), y_range=(-.4,.4))
        rng = random.Random(918)
        requests, expected = [], []
        for k in range(8):
            geometry = catalog[4*k]
            initial = task.PlanarPose(.05,rng.uniform(-.15,.15),rng.uniform(-math.pi,math.pi))
            access = (-.01,.4,-.06,.06)
            target = task.rectangle_polygon((-.12,-.02,-.05,.05))
            candidates = task._clearance_candidates(geometry,initial,access,rules,rng)
            good = [task._path_supported_clear(geometry,path,(target,),rules) for path in candidates]
            requests.append(task._PathRequest(geometry,candidates,target))
            expected.append(next((p for p,v in zip(candidates,good) if v),None))
            # Check every candidate, including rejected mid-rotation collisions.
            requests.extend(task._PathRequest(geometry,[p],target) for p in candidates)
            expected.extend(p if v else None for p,v in zip(candidates,good))
        bar = task.ObjectGeometry('bar',(-.10,-.01,0.),(.10,.01,.1))
        path = (task.PlanarMotion(task.PlanarPose(0.,0.,0.),task.PlanarPose(0.,0.,math.pi/2),(0.,0.)),)
        post = task.rectangle_polygon((.052,.065,.052,.065))
        requests.append(task._PathRequest(bar,[path],post))
        expected.append(None)  # Clear endpoints, but collision halfway through the pivot.
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            self.assertEqual(task._choose_clearance_paths(requests,rules,device),expected)

    def test_pivot_sweep_detects_mid_motion_obstacle(self):
        # Endpoints clear a thin post, but the rotating long bar crosses it.
        bar = task.ObjectGeometry("bar", (-.10, -.01, 0), (.10, .01, .1))
        motion = task.PlanarMotion(task.PlanarPose(0., 0., 0.), task.PlanarPose(0., 0., math.pi/2), (0., 0.))
        obstacle = task.rectangle_polygon((.052, .065, .052, .065))
        self.assertTrue(task.polygons_separated(task.footprint_polygon(bar, (0., 0.), 0.), obstacle))
        self.assertTrue(task.polygons_separated(task.footprint_polygon(bar, (0., 0.), math.pi/2), obstacle))
        self.assertTrue(any(not task.polygons_separated(sweep, obstacle)
                            for sweep in task.motion_sweeps(bar, motion)))

    def test_device_geometry_matches_independent_scalar_oracles_and_gap_boundaries(self):
        rng = np.random.default_rng(88)
        data = rng.uniform(-1.,1.,(128,10))
        data[:,[3,4,8,9]] = rng.uniform(.01,.14,(128,4))
        data[:4] = [0.,0.,0.,.03,.01,.06,0.,0.,.03,.01]
        data[:4,5] += np.array([-.00001,0.,.003,.003-2.e-9])
        regions = np.tile([-.3,.3,-.3,.3],(len(data),1))
        pairs=[]
        for row in data:
            a=task.ObjectGeometry('a',(-row[3],-row[4],0.),(row[3],row[4],.1))
            b=task.ObjectGeometry('b',(-row[8],-row[9],0.),(row[8],row[9],.1))
            pairs.append((task.footprint_polygon(a,tuple(row[:2]),float(row[2])),
                          task.footprint_polygon(b,tuple(row[5:7]),float(row[7]))))
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for gap in (0.,.003,.017):
                output=wp.empty((len(data),3),dtype=wp.float64,device=device)
                wp.launch(_layout_math_probe,dim=len(data),inputs=[wp.array(data,dtype=wp.float64,device=device),
                    wp.array(regions,dtype=wp.float64,device=device),gap,output],device=device)
                actual=output.numpy()
                expected=np.array([[task.polygons_separated(a,b,gap),task.polygon_distance(a,b),
                    task.overlap_area(a,task.rectangle_polygon(region))] for (a,b),region in zip(pairs,regions)])
                np.testing.assert_array_equal(actual[:,0],expected[:,0])
                # Distance helper is used only after collision separation.
                separated=actual[:,0].astype(bool)
                np.testing.assert_allclose(actual[separated,1],expected[separated,1],atol=1.e-12,rtol=1.e-12)
                np.testing.assert_allclose(actual[:,2],expected[:,2],atol=1.e-12,rtol=1.e-12)

    def test_local_repair_preserves_roles_and_all_constraints(self):
        import random
        # Keep a dense multi-copy case to exercise eviction/repair, even though
        # the current ten-object experiment needs fewer placements.
        catalog = self.catalog(width_scale=1.)
        rules = task.SpawnRules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35)
        stats={}
        result=task.sample_scenes(catalog,rules,[random.Random(50+i) for i in range(8)],
                                 [(0,4)]*8,[None]*8,statistics=stats)
        self.assertGreater(stats['evictions'],0)
        for sample in result:
            self.assertEqual((sample.target_id,sample.blocker_id),(0,4))
            task.validate_sample(sample,catalog,rules)

    def test_degenerate_sweep_edges_do_not_change_packing(self):
        import random
        catalog, rules = self.catalog(), task.SpawnRules()
        reference = task.sample_scene(catalog,rules,random.Random(1))
        plans=[]
        for seed in range(4):
            generator=task._scene_proposals(catalog,rules,random.Random(seed),reference.target_id,
                reference.blocker_id,reference.active_ids)
            plans.append(next(generator))
            generator.close()
        modified=[]
        for plan in plans:
            sweeps=[]
            for polygon in plan[5]:
                start, following=np.asarray(polygon[:2])
                tiny=tuple(start+(following-start)*1.e-10)
                sweeps.append((polygon[0],polygon[0],tiny,*polygon[1:]))
            modified.append((*plan[:5],tuple(sweeps),plan[-1]))
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            result=task._pack_layouts(plans+modified,catalog,rules,device)
            self.assertTrue(any(row is not None for row in result[:4]))
            for before,after in zip(result[:4],result[4:]):
                if before is None:
                    self.assertIsNone(after)
                else:
                    np.testing.assert_array_equal(before,after)

    def test_compiled_rotation_envelopes_match_scalar_hulls(self):
        box = task.ObjectGeometry('box', (-.10, -.02, 0.), (.10, .02, .1))
        for angle in (-math.pi / 2, math.pi / 4, math.pi):
            for pivot in (None, (.10, -.02)):
                motion = task.PlanarMotion(task.PlanarPose(.01, -.03, .2),
                                          task.PlanarPose(.04, .07, .2 + angle), pivot)
                previous = task._pose_polygon(box, motion.start)
                origin = pivot or (motion.start.x, motion.start.y)
                radius = max(math.hypot(x - origin[0], y - origin[1]) for x, y in previous)
                count = math.ceil(abs(angle) / math.radians(5.))
                padding = radius * (1. - math.cos(abs(angle) / (2 * count)))
                expected = []
                for step in range(1, count + 1):
                    current = task._pose_polygon(box, task._motion_pose(motion, step / count))
                    hull = task._convex_hull(previous + current)
                    expected.append(task._convex_hull(tuple((x + dx, y + dy) for x, y in hull
                        for dx, dy in ((-padding, -padding), (padding, -padding),
                                       (padding, padding), (-padding, padding)))))
                    previous = current
                self.assertEqual(task.motion_sweeps(box, motion), tuple(expected))

    def test_array_sweeps_match_scalar_pivot_translation_and_multistage_paths(self):
        box = task.ObjectGeometry('offset', (-.13, -.04, -.02), (.07, .02, .1))
        for angle in (0., 1.e-10, -math.pi/2, math.pi/4, math.pi):
            for pivot in (None, (.1, -.02)):
                motion = task.PlanarMotion(task.PlanarPose(.01, -.03, .2),
                    task.PlanarPose(.04, .07, .2+angle), pivot)
                path = (motion, task.PlanarMotion(motion.end, task.PlanarPose(.09, .12, motion.end.yaw)))
                expected = task._path_sweeps(box,path)
                actual = task._path_sweeps(box,path,packed=True)
                self.assertEqual(len(expected),len(actual))
                for polygon, vertices, count in zip(expected,actual.vertices,actual.counts):
                    np.testing.assert_allclose(vertices[:count],polygon,atol=1.e-14,rtol=1.e-14)

    def test_array_sweep_packing_matches_tuple_packing(self):
        import random
        catalog, rules = self.catalog(), task.SpawnRules()
        reference=task.sample_scene(catalog,rules,random.Random(1))
        plans=[]
        for seed in range(4):
            generator=task._scene_proposals(catalog,rules,random.Random(seed),reference.target_id,
                reference.blocker_id,reference.active_ids)
            plans.append(next(generator))
            generator.close()
        packed=[(*p[:5],task._path_sweeps(catalog[p[0][1]],p[4],packed=True),p[-1]) for p in plans]
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            result=task._pack_layouts(plans+packed,catalog,rules,device)
            self.assertTrue(any(row is not None for row in result[:4]))
            for before,after in zip(result[:4],result[4:]):
                if before is None:
                    self.assertIsNone(after)
                else:
                    np.testing.assert_array_equal(before,after)

    def test_tensor_start_regions_match_rotated_footprint_bounds(self):
        rng=np.random.default_rng(832)
        catalog=[task.ObjectGeometry(str(i),(-.02-i*.0005,-.04,0.),(.06,.03,.12)) for i in range(64)]
        spawn,rules=task.SpawnRules(),task.RobotStartRules()
        poses=np.column_stack((rng.uniform(-.65,-.52,64),rng.uniform(-.3,.3,64),
                               np.full(64,1.05),rng.uniform(-math.pi,math.pi,64)))
        expected=[]
        for i,pose in enumerate(poses):
            sample=task.SceneSample((i,),i,i,{i:tuple(pose[:3])},{i:pose[3]},(-1.,1.,-1.,1.),(0.,0.))
            expected.append(task.robot_start_region(sample,catalog,spawn,rules))
        half=(np.array([g.upper for g in catalog])-np.array([g.lower for g in catalog]))*.5
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for dtype in (torch.float64,torch.float32):
                actual=task._robot_start_regions(torch.tensor(poses,device=device,dtype=dtype),
                    torch.tensor(half,device=device,dtype=dtype),spawn,rules)
                torch.testing.assert_close(actual,torch.tensor(expected,device=device,dtype=dtype),
                    atol=1.e-7 if dtype==torch.float32 else 1.e-14,rtol=1.e-7 if dtype==torch.float32 else 1.e-14)

    def test_blocker_interval_matches_dense_independent_polygon_clipping(self):
        rng = np.random.default_rng(172)
        box = task.ObjectGeometry('blocker',(-.08,-.04,0.),(.08,.04,.1))
        access = np.array([-.02,.4,-.06,.06])
        for yaw in rng.uniform(-math.pi,math.pi,32):
            polygon = np.asarray(task.footprint_polygon(box,(.07,0.),float(yaw)))
            required = task.planar_polygon_area(tuple(map(tuple,polygon)))*.3
            low,high = task._blocker_y_interval(polygon,access,-.2,.2,required)
            self.assertLess(low,high)
            for y in np.linspace(low,high,15):
                actual = task.overlap_area(tuple(map(tuple,polygon+np.array([0.,y]))),task.rectangle_polygon(access))
                self.assertGreaterEqual(actual,required-1.e-9)
            for y in (-.2,low-1.e-5,high+1.e-5,.2):
                if -.2 <= y <= .2 and (y < low-1.e-7 or y > high+1.e-7):
                    actual = task.overlap_area(tuple(map(tuple,polygon+np.array([0.,y]))),task.rectangle_polygon(access))
                    self.assertLess(actual,required+1.e-9)
        low,high = task._blocker_y_interval(polygon,access,-.2,.2,1.)
        self.assertGreater(low,high)

    def test_resident_layouts_preserve_each_random_stream_and_subset_order(self):
        import random
        catalog, rules = self.catalog(), task.SpawnRules()
        def generate(indices, batch_size, device='cpu'):
            streams = [random.Random(50+i) for i in indices]
            result = task.sample_scenes(catalog,rules,streams,[(0,4)]*len(indices),
                                        [None]*len(indices),device=device,batch_size=batch_size)
            return result,[rng.getstate() for rng in streams]
        expected,states = generate(list(range(8)),1)
        actual,final_states = generate(list(reversed(range(8))),8)
        self.assertEqual(actual,expected[::-1])
        self.assertEqual(final_states,states[::-1])
        for sample in actual:
            task.validate_sample(sample,catalog,rules)
        if torch.cuda.is_available():
            gpu,_ = generate(list(range(8)),8,'cuda:0')
            for sample in gpu:
                task.validate_sample(sample,catalog,rules)
            # Independent device math may choose a different tied candidate;
            # within each device, ordering and batching cannot affect a seed.
            reversed_gpu,_ = generate(list(reversed(range(8))),3,'cuda:0')
            self.assertEqual(reversed_gpu,gpu[::-1])

    def test_two_object_configuration_and_fixed_composition_are_supported(self):
        import random
        catalog=self.catalog()[:1]+self.catalog()[4:5]
        rules=task.SpawnRules(min_objects=2,max_objects=2,min_asset_types=2,max_asset_types=2,all_asset_types=False,
            min_shelf_occupancy=0.,min_region_occupancy=0.,min_x_coverage=0.,min_y_coverage=0.,
            min_neighbor_fraction=0.)
        sample=task.sample_scene(catalog,rules,random.Random(44),0,1,(0,1))
        task.validate_sample(sample,catalog,rules)
        self.assertEqual(sample.active_ids,(0,1))

    def test_compiled_regional_overlap_matches_scalar_clipping_oracle(self):
        rng = np.random.default_rng(805)
        geometry = task.ObjectGeometry("box", (-.09, -.04, 0.), (.09, .04, .1))
        rules = task.SpawnRules()
        regions = task.shelf_regions(rules)
        polygons = [task.footprint_polygon(geometry, (float(x), float(y)), float(yaw))
                    for x, y, yaw in rng.uniform((-.9, -.5, -math.pi), (-.5, .5, math.pi), size=(96, 3))]
        expected = np.asarray([[task.overlap_area(polygon, task.rectangle_polygon(region)) for region in regions]
                               for polygon in polygons])
        actual = task._footprint_region_areas(np.asarray(polygons), np.asarray(regions))
        np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=2.e-16)
        self.assertEqual(task._footprint_region_areas(np.empty((0, 4, 2)), np.asarray(regions)).shape, (0, 6))

    def test_batched_placement_is_deterministic_and_independent_across_episodes(self):
        import random
        catalog, rules = self.catalog(), task.SpawnRules()
        first = task.sample_scene(catalog, rules, random.Random(1))
        repeated = task.sample_scene(catalog, rules, random.Random(1))
        self.assertEqual(first, repeated)
        other = task.sample_scene(catalog, rules, random.Random(103),
                                  target_id=first.target_id, blocker_id=first.blocker_id,
                                  active_ids=first.active_ids)
        self.assertEqual(other.active_ids, first.active_ids)
        self.assertNotEqual(other.positions, first.positions)
        task.validate_sample(other, catalog, rules)



def episode_reset_fixture(batch=5, selected=2):
    """Native asset-shaped buffers exposing indexed writes, without a simulator."""
    class Objects:
        def __init__(self):
            self.state = torch.full((batch, 4, 13), -7.)
            self.parked_mask = torch.zeros(batch, 4, dtype=torch.bool)
            self.writes = 0

        def refresh_parking(self):
            pass

        def write_object_state_to_sim(self, value, env_ids):
            self.state[env_ids] = value
            self.writes += 1

    class Robot:
        def __init__(self):
            default = torch.zeros(batch, 13)
            default[:, 0], default[:, 3], default[:, 7:] = .2, 1., 8.
            self.data = SimpleNamespace(default_root_state=default)
            self.root = torch.full((batch, 13), -8.)
            self.position = torch.full((batch, 12), -9.)
            self.velocity = torch.full_like(self.position, -10.)
            self.position_target = torch.full_like(self.position, -11.)
            self.velocity_target = torch.full_like(self.position, -12.)
            self.effort_target = torch.full_like(self.position, -13.)
            self.root_writes = self.joint_writes = 0

        def write_root_state_to_sim(self, value, env_ids):
            self.root[env_ids] = value
            self.root_writes += 1

        def write_joint_state_to_sim(self, position, velocity, env_ids):
            self.position[env_ids], self.velocity[env_ids] = position, velocity
            self.joint_writes += 1

        def set_joint_position_target(self, value, env_ids):
            self.position_target[env_ids] = value

        def set_joint_velocity_target(self, value, env_ids):
            self.velocity_target[env_ids] = value

        def set_joint_effort_target(self, value, env_ids):
            self.effort_target[env_ids] = value

    class Scene(dict):
        def __init__(self):
            super().__init__(object_collection=Objects(), robot=Robot(), shelf=Robot())
            self.env_origins = torch.zeros(batch, 3)
            self.env_origins[:, 0] = 2.5 * torch.arange(batch)
            self.env_origins[:, 1] = -1.5 * torch.arange(batch)
            self.writes = 0

        def write_data_to_sim(self):
            self.writes += 1

    runtime = SimpleNamespace(
        reset_counts=torch.zeros(batch, dtype=torch.long),
        robot_regions=torch.full((batch, 6), -21.), robot_tcp=torch.full((batch, 3), -22.),
        visibility=torch.full((batch, 2), -23.),
        camera_pose=torch.full((batch, 7), -24.), camera_intrinsics=torch.full((batch, 3, 3), -25.),
        episode_seeds=torch.full((batch,), -1, dtype=torch.long),
    )
    scene_data = SimpleNamespace(
        object_ids=torch.full((batch, 10), -1, dtype=torch.long),
        target_id=torch.full((batch,), -1, dtype=torch.long), blocker_id=torch.full((batch,), -1, dtype=torch.long),
        access_region=torch.full((batch, 4), -31.), geometry_b=torch.full((batch, 10, 10), -32.),
        geometry_valid=torch.zeros(batch, 10, dtype=torch.bool),
        observed_ids=torch.full((batch, 10), -1, dtype=torch.long),
    )
    from unittest.mock import Mock
    env = SimpleNamespace(num_envs=batch, device='cpu', scene=Scene(), task_scene_data=scene_data,
                          _episode_reset=runtime, task_sensor_reader=SimpleNamespace(reset=Mock()),
                          sim=SimpleNamespace(forward=Mock()))
    rows = torch.arange(selected)
    object_state = torch.zeros(selected, 4, 13)
    object_state[:, :, 0] = torch.arange(4)[None] * .1 + rows[:, None] * .013
    object_state[:, :, 2], object_state[:, :, 3] = 1.05, 1.
    slots = torch.full((selected, 10), -1, dtype=torch.long)
    # Vary roles and inactive bodies by episode, so accidental broadcasting fails.
    slots[:, :3] = (torch.tensor([2, 0, 3])[None] + rows[:, None]) % 4
    active = torch.zeros(selected, 4, dtype=torch.bool)
    active.scatter_(1, slots[:, :3], True)
    geometry = torch.zeros(selected, 10, 10)
    geometry[:, :, 0] = rows[:, None] + 100.
    geometry[:, :, 3], geometry[:, :, 7:] = 1., .1
    valid = torch.zeros(selected, 10, dtype=torch.bool)
    valid[:, :3] = True
    camera_pose = torch.zeros(selected, 7)
    camera_pose[:, 0], camera_pose[:, 3] = rows * .1, 1.
    episodes = task.EpisodeBatch(
        object_state=object_state, object_ids=slots, active=active,
        target_id=slots[:, 0].clone(), blocker_id=slots[:, 1].clone(),
        access_region=torch.tensor([-.8, -.5, -.1, .1])[None].repeat(selected, 1) + rows[:, None] * .001,
        joint_positions=torch.arange(selected * 12).reshape(selected, 12).float() / 100.,
        geometry_b=geometry, geometry_valid=valid, observed_ids=slots.clone(),
        robot_regions=torch.arange(selected * 6).reshape(selected, 6).float() / 10.,
        robot_tcp=torch.tensor([-.4, .1, 1.25])[None].repeat(selected, 1) + rows[:, None] * .01,
        visibility=torch.tensor([.45, .99])[None].repeat(selected, 1) + rows[:, None].remainder(4) * .001,
        camera_pose=camera_pose,
        camera_intrinsics=torch.eye(3)[None].repeat(selected, 1, 1) * (rows[:, None, None] + 1),
        seeds=rows + 43,
    )
    return env, episodes


class EpisodeResetTests(unittest.TestCase):
    @staticmethod
    def batch_fixture(geometries, rules, streams, roles, active_ids, **kwargs):
        return [task.sample_scene(geometries,rules,rng,target_id=t,blocker_id=b,active_ids=ids)
                for rng,(t,b),ids in zip(streams,roles,active_ids)]

    @staticmethod
    def builder_fixture():
        """Real proposal/commit buffers with rendering and collision search substituted."""
        from contextlib import nullcontext
        from unittest.mock import Mock

        env, _ = episode_reset_fixture()
        objects, robot = env.scene['object_collection'], env.scene['robot']
        default = torch.zeros(env.num_envs, 4, 13)
        default[:, :, 3] = 1.
        objects.data = SimpleNamespace(object_state_w=objects.state, default_object_state=default)
        for field, value in (
            ('root_state_w', robot.root), ('joint_pos', robot.position), ('joint_vel', robot.velocity),
            ('joint_pos_target', robot.position_target), ('joint_vel_target', robot.velocity_target),
            ('joint_effort_target', robot.effort_target),
        ):
            setattr(robot.data, field, value)
        robot.root_physx_view = SimpleNamespace(get_link_transforms=Mock(return_value=torch.zeros(5, 1, 7)))
        env.cfg = SimpleNamespace(seed=73, target_asset=None, blocker_asset=None,
                                  initial_vision=SimpleNamespace(max_layout_attempts=2))
        env.spawn_rules, env.spawn_geometries = task.SpawnRules(), ()
        env.task_scene_data.lower = torch.tensor([[-.05, -.05, 0.]]).repeat(4, 1)
        env.task_scene_data.upper = torch.tensor([[.05, .05, .1]]).repeat(4, 1)
        env.task_scene_data.upright = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        env._episode_reset.specs = tuple(SimpleNamespace(asset_name=f'type_{index}') for index in range(4))
        env._episode_reset.vision = SimpleNamespace(capture_session=nullcontext, capture_batch=Mock())
        proposal_count = 0

        def sample_scene(geometries, rules, rng, *, target_id, blocker_id, active_ids):
            nonlocal proposal_count
            proposal_count += 1
            return task.SceneSample(
                (0, 1, 2, 3), target_id, blocker_id,
                {index: (-.8 + proposal_count * .01, index * .03, 1.05) for index in range(4)},
                {index: 0. for index in range(4)}, (-.8, -.5, -.1, .1), (-.7, -.2),
            )

        def sample_robot_start(env, samples, seeds, **kwargs):
            return [(np.full(12, sample.positions[0][0]), (-.4, -.2, -.1, .1, 1.2, 1.3),
                     np.array([-.3, 0., 1.25]), .08, 1) for sample in samples]

        def snapshot(env_id, object_pose=None):
            geometry = torch.zeros(10, 10)
            geometry[:4, :3] = (objects.state[env_id, :, :3] if object_pose is None else object_pose[:, :3]) - env.scene.env_origins[env_id]
            geometry[:4, 3], geometry[:4, 7:] = 1., .1
            valid, observed = torch.zeros(10, dtype=torch.bool), torch.full((10,), -1, dtype=torch.long)
            valid[:4], observed[:4] = True, torch.arange(4)
            return SimpleNamespace(
                geometry_b=geometry, valid=valid, observed_ids=observed, visible_fractions=(.5, 1.),
                camera_pose_w=torch.tensor([env_id, 0., 2., 1., 0., 0., 0.]), intrinsics=torch.eye(3),
            )

        return env, sample_scene, sample_robot_start, snapshot

    def test_packing_applies_upright_once_corrects_off_center_roots_and_clears_all_velocity(self):
        default = torch.zeros(4, 13)
        default[:, :3] = torch.tensor([7., 8., 9.])
        default[:, 3], default[:, 7:] = 1., 42.
        lower = torch.tensor([[-.01, -.04, -.02], [-.1, -.1, 0.], [-.02, -.06, -.01], [-.1, -.1, 0.]])
        upper = torch.tensor([[.05, .02, .10], [.1, .1, .1], [.08, .02, .2], [.1, .1, .1]])
        upright = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        upright[2] = torch.tensor([math.sqrt(.5), -math.sqrt(.5), 0., 0.])
        sample = task.SceneSample((2, 0), 2, 0, {2: (-.7, .1, 1.06), 0: (-.8, -.1, 1.07)},
                                  {2: math.pi/2, 0: 0.}, (-.8, -.5, -.1, .1), (-.7, -.2))
        states, active, slots = task.pack_object_states(default, sample, lower, upper, upright)
        torch.testing.assert_close(states[2, :3], torch.tensor([-.72, .07, 1.06]))
        torch.testing.assert_close(states[0, :3], torch.tensor([-.82, -.09, 1.07]))
        expected_xyzw = (Rotation.from_euler('z', math.pi/2) * Rotation.from_euler('x', -math.pi/2)).as_quat()
        expected_wxyz = torch.tensor(expected_xyzw[[3, 0, 1, 2]], dtype=states.dtype)
        torch.testing.assert_close(states[2, 3:7], expected_wxyz)
        torch.testing.assert_close(states[[1, 3], :7], default[[1, 3], :7])
        self.assertEqual(int(states[:, 7:].count_nonzero()), 0)
        self.assertTrue(torch.equal(active, torch.tensor([True, False, True, False])))
        self.assertTrue(torch.equal(slots[:2], torch.tensor([2, 0])))
        self.assertTrue((slots[2:] == -1).all())
        self.assertTrue((default[:, 7:] == 42.).all())

    def test_independent_reset_writes_unsorted_selected_rows_and_clears_physical_targets(self):
        env, episodes = episode_reset_fixture()
        ids = torch.tensor([3, 1])
        untouched = torch.tensor([0, 2, 4])
        object_state_before = episodes.object_state.clone()
        task.apply_episode_batch(env, ids, episodes)
        objects, robot, data = env.scene['object_collection'], env.scene['robot'], env.task_scene_data
        expected = object_state_before.clone()
        expected[..., :3] += env.scene.env_origins[ids, None]
        torch.testing.assert_close(objects.state[ids], expected)
        torch.testing.assert_close(episodes.object_state, object_state_before)
        self.assertTrue((objects.state[untouched] == -7.).all())
        self.assertTrue((robot.root[untouched] == -8.).all())
        self.assertTrue((robot.position[untouched] == -9.).all())
        self.assertTrue((robot.velocity[untouched] == -10.).all())
        self.assertTrue((robot.position_target[untouched] == -11.).all())
        self.assertTrue((robot.velocity_target[untouched] == -12.).all())
        self.assertTrue((robot.effort_target[untouched] == -13.).all())
        torch.testing.assert_close(robot.root[ids, :3], env.scene.env_origins[ids] + torch.tensor([.2, 0., 0.]))
        torch.testing.assert_close(robot.position[ids], episodes.joint_positions)
        torch.testing.assert_close(robot.position_target[ids], episodes.joint_positions)
        for tensor in (robot.root[ids, 7:], robot.velocity[ids], robot.velocity_target[ids], robot.effort_target[ids]):
            self.assertEqual(int(tensor.count_nonzero()), 0)
        torch.testing.assert_close(objects.parked_mask[ids], ~episodes.active)
        self.assertEqual(int(objects.parked_mask[untouched].count_nonzero()), 0)
        for field in ('object_ids', 'target_id', 'blocker_id', 'access_region',
                      'geometry_b', 'geometry_valid', 'observed_ids'):
            torch.testing.assert_close(getattr(data, field)[ids], getattr(episodes, field))
        self.assertTrue((data.geometry_b[untouched] == -32.).all())
        self.assertTrue((data.object_ids[untouched] == -1).all())
        self.assertTrue((data.target_id[untouched] == -1).all())
        self.assertTrue((data.blocker_id[untouched] == -1).all())
        self.assertTrue((data.access_region[untouched] == -31.).all())
        self.assertFalse(data.geometry_valid[untouched].any())
        self.assertTrue((data.observed_ids[untouched] == -1).all())
        torch.testing.assert_close(env._episode_reset.robot_tcp[ids], episodes.robot_tcp + env.scene.env_origins[ids])
        for field in ('robot_regions', 'visibility', 'camera_pose', 'camera_intrinsics'):
            torch.testing.assert_close(getattr(env._episode_reset, field)[ids], getattr(episodes, field))
        torch.testing.assert_close(env._episode_reset.episode_seeds[ids], episodes.seeds)
        self.assertTrue((env._episode_reset.robot_tcp[untouched] == -22.).all())
        self.assertTrue((env._episode_reset.camera_pose[untouched] == -24.).all())
        self.assertTrue((env._episode_reset.episode_seeds[untouched] == -1).all())
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.tensor([0, 1, 0, 1, 0]))
        # Partial resets must not flush force buffers for untouched environments.
        self.assertEqual((objects.writes, robot.root_writes, robot.joint_writes, env.scene.writes), (1, 1, 1, 0))
        env.task_sensor_reader.reset.assert_called_once()
        torch.testing.assert_close(env.task_sensor_reader.reset.call_args.args[0], ids)
        env.sim.forward.assert_called_once()

    def test_batch_application_has_no_tensor_scalar_reads_or_per_environment_writes(self):
        env, episodes = episode_reset_fixture(1024, selected=512)
        ids = torch.arange(0, 1024, 2)
        def reject(*args, **kwargs):
            raise AssertionError('Host tensor read in batched episode application')
        with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'tolist', reject), patch.object(torch.Tensor, 'cpu', reject), \
             patch.object(torch.Tensor, 'numpy', reject):
            task.apply_episode_batch(env, ids, episodes)
        self.assertEqual(env.scene['object_collection'].writes, 1)
        self.assertEqual(env.scene['robot'].joint_writes, 1)
        torch.testing.assert_close(env._episode_reset.episode_seeds[ids], episodes.seeds)
        torch.testing.assert_close(env.scene['robot'].position[ids], episodes.joint_positions)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 512)

    @staticmethod
    def shared_fixture():
        env, template = episode_reset_fixture(selected=1)
        env.common_step_counter, env.max_episode_length = 0, 400
        env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
        env.cfg = SimpleNamespace(seed=73, initial_vision=SimpleNamespace(max_layout_attempts=2))
        env._episode_reset.specs = (None,) * 4
        env._episode_reset.support_heights = torch.ones(4)
        sample = task.SceneSample(
            (2, 0, 3), 2, 0, {i: (-.7, i * .03, 1.05) for i in (2, 0, 3)},
            {i: 0. for i in (2, 0, 3)}, (-.8, -.5, -.1, .1), (-.7, -.2))
        return env, template, sample

    @staticmethod
    def shared_starts(env, samples, seeds, **kwargs):
        return [(torch.full((12,), (seed % 10007) / 10007.), torch.arange(6).float(),
                 torch.tensor([-.3, (seed % 97) / 1000., 1.25]), .08, 1) for seed in seeds]

    def test_shared_scene_initializes_once_and_partial_reset_only_reruns_ik(self):
        env, template, sample = self.shared_fixture()
        with patch.object(task, '_build_episode_batch', return_value=(template, sample)) as build, \
             patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts) as ik:
            task.reset_scene(env)
            before = env.scene['object_collection'].state.clone()
            joints = env.scene['robot'].position.clone()
            vision = env.task_scene_data.geometry_b.clone()
            # Physical disturbance must be restored, not cached as the next layout.
            env.scene['object_collection'].state[4, :, :3] += .1
            env.common_step_counter = 17
            task.reset_scene(env, torch.tensor([4, 1]))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.call_args.kwargs, {'scene_generation': 0})
        torch.testing.assert_close(build.call_args.args[2], torch.tensor([0]))
        self.assertEqual(ik.call_count, 2)
        self.assertEqual(len(ik.call_args.args[1]), 2)
        torch.testing.assert_close(env.scene['object_collection'].state, before)
        torch.testing.assert_close(env.task_scene_data.geometry_b, vision)
        torch.testing.assert_close(env.scene['robot'].position[[0, 2, 3]], joints[[0, 2, 3]])
        self.assertFalse(torch.equal(env.scene['robot'].position[[4, 1]], joints[[4, 1]]))
        self.assertEqual(env.scene['robot'].position.unique(dim=0).shape[0], 5)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.tensor([1, 2, 1, 1, 2]))
        self.assertEqual(env._episode_reset.next_scene_step, 400)
        # Local camera template must not accumulate world origins on repeated resets.
        torch.testing.assert_close(env._episode_reset.camera_pose[:, :3],
                                   template.camera_pose[:, :3] + env.scene.env_origins)
        torch.testing.assert_close(env.task_scene_data.geometry_b,
                                   template.geometry_b.expand(5, -1, -1))

    def test_shared_round_boundary_truncates_recently_reset_rows_and_replaces_layout(self):
        from contact_manipulation.mdp.terminations import scene_time_out
        env, first, sample = self.shared_fixture()
        _, second = episode_reset_fixture(selected=1)
        second.object_state[:, :, 1] += .027
        second.geometry_b[:, :, 1] += .031
        with patch.object(task, '_build_episode_batch', side_effect=[(first, sample), (second, sample)]) as build, \
             patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts):
            task.reset_scene(env)
            env.common_step_counter = 399
            task.reset_scene(env, torch.tensor([4, 1]))
            self.assertFalse(scene_time_out(env).any())
            env.common_step_counter = 400
            self.assertTrue(scene_time_out(env).all())
            task.reset_scene(env, torch.tensor([4, 3, 2, 1, 0]))
        self.assertEqual(build.call_count, 2)
        self.assertEqual(build.call_args.kwargs, {'scene_generation': 1})
        self.assertEqual(env._episode_reset.next_scene_step, 800)
        self.assertFalse(scene_time_out(env).any())
        torch.testing.assert_close(env.task_scene_data.geometry_b, second.geometry_b.expand(5, -1, -1))
        local = env.scene['object_collection'].state[..., :3] - env.scene.env_origins[:, None]
        torch.testing.assert_close(local, second.object_state[..., :3].expand(5, -1, -1))

    def test_partial_reset_cannot_create_or_replace_a_shared_scene(self):
        env, _, _ = self.shared_fixture()
        with patch.object(task, '_build_episode_batch') as build:
            with self.assertRaisesRegex(RuntimeError, 'all environments reset together'):
                task.reset_scene(env, torch.tensor([4, 1]))
        build.assert_not_called()
        self.assertEqual(env.scene['object_collection'].writes, 0)

    def test_shared_robot_retry_keeps_accepted_rows_and_reuses_perception(self):
        env, template, sample = self.shared_fixture()
        calls = []
        def starts(env, samples, seeds, **kwargs):
            calls.append(seeds)
            result = self.shared_starts(env, samples, seeds)
            if len(calls) == 1:
                result[1] = None
            return result
        with patch.object(task, '_sample_robot_starts', side_effect=starts):
            batch = task._shared_reset_batch(env, env._episode_reset, torch.tensor([4, 1]), template, sample)
        self.assertEqual([len(seeds) for seeds in calls], [2, 1])
        expected = self.shared_starts(env, [sample], calls[0][:1])[0][0]
        torch.testing.assert_close(batch.joint_positions[0], expected)
        torch.testing.assert_close(batch.geometry_b, template.geometry_b.expand(2, -1, -1))
        self.assertEqual(env.scene['object_collection'].writes, 0)

    def test_shared_proposals_are_reused_until_the_template_changes(self):
        env, template, sample = self.shared_fixture()
        with patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts), \
             patch.object(task, '_episode_proposals', wraps=task._episode_proposals) as propose:
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([4, 1]), template, sample)
            cached = env._episode_reset.shared_inputs
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([2]), template, sample)
            self.assertIs(env._episode_reset.shared_inputs, cached)
            self.assertEqual(propose.call_count, 1)
            replacement = dataclasses.replace(template, geometry_b=template.geometry_b.clone())
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([0]), replacement, sample)
            self.assertIsNot(env._episode_reset.shared_inputs, cached)
            self.assertEqual(propose.call_count, 2)

    def test_shared_ik_exhaustion_does_not_commit_or_change_cache(self):
        env, template, sample = self.shared_fixture()
        env._episode_reset.shared_template, env._episode_reset.shared_sample = template, sample
        env._episode_reset.next_scene_step = 400
        with patch.object(task, '_sample_robot_starts', return_value=[None, None]):
            with self.assertRaisesRegex(RuntimeError, 'robot-start search exhausted'):
                task.reset_scene(env, torch.tensor([4, 1]))
        self.assertEqual(env.scene['object_collection'].writes, 0)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)
        self.assertIs(env._episode_reset.shared_template, template)

    def test_episode_seeds_are_reproducible_and_independent_of_reset_batch_order(self):
        expected = {(index, count): task._episode_seed(123, index, count)
                    for index in range(1024) for count in range(3)}
        self.assertEqual(len(set(expected.values())), 3072)
        for index in (1023, 7, 0, 83):
            for count in (2, 0, 1):
                value = task._episode_seed(123, index, count)
                self.assertEqual(value, expected[index, count])
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, 2**63)
        self.assertNotEqual(task._episode_seed(123, 7, 2), task._episode_seed(124, 7, 2))

    def test_builder_retries_only_rejected_rows_and_retains_corresponding_initial_vision(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        ids = torch.tensor([4, 1])
        captures, accepted_geometry = [], {}
        objects = env.scene['object_collection']
        untouched_before = objects.state[[0, 2, 3]].clone()

        def capture(**kwargs):
            env_id = kwargs['env_id']
            captures.append(env_id)
            if env_id == 1 and captures.count(1) == 1:
                return None
            result = snapshot(env_id, kwargs.get("object_pose"))
            accepted_geometry[env_id] = result.geometry_b.clone()
            return result

        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            capture(**request, object_pose=pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene) as propose, \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_check_physical_robot_starts', side_effect=lambda env,samples,*a: torch.ones(len(samples),dtype=torch.bool)):
            episodes = task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(captures, [4, 1, 1])
        self.assertEqual(propose.call_count, 3)
        self.assertEqual(objects.writes, 2)
        self.assertEqual(env.scene['robot'].root_physx_view.get_link_transforms.call_count, 2)
        torch.testing.assert_close(episodes.geometry_b[0], accepted_geometry[4])
        torch.testing.assert_close(episodes.geometry_b[1], accepted_geometry[1])
        torch.testing.assert_close(episodes.geometry_b[:, :4, :3], episodes.object_state[:, :, :3])
        self.assertNotEqual(float(episodes.object_state[0, 0, 0]), float(episodes.object_state[1, 0, 0]))
        torch.testing.assert_close(objects.state[[0, 2, 3]], untouched_before)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.zeros(5, dtype=torch.long))
        self.assertTrue((env.task_scene_data.geometry_b == -32.).all())
        env.task_sensor_reader.reset.assert_not_called()

    def test_failed_builder_restores_selected_physics_and_preserves_scene_metadata(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        ids = torch.tensor([4, 1])
        objects, robot = env.scene['object_collection'], env.scene['robot']
        physical = (objects.state, objects.parked_mask, robot.root, robot.position, robot.velocity,
                    robot.position_target, robot.velocity_target, robot.effort_target)
        physical_before = [value.clone() for value in physical]
        metadata_before = {key: value.clone() for key, value in vars(env.task_scene_data).items()
                           if isinstance(value, torch.Tensor)}
        captures = []

        def capture(**kwargs):
            captures.append(kwargs['env_id'])
            if kwargs['env_id'] == 1:
                return None
            return snapshot(kwargs['env_id'], kwargs.get('object_pose'))

        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            capture(**request, object_pose=pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_check_physical_robot_starts', side_effect=lambda env,samples,*a: torch.ones(len(samples),dtype=torch.bool)):
            with self.assertRaisesRegex(RuntimeError, 'Independent episode preparation exhausted retries'):
                task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(captures, [4, 1, 1])
        for actual, expected in zip(physical, physical_before):
            torch.testing.assert_close(actual, expected)
        for name, expected in metadata_before.items():
            torch.testing.assert_close(getattr(env.task_scene_data, name), expected)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.zeros(5, dtype=torch.long))
        self.assertTrue((env._episode_reset.episode_seeds == -1).all())
        self.assertEqual(env.scene.writes, 0)
        env.task_sensor_reader.reset.assert_not_called()

    def test_shelf_reset_restores_only_selected_poses_and_zeroes_velocity(self):
        env, episodes = episode_reset_fixture()
        ids = torch.tensor([4,1])
        shelf = env.scene['shelf']
        before = shelf.root.clone()
        task.apply_episode_batch(env, ids, episodes)
        expected = shelf.data.default_root_state[ids].clone()
        expected[:, :3] += env.scene.env_origins[ids]
        expected[:, 7:] = 0
        torch.testing.assert_close(shelf.root[ids], expected)
        torch.testing.assert_close(shelf.root[[0,2,3]], before[[0,2,3]])
        self.assertEqual(shelf.root_writes, 1)

    def test_empty_reset_does_not_initialize_resources_or_write_assets(self):
        env, _ = episode_reset_fixture()
        del env._episode_reset
        with patch.object(task, '_initialize_episode_reset', side_effect=AssertionError('Unexpected initialize')):
            task.reset_scene(env, torch.empty(0, dtype=torch.long))
        self.assertEqual(env.scene['object_collection'].writes, 0)



class PartialResetBatchTests(unittest.TestCase):
    def test_reset_snapshot_matches_used_transition_fields_without_dynamics_or_sensors(self):
        env, _, _ = native_fixture(7)
        task.reset_task(env)
        state = env._contact_mdp
        data = env.scene['robot'].data
        data.joint_pos += torch.arange(7)[:, None] * .01
        data.joint_vel += .023
        data.body_link_pos_w[:, 1, 0] += .04
        ids = torch.tensor([6, 1])
        expected = task._read_transition(env, state, ids)
        class PoseOnly(SimpleNamespace):
            def __getattribute__(self, name):
                if name in ('body_com_pos_w', 'body_com_acc_w', 'body_com_ang_vel_w',
                            'object_com_pos_w', 'object_com_vel_w', 'applied_torque'):
                    raise AssertionError(f'Reset fetched unused dynamics: {name}')
                return super().__getattribute__(name)
        env.scene['robot'].data = PoseOnly(**vars(data))
        objects = env.scene['object_collection']
        objects.data = PoseOnly(**vars(objects.data))
        with patch.object(env, 'task_sensor_reader', side_effect=AssertionError('Reset read sensors')):
            actual = task._read_reset_snapshot(env, state, ids)
        for name in ('center', 'corners', 'rotation', 'shelf_pose', 'sensor_position', 'tcp_pose_b'):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        torch.testing.assert_close(actual.sample[:, :31], expected.sample[:, :31], rtol=0, atol=0)

    def test_cached_local_com_follows_link_rotation_without_native_world_com_reads(self):
        env, _, _ = native_fixture(3)
        robot = env.scene['robot'].data
        objects = env.scene['object_collection'].data
        robot.body_com_pos_b[:, 3] = torch.tensor([.1, .02, -.03])
        objects.object_com_pos_b[:, 2] = torch.tensor([.03, -.04, .02])
        task.reset_task(env)
        state = env._contact_mdp
        robot.body_link_quat_w[:, 3] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        objects.object_link_pose_w[:, 2, 3:7] = robot.body_link_quat_w[:, 3]
        objects.object_com_vel_w[:, 2, 5] = 2.
        # Native world-COM buffers are deliberately stale/nonfinite. The fixed
        # local offsets must instead rotate with current link poses on device.
        robot.body_com_pos_w.fill_(math.nan)
        objects.object_com_pos_w.fill_(math.nan)
        with patch.object(task, 'self_motion_wrench', wraps=task.self_motion_wrench) as compensate:
            actual = task._read_transition(env, state, torch.arange(3))
        expected_hand_com = robot.body_link_pos_w[:, 3] + torch.tensor([-.02, .1, -.03])
        torch.testing.assert_close(compensate.call_args.args[2][:, 1], expected_hand_com)
        slot = 0  # Collection object 2 occupies the first active slot.
        pose = objects.object_link_pose_w[:, 2, :3] - env.scene.env_origins
        expected_com = pose + torch.tensor([.04, .03, .02])
        expected_velocity = torch.cross(torch.tensor([0., 0., 2.]).expand(3, -1),
                                        actual.center[:, slot] - expected_com, dim=-1)
        torch.testing.assert_close(actual.velocity[:, slot, :3], expected_velocity)
        self.assertTrue(actual.state_valid.all())

    @unittest.skipUnless(torch.cuda.is_available(), 'Transition CUDA graph requires CUDA.')
    def test_transition_graph_matches_eager_after_changed_measurements_and_subset_reset(self):
        env, action, _ = native_fixture(8)
        env.device = 'cuda'
        for holder in (env.scene, env.scene['robot'].data, env.scene['object_collection'].data, env.scene['shelf'].data,
                       env.task_scene_data, env.sensors, action):
            for name, value in vars(holder).items():
                if isinstance(value, torch.Tensor):
                    setattr(holder, name, value.cuda())
        task.reset_task(env)
        state = env._contact_mdp
        ids = torch.arange(8, device='cuda')
        with torch.inference_mode():
            previous = None
            for scale in (1., 1.3):
                env.scene['robot'].data.joint_vel.fill_(scale)
                env.scene['robot'].data.body_com_acc_w[..., :3].fill_(scale)
                env.sensors.ideal_wrench.add_(scale)
                expected = task._read_transition(env, state, ids)
                actual = task._read_transition(env, state)
                for name, value in vars(actual).items():
                    torch.testing.assert_close(value, getattr(expected, name), atol=0, rtol=0, equal_nan=True)
                if previous is not None:
                    for value, snapshot in previous:
                        torch.testing.assert_close(value, snapshot, atol=0, rtol=0, equal_nan=True)
                previous = [(v, v.clone()) for v in vars(actual).values()]
                task.reset_task(env, torch.tensor([1, 4], device='cuda'))

    def test_transition_subset_matches_full_batch_and_limits_geometry_and_self_load_work(self):
        env, action, _ = native_fixture(8)
        task.reset_task(env)
        state = env._contact_mdp
        robot = env.scene['robot'].data
        robot.joint_pos[:, 0] = torch.arange(8) * .01
        robot.joint_vel[:, 1] = torch.arange(8) * .02
        robot.body_com_acc_w[:, 3, 0] = torch.arange(8) * .1
        env.sensors.tactile_force[:, 2, 0] = torch.arange(8)
        env.scene["shelf"].data.root_link_pose_w[6, 0] += .02
        action.normalized_actions[:, 3] = torch.arange(8) * .01
        action.applied_torque[1, 0] = math.nan
        full = task._read_transition(env, state)
        ids = torch.tensor([6, 1])
        def reject(*args, **kwargs):
            raise AssertionError('Subset transition must not read device scalar values')
        with patch.object(task, 'self_motion_wrench', wraps=task.self_motion_wrench) as compensate, \
             patch.object(task, 'object_geometry', wraps=task.object_geometry) as geometry, \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'cpu', reject):
            subset = task._read_transition(env, state, ids)
        self.assertEqual(compensate.call_args.args[0].shape[0], 2)
        self.assertEqual(geometry.call_args.args[0].shape[0], 2)
        for field in dataclasses.fields(full):
            with self.subTest(field=field.name):
                self.assertEqual(getattr(subset, field.name).shape[0], 2)
                torch.testing.assert_close(getattr(subset, field.name), getattr(full, field.name)[ids])

    def test_selected_state_reset_matches_full_input_api_for_unsorted_ids(self):
        state_full, batch, values = fixture(6)
        state_selected, _, _ = fixture(6)
        batch.sample[:, :12] += torch.arange(6, dtype=batch.sample.dtype)[:, None]
        batch.tcp_pose_b[:, 0] = torch.arange(6, dtype=batch.sample.dtype) * .01
        ids = torch.tensor([4, 1])
        subset = task.TransitionBatch(**{field.name: getattr(batch, field.name)[ids]
                                        for field in dataclasses.fields(batch)})
        selected_values = {name: value[ids] for name, value in values.items()}
        state_full.reset(ids, batch, **values, step=0)
        state_selected.reset(ids, subset, **selected_values, step=0, batch_is_selected=True)
        for name, value in vars(state_full).items():
            if isinstance(value, torch.Tensor):
                with self.subTest(field=name):
                    torch.testing.assert_close(getattr(state_selected, name), value)

    def test_native_partial_reset_limits_all_preparation_to_selected_rows_and_preserves_neighbors(self):
        env, _, _ = native_fixture(8)
        task.reset_task(env)
        env.common_step_counter = 1
        state = task.get_task_state(env)
        ids = torch.tensor([6, 1])
        other_ids = torch.tensor([0, 2, 3, 4, 5, 7])
        preserved = {name: value[other_ids].clone() for name, value in vars(state).items()
                     if isinstance(value, torch.Tensor) and value.shape[:1] == (8,)}
        final_observation = env.extras['terminal_observation'].clone()
        env.task_scene_data.geometry_b[ids, :3, 0] += .01
        with patch.object(task, 'self_motion_wrench', wraps=task.self_motion_wrench) as compensate, \
             patch.object(task, '_read_reset_snapshot', wraps=task._read_reset_snapshot) as snapshot, \
             patch.object(task, 'prepare_initial_geometry', wraps=task.prepare_initial_geometry) as prepare, \
             patch.object(task, 'geometry_in_eef_frame', wraps=task.geometry_in_eef_frame) as transform, \
             patch.object(state, 'reset', wraps=state.reset) as commit:
            task.reset_task(env, ids)
        compensate.assert_not_called()
        self.assertEqual(snapshot.call_args.args[2].numel(), 2)
        self.assertEqual(prepare.call_args.args[0].shape[0], 2)
        self.assertEqual(transform.call_args.args[0].shape[0], 2)
        self.assertEqual(commit.call_args.args[1].sample.shape, (2, 72))
        self.assertEqual(commit.call_args.kwargs['geometry_b'].shape, (2, 10, 10))
        self.assertTrue(commit.call_args.kwargs['batch_is_selected'])
        for name, expected in preserved.items():
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(state, name)[other_ids], expected)
        torch.testing.assert_close(env.extras['terminal_observation'], final_observation)
        self.assertIsNot(env.extras['terminal_observation'], state.observation)


class ContactBatchUpdateTests(unittest.TestCase):
    def test_whole_batch_sampling_has_no_nonzero_and_partial_reset_keeps_other_histories(self):
        class Native:
            def _update_buffers_impl(self, ids):
                self.reads += 1
                self.history = self.history.roll(1, 1)
                self.history[:, 0] = self.force

            def reset(self, ids=None):
                self.history[slice(None) if ids is None else ids] = 0

        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TaskContactSensor')
        namespace = dict(ContactSensor=Native)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        for history_length in (0, 4):
            sensor = namespace['TaskContactSensor']()
            sensor.cfg = SimpleNamespace(update_period=0., track_air_time=False, track_pose=False,
                                         history_length=history_length)
            sensor._num_envs, sensor._is_visualizing, sensor.reads = 3, False, 0
            sensor._timestamp = torch.zeros(3)
            sensor._timestamp_last_update = torch.zeros(3)
            sensor._is_outdated = torch.ones(3, dtype=torch.bool)
            sensor.history, sensor.force = torch.zeros(3, 4), torch.tensor([1., 2., 3.])
            with patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('host synchronization')):
                for tick in range(4):
                    sensor.force += 1
                    sensor.update(.01)
                    sensor._update_outdated_buffers()
                    sensor._update_outdated_buffers()
                self.assertEqual(sensor.reads, 4)
                before = sensor.history.clone()
                sensor.reset(torch.tensor([1]))
                sensor._update_outdated_buffers()
                torch.testing.assert_close(sensor.history[[0, 2]], before[[0, 2]])
                self.assertEqual(sensor.history[1].count_nonzero(), 0)
                self.assertEqual(sensor.reads, 4)
                sensor.update(.01)
                sensor._update_outdated_buffers()
                self.assertEqual(sensor.reads, 5)
                torch.testing.assert_close(sensor.history[:, 0], sensor.force)

    def test_parking_uses_native_persistent_composer_and_refreshes_on_reset(self):
        class Native:
            def __init__(self, cfg):
                pass
            def write_data_to_sim(self):
                self.native_write_called = True
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TaskObjectPool')
        namespace = dict(RigidObjectCollection=Native, torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        pool = namespace['TaskObjectPool'](None)
        pool.device = 'cpu'
        pool.parked_mask = torch.tensor([[True, False], [False, True]])
        pool.gravity = torch.tensor([0., 0., -9.81])
        pool.data = SimpleNamespace(default_mass=torch.tensor([[[.2], [.4]], [[.3], [.5]]]))
        from unittest.mock import Mock
        pool.permanent_wrench_composer = Mock()
        for _ in range(2):
            pool.refresh_parking()
            kwargs = pool.permanent_wrench_composer.set_forces_and_torques.call_args.kwargs
            torch.testing.assert_close(kwargs['forces'], -pool.data.default_mass * pool.gravity * pool.parked_mask[..., None])
            self.assertTrue(kwargs['is_global'])
            self.assertNotIn('positions', kwargs)  # Native force is applied at COM without extra torque.
            pool.parked_mask.logical_not_()
        self.assertEqual(pool.permanent_wrench_composer.reset.call_count, 2)
        pool.parked_mask.zero_()
        pool.permanent_wrench_composer.set_forces_and_torques.reset_mock()
        pool.refresh_parking()
        pool.permanent_wrench_composer.set_forces_and_torques.assert_not_called()
        self.assertEqual(pool.permanent_wrench_composer.reset.call_count, 3)
        pool.write_data_to_sim()
        self.assertTrue(pool.native_write_called)


class FusedGeometryTests(unittest.TestCase):
    def test_projected_area_matches_tensor_oracle_for_tilt_degeneracy_and_padding(self):
        torch.manual_seed(613)
        q = torch.randn(12, 9, 4)
        q /= q.norm(dim=-1, keepdim=True)
        cube = torch.cartesian_prod(*[torch.tensor([-.5, .5])]*3)
        corners = obs.quat_apply(q[:, :, None].expand(-1, -1, 8, -1),
                                cube * torch.rand(12, 9, 1, 3) * .3) + torch.rand(12, 9, 1, 3)
        corners[0, :2] = 0  # Degenerate projected hulls must stay finite.
        active = torch.rand(12, 9) > .25
        regions = torch.tensor([.3, .7, .3, .7]).expand(12, -1).contiguous()
        polygon, count = task.convex_hull_xy(corners[..., :2])
        expected_area = task.polygon_area(polygon, count)
        expected_overlap = task.rectangle_overlap_area(polygon, count, regions[:, None])
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            wp.init()
            c, a, r = (t.to(device) for t in (corners, active, regions))
            area, overlap = torch.empty_like(a, dtype=torch.float32), torch.empty_like(a, dtype=torch.float32)
            kw = dict(stream=wp.stream_from_torch(torch.cuda.current_stream())) if device == 'cuda' else dict(device='cpu')
            wp.launch(task._projected_areas, dim=a.shape, inputs=[wp.from_torch(t) for t in (c, a, r, area, overlap)], **kw)
            torch.testing.assert_close(area.cpu(), expected_area * active, atol=2.e-7, rtol=2.e-5)
            torch.testing.assert_close(overlap.cpu(), expected_overlap * active, atol=2.e-7, rtol=2.e-5)

    def test_device_pose_mapping_keeps_roles_and_heights_without_downloading_clutter(self):
        poses = torch.tensor([[-.7, .1, .2], [-.6, .2, .3], [-.8, -.2, .4]])
        positions = task._PoseMapping({2: (-.7, .1, 1.), 0: (-.6, .2, 1.1)}, (2, 0, 1), poses,
                                      {2: 1., 0: 1.1, 1: 1.2})
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unnecessary transfer')):
            self.assertEqual(positions[2], (-.7, .1, 1.))
            self.assertEqual(positions.support_heights[1], 1.2)
        self.assertAlmostEqual(dict(positions)[1][0], -.8, places=6)


class SensorRuntimeTests(unittest.TestCase):
    """Execute the production sensor tensor path without initializing Kit."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        module = types.ModuleType('contact_manipulation.mdp._sensor_numerical_tests')
        module.__package__ = 'contact_manipulation.mdp'
        sys.modules[module.__name__] = module  # dataclasses resolves field types through this entry.
        exec(compile(path.read_text().split('# Native classes are imported only', 1)[0],
                     str(path), 'exec'), module.__dict__)
        cls.sensors = module

    @staticmethod
    def fixture(batch=4):
        tactile_names = tuple(f'touch_{index}' for index in range(17))
        ft_name = 'ToolForceTorqueLink'

        class Contact:
            def __init__(self, name):
                self.num_bodies, self.num_instances, self.body_names = 1, batch, [name]
                self.cfg = SimpleNamespace(history_length=4)
                self.data = SimpleNamespace(
                    net_forces_w=torch.zeros(batch, 1, 3),
                    force_matrix_w_history=torch.zeros(batch, 4, 1, 3, 3),
                )

            def reset(self, ids):
                self.data.net_forces_w[ids] = 0
                self.data.force_matrix_w_history[ids] = 0

            def find_bodies(self, names, preserve_order=False):
                assert preserve_order
                return [self.body_names.index(name) for name in names], list(names)

        # The force sensor is deliberately not body zero. The raw load includes
        # values above the torque sensing range; calibration belongs to the MDP.
        reaction = torch.arange(batch * 3 * 6, dtype=torch.float32).reshape(batch, 3, 6)
        scene = {'robot': SimpleNamespace(
            body_names=['base_link', ft_name, 'hand_base'],
            data=SimpleNamespace(body_incoming_joint_wrench_b=reaction),
            root_physx_view=SimpleNamespace(
                get_link_incoming_joint_force=lambda: reaction,
                link_paths=[['/World/envs/env_0/Robot/base_link',
                             '/World/envs/env_0/Robot/' + ft_name,
                             '/World/envs/env_0/Robot/ROASRightHand/base_link']],
            ),
        )}
        tactile = Contact('tactile')
        tactile.num_bodies, tactile.num_instances = 17, batch * 17
        # Deliberately use a different native body order from observation order.
        tactile.body_names = list(reversed(tactile_names))
        tactile.data.net_forces_w = torch.zeros(batch, 17, 3)
        scene['tactile_contact'] = tactile
        env = SimpleNamespace(scene=scene, num_envs=batch, device='cpu',
                              sim=SimpleNamespace(current_time=0.), cfg=SimpleNamespace(decimation=4))
        asset_module = types.ModuleType('contact_manipulation.config.ur5e.scene')
        asset_module.FORCE_TORQUE_BODY_NAME = ft_name
        asset_module.TACTILE_BODY_NAMES = tactile_names
        with patch.dict(sys.modules, {asset_module.__name__: asset_module}):
            reader = SensorRuntimeTests.sensors.SensorReader(env)
        return env, reader, reaction

    def test_initial_measurement_invalid_then_physics_valid_and_selected_reset_preserves_other_rows(self):
        env, reader, reaction = self.fixture()
        self.assertFalse(reader().wrench_valid.any())
        self.assertEqual(reader.distal_body_names, ('ToolForceTorqueLink', 'hand_base'))
        env.sim.current_time = .1

        def reject(*args, **kwargs):
            raise AssertionError('Sensor hot path read a tensor scalar on the host')

        with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'tolist', reject), patch.object(torch.Tensor, 'cpu', reject), \
             patch.object(torch.Tensor, 'numpy', reject):
            batch = reader()
        torch.testing.assert_close(batch.ideal_wrench, reaction[:, 1])
        self.assertTrue(batch.wrench_valid.all())
        reader.reset(torch.tensor([1, 3]))
        self.assertEqual(reader().wrench_valid.tolist(), [True, False, True, False])
        self.assertEqual(reader().ideal_wrench[1].count_nonzero(), 0)
        torch.testing.assert_close(reader().ideal_wrench[[0, 2]], reaction[[0, 2], 1])
        env.sim.current_time = .2
        self.assertTrue(reader().wrench_valid.all())

    def test_same_timestamp_read_does_not_revalidate_reset_rows_or_refresh_cached_others(self):
        env, reader, reaction = self.fixture()
        env.sim.current_time = .1
        reader.reset(torch.tensor([0]))
        before = reader().ideal_wrench[1].clone()
        reaction[1] += 3
        self.assertFalse(reader().wrench_valid[0])
        torch.testing.assert_close(reader().ideal_wrench[1], before)

    def test_nonfinite_ft_and_tactile_invalidate_only_affected_measurements(self):
        env, reader, reaction = self.fixture()
        reaction[0, 1, 2] = float('nan')
        native_index = env.scene['tactile_contact'].body_names.index('touch_3')
        env.scene['tactile_contact'].data.net_forces_w[2, native_index, 0] = float('inf')
        env.sim.current_time = .1
        batch = reader()
        self.assertEqual(batch.wrench_valid.tolist(), [False, True, True, True])
        self.assertFalse(batch.tactile_valid[2, 3])
        self.assertTrue(batch.tactile_valid[2, 2])
        self.assertTrue(batch.tactile_valid[1, 3])

    def test_grouped_tactile_preserves_every_channel_and_partial_reset(self):
        env, reader, _ = self.fixture()
        expected = torch.arange(4 * 17 * 3, dtype=torch.float32).reshape(4, 17, 3) / 1000
        expected[0, 0] = 0
        sensor = env.scene['tactile_contact']
        sensor.data.net_forces_w.copy_(expected.flip(1))
        env.sim.current_time = .1
        actual = reader()
        torch.testing.assert_close(actual.tactile_force, expected)
        torch.testing.assert_close(actual.tactile_force.norm(dim=-1) > .01, expected.norm(dim=-1) > .01)
        reader.reset(torch.tensor([1, 3]))
        torch.testing.assert_close(reader().tactile_force[[0, 2]], expected[[0, 2]])
        self.assertFalse(reader().tactile_valid[[1, 3]].any())
        torch.testing.assert_close(sensor.data.net_forces_w[[0, 2]], expected[[0, 2]].flip(1))

    def test_visible_depth_pca_is_translation_invariant_and_keeps_upright_orientation(self):
        generator = torch.Generator().manual_seed(2)
        points = torch.rand((1000, 3), generator=generator) * torch.tensor([.12, .04, .19])
        shift = torch.tensor([.1, -.2, .3])
        labels = torch.zeros((1, len(points)), dtype=torch.long)
        original = self.sensors.fit_visible_obbs(points[None], labels, 1, 16)[0][0, 0]
        translated = self.sensors.fit_visible_obbs((points + shift)[None], labels, 1, 16)[0][0, 0]
        torch.testing.assert_close(original[:3] + shift, translated[:3])
        torch.testing.assert_close(original[3:], translated[3:], atol=2.e-6, rtol=1.e-5)
        self.assertEqual(original[4:6].tolist(), [0., 0.])
        self.assertTrue((original[7:] > 0).all())

    def test_camera_config_rejects_invalid_acceptance_ranges_and_retry_counts(self):
        for kwargs in ({'min_target_visible_fraction': 1.1}, {'height_above_shelf': (.3, .2)},
                       {'max_camera_attempts': 0}, {'max_layout_attempts': 0},
                       {'roll_degrees': (float('nan'), 3)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sensors.InitialVisionCfg(**kwargs)


class BatchedRobotKinematicsTests(unittest.TestCase):
    def test_parallel_obb_axes_do_not_falsely_accept_overlapping_boxes(self):
        # A rounded rotation can produce spurious cross-axis residuals in
        # FP32. Relative-coordinate cancellation previously accepted this pair.
        rotation = np.array([[-.8018245697021484, .4289975166320801, .41597890853881836],
                             [-.13086608052253723, .5531646609306335, -.8227289915084839],
                             [-.5830535292625427, -.7141218185424805, -.3873997926712036]])
        ca = np.array([-.04387548938393593, -.18531934916973114, .15092703700065613])
        cb = np.array([-.23477087914943695, -.22714553773403168, .19318510591983795])
        ha = np.array([.08198848366737366, .031128162518143654, .12377825379371643])
        hb = np.array([.08862842619419098, .10184000432491302, .09672331064939499])
        self.assertFalse(task.oriented_box_separated(ca, rotation, ha, cb, rotation, hb, .005))
        for dtype in (torch.float32, torch.float64):
            args = [torch.tensor(value, dtype=dtype) for value in (ca, rotation, ha, cb, rotation, hb)]
            result = task.oriented_boxes_separated_batch(*args, .005)
            self.assertFalse(result)
            expanded = [value.expand(3, *value.shape) for value in args]
            self.assertFalse(task.oriented_boxes_separated_batch(*expanded, .005).any())

    @unittest.skipUnless(torch.cuda.is_available(), 'Fused clearance requires CUDA.')
    def test_fused_clearance_matches_eager_with_self_pairs_invalid_candidates_and_nan(self):
        rng = np.random.default_rng(831)
        n, colliders, obstacles = 512, 2, 5
        device = 'cuda'
        frames = torch.eye(4, device=device).expand(n, colliders, 4, 4).clone()
        frames[..., :3, :3] = torch.tensor(Rotation.random(n*colliders, random_state=rng).as_matrix(),
                                          device=device, dtype=torch.float32).reshape(n, colliders, 3, 3)
        frames[..., :3, 3] = torch.tensor(rng.normal(0., .15, (n, colliders, 3)), device=device)
        frames[..., 2, 3] += 1.1
        joints = torch.eye(4, device=device).expand(n, 4, 4, 4).clone()
        joints[:, 1:4, 2, 3] = torch.tensor([1., 1.4, 1.], device=device)
        joints[:, 2, 0, 3] = .2
        constants = dict(collider_owners=torch.arange(colliders, device=device),
                         collider_offsets=torch.eye(4, device=device).expand(colliders, 4, 4).clone(),
                         collider_half=torch.full((colliders, 3), .045, device=device),
                         self_pairs=torch.tensor([[0, 1]], device=device))
        model = SimpleNamespace(colliders=[None]*colliders, self_collision_pairs=lambda: [(0, 1)],
                                _tensor_constants=lambda *_: constants)
        env = SimpleNamespace(robot_pose_model=model, cfg=SimpleNamespace(robot_start=task.RobotStartRules()))
        centers = torch.tensor(rng.normal(0., .18, (n, obstacles, 3)), device=device, dtype=torch.float32)
        centers[..., 2] += 1.1
        rotations = torch.tensor(Rotation.random(n*obstacles, random_state=rng).as_matrix(),
                                 device=device, dtype=torch.float32).reshape(n, obstacles, 3, 3)
        half = torch.full((n, obstacles, 3), .05, device=device)
        obstacle_valid = torch.tensor(rng.random((n, obstacles)) > .2, device=device)
        candidate_valid = torch.tensor(rng.random(n) > .2, device=device)
        frames[-1, 0, 0, 0] = math.nan
        candidate_valid[-1] = True
        args = (env, frames, joints, frames[:, 0], centers, rotations, half, obstacle_valid)
        expected, expected_height = task._robot_candidate_clearance_eager(*args)
        actual, height = task._robot_candidate_clearance(*args, candidate_valid=candidate_valid)
        torch.testing.assert_close(actual, expected & candidate_valid, rtol=0, atol=0)
        torch.testing.assert_close(height, expected_height)

    @classmethod
    def setUpClass(cls):
        # Load production numerical methods without importing the simulator SDK.
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        tree = ast.parse(path.read_text())
        names = {'_RobotJoint', 'RobotColliderBox', 'RobotPoseModel'}
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [node for node in tree.body if (isinstance(node, ast.ClassDef) and node.name in names)
                  or (isinstance(node, ast.FunctionDef) and node.name in ('_arm_ik_kernel', '_ik_orientation_error'))
                  or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ('_ArmVector', '_ArmMatrix', '_ArmPoints'))]
        module = types.ModuleType('contact_test_robot_kinematics')
        sys.modules[module.__name__] = module
        module.__dict__.update(np=np, torch=torch, wp=wp, math=math, dataclass=dataclasses.dataclass,
                               FORCE_TORQUE_PROXIMAL_BODY_NAME='SensorMountProximal',
                               FORCE_TORQUE_BODY_NAME='ToolForceTorqueLink')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), module.__dict__)
        cls.numerics = module

    def model(self):
        module = self.numerics
        model = module.RobotPoseModel()
        model.joint_names = ('pan', 'lift', 'elbow', 'wrist1', 'wrist2', 'wrist3', 'finger')
        model.body_names = ('base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                            'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'palm', 'finger')
        model.arm_ids = np.arange(6)
        model.default_joint_positions = np.array([.1, -.8, 1.1, -.4, .3, -.2, .1])
        model.arm_limits = np.tile([-2*math.pi, 2*math.pi], (6, 1))
        model.root_body_name, model.tcp_body_name = 'base_link', 'palm'
        model.root_transform = np.eye(4)
        model.root_transform[:3, 3] = (.1, -.2, .8)
        joints = []
        for i in range(6):
            pre, inverse_post = np.eye(4), np.eye(4)
            pre[:3, 3] = (.06, .015, .09)
            pre[:3, :3] = Rotation.from_rotvec([.01*i, -.02*i, .03]).as_matrix()
            inverse_post[:3, 3] = (.02, -.03, .04)
            axis = np.eye(3)[[2, 1, 1, 0, 1, 2][i]]
            joints.append(module._RobotJoint(model.joint_names[i], model.body_names[i],
                                            model.body_names[i+1], pre, inverse_post, axis, i, i == 3))
        pre = np.eye(4)
        pre[:3, 3] = (.02, .03, .05)
        joints.append(module._RobotJoint('fixed_palm', 'wrist_3_link', 'palm', pre,
                                        np.eye(4), np.zeros(3), None))
        joints.append(module._RobotJoint('finger', 'palm', 'finger', pre,
                                        np.eye(4), np.array([1., 0., 0.]), 6))
        model.joints = tuple(joints)
        model.body_parents = {joint.child: joint.parent for joint in joints}
        model.colliders = [module.RobotColliderBox(name, np.array([.01, .02, .03]), np.eye(3),
                                                  np.array([.01, .015, .02]), name) for name in model.body_names]
        model.base_origin = np.zeros(3)
        return model

    @unittest.skipUnless(torch.cuda.is_available(), 'Fused IK requires CUDA.')
    def test_fused_ik_reaches_seeded_targets_and_rejects_unreachable_goals(self):
        model = self.model()
        generator = torch.Generator(device='cuda').manual_seed(192)
        truth = torch.randn(256, 6, device='cuda', generator=generator) * .3
        tcp = model.arm_forward_batch(truth)[2]
        positions, valid, _, _, result = model.solve_batch(tcp[:, :3, 3], tcp[:, :3, :3], truth + .08)
        self.assertTrue(valid.all())
        # TF32 FK accumulation can differ from full-float Warp by ~0.15 mm.
        torch.testing.assert_close(result, tcp, atol=5.e-4, rtol=0.)
        self.assertTrue(torch.isfinite(positions).all())
        _, valid, *_ = model.solve_batch(torch.full((8, 3), 9., device='cuda'),
                                         torch.eye(3, device='cuda').expand(8, -1, -1), truth[:8])
        self.assertFalse(valid.any())

    def test_collapsed_tensor_fk_matches_all_scalar_links_including_reverse_and_default_finger(self):
        model = self.model()
        q = torch.tensor(np.tile(model.default_joint_positions, (17, 1)), dtype=torch.float64)
        q[:, :6] += torch.randn((17, 6), generator=torch.Generator().manual_seed(11), dtype=q.dtype) * .5
        actual = model.forward_batch(q)
        expected = np.stack([[model.forward(row.numpy())[name] for name in model.body_names] for row in q])
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1.e-12, rtol=1.e-12)

    def test_hand_arm_pairs_cover_forearm_and_wrist_but_exclude_fixed_mount(self):
        model = self.model()
        pairs = {frozenset((model.colliders[a].body_name, model.colliders[b].body_name))
                 for a, b in model.self_collision_pairs()}
        for hand in ('palm', 'finger'):
            for arm in ('base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                        'wrist_1_link', 'wrist_2_link'):
                self.assertIn(frozenset((hand, arm)), pairs)
            self.assertNotIn(frozenset((hand, 'wrist_3_link')), pairs)
        self.assertNotIn(frozenset(('palm', 'finger')), pairs)

    def test_scalar_and_batch_reject_hand_arm_overlap_folded_forearm_and_low_elbow(self):
        batch = 4
        frames = torch.eye(4, dtype=torch.float64).repeat(batch, 2, 1, 1)
        frames[:, :, 2, 3] = 1.
        frames[:, 1, 0, 3] = .2
        frames[2, 1, 0, 3] = .03  # Hand intersects forearm.
        joints = torch.eye(4, dtype=torch.float64).repeat(batch, 6, 1, 1)
        joints[:, 1, :3, 3] = torch.tensor([.2, 0., 1.])
        joints[:, 2, :3, 3] = torch.tensor([0., 0., 1.3])
        joints[:, 3, :3, 3] = torch.tensor([-.2, 0., 1.])
        joints[1, 3, 0, 3] = .1  # Wrist behind elbow despite high elbow.
        joints[3, 2, 2, 3] = 1.  # No elbow-up margin.
        constants = dict(collider_owners=torch.tensor([0, 1]),
                         collider_offsets=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
                         collider_half=torch.full((2, 3), .02, dtype=torch.float64),
                         self_pairs=torch.tensor([[0, 1]]))
        colliders = [self.numerics.RobotColliderBox(name, np.zeros(3), np.eye(3), np.full(3, .02), name)
                     for name in ('forearm_link', 'palm')]
        model = SimpleNamespace(_tensor_constants=lambda *_: constants, colliders=colliders,
                                self_collision_pairs=lambda: np.array([[0, 1]]))
        env = SimpleNamespace(robot_pose_model=model, cfg=SimpleNamespace(robot_start=task.RobotStartRules()))
        valid, height = task._robot_candidate_clearance(
            env, frames, joints, frames[:, 1], frames.new_empty(batch, 0, 3),
            frames.new_empty(batch, 0, 3, 3), frames.new_empty(batch, 0, 3), torch.empty(batch, 0, dtype=torch.bool))
        self.assertEqual(valid.tolist(), [True, False, False, False])
        for row in range(batch):
            model.arm_landmarks = lambda _, row=row: joints[row, [1, 2, 3], :3, 3].numpy()
            transforms = {box.body_name: frames[row, index].numpy() for index, box in enumerate(colliders)}
            reason = task.robot_start_collision_reason(env, None, transforms, obstacles=[])
            scalar_valid = reason is None and float(height[row]) >= env.cfg.robot_start.min_elbow_height
            self.assertEqual(scalar_valid, bool(valid[row]))
            if row == 2:
                self.assertIn('forearm_link / palm', reason)

    def test_analytic_tensor_jacobian_matches_finite_difference_of_exact_usd_chain(self):
        model = self.model()
        arm = torch.tensor(model.default_joint_positions[:6][None], dtype=torch.float64)
        _, _, tcp, jacobian = model.arm_forward_batch(arm, jacobian=True)
        for index in range(6):
            shifted = arm.clone()
            shifted[:, index] += 1.e-6
            _, _, other = model.arm_forward_batch(shifted)
            linear = (other[0, :3, 3] - tcp[0, :3, 3]) / 1.e-6
            angular = Rotation.from_matrix((other[0, :3, :3] @ tcp[0, :3, :3].T).numpy()).as_rotvec() / 1.e-6
            torch.testing.assert_close(jacobian[0, :, index], torch.cat((linear, torch.tensor(angular))),
                                       atol=3.e-7, rtol=2.e-6)

    def test_batch_ik_reaches_distinct_targets_and_obeys_limits(self):
        model = self.model()
        truth = torch.tensor(np.tile(model.default_joint_positions[:6], (6, 1)), dtype=torch.float64)
        truth += torch.randn(truth.shape, generator=torch.Generator().manual_seed(22), dtype=truth.dtype) * .15
        _, _, target = model.arm_forward_batch(truth)
        q, valid, _, _, result = model.solve_batch(target[:, :3, 3], target[:, :3, :3], truth + .08)
        self.assertTrue(valid.all())
        self.assertTrue((q[:, :6].abs() <= 2*math.pi).all())
        self.assertTrue(((result[:, :3, 3] - target[:, :3, 3]).norm(dim=-1) < 5.e-4).all())
        torch.testing.assert_close(q[:, 6], torch.full((6,), .1, dtype=q.dtype))

    def test_physical_link_check_handles_distinct_origins_and_rejects_one_corrupt_link_only(self):
        model = self.model()
        q = np.tile(model.default_joint_positions, (3, 1))
        q[:, :6] += np.array([[-.03], [0.], [.04]])
        predicted = model.forward_batch(torch.tensor(q, dtype=torch.float64)).numpy()
        origins = torch.tensor([[0., 0., 0.], [5., -3., 0.], [-2., 8., 0.]], dtype=torch.float64)
        poses = torch.zeros((3, len(model.body_names), 7), dtype=torch.float64)
        poses[..., :3] = torch.tensor(predicted[..., :3, 3]) + origins[:, None]
        poses[..., 3:] = torch.tensor(Rotation.from_matrix(predicted[..., :3, :3].reshape(-1, 3, 3)).as_quat()).reshape(3, -1, 4)
        starts = [(row, None, None, None, None) for row in q]
        env = SimpleNamespace(robot_pose_model=model)
        result = task._check_physical_robot_starts(env, [None]*3, starts, poses, origins)
        self.assertTrue(result.all())
        poses[1, -1, 0] += .001
        torch.testing.assert_close(task._check_physical_robot_starts(env, [None]*3, starts, poses, origins),
                                   torch.tensor([True, False, True]))

    def test_unreachable_ik_targets_are_reported_invalid_without_invalid_joint_values(self):
        model = self.model()
        seed = torch.tensor(model.default_joint_positions[:6][None], dtype=torch.float64)
        q, valid, *_ = model.solve_batch(torch.tensor([[9., 9., 9.]], dtype=seed.dtype),
                                        torch.eye(3, dtype=seed.dtype)[None], seed)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q[:, :6].abs() <= 2*math.pi).all())

    def test_candidate_seed_stream_is_unchanged_for_small_and_large_memory_caps(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=80)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        expected = None
        for cap in (8, 1, 3, 7, 16, 32, 64, 1024):
            rules.ik_batch_size = cap
            proposals = []
            def solve(positions, rotations, seeds):
                proposals.append(seeds.clone())
                n = len(seeds)
                # Force all proposals to be visited, isolating RNG/batching
                # from solver convergence and geometric acceptance.
                tcp = torch.eye(4).expand(n, -1, -1).clone()
                tcp[:, :3, 3] = positions
                return (torch.zeros((n, len(model.joint_names))), torch.zeros(n, dtype=torch.bool),
                        torch.zeros((n, 7, 4, 4)), torch.zeros((n, 6, 4, 4)), tcp)
            def clear(_env, frames, *_args):
                return torch.ones(len(frames), dtype=torch.bool), torch.full((len(frames),), .1)
            with patch.object(model, 'solve_batch', side_effect=solve), \
                 patch.object(task, '_robot_candidate_clearance', side_effect=clear):
                self.assertEqual(task._sample_robot_starts(env, [sample], [571]), [None])
            actual = torch.cat(proposals)
            if expected is None:
                expected = actual
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_shared_robot_geometry_cache_reuses_boxes_and_invalidates_on_rule_change(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=2)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        proposals = task._episode_proposals([sample], 2, 'cpu', torch.float32)
        cache = SimpleNamespace(sample=sample, proposals=proposals, robot=None)
        before = [value.clone() for value in proposals]
        with patch.object(task, '_scene_boxes_tensor', wraps=task._scene_boxes_tensor) as boxes, \
             patch.object(task, '_robot_start_regions', wraps=task._robot_start_regions) as regions:
            first = task._sample_robot_starts(env, [sample]*3, [17, 18, 19], scene_cache=cache)
            again = task._sample_robot_starts(env, [sample]*3, [17, 18, 19], scene_cache=cache)
            self.assertEqual(boxes.call_count, 1)
            self.assertEqual(regions.call_count, 1)
            self.assertEqual(boxes.call_args.args[2][0].shape[0], 1)
            cached = cache.robot
            rules.lateral_half_width += .01
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache)
            self.assertEqual(boxes.call_count, 2)
            self.assertIsNot(cache.robot, cached)
        for left, right in zip(first, again):
            if left is None:
                self.assertIsNone(right)
            else:
                for actual, expected in zip(left, right):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(proposals, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_tensor_rotation_error_handles_exact_pi_without_silent_zero(self):
        model = self.model()
        current = torch.eye(3, dtype=torch.float64)[None]
        target = torch.tensor(Rotation.from_rotvec(np.array([1., 2., 3.]) / np.sqrt(14) * math.pi).as_matrix())[None]
        actual = model._rotation_error_batch(target, current)[0]
        self.assertAlmostEqual(float(actual.norm()), math.pi, places=10)
        np.testing.assert_allclose(Rotation.from_rotvec(actual.numpy()).as_matrix(), target[0].numpy(), atol=1.e-12)

    def test_batched_sat_matches_complete_scalar_obb_oracle_and_parallel_touches(self):
        rng = np.random.default_rng(90)
        n = 400
        first, second = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        rotation_a = Rotation.random(n, random_state=rng).as_matrix()
        rotation_b = Rotation.random(n, random_state=rng).as_matrix()
        half_a, half_b = rng.uniform(.01, .6, size=(n, 3)), rng.uniform(.01, .6, size=(n, 3))
        for gap in (0., .015):
            expected = torch.tensor([task.oriented_box_separated(a, ra, ha, b, rb, hb, gap)
                                    for a, ra, ha, b, rb, hb in zip(first, rotation_a, half_a, second, rotation_b, half_b)])
            actual = task.oriented_boxes_separated_batch(*[torch.tensor(value) for value in
                (first, rotation_a, half_a, second, rotation_b, half_b)], gap)
            torch.testing.assert_close(actual, expected)
        center = torch.zeros((3, 3), dtype=torch.float64)
        shifted = center.clone()
        shifted[:, 0] = torch.tensor([1., 1.-1.e-6, 1.+.015], dtype=center.dtype)
        rotation = torch.eye(3, dtype=center.dtype).expand(3, -1, -1)
        half = torch.full_like(center, .5)
        torch.testing.assert_close(task.oriented_boxes_separated_batch(center, rotation, half, shifted, rotation, half),
                                   torch.tensor([True, False, True]))
        torch.testing.assert_close(task.oriented_boxes_separated_batch(center, rotation, half, shifted, rotation, half, .015),
                                   torch.tensor([False, False, True]))





class GeometricVisionTests(unittest.TestCase):
    """Use the production Warp kernel on CPU; no Kit or renderer is involved."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import warp as wp
        cls.wp = wp
        cls.cache_dir = tempfile.TemporaryDirectory(prefix='contact_ray_tests_')
        wp.init()
        cls.old_cache_dir = wp.config.kernel_cache_dir
        wp.config.kernel_cache_dir = cls.cache_dir.name
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        module = types.ModuleType('contact_geometric_test_runtime')
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        numerical = path.read_text().split('# Native classes are imported only', 1)[0]
        module.__package__ = 'contact_manipulation.mdp'
        exec(compile(numerical, str(path), 'exec'), module.__dict__)
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in ('_surface_moments', '_surface_covariance', '_surface_extents', '_fit_visible_obbs_cuda',
                                   '_initial_viewpoints', '_initial_ray_frames', '_initial_surface_rays',
                                   'InitialVisionSensor')]
        module.__dict__.update(wp=wp, np=np)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
        cls.module = module

    @classmethod
    def tearDownClass(cls):
        cls.wp.config.kernel_cache_dir = cls.old_cache_dir
        cls.cache_dir.cleanup()

    def fixture(self, batch=1, device='cpu'):
        wp = self.wp
        # Two role surfaces and one optional distractor surface.
        # Avoid tangent rays exactly on an edge: float32 rigid transforms can
        # classify that measure-zero boundary on either side of the triangle.
        points = np.array([[-.2937, -.2937, 0], [.2937, -.2937, 0],
                           [.2937, .2937, 0], [-.2937, .2937, 0]], np.float32)
        mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                       indices=wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device))
        sensor = self.module.InitialVisionSensor.__new__(self.module.InitialVisionSensor)
        sensor.cfg = self.module.InitialVisionCfg(ray_width=80, ray_height=60)
        sensor.capacity = sensor.cfg.ray_batch_size
        sensor.object_count = 3
        sensor.meshes = [mesh]
        sensor.mesh_ids = wp.array([mesh.id] * 3, dtype=wp.uint64, device=device)
        sensor.mesh_bounds = wp.array(np.stack([np.stack((points.min(0), points.max(0)))] * 3),
                                      dtype=wp.vec3, device=device)
        rows, columns = torch.meshgrid(torch.arange(120), torch.arange(160), indexing='ij')
        sensor.directions = torch.stack(((columns + .5 - 80) / 60,
                                         (rows + .5 - 60) / 60, torch.ones_like(rows)), -1).float().reshape(-1, 3)
        sensor.crop_directions = sensor.directions.reshape(120, 160, 3)[30:90, 40:120].reshape(-1, 3).contiguous()
        sensor.intrinsics = torch.tensor([[60., 0., 40.], [0., 60., 30.], [0., 0., 1.]])
        poses = torch.zeros(batch, 3, 7)
        poses[..., 3] = 1
        poses[:, 0, 2] = 1.4
        poses[:, 1, 2] = .8
        poses[:, 2, 2] = .5
        enabled = torch.tensor([[0, 1, -1]], dtype=torch.int32).expand(batch, -1).clone()
        eyes = torch.zeros(batch, 3)
        rotation = torch.tensor([[1., 0., 0., 0.]]).expand(batch, -1).clone()
        roles = torch.tensor([[0, 1]], dtype=torch.int32).expand(batch, -1).clone()
        sensor.directions = sensor.directions.to(device)
        sensor.crop_directions = sensor.crop_directions.to(device)
        sensor.intrinsics = sensor.intrinsics.to(device)
        return (sensor, *(tensor.to(device) for tensor in (poses, enabled, eyes, rotation, roles)))

    def test_front_surface_occludes_target_but_isolated_denominator_remains(self):
        sensor, *args = self.fixture()
        depth, labels, full, visible, ratios, accepted = sensor._trace(*args)
        self.assertGreater(int(full[0, 0]), 0)
        self.assertEqual(int(visible[0, 0]), 0)
        self.assertEqual(float(ratios[0, 1]), 1.)
        self.assertFalse(accepted.any())
        torch.testing.assert_close(depth[labels == 1], torch.full_like(depth[labels == 1], .8))

    def test_inactive_body_does_not_occlude_and_active_distractor_is_distinct(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        enabled[:, 1] = -1
        first = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertEqual(float(first[4][0, 0]), 1.)
        enabled[:, 1] = 2
        second = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertEqual(int(second[3][0, 0]), 0)
        self.assertTrue((second[1] == 2).any())

    def test_batch_equals_individual_and_rigid_transform_does_not_change_visibility(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture(batch=2)
        poses[1, 1, 0] = .4
        together = sensor._trace(poses, enabled, eyes, rotation, roles)
        for row in range(2):
            alone = sensor._trace(poses[row:row+1], enabled[row:row+1], eyes[row:row+1], rotation[row:row+1], roles[row:row+1])
            for actual, expected in zip(together, alone):
                torch.testing.assert_close(actual[row:row+1], expected)
        q = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
        shift = torch.tensor([1., -.5, 2.])
        transformed = poses.clone()
        transformed[..., :3] = obs.quat_apply(q.expand(2, 3, -1), poses[..., :3]) + shift
        transformed[..., 3:] = q
        rotated = sensor._trace(transformed, enabled, eyes + shift, q.expand(2, -1), roles)
        torch.testing.assert_close(rotated[1], together[1])
        torch.testing.assert_close(rotated[2], together[2])

    def test_partly_off_frame_role_is_rejected_using_full_silhouette(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        poses[:, 1, 0] = .5
        result = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertGreater(int(result[2][0, 1]), int(result[3][0, 1]))
        self.assertFalse(result[-1].any())

    def test_off_frame_silhouettes_follow_role_ids_not_body_order(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        poses[:, 1, 0] = .5
        expected = sensor._trace(poses, enabled, eyes, rotation, roles)
        # Move the role meshes away from slots 0/1; the last slot is a role.
        permutation = [2, 0, 1]
        active_ids = torch.where(enabled >= 0, enabled + 1, -1)
        actual = sensor._trace(poses[:, permutation], active_ids,
                               eyes, rotation, roles + 1)
        for index in (0, 2, 3, 4, 5):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        remapped = torch.where(expected[1] >= 0, expected[1] + 1, expected[1])
        torch.testing.assert_close(actual[1], remapped, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Visible-point reductions require CUDA.')
    def test_gpu_visible_fit_matches_cpu_with_background_and_hidden_objects(self):
        generator = torch.Generator().manual_seed(882)
        points = torch.rand(3, 19200, 3, generator=generator) * torch.tensor([.18, .06, .2])
        labels = torch.randint(-1, 5, (3, 19200), generator=generator, dtype=torch.int32)
        labels[:, :16000] = -1  # Large background bucket used to dominate atomics.
        points[:, :16000] = float('nan')  # Background must never contribute to an OBB.
        # The CPU oracle's background NaNs stay in its extra background bucket.
        expected = self.module.fit_visible_obbs(points, labels, 6, 16)
        actual = self.module.fit_visible_obbs(points.cuda(), labels.cuda(), 6, 16)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2.e-5, rtol=2.e-4)
        torch.testing.assert_close(actual[1].cpu(), expected[1])
        self.assertFalse(actual[1][:, 5].any())

    def test_projected_bounds_preserve_ray_hits_including_boxes_crossing_camera_plane(self):
        sensor, poses, active, eyes, rotation, roles = self.fixture(batch=4)
        poses[1, 1, 0] = .5
        poses[2, 1, 0] = -.5
        poses[3, 0, 2] = -.1
        actual = sensor._trace(poses, active, eyes, rotation, roles)
        # Deliberately loose bounds disable the coarse rejection. The same
        # actual triangle BVHs provide an independent uncropped query oracle.
        sensor.mesh_bounds = self.wp.array(np.array([[[-1.e4]*3, [1.e4]*3]]*3, np.float32),
                                           dtype=self.wp.vec3, device='cpu')
        expected = sensor._trace(poses, active, eyes, rotation, roles)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_visible_fit_uses_only_hits_and_leaves_hidden_slots_zero(self):
        points = torch.tensor([[[0., 0., .1], [.1, 0., .1], [0., .2, .3], [.1, .2, .3], [100., 100., 100.]]])
        labels = torch.tensor([[0, 0, 0, 0, -1]])
        geometry, valid = self.module.fit_visible_obbs(points, labels, 3, 3)
        self.assertEqual(valid.tolist(), [[True, False, False]])
        self.assertEqual(geometry[:, 1:].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(geometry).all())
        torch.testing.assert_close(geometry[0, 0, :3], torch.tensor([.05, .1, .2]))
        self.assertLess(float(geometry[0, 0, 7:].max()), .3)

    def test_visible_fit_batch_and_translation_parity_without_host_scalar_reads(self):
        generator = torch.Generator().manual_seed(91)
        points = torch.rand(3, 100, 3, generator=generator)
        labels = torch.randint(-1, 4, (3, 100), generator=generator)
        def reject(*args, **kwargs):
            raise AssertionError('Visible-surface fitting synchronized with the host')
        with patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'tolist', reject), \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'cpu', reject):
            together = self.module.fit_visible_obbs(points, labels, 4, 4)
        for row in range(3):
            single = self.module.fit_visible_obbs(points[row:row+1], labels[row:row+1], 4, 4)
            torch.testing.assert_close(together[0][row:row+1], single[0])
        shifted = self.module.fit_visible_obbs(points + .5, labels, 4, 4)
        torch.testing.assert_close(shifted[0][..., :3], together[0][..., :3] + .5)
        torch.testing.assert_close(shifted[0][..., 3:], together[0][..., 3:], atol=2.e-5, rtol=1.e-4)

    def test_headless_task_has_no_camera_or_reference_resources(self):
        root = Path(__file__).resolve().parents[1]
        cfg = (root / 'contact_manipulation/config/ur5e/ur5e_env_cfg.py').read_text()
        self.assertNotIn('make_initial_camera_cfg', cfg)
        self.assertNotIn('VisionReferencesCfg', cfg)
        self.assertNotIn('sim.render(', (root / 'contact_manipulation/mdp/sensors.py').read_text())
        for kwargs in ({'ray_width': 0}, {'ray_height': 13}, {'ray_batch_size': True},
                       {'ray_candidates_per_env': 0}, {'ray_candidates_per_env': True}):
            with self.assertRaises(ValueError):
                self.module.InitialVisionCfg(**kwargs)

    def test_viewpoints_are_seeded_bounded_and_independent_of_request_order(self):
        sensor, *_ = self.fixture()
        sensor.cfg.max_camera_attempts = 7
        sensor.cfg.roll_degrees = (0., 0.)
        sensor.cfg.aim_y_jitter = sensor.cfg.aim_z_jitter = 0.
        requests = [dict(seed=seed, workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.)
                    for seed in (19, 923, (1 << 40) + 19)]
        positions = torch.tensor([[[-.8, .1, 1.1], [-.7, .05, 1.1]]]).repeat(3, 1, 1)
        eyes, rotations = sensor._viewpoints(requests, positions)
        self.assertTrue(((eyes[..., 0] >= .05) & (eyes[..., 0] <= .45)).all())
        self.assertTrue(((eyes[..., 1] >= -.4) & (eyes[..., 1] <= .4)).all())
        self.assertTrue(((eyes[..., 2] >= 1.18) & (eyes[..., 2] <= 1.48)).all())
        torch.testing.assert_close(rotations.norm(dim=-1), torch.ones(3, 7))
        target = torch.tensor([-.75, .075, 1.1])
        forward = (target - eyes) / (target - eyes).norm(dim=-1, keepdim=True)
        torch.testing.assert_close(obs.quat_apply(rotations, torch.tensor([0., 0., 1.]).expand_as(eyes)), forward)
        self.assertFalse(torch.equal(eyes[0], eyes[2]))
        for row in (2, 0, 1):
            one = sensor._viewpoints([requests[row]], positions[row:row+1])
            torch.testing.assert_close(one[0][0], eyes[row], atol=0, rtol=0)
            torch.testing.assert_close(one[1][0], rotations[row], atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA visibility parity requires a GPU.')
    def test_cuda_ray_and_viewpoint_results_match_cpu(self):
        expected, actual = self.fixture(batch=2), self.fixture(batch=2, device='cuda')
        for fixture in (expected, actual):
            fixture[1][1, 1, 0] = .4
        reference = expected[0]._trace(*expected[1:])
        result = actual[0]._trace(*actual[1:])
        for first, second in zip(reference, result):
            torch.testing.assert_close(first, second.cpu())
        requests = [dict(seed=seed, workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.) for seed in (93, 1823)]
        positions = torch.tensor([[[-.8, .1, 1.1], [-.7, .05, 1.1]]]).repeat(2, 1, 1)
        reference = expected[0]._viewpoints(requests, positions)
        def reject(*args, **kwargs):
            raise AssertionError('GPU viewpoint generation read a tensor on the host')
        with patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'tolist', reject), \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'cpu', reject):
            result = actual[0]._viewpoints(requests, positions.cuda())
        for first, second in zip(reference, result):
            torch.testing.assert_close(first, second.cpu(), atol=1.e-6, rtol=1.e-6)

    def test_queue_selects_first_valid_view_refills_and_honors_attempt_limit(self):
        sensor, poses, *_ = self.fixture(batch=5)
        sensor.capacity = 2
        sensor.cfg.max_camera_attempts = 6  # Last group is smaller than four.
        robot = SimpleNamespace(root_pos_w=torch.zeros(5, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1))
        class Scene(dict):
            env_origins = torch.zeros(5, 3)
        scene = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                      robot=SimpleNamespace(data=robot))
        sensor.env = SimpleNamespace(scene=scene, device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        requests = [dict(env_id=row, active_ids=(0, 1), target_id=0, blocker_id=1, seed=row,
                         workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.) for row in range(5)]
        required = [0, 5, 2, 6, 4]  # Row 3 must fail: candidate 6 is beyond the search budget.
        calls = []
        def viewpoints(requests, role_positions):
            eyes = torch.zeros(len(requests), 6, 3)
            eyes[..., 0] = torch.tensor([r['env_id'] for r in requests])[:, None]
            eyes[..., 1] = torch.arange(6)
            rotations = torch.zeros(len(requests), 6, 4)
            rotations[..., 0] = 1.
            return eyes, rotations
        def trace(poses, active, eyes, rotations, roles):
            rows, attempts = eyes[:, 0].long(), eyes[:, 1].long()
            calls.append(list(zip(rows.tolist(), attempts.tolist())))
            accepted = attempts >= torch.tensor(required)[rows]
            depth = torch.ones(len(rows), 80*60)
            labels = torch.zeros_like(depth, dtype=torch.int32)
            labels[:, 80*30:] = 1
            visible = torch.full((len(rows), 2), 80*30, dtype=torch.int64)
            return depth, labels, visible, visible, torch.ones(len(rows), 2), accepted
        with patch.object(sensor, '_viewpoints', viewpoints), patch.object(sensor, '_trace', trace):
            results = sensor.capture_batch(requests, retain_images=True)
            self.assertEqual([None if r is None else r.attempts for r in results], [1, 6, 3, None, 5])
            self.assertEqual({row for row, attempt in calls[1]}, {1, 2})
            self.assertTrue(all(len(call) <= 8 for call in calls))
            reference = results
            sensor.cfg.ray_candidates_per_env = 1
            sensor.capacity = 3
            order = [4, 2, 1, 0, 3]
            results = sensor.capture_batch([requests[row] for row in order], retain_images=True)
        for result, row in zip(results, order):
            if row == 3:
                self.assertIsNone(result)
                continue
            self.assertEqual(result.attempts, reference[row].attempts)
            torch.testing.assert_close(result.geometry_b, reference[row].geometry_b)
            torch.testing.assert_close(result.camera_pose_w, reference[row].camera_pose_w)
            self.assertEqual(result.pixel_counts, ((2400, 2400), (2400, 2400)))
            self.assertEqual(result.depth.shape, (60, 80, 1))
            self.assertEqual(result.role_masks.shape, (60, 80, 2))
        self.assertEqual(sensor.capture_batch([]), [])

class HandContactProxyTests(unittest.TestCase):
    """Use the supplied CAD/URDF as an independent pad-frame geometry oracle."""

    @classmethod
    def setUpClass(cls):
        from itertools import product
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_tactile_parent_proxies')
        namespace = dict(np=np, product=product)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        cls.make = staticmethod(namespace['_tactile_parent_proxies'])

    def test_supplied_hand_has_two_small_convexes_per_parent_and_exposed_pad_faces(self):
        import trimesh
        from xml.etree import ElementTree
        base = (Path(__file__).resolve().parents[3] / 'assets/robots/Roas_provided_inspire_urdf'
                / 'RH56E2 URDF/RH56E2_R_2025_9_11')
        root = ElementTree.parse(base / 'urdf/RH56E2_R_2025_9_11_relative.urdf').getroot()
        for name in ('right_plam_1', 'right_index_2', 'right_middle_2', 'right_ring_2', 'right_little_2'):
            points = trimesh.load(base / 'meshes' / (name+'.STL'), force='mesh').vertices
            pads, frames = [], []
            for joint in root.findall('joint'):
                if joint.find('parent').get('link') != name or 'force_sensor' not in joint.find('child').get('link'):
                    continue
                pose = joint.find('origin').attrib
                rotation = Rotation.from_euler('xyz', np.fromstring(pose['rpy'], sep=' ')).as_matrix()
                position = np.fromstring(pose['xyz'], sep=' ')
                local = trimesh.load(base / 'meshes' / (joint.find('child').get('link')+'.STL'), force='mesh').vertices
                pads.append(local @ rotation.T + position)
                frames.append(rotation)
            pieces = self.make(points, pads)
            self.assertEqual(len(pieces), 2)
            hulls = []
            for vertices, faces in pieces:
                self.assertLessEqual(len(vertices), 64)
                hull = ConvexHull(vertices)
                self.assertGreater(hull.volume, 0)
                self.assertEqual(len(faces), len(hull.simplices))
                hulls.append(hull.equations)
            # Do not silently remove a large external part of the supplied hand.
            error = np.minimum.reduce([(points @ h[:, :3].T+h[:, 3]).max(-1) for h in hulls])
            self.assertLess(float(error.max()), .002, name)
            for pad, rotation in zip(pads, frames):
                normal = rotation[:, 2]  # independent authored URDF outward face
                for u, v in ((0., 0.), (.3, 0.), (-.3, 0.), (0., .3), (0., -.3)):
                    origin = pad.mean(0) + u*np.ptp(pad@rotation[:, 0])*rotation[:, 0] + v*np.ptp(pad@rotation[:, 1])*rotation[:, 1]
                    front = []
                    for h in hulls:
                        slope = h[:, :3]@normal
                        residual = h[:, :3]@origin+h[:, 3]
                        if np.any((abs(slope)<1.e-8)&(residual>1.e-8)):
                            continue
                        lo = np.max(-residual[slope < -1.e-8]/slope[slope < -1.e-8], initial=-np.inf)
                        hi = np.min(-residual[slope > 1.e-8]/slope[slope > 1.e-8], initial=np.inf)
                        if lo <= hi:
                            front.append(hi)
                    self.assertTrue(front)
                    self.assertLess(max(front), float(((pad-origin)@normal).max())-.0005, name)


class NativeGeometrySharingTests(unittest.TestCase):
    """Run with Kit's USD modules available; ordinary CPU discovery skips these."""

    @classmethod
    def setUpClass(cls):
        try:
            from pxr import Usd, Sdf, UsdGeom, UsdPhysics, UsdShade
        except ImportError:
            raise unittest.SkipTest('Native USD geometry tests require an initialized Isaac Sim app.')
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in ('_deinstance_subtree', 'share_rigid_body_geometry')]
        namespace = dict(Usd=Usd, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        cls.share = staticmethod(namespace['share_rigid_body_geometry'])
        cls.Usd, cls.Sdf, cls.UsdGeom, cls.UsdPhysics, cls.UsdShade = Usd, Sdf, UsdGeom, UsdPhysics, UsdShade

    def fixture(self):
        stage = self.Usd.Stage.CreateInMemory()
        self.UsdGeom.Xform.Define(stage, '/World/envs/env_0')
        body = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/body')
        body.AddTranslateOp().Set((1., 2., 3.))
        self.UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        self.UsdGeom.Xform.Define(stage, '/Source')
        mesh = self.UsdGeom.Mesh.Define(stage, '/Source/mesh')
        mesh.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        material = self.UsdShade.Material.Define(stage, '/Looks/Paint')
        shader = self.UsdShade.Shader.Define(stage, '/Looks/Paint/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', self.Sdf.ValueTypeNames.Color3f).Set((.1, .2, .3))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        self.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        visual = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/body/visuals').GetPrim()
        visual.GetReferences().AddInternalReference('/Source')
        visual.SetInstanceable(True)
        collider = self.UsdGeom.Cube.Define(stage, '/World/envs/env_0/body/collider')
        collider.CreateSizeAttr(.25)
        self.UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        return stage, body.GetPrim()

    def test_nested_mesh_instances_material_and_collision_survive_and_replicas_share_geometry(self):
        import tempfile
        stage, body = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'geometry.usdc')), 1)
            self.assertTrue(body.IsInstance())
            mesh = self.UsdGeom.Mesh(stage.GetPrimAtPath('/World/envs/env_0/body/visuals/mesh'))
            self.assertEqual(list(mesh.GetFaceVertexIndicesAttr().Get()), [0, 1, 2])
            np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), [[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            material, _ = self.UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            self.assertTrue(material)
            surface = self.UsdShade.Shader(material.GetPrim().GetChild('Shader'))
            np.testing.assert_allclose(surface.GetInput('diffuseColor').Get(), [.1, .2, .3])
            collider = stage.GetPrimAtPath('/World/envs/env_0/body/collider')
            self.assertTrue(self.UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())
            self.assertEqual(self.UsdGeom.Cube(collider).GetSizeAttr().Get(), .25)
            copy = self.UsdGeom.Xform.Define(stage, '/World/envs/env_1').GetPrim()
            copy.GetInherits().AddInherit('/World/envs/env_0')
            other = stage.GetPrimAtPath('/World/envs/env_1/body')
            self.assertEqual(body.GetPrototype(), other.GetPrototype())
            other.GetAttribute('xformOp:translate').Set((4., 5., 6.))
            self.assertEqual(tuple(body.GetAttribute('xformOp:translate').Get()), (1., 2., 3.))
            self.assertTrue(self.UsdPhysics.RigidBodyAPI(other))
            self.assertFalse(other.IsInstanceProxy())

    def test_joint_or_nested_body_root_is_not_made_an_instance(self):
        import tempfile
        stage, body = self.fixture()
        self.UsdPhysics.FixedJoint.Define(stage, '/World/envs/env_0/body/joint')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'geometry.usdc')), 0)
            self.assertFalse(body.IsInstance())

    def test_hand_proxy_replaces_referenced_collision_after_sharing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = self.Usd.Stage.CreateNew(str(Path(directory) / 'source.usda'))
            self.UsdGeom.Xform.Define(source, '/Hand')
            visual = self.UsdGeom.Mesh.Define(source, '/Hand/visual')
            visual.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
            visual.CreateFaceVertexCountsAttr([3])
            visual.CreateFaceVertexIndicesAttr([0, 1, 2])
            original = self.UsdGeom.Cube.Define(source, '/Hand/collisions/original').GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(original).CreateCollisionEnabledAttr(True)
            source.GetRootLayer().Save()
            stage = self.Usd.Stage.CreateInMemory()
            body = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/hand').GetPrim()
            body.GetReferences().AddReference(source.GetRootLayer().identifier, '/Hand')
            self.UsdPhysics.RigidBodyAPI.Apply(body)
            body.CreateAttribute('contactManipulation:handProxy', self.Sdf.ValueTypeNames.Bool).Set(True)
            stage.GetPrimAtPath(body.GetPath().AppendPath('collisions/original')).SetActive(False)
            proxy = self.UsdGeom.Cube.Define(stage, body.GetPath().AppendChild('proxy')).GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'shared.usdc')), 1)
            colliders = [p.GetName() for p in self.Usd.PrimRange(body, self.Usd.TraverseInstanceProxies())
                         if p.HasAPI(self.UsdPhysics.CollisionAPI)]
            self.assertEqual(colliders, ['proxy'])
            self.assertTrue(body.IsInstance())

    def test_reference_collision_overrides_survive_sharing_without_hand_marker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = self.Usd.Stage.CreateNew(str(Path(directory) / 'source.usda'))
            body = self.UsdGeom.Xform.Define(source, '/Object').GetPrim()
            self.UsdPhysics.RigidBodyAPI.Apply(body)
            mesh = self.UsdGeom.Mesh.Define(source, '/Object/mesh')
            mesh.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            self.UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            mesh.GetPrim().CreateAttribute('physxCollision:contactOffset', self.Sdf.ValueTypeNames.Float).Set(.02)
            source.GetRootLayer().Save()
            stage = self.Usd.Stage.CreateInMemory()
            live = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/object').GetPrim()
            live.GetReferences().AddReference(source.GetRootLayer().identifier, '/Object')
            original = stage.GetPrimAtPath(live.GetPath().AppendChild('mesh'))
            self.UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
            original.GetAttribute('physxCollision:contactOffset').Set(.005)
            proxy = self.UsdGeom.Cube.Define(stage, live.GetPath().AppendChild('proxy')).GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'shared.usdc')), 1)
            original = stage.GetPrimAtPath(live.GetPath().AppendChild('mesh'))
            self.assertFalse(self.UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Get())
            self.assertAlmostEqual(original.GetAttribute('physxCollision:contactOffset').Get(), .005)
            self.assertEqual(len(self.UsdGeom.Mesh(original).GetPointsAttr().Get()), 3)


if __name__ == "__main__":
    unittest.main()
