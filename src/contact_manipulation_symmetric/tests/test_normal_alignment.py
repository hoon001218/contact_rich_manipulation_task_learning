"""CPU invariants for the directional tactile preparation reward; no Kit scene."""

from __future__ import annotations

import ast
from contextlib import ExitStack
import inspect
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

# Reuse the checkout's existing lightweight Isaac Lab math/MDP import fixture.
from test_mdp import Rotation, native_reward_compute, rewards, task


DECAY_45 = math.log(2.0) / (1.0 - math.cos(math.pi / 4.0))
FINGERTIP_INDICES = (4, 7, 10, 13, 16)
PALMAR_INDICES = (0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15)


def facing_inputs(batch=1, dtype=torch.float64):
    points = torch.zeros((batch, 17, 3), dtype=dtype)
    points[..., 0] = -1.0
    normals = torch.zeros_like(points)
    normals[..., 0] = 1.0
    center = torch.zeros((batch, 3), dtype=dtype)
    return points, normals, center


class NormalAlignmentQualityTests(unittest.TestCase):
    def test_facing_45_degrees_tangent_and_back_use_signed_cosine(self):
        points, normals, center = facing_inputs(batch=4)
        for row, angle in enumerate((0.0, math.pi / 4.0, math.pi / 2.0, math.pi)):
            normals[row, :, 0] = math.cos(angle)
            normals[row, :, 1] = math.sin(angle)
        actual = rewards.blocker_normal_alignment_quality(points, normals, center)
        expected = center.new_tensor([1.0, .5, math.exp(-DECAY_45), math.exp(-2 * DECAY_45)])
        torch.testing.assert_close(actual, expected)
        self.assertGreater(float(actual[0]), float(actual[-1]))
        self.assertGreater(float(actual[1]), float(actual[2]))

    def test_each_pad_is_exponentiated_before_the_fixed_17_pad_mean(self):
        points, normals, center = facing_inputs()
        normals[:, 8:, 0] = -1.0
        actual = rewards.blocker_normal_alignment_quality(points, normals, center, decay=2.0)
        expected = actual.new_tensor([(8.0 + 9.0 * math.exp(-4.0)) / 17.0])
        torch.testing.assert_close(actual, expected)
        exp_after_mean = math.exp(-2.0 * (1.0 - (8.0 - 9.0) / 17.0))
        self.assertGreater(abs(float(actual[0]) - exp_after_mean), .1)

    def test_direction_score_is_distance_invariant_and_normal_magnitude_invariant(self):
        points, normals, center = facing_inputs(batch=3)
        normals[..., 0] = math.cos(.7)
        normals[..., 1] = math.sin(.7)
        reference = rewards.blocker_normal_alignment_quality(points, normals, center)
        points *= points.new_tensor([.001, .4, 100.0])[:, None, None]
        normals *= normals.new_tensor([.2, 3.0, 100.0])[:, None, None]
        torch.testing.assert_close(rewards.blocker_normal_alignment_quality(points, normals, center), reference)

    def test_common_three_dimensional_rigid_transform_does_not_change_quality(self):
        generator = torch.Generator().manual_seed(23)
        points = torch.randn((3, 17, 3), generator=generator, dtype=torch.float64)
        normals = torch.randn((3, 17, 3), generator=generator, dtype=torch.float64)
        center = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        original = [value.clone() for value in (points, normals, center)]
        expected = rewards.blocker_normal_alignment_quality(points, normals, center)
        rotation = points.new_tensor(Rotation.from_rotvec([.33, -.57, 1.13]).as_matrix())
        translation = points.new_tensor([3.0, -2.5, 7.0])
        transformed = rewards.blocker_normal_alignment_quality(
            points @ rotation.T + translation, normals @ rotation.T, center @ rotation.T + translation)
        torch.testing.assert_close(transformed, expected)
        for before, after in zip(original, (points, normals, center)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_invalid_and_coincident_pads_are_zero_without_changing_denominator(self):
        points, normals, center = facing_inputs(batch=3)
        points[0, 0, 0] = math.nan
        normals[0, 1, 1] = math.inf
        normals[0, 2] = 0.0
        points[0, 3] = center[0]
        center[1, 0] = math.nan
        center[2, 2] = math.inf
        actual = rewards.blocker_normal_alignment_quality(points, normals, center)
        torch.testing.assert_close(actual, actual.new_tensor([13.0 / 17.0, 0.0, 0.0]))
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_no_direction_or_no_surface_normal_never_fabricates_alignment(self):
        points, normals, center = facing_inputs(batch=2)
        points[0] = center[0]
        normals[1] = 0.0
        actual = rewards.blocker_normal_alignment_quality(points, normals, center)
        torch.testing.assert_close(actual, torch.zeros(2, dtype=points.dtype))

    def test_any_positive_pad_count_preserves_fixed_mean_and_batch_contract(self):
        for batch in (0, 1024):
            for dtype in (torch.float32, torch.float64):
                for pads in (1, 12, 16, 17):
                    with self.subTest(batch=batch, dtype=dtype, pads=pads):
                        points, normals, center = facing_inputs(batch, dtype)
                        actual = rewards.blocker_normal_alignment_quality(points[:, :pads], normals[:, :pads], center)
                        self.assertEqual(actual.shape, (batch,))
                        self.assertEqual(actual.dtype, dtype)
                        self.assertEqual(actual.device, points.device)
                        torch.testing.assert_close(actual, torch.ones(batch, dtype=dtype))

    def test_hot_path_uses_no_host_transfer_or_tensor_scalar_condition(self):
        points, normals, center = facing_inputs(batch=1024, dtype=torch.float32)
        with ExitStack() as stack:
            for method in ("cpu", "item", "tolist", "numpy", "__bool__"):
                stack.enter_context(patch.object(torch.Tensor, method,
                    side_effect=AssertionError(f"Unexpected host/scalar operation: {method}")))
            actual = rewards.blocker_normal_alignment_quality(points, normals, center)
        torch.testing.assert_close(actual, torch.ones(1024))

    def test_malformed_shapes_are_rejected_instead_of_broadcasting_environments(self):
        points, normals, center = facing_inputs(batch=2)
        cases = ((points[:, :0], normals[:, :0], center),
                 (points, normals[:1], center),
                 (points, normals, center[:1]),
                 (points[..., :2], normals, center),
                 (points, normals[..., :2], center),
                 (points, normals, torch.zeros((2, 4), dtype=center.dtype)),
                 (points[0], normals[0], center[0]))
        for bad_points, bad_normals, bad_center in cases:
            with self.subTest(shapes=(bad_points.shape, bad_normals.shape, bad_center.shape)), \
                    self.assertRaises(ValueError):
                rewards.blocker_normal_alignment_quality(bad_points, bad_normals, bad_center)

    def test_nonpositive_and_nonfinite_decay_are_rejected(self):
        points, normals, center = facing_inputs()
        for decay in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(decay=decay), self.assertRaises(ValueError):
                rewards.blocker_normal_alignment_quality(points, normals, center, decay=decay)


class TactilePadFramesTests(unittest.TestCase):
    @staticmethod
    def cached_scene(batch=2):
        # Native articulation order deliberately differs from canonical pad
        # order, and local offsets are surface points rather than link origins.
        positions = torch.arange(batch * 19 * 3, dtype=torch.float64).reshape(batch, 19, 3) * .01
        rotations = torch.zeros((batch, 19, 4), dtype=positions.dtype)
        rotations[..., 0] = 1.0
        robot = SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions, body_link_quat_w=rotations))
        indices = torch.arange(17, 0, -1)
        offsets = torch.stack((torch.arange(17, dtype=positions.dtype) * .0001,
                               torch.full((17,), .002, dtype=positions.dtype),
                               torch.full((17,), .005, dtype=positions.dtype)), -1)
        binding = (robot, indices, offsets)
        env = SimpleNamespace(scene={"robot": robot}, device="cpu", _tactile_reward_geometry=binding)
        return env, robot, indices, offsets

    def test_cached_three_field_binding_reorders_native_bodies_and_rotates_surface_offsets(self):
        env, robot, indices, offsets = self.cached_scene()
        robot.data.body_link_quat_w[..., 0] = math.sqrt(.5)
        robot.data.body_link_quat_w[0, :, 2] = math.sqrt(.5)
        robot.data.body_link_quat_w[1, :, 2] = -math.sqrt(.5)
        points, normals = rewards.tactile_pad_frames(env)
        rotated_offsets = torch.stack((torch.stack((offsets[:, 2], offsets[:, 1], -offsets[:, 0]), -1),
                                       torch.stack((-offsets[:, 2], offsets[:, 1], offsets[:, 0]), -1)))
        expected_points = robot.data.body_link_pos_w[:, indices] + rotated_offsets
        expected_normals = torch.zeros_like(points)
        expected_normals[0, :, 0] = 1.0
        expected_normals[1, :, 0] = -1.0
        torch.testing.assert_close(points, expected_points)
        torch.testing.assert_close(normals, expected_normals, atol=1.e-14, rtol=0)
        torch.testing.assert_close(rewards.tactile_pad_points(env), points)
        self.assertEqual(len(env._tactile_reward_geometry), 3)

    def test_current_native_pose_updates_while_local_normal_axis_and_geometry_are_reused(self):
        env, robot, indices, offsets = self.cached_scene()
        initial_points, initial_normals = rewards.tactile_pad_frames(env)
        binding = env._tactile_reward_geometry
        axis = env._tactile_reward_normal_axis
        translation = initial_points.new_tensor([2.0, -1.0, .5])
        robot.data.body_link_pos_w.add_(translation)
        robot.data.body_link_quat_w[..., 0] = math.sqrt(.5)
        robot.data.body_link_quat_w[..., 2] = math.sqrt(.5)
        with ExitStack() as stack:
            for method in ("cpu", "item", "tolist", "numpy", "__bool__"):
                stack.enter_context(patch.object(torch.Tensor, method,
                    side_effect=AssertionError(f"Unexpected host/scalar operation: {method}")))
            updated_points, updated_normals = rewards.tactile_pad_frames(env)
        rotated_offsets = torch.stack((offsets[:, 2], offsets[:, 1], -offsets[:, 0]), -1)
        expected = robot.data.body_link_pos_w[:, indices] + rotated_offsets
        torch.testing.assert_close(updated_points, expected)
        expected_normals = torch.zeros_like(updated_normals)
        expected_normals[..., 0] = 1.0
        torch.testing.assert_close(updated_normals, expected_normals, atol=1.e-14, rtol=0)
        self.assertIs(env._tactile_reward_geometry, binding)
        self.assertIs(env._tactile_reward_normal_axis, axis)
        torch.testing.assert_close(axis, initial_normals[0])
        self.assertFalse(torch.allclose(updated_points, initial_points))

    def test_zero_native_quaternion_cannot_create_a_spurious_facing_pad(self):
        env, robot, indices, offsets = self.cached_scene(batch=1)
        robot.data.body_link_pos_w.zero_()
        robot.data.body_link_pos_w[..., 2] = -1.0
        offsets.zero_()
        robot.data.body_link_quat_w[:, indices[3]] = 0.0
        points, normals = rewards.tactile_pad_frames(env)
        center = points.new_zeros((1, 3))
        quality = rewards.blocker_normal_alignment_quality(points, normals, center)
        torch.testing.assert_close(quality, quality.new_tensor([16.0 / 17.0]))


class NormalAlignmentIntegrationTests(unittest.TestCase):
    @staticmethod
    def reward_scene(batch=1):
        class Scene(dict):
            pass

        scene = Scene()
        scene.env_origins = torch.zeros((batch, 3), dtype=torch.float64)
        env = SimpleNamespace(scene=scene, step_dt=.02,
                              _tactile_alignment_pad_indices=torch.tensor(PALMAR_INDICES))
        points, normals, center = facing_inputs(batch=batch)
        return env, points, normals, SimpleNamespace(blocker_center=center)

    def test_wrapper_uses_current_blocker_center_and_environment_origin_without_extra_gates(self):
        class Scene(dict):
            env_origins = torch.tensor([[10.0, -4.0, 2.0], [-7.0, 12.0, -3.0]], dtype=torch.float64)

        local_points, normals, local_center = facing_inputs(batch=2)
        env = SimpleNamespace(scene=Scene(), step_dt=.02,
                              _tactile_alignment_pad_indices=torch.tensor(PALMAR_INDICES))
        world_points = local_points + env.scene.env_origins[:, None]
        state = SimpleNamespace(blocker_center=local_center.clone(),
                                target_protected=torch.zeros(2, dtype=torch.bool),
                                failure=torch.ones(2, dtype=torch.bool))
        with patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(world_points, normals)):
            facing = rewards.blocker_normal_alignment(env)
            state.blocker_center[:, 0] = -2.0
            behind = rewards.blocker_normal_alignment(env)
        torch.testing.assert_close(facing, torch.ones(2, dtype=world_points.dtype))
        torch.testing.assert_close(behind, facing.new_full((2,), math.exp(-2 * DECAY_45)))

    def test_five_fingertip_positions_and_normals_do_not_affect_default_reward(self):
        env, points, normals, state = self.reward_scene(batch=2)
        originals = (points.clone(), normals.clone())
        with patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(points, normals)):
            expected = rewards.blocker_normal_alignment(env)
            points[:, FINGERTIP_INDICES] = math.nan
            normals[:, FINGERTIP_INDICES] = math.inf
            unchanged = rewards.blocker_normal_alignment(env)
        torch.testing.assert_close(expected, torch.ones(2, dtype=points.dtype))
        torch.testing.assert_close(unchanged, expected)
        # Selection is local to this reward; all 17 pad inputs still exist and
        # the retained pad geometry is untouched for other rewards/sensors.
        self.assertEqual(points.shape, (2, 17, 3))
        self.assertEqual(normals.shape, (2, 17, 3))
        torch.testing.assert_close(points[:, PALMAR_INDICES], originals[0][:, PALMAR_INDICES])
        torch.testing.assert_close(normals[:, PALMAR_INDICES], originals[1][:, PALMAR_INDICES])

    def test_invalid_retained_pad_is_zero_with_fixed_twelve_pad_denominator(self):
        env, points, normals, state = self.reward_scene()
        normals[:, PALMAR_INDICES[3]] = 0.0
        points[:, PALMAR_INDICES[5]] = state.blocker_center
        with patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(points, normals)):
            actual = rewards.blocker_normal_alignment(env)
        torch.testing.assert_close(actual, actual.new_tensor([10.0 / 12.0]))

    def test_facing_only_fingertips_cannot_replace_palmar_alignment_and_legacy_is_explicit(self):
        env, points, normals, state = self.reward_scene()
        normals[:, PALMAR_INDICES, 0] = -1.0
        with patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(points, normals)):
            selected = rewards.blocker_normal_alignment(env)
            legacy = rewards.blocker_normal_alignment(env, exclude_fingertips=False)
        opposed = math.exp(-2 * DECAY_45)
        torch.testing.assert_close(selected, selected.new_tensor([opposed]))
        torch.testing.assert_close(legacy, legacy.new_tensor([(12 * opposed + 5) / 17]))
        self.assertLess(float(selected[0]), float(legacy[0]))

    def test_cached_selection_reuses_indices_without_host_reads_on_current_frames(self):
        env, points, normals, state = self.reward_scene(batch=1024)
        indices = env._tactile_alignment_pad_indices
        with patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(points, normals)):
            with ExitStack() as stack:
                for method in ("cpu", "item", "tolist", "numpy", "__bool__"):
                    stack.enter_context(patch.object(torch.Tensor, method,
                        side_effect=AssertionError(f"Unexpected host/scalar operation: {method}")))
                actual = rewards.blocker_normal_alignment(env)
        self.assertIs(env._tactile_alignment_pad_indices, indices)
        torch.testing.assert_close(actual, torch.ones(1024, dtype=points.dtype))

    def test_canonical_fingertip_names_build_and_cache_only_the_twelve_retained_indices(self):
        source = Path(__file__).resolve().parents[1] / "contact_manipulation_symmetric/config/ur5e/scene.py"
        syntax = ast.parse(source.read_text())
        namespace = {}
        for assignment in syntax.body:
            if (isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                    and isinstance(assignment.targets[0], ast.Name)
                    and assignment.targets[0].id in {"TCP_BODY_NAME", "TACTILE_BODY_NAMES", "FINGERTIP_BODY_NAMES"}):
                exec(compile(ast.Module(body=[assignment], type_ignores=[]), str(source), "exec"), namespace)
        self.assertEqual(namespace["FINGERTIP_BODY_NAMES"], (
            "right_thumb_force_sensor_4", "right_index_force_sensor_3", "right_middle_force_sensor_3",
            "right_ring_force_sensor_3", "right_little_force_sensor_3"))
        self.assertEqual(tuple(namespace["TACTILE_BODY_NAMES"].index(name)
                               for name in namespace["FINGERTIP_BODY_NAMES"]), FINGERTIP_INDICES)
        env, points, normals, state = self.reward_scene()
        del env._tactile_alignment_pad_indices
        scene_name = rewards.__package__.rsplit(".", 1)[0] + ".config.ur5e.scene"
        fake_scene = SimpleNamespace(TACTILE_BODY_NAMES=namespace["TACTILE_BODY_NAMES"],
                                     FINGERTIP_BODY_NAMES=namespace["FINGERTIP_BODY_NAMES"])
        with patch.dict(sys.modules, {scene_name: fake_scene}), \
                patch.object(task, "get_task_state", return_value=state), \
                patch.object(rewards, "tactile_pad_frames", return_value=(points, normals)):
            actual = rewards.blocker_normal_alignment(env)
            cached = env._tactile_alignment_pad_indices
            rewards.blocker_normal_alignment(env)
        torch.testing.assert_close(actual, torch.ones(1, dtype=points.dtype))
        torch.testing.assert_close(cached, torch.tensor(PALMAR_INDICES))
        self.assertIs(env._tactile_alignment_pad_indices, cached)

    def test_native_reward_manager_applies_dt_once_to_the_unscaled_quality(self):
        points, normals, center = facing_inputs(batch=2)
        normals[1, :, 0] = 0.0
        normals[1, :, 1] = 1.0
        quality = rewards.blocker_normal_alignment_quality(points, normals, center)
        for dt in (.01, .02):
            env = SimpleNamespace(step_dt=dt)
            manager = SimpleNamespace(
                _env=env, _term_names=["blocker_normal_alignment"],
                _term_cfgs=[SimpleNamespace(func=lambda _env: quality, weight=1.0, params={})],
                _reward_buf=torch.zeros_like(quality), _step_reward=quality.new_zeros((2, 1)),
                _episode_sums={"blocker_normal_alignment": torch.zeros_like(quality)})
            actual = native_reward_compute()(manager, dt)
            torch.testing.assert_close(actual, quality * dt)
            torch.testing.assert_close(manager._episode_sums["blocker_normal_alignment"], quality * dt)
            torch.testing.assert_close(manager._step_reward[:, 0], quality)

    def test_active_config_has_unit_weight_and_45_degree_half_quality(self):
        source = Path(__file__).resolve().parents[1] / "contact_manipulation_symmetric/shelf_manipulation_env_cfg.py"
        syntax = ast.parse(source.read_text())
        config = next(node for node in syntax.body if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
        terms = []
        for assignment in config.body:
            if not isinstance(assignment, (ast.Assign, ast.AnnAssign)):
                continue
            value = assignment.value
            if not isinstance(value, ast.Call):
                continue
            keywords = {keyword.arg: keyword.value for keyword in value.keywords}
            function = keywords.get("func")
            if isinstance(function, ast.Attribute) and function.attr == "blocker_normal_alignment":
                terms.append(keywords)
        self.assertEqual(len(terms), 1)
        self.assertEqual(ast.literal_eval(terms[0]["weight"]), 1.0)
        parameters = eval(compile(ast.Expression(terms[0]["params"]), str(source), "eval"), {"math": math})
        self.assertAlmostEqual(parameters["decay"], DECAY_45)
        self.assertIs(parameters["exclude_fingertips"], True)
        self.assertAlmostEqual(math.exp(-parameters["decay"] * (1 - math.cos(math.pi / 4))), .5)
        default = inspect.signature(rewards.blocker_normal_alignment_quality).parameters["decay"].default
        self.assertAlmostEqual(default, DECAY_45)


if __name__ == "__main__":
    unittest.main()
