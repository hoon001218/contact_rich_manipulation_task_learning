"""Example-style object-drop coverage; source prepared without running training.

The generic native fixture has its floor at world Z=0, so it overrides the
height threshold to 0 m. Tests below explicitly restore the task's 1.04 m
threshold when checking actual world-height and object-origin semantics.
"""

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_mdp import PhysicalStateCapacityTests, fixture, identity, native_fixture, physical_inputs, task, terminations


def axis_quaternion(angle, axis=0, *, dtype=torch.float64):
    quaternion = torch.zeros(4, dtype=dtype)
    quaternion[0], quaternion[axis + 1] = math.cos(angle / 2), math.sin(angle / 2)
    return quaternion


def drop_term():
    return SimpleNamespace(func=terminations.failure, params={"reason": "object_drop"}, time_out=False)


class ObjectDropPredicateTests(unittest.TestCase):
    def test_height_or_either_signed_roll_pitch_is_immediate_but_yaw_is_free(self):
        height = torch.full((10, 2), 1.05, dtype=torch.float64)
        rotation = identity(10, 2)
        active = torch.ones_like(height, dtype=torch.bool)
        height[0, 1], height[1, 1] = 1.0399, 1.04
        for row, angle, axis in ((2, .9001, 0), (3, -.9001, 0),
                                 (4, .9001, 1), (5, -.9001, 1),
                                 (6, .8999, 0), (7, -.8999, 1), (8, math.pi, 2)):
            rotation[row, 1] = axis_quaternion(angle, axis)
        expected = torch.tensor([True, False, True, True, True, True, False, False, False, False])
        torch.testing.assert_close(terminations.object_drop_condition(height, rotation, active), expected)
        # A single completed sample is sufficient; no prior tilt/loss counter.
        torch.testing.assert_close(terminations.object_drop_condition(height, rotation, active), expected)

    def test_inactive_and_nonfinite_or_zero_quaternion_slots_do_not_invent_drop_failures(self):
        height = torch.tensor([[.9, 1.05], [math.nan, 1.05], [1.05, 1.05],
                               [1.05, 1.05], [.9, 1.05]], dtype=torch.float64)
        rotation = identity(5, 2)
        active = torch.ones_like(height, dtype=torch.bool)
        active[0, 0] = False
        rotation[2, 0] = math.nan
        rotation[3, 0] = 0.
        rotation[4, 0] *= 3.  # Finite nonzero quaternion drift is normalized.
        expected = torch.tensor([False, False, False, False, True])
        torch.testing.assert_close(terminations.object_drop_condition(height, rotation, active), expected)

    def test_upright_import_rotation_is_removed_before_example_roll_pitch_test(self):
        height = torch.full((1, 1), 1.05, dtype=torch.float64)
        active = torch.ones_like(height, dtype=torch.bool)
        # YCB's raw link frame uses -90 degrees roll in a physically upright pose.
        upright = axis_quaternion(-math.pi / 2).reshape(1, 1, 4)
        self.assertTrue(terminations.object_drop_condition(height, upright, active).all())
        canonical = task.quat_mul(upright, task.quat_conjugate(upright))
        self.assertFalse(terminations.object_drop_condition(height, canonical, active).any())
        tilted_raw = task.quat_mul(axis_quaternion(.91).reshape(1, 1, 4), upright)
        canonical = task.quat_mul(tilted_raw, task.quat_conjugate(upright))
        self.assertTrue(terminations.object_drop_condition(height, canonical, active).all())

    def test_thresholds_are_configurable_and_invalid_shapes_or_thresholds_are_rejected(self):
        height, rotation = torch.tensor([[1.03]], dtype=torch.float64), identity(1, 1)
        active = torch.ones_like(height, dtype=torch.bool)
        rotation[0, 0] = axis_quaternion(.95)
        self.assertTrue(terminations.object_drop_condition(height, rotation, active).all())
        self.assertFalse(terminations.object_drop_condition(
            height, rotation, active, height_condition=1., rotation_condition=1.).any())
        for kwargs in ({"height_condition": math.nan}, {"rotation_condition": 0.},
                       {"rotation_condition": math.pi + .1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                terminations.object_drop_condition(height, rotation, active, **kwargs)
        with self.assertRaises(ValueError):
            terminations.object_drop_condition(height[:, 0], rotation, active)
        with self.assertRaises(ValueError):
            terminations.object_drop_condition(height, rotation, active.to(torch.int64))

    def test_drop_prevents_success_only_when_enabled_and_diagnostic_cause_stays_visible(self):
        data = physical_inputs(2)
        data["object_drop"] = torch.tensor([True, False])
        index = terminations.FAILURE_REASON_NAMES.index("object_drop")
        for enabled in (True, False):
            mask, _ = terminations.configured_terminations(
                {"object_drop": drop_term() if enabled else None}, device="cpu")
            candidate, stable, _, reasons, failed = terminations.task_conditions(
                task.MdpCfg(), **data, failure_mask=mask)
            torch.testing.assert_close(reasons[:, index], torch.tensor([True, False]))
            torch.testing.assert_close(failed, torch.tensor([enabled, False]))
            torch.testing.assert_close(candidate, ~failed)
            torch.testing.assert_close(stable, (~failed).long())

    def test_predicate_has_no_tensor_scalar_reads_or_host_transfers(self):
        height = torch.full((3, 25), 1.05, dtype=torch.float64)
        rotation, active = identity(3, 25), torch.ones_like(height, dtype=torch.bool)
        height[1, 24] = 1.
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host condition")), \
             patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")):
            result = terminations.object_drop_condition(height, rotation, active)
        torch.testing.assert_close(result, torch.tensor([False, True, False]))


class ObjectDropIntegrationTests(unittest.TestCase):
    @staticmethod
    def native(n=3):
        env, _, _ = native_fixture(n)
        env.cfg.mdp.object_drop_height = 1.04
        env.cfg.terminations = {"object_drop": drop_term()}
        env.scene["object_collection"].data.object_link_pose_w[..., 2] = 1.05
        return env

    def test_native_default_is_link_origin_world_height_not_obb_center_or_collision_bottom(self):
        env = self.native(3)
        source = env.task_scene_data
        # Actual Blocker is physical object 0, mapped to active-state slot 1.
        # Its OBB center is well above the link origin; its bottom is even higher.
        source.lower[0, 2], source.upper[0, 2] = .3, .5
        task.reset_task(env)
        poses = env.scene["object_collection"].data.object_link_pose_w
        poses[0, 0, 2] = 1.039
        poses[1, 0, 2] = 1.04
        # Physical object 1 is unused in all three rows and must be ignored.
        poses[2, 1, 2] = .1
        env.common_step_counter = 1
        state = task.get_task_state(env)
        self.assertGreater(float(state.bindings.local_center[0, 1, 2]), .39)
        torch.testing.assert_close(terminations.failure(env, reason="object_drop"),
                                   torch.tensor([True, False, False]))
        torch.testing.assert_close(state.failure, torch.tensor([True, False, False]))
        torch.testing.assert_close(env.extras["terminated"], state.failure)

    def test_native_ycb_upright_frame_stays_valid_and_physical_roll_fails_immediately(self):
        env = self.native(2)
        poses = env.scene["object_collection"].data.object_link_pose_w
        upright = axis_quaternion(-math.pi / 2, dtype=poses.dtype)
        env.task_scene_data.upright[0] = upright
        poses[:, 0, 3:] = upright
        task.reset_task(env)
        poses[1, 0, 3:] = task.quat_mul(axis_quaternion(.91, dtype=poses.dtype), upright)
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([False, True]))
        self.assertEqual(int(state.tilt_count[1, 1]), 0)  # .91 rad is below legacy 80 degrees.

    def test_all_active_physical_objects_are_checked_even_outside_ten_observation_slots(self):
        env, _, _ = PhysicalStateCapacityTests.fixture()
        env.cfg.terminations = {"object_drop": drop_term()}
        task.reset_task(env)
        poses = env.scene["object_collection"].data.object_link_pose_w
        poses[0, 31, 2] = -1.  # Inactive pool slot.
        poses[1, 31, 2] = -.01  # Active but never observed.
        poses[2, 31, 3:] = axis_quaternion(-.91, axis=1, dtype=poses.dtype)
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([False, True, True]))
        self.assertFalse((env.task_scene_data.observed_ids == 31).any())

    def test_invalid_active_pose_is_owned_by_invalid_state_and_does_not_create_drop_reason(self):
        env = self.native(2)
        env.cfg.terminations["invalid_state"] = SimpleNamespace(
            func=terminations.failure, params={"reason": "invalid_state"}, time_out=False)
        task.reset_task(env)
        env.scene["object_collection"].data.object_link_pose_w[0, 0, 2] = math.nan
        env.scene["object_collection"].data.object_link_pose_w[1, 0, 3:] = 0.
        env.common_step_counter = 1
        state = task.get_task_state(env)
        self.assertTrue(state.failure.all())
        self.assertTrue(terminations.failure(env, reason="invalid_state").all())
        self.assertFalse(terminations.failure(env, reason="object_drop").any())

    def test_partial_reset_clears_selected_drop_flags_and_cached_reads_do_not_recompute(self):
        env = self.native(3)
        task.reset_task(env)
        poses = env.scene["object_collection"].data.object_link_pose_w
        poses[0, 0, 2] = 1.
        poses[1, 0, 3:] = axis_quaternion(.91, dtype=poses.dtype)
        env.common_step_counter = 1
        with patch.object(task, "object_drop_condition", wraps=terminations.object_drop_condition) as predicate:
            state = task.get_task_state(env)
            torch.testing.assert_close(state.failure, torch.tensor([True, True, False]))
            self.assertIs(task.get_task_state(env), state)
            terminations.failure(env, reason="object_drop")
            self.assertEqual(predicate.call_count, 1)
        index = terminations.FAILURE_REASON_NAMES.index("object_drop")
        poses[0, 0, 2] = 1.05
        task.reset_task(env, torch.tensor([0]))
        torch.testing.assert_close(state.failure_reasons[:, index], torch.tensor([False, True, False]))
        env.common_step_counter += 1
        torch.testing.assert_close(task.get_task_state(env).failure, torch.tensor([False, True, False]))

    def test_state_advance_keeps_standalone_callers_compatible_and_new_failure_masks_success(self):
        state, batch, reset = fixture(2)
        state.failure_mask[:] = False
        state.failure_mask[terminations.FAILURE_REASON_NAMES.index("object_drop")] = True
        state.advance(batch, 1, object_drop=torch.tensor([True, False]))
        torch.testing.assert_close(state.failure, torch.tensor([True, False]))
        self.assertFalse(state.success[0])
        state.reset(torch.arange(2), batch, **reset, step=1)
        state.advance(batch, 2)  # Legacy pure-batch callers provide no world-origin height.
        self.assertFalse(state.failure.any())


if __name__ == "__main__":
    unittest.main()
