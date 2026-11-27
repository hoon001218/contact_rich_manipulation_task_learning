"""Preserved clearance split helpers and positive-only active configuration.

Test source only: not executed while the user's training was active.
"""

import ast
import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_mdp import fixture, native_reward_compute, rewards, task
from contact_manipulation_symmetric import mdp


class ClearanceSplitTests(unittest.TestCase):
    @staticmethod
    def state():
        current = torch.tensor([.3, .1, .2, .4, math.nan, .5, .7, .8, -.3, .1],
                               dtype=torch.float64)
        previous = current.new_tensor([.2, .2, .2, .1, .2, math.nan, .1, .2, -.5, .4])
        overlap = torch.ones_like(current)
        overlap[6], overlap[7] = math.nan, math.inf
        state = SimpleNamespace(clearance_quality=current, previous_clearance_quality=previous,
                                blocker_overlap=overlap,
                                target_protected=torch.zeros(len(current), dtype=torch.bool),
                                failure=torch.ones(len(current), dtype=torch.bool))
        contacts = current.new_tensor([2., 17., 3., 0., 1., 1., 1., 1., 3., 0.])
        return state, contacts

    def test_improvement_uses_contacts_but_regression_pays_at_zero_contact_and_invalid_pairs_pay_zero(self):
        state, contacts = self.state()
        env = SimpleNamespace(step_dt=.02)
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            legacy = rewards.clearance_contact_progress(env)
            improvement = rewards.clearance_contact_improvement(env)
            regression = rewards.clearance_regression(env)
        torch.testing.assert_close(legacy, contacts.new_tensor([.2, -1.7, 0., 0., 0., 0., 0., 0., .6, 0.]))
        torch.testing.assert_close(improvement, contacts.new_tensor([.2, 0., 0., 0., 0., 0., 0., 0., .6, 0.]))
        torch.testing.assert_close(regression, contacts.new_tensor([0., .1, 0., 0., 0., 0., 0., 0., 0., .3]))
        self.assertTrue(torch.isfinite(improvement).all())
        self.assertTrue(torch.isfinite(regression).all())
        self.assertEqual(int((improvement * regression).count_nonzero()), 0)

    def test_one_pad_equal_weights_match_old_signed_term_without_changing_legacy_semantics(self):
        state, contacts = self.state()
        contacts.fill_(1.)
        env = SimpleNamespace()
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            legacy = rewards.clearance_contact_progress(env)
            split = 30. * rewards.clearance_contact_improvement(env) - 30. * rewards.clearance_regression(env)
        torch.testing.assert_close(split, 30. * legacy)

    def test_regression_does_not_call_contact_sensor_and_is_invariant_to_pad_count(self):
        state, contacts = self.state()
        expected = contacts.new_tensor([0., .1, 0., 0., 0., 0., 0., 0., 0., .3])
        with patch.object(task, "get_task_state", return_value=state):
            for count in (0., 1., 3., 17.):
                contacts.fill_(count)
                with self.subTest(pads=count), \
                     patch.object(rewards, "_current_blocker_contact_count",
                                  side_effect=AssertionError("Regression must not fetch contact sensors")):
                    torch.testing.assert_close(rewards.clearance_regression(SimpleNamespace()), expected)

    def test_independent_weights_change_only_the_matching_direction_and_never_double_pay(self):
        state, contacts = self.state()
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            positive = rewards.clearance_contact_improvement(SimpleNamespace())
            negative = rewards.clearance_regression(SimpleNamespace())
        baseline = 30. * positive - 30. * negative
        stronger_regression = 30. * positive - 60. * negative
        weaker_improvement = 15. * positive - 30. * negative
        torch.testing.assert_close(stronger_regression - baseline, -30. * negative)
        torch.testing.assert_close(weaker_improvement - baseline, -15. * positive)
        self.assertEqual(float(stronger_regression[0]), float(baseline[0]))
        self.assertEqual(float(weaker_improvement[1]), float(baseline[1]))
        self.assertEqual(int((positive * negative).count_nonzero()), 0)

    def test_repeat_reads_in_either_order_do_not_mutate_history_or_read_tensors_on_host(self):
        state, contacts = self.state()
        original = {name: value.clone() for name, value in vars(state).items()}
        env = SimpleNamespace(step_dt=.02)
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar read")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("host condition")), \
             patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")), \
             patch.object(torch.Tensor, "numpy", side_effect=AssertionError("host transfer")):
            first_positive = rewards.clearance_contact_improvement(env)
            first_negative = rewards.clearance_regression(env)
            second_negative = rewards.clearance_regression(env)
            second_positive = rewards.clearance_contact_improvement(env)
        torch.testing.assert_close(first_positive, second_positive)
        torch.testing.assert_close(first_negative, second_negative)
        for name, before in original.items():
            torch.testing.assert_close(getattr(state, name), before, equal_nan=True)

    def test_native_reward_manager_applies_dt_once_and_logs_two_signed_contributions(self):
        state, contacts = self.state()
        compute = native_reward_compute()
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            expected_positive = contacts.new_tensor([.2, 0., 0., 0., 0., 0., 0., 0., .6, 0.])
            expected_negative = contacts.new_tensor([0., .1, 0., 0., 0., 0., 0., 0., 0., .3])
            for dt in (.01, .02, .1):
                with self.subTest(dt=dt):
                    manager = SimpleNamespace(_env=SimpleNamespace(step_dt=dt),
                        _term_names=["clearance", "clearance_regression"],
                        _term_cfgs=[SimpleNamespace(func=rewards.clearance_contact_improvement,
                                                   weight=30., params={}),
                                    SimpleNamespace(func=rewards.clearance_regression,
                                                    weight=-30., params={})],
                        _reward_buf=torch.zeros_like(contacts), _step_reward=contacts.new_zeros(len(contacts), 2),
                        _episode_sums={"clearance": torch.zeros_like(contacts),
                                       "clearance_regression": torch.zeros_like(contacts)})
                    actual = compute(manager, dt)
                    torch.testing.assert_close(actual, 30. * (expected_positive - expected_negative) * dt)
                    torch.testing.assert_close(manager._episode_sums["clearance"], expected_positive * 30. * dt)
                    torch.testing.assert_close(manager._episode_sums["clearance_regression"],
                                               expected_negative * -30. * dt)
                    self.assertTrue((manager._episode_sums["clearance"] >= 0).all())
                    self.assertTrue((manager._episode_sums["clearance_regression"] <= 0).all())

    def test_only_positive_term_forwards_contact_threshold(self):
        state = SimpleNamespace(clearance_quality=torch.tensor([.2, .1]),
            previous_clearance_quality=torch.tensor([.1, .2]), blocker_overlap=torch.ones(2))
        env, contacts = SimpleNamespace(), torch.tensor([2., 3.])
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts) as count:
            positive = rewards.clearance_contact_improvement(env, threshold=.7)
            self.assertEqual(count.call_args.args, (env, .7))
            negative = rewards.clearance_regression(env)
            self.assertEqual(count.call_count, 1)
        torch.testing.assert_close(positive, torch.tensor([.2, 0.]))
        torch.testing.assert_close(negative, torch.tensor([0., .1]))

    @staticmethod
    def initialized_state(n=3):
        state, batch, reset = fixture(n)
        offset = .2 - batch.center[:, 1, 0]
        batch.center[:, 1, 0] += offset
        batch.corners[:, 1, :, 0] += offset[:, None]
        state.reset(torch.arange(n), batch, **reset, step=0)
        return state, batch, reset

    def test_partial_reset_rebases_only_reset_row_and_keeps_both_terms_idempotent(self):
        state, batch, reset = self.initialized_state()
        contacts = batch.center.new_full((3,), 2.)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            batch.center[:, 1, 0] += .01
            batch.corners[:, 1, :, 0] += .01
            state.advance(batch, 1)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_full((3,), .4))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_zeros(3))
            batch.center[:, 1, 0] -= .01
            batch.corners[:, 1, :, 0] -= .01
            state.advance(batch, 2)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(3))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_full((3,), .2))
            ids = torch.tensor([1])
            selected = task.TransitionBatch(**{field.name: getattr(batch, field.name)[ids].clone()
                                               for field in dataclasses.fields(batch)})
            state.reset(ids, selected, **{name: value[ids] for name, value in reset.items()},
                        step=2, batch_is_selected=True)
            expected = contacts.new_tensor([.2, 0., .2])
            for _ in range(2):
                torch.testing.assert_close(rewards.clearance_regression(env), expected)
                torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(3))
            state.advance(batch, 3)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(3))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_zeros(3))

    def test_no_contact_regression_is_penalized_and_recontact_does_not_repay_old_changes(self):
        state, batch, _ = self.initialized_state(1)
        contacts = batch.center.new_zeros(1)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, "get_task_state", return_value=state), \
             patch.object(rewards, "_current_blocker_contact_count", return_value=contacts):
            batch.center[:, 1, 0] += .01
            batch.corners[:, 1, :, 0] += .01
            state.advance(batch, 1)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(1))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_zeros(1))
            batch.center[:, 1, 0] -= .01
            batch.corners[:, 1, :, 0] -= .01
            state.advance(batch, 2)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(1))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_full((1,), .2))
            contacts.fill_(2.)
            state.advance(batch, 3)
            torch.testing.assert_close(rewards.clearance_contact_improvement(env), contacts.new_zeros(1))
            torch.testing.assert_close(rewards.clearance_regression(env), contacts.new_zeros(1))

    def test_split_helpers_stay_exported_but_only_improvement_is_configured(self):
        for name in ("clearance_contact_progress", "clearance_contact_improvement", "clearance_regression"):
            self.assertIn(name, rewards.__all__)
            self.assertIs(getattr(mdp, name), getattr(rewards, name))
        source = Path(__file__).resolve().parents[1] / "contact_manipulation_symmetric/shelf_manipulation_env_cfg.py"
        cfg = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
        declarations = {node.targets[0].id: {arg.arg: arg.value for arg in node.value.keywords}
                        for node in cfg.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)}
        self.assertEqual(ast.unparse(declarations["clearance"]["func"]), "mdp.clearance_contact_improvement")
        self.assertEqual(ast.literal_eval(declarations["clearance"]["weight"]), 30.)
        self.assertEqual(ast.literal_eval(declarations["clearance"]["params"]), {"contact_exponent": 1.2})
        self.assertNotIn("clearance_regression", declarations)


if __name__ == "__main__":
    unittest.main()
