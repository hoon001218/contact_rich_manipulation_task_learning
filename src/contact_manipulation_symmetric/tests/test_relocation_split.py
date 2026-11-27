"""Preserved relocation split helpers and positive-only active configuration.

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


class RelocationSplitTests(unittest.TestCase):
    @staticmethod
    def state(previous_x, current_x):
        previous = torch.tensor(previous_x, dtype=torch.float64)
        current = torch.tensor(current_x, dtype=torch.float64)
        count = current.numel()
        polygon = current.new_tensor([[0., -.1], [.2, -.1], [.2, .1], [0., .1]])
        return SimpleNamespace(
            blocker_center=torch.stack((current, torch.zeros_like(current), torch.ones_like(current)), -1),
            previous_blocker_center=torch.stack(
                (previous, torch.zeros_like(previous), torch.ones_like(previous)), -1),
            relocation_regions=polygon[None, None].expand(count, 1, -1, -1),
            relocation_vertex_counts=torch.full((count, 1), 4, dtype=torch.long),
            relocation_valid=torch.ones((count, 1), dtype=torch.bool),
        )

    @staticmethod
    def evaluate(state, contacts, dt=.02, **parameters):
        env = SimpleNamespace(step_dt=dt)
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count', return_value=contacts):
            return (rewards.blocker_relocation_distance_improvement(env, **parameters),
                    rewards.blocker_relocation_distance_regression(env, **parameters))

    def test_improvement_scales_with_current_pads_but_regression_survives_zero_contact(self):
        state = self.state([.4, .36, .32, .4, .36, .4], [.36, .4, .4, .36, .4, .4])
        contacts = state.blocker_center.new_tensor([3., 17., 0., 0., 1., 17.])
        positive, negative = self.evaluate(state, contacts)
        torch.testing.assert_close(positive, contacts.new_tensor([1.5, 0., 0., 0., 0., 0.]))
        torch.testing.assert_close(negative, contacts.new_tensor([0., .5, 1., 0., .5, 0.]))
        self.assertEqual(int((positive * negative).count_nonzero()), 0)

    def test_signed_fraction_clips_before_multiplying_pads_and_preserves_tiny_changes(self):
        state = self.state([.4, .32, .5, .3, .4, .4 - 1.e-8],
                           [.32, .4, .3, .5, .4 - 1.e-8, .4])
        contacts = state.blocker_center.new_tensor([2., 2., 17., 17., 3., 17.])
        positive, negative = self.evaluate(state, contacts)
        torch.testing.assert_close(positive, contacts.new_tensor([2., 0., 17., 0., 3.75e-7, 0.]),
                                   atol=1.e-12, rtol=1.e-8)
        torch.testing.assert_close(negative, contacts.new_tensor([0., 1., 0., 1., 0., 1.25e-7]),
                                   atol=1.e-12, rtol=1.e-8)

    def test_goal_entry_exit_and_motion_within_union_have_expected_signs(self):
        state = self.state([.25, .15, .05, .4], [.15, .25, .15, .4])
        contacts = state.blocker_center.new_full((4,), 2.)
        positive, negative = self.evaluate(state, contacts)
        torch.testing.assert_close(positive, contacts.new_tensor([1.25, 0., 0., 0.]))
        torch.testing.assert_close(negative, contacts.new_tensor([0., .625, 0., 0.]))
        corridor = state.blocker_center.new_tensor([[.2, -.1], [.5, -.1], [.5, .1], [.2, .1]])
        state = self.state([.6, .6, .45], [.55, .45, .55])
        state.relocation_regions = torch.cat((state.relocation_regions,
            corridor[None, None].expand(3, 1, -1, -1)), 1)
        state.relocation_vertex_counts = torch.full((3, 2), 4, dtype=torch.long)
        state.relocation_valid = torch.ones((3, 2), dtype=torch.bool)
        positive, negative = self.evaluate(state, contacts[:3])
        torch.testing.assert_close(positive, contacts.new_tensor([1.25, 2., 0.]))
        torch.testing.assert_close(negative, contacts.new_tensor([0., 0., .625]))

    def test_regression_never_reads_contacts_and_does_not_depend_on_pad_count(self):
        state = self.state([.36, .32], [.4, .4])
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count',
                          side_effect=AssertionError('Regression must not fetch contact sensors')):
            torch.testing.assert_close(rewards.blocker_relocation_distance_regression(
                SimpleNamespace(step_dt=.02)), state.blocker_center.new_tensor([.5, 1.]))
        for count in (0., 1., 3., 17.):
            with self.subTest(pads=count):
                _, negative = self.evaluate(state, state.blocker_center.new_full((2,), count))
                torch.testing.assert_close(negative, state.blocker_center.new_tensor([.5, 1.]))

    def test_equal_weights_match_legacy_for_one_pad_without_changing_legacy_semantics(self):
        state = self.state([.4, .36, .32], [.36, .4, .4])
        contacts = state.blocker_center.new_ones(3)
        positive, negative = self.evaluate(state, contacts)
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count', return_value=contacts):
            legacy = rewards.blocker_relocation_distance_progress(SimpleNamespace(step_dt=.02))
        torch.testing.assert_close(30. * positive - 30. * negative, 30. * legacy)

    def test_invalid_positions_regions_and_missing_goals_pay_finite_zero(self):
        state = self.state([math.nan, .4, .4, .4, .4, .4], [.3, math.inf, .3, .3, .3, .3])
        state.blocker_center[2, 2] = math.nan
        state.previous_blocker_center[3, 2] = math.inf
        state.relocation_valid = state.relocation_valid.clone()
        state.relocation_valid[4] = False
        state.relocation_regions = state.relocation_regions.clone()
        state.relocation_regions[5, 0, 1, 0] = math.nan
        contacts = state.blocker_center.new_full((6,), 17.)
        for missing in (False, True):
            if missing:
                state.relocation_regions = state.relocation_regions[:, :0]
                state.relocation_vertex_counts = state.relocation_vertex_counts[:, :0]
                state.relocation_valid = state.relocation_valid[:, :0]
            with self.subTest(missing=missing):
                for value in self.evaluate(state, contacts):
                    torch.testing.assert_close(value, contacts.new_zeros(6))

    def test_reference_distance_uses_actual_dt_and_speed_for_both_signs(self):
        state = self.state([.4, .38], [.38, .4])
        contacts = state.blocker_center.new_full((2,), 2.)
        for dt, speed, expected in ((.02, 4., .25), (.01, 4., .5), (.02, 2., .5)):
            with self.subTest(dt=dt, speed=speed):
                positive, negative = self.evaluate(state, contacts, dt=dt, reference_speed=speed)
                torch.testing.assert_close(positive, contacts.new_tensor([2. * expected, 0.]))
                torch.testing.assert_close(negative, contacts.new_tensor([0., expected]))
        for function in (rewards.blocker_relocation_distance_improvement,
                         rewards.blocker_relocation_distance_regression):
            for invalid in (0., -1., math.nan, math.inf, -math.inf):
                with self.subTest(function=function.__name__, speed=invalid), \
                     self.assertRaisesRegex(ValueError, 'reference speed'):
                    function(SimpleNamespace(step_dt=.02), reference_speed=invalid)
                with self.subTest(function=function.__name__, dt=invalid), \
                     self.assertRaisesRegex(ValueError, 'step_dt'):
                    function(SimpleNamespace(step_dt=invalid))
            with self.assertRaisesRegex(ValueError, 'reference distance'):
                function(SimpleNamespace(step_dt=1.e300), reference_speed=1.e300)

    def test_native_manager_applies_independent_weights_and_dt_once(self):
        state = self.state([.4, .4, .36, .32], [.36, .32, .4, .4])
        contacts = state.blocker_center.new_tensor([2., 3., 17., 0.])
        for dt, expected_improvement, expected_regression in (
            (.01, [2., 3., 0., 0.], [0., 0., 1., 1.]),
            (.02, [1., 3., 0., 0.], [0., 0., .5, 1.]),
        ):
            for positive_weight, negative_weight in ((30., -30.), (15., -60.)):
                manager = SimpleNamespace(_env=SimpleNamespace(step_dt=dt),
                    _term_names=['blocker_relocation', 'blocker_relocation_regression'],
                    _term_cfgs=[SimpleNamespace(func=rewards.blocker_relocation_distance_improvement,
                        weight=positive_weight, params={'reference_speed': 4., 'threshold': .01}),
                        SimpleNamespace(func=rewards.blocker_relocation_distance_regression,
                            weight=negative_weight, params={'reference_speed': 4.})],
                    _reward_buf=torch.zeros_like(contacts), _step_reward=contacts.new_zeros(4, 2),
                    _episode_sums={name: torch.zeros_like(contacts)
                                   for name in ('blocker_relocation', 'blocker_relocation_regression')})
                with self.subTest(dt=dt, positive=positive_weight, negative=negative_weight), \
                     patch.object(task, 'get_task_state', return_value=state), \
                     patch.object(rewards, '_current_blocker_contact_count', return_value=contacts):
                    actual = native_reward_compute()(manager, dt)
                expected_positive = contacts.new_tensor(expected_improvement) * positive_weight * dt
                expected_negative = contacts.new_tensor(expected_regression) * negative_weight * dt
                torch.testing.assert_close(actual, expected_positive + expected_negative)
                torch.testing.assert_close(manager._episode_sums['blocker_relocation'], expected_positive)
                torch.testing.assert_close(manager._episode_sums['blocker_relocation_regression'], expected_negative)

    def test_repeated_reads_are_read_only_and_never_transfer_tensors_to_host(self):
        state = self.state([.4, .3], [.36, .4])
        original = {name: value.clone() for name, value in vars(state).items()}
        contacts = state.blocker_center.new_tensor([3., 17.])
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar read')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host condition')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('host transfer')):
            first = self.evaluate(state, contacts)
            second = self.evaluate(state, contacts)
        for before, after in zip(first, second):
            torch.testing.assert_close(before, after)
        for name, value in original.items():
            torch.testing.assert_close(getattr(state, name), value)

    def test_snapshots_do_not_repay_noncontact_progress_and_partial_reset_rebases_only_its_row(self):
        state, batch, reset = fixture(3)
        state.dt = .02
        state.relocation_regions = batch.center.new_tensor(
            [[.3, -.1], [.4, -.1], [.4, .1], [.3, .1]])[None, None].expand(3, 1, -1, -1)
        state.relocation_vertex_counts = torch.full((3, 1), 4, dtype=torch.long)
        state.relocation_valid = torch.ones((3, 1), dtype=torch.bool)
        contacts = batch.center.new_zeros(3)
        batch.center[:, 1, 0] += .04
        batch.corners[:, 1, :, 0] += .04
        state.advance(batch, 1)
        for value in self.evaluate(state, contacts):
            torch.testing.assert_close(value, contacts.new_zeros(3))
        contacts.fill_(2.)
        state.advance(batch, 2)
        for value in self.evaluate(state, contacts):
            torch.testing.assert_close(value, contacts.new_zeros(3))
        contacts.zero_()
        batch.center[:, 1, 0] -= .04
        batch.corners[:, 1, :, 0] -= .04
        state.advance(batch, 3)
        positive, negative = self.evaluate(state, contacts)
        torch.testing.assert_close(positive, contacts.new_zeros(3))
        torch.testing.assert_close(negative, contacts.new_full((3,), .5))
        ids = torch.tensor([1])
        selected = task.TransitionBatch(**{field.name: getattr(batch, field.name)[ids].clone()
                                           for field in dataclasses.fields(batch)})
        state.reset(ids, selected, **{name: value[ids] for name, value in reset.items()},
                    step=3, batch_is_selected=True)
        for _ in range(2):
            positive, negative = self.evaluate(state, contacts)
            torch.testing.assert_close(positive, contacts.new_zeros(3))
            torch.testing.assert_close(negative, contacts.new_tensor([.5, 0., .5]))
        state.advance(batch, 4)
        for value in self.evaluate(state, contacts):
            torch.testing.assert_close(value, contacts.new_zeros(3))

    def test_split_helpers_stay_exported_but_only_improvement_is_configured(self):
        for name in ('blocker_relocation_distance_progress', 'blocker_relocation_distance_improvement',
                     'blocker_relocation_distance_regression'):
            self.assertIn(name, rewards.__all__)
            self.assertIs(getattr(mdp, name), getattr(rewards, name))
        source = Path(__file__).resolve().parents[1] / 'contact_manipulation_symmetric/shelf_manipulation_env_cfg.py'
        cfg = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == 'RewardsCfg')
        declarations = {node.targets[0].id: {arg.arg: arg.value for arg in node.value.keywords}
                        for node in cfg.body if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)}
        positive = declarations['blocker_relocation']
        self.assertEqual(ast.unparse(positive['func']), 'mdp.blocker_relocation_distance_improvement')
        self.assertEqual(ast.literal_eval(positive['weight']), 30.)
        self.assertEqual(ast.literal_eval(positive['params']),
                         {'reference_speed': 4., 'threshold': .01, 'contact_exponent': 1.2})
        self.assertNotIn('blocker_relocation_regression', declarations)
        state, contacts = self.state([.4], [.36]), torch.tensor([2.], dtype=torch.float64)
        env = SimpleNamespace(step_dt=.02)
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count', return_value=contacts) as count:
            rewards.blocker_relocation_distance_improvement(env, threshold=.7)
            self.assertEqual(count.call_args.args, (env, .7))
            rewards.blocker_relocation_distance_regression(env)
            self.assertEqual(count.call_count, 1)


if __name__ == '__main__':
    unittest.main()
