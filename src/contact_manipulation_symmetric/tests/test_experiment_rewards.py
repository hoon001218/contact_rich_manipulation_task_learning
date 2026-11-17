"""CPU checks for normalized relocation progress; no Kit or GPU scene required."""

import dataclasses
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

# Reuse the established Isaac Lab math bootstrap and numerical TaskState fixture.
# Importing the module does not execute its unittest suites or start Isaac Sim.
from test_mdp import fixture, native_reward_compute, rewards, task


class RelocationDistanceProgressTests(unittest.TestCase):
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
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count', return_value=contacts):
            return rewards.blocker_relocation_distance_progress(SimpleNamespace(step_dt=dt), **parameters)

    def test_80mm_reference_clips_fraction_before_multiplying_current_pad_count(self):
        state = self.state([.4] * 4, [.395, .36, .32, .2])
        contacts = state.blocker_center.new_tensor([1., 2., 3., 17.])
        actual = self.evaluate(state, contacts)
        torch.testing.assert_close(actual, contacts.new_tensor([.0625, 1., 3., 17.]))

    def test_signed_steps_are_symmetric_including_tiny_regressions(self):
        state = self.state([.4, .395, .4, .36, .4, .4 - 1.e-8],
                           [.395, .4, .36, .4, .4 - 1.e-8, .4])
        contacts = state.blocker_center.new_tensor([2., 2., 3., 3., 1., 1.])
        actual = self.evaluate(state, contacts)
        expected = contacts.new_tensor([.125, -.125, 1.5, -1.5, 1.25e-7, -1.25e-7])
        torch.testing.assert_close(actual, expected, atol=1.e-12, rtol=1.e-8)

    def test_signed_80mm_saturation_applies_before_contact_count(self):
        state = self.state([.4, .32, .5, .3], [.32, .4, .3, .5])
        contacts = state.blocker_center.new_tensor([2., 2., 17., 17.])
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([2., -2., 17., -17.]))

    def test_goal_entry_is_positive_exit_negative_and_stationary_or_noncontact_zero(self):
        state = self.state([.25, .15, .05, .4, .3, .4], [.15, .25, .15, .4, .4, .3])
        contacts = state.blocker_center.new_tensor([2., 2., 17., 17., 0., 0.])
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([1.25, -1.25, 0., 0., 0., 0.]))

    def test_minimum_distance_uses_union_including_front_corridor_and_reaches_zero_inside(self):
        state = self.state([.6, .6, .8], [.55, .45, .75])
        # Pocket: x=[0,.2], y=[-.1,.1]; front corridor: x=[.2,.5].
        # The third polygon is another independent destination farther right.
        corridor = state.blocker_center.new_tensor([[.2, -.1], [.5, -.1], [.5, .1], [.2, .1]])
        other = state.blocker_center.new_tensor([[.9, -.1], [1., -.1], [1., .1], [.9, .1]])
        state.relocation_regions = torch.cat((state.relocation_regions,
            corridor[None, None].expand(3, 1, -1, -1), other[None, None].expand(3, 1, -1, -1)), 1)
        state.relocation_vertex_counts = torch.full((3, 3), 4, dtype=torch.long)
        state.relocation_valid = torch.ones((3, 3), dtype=torch.bool)
        # 50 mm improvement, entry into corridor, then motion away from nearest goal.
        actual = self.evaluate(state, state.blocker_center.new_ones(3))
        torch.testing.assert_close(actual, actual.new_tensor([.625, 1., -.625]))

    def test_invalid_positions_regions_and_missing_goals_pay_finite_zero(self):
        state = self.state([math.nan, .4, .4, .4, .4, .4], [.3, math.inf, .3, .3, .3, .3])
        state.blocker_center[2, 2] = math.nan  # Finite XY alone is insufficient.
        state.previous_blocker_center[3, 2] = math.inf
        state.relocation_valid = state.relocation_valid.clone()
        state.relocation_valid[4] = False
        state.relocation_regions = state.relocation_regions.clone()
        state.relocation_regions[5, 0, 1, 0] = math.nan
        contacts = state.blocker_center.new_full((6,), 17.)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_zeros(6))
        state.relocation_regions = state.relocation_regions[:, :0]
        state.relocation_vertex_counts = state.relocation_vertex_counts[:, :0]
        state.relocation_valid = state.relocation_valid[:, :0]
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_zeros(6))

    def test_reference_distance_follows_control_dt_and_speed_without_extra_rate_conversion(self):
        state = self.state([.4], [.38])
        contacts = state.blocker_center.new_tensor([2.])
        torch.testing.assert_close(self.evaluate(state, contacts, dt=.02), contacts.new_tensor([.5]))
        torch.testing.assert_close(self.evaluate(state, contacts, dt=.01), contacts.new_tensor([1.]))
        torch.testing.assert_close(self.evaluate(state, contacts, dt=.02, reference_speed=2.),
                                   contacts.new_tensor([1.]))

    def test_native_reward_manager_applies_weight_and_control_dt_once(self):
        state = self.state([.4, .4, .36, .32], [.36, .32, .4, .4])
        contacts = state.blocker_center.new_tensor([2., 3., 2., 3.])
        env = SimpleNamespace(step_dt=.02)
        manager = SimpleNamespace(_env=env, _term_names=['blocker_relocation'],
            _term_cfgs=[SimpleNamespace(func=rewards.blocker_relocation_distance_progress,
                                       weight=3., params={'reference_speed': 4.})],
            _reward_buf=torch.zeros_like(contacts), _step_reward=contacts.new_zeros(4, 1),
            _episode_sums={'blocker_relocation': torch.zeros_like(contacts)})
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(rewards, '_current_blocker_contact_count', return_value=contacts):
            actual = native_reward_compute()(manager, env.step_dt)
        torch.testing.assert_close(actual, contacts.new_tensor([.06, .18, -.06, -.18]))
        torch.testing.assert_close(manager._episode_sums['blocker_relocation'], actual)

    def test_bad_scalar_reference_parameters_are_rejected_before_reading_scene(self):
        for value in (0., -1., math.nan, math.inf, -math.inf):
            with self.subTest(speed=value), self.assertRaisesRegex(ValueError, 'reference speed'):
                rewards.blocker_relocation_distance_progress(SimpleNamespace(step_dt=.02), reference_speed=value)
            with self.subTest(dt=value), self.assertRaisesRegex(ValueError, 'step_dt'):
                rewards.blocker_relocation_distance_progress(SimpleNamespace(step_dt=value))
        with self.assertRaisesRegex(ValueError, 'reference distance'):
            rewards.blocker_relocation_distance_progress(SimpleNamespace(step_dt=1.e300), reference_speed=1.e300)

    def test_tensor_path_has_no_host_scalar_reads_and_does_not_mutate_state(self):
        state = self.state([.4, .3], [.36, .4])
        original = {name: value.clone() for name, value in vars(state).items()}
        contacts = state.blocker_center.new_tensor([3., 17.])
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar read')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host condition')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')), \
             patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('host transfer')):
            first = self.evaluate(state, contacts)
            second = self.evaluate(state, contacts)
        torch.testing.assert_close(first, contacts.new_tensor([1.5, -17.]))
        torch.testing.assert_close(second, first)
        for name, value in original.items():
            torch.testing.assert_close(getattr(state, name), value)

    def test_shared_snapshots_do_not_pay_noncontact_progress_when_contact_returns(self):
        state, batch, _ = fixture(1)
        state.dt = .02
        state.relocation_regions = batch.center.new_tensor(
            [[.3, -.1], [.4, -.1], [.4, .1], [.3, .1]])[None, None]
        state.relocation_vertex_counts = torch.full((1, 1), 4, dtype=torch.long)
        state.relocation_valid = torch.ones((1, 1), dtype=torch.bool)
        batch.center[:, 1, 0] += .04
        batch.corners[:, 1, :, 0] += .04
        state.advance(batch, 1)
        contacts = batch.center.new_zeros(1)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts)
        contacts.fill_(2.)
        state.advance(batch, 2)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_zeros(1))
        batch.center[:, 1, 0] += .04
        batch.corners[:, 1, :, 0] += .04
        state.advance(batch, 3)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([1.]))
        state.advance(batch, 3)  # Multiple readers share the same completed step.
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([1.]))
        batch.center[:, 1, 0] -= .04
        batch.corners[:, 1, :, 0] -= .04
        state.advance(batch, 4)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([-1.]))
        state.advance(batch, 4)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([-1.]))

    def test_partial_reset_rebases_only_reset_rows_without_consuming_other_progress(self):
        state, batch, reset = fixture(3)
        state.dt = .02
        state.relocation_regions = batch.center.new_tensor(
            [[.3, -.1], [.4, -.1], [.4, .1], [.3, .1]])[None, None].expand(3, 1, -1, -1)
        state.relocation_vertex_counts = torch.full((3, 1), 4, dtype=torch.long)
        state.relocation_valid = torch.ones((3, 1), dtype=torch.bool)
        batch.center[:, 1, 0] += .04
        batch.corners[:, 1, :, 0] += .04
        state.advance(batch, 1)
        ids = torch.tensor([1])
        batch.center[1, 1, 0] = .25
        batch.corners[1, 1, :, 0] += .09
        selected = task.TransitionBatch(**{field.name: getattr(batch, field.name)[ids].clone()
                                           for field in dataclasses.fields(batch)})
        state.reset(ids, selected, **{name: value[ids] for name, value in reset.items()},
                    step=1, batch_is_selected=True)
        contacts = batch.center.new_full((3,), 2.)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([1., 0., 1.]))
        state.advance(batch, 2)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_zeros(3))
        batch.center[0, 1, 0] -= .04
        batch.corners[0, 1, :, 0] -= .04
        state.advance(batch, 3)
        torch.testing.assert_close(self.evaluate(state, contacts), contacts.new_tensor([-1., 0., 0.]))


if __name__ == '__main__':
    unittest.main()
