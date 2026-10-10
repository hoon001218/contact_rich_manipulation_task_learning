"""CPU checks for information boundaries, masks, and the trainable policy."""

from dataclasses import asdict
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.config.ur5e.agents.model import ContactRearrangementPolicy
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg


def observation(cfg: ModelCfg, batch: int = 3) -> torch.Tensor:
    geometry = torch.randn(batch, cfg.max_objects, 10)
    geometry[..., 3:7] = torch.nn.functional.normalize(geometry[..., 3:7], dim=-1)
    geometry[..., 7:] = geometry[..., 7:].abs() * 0.1 + 0.02
    valid = torch.ones(batch, cfg.max_objects)
    valid[:, -1] = 0
    roles = torch.zeros(batch, cfg.max_objects, 2)
    roles[:, 0, 0], roles[:, 1, 1] = 1, 1
    history = torch.randn(batch, cfg.history_steps, cfg.step_features)
    history_valid = torch.ones(batch, cfg.history_steps)
    history_valid[:, -2:] = 0
    return torch.cat((geometry.flatten(1), valid, roles.flatten(1), history.flatten(1), history_valid), dim=-1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(81)
        self.cfg = ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
        self.model = ContactRearrangementPolicy(self.cfg)
        self.observation = observation(self.cfg)

    def test_default_observation_layout(self):
        self.assertEqual(ModelCfg().observation_dim, 1960)
        self.assertEqual(asdict(ModelCfg())["history_steps"], 16)

    def test_object_slot_permutation_preserves_policy_and_value(self):
        perm = torch.tensor([4, 2, 0, 3, 1])
        geometry, valid, roles, history, history_valid = self.model._split(self.observation)
        reordered = torch.cat((geometry[:, perm].flatten(1), valid[:, perm], roles[:, perm].flatten(1),
                               history.flatten(1), history_valid), dim=-1)
        expected = self.model(self.observation)
        actual = self.model(reordered)
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_masked_object_and_history_nan_padding_is_ignored(self):
        geometry, valid, roles, history, history_valid = self.model._split(self.observation)
        geometry = torch.where(valid[..., None], geometry, torch.full_like(geometry, float("nan")))
        roles = torch.where(valid[..., None], roles, torch.full_like(roles, float("nan")))
        history = torch.where(history_valid[..., None], history, torch.full_like(history, float("nan")))
        poisoned = torch.cat((geometry.flatten(1), valid, roles.flatten(1), history.flatten(1), history_valid), dim=-1)
        for first, second in zip(self.model(self.observation), self.model(poisoned)):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(second).all())

    def test_all_padding_is_finite_without_inventing_observed_objects(self):
        geometry, valid, roles, history, history_valid = self.model._split(self.observation)
        geometry[:] = float("nan")
        roles[:] = float("nan")
        history[:] = float("nan")
        valid[:] = False
        history_valid[:] = False
        padded = torch.cat((geometry.flatten(1), valid, roles.flatten(1), history.flatten(1), history_valid), dim=-1)
        actions, log_probs, values = self.model.act(padded)
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(values).all())
        scene = self.model.scene_encoder(geometry, valid, roles)
        torch.testing.assert_close(scene, torch.zeros_like(scene))

    def test_history_is_ordered_and_complete_window_reaches_encoder(self):
        _, _, _, history, valid = self.model._split(self.observation)
        valid[:] = True
        baseline = self.model.history_encoder(history, valid)
        reversed_history = self.model.history_encoder(history.flip(1), valid)
        self.assertGreater(float((baseline - reversed_history).abs().max()), 1e-4)
        changed = history.clone()
        changed[:, -1] += 10.0
        oldest_changed = self.model.history_encoder(changed, valid)
        # The newest representation itself must receive the oldest valid step,
        # not merely an order-independent pooling contribution.
        self.assertGreater(float((baseline[:, :16] - oldest_changed[:, :16]).abs().max()), 1e-5)

    def test_action_bounds_and_tanh_log_probability_agree(self):
        actions, log_probs, values = self.model.act(self.observation)
        evaluated, entropy, evaluated_values = self.model.evaluate_actions(self.observation, actions)
        self.assertTrue((actions.abs() <= 1).all())
        torch.testing.assert_close(log_probs, evaluated, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(values, evaluated_values)
        self.assertTrue(torch.isfinite(entropy).all())
        first = self.model.act(self.observation, deterministic=True)
        second = self.model.act(self.observation, deterministic=True)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        near_limits = torch.full_like(actions, 1 - 1e-6)
        near_limits[:, ::2] *= -1
        self.assertTrue(torch.isfinite(self.model.evaluate_actions(self.observation, near_limits)[0]).all())

    def test_both_encoders_receive_policy_and_value_gradients(self):
        actions, _, _ = self.model.act(self.observation)
        logs, entropy, values = self.model.evaluate_actions(self.observation, actions.detach())
        (logs.mean() + 0.01 * entropy.mean() + values.square().mean()).backward()
        for module in (self.model.scene_encoder, self.model.history_encoder, self.model.actor, self.model.critic):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_saturated_actions_preserve_probability_ratios_and_finite_gradients(self):
        with torch.no_grad():
            self.model.actor.bias[::2] = 20.0
            self.model.actor.bias[1::2] = -20.0
        actions, logs, _ = self.model.act(self.observation)
        evaluated, entropy, values = self.model.evaluate_actions(self.observation, actions)
        self.assertTrue((actions.abs() < 1.0).all())
        torch.testing.assert_close(logs, evaluated, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(entropy).all())
        (evaluated.mean() + entropy.mean() + values.mean()).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_arbitrary_leading_batch_dimensions(self):
        repeated = self.observation[None].expand(2, -1, -1)
        actions, logs, values = self.model.act(repeated)
        self.assertEqual(actions.shape, (2, 3, 18))
        self.assertEqual(logs.shape, (2, 3))
        self.assertEqual(values.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
