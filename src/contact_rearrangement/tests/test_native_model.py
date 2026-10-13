"""CPU contract checks for the research model's native RSL-RL interface."""

from dataclasses import asdict
from itertools import chain
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.config.ur5e.agents.model import (
    ContactRearrangementPolicy, NativeContactActor, NativeContactCritic, build_policy_masks,
)
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg

from test_model import observation


class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)
        self.cfg = ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
        self.values = observation(self.cfg, batch=5)
        self.obs = TensorDict({"policy": self.values}, batch_size=[5])
        self.groups = {"actor": ["policy"], "critic": ["policy"]}
        self.actor = NativeContactActor(self.obs, self.groups, "actor", 18,
                                       model_cfg=asdict(self.cfg), max_forward_batch_size=2)
        self.critic = NativeContactCritic(self.obs, self.groups, "critic", 1,
                                         shared_policy=self.actor.policy, actor=self.actor,
                                         max_forward_batch_size=2)

    def expected(self):
        return self.actor.policy(self.values, masks=build_policy_masks(self.values, self.cfg))

    def test_default_architecture_and_observation_contract_are_preserved(self):
        cfg = ModelCfg()
        obs = TensorDict({"policy": observation(cfg, batch=1)}, batch_size=[1])
        actor = NativeContactActor(obs, self.groups, "actor", 18)
        self.assertEqual(actor.obs_dim, 544)
        self.assertEqual(sum(parameter.numel() for parameter in actor.parameters()), 247717)
        self.assertEqual(actor.policy.cfg.history_steps, 2)
        self.assertEqual(actor.policy.scene_encoder.node_encoder[0].in_features, 10)
        self.assertEqual(actor.policy.history_encoder.project.in_channels, 72)
        self.assertIs(self.actor.get_policy_observation(self.obs), self.values)
        with self.assertRaisesRegex(ValueError, "single public"):
            NativeContactActor(self.obs, {"actor": ["policy", "privileged"]}, "actor", 18)
        with self.assertRaisesRegex(ValueError, "physical values"):
            NativeContactActor(obs, self.groups, "actor", 18, obs_normalization=True)

    def test_native_optimizer_registers_the_complete_shared_policy_once(self):
        self.assertIs(self.critic.policy, self.actor.policy)
        self.assertEqual(list(self.critic.parameters()), [])
        self.assertEqual(self.critic.state_dict(), {})
        parameters = list(chain(self.actor.parameters(), self.critic.parameters()))
        self.assertEqual(len(parameters), len({id(parameter) for parameter in parameters}))
        self.assertEqual({id(parameter) for parameter in parameters},
                         {id(parameter) for parameter in self.actor.policy.parameters()})
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        self.actor(self.obs, stochastic_output=True)
        actions = torch.zeros(5, 18)
        values = self.critic(self.obs)
        loss = -self.actor.get_output_log_prob(actions).mean() + values.square().mean() - 0.01 * self.actor.output_entropy.mean()
        loss.backward()
        for module in (self.actor.policy.scene_encoder, self.actor.policy.history_encoder,
                       self.actor.policy.actor, self.actor.policy.critic):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        original = self.actor.policy.critic.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(original, self.actor.policy.critic.weight))

    def test_native_outputs_match_core_across_forward_chunks(self):
        mean, expected_value = self.expected()
        actions = self.actor(self.obs)
        torch.testing.assert_close(actions, self.actor.policy._bounded_action(mean), atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.critic(self.obs).squeeze(-1), expected_value, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.actor.output_mean, mean, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.actor.output_std,
                                   self.actor.policy.log_std.clamp(-5, 2).exp().expand_as(mean))
        self.assertEqual(self.actor.get_hidden_state(), None)
        self.assertEqual(self.critic.get_hidden_state(), None)
        self.assertFalse(self.actor.is_recurrent)

    def test_native_log_probability_and_fixed_entropy_match_core(self):
        noise = torch.randn(5, 18)
        actions = self.actor(self.obs, stochastic_output=True, entropy_noise=noise)
        self.assertTrue((actions.abs() < 1).all())
        expected_log, expected_entropy, expected_value = self.actor.policy.evaluate_actions(
            self.values, actions, masks=build_policy_masks(self.values, self.cfg), entropy_noise=noise,
        )
        torch.testing.assert_close(self.actor.get_output_log_prob(actions), expected_log, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(self.actor.output_entropy, expected_entropy, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(self.critic(self.obs).squeeze(-1), expected_value, atol=1e-6, rtol=1e-5)
        self.assertIs(self.actor.output_entropy, self.actor.output_entropy)
        self.assertTrue(torch.isfinite(self.actor.get_kl_divergence(
            self.actor.output_distribution_params,
            (self.actor.output_mean + 0.1, self.actor.output_std * 1.1),
        )).all())

    def test_saturation_keeps_native_log_probabilities_and_gradients_finite(self):
        with torch.no_grad():
            self.actor.policy.actor.bias[::2] = 20
            self.actor.policy.actor.bias[1::2] = -20
        actions = self.actor(self.obs, stochastic_output=True)
        log_probability = self.actor.get_output_log_prob(actions)
        entropy = self.actor.output_entropy
        values = self.critic(self.obs)
        self.assertTrue((actions.abs() < 1).all())
        self.assertTrue(torch.isfinite(log_probability).all())
        self.assertTrue(torch.isfinite(entropy).all())
        (log_probability.mean() + entropy.mean() + values.mean()).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.actor.parameters()))

    def test_value_cache_is_consumed_only_once(self):
        calls = []
        hook = self.actor.policy.scene_encoder.register_forward_pre_hook(lambda _module, _inputs: calls.append(True))
        try:
            self.actor(self.obs)
            self.assertEqual(len(calls), 3)
            first = self.critic(self.obs)
            self.assertEqual(len(calls), 3)
            second = self.critic(self.obs)
            self.assertEqual(len(calls), 6)
            torch.testing.assert_close(first, second)
        finally:
            hook.remove()

    def test_input_mutation_and_parameter_updates_invalidate_value_cache(self):
        self.actor(self.obs)
        self.values[:, 0] += 0.1
        _, expected = self.expected()
        torch.testing.assert_close(self.critic(self.obs).squeeze(-1), expected, atol=1e-6, rtol=1e-5)
        self.actor(self.obs)
        with torch.no_grad():
            self.actor.policy.critic.bias += 1
        _, expected = self.expected()
        torch.testing.assert_close(self.critic(self.obs).squeeze(-1), expected, atol=1e-6, rtol=1e-5)

    def test_new_observation_container_does_not_reuse_cached_values(self):
        self.actor(self.obs)
        self.assertIsNone(self.actor.consume_cached_value(self.obs.clone(recurse=False)))
        self.assertIsNone(self.actor.consume_cached_value(self.obs))

    def test_inference_observations_are_evaluated_without_unsafe_cache_reuse(self):
        with torch.inference_mode():
            obs = TensorDict({"policy": self.values.clone()}, batch_size=[5])
            self.actor(obs, stochastic_output=True)
            self.assertIsNone(self.actor._cached_values)
            first = self.critic(obs)
            obs["policy"][:, 0] += 1
            second = self.critic(obs)
            self.assertFalse(torch.equal(first, second))

    def test_native_actor_state_dict_preserves_every_shared_parameter(self):
        checkpoint = self.actor.state_dict()
        restored = NativeContactActor(self.obs, self.groups, "actor", 18, model_cfg=asdict(self.cfg))
        restored.load_state_dict(checkpoint)
        for name, expected in self.actor.policy.state_dict().items():
            torch.testing.assert_close(restored.policy.state_dict()[name], expected, rtol=0, atol=0)
        with self.assertRaises(NotImplementedError):
            restored.as_jit()
        with self.assertRaises(NotImplementedError):
            restored.as_onnx()


if __name__ == "__main__":
    unittest.main()
