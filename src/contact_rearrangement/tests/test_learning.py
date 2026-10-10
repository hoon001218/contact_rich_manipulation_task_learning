"""PPO episode-boundary and checkpoint checks without launching Isaac Sim."""

from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.config.ur5e.agents.ppo import PPOTrainer, generalized_advantage_estimate
from contact_rearrangement.config.ur5e.agents.model import ContactRearrangementPolicy
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ContactRearrangementPPORunnerCfg, ModelCfg, PPOCfg
from test_model import observation


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_runner_owns_independent_policy_and_algorithm_defaults(self):
        first, second = ContactRearrangementPPORunnerCfg(), ContactRearrangementPPORunnerCfg()
        self.assertEqual(first.policy.observation_dim, 1960)
        self.assertEqual(first.algorithm.rollout_steps, 32)
        self.assertEqual(first.max_iterations, 1000)
        self.assertEqual(first.save_interval, 50)
        self.assertEqual(first.experiment_name, "contact_rearrangement")
        first.policy.node_dim = 16
        first.algorithm.learning_rate = 1e-4
        self.assertEqual(second.policy.node_dim, 64)
        self.assertEqual(second.algorithm.learning_rate, 3e-4)

    def test_timeout_uses_final_state_value_and_stops_episode_carry(self):
        rewards = torch.tensor([[1.0], [100.0]])
        values = torch.tensor([[2.0], [0.0]])
        final_values = torch.tensor([[4.0], [0.0]])
        terminated = torch.tensor([[False], [True]])
        truncated = torch.tensor([[True], [False]])
        advantages, returns = generalized_advantage_estimate(
            rewards, values, final_values, terminated, truncated, gamma=0.5, gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([[1.0], [100.0]]))
        torch.testing.assert_close(returns, torch.tensor([[3.0], [100.0]]))

    def test_true_terminal_state_never_bootstraps_even_with_timeout(self):
        advantages, returns = generalized_advantage_estimate(
            torch.tensor([[2.0]]), torch.tensor([[1.0]]), torch.tensor([[1000.0]]),
            torch.tensor([[True]]), torch.tensor([[True]]),
        )
        torch.testing.assert_close(advantages, torch.tensor([[1.0]]))
        torch.testing.assert_close(returns, torch.tensor([[2.0]]))

    def test_gae_continues_only_within_an_episode(self):
        advantages, _ = generalized_advantage_estimate(
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]), torch.zeros(2, 2), torch.zeros(2, 2),
            torch.tensor([[False, True], [True, True]]), torch.zeros(2, 2, dtype=torch.bool),
            gamma=0.5, gae_lambda=1.0,
        )
        torch.testing.assert_close(advantages, torch.tensor([[2.0, 1.0], [2.0, 2.0]]))

    def test_update_trains_joint_model_and_roundtrips_optimizer(self):
        torch.manual_seed(192)
        cfg = ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
        model = ContactRearrangementPolicy(cfg)
        trainer = PPOTrainer(model, PPOCfg(rollout_steps=3, epochs=2, minibatches=2))
        # Real collection often uses inference_mode.  Updates must safely copy
        # those tensors before saving inputs for autograd.
        with torch.inference_mode():
            observations = observation(cfg, batch=6).reshape(3, 2, -1)
            actions, logs, values = model.act(observations)
            next_values = model.value(observations) + 0.1
            rewards = torch.tensor([[0.1, 0.2], [-0.1, 0.3], [1.0, 0.1]])
            terminated = torch.tensor([[False, False], [False, True], [True, False]])
            truncated = torch.tensor([[False, False], [False, False], [False, True]])
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        metrics = trainer.update(observations, actions, logs, rewards, values, next_values, terminated, truncated)
        self.assertEqual(metrics["minibatch_updates"], 4)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        for prefix in ("scene_encoder.", "history_encoder.", "actor.", "critic."):
            self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()
                                if name.startswith(prefix)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            trainer.save(path, experiment="cpu_smoke")
            loaded = PPOTrainer.load(path)
            self.assertEqual(loaded.update_count, 1)
            self.assertEqual(loaded.cfg, trainer.cfg)
            self.assertEqual(loaded.model.cfg, model.cfg)
            with torch.no_grad():
                expected = model.act(observations, deterministic=True)
                actual = loaded.model.act(observations, deterministic=True)
            for first, second in zip(expected, actual):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
            first_states, second_states = trainer.optimizer.state_dict(), loaded.optimizer.state_dict()
            self.assertEqual(first_states["param_groups"], second_states["param_groups"])
            for parameter_id, state in first_states["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, second_states["state"][parameter_id][key])

    def test_invalid_rollout_is_rejected_before_optimizer_mutation(self):
        cfg = ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
        model = ContactRearrangementPolicy(cfg)
        trainer = PPOTrainer(model)
        observations = observation(cfg, 2).reshape(1, 2, -1)
        with torch.no_grad():
            actions, logs, values = model.act(observations)
        actions[0, 0, 0] = 1.1
        before = [parameter.clone() for parameter in model.parameters()]
        with self.assertRaisesRegex(ValueError, "actions must be in"):
            trainer.update(observations, actions, logs, torch.zeros(1, 2), values, values,
                           torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool))
        for old, current in zip(before, model.parameters()):
            torch.testing.assert_close(old, current, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
