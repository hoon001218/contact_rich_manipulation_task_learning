"""Exercise the native RSL-RL runner and the research PPO extensions on CPU."""

from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.config.ur5e.agents.ppo import ContactRearrangementPPO
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg
from contact_rearrangement.src.catalog import SI_200_8_CALIBRATION
from test_model import observation

AGENTS = "contact_rearrangement.config.ur5e.agents"


def runner_config(model_cfg=None, **algorithm_overrides):
    model_cfg = model_cfg or ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
    algorithm = dict(
        class_name=f"{AGENTS}.ppo:ContactRearrangementPPO", num_learning_epochs=2, num_mini_batches=2,
        clip_param=.2, gamma=.99, lam=.95, value_loss_coef=.25, entropy_coef=.01,
        learning_rate=3e-4, max_grad_norm=.1, use_clipped_value_loss=True,
        schedule="fixed", desired_kl=None, normalize_advantage_per_mini_batch=False,
        max_microbatch_size=32, rnd_cfg=None, symmetry_cfg=None,
    )
    algorithm.update(algorithm_overrides)
    return dict(
        class_name="OnPolicyRunner", num_steps_per_env=4, max_iterations=2, save_interval=2,
        logger="tensorboard", experiment_name="contact_rearrangement", run_name="",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name=f"{AGENTS}.model:NativeContactActor", model_cfg=asdict(model_cfg),
                   max_forward_batch_size=32),
        critic=dict(class_name=f"{AGENTS}.model:NativeContactCritic", max_forward_batch_size=32),
        algorithm=algorithm, multi_gpu=None,
    )


class SmallVectorEnv:
    """A deterministic vector API fixture with real observations and auto-resets."""

    def __init__(self, model_cfg, num_envs=2):
        self.num_envs = num_envs
        self.num_actions = model_cfg.action_dim
        self.device = "cpu"
        self.cfg = SimpleNamespace(force_torque=SI_200_8_CALIBRATION)
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.initial = observation(model_cfg, num_envs)
        self.current = self.initial.clone()
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": self.current.clone()}, batch_size=[self.num_envs])

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions):
            raise AssertionError("Runner must preserve the 18-dimensional vector action.")
        self.steps += 1
        self.episode_length_buf += 1
        self.current[:, :3] += .001
        final = self.current.clone()
        done = self.episode_length_buf >= self.max_episode_length
        index = torch.arange(self.num_envs)
        terminated = done & (index % 2 == 0)
        truncated = done & ~terminated
        rewards = (index + 1).float()
        self.current[done] = self.initial[done]
        self.episode_length_buf[done] = 0
        extras = dict(
            terminal_observation=final, time_outs=truncated, terminated=terminated,
            log={"Episode_Reward/manipulation": torch.tensor(.5)},
        )
        return self.get_observations(), rewards, done.long(), extras


def make_algorithm(cfg=None, num_envs=2):
    cfg = runner_config() if cfg is None else cfg
    model_cfg = ModelCfg(**cfg["actor"]["model_cfg"])
    env = SmallVectorEnv(model_cfg, num_envs)
    with redirect_stdout(io.StringIO()):
        algorithm = ContactRearrangementPPO.construct_algorithm(env.get_observations(), env, deepcopy(cfg), "cpu")
    return algorithm, env


def collect(algorithm, env):
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(algorithm.storage.num_transitions_per_env):
            actions = algorithm.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            algorithm.process_env_step(obs, rewards, dones, extras)
        algorithm.compute_returns(obs)


class NativeLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(81)

    def test_native_runner_owns_collection_storage_gae_and_episode_logs(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            runner = OnPolicyRunner(env, deepcopy(cfg), log_dir=directory, device="cpu")
            before = [parameter.detach().clone() for parameter in runner.alg.actor.policy.parameters()]
            with patch.object(Logger, "_store_code_state", return_value=[]):
                runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            runner.logger.writer.flush()
            events = EventAccumulator(directory).Reload()
            runner.logger.writer.close()
            self.assertIs(type(runner), OnPolicyRunner)
            self.assertIsInstance(runner.alg, PPO)
            self.assertIs(type(runner.alg.storage), RolloutStorage)
            self.assertIs(ContactRearrangementPPO.act, PPO.act)
            self.assertIs(ContactRearrangementPPO.compute_returns, PPO.compute_returns)
            self.assertEqual(env.steps, 8)
            self.assertEqual(runner.alg.storage.step, 0)
            self.assertEqual(runner.logger.tot_timesteps, 16)
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(before, runner.alg.actor.policy.parameters())))
            self.assertIn("Learning iteration 0/2", console.getvalue())
            self.assertIn("Learning iteration 1/2", console.getvalue())
            self.assertEqual([entry.value for entry in events.Scalars("Train/mean_reward")], [3., 3.])
            self.assertEqual([entry.value for entry in events.Scalars("Train/mean_episode_length")], [2., 2.])
            for tag in ("Loss/value", "Loss/surrogate", "Policy/mean_std", "Perf/total_fps",
                        "Episode_Reward/manipulation"):
                self.assertIn(tag, events.Tags()["scalars"])
            self.assertTrue((Path(directory) / "model_0.pt").exists())
            self.assertTrue((Path(directory) / "model_1.pt").exists())

    def test_shared_parameters_are_registered_once_in_native_optimizer_and_checkpoint(self):
        algorithm, _ = make_algorithm()
        self.assertIs(algorithm.actor.policy, algorithm.critic.policy)
        self.assertEqual(list(algorithm.critic.parameters()), [])
        self.assertEqual(algorithm.critic.state_dict(), {})
        expected = list(algorithm.actor.policy.parameters())
        actual = algorithm.optimizer.param_groups[0]["params"]
        self.assertEqual([id(value) for value in actual], [id(value) for value in expected])
        self.assertEqual(len(actual), len({id(value) for value in actual}))
        saved = algorithm.save()
        self.assertEqual(saved["research_metadata"]["force_torque"], asdict(SI_200_8_CALIBRATION))
        self.assertIn("policy.critic.weight", saved["actor_state_dict"])
        self.assertEqual(saved["critic_state_dict"], {})

    @staticmethod
    def transition(algorithm, obs, values):
        algorithm.transition.observations = obs
        algorithm.transition.actions = torch.zeros(obs.batch_size[0], 18)
        algorithm.transition.values = torch.tensor(values).reshape(-1, 1)
        algorithm.transition.actions_log_prob = torch.zeros(obs.batch_size[0])
        algorithm.transition.distribution_params = (
            torch.zeros(obs.batch_size[0], 18), torch.ones(obs.batch_size[0], 18),
        )

    def test_timeout_bootstraps_final_state_and_physical_termination_wins(self):
        cfg = runner_config(gamma=.5, lam=1., num_mini_batches=1)
        cfg["num_steps_per_env"] = 2
        algorithm, env = make_algorithm(cfg)
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7.])
        rewards = torch.tensor([1., 5.])
        extras = dict(time_outs=torch.tensor([True, True]), terminated=torch.tensor([False, True]),
                      terminal_observation=obs["policy"].clone())
        with patch.object(algorithm.critic, "forward", return_value=torch.tensor([[4.]])) as critic:
            algorithm.process_env_step(obs, rewards, torch.ones(2, dtype=torch.long), extras)
        self.assertEqual(critic.call_args.args[0].batch_size, torch.Size([1]))
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([3., 5.]))
        torch.testing.assert_close(rewards, torch.tensor([1., 5.]))
        self.assertTrue(extras["time_outs"].all())
        self.transition(algorithm, obs, [0., 0.])
        algorithm.process_env_step(obs, torch.tensor([100., 200.]), torch.ones(2, dtype=torch.long), {})
        with patch.object(algorithm.critic, "forward", return_value=torch.zeros(2, 1)):
            algorithm.compute_returns(obs)
        # Native GAE stops at both episode boundaries. The next episode's
        # rewards cannot leak into either first transition's return.
        torch.testing.assert_close(algorithm.storage.returns[:, :, 0], torch.tensor([[3., 5.], [100., 200.]]))

    def test_timeout_without_final_observation_fails_before_storage_changes(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7.])
        with self.assertRaisesRegex(ValueError, "terminal_observation"):
            algorithm.process_env_step(obs, torch.ones(2), torch.ones(2),
                                       {"time_outs": torch.tensor([True, False])})
        self.assertEqual(algorithm.storage.step, 0)
        algorithm.process_env_step(obs, torch.ones(2), torch.zeros(2),
                                   {"time_outs": torch.tensor([False, False])})
        self.assertEqual(algorithm.storage.step, 1)

    def paired_algorithms(self, **overrides):
        cfg = runner_config(**overrides)
        first, env = make_algorithm(cfg, num_envs=3)
        second, _ = make_algorithm(cfg, num_envs=3)
        second.actor.policy.load_state_dict(first.actor.policy.state_dict())
        collect(first, env)
        second.storage = deepcopy(first.storage)
        return first, second

    def test_microbatches_preserve_original_adam_steps_losses_and_shared_gradients(self):
        full, split = self.paired_algorithms(max_microbatch_size=None)
        split.max_microbatch_size = 4
        torch.manual_seed(92)
        with patch.object(full.optimizer, "step", wraps=full.optimizer.step) as full_steps:
            full_metrics = full.update()
        torch.manual_seed(92)
        with patch.object(split.optimizer, "step", wraps=split.optimizer.step) as split_steps, \
             patch.object(split.actor.policy, "evaluate_actions", wraps=split.actor.policy.evaluate_actions) as forwards:
            split_metrics = split.update()
        self.assertEqual(full_steps.call_count, 4)
        self.assertEqual(split_steps.call_count, 4)
        self.assertEqual([call.args[0].shape[0] for call in forwards.call_args_list], [4, 2] * 4)
        for first, second in zip(full.actor.policy.parameters(), split.actor.policy.parameters()):
            torch.testing.assert_close(first, second, rtol=4e-5, atol=3e-6)
            torch.testing.assert_close(first.grad, second.grad, rtol=4e-4, atol=3e-7)
        for name in full_metrics:
            self.assertAlmostEqual(full_metrics[name], split_metrics[name], places=5)
        for first, second in zip(full.optimizer.state.values(), split.optimizer.state.values()):
            self.assertEqual(first["step"], second["step"])
            torch.testing.assert_close(first["exp_avg"], second["exp_avg"], rtol=5e-4, atol=2e-8)
            torch.testing.assert_close(first["exp_avg_sq"], second["exp_avg_sq"], rtol=1e-3, atol=2e-10)

    def test_unsplit_shared_loss_matches_native_ppo_update(self):
        extension, native = self.paired_algorithms(max_microbatch_size=None, entropy_coef=0.)
        torch.manual_seed(16)
        extension_metrics = extension.update()
        torch.manual_seed(16)
        native_metrics = PPO.update(native)
        for name in ("value", "surrogate"):
            self.assertAlmostEqual(extension_metrics[name], native_metrics[name], places=5)
        for first, second in zip(extension.actor.policy.parameters(), native.actor.policy.parameters()):
            torch.testing.assert_close(first, second, rtol=4e-5, atol=3e-6)

    def test_native_runner_checkpoint_restores_shared_policy_and_optimizer(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, deepcopy(cfg), log_dir=directory, device="cpu")
            with patch.object(Logger, "_store_code_state", return_value=[]):
                runner.learn(num_learning_iterations=2)
            checkpoint = str(Path(directory) / "model_1.pt")
            restored = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            restored.load(checkpoint, map_location="cpu")
            self.assertEqual(restored.current_learning_iteration, 1)
            self.assertIs(restored.alg.actor.policy, restored.alg.critic.policy)
            for first, second in zip(runner.alg.actor.policy.parameters(), restored.alg.actor.policy.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertTrue(restored.alg.optimizer.state)
            for first, second in zip(runner.alg.optimizer.state.values(), restored.alg.optimizer.state.values()):
                torch.testing.assert_close(first["exp_avg"], second["exp_avg"], rtol=0, atol=0)
            runner.logger.writer.flush()
            runner.logger.writer.close()

    def test_checkpoint_semantics_are_checked_before_loading_weights(self):
        algorithm, _ = make_algorithm()
        variants = []
        legacy = algorithm.save()
        del legacy["research_metadata"]
        variants.append(legacy)
        layout = algorithm.save()
        layout["research_metadata"]["model_cfg"]["observation_layout"] = "legacy_roles"
        variants.append(layout)
        calibration = algorithm.save()
        calibration["research_metadata"]["force_torque"]["resolution"] = (.01,) * 6
        variants.append(calibration)
        with patch.object(algorithm.actor, "load_state_dict", wraps=algorithm.actor.load_state_dict) as load:
            for checkpoint in variants:
                with self.subTest(metadata=checkpoint.get("research_metadata")), self.assertRaisesRegex(ValueError, "F/T"):
                    algorithm.load(checkpoint)
            load.assert_not_called()

    def test_inference_policy_uses_same_observable_input(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, deepcopy(cfg), device="cpu")
        policy = runner.get_inference_policy(device="cpu")
        with torch.no_grad():
            actions = policy(env.get_observations())
        self.assertEqual(actions.shape, (2, 18))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions.abs() <= 1.).all())

    def test_invalid_microbatch_sizes_and_unsupported_extensions_fail_explicitly(self):
        for limit in (0, -1, 2.5, True):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "max_microbatch_size"):
                make_algorithm(runner_config(max_microbatch_size=limit))
        for name in ("rnd_cfg", "symmetry_cfg"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                make_algorithm(runner_config(**{name: {"enabled": True}}))
        cfg = runner_config()
        cfg["multi_gpu"] = {"global_rank": 0, "world_size": 2}
        with self.assertRaisesRegex(ValueError, "multi_gpu_cfg"):
            make_algorithm(cfg)

    def test_native_rollout_sample_count_cannot_silently_drop_samples(self):
        cfg = runner_config(num_mini_batches=3)
        with self.assertRaisesRegex(ValueError, "divisible"):
            make_algorithm(cfg)
        cfg["num_steps_per_env"] = 1
        with self.assertRaisesRegex(ValueError, "one sample per PPO minibatch"):
            make_algorithm(cfg)


if __name__ == "__main__":
    unittest.main()
