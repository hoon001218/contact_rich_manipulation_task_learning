"""CPU checks for research GNN/attention and native RSL-RL learning integration."""

import ast
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import copy
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.logger import Logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensordict import TensorDict

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
from contact_manipulation.config.ur5e.agents.model import (
    ContactManipulationEncoder, NativeContactActor, NativeContactCritic, build_policy_masks,
)
from contact_manipulation.config.ur5e.agents.model import ModelCfg, _masked_softmax
from contact_manipulation.config.ur5e.agents.ppo import (
    ACTION_CONTRACT, load_checkpoint, prepare_inference_checkpoint,
)
from contact_manipulation.mdp.observations import SI_200_8_RANGES, SI_200_8_RESOLUTION


def pack(geometry, history):
    return torch.cat((geometry.flatten(1), history.flatten(1)), dim=-1)


def observation(cfg: ModelCfg, batch: int = 3) -> torch.Tensor:
    geometry = torch.randn(batch, cfg.max_objects, 10)
    geometry[..., 3:7] = torch.nn.functional.normalize(geometry[..., 3:7], dim=-1)
    geometry[..., 7:] = geometry[..., 7:].abs() * 0.1 + 0.02
    geometry[:, -1] = 0
    history = torch.randn(batch, cfg.history_steps, cfg.step_features)
    history[..., 27:31] = torch.nn.functional.normalize(history[..., 27:31], dim=-1)
    history[..., 37:54] = torch.randint(0, 2, (batch, cfg.history_steps, 17)).float()
    if cfg.history_steps > 1:
        history[-1, 1:] = 0
    return pack(geometry, history)


def dense_reference(encoder, geometry, valid, roles):
    """Original unoptimized edge concatenation as an independent math oracle."""
    geometry = torch.where(valid[..., None], geometry, 0.0)
    hidden = encoder.node_encoder(geometry) * valid[..., None]
    batch, count, width = hidden.shape
    edges = valid[:, :, None] & valid[:, None, :] & ~torch.eye(count, dtype=torch.bool)
    relations = torch.cat((geometry[:, None, :, :3] - geometry[:, :, None, :3],
                           geometry[:, None, :, 7:] - geometry[:, :, None, 7:]), -1)
    for message, update in zip(encoder.messages, encoder.updates):
        edge_input = torch.cat((hidden[:, :, None, :].expand(batch, count, count, width),
                                hidden[:, None, :, :].expand(batch, count, count, width), relations), -1)
        messages = message(edge_input) * edges[..., None]
        aggregate = messages.sum(-2) / edges.sum(-1, keepdim=True).clamp_min(1)
        hidden = (hidden + update(torch.cat((hidden, aggregate), -1))) * valid[..., None]
    target = (hidden * roles[..., 0, None]).sum(-2)
    blocker = (hidden * roles[..., 1, None]).sum(-2)
    queries = encoder.role_queries(torch.cat((target, blocker), -1)).reshape(batch, 2, width)
    scores = queries @ encoder.keys(hidden).transpose(-2, -1) / math.sqrt(width)
    attention = scores.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
    context = (attention @ encoder.values(hidden)).flatten(-2)
    pooled = hidden.sum(-2) / valid.sum(-1, keepdim=True).clamp_min(1)
    return torch.cat((target, blocker, context, pooled), -1)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(81)
        self.cfg = ModelCfg(max_objects=5, node_dim=16, latent_dim=24)
        self.observation = observation(self.cfg)
        self.model = NativeContactActor(
            TensorDict({"policy": self.observation}, batch_size=[3]),
            {"actor": ["policy"]}, "actor", self.cfg.action_dim, model_cfg=asdict(self.cfg),
            distribution_cfg={"class_name": "GaussianDistribution", "std_type": "log", "init_std": .3},
        )

    def masks(self, values=None):
        return build_policy_masks(self.observation if values is None else values, self.cfg)

    def forward(self, values):
        return self.model(TensorDict({"policy": values}, batch_size=values.shape[:-1]))

    def act(self, values, deterministic=False, *, masks):
        # Unit-test convenience only: production uses the native model interface.
        mean = self.forward(values)
        self.model.distribution.update(mean)
        action = mean if deterministic else self.model.distribution.sample()
        return action, self.model.get_output_log_prob(action)

    def evaluate_actions(self, values, actions, *, masks):
        self.model.distribution.update(self.forward(values))
        return self.model.get_output_log_prob(actions), self.model.output_entropy

    def test_default_observation_layout_excludes_mask_channels(self):
        self.assertEqual(ModelCfg().observation_dim, 244)
        self.assertEqual(asdict(ModelCfg())["history_steps"], 2)
        self.assertEqual(asdict(ModelCfg())["activation"], "relu")
        self.assertFalse(any(isinstance(module, torch.nn.SiLU) for module in self.model.modules()))
        self.assertEqual(asdict(ModelCfg())["observation_layout"], "fixed_roles_v1")
        self.assertEqual(ModelCfg().step_features, 72)
        self.assertEqual(self.model.encoder.scene_encoder.node_encoder[0].in_features, 10)
        self.assertFalse(hasattr(self.model, "history_encoder"))
        self.assertEqual(self.model.mlp[0].in_features, 5 * self.cfg.node_dim + 144)
        with self.assertRaisesRegex(ValueError, "72 physical observation"):
            ModelCfg(step_features=89)
        with self.assertRaisesRegex(ValueError, "At least two object slots"):
            ModelCfg(max_objects=1)
        with self.assertRaisesRegex(ValueError, "observation_layout"):
            ModelCfg(observation_layout="categorical_roles")

    def test_native_model_owns_head_distribution_and_common_interfaces(self):
        self.assertIsInstance(self.model, MLPModel)
        self.assertIsInstance(self.model.encoder, ContactManipulationEncoder)
        for name in ("forward", "get_output_log_prob", "get_kl_divergence", "reset",
                     "get_hidden_state", "detach_hidden_state"):
            self.assertIs(getattr(NativeContactActor, name), getattr(MLPModel, name))
        self.assertEqual(self.model.mlp[0].in_features, 5 * self.cfg.node_dim + 144)
        self.assertEqual(self.model.mlp[0].out_features, 2 * self.cfg.latent_dim)
        self.assertEqual(self.model.mlp[2].out_features, self.cfg.latent_dim)
        self.assertEqual(self.model.mlp[-1].out_features, 18)
        self.assertFalse(hasattr(self.model, "policy"))
        self.assertFalse(hasattr(self.model.encoder, "distribution"))
        self.assertFalse(hasattr(self.model.encoder, "fusion"))

    def test_default_native_gaussian_and_head_initialization_match_official_model(self):
        groups = {"actor": ["policy"]}
        obs = TensorDict({"policy": self.observation}, batch_size=[3])
        torch.manual_seed(123)
        actor = NativeContactActor(obs, groups, "actor", 18, model_cfg=asdict(self.cfg))
        self.assertEqual(actor.distribution.std_type, "scalar")
        torch.testing.assert_close(actor.distribution.std_param, torch.ones(18))
        # Account for encoder initialization before constructing the same native
        # MLP head. There must be no task-specific output gain/bias rewrite.
        torch.manual_seed(123)
        ContactManipulationEncoder(self.cfg)
        latent = torch.zeros(3, 5 * self.cfg.node_dim + 144)
        reference = MLPModel(
            TensorDict({"policy": latent}, batch_size=[3]), groups, "actor", 18,
            hidden_dims=[2 * self.cfg.latent_dim, self.cfg.latent_dim], activation="relu",
            distribution_cfg={"class_name": "GaussianDistribution", "std_type": "scalar", "init_std": 1.},
        )
        for actual, expected in zip(actor.mlp.parameters(), reference.mlp.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_head_matches_previous_linear_relu_mean_and_input_gradients(self):
        # Independent copy of the previous fusion+output expression. Copying
        # weights isolates the structure change from random initialization.
        reference = torch.nn.Sequential(
            torch.nn.Linear(5 * self.cfg.node_dim + 144, 2 * self.cfg.latent_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * self.cfg.latent_dim, self.cfg.latent_dim), torch.nn.ReLU(),
            torch.nn.Linear(self.cfg.latent_dim, 18),
        )
        reference.load_state_dict(self.model.mlp.state_dict())
        features = self.model.encoder(self.observation, masks=self.masks()).detach().requires_grad_()
        reference_features = features.detach().clone().requires_grad_()
        actual, expected = self.model.mlp(features), reference(reference_features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(features.grad, reference_features.grad, rtol=0, atol=0)
        for actual_parameter, expected_parameter in zip(self.model.mlp.parameters(), reference.parameters()):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)

    def test_masks_are_boolean_and_have_no_gradient_or_environment_dependency(self):
        inputs = self.observation.clone().requires_grad_(True)
        masks = self.masks(inputs)
        for mask in (masks.object_valid, masks.roles, masks.history_valid, masks.tactile_valid):
            self.assertEqual(mask.dtype, torch.bool)
            self.assertFalse(mask.requires_grad)
            self.assertIsNone(mask.grad_fn)
            self.assertEqual(mask.device, inputs.device)
        self.assertEqual(masks.object_valid.shape, (3, 5))
        self.assertEqual(masks.roles.shape, (3, 5, 2))
        self.assertEqual(masks.history_valid.shape, (3, 2))
        self.assertEqual(masks.tactile_valid.shape, (3, 2, 17))

    def test_valid_origin_geometry_and_zero_contact_are_not_padding(self):
        geometry, history = self.model.encoder._split(self.observation)
        geometry[:, 0, :3] = 0
        history[..., 37:54] = 0
        masks = self.masks(pack(geometry, history))
        self.assertTrue(masks.object_valid[:, 0].all())
        self.assertTrue(masks.tactile_valid[:, 0].all())
        self.assertTrue(masks.tactile_valid[:2, 1].all())
        self.assertFalse(masks.tactile_valid[-1, 1].any())

    def test_geometry_roles_and_padding_have_distinct_meanings(self):
        geometry, history = self.model.encoder._split(self.observation)
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.object_valid[:, -1].any())
        self.assertFalse(masks.roles[:, -1].any())
        self.assertTrue(masks.object_valid[:, 2].all())
        self.assertFalse(masks.roles[:, 2].any())
        self.assertTrue(masks.roles[:, 0, 0].all())
        self.assertTrue(masks.roles[:, 1, 1].all())

    def test_missing_role_slots_do_not_promote_other_observed_objects(self):
        geometry, history = self.model.encoder._split(self.observation)
        geometry[:, 0] = 0
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.object_valid[:, 0].any())
        self.assertFalse(masks.roles[..., 0].any())
        self.assertTrue(masks.roles[:, 1, 1].all())
        self.assertTrue(masks.object_valid[:, 2:4].all())
        self.assertFalse(masks.roles[:, 2:].any())
        geometry[:, 1] = float("nan")
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.roles.any())

    def test_invalid_geometry_extent_and_quaternion_are_excluded(self):
        geometry, history = self.model.encoder._split(self.observation)
        geometry[:, 0, 7] = 0
        geometry[:, 1, 8] = -0.1
        geometry[:, 2, 3:7] = 0
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.object_valid[:, :3].any())
        self.assertFalse(masks.roles[:, :3].any())

    def test_tactile_unavailable_is_separate_from_no_contact(self):
        geometry, history = self.model.encoder._split(self.observation)
        history[:, 0, 37:42] = torch.tensor([0, 1, -1, float("nan"), 0.5])
        masks = self.masks(pack(geometry, history))
        self.assertTrue(masks.history_valid[:, 0].all())
        torch.testing.assert_close(masks.tactile_valid[:, 0, :5],
                                   torch.tensor([[True, True, False, False, False]]).expand(3, -1))

    def test_history_validity_comes_from_robot_pose_and_finite_physical_values(self):
        geometry, history = self.model.encoder._split(self.observation)
        history[0, 0, 27:31] = 0
        history[1, 0, 31] = float("nan")
        history[2, 0, 54] = float("nan")
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.history_valid[:, 0].any())
        self.assertFalse(masks.tactile_valid[:, 0].any())

    def test_masks_are_explicit_for_encoder_and_derived_by_native_actor(self):
        with self.assertRaises(TypeError):
            self.model.encoder(self.observation)
        self.assertFalse(hasattr(self.model, "critic"))
        self.assertFalse(hasattr(self.model, "value"))
        masks = self.masks()
        with self.assertRaisesRegex(ValueError, "Boolean tensors"):
            self.model.encoder(self.observation, masks=replace(masks, object_valid=masks.object_valid.float()))
        latent = self.model.encoder(self.observation, masks=masks)
        torch.testing.assert_close(self.forward(self.observation), self.model.mlp(latent))

    def test_masks_do_not_become_learned_feature_channels(self):
        geometry, history = self.model.encoder._split(self.observation)
        masks = self.masks()
        node_inputs, history_inputs = [], []
        node_hook = self.model.encoder.scene_encoder.node_encoder[0].register_forward_pre_hook(
            lambda _module, inputs: node_inputs.append(inputs[0].detach().clone()))
        history_hook = self.model.mlp[0].register_forward_pre_hook(
            lambda _module, inputs: history_inputs.append(inputs[0].detach().clone()))
        try:
            self.forward(self.observation)
        finally:
            node_hook.remove()
            history_hook.remove()
        expected_geometry = torch.where(masks.object_valid[..., None], geometry, 0)
        expected_history = torch.where(masks.history_valid[..., None], history, 0)
        torch.testing.assert_close(node_inputs[0], expected_geometry)
        torch.testing.assert_close(history_inputs[0][:, -144:], expected_history.flatten(1))

    def test_distractor_slot_permutation_preserves_policy(self):
        perm = torch.tensor([0, 1, 4, 2, 3])
        geometry, history = self.model.encoder._split(self.observation)
        reordered = pack(geometry[:, perm], history)
        torch.testing.assert_close(self.forward(self.observation), self.forward(reordered), atol=1e-6, rtol=1e-5)

    def test_swapping_reserved_slots_changes_target_blocker_semantics(self):
        geometry, history = self.model.encoder._split(self.observation)
        swapped = pack(geometry[:, torch.tensor([1, 0, 2, 3, 4])], history)
        original_masks, swapped_masks = self.masks(), self.masks(swapped)
        torch.testing.assert_close(original_masks.roles, swapped_masks.roles)
        first, second = self.forward(self.observation), self.forward(swapped)
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_graph_permutation_is_invariant_when_geometry_and_selectors_move_together(self):
        geometry, _ = self.model.encoder._split(self.observation)
        masks = self.masks()
        perm = torch.tensor([4, 2, 0, 3, 1])
        first = self.model.encoder.scene_encoder(geometry, masks.object_valid, masks.roles)
        second = self.model.encoder.scene_encoder(geometry[:, perm], masks.object_valid[:, perm], masks.roles[:, perm])
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_nan_padding_is_ignored_when_masks_are_derived_from_the_poisoned_observation(self):
        geometry, history = self.model.encoder._split(self.observation)
        masks = self.masks()
        geometry = torch.where(masks.object_valid[..., None], geometry, torch.full_like(geometry, float("nan")))
        history = torch.where(masks.history_valid[..., None], history, torch.full_like(history, float("nan")))
        poisoned = pack(geometry, history)
        actual = self.forward(poisoned)
        torch.testing.assert_close(self.forward(self.observation), actual, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_all_padding_is_finite_without_inventing_observed_objects(self):
        padded = torch.full_like(self.observation, float("nan"))
        masks = self.masks(padded)
        actions, log_probs = self.act(padded, masks=masks)
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        geometry, _ = self.model.encoder._split(padded)
        scene = self.model.encoder.scene_encoder(geometry, masks.object_valid, masks.roles)
        torch.testing.assert_close(scene, torch.zeros_like(scene))

    def test_current_previous_samples_reach_final_mlp_in_order_and_receive_gradients(self):
        inputs = self.observation.clone().requires_grad_(True)
        captured = []
        hook = self.model.mlp[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0]))
        try:
            means = self.forward(inputs)
        finally:
            hook.remove()
        _, history = self.model.encoder._split(inputs)
        expected = torch.where(self.masks(inputs).history_valid[..., None], history, 0)
        torch.testing.assert_close(captured[0][:, -144:], expected.flatten(1))
        means.square().sum().backward()
        gradient = inputs.grad[:, -144:].reshape(-1, 2, 72)
        self.assertGreater(float(gradient[0, 0].abs().sum()), 0)
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0)
        self.assertEqual(float(gradient[-1, 1].abs().sum()), 0)

    def test_reset_padding_discards_previous_episode_but_observes_current_sample(self):
        geometry, history = self.model.encoder._split(self.observation)
        history = history.clone()
        history[:, 1] = 0
        inputs = pack(geometry, history)
        expected = self.forward(inputs)
        masks = self.masks(inputs)
        self.assertTrue(masks.history_valid[:, 0].all())
        self.assertFalse(masks.history_valid[:, 1].any())
        history[:, 1] = float("nan")
        actual = self.forward(pack(geometry, history))
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        history[:, 0, :12] += 10.0
        changed = self.forward(pack(geometry, history))
        self.assertGreater(float((expected - changed).abs().max()), 1e-6)

    def test_invalid_tactile_values_are_zeroed_before_projection(self):
        geometry, history = self.model.encoder._split(self.observation)
        history[:, 0, 37] = -1
        expected = self.forward(pack(geometry, history))
        history[:, 0, 37] = float("nan")
        actual = self.forward(pack(geometry, history))
        history[:, 0, 37] = 0
        no_contact = self.forward(pack(geometry, history))
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        torch.testing.assert_close(expected, no_contact, atol=0, rtol=0)

    def test_gaussian_actions_match_analytic_log_probability_and_entropy(self):
        masks = self.masks()
        actions, log_probs = self.act(self.observation, masks=masks)
        evaluated, entropy = self.evaluate_actions(self.observation, actions, masks=masks)
        mean = self.forward(self.observation)
        std = self.model.distribution.std
        expected_log_probability = -.5 * (
            ((actions - mean) / std).square() + 2 * std.log() + math.log(2 * math.pi)
        ).sum(-1)
        expected_entropy = (.5 * math.log(2 * math.pi * math.e) + std.log()).sum(-1)
        torch.testing.assert_close(log_probs, evaluated, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(evaluated, expected_log_probability)
        torch.testing.assert_close(entropy, expected_entropy)
        self.assertTrue(torch.isfinite(entropy).all())
        first = self.act(self.observation, deterministic=True, masks=masks)
        second = self.act(self.observation, deterministic=True, masks=masks)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        torch.testing.assert_close(first[0], mean, atol=0, rtol=0)

    def test_scene_encoder_attention_and_final_mlp_receive_policy_gradients(self):
        masks = self.masks()
        actions, _ = self.act(self.observation, masks=masks)
        logs, entropy = self.evaluate_actions(self.observation, actions.detach(), masks=masks)
        (logs.mean() + 0.01 * entropy.mean()).backward()
        for module in (self.model.encoder.scene_encoder, self.model.mlp, self.model.mlp[-1]):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_analytic_entropy_preserves_objective_across_subbatches_without_sampling(self):
        actions = torch.randn(3, self.model.cfg.action_dim)
        random_state = torch.get_rng_state().clone()
        full = self.evaluate_actions(self.observation, actions, masks=self.masks())
        chunks = [self.evaluate_actions(self.observation[start:stop], actions[start:stop],
                                              masks=self.masks(self.observation[start:stop]))
                  for start, stop in ((0, 2), (2, 3))]
        torch.testing.assert_close(torch.get_rng_state(), random_state, atol=0, rtol=0)
        for column, expected in enumerate(full):
            torch.testing.assert_close(torch.cat([result[column] for result in chunks]), expected,
                                       rtol=1e-5, atol=1e-6)

    def test_large_means_remain_unrestricted_with_finite_probability_and_gradients(self):
        with torch.no_grad():
            self.model.mlp[-1].bias[::2] = 20.0
            self.model.mlp[-1].bias[1::2] = -20.0
        masks = self.masks()
        actions, logs = self.act(self.observation, masks=masks)
        evaluated, entropy = self.evaluate_actions(self.observation, actions, masks=masks)
        self.assertTrue((actions[:, ::2] > 1.0).all())
        self.assertTrue((actions[:, 1::2] < -1.0).all())
        torch.testing.assert_close(logs, evaluated, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(entropy).all())
        deterministic, _ = self.act(self.observation, deterministic=True, masks=masks)
        mean = self.forward(self.observation)
        torch.testing.assert_close(deterministic, mean, atol=0, rtol=0)
        (evaluated.mean() + entropy.mean()).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_native_gaussian_standard_deviation_is_positive_and_learnable(self):
        self.assertIsInstance(self.model.distribution, GaussianDistribution)
        self.assertEqual(self.model.distribution.std_type, "log")
        parameter = self.model.distribution.log_std_param
        torch.testing.assert_close(parameter.exp(), torch.full((18,), .3))
        before = parameter.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        actions = torch.zeros(3, self.cfg.action_dim)
        _, entropy = self.evaluate_actions(self.observation, actions, masks=self.masks())
        (-entropy.mean()).backward()
        torch.testing.assert_close(parameter.grad, -torch.ones_like(parameter))
        optimizer.step()
        self.assertTrue((parameter > before).all())
        self.assertTrue((parameter.exp() > 0).all())

    def test_arbitrary_leading_batch_dimensions(self):
        repeated = self.observation[None].expand(2, -1, -1)
        masks = self.masks(repeated)
        actions, logs = self.act(repeated, masks=masks)
        self.assertEqual(actions.shape, (2, 3, 18))
        self.assertEqual(logs.shape, (2, 3))
        self.assertEqual(masks.roles.shape, (2, 3, 5, 2))

    def test_single_observation_without_batch_dimension(self):
        single = self.observation[0]
        actions, log_probability = self.act(single, masks=self.masks(single))
        self.assertEqual(actions.shape, (18,))
        self.assertEqual(log_probability.shape, ())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA fused edge reduction")
    def test_fused_edge_reduction_matches_dense_output_and_backward(self):
        from contact_manipulation.config.ur5e.agents.model import _edge_mean
        for batch, count, width in ((3, 5, 17), (17, 40, 64)):
            receiver = torch.randn(batch, count, width, device="cuda", requires_grad=True)
            sender = torch.randn_like(receiver, requires_grad=True)
            with torch.no_grad():
                receiver[0] = 0
                sender[0] = 0  # ReLU at exactly zero must have zero gradient.
            edges = torch.rand(batch, count, count, device="cuda") > .3
            edges[:, -1] = False  # Empty neighborhoods, even with other valid nodes.
            degree = edges.sum(-1, keepdim=True)
            # Non-contiguous upstream gradients also occur in composite models.
            gradient = torch.randn(width, count, batch, device="cuda").permute(2, 1, 0)
            expected = (torch.relu(receiver[:, :, None] + sender[:, None]) * edges[..., None]).sum(-2) / degree.clamp_min(1)
            expected_grad = torch.autograd.grad(expected, (receiver, sender), gradient)
            actual = _edge_mean(receiver, sender, edges, degree)
            actual_grad = torch.autograd.grad(actual, (receiver, sender), gradient)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
            for a, b in zip(actual_grad, expected_grad):
                torch.testing.assert_close(a, b, rtol=3e-6, atol=5e-7)
            self.assertEqual(actual_grad[0][0].count_nonzero(), 0)
            self.assertEqual(actual_grad[1][0].count_nonzero(), 0)

    def test_factorized_edge_messages_match_dense_reference_outputs_and_gradients(self):
        optimized = self.model.encoder.scene_encoder.double()
        reference = copy.deepcopy(optimized)
        geometry = self.model.encoder._split(self.observation)[0].double().requires_grad_()
        reference_geometry = geometry.detach().clone().requires_grad_()
        masks = self.masks()
        actual = optimized(geometry, masks.object_valid, masks.roles)
        expected = dense_reference(reference, reference_geometry, masks.object_valid, masks.roles)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        weights = torch.randn_like(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        torch.testing.assert_close(geometry.grad, reference_geometry.grad, atol=2e-11, rtol=2e-11)
        for optimized_parameter, reference_parameter in zip(optimized.parameters(), reference.parameters()):
            torch.testing.assert_close(optimized_parameter.grad, reference_parameter.grad,
                                       atol=2e-11, rtol=2e-11)

    def test_isolated_node_has_no_edge_affine_bias_or_gradient(self):
        geometry, history = self.model.encoder._split(self.observation)
        geometry[:, 1:] = 0
        masks = self.masks(pack(geometry, history))
        optimized = self.model.encoder.scene_encoder.double()
        for message in optimized.messages:
            with torch.no_grad():
                message[2].bias.fill_(.7)
        reference = copy.deepcopy(optimized)
        actual = optimized(geometry.double(), masks.object_valid, masks.roles)
        expected = dense_reference(reference, geometry.double(), masks.object_valid, masks.roles)
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
        actual.square().sum().backward()
        for message in optimized.messages:
            torch.testing.assert_close(message[2].bias.grad, torch.zeros_like(message[2].bias))
            torch.testing.assert_close(message[2].weight.grad, torch.zeros_like(message[2].weight))

    def test_masked_attention_normalizes_valid_nodes_and_empty_rows_are_zero(self):
        scores = torch.tensor([[2., 100., -2.], [0., 0., 0.]], requires_grad=True)
        mask = torch.tensor([[True, False, True], [False, False, False]])
        attention = _masked_softmax(scores, mask)
        torch.testing.assert_close(attention.sum(-1), torch.tensor([1., 0.]))
        self.assertEqual(float(attention[0, 1]), 0.0)
        self.assertTrue(torch.isfinite(attention).all())
        (attention * torch.tensor([1., 2., 3.])).sum().backward()
        torch.testing.assert_close(scores.grad[~mask], torch.zeros_like(scores.grad[~mask]))

    def test_attention_and_both_message_layers_receive_nonzero_gradients(self):
        means = self.forward(self.observation)
        means.square().sum().backward()
        graph = self.model.encoder.scene_encoder
        for module in (*graph.messages, graph.role_queries, graph.keys, graph.values):
            self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                                for parameter in module.parameters()))
            self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()), 0.0)

    def test_parallel_batch_matches_individual_observations(self):
        inputs = observation(self.cfg, batch=1024)
        with torch.no_grad():
            actions = self.forward(inputs)
            for index in (0, 1, 31, 32, 511, 1023):
                single_actions = self.forward(inputs[index])
                torch.testing.assert_close(actions[index], single_actions, atol=1e-7, rtol=1e-5)

    def test_sampling_and_evaluation_do_not_extract_tensor_scalars(self):
        def forbid(*_args, **_kwargs):
            raise AssertionError("Policy hot path extracted a tensor scalar.")

        with patch.object(torch.Tensor, "__bool__", forbid), patch.object(torch.Tensor, "item", forbid):
            masks = self.masks()
            actions, _ = self.act(self.observation, masks=masks)
            self.evaluate_actions(self.observation, actions, masks=masks)

    def test_research_history_and_action_contract_rejects_incompatible_config(self):
        for kwargs in ({"history_steps": 16}, {"action_dim": 12}, {"max_objects": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelCfg(**kwargs)


AGENTS = "contact_manipulation.config.ur5e.agents"


def runner_config(model_cfg=None, **algorithm_overrides):
    model_cfg = model_cfg or ModelCfg(max_objects=5, node_dim=16, latent_dim=24)
    algorithm = dict(
        class_name="PPO", num_learning_epochs=2, num_mini_batches=2,
        clip_param=.2, gamma=.99, lam=.95, value_loss_coef=.25, entropy_coef=.01,
        learning_rate=3e-4, max_grad_norm=.1, use_clipped_value_loss=True,
        schedule="fixed", desired_kl=None, normalize_advantage_per_mini_batch=False,
        rnd_cfg=None, symmetry_cfg=None,
    )
    algorithm.update(algorithm_overrides)
    return dict(
        class_name="OnPolicyRunner", num_steps_per_env=4, max_iterations=2, save_interval=2,
        logger="tensorboard", experiment_name="contact_manipulation", run_name="",
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        actor=dict(class_name=f"{AGENTS}.model:NativeContactActor", model_cfg=asdict(model_cfg),
                   max_forward_batch_size=32,
                   distribution_cfg={"class_name": "GaussianDistribution", "std_type": "log", "init_std": .3}),
        critic=dict(class_name=f"{AGENTS}.model:NativeContactCritic", model_cfg=asdict(model_cfg),
                    max_forward_batch_size=32,
                    hidden_dims=[2 * model_cfg.latent_dim, model_cfg.latent_dim], activation="relu",
                    obs_normalization=False, distribution_cfg=None),
        algorithm=algorithm, multi_gpu=None,
    )


class SmallVectorEnv:
    """A deterministic vector API fixture with real observations and auto-resets."""

    def __init__(self, model_cfg, num_envs=2, critic_cfg=None):
        self.num_envs = num_envs
        self.num_actions = model_cfg.action_dim
        self.device = "cpu"
        self.cfg = SimpleNamespace()
        self.max_episode_length = 2
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.initial = observation(model_cfg, num_envs)
        self.current = self.initial.clone()
        critic_cfg = critic_cfg or model_cfg
        self.critic_initial = observation(critic_cfg, num_envs)
        self.critic_current = self.critic_initial.clone()
        self.steps = 0

    def get_observations(self):
        critic = torch.cat((self.critic_current[:, :-144], self.current[:, -144:]), dim=-1)
        return TensorDict({"policy": self.current.clone(), "critic": critic}, batch_size=[self.num_envs])

    def step(self, actions):
        if actions.shape != (self.num_envs, self.num_actions):
            raise AssertionError("Runner must preserve the 18-dimensional vector action.")
        self.steps += 1
        self.episode_length_buf += 1
        self.current[:, :3] += .001
        self.critic_current[:, :3] += .003
        final = self.current.clone()
        done = self.episode_length_buf >= self.max_episode_length
        index = torch.arange(self.num_envs)
        terminated = done & (index % 2 == 0)
        truncated = done & ~terminated
        rewards = (index + 1).float()
        self.current[done] = self.initial[done]
        self.critic_current[done] = self.critic_initial[done]
        self.episode_length_buf[done] = 0
        extras = dict(
            terminal_observation=final, time_outs=truncated, terminated=terminated,
            log={"Episode_Reward/manipulation": torch.tensor(.5)},
        )
        return self.get_observations(), rewards, done.long(), extras


def make_algorithm(cfg=None, num_envs=2):
    cfg = runner_config() if cfg is None else cfg
    model_cfg = ModelCfg(**cfg["actor"]["model_cfg"])
    critic_cfg = ModelCfg(**cfg["critic"]["model_cfg"]) if "model_cfg" in cfg["critic"] else None
    env = SmallVectorEnv(model_cfg, num_envs, critic_cfg=critic_cfg)
    with redirect_stdout(io.StringIO()):
        algorithm = PPO.construct_algorithm(env.get_observations(), env, deepcopy(cfg), "cpu")
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

    def test_training_configuration_matches_registered_sweeping_random_example(self):
        workspace = Path(__file__).resolve().parents[3]
        baseline = workspace / "examples/Sweeping-Policy-DRL-Sim-Train/sweeping_policy/config/ur5e/agents/rsl_rl_ppo_cfg_02.py"
        production = Path(__file__).resolve().parents[1] / "contact_manipulation/config/ur5e/agents/rsl_rl_ppo_cfg.py"

        def class_settings(path, class_name):
            node = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.ClassDef) and node.name == class_name)
            values = {}
            for statement in node.body:
                if isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name):
                    name, value = statement.targets[0].id, statement.value
                elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                    name, value = statement.target.id, statement.value
                else:
                    continue
                try:
                    if isinstance(value, ast.Call):
                        values[name] = {keyword.arg: ast.literal_eval(keyword.value) for keyword in value.keywords}
                    else:
                        values[name] = ast.literal_eval(value)
                except (ValueError, TypeError):
                    pass
            return values

        expected = class_settings(baseline, "UR5eSweepPPORunnerCfg")
        actual = class_settings(production, "ContactManipulationPPORunnerCfg")
        for name in ("num_steps_per_env", "max_iterations", "save_interval"):
            self.assertEqual(actual[name], expected[name], name)
        self.assertEqual(actual["algorithm"]["class_name"], "PPO")
        self.assertEqual({key: value for key, value in actual["algorithm"].items() if key != "class_name"},
                         expected["algorithm"])
        self.assertNotIn("seed", actual)  # Inherit the same native runner default.
        self.assertIsNone(actual["clip_actions"])
        self.assertEqual(actual["obs_groups"], {"actor": ["policy"], "critic": ["critic"]})
        actor = class_settings(production, "ContactActorCfg")
        self.assertEqual(actor["obs_normalization"], expected["policy"]["actor_obs_normalization"])
        self.assertEqual(actor["distribution_cfg"], {
            "class_name": "GaussianDistribution", "std_type": "scalar",
            "init_std": expected["policy"]["init_noise_std"],
        })
        # Both research feature extractors and final heads are deliberate exceptions.
        self.assertEqual(actor["hidden_dims"], [256, 128])
        self.assertEqual(actor["activation"], "relu")
        critic = class_settings(production, "ContactCriticCfg")
        self.assertEqual(critic["class_name"], f"{AGENTS}.model:NativeContactCritic")
        self.assertIsNone(critic["distribution_cfg"])
        # The critic inherits the actor's feature/head configuration, rather than
        # diverging silently from the independently instantiated actor.
        production_tree = ast.parse(production.read_text())
        critic_class = next(node for node in production_tree.body
                            if isinstance(node, ast.ClassDef) and node.name == "ContactCriticCfg")
        self.assertIn("ContactActorCfg", [base.id for base in critic_class.bases if isinstance(base, ast.Name)])

    def test_native_runner_owns_collection_storage_gae_and_episode_logs(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as console:
            runner = OnPolicyRunner(env, deepcopy(cfg), log_dir=directory, device="cpu")
            before = [parameter.detach().clone() for parameter in runner.alg.actor.parameters()]
            with patch.object(Logger, "_store_code_state", return_value=[]):
                runner.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            runner.logger.writer.flush()
            events = EventAccumulator(directory).Reload()
            runner.logger.writer.close()
            self.assertIs(type(runner), OnPolicyRunner)
            self.assertIsInstance(runner.alg, PPO)
            self.assertIs(type(runner.alg.storage), RolloutStorage)
            self.assertIs(type(runner.alg), PPO)
            for name in ("act", "process_env_step", "compute_returns", "update", "save", "load"):
                self.assertIs(getattr(type(runner.alg), name), getattr(PPO, name))
            self.assertEqual(env.steps, 8)
            self.assertEqual(runner.alg.storage.step, 0)
            self.assertEqual(runner.logger.tot_timesteps, 16)
            self.assertTrue(any(not torch.equal(old, new) for old, new in
                                zip(before, runner.alg.actor.parameters())))
            self.assertIn("Learning iteration 0/2", console.getvalue())
            self.assertIn("Learning iteration 1/2", console.getvalue())
            self.assertEqual([entry.value for entry in events.Scalars("Train/mean_reward")], [3., 3.])
            self.assertEqual([entry.value for entry in events.Scalars("Train/mean_episode_length")], [2., 2.])
            for tag in ("Loss/value", "Loss/surrogate", "Policy/mean_std", "Perf/total_fps",
                        "Episode_Reward/manipulation"):
                self.assertIn(tag, events.Tags()["scalars"])
            self.assertTrue((Path(directory) / "model_0.pt").exists())
            self.assertTrue((Path(directory) / "model_1.pt").exists())

    def test_independent_native_critic_has_separate_parameters_and_checkpoint(self):
        algorithm, _ = make_algorithm()
        self.assertIs(type(algorithm.critic), NativeContactCritic)
        self.assertIsInstance(algorithm.critic, MLPModel)
        self.assertIsNone(algorithm.critic.distribution)
        self.assertIsNot(algorithm.actor.encoder, algorithm.critic.encoder)
        self.assertIsNot(algorithm.actor.encoder.scene_encoder, algorithm.critic.encoder.scene_encoder)
        self.assertFalse(hasattr(algorithm.actor, "critic"))
        self.assertFalse(hasattr(algorithm.actor, "consume_cached_value"))
        self.assertIsInstance(algorithm.actor.distribution, GaussianDistribution)
        self.assertEqual(sum(isinstance(module, GaussianDistribution)
                             for module in algorithm.actor.modules()), 1)
        actor_parameters = list(algorithm.actor.parameters())
        critic_parameters = list(algorithm.critic.parameters())
        self.assertTrue(critic_parameters)
        self.assertTrue({id(value) for value in actor_parameters}.isdisjoint(
            id(value) for value in critic_parameters))
        self.assertTrue({value.data_ptr() for value in actor_parameters}.isdisjoint(
            value.data_ptr() for value in critic_parameters))
        expected = actor_parameters + critic_parameters
        actual = algorithm.optimizer.param_groups[0]["params"]
        self.assertEqual([id(value) for value in actual], [id(value) for value in expected])
        self.assertEqual(len(actual), len({id(value) for value in actual}))
        saved = algorithm.save()
        metadata = saved["actor_state_dict"]["_extra_state"]
        self.assertEqual(metadata["architecture"], "native_graph_actor_v3")
        self.assertEqual(metadata["action_contract"], ACTION_CONTRACT)
        self.assertEqual(metadata["model_cfg"], asdict(algorithm.actor.cfg))
        self.assertEqual(metadata["actor_model"], algorithm.actor.native_model_cfg)
        self.assertNotIn("research_metadata", saved)
        self.assertFalse(any("critic" in name for name in saved["actor_state_dict"]))
        self.assertIn("distribution.log_std_param", saved["actor_state_dict"])
        self.assertTrue(saved["critic_state_dict"])
        critic_metadata = saved["critic_state_dict"]["_extra_state"]
        self.assertEqual(critic_metadata["architecture"], "native_graph_critic_v2")
        self.assertEqual(critic_metadata["action_contract"], ACTION_CONTRACT)
        self.assertEqual(critic_metadata["observation_contract"], "tracked_initial_objects_v1")
        self.assertEqual(critic_metadata["model_cfg"], asdict(algorithm.critic.cfg))
        self.assertEqual(algorithm.critic.cfg.max_objects, algorithm.actor.cfg.max_objects)
        self.assertEqual(critic_metadata["critic_model"], algorithm.critic.native_model_cfg)
        self.assertEqual(algorithm.critic.mlp[0].in_features, algorithm.actor.mlp[0].in_features)
        self.assertEqual(algorithm.critic.mlp[0].out_features, algorithm.actor.mlp[0].out_features)
        self.assertEqual(algorithm.critic.mlp[2].out_features, algorithm.actor.mlp[2].out_features)
        self.assertEqual(algorithm.critic.mlp[-1].out_features, 1)
        self.assertIsInstance(algorithm.critic.mlp[1], torch.nn.ReLU)
        self.assertFalse(any("distribution" in name for name in saved["critic_state_dict"]))
        for name in ("forward", "reset", "get_hidden_state", "detach_hidden_state"):
            self.assertIs(getattr(NativeContactCritic, name), getattr(MLPModel, name))

    def test_actor_and_critic_execute_scene_encoder_before_final_native_mlp(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        for model, width in ((algorithm.actor, 18), (algorithm.critic, 1)):
            calls, scene_features, head_inputs = [], [], []

            def scene_hook(_module, _args, output):
                calls.append("scene")
                scene_features.append(output.detach().clone())

            def head_hook(_module, args):
                calls.append("mlp")
                head_inputs.append(args[0].detach().clone())

            hooks = (model.encoder.scene_encoder.register_forward_hook(scene_hook),
                     model.mlp.register_forward_pre_hook(head_hook))
            try:
                output = model(obs)
            finally:
                for hook in hooks:
                    hook.remove()
            with self.subTest(model=type(model).__name__):
                self.assertEqual(calls, ["scene", "mlp"])
                self.assertEqual(output.shape, (env.num_envs, width))
                torch.testing.assert_close(head_inputs[0][:, :5 * model.cfg.node_dim], scene_features[0])
                source = obs["policy" if model is algorithm.actor else "critic"]
                masks = build_policy_masks(source, model.cfg)
                expected_history = model.encoder.history_input(source, masks=masks)
                torch.testing.assert_close(head_inputs[0][:, -144:], expected_history)
                self.assertFalse(hasattr(model.encoder, "history_encoder"))
                self.assertFalse(hasattr(model.encoder, "fusion"))

    def test_asymmetric_groups_keep_tracked_current_geometry_out_of_actor(self):
        algorithm, env = make_algorithm()
        inputs = env.get_observations()
        self.assertEqual(inputs["policy"].shape[-1], algorithm.actor.cfg.observation_dim)
        self.assertEqual(inputs["critic"].shape[-1], algorithm.critic.cfg.observation_dim)
        self.assertEqual(inputs["policy"].shape, inputs["critic"].shape)
        self.assertFalse(torch.equal(inputs["policy"][:, :-144], inputs["critic"][:, :-144]))
        torch.testing.assert_close(inputs["policy"][:, -144:], inputs["critic"][:, -144:])
        with torch.no_grad():
            actions, values = algorithm.actor(inputs), algorithm.critic(inputs)
            hidden_truth = inputs.clone()
            hidden_truth["critic"].fill_(float("nan"))
            torch.testing.assert_close(algorithm.actor(hidden_truth), actions, rtol=0, atol=0)
            hidden_vision = inputs.clone()
            hidden_vision["policy"].fill_(float("nan"))
            torch.testing.assert_close(algorithm.critic(hidden_vision), values, rtol=0, atol=0)
        captured = []
        hook = algorithm.critic.encoder.register_forward_pre_hook(
            lambda _module, args: captured.append(args[0].clone()))
        try:
            algorithm.critic(inputs)
        finally:
            hook.remove()
        torch.testing.assert_close(captured[0], inputs["critic"], rtol=0, atol=0)
        collect(algorithm, env)
        stored = algorithm.storage.observations
        self.assertEqual(stored["policy"].shape[-1], algorithm.actor.cfg.observation_dim)
        self.assertEqual(stored["critic"].shape[-1], algorithm.critic.cfg.observation_dim)

    def test_independent_graph_models_do_not_share_normalization_buffers(self):
        cfg = runner_config()
        cfg["actor"]["obs_normalization"] = True
        cfg["critic"]["obs_normalization"] = True
        algorithm, env = make_algorithm(cfg)
        actor, critic = algorithm.actor, algorithm.critic
        actor_buffers = dict(actor.obs_normalizer.named_buffers())
        critic_buffers = dict(critic.obs_normalizer.named_buffers())
        self.assertEqual(actor_buffers.keys(), critic_buffers.keys())
        self.assertTrue(actor_buffers)
        for name in actor_buffers:
            self.assertIsNot(actor_buffers[name], critic_buffers[name], name)
            self.assertNotEqual(actor_buffers[name].data_ptr(), critic_buffers[name].data_ptr(), name)
        obs = env.get_observations()
        actor.update_normalization(obs)
        self.assertEqual(int(actor.obs_normalizer.count), env.num_envs)
        self.assertEqual(int(critic.obs_normalizer.count), 0)
        actor_statistics = {name: value.clone() for name, value in actor.obs_normalizer.named_buffers()}
        changed_obs = obs.clone()
        changed_obs["critic"] *= 2.
        critic.update_normalization(changed_obs)
        for name, value in actor.obs_normalizer.named_buffers():
            torch.testing.assert_close(value, actor_statistics[name], rtol=0, atol=0)
        self.assertEqual(int(critic.obs_normalizer.count), env.num_envs)
        self.assertFalse(torch.equal(actor.obs_normalizer.mean, critic.obs_normalizer.mean))

    @staticmethod
    def transition(algorithm, obs, values):
        algorithm.transition.observations = obs
        algorithm.transition.actions = torch.zeros(obs.batch_size[0], 18)
        algorithm.transition.values = torch.tensor(values).reshape(-1, 1)
        algorithm.transition.actions_log_prob = torch.zeros(obs.batch_size[0])
        algorithm.transition.distribution_params = (
            torch.zeros(obs.batch_size[0], 18), torch.ones(obs.batch_size[0], 18),
        )

    def test_native_actor_normalizes_history_after_raw_masks_and_preserves_padding(self):
        cfg = runner_config()
        cfg["actor"]["obs_normalization"] = True
        algorithm, env = make_algorithm(cfg, num_envs=3)
        actor = algorithm.actor
        raw = env.get_observations()["policy"].clone()
        geometry = raw[:, :(actor.cfg.max_objects * actor.cfg.node_features)].reshape(3, actor.cfg.max_objects, 10)
        history = raw[:, (actor.cfg.max_objects * actor.cfg.node_features):].reshape(3, 2, 72)
        geometry[:, -1] = float("nan")
        history[-1, 1] = float("nan")
        history[0, 0, 37] = -1
        obs = TensorDict({"policy": raw}, batch_size=[3])
        masks = build_policy_masks(raw, actor.cfg)
        clean = actor.encoder.history_input(raw, masks=masks)
        actor.update_normalization(obs)
        self.assertEqual(int(actor.obs_normalizer.count), 3)
        torch.testing.assert_close(actor.obs_normalizer.mean, clean.mean(0))
        self.assertTrue(all(torch.isfinite(value).all() for value in actor.obs_normalizer.buffers()))
        # Nonzero means would turn absent history/pads into learned constants
        # unless masking is reapplied after normalization.
        with torch.no_grad():
            actor.obs_normalizer._mean.fill_(2.)
            actor.obs_normalizer._std.fill_(3.)
        scene_inputs = []
        hook = actor.encoder.scene_encoder.node_encoder[0].register_forward_pre_hook(
            lambda _module, args: scene_inputs.append(args[0].detach().clone()))
        try:
            latent = actor.get_latent(obs)
        finally:
            hook.remove()
        expected = actor.encoder(raw, masks=masks, history=actor.obs_normalizer(clean))
        torch.testing.assert_close(latent, expected)
        torch.testing.assert_close(scene_inputs[0], torch.where(masks.object_valid[..., None], geometry, 0))
        normalized_history = latent[:, -144:].reshape(3, 2, 72)
        self.assertEqual(normalized_history[-1, 1].count_nonzero(), 0)
        self.assertEqual(float(normalized_history[0, 0, 37]), 0.)
        self.assertTrue(torch.isfinite(actor(obs, stochastic_output=True)).all())
        self.assertTrue(torch.isfinite(actor.get_output_log_prob(actor.output_mean)).all())

    def test_native_timeout_bootstraps_stored_value_and_gae_stops_at_done(self):
        cfg = runner_config(gamma=.5, lam=1., num_mini_batches=1)
        cfg["num_steps_per_env"] = 2
        algorithm, env = make_algorithm(cfg)
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7.])
        rewards = torch.tensor([1., 5.])
        # Native PPO uses the value saved by act(o_t). It does not evaluate
        # terminal observations or apply this project's former timeout override.
        extras = dict(time_outs=torch.tensor([True, False]),
                      terminal_observation=obs["policy"].clone())
        with patch.object(algorithm.critic, "forward", side_effect=AssertionError("Extra critic call")):
            algorithm.process_env_step(obs, rewards, torch.ones(2, dtype=torch.long), extras)
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([2., 5.]))
        torch.testing.assert_close(rewards, torch.tensor([1., 5.]))
        self.transition(algorithm, obs, [0., 0.])
        algorithm.process_env_step(obs, torch.tensor([100., 200.]), torch.ones(2, dtype=torch.long), {})
        with patch.object(algorithm.critic, "forward", return_value=torch.zeros(2, 1)):
            algorithm.compute_returns(obs)
        torch.testing.assert_close(algorithm.storage.returns[:, :, 0], torch.tensor([[2., 5.], [100., 200.]]))

    def test_native_runner_checkpoint_restores_actor_critic_and_optimizer(self):
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
            self.assertIs(type(restored.alg.critic), NativeContactCritic)
            for first, second in zip(runner.alg.actor.parameters(), restored.alg.actor.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            for first, second in zip(runner.alg.critic.parameters(), restored.alg.critic.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertTrue(restored.alg.optimizer.state)
            for first, second in zip(runner.alg.optimizer.state.values(), restored.alg.optimizer.state.values()):
                torch.testing.assert_close(first["exp_avg"], second["exp_avg"], rtol=0, atol=0)
            # Play uses the native runner with strict weights/metadata checks,
            # but skips the training optimizer and returns deterministic actions.
            player = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            player.load(checkpoint, map_location="cpu", load_cfg={
                "actor": True, "critic": True, "iteration": True, "optimizer": False, "rnd": False,
            })
            self.assertFalse(player.alg.optimizer.state)
            self.assertEqual(player.current_learning_iteration, 1)
            policy = player.get_inference_policy(device="cpu")
            with torch.inference_mode():
                inputs = env.get_observations()
                expected = runner.get_inference_policy(device="cpu")(inputs)
                torch.testing.assert_close(policy(inputs), expected, rtol=0, atol=0)
                torch.testing.assert_close(policy(inputs), expected, rtol=0, atol=0)
                policy.reset(torch.ones(env.num_envs, dtype=torch.bool))
            runner.logger.writer.flush()
            runner.logger.writer.close()

    def test_native_actor_normalization_statistics_roundtrip_with_checkpoint(self):
        cfg = runner_config()
        cfg["actor"]["obs_normalization"] = True
        cfg["critic"]["obs_normalization"] = True
        original, env = make_algorithm(cfg)
        obs = env.get_observations()
        original.actor.update_normalization(obs)
        original.actor.update_normalization(TensorDict({"policy": obs["policy"] * 1.2}, batch_size=obs.batch_size))
        original.critic.update_normalization(TensorDict({"critic": obs["critic"] * 1.5}, batch_size=obs.batch_size))
        saved = deepcopy(original.save())
        restored, _ = make_algorithm(cfg)
        restored.load(saved, load_cfg=None, strict=True)
        self.assertTrue(saved["actor_state_dict"]["_extra_state"]["actor_model"]["obs_normalization"])
        torch.testing.assert_close(restored.actor.obs_normalizer.mean, original.actor.obs_normalizer.mean,
                                   rtol=0, atol=0)
        torch.testing.assert_close(restored.actor.obs_normalizer.std, original.actor.obs_normalizer.std,
                                   rtol=0, atol=0)
        self.assertEqual(restored.actor.obs_normalizer.count, original.actor.obs_normalizer.count)
        self.assertTrue(saved["critic_state_dict"]["_extra_state"]["critic_model"]["obs_normalization"])
        torch.testing.assert_close(restored.critic.obs_normalizer.mean, original.critic.obs_normalizer.mean,
                                   rtol=0, atol=0)
        torch.testing.assert_close(restored.critic.obs_normalizer.std, original.critic.obs_normalizer.std,
                                   rtol=0, atol=0)
        self.assertEqual(restored.critic.obs_normalizer.count, original.critic.obs_normalizer.count)
        with torch.no_grad():
            torch.testing.assert_close(restored.actor(obs), original.actor(obs), rtol=0, atol=0)
            torch.testing.assert_close(restored.critic(obs), original.critic(obs), rtol=0, atol=0)

    def test_playback_restores_legacy_distribution_without_resuming_custom_optimizer(self):
        legacy_cfg = runner_config()
        legacy_cfg["obs_groups"]["critic"] = ["policy"]
        legacy_cfg["critic"] = dict(class_name="MLPModel", hidden_dims=[256, 128, 64], activation="elu",
                                    obs_normalization=False, distribution_cfg=None)
        original, env = make_algorithm(legacy_cfg)
        saved = deepcopy(original.save())
        saved["actor_state_dict"].pop("_extra_state")
        saved["iter"], saved["infos"] = 19100, {"test": "legacy"}
        saved["research_metadata"] = {
            "action_contract": ACTION_CONTRACT,
            "actor_critic_architecture": "separate_native_graph_actor_mlp_critic_v2",
            "model_cfg": {**asdict(original.actor.cfg), "initial_std": .3},
            "actor_model": deepcopy(original.actor.native_model_cfg),
            "output_distribution": deepcopy(original.actor.native_model_cfg["distribution_cfg"]),
            "critic_model": deepcopy(legacy_cfg["critic"]),
            "force_torque": {"name": "ATI Axia80 SI-200-8", "ranges": SI_200_8_RANGES,
                             "resolution": SI_200_8_RESOLUTION},
        }
        prefixes = (("encoder.scene_encoder.", "policy.scene_encoder."),
                    ("mlp.0.", "policy.fusion.0."), ("mlp.2.", "policy.fusion.2."),
                    ("mlp.4.", "policy.actor."), ("distribution.", "policy.distribution."))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            for version in (1, 2):
                checkpoint = deepcopy(saved)
                if version == 1:
                    checkpoint["research_metadata"]["actor_critic_architecture"] = "separate_actor_mlp_critic_v1"
                    checkpoint["research_metadata"].pop("actor_model")
                    converted = {}
                    for name, value in checkpoint["actor_state_dict"].items():
                        for current, legacy in prefixes:
                            if name.startswith(current):
                                converted[legacy + name[len(current):]] = value
                                break
                        else:
                            self.fail(f"Unexpected legacy state key: {name}")
                    checkpoint["actor_state_dict"] = converted
                path = Path(directory) / f"legacy_v{version}.pt"
                torch.save(checkpoint, path)
                cfg = runner_config()
                cfg["actor"]["distribution_cfg"] = {
                    "class_name": "GaussianDistribution", "std_type": "scalar", "init_std": 1.,
                }
                agent_cfg = SimpleNamespace(obs_groups=deepcopy(cfg["obs_groups"]),
                                            actor=SimpleNamespace(**cfg["actor"]),
                                            critic=SimpleNamespace(**cfg["critic"]))
                loaded = prepare_inference_checkpoint(agent_cfg, path)
                self.assertNotIn("initial_std", agent_cfg.actor.model_cfg)
                self.assertEqual(agent_cfg.actor.distribution_cfg["std_type"], "log")
                self.assertEqual(agent_cfg.critic.class_name, "MLPModel")
                self.assertEqual(agent_cfg.obs_groups["critic"], ["policy"])
                cfg["actor"], cfg["critic"] = vars(agent_cfg.actor), vars(agent_cfg.critic)
                cfg["obs_groups"] = agent_cfg.obs_groups
                player = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
                with self.subTest(version=version):
                    with self.assertRaisesRegex(ValueError, "Legacy optimizer resume"):
                        load_checkpoint(player, path, checkpoint=loaded)
                    info = load_checkpoint(player, path, inference=True, checkpoint=loaded)
                    self.assertEqual(info, {"test": "legacy"})
                    self.assertEqual(player.current_learning_iteration, 19100)
                    self.assertFalse(player.alg.optimizer.state)
                    self.assertNotIn("_extra_state", loaded["actor_state_dict"])
                    with torch.no_grad():
                        torch.testing.assert_close(player.get_inference_policy()(env.get_observations()),
                                                   original.actor(env.get_observations()), rtol=0, atol=0)

    def test_native_checkpoint_helpers_keep_native_training_resume_and_model_semantics(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            collect(source.alg, env)
            source.alg.update()
            source.current_learning_iteration = 4
            path = Path(directory) / "native.pt"
            with patch.object(Logger, "save_model"):
                source.save(str(path), infos={"test": "native"})
            restored = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            load_checkpoint(restored, path)
            self.assertEqual(restored.current_learning_iteration, 4)
            self.assertTrue(restored.alg.optimizer.state)
            agent_cfg = SimpleNamespace(obs_groups=deepcopy(cfg["obs_groups"]),
                                        actor=SimpleNamespace(**deepcopy(cfg["actor"])),
                                        critic=SimpleNamespace(**deepcopy(cfg["critic"])))
            agent_cfg.actor.distribution_cfg = {"class_name": "GaussianDistribution", "std_type": "scalar", "init_std": 1.}
            checkpoint = prepare_inference_checkpoint(agent_cfg, path)
            self.assertEqual(agent_cfg.actor.distribution_cfg, cfg["actor"]["distribution_cfg"])
            self.assertEqual(agent_cfg.obs_groups, cfg["obs_groups"])
            self.assertEqual(agent_cfg.critic.model_cfg, cfg["critic"]["model_cfg"])
            for name in ("class_name", "hidden_dims", "activation", "obs_normalization", "distribution_cfg"):
                self.assertEqual(getattr(agent_cfg.critic, name), cfg["critic"][name], name)
            checkpoint["actor_state_dict"]["_extra_state"]["action_contract"] = "old_base_frame"
            invalid = Path(directory) / "old_frame.pt"
            torch.save(checkpoint, invalid)
            with self.assertRaisesRegex(ValueError, "action contract"):
                prepare_inference_checkpoint(agent_cfg, invalid)
            with self.assertRaisesRegex(ValueError, "action contract"):
                load_checkpoint(restored, invalid)

    def test_previous_native_actor_mlp_critic_checkpoint_still_plays_without_resuming_graph_critic(self):
        legacy_cfg = runner_config()
        legacy_cfg["obs_groups"]["critic"] = ["policy"]
        legacy_cfg["critic"] = dict(class_name="MLPModel", hidden_dims=[256, 128, 64], activation="elu",
                                    obs_normalization=False, distribution_cfg=None)
        env = SmallVectorEnv(ModelCfg(**legacy_cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = OnPolicyRunner(env, deepcopy(legacy_cfg), log_dir=None, device="cpu")
            collect(source.alg, env)
            source.alg.update()
            path = Path(directory) / "native_mlp_critic.pt"
            with patch.object(Logger, "save_model"):
                source.save(str(path), infos={"test": "previous_native"})
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("_extra_state", checkpoint["actor_state_dict"])
            self.assertNotIn("_extra_state", checkpoint["critic_state_dict"])
            self.assertNotIn("research_metadata", checkpoint)
            current_cfg = runner_config()
            current = OnPolicyRunner(env, deepcopy(current_cfg), log_dir=None, device="cpu")
            before = {name: value.clone() for name, value in current.alg.actor.named_parameters()}
            with self.assertRaisesRegex(ValueError, "[Cc]ritic"):
                load_checkpoint(current, path)
            for name, value in current.alg.actor.named_parameters():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertFalse(current.alg.optimizer.state)
            agent_cfg = SimpleNamespace(obs_groups=deepcopy(current_cfg["obs_groups"]),
                                        actor=SimpleNamespace(**deepcopy(current_cfg["actor"])),
                                        critic=SimpleNamespace(**deepcopy(current_cfg["critic"])))
            loaded = prepare_inference_checkpoint(agent_cfg, path)
            self.assertEqual(agent_cfg.critic.class_name, "MLPModel")
            self.assertEqual(agent_cfg.critic.hidden_dims, [256, 128, 64])
            self.assertEqual(agent_cfg.critic.activation, "elu")
            self.assertEqual(agent_cfg.obs_groups["critic"], ["policy"])
            current_cfg["actor"], current_cfg["critic"] = vars(agent_cfg.actor), vars(agent_cfg.critic)
            current_cfg["obs_groups"] = agent_cfg.obs_groups
            player = OnPolicyRunner(env, deepcopy(current_cfg), log_dir=None, device="cpu")
            infos = load_checkpoint(player, path, inference=True, checkpoint=loaded)
            self.assertEqual(infos, {"test": "previous_native"})
            self.assertFalse(player.alg.optimizer.state)
            with torch.no_grad():
                torch.testing.assert_close(player.get_inference_policy()(env.get_observations()),
                                           source.get_inference_policy()(env.get_observations()), rtol=0, atol=0)

    def test_previous_vision_graph_critic_checkpoint_restores_policy_group_for_play_only(self):
        old_cfg = runner_config()
        old_cfg["obs_groups"]["critic"] = ["policy"]
        old_cfg["critic"]["model_cfg"] = deepcopy(old_cfg["actor"]["model_cfg"])
        old_cfg["critic"]["observation_contract"] = "initial_vision_v1"
        source, env = make_algorithm(old_cfg)
        saved = deepcopy(source.save())
        saved["iter"] = 450
        saved["infos"] = None
        contract = saved["critic_state_dict"]["_extra_state"]
        self.assertEqual(contract["architecture"], "native_graph_critic_v1")
        self.assertNotIn("observation_contract", contract)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "vision_graph_critic.pt"
            torch.save(saved, path)
            cfg = runner_config()
            current_env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]),
                                         critic_cfg=ModelCfg(**cfg["critic"]["model_cfg"]))
            current = OnPolicyRunner(current_env, deepcopy(cfg), log_dir=None, device="cpu")
            before = {name: value.clone() for name, value in current.alg.actor.named_parameters()}
            with self.assertRaisesRegex(ValueError, "[Cc]ritic"):
                load_checkpoint(current, path)
            for name, value in current.alg.actor.named_parameters():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            agent_cfg = SimpleNamespace(obs_groups=deepcopy(cfg["obs_groups"]),
                                        actor=SimpleNamespace(**deepcopy(cfg["actor"])),
                                        critic=SimpleNamespace(**deepcopy(cfg["critic"])))
            loaded = prepare_inference_checkpoint(agent_cfg, path)
            self.assertEqual(agent_cfg.obs_groups["critic"], ["policy"])
            self.assertEqual(agent_cfg.critic.observation_contract, "initial_vision_v1")
            cfg["obs_groups"] = agent_cfg.obs_groups
            cfg["actor"], cfg["critic"] = vars(agent_cfg.actor), vars(agent_cfg.critic)
            player = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            load_checkpoint(player, path, inference=True, checkpoint=loaded)
            self.assertEqual(player.current_learning_iteration, 450)
            self.assertFalse(player.alg.optimizer.state)
            inputs = env.get_observations()
            with torch.no_grad():
                torch.testing.assert_close(player.alg.actor(inputs), source.actor(inputs), rtol=0, atol=0)
                torch.testing.assert_close(player.alg.critic(inputs), source.critic(inputs), rtol=0, atol=0)

    def test_graph_critic_contract_mismatch_rejects_resume_before_actor_mutation(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            source = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            saved = deepcopy(source.alg.save())
            saved["critic_state_dict"]["_extra_state"]["model_cfg"]["graph_layers"] += 1
            path = Path(directory) / "incompatible_critic.pt"
            torch.save(saved, path)
            target = OnPolicyRunner(env, deepcopy(cfg), log_dir=None, device="cpu")
            before = {name: value.clone() for name, value in target.alg.actor.named_parameters()}
            with self.assertRaisesRegex(ValueError, "[Cc]ritic"):
                load_checkpoint(target, path)
            for name, value in target.alg.actor.named_parameters():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_native_actor_checkpoint_rejects_changed_input_or_action_contract(self):
        algorithm, _ = make_algorithm()
        original = deepcopy(algorithm.actor.state_dict())
        for field, value in (("action_contract", "old_base_frame"), ("architecture", "shared_critic_v0")):
            changed = deepcopy(original)
            changed["_extra_state"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "input/action/model"):
                algorithm.actor.load_state_dict(changed)
        algorithm.actor.load_state_dict(original)

    def test_inference_policy_uses_same_observable_input(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, deepcopy(cfg), device="cpu")
        with torch.no_grad():
            runner.alg.actor.mlp[-1].bias.fill_(3.)
        policy = runner.get_inference_policy(device="cpu")
        with torch.no_grad():
            inputs = env.get_observations()
            actions = policy(inputs)
            expected = runner.alg.actor(inputs)
        self.assertEqual(actions.shape, (2, 18))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions > 1.).all())
        torch.testing.assert_close(actions, expected, atol=0, rtol=0)

    def test_native_act_uses_each_independent_encoder_once(self):
        algorithm, env = make_algorithm()
        with torch.inference_mode():
            obs = env.get_observations()
            with patch.object(algorithm.actor.encoder, "forward", wraps=algorithm.actor.encoder.forward) as forward, \
                 patch.object(algorithm.critic.encoder, "forward", wraps=algorithm.critic.encoder.forward) as critic:
                algorithm.act(obs)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(critic.call_count, 1)
            obs["policy"].add_(.01)
            with patch.object(algorithm.actor.encoder, "forward", side_effect=AssertionError("Critic called actor")):
                values = algorithm.critic(obs)
            self.assertEqual(values.shape, (env.num_envs, 1))

    def test_critic_loss_cannot_update_actor_graph_attention_fusion_action_or_std(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        before = {name: value.detach().clone() for name, value in algorithm.actor.named_parameters()}
        algorithm.optimizer.zero_grad(set_to_none=True)
        (algorithm.critic(obs) - 100.).square().mean().backward()
        self.assertTrue(all(value.grad is None for value in algorithm.actor.parameters()))
        critic_gradients = [value.grad for value in algorithm.critic.parameters()]
        self.assertTrue(all(value is not None and torch.isfinite(value).all() for value in critic_gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in critic_gradients), 0.)
        graph = algorithm.critic.encoder.scene_encoder
        for module in (*graph.messages, graph.role_queries, graph.keys, graph.values):
            self.assertGreater(sum(float(value.grad.abs().sum()) for value in module.parameters()), 0.)
        algorithm.optimizer.step()
        for name, value in algorithm.actor.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_actor_loss_cannot_update_critic(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        before = {name: value.detach().clone() for name, value in algorithm.critic.named_parameters()}
        algorithm.optimizer.zero_grad(set_to_none=True)
        algorithm.actor(obs, stochastic_output=True)
        actions = algorithm.actor.output_mean.detach() + .1
        loss = -algorithm.actor.get_output_log_prob(actions).mean() - .01 * algorithm.actor.output_entropy.mean()
        loss.backward()
        self.assertTrue(all(value.grad is None for value in algorithm.critic.parameters()))
        self.assertGreater(sum(float(value.grad.abs().sum()) for value in algorithm.actor.parameters()
                               if value.grad is not None), 0.)
        algorithm.optimizer.step()
        for name, value in algorithm.critic.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_compilation_is_optional_on_cpu_and_does_not_register_checkpoint_wrappers(self):
        cfg = runner_config()
        cfg["actor"]["compile_model"] = True
        cfg["critic"]["compile_model"] = True
        algorithm, env = make_algorithm(cfg)
        with patch.object(torch, "compile", side_effect=AssertionError("CPU must remain eager")):
            algorithm.act(env.get_observations())
        self.assertIsNone(algorithm.actor._compiled_features)
        self.assertIsNone(algorithm.critic._compiled_features)
        self.assertTrue(all(name == "_extra_state" or name.startswith(("encoder.", "mlp.", "distribution."))
                            for name in algorithm.actor.state_dict()))
        cfg["actor"]["compile_model"] = "yes"
        with self.assertRaisesRegex(ValueError, "compile_model"):
            make_algorithm(cfg)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA Inductor model integration")
    def test_cuda_compilation_preserves_inference_training_gradients_and_checkpoint_keys(self):
        algorithm, env = make_algorithm(num_envs=4)
        actor = algorithm.actor.cuda()
        reference = deepcopy(actor)
        critic = algorithm.critic.cuda()
        actor.compile_model = True
        obs = env.get_observations().to("cuda")
        keys = list(actor.state_dict())
        with torch.inference_mode():
            expected_actions = reference(obs).clone()
            actions = actor(obs).clone()
            values = critic(obs).clone()
            torch.testing.assert_close(actions, expected_actions, rtol=1e-4, atol=2e-6)
            self.assertTrue(torch.isfinite(values).all())
            self.assertEqual(values.shape, (4, 1))
        # Native runner alternates inference rollout with gradient-enabled PPO.
        for candidate in (reference, actor):
            candidate.zero_grad(set_to_none=True)
            output = candidate(obs)
            output.square().mean().backward()
        for actual, expected in zip(actor.parameters(), reference.parameters()):
            if expected.grad is not None:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-3, atol=3e-6)
        with torch.no_grad():
            actor.mlp[-1].bias.add_(.01)
            reference.mlp[-1].bias.add_(.01)
        # Repeated inference must see parameter changes and retain prior outputs.
        with torch.inference_mode():
            torch.testing.assert_close(actor(obs), reference(obs), rtol=1e-4, atol=2e-6)
            torch.testing.assert_close(actions, expected_actions, rtol=1e-4, atol=2e-6)
        self.assertEqual(list(actor.state_dict()), keys)
        self.assertTrue(all(name == "_extra_state" or name.startswith(("encoder.", "mlp.", "distribution.")) for name in keys))

    def test_rollout_and_timeout_have_no_tensor_scalar_reads(self):
        algorithm, env = make_algorithm()
        with torch.inference_mode(), patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("tensor bool")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("tensor item")):
            obs = env.get_observations()
            algorithm.act(obs)
            final = obs["policy"].clone()
            algorithm.process_env_step(obs, torch.ones(2), torch.ones(2, dtype=torch.long), {
                "time_outs": torch.tensor([True, False]), "terminated": torch.tensor([False, True]),
                "terminal_observation": final,
            })

    def test_model_rollout_chunks_do_not_change_distribution_samples_or_critic_values(self):
        algorithm, env = make_algorithm(num_envs=7)
        obs = env.get_observations()
        with torch.no_grad():
            torch.manual_seed(45)
            full = algorithm.actor(obs, stochastic_output=True)
            full_values = algorithm.critic(obs)
            algorithm.actor.max_forward_batch_size = 3
            algorithm.critic.max_forward_batch_size = 3
            torch.manual_seed(45)
            with patch.object(algorithm.actor.encoder, "forward", wraps=algorithm.actor.encoder.forward) as forward, \
                 patch.object(algorithm.critic.encoder, "forward", wraps=algorithm.critic.encoder.forward) as critic_forward:
                chunked = algorithm.actor(obs, stochastic_output=True)
                chunked_values = algorithm.critic(obs)
        self.assertEqual([call.args[0].shape[0] for call in forward.call_args_list], [3, 3, 1])
        self.assertEqual([call.args[0].shape[0] for call in critic_forward.call_args_list], [3, 3, 1])
        torch.testing.assert_close(chunked, full, atol=2e-7, rtol=2e-6)
        torch.testing.assert_close(chunked_values, full_values, atol=2e-7, rtol=2e-6)

    def test_native_distribution_configuration_preserves_gaussian_action_contract(self):
        cfg = runner_config()
        cfg["actor"]["distribution_cfg"] = {
            "class_name": "GaussianDistribution", "std_type": "log", "init_std": .3,
        }
        cfg["actor"]["stochastic"] = False
        cfg["critic"]["distribution_cfg"] = None
        algorithm, _ = make_algorithm(cfg)
        torch.testing.assert_close(algorithm.actor.distribution.log_std_param.exp(), torch.full((18,), .3))
        cfg["actor"]["distribution_cfg"] = {
            "class_name": "GaussianDistribution", "std_type": "scalar", "init_std": .7,
        }
        scalar, _ = make_algorithm(cfg)
        self.assertEqual(scalar.actor.distribution.std_type, "scalar")
        torch.testing.assert_close(scalar.actor.distribution.std_param, torch.full((18,), .7))
        self.assertEqual(scalar.save()["actor_state_dict"]["_extra_state"]["actor_model"]["distribution_cfg"],
                         cfg["actor"]["distribution_cfg"])
        cfg["actor"]["distribution_cfg"] = {"class_name": "tanh_gaussian", "init_std": .3}
        with self.assertRaisesRegex(ValueError, "Could not resolve.*tanh_gaussian"):
            make_algorithm(cfg)

    def test_native_actor_entropy_is_analytic_without_extra_random_draws(self):
        algorithm, env = make_algorithm()
        with torch.no_grad():
            algorithm.actor.mlp[-1].bias.fill_(3.)
        obs = env.get_observations()
        mean = algorithm.actor(obs)
        algorithm.actor(obs, stochastic_output=True)
        random_state = torch.get_rng_state().clone()
        actions = algorithm.actor.output_mean
        entropy = algorithm.actor.output_entropy
        expected = (.5 * math.log(2 * math.pi * math.e) + algorithm.actor.output_std.log()).sum(-1)
        torch.testing.assert_close(entropy, expected)
        torch.testing.assert_close(torch.get_rng_state(), random_state, atol=0, rtol=0)
        torch.testing.assert_close(actions, mean, atol=0, rtol=0)
        self.assertTrue((actions > 1.).all())


if __name__ == "__main__":
    unittest.main()
