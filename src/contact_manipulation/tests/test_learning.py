"""CPU checks for research GNN/attention and native RSL-RL learning integration."""

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
    ContactManipulationPolicy, build_policy_masks,
)
from contact_manipulation.config.ur5e.agents.model import ModelCfg, _masked_softmax
from contact_manipulation.config.ur5e.agents.ppo import ACTION_CONTRACT, ContactManipulationPPO
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
        self.model = ContactManipulationPolicy(self.cfg)
        self.observation = observation(self.cfg)

    def masks(self, values=None):
        return build_policy_masks(self.observation if values is None else values, self.cfg)

    def forward(self, values):
        return self.model(values, masks=self.masks(values))

    def test_default_observation_layout_excludes_mask_channels(self):
        self.assertEqual(ModelCfg().observation_dim, 244)
        self.assertEqual(asdict(ModelCfg())["history_steps"], 2)
        self.assertEqual(asdict(ModelCfg())["activation"], "relu")
        self.assertFalse(any(isinstance(module, torch.nn.SiLU) for module in self.model.modules()))
        self.assertEqual(asdict(ModelCfg())["observation_layout"], "fixed_roles_v1")
        self.assertEqual(ModelCfg().step_features, 72)
        self.assertEqual(self.model.scene_encoder.node_encoder[0].in_features, 10)
        self.assertFalse(hasattr(self.model, "history_encoder"))
        self.assertEqual(self.model.fusion[0].in_features, 5 * self.cfg.node_dim + 144)
        with self.assertRaisesRegex(ValueError, "72 physical observation"):
            ModelCfg(step_features=89)
        with self.assertRaisesRegex(ValueError, "At least two object slots"):
            ModelCfg(max_objects=1)
        with self.assertRaisesRegex(ValueError, "observation_layout"):
            ModelCfg(observation_layout="categorical_roles")

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
        geometry, history = self.model._split(self.observation)
        geometry[:, 0, :3] = 0
        history[..., 37:54] = 0
        masks = self.masks(pack(geometry, history))
        self.assertTrue(masks.object_valid[:, 0].all())
        self.assertTrue(masks.tactile_valid[:, 0].all())
        self.assertTrue(masks.tactile_valid[:2, 1].all())
        self.assertFalse(masks.tactile_valid[-1, 1].any())

    def test_geometry_roles_and_padding_have_distinct_meanings(self):
        geometry, history = self.model._split(self.observation)
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.object_valid[:, -1].any())
        self.assertFalse(masks.roles[:, -1].any())
        self.assertTrue(masks.object_valid[:, 2].all())
        self.assertFalse(masks.roles[:, 2].any())
        self.assertTrue(masks.roles[:, 0, 0].all())
        self.assertTrue(masks.roles[:, 1, 1].all())

    def test_missing_role_slots_do_not_promote_other_observed_objects(self):
        geometry, history = self.model._split(self.observation)
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
        geometry, history = self.model._split(self.observation)
        geometry[:, 0, 7] = 0
        geometry[:, 1, 8] = -0.1
        geometry[:, 2, 3:7] = 0
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.object_valid[:, :3].any())
        self.assertFalse(masks.roles[:, :3].any())

    def test_tactile_unavailable_is_separate_from_no_contact(self):
        geometry, history = self.model._split(self.observation)
        history[:, 0, 37:42] = torch.tensor([0, 1, -1, float("nan"), 0.5])
        masks = self.masks(pack(geometry, history))
        self.assertTrue(masks.history_valid[:, 0].all())
        torch.testing.assert_close(masks.tactile_valid[:, 0, :5],
                                   torch.tensor([[True, True, False, False, False]]).expand(3, -1))

    def test_history_validity_comes_from_robot_pose_and_finite_physical_values(self):
        geometry, history = self.model._split(self.observation)
        history[0, 0, 27:31] = 0
        history[1, 0, 31] = float("nan")
        history[2, 0, 54] = float("nan")
        masks = self.masks(pack(geometry, history))
        self.assertFalse(masks.history_valid[:, 0].any())
        self.assertFalse(masks.tactile_valid[:, 0].any())

    def test_masks_must_be_passed_explicitly_before_every_policy_call(self):
        with self.assertRaises(TypeError):
            self.model(self.observation)
        with self.assertRaises(TypeError):
            self.model.act(self.observation)
        with self.assertRaises(TypeError):
            self.model.evaluate_actions(self.observation, torch.zeros(3, 18))
        self.assertFalse(hasattr(self.model, "critic"))
        self.assertFalse(hasattr(self.model, "value"))
        masks = self.masks()
        with self.assertRaisesRegex(ValueError, "Boolean tensors"):
            self.model(self.observation, masks=replace(masks, object_valid=masks.object_valid.float()))

    def test_masks_do_not_become_learned_feature_channels(self):
        geometry, history = self.model._split(self.observation)
        masks = self.masks()
        node_inputs, history_inputs = [], []
        node_hook = self.model.scene_encoder.node_encoder[0].register_forward_pre_hook(
            lambda _module, inputs: node_inputs.append(inputs[0].detach().clone()))
        history_hook = self.model.fusion[0].register_forward_pre_hook(
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
        geometry, history = self.model._split(self.observation)
        reordered = pack(geometry[:, perm], history)
        torch.testing.assert_close(self.forward(self.observation), self.forward(reordered), atol=1e-6, rtol=1e-5)

    def test_swapping_reserved_slots_changes_target_blocker_semantics(self):
        geometry, history = self.model._split(self.observation)
        swapped = pack(geometry[:, torch.tensor([1, 0, 2, 3, 4])], history)
        original_masks, swapped_masks = self.masks(), self.masks(swapped)
        torch.testing.assert_close(original_masks.roles, swapped_masks.roles)
        first, second = self.forward(self.observation), self.forward(swapped)
        self.assertGreater(float((first - second).abs().max()), 1e-6)

    def test_graph_permutation_is_invariant_when_geometry_and_selectors_move_together(self):
        geometry, _ = self.model._split(self.observation)
        masks = self.masks()
        perm = torch.tensor([4, 2, 0, 3, 1])
        first = self.model.scene_encoder(geometry, masks.object_valid, masks.roles)
        second = self.model.scene_encoder(geometry[:, perm], masks.object_valid[:, perm], masks.roles[:, perm])
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_nan_padding_is_ignored_when_masks_are_derived_from_the_poisoned_observation(self):
        geometry, history = self.model._split(self.observation)
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
        actions, log_probs = self.model.act(padded, masks=masks)
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        geometry, _ = self.model._split(padded)
        scene = self.model.scene_encoder(geometry, masks.object_valid, masks.roles)
        torch.testing.assert_close(scene, torch.zeros_like(scene))

    def test_current_previous_samples_reach_final_mlp_in_order_and_receive_gradients(self):
        inputs = self.observation.clone().requires_grad_(True)
        captured = []
        hook = self.model.fusion[0].register_forward_pre_hook(
            lambda _module, args: captured.append(args[0]))
        try:
            means = self.forward(inputs)
        finally:
            hook.remove()
        _, history = self.model._split(inputs)
        expected = torch.where(self.masks(inputs).history_valid[..., None], history, 0)
        torch.testing.assert_close(captured[0][:, -144:], expected.flatten(1))
        means.square().sum().backward()
        gradient = inputs.grad[:, -144:].reshape(-1, 2, 72)
        self.assertGreater(float(gradient[0, 0].abs().sum()), 0)
        self.assertGreater(float(gradient[0, 1].abs().sum()), 0)
        self.assertEqual(float(gradient[-1, 1].abs().sum()), 0)

    def test_reset_padding_discards_previous_episode_but_observes_current_sample(self):
        geometry, history = self.model._split(self.observation)
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
        geometry, history = self.model._split(self.observation)
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
        actions, log_probs = self.model.act(self.observation, masks=masks)
        evaluated, entropy = self.model.evaluate_actions(self.observation, actions, masks=masks)
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
        first = self.model.act(self.observation, deterministic=True, masks=masks)
        second = self.model.act(self.observation, deterministic=True, masks=masks)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        torch.testing.assert_close(first[0], mean, atol=0, rtol=0)

    def test_scene_encoder_attention_and_final_mlp_receive_policy_gradients(self):
        masks = self.masks()
        actions, _ = self.model.act(self.observation, masks=masks)
        logs, entropy = self.model.evaluate_actions(self.observation, actions.detach(), masks=masks)
        (logs.mean() + 0.01 * entropy.mean()).backward()
        for module in (self.model.scene_encoder, self.model.fusion, self.model.actor):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_analytic_entropy_preserves_objective_across_subbatches_without_sampling(self):
        actions = torch.randn(3, self.model.cfg.action_dim)
        random_state = torch.get_rng_state().clone()
        full = self.model.evaluate_actions(self.observation, actions, masks=self.masks())
        chunks = [self.model.evaluate_actions(self.observation[start:stop], actions[start:stop],
                                              masks=self.masks(self.observation[start:stop]))
                  for start, stop in ((0, 2), (2, 3))]
        torch.testing.assert_close(torch.get_rng_state(), random_state, atol=0, rtol=0)
        for column, expected in enumerate(full):
            torch.testing.assert_close(torch.cat([result[column] for result in chunks]), expected,
                                       rtol=1e-5, atol=1e-6)

    def test_large_means_remain_unrestricted_with_finite_probability_and_gradients(self):
        with torch.no_grad():
            self.model.actor.bias[::2] = 20.0
            self.model.actor.bias[1::2] = -20.0
        masks = self.masks()
        actions, logs = self.model.act(self.observation, masks=masks)
        evaluated, entropy = self.model.evaluate_actions(self.observation, actions, masks=masks)
        self.assertTrue((actions[:, ::2] > 1.0).all())
        self.assertTrue((actions[:, 1::2] < -1.0).all())
        torch.testing.assert_close(logs, evaluated, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(entropy).all())
        deterministic, _ = self.model.act(self.observation, deterministic=True, masks=masks)
        mean = self.forward(self.observation)
        torch.testing.assert_close(deterministic, mean, atol=0, rtol=0)
        (evaluated.mean() + entropy.mean()).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_native_gaussian_standard_deviation_is_positive_and_learnable(self):
        self.assertIsInstance(self.model.distribution, GaussianDistribution)
        self.assertEqual(self.model.distribution.std_type, "log")
        parameter = self.model.distribution.log_std_param
        torch.testing.assert_close(parameter.exp(), torch.full((18,), self.cfg.initial_std))
        before = parameter.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        actions = torch.zeros(3, self.cfg.action_dim)
        _, entropy = self.model.evaluate_actions(self.observation, actions, masks=self.masks())
        (-entropy.mean()).backward()
        torch.testing.assert_close(parameter.grad, -torch.ones_like(parameter))
        optimizer.step()
        self.assertTrue((parameter > before).all())
        self.assertTrue((parameter.exp() > 0).all())

    def test_arbitrary_leading_batch_dimensions(self):
        repeated = self.observation[None].expand(2, -1, -1)
        masks = self.masks(repeated)
        actions, logs = self.model.act(repeated, masks=masks)
        self.assertEqual(actions.shape, (2, 3, 18))
        self.assertEqual(logs.shape, (2, 3))
        self.assertEqual(masks.roles.shape, (2, 3, 5, 2))

    def test_single_observation_without_batch_dimension(self):
        single = self.observation[0]
        actions, log_probability = self.model.act(single, masks=self.masks(single))
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
        optimized = self.model.scene_encoder.double()
        reference = copy.deepcopy(optimized)
        geometry = self.model._split(self.observation)[0].double().requires_grad_()
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
        geometry, history = self.model._split(self.observation)
        geometry[:, 1:] = 0
        masks = self.masks(pack(geometry, history))
        optimized = self.model.scene_encoder.double()
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
        graph = self.model.scene_encoder
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
            actions, _ = self.model.act(self.observation, masks=masks)
            self.model.evaluate_actions(self.observation, actions, masks=masks)

    def test_research_history_and_action_contract_rejects_incompatible_config(self):
        for kwargs in ({"history_steps": 16}, {"action_dim": 12}, {"max_objects": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelCfg(**kwargs)


AGENTS = "contact_manipulation.config.ur5e.agents"


def runner_config(model_cfg=None, **algorithm_overrides):
    model_cfg = model_cfg or ModelCfg(max_objects=5, node_dim=16, latent_dim=24)
    algorithm = dict(
        class_name=f"{AGENTS}.ppo:ContactManipulationPPO", num_learning_epochs=2, num_mini_batches=2,
        clip_param=.2, gamma=.99, lam=.95, value_loss_coef=.25, entropy_coef=.01,
        learning_rate=3e-4, max_grad_norm=.1, use_clipped_value_loss=True,
        schedule="fixed", desired_kl=None, normalize_advantage_per_mini_batch=False,
        max_microbatch_size=32, rnd_cfg=None, symmetry_cfg=None,
    )
    algorithm.update(algorithm_overrides)
    return dict(
        class_name="OnPolicyRunner", num_steps_per_env=4, max_iterations=2, save_interval=2,
        logger="tensorboard", experiment_name="contact_manipulation", run_name="",
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
        actor=dict(class_name=f"{AGENTS}.ppo:NativeContactActor", model_cfg=asdict(model_cfg),
                   max_forward_batch_size=32),
        critic=dict(class_name="MLPModel", hidden_dims=[256, 128, 64], activation="elu",
                    obs_normalization=False, distribution_cfg=None),
        algorithm=algorithm, multi_gpu=None,
    )


class SmallVectorEnv:
    """A deterministic vector API fixture with real observations and auto-resets."""

    def __init__(self, model_cfg, num_envs=2):
        self.num_envs = num_envs
        self.num_actions = model_cfg.action_dim
        self.device = "cpu"
        self.cfg = SimpleNamespace()
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
        algorithm = ContactManipulationPPO.construct_algorithm(env.get_observations(), env, deepcopy(cfg), "cpu")
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
            self.assertIs(ContactManipulationPPO.act, PPO.act)
            self.assertIs(ContactManipulationPPO.compute_returns, PPO.compute_returns)
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

    def test_independent_native_critic_has_separate_parameters_and_checkpoint(self):
        algorithm, _ = make_algorithm()
        self.assertIs(type(algorithm.critic), MLPModel)
        self.assertFalse(hasattr(algorithm.actor.policy, "critic"))
        self.assertFalse(hasattr(algorithm.actor, "consume_cached_value"))
        self.assertIsInstance(algorithm.actor.policy.distribution, GaussianDistribution)
        self.assertEqual(sum(isinstance(module, GaussianDistribution)
                             for module in algorithm.actor.modules()), 1)
        actor_parameters = list(algorithm.actor.parameters())
        critic_parameters = list(algorithm.critic.parameters())
        self.assertTrue(critic_parameters)
        self.assertTrue({id(value) for value in actor_parameters}.isdisjoint(
            id(value) for value in critic_parameters))
        expected = actor_parameters + critic_parameters
        actual = algorithm.optimizer.param_groups[0]["params"]
        self.assertEqual([id(value) for value in actual], [id(value) for value in expected])
        self.assertEqual(len(actual), len({id(value) for value in actual}))
        saved = algorithm.save()
        self.assertEqual(saved["research_metadata"]["force_torque"], {"name": "ATI Axia80 SI-200-8", "ranges": SI_200_8_RANGES, "resolution": SI_200_8_RESOLUTION})
        self.assertEqual(saved["research_metadata"]["action_contract"], ACTION_CONTRACT)
        self.assertEqual(saved["research_metadata"]["output_distribution"], {
            "class_name": "GaussianDistribution", "std_type": "log", "init_std": .3,
        })
        self.assertFalse(any("critic" in name for name in saved["actor_state_dict"]))
        self.assertIn("policy.distribution.log_std_param", saved["actor_state_dict"])
        self.assertNotIn("policy.log_std", saved["actor_state_dict"])
        self.assertTrue(saved["critic_state_dict"])
        self.assertEqual(saved["research_metadata"]["critic_model"], {
            "class_name": "MLPModel", "hidden_dims": [256, 128, 64], "activation": "elu",
            "obs_normalization": False, "distribution_cfg": None,
        })

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
        with patch.object(algorithm.critic, "forward", return_value=torch.tensor([[4.], [400.]])) as critic:
            algorithm.process_env_step(obs, rewards, torch.ones(2, dtype=torch.long), extras)
        self.assertEqual(critic.call_args.args[0].batch_size, torch.Size([2]))
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
                                   {"time_outs": torch.tensor([False, False]), "terminal_observation": obs["policy"],
                                    "terminal_env_ids": torch.empty(0, dtype=torch.long)})
        self.assertEqual(algorithm.storage.step, 1)

    def paired_algorithms(self, **overrides):
        cfg = runner_config(**overrides)
        first, env = make_algorithm(cfg, num_envs=3)
        second, _ = make_algorithm(cfg, num_envs=3)
        second.actor.policy.load_state_dict(first.actor.policy.state_dict())
        second.critic.load_state_dict(first.critic.state_dict())
        collect(first, env)
        second.storage = deepcopy(first.storage)
        return first, second

    def test_microbatches_preserve_adam_steps_losses_and_actor_critic_gradients(self):
        full, split = self.paired_algorithms(max_microbatch_size=None)
        split.max_microbatch_size = 4
        torch.manual_seed(92)
        with patch.object(full.optimizer, "step", wraps=full.optimizer.step) as full_steps:
            full_metrics = full._update_microbatches()
        torch.manual_seed(92)
        with patch.object(split.optimizer, "step", wraps=split.optimizer.step) as split_steps, \
             patch.object(split.actor, "forward", wraps=split.actor.forward) as forwards:
            split_metrics = split.update()
        self.assertEqual(full_steps.call_count, 4)
        self.assertEqual(split_steps.call_count, 4)
        self.assertEqual([call.args[0].batch_size[0] for call in forwards.call_args_list], [4, 2] * 4)
        full_parameters = list(full.actor.parameters()) + list(full.critic.parameters())
        split_parameters = list(split.actor.parameters()) + list(split.critic.parameters())
        for first, second in zip(full_parameters, split_parameters):
            torch.testing.assert_close(first, second, rtol=4e-5, atol=3e-6)
            torch.testing.assert_close(first.grad, second.grad, rtol=4e-4, atol=3e-7)
        for name in full_metrics:
            self.assertAlmostEqual(full_metrics[name], split_metrics[name], places=5)
        for first, second in zip(full.optimizer.state.values(), split.optimizer.state.values()):
            self.assertEqual(first["step"], second["step"])
            torch.testing.assert_close(first["exp_avg"], second["exp_avg"], rtol=5e-4, atol=2e-8)
            torch.testing.assert_close(first["exp_avg_sq"], second["exp_avg_sq"], rtol=1e-3, atol=2e-10)

    def test_native_update_is_used_when_minibatch_fits_memory_limit(self):
        for limit, native_expected in ((None, True), (6, True), (5, False)):
            algorithm, _ = self.paired_algorithms(max_microbatch_size=limit)
            with self.subTest(limit=limit), patch.object(PPO, "update", return_value={"value": 1.}) as native, \
                 patch.object(algorithm, "_update_microbatches", return_value={"value": 2.}) as split:
                result = algorithm.update()
                self.assertEqual(native.call_count, int(native_expected))
                self.assertEqual(split.call_count, int(not native_expected))
                self.assertEqual(result["value"], 1. if native_expected else 2.)

    def test_unsplit_independent_losses_match_native_ppo_update(self):
        extension, native = self.paired_algorithms(max_microbatch_size=None, entropy_coef=.01)
        torch.manual_seed(16)
        extension_metrics = extension._update_microbatches()
        torch.manual_seed(16)
        native_metrics = PPO.update(native)
        for name in ("value", "surrogate"):
            self.assertAlmostEqual(extension_metrics[name], native_metrics[name], places=5)
        extension_parameters = list(extension.actor.parameters()) + list(extension.critic.parameters())
        native_parameters = list(native.actor.parameters()) + list(native.critic.parameters())
        for first, second in zip(extension_parameters, native_parameters):
            torch.testing.assert_close(first, second, rtol=4e-5, atol=3e-6)

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
            self.assertIs(type(restored.alg.critic), MLPModel)
            for first, second in zip(runner.alg.actor.policy.parameters(), restored.alg.actor.policy.parameters()):
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

    def test_checkpoint_semantics_are_checked_before_loading_weights(self):
        algorithm, _ = make_algorithm()
        variants = []
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

    def test_previous_action_checkpoints_are_rejected_before_loading_weights(self):
        algorithm, _ = make_algorithm()
        self.assertEqual(ACTION_CONTRACT, "isaaclab_ur5e_wrist_osc_variable_kp_gravity_v3")
        variants = []
        unversioned = algorithm.save()
        del unversioned["research_metadata"]
        variants.append(unversioned)
        legacy = algorithm.save()
        del legacy["research_metadata"]["action_contract"]
        variants.append(legacy)
        custom = algorithm.save()
        custom["research_metadata"]["action_contract"] = "custom_osc_and_bounded_hand_v1"
        variants.append(custom)
        base_frame = algorithm.save()
        base_frame["research_metadata"]["action_contract"] = "isaaclab_min_osc_variable_kp_joint_position_to_limits_v1"
        variants.append(base_frame)
        palm_frame = algorithm.save()
        palm_frame["research_metadata"]["action_contract"] = "isaaclab_eef_osc_variable_kp_joint_position_to_limits_v2"
        variants.append(palm_frame)
        before = {name: value.clone() for name, value in algorithm.actor.state_dict().items()}
        with patch.object(algorithm.actor, "load_state_dict") as actor_load, \
             patch.object(algorithm.critic, "load_state_dict") as critic_load, \
             patch.object(algorithm.optimizer, "load_state_dict") as optimizer_load:
            for checkpoint in variants:
                with self.subTest(metadata=checkpoint.get("research_metadata")), \
                     self.assertRaisesRegex(ValueError, "action contract.*native Isaac Lab OSC"):
                    algorithm.load(checkpoint)
            actor_load.assert_not_called()
            critic_load.assert_not_called()
            optimizer_load.assert_not_called()
        for name, value in algorithm.actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_previous_output_distributions_are_rejected_before_loading_any_state(self):
        algorithm, _ = make_algorithm()
        variants = []
        missing = algorithm.save()
        del missing["research_metadata"]["output_distribution"]
        variants.append(missing)
        squashed = algorithm.save()
        squashed["research_metadata"]["output_distribution"] = {
            "class_name": "tanh_gaussian", "init_std": .3,
        }
        variants.append(squashed)
        scalar = algorithm.save()
        scalar["research_metadata"]["output_distribution"]["std_type"] = "scalar"
        variants.append(scalar)
        before = {name: value.clone() for name, value in algorithm.actor.state_dict().items()}
        with patch.object(algorithm.actor, "load_state_dict") as actor_load, \
             patch.object(algorithm.critic, "load_state_dict") as critic_load, \
             patch.object(algorithm.optimizer, "load_state_dict") as optimizer_load:
            for checkpoint in variants:
                with self.subTest(metadata=checkpoint["research_metadata"]), \
                     self.assertRaisesRegex(ValueError, "[Oo]utput distribution.*Gaussian"):
                    algorithm.load(checkpoint)
            actor_load.assert_not_called()
            critic_load.assert_not_called()
            optimizer_load.assert_not_called()
        for name, value in algorithm.actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_shared_critic_checkpoints_are_rejected_before_loading_any_state(self):
        algorithm, _ = make_algorithm()
        shared = algorithm.save()
        shared["research_metadata"].pop("critic_model", None)
        shared["research_metadata"].pop("actor_critic_architecture", None)
        shared["critic_state_dict"] = {}
        variants = [shared]
        wrong_critic = algorithm.save()
        wrong_critic["research_metadata"]["critic_model"]["hidden_dims"] = [64, 64]
        variants.append(wrong_critic)
        with patch.object(algorithm.actor, "load_state_dict") as actor_load, \
             patch.object(algorithm.critic, "load_state_dict") as critic_load, \
             patch.object(algorithm.optimizer, "load_state_dict") as optimizer_load:
            for checkpoint in variants:
                with self.subTest(metadata=checkpoint["research_metadata"]), \
                     self.assertRaisesRegex(ValueError, "[Cc]ritic architecture"):
                    algorithm.load(checkpoint)
            actor_load.assert_not_called()
            critic_load.assert_not_called()
            optimizer_load.assert_not_called()

    def test_inference_policy_uses_same_observable_input(self):
        cfg = runner_config()
        env = SmallVectorEnv(ModelCfg(**cfg["actor"]["model_cfg"]))
        with redirect_stdout(io.StringIO()):
            runner = OnPolicyRunner(env, deepcopy(cfg), device="cpu")
        with torch.no_grad():
            runner.alg.actor.policy.actor.bias.fill_(3.)
        policy = runner.get_inference_policy(device="cpu")
        with torch.no_grad():
            inputs = env.get_observations()
            actions = policy(inputs)
            expected = runner.alg.actor.policy(inputs["policy"], masks=build_policy_masks(
                inputs["policy"], runner.alg.actor.policy.cfg))
        self.assertEqual(actions.shape, (2, 18))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue((actions > 1.).all())
        torch.testing.assert_close(actions, expected, atol=0, rtol=0)

    def test_native_act_uses_actor_encoder_once_and_critic_never_calls_it(self):
        algorithm, env = make_algorithm()
        with torch.inference_mode():
            obs = env.get_observations()
            with patch.object(algorithm.actor.policy, "forward", wraps=algorithm.actor.policy.forward) as forward, \
                 patch.object(algorithm.critic, "forward", wraps=algorithm.critic.forward) as critic:
                algorithm.act(obs)
            self.assertEqual(forward.call_count, 1)
            self.assertEqual(critic.call_count, 1)
            obs["policy"].add_(.01)
            with patch.object(algorithm.actor.policy, "forward", side_effect=AssertionError("Critic called actor")):
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
        algorithm.optimizer.step()
        for name, value in algorithm.actor.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_actor_loss_cannot_update_critic(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        before = {name: value.detach().clone() for name, value in algorithm.critic.named_parameters()}
        algorithm.optimizer.zero_grad(set_to_none=True)
        algorithm.actor(obs)
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
        algorithm, env = make_algorithm(cfg)
        with patch.object(torch, "compile", side_effect=AssertionError("CPU must remain eager")):
            algorithm.act(env.get_observations())
        self.assertIsNone(algorithm.actor._compiled_forward)
        self.assertTrue(all(name.startswith("policy.") for name in algorithm.actor.state_dict()))
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
            actor.policy.actor.bias.add_(.01)
            reference.policy.actor.bias.add_(.01)
        # Repeated inference must see parameter changes and retain prior outputs.
        with torch.inference_mode():
            torch.testing.assert_close(actor(obs), reference(obs), rtol=1e-4, atol=2e-6)
            torch.testing.assert_close(actions, expected_actions, rtol=1e-4, atol=2e-6)
        self.assertEqual(list(actor.state_dict()), keys)
        self.assertTrue(all(name.startswith("policy.") for name in keys))

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

    def test_timeout_uses_native_reset_indices_and_skips_empty_batches(self):
        algorithm, env = make_algorithm(num_envs=4)
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7., 1., 3.])
        extras = dict(time_outs=torch.tensor([False, True, False, True]),
                      terminated=torch.tensor([False, False, False, True]),
                      terminal_observation=obs["policy"].clone(), terminal_env_ids=torch.tensor([1, 3]))
        with patch.object(algorithm.critic, "forward", return_value=torch.tensor([[4.], [400.]])) as critic:
            algorithm.process_env_step(obs, torch.ones(4), torch.tensor([0, 1, 0, 1]), extras)
        torch.testing.assert_close(critic.call_args.args[0]["policy"], extras["terminal_observation"][[1, 3]])
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([1., 4.96, 1., 1.]))
        self.transition(algorithm, obs, [2., 7., 1., 3.])
        extras["terminal_env_ids"] = torch.empty(0, dtype=torch.long)
        extras["time_outs"] = torch.zeros(4, dtype=torch.bool)
        with patch.object(algorithm.critic, "forward") as critic:
            algorithm.process_env_step(obs, torch.ones(4), torch.zeros(4, dtype=torch.long), extras)
        critic.assert_not_called()

    def test_native_round_without_possible_timeouts_skips_terminal_critic_without_scalar_reads(self):
        algorithm, env = make_algorithm(num_envs=4)
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7., 1., 3.])
        rewards = torch.tensor([1., 2., 3., 4.])
        extras = dict(time_outs=torch.zeros(4, dtype=torch.bool),
                      time_outs_possible=False, terminated=torch.tensor([False, True, False, True]),
                      terminal_observation=obs["policy"].clone(), terminal_env_ids=torch.tensor([1, 3]))
        with patch.object(algorithm.critic, "forward") as critic, \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("tensor bool")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("tensor item")):
            algorithm.process_env_step(obs, rewards, torch.tensor([0, 1, 0, 1]), extras)
        critic.assert_not_called()
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], rewards)
        self.assertIs(extras["time_outs_possible"], False)
        self.assertIn("time_outs", extras)

    def test_native_round_boundary_keeps_timeout_final_state_bootstrap(self):
        algorithm, env = make_algorithm()
        obs = env.get_observations()
        self.transition(algorithm, obs, [2., 7.])
        extras = dict(time_outs=torch.tensor([True, True]), time_outs_possible=True,
                      terminated=torch.tensor([False, True]), terminal_observation=obs["policy"].clone(),
                      terminal_env_ids=torch.tensor([0, 1]))
        with patch.object(algorithm.critic, "forward", return_value=torch.tensor([[4.], [400.]])) as critic:
            algorithm.process_env_step(obs, torch.ones(2), torch.ones(2, dtype=torch.long), extras)
        critic.assert_called_once()
        torch.testing.assert_close(algorithm.storage.rewards[0, :, 0], torch.tensor([4.96, 1.]))

    def test_timeout_host_hint_tracks_scene_boundary_and_external_fallback(self):
        from contact_manipulation.mdp.terminations import scene_time_out

        env = SimpleNamespace(episode_length_buf=torch.tensor([15, 399]), max_episode_length=400,
                              common_step_counter=399, extras={}, _episode_reset=SimpleNamespace(
                                  scene_start_step=0, next_scene_step=400, shared_template=object()))
        with patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("tensor bool")), \
             patch.object(torch.Tensor, "item", side_effect=AssertionError("tensor item")):
            before = scene_time_out(env)
            self.assertIs(env.extras["time_outs_possible"], False)
            env.common_step_counter = 400
            boundary = scene_time_out(env)
            self.assertIs(env.extras["time_outs_possible"], True)
            # Reset replaces the deadline but must not change this step's hint.
            env._episode_reset.scene_start_step = 400
            env._episode_reset.next_scene_step = 800
            self.assertIs(env.extras["time_outs_possible"], True)
            env.common_step_counter = 401
            env.episode_length_buf.zero_()
            after = scene_time_out(env)
            self.assertIs(env.extras["time_outs_possible"], False)
            del env._episode_reset.scene_start_step
            env.episode_length_buf[1] = 400
            fallback = scene_time_out(env)
            self.assertIs(env.extras["time_outs_possible"], True)
        torch.testing.assert_close(before, torch.tensor([False, False]))
        torch.testing.assert_close(boundary, torch.tensor([True, True]))
        torch.testing.assert_close(after, torch.tensor([False, False]))
        torch.testing.assert_close(fallback, torch.tensor([False, True]))

    def test_actor_rollout_chunks_do_not_change_distribution_or_random_samples(self):
        algorithm, env = make_algorithm(num_envs=7)
        obs = env.get_observations()
        with torch.no_grad():
            torch.manual_seed(45)
            full = algorithm.actor(obs, stochastic_output=True)
            full_values = algorithm.critic(obs)
            algorithm.actor.max_forward_batch_size = 3
            torch.manual_seed(45)
            with patch.object(algorithm.actor.policy, "forward", wraps=algorithm.actor.policy.forward) as forward:
                chunked = algorithm.actor(obs, stochastic_output=True)
            chunked_values = algorithm.critic(obs)
        self.assertEqual([call.args[0].shape[0] for call in forward.call_args_list], [3, 3, 1])
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
        torch.testing.assert_close(algorithm.actor.policy.distribution.log_std_param.exp(), torch.full((18,), .3))
        cfg["actor"]["distribution_cfg"] = {"class_name": "tanh_gaussian", "init_std": .3}
        with self.assertRaisesRegex(ValueError, "GaussianDistribution"):
            make_algorithm(cfg)

    def test_native_actor_entropy_is_analytic_without_extra_random_draws(self):
        algorithm, env = make_algorithm()
        with torch.no_grad():
            algorithm.actor.policy.actor.bias.fill_(3.)
        obs = env.get_observations()
        actions = algorithm.actor(obs)
        random_state = torch.get_rng_state().clone()
        entropy = algorithm.actor.output_entropy
        expected = (.5 * math.log(2 * math.pi * math.e) + algorithm.actor.output_std.log()).sum(-1)
        torch.testing.assert_close(entropy, expected)
        torch.testing.assert_close(torch.get_rng_state(), random_state, atol=0, rtol=0)
        torch.testing.assert_close(actions, algorithm.actor.output_mean, atol=0, rtol=0)
        self.assertTrue((actions > 1.).all())

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

    def test_minibatch_normalization_rejects_single_sample_variance(self):
        cfg = runner_config(num_mini_batches=8, normalize_advantage_per_mini_batch=True)
        with self.assertRaisesRegex(ValueError, "at least two samples per minibatch"):
            make_algorithm(cfg)
        cfg["algorithm"]["num_mini_batches"] = 4
        algorithm, env = make_algorithm(cfg)
        collect(algorithm, env)
        self.assertTrue(all(math.isfinite(value) for value in algorithm.update().values()))


if __name__ == "__main__":
    unittest.main()
