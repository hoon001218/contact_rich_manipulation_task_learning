"""CPU checks for observation-derived masks and the trainable policy."""

from dataclasses import asdict, replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.config.ur5e.agents.model import (
    ContactRearrangementPolicy, build_policy_masks,
)
from contact_rearrangement.config.ur5e.agents.ppo_cfg import ModelCfg


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


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(81)
        self.cfg = ModelCfg(max_objects=5, node_dim=16, temporal_dim=16, latent_dim=24)
        self.model = ContactRearrangementPolicy(self.cfg)
        self.observation = observation(self.cfg)

    def masks(self, values=None):
        return build_policy_masks(self.observation if values is None else values, self.cfg)

    def forward(self, values):
        return self.model(values, masks=self.masks(values))

    def test_default_observation_layout_excludes_mask_channels(self):
        self.assertEqual(ModelCfg().observation_dim, 544)
        self.assertEqual(asdict(ModelCfg())["history_steps"], 2)
        self.assertEqual(asdict(ModelCfg())["observation_layout"], "fixed_roles_v1")
        self.assertEqual(ModelCfg().step_features, 72)
        self.assertEqual(self.model.scene_encoder.node_encoder[0].in_features, 10)
        self.assertEqual(self.model.history_encoder.project.in_channels, 72)
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
            self.model.value(self.observation)
        with self.assertRaises(TypeError):
            self.model.evaluate_actions(self.observation, torch.zeros(3, 18))
        masks = self.masks()
        with self.assertRaisesRegex(ValueError, "Boolean tensors"):
            self.model(self.observation, masks=replace(masks, object_valid=masks.object_valid.float()))

    def test_masks_do_not_become_learned_feature_channels(self):
        geometry, history = self.model._split(self.observation)
        masks = self.masks()
        node_inputs, history_inputs = [], []
        node_hook = self.model.scene_encoder.node_encoder[0].register_forward_pre_hook(
            lambda _module, inputs: node_inputs.append(inputs[0].detach().clone()))
        history_hook = self.model.history_encoder.project.register_forward_pre_hook(
            lambda _module, inputs: history_inputs.append(inputs[0].detach().clone()))
        try:
            self.forward(self.observation)
        finally:
            node_hook.remove()
            history_hook.remove()
        expected_geometry = torch.where(masks.object_valid[..., None], geometry, 0)
        expected_history = torch.where(masks.history_valid[..., None], history, 0)
        torch.testing.assert_close(node_inputs[0], expected_geometry)
        torch.testing.assert_close(history_inputs[0], expected_history.flip(1).transpose(1, 2))

    def test_distractor_slot_permutation_preserves_policy_and_value(self):
        perm = torch.tensor([0, 1, 4, 2, 3])
        geometry, history = self.model._split(self.observation)
        reordered = pack(geometry[:, perm], history)
        for first, second in zip(self.forward(self.observation), self.forward(reordered)):
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)

    def test_swapping_reserved_slots_changes_target_blocker_semantics(self):
        geometry, history = self.model._split(self.observation)
        swapped = pack(geometry[:, torch.tensor([1, 0, 2, 3, 4])], history)
        original_masks, swapped_masks = self.masks(), self.masks(swapped)
        torch.testing.assert_close(original_masks.roles, swapped_masks.roles)
        first, second = self.forward(self.observation), self.forward(swapped)
        self.assertGreater(float((first[1] - second[1]).abs().max()), 1e-6)

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
        for first, second in zip(self.forward(self.observation), self.forward(poisoned)):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertTrue(torch.isfinite(second).all())

    def test_all_padding_is_finite_without_inventing_observed_objects(self):
        padded = torch.full_like(self.observation, float("nan"))
        masks = self.masks(padded)
        actions, log_probs, values = self.model.act(padded, masks=masks)
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(values).all())
        geometry, _ = self.model._split(padded)
        scene = self.model.scene_encoder(geometry, masks.object_valid, masks.roles)
        torch.testing.assert_close(scene, torch.zeros_like(scene))

    def test_current_previous_order_matters_and_previous_reaches_newest_representation(self):
        _, history = self.model._split(self.observation)
        valid = torch.ones(history.shape[:2], dtype=torch.bool)
        tactile_valid = torch.ones((*history.shape[:2], 17), dtype=torch.bool)
        history = history.clone()
        self.assertEqual(history.shape[1], 2)
        baseline = self.model.history_encoder(history, valid, tactile_valid)
        reversed_history = self.model.history_encoder(history.flip(1), valid, tactile_valid)
        self.assertGreater(float((baseline - reversed_history).abs().max()), 1e-4)
        changed = history.clone()
        changed[:, -1] += 10.0
        previous_changed = self.model.history_encoder(changed, valid, tactile_valid)
        width = self.cfg.temporal_dim
        self.assertGreater(float((baseline[:, :width] - previous_changed[:, :width]).abs().max()), 1e-5)
        changed = history.clone()
        changed[:, 0] += 10.0
        current_changed = self.model.history_encoder(changed, valid, tactile_valid)
        self.assertGreater(float((baseline[:, :width] - current_changed[:, :width]).abs().max()), 1e-5)

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
        for first, second in zip(expected, actual):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
        history[:, 0, :12] += 10.0
        changed = self.forward(pack(geometry, history))
        self.assertGreater(float((expected[1] - changed[1]).abs().max()), 1e-5)

    def test_invalid_tactile_values_are_zeroed_before_projection(self):
        geometry, history = self.model._split(self.observation)
        history[:, 0, 37] = -1
        expected = self.forward(pack(geometry, history))
        history[:, 0, 37] = float("nan")
        actual = self.forward(pack(geometry, history))
        history[:, 0, 37] = 0
        no_contact = self.forward(pack(geometry, history))
        for first, second, third in zip(expected, actual, no_contact):
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            torch.testing.assert_close(first, third, atol=0, rtol=0)

    def test_action_bounds_and_tanh_log_probability_agree(self):
        masks = self.masks()
        actions, log_probs, values = self.model.act(self.observation, masks=masks)
        evaluated, entropy, evaluated_values = self.model.evaluate_actions(self.observation, actions, masks=masks)
        self.assertTrue((actions.abs() <= 1).all())
        torch.testing.assert_close(log_probs, evaluated, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(values, evaluated_values)
        self.assertTrue(torch.isfinite(entropy).all())
        first = self.model.act(self.observation, deterministic=True, masks=masks)
        second = self.model.act(self.observation, deterministic=True, masks=masks)
        torch.testing.assert_close(first[0], second[0], atol=0, rtol=0)
        near_limits = torch.full_like(actions, 1 - 1e-6)
        near_limits[:, ::2] *= -1
        self.assertTrue(torch.isfinite(self.model.evaluate_actions(self.observation, near_limits, masks=masks)[0]).all())

    def test_both_encoders_receive_policy_and_value_gradients(self):
        masks = self.masks()
        actions, _, _ = self.model.act(self.observation, masks=masks)
        logs, entropy, values = self.model.evaluate_actions(self.observation, actions.detach(), masks=masks)
        (logs.mean() + 0.01 * entropy.mean() + values.square().mean()).backward()
        for module in (self.model.scene_encoder, self.model.history_encoder, self.model.actor, self.model.critic):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)

    def test_saturated_actions_preserve_probability_ratios_and_finite_gradients(self):
        with torch.no_grad():
            self.model.actor.bias[::2] = 20.0
            self.model.actor.bias[1::2] = -20.0
        masks = self.masks()
        actions, logs, _ = self.model.act(self.observation, masks=masks)
        evaluated, entropy, values = self.model.evaluate_actions(self.observation, actions, masks=masks)
        self.assertTrue((actions.abs() < 1.0).all())
        torch.testing.assert_close(logs, evaluated, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(entropy).all())
        (evaluated.mean() + entropy.mean() + values.mean()).backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                            for parameter in self.model.parameters()))

    def test_arbitrary_leading_batch_dimensions(self):
        repeated = self.observation[None].expand(2, -1, -1)
        masks = self.masks(repeated)
        actions, logs, values = self.model.act(repeated, masks=masks)
        self.assertEqual(actions.shape, (2, 3, 18))
        self.assertEqual(logs.shape, (2, 3))
        self.assertEqual(values.shape, (2, 3))
        self.assertEqual(masks.roles.shape, (2, 3, 5, 2))

    def test_single_observation_without_batch_dimension(self):
        single = self.observation[0]
        actions, log_probability, value = self.model.act(single, masks=self.masks(single))
        self.assertEqual(actions.shape, (18,))
        self.assertEqual(log_probability.shape, ())
        self.assertEqual(value.shape, ())


if __name__ == "__main__":
    unittest.main()
