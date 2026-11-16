"""CPU checks for independent experiment evaluation geometry; no simulator."""

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_mdp as fixtures

task, rewards = fixtures.task, fixtures.rewards


class ExperimentRelocationRegionTests(unittest.TestCase):
    def simple_scene(self):
        pockets = ((-.75, -.55, -.5, -.2), (-.75, -.55, .2, .5))
        corridors = ((-.55, 0., -.5, -.2), (-.55, 0., .2, .5))
        sample = task.SceneSample((0, 1), 0, 1, {}, {}, (-.8, 0., -.1, .1), (0., 0.),
            reserved_regions=pockets, parameter_certificate={
                'sequential_state': {'pocket_front_corridors': list(corridors)}})
        rules = task.SpawnRules(generation_mode='sequential', x_range=(-1., 0.))
        return sample, rules, pockets, corridors

    def test_corridors_are_separate_polygons_and_do_not_bridge_the_access_region(self):
        sample, rules, pockets, corridors = self.simple_scene()
        vertices, counts, valid = task.build_initial_relocation_regions(
            sample, (), rules, device='cpu', dtype=torch.float64, include_front_corridors=True)
        expected = torch.tensor([task.rectangle_polygon(region) for region in (*pockets, *corridors)],
                                dtype=torch.float64)
        torch.testing.assert_close(vertices, expected, atol=0., rtol=0.)
        torch.testing.assert_close(counts, torch.full((4,), 4))
        self.assertTrue(valid.all())
        points = vertices.new_tensor([[-.25, -.3], [-.25, .3], [-.25, 0.]])
        distance = rewards.relocation_distance(points, vertices[None].expand(3, -1, -1, -1),
            counts[None].expand(3, -1), valid[None].expand(3, -1))
        torch.testing.assert_close(distance, points.new_tensor([0., 0., .2]))
        # Historical comparisons can retain destination-only distance with corridor metadata.
        baseline = task.build_initial_relocation_regions(sample, (), rules, device='cpu', dtype=torch.float64)
        self.assertEqual(baseline[0].shape, (2, 4, 2))
        baseline_distance = rewards.relocation_distance(points[:1], baseline[0][None],
                                                       baseline[1][None], baseline[2][None])
        torch.testing.assert_close(baseline_distance, points.new_tensor([.3]))

    def test_actual_generator_metadata_is_used_without_mutating_the_sample(self):
        fixtures.SequentialSceneGenerationTests.setUpClass()
        source = fixtures.SequentialSceneGenerationTests()
        for seed in (7, 11, 18):
            with self.subTest(seed=seed):
                sample = source.sample(seed)
                saved = (sample.access_region, sample.reserved_regions, dict(sample.positions), dict(sample.yaws))
                corridors = sample.parameter_certificate['sequential_state']['pocket_front_corridors']
                vertices, counts, valid = task.build_initial_relocation_regions(sample,
                    source.geometries, source.rules, device='cpu', dtype=torch.float64,
                    include_front_corridors=True)
                expected = [task.rectangle_polygon(region) for region in (*sample.reserved_regions, *corridors)]
                np.testing.assert_allclose(vertices.numpy(), expected, rtol=0., atol=0.)
                self.assertTrue(valid.all())
                self.assertTrue((counts == 4).all())
                self.assertEqual(saved, (sample.access_region, sample.reserved_regions,
                                         dict(sample.positions), dict(sample.yaws)))

    def test_missing_or_wrong_reserved_corridors_reject_instead_of_inventing_empty_space(self):
        sample, rules, _, corridors = self.simple_scene()
        invalid_metadata = (None, {}, {'sequential_state': {}},
            {'sequential_state': {'pocket_front_corridors': [corridors[0]]}},
            {'sequential_state': {'pocket_front_corridors': [(-.6, 0., -.5, -.2), corridors[1]]}})
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                task.build_initial_relocation_regions(replace(sample, parameter_certificate=metadata),
                    (), rules, device='cpu', dtype=torch.float64, include_front_corridors=True)
        with self.assertRaisesRegex(ValueError, 'sequential'):
            task.build_initial_relocation_regions(sample, (), replace(rules, generation_mode='clutter'),
                device='cpu', dtype=torch.float64, include_front_corridors=True)

    def test_relocation_flag_reaches_region_builder_without_extending_access_geometry(self):
        env, template, sample = fixtures.EpisodeResetTests.shared_fixture()
        env.cfg.relocation_include_front_corridors = True
        relocation = (torch.tensor([task.rectangle_polygon((-.8, -.5, -.4, -.2))]),
                      torch.tensor([4]), torch.tensor([True]))
        saved = template.access_region.clone()
        with patch.object(task, '_build_episode_batch', return_value=(template, sample)), \
             patch.object(task, '_sample_robot_starts', side_effect=fixtures.EpisodeResetTests.shared_starts), \
             patch.object(task, 'build_initial_relocation_regions', return_value=relocation) as regions:
            task.reset_scene(env)
            shared = env.task_scene_data.relocation_regions
            env.common_step_counter = 17
            task.reset_scene(env, torch.tensor([4, 1]))
        self.assertEqual(regions.call_count, 1)
        self.assertTrue(regions.call_args.kwargs['include_front_corridors'])
        self.assertIs(env.task_scene_data.relocation_regions, shared)
        torch.testing.assert_close(env.task_scene_data.access_region, saved.expand(5, -1))
        torch.testing.assert_close(template.access_region, saved)


class ExperimentBackWallTests(unittest.TestCase):
    def test_extension_changes_only_rear_limit_and_preserves_baseline_identity(self):
        access = torch.tensor([[-.65, -.5, -.1, .1], [-.72, -.51, .05, .15]], dtype=torch.float64)
        bounds = torch.tensor([[-.9, -.5, -.4, .4], [-.91, -.51, -.4, .4]], dtype=torch.float64)
        saved_access, saved_bounds = access.clone(), bounds.clone()
        self.assertIs(task.evaluation_access_region(access, bounds), access)
        expanded = task.evaluation_access_region(access, bounds, to_back_wall=True)
        expected = access.clone()
        expected[:, 0] = bounds[:, 0]
        torch.testing.assert_close(expanded, expected)
        torch.testing.assert_close(access, saved_access)
        torch.testing.assert_close(bounds, saved_bounds)
        with self.assertRaises(ValueError):
            task.evaluation_access_region(access, bounds[:1], to_back_wall=True)

    def test_backward_motion_retains_overlap_and_cannot_claim_clearance_or_success(self):
        _, batch, reset = fixtures.fixture(n=2)
        reset['active'].zero_()
        reset['active'][:, :2] = True
        reset['geometry_valid'] = reset['active'].clone()
        batch.center.zero_()
        batch.center[..., 2] = .02
        batch.center[:, 0, 0], batch.center[:, 1, 0] = -.72, -.58
        signs = batch.corners[0, 0] - batch.corners[0, 0].mean(0)
        signs = signs.sign() * .02
        batch.corners[:] = batch.center[..., None, :] + signs
        reset['workspace_bounds'] = batch.center.new_tensor([-.9, -.5, -.4, .4]).expand(2, -1)
        original_access = batch.center.new_tensor([-.65, -.5, -.1, .1]).expand(2, -1).clone()
        reset['access_region'] = original_access.clone()
        reset['access_region'][1:] = task.evaluation_access_region(original_access[1:],
            reset['workspace_bounds'][1:], to_back_wall=True)
        state = task.TaskState(2, 'cpu', task.MdpCfg(), .02, dtype=batch.center.dtype)
        state.reset(torch.arange(2), batch, **reset, step=0)
        torch.testing.assert_close(state.initial_blocker_overlap, batch.center.new_full((2,), .04**2))
        moved = fixtures.clone_batch(batch)
        moved.center[:, 1, 0] = -.82
        moved.corners[:, 1] += moved.center[:, 1, None] - batch.center[:, 1, None]
        state.advance(moved, 1)
        torch.testing.assert_close(state.blocker_overlap, batch.center.new_tensor([0., .04**2]))
        torch.testing.assert_close(state.clearance_quality, batch.center.new_tensor([1., 0.]))
        self.assertTrue(state.success[0])
        self.assertFalse(state.success[1])
        self.assertFalse(state.failure.any())
        torch.testing.assert_close(original_access[:, 0], batch.center.new_full((2,), -.65))

    def test_partial_commit_updates_only_selected_rows_and_keeps_all_spawn_inputs(self):
        env, batch = fixtures.episode_reset_fixture()
        env.cfg = SimpleNamespace(access_region_to_back_wall=True)
        env.task_scene_data.workspace_bounds = torch.tensor([-.95, -.5, -.4, .4]).expand(5, -1)
        saved = {field.name: getattr(batch, field.name).clone() for field in fixtures.dataclasses.fields(batch)}
        ids = torch.tensor([3, 1])
        untouched = torch.tensor([0, 2, 4])
        def reject(*args, **kwargs):
            raise AssertionError('Evaluation-only geometry must not read tensors back to the host.')
        with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'tolist', reject), patch.object(torch.Tensor, 'cpu', reject), \
             patch.object(torch.Tensor, 'numpy', reject):
            task.apply_episode_batch(env, ids, batch)
        torch.testing.assert_close(env.task_scene_data.access_region[ids, 0], torch.full((2,), -.95))
        torch.testing.assert_close(env.task_scene_data.access_region[ids, 1:], batch.access_region[:, 1:])
        torch.testing.assert_close(env.task_scene_data.access_region[untouched], torch.full((3, 4), -31.))
        for name, value in saved.items():
            torch.testing.assert_close(getattr(batch, name), value, atol=0., rtol=0.)
        torch.testing.assert_close(env.scene['object_collection'].state[ids, :, :3],
                                   batch.object_state[:, :, :3] + env.scene.env_origins[ids, None])

    def test_shared_partial_resets_reuse_original_template_and_expanded_evaluation(self):
        env, template, sample = fixtures.EpisodeResetTests.shared_fixture()
        env.cfg.access_region_to_back_wall = True
        env.cfg.relocation_include_front_corridors = True
        env.task_scene_data.workspace_bounds = torch.tensor([-.95, -.5, -.4, .4]).expand(5, -1)
        saved = template.access_region.clone()
        relocation = (torch.tensor([task.rectangle_polygon(region) for region in
                                   ((-.8, -.6, -.4, -.2), (-.6, -.5, -.4, -.2))]),
                      torch.full((2,), 4), torch.ones(2, dtype=torch.bool))
        with patch.object(task, '_build_episode_batch', return_value=(template, sample)) as builder, \
             patch.object(task, '_sample_robot_starts', side_effect=fixtures.EpisodeResetTests.shared_starts), \
             patch.object(task, 'build_initial_relocation_regions', return_value=relocation) as regions:
            task.reset_scene(env)
            before = env.task_scene_data.access_region.clone()
            relocation_before = env.task_scene_data.relocation_regions
            env.common_step_counter = 17
            task.reset_scene(env, torch.tensor([4, 1]))
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(regions.call_count, 1)
        self.assertTrue(regions.call_args.kwargs['include_front_corridors'])
        self.assertIs(env._episode_reset.shared_template, template)
        self.assertIs(env._episode_reset.shared_sample, sample)
        self.assertIs(env.task_scene_data.relocation_regions, relocation_before)
        torch.testing.assert_close(relocation_before, relocation[0][None].expand(5, -1, -1, -1))
        torch.testing.assert_close(env.task_scene_data.access_region, before)
        torch.testing.assert_close(template.access_region, saved)
        self.assertEqual(sample.access_region, (-.8, -.5, -.1, .1))


if __name__ == '__main__':
    unittest.main()
