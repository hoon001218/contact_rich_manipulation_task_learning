"""CPU geometry and packing checks for the configured spawn clearance."""

import ast
import dataclasses
import math
from pathlib import Path
import random
import unittest

import torch

import test_mdp as mdp_tests
from test_mdp import episode_reset_fixture, task


OFFSET = 0.001


class SpawnHeightOffsetTests(unittest.TestCase):
    def test_common_height_uses_metres_and_accepts_attribute_or_mapping_inputs(self):
        geometry = task.ObjectGeometry('offset-box', (-.02, -.03, -.007), (.04, .05, .10))
        baseline = task.SpawnRules()
        rules = dataclasses.replace(baseline, spawn_height_offset=OFFSET)
        self.assertEqual(baseline.spawn_height_offset, 0.0)
        self.assertAlmostEqual(task.object_spawn_height(rules, geometry), 1.058, places=12)
        self.assertAlmostEqual(task.object_spawn_height(rules, geometry)
                               - task.object_spawn_height(baseline, geometry), OFFSET, places=12)
        mapping_rules = dict(shelf_top=1.05, spawn_height_offset=OFFSET)
        mapping_geometry = dict(lower=geometry.lower)
        self.assertAlmostEqual(task.object_spawn_height(mapping_rules, mapping_geometry), 1.058, places=12)
        self.assertAlmostEqual(task.object_spawn_height(dict(shelf_top=1.05), mapping_geometry), 1.057,
                               places=12)

    def test_invalid_clearance_is_rejected_instead_of_making_a_bad_spawn_pose(self):
        geometry = task.ObjectGeometry('box', (-.02, -.03, -.007), (.04, .05, .10))
        for value in (-OFFSET, math.nan, math.inf, -math.inf):
            with self.subTest(offset=value), self.assertRaises(ValueError):
                task.object_spawn_height(dataclasses.replace(task.SpawnRules(), spawn_height_offset=value), geometry)

    def test_sequential_same_seed_preserves_layout_random_stream_and_changes_only_root_height(self):
        catalog = mdp_tests.EpisodeLayoutTests.catalog(copies=3)
        rules = task.SpawnRules(generation_mode='sequential', min_objects=10, max_objects=10,
                               x_range=(-1.2, -.5), y_range=(-.455, .455))
        elevated_rules = dataclasses.replace(rules, spawn_height_offset=OFFSET)
        for seed in (4, 11, 23):
            with self.subTest(seed=seed):
                baseline_rng, raised_rng = random.Random(seed), random.Random(seed)
                original = task.sample_scene(catalog, rules, baseline_rng)
                elevated = task.sample_scene(catalog, elevated_rules, raised_rng)
                self.assertEqual(original.active_ids, elevated.active_ids)
                self.assertEqual((original.target_id, original.blocker_id),
                                 (elevated.target_id, elevated.blocker_id))
                self.assertEqual(original.yaws, elevated.yaws)
                self.assertEqual(original.access_region, elevated.access_region)
                self.assertEqual(original.reserved_regions, elevated.reserved_regions)
                self.assertEqual(original.camera_pose, elevated.camera_pose)
                self.assertEqual(baseline_rng.random(), raised_rng.random())
                for index in original.active_ids:
                    self.assertEqual(original.positions[index][:2], elevated.positions[index][:2])
                    self.assertAlmostEqual(elevated.positions[index][2] - original.positions[index][2],
                                           OFFSET, places=12)
                    self.assertAlmostEqual(elevated.positions[index][2] + catalog[index].lower[2],
                                           elevated_rules.shelf_top + OFFSET, places=12)

    def test_device_proposal_support_cache_and_packing_add_clearance_exactly_once(self):
        catalog = mdp_tests.EpisodeLayoutTests.catalog(copies=3)
        rules = task.SpawnRules(generation_mode='sequential', min_objects=10, max_objects=10,
                               x_range=(-1.2, -.5), spawn_height_offset=OFFSET)
        sample = task.sample_scene(catalog, rules, random.Random(11))
        lower, upper = (torch.tensor([getattr(g, name) for g in catalog]) for name in ('lower', 'upper'))
        upright = torch.zeros(len(catalog), 4)
        upright[:, 0] = 1.0
        default = torch.zeros(len(catalog), 13)
        default[:, 0], default[:, 2], default[:, 3] = 4.0, -.3, 1.0
        reference = task.pack_episode_states(default, [sample], lower, upper, upright)
        order = sample.active_ids[::-1]
        resident = dataclasses.replace(sample, planar_ids=order,
            planar_poses=torch.tensor([(*sample.positions[i][:2], sample.yaws[i]) for i in order]),
            positions={})
        heights = lower.new_tensor([task.object_spawn_height(rules, g) for g in catalog])
        proposed = task._episode_proposals([resident], len(catalog), 'cpu', torch.float32,
                                           support_heights=heights)
        actual = task.pack_episode_states(default, [resident], lower, upper, upright, proposals=proposed)
        for value, expected in zip(actual, reference):
            torch.testing.assert_close(value, expected, rtol=0, atol=1.e-6)
        states, active, _ = actual
        torch.testing.assert_close(states[0, active[0], 2] + lower[active[0], 2],
            lower.new_full((int(active.sum()),), rules.shelf_top + OFFSET), rtol=0, atol=1.e-6)
        torch.testing.assert_close(states[0, ~active[0], :7], default[~active[0], :7], rtol=0, atol=0)
        torch.testing.assert_close(states[..., 7:], torch.zeros_like(states[..., 7:]), rtol=0, atol=0)

    def test_indexed_native_commit_preserves_origin_height_spares_and_unselected_rows(self):
        env, batch = episode_reset_fixture(batch=5, selected=2)
        ids = torch.tensor([4, 1])
        # Include Z origins as well as XY origins to expose world/local mistakes.
        env.scene.env_origins[:, 2] = torch.arange(5) * .125
        source = batch.object_state.clone()
        source[..., 2] += OFFSET
        source[..., 7:] = 0.0
        # Parked bodies keep their existing template position, without receiving an offset.
        source[..., 2].masked_fill_(~batch.active, -.4)
        original_source = source.clone()
        before = env.scene['object_collection'].state.clone()
        task._write_physical_reset(env, ids, source, batch.joint_positions, batch.active)
        actual = env.scene['object_collection'].state
        expected = source.clone()
        expected[..., :3] += env.scene.env_origins[ids, None]
        torch.testing.assert_close(actual[ids], expected, rtol=0, atol=0)
        for row, native_id in enumerate(ids):
            active = batch.active[row]
            torch.testing.assert_close(actual[native_id, active, 2],
                source[row, active, 2] + env.scene.env_origins[native_id, 2], rtol=0, atol=1.e-6)
        torch.testing.assert_close(actual[torch.tensor([0, 2, 3])], before[torch.tensor([0, 2, 3])],
                                   rtol=0, atol=0)
        torch.testing.assert_close(actual[ids, :, 7:], torch.zeros_like(source[..., 7:]), rtol=0, atol=0)
        torch.testing.assert_close(source, original_source, rtol=0, atol=0)

    def test_current_task_explicitly_selects_one_millimetre(self):
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation_symmetric/shelf_manipulation_env_cfg.py'
        module = ast.parse(path.read_text())
        cfg = next(node for node in module.body
                   if isinstance(node, ast.ClassDef) and node.name == 'ShelfManipulationEnvCfg')
        declaration = next(node for node in cfg.body if isinstance(node, ast.AnnAssign)
                           and isinstance(node.target, ast.Name) and node.target.id == 'spawn_rules')
        self.assertIsInstance(declaration.value, ast.Call)
        keyword = next(value for value in declaration.value.keywords if value.arg == 'spawn_height_offset')
        self.assertEqual(ast.literal_eval(keyword.value), OFFSET)


if __name__ == '__main__':
    unittest.main()
