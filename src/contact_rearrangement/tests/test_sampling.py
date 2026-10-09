"""Geometry/task invariant tests independent of Isaac Sim and remote assets."""

from __future__ import annotations

import math
import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_rearrangement.sampling import (
    ObjectGeometry,
    PlanarMotion,
    PlanarPose,
    SceneSample,
    SpawnRules,
    cluster_metrics,
    footprint_polygon,
    overlap_area,
    motion_sweeps,
    polygon_area,
    polygon_distance,
    polygons_separated,
    rectangle_polygon,
    sample_scene,
    scene_metrics,
    validate_sample,
)


def make_catalog() -> tuple[ObjectGeometry, ...]:
    # Mixed sizes with intentionally noncentered asset roots and height bounds.
    sizes = ((0.05, 0.05, 0.14), (0.085, 0.06, 0.19), (0.075, 0.10, 0.12),
             (0.065, 0.065, 0.11), (0.10, 0.08, 0.17), (0.055, 0.055, 0.15),
             (0.10, 0.10, 0.14), (0.06, 0.16, 0.21), (0.045, 0.095, 0.18),
             (0.065, 0.065, 0.10), (0.065, 0.10, 0.19), (0.045, 0.09, 0.09),
             (0.03, 0.08, 0.09), (0.06, 0.10, 0.07), (0.09, 0.11, 0.25))
    return tuple(
        ObjectGeometry(f"object_{index}", (0.013 - x / 2, -0.02 - y / 2, -0.007),
                       (0.013 + x / 2, -0.02 + y / 2, z - 0.007))
        for index, (x, y, z) in enumerate(sizes)
    )


class PolygonTests(unittest.TestCase):
    def test_clipped_overlap_is_symmetric_for_rotated_footprint(self):
        box = ObjectGeometry("box", (-0.5, -0.5, 0), (0.5, 0.5, 1))
        square = rectangle_polygon((-0.5, 0.5, -0.5, 0.5))
        rotated = footprint_polygon(box, (0, 0), math.pi / 4)
        self.assertAlmostEqual(polygon_area(rotated), 1.0)
        self.assertAlmostEqual(overlap_area(square, rotated), 2 * math.sqrt(2) - 2)
        self.assertAlmostEqual(overlap_area(square, rotated), overlap_area(rotated, square))

    def test_sat_detects_overlap_gap_and_boundary_contact(self):
        first = rectangle_polygon((0, 1, 0, 1))
        touching = rectangle_polygon((1, 2, 0, 1))
        separated = rectangle_polygon((1.02, 2, 0, 1))
        overlapping = rectangle_polygon((0.9, 2, 0, 1))
        self.assertTrue(polygons_separated(first, touching))
        self.assertFalse(polygons_separated(first, touching, 0.01))
        self.assertTrue(polygons_separated(first, separated, 0.02))
        self.assertFalse(polygons_separated(first, separated, 0.021))
        self.assertFalse(polygons_separated(first, overlapping))
        self.assertAlmostEqual(overlap_area(first, touching), 0.0)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = make_catalog()
        self.rules = SpawnRules()

    def test_seeded_catalog_preserves_roles_count_and_whole_shelf_geometry(self):
        counts, targets, blockers, path_kinds = set(), set(), set(), set()
        rng = random.Random(153)
        for _ in range(200):
            expected_rng = random.Random()
            expected_rng.setstate(rng.getstate())
            expected_target = expected_rng.choice(tuple(range(len(self.catalog))))
            expected_blocker = expected_rng.choice(tuple(index for index in range(len(self.catalog))
                                                        if index != expected_target))
            expected_count = expected_rng.randint(self.rules.min_objects, self.rules.max_objects)
            sample = sample_scene(self.catalog, self.rules, rng)
            self.assertEqual(len(sample.active_ids), expected_count)
            self.assertEqual((sample.target_id, sample.blocker_id), (expected_target, expected_blocker))
            validate_sample(sample, self.catalog, self.rules)
            metrics = scene_metrics(sample, self.catalog, self.rules)
            self.assertGreaterEqual(metrics.x_coverage, self.rules.min_x_coverage)
            self.assertGreaterEqual(metrics.y_coverage, self.rules.min_y_coverage)
            self.assertGreaterEqual(metrics.local_neighbor_fraction, self.rules.min_neighbor_fraction)
            self.assertGreater(metrics.shelf_occupancy, 0)
            self.assertLess(metrics.shelf_occupancy, 1)
            counts.add(len(sample.active_ids))
            targets.add(sample.target_id)
            blockers.add(sample.blocker_id)
            path_kinds.add(sample.clearance_kind)
            for index in sample.active_ids:
                self.assertAlmostEqual(sample.positions[index][2] + self.catalog[index].lower[2], 1.05)
        self.assertEqual(counts, set(range(4, 11)))
        self.assertEqual(targets, set(range(15)))
        self.assertEqual(blockers, set(range(15)))
        self.assertEqual(path_kinds, {"translation", "pivot", "rotate_translate"})

    def test_identical_seed_reproduces_complete_scene(self):
        self.assertEqual(sample_scene(self.catalog, self.rules, random.Random(9)),
                         sample_scene(self.catalog, self.rules, random.Random(9)))

    def test_explicit_roles_and_individually_missing_roles(self):
        for target, blocker in ((2, 7), (2, None), (None, 7)):
            sample = sample_scene(self.catalog, self.rules, random.Random(20), target_id=target, blocker_id=blocker)
            validate_sample(sample, self.catalog, self.rules)
            if target is not None:
                self.assertEqual(sample.target_id, target)
            if blocker is not None:
                self.assertEqual(sample.blocker_id, blocker)
            self.assertNotEqual(sample.target_id, sample.blocker_id)
        for target, blocker in ((-1, 2), (len(self.catalog), None), (None, -1), (1, len(self.catalog))):
            with self.assertRaisesRegex(ValueError, "index the catalog"):
                sample_scene(self.catalog, self.rules, random.Random(0), target_id=target, blocker_id=blocker)
        with self.assertRaisesRegex(ValueError, "distinct"):
            sample_scene(self.catalog, self.rules, random.Random(0), target_id=2, blocker_id=2)

    def test_rejected_layout_preserves_the_first_chosen_pair_and_count(self):
        class RetryRandom(random.Random):
            def __init__(self):
                super().__init__(12)
                self.choice_populations = []
                self.sample_populations = []
                self.yaw_calls = 0
                self.count_calls = 0

            def choice(self, population):
                self.choice_populations.append(tuple(population))
                if len(self.choice_populations) <= 2:
                    return len(self.choice_populations) - 1
                return super().choice(population)

            def sample(self, population, k, **kwargs):
                self.sample_populations.append(tuple(population))
                return super().sample(population, k, **kwargs)

            def randint(self, low, high):
                self.count_calls += 1
                return super().randint(low, high)

            def uniform(self, lower, upper):
                if lower == -math.pi and upper == math.pi:
                    self.yaw_calls += 1
                    if self.yaw_calls <= 2:
                        return 0.0  # Both long boxes span X; the first pair cannot fit.
                return super().uniform(lower, upper)

        catalog = tuple(ObjectGeometry(str(index), (-0.125, -0.02, 0), (0.125, 0.02, 0.10))
                        if index < 2 else self.catalog[index] for index in range(6))
        rng = RetryRandom()
        rules = replace(self.rules, min_objects=4, max_objects=4)
        sample = sample_scene(catalog, rules, rng)
        self.assertEqual((sample.target_id, sample.blocker_id), (0, 1))
        self.assertEqual(rng.count_calls, 1)
        self.assertGreaterEqual(len(rng.sample_populations), 2)
        self.assertTrue(all(set(population) == {2, 3, 4, 5} for population in rng.sample_populations))

    def test_active_count_can_be_fixed_up_to_ten(self):
        for count in (4, 6, 10):
            rules = replace(self.rules, min_objects=count, max_objects=count)
            sample = sample_scene(self.catalog, rules, random.Random(count))
            self.assertEqual(len(sample.active_ids), count)
            validate_sample(sample, self.catalog, rules)

    def test_target_type_unique_but_other_types_can_repeat(self):
        catalog = tuple(replace(self.catalog[asset], name=f"asset_{asset}_instance_{instance}",
                                asset_name=f"asset_{asset}") for asset in range(2) for instance in range(6))
        rules = replace(self.rules, max_objects=6)
        for seed in range(50):
            sample = sample_scene(catalog, rules, random.Random(seed))
            identities = [catalog[index].asset_name for index in sample.active_ids]
            self.assertEqual(identities.count(catalog[sample.target_id].asset_name), 1)
            self.assertEqual(identities.count(catalog[sample.blocker_id].asset_name), len(identities) - 1)
            self.assertEqual(len(set(sample.active_ids)), len(sample.active_ids))
        with self.assertRaisesRegex(ValueError, "cannot also be"):
            sample_scene(catalog, rules, random.Random(0), target_id=0, blocker_id=1)

    def test_all_ten_types_mode_uses_one_instance_of_every_type(self):
        catalog = tuple(replace(geometry, name=f"{geometry.name}_copy_{copy}", asset_name=geometry.name)
                        for geometry in self.catalog[:10] for copy in range(3))
        rules = replace(self.rules, min_objects=10, max_objects=10, all_asset_types=True)
        for seed in range(30):
            sample = sample_scene(catalog, rules, random.Random(seed))
            validate_sample(sample, catalog, rules)
            identities = [catalog[index].asset_name for index in sample.active_ids]
            self.assertEqual(len(identities), 10)
            self.assertEqual(set(identities), {geometry.name for geometry in self.catalog[:10]})
        with self.assertRaisesRegex(ValueError, "count must satisfy"):
            sample_scene(catalog, replace(rules, max_objects=6, min_objects=4), random.Random(0))

    def test_validation_rejects_a_second_instance_of_the_target_type(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(5))
        catalog = list(self.catalog)
        surrounding = sample.active_ids[2]
        catalog[surrounding] = replace(catalog[surrounding], asset_name=catalog[sample.target_id].name)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_sample(sample, catalog, self.rules)

    def test_target_can_approach_front_of_full_feasible_range(self):
        advances = []
        for seed in range(100):
            sample = sample_scene(self.catalog, self.rules, random.Random(seed), target_id=0, blocker_id=5)
            target_polygon = footprint_polygon(self.catalog[0], sample.positions[0][:2], sample.yaws[0])
            rear_edge = min(x for x, _ in target_polygon)
            advances.append(rear_edge - self.rules.x_range[0] - self.rules.edge_margin)
        self.assertGreater(max(advances), 0.18)  # Earlier proposal was capped at rear + 0.10 m.
        self.assertLess(min(advances), 0.02)

    def test_unworkable_geometry_fails_after_bounded_attempts(self):
        impossible = tuple(ObjectGeometry(str(index), (-1, -1, 0), (1, 1, 0.1)) for index in range(10))
        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            sample_scene(impossible, replace(self.rules, max_attempts=3), random.Random(0))

    def test_invalid_catalog_and_rules_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            sample_scene((self.catalog[0],) * 10, self.rules, random.Random(0))
        with self.assertRaisesRegex(ValueError, "Active object limits"):
            sample_scene(self.catalog, replace(self.rules, min_objects=1), random.Random(0))
        invalid = replace(self.catalog[0], lower=(math.nan, 0, 0))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            sample_scene((invalid,) + self.catalog[1:], self.rules, random.Random(0))
        with self.assertRaisesRegex(ValueError, "neighbor fraction"):
            sample_scene(self.catalog, replace(self.rules, min_x_coverage=1.01), random.Random(0))

    def test_witness_must_clear_access_and_not_leave_shelf(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(3))
        blocker = sample.positions[sample.blocker_id]
        with self.assertRaisesRegex(ValueError, "still intersects"):
            validate_sample(replace(sample, clearance_path=(), clearance_witness=(blocker[0], blocker[1] + 0.001)),
                            self.catalog, self.rules)
        with self.assertRaisesRegex(ValueError, "leaves supported"):
            validate_sample(replace(sample, clearance_path=(), clearance_witness=(blocker[0], 5.0)),
                            self.catalog, self.rules)

    def test_clear_endpoint_does_not_hide_collision_during_translation(self):
        catalog = tuple(ObjectGeometry(str(index), (-0.025, -0.025, 0), (0.025, 0.025, 0.05))
                        for index in range(4))
        sample = SceneSample(
            active_ids=(0, 1, 2, 3), target_id=0, blocker_id=1,
            positions={0: (-0.82, 0.0, 1.05), 1: (-0.70, 0.0, 1.05),
                       2: (-0.70, 0.14, 1.05), 3: (-0.84, -0.18, 1.05)},
            yaws={index: 0.0 for index in range(4)},
            access_region=(-0.785, -0.51, -0.06, 0.06), clearance_witness=(-0.70, 0.25),
        )
        rules = replace(self.rules, min_objects=4, max_objects=4, min_x_coverage=0,
                        min_y_coverage=0, min_neighbor_fraction=0)
        with self.assertRaisesRegex(ValueError, "path is obstructed"):
            validate_sample(sample, catalog, rules)
        positions = dict(sample.positions)
        positions[2] = (-0.70, -0.14, 1.05)
        validate_sample(replace(sample, positions=positions), catalog, rules)

    def test_pivot_sweep_catches_obstacle_between_clear_endpoints(self):
        bar = ObjectGeometry("bar", (-0.10, -0.01, 0), (0.10, 0.01, 0.1))
        motion = PlanarMotion(PlanarPose(0, 0, 0), PlanarPose(-0.11, 0.09, math.pi / 2), (-0.10, -0.01))
        obstacle = rectangle_polygon((0.023, 0.035, 0.133, 0.145))
        initial = footprint_polygon(bar, (motion.start.x, motion.start.y), motion.start.yaw)
        final = footprint_polygon(bar, (motion.end.x, motion.end.y), motion.end.yaw)
        self.assertTrue(polygons_separated(initial, obstacle))
        self.assertTrue(polygons_separated(final, obstacle))
        self.assertTrue(any(not polygons_separated(sweep, obstacle) for sweep in motion_sweeps(bar, motion)))

    def test_rotating_sweep_contains_corners_between_subdivision_endpoints(self):
        bar = ObjectGeometry("bar", (-0.10, -0.01, 0), (0.10, 0.01, 0.1))
        motion = PlanarMotion(PlanarPose(0, 0, 0), PlanarPose(0.03, 0.02, math.pi / 2))
        sweeps = motion_sweeps(bar, motion)
        # Mid-interval corners bulge outside their straight chords. They must
        # remain inside the certified envelopes, rather than being missed.
        for step in range(18):
            fraction = (step + 0.5) / 18
            polygon = footprint_polygon(bar, (0.03 * fraction, 0.02 * fraction), fraction * math.pi / 2)
            for x, y in polygon:
                tiny_box = rectangle_polygon((x - 1e-10, x + 1e-10, y - 1e-10, y + 1e-10))
                self.assertFalse(polygons_separated(sweeps[step], tiny_box))

    def test_coverage_and_local_clutter_are_separate_requirements(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(0))
        metrics = scene_metrics(sample, self.catalog, self.rules)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_sample(sample, self.catalog, replace(self.rules, min_x_coverage=metrics.x_coverage + 0.001))
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_sample(sample, self.catalog, replace(self.rules, min_y_coverage=metrics.y_coverage + 0.001))
        with self.assertRaisesRegex(ValueError, "nearby clutter"):
            validate_sample(sample, self.catalog, replace(self.rules, min_neighbor_fraction=1.0))
        # Whole-shelf occupancy below the old 35% is deliberately valid.
        self.assertLess(metrics.shelf_occupancy, 0.35)

    def test_polygon_distance_uses_true_corner_gap(self):
        first = rectangle_polygon((0, 1, 0, 1))
        diagonal = rectangle_polygon((1.03, 2, 1.04, 2))
        self.assertAlmostEqual(polygon_distance(first, diagonal), 0.05)
        self.assertAlmostEqual(polygon_distance(first, rectangle_polygon((0.5, 2, 0.5, 2))), 0.0)

    def test_role_overlap_support_and_access_metadata_are_checked(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(14))
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_sample(replace(sample, blocker_id=sample.target_id), self.catalog, self.rules)
        positions = dict(sample.positions)
        surrounding = sample.active_ids[2]
        positions[surrounding] = (*sample.positions[sample.blocker_id][:2], positions[surrounding][2])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sample(replace(sample, positions=positions), self.catalog, self.rules)
        positions[surrounding] = (self.rules.x_range[1], 0.0, positions[surrounding][2])
        with self.assertRaisesRegex(ValueError, "outside supported"):
            validate_sample(replace(sample, positions=positions), self.catalog, self.rules)
        with self.assertRaisesRegex(ValueError, "fixed"):
            validate_sample(replace(sample, access_region=(-0.85, -0.5, -0.1, 0.1)), self.catalog, self.rules)
        motion = sample.clearance_path[0]
        with self.assertRaisesRegex(ValueError, "continuous"):
            validate_sample(replace(sample, clearance_path=(replace(motion, start=PlanarPose(0, 0, 0)),)),
                            self.catalog, self.rules)


if __name__ == "__main__":
    unittest.main()
