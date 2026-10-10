"""Geometry/task invariant tests independent of Isaac Sim and remote assets."""

from __future__ import annotations

import math
import random
import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contact_rearrangement.src.sampling import (
    ObjectGeometry,
    PlanarMotion,
    PlanarPose,
    SceneSample,
    RobotStartRules,
    SpawnRules,
    cluster_metrics,
    footprint_polygon,
    elbow_above_chord,
    oriented_box_separated,
    overlap_area,
    motion_sweeps,
    polygon_area,
    polygon_distance,
    polygons_separated,
    rectangle_polygon,
    robot_start_region,
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
        ObjectGeometry(f"object_{asset}_copy_{copy}", (0.013 - x / 2, -0.02 - y / 2, -0.007),
                       (0.013 + x / 2, -0.02 + y / 2, z - 0.007), asset_name=f"object_{asset}")
        for asset, (x, y, z) in enumerate(sizes[:10]) for copy in range(10)
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


class RobotStartGeometryTests(unittest.TestCase):
    IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    @staticmethod
    def rotation_xyz(degrees):
        x, y, z = (math.radians(value) for value in degrees)
        cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
        return (
            (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
            (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
            (-sy, cy * sx, cy * cx),
        )

    def make_roles(self, position=(-.75, .1, 1.05), yaw=math.pi / 2):
        # An off-center asset root must not change the footprint-centered front.
        geometries = (
            ObjectGeometry("target", (-.025, -.025, 0), (.025, .025, .1)),
            ObjectGeometry("blocker", (-.08, -.06, 0), (.12, .04, .1)),
        )
        sample = SceneSample((0, 1), 0, 1, {0: (-.85, .1, 1.05), 1: position},
                             {0: 0.0, 1: yaw}, (-.8, -.5, .04, .16), (-.75, -.1))
        return sample, geometries

    def test_start_region_uses_rotated_front_and_shelf_front_clearance(self):
        sample, geometries = self.make_roles()
        shelf = replace(SpawnRules(), x_range=(-.9, -.5))
        bounds = robot_start_region(sample, geometries, shelf, RobotStartRules())
        expected = (-.44, -.30, .02, .18, 1.25, 1.33)
        for actual, wanted in zip(bounds, expected):
            self.assertAlmostEqual(actual, wanted)
        # When the Blocker lies further forward, its front-distance bound wins.
        sample, geometries = self.make_roles(position=(-.58, -.1, 1.05), yaw=0)
        bounds = robot_start_region(sample, geometries, shelf, RobotStartRules())
        self.assertAlmostEqual(bounds[0], -.36)
        self.assertAlmostEqual(bounds[1], -.08)
        self.assertAlmostEqual(bounds[2], -.18)
        self.assertAlmostEqual(bounds[3], -.02)

    def test_impossible_or_invalid_start_region_fails_without_relaxation(self):
        sample, geometries = self.make_roles()
        rules = RobotStartRules()
        for invalid in (
            replace(rules, front_distance_range=(.3, .2)),
            replace(rules, front_distance_range=(-.1, .4)),
            replace(rules, height_above_shelf=(.2, .2)),
            replace(rules, shelf_front_clearance=-.01),
            replace(rules, lateral_half_width=0),
            replace(rules, collision_clearance=math.nan),
            replace(rules, min_elbow_height=-.01),
            replace(rules, max_attempts=0),
            replace(rules, max_attempts=1.5),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                robot_start_region(sample, geometries, SpawnRules(), invalid)
        with self.assertRaisesRegex(ValueError, "empty"):
            robot_start_region(sample, geometries, SpawnRules(), replace(rules, front_distance_range=(.12, .20)))
        with self.assertRaisesRegex(ValueError, "active"):
            robot_start_region(replace(sample, blocker_id=3), geometries, SpawnRules(), rules)
        with self.assertRaisesRegex(ValueError, "finite"):
            robot_start_region(sample, geometries, replace(SpawnRules(), shelf_top=math.inf), rules)

    def test_elbow_branch_uses_world_height_above_chord(self):
        self.assertAlmostEqual(elbow_above_chord((0, 0, 1), (.3, 0, 1.4), (.6, 0, 1)), .4)
        self.assertAlmostEqual(elbow_above_chord((0, 0, 1), (.3, 0, .6), (.6, 0, 1)), -.4)
        # A rising shoulder-to-wrist chord must be subtracted at the projected point.
        self.assertAlmostEqual(elbow_above_chord((0, 0, 1), (.4, 0, 1.4), (.6, 0, 1.2)), .24)
        # The shoulder/wrist segment, not an unbounded line, defines the reference.
        self.assertAlmostEqual(elbow_above_chord((0, 0, 1), (-.1, 0, 1.2), (.6, 0, 1)), .2)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            elbow_above_chord((0, 0, 1), (.1, 0, 1.2), (0, 0, 1))
        with self.assertRaisesRegex(ValueError, "finite"):
            elbow_above_chord((0, 0, 1), (.1, 0, math.nan), (.6, 0, 1))

    def test_obb_detects_overlap_touching_and_required_clearance(self):
        first = ((0, 0, 0), self.IDENTITY, (.5, .5, .5))
        self.assertTrue(oriented_box_separated(*first, (1, 0, 0), self.IDENTITY, (.5, .5, .5)))
        self.assertFalse(oriented_box_separated(*first, (1, 0, 0), self.IDENTITY, (.5, .5, .5), .01))
        self.assertTrue(oriented_box_separated(*first, (1.02, 0, 0), self.IDENTITY, (.5, .5, .5), .02))
        self.assertFalse(oriented_box_separated(*first, (1.02, 0, 0), self.IDENTITY, (.5, .5, .5), .021))
        self.assertFalse(oriented_box_separated(*first, (.9, .2, .1), self.IDENTITY, (.5, .5, .5)))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            oriented_box_separated(*first, (2, 0, 0), self.IDENTITY, (.5, .5, .5), -.01)

    def test_tcp_clearance_does_not_hide_tilted_finger_collision(self):
        obstacle = ((.14, 0, .25), self.IDENTITY, (.05, .05, .05))
        tcp = ((0, 0, .4), self.IDENTITY, (.03, .03, .03))
        finger = ((.08, 0, .32), self.rotation_xyz((0, 45, 0)), (.015, .015, .12))
        self.assertTrue(oriented_box_separated(*tcp, *obstacle, .015))
        self.assertFalse(oriented_box_separated(*finger, *obstacle, .015))

    def test_edge_cross_axes_detect_separation_missed_by_all_face_axes(self):
        first = ((0, 0, 0), self.rotation_xyz((60, 75, 45)), (.30, .045, .045))
        second = ((-.01, -.12, -.09), self.rotation_xyz((75, 45, 15)), (.25, .04, .06))
        axes_a = tuple(zip(*first[1]))
        axes_b = tuple(zip(*second[1]))
        # World AABB axes and all six face normals overlap in this fixture.
        # Only one of the nine edge-cross-product axes proves separation.
        for axis in (*self.IDENTITY, *axes_a, *axes_b):
            distance = abs(sum(second[0][index] * axis[index] for index in range(3)))
            radius = sum(first[2][index] * abs(sum(axes_a[index][j] * axis[j] for j in range(3))) +
                         second[2][index] * abs(sum(axes_b[index][j] * axis[j] for j in range(3)))
                         for index in range(3))
            self.assertLess(distance, radius)
        self.assertTrue(oriented_box_separated(*first, *second))
        self.assertTrue(oriented_box_separated(*second, *first))
        self.assertFalse(oriented_box_separated(*first, (0, 0, 0), second[1], second[2]))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.catalog = make_catalog()
        self.rules = SpawnRules()

    def test_kind_count_is_separate_from_dense_instance_count(self):
        kinds, targets, path_kinds = set(), set(), set()
        for seed in range(80):
            sample = sample_scene(self.catalog, self.rules, random.Random(seed))
            validate_sample(sample, self.catalog, self.rules)
            identities = [self.catalog[index].asset_name for index in sample.active_ids]
            kinds.add(len(set(identities)))
            targets.add(self.catalog[sample.target_id].asset_name)
            path_kinds.add(sample.clearance_kind)
            self.assertGreater(len(sample.active_ids), len(set(identities)))
            self.assertLessEqual(len(sample.active_ids), self.rules.max_objects)
            self.assertEqual(identities.count(self.catalog[sample.target_id].asset_name), 1)
            metrics = scene_metrics(sample, self.catalog, self.rules)
            self.assertGreaterEqual(metrics.shelf_occupancy + 1e-9, self.rules.min_shelf_occupancy)
            self.assertGreaterEqual(min(metrics.region_occupancies) + 1e-9, self.rules.min_region_occupancy)
            self.assertGreaterEqual(metrics.local_neighbor_fraction, self.rules.min_neighbor_fraction)
            for index in sample.active_ids:
                self.assertAlmostEqual(sample.positions[index][2] + self.catalog[index].lower[2], 1.05)
        self.assertEqual(kinds, set(range(4, 11)))
        self.assertEqual(targets, {f"object_{index}" for index in range(10)})
        self.assertEqual(path_kinds, {"translation", "pivot", "rotate_translate"})

    def test_all_types_allows_duplicates_to_fill_shelf(self):
        rules = replace(self.rules, all_asset_types=True)
        for seed in range(10):
            sample = sample_scene(self.catalog, rules, random.Random(seed))
            validate_sample(sample, self.catalog, rules)
            identities = [self.catalog[index].asset_name for index in sample.active_ids]
            self.assertEqual(set(identities), {f"object_{index}" for index in range(10)})
            self.assertGreater(len(sample.active_ids), 10)
            self.assertEqual(identities.count(self.catalog[sample.target_id].asset_name), 1)

    def test_density_duplicates_balance_counts_despite_different_footprint_sizes(self):
        # Mixed footprint areas previously made the largest type dominate.
        for kinds in (4, 6, 10):
            rules = replace(self.rules, min_asset_types=kinds, max_asset_types=kinds)
            for seed in range(5):
                sample = sample_scene(self.catalog, rules, random.Random(seed), target_id=0, blocker_id=10)
                counts = Counter(self.catalog[index].asset_name for index in sample.active_ids)
                target_type = self.catalog[sample.target_id].asset_name
                self.assertEqual(counts.pop(target_type), 1)
                self.assertEqual(len(counts), kinds - 1)
                self.assertGreater(sum(counts.values()), kinds - 1)
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
                self.assertGreaterEqual(scene_metrics(sample, self.catalog, rules).shelf_occupancy + 1e-9, .35)

    def test_balanced_duplicates_continue_when_one_type_has_no_spare_copies(self):
        catalog = tuple(geometry for index, geometry in enumerate(self.catalog)
                        if geometry.asset_name != "object_1" or index == 10)
        rules = replace(self.rules, all_asset_types=True)
        sample = sample_scene(catalog, rules, random.Random(3), target_id=0, blocker_id=10)
        counts = Counter(catalog[index].asset_name for index in sample.active_ids)
        self.assertEqual(counts.pop("object_0"), 1)  # Target remains unique.
        self.assertEqual(counts.pop("object_1"), 1)  # This kind's only copy is the Blocker.
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertGreater(max(counts.values()), 1)
        validate_sample(sample, catalog, rules)

    def test_identical_seed_reproduces_complete_scene(self):
        self.assertEqual(sample_scene(self.catalog, self.rules, random.Random(9)),
                         sample_scene(self.catalog, self.rules, random.Random(9)))

    def test_new_layout_retains_explicit_episode_composition_and_id_order(self):
        class LayoutRandom(random.Random):
            def randint(self, low, high):
                raise AssertionError("A fixed episode must not draw a new type count.")

            def sample(self, population, k, **kwargs):
                raise AssertionError("A fixed episode must not draw a new type subset.")

        original = sample_scene(self.catalog, self.rules, random.Random(9))
        for ids in (original.active_ids, tuple(reversed(original.active_ids))):
            with self.subTest(ids=ids):
                proposal = sample_scene(self.catalog, self.rules, LayoutRandom(4),
                                        target_id=original.target_id, blocker_id=original.blocker_id,
                                        active_ids=ids)
                validate_sample(proposal, self.catalog, self.rules)
                self.assertEqual(proposal.active_ids, ids)
                self.assertEqual((proposal.target_id, proposal.blocker_id),
                                 (original.target_id, original.blocker_id))
                self.assertEqual(Counter(self.catalog[index].asset_name for index in proposal.active_ids),
                                 Counter(self.catalog[index].asset_name for index in original.active_ids))
                self.assertNotEqual(proposal.positions, original.positions)
                self.assertNotEqual(proposal.yaws, original.yaws)

    def test_invalid_fixed_composition_fails_without_changing_ids_or_density(self):
        original = sample_scene(self.catalog, self.rules, random.Random(9))
        target, blocker = original.target_id, original.blocker_id
        another_target = 10 * (target // 10) + (target + 1) % 10
        malformed = (
            (original.active_ids, None, blocker, "explicit"),
            (original.active_ids, target, None, "explicit"),
            (original.active_ids + (original.active_ids[-1],), target, blocker, "unique"),
            (original.active_ids[:-1] + (len(self.catalog),), target, blocker, "index the catalog"),
            (tuple(index for index in original.active_ids if index != blocker), target, blocker, "present"),
            (original.active_ids[:-1] + (another_target,), target, blocker, "exactly once"),
            ((target, blocker), target, blocker, "active object limits"),
            ((target, blocker, 10 * (blocker // 10) + (blocker + 1) % 10,
              10 * (blocker // 10) + (blocker + 2) % 10), target, blocker, "asset type limits"),
            ((0, 10, 20, 30), 0, 10, "whole-shelf density"),
        )
        for ids, fixed_target, fixed_blocker, message in malformed:
            with self.subTest(ids=ids, message=message), self.assertRaisesRegex(ValueError, message):
                sample_scene(self.catalog, self.rules, random.Random(4),
                             target_id=fixed_target, blocker_id=fixed_blocker, active_ids=ids)
        with self.assertRaisesRegex(ValueError, "every catalog asset type"):
            sample_scene(self.catalog, replace(self.rules, all_asset_types=True), random.Random(4),
                         target_id=0, blocker_id=10, active_ids=(0, 10, 20, 30))

    def test_explicit_roles_and_individually_missing_roles(self):
        for target, blocker in ((2, 17), (2, None), (None, 17)):
            sample = sample_scene(self.catalog, self.rules, random.Random(20), target_id=target, blocker_id=blocker)
            validate_sample(sample, self.catalog, self.rules)
            if target is not None:
                self.assertEqual(sample.target_id, target)
            if blocker is not None:
                self.assertEqual(sample.blocker_id, blocker)
        for target, blocker in ((-1, 2), (len(self.catalog), None), (None, -1), (1, len(self.catalog))):
            with self.assertRaisesRegex(ValueError, "index the catalog"):
                sample_scene(self.catalog, self.rules, random.Random(0), target_id=target, blocker_id=blocker)
        with self.assertRaisesRegex(ValueError, "distinct"):
            sample_scene(self.catalog, self.rules, random.Random(0), target_id=2, blocker_id=2)
        with self.assertRaisesRegex(ValueError, "cannot also be"):
            sample_scene(self.catalog, self.rules, random.Random(0), target_id=2, blocker_id=7)

    def test_rejected_layout_preserves_roles_types_and_instance_composition(self):
        class RetryRandom(random.Random):
            def __init__(self):
                super().__init__(12)
                self.choice_calls = 0
                self.type_samples = []
                self.yaw_calls = 0
                self.count_calls = 0

            def choice(self, population):
                self.choice_calls += 1
                if self.choice_calls == 1:
                    return 0
                if self.choice_calls == 2:
                    return 10
                return super().choice(population)

            def sample(self, population, k, **kwargs):
                self.type_samples.append(tuple(population))
                return super().sample(population, k, **kwargs)

            def randint(self, low, high):
                self.count_calls += 1
                return super().randint(low, high)

            def uniform(self, lower, upper):
                if lower == -math.pi and upper == math.pi:
                    self.yaw_calls += 1
                    if self.yaw_calls <= 2:
                        return 0.0  # First two long roles cannot fit together along X.
                return super().uniform(lower, upper)

        catalog = tuple(replace(geometry, lower=(-0.125, -0.02, 0), upper=(0.125, 0.02, 0.1))
                        if index < 20 else geometry for index, geometry in enumerate(self.catalog))
        rng = RetryRandom()
        rules = replace(self.rules, min_asset_types=4, max_asset_types=4)
        sample = sample_scene(catalog, rules, rng)
        self.assertEqual((sample.target_id, sample.blocker_id), (0, 10))
        self.assertEqual(rng.count_calls, 1)  # Only the kind count is drawn once.
        self.assertEqual(len(rng.type_samples), 1)  # Kind subset does not change on retry.
        self.assertGreater(rng.yaw_calls, len(sample.active_ids))

    def test_instance_cap_does_not_silently_increase_for_density(self):
        rules = replace(self.rules, all_asset_types=True, max_objects=10)
        with self.assertRaisesRegex(RuntimeError, "fixed instance cap"):
            sample_scene(self.catalog, rules, random.Random(0))

    def test_regional_density_controls_composition_when_higher_than_global(self):
        # All cells at 20% imply a global minimum of 20%, even when the
        # explicitly requested global threshold is only 5%.
        rules = replace(self.rules, min_shelf_occupancy=.05, min_region_occupancy=.20,
                        min_asset_types=4, max_asset_types=4)
        sample = sample_scene(self.catalog, rules, random.Random(0))
        metrics = scene_metrics(sample, self.catalog, rules)
        self.assertGreater(len(sample.active_ids), 4)
        self.assertGreaterEqual(metrics.shelf_occupancy + 1e-9, .20)
        self.assertGreaterEqual(min(metrics.region_occupancies) + 1e-9, .20)
        validate_sample(sample, self.catalog, rules)

    def test_insufficient_type_capacity_fails_before_layout(self):
        tiny = tuple(ObjectGeometry(f"kind_{kind}_copy_{copy}", (-.02, -.02, 0), (.02, .02, .1),
                                    f"kind_{kind}") for kind in range(4) for copy in range(10))
        rules = replace(self.rules, min_asset_types=4, max_asset_types=4, min_shelf_occupancy=.45)
        with self.assertRaisesRegex(RuntimeError, "fixed instance cap"):
            sample_scene(tiny, rules, random.Random(0))

    def test_unworkable_geometry_fails_after_bounded_attempts(self):
        impossible = tuple(ObjectGeometry(str(index), (-1, -1, 0), (1, 1, .1)) for index in range(10))
        with self.assertRaisesRegex(RuntimeError, "after 3 attempts"):
            sample_scene(impossible, replace(self.rules, max_objects=10, max_attempts=3), random.Random(0))

    def test_invalid_catalog_rules_and_kind_limits_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            sample_scene((self.catalog[1],) + self.catalog[1:], self.rules, random.Random(0))
        with self.assertRaisesRegex(ValueError, "Active object limits"):
            sample_scene(self.catalog, replace(self.rules, min_objects=1), random.Random(0))
        with self.assertRaisesRegex(ValueError, "Asset type limits"):
            sample_scene(self.catalog, replace(self.rules, max_asset_types=11), random.Random(0))
        invalid = replace(self.catalog[0], lower=(math.nan, 0, 0))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            sample_scene((invalid,) + self.catalog[1:], self.rules, random.Random(0))
        with self.assertRaisesRegex(ValueError, "Region grid"):
            sample_scene(self.catalog, replace(self.rules, region_grid=(0, 3)), random.Random(0))

    def test_global_and_regional_density_are_independent_hard_requirements(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(0))
        metrics = scene_metrics(sample, self.catalog, self.rules)
        with self.assertRaisesRegex(ValueError, "Whole-shelf footprint occupancy"):
            validate_sample(sample, self.catalog, replace(self.rules, min_shelf_occupancy=metrics.shelf_occupancy + .001))
        with self.assertRaisesRegex(ValueError, "shelf region footprint occupancy"):
            validate_sample(sample, self.catalog, replace(self.rules, min_region_occupancy=min(metrics.region_occupancies) + .001))
        self.assertAlmostEqual(sum(metrics.region_occupancies) / len(metrics.region_occupancies), metrics.shelf_occupancy)

    def test_sparse_large_span_scene_cannot_pass_global_density(self):
        sparse = replace(self.rules, min_objects=4, max_objects=4, min_asset_types=4, max_asset_types=4,
                         min_shelf_occupancy=0, min_region_occupancy=0, min_neighbor_fraction=0)
        sample = sample_scene(self.catalog, sparse, random.Random(0))
        self.assertGreaterEqual(scene_metrics(sample, self.catalog, sparse).y_coverage, sparse.min_y_coverage)
        with self.assertRaisesRegex(ValueError, "Whole-shelf footprint occupancy"):
            validate_sample(sample, self.catalog, replace(sparse, min_shelf_occupancy=.35))

    def test_dense_scene_with_large_empty_shelf_region_is_rejected(self):
        # Eight supported, mutually separated boxes fill >50% of the shelf and
        # cover >65% of its Y span, yet leave its entire bottom third empty.
        # The designated Blocker can translate into that empty part safely.
        catalog = tuple(ObjectGeometry(str(index), (-.08, -.065, 0), (.08, .065, .1))
                        for index in range(8))
        positions = {2 * row + col: (x, y, 1.05)
                     for row, y in enumerate((-.0816, .0706, .2228, .375))
                     for col, x in enumerate((-.7875, -.6125))}
        sample = SceneSample(tuple(range(8)), 0, 1, positions, {index: 0 for index in range(8)},
                             (-.6975, -.51, -.1416, -.0216), (-.6125, -.27))
        rules = replace(self.rules, max_objects=8, max_asset_types=8, min_region_occupancy=0)
        validate_sample(sample, catalog, rules)
        metrics = scene_metrics(sample, catalog, rules)
        self.assertGreater(metrics.shelf_occupancy, .50)
        self.assertGreater(metrics.y_coverage, .65)
        self.assertAlmostEqual(min(metrics.region_occupancies), 0)
        with self.assertRaisesRegex(ValueError, "shelf region footprint occupancy"):
            validate_sample(sample, catalog, replace(rules, min_region_occupancy=.10))

    def test_validation_rejects_a_second_instance_of_target_type(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(5))
        catalog = list(self.catalog)
        surrounding = sample.active_ids[2]
        catalog[surrounding] = replace(catalog[surrounding], asset_name=catalog[sample.target_id].asset_name)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_sample(sample, catalog, self.rules)

    def test_blocker_front_reservation_rejects_clutter_outside_target_access(self):
        catalog = (
            ObjectGeometry("target", (-.025, -.025, 0), (.025, .025, .1)),
            ObjectGeometry("blocker", (-.05, -.10, 0), (.05, .10, .1)),
            ObjectGeometry("front_clutter", (-.02, -.02, 0), (.02, .02, .1)),
            ObjectGeometry("rear_clutter", (-.025, -.025, 0), (.025, .025, .1)),
        )
        sample = SceneSample((0, 1, 2, 3), 0, 1,
            {0: (-.82, 0, 1.05), 1: (-.68, .05, 1.05),
             2: (-.57, .11, 1.05), 3: (-.84, .28, 1.05)},
            {index: 0.0 for index in range(4)}, (-.785, -.51, -.06, .06), (-.68, -.20))
        rules = replace(self.rules, min_objects=4, max_objects=4, min_asset_types=4, max_asset_types=4,
                        min_x_coverage=0, min_y_coverage=0, min_neighbor_fraction=0,
                        min_shelf_occupancy=0, min_region_occupancy=0, reserve_blocker_front=False)
        # This distractor is outside A0 yet covers the wide Blocker's front.
        validate_sample(sample, catalog, rules)
        enabled = replace(rules, reserve_blocker_front=True)
        with self.assertRaisesRegex(ValueError, "Blocker frontal-view corridor"):
            validate_sample(sample, catalog, enabled)
        positions = dict(sample.positions)
        positions[2] = (-.57, .22, 1.05)
        validate_sample(replace(sample, positions=positions), catalog, enabled)

    def test_clear_endpoint_does_not_hide_mid_translation_collision(self):
        catalog = tuple(ObjectGeometry(str(index), (-.025, -.025, 0), (.025, .025, .05)) for index in range(4))
        sample = SceneSample((0, 1, 2, 3), 0, 1,
            {0: (-.82, 0, 1.05), 1: (-.70, 0, 1.05), 2: (-.70, .14, 1.05), 3: (-.84, -.18, 1.05)},
            {index: 0.0 for index in range(4)}, (-.785, -.51, -.06, .06), (-.70, .25))
        rules = replace(self.rules, min_objects=4, max_objects=4, min_asset_types=4, max_asset_types=4,
                        min_x_coverage=0, min_y_coverage=0, min_neighbor_fraction=0,
                        min_shelf_occupancy=0, min_region_occupancy=0)
        with self.assertRaisesRegex(ValueError, "path is obstructed"):
            validate_sample(sample, catalog, rules)
        positions = dict(sample.positions)
        positions[2] = (-.70, -.14, 1.05)
        validate_sample(replace(sample, positions=positions), catalog, rules)

    def test_pivot_sweep_catches_obstacle_between_clear_endpoints(self):
        bar = ObjectGeometry("bar", (-.10, -.01, 0), (.10, .01, .1))
        motion = PlanarMotion(PlanarPose(0, 0, 0), PlanarPose(-.11, .09, math.pi / 2), (-.10, -.01))
        obstacle = rectangle_polygon((.023, .035, .133, .145))
        self.assertTrue(polygons_separated(footprint_polygon(bar, (0, 0), 0), obstacle))
        self.assertTrue(polygons_separated(footprint_polygon(bar, (-.11, .09), math.pi / 2), obstacle))
        self.assertTrue(any(not polygons_separated(sweep, obstacle) for sweep in motion_sweeps(bar, motion)))

    def test_rotation_envelopes_include_mid_interval_corner_arcs(self):
        bar = ObjectGeometry("bar", (-.10, -.01, 0), (.10, .01, .1))
        motion = PlanarMotion(PlanarPose(0, 0, 0), PlanarPose(.03, .02, math.pi / 2))
        sweeps = motion_sweeps(bar, motion)
        for step in range(18):
            fraction = (step + .5) / 18
            for x, y in footprint_polygon(bar, (.03 * fraction, .02 * fraction), fraction * math.pi / 2):
                self.assertFalse(polygons_separated(sweeps[step], rectangle_polygon((x - 1e-10, x + 1e-10, y - 1e-10, y + 1e-10))))

    def test_polygon_distance_uses_true_corner_gap(self):
        first = rectangle_polygon((0, 1, 0, 1))
        self.assertAlmostEqual(polygon_distance(first, rectangle_polygon((1.03, 2, 1.04, 2))), .05)

    def test_role_overlap_support_and_access_metadata_are_checked(self):
        sample = sample_scene(self.catalog, self.rules, random.Random(14))
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_sample(replace(sample, blocker_id=sample.target_id), self.catalog, self.rules)
        positions = dict(sample.positions)
        surrounding = sample.active_ids[2]
        second_surrounding = sample.active_ids[3]
        positions[surrounding] = (-.7, 0, positions[surrounding][2])
        positions[second_surrounding] = (-.7, 0, positions[second_surrounding][2])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sample(replace(sample, positions=positions), self.catalog, self.rules)
        positions[surrounding] = (self.rules.x_range[1], 0, positions[surrounding][2])
        with self.assertRaisesRegex(ValueError, "outside supported"):
            validate_sample(replace(sample, positions=positions), self.catalog, self.rules)
        with self.assertRaisesRegex(ValueError, "fixed"):
            validate_sample(replace(sample, access_region=(-.85, -.5, -.1, .1)), self.catalog, self.rules)


if __name__ == "__main__":
    unittest.main()
