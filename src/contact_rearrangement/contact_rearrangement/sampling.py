"""Seeded, simulator-independent sampling for a single-blocker shelf baseline.

Geometry is the conservative local AABB in each asset's canonical upright frame.
Returned XY positions describe its rotated footprint center, while Z describes
the asset root. The caller must subtract the rotated local AABB center from XY
when writing root poses. The clearance witness is sampler metadata, not a policy
goal: it proves a supported collision-free planar pose path exists, without
proving robot reachability or physical control success.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

Point = tuple[float, float]
Polygon = tuple[Point, ...]
Region = tuple[float, float, float, float]
_EPS = 1.0e-9


@dataclass(frozen=True)
class ObjectGeometry:
    name: str
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    asset_name: str | None = None


@dataclass(frozen=True)
class SpawnRules:
    x_range: tuple[float, float] = (-0.89, -0.51)
    y_range: tuple[float, float] = (-0.455, 0.455)
    shelf_top: float = 1.05
    ceiling: float = 1.35
    min_objects: int = 4
    max_objects: int = 10
    all_asset_types: bool = False
    edge_margin: float = 0.015
    gap: float = 0.003
    hand_width: float = 0.10
    approach_margin: float = 0.01
    min_blocked_ratio: float = 0.30
    min_x_coverage: float = 0.55
    min_y_coverage: float = 0.65
    neighbor_distance: float = 0.04
    min_neighbor_fraction: float = 0.50
    max_attempts: int = 300


@dataclass(frozen=True)
class PlanarPose:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class PlanarMotion:
    """One supported SE(2) motion; ``pivot`` specifies an exact fixed-pivot arc.

    Without a pivot the bounding center interpolates linearly while yaw changes.
    A pivot rotates the entire footprint about that world XY point instead.
    """

    start: PlanarPose
    end: PlanarPose
    pivot: Point | None = None


@dataclass(frozen=True)
class SceneMetrics:
    x_coverage: float
    y_coverage: float
    shelf_occupancy: float
    local_neighbor_fraction: float


@dataclass(frozen=True)
class SceneSample:
    active_ids: tuple[int, ...]
    target_id: int
    blocker_id: int
    positions: dict[int, tuple[float, float, float]]
    yaws: dict[int, float]
    access_region: Region
    clearance_witness: Point
    clearance_path: tuple[PlanarMotion, ...] = ()
    clearance_kind: str = "translation"


def rectangle_polygon(region: Region) -> Polygon:
    """Return a counterclockwise polygon for (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = region
    return ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))


def footprint_polygon(geometry: ObjectGeometry, center_xy: Point, yaw: float) -> Polygon:
    """Rotate the conservative upright footprint about its own bounding center."""
    hx = 0.5 * (geometry.upper[0] - geometry.lower[0])
    hy = 0.5 * (geometry.upper[1] - geometry.lower[1])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return tuple(
        (center_xy[0] + cosine * x - sine * y, center_xy[1] + sine * x + cosine * y)
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    )


def polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(sum(a[0] * b[1] - a[1] * b[0] for a, b in zip(polygon, polygon[1:] + polygon[:1]))) * 0.5


def polygons_separated(first: Polygon, second: Polygon, gap: float = 0.0) -> bool:
    """SAT separation with at least ``gap`` on one normalized edge axis.

    Requiring a gap on an edge axis is conservative for diagonal corner gaps.
    Boundary touching counts as separated when gap is zero.
    """
    for polygon in (first, second):
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if length <= _EPS:
                continue
            axis = (-dy / length, dx / length)
            projection_a = [p[0] * axis[0] + p[1] * axis[1] for p in first]
            projection_b = [p[0] * axis[0] + p[1] * axis[1] for p in second]
            if max(projection_a) + gap <= min(projection_b) + _EPS:
                return True
            if max(projection_b) + gap <= min(projection_a) + _EPS:
                return True
    return False


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def overlap_area(first: Polygon, second: Polygon) -> float:
    """Clip counterclockwise convex polygons and return their intersection area."""
    output = list(first)
    for start, end in zip(second, second[1:] + second[:1]):
        edge = (end[0] - start[0], end[1] - start[1])

        def signed_distance(point: Point) -> float:
            return _cross(edge, (point[0] - start[0], point[1] - start[1]))

        input_points, output = output, []
        if not input_points:
            break
        previous = input_points[-1]
        previous_distance = signed_distance(previous)
        for current in input_points:
            current_distance = signed_distance(current)
            previous_inside, current_inside = previous_distance >= -_EPS, current_distance >= -_EPS
            if previous_inside != current_inside:
                fraction = previous_distance / (previous_distance - current_distance)
                output.append(
                    (
                        previous[0] + fraction * (current[0] - previous[0]),
                        previous[1] + fraction * (current[1] - previous[1]),
                    )
                )
            if current_inside:
                output.append(current)
            previous, previous_distance = current, current_distance
    return polygon_area(tuple(output))


def _convex_hull(points: Sequence[Point]) -> Polygon:
    unique = sorted(set(points))

    def turn(origin: Point, first: Point, second: Point) -> float:
        return _cross((first[0] - origin[0], first[1] - origin[1]), (second[0] - origin[0], second[1] - origin[1]))

    lower: list[Point] = []
    upper: list[Point] = []
    for point in unique:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], point) <= _EPS:
            lower.pop()
        lower.append(point)
    for point in reversed(unique):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], point) <= _EPS:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _swept_polygon(polygon: Polygon, displacement: Point) -> Polygon:
    translated = tuple((x + displacement[0], y + displacement[1]) for x, y in polygon)
    return _convex_hull(polygon + translated)


def _half_extents(geometry: ObjectGeometry, yaw: float) -> Point:
    hx = 0.5 * (geometry.upper[0] - geometry.lower[0])
    hy = 0.5 * (geometry.upper[1] - geometry.lower[1])
    cosine, sine = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (cosine * hx + sine * hy, sine * hx + cosine * hy)


def _supported(polygon: Polygon, rules: SpawnRules) -> bool:
    return all(
        rules.x_range[0] + rules.edge_margin - _EPS <= x <= rules.x_range[1] - rules.edge_margin + _EPS
        and rules.y_range[0] + rules.edge_margin - _EPS <= y <= rules.y_range[1] - rules.edge_margin + _EPS
        for x, y in polygon
    )


def _cluster_measurements(polygons: Sequence[Polygon]) -> Point:
    points = tuple(point for polygon in polygons for point in polygon)
    x_span = max(point[0] for point in points) - min(point[0] for point in points)
    y_span = max(point[1] for point in points) - min(point[1] for point in points)
    return y_span, sum(polygon_area(polygon) for polygon in polygons) / (x_span * y_span)


def cluster_metrics(sample: SceneSample, geometries: Sequence[ObjectGeometry]) -> Point:
    """Return legacy diagnostic Y span and fill of the tight bounding rectangle.

    Neither quantity is an acceptance threshold. Only active footprints count;
    the reserved future rearrangement space
    is excluded. Footprints are conservative boxes rather than occupied mesh
    surface area, so this metric is a reproducible spawn proxy for clutter.
    """
    return _cluster_measurements(tuple(
        footprint_polygon(geometries[index], sample.positions[index][:2], sample.yaws[index])
        for index in sample.active_ids
    ))


def _asset_type(geometry: ObjectGeometry) -> str:
    return geometry.asset_name or geometry.name


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq))
    return math.hypot(point[0] - start[0] - fraction * dx, point[1] - start[1] - fraction * dy)


def polygon_distance(first: Polygon, second: Polygon) -> float:
    """Euclidean boundary distance for convex polygons (zero if intersecting)."""
    if not polygons_separated(first, second):
        return 0.0
    return min(_point_segment_distance(point, start, end)
               for points, edges in ((first, second), (second, first))
               for point in points for start, end in zip(edges, edges[1:] + edges[:1]))


def _scene_measurements(polygons: Sequence[Polygon], rules: SpawnRules) -> SceneMetrics:
    points = tuple(point for polygon in polygons for point in polygon)
    usable_x = rules.x_range[1] - rules.x_range[0] - 2 * rules.edge_margin
    usable_y = rules.y_range[1] - rules.y_range[0] - 2 * rules.edge_margin
    neighbors = sum(any(other != index and polygon_distance(polygon, polygons[other])
                        <= rules.neighbor_distance + _EPS for other in range(len(polygons)))
                    for index, polygon in enumerate(polygons))
    return SceneMetrics(
        (max(x for x, _ in points) - min(x for x, _ in points)) / usable_x,
        (max(y for _, y in points) - min(y for _, y in points)) / usable_y,
        sum(polygon_area(polygon) for polygon in polygons) / (usable_x * usable_y),
        neighbors / len(polygons),
    )


def scene_metrics(sample: SceneSample, geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> SceneMetrics:
    """Whole-shelf coverage/occupancy and independent local-clutter measurements."""
    return _scene_measurements(tuple(footprint_polygon(geometries[index], sample.positions[index][:2],
                                                       sample.yaws[index]) for index in sample.active_ids), rules)


def _pose_polygon(geometry: ObjectGeometry, pose: PlanarPose) -> Polygon:
    return footprint_polygon(geometry, (pose.x, pose.y), pose.yaw)


def _motion_pose(motion: PlanarMotion, fraction: float) -> PlanarPose:
    delta = motion.end.yaw - motion.start.yaw
    yaw = motion.start.yaw + fraction * delta
    if motion.pivot is None:
        return PlanarPose(motion.start.x + fraction * (motion.end.x - motion.start.x),
                          motion.start.y + fraction * (motion.end.y - motion.start.y), yaw)
    px, py = motion.pivot
    dx, dy = motion.start.x - px, motion.start.y - py
    cosine, sine = math.cos(fraction * delta), math.sin(fraction * delta)
    return PlanarPose(px + cosine * dx - sine * dy, py + sine * dx + cosine * dy, yaw)


def motion_sweeps(geometry: ObjectGeometry, motion: PlanarMotion) -> tuple[Polygon, ...]:
    """Enclose every footprint during a motion, including rotation between poses.

    Each rotation interval is <= 5 degrees. Its endpoint convex hull is expanded
    by R*(1-cos(angle/2)), the maximum corner-arc departure from its chord. The
    square expansion contains that circular error bound. Fixed-pivot R measures
    corners from the pivot; ordinary SE(2) R measures corners from the center.
    Thus checking only clear endpoints cannot conceal a mid-rotation collision.
    """
    delta = motion.end.yaw - motion.start.yaw
    start_polygon = _pose_polygon(geometry, motion.start)
    if abs(delta) <= _EPS:
        return (_convex_hull(start_polygon + _pose_polygon(geometry, motion.end)),)
    origin = motion.pivot or (motion.start.x, motion.start.y)
    radius = max(math.hypot(x - origin[0], y - origin[1]) for x, y in start_polygon)
    divisions = max(1, math.ceil(abs(delta) / math.radians(5.0)))
    padding = radius * (1.0 - math.cos(abs(delta) / (2 * divisions)))
    envelopes = []
    previous = start_polygon
    for step in range(1, divisions + 1):
        current = _pose_polygon(geometry, _motion_pose(motion, step / divisions))
        hull = _convex_hull(previous + current)
        envelopes.append(_convex_hull(tuple((x + dx, y + dy) for x, y in hull
                                           for dx, dy in ((-padding, -padding), (padding, -padding),
                                                          (padding, padding), (-padding, padding)))))
        previous = current
    return tuple(envelopes)


def _path_sweeps(geometry: ObjectGeometry, path: Sequence[PlanarMotion]) -> tuple[Polygon, ...]:
    return tuple(polygon for motion in path for polygon in motion_sweeps(geometry, motion))


def _path_supported_clear(geometry: ObjectGeometry, path: Sequence[PlanarMotion], obstacles: Sequence[Polygon],
                          rules: SpawnRules) -> bool:
    return all(_supported(polygon, rules) and all(polygons_separated(polygon, obstacle, rules.gap)
                                                  for obstacle in obstacles)
               for polygon in _path_sweeps(geometry, path))


def _check_inputs(geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> None:
    values = (*rules.x_range, *rules.y_range, rules.shelf_top, rules.ceiling, rules.edge_margin,
              rules.gap, rules.hand_width, rules.approach_margin, rules.min_blocked_ratio,
              rules.min_x_coverage, rules.min_y_coverage, rules.neighbor_distance, rules.min_neighbor_fraction)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Spawn rules must have finite dimensions and thresholds.")
    if not 2 <= rules.min_objects <= rules.max_objects <= len(geometries):
        raise ValueError("Active object limits must satisfy 2 <= min <= max <= catalog size.")
    if rules.max_attempts < 1 or not 0.0 < rules.min_blocked_ratio <= 1.0:
        raise ValueError("max_attempts must be positive and min_blocked_ratio must be in (0, 1].")
    if any(not 0.0 <= value <= 1.0 for value in
           (rules.min_x_coverage, rules.min_y_coverage, rules.min_neighbor_fraction)):
        raise ValueError("Coverage and neighbor fraction thresholds must be in [0, 1].")
    if min(rules.edge_margin, rules.gap, rules.approach_margin, rules.neighbor_distance) < 0 or rules.hand_width <= 0:
        raise ValueError("Margins, neighbor distance and gap must be nonnegative; hand_width must be positive.")
    if rules.x_range[1] - rules.x_range[0] <= 2.0 * rules.edge_margin:
        raise ValueError("Shelf X range has no usable support area.")
    if rules.y_range[1] - rules.y_range[0] <= 2.0 * rules.edge_margin:
        raise ValueError("Shelf Y range has no usable support area.")
    if rules.ceiling <= rules.shelf_top:
        raise ValueError("Ceiling must be above shelf support.")
    if len({geometry.name for geometry in geometries}) != len(geometries):
        raise ValueError("Catalog geometry names must be unique.")
    for geometry in geometries:
        if not all(math.isfinite(value) for value in (*geometry.lower, *geometry.upper)):
            raise ValueError(f"Nonfinite bounds for {geometry.name}.")
        if any(high <= low for low, high in zip(geometry.lower, geometry.upper)):
            raise ValueError(f"Bounds must have positive extent for {geometry.name}.")


def _validated_path(sample: SceneSample) -> tuple[PlanarMotion, ...]:
    start = PlanarPose(*sample.positions[sample.blocker_id][:2], sample.yaws[sample.blocker_id])
    # Legacy manually specified witnesses still describe a genuine translation.
    path = sample.clearance_path or (PlanarMotion(start, PlanarPose(*sample.clearance_witness, start.yaw)),)
    previous = start
    for motion in path:
        values = (motion.start.x, motion.start.y, motion.start.yaw, motion.end.x, motion.end.y, motion.end.yaw)
        if motion.pivot is not None:
            values += motion.pivot
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Clearance path poses and pivots must be finite.")
        if any(abs(a - b) > _EPS for a, b in zip((previous.x, previous.y, previous.yaw),
                                                (motion.start.x, motion.start.y, motion.start.yaw))):
            raise ValueError("Clearance path must start at the initial Blocker and remain continuous.")
        if motion.pivot is not None:
            endpoint = _motion_pose(motion, 1.0)
            if math.hypot(endpoint.x - motion.end.x, endpoint.y - motion.end.y) > _EPS:
                raise ValueError("Pivot motion endpoint must agree with its fixed-pivot arc.")
        previous = motion.end
    if any(abs(a - b) > _EPS for a, b in zip((previous.x, previous.y), sample.clearance_witness)):
        raise ValueError("Clearance witness must agree with the final path pose.")
    return path


def validate_sample(sample: SceneSample, geometries: Sequence[ObjectGeometry], rules: SpawnRules) -> None:
    """Raise ValueError if a scene violates role, coverage, clutter or SE(2) path rules."""
    _check_inputs(geometries, rules)
    ids = sample.active_ids
    if not rules.min_objects <= len(ids) <= rules.max_objects or len(set(ids)) != len(ids):
        raise ValueError("Active IDs must be unique and respect active object limits.")
    if any(index < 0 or index >= len(geometries) for index in ids):
        raise ValueError("Active IDs must index the catalog.")
    if sample.target_id not in ids or sample.blocker_id not in ids or sample.target_id == sample.blocker_id:
        raise ValueError("Target and Blocker must be distinct active objects.")
    target_type = _asset_type(geometries[sample.target_id])
    if sum(_asset_type(geometries[index]) == target_type for index in ids) != 1:
        raise ValueError("Target asset type must occur exactly once among active objects.")
    if rules.all_asset_types and { _asset_type(geometries[index]) for index in ids } != {
            _asset_type(geometry) for geometry in geometries}:
        raise ValueError("All-asset-types scenes must contain every catalog asset type.")
    if rules.all_asset_types and len(ids) != len({_asset_type(geometry) for geometry in geometries}):
        raise ValueError("All-asset-types scenes must contain exactly one instance per type.")
    if set(sample.positions) != set(ids) or set(sample.yaws) != set(ids):
        raise ValueError("Sampled positions and yaws must contain exactly the active objects.")
    polygons: dict[int, Polygon] = {}
    for index in ids:
        position, yaw, geometry = sample.positions[index], sample.yaws[index], geometries[index]
        if not all(math.isfinite(value) for value in (*position, yaw)):
            raise ValueError("Sampled poses must be finite.")
        expected_z = rules.shelf_top - geometry.lower[2]
        if abs(position[2] - expected_z) > _EPS:
            raise ValueError("Root Z must put the canonical lower bound on the shelf.")
        if position[2] + geometry.upper[2] > rules.ceiling - rules.gap + _EPS:
            raise ValueError("An active object lacks ceiling clearance.")
        polygons[index] = footprint_polygon(geometry, position[:2], yaw)
        if not _supported(polygons[index], rules):
            raise ValueError("An active footprint is outside supported shelf margins.")
    for first, second in combinations(ids, 2):
        if not polygons_separated(polygons[first], polygons[second], rules.gap):
            raise ValueError("Active footprints overlap or lack the required gap.")
    target, blocker = sample.target_id, sample.blocker_id
    target_front = max(x for x, _ in polygons[target])
    if min(x for x, _ in polygons[blocker]) < target_front + rules.gap - _EPS:
        raise ValueError("Blocker must be on the robot side of Target with a positive planar gap.")
    half_width = rules.hand_width * 0.5 + rules.approach_margin
    target_y = sample.positions[target][1]
    expected_access = (target_front + rules.approach_margin, rules.x_range[1],
                       target_y - half_width, target_y + half_width)
    if not all(math.isfinite(value) for value in sample.access_region):
        raise ValueError("Access region must be finite.")
    if any(abs(actual - expected) > _EPS for actual, expected in zip(sample.access_region, expected_access)):
        raise ValueError("Access region must stay fixed to the initial Target and +X approach.")
    if expected_access[0] >= expected_access[1]:
        raise ValueError("Access region has no positive length.")
    access = rectangle_polygon(sample.access_region)
    if overlap_area(polygons[blocker], access) / polygon_area(polygons[blocker]) < rules.min_blocked_ratio - _EPS:
        raise ValueError("Designated Blocker does not sufficiently block the initial access region.")
    for index in ids:
        if index not in (target, blocker) and not polygons_separated(polygons[index], access, rules.gap):
            raise ValueError("Surrounding objects must begin outside the initial access region.")
    if not all(math.isfinite(value) for value in sample.clearance_witness):
        raise ValueError("Clearance witness must be finite.")
    path = _validated_path(sample)
    final = _pose_polygon(geometries[blocker], path[-1].end)
    if not polygons_separated(final, access, rules.gap):
        raise ValueError("Clearance witness still intersects the access region.")
    for swept in _path_sweeps(geometries[blocker], path):
        if not _supported(swept, rules):
            raise ValueError("Blocker clearance path leaves supported shelf margins.")
        if any(index != blocker and not polygons_separated(swept, polygon, rules.gap)
               for index, polygon in polygons.items()):
            raise ValueError("Blocker clearance path is obstructed by another object.")
    metrics = _scene_measurements(tuple(polygons.values()), rules)
    if metrics.x_coverage < rules.min_x_coverage - _EPS or metrics.y_coverage < rules.min_y_coverage - _EPS:
        raise ValueError("Active footprints do not meet whole-shelf coverage thresholds.")
    if metrics.local_neighbor_fraction < rules.min_neighbor_fraction - _EPS:
        raise ValueError("Too few active footprints have a nearby clutter neighbor.")


def _clearance_candidates(geometry: ObjectGeometry, initial: PlanarPose, access: Region,
                          rules: SpawnRules, rng: random.Random) -> list[tuple[PlanarMotion, ...]]:
    """Finite translation, pivot, rotate/translate, and multistage proposals."""
    prefixes: list[tuple[PlanarMotion, ...]] = [()]
    for angle in (-math.pi / 2, math.pi / 2, -math.pi / 4, math.pi / 4):
        prefixes.append((PlanarMotion(initial, PlanarPose(initial.x, initial.y, initial.yaw + angle)),))
        for pivot in _pose_polygon(geometry, initial):
            provisional = PlanarMotion(initial, PlanarPose(0, 0, initial.yaw + angle), pivot)
            prefixes.append((PlanarMotion(initial, _motion_pose(provisional, 1.0), pivot),))
    # Move away from the rear Target before turning, then translate to clear A0.
    for dx in (0.015, 0.03):
        shifted = PlanarPose(initial.x + dx, initial.y, initial.yaw)
        for angle in (-math.pi / 2, math.pi / 2):
            prefixes.append((PlanarMotion(initial, shifted),
                             PlanarMotion(shifted, PlanarPose(shifted.x, shifted.y, shifted.yaw + angle))))
    candidates = []
    for prefix in prefixes:
        pose = prefix[-1].end if prefix else initial
        if prefix and polygons_separated(_pose_polygon(geometry, pose), rectangle_polygon(access), rules.gap):
            candidates.append(prefix)
        _, hy = _half_extents(geometry, pose.yaw)
        for y in (access[2] - hy - rules.gap - 0.001, access[3] + hy + rules.gap + 0.001):
            candidates.append((*prefix, PlanarMotion(pose, PlanarPose(pose.x, y, pose.yaw))))
    rng.shuffle(candidates)
    return candidates


def sample_scene(geometries: Sequence[ObjectGeometry], rules: SpawnRules, rng: random.Random,
                 target_id: int | None = None, blocker_id: int | None = None) -> SceneSample:
    """Choose episode roles/count once, then sample whole-shelf clutter around them.

    The Target species appears only once. Other species may repeat unless
    all_asset_types requests exactly one of every type. A conservative finite
    SE(2) witness reserves rearrangement room but is not an actor goal or proof
    of reachable physical manipulation. Camera visibility must be checked by
    future rendered instance masks; planar blockage is not a visibility test.
    """
    _check_inputs(geometries, rules)
    types = {_asset_type(geometry) for geometry in geometries}
    for index in (target_id, blocker_id):
        if index is not None and (not isinstance(index, int) or not 0 <= index < len(geometries)):
            raise ValueError("Explicit role IDs must index the catalog.")
    if target_id is not None and target_id == blocker_id:
        raise ValueError("Target and Blocker must be distinct instance IDs.")
    if target_id is None:
        choices = tuple(index for index, geometry in enumerate(geometries)
                        if blocker_id is None or _asset_type(geometry) != _asset_type(geometries[blocker_id]))
        if not choices:
            raise ValueError("Target requires an asset type different from Blocker.")
        target_id = rng.choice(choices)
    target_type = _asset_type(geometries[target_id])
    non_target = tuple(index for index, geometry in enumerate(geometries) if _asset_type(geometry) != target_type)
    if blocker_id is None:
        if not non_target:
            raise ValueError("Target uniqueness leaves no eligible Blocker.")
        blocker_id = rng.choice(non_target)
    if _asset_type(geometries[blocker_id]) == target_type:
        raise ValueError("Target asset type cannot also be the Blocker or a surrounding object.")
    target, blocker = target_id, blocker_id
    surrounding_pool = tuple(index for index in non_target if index != blocker)
    if rules.all_asset_types:
        active_count = len(types)
        if not rules.min_objects <= active_count <= rules.max_objects:
            raise ValueError("All-asset-types count must satisfy active object limits.")
        remaining_types = sorted(types - {target_type, _asset_type(geometries[blocker])})
    else:
        active_count = rng.randint(rules.min_objects, rules.max_objects)
        if active_count > len(non_target) + 1:
            raise ValueError("Target uniqueness leaves too few eligible instances for the episode count.")
        remaining_types = []
    for _ in range(rules.max_attempts):
        if rules.all_asset_types:
            surrounding_ids = [rng.choice(tuple(index for index in surrounding_pool
                                               if _asset_type(geometries[index]) == asset))
                               for asset in remaining_types]
            rng.shuffle(surrounding_ids)
        else:
            surrounding_ids = rng.sample(surrounding_pool, active_count - 2)
        ids = (target, blocker, *surrounding_ids)
        yaws = {index: rng.uniform(-math.pi, math.pi) for index in ids}
        extents = {index: _half_extents(geometries[index], yaws[index]) for index in ids}
        if any(rules.shelf_top + geometries[index].upper[2] - geometries[index].lower[2]
               > rules.ceiling - rules.gap for index in ids):
            continue
        target_hx, target_hy = extents[target]
        blocker_hx, blocker_hy = extents[blocker]
        pair_gap = rng.uniform(rules.gap + 0.002, rules.gap + 0.012)
        xmin = rules.x_range[0] + rules.edge_margin + target_hx
        xmax = rules.x_range[1] - rules.edge_margin - target_hx - 2 * blocker_hx - pair_gap
        half_width = rules.hand_width / 2 + rules.approach_margin
        y_extent = max(target_hy, half_width)
        ymin, ymax = rules.y_range[0] + rules.edge_margin + y_extent, rules.y_range[1] - rules.edge_margin - y_extent
        if xmin > xmax or ymin > ymax:
            continue
        target_x, target_y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        blocker_x = target_x + target_hx + blocker_hx + pair_gap
        blocker_ymin = max(rules.y_range[0] + rules.edge_margin + blocker_hy, target_y - half_width - blocker_hy)
        blocker_ymax = min(rules.y_range[1] - rules.edge_margin - blocker_hy, target_y + half_width + blocker_hy)
        if blocker_ymin > blocker_ymax:
            continue
        blocker_y = rng.uniform(blocker_ymin, blocker_ymax)
        access_region = (target_x + target_hx + rules.approach_margin, rules.x_range[1],
                         target_y - half_width, target_y + half_width)
        access = rectangle_polygon(access_region)
        positions = {target: (target_x, target_y, rules.shelf_top - geometries[target].lower[2]),
                     blocker: (blocker_x, blocker_y, rules.shelf_top - geometries[blocker].lower[2])}
        polygons = {index: footprint_polygon(geometries[index], positions[index][:2], yaws[index])
                    for index in (target, blocker)}
        if overlap_area(polygons[blocker], access) / polygon_area(polygons[blocker]) < rules.min_blocked_ratio:
            continue
        initial = PlanarPose(blocker_x, blocker_y, yaws[blocker])
        path = next((candidate for candidate in _clearance_candidates(geometries[blocker], initial, access_region, rules, rng)
                     if _path_supported_clear(geometries[blocker], candidate, (polygons[target],), rules)), None)
        if path is None:
            continue
        sweeps = _path_sweeps(geometries[blocker], path)
        for order, index in enumerate(ids[2:]):
            hx, hy = extents[index]
            placed = False
            for placement_attempt in range(160):
                # Early stratified proposals seek both shelf sides; later trials
                # can use the entire support. Additional objects often form pairs.
                global_proposal = order < 2 or rng.random() < 0.35
                if global_proposal:
                    x_low, x_high = rules.x_range[0] + rules.edge_margin + hx, rules.x_range[1] - rules.edge_margin - hx
                    y_low, y_high = rules.y_range[0] + rules.edge_margin + hy, rules.y_range[1] - rules.edge_margin - hy
                    if x_low > x_high or y_low > y_high:
                        break
                    if order < 2 and placement_attempt < 100:
                        fraction = rng.uniform(0.0, 0.25) if order == 0 else rng.uniform(0.75, 1.0)
                        y = y_low + fraction * (y_high - y_low)
                        # Opposing X bands also encourage depth-wide coverage.
                        fraction_x = rng.uniform(0.0, 0.40) if order == 0 else rng.uniform(0.60, 1.0)
                        x = x_low + fraction_x * (x_high - x_low)
                    else:
                        x, y = rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)
                else:
                    anchor = rng.choice(tuple(positions))
                    ax, ay, _ = positions[anchor]
                    ahx, ahy = extents[anchor]
                    axis, sign = rng.randrange(2), rng.choice((-1, 1))
                    gap = rng.uniform(rules.gap + 0.001, rules.gap + 0.025)
                    if axis == 0:
                        x, y = ax + sign * (ahx + hx + gap), ay + rng.uniform(-max(ahy, hy), max(ahy, hy))
                    else:
                        x, y = ax + rng.uniform(-max(ahx, hx), max(ahx, hx)), ay + sign * (ahy + hy + gap)
                polygon = footprint_polygon(geometries[index], (x, y), yaws[index])
                if not _supported(polygon, rules) or not polygons_separated(polygon, access, rules.gap):
                    continue
                if any(not polygons_separated(polygon, sweep, rules.gap) for sweep in sweeps):
                    continue
                if any(not polygons_separated(polygon, other, rules.gap) for other in polygons.values()):
                    continue
                positions[index] = (x, y, rules.shelf_top - geometries[index].lower[2])
                polygons[index] = polygon
                placed = True
                break
            if not placed:
                break
        if len(positions) != len(ids):
            continue
        metrics = _scene_measurements(tuple(polygons.values()), rules)
        if metrics.x_coverage < rules.min_x_coverage or metrics.y_coverage < rules.min_y_coverage:
            continue
        if metrics.local_neighbor_fraction < rules.min_neighbor_fraction:
            continue
        kind = "pivot" if any(motion.pivot is not None for motion in path) else (
            "rotate_translate" if any(abs(motion.end.yaw - motion.start.yaw) > _EPS for motion in path)
            else "translation")
        sample = SceneSample(ids, target, blocker, positions, yaws, access_region,
                             (path[-1].end.x, path[-1].end.y), path, kind)
        validate_sample(sample, geometries, rules)
        return sample
    raise RuntimeError(
        f"Could not sample a whole-shelf, supported, collision-free, initially blocked and geometrically clearable scene "
        f"after {rules.max_attempts} attempts. Check fixed roles/count, catalog geometry, shelf size and spawn thresholds."
    )
