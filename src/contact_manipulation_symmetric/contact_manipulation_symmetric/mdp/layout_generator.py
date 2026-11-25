"""Construct shelf layouts with a geometric witness for each allowed exit."""

from collections import Counter
from functools import lru_cache
from itertools import combinations_with_replacement
import math
from typing import NamedTuple

import numpy as np

from .curriculums import layout_difficulty
from .layout_geometry import _dilated_cut_interval, inscribed_radius, lateral_cut_blocked


# RH56E2 collision meshes through coupled joint limits and the nominal
# approach rotation-vector range (10 degrees per axis), rounded up to mm.
HAND_APPROACH_WIDTH_METERS = 0.202


class _Guard(NamedTuple):
    name: str
    yaw: float
    y: float
    min_x: float
    depth: float
    cut_reach: float


def _guard_profiles(shapes_by_type, available, side, inner, cut, radius, gap, edge):
    """Find a feasible angle per asset from its actual convex collision footprint."""
    guards = []
    min_reach = max(gap*.5, edge-radius, 0.)
    for name, orientations in shapes_by_type.items():
        if available[name] <= 0:
            continue
        for yaw, shape in orientations:
            x0, x1, y0, y1 = shape["bounds"]
            polygon = shape["collision"] * (1, side)
            inside_edge = y0 if side == 1 else -y1
            center_y = (polygon[:, 1].min() + polygon[:, 1].max())*.5
            outward_y = max(inner-inside_edge, cut-center_y)
            covered = _dilated_cut_interval(polygon, cut-outward_y, radius)
            if covered is None:
                continue
            # The actual expanded polygon covers this much beyond both ends
            # of the occupied depth on the cut. Adjacent reaches can bridge
            # a gap without treating a rounded object's enclosing box as solid.
            reach = min(x0-covered[0], covered[1]-x1)-1.e-8
            if reach >= min_reach:
                guards.append(_Guard(name, yaw, side*outward_y, x0, x1-x0, reach))
                break
    return guards


@lru_cache(maxsize=64)
def _wall_combinations(type_count, body_count):
    combinations = tuple(combinations_with_replacement(range(type_count), body_count))
    return np.asarray(combinations, dtype=np.intp).reshape(-1, body_count)


def _choose_wall(guards, available, lower, upper, gap, budget, fixed_types, max_types, rng):
    """Sample an asset and fit a rail containing it when geometry allows."""
    if not guards or upper < lower:
        return None
    depths = np.array([g.depth for g in guards])
    reaches = np.array([g.cut_reach for g in guards])
    max_count = min(budget, int((upper+gap)/(depths.min()+gap)))
    preferred = rng.randrange(len(guards))
    fallback = None
    for count in range(1, max_count+1):
        combinations = _wall_combinations(len(guards), count)
        depths_sum = depths[combinations].sum(axis=1)
        low = np.maximum(lower, depths_sum+(count-1)*gap)
        high = np.minimum(upper, depths_sum+(count-1)*2*reaches[combinations].min(axis=1))
        valid = np.flatnonzero(low <= high+1.e-10)
        preferred_rows = (combinations[valid] == preferred).any(axis=1)
        for indices in (valid[preferred_rows], valid[~preferred_rows] if fallback is None else ()):
            for index in rng.sample(list(indices), len(indices)):
                pieces = [guards[i] for i in combinations[index]]
                names = Counter(g.name for g in pieces)
                if len(fixed_types | names.keys()) > max_types or any(n > available[name] for name, n in names.items()):
                    continue
                rng.shuffle(pieces)
                choice = (pieces, rng.uniform(low[index], high[index]))
                if preferred in combinations[index]:
                    return choice
                fallback = choice
                break
    return fallback


def _projected_blocker_offset(target_shape, blocker_shape, tx, bx, camera_x, side):
    """Signed Blocker center intervals for 10–50% target frontal overlap."""
    target = target_shape["visual"]
    u = side * target[:, 1] / (camera_x-tx-target[:, 0])
    target_low, target_high = float(u.min()), float(u.max())
    target_width = target_high-target_low
    blocker = blocker_shape["visual"]
    denom = camera_x-bx-blocker[:, 0]
    y = side * blocker[:, 1]
    lo = max(float(np.max((target_high-.5*target_width)*denom-y)),
             float(np.min(target_high*denom-y)))
    hi = float(np.max((target_high-.1*target_width)*denom-y))
    return lo, hi


def populate_layout(builder, layout, hand_width):
    """Choose constructible parameter domains before committing a scene."""
    from .events import PlanarMotion, PlanarPose, _path_supported_clear, _sequential_inflate, _sequential_translate

    near, near_status, far_status = layout
    if near not in (-1, 1):
        raise ValueError("Layout near side must be -1 or +1.")
    difficulty = layout_difficulty(near_status, far_status)
    if not math.isfinite(hand_width) or hand_width <= 0.0:
        raise ValueError("Layout curriculum needs a positive Hand approach width.")
    rules, rng = builder.rules, builder.rng
    gap, edge = rules.gap, rules.edge_margin
    shelf = (*rules.x_range, *rules.y_range)
    x0, x1, y0, y1 = (shelf[0]+edge, shelf[1]-edge, shelf[2]+edge, shelf[3]-edge)
    builder.shelf = (x0, x1, y0, y1)
    builder.by_type = {}
    for index, geometry in enumerate(builder.geometries):
        builder.by_type.setdefault(geometry.asset_name or geometry.name, []).append(index)
    builder.types = tuple(builder.by_type)
    asset_names = {i: name for name, ids in builder.by_type.items() for i in ids}
    sides = ((near, near_status), (-near, far_status))
    yaw = {name: rng.uniform(-math.pi, math.pi) for name in builder.types}
    guard_shapes = {name: [(angle, builder._shape(ids[0], angle))
                           for angle in (yaw[name], 0., math.pi/2, math.pi, -math.pi/2)]
                    for name, ids in builder.by_type.items()}
    targets = (builder.requested_target_id,) if builder.requested_target_id is not None else tuple(
        ids[0] for name, ids in builder.by_type.items() if name != builder.blocker_asset)
    blockers = (builder.requested_blocker_id,) if builder.requested_blocker_id is not None else tuple(
        ids[0] for name, ids in builder.by_type.items() if builder.blocker_asset is None or name == builder.blocker_asset)
    pairs = [(t, b) for t in targets for b in blockers if asset_names[t] != asset_names[b]]
    rng.shuffle(pairs)
    camera_x = x1+.75
    # Sample one continuous parameter vector; compatibility tests below select
    # its geometric domain before allocating any background, camera or IK work.
    depth_fraction, offset_fraction, lateral_fraction, turn_fraction = [rng.random() for _ in range(4)]
    for target, blocker in pairs:
        target_name, blocker_name = asset_names[target], asset_names[blocker]
        tyaw, byaw = yaw[target_name], yaw[blocker_name]
        t, b = builder._shape(target, tyaw), builder._goal_shape(blocker, byaw)
        tx0, tx1, ty0, ty1 = t["bounds"]
        bx0, bx1, by0, by1 = b["bounds"]
        width = max(t["visual_bounds"][3]-t["visual_bounds"][2], hand_width)+2*builder.margin
        center = (t["visual_bounds"][2]+t["visual_bounds"][3])*.5
        access_y = (center-width*.5, center+width*.5)
        radius = inscribed_radius(b["collision"], (0.0, 0.0))
        if radius < edge*.6:
            continue
        # Role depth must leave space to move Blocker ahead of the guard run.
        tx = x0-tx0
        begin_x = tx+tx1+gap-bx0
        end_x = x1-bx1-2*gap
        forward_margin = max(2*gap, (bx1-bx0)*.20)
        front_limit = end_x+bx0-gap
        initial_hi = min(end_x, front_limit-forward_margin-gap) if any(s == "front" for _, s in sides) else end_x
        if initial_hi <= begin_x+gap:
            continue
        bx = begin_x+(initial_hi-begin_x)*depth_fraction
        # A modest common translation makes the role depth vary while retaining
        # the available guard and turning domain.
        tx += (bx-begin_x)*.45
        offset_lo, offset_hi = _projected_blocker_offset(t, b, tx, bx, camera_x, near)
        offset_lo = max(offset_lo, near*center+gap)
        cut_y = {}
        for side, status in sides:
            if status == "direct":
                continue
            boundary = access_y[1] if side == 1 else -access_y[0]
            outward_t = ty1 if side == 1 else -ty0
            outward_b = by1 if side == 1 else -by0
            cut_y[side] = boundary + radius - .25*gap
            # The wall can start outside the Blocker's starting silhouette,
            # while its dilation still reaches the center cut at every yaw.
            wall_limit = cut_y[side] + .60*radius
            if max(boundary, outward_t)+gap > wall_limit:
                break
            if side == near:
                offset_hi = min(offset_hi, wall_limit-outward_b-gap, cut_y[side]-gap)
            else:
                offset_lo = max(offset_lo, outward_b+gap-wall_limit, -cut_y[side]+gap)
        else:
            if offset_lo >= offset_hi:
                continue
            offset = offset_lo+(offset_hi-offset_lo)*offset_fraction
            by = near*offset
            available = Counter({name: len(ids) for name, ids in builder.by_type.items()})
            available[blocker_name] -= 1
            available[target_name] = 0
            fixed_types = {target_name, blocker_name}
            guard_budget = rules.max_objects-2
            walls = {}
            for side, status in sides:
                if status == "direct":
                    continue
                boundary = access_y[1] if side == 1 else -access_y[0]
                outward_t = ty1 if side == 1 else -ty0
                outward_b = by1 if side == 1 else -by0
                inner = max(boundary, outward_t, side*by+outward_b)+gap
                profiles = _guard_profiles(guard_shapes, available, side, inner, cut_y[side], radius, gap, edge)
                upper = (x1-x0) if status == "blocked" else front_limit-x0
                lower = upper if status == "blocked" else bx+forward_margin-x0
                selected = _choose_wall(profiles, available, lower, upper, gap, guard_budget,
                                        fixed_types, rules.max_asset_types, rng)
                if selected is None:
                    break
                walls[side] = selected
                names = [guard.name for guard in selected[0]]
                available.subtract(names)
                fixed_types.update(names)
                guard_budget -= len(names)
            if len(walls) != sum(status != "direct" for _, status in sides):
                continue

            front_ends = [x0+walls[side][1] for side, status in sides if status == "front"]
            turn_min = max(front_ends, default=bx)+gap-bx0
            turn_max = x1-bx1-gap
            if front_ends and turn_min >= turn_max:
                continue
            turn_x = turn_min+(turn_max-turn_min)*turn_fraction
            translated = [(target, tx, 0., tyaw), (blocker, bx, by, byaw)]
            goal_rows, lane_rectangles, closed = [], [], []
            guards_by_side = {}
            # Allocate a distinct physical object for each guard.
            used = {target, blocker}
            for side, status in sides:
                boundary = access_y[1] if side == 1 else -access_y[0]
                inward_b = -by0 if side == 1 else by1
                if status in ("front", "blocked"):
                    pieces, span = walls[side]
                    spacing = (span-sum(piece.depth for piece in pieces))/max(1, len(pieces)-1)
                    wall_x = x0
                    guards_by_side[str(side)] = []
                    for piece in pieces:
                        guard = next(i for i in builder.by_type[piece.name] if i not in used)
                        used.add(guard)
                        guards_by_side[str(side)].append(guard)
                        translated.append((guard, wall_x-piece.min_x, piece.y, piece.yaw))
                        wall_x += piece.depth+spacing
                    cutoff = x0+span if status == "front" else None
                    closed.append((side*cut_y[side], cutoff))
                if status == "blocked":
                    continue
                # Same current yaw witnesses a feasible solution. Steering or
                # rotating elsewhere in the available space remains possible.
                goal_x = bx if status == "direct" else turn_x
                goal_y = side*(boundary+gap+inward_b)
                points = ((bx, by), (goal_x, goal_y)) if status == "direct" else (
                    (bx, by), (goal_x, by), (goal_x, goal_y))
                pocket = _sequential_translate(b["bounds"], goal_x, goal_y)
                goal_rows.append((side, pocket, (goal_x, goal_y), points))
                for au, av in zip(points, points[1:]):
                    lane_rectangles.append((min(au[0], av[0])+bx0, max(au[0], av[0])+bx1,
                                            min(au[1], av[1])+by0, max(au[1], av[1])+by1))

            extents = [access_y]
            for index, _, y, angle in translated:
                shape = builder._shape(index, angle)
                extents.append((y+shape["bounds"][2], y+shape["bounds"][3]))
            extents.extend(r[2:4] for r in lane_rectangles)
            shift_min = y0 - min(low for low, _ in extents)
            shift_max = y1 - max(high for _, high in extents)
            if shift_min+gap >= shift_max:
                continue
            dy = shift_min+(shift_max-shift_min)*lateral_fraction
            world = [(index, x, y+dy, angle) for index, x, y, angle in translated]
            obstacles = [tuple(map(tuple, builder._shape(index, angle)["collision"]+(x, y)))
                         for index, x, y, angle in world if index != blocker]
            if any(not lateral_cut_blocked(shelf, cut+dy, radius, obstacles, x_upper=cutoff)
                   for cut, cutoff in closed):
                continue
            # Check positive witnesses with the complete swept footprint. For
            # impossibility use the inscribed disk and all possible yaws, even
            # for destinations outside the reserved pockets.
            paths = {str(side): [(x, y+dy) for x, y in points]
                     for side, _, _, points in goal_rows}
            if any(not _path_supported_clear(builder.geometries[blocker],
                       tuple(PlanarMotion(PlanarPose(*a, byaw), PlanarPose(*p, byaw))
                             for a, p in zip(points, points[1:])), obstacles, rules)
                   for points in paths.values()):
                continue
            break
    else:
        raise ValueError(f"No constructive parameter domain for layout {layout} on this shelf and asset pool.")

    builder.target, builder.blocker = target, blocker
    builder.target_type, builder.blocker_type = target_name, blocker_name
    builder.target_yaw, builder.blocker_yaw = tyaw, byaw
    builder.camera = (camera_x, dy, rules.shelf_top+.18, .5, -.5, -.5, .5)
    access = (tx+t["bounds"][1]+rules.approach_margin, x1, access_y[0]+dy, access_y[1]+dy)
    # Access must initially have a positive Blocker overlap. The initial
    # placement calculation permits the visual sliver used in the task.
    state = builder.state
    state.update(target_id=target, blocker_id=blocker, clearance=access, camera_pose=builder.camera,
                 shelf_bounds=builder.shelf)
    guard_ids = []
    for index, x, y, angle in world:
        builder._add(index, x, y, angle)
        if index not in (target, blocker):
            guard_ids.append(index)
    # Additional padding keeps the swept convex polygons SAT-separated even
    # when a rotated background body's AABB just touches the packing exclusion.
    state["protected_layout_rectangles"] = [
        _sequential_inflate(_sequential_translate(region, 0, dy), gap) for region in lane_rectangles
    ]
    # Preserve the native front-of-Blocker clearance used by starts as well.
    state["blocker_front_corridor"] = (bx+bx1, shelf[1], by+dy+by0-gap, by+dy+by1+gap)
    for _, pocket, goal, _ in goal_rows:
        pocket = _sequential_translate(pocket, 0, dy)
        state["pockets"].append(pocket)
        state["pocket_goals"].append((goal[0], goal[1]+dy, b["z"], byaw))
        state["pocket_front_corridors"].append((pocket[1], shelf[1], pocket[2], pocket[3]))
    for region in [builder._world_rect(i) for i in guard_ids]+state["pockets"]:
        # Leave the frontal projection clear all the way from the feature to
        # the shelf lip; the native fixed camera samples this view.
        x, end, low, high = region
        front = shelf[1]
        ratios = ((camera_x-front)/(camera_x-x), (camera_x-front)/(camera_x-end))
        projected = [dy+(y-dy)*ratio for y in (low, high) for ratio in ratios]
        state["protected_layout_rectangles"].append((end, front, min(low, *projected), max(high, *projected)))
    state["metadata"].update(layout_difficulty=difficulty, layout_near_side=near,
        layout_near_status=near_status, layout_far_status=far_status, layout_paths=paths,
        layout_guards=guard_ids, layout_guards_by_side=guards_by_side,
        layout_barriers=[(cut+dy, limit) for cut, limit in closed],
        layout_inscribed_radius_m=radius, hand_width_m=hand_width,
        target_asset=target_name, blocker_asset=blocker_name,
        target_frontal_world_width_m=t["visual_bounds"][3]-t["visual_bounds"][2])
