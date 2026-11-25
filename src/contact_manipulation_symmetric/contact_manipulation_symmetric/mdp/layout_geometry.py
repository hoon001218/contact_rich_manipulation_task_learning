"""Conservative planar separation proofs for constructive shelf layouts."""

import math

import numpy as np


_NUMERIC_MARGIN = 1.0e-10


def inscribed_radius(blocker_polygon, center):
    """Radius of a disk around ``center`` contained in every rotated footprint.

    The convex polygon may have either winding. A center on or outside its
    boundary cannot provide a positive disk certificate.
    """
    vertices = np.asarray(blocker_polygon, dtype=np.float64) - center
    following = np.roll(vertices, -1, axis=0)
    lengths = np.linalg.norm(following - vertices, axis=1)
    cross = vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    if np.count_nonzero(lengths > 0.0) < 3:
        return 0.0
    used = lengths > 0.0
    orientation = np.sign(cross.sum())
    return max(0.0, float(np.min(orientation * cross[used] / lengths[used])))


def _dilated_cut_interval(polygon, cut_y, radius):
    """Intersect a convex polygon dilated by a disk with a horizontal line."""
    starts = np.asarray(polygon, dtype=np.float64)
    ends = np.roll(starts, -1, axis=0)
    edges = ends - starts
    lengths = np.linalg.norm(edges, axis=1)
    nonzero = lengths > 0.0
    normals = np.column_stack((-edges[nonzero, 1], edges[nonzero, 0]))
    normals *= (radius / lengths[nonzero])[:, None]
    vertical_offset = starts[:, 1] - cut_y
    reaches_cut = np.abs(vertical_offset) <= radius
    half_chords = np.sqrt(np.maximum(0.0, radius**2 - vertical_offset[reaches_cut] ** 2))
    circle_centers = starts[reaches_cut, 0]
    cut_x = [circle_centers - half_chords, circle_centers + half_chords]
    # A capsule around each edge contributes two parallel segments and two
    # endpoint disks. An extreme point of the convex dilation lies on one.
    for offset in (normals, -normals):
        start = starts[nonzero] + offset
        end = ends[nonzero] + offset
        spans_cut = (np.minimum(start[:, 1], end[:, 1]) <= cut_y) & (
            cut_y <= np.maximum(start[:, 1], end[:, 1]))
        flat = spans_cut & (start[:, 1] == end[:, 1])
        sloped = spans_cut & ~flat
        weight = (cut_y - start[sloped, 1]) / (end[sloped, 1] - start[sloped, 1])
        cut_x.extend((start[flat, 0], end[flat, 0],
                      start[sloped, 0] + weight * (end[sloped, 0] - start[sloped, 0])))
    cut_x = np.concatenate(cut_x)
    return (float(cut_x.min()), float(cut_x.max())) if cut_x.size else None


def lateral_cut_blocked(shelf, cut_y, radius, other_polygons, *, x_upper=None, gap=0.0):
    """Prove that no supported blocker-center position can cross ``cut_y``.

    ``shelf`` is (xmin, xmax, ymin, ymax). The disk must be inside the actual
    Blocker at every yaw. Its center has a superset of the feasible positions
    of the real footprint, so complete coverage along a cut rules out all
    planar paths across that cut. An uncovered cut is inconclusive. ``x_upper``
    optionally restricts center X to prove that a forward excursion is required.
    """
    if not math.isfinite(radius) or radius <= _NUMERIC_MARGIN:
        return False
    if not math.isfinite(gap) or gap < 0.0:
        raise ValueError("The requested clearance gap must be finite and nonnegative.")
    xmin, xmax, ymin, ymax = shelf
    radius -= _NUMERIC_MARGIN  # Keep the computed obstacle coverage a lower bound.
    if cut_y < ymin + radius or cut_y > ymax - radius:
        return True
    start = xmin + radius
    end = xmax - radius
    if x_upper is not None:
        end = min(end, x_upper)
    if end < start:
        return True
    covered = []
    for polygon in other_polygons:
        interval = _dilated_cut_interval(polygon, cut_y, radius + gap)
        if interval is not None and interval[1] >= start and interval[0] <= end:
            covered.append(interval)
    for low, high in sorted(covered):
        if low > start:
            return False
        start = max(start, high)
        if start >= end:
            return True
    return False
