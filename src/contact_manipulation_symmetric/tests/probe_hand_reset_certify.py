"""Build continuous raw-action reset boxes from exported cooked hand colliders."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import time

import hppfcl
import numpy as np
from scipy.stats import qmc

import probe_hand_reset_geometry as geometry


RAW_MIN, RAW_MAX = -2.0, 2.0
NUMERIC_PADDING_M = 1.0e-8


def _convex_hull(vertices):
    points = hppfcl.StdVec_Vec3f()
    for point in vertices:
        points.append(np.asarray(point, dtype=np.float64))
    return hppfcl.Convex.convexHull(points, False, "")


def _urdf_link_pose(frame):
    rotation, translation = frame
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation[0], translation[0]
    return pose


def _native_link_frames(cooked, frames):
    """Re-express each USD body-local frame in its canonical URDF link frame."""
    native_poses = cooked["reference_body_transforms"]
    hand_from_world = np.linalg.inv(native_poses["base_link"])
    return {link: np.linalg.inv(_urdf_link_pose(frames[link])) @ hand_from_world @ native_poses[link]
            for link in cooked["bodies"]}


def _native_hull_vertices(bodies, link_frames):
    pieces = []
    for link, hulls in bodies.items():
        pose = link_frames[link]
        for hull in hulls:
            vertices = np.asarray(hull, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
            pieces.append((link, vertices))
    return pieces


def _piece_motion_bounds(body, vertices, edges, relation, limits):
    """Bound the swept vertex distance per unit raw action for each ancestor."""
    radius = np.linalg.norm(vertices, axis=1).max()
    node = body
    result = {}
    while node in edges:
        joint, parent, _, translation, _, kind = edges[node]
        if kind == "revolute":
            motor, gain, _ = relation[joint]
            # Native hand scale is 0.5 and its normalized-to-URDF map divides by 2.
            result[node] = (motor, radius * abs(gain) * (limits[motor, 1] - limits[motor, 0]) / 4.0)
        radius += np.linalg.norm(translation)
        node = parent
    return result


class CookedHandCertificate:
    """Verify a lower bound on every convex-piece gap over a raw-action box."""

    def __init__(self, cooked_path):
        cooked = json.loads(cooked_path.read_text())
        self.names, self.limits, self.relation = geometry.production_hand_mapping(geometry.URDF)
        self.urdf = geometry.ElementTree.parse(geometry.URDF).getroot()
        reference = np.array([[cooked["reference_pose"][name] for name in self.names]])
        reference_frames, edges = geometry.hand_frames(self.urdf, reference, self.relation)
        link_frames = _native_link_frames(cooked, reference_frames)
        self.urdf_tcp_from_native = link_frames["right_plam_force_sensor"]
        self.fk_agreement = self._fk_checks(cooked.get("fk_checks", ()), link_frames)
        pieces = _native_hull_vertices(cooked["bodies"], link_frames)
        self.links = [name for name, _ in pieces]
        self.vertices = [vertices for _, vertices in pieces]
        self.hulls = [_convex_hull(vertices) for vertices in self.vertices]
        low = np.array([vertices.min(0) for vertices in self.vertices])
        high = np.array([vertices.max(0) for vertices in self.vertices])
        self.centers = (high + low) * 0.5
        self.half_extents = (high - low) * 0.5
        self.pairs, _ = geometry.collision_pairs(self.links, edges)
        self.first, self.second = np.array(self.pairs, dtype=np.intp).T
        self.motion = np.zeros((len(self.pairs), len(self.names)))
        bounds = [_piece_motion_bounds(name, vertices, edges, self.relation, self.limits)
                  for name, vertices in pieces]
        self.piece_motion = np.zeros((len(pieces), len(self.names)))
        for row, chain in zip(self.piece_motion, bounds):
            for motor, amount in chain.values():
                row[motor] += amount
        for pair, (left, right) in enumerate(self.pairs):
            # Shared ancestor rotations move both pieces together. Only the
            # relative branch motion can close the separating projection gap.
            a, b = bounds[left], bounds[right]
            for motor, amount in [a[name] for name in a.keys() - b.keys()] + [
                b[name] for name in b.keys() - a.keys()
            ]:
                self.motion[pair, motor] += amount
        self.distance_request = hppfcl.DistanceRequest()
        self.distance_request.enable_nearest_points = True
        self.distance_result = hppfcl.DistanceResult()

    def tcp_envelopes(self, lower, upper):
        """Enclose the full swept Hand shape of each action box in palm TCP coordinates."""
        raw = (lower + upper) * 0.5
        master = self.limits[:, 0] + (raw + 2.0) * (self.limits[:, 1] - self.limits[:, 0]) / 4.0
        frames, _ = geometry.hand_frames(self.urdf, master, self.relation)
        hand_from_tcp = _urdf_link_pose(frames["right_plam_force_sensor"]) @ self.urdf_tcp_from_native
        native_rotation, native_origin = hand_from_tcp[:3, :3], hand_from_tcp[:3, 3]
        motion_allowance = ((upper - lower) * 0.5) @ self.piece_motion.T + NUMERIC_PADDING_M
        low, high = np.full((len(lower), 3), np.inf), np.full((len(lower), 3), -np.inf)
        for index, (link, vertices) in enumerate(zip(self.links, self.vertices)):
            rotation, translation = frames[link]
            points = vertices @ np.swapaxes(rotation, 1, 2) + (translation - native_origin)[:, None, :]
            points = points @ native_rotation
            allowance = motion_allowance[:, index, None]
            low = np.minimum(low, points.min(axis=1) - allowance)
            high = np.maximum(high, points.max(axis=1) + allowance)
        return low, high

    def _fk_checks(self, records, link_frames):
        checks = []
        for record in records:
            master_positions = np.array([[record["joint_positions"][name] for name in self.names]])
            frames, _ = geometry.hand_frames(self.urdf, master_positions, self.relation)
            native = record["body_transforms"]
            hand_from_world = np.linalg.inv(native["base_link"])
            errors = []
            for name, link_frame in link_frames.items():
                expected = hand_from_world @ native[name]
                delta = _urdf_link_pose(frames[name]) @ link_frame - expected
                errors.append((np.linalg.norm(delta[:3, 3]), np.linalg.norm(delta[:3, :3])))
            translation, rotation = np.max(errors, axis=0)
            if max(translation, rotation) > 1.0e-5:
                raise ValueError(f"URDF and exported PhysX FK differ at raw pose {record['raw']}.")
            checks.append({"raw": record["raw"], "max_translation_difference_m": float(translation),
                           "max_rotation_difference_frobenius": float(rotation)})
        return checks

    def _separation_axis_gap(self, rotations, translations, pair):
        i, j = self.pairs[pair]
        pose_a = hppfcl.Transform3f(rotations[i], translations[i])
        pose_b = hppfcl.Transform3f(rotations[j], translations[j])
        result = self.distance_result
        result.clear()
        distance = hppfcl.distance(
            self.hulls[i], pose_a, self.hulls[j], pose_b, self.distance_request, result)
        if distance <= 0.0:
            return 0.0
        axis = result.getNearestPoint2() - result.getNearestPoint1()
        norm = np.linalg.norm(axis)
        if not np.isfinite(norm) or norm <= NUMERIC_PADDING_M:
            return 0.0
        axis /= norm
        # Recompute both support extrema explicitly. The GJK result suggests
        # an axis; its reported distance is never used as a certified gap.
        maximum = (self.vertices[i] @ (rotations[i].T @ axis)).max()
        minimum = (self.vertices[j] @ (rotations[j].T @ axis)).min()
        return float(minimum - maximum + (translations[j] - translations[i]) @ axis)

    def verify(self, lower, upper, clearance):
        """Return certified margins and limiting piece pair for each candidate box."""
        lo, hi = np.asarray(lower), np.asarray(upper)
        raw = (lo + hi) * 0.5
        radius = (hi - lo) * 0.5
        master = self.limits[:, 0] + (1.0 + raw * 0.5) * 0.5 * np.diff(self.limits, axis=1).T
        frames, _ = geometry.hand_frames(self.urdf, master, self.relation)
        rotations = np.stack([frames[name][0] for name in self.links], axis=1)
        translations = np.stack([frames[name][1] for name in self.links], axis=1)
        centers = np.einsum("nijk,ik->nij", rotations, self.centers) + translations
        half_extents = np.einsum("nijk,ik->nij", np.abs(rotations), self.half_extents)
        gap_aabb = (np.abs(centers[:, self.first] - centers[:, self.second])
                    - half_extents[:, self.first] - half_extents[:, self.second]).max(axis=-1)
        travel = radius @ self.motion.T + NUMERIC_PADDING_M
        margin = gap_aabb - travel
        limiting = np.argmin(margin, axis=-1)
        for row, possible in enumerate(margin):
            unresolved = np.flatnonzero(possible < clearance)
            for pair in unresolved[np.argsort(possible[unresolved])]:
                gap = self._separation_axis_gap(rotations[row], translations[row], pair)
                possible[pair] = max(possible[pair], gap - travel[row, pair])
                if possible[pair] < clearance:
                    limiting[row] = pair
                    break
            else:
                limiting[row] = possible.argmin()
        return margin[np.arange(len(lo)), limiting], limiting

    def pair_name(self, index):
        return " / ".join(self.links[piece] for piece in self.pairs[index])


def _anchors(seed, sobol_power):
    """Try the central hand, broad thumb/index/middle variants, and low discrepancy seeds."""
    structured = [[a, b, c, d, 0.0, 0.0]
                  for a, b, c, d in product((-1.7, 0.0, 1.7), (-1.7, 0.0, 1.7),
                                           (-1.7, 0.0), (-1.7, 0.0))]
    thumb_endpoints = [[a, b, c, d, 0.0, 0.0]
                       for a, b, c, d in product((-1.95, 0.0, 1.95), (-1.95, -1.90, 1.90, 1.95),
                                                (-1.0, 0.0), (0.0, 1.0))]
    sobol = RAW_MIN + (RAW_MAX - RAW_MIN) * qmc.Sobol(6, scramble=True, seed=seed).random_base2(sobol_power)
    return np.vstack((np.zeros((1, 6)), sorted(structured, key=np.linalg.norm), thumb_endpoints, sobol))


def _union_gaps(lower, upper):
    gaps, extent = [], RAW_MIN
    for lo, hi in sorted(zip(lower, upper)):
        if lo > extent + 1.0e-10:
            gaps.append([extent, float(lo)])
        extent = max(extent, float(hi))
    if extent < RAW_MAX - 1.0e-10:
        gaps.append([extent, RAW_MAX])
    return gaps


def build_boxes(certificate, anchors, bins, refinement, companion_width, minimum_clearance):
    """Cover each motor axis with continuous intervals and nonzero companion widths."""
    bounds = np.linspace(RAW_MIN, RAW_MAX, bins + 1)
    orthogonal = np.clip(anchors, RAW_MIN + companion_width, RAW_MAX - companion_width)
    boxes, failures = [], []
    for axis, name in enumerate(certificate.names):
        pending = np.stack((bounds[:-1], bounds[1:]), axis=1)
        recorded = Counter()
        for level in range(refinement + 1):
            missing = pending
            for anchor in orthogonal:
                if not len(missing):
                    break
                low = np.broadcast_to(anchor - companion_width, (len(missing), 6)).copy()
                high = np.broadcast_to(anchor + companion_width, low.shape).copy()
                low[:, axis], high[:, axis] = missing.T
                margins, limiting = certificate.verify(low, high, minimum_clearance)
                accepted = margins >= minimum_clearance
                boxes.extend((a, b, float(gap), axis) for a, b, gap in zip(
                    low[accepted], high[accepted], margins[accepted]))
                if level == refinement:
                    recorded.update(certificate.pair_name(i) for i in limiting[~accepted])
                missing = missing[~accepted]
            if not len(missing):
                break
            mids = missing.mean(axis=1)
            pending = np.column_stack((missing[:, 0], mids, mids, missing[:, 1])).reshape(-1, 2)
        failures.append({"motor": name, "uncertified_segments": missing.tolist(),
                         "frequent_limiting_pairs": recorded.most_common(5)})
        print(f"{name}: {len(missing)} uncertified terminal segments", flush=True)
    return boxes, failures


def broad_boxes(certificate, seed, sobol_power, halfwidth, minimum_clearance):
    """Keep the largest tested safe box for each center of a bounded Sobol draw."""
    if sobol_power == 0:
        return [], []
    unit = qmc.Sobol(6, scramble=True, seed=seed).random_base2(sobol_power)
    centers = RAW_MIN + halfwidth + (RAW_MAX - RAW_MIN - 2.0 * halfwidth) * unit
    boxes, results = [], []
    for factor in (1.0, 0.5, 0.25):
        width = halfwidth * factor
        lower, upper = centers - width, centers + width
        margins, _ = certificate.verify(lower, upper, minimum_clearance)
        accepted = margins >= minimum_clearance
        boxes.extend((lo, hi, float(margin), -1) for lo, hi, margin in zip(
            lower[accepted], upper[accepted], margins[accepted]))
        results.append({"raw_halfwidth": width, "accepted": int(accepted.sum()), "tried": len(accepted)})
        centers = centers[~accepted]
        if not len(centers):
            break
    print(f"Broad boxes: {len(boxes)} / {len(unit)} certified", flush=True)
    return boxes, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cooked", type=Path, help="Cooked convex pieces and reference pose exported from Isaac Sim")
    parser.add_argument("--bins", type=int, default=48)
    parser.add_argument("--refinements", type=int, default=3)
    parser.add_argument("--anchors-power", type=int, default=5)
    parser.add_argument("--companion-halfwidth", type=float, default=0.025)
    parser.add_argument("--broad-power", type=int, default=9, help="Set to 0 to disable broad Sobol boxes")
    parser.add_argument("--broad-halfwidth", type=float, default=0.08)
    parser.add_argument("--minimum-gap-mm", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.bins < 1 or args.refinements < 0 or args.anchors_power < 0 or args.broad_power < 0:
        parser.error("Choose a positive bin count and nonnegative refinements and Sobol powers.")
    if not 0 < args.companion_halfwidth < 2 or not 0 < args.broad_halfwidth < 2 or args.minimum_gap_mm <= 0:
        parser.error("Choose both half-widths in (0, 2) and a positive minimum gap.")
    started = time.perf_counter()
    certificate = CookedHandCertificate(args.cooked)
    boxes, failures = build_boxes(certificate, _anchors(args.seed, args.anchors_power), args.bins,
                                  args.refinements, args.companion_halfwidth, args.minimum_gap_mm / 1000)
    wide_boxes, wide_results = broad_boxes(certificate, args.seed + 1, args.broad_power,
                                          args.broad_halfwidth, args.minimum_gap_mm / 1000)
    boxes.extend(wide_boxes)
    low = np.array([box[0] for box in boxes], dtype=np.float64).reshape(-1, 6)
    high = np.array([box[1] for box in boxes], dtype=np.float64).reshape(-1, 6)
    margin = np.array([box[2] for box in boxes], dtype=np.float64)
    axis = np.array([box[3] for box in boxes], dtype=np.int64)
    envelope_low, envelope_high = certificate.tcp_envelopes(low, high) if len(low) else (low[:, :3], high[:, :3])
    gaps = {name: _union_gaps(low[:, i], high[:, i]) for i, name in enumerate(certificate.names)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, lower=low, upper=high, clearance_m=margin,
                        axis=axis, motor_names=np.array(certificate.names),
                        tcp_envelope_lower_m=envelope_low, tcp_envelope_upper_m=envelope_high)
    summary = {"cooked_geometry": str(args.cooked), "parameters": {
        "bins": args.bins, "refinements": args.refinements, "anchors": len(_anchors(args.seed, args.anchors_power)),
        "companion_halfwidth": args.companion_halfwidth, "minimum_gap_mm": args.minimum_gap_mm,
        "broad_halfwidth": args.broad_halfwidth, "seed": args.seed},
        "pieces": len(certificate.hulls), "piece_pairs": len(certificate.pairs),
        "native_fk_spot_checks": certificate.fk_agreement,
        "boxes": len(boxes), "minimum_certificate_mm": float(margin.min() * 1000) if len(margin) else None,
        "tcp_envelope_min_m": envelope_low.min(axis=0).tolist() if len(low) else None,
        "tcp_envelope_max_m": envelope_high.max(axis=0).tolist() if len(low) else None,
        "axis_cover_boxes": len(boxes) - len(wide_boxes), "broad_boxes": len(wide_boxes),
        "broad_search": wide_results,
        "axis_projection_gaps_raw": gaps, "complete_axis_coverage": all(not values for values in gaps.values()),
        "per_axis_search": failures, "duration_seconds": round(time.perf_counter() - started, 3)}
    report = json.dumps(summary, indent=2)
    args.output.with_suffix(".json").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
