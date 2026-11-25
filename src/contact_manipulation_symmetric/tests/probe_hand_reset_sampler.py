"""Draw and time reset poses from a union of certified Hand action boxes."""

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np

from probe_hand_reset_geometry import ROOT, URDF, production_hand_mapping


OUTPUT = ROOT / "outputs/contact_manipulation_symmetric/hand_reset_full_range_2026-11-25"


def gaps_by_axis(lower, upper):
    """Find gaps in the projection of the box union onto each raw action axis."""
    gaps = []
    for lo, hi in zip(lower.T, upper.T):
        end = -2.0
        axis_gaps = []
        for left, right in sorted(zip(lo, hi)):
            if left > end + 1e-12:
                axis_gaps.append([float(end), float(left)])
            end = max(end, right)
        if end < 2.0 - 1e-12:
            axis_gaps.append([float(end), 2.0])
        gaps.append(axis_gaps)
    return gaps


def draw_poses(rng, count, lower, width, joint_bias, joint_slope, master_ids):
    """Draw each box and each in-box coordinate once, and apply URDF coupling."""
    box_ids = rng.integers(len(lower), size=count)
    raw = lower[box_ids] + rng.random((count, lower.shape[1])) * width[box_ids]
    q = joint_bias + raw[:, master_ids] * joint_slope
    return box_ids, raw, q


def endpoint_witnesses(names, lower, upper, clearance):
    """Provide one physical test pose for each endpoint covered by the boxes."""
    witnesses = []
    for axis, name in enumerate(names):
        for edge, endpoint in ((lower, -2.0), (upper, 2.0)):
            ids = np.flatnonzero(edge[:, axis] == endpoint)
            if ids.size:
                box = ids[clearance[ids].argmax()]
                raw = (lower[box] + upper[box]) * 0.5
                raw[axis] = endpoint
                witnesses.append({"joint": name, "endpoint": endpoint, "box_id": int(box), "raw": raw.tolist()})
    return witnesses


def measure(rng, count, repetitions, sampler_args):
    for _ in range(100):
        draw_poses(rng, count, *sampler_args)
    timings_ns = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        draw_poses(rng, count, *sampler_args)
        timings_ns.append(time.perf_counter_ns() - start)
    mean = statistics.mean(timings_ns) / 1000
    return {
        "batch_size": count,
        "repetitions": repetitions,
        "mean_batch_us": mean,
        "mean_pose_us": mean / count,
        "median_batch_us": statistics.median(timings_ns) / 1000,
        "p95_batch_us": float(np.percentile(timings_ns, 95)) / 1000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("boxes", type=Path, help="NPZ with lower and upper raw action bounds")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.count <= 0 or args.repetitions <= 0:
        parser.error("Pose count and repetitions must be positive.")

    start = time.perf_counter_ns()
    with np.load(args.boxes) as boxes:
        lower, upper, clearance = boxes["lower"], boxes["upper"], boxes["clearance_m"]
        box_names = tuple(boxes["motor_names"])
    box_load_us = (time.perf_counter_ns() - start) / 1000
    names, limits, relations = production_hand_mapping(URDF)
    if box_names != names:
        raise ValueError("Box axes do not match the configured URDF motor order.")
    if (lower.ndim != 2 or lower.shape[1] != len(names) or lower.shape != upper.shape or not len(lower)
            or clearance.shape != (len(lower),) or not np.isfinite(clearance).all() or np.any(clearance <= 0)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower < -2.0)
            or np.any(upper > 2.0) or np.any(upper <= lower)):
        raise ValueError("Expected nonempty boxes within the full [-2, 2] raw range.")

    driven_names = list(relations)
    master_ids, multiplier, offset = (np.asarray(part) for part in zip(*relations.values()))
    joint_bias = limits[master_ids].mean(axis=1) * multiplier + offset
    joint_slope = (limits[master_ids, 1] - limits[master_ids, 0]) * 0.25 * multiplier
    sampler_args = (lower, upper - lower, joint_bias, joint_slope, master_ids)
    ids, raw, pos = draw_poses(np.random.default_rng(args.seed), args.count, *sampler_args)
    endpoints = endpoint_witnesses(names, lower, upper, clearance)
    rng = np.random.default_rng(args.seed + 1)
    summary = {
        "seed": args.seed,
        "count": args.count,
        "box_count": len(lower),
        "box_load_us": box_load_us,
        "master_order": names,
        "driven_order": driven_names,
        "raw_support_gaps_by_axis": dict(zip(names, gaps_by_axis(lower, upper))),
        "raw_sample_min": raw.min(axis=0).tolist(),
        "raw_sample_max": raw.max(axis=0).tolist(),
        "endpoint_witness_count": len(endpoints),
        "batch_timings": [measure(rng, size, args.repetitions, sampler_args) for size in (1, args.count, 1024)],
    }

    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / "raw_poses.json").write_text(json.dumps({"raw_actions": raw.tolist()}, indent=2) + "\n")
    native_input = {"raw_actions": raw.tolist() + [item["raw"] for item in endpoints],
                    "endpoint_witnesses": endpoints}
    (args.output / "native_input.json").write_text(json.dumps(native_input, indent=2) + "\n")
    (args.output / "sampling_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output / "poses.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["pose_id", "box_id", "certified_min_clearance_mm", *(f"raw_{name}" for name in names),
                         *(f"rad_{name}" for name in driven_names)])
        for row, (box, action, angles) in enumerate(zip(ids, raw, pos), 1):
            writer.writerow([row, box, float(clearance[box] * 1000), *action, *angles])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
