"""Build the packaged Warp Hand geometry from an Isaac Sim export and certified reset boxes."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from probe_hand_reset_certify import CookedHandCertificate, geometry


# Keep 0.10 mm of clearance plus the sampler's 0.004 mm float32 allowance.
MINIMUM_FALLBACK_GAP_M = 0.000104


def _face_axes(vertices):
    hull = ConvexHull(vertices)
    triangle = vertices[hull.simplices]
    areas = np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
    directions = []
    for index in np.argsort(-areas):
        axis = hull.equations[index, :3]
        if all(abs(previous @ axis) < 0.999999 for previous in directions):
            directions.append(axis)
    return np.asarray(directions)


def _pack_vectors(rows):
    lengths = np.array([len(row) for row in rows], dtype=np.int32)
    padded = np.zeros((len(rows), lengths.max(), 3), dtype=np.float32)
    for index, row in enumerate(rows):
        padded[index, :len(row)] = row
    return padded, lengths


def _fallback_arrays(certificate, boxes_path):
    with np.load(boxes_path, allow_pickle=False) as boxes:
        if tuple(boxes["motor_names"]) != certificate.names:
            raise ValueError("Fallback box axes differ from the configured Hand motor order.")
        cover = np.flatnonzero(boxes["axis"] == 0)
        cover = cover[np.argsort(boxes["lower"][cover, 0], kind="stable")]
        lower, upper, gap = (boxes[key][cover] for key in ("lower", "upper", "clearance_m"))
    if (not len(lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or not np.isfinite(gap).all() or np.any(upper <= lower) or np.any(lower < -2) or np.any(upper > 2)
            or lower[0, 0] != -2 or upper[-1, 0] != 2 or not np.array_equal(lower[1:, 0], upper[:-1, 0])):
        raise ValueError("Expected contiguous thumb-first boxes covering the full [-2, 2] raw interval.")

    # Check the emitted float32 endpoints as well as the supplied certificate.
    lower, upper = lower.astype(np.float32), upper.astype(np.float32)
    verified, _ = certificate.verify(lower.astype(np.float64), upper.astype(np.float64), MINIMUM_FALLBACK_GAP_M)
    if np.any(gap < MINIMUM_FALLBACK_GAP_M) or np.any(verified < MINIMUM_FALLBACK_GAP_M):
        raise ValueError("The supplied fallback boxes lack the required clearance with the supplied geometry.")
    return {"fallback_lower": lower, "fallback_upper": upper, "fallback_gap_m": gap}


def build(cooked_path: Path, boxes_path: Path, output_path: Path):
    certificate = CookedHandCertificate(cooked_path)
    links = list(dict.fromkeys(certificate.links))
    _, edges = geometry.hand_frames(certificate.urdf, certificate.limits.mean(1)[None], certificate.relation)
    joint_ids = {joint: index for index, joint in enumerate(edges)}
    chains = []
    for link in links:
        chain = []
        while link in edges:
            chain.append(joint_ids[link])
            link = edges[link][1]
        chains.append(chain[::-1])
    chain_lengths = [len(chain) for chain in chains]
    width = max(chain_lengths)
    padded_chains = [chain + [0] * (width - len(chain)) for chain in chains]

    rotations, positions, axes, motors, biases, gains = [], [], [], [], [], []
    for joint, _, rotation, position, axis, kind in edges.values():
        rotations.append(rotation)
        positions.append(position)
        if kind == "revolute":
            motor, multiplier, offset = certificate.relation[joint]
            lower, upper = certificate.limits[motor]
            motors.append(motor)
            axes.append(axis / np.linalg.norm(axis))
            biases.append((lower + upper) * 0.5 * multiplier + offset)
            gains.append((upper - lower) * 0.25 * multiplier)
        else:
            motors.append(-1)
            axes.append([0.0, 0.0, 1.0])
            biases.append(0.0)
            gains.append(0.0)

    vertex, vertex_count = _pack_vectors(certificate.vertices)
    face_axis, face_axis_count = _pack_vectors([_face_axes(vertices) for vertices in certificate.vertices])
    arrays = {
        "urdf_sha256": np.asarray(sha256(geometry.URDF.read_bytes()).hexdigest()),
        "chain": np.asarray(padded_chains, dtype=np.int32),
        "chain_length": np.asarray(chain_lengths, dtype=np.int32),
        "joint_rot": np.asarray(rotations, dtype=np.float32),
        "joint_pos": np.asarray(positions, dtype=np.float32),
        "joint_axis": np.asarray(axes, dtype=np.float32),
        "joint_motor": np.asarray(motors, dtype=np.int32),
        "joint_bias": np.asarray(biases, dtype=np.float32),
        "joint_gain": np.asarray(gains, dtype=np.float32),
        "piece_link": np.asarray([links.index(link) for link in certificate.links], dtype=np.int32),
        "piece_center": np.asarray(certificate.centers, dtype=np.float32),
        "piece_half_extents": np.asarray(certificate.half_extents, dtype=np.float32),
        "vertex": vertex,
        "vertex_count": vertex_count,
        "face_axis": face_axis,
        "face_axis_count": face_axis_count,
    }
    labels = [tuple(sorted(geometry.finger_group(certificate.links[i]) for i in pair)) for pair in certificate.pairs]
    thumb_groups = (("palm", "thumb"), ("index", "thumb"), ("middle", "thumb"))
    pairs = np.asarray(certificate.pairs, dtype=np.int32)
    for stage, groups in (("thumb", thumb_groups), ("index", (thumb_groups[1],)), ("middle", (thumb_groups[2],))):
        active = np.asarray([label in groups for label in labels])
        arrays[f"{stage}_pairs"] = pairs[active]
        arrays[f"{stage}_motion_m"] = certificate.motion[active]
    arrays.update(_fallback_arrays(certificate, boxes_path))
    np.savez_compressed(output_path, **arrays)
    print(f"{len(links)} links, {len(vertex)} convex pieces, {len(arrays['fallback_gap_m'])} thumb fallback boxes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cooked", type=Path, help="JSON from probe_hand_reset_randomization.py --export-hand-hulls")
    parser.add_argument("boxes", type=Path, help="NPZ from probe_hand_reset_certify.py --output")
    parser.add_argument("output", type=Path, help="Path to write hand_reset_geometry.npz")
    args = parser.parse_args()
    build(args.cooked, args.boxes, args.output)
