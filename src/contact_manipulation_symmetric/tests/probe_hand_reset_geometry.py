"""Screen random RH56E2 reset commands with URDF collision meshes on CPU.

The triangle surface test and single-convex-hull screen are complementary.
Neither reproduces the five convex decompositions cooked by PhysX. The
single-hull screen can overestimate interference in concave regions.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from itertools import combinations, product
import json
import math
from pathlib import Path
import time
from xml.etree import ElementTree

import hppfcl
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "src/contact_manipulation_symmetric/contact_manipulation_symmetric/config/ur5e/scene.py"
URDF = next((ROOT / "assets/robots/Roas_provided_inspire_urdf").rglob("RH56E2_R_2025_9_11_relative.urdf"))
OUTPUT = ROOT / "outputs/contact_manipulation_symmetric/hand_reset_randomization_2026-11-25"
HOUSES = {"right_plam_1", "right_plam_2"}
ROOT_FINGERS = {f"right_{finger}_1" for finger in ("thumb", "index", "middle", "ring", "little")}


def production_hand_mapping(path):
    """Read the exported mapping functions without starting Isaac Sim."""
    source = ast.parse(SCENE.read_text())
    wanted = {"hand_joint_mapping", "hand_joint_limits"}
    declarations = [
        node for node in source.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted)
        or any(isinstance(target, ast.Name) and target.id == "HAND_JOINT_NAMES"
               for target in getattr(node, "targets", ()))
    ]
    namespace = {"ElementTree": ElementTree, "math": math}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(SCENE), "exec"), namespace)
    names = namespace["HAND_JOINT_NAMES"]
    limits, _ = namespace["hand_joint_limits"](str(path))
    relation = {joint: (index, gain, offset) for joint, index, gain, offset
                in zip(*namespace["hand_joint_mapping"](str(path)))}
    return tuple(names), np.array([limits[name] for name in names]), relation


def vector(node, key, default=(0.0, 0.0, 0.0)):
    if node is None or key not in node.attrib:
        return np.asarray(default)
    return np.array([float(value) for value in node.get(key).split()])


def origin(node):
    return Rotation.from_euler("xyz", vector(node, "rpy")).as_matrix(), vector(node, "xyz")


def finger_group(link):
    for finger in ("thumb", "index", "middle", "ring", "little"):
        if link.startswith(f"right_{finger}_"):
            return finger
    return "palm"


def hand_frames(urdf, master_positions, relation):
    """URDF FK for every link at each candidate's coupled angles."""
    edges = {
        joint.find("child").get("link"):
        (joint.get("name"), joint.find("parent").get("link"), *origin(joint.find("origin")),
         vector(joint.find("axis"), "xyz", (1.0, 0.0, 0.0)), joint.get("type"))
        for joint in urdf.findall("joint")
    }
    frames = {}

    def visit(link):
        if link in frames:
            return frames[link]
        count = len(master_positions)
        if link not in edges:
            result = (np.broadcast_to(np.eye(3), (count, 3, 3)), np.zeros((count, 3)))
        else:
            name, parent, joint_rotation, joint_translation, axis, kind = edges[link]
            parent_rotation, parent_translation = visit(parent)
            rotation = parent_rotation @ joint_rotation
            translation = parent_rotation @ joint_translation + parent_translation
            if kind == "revolute":
                index, gain, offset = relation[name]
                angle = master_positions[:, index] * gain + offset
                rotation = rotation @ Rotation.from_rotvec(angle[:, None] * (axis / np.linalg.norm(axis))).as_matrix()
            result = (rotation, translation)
        frames[link] = result
        return result

    for link in urdf.findall("link"):
        visit(link.get("name"))
    return frames, edges


def collision_pairs(links, edges):
    """Skip rigidly attached/direct joint groups and authored mounting overlaps."""
    def fixed_root(link):
        while link in edges and edges[link][-1] == "fixed":
            link = edges[link][1]
        return link

    connected = {frozenset((fixed_root(child), fixed_root(data[1])))
                 for child, data in edges.items() if data[-1] != "fixed"}
    groups = [fixed_root(link) for link in links]
    columns, kinds = [], []
    for i, j in combinations(range(len(links)), 2):
        a, b = links[i], links[j]
        if (groups[i] == groups[j] or frozenset((groups[i], groups[j])) in connected
                or (a in HOUSES and b in ROOT_FINGERS) or (b in HOUSES and a in ROOT_FINGERS)):
            continue
        ga, gb = finger_group(a), finger_group(b)
        columns.append((i, j))
        kinds.append("same_finger" if ga == gb else "palm_finger" if "palm" in (ga, gb) else "cross_finger")
    return columns, kinds


def load_colliders(path, urdf, frames):
    """Each entry has its source BVH, convex envelope, poses and world AABBs."""
    loader = hppfcl.MeshLoader()
    links, meshes, hulls, all_rot, all_trans, all_centers, all_half_extents = ([] for _ in range(7))
    for link in urdf.findall("link"):
        name = link.get("name")
        for collider in link.findall("collision"):
            mesh = collider.find("./geometry/mesh")
            if mesh is None:
                raise ValueError(f"Expected production hand mesh geometry on {name}.")
            mesh_path = (path.parent / mesh.get("filename")).resolve()
            native = loader.load(str(mesh_path), vector(mesh, "scale", (1.0, 1.0, 1.0)))
            vertices = native.vertices()
            lower, upper = vertices.min(0), vertices.max(0)
            midpoint, half_extent = (lower + upper) * .5, (upper - lower) * .5
            native.buildConvexHull(False, "")
            if native.convex is None:
                raise RuntimeError(f"Cannot build convex envelope of {mesh_path}.")
            local_r, local_t = origin(collider.find("origin"))
            r, t = frames[name]
            link_r, link_t = r @ local_r, r @ local_t + t
            links.append(name)
            meshes.append(native)
            hulls.append(native.convex)
            all_rot.append(link_r)
            all_trans.append(link_t)
            all_centers.append(link_r @ midpoint + link_t)
            all_half_extents.append(np.abs(link_r) @ half_extent)
    return (links, meshes, hulls, np.stack(all_rot, 1), np.stack(all_trans, 1),
            np.stack(all_centers, 1), np.stack(all_half_extents, 1))


def intersections(urdf, master_positions, relation, path):
    frames, edges = hand_frames(urdf, master_positions, relation)
    links, meshes, hulls, rotations, translations, centers, half_extents = load_colliders(path, urdf, frames)
    columns, kinds = collision_pairs(links, edges)
    count = len(master_positions)
    hits_hull = np.zeros((count, len(columns)), dtype=bool)
    hits_triangle = np.zeros_like(hits_hull)
    query = hppfcl.CollisionRequest()
    result = hppfcl.CollisionResult()
    query.num_max_contacts = 1
    query.enable_contact = False
    for column, (i, j) in enumerate(columns):
        aabb_overlap = (np.abs(centers[:, i] - centers[:, j]) <= half_extents[:, i] + half_extents[:, j]).all(-1)
        for row in np.flatnonzero(aabb_overlap):
            left = hppfcl.Transform3f(rotations[row, i], translations[row, i])
            right = hppfcl.Transform3f(rotations[row, j], translations[row, j])
            result.clear()
            if hppfcl.collide(hulls[i], left, hulls[j], right, query, result) == 0:
                continue
            hits_hull[row, column] = True
            result.clear()
            hits_triangle[row, column] = hppfcl.collide(meshes[i], left, meshes[j], right, query, result) > 0
    pairs = [f"{links[i]} / {links[j]}" for i, j in columns]
    return pairs, np.asarray(kinds), hits_hull, hits_triangle


def summarize(hits, pairs, kinds, mask):
    rows = hits[mask]
    relative = rows & ~hits[0]
    sectors = {}
    for name in sorted(set(kinds)):
        subset = kinds == name
        sectors[str(name)] = {
            "pairs": int(subset.sum()),
            "any": int(rows[:, subset].any(1).sum()),
            "new_relative_to_raw_zero": int(relative[:, subset].any(1).sum()),
        }
    common = Counter({name: int(num) for name, num in zip(pairs, rows.sum(0)) if num > 0})
    return {
        "count": len(rows),
        "any": int(rows.any(1).sum()),
        "new_relative_to_raw_zero": int(relative.any(1).sum()),
        "pairs_from_raw_zero": [pair for pair, hit in zip(pairs, hits[0]) if hit],
        "sectors": sectors,
        "most_frequent_pairs": common.most_common(12),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sobol-power", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--urdf", type=Path, default=URDF)
    args = parser.parse_args()
    start = time.perf_counter()
    names, limits, relation = production_hand_mapping(args.urdf)
    samples = 2.0 * qmc.Sobol(len(names), scramble=True, seed=args.seed).random_base2(args.sobol_power) - 1.0
    corners = np.array(tuple(product((-1.0, 1.0), repeat=len(names))))
    raw = np.vstack((np.zeros((1, len(names))), samples, corners))
    # Hand scale 0.5, then native affine map from normalized [-1,1] to the
    # coupled master limits. Raw [-1,1] covers the central 25–75% of travel.
    master = limits[:, 0] + (1.0 + raw * .5) * .5 * np.diff(limits, axis=1).T
    urdf = ElementTree.parse(args.urdf).getroot()
    pairs, kinds, hull, triangles = intersections(urdf, master, relation, args.urdf.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "static_geometry_samples.npz", raw_action=raw, master_angle_rad=master,
                        pairs=pairs, pair_kind=kinds, convex_hull_intersections=hull, triangle_intersections=triangles)
    random_rows = slice(1, 1 + len(samples))
    corner_rows = slice(1 + len(samples), None)
    summary = {
        "urdf": str(args.urdf.resolve().relative_to(ROOT)) if args.urdf.resolve().is_relative_to(ROOT) else str(args.urdf),
        "seed": args.seed, "raw_units": "policy raw action before scale 0.5",
        "master_names": names, "master_limits_rad": limits.tolist(),
        "pairs_checked": len(pairs),
        "geometry": "URDF mesh triangles and per-collider convex hulls; fixed/directly jointed rigid groups and authored root/mount-filtered pairs omitted",
        "duration_seconds": round(time.perf_counter() - start, 3),
        "source_triangles": {
            "sobol": summarize(triangles, pairs, kinds, random_rows),
            "corners": summarize(triangles, pairs, kinds, corner_rows),
        },
        "convex_hulls": {
            "sobol": summarize(hull, pairs, kinds, random_rows),
            "corners": summarize(hull, pairs, kinds, corner_rows),
        },
    }
    for name, records in (("source_triangles", triangles), ("convex_hulls", hull)):
        new = records & ~records[0]
        extra_hand_pairs = new & (kinds != "same_finger")
        score = extra_hand_pairs.sum(1)
        picked = np.argsort(-score, kind="stable")[:8]
        summary[name]["example_poses"] = [
            {"row": int(i), "raw_action": raw[i].tolist(),
             "new_cross_finger_or_palm_pairs": [pair for pair, hit in zip(pairs, extra_hand_pairs[i]) if hit]}
            for i in picked if score[i] > 0
        ]
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    (args.output / "static_geometry_summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
