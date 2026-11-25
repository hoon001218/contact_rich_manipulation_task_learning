"""Generate a CPU top-view gallery using the training sampler and exported asset geometry."""

from __future__ import annotations

import argparse
from itertools import cycle
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from test_mdp import task
from contact_manipulation_symmetric.mdp.curriculums import SCENE_LAYOUTS_BY_DIFFICULTY


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESOURCES = ROOT / "docs/figures/sequential_spawn_layouts30_blocker_front_2026-11-08/resources.json"
DEFAULT_OUTPUT = ROOT / "outputs/contact_manipulation_symmetric/layout_curriculum"
DIFFICULTIES = dict(zip(("easy", "medium", "hard"), SCENE_LAYOUTS_BY_DIFFICULTY, strict=True))
SIDE_TEXT = {-1: "LEFT", 1: "RIGHT"}
STATUS_TEXT = {"direct": "DIRECT", "front": "FRONT ROUTE", "blocked": "BLOCKED"}

PAPER = "#ecf1f5"
INK = "#162739"
MUTED = "#5b6c7d"
SHELF = "#ffffff"
EDGE = "#8c9dad"
ACCESS = "#3d6da5"
DESTINATION = "#218479"
TARGET = "#33755c"
BLOCKER = "#c54539"
GUARD = "#a06418"
BACKGROUND_OBJECT = "#667988"
NEAR_PATH = "#067cce"
FAR_PATH = "#8e3da6"
FRAME_SIZE = (870, 500)


def font(size, *, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


TITLE_FONT = font(27, bold=True)
LABEL_FONT = font(17, bold=True)
SMALL_FONT = font(15)
TINY_FONT = font(13)


def read_resources(path):
    data = json.loads(path.read_text())
    geometries = tuple(task.ObjectGeometry(
        name=g["name"], asset_name=g["asset_name"], lower=tuple(g["lower"]), upper=tuple(g["upper"]),
        collision_footprint_vertices=tuple(map(tuple, g["collision_footprint_vertices"])),
        visual_footprint_vertices=tuple(map(tuple, g["visual_footprint_vertices"])),
    ) for g in data["geometries"])
    rules_data = data["rules"] | {"generation_mode": "sequential", "spawn_height_offset": 0.001}
    return geometries, task.SpawnRules(**rules_data)


def footprints(sample, geometries):
    return {
        ident: task._collision_polygon(geometries[ident], sample.yaws[ident]) + sample.positions[ident][:2]
        for ident in sample.active_ids
    }


def background_fill(ident, sample, guards):
    if ident == sample.target_id:
        return "#abd6bc", TARGET, "T"
    if ident == sample.blocker_id:
        return "#ffc2b6", BLOCKER, "B"
    if ident in guards:
        return "#f6d392", GUARD, "G"
    return "#d2dce5", BACKGROUND_OBJECT, ""


def dashed(draw, points, color, *, width=2, dash=8.0, space=5.0):
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0)
        if length == 0.0:
            continue
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        distance = 0.0
        while distance < length:
            end = min(distance + dash, length)
            draw.line((x0 + dx * distance, y0 + dy * distance, x0 + dx * end, y0 + dy * end),
                      fill=color, width=width)
            distance += dash + space


def draw_arrow(draw, points, color):
    draw.line(points, fill="#ffffff", width=8, joint="curve")
    dashed(draw, points, color, width=4, dash=12.0, space=6.0)
    x0, y0 = points[-2]
    x1, y1 = points[-1]
    length = math.hypot(x1 - x0, y1 - y0)
    if length > 0.0:
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        triangle = [(x1, y1), (x1 - 13 * dx - 6 * dy, y1 - 13 * dy + 6 * dx),
                    (x1 - 13 * dx + 6 * dy, y1 - 13 * dy - 6 * dx)]
        draw.polygon(triangle, fill=color, outline="#ffffff", width=1)


def verify_paths(sample, rules, geometries, polygons, layout, paths):
    near_side, near, far = layout
    statuses = {near_side: near, -near_side: far}
    paths = {int(side): tuple(path) for side, path in paths.items()}
    if set(paths) != {side for side, status in statuses.items() if status != "blocked"}:
        raise RuntimeError(f"Expected a witness for every open boundary in {layout}.")

    blocker = geometries[sample.blocker_id]
    initial = sample.positions[sample.blocker_id][:2]
    initial_yaw = sample.yaws[sample.blocker_id]
    obstacles = [tuple(map(tuple, polygon)) for ident, polygon in polygons.items() if ident != sample.blocker_id]
    routes = {}
    for side, path in paths.items():
        if not path:
            raise RuntimeError(f"Empty witness for {SIDE_TEXT[side]} boundary.")
        if hasattr(path[0], "start"):
            motions = path
        else:
            poses = [task.PlanarPose(float(point[0]), float(point[1]),
                                    float(point[2]) if len(point) > 2 else initial_yaw) for point in path]
            motions = tuple(task.PlanarMotion(start, finish) for start, finish in zip(poses, poses[1:]))
        if not motions:
            raise RuntimeError(f"Expected at least two poses in {SIDE_TEXT[side]} witness.")
        begin, finish = motions[0].start, motions[-1].end
        if (not np.allclose((begin.x, begin.y, begin.yaw), (*initial, initial_yaw), rtol=0.0, atol=1e-8)
                or any(first.end != second.start for first, second in zip(motions, motions[1:]))
                or not task._path_supported_clear(blocker, motions, obstacles, rules)):
            raise RuntimeError(f"Unsupported, disconnected or obstructed {SIDE_TEXT[side]} witness.")

        footprint = task.footprint_polygon(blocker, (finish.x, finish.y), finish.yaw)
        boundary = sample.access_region[2 if side < 0 else 3]
        if any(side * (y - boundary) < -1.e-8 for _, y in footprint):
            raise RuntimeError(f"The {SIDE_TEXT[side]} last footprint still overlaps Access.")
        if statuses[side] == "direct" and any(
            not np.allclose((motion.end.x, motion.end.yaw), (begin.x, begin.yaw), rtol=0.0, atol=1.e-8)
            for motion in motions
        ):
            raise RuntimeError(f"The {SIDE_TEXT[side]} direct route changes depth or orientation.")
        routes[side] = [[begin.x, begin.y], *[[motion.end.x, motion.end.y] for motion in motions]]
    return routes


def draw_legend(draw, y):
    items = (("Target", TARGET), ("Blocker", BLOCKER), ("Guard", GUARD), ("Access", ACCESS),
             ("Destination", DESTINATION), ("Near path", NEAR_PATH), ("Far path", FAR_PATH))
    x = 34
    for name, color in items:
        draw.rounded_rectangle((x, y + 4, x + 12, y + 16), radius=2, fill=color)
        draw.text((x + 18, y), name, font=SMALL_FONT, fill=INK)
        x += int(draw.textlength(name, font=SMALL_FONT)) + 36


def draw_scene(path, level, index, seed, layout, sample, rules, polygons, guards, paths, state):
    image = Image.new("RGB", FRAME_SIZE, PAPER)
    draw = ImageDraw.Draw(image)
    xmin, xmax = rules.x_range
    ymin, ymax = rules.y_range
    scale = min(795 / (ymax - ymin), 342 / (xmax - xmin))
    width, height = (ymax - ymin) * scale, (xmax - xmin) * scale
    left, top = (FRAME_SIZE[0] - width) / 2, 79

    def screen(point):
        x, y = point
        return (left + (y - ymin) * scale, top + (x - xmin) * scale)

    def box(region):
        return [screen(point) for point in task.rectangle_polygon(region)]

    near_side, near, far = layout
    draw.text((34, 16), f"{level.upper()}  /  {index:02d}", font=TITLE_FONT, fill=INK)
    note = f"Seed {seed}  •  {len(sample.active_ids)} objects"
    draw.text((FRAME_SIZE[0] - 34, 24), note, font=SMALL_FONT, fill=MUTED, anchor="ra")
    draw.text((34, 51), f"Near {SIDE_TEXT[near_side]}  {STATUS_TEXT[near]}", font=LABEL_FONT, fill=NEAR_PATH)
    draw.text((FRAME_SIZE[0] - 34, 51), f"Far {SIDE_TEXT[-near_side]}  {STATUS_TEXT[far]}",
              font=LABEL_FONT, fill=FAR_PATH, anchor="ra")
    draw.rounded_rectangle((left, top, left + width, top + height), radius=4, fill=SHELF, outline=EDGE, width=2)

    access = (xmin, sample.access_region[1], *sample.access_region[2:])
    outline = box(access)
    draw.polygon(outline, fill="#e5f0fc")
    dashed(draw, [*outline, outline[0]], ACCESS)
    for corridor in state["pocket_front_corridors"]:
        outline = box(corridor)
        draw.polygon(outline, fill="#e4f5f1")
        dashed(draw, [*outline, outline[0]], DESTINATION, width=1, dash=5.0, space=5.0)
    for region in sample.reserved_regions:
        outline = box(region)
        draw.polygon(outline, fill="#bfe5dd", outline=DESTINATION, width=3)

    roles = (sample.target_id, sample.blocker_id)
    labels = []
    for ident in (*[i for i in sample.active_ids if i not in roles], *roles):
        polygon = [screen(point) for point in polygons[ident]]
        fill, edge, label = background_fill(ident, sample, guards)
        draw.polygon(polygon, fill=fill, outline=edge, width=3 if label else 2)
        if label:
            labels.append((screen(sample.positions[ident][:2]), label, edge))

    for side, waypoints in paths.items():
        draw_arrow(draw, [screen(point) for point in waypoints], NEAR_PATH if side == near_side else FAR_PATH)
    for (x, y), label, color in labels:
        draw.rounded_rectangle((x - 9, y - 10, x + 9, y + 10), radius=3, fill="#ffffff")
        draw.text((x, y), label, font=TINY_FONT, fill=color, anchor="mm")

    draw.text((FRAME_SIZE[0] / 2, top + height + 12), "SHELF FRONT", font=LABEL_FONT, fill=MUTED, anchor="ma")
    draw_legend(draw, FRAME_SIZE[1] - 35)
    image.save(path)


def draw_contact_sheet(level, paths, output):
    columns = 5
    thumb_width, thumb_height = 470, 270
    margin, spacing = 30, 10
    rows = math.ceil(len(paths) / columns)
    width = 2 * margin + columns * thumb_width + (columns - 1) * spacing
    height = 126 + rows * thumb_height + (rows - 1) * spacing + margin
    sheet = Image.new("RGB", (width, height), "#d4dfe8")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 20), f"{level.upper()}  /  TOP VIEWS", font=font(44, bold=True), fill=INK)
    draw.text((margin, 78), f"{len(paths)} scenes generated with training geometry",
              font=font(24), fill=MUTED)
    for i, source in enumerate(paths):
        x = margin + i % columns * (thumb_width + spacing)
        y = 126 + i // columns * (thumb_height + spacing)
        with Image.open(source) as image:
            sheet.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
    sheet.save(output / f"{level}_top_views.png")


def generate_gallery(args):
    geometries, rules = read_resources(args.resources)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for level_index, (level, combinations) in enumerate(DIFFICULTIES.items()):
        directory = args.output / level
        directory.mkdir(exist_ok=True)
        balanced_layouts = cycle((side, near, far) for near, far in combinations for side in (-1, 1))
        paths = []
        for index in range(args.count):
            layout = next(balanced_layouts)
            seed = args.seed + level_index * args.count + index
            started = time.perf_counter()
            kwargs = {"hand_width_m": args.hand_width_m} if args.hand_width_m is not None else {}
            sample = task.sample_sequential_scene(geometries, rules, seed=seed, layout=layout, **kwargs)
            elapsed_ms = 1e3 * (time.perf_counter() - started)
            certificate = sample.parameter_certificate
            state = certificate["sequential_state"]
            details = state["metadata"]
            actual_layout = tuple(details[key] for key in ("layout_near_side", "layout_near_status", "layout_far_status"))
            if actual_layout != layout:
                raise RuntimeError(f"Seed {seed} expected {layout}, got {actual_layout}.")
            guards = tuple(details["layout_guards"])
            polygons = footprints(sample, geometries)
            routes = verify_paths(sample, rules, geometries, polygons, layout, details["layout_paths"])
            path = directory / f"layout_{index + 1:02d}.png"
            draw_scene(path, level, index + 1, seed, layout, sample, rules, polygons, guards, routes, state)
            paths.append(path)
            records.append({
                "difficulty": level, "index": index + 1, "seed": seed, "near_side": layout[0],
                "near_condition": layout[1], "far_condition": layout[2], "generation_ms": elapsed_ms,
                "image": str(path.relative_to(args.output)), "object_count": len(sample.active_ids),
                "target_asset": geometries[sample.target_id].asset_name,
                "blocker_asset": geometries[sample.blocker_id].asset_name,
                "guard_assets": [geometries[ident].asset_name for ident in guards],
                "blocker_path_xy_m": routes, "access_region_m": sample.access_region,
                "destinations_m": sample.reserved_regions,
                "hand_width_m": details.get("hand_width_m", args.hand_width_m),
            })
        draw_contact_sheet(level, paths, args.output)
        print(f"{level}: {len(paths)} top views saved to {directory}", flush=True)
    try:
        resources = args.resources.resolve().relative_to(ROOT)
    except ValueError:
        resources = args.resources.resolve()
    manifest = {"asset_geometry": str(resources), "scenes": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=30, help="Layouts per difficulty")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--hand-width-m", type=float)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generate_gallery(args)


if __name__ == "__main__":
    main()
