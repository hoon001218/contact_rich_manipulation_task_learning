"""Add the hand reward measurements to a two-view Isaac Sim diagnostic video."""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


GROUPS = ("palm", "thumb", "index", "middle", "ring", "little")
BACKGROUND = "#101925"
PANEL = "#1a2737"
TEXT = "#f1f6fc"
MUTED = "#aebed0"
DISTANCE = "#74ddf5"
ALIGNMENT = "#c8aeff"
CONTACT = "#82edb2"
NEGATIVE = "#ffb998"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = FONT_DIR / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _number(value, precision: int, *, signed: bool = False) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    value = float(value)
    if abs(value) < 0.5 * 10 ** -precision:
        value = 0.0
    return f"{value:+.{precision}f}" if signed else f"{value:.{precision}f}"


def _group_indices(names):
    groups = {name: [] for name in GROUPS}
    for index, name in enumerate(names):
        segments = name.lower().split("_")
        group = "palm" if "plam" in segments else next(
            (key for key in GROUPS if key in segments), None)
        if group is None:
            raise ValueError(f"Unrecognized tactile pad name {name!r}.")
        groups[group].append(index)
    return groups


def _mean(values, ids):
    finite = [float(values[i]) for i in ids if values[i] is not None and math.isfinite(float(values[i]))]
    return sum(finite) / len(finite) if finite else None


def _draw_overlay(frame, row, groups, alignment_mask, weights):
    views = Image.fromarray(frame).convert("RGB")
    width, height = views.size
    header, metrics_height = 58, 166
    image = Image.new("RGB", (width, height + header + metrics_height), BACKGROUND)
    image.paste(views, (0, header))
    draw = ImageDraw.Draw(image)
    draw.line((width // 2, header, width // 2, header + height), fill=BACKGROUND, width=2)
    contact_trial = row["mode"] == "contact"
    approach = "phase" in row
    grasp_candidate = "hand_raw" in row
    mode = (f"{row['phase']}   ·   {row['time_s']:.2f} s" if approach else
            "Live contact" if contact_trial else "Fixed-pose geometry")
    asset = row["asset"].replace("ycb_", "YCB ").replace("_", " ")
    draw.text((18, 8), asset, font=_font(22, True), fill=TEXT)
    draw.text((18, 36), f"{mode}    ·    {row['case']}", font=_font(13), fill=MUTED)
    command_label = (f"Closure {100. * row['raw']:.0f}%" if grasp_candidate else
                     f"{row['joint']}  {_number(row['raw'], 2, signed=True)} raw")
    draw.text((width - 18, 10), command_label,
              anchor="ra", font=_font(17, True), fill=TEXT)
    if approach:
        draw.text((width - 18, 36), f"Projected gap  {_number(row['approach_gap_mm'], 1)} mm",
                  anchor="ra", font=_font(13), fill=MUTED)
    elif grasp_candidate:
        readout = "Thumb and finger forces oppose" if row["opposed"] else "Blocker touch"
        if not any(row["contacts"]):
            readout = "No tactile contact"
        draw.text((width - 18, 36), readout, anchor="ra", font=_font(13), fill=MUTED)
    elif row["gap_mm"] is not None:
        draw.text((width - 18, 36), f"Initial pose gap  {_number(row['gap_mm'], 1)} mm",
                  anchor="ra", font=_font(13), fill=MUTED)

    distance_delta, alignment_delta = float(row["d_delta"]), float(row["a_delta"])
    total = distance_delta + alignment_delta
    cards = [
        ("Blocker distance", _number(row["distance"], 3), _number(distance_delta, 3, signed=True), DISTANCE),
        ("Pad normal alignment", _number(row["alignment"], 3), _number(alignment_delta, 3, signed=True), ALIGNMENT),
        ("Sum of weighted Δ", _number(total, 3, signed=True), "", CONTACT if total >= 0 else NEGATIVE),
    ]
    if contact_trial:
        count = sum(map(bool, row["contacts"]))
        forces = [float(value) for value in row["force_N"] if value is not None and math.isfinite(float(value))]
        peak = _number(max(forces, default=0.0), 2)
        hold = _number(weights[2] * float(row["contact_hold"]), 1, signed=True)
        cards.append(("Blocker tactile", f"{count} / 17 ON", f"{peak} N peak  ·  Hold {hold}", CONTACT))
    heading = "Quality and weighted Δ from the case baseline (points/s)"
    if approach:
        distance, alignment, hold = (weight * float(row[field]) for weight, field in
                                     zip(weights, ("distance", "alignment", "contact_hold")))
        cards = [
            ("Blocker distance", _number(distance, 3), "", DISTANCE),
            ("Pad normal alignment", _number(alignment, 3), "", ALIGNMENT),
            ("Contact Hold", _number(hold, 3), f"{count} / 17 pads", CONTACT),
            ("Sum of these three", _number(distance + alignment + hold, 3), "", TEXT),
        ]
        heading = "Weighted reward (points/s)"

    start_y = height + header
    draw.text((18, start_y + 7), heading, font=_font(13), fill=MUTED)
    card_width = (width - 36 - (len(cards) - 1) * 8) / len(cards)
    for index, (title, value, delta, color) in enumerate(cards):
        x = 18 + index * (card_width + 8)
        y = start_y + 25
        draw.rounded_rectangle((x, y, x + card_width, y + 51), radius=7, fill=PANEL)
        draw.text((x + 10, y + 5), title, font=_font(13), fill=MUTED)
        draw.text((x + 10, y + 21), value, font=_font(20, True), fill=color)
        if delta:
            draw.text((x + card_width - 10, y + 25), delta, anchor="ra", font=_font(14), fill=TEXT)

    y = start_y + 86
    pad_legend = "Pad group means  ·  OBB distance (mm)  ·  cosine of alignment pads"
    if contact_trial:
        pad_legend += "  ·  green = Blocker contact"
    draw.text((18, y), pad_legend, font=_font(13), fill=MUTED)
    group_width = (width - 36 - (len(groups) - 1) * 8) / len(groups)
    for index, (title, ids) in enumerate(groups.items()):
        x = 18 + index * (group_width + 8)
        top = y + 19
        draw.rounded_rectangle((x, top, x + group_width, top + 52), radius=7, fill=PANEL)
        draw.text((x + 9, top + 5), title.capitalize(), font=_font(14, True), fill=TEXT)
        distance = _mean(row["distance_mm"], ids)
        cosine = _mean(row["cosine"], [i for i in ids if alignment_mask[i]])
        draw.text((x + 9, top + 28), _number(distance, 1), font=_font(17, True), fill=DISTANCE)
        draw.text((x + group_width - 9, top + 28), _number(cosine, 2, signed=True),
                  anchor="ra", font=_font(17, True), fill=ALIGNMENT)
        if contact_trial:
            for j, pad in enumerate(reversed(ids)):
                dot_x = x + group_width - 12 - j * 10
                draw.ellipse((dot_x - 3, top + 9, dot_x + 3, top + 15), outline=MUTED,
                             fill=CONTACT if row["contacts"][pad] else None)
    return np.asarray(image)


def render_hand_reward_sweep(raw_video: Path, metrics: Path | dict, output: Path) -> Path:
    """Encode the two native views plus synchronized reward and tactile measurements."""
    data = metrics if isinstance(metrics, dict) else json.loads(Path(metrics).read_text())
    rows, pad_names, alignment_mask = data["frames"], data["pad_names"], data["alignment_mask"]
    fps = float(data["video_fps"])
    weights = tuple(map(float, data["weights"]))
    if not rows or len(pad_names) != 17 or len(alignment_mask) != 17 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Expected nonempty measurements, 17 tactile pads and a finite positive video FPS.")
    groups = _group_indices(pad_names)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.rendering{output.suffix}")
    try:
        with imageio.get_reader(str(raw_video)) as reader, imageio.get_writer(
            str(temporary), fps=fps, codec="libx264", quality=8, macro_block_size=2,
        ) as writer:
            frames = iter(reader)
            for index, row in enumerate(rows):
                if row["mode"] not in ("geometry", "contact"):
                    raise ValueError(f"Unrecognized measurement mode at frame {index}: {row['mode']!r}.")
                if any(len(row[key]) != 17 for key in ("contacts", "distance_mm", "cosine", "force_N")):
                    raise ValueError(f"Expected 17 tactile measurements at frame {index}.")
                frame = next(frames, None)
                if frame is None:
                    raise ValueError(f"Video ended at frame {index}, before all {len(rows)} measurements.")
                writer.append_data(_draw_overlay(frame, row, groups, alignment_mask, weights))
            if next(frames, None) is not None:
                raise ValueError(f"Video contains more than {len(rows)} measured frames.")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_video", type=Path)
    parser.add_argument("metrics_json", type=Path)
    parser.add_argument("output_video", type=Path)
    args = parser.parse_args()
    print(render_hand_reward_sweep(args.raw_video, args.metrics_json, args.output_video))
