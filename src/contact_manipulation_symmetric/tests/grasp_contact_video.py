"""Offline, measured-contact overlay for the production hand grasp probe.

This module only reads an already recorded video and diagnostic samples. It
creates no simulation objects and never infers contact from rendered geometry.
"""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path


BACKGROUND = (18, 24, 33)
PANEL = (28, 36, 47)
TEXT = (230, 237, 245)
MUTED = (158, 174, 192)
GRID = (63, 76, 92)
OBJECT = (77, 222, 140)
OTHER = (255, 190, 72)
OFF = (66, 77, 92)


def _font(size):
    from PIL import ImageFont

    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _joint_label(name):
    result = str(name).removeprefix("right_").removesuffix("_joint")
    for finger, short in (("thumb", "Th"), ("index", "In"), ("middle", "Mi"),
                          ("ring", "Ri"), ("little", "Li")):
        result = result.replace(finger, short)
    return result.replace("_", "")


def _groups(pad_names):
    """Keep the canonical row IDs while allowing an explicit name ordering."""
    result = []
    for label, token in (("Palm", "plam"), ("Thumb", "thumb"), ("Index", "index"),
                         ("Middle", "middle"), ("Ring", "ring"), ("Little", "little")):
        indices = [i for i, name in enumerate(pad_names) if token in name]
        if label == "Palm" and not indices:
            indices = [i for i, name in enumerate(pad_names) if "palm" in name]
        result.append((label, indices))
    flattened = [i for _, indices in result for i in indices]
    if sorted(flattened) != list(range(17)):
        raise ValueError("The grasp overlay requires Palm 1, Thumb 4 and four fingers with 3 pads.")
    if [len(indices) for _, indices in result] != [1, 4, 3, 3, 3, 3]:
        raise ValueError("Unexpected tactile group sizes in diagnostic pad names.")
    return result


def render_overlay(raw_video, output_video, samples, frame_indices, metadata):
    """Render canonical pad contact states, forces and measured joint speeds.

    Green means contact with the test object in the independent filtered
    contact view. Amber means the production net-contact bit is ON while the
    object-contact bit is OFF. The production and object counts are both shown;
    neither is substituted for the other. The force bars use one fixed scale
    across the whole recording, with a minimum full scale of 1 N.

    Returns paths and recording dimensions for the caller's diagnostic report.
    """
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    pad_names = list(metadata["pad_names"])
    joint_names = list(metadata["joint_names"])
    if len(pad_names) != 17 or len(joint_names) != 12:
        raise ValueError("The grasp overlay requires 17 tactile pads and 12 measured hand joints.")
    groups = _groups(pad_names)
    if not samples or len(frame_indices) == 0:
        raise ValueError("The grasp overlay needs recorded samples and video sample indices.")
    if any(not isinstance(i, (int, np.integer)) or i < 0 or i >= len(samples) for i in frame_indices):
        raise ValueError("A video frame points outside the recorded diagnostic samples.")
    threshold = float(metadata["threshold_N"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("The tactile threshold must be finite and nonnegative.")

    series = defaultdict(list)
    max_force = 0.0
    for sample in samples:
        if any(len(sample[key]) != 17 for key in
               ("binary", "object_binary", "pad_force_N", "object_pad_force_N")):
            raise ValueError("A diagnostic sample has an incomplete 17-pad measurement.")
        if len(sample["joint_velocity_rad_s"]) != 12:
            raise ValueError("A diagnostic sample has an incomplete 12-joint velocity measurement.")
        key = str(sample["trial_name"])
        series[key].append((float(sample["time_s"]), sum(bool(x) for x in sample["object_binary"]),
                            sum(bool(x) for x in sample["binary"])))
        finite = [float(x) for x in sample["object_pad_force_N"] if math.isfinite(float(x))]
        max_force = max(max_force, max(finite, default=0.0))
    force_scale = max(1.0, max_force)

    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = output.with_name(output.stem + "_preview.png")
    font, small, tiny = _font(22), _font(18), _font(14)
    large = _font(30)
    reader = imageio.get_reader(str(raw_video))
    writer = None
    frame_count, best_preview = 0, (-1, -1.0)
    size = None
    try:
        fps = float(metadata.get("video_fps", reader.get_meta_data().get("fps", 25)))
        writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8, macro_block_size=1)
        for frame_index, frame in enumerate(reader):
            if frame_index >= len(frame_indices):
                raise ValueError("The recorded video contains more frames than its sample index map.")
            sample = samples[int(frame_indices[frame_index])]
            raw = Image.fromarray(np.asarray(frame)[..., :3]).convert("RGB")
            # The two rendered views stay unchanged above the offline footer.
            # Normal production output is 1280 pixels wide; scale this layout
            # for a caller that records another camera resolution.
            width, raw_height = raw.size
            scale = width / 1280.0
            footer_height = max(400, round(420 * scale))
            image = Image.new("RGB", (width, raw_height + footer_height), BACKGROUND)
            image.paste(raw, (0, 0))
            draw = ImageDraw.Draw(image)

            def xy(x, y):
                return round(x * scale), raw_height + round(y * scale)

            def label(x, y, value, fill=TEXT, face=small):
                draw.text(xy(x, y), str(value), font=face, fill=fill)

            def box(x0, y0, x1, y1, **kwargs):
                draw.rectangle((*xy(x0, y0), *xy(x1, y1)), **kwargs)

            object_bits = [bool(x) for x in sample["object_binary"]]
            net_bits = [bool(x) for x in sample["binary"]]
            object_count = sum(object_bits)
            net_count = sum(net_bits)
            other_count = sum(net and not obj for net, obj in zip(net_bits, object_bits))
            phase = str(sample["phase"]).replace("_", " ")
            trial = str(sample["trial_name"]).replace("_", " ")
            fractions = np.asarray(sample["hand_fraction"], dtype=float).reshape(-1)
            if len(fractions) not in (1, 6) or not np.isfinite(fractions).all():
                raise ValueError("Hand command must contain six finite motor fractions.")
            opposition = float(fractions[0])
            flexion = float(fractions[1:].mean()) if len(fractions) == 6 else opposition
            label(18, 8, f"Object {object_count:2d} / 17", OBJECT, large)
            label(274, 13, f"Net {net_count:2d} / 17", TEXT, font)
            label(466, 13, f"Other {other_count:2d}", OTHER, font)
            label(674, 8, f"{trial}  |  {phase}", TEXT, font)
            label(674, 37, f"Thumb opposition {100 * opposition:.0f}%  |  Flexion {100 * flexion:.0f}%")

            label(18, 48, "Pad / object normal force [N]", MUTED, tiny)
            label(354, 48, f"ON > {threshold:g} N   |   bar 0 - {force_scale:.2f} N", MUTED, tiny)
            for row, (group, indices) in enumerate(groups):
                y = 72 + row * 43
                label(18, y + 7, group)
                for col, pad in enumerate(indices):
                    x = 115 + col * 127
                    color = OBJECT if object_bits[pad] else OTHER if net_bits[pad] else OFF
                    box(x, y, x + 117, y + 34, fill=PANEL, outline=color, width=2)
                    label(x + 7, y + 3, f"{pad:02d}", color, small)
                    force = float(sample["object_pad_force_N"][pad])
                    value = f"{force:.2f}" if math.isfinite(force) else "invalid"
                    label(x + 40, y + 4, value, TEXT, tiny)
                    box(x + 4, y + 28, x + 113, y + 31, fill=GRID)
                    if math.isfinite(force) and force > 0:
                        box(x + 4, y + 28, x + 4 + 109 * min(force / force_scale, 1.0), y + 31,
                            fill=color)

            # One time axis per independent trial, with future measurements
            # hidden until their frame is reached.
            graph = (710, 100, 1252, 220)
            x0, y0, x1, y1 = graph
            label(674, 74, "Simultaneous pads", TEXT, small)
            entries = series[str(sample["trial_name"])]
            start, end = entries[0][0], entries[-1][0]
            duration = max(end - start, float(metadata.get("dt", .01)))
            for count in (0, 5, 10, 17):
                y = y1 - (y1 - y0) * count / 17.0
                draw.line((xy(x0, y), xy(x1, y)), fill=GRID, width=1)
                label(x0 - 28, y - 8, count, MUTED, tiny)
            visible = [entry for entry in entries if entry[0] <= float(sample["time_s"]) + 1.e-8]
            # Net is drawn first so equal traces leave object green visible.
            for component, color in ((2, OTHER), (1, OBJECT)):
                points = [xy(x0 + (x1 - x0) * (t - start) / duration,
                             y1 - (y1 - y0) * entry[component] / 17.0)
                          for entry in visible for t in [entry[0]]]
                if len(points) > 1:
                    draw.line(points, fill=color, width=2)
            now = x0 + (x1 - x0) * (float(sample["time_s"]) - start) / duration
            draw.line((xy(now, y0), xy(now, y1)), fill=TEXT, width=1)
            label(x0, y1 + 5, "0 s", MUTED, tiny)
            label(x1 - 64, y1 + 5, f"{duration:.1f} s", MUTED, tiny)
            label(880, y1 + 5, "Object", OBJECT, tiny)
            label(972, y1 + 5, "Net", OTHER, tiny)

            speeds = [float(x) for x in sample["joint_velocity_rad_s"]]
            finite_speeds = [(abs(speed), joint_names[i]) for i, speed in enumerate(speeds)
                             if math.isfinite(speed)]
            maximum, fastest_name = max(finite_speeds, default=(math.nan, "invalid"))
            label(674, 254, f"Max |joint velocity| {maximum:.3f} rad/s  ({_joint_label(fastest_name)})", TEXT, small)
            for index, (name, speed) in enumerate(zip(joint_names, speeds)):
                x, y = 680 + (index % 6) * 96, 289 + (index // 6) * 38
                label(x, y, _joint_label(name), MUTED, tiny)
                label(x, y + 16, f"{speed:+.3f}" if math.isfinite(speed) else "invalid", TEXT, tiny)

            settings = metadata.get("hand_setting", {})
            label(18, 346, "Object contact", OBJECT, tiny)
            label(151, 346, "Other net contact", OTHER, tiny)
            label(316, 346, "OFF", MUTED, tiny)
            hand_bodies = {str(name): float(force) for name, force in
                           sample.get("hand_body_object_force_N", {}).items()
                           if math.isfinite(float(force)) and float(force) > threshold}
            tactile_names = set(pad_names)
            non_pad = [name for name in hand_bodies if name not in tactile_names]
            label(18, 371, f"Non-pad links on object: {len(non_pad)}", MUTED, small)
            label(674, 371, f"Kp {settings.get('stiffness', '?')}  |  Kd {settings.get('damping', '?')}  |  "
                  f"Armature {settings.get('armature', '?')}", MUTED, tiny)
            if "objects" in metadata:
                fixture = metadata["objects"][int(sample["trial_index"])]
            else:
                fixture = metadata.get("object", {})
            body_type = "fixed" if fixture.get("fixed") else "dynamic"
            mass = fixture.get("mass_kg", "?")
            dimensions = np.asarray(fixture.get("size_m", []), dtype=float).reshape(-1)
            size_label = " x ".join(f"{1000 * x:g}" for x in dimensions)
            object_label = f"{size_label} mm {fixture.get('shape', 'Object')}" if size_label else fixture.get("shape", "Object")
            label(18, 397, f"{object_label}  |  {mass} kg  |  {body_type}", MUTED, tiny)
            badge = f"{'FIXED' if fixture.get('fixed') else 'DYNAMIC'} OBJECT"
            if size_label:
                badge += f"  |  {size_label} mm"
            # An explicit fixture badge remains visible in the rendered view
            # even when a viewer is inspecting only the upper video region.
            # This is offline image annotation, without scene geometry.
            badge_position = (18, 12)
            badge_bounds = draw.textbbox(badge_position, badge, font=font)
            draw.rounded_rectangle((badge_bounds[0] - 8, badge_bounds[1] - 6,
                                    badge_bounds[2] + 8, badge_bounds[3] + 6),
                                   radius=5, fill=(10, 16, 24))
            draw.text(badge_position, badge, font=font, fill=OTHER if fixture.get("fixed") else OBJECT)
            dt = float(metadata.get("dt", .01))
            decimation = int(metadata.get("policy_decimation", 1))
            label(674, 397, f"t = {float(sample['time_s']):.2f} s  |  Physics {1. / dt:g} Hz  |  "
                  f"Policy {1. / (dt * decimation):g} Hz", MUTED, tiny)

            writer.append_data(np.asarray(image))
            score = (object_count, sum(float(x) for x in sample["object_pad_force_N"]
                                       if math.isfinite(float(x))))
            if score > best_preview:
                image.save(preview)
                best_preview = score
            frame_count += 1
            size = list(image.size)
        if frame_count != len(frame_indices):
            raise ValueError("The recorded video contains fewer frames than its sample index map.")
    finally:
        reader.close()
        if writer is not None:
            writer.close()
    return {"video": str(output), "preview": str(preview), "frames": frame_count,
            "output_size": size, "fps": fps, "pad_force_bar_full_scale_N": force_scale,
            "preview_object_contact_count": best_preview[0]}
