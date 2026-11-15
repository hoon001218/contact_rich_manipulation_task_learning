"""Record the production F/T compensation during physical, noncontact motion.

Run through Isaac Lab after other GPU simulations have stopped::

    ./IsaacLab/isaaclab.sh -p src/contact_manipulation_symmetric/tests/probe_ft_compensation.py --headless

The assembled UR5e, load cell, hand, gravity, masses, inertias, actuators and
physics timestep come from contact_manipulation_symmetric. The actual production
SensorFrameWrenchAction/native OSC executes smooth sensor-frame wrench commands,
and UrdfCoupledHandAction controls the fingers. In the default F/T mode,
joint/root states are never overwritten after initial reset and no external
object touches the hand.

Incoming joint wrench is acquired through the production SensorReader, then
the unchanged calibrate_wrench/self_motion_wrench helpers produce exactly the
sensor observation's clip -> quantize -> subtract-self-load sequence. Video
axes and overlays are visual only. The report contains native raw reactions,
calibrated reactions, gravity/inertia predictions and compensated residuals at
every physics step, and reports measurements without invented pass thresholds.

Add --tactile-scan to instead record all 17 production binary tactile channels.
A dynamic, gravity-free sphere follows a native prismatic drive into each pad
and withdraws. Robot/probe resets occur only between independent pad cases;
within a case every contact and displacement comes from actual PhysX dynamics.

Add --osc-push for a supported 300 g box pushed by the assembled robot's actual
production wrench OSC. Only initial states are written: no hand guide, external
body force, or pose replacement drives either the hand or the box.

Add --osc-torque-staircase for independent sensor +Z torque trials. Each level
starts from the same pose at rest; all six arm joints remain torque controlled.
The training overspeed threshold is displayed, not used to reset the trial.
Add --osc-wrench-sweep to repeat the same independent trials for every sensor
force/torque axis and compare all six UR5e joint velocities.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", REPOSITORY_ROOT / "IsaacLab")).expanduser().resolve()
sys.path[:0] = [str(REPOSITORY_ROOT / "src" / "contact_manipulation_symmetric"),
               *(str(ISAACLAB_ROOT / "source" / package) for package in
                 ("isaaclab", "isaaclab_tasks", "isaaclab_rl"))]

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-output", type=Path)
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-steps", type=int, help="Optional early stop for a short physical smoke test.")
    parser.add_argument("--force-amplitude", type=float, default=.3,
                        help="Coefficient in A*(cos(w*t)-cos(2*w*t)); peak magnitude is at most 2*A N.")
    parser.add_argument("--torque-amplitude", type=float, default=.012,
                        help="Same waveform coefficient for torque axes in N m.")
    parser.add_argument("--frequency", type=float, default=1., help="Oscillation base frequency in Hz.")
    parser.add_argument("--max-arm-speed", type=float, default=2., help="Diagnostic stop threshold in rad/s.")
    parser.add_argument("--tactile-scan", action="store_true", help="Record real contact / release on all 17 tactile pads.")
    parser.add_argument("--osc-push", action="store_true", help="Push a supported dynamic box with the assembled robot's OSC.")
    parser.add_argument("--osc-torque-staircase", action="store_true",
                        help="Compare signed sensor +Z torque levels from the same initial robot state.")
    parser.add_argument("--osc-wrench-sweep", action="store_true",
                        help="Compare signed sensor force/torque inputs and all six UR5e velocities.")
    parser.add_argument("--wrench-axes", nargs="+", choices=("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"),
                        default=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
    parser.add_argument("--force-levels", type=float, nargs="+", default=[.1, .5, 1., 2., 5., 10.],
                        help="Nonnegative force magnitudes in N, applied separately in both directions.")
    parser.add_argument("--torque-levels", type=float, nargs="+", default=[0., .001, .002, .005, .01, .02, .05, .1],
                        help="Nonnegative torque magnitudes in N m; positive and negative trials are generated.")
    parser.add_argument("--torque-hold", "--wrench-hold", type=float, default=.8,
                        help="Duration of each constant input pulse, in seconds.")
    parser.add_argument("--push-force", type=float, default=1.8, help="Peak scripted horizontal push wrench force, in N.")
    parser.add_argument("--push-hand-fraction", type=float, default=.15, help="Fixed hand position fraction within native limits.")
    parser.add_argument("--push-orientation-kp", type=float, default=.5, help="Diagnostic wrench-generator angular gain, N m/rad.")
    parser.add_argument("--push-orientation-kd", type=float, default=.02, help="Diagnostic wrench-generator angular damping, N m s/rad.")
    parser.add_argument("--tactile-count", type=int, default=17, help="Number of pad cases; use 2 for an initial smoke test.")
    parser.add_argument("--tactile-start", type=int, default=0, help="First canonical pad channel, for diagnostic reruns.")
    parser.add_argument("--tactile-penetration", type=float, default=.0015,
                        help="Diagnostic drive target beyond first-touch position, in m; actual penetration is dynamic.")
    parser.add_argument("--tactile-radius", type=float, default=.0015,
                        help="Radius of the dynamic external probe; small enough to avoid covering adjacent fingertip housing.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video_fps < 1:
        parser.error("--video-fps must be positive.")
    if sum((args.osc_push, args.tactile_scan, args.osc_torque_staircase, args.osc_wrench_sweep)) > 1:
        parser.error("OSC push, torque staircase, wrench sweep and tactile scan are separate diagnostic modes.")
    if any(not math.isfinite(value) or not 0 <= value <= 8. for value in args.torque_levels):
        parser.error("Torque magnitudes must be finite and within [0, 8] N m.")
    if any(not math.isfinite(value) or not 0 <= value <= 200. for value in args.force_levels):
        parser.error("Force magnitudes must be finite and within [0, 200] N.")
    if not math.isfinite(args.torque_hold) or args.torque_hold <= 0:
        parser.error("--torque-hold must be finite and positive.")
    if not math.isfinite(args.push_force) or not 0 < args.push_force <= 8.:
        parser.error("--push-force must be in (0, 8] N.")
    if not math.isfinite(args.push_hand_fraction) or not 0 <= args.push_hand_fraction <= .5:
        parser.error("--push-hand-fraction must be in [0, .5].")
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in
           ("push_orientation_kp", "push_orientation_kd")):
        parser.error("Diagnostic angular gains must be finite and nonnegative.")
    if not 0 <= args.tactile_start < 17 or not 1 <= args.tactile_count <= 17 - args.tactile_start:
        parser.error("Tactile channel range must be a nonempty subset of 0..16.")
    if not math.isfinite(args.tactile_penetration) or not 0 < args.tactile_penetration <= .003:
        parser.error("--tactile-penetration must be in (0, .003] m.")
    if not math.isfinite(args.tactile_radius) or not .0005 <= args.tactile_radius <= .006:
        parser.error("--tactile-radius must be in [.0005, .006] m.")
    if any(not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0. for name in
           ("force_amplitude", "torque_amplitude", "frequency", "max_arm_speed")):
        parser.error("Wrench amplitudes, frequency and speed threshold must be finite and positive.")
    if not args.no_video:
        args.enable_cameras = True
    directory = ("osc_wrench_sweep_2026-11-14" if args.osc_wrench_sweep else
                 "osc_torque_staircase_2026-11-14" if args.osc_torque_staircase else
                 "osc_hand_push_2026-11-08" if args.osc_push else
                 "tactile_contact_2026-11-08" if args.tactile_scan else "ft_compensation_2026-11-08")
    if args.output is None:
        args.output = REPOSITORY_ROOT / "logs/diagnostics" / directory / "report.json"
    if args.video_output is None:
        filename = ("wrench_sweep.mp4" if args.osc_wrench_sweep else
                    "torque_staircase.mp4" if args.osc_torque_staircase else
                    "osc_push.mp4" if args.osc_push else "all_pads.mp4" if args.tactile_scan else "ft_shake.mp4")
        args.video_output = REPOSITORY_ROOT / "docs/videos" / directory / filename
    return args


PHASES = (("zero_wrench", 2.), ("force_x", 2.), ("force_y", 2.), ("force_z", 2.),
          ("torque_x", 2.), ("torque_y", 2.), ("torque_z", 2.),
          ("finger_motion", 5.), ("final_zero_wrench", 2.))


def smooth_step(value):
    value = max(0., min(1., value))
    return value**3 * (10. - 15. * value + 6. * value**2)


def motion_command(phase, time, duration, force_amplitude, torque_amplitude, frequency):
    """Zero-impulse full-period wrench bursts; no position stabilization added."""
    wrench, hand_fraction = [0.] * 6, 0.
    if phase.startswith(("force_", "torque_")):
        axis = "xyz".index(phase[-1]) + (3 if phase.startswith("torque_") else 0)
        amplitude = torque_amplitude if axis >= 3 else force_amplitude
        omega = 2. * math.pi * frequency
        wrench[axis] = amplitude * (math.cos(omega * time) - math.cos(2. * omega * time))
    elif phase == "finger_motion":
        # Slow free-space closure/reopening remains inside every source limit.
        envelope = smooth_step(time / .6) * smooth_step((duration - time) / .6)
        hand_fraction = .45 * envelope * .5 * (1. - math.cos(2. * math.pi * .4 * time))
    return wrench, hand_fraction


def torque_trial_command(time, level, hold):
    """An exact step pulse, with baseline/release and no motion feedback added."""
    baseline = .4
    stage = "baseline" if time < baseline - 1.e-9 else "torque" if time < baseline + hold - 1.e-9 else "release"
    return ([0.] * 5 + [level if stage == "torque" else 0.]), stage


def summarize_wrench_trials(samples, metadata, dt):
    """Offline per-axis response; only the six UR5e joints are evaluated."""
    import numpy as np

    result = []
    names = metadata["arm_joint_names"]
    limits = np.asarray(metadata["arm_joint_velocity_limits_rad_s"])
    reference = metadata["trained_velocity_threshold_rad_s"]
    for index, trial in enumerate(metadata["wrench_trials"]):
        rows = [sample for sample in samples if sample["trial_index"] == index]
        if not rows:
            continue
        velocities = np.asarray([sample["arm_joint_velocity_rad_s"] for sample in rows])
        times = np.asarray([sample["trial_time_s"] for sample in rows])
        active = np.asarray([sample["trial_stage"] == "command" for sample in rows])
        active_rows = np.flatnonzero(active)
        baseline_rows = np.flatnonzero(times <= .4 + 1.e-9)
        raw = np.asarray([sample["osc_raw_joint_effort_Nm"] for sample in rows])
        applied = np.asarray([sample["arm_applied_joint_effort_Nm"] for sample in rows])
        joints = {}
        for column, name in enumerate(names):
            values = velocities[:, column]
            above = np.flatnonzero(np.abs(values) > reference)
            joints[name] = {
                "speed_min_rad_s": float(values.min()), "speed_max_rad_s": float(values.max()),
                "speed_max_abs_rad_s": float(np.abs(values).max()),
                "rms_rad_s": float(np.sqrt(np.mean(values**2))),
                "native_velocity_limit_rad_s": float(limits[column]),
                "samples_over_native_limit": int((np.abs(values) > limits[column] + 1.e-5).sum()),
                "samples_over_training_reference": int(len(above)),
                "first_over_training_reference_after_onset_s": float(times[above[0]] - .4) if len(above) else None,
                "speed_on_first_command_step_rad_s": float(values[active_rows[0]]) if len(active_rows) else None,
                "initial_acceleration_estimate_rad_s2": float((values[active_rows[0]] - values[baseline_rows[-1]]) / dt)
                    if len(active_rows) and len(baseline_rows) else None,
                "peak_raw_osc_effort_Nm": float(np.abs(raw[:, column]).max()),
                "peak_applied_effort_Nm": float(np.abs(applied[:, column]).max()),
                "max_raw_applied_difference_Nm": float(np.abs(raw[:, column] - applied[:, column]).max()),
            }
        peak_column = int(np.abs(velocities).max(0).argmax())
        result.append({
            "trial_index": index, "axis": ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[trial["axis"]],
            "level": trial["level"], "unit": "N" if trial["axis"] < 3 else "N m", "samples": len(rows),
            "applied_input_duration_s": int(active.sum()) * dt,
            "completed_full_hold": int(active.sum()) >= round(metadata["trial_timing"]["torque_hold_s"] / dt),
            "maximum_speed_joint": names[peak_column],
            "maximum_speed_rad_s": joints[names[peak_column]]["speed_max_abs_rad_s"],
            "maximum_sensor_displacement_m": max(float(np.linalg.norm(s["sensor_displacement_w_m"])) for s in rows),
            "minimum_body_origin_height_m": min(s["minimum_body_origin_height_m"] for s in rows),
            "maximum_tactile_force_N": max(s["max_tactile_normal_force_N"] for s in rows),
            "all_arm_states_finite": bool(np.isfinite(velocities).all() and
                np.isfinite([s["arm_joint_position_rad"] for s in rows]).all()),
            "joints": joints,
        })
    return result


def write_wrench_velocity_csv(directory, samples, metadata, summaries):
    """Keep every physics sample and per-trial/joint limits reviewable in CSV."""
    import csv

    names = metadata["arm_joint_names"]
    with (directory / "ur5e_velocities.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "axis", "level", "time_s", "trial_time_s", "stage",
                         *[f"{name}_rad_s" for name in names]])
        for sample in samples:
            writer.writerow([sample["trial_index"], ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[sample["trial_axis"]],
                             sample["trial_level"], sample["time_s"], sample["trial_time_s"], sample["trial_stage"],
                             *sample["arm_joint_velocity_rad_s"]])
    with (directory / "ur5e_velocity_summary.csv").open("w", newline="") as stream:
        fields = list(summaries[0]["joints"][names[0]]) if summaries else []
        writer = csv.writer(stream)
        writer.writerow(["trial", "axis", "level", "unit", "joint", *fields])
        for trial in summaries:
            for name, values in trial["joints"].items():
                writer.writerow([trial["trial_index"], trial["axis"], trial["level"], trial["unit"], name,
                                 *[values[key] for key in fields]])


def _phase_statistics(samples, phases=PHASES):
    import numpy as np

    result = {}
    for phase, _ in phases:
        selected = [s for s in samples if s["phase"] == phase and s["time_s"] >= 1.]
        if not selected:
            continue
        entry = {"samples": len(selected)}
        for field in ("raw_wrench", "measured_wrench", "self_load", "compensated_wrench", "ideal_compensated_wrench"):
            values = np.asarray([s[field] for s in selected])
            entry[field] = {"mean": values.mean(0).tolist(), "rms": np.sqrt((values**2).mean(0)).tolist(),
                            "max_abs": np.abs(values).max(0).tolist()}
        entry["max_tactile_normal_force_N"] = max(s["max_tactile_normal_force_N"] for s in selected)
        entry["overrange_samples"] = sum(any(s["overrange"]) for s in selected)
        entry["arm_max_speed_rad_s"] = max(s["arm_max_speed_rad_s"] for s in selected)
        poses = np.asarray([s["sensor_pose_w"] for s in selected])
        entry["sensor_translation_range_m"] = np.ptp(poses[:, :3], axis=0).tolist()
        dots = np.abs(poses[:, 3:] @ poses[0, 3:])
        entry["sensor_max_rotation_from_phase_start_rad"] = float(2. * np.arccos(np.clip(dots, 0., 1.)).max())
        result[phase] = entry
    return result


def encode_overlay(scene_video, output, samples, sample_indices, fps, metadata):
    """Draw plots afterwards so every frame uses the same measured axis scale."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    tiny = ImageFont.truetype("DejaVuSans.ttf", 13)
    pushing = metadata.get("experiment") == "osc_push"
    times = np.asarray([s["time_s"] for s in samples])
    # The first native impulse when constraints initialise is retained in JSON,
    # but does not hide the subsequent motion behind an enormous plot scale.
    valid = times >= 1.
    channels = {key: np.asarray([s[key] for s in samples]) for key in
                ("measured_wrench", "self_load", "compensated_wrench")}
    scale_values = list(channels.values())
    commands = np.asarray([s["wrench_command"] for s in samples])
    if pushing:
        scale_values.append(commands)
    limits = np.maximum(np.max(np.abs(np.stack([v[valid] for v in scale_values])), axis=(0, 1)) * 1.1,
                        np.asarray([1., 1., 1., .08, .08, .08]))
    palette = {"measured_wrench": (248, 192, 75), "self_load": (160, 185, 230),
               "compensated_wrench": (90, 235, 155)}
    labels = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    command_color = (210, 145, 250)
    tactile_labels = ("Palm", "T1", "T2", "T3", "T4", "I1", "I2", "I3", "M1", "M2", "M3",
                      "R1", "R2", "R3", "L1", "L2", "L3")
    reader = imageio.get_reader(str(scene_video))
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8,
                               ffmpeg_params=["-movflags", "+faststart"])
    try:
        for frame_index, frame in enumerate(reader):
            index = sample_indices[frame_index]
            sample, now = samples[index], times[index]
            image = Image.new("RGB", (1280, 1264 if pushing else 1088), (18, 23, 32))
            image.paste(Image.fromarray(frame), (0, 64))
            draw = ImageDraw.Draw(image)
            title = "UR5e + SI-200-8 F/T + Hand | OSC push" if pushing else "SI-200-8 F/T"
            draw.text((16, 8), f"{title} | {sample['phase'].upper()} | t = {now:.2f} s | 1x real time", font=font,
                      fill=(240, 245, 250))
            if pushing:
                friction = metadata.get("box_friction", (.5, .5))
                subtitle = (f"Box {metadata.get('box_mass_kg', .3) * 1000.:.0f} g | friction static/dynamic "
                            f"{friction[0]:.2f}/{friction[1]:.2f} | sensor XYZ = red / green / blue")
            else:
                subtitle = "No external contact | gravity ON | native physical motion | sensor XYZ = red / green / blue"
            draw.text((16, 36), subtitle,
                      font=small, fill=(185, 205, 225))
            if pushing:
                draw.text((16, 474),
                    f"Box push {sample['box_push_displacement_m'] * 1000.:+.1f} mm  |  "
                    f"XY displacement {sample['box_xy_displacement_m'] * 1000.:.1f} mm  |  "
                    f"Tilt {math.degrees(sample['box_tilt_rad']):.2f} deg  |  "
                    f"Hand-box normal resultant {sample['box_hand_normal_force_N']:.2f} N",
                    font=small, fill=(235, 240, 245))
                effort_text = ""
                if "osc_raw_joint_effort_Nm" in sample and "arm_applied_joint_effort_Nm" in sample:
                    raw_peak = max(abs(value) for value in sample["osc_raw_joint_effort_Nm"])
                    applied_peak = max(abs(value) for value in sample["arm_applied_joint_effort_Nm"])
                    effort_text = f"  |  Peak arm torque OSC / submitted {raw_peak:.2f}/{applied_peak:.2f} N m"
                draw.text((16, 501),
                    f"Hand target error max {math.degrees(sample['hand_max_target_error_rad']):.2f} deg  |  "
                    f"URDF limit breach max {math.degrees(sample['hand_max_limit_violation_rad']):.3f} deg  |  "
                    f"Arm |qdot| max {sample['arm_max_speed_rad_s']:.2f} rad/s{effort_text}",
                    font=small, fill=(185, 205, 225))
                contact = sample["tactile_binary"]
                draw.text((16, 528), f"Tactile: {sum(bool(value) for value in contact)}/17 ON | "
                          "Grouped native contact forces, production binary threshold | green ON / gray OFF",
                          font=small, fill=(185, 205, 225))
                for channel, (label, active) in enumerate(zip(tactile_labels, contact)):
                    left = 16 + channel * 73
                    draw.rounded_rectangle((left, 554, left + 68, 585), radius=5,
                                           fill=(40, 150, 100) if active else (48, 56, 70))
                    box = draw.textbbox((0, 0), label, font=small)
                    draw.text((left + (68 - box[2]) / 2., 560), label, font=small, fill=(235, 245, 240))
            legend_y, graph_y = (607, 632) if pushing else (488, 510)
            for offset, (key, label) in enumerate((("measured_wrench", "Sensor total"), ("self_load", "Predicted self-load"),
                                                  ("compensated_wrench", "Compensated observation"))):
                x = 16 + offset * (315 if pushing else 400)
                draw.line((x, legend_y, x + 32, legend_y), fill=palette[key], width=3)
                draw.text((x + 40, legend_y - 11), label, font=small, fill=palette[key])
            if pushing:
                draw.line((1030, legend_y, 1043, legend_y), fill=command_color, width=2)
                draw.line((1050, legend_y, 1062, legend_y), fill=command_color, width=2)
                draw.text((1070, legend_y - 11), "OSC command", font=small, fill=command_color)
            window_start, window_end = max(0., now - 6.), max(6., now)
            first = int(np.searchsorted(times, window_start))
            show_times = times[first:index + 1]
            for axis, label in enumerate(labels):
                column, row = axis % 3, axis // 3
                left, top, width, height = 12 + column * 424, graph_y + row * 262, 412, 250
                draw.rounded_rectangle((left, top, left + width, top + height), radius=8, fill=(26, 33, 45))
                unit, limit = ("N" if axis < 3 else "N m"), limits[axis]
                draw.text((left + 10, top + 6), f"{label} [{unit}]", font=font, fill=(235, 240, 245))
                draw.text((left + 170, top + 10), f"OSC cmd {sample['wrench_command'][axis]:+.3f}",
                          font=small, fill=(210, 210, 225))
                x0, x1, y0, y1 = left + 53, left + width - 12, top + 78, top + height - 25
                middle = (y0 + y1) / 2.
                draw.line((x0, y0, x0, y1), fill=(75, 90, 110), width=1)
                draw.line((x0, middle, x1, middle), fill=(80, 95, 115), width=1)
                draw.text((left + 3, y0 - 5), f"{limit:.2f}", font=tiny, fill=(140, 155, 175))
                draw.text((left + 2, y1 - 9), f"{-limit:.2f}", font=tiny, fill=(140, 155, 175))
                for line, (key, values) in enumerate(channels.items()):
                    value = values[index, axis]
                    draw.text((left + 10 + line * 133, top + 39), f"{value:+.3f}", font=small, fill=palette[key])
                    xs = x0 + (show_times - window_start) / (window_end - window_start) * (x1 - x0)
                    ys = middle - np.clip(values[first:index + 1, axis] / limit, -1., 1.) * (y1 - y0) / 2.
                    points = list(zip(xs.tolist(), ys.tolist()))
                    if len(points) > 1:
                        draw.line(points, fill=palette[key], width=2)
                if pushing:
                    ys = middle - np.clip(commands[first:index + 1, axis] / limit, -1., 1.) * (y1 - y0) / 2.
                    command_points = list(zip(xs.tolist(), ys.tolist()))
                    for start in range(0, len(command_points) - 1, 5):
                        draw.line(command_points[start:start + 3], fill=command_color, width=2)
                draw.text((x0, y1 + 4), f"{window_start:.1f} s", font=tiny, fill=(140, 155, 175))
                draw.text((x1 - 40, y1 + 4), f"{window_end:.1f} s", font=tiny, fill=(140, 155, 175))
            if pushing:
                draw.text((16, 1170), "F/T = calibrated joint reaction at the sensor datum; compensated = total - predicted self-load.",
                          font=small, fill=(185, 205, 225))
                draw.text((16, 1196), "Command, transmitted F/T and normal-only contact resultant are distinct measurements.",
                          font=small, fill=(185, 205, 225))
                draw.text((16, 1222), "Native OSC moves the assembled robot; no robot/object pose overrides during the push.",
                          font=tiny, fill=(155, 175, 195))
            else:
                draw.text((16, 1042), "Raw = calibrated joint reaction; compensated = raw - predicted distal gravity / inertia.",
                          font=small, fill=(185, 205, 225))
            if pushing and metadata.get("orientation_wrench_generator"):
                controller_note = ("Scripted wrench + angular PD command generator | native OSC / IdealPD | "
                                   f"inner F/T feedback OFF | physics {1 / metadata['physics_dt_s']:.0f} Hz")
            else:
                controller_note = (f"Physics {1 / metadata['physics_dt_s']:.0f} Hz | production wrench OSC / IdealPD | "
                                   "inner F/T feedback OFF")
            draw.text((16, 1243 if pushing else 1066), controller_note,
                      font=tiny, fill=(155, 175, 195))
            writer.append_data(np.asarray(image))
    finally:
        reader.close()
        writer.close()


def encode_torque_overlay(scene_video, output, samples, sample_indices, fps, metadata):
    """Render measured velocities/angles alongside the commanded sensor Tz.

    ``sample_indices`` is the acquisition-time mapping of each rendered frame
    to its exact post-physics sample. Every trial starts from its own reset;
    plots display only the current trial and never connect different resets.
    The training velocity threshold is a reference line, not an assertion
    that this independent diagnostic automatically terminates there.
    """
    import math
    import os
    from pathlib import Path
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(samples) == 0 or len(sample_indices) == 0:
        raise ValueError('Torque overlay requires acquired samples and video/sample associations.')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Video FPS must be finite and positive.')
    names = list(metadata.get('arm_joint_names', (
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')))
    if len(names) != 6:
        raise ValueError('Expected exactly six ordered arm_joint_names.')
    wrist = next((i for i, name in enumerate(names) if name == 'wrist_3_joint'), 5)
    reference = float(metadata.get('trained_velocity_threshold_rad_s', 1.0))
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError('Training velocity reference must be finite and positive.')
    qd = np.asarray([s['arm_joint_velocity_rad_s'] for s in samples], dtype=float)
    command = np.asarray([s['wrench_command'][5] for s in samples], dtype=float)
    angle = np.rad2deg(np.asarray([s['wrist3_delta_rad'] for s in samples], dtype=float))
    trial_time = np.asarray([s['trial_time_s'] for s in samples], dtype=float)
    time = np.asarray([s['time_s'] for s in samples], dtype=float)
    if qd.shape != (len(samples), 6):
        raise ValueError('Expected six measured arm velocities at every sample.')
    trial_ids = [s['trial_index'] for s in samples]
    trial_order = list(dict.fromkeys(trial_ids))
    trial_rows = {key: np.asarray([i for i, value in enumerate(trial_ids) if value == key])
                  for key in trial_order}
    trials_count = len(metadata.get('torque_trials_Nm', [])) or len(trial_order)
    index_lookup = {key: n + 1 for n, key in enumerate(trial_order)}

    def finite_max(values, fallback):
        finite = np.abs(np.asarray(values))[np.isfinite(values)]
        return max(float(finite.max()) if finite.size else 0., fallback)

    command_limit = finite_max(command, .02) * 1.1
    speed_limit = finite_max(qd[:, wrist], reference * 1.12) * 1.08
    bar_limit = max(2 * reference, min(finite_max(qd, 2 * reference), 5 * reference))
    trial_duration = {key: max(.1, float(np.nanmax(trial_time[rows]))) for key, rows in trial_rows.items()}
    angle_limits = {}
    for key, rows in trial_rows.items():
        finite = angle[rows][np.isfinite(angle[rows])]
        low = min(float(finite.min()) if finite.size else 0., 0.)
        high = max(float(finite.max()) if finite.size else 0., 0.)
        span = max(high - low, 10.)
        angle_limits[key] = (low - .12 * span, max(high + .12 * span, low + 10.))

    def load_font(size):
        for face in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 'DejaVuSans.ttf'):
            try:
                return ImageFont.truetype(face, size)
            except OSError:
                pass
        return ImageFont.load_default()

    title_font, font, small, tiny = (load_font(n) for n in (23, 19, 16, 13))
    background = (18, 23, 32)
    white, muted = (240, 245, 250), (174, 191, 211)
    violet, teal, blue = (204, 150, 255), (83, 226, 178), (117, 183, 255)
    red, yellow = (255, 91, 100), (255, 214, 128)
    canvas_size = (1280, 1088)

    def dashed_horizontal(draw, x0, x1, y, color, width=1):
        for left in range(int(x0), int(x1), 10):
            draw.line((left, y, min(left + 6, x1), y), fill=color, width=width)

    def plot(draw, box, title, xvalues, yvalues, xend, yrange, color, *, thresholds=(), current=None):
        left, top, width, height = box
        draw.rounded_rectangle((left, top, left + width, top + height), radius=8, fill=(26, 34, 46))
        draw.text((left + 12, top + 9), title, font=font, fill=white)
        if current is not None:
            draw.text((left + 12, top + 37), current, font=small, fill=color)
        x0, x1 = left + 57, left + width - 14
        y0, y1 = top + 78, top + height - 31
        low, high = yrange
        def yy(value):
            return y1 - (float(value) - low) / (high - low) * (y1 - y0)
        draw.line((x0, y0, x0, y1), fill=(100, 113, 132))
        draw.line((x0, y1, x1, y1), fill=(100, 113, 132))
        if low <= 0 <= high:
            draw.line((x0, yy(0.), x1, yy(0.)), fill=(70, 82, 99))
        for threshold in thresholds:
            if low <= threshold <= high:
                dashed_horizontal(draw, x0, x1, yy(threshold), yellow)
                draw.text((x0 + 5, yy(threshold) - 17), f'{threshold:+.1f}', font=tiny, fill=yellow)
        draw.text((left + 4, y0 - 6), f'{high:.2g}', font=tiny, fill=muted)
        draw.text((left + 4, y1 - 8), f'{low:.2g}', font=tiny, fill=muted)
        finite = np.isfinite(xvalues) & np.isfinite(yvalues)
        points = [(float(x0 + t / xend * (x1 - x0)), float(yy(v)))
                  for t, v in zip(xvalues[finite], yvalues[finite])]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        if points:
            x, y = points[-1]
            draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        draw.text((x0, y1 + 8), '0 s', font=tiny, fill=muted)
        draw.text((x1 - 40, y1 + 8), f'{xend:.1f}s', font=tiny, fill=muted)

    reader = imageio.get_reader(str(scene_video))
    writer = imageio.get_writer(str(output), fps=fps, codec='libx264', quality=8,
                               macro_block_size=None,
                               ffmpeg_params=['-movflags', '+faststart', '-threads', '2'])
    frame_count = 0
    try:
        for frame_index, frame in enumerate(reader):
            if frame_index >= len(sample_indices):
                raise ValueError('Video has more frames than recorded sample associations.')
            sample_index = int(sample_indices[frame_index])
            if not 0 <= sample_index < len(samples):
                raise IndexError(f'Frame {frame_index} has invalid sample index {sample_index}.')
            sample = samples[sample_index]
            trial = sample['trial_index']
            rows = trial_rows[trial]
            history = rows[rows <= sample_index]
            stage = str(sample['trial_stage'])
            speed = float(qd[sample_index, wrist])
            measured_over = abs(speed) > reference
            current_color = red if measured_over else teal
            canvas = Image.new('RGB', canvas_size, background)
            scene = Image.fromarray(frame[..., :3]).convert('RGB')
            if scene.size != (1280, 400):
                # Preserve camera aspect ratio if a temporary probe changes resolution.
                scene.thumbnail((1280, 400))
                canvas.paste(scene, ((1280 - scene.width)//2, 100 + (400 - scene.height)//2))
            else:
                canvas.paste(scene, (0, 100))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 8), 'UR5e + F/T + Hand | free-space sensor +Z torque step', font=title_font, fill=white)
            draw.text((16, 40), f'COMMAND Tz  {command[sample_index]:+.4f} N m', font=font, fill=violet)
            draw.text((460, 40), f'MEASURED wrist_3  {speed:+.4f} rad/s  ({math.degrees(speed):+.1f} deg/s)',
                      font=font, fill=current_color)
            draw.text((16, 70), f'Wrist_3 angle from case reset: {angle[sample_index]:+.2f} deg', font=small, fill=blue)
            draw.text((650, 70), f'Sim time {time[sample_index]:.2f}s | case time {trial_time[sample_index]:.2f}s', font=small, fill=muted)
            # These are explanatory view labels, with no physical scene geometry.
            draw.rectangle((8, 108, 190, 131), fill=(18,23,32))
            draw.text((15, 110), 'Overview', font=small, fill=white)
            draw.rectangle((648, 108, 1268, 131), fill=(18,23,32))
            draw.text((655, 110), 'Axial close-up | sensor XYZ = red / green / blue', font=small, fill=white)
            reset_note = ' | SAME INITIAL q, qdot RESET' if trial_time[sample_index] <= .45 else ''
            draw.text((16, 509), f"CASE {index_lookup[trial]}/{trials_count} | level {float(sample['trial_torque_Nm']):+.4f} N m | {stage.upper()}{reset_note}",
                      font=font, fill=yellow if reset_note else white)
            stopped = any(stop['trial_index'] == trial for stop in metadata.get('trial_stops', []))
            if stopped and sample_index == rows[-1]:
                guard = metadata.get('diagnostic_speed_guard_rad_s', float('nan'))
                draw.text((835, 509), f'TRIAL GUARD STOP > {guard:g} rad/s', font=small, fill=red)
            raw = float(sample['osc_raw_joint_effort_Nm'][wrist])
            applied = float(sample['arm_applied_joint_effort_Nm'][wrist])
            draw.text((16, 538), f'wrist_3 JOINT torque: OSC raw {raw:+.3f} / actuator submitted {applied:+.3f} N m', font=small, fill=muted)
            axial = float(sample.get('sensor_axial_angular_velocity_rad_s', float('nan')))
            draw.text((820, 538), f'sensor axial speed {axial:+.3f} rad/s', font=small, fill=muted)
            x = trial_time[history]
            duration = trial_duration[trial]
            plot(draw, (12, 568, 412, 303), 'Sensor Tz COMMAND [N m]', x, command[history], duration,
                 (-command_limit, command_limit), violet, current=f'{command[sample_index]:+.4f} N m')
            plot(draw, (434, 568, 412, 303), 'wrist_3 MEASURED [rad/s]', x, qd[history, wrist], duration,
                 (-speed_limit, speed_limit), teal, thresholds=(-reference, reference),
                 current=f'{speed:+.4f} rad/s | ref +/-{reference:g}')
            plot(draw, (856, 568, 412, 303), 'wrist_3 relative angle [deg]', x, angle[history], duration,
                 angle_limits[trial], blue, current=f'{angle[sample_index]:+.2f} deg from case start')
            for j, name in enumerate(names):
                col, row = j % 3, j // 3
                left, top = 16 + col*423, 889 + row*68
                value = float(qd[sample_index, j])
                color = red if abs(value) > reference else teal
                label = name.removesuffix('_joint')
                draw.text((left, top), f'{label:<14} {value:+.4f} rad/s', font=small, fill=color)
                bx, by, width = left, top+29, 394
                draw.rectangle((bx, by, bx+width, by+9), fill=(47,55,69))
                amount = min(abs(value)/bar_limit, 1.)
                draw.rectangle((bx, by, bx+width*amount, by+9), fill=color)
                line = bx+width*reference/bar_limit
                draw.line((line,by-3,line,by+12), fill=yellow, width=2)
            alignment=float(sample.get('axial_alignment',float('nan')))
            contact=float(sample.get('max_tactile_normal_force_N',float('nan')))
            draw.text((16, 1029), f'Axial alignment dot {alignment:+.4f} | tactile peak {contact:.3f} N | yellow: training speed reference',font=tiny,fill=muted)
            draw.text((16, 1051), 'Only case boundaries reset state. Within a case: native wrench OSC / IdealPD; no pose replacement.',font=tiny,fill=muted)
            draw.text((16, 1070), 'Tz is a command; velocities/angles are measured. Raw/submitted joint torques include gravity compensation.',font=tiny,fill=muted)
            writer.append_data(np.asarray(canvas))
            frame_count += 1
    finally:
        reader.close()
        writer.close()

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact_osc_torque_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    colors = ('#318d80', '#5e60ce', '#d17b0f', '#b04177', '#698f3f', '#e63946')
    for key in trial_order:
        rows = trial_rows[key]
        level=float(samples[int(rows[0])]['trial_torque_Nm'])
        axes[0].plot(time[rows],command[rows],color='#9359c5',lw=1.4)
        for j,name in enumerate(names):
            axes[1].plot(time[rows],qd[rows,j],color=colors[j],lw=1.4 if j==wrist else .7,
                         alpha=1 if j==wrist else .55,label=name.removesuffix('_joint') if key==trial_order[0] else None)
        axes[2].plot(time[rows],angle[rows],color='#3286bf',lw=1.2)
        for ax in axes:ax.axvline(time[rows[0]],color='gray',ls=':',lw=.8,alpha=.6)
        axes[0].annotate(f'case {index_lookup[key]}\n{level:+g} N m',(time[rows[0]],level),
                         xytext=(3,7),textcoords='offset points',fontsize=8)
    axes[0].set_ylabel('Command sensor Tz [N m]')
    axes[1].set_ylabel('Measured joint velocity [rad/s]')
    axes[1].axhline(reference,color='#c34d27',ls='--',lw=1.,label=f'training reference +/-{reference:g}')
    axes[1].axhline(-reference,color='#c34d27',ls='--',lw=1.)
    axes[1].legend(loc='upper left',ncol=4,fontsize=8)
    axes[2].set_ylabel('wrist_3 angle from case reset [deg]')
    axes[2].set_xlabel('Simulation time [s]')
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Sensor-axis torque steps: independent same-state trials; no interpolation across resets')
    fig.tight_layout()
    plot_path=output.with_name(output.stem+'_trajectory.png')
    fig.savefig(plot_path,dpi=160);plt.close(fig)
    return {'video':str(output),'trajectory_plot':str(plot_path),'video_frames':frame_count,
            'output_size':[1280,1088],'fps':fps,'trial_count':len(trial_order)}


def encode_wrench_sweep_overlay(scene_video, output, samples, sample_indices, fps, metadata):
    """Overlay physical joint velocities; preserve the supplied frame/sample map.

    Sensor wrench commands and measured joint velocities are explicitly distinct.
    Only the six UR5e joints are analyzed; the assembled hand remains physical.
    Diagnostic guard information comes from metadata.trial_stops; it is not
    inferred as a training termination. Each plot resets at a new trial.
    """
    import math
    import os
    from pathlib import Path
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(samples) == 0 or len(sample_indices) == 0:
        raise ValueError('Wrench overlay requires acquired samples and frame/sample associations.')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS must be finite and positive.')
    default_arm = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
    arm_names = list(metadata.get('arm_joint_names', default_arm))
    if len(arm_names) != 6:
        raise ValueError('Expected six named UR5e joints.')
    names = arm_names
    display_order = list(range(6))
    qd = np.asarray([s['arm_joint_velocity_rad_s'] for s in samples], dtype=float)
    arm_qd = qd
    if qd.shape != (len(samples),6):
        raise ValueError('Every sample must contain six measured UR5e joint velocities.')
    commands = np.asarray([s['wrench_command'] for s in samples], dtype=float)
    times = np.asarray([s['time_s'] for s in samples], dtype=float)
    case_time = np.asarray([s['trial_time_s'] for s in samples], dtype=float)
    axes = np.asarray([s['trial_axis'] for s in samples], dtype=int)
    axis_names = ('Fx','Fy','Fz','Tx','Ty','Tz')
    if commands.shape != (len(samples),6) or np.any((axes<0)|(axes>5)):
        raise ValueError('Expected six-axis wrench commands and trial_axis in 0..5.')
    arm_reference = float(metadata.get('trained_velocity_threshold_rad_s',1.))
    guard = float(metadata.get('diagnostic_speed_guard_rad_s',4.))
    if not (math.isfinite(arm_reference) and arm_reference>0 and math.isfinite(guard) and guard>0):
        raise ValueError('Velocity reference and guard must be finite and positive.')
    keys = [s['trial_index'] for s in samples]
    order = list(dict.fromkeys(keys))
    groups = {key:np.asarray([i for i,k in enumerate(keys) if k==key]) for key in order}
    numbers = {key:i+1 for i,key in enumerate(order)}
    ntrials = len(metadata.get('wrench_trials',[])) or len(order)
    stops = {s['trial_index']:s for s in metadata.get('trial_stops',[])}

    def maximum(values,fallback):
        values=np.abs(np.asarray(values,dtype=float));values=values[np.isfinite(values)]
        return max(float(values.max()) if values.size else 0.,fallback)

    command_limits = {axis:maximum(commands[axes==axis,axis],.01 if axis<3 else .001)*1.12
                      for axis in range(6)}
    arm_plot_limit = maximum(arm_qd,arm_reference*1.15)*1.06
    bar_limit = maximum(qd,guard)*1.06
    durations = {key:max(.1,float(case_time[rows].max())) for key,rows in groups.items()}
    sensor_positions = np.asarray([s['sensor_pose_w'][:3] for s in samples],dtype=float)
    displacement = sensor_positions.copy()
    for rows in groups.values():
        displacement[rows] = (sensor_positions[rows] - sensor_positions[int(rows[0])]) * 1000.
    displacement_limit = maximum(displacement,.5)*1.12
    arm_colors = ((80,217,177),(100,179,255),(255,184,92),(228,123,191),(180,211,111),(255,112,108))
    purple,teal,white,muted,red,yellow=(204,150,255),(83,226,178),(240,245,250),(173,191,210),(255,91,100),(255,214,128)

    def font(size):
        for filename in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','DejaVuSans.ttf'):
            try:return ImageFont.truetype(filename,size)
            except OSError:pass
        return ImageFont.load_default()

    f24,f19,f16,f14,f12=(font(n) for n in (24,19,16,14,12))
    def short(name):
        return name.removeprefix('right_').removesuffix('_joint')

    def plot(draw,box,title,xx,series,xend,limit,colors,*,reference=None,subtitle=''):
        left,top,width,height=box
        draw.rounded_rectangle((left,top,left+width,top+height),radius=8,fill=(26,34,46))
        draw.text((left+11,top+10),title,font=f19,fill=white)
        draw.text((left+11,top+39),subtitle,font=f14,fill=muted)
        x0,x1,y0,y1=left+58,left+width-13,top+81,top+height-31
        middle=(y0+y1)/2.
        draw.line((x0,y0,x0,y1),fill=(100,113,132));draw.line((x0,y1,x1,y1),fill=(100,113,132))
        draw.line((x0,middle,x1,middle),fill=(70,82,99))
        if reference is not None:
            for value in (-reference,reference):
                y=middle-value/limit*(y1-y0)/2
                if y0<=y<=y1:
                    for x in range(int(x0),int(x1),10):draw.line((x,y,min(x+6,x1),y),fill=yellow)
                    draw.text((x0+3,y-16),f'{value:+g}',font=f12,fill=yellow)
        draw.text((left+3,y0-5),f'{limit:.2g}',font=f12,fill=muted)
        draw.text((left+3,y1-8),f'{-limit:.2g}',font=f12,fill=muted)
        yy=np.asarray(series,dtype=float)
        if yy.ndim==1:yy=yy[:,None]
        for j in range(yy.shape[1]):
            good=np.isfinite(xx)&np.isfinite(yy[:,j])
            points=[(float(x0+x/xend*(x1-x0)),float(middle-np.clip(v/limit,-1.,1.)*(y1-y0)/2))
                    for x,v in zip(xx[good],yy[good,j])]
            color=colors[j%len(colors)]
            if len(points)>1:draw.line(points,fill=color,width=2)
            if len(points):
                x,y=points[-1];draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
        draw.text((x0,y1+9),'0 s',font=f12,fill=muted)
        draw.text((x1-42,y1+9),f'{xend:.2f}s',font=f12,fill=muted)

    reader=imageio.get_reader(str(scene_video))
    writer=imageio.get_writer(str(output),fps=fps,codec='libx264',quality=8,macro_block_size=None,
                              ffmpeg_params=['-movflags','+faststart','-threads','2'])
    frame_count=0
    try:
        for frame_index,frame in enumerate(reader):
            if frame_index>=len(sample_indices):raise ValueError('More video frames than sample associations.')
            index=int(sample_indices[frame_index])
            if not 0<=index<len(samples):raise IndexError(f'Invalid video sample index {index}.')
            sample=samples[index];key=sample['trial_index'];axis=int(sample['trial_axis'])
            unit='N' if axis<3 else 'N m';name=axis_names[axis]
            rows=groups[key];past=rows[rows<=index];duration=durations[key]
            now_qd=qd[index];peak_index=int(np.nanargmax(np.abs(now_qd)))
            peak=float(now_qd[peak_index]);peak_joint=short(names[peak_index])
            warning=abs(peak)>arm_reference
            image=Image.new('RGB',(1280,1088),(18,23,32));scene=Image.fromarray(frame[...,:3]).convert('RGB')
            if scene.size!=(1280,400):
                scene.thumbnail((1280,400));image.paste(scene,((1280-scene.width)//2,100+(400-scene.height)//2))
            else:image.paste(scene,(0,100))
            d=ImageDraw.Draw(image)
            d.text((16,8),'UR5e + F/T + Hand | independent six-axis wrench trials',font=f24,fill=white)
            d.text((16,43),f'COMMAND {name} {commands[index,axis]:+.4f} {unit}',font=f19,fill=purple)
            d.text((505,43),f'MAX MEASURED |qdot|: {peak_joint} {peak:+.4f} rad/s',font=f19,fill=red if warning else teal)
            d.text((16,73),f'Sim t={times[index]:.2f}s | case t={case_time[index]:.2f}s | 6 measured UR5e joints',font=f16,fill=muted)
            d.text((795,73),f'ARM train reference +/-{arm_reference:g}; ARM guard {guard:g}',font=f16,fill=yellow)
            for x,label in ((8,'Overview'),(648,'Sensor/hand close-up | XYZ = red / green / blue')):
                d.rectangle((x,108,x+(180 if x==8 else 620),132),fill=(18,23,32));d.text((x+7,110),label,font=f16,fill=white)
            stage=str(sample['trial_stage']).upper()
            label=f"CASE {numbers[key]}/{ntrials} | {name} level {float(sample['trial_level']):+.4f} {unit} | {stage}"
            d.text((16,509),label,font=f19,fill=white)
            is_guard=key in stops and index==int(rows[-1])
            if is_guard:
                cause = f'SPEED GUARD > {guard:g} rad/s' if 'speed' in stops[key]['reason'].lower() else 'DISPLACEMENT GUARD > 0.25 m'
                d.text((795,509),cause,font=f16,fill=red)
            elif case_time[index]<=.38:d.text((830,509),'RESET: same initial q / qdot',font=f16,fill=yellow)
            raw=maximum(sample['osc_raw_joint_effort_Nm'],0.)
            submitted=maximum(sample['arm_applied_joint_effort_Nm'],0.)
            contact=float(sample.get('max_tactile_normal_force_N',float('nan')))
            d.text((16,539),f'Arm joint effort |max|: OSC raw {raw:.3f} / submitted {submitted:.3f} N m',font=f16,fill=muted)
            d.text((886,539),f'Tactile peak {contact:.4f} N',font=f16,fill=muted)
            xx=case_time[past]
            plot(d,(12,568,412,303),f'{name} COMMAND [{unit}]',xx,commands[past,axis],duration,command_limits[axis],(purple,),
                 subtitle=f'Current {commands[index,axis]:+.4f} {unit}; sensor axes')
            plot(d,(434,568,412,303),'ARM 6 measured [rad/s]',xx,arm_qd[past],duration,arm_plot_limit,arm_colors,reference=arm_reference,
                 subtitle=f'Yellow +/-{arm_reference:g}: training reference')
            current_delta=displacement[index]
            plot(d,(856,568,412,303),'Sensor displacement [mm]',xx,displacement[past],duration,displacement_limit,
                 ((255,112,108),(110,221,144),(117,183,255)),
                 subtitle=f'X {current_delta[0]:+.1f} | Y {current_delta[1]:+.1f} | Z {current_delta[2]:+.1f}')
            d.text((16,886),f'Measured UR5e joints | red: >{arm_reference:g} rad/s training reference | diagnostic guard: {guard:g} rad/s',font=f16,fill=yellow)
            for position,j in enumerate(display_order):
                col,row=position%3,position//3;left,top=16+col*423,918+row*51
                value=float(now_qd[j]);threshold=arm_reference
                color=red if abs(value)>threshold else arm_colors[position]
                d.text((left,top),f'{short(names[j]):<15} {value:+.4f} rad/s',font=f16,fill=color)
                bx,by,width=left,top+29,394
                d.rectangle((bx,by,bx+width,by+9),fill=(47,55,69))
                d.rectangle((bx,by,bx+width*min(abs(value)/bar_limit,1.),by+9),fill=color)
                line=bx+width*threshold/bar_limit;d.line((line,by-3,line,by+12),fill=yellow,width=2)
            d.text((16,1033),'Sensor displacement is relative to the first measured sample of this case. XYZ curves are world-frame translations.',font=f12,fill=muted)
            d.text((16,1053),'Same initial state is restored only BETWEEN cases. Within a case: native wrench OSC, assembled physical hand, no pose replacement.',font=f12,fill=muted)
            d.text((16,1073),'Wrench COMMAND, OSC joint effort and measured joint velocity are different quantities. Hand velocities are outside this diagnostic.',font=f12,fill=muted)
            writer.append_data(np.asarray(image));frame_count+=1
    finally:
        reader.close();writer.close()

    os.environ.setdefault('MPLCONFIGDIR','/tmp/contact_osc_wrench_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(15,10),sharex=True)
    for position,j in enumerate(display_order):
        ax=axs.flat[position]
        for key,rows in groups.items():
            ax.plot(times[rows],qd[rows,j],lw=.85,color='#16826e')
        limit=maximum(qd[:,j],arm_reference*1.1)*1.12
        ax.set_ylim(-limit,limit)
        ax.axhline(arm_reference,color='#c77b26',ls='--',lw=.75)
        ax.axhline(-arm_reference,color='#c77b26',ls='--',lw=.75)
        if guard<=limit:
            ax.axhline(guard,color='#c44747',ls=':',lw=.75)
            ax.axhline(-guard,color='#c44747',ls=':',lw=.75)
        ax.set_title('UR5e '+names[j],fontsize=9)
        ax.set_ylabel('rad/s');ax.grid(alpha=.2)
        if position>=4:ax.set_xlabel('Simulation time [s]')
    fig.suptitle('Six measured UR5e joint velocities | independent Y scales | lines break at case resets\nDashed: training +/-1 rad/s; red dotted: separate diagnostic guard where visible',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));plot_path=output.with_name(output.stem+'_ur5e_joint_velocities.png')
    fig.savefig(plot_path,dpi=160);plt.close(fig)
    return {'video':str(output),'ur5e_joint_velocity_plot':str(plot_path),'video_frames':frame_count,
            'output_size':[1280,1088],'fps':fps,'trial_count':len(order),'measured_joint_count':len(names),
            'arm_joint_names':arm_names}


def run_probe(args):
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab.utils.math import compute_pose_error

    scene = importlib.import_module("contact_manipulation_symmetric.config.ur5e.scene")
    env_module = importlib.import_module("contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module("contact_manipulation_symmetric.mdp.events")
    sensors = importlib.import_module("contact_manipulation_symmetric.mdp.sensors")
    observations = importlib.import_module("contact_manipulation_symmetric.mdp.observations")
    rewards = importlib.import_module("contact_manipulation_symmetric.mdp.rewards")
    production_cfg = env_module.UR5eContactManipulationEnvCfg()
    independent_trials = args.osc_torque_staircase or args.osc_wrench_sweep
    robot_cfg = production_cfg.scene.robot.copy()
    # The official OSC Action uses this native cloned-environment namespace
    # when resolving its task FrameTransformer; keep it unchanged in this probe.
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    sim_cfg = production_cfg.sim.copy()
    sim_cfg.device, sim_cfg.render_interval = args.device, 2**31 - 1
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    robot = Articulation(robot_cfg)
    tactile_cfg = sensors.make_tactile_contact_cfg(scene.TACTILE_BODY_NAMES)
    tactile_cfg.prim_path = tactile_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    tactile = tactile_cfg.class_type(tactile_cfg)
    box = None
    box_size, floor_z = (.22, .18, .20), 1.36
    if args.osc_push:
        # A never-moved support has its final authored position before PhysX
        # starts. The dynamic box alone is placed once after native FK is known.
        support = sim_utils.CuboidCfg(size=(.50, 1.20, .04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25, .30, .36), roughness=.8))
        support.func("/World/PushSupport", support, translation=(-.60, -.20, floor_z - .02))
        box = RigidObject(RigidObjectCfg(prim_path="/World/PushBox",
            spawn=sim_utils.CuboidCfg(size=box_size, activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=.1),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
                mass_props=sim_utils.MassPropertiesCfg(mass=.3),
                physics_material=sim_utils.RigidBodyMaterialCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .52, .15), roughness=.6)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 8.))))
    # The standard ground has a collider; do not confuse it with the visual-only
    # sensor gizmo below. Diagnostic motion bounds keep the hand near its start.
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/Ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.9, .94, 1.))
    light.func("/World/ProbeLight", light)
    camera = marker = None
    if not args.no_video:
        for index in range(2):
            sim_utils.create_prim(f"/World/ProbeViews/View_{index}", "Xform")
        camera = Camera(CameraCfg(
            prim_path="/World/ProbeViews/View_.*/Camera", width=640, height=400, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=24., clipping_range=(.01, 15.))))
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/World/ProbeSensorFrame"
        marker_cfg.markers = {"frame": marker_cfg.markers["frame"]}
        marker_cfg.markers["frame"].scale = (.06, .06, .06)
        marker = VisualizationMarkers(marker_cfg)
        for prim in Usd.PrimRange(sim_utils.get_current_stage().GetPrimAtPath(marker_cfg.prim_path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                raise RuntimeError("The visual frame marker must not contain physical schemas.")
    sim.reset()
    # A bare Articulation does not execute ManagerBased reset events. Apply the
    # configured elbow-up pose once, as the official standalone OSC tutorial
    # does at reset; every subsequent motion is generated by torque dynamics.
    initial_q = robot.data.default_joint_pos.clone()
    if args.osc_push or independent_trials:
        # Default palm faces down. Roll the wrist once so the pad face is
        # horizontal, retaining the elbow-up shoulder/elbow configuration.
        initial_q[:, robot.joint_names.index("wrist_3_joint")] += math.pi / 2.
    robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
    robot.reset()
    sim.forward()
    tactile.reset()
    events.configure_hand_action_limits(SimpleNamespace(scene={"robot": robot}))
    class ProbeScene(dict):
        env_prim_paths = ["/World/envs/env_0"]

    env = SimpleNamespace(scene=ProbeScene(robot=robot, tactile_contact=tactile), num_envs=1, device=sim.device,
                          sim=sim, cfg=production_cfg)
    reader = sensors.SensorReader(env)
    arm_action = production_cfg.actions.arm_action.class_type(production_cfg.actions.arm_action.copy(), env)
    hand_action = production_cfg.actions.hand_action.class_type(production_cfg.actions.hand_action.copy(), env)
    if args.osc_push and args.push_hand_fraction:
        hand_raw = initial_q.new_full((1, 6), (2. * args.push_hand_fraction - 1.) / production_cfg.actions.hand_action.scale)
        hand_action.process_actions(hand_raw)
        initial_q[:, hand_action._driven_joint_ids] = hand_action._driven_targets
        robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
        robot.reset()
        sim.forward()
        tactile.reset()
        reader.reset(torch.tensor([0], device=sim.device))
    dt = sim.get_physics_dt()
    stride = round(1. / (dt * args.video_fps))
    if stride < 1 or not math.isclose(stride * dt * args.video_fps, 1., abs_tol=1.e-6):
        raise ValueError("Video FPS must exactly divide the physical frequency.")
    names, paths = robot.body_names, robot.root_physx_view.link_paths[0]
    sensor_id = names.index(scene.FORCE_TORQUE_BODY_NAME)
    payload_ids = torch.tensor([i for i, (name, path) in enumerate(zip(names, paths))
                                if name == scene.FORCE_TORQUE_BODY_NAME or "/ROASRightHand/" in str(path)],
                               dtype=torch.long, device=sim.device)
    if payload_ids.numel() < 2:
        raise RuntimeError("Cannot resolve the production distal load-cell / hand payload.")
    mass = robot.root_physx_view.get_masses().to(sim.device).index_select(1, payload_ids).clone()
    inertia = robot.root_physx_view.get_inertias().to(sim.device).index_select(1, payload_ids)
    inertia = inertia.reshape(1, -1, 3, 3).transpose(-1, -2).contiguous()
    local_com = robot.data.body_com_pos_b.index_select(1, payload_ids).clone()
    mount = robot_cfg.spawn
    _, expected_mass, expected_center, _, _ = scene._sensor_mount_half_properties(
        mount.sensor_mount_length, mount.sensor_mount_radius, mount.sensor_mount_mass,
        mount.sensor_body_length, mount.sensor_flange_radius, mount.sensor_flange_mass)
    expected_com = mass.new_tensor([0., 0., expected_center - mount.sensor_body_length * .5])
    native_sensor_mass = robot.root_physx_view.get_masses()[0, sensor_id]
    native_sensor_com = robot.data.body_com_pos_b[0, sensor_id]
    if (not math.isclose(float(native_sensor_mass), expected_mass, abs_tol=1.e-6)
            or not torch.allclose(native_sensor_com, expected_com, atol=1.e-7, rtol=1.e-5)):
        raise RuntimeError("The native sensor mass/COM do not match the current tool-side-reference mount.")
    gravity = mass.new_tensor(sim.cfg.gravity)
    ranges, resolution = mass.new_tensor(observations.SI_200_8_RANGES), mass.new_tensor(observations.SI_200_8_RESOLUTION)
    arm_ids, _ = robot.find_joints(list(scene.ARM_JOINT_NAMES), preserve_order=True)
    hand_ids = hand_action._driven_joint_ids
    pair_view = None
    if args.osc_push:
        pad_ids, _ = robot.find_bodies(list(scene.TACTILE_BODY_NAMES), preserve_order=True)
        palm_rotation = robot.data.body_link_quat_w[:, pad_ids[0]]
        normal = observations.quat_apply(palm_rotation, mass.new_tensor([[0., 0., 1.]]))[0]
        if abs(float(normal[2])) > .05:
            raise RuntimeError("The diagnostic palm must face horizontally before box placement.")
        push_direction = normal.clone()
        push_direction[2] = 0.
        push_direction /= push_direction.norm()
        initial_sensor_rotation = robot.data.body_link_quat_w[:, sensor_id].clone()
        lateral = torch.cross(mass.new_tensor([0., 0., 1.]), push_direction, dim=-1)
        pad_points = rewards.tactile_pad_points(env)[0]
        contact_center = pad_points[5:].mean(0)
        # Clear the real Hand collision geometry in the box's lateral span,
        # including parent housings rather than only ideal tactile surfaces.
        stage = sim_utils.get_current_stage()
        cache = UsdGeom.XformCache()
        vertices = []
        for body_id, path in enumerate(paths):
            if "/ROASRightHand/" not in str(path):
                continue
            body = stage.GetPrimAtPath(str(path))
            inverse = cache.GetLocalToWorldTransform(body).GetInverse()
            for prim in Usd.PrimRange(body):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                collider = prim
                while collider and collider != body and not collider.HasAPI(UsdPhysics.CollisionAPI):
                    collider = collider.GetParent()
                if (not collider or collider == body
                        or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False):
                    continue
                owner = prim.GetParent()
                while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                    owner = owner.GetParent()
                if owner != body:
                    continue
                matrix = cache.GetLocalToWorldTransform(prim) * inverse
                matrix_np = np.asarray(matrix)
                points_np = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                local = mass.new_tensor(points_np @ matrix_np[:3, :3] + matrix_np[3, :3])
                world = robot.data.body_link_pos_w[0, body_id] + observations.quat_apply(
                    robot.data.body_link_quat_w[:, body_id].expand(local.shape[0], -1), local)
                vertices.append(world)
        if not vertices:
            raise RuntimeError("Cannot locate the assembled Hand collision geometry.")
        vertices = torch.cat(vertices)
        hand_lowest = float(vertices[:, 2].min())
        if hand_lowest < floor_z + .005:
            raise RuntimeError(f"Initial Hand collides with the support: minimum z={hand_lowest:g}.")
        in_span = ((vertices - contact_center) @ lateral).abs() < box_size[1] * .5 + .005
        in_span &= (vertices[:, 2] >= floor_z) & (vertices[:, 2] <= floor_z + box_size[2])
        front = (vertices[in_span] @ push_direction).max()
        box_center = contact_center + push_direction * (front - contact_center @ push_direction + .012 + box_size[0] * .5)
        box_center[2] = floor_z + box_size[2] * .5 + .0005
        yaw = math.atan2(float(push_direction[1]), float(push_direction[0]))
        box_state = box.data.default_root_state.clone()
        box_state[:, :3] = box_center
        box_state[:, 3:7] = mass.new_tensor([math.cos(yaw / 2.), 0., 0., math.sin(yaw / 2.)])
        box_state[:, 7:] = 0.
        box.write_root_state_to_sim(box_state)
        box.reset()
        sim.forward()
        box_initial_position = box.data.root_pos_w[0].clone()
        box_local_corners = mass.new_tensor([[x, y, z] for x in (-1., 1.) for y in (-1., 1.)
                                           for z in (-1., 1.)]) * mass.new_tensor(box_size) * .5
        box_mass = float(box.root_physx_view.get_masses()[0, 0])
        if not math.isclose(box_mass, .3, abs_tol=1.e-6):
            raise RuntimeError(f"Native box mass is {box_mass:g} kg, expected 0.3.")
        hand_paths = [str(path) for path in paths if "/ROASRightHand/" in str(path)]
        pair_view = robot._physics_sim_view.create_rigid_contact_view(
            hand_paths, filter_patterns=[[box.cfg.prim_path]] * len(hand_paths), max_contact_data_count=0)
        if (list(pair_view.sensor_paths) != hand_paths or pair_view.sensor_count != len(hand_paths)
                or pair_view.filter_count != 1):
            raise RuntimeError(f"Hand/box witness view changed the explicit body path order: {pair_view.sensor_paths}.")
    wrench_scale = mass.new_tensor(production_cfg.actions.arm_action.wrench_scale)
    wrench_command = mass.new_zeros((1, 6))
    policy_decimation = production_cfg.decimation
    if not math.isclose(2. * args.frequency, round(2. * args.frequency), abs_tol=1.e-6):
        raise ValueError("Two-second bursts must contain an integer number of base-frequency periods.")
    stage = sim_utils.get_current_stage()
    physics_properties = {}
    for name, path in zip(names, paths):
        prim = stage.GetPrimAtPath(str(path))
        api = PhysxSchema.PhysxRigidBodyAPI(prim)
        properties = {}
        for key, getter in (("linear_damping", "GetLinearDampingAttr"),
                            ("angular_damping", "GetAngularDampingAttr"),
                            ("enable_gyroscopic_forces", "GetEnableGyroscopicForcesAttr")):
            attribute = getattr(api, getter)()
            properties[key] = {"value": attribute.Get(), "authored": attribute.HasAuthoredValueOpinion()}
        physics_properties[str(path)] = {"body_name": name, **properties}
    metadata = {
        "status": "running", "experiment": "osc_push" if args.osc_push else "ft_shake",
        "package": "contact_manipulation_symmetric", "physics_dt_s": dt, "gravity_m_s2": list(sim.cfg.gravity),
        "external_contact": "No objects; grouped tactile forces and hand self-contact still monitored.",
        "motion": "Production SensorFrameWrenchAction + official OSC + IdealPD; no joint/root overwrites.",
        "arm_action": {"type": type(arm_action).__name__, "wrench_scale": wrench_scale.cpu().tolist(),
                       "inner_wrench_feedback": production_cfg.actions.arm_action.controller_cfg.contact_wrench_stiffness_task,
                       "gravity_compensation": production_cfg.actions.arm_action.controller_cfg.gravity_compensation},
        "policy_decimation": policy_decimation, "policy_dt_s": policy_decimation * dt,
        "initial_joint_positions_rad": robot.data.joint_pos[0].cpu().tolist(),
        "configured_default_joint_positions_rad": robot.data.default_joint_pos[0].cpu().tolist(),
        "wrench_waveform": {"force_amplitude_N": args.force_amplitude, "torque_amplitude_Nm": args.torque_amplitude,
                            "frequency_Hz": args.frequency, "formula": "A*(cos(2*pi*f*t)-cos(4*pi*f*t))"},
        "hand_drive": {"stiffness": robot_cfg.actuators["hand"].stiffness,
                       "damping": robot_cfg.actuators["hand"].damping, "armature": robot_cfg.actuators["hand"].armature},
        "sensor_mount": {key: getattr(robot_cfg.spawn, key) for key in
                         ("sensor_mount_length", "sensor_mount_radius", "sensor_mount_mass", "sensor_body_length",
                          "sensor_flange_radius", "sensor_flange_mass")},
        "sensor_ranges": ranges.cpu().tolist(), "sensor_resolution": resolution.cpu().tolist(),
        "sensor_body": names[sensor_id],
        "sensor_reference": "Manufacturer tool-side face origin; equivalent fixed-joint and actor frames coincide.",
        "sensor_reference_offset_from_wrist_m": (mount.sensor_mount_length + mount.sensor_body_length) * .5,
        "native_sensor_mass_kg": float(native_sensor_mass), "native_sensor_com_local_m": native_sensor_com.cpu().tolist(),
        "payload_names": [names[i] for i in payload_ids.cpu().tolist()],
        "payload_mass_kg": mass[0].cpu().tolist(), "payload_total_mass_kg": float(mass.sum()),
        "payload_inertia_kg_m2": inertia[0].cpu().tolist(), "payload_com_local_m": local_com[0].cpu().tolist(),
        "native_body_physics_properties": physics_properties,
        "frame_gizmo": "Visual-only native marker at the sensor's actor origin, XYZ red/green/blue.",
        "camera_views": {"overview": "Root/sensor midpoint, world offset (1.5,-1.8,0.8) m.",
                         "detail": "Sensor world offset (-0.45,-0.50,0.20) m; aim 0.06 m along sensor +Z."},
        "spec_model": "SI-200-8 range clipping and deterministic quantization; no noise, bias, filtering or delay model.",
        "phases": [{"name": phase, "duration_s": duration} for phase, duration in PHASES],
    }
    phases = PHASES
    torque_trials = []
    wrench_trials = []
    if independent_trials:
        for axis in (range(6) if args.osc_wrench_sweep else [5]):
            label = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")[axis]
            if args.osc_wrench_sweep and label not in args.wrench_axes:
                continue
            magnitudes = sorted(set(args.force_levels if axis < 3 else args.torque_levels) - {0.})
            levels = [0., *magnitudes, *(-value for value in magnitudes)]
            wrench_trials.extend({"axis": axis, "level": value} for value in levels)
        torque_trials = [trial["level"] for trial in wrench_trials]
        phases = tuple((f"trial_{index:02d}", .4 + args.torque_hold + .3)
                       for index in range(len(torque_trials)))
        metadata.update({
            "experiment": "osc_wrench_sweep" if args.osc_wrench_sweep else "osc_torque_staircase",
            "torque_trials_Nm": torque_trials if args.osc_torque_staircase else [],
            "wrench_trials": wrench_trials,
            "arm_joint_names": list(scene.ARM_JOINT_NAMES),
            "arm_joint_velocity_limits_rad_s": robot.data.joint_vel_limits[0, arm_ids].cpu().tolist(),
            "trained_velocity_threshold_rad_s": 1., "diagnostic_speed_guard_rad_s": args.max_arm_speed,
            "trial_timing": {"baseline_s": .4, "torque_hold_s": args.torque_hold, "release_s": .3},
            "state_writes": "Same initial q and zero qdot only between independent torque levels; none during a trial.",
            "command": "One sensor wrench axis at a time; all other axes zero; no orientation PD or joint locking.",
            "initial_wrist_3_roll_added_rad": math.pi / 2.,
            "training_termination": "Not active in this standalone probe; 1 rad/s is a visual reference only.",
            "phases": [{"name": phase, "duration_s": duration} for phase, duration in phases],
            "trial_stops": [],
        })
        if args.osc_wrench_sweep:
            metadata["sensor_displacement_guard_m"] = .25
            metadata["external_contact"] = "No scene objects; physical ground and robot self-collision remain enabled."
    if args.osc_push:
        phases = (("baseline", 1.2), ("approach", 1.6), ("push", 1.6), ("brake", .8), ("settle", 2.4))
        metadata.update({"external_contact": "Dynamic 300 g box on a fixed horizontal support.",
            "box_mass_kg": box_mass, "box_friction": [.5, .5], "box_size_m": box_size,
            "support_top_z_m": floor_z, "initial_hand_lowest_z_m": hand_lowest,
            "box_initial_position_w_m": box_initial_position.cpu().tolist(),
            "push_direction_w": push_direction.cpu().tolist(), "push_force_N": args.push_force,
            "push_hand_fraction": args.push_hand_fraction,
            "hand_box_contact_body_paths": hand_paths,
            "initial_wrist_3_roll_added_rad": math.pi / 2.,
            "state_writes": "Initial robot and box setup only, before the first physics step. No physical guide or external forces.",
            "script": "Horizontal force schedule and contact-only r cross F; diagnostic angular PD generates torque commands to native pure-wrench OSC.",
            "orientation_wrench_generator": {"kp_Nm_rad": args.push_orientation_kp,
                "kd_Nms_rad": args.push_orientation_kd, "maximum_correction_norm_Nm": 1.5,
                "target_quaternion_wxyz": initial_sensor_rotation[0].cpu().tolist(),
                "scope": "Diagnostic command generator only; production OSC and F/T feedback settings unchanged."},
            "braking": {"trigger_box_travel_m": .08, "force_N": -1., "duration_s": .6},
            "collision_vertices_used": int(vertices.shape[0]), "front_plane_vertices": int(in_span.sum()),
            "camera_views": {"overview": "Root/contact midpoint plus (1.1,-1.4,.85) m.",
                "detail": "Hand/box midpoint plus .65*push_direction + .55*lateral + world (0,0,.32) m."},
            "phases": [{"name": phase, "duration_s": duration} for phase, duration in phases]})
    sample_indices, samples = [], []
    temporary = tempfile.TemporaryDirectory(prefix="contact_ft_video_")
    scene_path = Path(temporary.name) / "scene.mp4"
    writer = None

    def set_camera_views(sensor_position, sensor_rotation):
        center = sensor_position[0]
        midpoint = (robot.data.root_pos_w[0] + center) * .5
        detail_aim = center + observations.quat_apply(sensor_rotation, center.new_tensor([[0., 0., .06]]))[0]
        if independent_trials:
            # Static world camera: do not rotate the viewer along with the hand.
            reference_q = initial_sensor_camera_rotation
            axial = observations.quat_apply(reference_q, center.new_tensor([[0., 0., 1.]]))[0]
            lateral = observations.quat_apply(reference_q, center.new_tensor([[1., 0., 0.]]))[0]
            vertical = observations.quat_apply(reference_q, center.new_tensor([[0., 1., 0.]]))[0]
            reference_aim = initial_sensor_camera_position[0] + axial * .08
            detail_aim = (center if args.osc_wrench_sweep else initial_sensor_camera_position[0]) + axial * .08
            midpoint = (robot.data.root_pos_w[0] + reference_aim) * .5
            eyes = torch.stack((midpoint + center.new_tensor([1.4, -1.6, .7]),
                                detail_aim + .70 * axial + .12 * lateral + .10 * vertical))
            camera.set_world_poses_from_view(eyes, torch.stack((midpoint, detail_aim)))
            return
        if args.osc_push:
            detail_aim = (rewards.tactile_pad_points(env)[0, 5:].mean(0) + box.data.root_pos_w[0]) * .5
            midpoint = (robot.data.root_pos_w[0] + detail_aim) * .5
            eyes = torch.stack((midpoint + center.new_tensor([1.1, -1.4, .85]),
                                detail_aim + .65 * push_direction + .55 * lateral + center.new_tensor([0., 0., .32])))
            camera.set_world_poses_from_view(eyes, torch.stack((midpoint, detail_aim)))
            return
        eyes = torch.stack((midpoint + center.new_tensor([1.5, -1.8, .8]),
                            center + center.new_tensor([-.45, -.50, .20])))
        camera.set_world_poses_from_view(eyes, torch.stack((midpoint, detail_aim)))

    initial_sensor_camera_position = robot.data.body_link_pos_w[:, sensor_id].clone()
    initial_sensor_camera_rotation = robot.data.body_link_quat_w[:, sensor_id].clone()
    if camera is not None:
        sensor_position = robot.data.body_link_pos_w[:, sensor_id]
        sensor_rotation = robot.data.body_link_quat_w[:, sensor_id]
        set_camera_views(sensor_position, sensor_rotation)
        marker.visualize(translations=sensor_position, orientations=sensor_rotation)
        for _ in range(8):
            sim.render()
        writer = imageio.get_writer(str(scene_path), fps=args.video_fps, codec="libx264", quality=8)
    try:
        stop_reason = None
        push_braking_step = None
        for trial_index, (phase, duration) in enumerate(phases):
            if independent_trials:
                robot.write_joint_state_to_sim(initial_q, robot.data.default_joint_vel)
                robot.reset()
                sim.forward()
                tactile.reset()
                reader.reset(torch.tensor([0], device=sim.device))
                arm_action.reset()
                hand_action.reset()
                wrench_command.zero_()
                arm_action.process_actions(wrench_command)
                hand_action.process_actions(mass.new_full((1, 6), -1. / production_cfg.actions.hand_action.scale))
                wrist3_initial = float(initial_q[0, arm_ids[-1]])
                trial_sensor_origin = robot.data.body_link_pos_w[0, sensor_id].clone()
            print(f"[F/T probe] {phase}: {duration:g} simulated seconds", flush=True)
            count = round(duration / dt)
            for step in range(count):
                if (step if independent_trials else len(samples)) % policy_decimation == 0:
                    if independent_trials:
                        command, trial_stage = torque_trial_command(step * dt, torque_trials[trial_index], args.torque_hold)
                        if args.osc_wrench_sweep:
                            value = command[5]
                            command = [0.] * 6
                            command[wrench_trials[trial_index]["axis"]] = value
                            trial_stage = "command" if trial_stage == "torque" else trial_stage
                        wrench_command[:] = wrench_command.new_tensor(command)
                        fraction = 0.
                    elif args.osc_push:
                        progress = smooth_step((step + 1) * dt / min(.4, duration))
                        travel = float((box.data.root_pos_w[0] - box_initial_position) @ push_direction)
                        if push_braking_step is None and travel >= .08:
                            push_braking_step = len(samples)
                        magnitude = (.15 * progress if phase == "approach" else
                                     .15 + (args.push_force - .15) * progress if phase == "push" else
                                     -1. * math.sin(math.pi * (step + 1) * dt / duration) if phase == "brake" else 0.)
                        if push_braking_step is not None:
                            brake_time = (len(samples) - push_braking_step) * dt
                            magnitude = -1. * math.sin(math.pi * brake_time / .6) if brake_time < .6 else 0.
                        force_w = push_direction * magnitude
                        contact = rewards.tactile_pad_points(env)[0, 5:].mean(0)
                        sensor_pos = robot.data.body_link_pos_w[0, sensor_id]
                        in_contact = bool(samples and samples[-1].get("box_hand_normal_force_N", 0.) > .01)
                        torque_w = torch.cross(contact - sensor_pos, force_w, dim=-1) if in_contact else torch.zeros_like(force_w)
                        _, angle_error = compute_pose_error(sensor_pos[None], robot.data.body_link_quat_w[:, sensor_id],
                            sensor_pos[None], initial_sensor_rotation)
                        correction = (args.push_orientation_kp * angle_error[0]
                            - args.push_orientation_kd * robot.data.body_com_ang_vel_w[0, sensor_id])
                        correction *= (1.5 / correction.norm().clamp_min(1.5))
                        torque_w += correction
                        rotation_inverse = observations.quat_conjugate(robot.data.body_link_quat_w[:, sensor_id])
                        wrench_command[:, :3] = observations.quat_apply(rotation_inverse, force_w[None])
                        wrench_command[:, 3:] = observations.quat_apply(rotation_inverse, torque_w[None])
                        fraction = args.push_hand_fraction
                    else:
                        command, fraction = motion_command(phase, step * dt, duration, args.force_amplitude,
                                                           args.torque_amplitude, args.frequency)
                        wrench_command[:] = wrench_command.new_tensor(command)
                    arm_action.process_actions(wrench_command / wrench_scale)
                    hand_raw = mass.new_full((1, 6), (2. * fraction - 1.) / production_cfg.actions.hand_action.scale)
                    hand_action.process_actions(hand_raw)
                arm_action.apply_actions()
                hand_action.apply_actions()
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(dt)
                if box is not None:
                    box.update(dt)
                tactile.update(dt)
                sensor = reader()
                raw = sensor.ideal_wrench
                measured, overrange = observations.calibrate_wrench(raw, ranges, resolution)
                positions, rotations = robot.data.body_link_pos_w, robot.data.body_link_quat_w
                payload_rotation = rotations.index_select(1, payload_ids)
                com = positions.index_select(1, payload_ids) + observations.quat_apply(payload_rotation, local_com)
                acceleration = robot.data.body_com_acc_w.index_select(1, payload_ids)
                omega = robot.data.body_com_ang_vel_w.index_select(1, payload_ids)
                sensor_position, sensor_rotation = positions[:, sensor_id], rotations[:, sensor_id]
                self_load = observations.self_motion_wrench(mass, inertia, com, payload_rotation, acceleration,
                                                            omega, sensor_position, sensor_rotation, gravity)
                gravity_load = observations.self_motion_wrench(mass, inertia, com, payload_rotation,
                                                               torch.zeros_like(acceleration), torch.zeros_like(omega),
                                                               sensor_position, sensor_rotation, gravity)
                compensated = measured - self_load
                arm_max_speed = float(robot.data.joint_vel[:, arm_ids].abs().max())
                finite = bool(sensor.wrench_valid.all() & torch.isfinite(self_load).all()
                              & torch.isfinite(robot.data.joint_pos).all() & torch.isfinite(robot.data.joint_vel).all())
                if not finite:
                    raise RuntimeError(f"Invalid F/T sample at physics step {len(samples)}.")
                samples.append({
                    "time_s": (len(samples) + 1) * dt, "phase": phase, "phase_time_s": (step + 1) * dt,
                    "raw_wrench": raw[0].cpu().tolist(), "measured_wrench": measured[0].cpu().tolist(),
                    "self_load": self_load[0].cpu().tolist(), "gravity_load": gravity_load[0].cpu().tolist(),
                    "inertial_load": (self_load - gravity_load)[0].cpu().tolist(),
                    "compensated_wrench": compensated[0].cpu().tolist(), "overrange": overrange[0].cpu().tolist(),
                    "ideal_compensated_wrench": (raw - self_load)[0].cpu().tolist(),
                    "sensor_pose_w": torch.cat((sensor_position, sensor_rotation), -1)[0].cpu().tolist(),
                    "joint_position_rad": robot.data.joint_pos[0].cpu().tolist(),
                    "joint_velocity_rad_s": robot.data.joint_vel[0].cpu().tolist(),
                    "wrench_command": wrench_command[0].cpu().tolist(),
                    "osc_raw_joint_effort_Nm": arm_action._joint_efforts[0].cpu().tolist(),
                    "arm_applied_joint_effort_Nm": robot.data.applied_torque[0, arm_ids].cpu().tolist(),
                    "arm_max_speed_rad_s": arm_max_speed,
                    "max_tactile_normal_force_N": float(torch.linalg.vector_norm(sensor.tactile_force, dim=-1).max()),
                })
                if independent_trials:
                    angular_axis = observations.quat_apply(observations.quat_conjugate(arm_action._ee_pose_b[:, 3:7]),
                                                          arm_action._jacobian_b[:, 3:6, -1])
                    alignment = float(angular_axis[0, 2])
                    if abs(abs(alignment) - 1.) > 1.e-4 or float(angular_axis[0, :2].norm()) > 1.e-4:
                        raise RuntimeError(f"Sensor +Z is not the wrist3 angular Jacobian axis: {angular_axis.tolist()}")
                    metadata.setdefault("wrist3_axis_in_sensor", angular_axis[0].cpu().tolist())
                    angular_velocity = observations.quat_apply(observations.quat_conjugate(sensor_rotation),
                                                             robot.data.body_com_ang_vel_w[:, sensor_id])
                    samples[-1].update({"trial_index": trial_index,
                        **({"trial_torque_Nm": torque_trials[trial_index]} if args.osc_torque_staircase else {}),
                        "trial_axis": wrench_trials[trial_index]["axis"], "trial_level": torque_trials[trial_index],
                        "trial_time_s": (step + 1) * dt, "trial_stage": trial_stage,
                        "arm_joint_position_rad": robot.data.joint_pos[0, arm_ids].cpu().tolist(),
                        "arm_joint_velocity_rad_s": robot.data.joint_vel[0, arm_ids].cpu().tolist(),
                        "wrist3_delta_rad": float(robot.data.joint_pos[0, arm_ids[-1]]) - wrist3_initial,
                        "sensor_axial_angular_velocity_rad_s": float(angular_velocity[0, 2]),
                        "axial_alignment": alignment})
                    if args.osc_wrench_sweep:
                        displacement = sensor_position[0] - trial_sensor_origin
                        samples[-1]["sensor_displacement_w_m"] = displacement.cpu().tolist()
                        samples[-1]["minimum_body_origin_height_m"] = float(positions[0, :, 2].min())
                        if float(displacement.norm()) > metadata["sensor_displacement_guard_m"]:
                            stop_reason = "Sensor displacement exceeded the 0.25 m diagnostic region."
                if args.osc_push:
                    binary, valid, _, _ = sensors.sensor_monitor_snapshot(sensor.tactile_force, sensor.tactile_valid,
                        raw, sensor.wrench_valid, production_cfg.mdp.tactile_threshold)
                    pair = pair_view.get_contact_force_matrix(dt=dt).reshape(-1, 3)
                    offset = box.data.root_pos_w[0] - box_initial_position
                    box_up = observations.quat_apply(box.data.root_quat_w, mass.new_tensor([[0., 0., 1.]]))[0]
                    corners = box.data.root_pos_w + observations.quat_apply(
                        box.data.root_quat_w.expand(8, -1), box_local_corners)
                    support_xy = mass.new_tensor([-.60, -.20])
                    support_half = mass.new_tensor([.25, .60])
                    outside = ((corners[:, :2] - support_xy).abs() - support_half).clamp_min(0.)
                    hand_q = robot.data.joint_pos[:, hand_ids]
                    bounds = robot.data.joint_pos_limits[:, hand_ids]
                    violation = torch.maximum(bounds[..., 0] - hand_q, hand_q - bounds[..., 1]).clamp_min(0.)
                    samples[-1].update({"box_position_w_m": box.data.root_pos_w[0].cpu().tolist(),
                        "box_quaternion_wxyz": box.data.root_quat_w[0].cpu().tolist(),
                        "box_linear_velocity_w_m_s": box.data.root_lin_vel_w[0].cpu().tolist(),
                        "box_lowest_corner_z_m": float(corners[:, 2].min()),
                        "box_support_xy_exit_m": float(outside.max()),
                        "box_xy_displacement_m": float(offset[:2].norm()),
                        "box_push_displacement_m": float(offset @ push_direction),
                        "box_tilt_rad": float(torch.acos(box_up[2].clamp(-1., 1.))),
                        "box_hand_normal_force_N": float(pair.sum(0).norm()),
                        "hand_box_body_normal_force_N": pair.norm(dim=-1).cpu().tolist(),
                        "tactile_normal_force_N": sensor.tactile_force[0].norm(dim=-1).cpu().tolist(),
                        "tactile_binary": binary[0].cpu().tolist(), "tactile_valid": valid[0].cpu().tolist(),
                        "hand_max_target_error_rad": float((hand_q - hand_action._driven_targets).abs().max()),
                        "hand_max_limit_violation_rad": float(violation.max())})
                    if float(offset[:2].norm()) > .35 or abs(float(offset[2])) > .08:
                        stop_reason = "Box left the bounded diagnostic pushing region."
                if writer is not None and (len(samples) % stride == 0 or
                        (independent_trials and (arm_max_speed > args.max_arm_speed or stop_reason is not None))):
                    marker.visualize(translations=sensor_position, orientations=sensor_rotation)
                    # Track close-up translation; world-fixed eye-to-target direction
                    # avoids concealing rotation by rotating the camera with the hand.
                    set_camera_views(sensor_position, sensor_rotation)
                    sim.render()
                    camera.update(stride * dt, force_recompute=True)
                    rgb = camera.data.output["rgb"][..., :3].cpu().numpy()
                    writer.append_data(np.concatenate((rgb[0], rgb[1]), axis=1))
                    sample_indices.append(len(samples) - 1)
                if arm_max_speed > args.max_arm_speed:
                    stop_reason = f"Arm speed {arm_max_speed:.3f} rad/s exceeded diagnostic threshold {args.max_arm_speed:g}."
                if args.max_steps is not None and len(samples) >= args.max_steps:
                    stop_reason = "Requested maximum physics steps reached (smoke run)."
                if stop_reason is not None:
                    break
            if stop_reason is not None:
                if independent_trials and (args.max_steps is None or len(samples) < args.max_steps):
                    metadata["trial_stops"].append({"trial_index": trial_index,
                        **wrench_trials[trial_index],
                        **({"torque_Nm": torque_trials[trial_index]} if args.osc_torque_staircase else {}),
                        "trial_time_s": (step + 1) * dt, "reason": stop_reason})
                    stop_reason = None
                else:
                    break
        metadata["joint_names"] = list(robot.joint_names)
        metadata["status"] = "complete" if stop_reason is None else "stopped"
        metadata["stop_reason"] = stop_reason
    finally:
        if writer is not None:
            writer.close()
        if metadata["status"] == "running":
            metadata["status"] = "failed"
            metadata["stop_reason"] = "Exception during acquisition; see the console traceback."
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {"metadata": metadata, "phase_statistics": _phase_statistics(samples, phases), "samples": samples}
        if args.osc_torque_staircase:
            report["video_sample_indices"] = sample_indices
            summaries = []
            for index, level in enumerate(torque_trials):
                selected = [s for s in samples if s["trial_index"] == index]
                if not selected:
                    continue
                active = [s for s in selected if s["trial_stage"] == "torque"]
                crossing = next((s["trial_time_s"] for s in selected if abs(s["arm_joint_velocity_rad_s"][-1]) > 1.), None)
                summaries.append({"trial_index": index, "torque_Nm": level, "samples": len(selected),
                    "first_wrist3_over_1_rad_s_at_trial_time_s": crossing,
                    "wrist3_speed_max_rad_s": max(abs(s["arm_joint_velocity_rad_s"][-1]) for s in selected),
                    "applied_torque_duration_s": len(active) * dt,
                    "completed_full_hold": len(active) >= round(args.torque_hold / dt),
                    "wrist3_speed_at_last_active_sample_rad_s": active[-1]["arm_joint_velocity_rad_s"][-1] if active else None,
                    "wrist3_angle_at_last_active_sample_deg": math.degrees(active[-1]["wrist3_delta_rad"]) if active else None,
                    "first_wrist3_over_1_rad_s_after_torque_onset_s": None if crossing is None else crossing - .4,
                    "max_other_arm_speed_rad_s": max(max(abs(v) for v in s["arm_joint_velocity_rad_s"][:-1]) for s in selected),
                    "max_tactile_force_N": max(s["max_tactile_normal_force_N"] for s in selected),
                    "peak_wrist3_osc_effort_Nm": max(abs(s["osc_raw_joint_effort_Nm"][-1]) for s in selected),
                    "peak_wrist3_applied_effort_Nm": max(abs(s["arm_applied_joint_effort_Nm"][-1]) for s in selected)})
            report["torque_summary"] = summaries
        if args.osc_wrench_sweep:
            report["video_sample_indices"] = sample_indices
            report["wrench_summary"] = summarize_wrench_trials(samples, metadata, dt)
            write_wrench_velocity_csv(args.output.parent, samples, metadata, report["wrench_summary"])
        if args.osc_push and samples:
            report["push_summary"] = {
                "max_box_push_displacement_m": max(s["box_push_displacement_m"] for s in samples),
                "final_box_push_displacement_m": samples[-1]["box_push_displacement_m"],
                "max_box_tilt_rad": max(s["box_tilt_rad"] for s in samples),
                "max_hand_box_normal_force_N": max(s["box_hand_normal_force_N"] for s in samples),
                "tactile_contact_samples": sum(any(s["tactile_binary"]) for s in samples),
                "max_hand_limit_violation_rad": max(s["hand_max_limit_violation_rad"] for s in samples)}
            report["push_summary"].update({
                "min_box_lowest_corner_z_m": min(s["box_lowest_corner_z_m"] for s in samples),
                "max_box_support_xy_exit_m": max(s["box_support_xy_exit_m"] for s in samples),
                "max_contact_force_by_hand_body_N": {
                    path.rsplit("/", 1)[-1]: max(s["hand_box_body_normal_force_N"][i] for s in samples)
                    for i, path in enumerate(hand_paths)}})
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if camera is not None:
        print(f"[F/T probe] Encoding six-axis overlay: {args.video_output}", flush=True)
        if independent_trials:
            import shutil
            shutil.copyfile(scene_path, args.output.parent / "scene.mp4")
        (encode_wrench_sweep_overlay if args.osc_wrench_sweep else
         encode_torque_overlay if args.osc_torque_staircase else encode_overlay)(
            scene_path, args.video_output, samples, sample_indices, args.video_fps, metadata)
    temporary.cleanup()
    print(f"[F/T probe] Report: {args.output}", flush=True)


def encode_tactile_overlay(scene_video, output, samples, sample_indices, fps, metadata):
    """Render the measured 17 observation bits, not a geometric contact guess."""
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    tiny = ImageFont.truetype("DejaVuSans.ttf", 13)
    green, gray, amber, cyan = (75, 220, 135), (70, 80, 95), (245, 180, 65), (90, 200, 245)
    groups = (("Palm", (0,)), ("Thumb", range(1, 5)), ("Index", range(5, 8)),
              ("Middle", range(8, 11)), ("Ring", range(11, 14)), ("Little", range(14, 17)))
    reader = imageio.get_reader(str(scene_video))
    writer = imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8,
                               ffmpeg_params=["-movflags", "+faststart"])
    best_contact_frames = {}
    try:
        for frame_index, frame in enumerate(reader):
            index = sample_indices[frame_index]
            sample = samples[index]
            channel, now = sample["channel"], sample["case_time_s"]
            image = Image.new("RGB", (1280, 1040), (18, 23, 32))
            image.paste(Image.fromarray(frame), (0, 64))
            draw = ImageDraw.Draw(image)
            pad_name = metadata["pad_names"][channel].removeprefix("right_")
            draw.text((16, 7), f"17-pad tactile scan | channel {channel:02d}: {pad_name} | {sample['phase'].upper()}",
                      font=font, fill=(235, 245, 250))
            draw.text((16, 36), "Green = measured ON | gray = OFF | amber = invalid | cyan outline = pad under test | 1x real time",
                      font=small, fill=(185, 205, 225))
            draw.text((16, 468), "Production grouped ContactSensor -> SensorReader -> 17 binary observation channels", font=font,
                      fill=(235, 240, 245))
            for group_index, (title, channels) in enumerate(groups):
                left = 16 + group_index * 210
                draw.text((left + 5, 503), title, font=font, fill=(220, 235, 245))
                for row, pad in enumerate(channels):
                    top = 536 + row * 42
                    valid, on = sample["valid"][pad], sample["binary"][pad]
                    color = amber if not valid else green if on else gray
                    border = cyan if pad == channel else (40, 51, 66)
                    draw.rounded_rectangle((left, top, left + 194, top + 36), radius=5,
                                           fill=color, outline=border, width=4 if pad == channel else 1)
                    value = "invalid" if not valid else "ON" if on else "OFF"
                    draw.text((left + 8, top + 8), f"{pad:02d}  {value}", font=small, fill=(15, 25, 32) if on else (235, 240, 245))
                    draw.text((left + 98, top + 9), f"{sample['normal_force_N'][pad]:.3f} N", font=tiny,
                              fill=(15, 25, 32) if on else (235, 240, 245))
            draw.text((16, 721), f"Selected pad: {sample['normal_force_N'][channel]:.4f} N | "
                      f"probe-only witness: {sample['probe_pair_force_N'][channel]:.4f} N | "
                      f"{sum(sample['binary'])}/17 ON", font=font, fill=green if sample["binary"][channel] else (215, 225, 235))
            selected = [s for s in samples[:index + 1] if s["channel"] == channel]
            force = np.asarray([s["normal_force_N"][channel] for s in selected])
            times = np.asarray([s["case_time_s"] for s in selected])
            # This one-case scale includes the physical approach impulse rather
            # than silently removing it; the numerical threshold is always drawn.
            all_case = [s for s in samples if s["channel"] == channel]
            limit = max(.1, 1.1 * max(s["normal_force_N"][channel] for s in all_case))
            x0, x1, y0, y1 = 80, 1245, 785, 950
            draw.line((x0, y0, x0, y1, x1, y1), fill=(90, 110, 135), width=1)
            threshold_y = y1 - metadata["threshold_N"] / limit * (y1 - y0)
            draw.line((x0, threshold_y, x1, threshold_y), fill=amber, width=2)
            draw.text((x0 + 8, threshold_y - 20), f"Observation threshold: {metadata['threshold_N']:.2f} N", font=tiny, fill=amber)
            draw.text((12, y0 - 8), f"{limit:.2f} N", font=tiny, fill=(180, 200, 220))
            draw.text((12, y1 - 8), "0 N", font=tiny, fill=(180, 200, 220))
            points = list(zip((x0 + times / metadata["case_duration_s"] * (x1 - x0)).tolist(),
                              (y1 - np.clip(force / limit, 0., 1.) * (y1 - y0)).tolist()))
            if len(points) > 1:
                draw.line(points, fill=green, width=3)
            draw.text((x0, y1 + 5), "0 s", font=tiny, fill=(180, 200, 220))
            draw.text((x1 - 40, y1 + 5), f"{metadata['case_duration_s']:.1f} s", font=tiny, fill=(180, 200, 220))
            draw.text((16, 989), f"Dynamic {metadata['fixture']['radius_m'] * 2000:g} mm sphere; native prismatic position drive | "
                      "real normal-contact resultant, not pressure or friction.",
                      font=small, fill=(185, 205, 225))
            draw.text((16, 1014), "Robot/probe reset only between pad cases. Production OSC/hand drives stay active. XYZ frame gizmo is visual only.",
                      font=tiny, fill=(155, 175, 195))
            writer.append_data(np.asarray(image))
            if (sample["binary"][channel]
                    and sample["normal_force_N"][channel] > best_contact_frames.get(channel, -1.)):
                best_contact_frames[channel] = sample["normal_force_N"][channel]
                image.save(output.parent / f"pad_{channel:02d}.png")
                if channel == metadata["tested_channels"][0]:
                    image.save(output.parent / "preview.png")
    finally:
        reader.close()
        writer.close()


def run_tactile_scan(args):
    """Reuse the production sensor/actions; only the external fixture is new."""
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from pxr import Gf, Sdf, Usd, UsdPhysics
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.sensors import Camera, CameraCfg

    scene = importlib.import_module("contact_manipulation_symmetric.config.ur5e.scene")
    env_module = importlib.import_module("contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg")
    events = importlib.import_module("contact_manipulation_symmetric.mdp.events")
    sensors = importlib.import_module("contact_manipulation_symmetric.mdp.sensors")
    observations = importlib.import_module("contact_manipulation_symmetric.mdp.observations")
    rewards = importlib.import_module("contact_manipulation_symmetric.mdp.rewards")
    cfg = env_module.UR5eContactManipulationEnvCfg()
    robot_cfg = cfg.scene.robot.copy()
    robot_cfg.prim_path = "/World/envs/env_0/Robot"
    sim_cfg = cfg.sim.copy()
    sim_cfg.device, sim_cfg.render_interval = args.device, 2**31 - 1
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.create_prim("/World/envs/env_0", "Xform")
    robot = Articulation(robot_cfg)
    tactile_cfg = sensors.make_tactile_contact_cfg(scene.TACTILE_BODY_NAMES)
    tactile_cfg.prim_path = tactile_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    tactile = tactile_cfg.class_type(tactile_cfg)
    radius, gap, probe_mass = args.tactile_radius, .006, .01
    probe = RigidObject(RigidObjectCfg(
        prim_path="/World/TactileProbe", spawn=sim_utils.SphereCfg(radius=radius,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=.1),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.0002, rest_offset=0.),
            mass_props=sim_utils.MassPropertiesCfg(mass=probe_mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.5, dynamic_friction=.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.95, .55, .12), roughness=.5)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 8.))))
    stage = sim_utils.get_current_stage()
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/TactileGuide")
    # World-anchored native guide. Its frame is moved only between independent
    # cases; the GPU tensor backend does not implement kinematic target writes.
    joint.CreateBody1Rel().SetTargets([Sdf.Path(probe.cfg.prim_path)])
    joint.CreateAxisAttr("Z")
    for suffix in ("0", "1"):
        getattr(joint, f"CreateLocalPos{suffix}Attr")(Gf.Vec3f(0., 0., 0.))
        getattr(joint, f"CreateLocalRot{suffix}Attr")(Gf.Quatf(1., Gf.Vec3f(0., 0., 0.)))
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(0., 0., 8.))
    joint.CreateLowerLimitAttr(-.001)
    joint.CreateUpperLimitAttr(.015)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(200.)
    drive.CreateDampingAttr(2.)
    drive.CreateMaxForceAttr(1.)
    target = drive.CreateTargetPositionAttr(0.)
    drive.CreateTargetVelocityAttr(0.)
    light = sim_utils.DomeLightCfg(intensity=1800., color=(.9, .94, 1.))
    light.func("/World/ProbeLight", light)
    camera = frames = pads = None
    if not args.no_video:
        for index in range(2):
            sim_utils.create_prim(f"/World/ProbeViews/View_{index}", "Xform")
        camera = Camera(CameraCfg(prim_path="/World/ProbeViews/View_.*/Camera", width=640, height=400,
            data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=24., horizontal_aperture=24.,
                                                               clipping_range=(.001, 15.))))
        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/World/TactileSensorFrame"
        frame_cfg.markers = {"frame": frame_cfg.markers["frame"]}
        frame_cfg.markers["frame"].scale = (.035, .035, .035)
        frames = VisualizationMarkers(frame_cfg)
        colors = ((.25, .85, .45), (.18, .23, .30), (.95, .65, .12))
        pads = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/World/TactileMeasuredBits", markers={
            name: sim_utils.SphereCfg(radius=.0008, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))
            for name, color in zip(("on", "off", "invalid"), colors)}))
        for path in (frame_cfg.prim_path, "/World/TactileMeasuredBits"):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError("Diagnostic markers must have no physical schemas.")
    sim.reset()
    events.configure_hand_action_limits(SimpleNamespace(scene={"robot": robot}))
    class ProbeScene(dict):
        env_prim_paths = ["/World/envs/env_0"]

    env = SimpleNamespace(scene=ProbeScene(robot=robot, tactile_contact=tactile), num_envs=1,
                          device=sim.device, sim=sim, cfg=cfg)
    reader = sensors.SensorReader(env)
    arm_action = cfg.actions.arm_action.class_type(cfg.actions.arm_action.copy(), env)
    hand_action = cfg.actions.hand_action.class_type(cfg.actions.hand_action.copy(), env)
    body_ids, resolved = robot.find_bodies(list(scene.TACTILE_BODY_NAMES), preserve_order=True)
    if tuple(resolved) != scene.TACTILE_BODY_NAMES:
        raise RuntimeError("Canonical tactile channel/body mapping changed.")
    body_ids = torch.tensor(body_ids, device=sim.device, dtype=torch.long)
    sensor_id = robot.body_names.index(scene.FORCE_TORQUE_BODY_NAME)
    joint_ids, _ = robot.find_joints(list(scene.ARM_JOINT_NAMES) + list(scene.HAND_JOINT_NAMES), preserve_order=True)
    pair_view = robot._physics_sim_view.create_rigid_contact_view(
        list(tactile.contact_physx_view.sensor_paths), filter_patterns=[[probe.cfg.prim_path]] * 17,
        max_contact_data_count=0)
    if (pair_view.sensor_count != 17 or pair_view.filter_count != 1
            or list(pair_view.sensor_paths) != list(tactile.contact_physx_view.sensor_paths)):
        raise RuntimeError("Probe witness view must preserve all 17 native sensor rows and one filter.")
    dt = sim.get_physics_dt()
    stride = round(1. / (dt * args.video_fps))
    if stride < 1 or not math.isclose(stride * dt * args.video_fps, 1., abs_tol=1.e-6):
        raise ValueError("Video FPS must exactly divide the physics frequency.")
    phases = (("baseline", .4), ("approach", .7), ("hold", .8), ("release", .7), ("clear", .4))
    duration = sum(value for _, value in phases)
    channels = list(range(args.tactile_start, args.tactile_start + args.tactile_count))
    metadata = {"status": "running", "package": "contact_manipulation_symmetric", "physics_dt_s": dt,
        "policy_decimation": cfg.decimation, "threshold_N": cfg.mdp.tactile_threshold,
        "pad_names": list(scene.TACTILE_BODY_NAMES), "tested_channels": channels, "case_duration_s": duration,
        "phases": [{"name": name, "duration_s": seconds} for name, seconds in phases],
        "fixture": {"radius_m": radius, "mass_kg": probe_mass, "initial_gap_m": gap,
                    "drive_stiffness_N_m": 200., "drive_damping_N_s_m": 2., "drive_force_limit_N": 1.,
                    "target_extra_stroke_m": args.tactile_penetration, "gravity_enabled": False,
                    "type": "Dynamic sphere on native driven world-anchored prismatic joint."},
        "hand_drive": {key: getattr(robot_cfg.actuators["hand"], key) for key in
                       ("stiffness", "damping", "armature", "effort_limit_sim", "velocity_limit_sim")},
        "robot_motion": "Unchanged production wrench OSC / IdealPD and 6-to-12 hand targets; zero wrench / open hand.",
        "state_writes": "One robot/probe native reset between independent cases; none within approach/hold/release.",
        "binary_source": "Canonical reordered grouped SensorReader net_forces_w norm > production threshold.",
        "pair_witness": "Independent native rigid contact view filtered to the physical probe; normal forces only.",
        "markers": "Visual-only RGB sensor frame and green/gray/amber actual tactile bits.",
    }
    samples, sample_indices, results = [], [], []
    temporary = tempfile.TemporaryDirectory(prefix="contact_tactile_video_")
    scene_video = Path(temporary.name) / "scene.mp4"
    writer = imageio.get_writer(str(scene_video), fps=args.video_fps, codec="libx264", quality=8) if camera else None
    zero_wrench = torch.zeros((1, 6), device=sim.device)
    hand_raw = zero_wrench.new_full((1, 6), -1. / cfg.actions.hand_action.scale)
    accepted_action = torch.cat((zero_wrench, hand_raw), -1)

    def update_visuals(channel, binary, valid):
        positions = rewards.tactile_pad_points(env)[0]
        rotations = robot.data.body_link_quat_w[0].index_select(0, body_ids)
        normal = observations.quat_apply(rotations, rotations.new_tensor([[0., 0., 1.]]).expand(17, -1))
        colors = torch.where(valid, torch.where(binary, 0, 1), 2)
        pads.visualize(translations=positions + normal * .0003, marker_indices=colors)
        frames.visualize(translations=robot.data.body_link_pos_w[:, sensor_id],
                         orientations=robot.data.body_link_quat_w[:, sensor_id])
        palm = positions[0]
        palm_normal = normal[0]
        palm_up = observations.quat_apply(rotations[:1], rotations.new_tensor([[0., 1., 0.]]))[0]
        overview = (positions.mean(0) + robot.data.body_link_pos_w[0, sensor_id]) * .5
        close = positions[channel]
        eyes = torch.stack((overview + .43 * palm_normal + .10 * palm_up,
                            close + .10 * normal[channel] + .06 * palm_normal + .02 * palm_up))
        camera.set_world_poses_from_view(eyes, torch.stack((overview, close)))

    def save_report():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"metadata": metadata, "pad_results": results, "samples": samples}, indent=2) + "\n")

    try:
        for channel in channels:
            print(f"[Tactile probe] channel {channel:02d}: {scene.TACTILE_BODY_NAMES[channel]}", flush=True)
            target.Set(0.)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            robot.reset()
            sim.forward()
            points = rewards.tactile_pad_points(env)[0]
            rotation = robot.data.body_link_quat_w[0, body_ids[channel]]
            normal = observations.quat_apply(rotation[None], rotation.new_tensor([[0., 0., 1.]]))[0]
            point = points[channel]
            quaternion = Gf.Rotation(Gf.Vec3d(0., 0., 1.), Gf.Vec3d(*(-normal).cpu().tolist())).GetQuat()
            state = probe.data.default_root_state.clone()
            state[:, :3] = point + normal * (radius + gap)
            state[:, 3:7] = state.new_tensor([quaternion.GetReal(), *quaternion.GetImaginary()])
            state[:, 7:] = 0.
            joint.GetLocalPos0Attr().Set(Gf.Vec3f(*state[0, :3].cpu().tolist()))
            joint.GetLocalRot0Attr().Set(Gf.Quatf(quaternion.GetReal(), Gf.Vec3f(*quaternion.GetImaginary())))
            probe.write_root_state_to_sim(state)
            probe.reset()
            tactile.reset()
            sim.forward()
            reader.reset(torch.tensor([0], device=sim.device))
            arm_action.reset()
            hand_action.reset()
            arm_action.process_actions(zero_wrench)
            hand_action.process_actions(hand_raw)
            if camera:
                update_visuals(channel, torch.zeros(17, device=sim.device, dtype=torch.bool),
                               torch.ones(17, device=sim.device, dtype=torch.bool))
                for _ in range(4):
                    sim.render()
            case_start = len(samples)
            case_steps = 0
            for phase, seconds in phases:
                count = round(seconds / dt)
                for step in range(count):
                    amount = smooth_step((step + 1) / count)
                    stroke = gap + args.tactile_penetration
                    position_target = (stroke * amount if phase == "approach" else stroke if phase == "hold"
                                       else stroke * (1. - amount) if phase == "release" else 0.)
                    target.Set(position_target)
                    if case_steps % cfg.decimation == 0:
                        arm_action.process_actions(zero_wrench)
                        hand_action.process_actions(hand_raw)
                    arm_action.apply_actions()
                    hand_action.apply_actions()
                    robot.write_data_to_sim()
                    sim.step(render=False)
                    robot.update(dt)
                    probe.update(dt)
                    tactile.update(dt)
                    batch = reader()
                    binary, valid, _, _ = sensors.sensor_monitor_snapshot(batch.tactile_force, batch.tactile_valid,
                        batch.ideal_wrench, batch.wrench_valid, cfg.mdp.tactile_threshold)
                    packed = observations.dynamic_sample(robot.data.joint_pos[:, joint_ids], robot.data.joint_vel[:, joint_ids],
                        torch.cat((robot.data.body_link_pos_w[:, sensor_id], robot.data.body_link_quat_w[:, sensor_id]), -1),
                        zero_wrench, binary, valid, accepted_action)
                    if not torch.equal(packed[:, 37:54], torch.where(valid, binary.float(), -1.)):
                        raise RuntimeError("Displayed tactile bits differ from the packed observation channels.")
                    force = batch.tactile_force[0].norm(dim=-1)
                    witness = pair_view.get_contact_force_matrix(dt=dt).reshape(1, 17, 3).index_select(1, reader.tactile_ids)[0].norm(dim=-1)
                    if not bool(torch.isfinite(robot.data.joint_pos).all() & torch.isfinite(force).all()
                                & torch.isfinite(witness).all() & valid.all()):
                        raise RuntimeError("Invalid native joint/contact state in tactile scan.")
                    case_steps += 1
                    samples.append({"time_s": (len(samples) + 1) * dt, "channel": channel, "phase": phase,
                        "case_time_s": case_steps * dt, "drive_target_m": position_target,
                        "policy_sample": case_steps % cfg.decimation == 0,
                        "normal_force_N": force.cpu().tolist(), "probe_pair_force_N": witness.cpu().tolist(),
                        "binary": binary[0].cpu().tolist(), "valid": valid[0].cpu().tolist(),
                        "probe_position_w_m": probe.data.root_pos_w[0].cpu().tolist(),
                        "selected_pad_position_w_m": rewards.tactile_pad_points(env)[0, channel].cpu().tolist(),
                        "arm_max_speed_rad_s": float(robot.data.joint_vel[:, joint_ids[:6]].abs().max())})
                    if writer is not None and len(samples) % stride == 0:
                        update_visuals(channel, binary[0], valid[0])
                        sim.render()
                        camera.update(dt * stride, force_recompute=True)
                        rgb = camera.data.output["rgb"][..., :3].cpu().numpy()
                        writer.append_data(np.concatenate((rgb[0], rgb[1]), axis=1))
                        sample_indices.append(len(samples) - 1)
                    if args.max_steps is not None and len(samples) >= args.max_steps:
                        break
                if args.max_steps is not None and len(samples) >= args.max_steps:
                    break
            case = samples[case_start:]
            baseline = [s for s in case if s["phase"] == "baseline"]
            contact = [s for s in case if s["phase"] in ("approach", "hold")]
            clear = [s for s in case if s["phase"] == "clear"]
            entry = {"channel": channel, "pad_name": scene.TACTILE_BODY_NAMES[channel], "samples": len(case),
                "baseline_off": bool(baseline) and not any(s["binary"][channel] for s in baseline),
                "contact_on": any(s["binary"][channel] for s in contact),
                "probe_contact_on": any(s["binary"][channel]
                    and s["probe_pair_force_N"][channel] > cfg.mdp.tactile_threshold for s in contact),
                "released_off": bool(clear) and not any(s["binary"][channel] for s in clear[-10:]),
                "contact_samples": sum(s["binary"][channel] for s in contact),
                "policy_contact_samples": sum(s["binary"][channel] and s["policy_sample"] for s in contact),
                "max_normal_force_N": max(s["normal_force_N"][channel] for s in case),
                "max_probe_pair_force_N": max(s["probe_pair_force_N"][channel] for s in case),
                "simultaneous_channels": sorted({i for s in contact for i, on in enumerate(s["binary"]) if on})}
            entry["passed"] = (entry["baseline_off"] and entry["contact_on"]
                               and entry["probe_contact_on"] and entry["released_off"])
            results.append(entry)
            print(f"[Tactile probe] {entry}", flush=True)
            save_report()
            if args.max_steps is not None and len(samples) >= args.max_steps:
                break
        metadata["status"] = ("complete" if len(results) == len(channels) and all(r["passed"] for r in results)
                              else "contact_check_failed")
    except Exception as error:
        import traceback

        metadata["error"] = repr(error)
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if metadata["status"] == "running":
            metadata["status"] = "failed"
        save_report()
    if camera is not None:
        print(f"[Tactile probe] Encoding measured tactile overlay: {args.video_output}", flush=True)
        encode_tactile_overlay(scene_video, args.video_output, samples, sample_indices, args.video_fps, metadata)
    temporary.cleanup()
    print(f"[Tactile probe] {metadata['status']}: {sum(r['passed'] for r in results)}/{len(results)} pad cases.", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    app_launcher = AppLauncher(arguments)
    try:
        (run_tactile_scan if arguments.tactile_scan else run_probe)(arguments)
    except Exception as error:
        import traceback

        detail = {"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.with_suffix(".error.json").write_text(json.dumps(detail, indent=2) + "\n")
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # MP4/report writes above are synchronous. Match the existing hand
        # contact probe's close mode to avoid its known contact-view teardown hang.
        app_launcher.app.close(wait_for_replicator=False, skip_cleanup=True)
