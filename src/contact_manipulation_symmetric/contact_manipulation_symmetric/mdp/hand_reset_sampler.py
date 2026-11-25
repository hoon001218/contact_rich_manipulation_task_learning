"""Reset-only sampling of coupled Hand actions from convex separation intervals.

Geometry is cooked from the default RH56E2 URDF and expressed in its link frames.
The bundled data stores geometry, swept motion bounds and a thumb-axis covering
set for deterministic fallback. Hand/world clearance is handled by reset IK.

To regenerate the package data, export hulls with
``tests/probe_hand_reset_randomization.py --export-hand-hulls``, certify boxes
with ``tests/probe_hand_reset_certify.py``, then run
``tests/build_hand_reset_geometry.py COOKED_JSON CERTIFIED_BOXES_NPZ OUTPUT_NPZ``.
Regenerate after changing the Hand URDF, collision meshes or cooking settings.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
import warp as wp


_GEOMETRY = Path(__file__).resolve().parents[1] / "data" / "hand_reset_geometry.npz"
_THUMB_BINS = 64
_FINGER_BINS = 32
_WITNESS = -1.98
_WITNESS_HALF_WIDTH = 0.016
_CLEARANCE_M = 0.0001
_FLOAT_PADDING_M = 0.000004

@wp.kernel(enable_backward=False)
def _forward_links(raw: wp.array2d(dtype=float),
                  chains: wp.array2d(dtype=wp.int32),
                  counts: wp.array(dtype=wp.int32),
                  origin_r: wp.array(dtype=wp.mat33),
                  origin_t: wp.array(dtype=wp.vec3),
                  axis: wp.array(dtype=wp.vec3),
                  motors: wp.array(dtype=wp.int32),
                  slopes: wp.array(dtype=float),
                  bias: wp.array(dtype=float),
                  rotations: wp.array2d(dtype=wp.mat33),
                  translations: wp.array2d(dtype=wp.vec3)):
    row, link = wp.tid()
    r = wp.identity(n=3, dtype=float)
    p = wp.vec3(0.)
    for k in range(counts[link]):
        joint = chains[link, k]
        p = p + r * origin_t[joint]
        r = r * origin_r[joint]
        motor = motors[joint]
        if motor >= 0:
            angle = raw[row, motor] * slopes[joint] + bias[joint]
            r = r * wp.quat_to_matrix(wp.quat_from_axis_angle(axis[joint], angle))
    rotations[row, link] = r
    translations[row, link] = p


@wp.func
def _extent(r: wp.mat33, e: wp.vec3) -> wp.vec3:
    return wp.vec3(
        wp.abs(r[0, 0])*e[0]+wp.abs(r[0, 1])*e[1]+wp.abs(r[0, 2])*e[2],
        wp.abs(r[1, 0])*e[0]+wp.abs(r[1, 1])*e[1]+wp.abs(r[1, 2])*e[2],
        wp.abs(r[2, 0])*e[0]+wp.abs(r[2, 1])*e[1]+wp.abs(r[2, 2])*e[2])


@wp.func
def _proves_axis(vertices: wp.array2d(dtype=wp.vec3), vertex_counts: wp.array(dtype=wp.int32),
                a: int, b: int, ra_t: wp.mat33, rb_t: wp.mat33, delta: wp.vec3,
                axis: wp.vec3, required: float) -> bool:
    d_a = ra_t * axis
    d_b = rb_t * axis
    a_min = float(10.0)
    b_min = float(10.0)
    a_max = float(-10.0)
    b_max = float(-10.0)
    for k in range(vertex_counts[a]):
        x = wp.dot(vertices[a, k], d_a)
        a_min = wp.min(a_min, x)
        a_max = wp.max(a_max, x)
    for k in range(vertex_counts[b]):
        x = wp.dot(vertices[b, k], d_b)
        b_min = wp.min(b_min, x)
        b_max = wp.max(b_max, x)
    offset = wp.dot(delta, axis)
    return wp.max(b_min - a_max + offset, a_min - b_max - offset) >= required


@wp.kernel(enable_backward=False)
def _check_intervals(rotations: wp.array2d(dtype=wp.mat33),
                    translations: wp.array2d(dtype=wp.vec3),
                    piece_link: wp.array(dtype=wp.int32),
                    local_center: wp.array(dtype=wp.vec3),
                    local_half: wp.array(dtype=wp.vec3),
                    vertices: wp.array2d(dtype=wp.vec3),
                    vertex_counts: wp.array(dtype=wp.int32),
                    face_axes: wp.array2d(dtype=wp.vec3),
                    face_counts: wp.array(dtype=wp.int32),
                    pairs: wp.array2d(dtype=wp.int32),
                    required: wp.array(dtype=float),
                    valid: wp.array(dtype=wp.int32)):
    row, pair = wp.tid()
    a, b = pairs[pair, 0], pairs[pair, 1]
    a_link, b_link = piece_link[a], piece_link[b]
    ra, rb = rotations[row, a_link], rotations[row, b_link]
    delta = translations[row, b_link] - translations[row, a_link]
    cd = delta + rb * local_center[b] - ra * local_center[a]
    gap = _extent(ra, local_half[a]) + _extent(rb, local_half[b]) + wp.vec3(required[pair])
    if wp.abs(cd[0]) >= gap[0] or wp.abs(cd[1]) >= gap[1] or wp.abs(cd[2]) >= gap[2]:
        return
    ra_t, rb_t = wp.transpose(ra), wp.transpose(rb)
    if wp.length_sq(cd) > 1.0e-12:
        if _proves_axis(vertices, vertex_counts, a, b, ra_t, rb_t, delta, wp.normalize(cd), required[pair]):
            return
    for k in range(face_counts[a]):
        if _proves_axis(vertices, vertex_counts, a, b, ra_t, rb_t, delta, ra * face_axes[a, k], required[pair]):
            return
    for k in range(face_counts[b]):
        if _proves_axis(vertices, vertex_counts, a, b, ra_t, rb_t, delta, rb * face_axes[b, k], required[pair]):
            return
    wp.atomic_min(valid, row, 0)



class HandResetSampler:
    """Choose a thumb value, then draw subsequent channels in feasible intervals."""

    uniform_dim = 9
    action_dim = 6

    def __init__(self, device: str | torch.device, *, urdf_path: str | Path | None = None):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._wp_device = wp.get_device(str(self.device))
        with np.load(_GEOMETRY, allow_pickle=False) as geometry:
            if urdf_path is not None:
                actual_digest = sha256(Path(urdf_path).read_bytes()).hexdigest()
                if actual_digest != str(geometry["urdf_sha256"]):
                    raise ValueError("Hand reset collision geometry does not match the configured URDF.")
            self._load_geometry(geometry)
        self._thumb_centers = self._bin_centers(_THUMB_BINS)
        self._finger_centers = self._bin_centers(_FINGER_BINS)
        self._capacity = 0
        self.last_fallback = torch.empty((0, 3), device=self.device, dtype=torch.bool)

    def _load_geometry(self, geometry):
        def to_warp(key, dtype):
            return wp.array(geometry[key], dtype=dtype, device=self._wp_device)

        self._link_count = len(geometry["chain_length"])
        self._kinematics = (
            to_warp("chain", wp.int32),
            to_warp("chain_length", wp.int32),
            to_warp("joint_rot", wp.mat33),
            to_warp("joint_pos", wp.vec3),
            to_warp("joint_axis", wp.vec3),
            to_warp("joint_motor", wp.int32),
            to_warp("joint_gain", float),
            to_warp("joint_bias", float),
        )
        self._collision_geometry = (
            to_warp("piece_link", wp.int32),
            to_warp("piece_center", wp.vec3),
            to_warp("piece_half_extents", wp.vec3),
            to_warp("vertex", wp.vec3),
            to_warp("vertex_count", wp.int32),
            to_warp("face_axis", wp.vec3),
            to_warp("face_axis_count", wp.int32),
        )
        # Other pair groups separate across their entire joint domain in the
        # offline certificate. Check the three coupled thumb groups at reset.
        half_widths = {
            "thumb": [0, 2 / _THUMB_BINS, _WITNESS_HALF_WIDTH, _WITNESS_HALF_WIDTH, 0, 0],
            "index": [0, 0, 2 / _FINGER_BINS, 0, 0, 0],
            "middle": [0, 0, 0, 2 / _FINGER_BINS, 0, 0],
        }
        self._stages = {}
        for stage, half_width in half_widths.items():
            key = f"{stage}_pairs"
            motion = geometry[f"{stage}_motion_m"] @ half_width
            minimum = motion + _CLEARANCE_M + _FLOAT_PADDING_M
            self._stages[stage] = (to_warp(key, wp.int32), wp.array(minimum, dtype=float, device=self._wp_device),
                                   len(geometry[key]))
        lower, upper = geometry["fallback_lower"], geometry["fallback_upper"]
        if (lower[0, 0] != -2 or upper[-1, 0] != 2 or not np.array_equal(lower[1:, 0], upper[:-1, 0])
                or geometry["fallback_gap_m"].min() < _CLEARANCE_M + _FLOAT_PADDING_M):
            raise ValueError("Hand reset fallback must cover the thumb axis with certified clearance.")
        self._fallback_lower = torch.as_tensor(lower[:, 1:].copy(), device=self.device)
        self._fallback_upper = torch.as_tensor(upper[:, 1:].copy(), device=self.device)
        self._fallback_thumb_ends = torch.as_tensor(upper[:, 0].copy(), device=self.device)

    def _bin_centers(self, count: int) -> torch.Tensor:
        return torch.linspace(-2 + 2 / count, 2 - 2 / count, count, device=self.device, dtype=torch.float32)

    def _verify_intervals(self, raw: torch.Tensor, stage: str, stream) -> torch.Tensor:
        rows = raw.shape[0]
        if rows > self._capacity:
            # A reset can have any batch size. Keep only one growable workspace.
            self._capacity = 1 << (rows - 1).bit_length()
            shape = (self._capacity, self._link_count)
            self._link_rotations = wp.empty(shape, dtype=wp.mat33, device=self._wp_device)
            self._link_positions = wp.empty(shape, dtype=wp.vec3, device=self._wp_device)
        valid = torch.ones(rows, device=self.device, dtype=torch.int32)
        rotations, positions = self._link_rotations, self._link_positions
        wp.launch(_forward_links, dim=(rows, self._link_count),
                  inputs=[wp.from_torch(raw), *self._kinematics, rotations, positions],
                  device=self._wp_device, stream=stream)
        pairs, minimum, count = self._stages[stage]
        wp.launch(_check_intervals, dim=(rows, count),
                  inputs=[rotations, positions, *self._collision_geometry, pairs, minimum, wp.from_torch(valid)],
                  device=self._wp_device, stream=stream)
        return valid > 0

    @staticmethod
    def _select_interval(valid: torch.Tensor, centers: torch.Tensor, rank: torch.Tensor,
                         offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = valid.sum(dim=1)
        # Clamping the rank also handles a uniform rounded to 1 in float32.
        rank = ((count * rank).floor().long() + 1).minimum(count)
        selected = (valid.cumsum(dim=1) >= rank[:, None]).to(torch.int32).argmax(dim=1)
        action = centers[selected] + (offset - 0.5) * (4 / centers.numel())
        return action, count > 0

    def sample(self, raw_uniform: torch.Tensor) -> torch.Tensor:
        """Map N by 9 uniform draws to N by 6 raw actions without resampling.

        The caller supplies random numbers so the reset seed and environment
        order determine each sample. The result stays on the configured device.
        """
        if raw_uniform.ndim != 2 or raw_uniform.shape[1] != self.uniform_dim:
            raise ValueError(f"Expected one row of {self.uniform_dim} uniform draws per reset.")
        uniform = raw_uniform.to(device=self.device, dtype=torch.float32)
        count = uniform.shape[0]
        self.last_fallback = torch.empty((0, 3), device=self.device, dtype=torch.bool)
        if count == 0:
            return uniform[:, :self.action_dim]
        stream = (wp.stream_from_torch(torch.cuda.current_stream(self.device))
                  if self.device.type == "cuda" else None)

        thumb_first = uniform[:, 0] * 4 - 2
        thumb_candidates = uniform.new_zeros((count, _THUMB_BINS, self.action_dim))
        thumb_candidates[:, :, 0] = thumb_first[:, None]
        thumb_candidates[:, :, 1] = self._thumb_centers
        thumb_candidates[:, :, 2:4] = _WITNESS
        valid = self._verify_intervals(thumb_candidates.flatten(0, 1), "thumb", stream).reshape(count, _THUMB_BINS)
        thumb_second, has_thumb_interval = self._select_interval(valid, self._thumb_centers,
                                                                 uniform[:, 1], uniform[:, 2])

        # A valid thumb pair leaves the two witness intervals feasible. Scan
        # each finger more broadly, and use the witness only when no scan cell
        # fits. The finger-to-finger pairs are covered by the offline proof.
        candidates = uniform.new_zeros((count, _FINGER_BINS, self.action_dim))
        candidates[:, :, 0] = thumb_first[:, None]
        candidates[:, :, 1] = thumb_second[:, None]
        finger_values, finger_fallbacks = [], []
        for stage, axis, column in (("index", 2, 3), ("middle", 3, 5)):
            candidates[:, :, axis] = self._finger_centers
            valid = self._verify_intervals(candidates.flatten(0, 1), stage, stream).reshape(count, _FINGER_BINS)
            value, has_interval = self._select_interval(valid, self._finger_centers,
                                                        uniform[:, column], uniform[:, column + 1])
            witness = _WITNESS + (uniform[:, column + 1] - 0.5) * (2 * _WITNESS_HALF_WIDTH)
            finger_values.append(torch.where(has_interval, value, witness))
            finger_fallbacks.append(has_thumb_interval & ~has_interval)
        sampled_tail = torch.stack((thumb_second, *finger_values, uniform[:, 7] * 4 - 2, uniform[:, 8] * 4 - 2), dim=1)

        # The offline coverage boxes are contiguous in the first thumb channel.
        # A rare failed scan keeps thumb_first and uses the compatible box slice.
        box = torch.searchsorted(self._fallback_thumb_ends, thumb_first)
        lower, upper = self._fallback_lower[box], self._fallback_upper[box]
        covered_tail = lower + uniform[:, (2, 4, 6, 7, 8)] * (upper - lower)
        tail = torch.where(has_thumb_interval[:, None], sampled_tail, covered_tail)
        self.last_fallback = torch.stack((~has_thumb_interval, *finger_fallbacks), dim=1)
        return torch.cat((thumb_first[:, None], tail), dim=1)
