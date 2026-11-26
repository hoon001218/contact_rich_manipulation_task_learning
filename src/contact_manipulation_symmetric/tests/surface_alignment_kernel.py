"""Diagnostic Warp kernels for nearest collision-face alignment at tactile pads.

Meshes use metric body-local vertices and consistently outward face winding.
Closest-point queries are unsigned. Points inside a collision mesh can also
return a face, so this diagnostic does not measure penetration or contact.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from contact_manipulation_symmetric.mdp.surface_alignment import nearest_face_score


@wp.kernel(enable_backward=False)
def _query_pads(
    points: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    body_poses: wp.array2d(dtype=float),
    mesh_ids: wp.array(dtype=wp.uint64),
    decay: float,
    max_distance: float,
    scores: wp.array2d(dtype=float),
):
    env, pad = wp.tid()
    origin = wp.vec3(body_poses[env, 0], body_poses[env, 1], body_poses[env, 2])
    # Isaac Lab stores wxyz. Warp quaternions take xyzw.
    rotation = wp.quat(body_poses[env, 4], body_poses[env, 5], body_poses[env, 6], body_poses[env, 3])
    if not wp.isfinite(origin) or not wp.isfinite(rotation) or wp.length_sq(rotation) <= 1.0e-16:
        scores[env, pad] = 0.0
        return
    scores[env, pad] = nearest_face_score(
        mesh_ids[env], points[env, pad], normals[env, pad], origin, wp.normalize(rotation), decay, max_distance,
    )


@wp.kernel(enable_backward=False)
def _mean_pad_scores(scores: wp.array2d(dtype=float), output: wp.array(dtype=float)):
    env = wp.tid()
    summed = float(0.0)
    for pad in range(scores.shape[1]):
        summed += scores[env, pad]
    output[env] = summed / float(scores.shape[1])


@wp.kernel(enable_backward=False)
def _query_and_mean(
    points: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    body_poses: wp.array2d(dtype=float),
    mesh_ids: wp.array(dtype=wp.uint64),
    decay: float,
    max_distance: float,
    output: wp.array(dtype=float),
):
    env = wp.tid()
    origin = wp.vec3(body_poses[env, 0], body_poses[env, 1], body_poses[env, 2])
    rotation = wp.quat(body_poses[env, 4], body_poses[env, 5], body_poses[env, 6], body_poses[env, 3])
    if not wp.isfinite(origin) or not wp.isfinite(rotation) or wp.length_sq(rotation) <= 1.0e-16:
        output[env] = 0.0
        return
    rotation = wp.normalize(rotation)
    mesh_id = mesh_ids[env]
    summed = float(0.0)
    for pad in range(points.shape[1]):
        summed += nearest_face_score(mesh_id, points[env, pad], normals[env, pad], origin, rotation, decay, max_distance)
    output[env] = summed / float(points.shape[1])


class SurfaceAlignmentKernel:
    """Bind tensor storage once and return the same score tensor on each launch.

    Points and normals are world-frame float32 tensors of shape ``[N, P, 3]``.
    ``body_poses_w`` is ``[N, 7]`` with xyz positions and wxyz rotations. Each
    ``mesh_ids`` row refers to a Warp mesh in that environment's body frame.
    Keep the owning Warp meshes alive, and update the bound inputs in place.
    """

    def __init__(self, points_w: torch.Tensor, normals_w: torch.Tensor, body_poses_w: torch.Tensor,
                 mesh_ids: wp.array, decay: float, *, variant: str = "parallel", max_distance: float = 1.0e4):
        if (points_w.ndim != 3 or points_w.shape[1] < 1 or points_w.shape[2] != 3
                or normals_w.shape != points_w.shape):
            raise ValueError("Expected pad points and normals [N, P, 3].")
        self.count, num_pads = points_w.shape[:2]
        if body_poses_w.shape != (self.count, 7):
            raise ValueError("Expected Blocker link poses [N, 7] with xyz and wxyz.")
        tensors = (points_w, normals_w, body_poses_w)
        device = points_w.device
        if any(value.dtype != torch.float32 or value.device != device or not value.is_contiguous() for value in tensors):
            raise ValueError("Pad points, pad normals and body poses must be contiguous float32 on one device.")
        self.device = wp.get_device(str(device))
        if (mesh_ids.dtype != wp.uint64 or mesh_ids.shape != (self.count,) or mesh_ids.device != self.device):
            raise ValueError("Expected one Warp uint64 mesh ID per environment on the tensor device.")
        if not math.isfinite(decay) or decay <= 0.0 or not math.isfinite(max_distance) or max_distance <= 0.0:
            raise ValueError("Decay and query limit must be finite and positive.")
        if variant not in ("parallel", "serial"):
            raise ValueError("Choose the parallel or serial diagnostic kernel.")
        self.variant = variant
        self._torch_inputs = tensors
        self.output = torch.empty(self.count, device=device, dtype=torch.float32)
        self._launches = ()
        if self.count:
            self._query_args = [wp.from_torch(points_w, dtype=wp.vec3), wp.from_torch(normals_w, dtype=wp.vec3),
                                wp.from_torch(body_poses_w), mesh_ids, float(decay), float(max_distance)]
            self._output_view = wp.from_torch(self.output)
            if variant == "serial":
                self._launches = (
                    wp.launch(_query_and_mean, dim=self.count, inputs=[*self._query_args, self._output_view],
                              device=self.device, record_cmd=True),
                )
            else:
                self._scores = wp.empty((self.count, num_pads), dtype=float, device=self.device)
                self._launches = (
                    wp.launch(_query_pads, dim=(self.count, num_pads), inputs=[*self._query_args, self._scores],
                              device=self.device, record_cmd=True),
                    wp.launch(_mean_pad_scores, dim=self.count, inputs=[self._scores, self._output_view],
                              device=self.device, record_cmd=True),
                )

    def launch(self, stream: wp.Stream | None = None) -> torch.Tensor:
        """Launch on the caller's Warp stream or the current Torch CUDA stream."""
        if stream is None and self.output.is_cuda:
            stream = wp.stream_from_torch(torch.cuda.current_stream(self.output.device))
        for command in self._launches:
            command.launch(stream)
        return self.output
