"""GPU-batched tactile alignment against the current Blocker collision surface."""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp


@wp.func
def nearest_face_score(mesh_id: wp.uint64, point_world: wp.vec3, normal_world: wp.vec3,
                       body_origin: wp.vec3, body_rotation: wp.quat, decay: float, max_distance: float) -> float:
    if mesh_id == wp.uint64(0) or not wp.isfinite(point_world) or not wp.isfinite(normal_world):
        return 0.0
    if wp.length_sq(normal_world) <= 1.0e-16:
        return 0.0
    point_local = wp.quat_rotate_inv(body_rotation, point_world - body_origin)
    nearest = wp.mesh_query_point_no_sign(mesh_id, point_local, max_distance)
    if not nearest.result:
        return 0.0
    surface = wp.mesh_eval_face_normal(mesh_id, nearest.face)
    if not wp.isfinite(surface) or wp.length_sq(surface) <= 1.0e-16:
        return 0.0
    pad = wp.quat_rotate_inv(body_rotation, wp.normalize(normal_world))
    cosine = wp.clamp(wp.dot(pad, wp.normalize(surface)), -1.0, 1.0)
    return wp.exp(-decay * (1.0 + cosine))


@wp.kernel(enable_backward=False)
def _score_pads(
    points: wp.array2d(dtype=wp.vec3),
    normals: wp.array2d(dtype=wp.vec3),
    poses: wp.array3d(dtype=float),
    blocker_ids: wp.array(dtype=wp.int64),
    mesh_ids: wp.array(dtype=wp.uint64),
    decay: float,
    scores: wp.array2d(dtype=float),
):
    env, pad = wp.tid()
    blocker_id = wp.int32(blocker_ids[env])
    if blocker_id < 0 or blocker_id >= mesh_ids.shape[0]:
        scores[env, pad] = 0.0
        return

    origin = wp.vec3(poses[env, blocker_id, 0], poses[env, blocker_id, 1], poses[env, blocker_id, 2])
    # Native object poses store wxyz. Warp quaternions take xyzw.
    rotation = wp.quat(poses[env, blocker_id, 4], poses[env, blocker_id, 5],
                       poses[env, blocker_id, 6], poses[env, blocker_id, 3])
    if not wp.isfinite(origin) or not wp.isfinite(rotation) or wp.length_sq(rotation) <= 1.0e-16:
        scores[env, pad] = 0.0
        return

    # The mesh API requires a finite query limit. 10 km covers all shelf poses.
    scores[env, pad] = nearest_face_score(
        mesh_ids[blocker_id], points[env, pad], normals[env, pad], origin, wp.normalize(rotation), decay, 1.0e4,
    )


@wp.kernel(enable_backward=False)
def _mean_pad_scores(scores: wp.array2d(dtype=float), means: wp.array(dtype=float)):
    env = wp.tid()
    total = float(0.0)
    for pad in range(scores.shape[1]):
        total += scores[env, pad]
    means[env] = total / float(scores.shape[1])


class BlockerSurfaceAlignment:
    """Cache one body-local Warp mesh per asset and batch queries over current poses."""

    def __init__(self, env, num_pads: int):
        from pxr import Usd, UsdPhysics

        from ..config.ur5e.scene import OBJECT_SPECS
        from .collision_surfaces import collision_mesh

        self.objects = env.scene["object_collection"]
        self.device = str(env.device)
        self.num_pads = num_pads
        if tuple(self.objects.object_names) != tuple(spec.name for spec in OBJECT_SPECS):
            raise RuntimeError("Object collection order differs from the Blocker surface catalog.")
        # PhysX stores the collection in object-major order. Use env_0's copy.
        paths = self.objects.root_physx_view.prim_paths[::self.objects.num_instances]
        if len(paths) != len(OBJECT_SPECS):
            raise RuntimeError("Unexpected number of native Blocker object roots.")

        stage = env.scene.stage
        self.meshes = {}
        mesh_ids = []
        for path, spec in zip(paths, OBJECT_SPECS):
            if spec.asset_name not in self.meshes:
                body = stage.GetPrimAtPath(str(path))
                if not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    (body,) = (
                        prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    )
                vertices, triangles, _ = collision_mesh(stage, body)
                self.meshes[spec.asset_name] = wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=self.device),
                    indices=wp.array(triangles.reshape(-1), dtype=wp.int32, device=self.device),
                )
            mesh_ids.append(self.meshes[spec.asset_name].id)
        self.mesh_ids = wp.array(np.asarray(mesh_ids, dtype=np.uint64), dtype=wp.uint64, device=self.device)
        self.scores = wp.empty((env.num_envs, num_pads), dtype=float, device=self.device)
        with torch.inference_mode(False):
            self.means = torch.empty(env.num_envs, dtype=torch.float32, device=env.device)
        self.means_view = wp.from_torch(self.means)

    def compute(self, points, normals, blocker_ids, *, decay: float):
        if not math.isfinite(decay) or decay <= 0.0:
            raise ValueError("Alignment decay must be finite and positive.")
        poses = self.objects.data.object_link_pose_w
        # Share Torch's CUDA stream so the inputs and results need no device sync.
        stream = wp.stream_from_torch(torch.cuda.current_stream(points.device)) if points.is_cuda else None
        wp.launch(
            _score_pads, dim=(points.shape[0], self.num_pads),
            inputs=[wp.from_torch(points, dtype=wp.vec3), wp.from_torch(normals, dtype=wp.vec3),
                    wp.from_torch(poses), wp.from_torch(blocker_ids), self.mesh_ids, decay, self.scores],
            device=self.device, stream=stream, record_tape=False,
        )
        wp.launch(_mean_pad_scores, dim=points.shape[0], inputs=[self.scores, self.means_view],
                  device=self.device, stream=stream, record_tape=False)
        return self.means
