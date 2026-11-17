"""Scene-boundary, per-object friction using native PhysX material buffers."""

from __future__ import annotations

import math

import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import sample_uniform


class SceneFrictionRandomization(ManagerTermBase):
    """Resample each environment/object pair once after a shared scene commit.

    Isaac Lab's material randomizer does not accept RigidObjectCollection.
    Use its native uniform sampler, finite-bucket approach and PhysX getter/
    setter, with the collection's public reshape method for body ordering.
    Every shape/hull belonging to one body receives the same coefficients.
    Partial episode resets perform no CPU tensor copies or material writes.

    Shelf MIN (.4/.32) + object AVERAGE gives each sampled object coefficient.
    Hand MAX (.8/.64) keeps the higher rubber coefficient for hand contacts.
    Each environment owns three base materials, outside the body prototypes;
    geometry references never embed them per body. This event changes only
    the scalar material coefficients on the existing native actors/shapes.
    """

    def __init__(self, cfg, env):
        ratio = float(cfg.params.get("dynamic_friction_ratio", 0.8))
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("dynamic_friction_ratio must be finite and in [0, 1].")
        if cfg.params["asset_cfg"].name != "object_collection":
            raise ValueError("Scene friction randomization must target object_collection.")
        buckets = cfg.params.get("num_buckets", 128)
        if isinstance(buckets, bool) or not isinstance(buckets, int) or not 1 <= buckets <= 64000:
            raise ValueError("num_buckets must be an integer in [1, 64000].")
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        if not isinstance(self.asset, RigidObjectCollection):
            raise ValueError("Per-object friction requires a native RigidObjectCollection.")
        if self.asset.num_instances != env.num_envs:
            raise ValueError("Object collection and environment counts must match.")
        lower, upper = cfg.params["static_friction_range"]
        if (not math.isfinite(lower) or not math.isfinite(upper)
                or not 0.0 <= lower <= upper):
            raise ValueError("Static friction bounds must be finite, ordered and nonnegative.")
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        if tuple(restitution_range) != (0.0, 0.0):
            raise ValueError("Per-object friction variation retains zero restitution.")
        self.dynamic_friction_ratio = ratio
        static = sample_uniform(lower, upper, (buckets,), device="cpu")
        self.material_buckets = torch.stack((static, ratio * static, torch.zeros_like(static)), -1)
        self._all_view_ids = torch.arange(self.asset.root_physx_view.count, device="cpu")
        self.last_scene_generation = None
        self.last_bucket_ids = None

    def __call__(self, env, env_ids, static_friction_range, dynamic_friction_range,
                 restitution_range, num_buckets, asset_cfg, make_consistent=False,
                 dynamic_friction_ratio=0.8):
        runtime = getattr(env, "_episode_reset", None)
        generation = getattr(runtime, "scene_generation", None)
        if generation is None:
            raise RuntimeError("Scene friction must run after reset_scene commits the shared scene.")
        if generation == self.last_scene_generation:
            return
        if dynamic_friction_ratio != self.dynamic_friction_ratio:
            raise ValueError("Changing the friction ratio requires recreating the material event.")
        if env_ids is not None and not (isinstance(env_ids, slice) and env_ids == slice(None)):
            ids = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")
            if (ids.ndim != 1 or len(ids) != env.num_envs
                    or not torch.equal(ids.sort().values, torch.arange(env.num_envs))):
                raise ValueError("A new shared scene must randomize all object instances together.")
        view = self.asset.root_physx_view
        bucket_ids = torch.randint(len(self.material_buckets),
            (self.asset.num_instances, self.asset.num_objects, 1), device="cpu")
        # Native collection rows are object-major, not environment-major.
        view_bucket_ids = self.asset.reshape_data_to_view(bucket_ids).squeeze(-1)
        samples = self.material_buckets[view_bucket_ids]
        materials = view.get_material_properties()
        if materials.shape != (view.count, view.max_shapes, 3):
            raise RuntimeError("Unexpected native collection material buffer shape.")
        materials[:] = samples[:, None, :]
        view.set_material_properties(materials, self._all_view_ids)
        # Retain only the small bucket-ID table for inspection after success.
        self.last_bucket_ids = bucket_ids.squeeze(-1)
        self.last_scene_generation = generation
