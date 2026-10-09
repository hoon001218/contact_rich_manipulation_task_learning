"""Spawn YCB boxes with flat contact faces while retaining their scanned visuals."""

from __future__ import annotations

import math
from collections.abc import Callable

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass


YCB_BOX_ASSETS = frozenset({"ycb_003_cracker_box", "ycb_004_sugar_box"})


@clone
def spawn_flat_box_from_usd(
    prim_path: str,
    cfg: "FlatBoxUsdCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Replace the scanned mesh collider with a body-axis-aligned box at spawn.

    The outer clone decorator copies the completed collision override to every
    environment before PhysX builds its views. No source USD file is modified.
    """
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise RuntimeError(f"Flat-box asset {prim_path} must contain exactly one rigid body.")
    body = bodies[0]
    colliders = [
        prim for prim in Usd.PrimRange(body)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    ]
    if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Flat-box asset {prim_path} must contain one enabled scanned mesh collider.")
    original = colliders[0]
    # Fit in the unscaled body frame. The proxy inherits body/ancestor scale;
    # geometry inspection applies that scale and the upright rotation once.
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False, ignoreVisibility=True,
    )
    bounds = cache.ComputeUntransformedBound(body).ComputeAlignedRange()
    lower, upper = bounds.GetMin(), bounds.GetMax()
    if bounds.IsEmpty() or any(
        not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo
        for lo, hi in zip(lower, upper)
    ):
        raise RuntimeError(f"Flat-box asset {prim_path} has invalid local bounds.")
    proxy = UsdGeom.Cube.Define(root.GetStage(), body.GetPath().AppendChild("FlatBoxCollision"))
    proxy.CreateSizeAttr(1.0)
    proxy.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    proxy.AddTranslateOp().Set((lower + upper) * 0.5)
    proxy.AddScaleOp().Set(Gf.Vec3f(*(upper - lower)))
    proxy.CreatePurposeAttr(UsdGeom.Tokens.proxy)
    proxy.MakeInvisible()
    proxy_prim = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(proxy_prim).CreateCollisionEnabledAttr(True)
    PhysxSchema.PhysxCollisionAPI.Apply(proxy_prim)
    for attribute in original.GetAttributes():
        if attribute.GetName().startswith("physxCollision:") and attribute.HasAuthoredValueOpinion():
            proxy_prim.CreateAttribute(attribute.GetName(), attribute.GetTypeName()).Set(attribute.Get())
    material, _ = UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(materialPurpose="physics")
    if material and material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        UsdShade.MaterialBindingAPI.Apply(proxy_prim).Bind(material, materialPurpose="physics")
    UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
    return root


@configclass
class FlatBoxUsdCfg(sim_utils.UsdFileCfg):
    """Keep authored rigid-body properties and visuals, replacing only collision geometry."""

    func: Callable = spawn_flat_box_from_usd
