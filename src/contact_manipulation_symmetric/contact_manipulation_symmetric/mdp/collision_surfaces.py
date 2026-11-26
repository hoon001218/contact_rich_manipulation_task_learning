"""Extract body-local collision surfaces for a Warp nearest-point query."""

from __future__ import annotations

import numpy as np


def _convex_pieces(stage, prim):
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, UsdUtils

    pieces, status = [], []

    def capture(result, convexes):
        status.append(str(result))
        if result == PhysxCollisionRepresentationResult.RESULT_VALID:
            pieces.extend(np.array([(v.x, v.y, v.z) for v in hull.vertices]) for hull in convexes)

    cache = UsdUtils.StageCache.Get()
    stage_id = cache.GetId(stage)
    inserted = not stage_id.IsValid()
    if inserted:
        stage_id = cache.Insert(stage)
    try:
        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=stage_id.ToLongInt(),
            collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
            run_asynchronously=False,
            on_result=capture,
        )
    finally:
        if inserted:
            cache.Erase(stage_id)
    return pieces, status


def _cook_mesh(stage, prim, collider):
    from pxr import Usd, UsdGeom

    pieces, status = _convex_pieces(stage, prim) if prim == collider else ([], [])
    if pieces:
        return pieces, "physx_cooked"

    # Some imported schemas live on a collider Xform above the actual mesh.
    # Recreate the geometry and its composed physics attributes on a mesh prim.
    scratch = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(scratch, UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.SetStageUpAxis(scratch, UsdGeom.GetStageUpAxis(stage))
    mesh = UsdGeom.Mesh.Define(scratch, "/CollisionMesh")
    source = UsdGeom.Mesh(prim)
    mesh.CreatePointsAttr(source.GetPointsAttr().Get())
    mesh.CreateFaceVertexCountsAttr(source.GetFaceVertexCountsAttr().Get())
    mesh.CreateFaceVertexIndicesAttr(source.GetFaceVertexIndicesAttr().Get())
    mesh.CreateOrientationAttr(source.GetOrientationAttr().Get())
    holes = source.GetHoleIndicesAttr().Get()
    if holes:
        mesh.CreateHoleIndicesAttr(holes)
    for authored in (collider, prim) if collider != prim else (prim,):
        for schema in authored.GetAppliedSchemas():
            mesh.GetPrim().AddAppliedSchema(schema)
        for attr in authored.GetAuthoredAttributes():
            if attr.GetName().startswith(("physics:", "physx")) and attr.Get() is not None:
                mesh.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
    pieces, scratch_status = _convex_pieces(scratch, mesh.GetPrim())
    if not pieces:
        raise RuntimeError(f"No cooked convex surface for {prim.GetPath()}: {status + scratch_status}")
    return pieces, "physx_cooked_scratch"


def _shape_surfaces(stage, prim, collider):
    from pxr import UsdGeom
    from scipy.spatial import ConvexHull
    import trimesh

    if prim.IsA(UsdGeom.Mesh):
        pieces, method = _cook_mesh(stage, prim, collider)
        surfaces = []
        for points in pieces:
            hull = ConvexHull(points)
            faces = hull.simplices.copy()
            a, b, c = points[faces].transpose(1, 0, 2)
            inward = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0.0
            faces[inward] = faces[inward, ::-1]
            surfaces.append((points, faces))
        return surfaces, method

    axis = "Z"
    if prim.IsA(UsdGeom.Cube):
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        shape, method = trimesh.creation.box(extents=[size] * 3), "box"
    elif prim.IsA(UsdGeom.Sphere):
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
        shape, method = trimesh.creation.icosphere(subdivisions=3, radius=radius), "tessellated_sphere"
    else:
        capsule = prim.IsA(UsdGeom.Capsule)
        cfg = UsdGeom.Capsule(prim) if capsule else UsdGeom.Cylinder(prim)
        radius, height = float(cfg.GetRadiusAttr().Get()), float(cfg.GetHeightAttr().Get())
        if capsule:
            shape = trimesh.creation.capsule(radius=radius, height=height, count=[32, 64])
        else:
            shape = trimesh.creation.cylinder(radius=radius, height=height, sections=64)
        axis = str(cfg.GetAxisAttr().Get())
        method = "tessellated_capsule" if capsule else "tessellated_cylinder"
    points, faces = np.asarray(shape.vertices), np.asarray(shape.faces)
    if axis == "X":
        points, faces = points[:, [2, 1, 0]], faces[:, ::-1]
    elif axis == "Y":
        points, faces = points[:, [0, 2, 1]], faces[:, ::-1]
    return [(points, faces)], method


def collision_mesh(stage, rigid_body_prim):
    """Return body-local metric vertices, outward triangles and source details.

    Each cooked convex piece keeps its own surface, as do enabled primitives.
    Curved primitives are tessellated. Intersecting pieces retain internal faces,
    so a nearest-surface query from inside the compound can hit an internal face.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    if not rigid_body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise ValueError(f"Expected rigid body, got {rigid_body_prim.GetPath()}")
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body = np.array(cache.GetLocalToWorldTransform(rigid_body_prim)).T
    left, _, right = np.linalg.svd(body[:3, :3])
    if np.linalg.det(left @ right) < 0.0:
        left[:, -1] *= -1.0
    body[:3, :3] = left @ right
    to_body = np.linalg.inv(body)
    units = UsdGeom.GetStageMetersPerUnit(stage)
    types = (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder, UsdGeom.Capsule)
    vertices, indices, sources, offset = [], [], [], 0

    for prim in Usd.PrimRange(rigid_body_prim, Usd.TraverseInstanceProxies()):
        if not any(prim.IsA(kind) for kind in types):
            continue
        collider = prim
        while collider and not collider.HasAPI(UsdPhysics.CollisionAPI) and collider != rigid_body_prim:
            collider = collider.GetParent()
        if (not collider or not collider.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False):
            continue
        surfaces, method = _shape_surfaces(stage, prim, collider)
        transform = to_body @ np.array(cache.GetLocalToWorldTransform(prim)).T
        mirrored = np.linalg.det(transform[:3, :3]) < 0.0
        for points, faces in surfaces:
            vertices.append(units * (points @ transform[:3, :3].T + transform[:3, 3]))
            indices.append((faces[:, ::-1] if mirrored else faces) + offset)
            offset += len(points)
        sources.append({"prim": str(prim.GetPath()), "collider": str(collider.GetPath()),
                        "method": method, "pieces": len(surfaces),
                        "vertices": sum(len(points) for points, _ in surfaces),
                        "triangles": sum(len(faces) for _, faces in surfaces)})
    if not vertices:
        raise RuntimeError(f"No enabled collider surfaces for {rigid_body_prim.GetPath()}")
    points = np.concatenate(vertices).astype(np.float32)
    faces = np.concatenate(indices).astype(np.int32)
    info = {"body": str(rigid_body_prim.GetPath()), "vertices": len(points), "triangles": len(faces),
            "pieces": sum(source["pieces"] for source in sources), "colliders": sources}
    return points, faces, info
