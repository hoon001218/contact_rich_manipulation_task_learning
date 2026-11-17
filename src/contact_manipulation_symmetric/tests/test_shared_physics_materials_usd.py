"""Real USD composition regression for the PhysX 64K material startup failure.

Runs without Kit/PhysX when standalone pxr is on PYTHONPATH. The ordinary
CPU-only test environment can skip it; this verifies actual USD references,
instance proxies and material resolution rather than a mocked binding API.
"""

from __future__ import annotations

import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None


SCENE = (Path(__file__).resolve().parents[1]
         / "contact_manipulation_symmetric/config/ur5e/scene.py")


@unittest.skipIf(Usd is None, "Requires standalone USD Python bindings; no Kit is started.")
class SharedPhysicsMaterialUsdTests(unittest.TestCase):
    def helpers(self, object_names):
        syntax = ast.parse(SCENE.read_text())
        functions = {node.name: node for node in syntax.body if isinstance(node, ast.FunctionDef)}
        needed = {"configure_contact_materials", "share_rigid_body_geometry", "_deinstance_subtree"}
        previous = set()
        while previous != needed:
            previous = needed.copy()
            needed.update(node.func.id for name in previous for node in ast.walk(functions[name])
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                          and node.func.id in functions)
        selected = [functions[name] for name in functions if name in needed]
        for node in selected:
            node.decorator_list = []
        namespace = dict(Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
                         UsdShade=UsdShade, Sdf=Sdf, Gf=Gf,
                         PhysxSchema=PhysxSchema, OBJECT_NAMES=object_names)
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(SCENE), "exec"), namespace)
        return namespace

    def fixture(self, object_count=80):
        stage = Usd.Stage.CreateInMemory()
        template = "/World/envs/env_0"
        object_names = tuple(f"Object_{index:02d}" for index in range(object_count))
        roots = [("shelf", template + "/Shelf"),
                 *(("object", template + "/" + name) for name in object_names),
                 ("hand", template + "/Robot/ROASRightHand")]
        original_material = UsdShade.Material.Define(stage, "/Looks/VisualAndLegacyPhysics")
        UsdPhysics.MaterialAPI.Apply(original_material.GetPrim()).CreateStaticFrictionAttr().Set(.5)
        shader = UsdShade.Shader.Define(stage, "/Looks/VisualAndLegacyPhysics/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shapes = {}
        for category, path in roots:
            UsdGeom.Xform.Define(stage, path)
            body_path = path if category != "hand" else path + "/Palm"
            body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mesh_schema = UsdGeom.Mesh.Define(stage, body_path + "/Mesh")
            mesh_schema.CreatePointsAttr([(-.1, -.2, 0.), (.1, -.2, 0.), (.1, .2, 0.), (-.1, .2, 0.)])
            mesh_schema.CreateFaceVertexCountsAttr([4])
            mesh_schema.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            mesh = mesh_schema.GetPrim()
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr().Set(True)
            binding = UsdShade.MaterialBindingAPI.Apply(mesh)
            binding.Bind(original_material)
            binding.Bind(original_material, materialPurpose="physics")
            local = UsdShade.Material.Define(stage, path + "/_ContactMaterial")
            UsdPhysics.MaterialAPI.Apply(local.GetPrim()).CreateStaticFrictionAttr().Set(.25)
            shapes[path] = str(mesh.GetPath())

        # Arm geometry/material behavior must survive the Hand-only cleanup.
        arm = UsdGeom.Xform.Define(stage, template + "/Robot/Arm").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(arm)
        arm_mesh = UsdGeom.Mesh.Define(stage, str(arm.GetPath()) + "/Mesh").GetPrim()
        UsdPhysics.CollisionAPI.Apply(arm_mesh).CreateCollisionEnabledAttr().Set(True)
        UsdShade.MaterialBindingAPI.Apply(arm_mesh).Bind(original_material, materialPurpose="physics")

        def spawn(path, cfg):
            material = UsdShade.Material.Define(stage, path)
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            api.CreateStaticFrictionAttr().Set(cfg.static_friction)
            api.CreateDynamicFrictionAttr().Set(cfg.dynamic_friction)

        cfg = SimpleNamespace(**{role + "_contact_material": SimpleNamespace(func=spawn,
            static_friction=static, dynamic_friction=dynamic)
            for role, static, dynamic in (("shelf", .4, .32), ("object", .25, .2), ("hand", .8, .64))})
        namespace = self.helpers(object_names)
        return stage, template, roots, shapes, cfg, namespace

    def test_material_count_stays_bounded_after_real_reference_cloning(self):
        stage, template, roots, shapes, cfg, helpers = self.fixture()
        counts = helpers["configure_contact_materials"](stage, template, cfg)
        self.assertEqual(counts, {"shelf": 1, "object": 80, "hand": 1})
        expected = {}
        for role, path in roots:
            mesh = stage.GetPrimAtPath(shapes[path])
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial("physics")
            self.assertTrue(material)
            expected.setdefault(role, str(material.GetPath()))
            self.assertEqual(str(material.GetPath()), expected[role])
            self.assertEqual(str(material.GetPath()), f"{template}/ContactMaterials/{role}")
            # The visual binding remains authored and usable after stripping
            # physics schemas from mixed visual/physical legacy materials.
            visual, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            self.assertTrue(visual)

        with tempfile.TemporaryDirectory(prefix="shared_material_usd_") as directory:
            output = str(Path(directory) / "geometry.usda")
            helpers["share_rigid_body_geometry"](stage, template, output)
            # 80 physical pool objects x 1,024 clones previously exceeded 64K
            # even before runtime randomization; clone that same topology.
            for index in range(1, 1024):
                clone = UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}").GetPrim()
                clone.GetReferences().AddInternalReference(template)
            for index in (0, 1, 1023):
                for role, path in (roots[0], roots[1], roots[-2], roots[-1]):
                    copied = shapes[path].replace(template, f"/World/envs/env_{index}")
                    prim = stage.GetPrimAtPath(copied)
                    self.assertTrue(prim.IsInstanceProxy(), copied)
                    self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
                    self.assertEqual(list(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get()), [0, 1, 2, 3])
                    self.assertEqual(len(UsdGeom.Mesh(prim).GetPointsAttr().Get()), 4)
                    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                    self.assertEqual(str(material.GetPath()),
                                     expected[role].replace(template, f"/World/envs/env_{index}"))
                    asset_root = stage.GetPrimAtPath(path.replace(template, f"/World/envs/env_{index}"))
                    self.assertFalse(any(p.HasAPI(UsdPhysics.MaterialAPI)
                                         or p.HasAPI(PhysxSchema.PhysxMaterialAPI) for p in Usd.PrimRange(
                        asset_root, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))))
            # Count composed material prims, including instance proxies, just
            # as an importer can observe them. Arm materials are intentionally
            # untouched and contribute at most one copied material per clone.
            materials = [p for p in Usd.PrimRange.Stage(
                stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate))
                         if p.HasAPI(UsdPhysics.MaterialAPI)]
            # Three contact materials per environment plus one untouched arm
            # material per environment and its original source visual material.
            self.assertEqual(len(materials), 3 * 1024 + 1024 + 1)
            contact_materials = [p for p in materials if "/ContactMaterials/" in str(p.GetPath())]
            self.assertEqual(len(contact_materials), 3 * 1024)
            self.assertEqual(len({str(stage.GetPrimAtPath(path).GetPath())
                                  for path in expected.values()}), 3)


if __name__ == "__main__":
    unittest.main()
