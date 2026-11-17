"""CPU per-object material contracts, without starting Kit.

Native collection indexing and math run against a tensor-backed PhysX view.
USD authoring is checked independently through the task's real binding helper;
these tests do not claim to measure physical sliding or GPU material updates.
"""

from __future__ import annotations

import ast
from abc import ABC
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "contact_manipulation_symmetric"
LAB = PROJECT.parents[1] / "IsaacLab/source/isaaclab/isaaclab"
SCENE = PACKAGE / "config/ur5e/scene.py"


def execute_definitions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    source = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(path), "exec"), namespace)
    return namespace


class TensorMaterialView:
    def __init__(self, count=4, shapes=3):
        self.max_shapes = shapes
        self.count = count
        self.materials = torch.zeros((count, shapes, 3))
        self.get_calls = 0
        self.writes = []

    def get_material_properties(self):
        self.get_calls += 1
        return self.materials.clone()

    def set_material_properties(self, value, ids):
        self.materials[ids] = value[ids]
        self.writes.append((value.clone(), ids.clone()))


class NativeSceneFrictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Official randomize_rigid_body_material does not support collections.
        # Execute native ManagerTermBase, sampling, and collection reshaping;
        # only the simulator asset/view boundary is represented by a fixture.
        native = execute_definitions(LAB / "managers/manager_base.py",
            {"ManagerTermBase"}, dict(ABC=ABC))["ManagerTermBase"]
        sample_uniform = execute_definitions(LAB / "utils/math.py", {"sample_uniform"},
            dict(torch=torch))["sample_uniform"]
        collection_path = LAB / "assets/rigid_object_collection/rigid_object_collection.py"
        collection = next(node for node in ast.parse(collection_path.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == "RigidObjectCollection")
        collection.body = [node for node in collection.body if isinstance(node, ast.FunctionDef)
                          and node.name in ("reshape_data_to_view", "reshape_view_to_data")]
        collection.bases, collection.decorator_list = [], []
        syntax = ast.Module(body=[ast.ImportFrom(module="__future__",
            names=[ast.alias(name="annotations")], level=0), collection], type_ignores=[])
        native_namespace = dict(torch=torch)
        exec(compile(ast.fix_missing_locations(syntax), str(collection_path), "exec"), native_namespace)
        cls.Collection = native_namespace["RigidObjectCollection"]
        namespace = dict(torch=torch, math=math, ManagerTermBase=native,
            RigidObjectCollection=cls.Collection, sample_uniform=sample_uniform,
            math_utils=SimpleNamespace(sample_uniform=sample_uniform))
        cls.Term = execute_definitions(PACKAGE / "mdp/materials.py",
            {"SceneFrictionRandomization"}, namespace)["SceneFrictionRandomization"]
        cls.Native = native

    def fixture(self, *, ratio=.8, asset_name="object_collection", instances=4, objects=3, shapes=4):
        view = TensorMaterialView(count=instances * objects, shapes=shapes)
        pool = self.Collection()
        pool.root_physx_view = view
        pool.num_instances, pool.num_objects = instances, objects

        class Scene(dict):
            num_envs = instances

        scene = Scene(object_collection=pool)
        scene[asset_name] = pool
        env = SimpleNamespace(scene=scene, num_envs=instances, device="cpu",
                              _episode_reset=SimpleNamespace(scene_generation=0))
        params = dict(static_friction_range=(.25, .4),
            dynamic_friction_range=(.20, .32), restitution_range=(0., 0.),
            num_buckets=128, asset_cfg=SimpleNamespace(name=asset_name, body_ids=slice(None)),
            make_consistent=True, dynamic_friction_ratio=ratio)
        cfg = SimpleNamespace(params=params)
        return env, cfg, view

    def test_native_manager_term_uses_exact_linked_dynamic_coefficients(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        self.assertIsInstance(term, self.Native)
        buckets = term.material_buckets
        self.assertEqual(buckets.shape, (128, 3))
        self.assertEqual(buckets.device.type, "cpu")
        self.assertTrue(bool(((buckets[:, 0] >= .25) & (buckets[:, 0] <= .4)).all()))
        torch.testing.assert_close(buckets[:, 1], .8 * buckets[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(buckets[:, 2], torch.zeros(128), rtol=0, atol=0)
        self.assertEqual(view.get_calls, 0)
        term(env, None, **cfg.params)
        self.assertEqual(view.get_calls, 1)
        self.assertEqual(len(view.writes), 1)
        torch.testing.assert_close(view.materials[..., 1], .8 * view.materials[..., 0], rtol=0, atol=0)
        torch.testing.assert_close(view.writes[0][1], torch.arange(view.count))
        # Every actual shape assignment comes from the finite native buckets.
        for sample in view.materials.reshape(-1, 3):
            self.assertTrue(bool((buckets == sample).all(-1).any()))

    def test_independent_env_object_draws_use_native_object_major_order_and_shared_hull_material(self):
        env, cfg, view = self.fixture(instances=2, objects=3, shapes=5)
        term = self.Term(cfg, env)
        # Each [environment, object] gets a separate ID; all convex shapes of
        # that body must share it. PhysX rows are object-major, not env-major.
        bucket_ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
        with patch.object(torch, "randint", return_value=bucket_ids.unsqueeze(-1)) as draw:
            term(env, torch.tensor([1, 0]), **cfg.params)
        self.assertEqual(draw.call_count, 1)
        expected_ids = torch.tensor([0, 3, 1, 4, 2, 5])
        expected = term.material_buckets[expected_ids, None].expand(-1, view.max_shapes, -1)
        torch.testing.assert_close(view.materials, expected, rtol=0, atol=0)
        torch.testing.assert_close(term.last_bucket_ids, bucket_ids, rtol=0, atol=0)
        self.assertEqual(view.writes[0][1].tolist(), list(range(6)))

    def test_partial_resets_and_duplicate_calls_keep_scene_materials_unchanged(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        term(env, None, **cfg.params)
        before = view.materials.clone()
        torch_state = torch.random.get_rng_state().clone()
        partial_ids, full_ids = torch.tensor([3, 1]), torch.arange(env.num_envs)
        with patch.object(torch, "as_tensor", side_effect=AssertionError("No material ID host transfer")):
            term(env, partial_ids, **cfg.params)
            term(env, full_ids, **cfg.params)
            term(env, None, **cfg.params)
        self.assertEqual(len(view.writes), 1)
        self.assertEqual(view.get_calls, 1)
        torch.testing.assert_close(view.materials, before, rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)

    def test_new_scene_resamples_assignments_once_without_recreating_buckets(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        buckets, pointer = term.material_buckets.clone(), term.material_buckets.data_ptr()
        term(env, None, **cfg.params)
        env._episode_reset.scene_generation = 1
        term(env, torch.tensor([3, 2, 1, 0]), **cfg.params)
        self.assertEqual(len(view.writes), 2)
        self.assertEqual(term.material_buckets.data_ptr(), pointer)
        torch.testing.assert_close(term.material_buckets, buckets, rtol=0, atol=0)
        torch.testing.assert_close(view.writes[-1][1], torch.arange(view.count))
        term(env, torch.tensor([2]), **cfg.params)
        self.assertEqual(len(view.writes), 2)

    def test_new_scene_cannot_be_randomized_on_partial_reset(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        with self.assertRaises((ValueError, RuntimeError)):
            term(env, torch.tensor([2, 0]), **cfg.params)
        self.assertEqual(view.get_calls, 0)
        self.assertEqual(view.writes, [])
        # Rejected updates must leave the event retryable for the full reset.
        term(env, None, **cfg.params)
        self.assertEqual(len(view.writes), 1)

    def test_native_failure_does_not_mark_scene_as_already_randomized(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        previous_ids = term.last_bucket_ids
        native_set = view.set_material_properties
        view.set_material_properties = Mock(side_effect=RuntimeError("native material write failed"))
        with self.assertRaisesRegex(RuntimeError, "native material write failed"):
            term(env, None, **cfg.params)
        self.assertIs(getattr(term, "last_bucket_ids", None), previous_ids)
        view.set_material_properties = native_set
        term(env, None, **cfg.params)
        self.assertEqual(len(view.writes), 1)

    def test_malformed_native_buffer_or_duplicate_full_ids_cannot_commit(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        with self.assertRaises(ValueError):
            term(env, torch.tensor([0, 0, 1, 2]), **cfg.params)
        self.assertEqual(view.get_calls, 0)
        view.materials = view.materials[..., :2]
        with self.assertRaisesRegex(RuntimeError, "buffer shape"):
            term(env, None, **cfg.params)
        self.assertIsNone(term.last_bucket_ids)
        self.assertIsNone(term.last_scene_generation)
        self.assertEqual(view.writes, [])

    def test_non_collection_or_mismatched_instance_count_is_rejected(self):
        env, cfg, _ = self.fixture()
        env.scene["object_collection"] = SimpleNamespace()
        with self.assertRaisesRegex(ValueError, "RigidObjectCollection"):
            self.Term(cfg, env)
        env, cfg, _ = self.fixture()
        env.scene["object_collection"].num_instances += 1
        with self.assertRaisesRegex(ValueError, "counts must match"):
            self.Term(cfg, env)

    def test_event_cannot_run_before_a_shared_scene_is_committed(self):
        env, cfg, view = self.fixture()
        term = self.Term(cfg, env)
        del env._episode_reset.scene_generation
        with self.assertRaisesRegex(RuntimeError, "after reset_scene"):
            term(env, None, **cfg.params)
        self.assertEqual(view.get_calls, 0)
        self.assertEqual(view.writes, [])
        env._episode_reset.scene_generation = 0
        term(env, None, **cfg.params)
        self.assertEqual(len(view.writes), 1)

    def test_invalid_ratios_and_non_collection_asset_are_rejected(self):
        for ratio in (-.01, 1.01, math.nan, math.inf):
            env, cfg, _ = self.fixture(ratio=ratio)
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.Term(cfg, env)
        for name in ("robot", "shelf"):
            env, cfg, _ = self.fixture(asset_name=name)
            with self.subTest(asset=name), self.assertRaises(ValueError):
                self.Term(cfg, env)


class ContactMaterialBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = ast.parse(SCENE.read_text())

    @staticmethod
    def fixture(objects_count=2):
        collision_schema = object()

        class Relationship:
            def __init__(self, prim, name, targets):
                self.prim, self.name, self.targets = prim, name, targets

            def GetName(self):
                return self.name

            def GetTargets(self):
                return self.targets

            def GetPrim(self):
                return self.prim

        class Prim:
            def __init__(self, path, *, collision=False, enabled=True, material=False, defined=True):
                self.path, self.collision, self.enabled = path, collision, enabled
                self.material, self.defined = material, defined
                self.children, self.bindings, self.schemas = [], {"visual": "/Looks/Original"}, set()

            def GetPath(self):
                return self.path

            def IsValid(self):
                return True

            def HasAPI(self, schema):
                return (schema is collision_schema and self.collision) or schema in self.schemas

            def RemoveAPI(self, schema):
                self.schemas.discard(schema)

            def IsA(self, schema):
                return self.material and schema is Material

            def GetRelationships(self):
                return [Relationship(self, "material:binding" if purpose == "visual"
                    else "material:binding:" + purpose, [target]) for purpose, target in self.bindings.items()]

            def RemoveProperty(self, name):
                purpose = "visual" if name == "material:binding" else name.removeprefix("material:binding:")
                self.bindings.pop(purpose, None)

        nodes = {}

        def add(path, *, collision=False, enabled=True, material=False, defined=True):
            prim = nodes[path] = Prim(path, collision=collision, enabled=enabled,
                                     material=material, defined=defined)
            parent = nodes.get(path.rsplit("/", 1)[0])
            if parent is not None:
                parent.children.append(prim)
            return prim

        template = "/World/envs/env_0"
        shelf = add(template + "/Shelf")
        shelf_shape = add(shelf.path + "/support", collision=True)
        objects = tuple(f"object_{index}" for index in range(objects_count))
        object_shapes = []
        roots = [shelf]
        for name in objects:
            root = add(template + "/" + name)
            roots.append(root)
            object_shapes.append(add(root.path + "/shape", collision=True))
        arm = add(template + "/Robot")
        arm_shape = add(arm.path + "/wrist_3_collision", collision=True)
        sensor_shape = add(arm.path + "/ToolForceTorqueLink", collision=True)
        hand = add(arm.path + "/ROASRightHand")
        roots.append(hand)
        palm = add(hand.path + "/palm")
        palm_shape = add(palm.path + "/shape", collision=True)
        finger = add(hand.path + "/index")
        finger_shape = add(finger.path + "/shape", collision=True)
        disabled = add(finger.path + "/disabled", collision=True, enabled=False)
        visual = add(hand.path + "/visual")
        stage = SimpleNamespace(GetPrimAtPath=lambda path: nodes.get(path, SimpleNamespace(IsValid=lambda: False)))

        all_prims = object()

        def prim_range(prim, *predicates):
            # USD's default predicate prunes an undefined over and its subtree.
            # The all-prims predicate must be explicit to find copied materials
            # beneath an over-only _SharedMaterials ancestor.
            if not prim.defined and all_prims not in predicates:
                return
            yield prim
            for child in prim.children:
                yield from prim_range(child, *predicates)

        usd = SimpleNamespace(PrimRange=prim_range, PrimAllPrimsPredicate=all_prims,
                              TraverseInstanceProxies=lambda predicate=None: predicate)

        class CollisionAPI:
            def __new__(cls, prim):
                return SimpleNamespace(GetCollisionEnabledAttr=lambda: SimpleNamespace(Get=lambda: prim.enabled))

        # HasAPI compares the same public schema object passed by the helper.
        collision_schema = CollisionAPI
        spawns, binds = [], []

        class MaterialAPI:
            pass

        class PhysxMaterialAPI:
            pass

        class Material:
            def __init__(self, prim):
                self.prim = prim

            def GetPrim(self):
                return self.prim

            def GetPath(self):
                return self.prim.GetPath()

        class MaterialBindingAPI:
            def __init__(self, prim):
                self.prim = prim

            @classmethod
            def Apply(cls, prim):
                return cls(prim)

            def Bind(self, material, *, bindingStrength, materialPurpose):
                self.prim.bindings[materialPurpose] = str(material.GetPath())
                binds.append((self.prim.path, str(material.GetPath()), bindingStrength, materialPurpose))

        def spawn(path, cfg):
            spawns.append((path, cfg))
            prim = add(path, material=True)
            prim.schemas.update((MaterialAPI, PhysxMaterialAPI))
            return prim

        # Legacy physics materials also contain visual data. Remove their
        # physics schemas/bindings without deleting the authored visual prims.
        legacy = []
        nested_legacy = []
        for root in roots:
            material = add(root.path + "/LegacyPaint", material=True)
            material.schemas.update((MaterialAPI, PhysxMaterialAPI))
            legacy.append(material)
            shared_over = add(root.path + "/_SharedMaterials", defined=False)
            copied_material = add(shared_over.path + "/CopiedPaint", material=True)
            copied_material.schemas.update((MaterialAPI, PhysxMaterialAPI))
            copied_material.bindings["physics"] = material.path
            legacy.append(copied_material)
            nested_legacy.append(copied_material)
            for prim in prim_range(root):
                if prim.collision:
                    prim.bindings["physics"] = material.path
        arm_shape.bindings["physics"] = "/Looks/ArmPhysics"
        sensor_shape.bindings["physics"] = "/Looks/SensorPhysics"

        cfg = SimpleNamespace(**{name + "_contact_material": SimpleNamespace(func=spawn)
                                for name in ("shelf", "object", "hand")})
        namespace = dict(Usd=usd, UsdPhysics=SimpleNamespace(CollisionAPI=CollisionAPI, MaterialAPI=MaterialAPI),
            PhysxSchema=SimpleNamespace(PhysxMaterialAPI=PhysxMaterialAPI),
            UsdShade=SimpleNamespace(Material=Material, MaterialBindingAPI=MaterialBindingAPI,
                Tokens=SimpleNamespace(strongerThanDescendants="strongerThanDescendants")),
            OBJECT_NAMES=objects, sim_utils=SimpleNamespace())
        helper = execute_definitions(SCENE, {"configure_contact_materials"}, namespace)["configure_contact_materials"]
        return SimpleNamespace(helper=helper, stage=stage, template=template, cfg=cfg,
            nodes=nodes, spawns=spawns, binds=binds, hand=hand, roots=roots, legacy=legacy,
            nested_legacy=nested_legacy, usd=usd,
            expected=[shelf_shape, *object_shapes, palm_shape, finger_shape],
            excluded=[arm_shape, sensor_shape, disabled, visual])

    def test_binds_entire_hand_and_pool_without_changing_arm_sensor_or_visuals(self):
        fixture = self.fixture()
        before = {path: (prim.collision, prim.enabled, prim.bindings["visual"])
                  for path, prim in fixture.nodes.items()}
        counts = fixture.helper(fixture.stage, fixture.template, fixture.cfg)
        self.assertEqual(counts, {"shelf": 1, "object": 2, "hand": 2})
        self.assertEqual(len(fixture.spawns), 3)
        self.assertEqual({call[0] for call in fixture.binds}, {prim.path for prim in fixture.roots})
        for path, target, strength, purpose in fixture.binds:
            self.assertEqual(target, fixture.template + "/ContactMaterials/" + (
                "hand" if path.endswith("/ROASRightHand") else "shelf" if path.endswith("/Shelf") else "object"))
            self.assertEqual((strength, purpose), ("strongerThanDescendants", "physics"))
        for prim in fixture.expected:
            self.assertNotIn("physics", prim.bindings)
        self.assertEqual(fixture.excluded[0].bindings["physics"], "/Looks/ArmPhysics")
        self.assertEqual(fixture.excluded[1].bindings["physics"], "/Looks/SensorPhysics")
        for material in fixture.legacy:
            self.assertEqual(material.schemas, set())
        for path, prim in fixture.nodes.items():
            if path in before:
                self.assertEqual((prim.collision, prim.enabled, prim.bindings["visual"]), before[path])

    def test_cleanup_reaches_legacy_materials_below_undefined_shared_material_overs(self):
        fixture = self.fixture()
        for root, nested in zip(fixture.roots, fixture.nested_legacy):
            self.assertNotIn(nested, list(fixture.usd.PrimRange(root)))
            self.assertIn(nested, list(fixture.usd.PrimRange(root, fixture.usd.PrimAllPrimsPredicate)))
            self.assertTrue(nested.schemas)
            self.assertIn("physics", nested.bindings)
        fixture.helper(fixture.stage, fixture.template, fixture.cfg)
        for nested in fixture.nested_legacy:
            self.assertEqual(nested.schemas, set())
            self.assertNotIn("physics", nested.bindings)
            self.assertEqual(nested.bindings["visual"], "/Looks/Original")

    def test_three_materials_per_environment_are_shared_across_the_whole_object_pool(self):
        for pool_size in (1, 80):
            with self.subTest(pool_size=pool_size):
                fixture = self.fixture(objects_count=pool_size)
                counts = fixture.helper(fixture.stage, fixture.template, fixture.cfg)
                self.assertEqual(counts["object"], pool_size)
                self.assertEqual({path for path, _ in fixture.spawns}, {
                    fixture.template + "/ContactMaterials/shelf",
                    fixture.template + "/ContactMaterials/object",
                    fixture.template + "/ContactMaterials/hand"})
                # Each clone remaps three template-level materials to its own
                # environment. Material count grows with environments, never
                # with the number of object bodies or collision hulls.
                for clones in (1, 1024):
                    targets = set()
                    for environment in range(clones):
                        prefix = f"/World/envs/env_{environment}"
                        own_targets = {target.replace(fixture.template, prefix)
                                       for _, target, _, _ in fixture.binds}
                        self.assertEqual(own_targets, {prefix + "/ContactMaterials/" + category
                            for category in ("shelf", "object", "hand")})
                        targets.update(own_targets)
                    self.assertEqual(len(targets), 3 * clones)

    def test_missing_asset_or_no_enabled_colliders_is_an_error(self):
        fixture = self.fixture()
        del fixture.nodes[fixture.template + "/object_1"]
        with self.assertRaisesRegex(RuntimeError, "missing"):
            fixture.helper(fixture.stage, fixture.template, fixture.cfg)
        fixture = self.fixture()
        for prim in fixture.expected[-2:]:
            prim.enabled = False
        with self.assertRaisesRegex(RuntimeError, "no enabled collision"):
            fixture.helper(fixture.stage, fixture.template, fixture.cfg)

    def test_materials_are_bound_after_deinstancing_and_before_geometry_sharing(self):
        function = next(node for node in self.module.body
                        if isinstance(node, ast.FunctionDef) and node.name == "spawn_shared_geometry")
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name)
                 and node.func.id in {"_deinstance_subtree", "configure_contact_materials", "share_rigid_body_geometry"}]
        calls.sort(key=lambda node: node.lineno)
        self.assertEqual([node.func.id for node in calls],
            ["_deinstance_subtree", "configure_contact_materials", "share_rigid_body_geometry"])

    def test_material_combine_modes_preserve_object_pairs_and_enforce_requested_contacts(self):
        cfg_class = next(node for node in self.module.body
                         if isinstance(node, ast.ClassDef) and node.name == "SharedGeometryCfg")
        declared = {}
        for node in cfg_class.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.target.id.endswith("_contact_material"):
                declared[node.target.id] = {item.arg: ast.literal_eval(item.value) for item in node.value.keywords}
        self.assertEqual(declared, {
            "shelf_contact_material": {"static_friction": .4, "dynamic_friction": .32,
                                       "friction_combine_mode": "min"},
            "object_contact_material": {"static_friction": .25, "dynamic_friction": .20,
                                        "friction_combine_mode": "average"},
            "hand_contact_material": {"static_friction": .8, "dynamic_friction": .64,
                                      "friction_combine_mode": "max"},
        })
        # PhysX combine-mode priority is average < min < multiply < max.
        # Shelf coefficients cap the full randomization range; hand exceeds it.
        for obj in (.25, .325, .4):
            self.assertEqual(min(.4, obj), obj)
            self.assertAlmostEqual(min(.32, .8 * obj), .8 * obj)
            self.assertEqual(max(.8, obj), .8)
            self.assertEqual(max(.64, .8 * obj), .64)
        # Two independently randomized objects use their coefficient average.
        self.assertAlmostEqual((.25 + .4) / 2, .325)

    def test_scene_event_follows_reset_scene_and_precedes_mdp_snapshot(self):
        source = PACKAGE / "shelf_manipulation_env_cfg.py"
        cfg = next(node for node in ast.parse(source.read_text()).body
                   if isinstance(node, ast.ClassDef) and node.name == "EventCfg")
        fields = [(node.targets[0].id, node.value) for node in cfg.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)]
        names = [name for name, _ in fields]
        self.assertLess(names.index("reset_scene"), names.index("scene_friction"))
        self.assertLess(names.index("scene_friction"), names.index("reset_mdp"))
        term = dict(fields)["scene_friction"]
        values = {item.arg: item.value for item in term.keywords}
        self.assertEqual(ast.literal_eval(values["mode"]), "reset")
        params = values["params"]
        pairs = {ast.literal_eval(key): value for key, value in zip(params.keys, params.values)}
        self.assertEqual(ast.literal_eval(pairs["static_friction_range"]), (.25, .4))
        self.assertEqual(ast.literal_eval(pairs["dynamic_friction_range"]), (.20, .32))
        self.assertEqual(ast.literal_eval(pairs["dynamic_friction_ratio"]), .8)
        self.assertEqual(ast.literal_eval(pairs["restitution_range"]), (0., 0.))
        self.assertEqual(ast.literal_eval(pairs["num_buckets"]), 128)
        self.assertEqual(ast.literal_eval(pairs["asset_cfg"].args[0]), "object_collection")


if __name__ == "__main__":
    unittest.main()
