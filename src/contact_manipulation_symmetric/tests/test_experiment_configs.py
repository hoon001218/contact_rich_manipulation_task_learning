"""CPU-only experiment isolation/CLI checks; no Kit, CUDA or simulation imports.

Execute the real config class bodies with Isaac Lab's real configclass decorator.
Only external scene/controller/model configuration constructors are replaced by
plain records, so all baseline and experiment post-init mutations are exercised.
"""

import ast
from dataclasses import asdict, dataclass
import importlib.util
import math
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "contact_manipulation_symmetric"
WORKSPACE = PROJECT.parents[1]
EXPERIMENTS = {
    "relocation_distance": ("RelocationDistanceEnvCfg", "RelocationDistance"),
    "access_back_wall": ("AccessBackWallEnvCfg", "AccessBackWall"),
    "small_wrench": ("SmallWrenchEnvCfg", "SmallWrench"),
}


def execute_classes(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and (names is None or node.name in names)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)


def native_configclass():
    path = WORKSPACE / "IsaacLab/source/isaaclab/isaaclab/utils/configclass.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.ImportFrom) and node.level)]
    namespace = {"__name__": __name__}
    # Conversion helpers are not used by construction or by these comparisons.
    namespace.update(class_to_dict=asdict, update_class_from_dict=lambda *args: None)
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["configclass"]


def record(*args, **kwargs):
    if args:
        kwargs["name"] = args[0]
    return SimpleNamespace(**kwargs)


def freeze(value):
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(item) for item in value)
    if hasattr(value, "__dict__") and not callable(value):
        return {key: freeze(item) for key, item in vars(value).items() if not key.startswith("__")}
    return value


def load_configs():
    configclass = native_configclass()

    @configclass
    class ManagerBasedRLEnvCfg:
        sim: object = record(physx=record())
        viewer: object = record()

    @configclass
    class RunnerBase:
        seed: int = 42
        device: str = "cuda:0"

    @dataclass
    class ModelCfg:
        marker: str = "unchanged graph encoder"
        scene_readout: str = "full"

    @configclass
    class TermCfg:
        func: object = None
        weight: float = 0.0
        params: dict = {}
        time_out: bool = False

    mdp_names = {node.attr for path in (
        PACKAGE / "shelf_manipulation_env_cfg.py",
        *(PACKAGE / f"experiments/{variant}/env_cfg.py" for variant in EXPERIMENTS),
    ) for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "mdp"}
    mdp = SimpleNamespace(**{name: type(name, (), {}) for name in mdp_names})
    namespace = dict(
        __name__=__name__, configclass=configclass, MISSING=__import__("dataclasses").MISSING,
        ManagerBasedRLEnvCfg=ManagerBasedRLEnvCfg, ShelfManipulationEnvWindow=object,
        RewardTermCfg=TermCfg, TerminationTermCfg=TermCfg, SceneEntityCfg=record,
        EventTermCfg=record, ActionTermCfg=object, joint_vel_out_of_manual_limit=object,
        mdp=mdp, math=math, os=os, BlockerContactHold=type("BlockerContactHold", (), {}),
        ShelfManipulationSceneCfg=record,
        ObservationsCfg=lambda: record(policy=record(marker="actor"), critic=record(marker="critic")),
        EventCfg=record, CurriculumCfg=lambda: record(time_penalty=None),
        MdpCfg=record, RobotStartRules=record, InitialVisionCfg=record,
        SpawnRules=lambda **kwargs: record(**{"generation_mode": "sequential", **kwargs}),
    )
    execute_classes(PACKAGE / "shelf_manipulation_env_cfg.py", namespace,
                    {"ActionsCfg", "RewardsCfg", "TerminationsCfg", "ShelfManipulationEnvCfg"})
    namespace.update(
        UR5eContactSceneCfg=lambda **kwargs: record(
            **kwargs, robot=record(actuators={"arm": record(stiffness=100.0, damping=10.0)})),
        OperationalSpaceControllerActionCfg=record, OperationalSpaceControllerCfg=record,
        JointPositionToLimitsActionCfg=record, SensorFrameWrenchAction=type("SensorFrameWrenchAction", (), {}),
        UrdfCoupledHandAction=type("UrdfCoupledHandAction", (), {}),
        FORCE_TORQUE_BODY_NAME="ToolForceTorqueLink", configure_hand_action_limits=object,
    )
    execute_classes(PACKAGE / "config/ur5e/ur5e_env_cfg.py", namespace,
                    {"UR5eContactManipulationEnvCfg"})
    namespace.update(
        asdict=asdict, ModelCfg=ModelCfg, CRITIC_OBSERVATION_CONTRACT="unchanged critic contract",
        RslRlMLPModelCfg=RunnerBase, RslRlOnPolicyRunnerCfg=RunnerBase, RslRlPpoAlgorithmCfg=record,
    )
    execute_classes(PACKAGE / "config/ur5e/agents/rsl_rl_ppo_cfg.py", namespace)
    baseline_runner = namespace["ContactManipulationPPORunnerCfg"]
    execute_classes(PACKAGE / "experiments/common.py", namespace)
    variants = {}
    for variant, (class_name, _) in EXPERIMENTS.items():
        local = dict(namespace)
        execute_classes(PACKAGE / f"experiments/{variant}/env_cfg.py", local)
        variants[variant] = (local[class_name], local["ExperimentPPORunnerCfg"])
    return namespace["UR5eContactManipulationEnvCfg"], baseline_runner, variants, mdp


class ExperimentConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline_env, cls.baseline_runner, cls.variants, cls.mdp = load_configs()

    def test_variants_change_only_their_declared_baseline_fields(self):
        baseline = freeze(self.baseline_env())
        # The baseline terminal payment is user-selectable; the three
        # experiments below must still explicitly disable it in either case.
        terminal = baseline["rewards"].get("terminal")
        if terminal is not None:
            self.assertIs(terminal["func"], self.mdp.terminal)
        self.assertTrue(baseline["access_region_to_back_wall"])
        self.assertTrue(baseline["relocation_include_front_corridors"])
        self.assertEqual(baseline["mdp"]["finish_contact_confirmation_steps"], 3)
        relocation = baseline["rewards"]["blocker_relocation"]
        self.assertIs(relocation["func"], self.mdp.blocker_relocation_distance_improvement)
        self.assertEqual(relocation["params"], {"reference_speed": 4.0, "threshold": 0.01, "contact_exponent": 1.2})
        for name in ("clearance", "blocker_distance", "blocker_contact_hold", "blocker_normal_alignment"):
            self.assertFalse(baseline["rewards"][name]["params"].get("stop_on_success", False))
        pose_hold = baseline["rewards"]["blocker_pose_hold"]
        self.assertIs(pose_hold["func"], self.mdp.blocker_pose_hold)
        self.assertGreater(pose_hold["weight"], 0.0)
        self.assertEqual(pose_hold["params"], {"half_distance": 0.001, "success_confirmation_steps": 3})
        eef_front = baseline["rewards"]["eef_access_front"]
        self.assertIs(eef_front["func"], self.mdp.eef_access_front)
        self.assertNotIn("asset_cfg", eef_front["params"])
        self.assertEqual(eef_front["params"]["success_confirmation_steps"],
                         pose_hold["params"]["success_confirmation_steps"])
        self.assertNotIn("clearance_regression", baseline["rewards"])
        self.assertNotIn("blocker_relocation_regression", baseline["rewards"])
        # Research weights may be tuned independently; variants below must
        # inherit them without modifying the user's current baseline values.
        for variant, (cfg_class, _) in self.variants.items():
            with self.subTest(variant=variant):
                expected = freeze(self.baseline_env())
                expected["rewards"]["terminal"] = None
                expected["rewards"]["clearance"]["params"]["contact_exponent"] = 1.0
                # Existing Learning 2/3 comparisons retain their historical
                # relocation definition even after the default becomes Learning 1.
                if variant != "relocation_distance":
                    expected["relocation_include_front_corridors"] = False
                    expected["rewards"]["blocker_relocation"]["func"] = (
                        self.mdp.blocker_relocation_contact_progress)
                    expected["rewards"]["blocker_relocation"]["params"] = {"decay": 10.0}
                else:
                    expected["rewards"]["blocker_relocation"]["params"].pop("contact_exponent")
                # Learning 1/3 keep the original access depth; only the main
                # task combines metric relocation with rear-wall access now.
                expected["access_region_to_back_wall"] = variant == "access_back_wall"
                if variant == "small_wrench":
                    expected["actions"]["arm_action"]["wrench_scale"] = (
                        0.1, 0.1, 0.1, 0.001, 0.001, 0.001,
                    )
                self.assertEqual(freeze(cfg_class()), expected)
        self.assertEqual(freeze(self.baseline_env()), baseline)

    def test_instance_mutation_does_not_leak_to_baseline_or_other_variants(self):
        before = freeze(self.baseline_env())
        variants_before = {
            name: freeze(cfg_class()) for name, (cfg_class, _) in self.variants.items()
        }
        modified = self.variants["relocation_distance"][0]()
        modified.rewards.clearance.weight = 999.0
        modified.rewards.blocker_relocation.params["reference_speed"] = 999.0
        modified.relocation_include_front_corridors = False
        modified.access_region_to_back_wall = True
        modified.actions.hand_action.scale = 0.99
        modified.terminations.hand_velocity.params["max_velocity"] = 999.0
        modified.spawn_rules.max_objects = 999
        self.assertEqual(freeze(self.baseline_env()), before)
        for name, (cfg_class, _) in self.variants.items():
            with self.subTest(variant=name):
                self.assertEqual(freeze(cfg_class()), variants_before[name])

    def test_runner_ppo_models_and_observation_settings_are_identical_except_log_name(self):
        baseline = freeze(self.baseline_runner())
        self.assertEqual(baseline["actor"]["model_cfg"]["scene_readout"], "attention")
        self.assertEqual(baseline["critic"]["model_cfg"]["scene_readout"], "attention")
        for variant, (_, runner_class) in self.variants.items():
            with self.subTest(variant=variant):
                actual = freeze(runner_class())
                self.assertEqual(actual.pop("experiment_name"), f"contact_manipulation_symmetric_{variant}")
                expected = dict(baseline)
                expected.pop("experiment_name")
                self.assertEqual(actual, expected)

    def test_registration_uses_deferred_native_entry_points(self):
        registrations = []
        tree = ast.parse((PACKAGE / "config/ur5e/__init__.py").read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        scope = {"__name__": "contact_manipulation_symmetric.config.ur5e",
                 "gym": record(register=lambda **kwargs: registrations.append(kwargs))}
        exec(compile(tree, "registration", "exec"), scope)
        self.assertEqual(len(registrations), 4)
        for variant, (class_name, suffix) in EXPERIMENTS.items():
            task = f"Isaac-Contact-Manipulation-Symmetric-UR5e-{suffix}-v0"
            registered = next(item for item in registrations if item["id"] == task)
            module = f"contact_manipulation_symmetric.experiments.{variant}.env_cfg"
            self.assertEqual(registered["entry_point"], "isaaclab.envs:ManagerBasedRLEnv")
            self.assertTrue(registered["disable_env_checker"])
            self.assertEqual(registered["kwargs"], {
                "env_cfg_entry_point": f"{module}:{class_name}",
                "rsl_rl_cfg_entry_point": f"{module}:ExperimentPPORunnerCfg",
            })


class ExperimentPlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("experiment_play_helper", PACKAGE / "experiments/_play.py")
        cls.helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.helper)

    def test_all_options_and_hydra_overrides_reach_shared_playback(self):
        task = "Isaac-Contact-Manipulation-Symmetric-UR5e-SmallWrench-v0"
        options = ["--checkpoint", "/tmp/model with spaces.pt", "--video", "--video_length", "123",
                   "--seed", "7", "--num_envs", "2", "--headless", "env.scene.num_envs=2"]
        original = sys.argv
        forwarded = []
        with patch.object(self.helper.runpy, "run_path", side_effect=lambda *args, **kwargs: (
                forwarded.append((args, kwargs, sys.argv[:])))):
            self.helper.run_experiment_play(task, options)
        script = PROJECT / "scripts/play.py"
        self.assertEqual(forwarded, [((str(script),), {"run_name": "__main__"},
                                      [str(script), "--task", task, *options])])
        self.assertIs(sys.argv, original)

    def test_matching_task_spelling_is_accepted_but_another_task_is_rejected(self):
        task = "Isaac-Contact-Manipulation-Symmetric-UR5e-AccessBackWall-v0"
        with patch.object(self.helper.runpy, "run_path") as delegate:
            self.helper.run_experiment_play(task, [f"--task={task}", "--help"])
            delegate.assert_called_once()
        with patch.object(self.helper.runpy, "run_path") as delegate:
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                self.helper.run_experiment_play(task, ["--task", "wrong-task"])
            self.assertEqual(error.exception.code, 2)
            delegate.assert_not_called()

    def test_variant_entry_points_select_their_own_task_without_env_imports(self):
        for variant, (_, suffix) in EXPERIMENTS.items():
            with self.subTest(variant=variant):
                fake = record(run_experiment_play=lambda task: selected.append(task))
                selected = []
                original_path = sys.path[:]
                try:
                    with patch.dict(sys.modules, {"contact_manipulation_symmetric.experiments._play": fake}):
                        runpy.run_path(str(PACKAGE / f"experiments/{variant}/play.py"), run_name="__main__")
                finally:
                    sys.path[:] = original_path
                self.assertEqual(selected, [f"Isaac-Contact-Manipulation-Symmetric-UR5e-{suffix}-v0"])


if __name__ == "__main__":
    unittest.main()
