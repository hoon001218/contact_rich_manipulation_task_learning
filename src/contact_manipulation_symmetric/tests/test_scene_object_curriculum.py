"""CPU regressions for staged clutter, fixed tensor capacity, and full-scene play.

These checks reuse the numerical scene fixtures; they do not start Isaac Sim.
The test source was added without running it while training was active.
"""

import ast
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import test_mdp as fixtures
from contact_manipulation_symmetric.mdp import curriculums


task, observations = fixtures.task, fixtures.obs
PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT / "contact_manipulation_symmetric"
STEPS_PER_ITERATION = 36
SCHEDULE = dict(start_step=10_000 * STEPS_PER_ITERATION,
                end_step=45_000 * STEPS_PER_ITERATION, final_min=15, final_max=25)


def scheduled_range(iteration):
    return curriculums.object_count_range(iteration * STEPS_PER_ITERATION, **SCHEDULE)


def training_helpers():
    """Load only clock/config helpers, without executing AppLauncher or training."""
    path = PROJECT / "scripts/train.py"
    module = ast.parse(path.read_text())
    selected = [node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name in ("prepare_scene_object_curriculum", "checkpoint_completed_iterations")]
    namespace = dict(torch=torch, object_count_range=curriculums.object_count_range,
                     scene_object_count_curriculum=curriculums.scene_object_count_curriculum,
                     CurriculumTermCfg=lambda **kwargs: SimpleNamespace(**kwargs))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class ObjectCountScheduleTests(unittest.TestCase):
    def test_target_blocker_warmup_floor_midpoint_and_full_scene_endpoints(self):
        expected = {0: (2, 2), 9_999: (2, 2), 10_000: (2, 2),
                    27_500: (8, 13), 44_999: (14, 24), 45_000: (15, 25),
                    90_000: (15, 25)}
        for iteration, bounds in expected.items():
            with self.subTest(iteration=iteration):
                self.assertEqual(scheduled_range(iteration), bounds)

    def test_integer_schedule_is_monotone_and_does_not_depend_on_parallel_env_count(self):
        previous = (2, 2)
        for iteration in range(0, 90_001, 137):
            actual = scheduled_range(iteration)
            self.assertTrue(all(isinstance(value, int) for value in actual))
            self.assertLessEqual(2, actual[0])
            self.assertLessEqual(actual[0], actual[1])
            self.assertLessEqual(actual[1], 25)
            self.assertGreaterEqual(actual[0], previous[0])
            self.assertGreaterEqual(actual[1], previous[1])
            previous = actual
        for num_envs in (1, 128, 1024):
            env = SimpleNamespace(common_step_counter=27_500 * STEPS_PER_ITERATION,
                                  num_envs=num_envs, cfg=SimpleNamespace())
            state = curriculums.scene_object_count_curriculum(env, [0], **SCHEDULE)
            self.assertEqual(env.cfg.spawn_object_count_range, (8, 13))
            self.assertEqual((state["min_objects"], state["max_objects"]), (8, 13))
            self.assertEqual(state["progress"], 0.5)

    def test_resume_offset_applies_before_initial_reset_without_changing_environment_clocks(self):
        env = SimpleNamespace(common_step_counter=0, cfg=SimpleNamespace())
        state = curriculums.scene_object_count_curriculum(
            env, slice(None), **SCHEDULE, resume_steps=45_000 * STEPS_PER_ITERATION)
        self.assertEqual(env.cfg.spawn_object_count_range, (15, 25))
        self.assertEqual(state["progress"], 1.0)
        self.assertEqual(env.common_step_counter, 0)
        env.common_step_counter = 12_500 * STEPS_PER_ITERATION
        curriculums.scene_object_count_curriculum(
            env, [3, 1], **SCHEDULE, resume_steps=15_000 * STEPS_PER_ITERATION)
        self.assertEqual(env.cfg.spawn_object_count_range, scheduled_range(27_500))

    def test_bad_schedule_cannot_silently_select_invalid_counts(self):
        for name, value in (("start_step", True), ("start_step", -1),
                            ("end_step", 0), ("final_min", 1),
                            ("final_max", 14), ("final_max", 25.0)):
            arguments = {**SCHEDULE, name: value}
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                curriculums.object_count_range(0, **arguments)
        for step in (-1, True, 0.5):
            with self.subTest(step=step), self.assertRaises(ValueError):
                curriculums.object_count_range(step, **SCHEDULE)


class ObjectCountSceneBoundaryTests(unittest.TestCase):
    def test_callback_does_not_edit_existing_shared_layout_or_physical_capacity(self):
        sample = task.SceneSample((7, 2), 7, 2,
            {7: (-.8, .1, 1.05), 2: (-.65, .12, 1.05)}, {7: .2, 2: -.3},
            (-.78, -.5, .02, .18), (-.6, -.2))
        original = asdict(sample)
        rules = task.SpawnRules(generation_mode="sequential", min_objects=15, max_objects=25)
        runtime = SimpleNamespace(shared_sample=sample, shared_template=object(),
                                  next_scene_step=1000, scene_generation=3)
        env = SimpleNamespace(common_step_counter=45_000 * STEPS_PER_ITERATION,
                              cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=10)),
                              spawn_rules=rules, _episode_reset=runtime)
        before = dict(vars(runtime))
        curriculums.scene_object_count_curriculum(env, [0], **SCHEDULE)
        self.assertIs(runtime.shared_sample, sample)
        self.assertEqual(asdict(sample), original)
        self.assertEqual(vars(runtime), before)
        self.assertIs(env.spawn_rules, rules)
        self.assertEqual((rules.min_objects, rules.max_objects), (15, 25))
        self.assertEqual(task._active_object_capacity(env), 25)
        env.cfg.spawn_object_count_range = (2, 2)
        temporary = curriculums.curriculum_spawn_rules(env)
        self.assertIsNot(temporary, rules)
        self.assertEqual(asdict(temporary), {**asdict(rules), "min_objects": 2, "max_objects": 2})
        self.assertEqual(task._active_object_capacity(env), 25)
        self.assertEqual(env.cfg.mdp.max_objects, 10)

    def test_disabled_override_returns_original_full_rules_and_rejects_invalid_overrides(self):
        rules = task.SpawnRules(generation_mode="sequential", min_objects=15, max_objects=25)
        env = SimpleNamespace(spawn_rules=rules, cfg=SimpleNamespace(spawn_object_count_range=None))
        self.assertIs(curriculums.curriculum_spawn_rules(env), rules)
        for bounds in ((1, 2), (3, 2), (2, 26), (True, 2), (2.0, 3), (2,), "2,3"):
            env.cfg.spawn_object_count_range = bounds
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                curriculums.curriculum_spawn_rules(env)
        env.cfg.spawn_object_count_range = (2, 2)
        env.spawn_rules = replace(rules, generation_mode="clutter")
        with self.assertRaises(ValueError):
            curriculums.curriculum_spawn_rules(env)

    def test_actual_two_and_three_object_scenes_keep_roles_corridors_and_fixed_output_shapes(self):
        geometries = fixtures.EpisodeLayoutTests.catalog(copies=8)
        persistent = task.SpawnRules(generation_mode="sequential", min_objects=15, max_objects=25,
                                     x_range=(-1.2, -.5), y_range=(-.455, .455))
        lower = torch.tensor([g.lower for g in geometries], dtype=torch.float64)
        upper = torch.tensor([g.upper for g in geometries], dtype=torch.float64)
        upright = torch.zeros((len(geometries), 4), dtype=torch.float64)
        upright[:, 0] = 1.0
        parked = torch.zeros((len(geometries), 13), dtype=torch.float64)
        parked[:, 0], parked[:, 2], parked[:, 3] = 4.0, -.3, 1.0
        for count in (2, 3):
            env = SimpleNamespace(spawn_rules=persistent, cfg=SimpleNamespace(
                spawn_object_count_range=(count, count), mdp=SimpleNamespace(max_objects=10)))
            sampling_rules = curriculums.curriculum_spawn_rules(env)
            for seed in (7, 11, 18, 25):
                with self.subTest(count=count, seed=seed):
                    sample = task.sample_scene(geometries, sampling_rules, random.Random(seed))
                    self.assertEqual(len(sample.active_ids), count)
                    self.assertEqual(len(set(sample.active_ids)), count)
                    self.assertIn(sample.target_id, sample.active_ids)
                    self.assertIn(sample.blocker_id, sample.active_ids)
                    target_type = geometries[sample.target_id].asset_name
                    blocker_type = geometries[sample.blocker_id].asset_name
                    self.assertNotEqual(target_type, blocker_type)
                    types = Counter(geometries[index].asset_name for index in sample.active_ids)
                    self.assertEqual(types[target_type], 1)
                    metadata = sample.parameter_certificate
                    self.assertEqual(metadata["requested_count"], count)
                    self.assertEqual(metadata["actual_count"], count)
                    self.assertEqual(metadata["minimum_count_shortfall"], 0)
                    self.assertGreaterEqual(len(sample.reserved_regions), 1)
                    self.assertEqual(len(metadata["sequential_state"]["pocket_front_corridors"]),
                                     len(sample.reserved_regions))
                    states, active, slots = task.pack_episode_states(
                        parked, [sample], lower, upper, upright, max_objects=25)
                    self.assertEqual(states.shape, (1, len(geometries), 13))
                    self.assertEqual(slots.shape, (1, 25))
                    self.assertEqual(int(active.sum()), count)
                    self.assertEqual(int((slots >= 0).sum()), count)
                    torch.testing.assert_close(states[0, ~active[0], :7], parked[~active[0], :7])
                    geometry = torch.cat((states[..., :7], (upper-lower)[None]), dim=-1)
                    observed, valid = observations.prepare_initial_geometry(
                        geometry, active, torch.tensor([sample.target_id]),
                        torch.tensor([sample.blocker_id]), capacity=10)
                    self.assertEqual(observed.shape, (1, 10, 10))
                    self.assertEqual(valid.shape, (1, 10))
                    self.assertEqual(int(valid.sum()), count)
                    self.assertTrue(valid[0, :2].all())
                    self.assertEqual(int(observed[0, ~valid[0]].count_nonzero()), 0)
                    self.assertEqual(task._active_object_capacity(env), 25)


class ObjectCountEntrypointTests(unittest.TestCase):
    def test_train_prepares_fresh_and_resumed_scene_ranges_without_reducing_full_capacity(self):
        prepare = training_helpers()["prepare_scene_object_curriculum"]
        runner = SimpleNamespace(num_steps_per_env=STEPS_PER_ITERATION)
        for completed, expected in ((0, (2, 2)), (27_500, (8, 13)), (45_000, (15, 25))):
            cfg = SimpleNamespace(scene_object_curriculum_start_iteration=10_000,
                scene_object_curriculum_end_iteration=45_000,
                curriculum=SimpleNamespace(scene_object_count=None),
                spawn_rules=task.SpawnRules(generation_mode="sequential", min_objects=15, max_objects=25))
            with self.subTest(completed=completed):
                phase = prepare(cfg, runner, completed_iterations=completed)
                self.assertEqual(phase, (10_000, 45_000))
                self.assertEqual(cfg.spawn_object_count_range, expected)
                self.assertEqual((cfg.spawn_rules.min_objects, cfg.spawn_rules.max_objects), (15, 25))
                term = cfg.curriculum.scene_object_count
                self.assertIs(term.func, curriculums.scene_object_count_curriculum)
                self.assertEqual(term.params, {**SCHEDULE, "resume_steps": completed * STEPS_PER_ITERATION})
                # The first native curriculum/reset must preserve the phase
                # already selected before gym.make, including resumed runs.
                env = SimpleNamespace(common_step_counter=0, cfg=cfg)
                term.func(env, slice(None), **term.params)
                self.assertEqual(cfg.spawn_object_count_range, expected)

    def test_disabling_curriculum_clears_a_stale_runtime_override(self):
        prepare = training_helpers()["prepare_scene_object_curriculum"]
        cfg = SimpleNamespace(scene_object_curriculum_start_iteration=None,
            scene_object_curriculum_end_iteration=None,
            curriculum=SimpleNamespace(scene_object_count=object()), spawn_object_count_range=(2, 2))
        self.assertIsNone(prepare(cfg, SimpleNamespace()))
        self.assertIsNone(cfg.curriculum.scene_object_count)
        self.assertIsNone(cfg.spawn_object_count_range)

    def test_native_checkpoint_clock_is_read_on_cpu_and_converted_from_zero_based_label(self):
        completed = training_helpers()["checkpoint_completed_iterations"]
        with patch.object(torch, "load", return_value={"iter": 44_999}) as load:
            self.assertEqual(completed("model_44999.pt"), 45_000)
            self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertEqual(completed(None), 0)
        for invalid in (-1, True, 3.5, None):
            with self.subTest(iteration=invalid), patch.object(torch, "load", return_value={"iter": invalid}):
                with self.assertRaises(ValueError):
                    completed("bad_clock.pt")

    def test_default_schedule_and_play_full_scene_override_are_explicit(self):
        module = ast.parse((PACKAGE / "shelf_manipulation_env_cfg.py").read_text())
        cfg = next(node for node in module.body if isinstance(node, ast.ClassDef)
                   and node.name == "ShelfManipulationEnvCfg")
        fields = {node.target.id: node.value for node in cfg.body
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)}
        self.assertEqual(ast.literal_eval(fields["scene_object_curriculum_start_iteration"]), 10_000)
        self.assertEqual(ast.literal_eval(fields["scene_object_curriculum_end_iteration"]), 45_000)
        self.assertIsNone(ast.literal_eval(fields["spawn_object_count_range"]))
        module = ast.parse((PROJECT / "scripts/play.py").read_text())
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        gym_make = next(node for node in ast.walk(main) if isinstance(node, ast.Call)
                        and ast.unparse(node.func) == "gym.make")
        for attribute in ("env_cfg.curriculum.scene_object_count",
                          "env_cfg.spawn_object_count_range",
                          "env_cfg.scene_object_curriculum_start_iteration"):
            assignment = next(node for node in ast.walk(main) if isinstance(node, ast.Assign)
                              and any(ast.unparse(target) == attribute for target in node.targets))
            self.assertIsNone(ast.literal_eval(assignment.value))
            self.assertLess(assignment.lineno, gym_make.lineno)
        train = ast.parse((PROJECT / "scripts/train.py").read_text())
        main = next(node for node in train.body if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        checkpoint = next(node for node in calls if ast.unparse(node.func) == "checkpoint_completed_iterations")
        prepare = next(node for node in calls if ast.unparse(node.func) == "prepare_scene_object_curriculum")
        gym_make = next(node for node in calls if ast.unparse(node.func) == "gym.make")
        self.assertLess(checkpoint.lineno, prepare.lineno)
        self.assertLess(prepare.lineno, gym_make.lineno)


if __name__ == "__main__":
    unittest.main()
