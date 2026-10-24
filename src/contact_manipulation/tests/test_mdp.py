"""CPU MDP invariants and numerical controller checks; no Isaac Sim scene required."""

import ast
from contextlib import contextmanager
import dataclasses
from itertools import product
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import torch
import warp as wp

# Use the actual checkout math implementation without starting Kit for CPU tests.
# Production imports the public Isaac Lab module normally after AppLauncher.
if "isaaclab.utils.math" not in sys.modules:
    import importlib.util
    import os
    lab_root = Path(os.environ.get("ISAACLAB_PATH", Path(__file__).resolve().parents[3] / "IsaacLab"))
    math_path = lab_root / "source/isaaclab/isaaclab/utils/math.py"
    spec = importlib.util.spec_from_file_location("isaaclab.utils.math", math_path)
    lab_math = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = lab_math
    spec.loader.exec_module(lab_math)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_manipulation.mdp import events as task, observations as obs, rewards, terminations

def scene(batch=2, objects=5):
    geometry = torch.zeros((batch, objects, 10), dtype=torch.float64)
    geometry[..., 0] = torch.arange(objects)
    geometry[..., 3] = 1.0
    geometry[..., 7:] = 0.1
    return geometry


def identity(batch, bodies=None):
    shape = (batch, 4) if bodies is None else (batch, bodies, 4)
    quaternion = torch.zeros(shape, dtype=torch.float64)
    quaternion[..., 0] = 1.0
    return quaternion


def rotation_matrix(q):
    """Independent quaternion-to-matrix formula used only by the force oracle."""
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


class InitialGeometryTests(unittest.TestCase):
    def test_roles_reorder_once_distractors_follow_proximity_and_padding_is_zero(self):
        source = scene()
        original = source.clone()
        valid = torch.tensor([[True, True, False, True, True], [True, False, True, True, True]])
        geometry, ordered_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([3, 2]), torch.tensor([0, 4]), capacity=7
        )
        torch.testing.assert_close(geometry[0, :4, 0], torch.tensor([3., 0., 1., 4.], dtype=source.dtype))
        torch.testing.assert_close(geometry[1, :4, 0], torch.tensor([2., 4., 3., 0.], dtype=source.dtype))
        self.assertTrue(ordered_valid[:, :4].all())
        self.assertFalse(ordered_valid[:, 4:].any())
        self.assertEqual(int(geometry[:, 4:].count_nonzero()), 0)
        torch.testing.assert_close(source, original)

    def test_unobserved_target_and_unassociated_blocker_leave_reserved_slots_empty(self):
        source = scene(batch=1)
        valid = torch.tensor([[True, True, False, True, False]])
        geometry, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([2]), torch.tensor([-1]), capacity=5
        )
        self.assertEqual(int(geometry[:, :2].count_nonzero()), 0)
        self.assertFalse(result_valid[:, :2].any())
        torch.testing.assert_close(geometry[0, 2:, 0], source[0, [0, 1, 3], 0])

    def test_source_can_be_larger_than_output_when_only_visible_objects_fit(self):
        source = scene(batch=1, objects=100)
        valid = torch.zeros((1, 100), dtype=torch.bool)
        valid[:, [6, 30, 90]] = True
        geometry, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([90]), torch.tensor([30]), capacity=4
        )
        torch.testing.assert_close(geometry[0, :3, 0], source[0, [90, 30, 6], 0])
        self.assertEqual(int(result_valid.sum()), 3)

    def test_invalid_role_assignments_fail_but_excess_recognized_objects_are_selected(self):
        source, valid = scene(batch=1), torch.ones((1, 5), dtype=torch.bool)
        for target, blocker in ((0, 0), (-2, 1), (5, 0)):
            with self.subTest(target=target, blocker=blocker), self.assertRaises(ValueError):
                obs.prepare_initial_geometry(source, valid, torch.tensor([target]), torch.tensor([blocker]))
        result, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([0]), torch.tensor([1]), capacity=4)
        torch.testing.assert_close(result[0, :, 0], torch.tensor([0., 1., 2., 3.], dtype=source.dtype))
        self.assertTrue(result_valid.all())

    def test_valid_geometry_rejects_nonfinite_nonpositive_extent_and_nonunit_quaternion(self):
        valid = torch.ones((1, 5), dtype=torch.bool)
        for component, value in ((0, float("nan")), (8, float("inf")), (7, 0.0), (9, -.2),
                                 (3, 0.0), (3, 2.0)):
            source = scene(batch=1)
            source[0, 2, component] = value
            with self.subTest(component=component, value=value), self.assertRaisesRegex(
                ValueError, "finite.*positive extents.*unit quaternions"
            ):
                obs.prepare_initial_geometry(source, valid, torch.tensor([0]), torch.tensor([1]))

    def test_slight_quaternion_drift_is_normalized_without_mutating_source_and_invalid_nan_is_padding(self):
        source = scene(batch=1)
        source[0, 0, 3] = 1.0005
        source[0, 2] = float("nan")
        valid = torch.tensor([[True, True, False, True, True]])
        result, result_valid = obs.prepare_initial_geometry(
            source, valid, torch.tensor([0]), torch.tensor([1]), capacity=6
        )
        self.assertEqual(float(result[0, 0, 3]), 1.0)
        self.assertEqual(float(source[0, 0, 3]), 1.0005)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(int(result[:, 4:].count_nonzero()), 0)
        self.assertFalse(result_valid[:, 4:].any())

    def test_eef_coordinates_follow_robot_measurement_without_mutating_frozen_geometry(self):
        source = scene(batch=1, objects=2)
        source[0, 0, :3] = torch.tensor([2., 2., 3.])
        original = source.clone()
        tcp = torch.tensor([[1., 2., 3., math.sqrt(.5), 0., 0., math.sqrt(.5)]], dtype=source.dtype)
        result = obs.geometry_in_eef_frame(source, torch.tensor([[True, False]]), tcp)
        torch.testing.assert_close(result[0, 0, :3], torch.tensor([0., -1., 0.], dtype=source.dtype))
        torch.testing.assert_close(result[0, 0, 3:7], torch.tensor([math.sqrt(.5), 0., 0., -math.sqrt(.5)], dtype=source.dtype))
        torch.testing.assert_close(result[0, 0, 7:], source[0, 0, 7:])
        self.assertEqual(int(result[0, 1].count_nonzero()), 0)
        torch.testing.assert_close(source, original)

    def test_quaternion_sign_equivalent_geometry_has_same_policy_encoding(self):
        source = scene(batch=1, objects=2)
        source[0, 1, 0] = 0.0
        source[0, 1, 3] = -1.0
        tcp = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]], dtype=source.dtype)
        result = obs.geometry_in_eef_frame(source, torch.ones((1, 2), dtype=torch.bool), tcp)
        torch.testing.assert_close(result[:, 0], result[:, 1])


class DynamicObservationTests(unittest.TestCase):
    def values(self, batch=3):
        tcp = torch.zeros((batch, 7))
        tcp[:, 3] = 1.0
        return dict(joint_pos=torch.zeros((batch, 12)), joint_vel=torch.zeros((batch, 12)),
                    tcp_pose_b=tcp, measured_wrench=torch.zeros((batch, 6)),
                    tactile_contacts=torch.zeros((batch, 17), dtype=torch.bool),
                    tactile_valid=torch.ones((batch, 17), dtype=torch.bool),
                    previous_action=torch.zeros((batch, 18)))

    def test_72_channels_have_agreed_units_sensor_axes_and_missing_tactile(self):
        values = self.values(batch=1)
        values["joint_pos"][:] = math.pi
        values["joint_vel"][:] = 2.0
        values["measured_wrench"][:] = torch.tensor([200., -200., 360., -8., 8., 80.])
        values["tactile_contacts"][:, [0, 5]] = True
        values["tactile_valid"][:, [5, 16]] = False
        values["previous_action"][:, 0] = 2.0
        sample = obs.dynamic_sample(**values)
        self.assertEqual(sample.shape, (1, 72))
        torch.testing.assert_close(sample[:, :24], torch.ones((1, 24)))
        torch.testing.assert_close(sample[:, 24:31], values["tcp_pose_b"])
        torch.testing.assert_close(sample[0, 31:37], torch.tensor([1., -1., 1., -1., 1., 5.]))
        torch.testing.assert_close(sample[0, [37, 42, 53]], torch.tensor([1., -1., -1.]))
        self.assertEqual(float(sample[0, 54]), 2.0)

    def test_pack_has_exactly_geometry_current_previous_with_no_role_or_mask_channels(self):
        geometry = scene(batch=3, objects=10)
        current = obs.dynamic_sample(**self.values())
        previous = current.clone()
        previous[:, 0] = -0.5
        history = torch.stack((current, previous), dim=1).double()
        packed = obs.pack_policy_observation(geometry, history)
        self.assertEqual(packed.shape, (3, 244))
        torch.testing.assert_close(packed[:, :100], geometry.flatten(1))
        torch.testing.assert_close(packed[:, 100:172], current.double())
        torch.testing.assert_close(packed[:, 172:], previous.double())

    def test_invalid_mimic_joint_or_long_history_dimensions_are_rejected(self):
        values = self.values()
        values["joint_pos"] = torch.zeros((3, 18))
        with self.assertRaisesRegex(ValueError, "12 independent"):
            obs.dynamic_sample(**values)
        with self.assertRaisesRegex(ValueError, "current and previous"):
            obs.pack_policy_observation(scene(), torch.zeros((2, 16, 72)))

    def test_hot_path_batches_1024_environments_without_tensor_scalar_reads(self):
        batch = 1024
        values = self.values(batch)
        geometry = scene(batch, 10).float()
        valid = torch.ones((batch, 10), dtype=torch.bool)
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("GPU scalar synchronization")), \
             patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("GPU condition synchronization")):
            current = obs.dynamic_sample(**values)
            transformed = obs.geometry_in_eef_frame(geometry, valid, values["tcp_pose_b"])
            packed = obs.pack_policy_observation(transformed, torch.stack((current, current), dim=1))
            measured, _ = obs.calibrate_wrench(values["measured_wrench"])
        self.assertEqual(packed.shape, (batch, 244))
        torch.testing.assert_close(measured, values["measured_wrench"])


class WrenchTests(unittest.TestCase):
    def test_saturation_resolution_and_overload_apply_to_sensor_frame_measurement(self):
        ideal = torch.tensor([[200.2, -.14, 360.0, 8.2, .011, -.014]], dtype=torch.float64)
        measured, exceeded = obs.calibrate_wrench(ideal)
        torch.testing.assert_close(measured, torch.tensor([[200., -.1, 360., 8., .01, -.015]], dtype=ideal.dtype))
        torch.testing.assert_close(exceeded, torch.tensor([[True, False, False, True, False, False]]))

def settings():
    return task.MdpCfg()


def success_steps(cfg, step_dt=.01):
    return max(1, math.ceil(cfg.success_duration / step_dt - 1.e-9))




@contextmanager
def native_motion_rewards():
    """Execute Isaac Lab's actual reward arithmetic without importing Kit."""
    import os
    lab = Path(os.environ.get("ISAACLAB_PATH", Path(__file__).resolve().parents[3] / "IsaacLab"))
    source = lab / "source/isaaclab/isaaclab/envs/mdp/rewards.py"
    names = ('action_rate_l2', 'joint_vel_l2')
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    syntax = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *functions], type_ignores=[])
    namespace = {'torch': torch, 'SceneEntityCfg': lambda name: SimpleNamespace(name=name, joint_ids=slice(None))}
    exec(compile(ast.fix_missing_locations(syntax), str(source), 'exec'), namespace)
    native_module = types.ModuleType('isaaclab.envs.mdp.rewards')
    for name in names:
        setattr(native_module, name, namespace[name])
    with patch.dict(sys.modules, {'isaaclab.envs.mdp.rewards': native_module}):
        yield native_module


def native_reward_compute():
    """Load the installed RewardManager sum without requiring a Kit application."""
    source = Path(__file__).resolve().parents[3] / 'IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py'
    definition = next(n for n in ast.parse(source.read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == 'RewardManager')
    compute = next(n for n in definition.body if isinstance(n, ast.FunctionDef) and n.name == 'compute')
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[compute], type_ignores=[])), str(source), 'exec'), namespace)
    return namespace['compute']


def native_velocity_termination():
    """Execute the checkout's official velocity predicate without importing Kit."""
    source = Path(__file__).resolve().parents[3] / 'IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'joint_vel_out_of_manual_limit')
    syntax = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              function], type_ignores=[])
    namespace = {'torch': torch, 'SceneEntityCfg': lambda name: SimpleNamespace(name=name, joint_ids=slice(None))}
    exec(compile(ast.fix_missing_locations(syntax), str(source), 'exec'), namespace)
    return namespace[function.name]




def physical_inputs(n=1, objects=3):
    center = torch.zeros(n, objects, 3, dtype=torch.float64)
    center[..., 2] = 0.1
    signs = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=center.dtype)
    corners = center[..., None, :] + 0.1 * signs
    rotation = torch.zeros(n, objects, 4, dtype=center.dtype)
    rotation[..., 0] = 1.0
    active = torch.ones(n, objects, dtype=torch.bool)
    target = torch.zeros_like(active)
    target[:, 0] = True
    return dict(center=center, corners=corners, rotation=rotation,
                velocity=torch.zeros(n, objects, 6, dtype=center.dtype), active=active, target=target,
                initial_center=center.clone(), initial_rotation=rotation.clone(),
                blocker_overlap=torch.zeros(n, dtype=center.dtype),
                sensor_wrench=torch.zeros(n, 6, dtype=center.dtype),
                stable_count=torch.zeros(n, dtype=torch.long),
                tilt_count=torch.zeros(n, objects, dtype=torch.long), step_dt=.02,
                workspace_bounds=torch.tensor([-1., 1., -1., 1.], dtype=center.dtype),
                shelf_top=0.0, ceiling=0.8)


class ContactRewardTests(unittest.TestCase):
    def test_distance_adapter_reads_current_collection_pose_with_upright_origin_correction(self):
        dtype = torch.float64
        lower = torch.tensor([[.1, -.1, .05]] * 4, dtype=dtype)
        upper = torch.tensor([[.5, .1, .15]] * 4, dtype=dtype)
        upright_rotations = Rotation.from_euler('x', [0., math.pi / 2, 0., -math.pi / 2])
        upright = torch.tensor(upright_rotations.as_quat()[:, [3, 0, 1, 2]], dtype=dtype)
        blocker = torch.tensor([3, 1])  # Actual collection IDs, distinct from the role slot 1.
        canonical = Rotation.from_euler('z', [math.pi / 2, -math.pi / 4])
        native_rotation = canonical * upright_rotations[[3, 1]]
        poses = torch.zeros((2, 4, 7), dtype=dtype)
        poses[..., 3] = 1.
        poses[torch.arange(2), blocker, :3] = torch.tensor([[1., 2., 3.], [-1., .5, 2.]], dtype=dtype)
        poses[torch.arange(2), blocker, 3:] = torch.tensor(
            native_rotation.as_quat()[:, [3, 0, 1, 2]], dtype=dtype)
        centers = poses[torch.arange(2), blocker, :3].numpy() + canonical.apply(
            ((lower[blocker] + upper[blocker]) * .5).numpy())
        points = torch.full((2, 17, 3), 99., dtype=dtype)
        points[torch.arange(2), [2, 14]] = torch.tensor(
            centers + canonical.apply([[.3, 0., 0.], [.4, 0., 0.]]), dtype=dtype)
        env = SimpleNamespace(
            step_dt=.02,
            task_scene_data=SimpleNamespace(blocker_id=blocker, lower=lower, upper=upper, upright=upright),
            scene={'object_collection': SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses))},
        )
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(2, dtype=torch.bool))) as state, \
             patch.object(rewards, 'tactile_pad_points', return_value=points):
            score = rewards.blocker_distance(env)
            torch.testing.assert_close(score, torch.tensor([.5, .25], dtype=dtype))
            state.assert_called_once_with(env)
            # The native pose moves during the episode, even though initial
            # vision stays frozen. Distance must track this actual new pose.
            poses[torch.arange(2), blocker, :3] += torch.tensor(
                canonical.apply([[.05, 0., 0.], [.05, 0., 0.]]), dtype=dtype)
            score = rewards.blocker_distance(env)
            torch.testing.assert_close(score, torch.exp(
                -math.log(2.) / .1 * torch.tensor([.05, .15], dtype=dtype)))
            state.return_value.target_protected[0] = False
            masked = rewards.blocker_distance(env)
            torch.testing.assert_close(masked, torch.stack((score.new_zeros(()), score[1])))
            state.return_value.target_protected[0] = True
            torch.testing.assert_close(rewards.blocker_distance(env), score)

    def test_distance_uses_nearest_point_and_rotated_blocker_surface(self):
        # An asymmetric box checks that its orientation, not an axis-aligned
        # bound, determines the nearest surface. All 17 pads start far away.
        points = torch.full((4, 17, 3), 10., dtype=torch.float64)
        center = torch.zeros((4, 3), dtype=points.dtype)
        rotation = identity(4)
        half = torch.tensor([[.2, .1, .05]], dtype=points.dtype).expand(4, -1)
        points[0, 3] = points.new_tensor([.2, 0., 0.])        # On surface.
        points[1, 8] = points.new_tensor([.3, 0., 0.])        # 10 cm beyond +X.
        points[2, 12] = points.new_tensor([.3, .2, .05])      # 10 cm along two axes.
        rotation[3] = points.new_tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        center[3] = points.new_tensor([1., 2., 3.])
        points[3, 16] = points.new_tensor([1., 2.3, 3.])       # Rotated box +X is world +Y.
        quality, distance = rewards.blocker_distance_quality(points, center, rotation, half)
        expected = torch.tensor([0., .1, math.sqrt(.02), .1], dtype=points.dtype)
        torch.testing.assert_close(distance, expected)
        torch.testing.assert_close(quality, torch.exp(-math.log(2.) / .1 * expected))

    def test_distance_inside_is_zero_and_invalid_geometry_cannot_receive_quality(self):
        points = torch.ones((5, 17, 3), dtype=torch.float64)
        center, rotation = torch.zeros((5, 3), dtype=points.dtype), identity(5)
        half = torch.full((5, 3), .1, dtype=points.dtype)
        points[0, 0] = 0.  # A point inside the solid OBB has distance zero.
        points[0, 1] = math.nan  # Ignore a missing pad rather than poison valid pads.
        points[1] = math.nan
        center[2, 0] = math.inf
        half[3, 2] = 0.
        rotation[4] = 0.
        quality, distance = rewards.blocker_distance_quality(points, center, rotation, half)
        torch.testing.assert_close(quality, torch.tensor([1., 0., 0., 0., 0.], dtype=points.dtype))
        self.assertEqual(float(distance[0]), 0.)
        self.assertTrue(torch.isinf(distance[1:]).all())

    def test_hold_counts_same_pad_only_and_invalid_reading_breaks_continuity(self):
        force = torch.zeros((3, 17, 3), dtype=torch.float64)
        previous = torch.zeros((3, 17), dtype=torch.bool)
        previous[0, [0, 1, 2, 3, 4]] = True
        force[0, [0, 1, 5], 0] = .02
        force[0, 2, 0] = .01  # Strictly above threshold is required.
        force[0, 3, 0] = math.nan
        force[0, 4, 0] = math.inf
        force[1, 6, 0] = .02  # First contact is unscored.
        previous[2] = True
        force[2, :, 1] = .5
        valid = torch.ones_like(previous)
        valid[2, [3, 9]] = False
        before = previous.clone()
        count, current = rewards.blocker_contact_hold(force, previous, valid=valid)
        torch.testing.assert_close(count, torch.tensor([2., 0., 15.], dtype=force.dtype))
        torch.testing.assert_close(previous, before)  # History update belongs to the caller.
        self.assertTrue(current[0, 5])
        self.assertFalse(current[0, 2:5].any())
        # Retaining only pad 1 while transferring pad 0's contact to pad 7 does
        # not count a hold at pad 7. A cleared/reset row likewise earns zero.
        force[0] = 0.
        force[0, [1, 7], 2] = .02
        current[2] = False
        next_count, _ = rewards.blocker_contact_hold(force, current, valid=valid)
        torch.testing.assert_close(next_count, torch.tensor([1., 1., 0.], dtype=force.dtype))

    def test_contact_reward_helpers_preserve_dtype_and_never_read_device_scalars(self):
        def reject(*args, **kwargs):
            raise AssertionError("Contact rewards must not read scalars or transfer tensors to CPU")

        for dtype in (torch.float32, torch.float64):
            points = torch.full((1024, 17, 3), .2, dtype=dtype)
            center = torch.zeros((1024, 3), dtype=dtype)
            rotation = identity(1024).to(dtype)
            half = torch.full_like(center, .1)
            previous = torch.ones((1024, 17), dtype=torch.bool)
            with patch.object(torch.Tensor, '__bool__', reject), \
                 patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'cpu', reject):
                quality, distance = rewards.blocker_distance_quality(points, center, rotation, half)
                count, current = rewards.blocker_contact_hold(points, previous)
            self.assertEqual(quality.dtype, dtype)
            self.assertEqual(distance.dtype, dtype)
            self.assertEqual(count.dtype, dtype)
            self.assertEqual(current.dtype, torch.bool)
            self.assertEqual(tuple(current.shape), (1024, 17))


class RewardTests(unittest.TestCase):
    def test_clearance_remains_initial_relative_and_masks_only_positive_progress(self):
        quality = torch.tensor([.8, .8, .2, .2], dtype=torch.float64)
        state = SimpleNamespace(clearance_quality=quality, initial_clearance_quality=torch.full_like(quality, .5),
                                clearance_progress_valid=torch.tensor([True, False, True, False]))
        expected = torch.tensor([.3, 0., -.3, -.3], dtype=quality.dtype)
        for dt in (.01, .02, .1):
            with patch.object(task, 'get_task_state', return_value=state):
                torch.testing.assert_close(rewards.clearance(SimpleNamespace(step_dt=dt)), expected)
        state.clearance_progress_valid.fill_(True)
        with patch.object(task, 'get_task_state', return_value=state):
            torch.testing.assert_close(rewards.clearance(SimpleNamespace(step_dt=.01)), quality - .5)

    def test_force_and_torque_are_separate_squared_step_increases_without_rate_or_saturation(self):
        force = torch.tensor([0., .01, 1., 10., 100.], dtype=torch.float64)
        torque = torch.tensor([.1, 0., .5, 2., 8.], dtype=torch.float64)
        state = SimpleNamespace(force_growth=force, torque_growth=torque)
        for dt in (.01, .02, .1):
            with patch.object(task, 'get_task_state', return_value=state):
                env = SimpleNamespace(step_dt=dt)
                torch.testing.assert_close(rewards.motion_force_growth(env), force.square())
                torch.testing.assert_close(rewards.motion_torque_growth(env), torque.square())
        self.assertGreater(float(force[-1].square()), 1.)

    def test_native_action_rate_and_joint_velocity_preserve_per_environment_raw_l2_costs(self):
        action = torch.zeros(3, 18, dtype=torch.float64)
        action[0, [0, 7, 17]] = torch.tensor([1., 2., 3.], dtype=action.dtype)
        action[1, 5] = 2.
        velocity = torch.full((3, 18), 1000., dtype=action.dtype)
        velocity[:, :12] = 0.
        velocity[0, :6] = 1.
        velocity[1, 6:12] = 2.
        env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_vel=velocity))},
                              action_manager=SimpleNamespace(action=action, prev_action=torch.zeros_like(action)))
        with native_motion_rewards() as native:
            torch.testing.assert_close(native.action_rate_l2(env), action.new_tensor([14., 4., 0.]))
            arm = SimpleNamespace(name='robot', joint_ids=slice(0, 6))
            hand = SimpleNamespace(name='robot', joint_ids=[6, 7, 8, 9, 10, 11])
            torch.testing.assert_close(native.joint_vel_l2(env, arm), action.new_tensor([6., 0., 0.]))
            torch.testing.assert_close(native.joint_vel_l2(env, hand), action.new_tensor([0., 24., 0.]))

    def test_native_motion_adapters_delegate_and_sanitize_invalid_rows_only(self):
        for dtype in (torch.float32, torch.float64):
            action = torch.arange(18, dtype=dtype).expand(5, -1).clone() / 10
            velocity = torch.ones(5, 18, dtype=dtype)
            velocity[:, 12:] = 1000.  # Unselected mimic velocities stay excluded.
            action[2, 0] = math.nan
            velocity[3, 0] = math.inf
            action[4, 0] = -math.inf
            velocity[4, 0] = math.nan
            asset_cfg = SimpleNamespace(name='robot', joint_ids=slice(0, 12))
            env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_vel=velocity))},
                                  action_manager=SimpleNamespace(action=action, prev_action=torch.zeros_like(action)))
            with native_motion_rewards() as native:
                expected_action, expected_velocity = native.action_rate_l2(env), native.joint_vel_l2(env, asset_cfg)
                with patch.object(native, 'action_rate_l2', wraps=native.action_rate_l2) as action_call, \
                     patch.object(native, 'joint_vel_l2', wraps=native.joint_vel_l2) as velocity_call:
                    actual_action = rewards.motion_action_rate(env)
                    actual_velocity = rewards.motion_joint_velocity(env, asset_cfg)
                action_call.assert_called_once_with(env)
                velocity_call.assert_called_once_with(env, asset_cfg)
            # Native arithmetic is unchanged for every finite environment row.
            torch.testing.assert_close(actual_action[:2], expected_action[:2])
            torch.testing.assert_close(actual_velocity[:3], expected_velocity[:3])
            torch.testing.assert_close(actual_action[[2, 4]], action.new_zeros(2))
            torch.testing.assert_close(actual_velocity[[3, 4]], velocity.new_zeros(2))
            torch.testing.assert_close(actual_action[3], expected_action[3])
            self.assertTrue(torch.isfinite(actual_action).all())
            self.assertTrue(torch.isfinite(actual_velocity).all())
            self.assertEqual(actual_action.dtype, dtype)

    def test_native_manager_keeps_reward_finite_when_invalid_rows_still_compute_costs(self):
        action = torch.ones(3, 18)
        velocity = torch.ones(3, 12)
        action[1, 0], velocity[2, 0] = math.nan, math.inf
        asset_cfg = SimpleNamespace(name='robot', joint_ids=slice(0, 12))
        env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_vel=velocity))},
                              action_manager=SimpleNamespace(action=action, prev_action=torch.zeros_like(action)))
        configs = [SimpleNamespace(func=rewards.motion_action_rate, weight=-.03, params={}),
                   SimpleNamespace(func=rewards.motion_joint_velocity, weight=-.03, params={'asset_cfg': asset_cfg})]
        manager = SimpleNamespace(_env=env, _term_names=['action', 'velocity'], _term_cfgs=configs,
                                  _reward_buf=torch.zeros(3), _step_reward=torch.zeros(3, 2),
                                  _episode_sums={'action': torch.zeros(3), 'velocity': torch.zeros(3)})
        with native_motion_rewards():
            actual = native_reward_compute()(manager, .01)
        torch.testing.assert_close(actual, -.03 * .01 * torch.tensor([30., 12., 18.]))
        self.assertTrue(torch.isfinite(manager._step_reward).all())

    def test_dense_terms_use_native_manager_dt_but_success_bonus_remains_250(self):
        # Execute the installed manager's actual arithmetic, not a mirrored sum.
        compute = native_reward_compute()
        state = SimpleNamespace(clearance_quality=torch.tensor([.8, .2]), initial_clearance_quality=torch.tensor([.5, .5]),
                                clearance_progress_valid=torch.ones(2, dtype=torch.bool),
                                force_growth=torch.tensor([2., 0.]), torque_growth=torch.tensor([.5, 0.]),
                                terminal_reward=torch.tensor([250., 0.]))
        for dt in (.01, .02, .1):
            env = SimpleNamespace(step_dt=dt)
            configs = [SimpleNamespace(func=fn, weight=w, params={}) for fn, w in
                       ((rewards.clearance, 8.), (rewards.motion_force_growth, -.15),
                        (rewards.motion_torque_growth, -.15), (rewards.terminal, 1.))]
            manager = SimpleNamespace(_env=env, _term_names=['clear', 'force', 'torque', 'terminal'], _term_cfgs=configs,
                                      _reward_buf=torch.zeros(2), _step_reward=torch.zeros(2, 4),
                                      _episode_sums={name: torch.zeros(2) for name in ['clear', 'force', 'torque', 'terminal']})
            with patch.object(task, 'get_task_state', return_value=state):
                actual = compute(manager, dt)
            expected = 8 * (state.clearance_quality - .5) * dt
            expected -= .15 * (state.force_growth.square() + state.torque_growth.square()) * dt
            expected += state.terminal_reward
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(manager._episode_sums['terminal'], state.terminal_reward)

    def test_growth_costs_preserve_dtype_and_do_not_synchronize_the_host(self):
        def reject(*args, **kwargs):
            raise AssertionError('Reward arithmetic must stay on the tensor device')
        for dtype in (torch.float32, torch.float64):
            force = torch.arange(1024, dtype=dtype) / 100
            state = SimpleNamespace(force_growth=force, torque_growth=force / 10)
            with patch.object(task, 'get_task_state', return_value=state), \
                 patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
                 patch.object(torch.Tensor, 'cpu', reject):
                f = rewards.motion_force_growth(SimpleNamespace(step_dt=.01))
                t = rewards.motion_torque_growth(SimpleNamespace(step_dt=.01))
            torch.testing.assert_close(f, force.square())
            torch.testing.assert_close(t, (force / 10).square())
            self.assertEqual(f.dtype, dtype)

    def test_reward_config_delegates_native_l2_and_uses_negative_cost_weights(self):
        source = Path(__file__).resolve().parents[1] / 'contact_manipulation/shelf_manipulation_env_cfg.py'
        definition = next(n for n in ast.parse(source.read_text()).body
                          if isinstance(n, ast.ClassDef) and n.name == 'RewardsCfg')
        terms = {}
        for node in definition.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fields = {item.arg: item.value for item in node.value.keywords}
                function = fields['func']
                terms[node.targets[0].id] = (function.attr if isinstance(function, ast.Attribute) else function.id,
                                           ast.literal_eval(fields['weight']))
        self.assertNotIn('rearrangement', terms)
        for name, fn in (('motion_action_rate', 'motion_action_rate'), ('motion_joint_velocity', 'motion_joint_velocity'),
                         ('motion_force_growth', 'motion_force_growth'), ('motion_torque_growth', 'motion_torque_growth'),
                         ('object_collision', 'object_collision'), ('shelf_collision', 'shelf_collision')):
            self.assertEqual(terms[name][0], fn)
            self.assertLess(terms[name][1], 0.)
        self.assertEqual(terms['clearance'], ('clearance', 8.))
        self.assertEqual(terms['blocker_distance'], ('blocker_distance', 1.))
        self.assertEqual(terms['blocker_contact_hold'], ('BlockerContactHold', .2))
        self.assertEqual(terms['object_stability'], ('object_stability', .5))
        self.assertEqual(terms['terminal'], ('terminal', 1.))
        for removed in ('reward_components', 'corner_path_increment', 'rearrangement'):
            self.assertFalse(hasattr(rewards, removed))

    def test_time_integral_is_zero_at_reset_and_independent_of_control_partition(self):
        cfg = settings()
        self.assertEqual((cfg.time_penalty_scale, cfg.time_penalty_growth_rate), (.02, .7))
        end = 8.
        expected_total = -.02 / .7 * (math.expm1(.7 * end) - .7 * end)
        for dt in (.01, .02, .1):
            elapsed = torch.arange(round(end / dt) + 1, dtype=torch.float64) * dt
            state = SimpleNamespace(cfg=cfg, elapsed_time=elapsed,
                                    clearance_quality=torch.zeros(len(elapsed), dtype=torch.float64))
            env = SimpleNamespace(step_dt=dt)
            with patch.object(task, 'get_task_state', return_value=state), \
                 patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host condition')), \
                 patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')):
                payments = rewards.time_penalty(env) * dt
            self.assertEqual(float(payments[0]), 0.)
            self.assertTrue((payments[1:] < 0).all())
            self.assertTrue((payments[2:] < payments[1:-1]).all())
            self.assertAlmostEqual(float(payments.sum()), expected_total, places=11)
            self.assertAlmostEqual(float(payments[1]),
                                   -.02 / .7 * (math.expm1(.7 * dt) - .7 * dt), places=13)


    def test_time_clock_partial_reset_restarts_only_selected_rows(self):
        state, batch, reset = fixture(4)
        state.advance(batch, 1)
        state.advance(batch, 2)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            previous = rewards.time_penalty(env).clone()
            ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
            state.reset(ids, batch, **reset, step=2)
            current = rewards.time_penalty(env)
            torch.testing.assert_close(current[ids], torch.zeros(2, dtype=current.dtype))
            torch.testing.assert_close(current[untouched], previous[untouched])
            state.advance(batch, 3)
            current = rewards.time_penalty(env)
            self.assertTrue((current[untouched] < previous[untouched]).all())
            self.assertTrue((current[ids] > previous[ids]).all())


    def test_object_stability_distance_deadband_half_cost_padding_and_dt(self):
        cfg = settings()
        distance = torch.tensor([[0., .010, .009], [.020, .005, .010],
                                 [.030, 0., 0.], [.020, .020, .020],
                                 [.020, math.nan, math.inf]], dtype=torch.float64)
        active = torch.ones_like(distance, dtype=torch.bool)
        active[-1, 1:] = False
        state = SimpleNamespace(cfg=cfg, active=active, stability_distance=distance,
                                tilt_angle=torch.zeros_like(distance))
        expected = torch.tensor([0., -.5, -.9375, -.5, -.5], dtype=distance.dtype)
        for dt in (.01, .02, .1):
            env = SimpleNamespace(step_dt=dt)
            with patch.object(task, 'get_task_state', return_value=state), \
                 patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host condition')), \
                 patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')):
                penalty = rewards.object_stability(env)
            torch.testing.assert_close(penalty, expected)
        # Repeating a safe/risky object never multiplies the worst-object cost.
        self.assertEqual(float(penalty[1]), float(penalty[3]))
        # Active invalid geometry costs the bound; padding was ignored above.
        active[-1, 1] = True
        with patch.object(task, 'get_task_state', return_value=state):
            invalid = rewards.object_stability(SimpleNamespace(step_dt=.01))
        self.assertEqual(float(invalid[-1]), -1.)


    def test_object_stability_linear_tilt_half_angle_and_same_object_combination(self):
        cfg = settings()
        angles = torch.tensor([[0., 0.], [math.pi / 6, 0.], [math.pi / 4, 0.],
                               [math.pi / 2, 0.], [math.pi / 4, 0.],
                               [0., math.pi / 4], [math.nan, 0.]], dtype=torch.float64)
        distance = torch.zeros_like(angles)
        distance[4:6, 0] = .02
        active = torch.ones_like(angles, dtype=torch.bool)
        active[-1, 0] = False
        state = SimpleNamespace(cfg=cfg, active=active, stability_distance=distance, tilt_angle=angles)
        self.assertAlmostEqual(cfg.stability_tilt_decay, math.log(2.) / (math.pi / 4))
        env = SimpleNamespace(step_dt=.01)
        with patch.object(task, 'get_task_state', return_value=state):
            penalty = rewards.object_stability(env)
        expected = torch.tensor([0., 2**(-2/3) - 1, -.5, -.75, -.75, -.5, 0.], dtype=angles.dtype)
        torch.testing.assert_close(penalty, expected)
        # Distance and tilt compound when affecting one object; risks on two
        # different objects use the worst one, rather than summing their maxima.
        self.assertLess(float(penalty[4]), float(penalty[5]))
        active[-1, 0] = True
        with patch.object(task, 'get_task_state', return_value=state):
            invalid = rewards.object_stability(env)
        self.assertEqual(float(invalid[-1]), -1.)

class SceneMotionRewardTests(unittest.TestCase):
    @staticmethod
    def motion_inputs(n=1, objects=3, dtype=torch.float64):
        data = physical_inputs(n, objects)
        center = data['center'].to(dtype)
        corners = center[..., None, :] + .5 * (data['corners'].to(dtype) - center[..., None, :])
        blocker = torch.zeros_like(data['active'])
        blocker[:, 1] = True
        return center, corners, data['velocity'].to(dtype), data['active'], blocker

    def test_object_cost_is_per_environment_and_ignores_only_blocker_and_inactive_slots(self):
        center, corners, velocity, active, blocker = self.motion_inputs(5, 4)
        velocity[0, 0, :3] = velocity.new_tensor([.1, -.2, .3])  # Protected Target is included.
        velocity[1, 2, 0] = .4  # A distractor need not be observed to contribute.
        center[2, 1], corners[2, 1], velocity[2, 1] = math.nan, math.inf, math.nan
        center[3, 3], corners[3, 3], velocity[3, 3] = math.nan, math.inf, math.nan
        active[3, 3] = False
        velocity[4, 2, 0] = math.nan  # Invalid state termination handles this body.
        velocity[4, 3, 0] = .25  # Valid motion in the same environment is preserved.
        expected = velocity.new_tensor([math.sqrt(.14), .4, 0., 0., .25])
        torch.testing.assert_close(rewards.object_motion_cost(center, corners, velocity, active, blocker), expected)
        # A change in one scene must not change any other scene's reward.
        velocity[1, 2, 0] = 2.
        actual = rewards.object_motion_cost(center, corners, velocity, active, blocker)
        torch.testing.assert_close(actual[[0, 2, 3, 4]], expected[[0, 2, 3, 4]])
        self.assertEqual(float(actual[1]), 2.)  # No tanh bound or clipping.

    def test_object_cost_does_not_round_away_small_motion(self):
        center, corners, velocity, active, blocker = self.motion_inputs(2)
        velocity[1, 2, 0] = .001
        torch.testing.assert_close(rewards.object_motion_cost(center, corners, velocity, active, blocker),
                                   velocity.new_tensor([0., .001]))

    def test_invalid_active_geometry_does_not_discard_another_valid_objects_cost(self):
        center, corners, velocity, active, blocker = self.motion_inputs(3, 4)
        velocity[:, 0, 0], velocity[:, 2, 0] = .3, .2
        center[0, 0, 0] = math.nan
        corners[1, 0, 0, 1] = math.inf
        velocity[2, 0, 4] = math.inf
        torch.testing.assert_close(rewards.object_motion_cost(center, corners, velocity, active, blocker),
                                   velocity.new_full((3,), .2))

    def test_translation_is_euclidean_and_sums_objects_without_counting_eight_corners(self):
        center, corners, velocity, active, blocker = self.motion_inputs(2)
        velocity[:, 0, :3] = velocity.new_tensor([3., 4., 0.])
        velocity[1, 2, :3] = velocity.new_tensor([0., 0., 2.])
        torch.testing.assert_close(rewards.object_motion_cost(center, corners, velocity, active, blocker),
                                   velocity.new_tensor([5., 7.]))

    def test_pure_yaw_matches_cube_corner_radius_and_quarter_turn_path(self):
        center, corners, velocity, active, blocker = self.motion_inputs(2)
        velocity[:, 0, 5] = velocity.new_tensor([1., math.pi / 2])
        expected = velocity.new_tensor([math.sqrt(.05**2 + .05**2), math.pi * math.sqrt(.005) / 2])
        cost = rewards.object_motion_cost(center, corners, velocity, active, blocker)
        torch.testing.assert_close(cost, expected)
        self.assertAlmostEqual(float(-.5 * cost[1]), -.05553603672697958)

    def test_pivot_combines_linear_and_angular_velocity_before_taking_norm(self):
        center, corners, velocity, active, blocker = self.motion_inputs()
        # Pivot about the vertical edge at (+.05,+.05): the two edge corners
        # are stationary, four travel at .1 m/s and two at sqrt(.02) m/s.
        velocity[0, 0, :3] = velocity.new_tensor([.05, -.05, 0.])
        velocity[0, 0, 5] = 1.
        expected = (4 * .1 + 2 * math.sqrt(.02)) / 8
        torch.testing.assert_close(rewards.object_motion_cost(center, corners, velocity, active, blocker),
                                   velocity.new_tensor([expected]))

    def test_world_rotation_and_translation_preserve_rigid_body_motion_cost(self):
        center, corners, velocity, active, blocker = self.motion_inputs(3, 4)
        generator = torch.Generator().manual_seed(2104)
        center += torch.randn(center.shape, dtype=center.dtype, generator=generator)
        corners = center[..., None, :] + corners - corners.mean(-2, keepdim=True)
        velocity.copy_(torch.randn(velocity.shape, dtype=velocity.dtype, generator=generator))
        original = rewards.object_motion_cost(center, corners, velocity, active, blocker)
        rotation = torch.tensor(Rotation.from_rotvec([.3, -.7, 1.2]).as_matrix(), dtype=center.dtype)
        origin = center.new_tensor([10., -8., 4.])
        transformed = rewards.object_motion_cost(center @ rotation.T + origin, corners @ rotation.T + origin,
            torch.cat((velocity[..., :3] @ rotation.T, velocity[..., 3:] @ rotation.T), -1), active, blocker)
        torch.testing.assert_close(transformed, original)

    def test_reward_manager_integrates_same_translation_path_for_fast_and_slow_motion(self):
        center, corners, velocity, active, blocker = self.motion_inputs(2)
        velocity[:, 0, 0] = velocity.new_tensor([.1, 1.])
        state = SimpleNamespace(object_motion_cost=rewards.object_motion_cost(center, corners, velocity, active, blocker))
        env = SimpleNamespace()
        compute = native_reward_compute()
        config = SimpleNamespace(func=rewards.object_collision, weight=-.5, params={})
        manager = SimpleNamespace(_env=env, _term_names=['object_collision'], _term_cfgs=[config],
            _reward_buf=torch.zeros(2, dtype=center.dtype), _step_reward=torch.zeros(2, 1, dtype=center.dtype),
            _episode_sums={'object_collision': torch.zeros(2, dtype=center.dtype)})
        with patch.object(task, 'get_task_state', return_value=state):
            step = compute(manager, .01).clone()
        torch.testing.assert_close(step, center.new_tensor([-.0005, -.005]))
        torch.testing.assert_close(step * center.new_tensor([100., 10.]), center.new_tensor([-.05, -.05]))

    def test_shelf_flag_distinguishes_translation_rotation_and_both_velocity_units(self):
        initial = torch.cat((torch.zeros(9, 3, dtype=torch.float64), identity(9)), -1)
        pose, velocity = initial.clone(), torch.zeros(9, 6, dtype=initial.dtype)
        pose[0, 0] = .004
        pose[1, 0] = .006
        angle = math.radians(1.1)
        pose[2, 3:] = pose.new_tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        pose[3, 3:] *= -1  # q and -q describe the same pose.
        velocity[4, 0] = .006
        velocity[5, 3] = .051
        pose[6, 0] = math.nan
        velocity[7, 0] = math.inf
        expected = pose.new_tensor([0., 1., 1., 0., 1., 1., 1., 1., 0.])
        torch.testing.assert_close(rewards.shelf_motion_cost(pose, velocity, initial), expected)
        relaxed = rewards.shelf_motion_cost(pose, velocity, initial, translation_threshold=.01,
                                            rotation_threshold=.1, linear_velocity_threshold=.01,
                                            angular_velocity_threshold=.1)
        torch.testing.assert_close(relaxed, pose.new_tensor([0., 0., 0., 0., 0., 0., 1., 1., 0.]))

    def test_shelf_adapter_uses_public_state_and_episode_reference(self):
        initial = torch.cat((torch.zeros(3, 3), identity(3).float()), -1)
        pose, velocity = initial.clone(), torch.zeros(3, 6)
        pose[0, 0] = .006
        velocity[1, 3] = .06
        shelf = SimpleNamespace(data=SimpleNamespace(root_link_pose_w=pose, root_com_vel_w=velocity))
        env = SimpleNamespace(scene={'shelf': shelf})
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(initial_shelf_pose=initial)):
            torch.testing.assert_close(rewards.shelf_collision(env, SimpleNamespace(name='shelf')),
                                       torch.tensor([1., 1., 0.]))

    def test_parallel_scene_costs_preserve_dtype_chunks_and_never_read_host_scalars(self):
        def reject(*args, **kwargs):
            raise AssertionError('Scene reward arithmetic must stay on the tensor device')
        generator = torch.Generator().manual_seed(4096)
        for dtype in (torch.float32, torch.float64):
            center, corners, velocity, active, blocker = self.motion_inputs(1024, 40, dtype)
            velocity.copy_(torch.randn(velocity.shape, dtype=dtype, generator=generator) * .01)
            active = torch.rand(1024, 40, generator=generator) > .2
            pose = torch.cat((torch.zeros(1024, 3, dtype=dtype), identity(1024).to(dtype)), -1)
            with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
                 patch.object(torch.Tensor, 'cpu', reject):
                actual = rewards.object_motion_cost(center, corners, velocity, active, blocker)
                shelf = rewards.shelf_motion_cost(pose, velocity[:, 0], pose)
            chunked = torch.cat([rewards.object_motion_cost(c, p, v, a, b) for c, p, v, a, b in
                zip(center.split(73), corners.split(73), velocity.split(73), active.split(73), blocker.split(73))])
            torch.testing.assert_close(actual, chunked)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(shelf.dtype, dtype)
            self.assertEqual(actual.shape, (1024,))


class SingularityRewardTests(unittest.TestCase):
    @staticmethod
    def physical_jacobian(normalized, length=.25):
        result = normalized.clone()
        result[..., :3, :] *= length
        return result

    def test_margin_matches_inverse_frobenius_oracle_and_bounds_smallest_singular_value(self):
        diagonal = torch.tensor([[1., 1., 1., 1., 1., 1.],
                                 [.02, .2, .5, 1., 1.5, 2.],
                                 [1.e-8, .2, .5, 1., 1.5, 2.]], dtype=torch.float64)
        normalized = torch.diag_embed(diagonal)
        actual = rewards.singularity_margin(self.physical_jacobian(normalized))
        expected = diagonal.reciprocal().square().sum(-1).rsqrt()
        torch.testing.assert_close(actual, expected)
        minimum = diagonal.amin(-1)
        self.assertTrue((actual <= minimum).all())
        self.assertTrue((actual >= minimum / math.sqrt(6)).all())

    def test_singular_and_nonfinite_rows_have_zero_margin_without_poisoning_healthy_rows(self):
        normalized = torch.eye(6, dtype=torch.float64).expand(6, 6, 6).clone()
        normalized[1, 5] = 0.
        normalized[2] = 0.
        normalized[3, 1, 2] = math.nan
        normalized[4, 3, 4] = math.inf
        normalized[5, 1] = normalized[5, 0]
        actual = rewards.singularity_margin(self.physical_jacobian(normalized))
        torch.testing.assert_close(actual, torch.tensor([1 / math.sqrt(6), 0., 0., 0., 0., 0.],
                                                       dtype=normalized.dtype))
        self.assertTrue(torch.isfinite(actual).all())

    def test_margin_is_invariant_under_axis_rotation_and_consistent_length_unit_conversion(self):
        generator = torch.Generator().manual_seed(173)
        normalized = torch.randn(12, 6, 6, generator=generator, dtype=torch.float64)
        normalized += 2. * torch.eye(6, dtype=normalized.dtype)
        physical = self.physical_jacobian(normalized)
        rotation = torch.tensor(Rotation.from_euler('xyz', [.7, -.4, .2]).as_matrix())
        rotated = torch.cat((rotation @ physical[:, :3], rotation @ physical[:, 3:]), dim=1)
        expected = rewards.singularity_margin(physical)
        torch.testing.assert_close(rewards.singularity_margin(rotated), expected)
        # Millimeters and meters describe the same robot when the characteristic
        # length is expressed in the same unit as the translational Jacobian.
        millimeters = physical.clone()
        millimeters[:, :3] *= 1000.
        torch.testing.assert_close(rewards.singularity_margin(millimeters, characteristic_length=250.), expected)

    def test_penalty_has_a_deadband_half_cost_point_and_bounded_invalid_fallback(self):
        margin = torch.tensor([.2, .07, .04, 0., -.1, math.nan, math.inf], dtype=torch.float64)
        actual = rewards.singularity_penalty(margin)
        maximum = math.exp(-math.log(2) * (.07 / .03)**2) - 1.
        expected = margin.new_tensor([0., 0., -.5, maximum, maximum, maximum, maximum])
        torch.testing.assert_close(actual, expected)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertTrue(((actual <= 0.) & (actual >= -1.)).all())

    def test_parallel_margin_and_penalty_preserve_dtype_and_avoid_host_reads(self):
        def reject(*args, **kwargs):
            raise AssertionError('Singularity helpers must not read device tensor values on the host')

        for dtype in (torch.float32, torch.float64):
            diagonal = torch.ones(1024, 6, dtype=dtype)
            diagonal[:, 0] = torch.linspace(0., .1, 1024, dtype=dtype)
            jacobian = self.physical_jacobian(torch.diag_embed(diagonal))
            with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
                 patch.object(torch.Tensor, 'cpu', reject), patch.object(torch.linalg, 'svdvals', reject), \
                 patch.object(torch.linalg, 'eigvalsh', reject):
                margin = rewards.singularity_margin(jacobian)
                penalty = rewards.singularity_penalty(margin)
            chunked = torch.cat([rewards.singularity_margin(chunk) for chunk in jacobian.split(73)])
            torch.testing.assert_close(margin, chunked)
            self.assertEqual(margin.dtype, dtype)
            self.assertEqual(penalty.dtype, dtype)
            self.assertTrue(torch.isfinite(penalty).all())

    def test_native_adapter_preserves_raw_singularity_and_manager_integrates_dt(self):
        cfg = task.MdpCfg()
        margin = torch.tensor([.1, .06, .04, .005, 0.], dtype=torch.float64)
        state = SimpleNamespace(cfg=cfg, singularity_margin=margin)
        expected = rewards.singularity_penalty(
            margin, threshold=cfg.motion_singularity_threshold, decay=cfg.motion_singularity_decay)
        for step_dt in (.01, .02):
            env = SimpleNamespace(step_dt=step_dt)
            with patch.object(task, 'get_task_state', return_value=state):
                actual = rewards.motion_singularity(env) * .30
            torch.testing.assert_close(actual, expected * .30)
            torch.testing.assert_close(actual[1:4], margin.new_tensor(
                [-.022237586313812906, -.15, -.2884134893819129]))


class TerminationTests(unittest.TestCase):
    def test_success_requires_half_second_of_continuous_valid_completion_at_each_control_dt(self):
        cfg = settings()
        self.assertEqual(task.MdpCfg().success_duration, .5)
        for dt in (.01, .02, .03, .1):
            data = physical_inputs()
            data['step_dt'] = dt
            paid = torch.zeros(1, dtype=torch.bool)
            threshold = success_steps(cfg, dt)
            for step in range(1, threshold + 1):
                candidate, data['stable_count'], data['tilt_count'], _, failed = terminations.task_conditions(cfg, **data)
                successful, payment, paid = terminations.terminal_transition(
                    candidate, failed, data['stable_count'], paid, cfg, step_dt=dt)
                with self.subTest(dt=dt, step=step):
                    self.assertEqual(bool(successful[0]), step == threshold)
                    self.assertEqual(float(payment[0]), cfg.success_reward if step == threshold else 0.)
            # Moving one neighbor breaks the whole hold window, even if the
            # protected target and Blocker overlap still satisfy the goal.
            data['velocity'][0, 2, 0] = cfg.stable_linear_velocity + .001
            candidate, count, _, _, _ = terminations.task_conditions(cfg, **data)
            self.assertFalse(candidate.any())
            self.assertFalse(count.any())

    def test_terminal_pays_success_once_and_enabled_failure_always_suppresses_success(self):
        cfg = task.MdpCfg()
        self.assertEqual(cfg.success_reward, 250.)
        self.assertFalse(hasattr(cfg, 'failure_penalty'))
        candidate = torch.tensor([True, True, False, False, True])
        failed = torch.tensor([False, True, True, False, False])
        stable = torch.tensor([success_steps(cfg), success_steps(cfg), 0, 0, success_steps(cfg) - 1])
        paid = torch.zeros(5, dtype=torch.bool)
        successful, terminal, paid = terminations.terminal_transition(candidate, failed, stable, paid, cfg)
        self.assertEqual(successful.tolist(), [True, False, False, False, False])
        self.assertEqual(terminal.tolist(), [250., 0., 0., 0., 0.])
        self.assertEqual(paid.tolist(), [True, True, True, False, False])
        state = SimpleNamespace(terminal_reward=torch.zeros(5))
        state.terminal_reward = terminal
        for step_dt in (.01, .02):
            env = SimpleNamespace(step_dt=step_dt)
            with patch.object(task, 'get_task_state', return_value=state):
                torch.testing.assert_close(rewards.terminal(env) * step_dt, terminal)
        _, repeated, _ = terminations.terminal_transition(candidate, failed, stable, paid, cfg)
        torch.testing.assert_close(repeated, torch.zeros_like(repeated))

    def test_target_blocker_and_all_object_velocity_gate_success(self):
        data = physical_inputs(5)
        data["center"][1, 0, 0] += 0.021
        data["rotation"][2, 0] = torch.tensor([math.cos(0.08), 0., 0., math.sin(0.08)])
        data["blocker_overlap"][3] = 1.e-4
        data["velocity"][4, 2, 0] = 0.051
        data["stable_count"][:] = 2
        candidate, stable, _, _, failed = terminations.task_conditions(settings(), **data)
        self.assertEqual(candidate.tolist(), [True, False, False, False, False])
        self.assertEqual(stable.tolist(), [3, 0, 0, 0, 0])
        self.assertFalse(failed.any())

    def test_each_failure_reason_and_invalid_state_is_independent(self):
        data, cfg = physical_inputs(4), settings()
        data['corners'][0, 1, :, 2] -= .061
        data['rotation'][1, 1] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        data['tilt_count'][1, 1] = 9
        data['initial_shelf_pose'] = torch.zeros(4, 7, dtype=torch.float64)
        data['initial_shelf_pose'][:, 3] = 1.
        data['shelf_pose'] = data['initial_shelf_pose'].clone()
        data['shelf_pose'][2, 0] = .011
        data['center'][3, 1, 1] = math.nan
        support = ObjectSupportValidityTests.inputs(4)
        support['corners_shelf'] = data['corners']
        support['center_shelf'] = data['center']
        support['support_loss_count'][0, 1] = 4
        data['support_validity'] = terminations.object_support_validity(cfg, **support)
        candidate, stable, _, reasons, failed = terminations.task_conditions(cfg, **data)
        torch.testing.assert_close(reasons[:, :4], torch.eye(4, dtype=torch.bool))
        self.assertFalse(reasons[:, 4].any())
        self.assertTrue(failed.all())
        self.assertFalse(candidate.any())
        self.assertFalse(stable.any())

    def test_singularity_is_separate_inclusive_failure_and_only_enabled_causes_gate_success(self):
        data, cfg = physical_inputs(7), settings()
        data['singularity'] = torch.tensor([.1, .005001, .005, .004, 0., math.nan, math.inf],
                                           dtype=data['center'].dtype)
        candidate, _, _, reasons, failed = terminations.task_conditions(cfg, **data)
        expected = torch.tensor([False, False, True, True, True, True, True])
        torch.testing.assert_close(candidate, ~expected)
        self.assertFalse(reasons[:, :4].any())
        torch.testing.assert_close(reasons[:, 4], expected)
        torch.testing.assert_close(failed, expected)

        # Disabling native termination preserves the diagnostic cause, while
        # removing its effect on aggregate failure and timeout bootstrapping.
        data['failure_mask'] = torch.tensor([True, True, True, True, False])
        masked_candidate, _, _, masked_reasons, masked_failure = terminations.task_conditions(cfg, **data)
        torch.testing.assert_close(masked_reasons, reasons)
        self.assertFalse(masked_failure.any())
        self.assertTrue(masked_candidate.all())

    def test_fallen_threshold_inclusive_and_duration_uses_control_time(self):
        cfg, data = settings(), physical_inputs(2)
        self.assertAlmostEqual(task.MdpCfg().max_tilt, math.radians(80))
        self.assertEqual(task.MdpCfg().tilt_duration, .2)
        angles = torch.tensor([80., 79.9], dtype=data['rotation'].dtype).deg2rad() / 2
        data['rotation'][:, 1, 0] = angles.cos()
        data['rotation'][:, 1, 1] = angles.sin()
        for step in range(1, 11):
            _, _, count, reasons, failed = terminations.task_conditions(cfg, **data)
            self.assertEqual(count[:, 1].tolist(), [step, 0])
            self.assertEqual(reasons[:, 1].tolist(), [step >= 10, False])
            self.assertEqual(failed.tolist(), [step >= 10, False])
            data['tilt_count'] = count

        # Duration is seconds, rather than a fixed ten-step threshold.
        data = physical_inputs(1)
        data['step_dt'] = .03
        data['rotation'][0, 1] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        for step in range(1, 8):
            _, _, data['tilt_count'], _, failed = terminations.task_conditions(cfg, **data)
            self.assertEqual(bool(failed[0]), step >= 7)

    def test_transient_tilt_recovery_restarts_continuous_duration(self):
        cfg, data = settings(), physical_inputs(1)
        fallen = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.], dtype=data['rotation'].dtype)
        data['rotation'][0, 1] = fallen
        for _ in range(9):
            _, _, data['tilt_count'], _, failed = terminations.task_conditions(cfg, **data)
            self.assertFalse(failed.any())
        data['rotation'][0, 1] = torch.tensor([1., 0., 0., 0.])
        _, _, data['tilt_count'], _, failed = terminations.task_conditions(cfg, **data)
        self.assertFalse(data['tilt_count'].any())
        self.assertFalse(failed.any())
        data['rotation'][0, 1] = fallen
        _, _, count, _, failed = terminations.task_conditions(cfg, **data)
        self.assertEqual(count[0, 1], 1)
        self.assertFalse(failed.any())

    def test_current_other_object_fall_blocks_success_before_failure_debounce(self):
        cfg, data = settings(), physical_inputs(1)
        data['stable_count'].fill_(success_steps(cfg))
        data['rotation'][0, 1] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        candidate, stable, count, reasons, failed = terminations.task_conditions(cfg, **data)
        self.assertFalse(candidate.any())
        self.assertFalse(stable.any())
        self.assertEqual(count[0, 1], 1)
        self.assertFalse(reasons.any())
        self.assertFalse(failed.any())
        data.update(stable_count=stable, tilt_count=count)
        data['rotation'][0, 1] = torch.tensor([1., 0., 0., 0.])
        candidate, stable, count, _, failed = terminations.task_conditions(cfg, **data)
        self.assertTrue(candidate.all())
        self.assertEqual(stable[0], 1)
        self.assertFalse(count.any())
        self.assertFalse(failed.any())

    def test_nonfinite_target_pose_or_overlap_cannot_report_task_success(self):
        data = physical_inputs(6)
        data['center'][0, 0, 0] = math.nan
        data['initial_center'][1, 0, 1] = math.inf
        data['rotation'][2, 0] = 0.
        data['initial_rotation'][3, 0] = math.nan
        data['blocker_overlap'][4] = math.nan
        data['blocker_overlap'][5] = -1.
        candidate, stable, _, _, _ = terminations.task_conditions(settings(), **data)
        self.assertFalse(candidate.any())
        self.assertFalse(stable.any())

    def test_finite_large_ft_load_does_not_restore_removed_hard_load_termination(self):
        cfg, data = settings(), physical_inputs(2)
        data['stable_count'].fill_(success_steps(cfg, data['step_dt']) - 1)
        data['sensor_wrench'][0, 0] = 1000.
        data['sensor_wrench'][1, 3] = 1000.
        candidate, stable, _, reasons, failed = terminations.task_conditions(cfg, **data)
        self.assertTrue(candidate.all())
        self.assertFalse(reasons.any())
        self.assertFalse(failed.any())
        successful, payment, _ = terminations.terminal_transition(
            candidate, failed, stable, torch.zeros(2, dtype=torch.bool), cfg, step_dt=data['step_dt'])
        self.assertTrue(successful.all())
        torch.testing.assert_close(payment, torch.full((2,), 250.))
        self.assertNotIn('hard_load_or_overrange', terminations.FAILURE_REASON_NAMES)

    def test_enabled_native_failure_configuration_handles_alias_aggregate_and_timeouts(self):
        term = lambda fn, **params: SimpleNamespace(func=fn, params=params, time_out=False)
        config = SimpleNamespace(
            completion=term(terminations.success),
            shelf=term(terminations.failure, reason='shelf_displacement'),
            disabled=None,
            load_as_timeout=SimpleNamespace(func=terminations.failure,
                                           params={'reason': 'hard_load_or_overrange'}, time_out=True),
        )
        mask, enabled = terminations.configured_terminations(config, device='cpu')
        self.assertEqual(mask.tolist(), [False, False, True, False, False])
        self.assertTrue(enabled)
        mask, enabled = terminations.configured_terminations({'all_failures': term(terminations.failure)}, device='cpu')
        self.assertTrue(mask.all())
        self.assertFalse(enabled)

    def test_disabled_success_does_not_pay_or_mark_terminal_but_keeps_task_predicate(self):
        cfg = task.MdpCfg()
        successful, payment, paid = terminations.terminal_transition(
            torch.tensor([True]), torch.tensor([False]), torch.tensor([success_steps(cfg)]),
            torch.tensor([False]), cfg, success_enabled=False)
        self.assertTrue(successful.all())
        self.assertFalse(payment.any())
        self.assertFalse(paid.any())

    def test_alternating_fallen_objects_never_share_a_duration_counter(self):
        cfg, data = settings(), physical_inputs(1)
        for step in range(24):
            data['rotation'].zero_()
            data['rotation'][..., 0] = 1.
            index = 1 + step % 2
            data['rotation'][0, index] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
            _, _, data['tilt_count'], _, failed = terminations.task_conditions(cfg, **data)
            self.assertFalse(failed.any())
            self.assertEqual(data['tilt_count'][0, index], 1)
            self.assertEqual(data['tilt_count'][0, 3 - index], 0)

    def test_workspace_lift_and_ceiling_crossing_never_count_as_completion(self):
        data = physical_inputs(3)
        data['center'][0, 1, 0] += 2.
        data['corners'][0, 1, :, 0] += 2.
        data['center'][1, 1, 2] += .05
        data['corners'][1, 1, :, 2] += .05
        data['center'][2, 1, 2] += 1.
        data['corners'][2, 1, :, 2] += 1.
        data['velocity'][:, 1, 0] = .5
        candidate, stable, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertFalse(candidate.any())
        self.assertFalse(stable.any())
        # Hard failure uses object-local timers; these direct one-step inputs
        # have not yet completed either ejection/off-support grace period.
        self.assertFalse(reasons.any())
        self.assertFalse(failed.any())

    def test_nonfinite_initial_geometry_wrench_velocity_and_robot_fail(self):
        data = physical_inputs(5)
        data["initial_rotation"][0, 1] = math.nan
        data["sensor_wrench"][1, 0] = math.nan
        data["velocity"][2, 1, 0] = math.inf
        data["state_valid"] = torch.tensor([True, True, True, False, True])
        data["rotation"][4, 1] = 0.0  # Invalid quaternion must not count as upright.
        _, _, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(reasons[:, 3].all())
        self.assertTrue(failed.all())

    def test_inactive_nan_slots_do_not_poison_conditions(self):
        data = physical_inputs(2)
        data["active"][:, 2] = False
        for key in ("center", "corners", "rotation", "velocity", "initial_center", "initial_rotation"):
            data[key][:, 2] = math.nan
        data["tilt_count"][:, 2] = 9
        candidate, stable, tilt, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(candidate.all())
        self.assertFalse(reasons.any())
        self.assertFalse(failed.any())
        self.assertFalse(tilt.any())
        torch.testing.assert_close(stable, torch.ones_like(stable))

    def test_shelf_displacement_rotation_sign_and_invalid_pose(self):
        data = physical_inputs(8)
        initial = torch.zeros(8, 7, dtype=torch.float64)
        initial[:, :3] = torch.arange(8, dtype=torch.float64)[:, None] * 2.5
        initial[:, 3] = 1.
        pose = initial.clone()
        pose[0, 0] += .009  # Below translation tolerance.
        pose[1, 0] += .011
        for row, angle in ((2, 1.9), (3, 2.1)):
            half = math.radians(angle) / 2
            pose[row, 3:] = torch.tensor([math.cos(half), 0., 0., math.sin(half)], dtype=pose.dtype)
        pose[4, 3:] *= -1  # Quaternion sign must not trigger a rotation.
        pose[5, 3:] *= 2   # Normalize before comparing.
        pose[6, 0] = math.nan
        pose[7, 3:] = 0
        data.update(shelf_pose=pose, initial_shelf_pose=initial)
        _, _, _, reasons, failed = terminations.task_conditions(settings(), **data)
        self.assertEqual(failed.tolist(), [False, True, False, True, False, False, True, True])
        self.assertEqual(reasons[:, 2].tolist(), [False, True, False, True, False, False, False, True])
        self.assertTrue(reasons[6:, 3].all())



    def test_per_environment_workspace_and_shelf_offsets(self):
        data = physical_inputs(2)
        data["corners"][1, :, :, 2] += 2.0
        data["shelf_top"] = torch.tensor([0.0, 2.0], dtype=torch.float64)
        data["ceiling"] = torch.tensor([0.8, 2.8], dtype=torch.float64)
        data["workspace_bounds"] = data["workspace_bounds"].repeat(2, 1)
        candidate, _, _, _, failed = terminations.task_conditions(settings(), **data)
        self.assertTrue(candidate.all())
        self.assertFalse(failed.any())

    def test_large_parallel_conditions_equal_independent_chunks(self):
        torch.manual_seed(19)
        data = physical_inputs(2048, objects=10)
        data["velocity"].normal_(mean=0.0, std=0.006)
        data["blocker_overlap"].uniform_(0.0, 2.e-5)
        data["stable_count"].random_(0, 8)
        data["tilt_count"].random_(0, 12)
        angles = torch.empty(2048, 10, dtype=torch.float64).uniform_(0., math.pi)
        data["rotation"][..., 0] = torch.cos(angles / 2)
        data["rotation"][..., 1] = torch.sin(angles / 2)
        data["sensor_wrench"].normal_(mean=0.0, std=1.0)
        full = terminations.task_conditions(settings(), **data)
        chunks = [[], [], [], [], []]
        for start in range(0, 2048, 257):
            chunk = {k: v[start:start + 257] if isinstance(v, torch.Tensor) and k != "workspace_bounds" else v
                     for k, v in data.items()}
            partial = terminations.task_conditions(settings(), **chunk)
            for output, value in zip(chunks, partial):
                output.append(value)
        for expected, parts in zip(full, chunks):
            torch.testing.assert_close(expected, torch.cat(parts))


class ObjectSupportValidityTests(unittest.TestCase):
    @staticmethod
    def inputs(n=1, objects=3):
        values = physical_inputs(n, objects)
        return dict(center_shelf=values['center'], corners_shelf=values['corners'],
                    velocity=values['velocity'], active=values['active'],
                    upright_z=torch.ones(n, objects, dtype=values['center'].dtype),
                    support_bounds=values['workspace_bounds'].expand(n, -1),
                    support_height=torch.zeros(n, dtype=values['center'].dtype),
                    support_loss_count=torch.zeros(n, objects, dtype=torch.long), step_dt=.02)

    @staticmethod
    def advance(values):
        result = terminations.object_support_validity(settings(), **values)
        values.update(support_loss_count=result.support_loss_count)
        return result

    def test_moving_supported_push_and_small_pivot_can_progress_but_completion_waits(self):
        values = self.inputs(4)
        values['velocity'][0, 1, 0] = .5
        values['velocity'][1, 2, 3] = .31
        # A 7 mm support gap is permitted while manipulating but not settled.
        values['center_shelf'][2, 1, 2] += .007
        values['corners_shelf'][2, 1, :, 2] += .007
        values['upright_z'][3, 1] = math.cos(math.radians(35))
        result = self.advance(values)
        self.assertTrue(result.progress_valid.all())
        self.assertEqual(result.completion_valid.tolist(), [False, False, False, True])
        self.assertFalse(result.lost.any())

    def test_single_support_distance_unifies_ejection_flight_and_drop_with_one_timer(self):
        values = self.inputs(5)
        values['center_shelf'][0, 1, 0] += 2.
        values['corners_shelf'][0, 1, :, 0] += 2.
        values['corners_shelf'][1, 2, :, 2] += .04
        values['center_shelf'][1, 2, 2] += .04
        values['corners_shelf'][2, 2, :, 2] -= .04
        values['center_shelf'][2, 2, 2] -= .04
        # Each axis is below 3 cm, but their Euclidean correction exceeds 3 cm.
        values['center_shelf'][3, 1, 0] = 1.024
        values['corners_shelf'][3, 1, :, 0] += 1.024
        values['corners_shelf'][3, 1, :, 2] += .024
        values['center_shelf'][3, 1, 2] += .024
        values['corners_shelf'][4, 1, :, 2] += .030
        for step in range(1, 6):
            result = self.advance(values)
            self.assertFalse(result.progress_valid.any())
            self.assertFalse(result.completion_valid.any())
            self.assertEqual(result.lost.tolist(), [step >= 5] * 5)
        self.assertAlmostEqual(float(result.support_distance[3, 1]), math.hypot(.024, .024))
        self.assertAlmostEqual(float(result.support_distance[1, 2]), .04)
        self.assertAlmostEqual(float(result.support_distance[2, 2]), .04)

    def test_stability_uses_outer_obb_corners_and_collision_bottom_without_changing_failure_distance(self):
        values = self.inputs(4)
        # Both centers remain inside the finite panel, but OBB corners extend
        # beyond its edge. The new reward sees this before support-loss failure.
        for row, delta in ((0, (.95, 0., 0.)), (1, (.95, .96, .02))):
            offset = values['center_shelf'].new_tensor(delta)
            values['center_shelf'][row, 1] += offset
            values['corners_shelf'][row, 1] += offset
        values['corners_shelf'][2, 1, :, 2] -= .025
        # Exact collision bottom overrides the conservative OBB bottom.
        bottom = values['corners_shelf'].amin(-2)[..., 2]
        bottom[3, 1] = .011
        values['support_bottom'] = bottom
        for _ in range(8):
            result = self.advance(values)
            self.assertFalse(result.lost.any())
        expected_warning = torch.tensor([.05, math.sqrt(.05**2 + .06**2 + .02**2), .025, .011],
                                        dtype=bottom.dtype)
        expected_failure = torch.tensor([0., .02, .025, .011], dtype=bottom.dtype)
        torch.testing.assert_close(result.stability_distance[:, 1], expected_warning)
        torch.testing.assert_close(result.support_distance[:, 1], expected_failure)
        self.assertFalse(result.support_loss_count.any())

    def test_support_tilt_is_shelf_relative_upright_angle_with_clamp_and_padding(self):
        values = self.inputs(4)
        values['upright_z'][:, 1] = values['upright_z'].new_tensor(
            [math.cos(math.pi / 4), -1., 1. + 1.e-14, math.nan])
        values['active'][3, 1] = False
        result = self.advance(values)
        torch.testing.assert_close(result.tilt_angle[:3, 1],
                                   values['upright_z'].new_tensor([math.pi / 4, math.pi, 0.]))
        self.assertEqual(float(result.tilt_angle[3, 1]), 0.)

    def test_different_airborne_objects_do_not_share_the_same_hold_counter(self):
        values = self.inputs()
        original_center, original_corners = values['center_shelf'].clone(), values['corners_shelf'].clone()
        for step in range(15):
            values['center_shelf'].copy_(original_center)
            values['corners_shelf'].copy_(original_corners)
            index = 1 + step % 2
            values['center_shelf'][0, index, 2] += .04
            values['corners_shelf'][0, index, :, 2] += .04
            result = self.advance(values)
            self.assertFalse(result.lost.any())
            self.assertEqual(int(result.support_loss_count[0, index]), 1)
            self.assertEqual(int(result.support_loss_count[0, 3 - index]), 0)

    def test_recovery_and_padding_clear_only_the_corresponding_object_counters(self):
        values = self.inputs()
        values['corners_shelf'][0, 1:, :, 2] += .04
        values['center_shelf'][0, 1:, 2] += .04
        for _ in range(3):
            self.advance(values)
        values['corners_shelf'][0, 1, :, 2] -= .04
        values['center_shelf'][0, 1, 2] -= .04
        values['active'][0, 2] = False
        for name in ('center_shelf', 'corners_shelf', 'velocity', 'upright_z'):
            values[name][0, 2] = math.nan
        result = self.advance(values)
        self.assertTrue(result.progress_valid.all())
        self.assertTrue(result.completion_valid.all())
        self.assertFalse(result.support_loss_count.any())

    def test_signed_support_gap_rejects_below_board_without_treating_obb_as_mesh_contact(self):
        values = self.inputs(3)
        values['corners_shelf'][0, 1, :, 2] -= .02
        # A tilted/concave mesh may have OBB corners below the panel while its
        # collision geometry rests correctly. The measured mesh minimum wins.
        values['corners_shelf'][1, 1, :, 2] -= .05
        values['support_bottom'] = torch.zeros(3, 3, dtype=values['center_shelf'].dtype)
        values['support_bottom'][0, 1] = -.02
        values['support_bottom'][2, 1] = .011
        result = self.advance(values)
        self.assertEqual(result.progress_valid.tolist(), [False, True, False])
        self.assertEqual(result.completion_valid.tolist(), [False, True, False])

    def test_fallen_neighbor_blocks_progress_and_completion_before_tilt_failure(self):
        values = self.inputs()
        values['upright_z'][0, 2] = 0.
        result = self.advance(values)
        self.assertFalse(result.progress_valid.any())
        self.assertFalse(result.completion_valid.any())
        self.assertFalse(result.lost.any())

    def test_1024_environment_support_update_has_no_tensor_host_reads(self):
        values = self.inputs(1024, 10)
        values['active'][::2, 9] = False
        values['center_shelf'][::2, 9] = math.nan
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('GPU scalar synchronization')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('GPU conditional synchronization')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Unexpected host transfer')):
            result = self.advance(values)
        self.assertEqual(result.support_loss_count.shape, (1024, 10))
        self.assertTrue(result.completion_valid.all())
        self.assertFalse(result.support_distance[::2, 9].any())
        self.assertFalse(result.stability_distance[::2, 9].any())
        self.assertFalse(result.tilt_angle[::2, 9].any())
        values['center_shelf'][1, 2] = math.nan
        invalid = self.advance(values)
        self.assertTrue(torch.isinf(invalid.support_distance[1, 2]))
        self.assertFalse(invalid.progress_valid[1])

    @unittest.skipUnless(torch.cuda.is_available(), 'Support CUDA graph requires CUDA.')
    def test_gpu_support_kernel_is_cuda_graph_capturable(self):
        values = {name: value.cuda() if isinstance(value, torch.Tensor) else value
                  for name, value in self.inputs(1024, 10).items()}
        cfg = settings()
        for _ in range(3):
            terminations.object_support_validity(cfg, **values)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = terminations.object_support_validity(cfg, **values)
        # Graph must see new device values, rather than a CPU side index copy.
        values['corners_shelf'][::2, 1, :, 2] += .04
        graph.replay()
        expected = terminations.object_support_validity(cfg, **values)
        for field in actual._fields:
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field))

    @unittest.skipUnless(torch.cuda.is_available(), 'Collision support reduction requires CUDA.')
    def test_catalog_collision_support_matches_scalar_oracle_and_captures_without_vertex_expansion(self):
        generator = torch.Generator().manual_seed(740)
        vertices = torch.randn(3, 129, 3, generator=generator) * .1
        counts = torch.tensor([1, 3, 129], dtype=torch.long)
        ids = torch.tensor([[0, 1, 2, -1], [2, 0, 1, 2]], dtype=torch.long)
        normals = torch.randn(2, 4, 3, generator=generator)
        normals /= normals.norm(dim=-1, keepdim=True)
        centers = torch.randn(2, 4, 3, generator=generator) * .03
        expected = torch.full((2, 4), math.inf)
        for row in range(2):
            for slot in range(4):
                index = int(ids[row, slot])
                if index >= 0:
                    points = vertices[index, :counts[index]] - centers[row, slot]
                    expected[row, slot] = (points * normals[row, slot]).sum(-1).min()
        values = [value.cuda() for value in (vertices, counts, ids, normals, centers)]
        actual = task.collision_support_minimum(*values)
        torch.testing.assert_close(actual.cpu(), expected, rtol=1.e-5, atol=1.e-7)
        # This strict capture catches implicit CPU index tensors and hidden
        # scalar reads; the catalog is shared regardless of environment count.
        graph = torch.cuda.CUDAGraph()
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('host condition')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('host transfer')):
            with torch.cuda.graph(graph):
                captured = task.collision_support_minimum(*values)
            graph.replay()
        torch.testing.assert_close(captured.cpu(), expected, rtol=1.e-5, atol=1.e-7)

class HandUrdfBoundsTests(unittest.TestCase):
    def test_native_action_endpoints_satisfy_every_coupled_urdf_joint_limit(self):
        from xml.etree import ElementTree

        root = Path(__file__).resolve().parents[3]
        source = root / 'src/contact_manipulation/contact_manipulation/config/ur5e/scene.py'
        nodes = [node for node in ast.parse(source.read_text()).body
                 if (isinstance(node, ast.FunctionDef) and node.name == 'hand_joint_limits')
                 or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id == 'HAND_JOINT_NAMES')]
        namespace = dict(ElementTree=ElementTree)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        urdf = root / ('assets/robots/Roas_provided_inspire_urdf/RH56E2 URDF/'
                       'RH56E2_R_2025_9_11/urdf/RH56E2_R_2025_9_11_relative.urdf')
        joints = {joint.attrib['name']: joint for joint in ElementTree.parse(urdf).findall('joint')}
        masters = namespace['HAND_JOINT_NAMES']
        positions, velocities = namespace['hand_joint_limits'](str(urdf))

        def angle(name, independent):
            mimic = joints[name].find('mimic')
            if mimic is None:
                return independent[name]
            return (float(mimic.attrib.get('multiplier', 1.0)) * angle(mimic.attrib['joint'], independent)
                    + float(mimic.attrib.get('offset', 0.0)))

        for master in masters:
            for position in positions[master]:
                independent = dict.fromkeys(masters, 0.0)
                independent[master] = position
                for name, joint in joints.items():
                    if joint.attrib['type'] != 'revolute':
                        continue
                    bounds = joint.find('limit').attrib
                    with self.subTest(master=master, position=position, joint=name):
                        self.assertGreaterEqual(angle(name, independent), float(bounds['lower']) - 1.e-10)
                        self.assertLessEqual(angle(name, independent), float(bounds['upper']) + 1.e-10)
        self.assertEqual(positions['right_thumb_1_joint'], (0.0, 1.658))
        self.assertLess(positions['right_thumb_2_joint'][1], .62)
        for name in masters[2:]:
            self.assertLess(positions[name][1], 1.4381)
            self.assertLess(velocities[name], 1.0)
            child = name.replace('_1_joint', '_2_joint')
            self.assertLessEqual(velocities[name] * float(joints[child].find('mimic').attrib['multiplier']),
                                 float(joints[child].find('limit').attrib['velocity']) + 1.e-10)


class NativeArmEffortLimitTests(unittest.TestCase):
    def test_explicit_arm_clips_physx_submission_while_implicit_hand_keeps_position_targets(self):
        from unittest.mock import Mock

        # Execute checkout methods; only SDK initialization/config containers are replaced.
        root = Path(__file__).resolve().parents[3]
        lab = root / 'IsaacLab/source/isaaclab/isaaclab'

        def native_method(relative, owner, method):
            path = lab / relative
            cls = next(n for n in ast.parse(path.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == owner)
            node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
            namespace = dict(torch=torch, ArticulationActions=SimpleNamespace)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
            return namespace[method]

        class Config(SimpleNamespace):
            InitialStateCfg = SimpleNamespace
        class IdealConfig(SimpleNamespace):
            pass
        class ImplicitConfig(SimpleNamespace):
            pass

        path = root / 'src/contact_manipulation/contact_manipulation/config/ur5e/scene.py'
        nodes = [node for node in ast.parse(path.read_text()).body
                 if (isinstance(node, ast.FunctionDef) and node.name == 'make_ur5e_hand_cfg')
                 or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in ('ARM_JOINT_NAMES', 'HAND_JOINT_NAMES'))]
        namespace = dict(os=SimpleNamespace(environ={}), DEFAULT_HAND_URDF_PATH='hand.urdf',
                         DEFAULT_UR5E_USD_PATH='ur5e.usd', Ur5eHandSpawnerCfg=SimpleNamespace,
                         ArticulationCfg=Config, IdealPDActuatorCfg=IdealConfig,
                         ImplicitActuatorCfg=ImplicitConfig)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        cfg = namespace['make_ur5e_hand_cfg']()
        arm_cfg, hand_cfg = cfg.actuators['arm'], cfg.actuators['hand']
        self.assertIs(type(arm_cfg), IdealConfig)
        self.assertIs(type(hand_cfg), ImplicitConfig)
        names = namespace['ARM_JOINT_NAMES']
        expected_limits = dict(zip(names, (150., 150., 150., 28., 28., 28.)))
        self.assertEqual(arm_cfg.effort_limit, expected_limits)
        self.assertEqual(arm_cfg.effort_limit_sim, expected_limits)
        self.assertEqual(list(arm_cfg.velocity_limit_sim.values()), [3.14]*3 + [6.28]*3)
        self.assertEqual((arm_cfg.stiffness, arm_cfg.damping), (0., 0.))
        self.assertEqual((hand_cfg.effort_limit_sim, hand_cfg.velocity_limit_sim), (None, None))

        clip = native_method('actuators/actuator_base.py', 'ActuatorBase', '_clip_effort')
        arm_compute = native_method('actuators/actuator_pd.py', 'IdealPDActuator', 'compute')
        hand_compute = native_method('actuators/actuator_pd.py', 'ImplicitActuator', 'compute')
        arm_ids, hand_ids = torch.arange(0, 12, 2), torch.arange(1, 12, 2)
        arm = SimpleNamespace(joint_indices=arm_ids, stiffness=arm_cfg.stiffness,
                              damping=arm_cfg.damping, effort_limit=torch.tensor(list(expected_limits.values())),
                              velocity_limit=torch.tensor(list(arm_cfg.velocity_limit_sim.values())))
        # Native initialization resolves None from the imported URDF/USD limits.
        hand = SimpleNamespace(joint_indices=hand_ids, effort_limit=torch.ones(6),
                               velocity_limit=torch.tensor([2., 2., 1., 1., 1., 1.]),
                               stiffness=torch.tensor([hand_cfg.stiffness[n] for n in namespace['HAND_JOINT_NAMES']]),
                               damping=torch.tensor([hand_cfg.damping[n] for n in namespace['HAND_JOINT_NAMES']]))
        for actuator, compute in ((arm, arm_compute), (hand, hand_compute)):
            actuator.compute = types.MethodType(compute, actuator)
            actuator._clip_effort = types.MethodType(clip, actuator)
        raw = torch.tensor([[800880., -800880., 100., -100., 29., -29.],
                            [-150., 150., -149., 27., -27., 0.],
                            [1., -1., 0., 28., -28., 1.e6]])
        expected = torch.tensor([[150., -150., 100., -28., 28., -28.],
                                 [-150., 150., -149., 27., -27., 0.],
                                 [1., -1., 0., 28., -28., 28.]])
        data = SimpleNamespace(joint_pos_target=torch.full((3, 12), .8),
                               joint_vel_target=torch.full((3, 12), .4),
                               **{name: torch.zeros(3, 12) for name in (
                                   'joint_pos', 'joint_vel', 'joint_effort_target', 'computed_torque',
                                   'applied_torque', 'soft_joint_vel_limits')})
        data.joint_effort_target[:, arm_ids] = raw
        view = SimpleNamespace(set_dof_actuation_forces=Mock(), set_dof_position_targets=Mock(),
                               set_dof_velocity_targets=Mock())
        asset = SimpleNamespace(_data=data, actuators={'arm': arm, 'hand': hand}, root_physx_view=view,
                                _joint_effort_target_sim=torch.zeros(3, 12),
                                _joint_pos_target_sim=torch.full((3, 12), -123.),
                                _joint_vel_target_sim=torch.full((3, 12), -321.),
                                _has_implicit_actuators=True, _ALL_INDICES=torch.arange(3),
                                _instantaneous_wrench_composer=SimpleNamespace(active=False, reset=Mock()),
                                _permanent_wrench_composer=SimpleNamespace(active=False))
        apply = native_method('assets/articulation/articulation.py', 'Articulation', '_apply_actuator_model')
        asset._apply_actuator_model = types.MethodType(apply, asset)
        write = native_method('assets/articulation/articulation.py', 'Articulation', 'write_data_to_sim')
        write(asset)
        submitted = view.set_dof_actuation_forces.call_args.args[0]
        torch.testing.assert_close(submitted[:, arm_ids], expected)
        torch.testing.assert_close(data.computed_torque[:, arm_ids], raw)
        torch.testing.assert_close(data.joint_effort_target[:, arm_ids], raw)
        torch.testing.assert_close(data.applied_torque[:, arm_ids], expected)
        # Hand's implicit PD stays in PhysX: telemetry is clipped, feed-forward remains zero.
        torch.testing.assert_close(data.computed_torque[:, hand_ids], torch.full((3, 6), 16.4))
        torch.testing.assert_close(data.applied_torque[:, hand_ids], torch.ones(3, 6))
        torch.testing.assert_close(submitted[:, hand_ids], torch.zeros(3, 6))
        for method, field in ((view.set_dof_position_targets, 'joint_pos_target'),
                              (view.set_dof_velocity_targets, 'joint_vel_target')):
            torch.testing.assert_close(method.call_args.args[0][:, hand_ids], getattr(data, field)[:, hand_ids])
        torch.testing.assert_close(asset._joint_pos_target_sim[:, arm_ids], torch.full((3, 6), -123.))
        torch.testing.assert_close(asset._joint_vel_target_sim[:, arm_ids], torch.full((3, 6), -321.))


class NativeVelocityTerminationTests(unittest.TestCase):
    def test_official_threshold_checks_arm_and_hand_independent_joints_without_host_reads(self):
        native = native_velocity_termination()
        # Interleaved PhysX ordering makes an accidental [:6] or [:12] selection visible.
        ids = [5, 0, 14, 3, 16, 8, 1, 17, 6, 2, 13, 10]
        omitted = [index for index in range(18) if index not in ids]
        for dtype in (torch.float32, torch.float64):
            velocity = torch.zeros(1024, 18, dtype=dtype)
            velocity[0, ids] = 3.0  # The Inspire variant uses a strict > boundary.
            velocity[1, ids[4]] = 3.0001  # Arm.
            velocity[2, ids[11]] = -3.0001  # Independent finger.
            velocity[3, omitted] = 1.e8  # Mimic joints are not the selected independent DOFs.
            velocity[4, ids] = -2.0  # The former 1 rad/s cutoff no longer applies.
            env = SimpleNamespace(scene={'robot': SimpleNamespace(data=SimpleNamespace(joint_vel=velocity))})
            asset_cfg = SimpleNamespace(name='robot', joint_ids=ids)
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Tensor scalar read')), \
                 patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Tensor condition read')), \
                 patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('CPU transfer')):
                actual = native(env, max_velocity=3.0, asset_cfg=asset_cfg)
            expected = torch.zeros(1024, dtype=torch.bool)
            expected[[1, 2]] = True
            torch.testing.assert_close(actual, expected)

    def test_ur5e_configuration_uses_same_twelve_named_joints_for_reward_and_termination(self):
        source = Path(task.__file__).parent.parent / 'config/ur5e/ur5e_env_cfg.py'
        definition = next(node for node in ast.parse(source.read_text()).body
                          if isinstance(node, ast.ClassDef) and node.name == 'UR5eContactManipulationEnvCfg')
        method = next(node for node in definition.body
                      if isinstance(node, ast.FunctionDef) and node.name == '__post_init__')
        action_names = {}
        for node in method.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                target = ast.unparse(node.targets[0])
                if target in ('self.actions.arm_action', 'self.actions.hand_action'):
                    action_names[target.rsplit('.', 1)[-1]] = next(
                        ast.literal_eval(keyword.value) for keyword in node.value.keywords
                        if keyword.arg == 'joint_names')
        start = next(index for index, node in enumerate(method.body)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'independent_joint_names'
                             for target in node.targets))
        # Execute the original specialization's joint binding; no copied assignment formula.
        code = compile(ast.fix_missing_locations(ast.Module(body=method.body[start:], type_ignores=[])),
                       str(source), 'exec')
        reward_asset = SimpleNamespace(joint_names=None)
        velocity_asset = SimpleNamespace(joint_names=dataclasses.MISSING)
        cfg = SimpleNamespace(
            actions=SimpleNamespace(**{name: SimpleNamespace(joint_names=names)
                                       for name, names in action_names.items()}),
            rewards=SimpleNamespace(motion_joint_velocity=SimpleNamespace(params={'asset_cfg': reward_asset})),
            terminations=SimpleNamespace(hand_velocity=SimpleNamespace(params={'asset_cfg': velocity_asset})),
        )
        exec(code, {'self': cfg})
        expected = action_names['arm_action'] + action_names['hand_action']
        self.assertEqual(len(set(expected)), 12)
        self.assertEqual(reward_asset.joint_names, expected)
        self.assertEqual(velocity_asset.joint_names, expected)
        self.assertIn('right_little_1_joint', velocity_asset.joint_names)
        cfg.terminations.hand_velocity = None
        exec(code, {'self': cfg})  # Disabling only the termination keeps the velocity reward valid.
        self.assertEqual(reward_asset.joint_names, expected)

    def test_velocity_failure_wins_over_settled_success_without_adding_geometry_failure_columns(self):
        state, batch, _ = fixture(3)
        batch.center[:, 1, 0] += .4
        batch.corners[:, 1, :, 0] += .4
        state.stable_count.fill_(success_steps(state.cfg, state.dt) - 1)
        violation = torch.tensor([True, False, True])
        with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Tensor scalar read')), \
             patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('Tensor condition read')):
            state.advance(batch, 1, hand_velocity=violation)
        torch.testing.assert_close(state.failure, violation)
        torch.testing.assert_close(state.success, ~violation)
        torch.testing.assert_close(state.terminal_reward, batch.center.new_tensor([0., 250., 0.]))
        torch.testing.assert_close(state.stable_count[violation], torch.zeros(2, dtype=torch.long))
        self.assertFalse(state.clearance_progress_valid[violation].any())
        self.assertEqual(tuple(state.failure_reasons.shape), (3, 5))
        self.assertFalse(state.failure_reasons.any())


@unittest.skipUnless("isaaclab.controllers" in sys.modules, "Requires Isaac Sim AppLauncher")
class NativeControllerTests(unittest.TestCase):
    def test_project_osc_task_frame_matches_the_observation_eef(self):
        from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
        from contact_manipulation.config.ur5e.scene import TCP_BODY_NAME, TACTILE_BODY_NAMES
        from contact_manipulation.config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg

        cfg = UR5eContactManipulationEnvCfg()
        arm = cfg.actions.arm_action
        observation_eef = cfg.scene.ee_frame.target_frames[0].prim_path
        self.assertIs(arm.class_type, OperationalSpaceControllerAction)
        self.assertEqual(arm.body_name, "wrist_3_link")
        self.assertEqual(arm.task_frame_rel_path, "Robot/wrist_3_link")
        self.assertEqual(observation_eef, "{ENV_REGEX_NS}/Robot/wrist_3_link")
        self.assertEqual("{ENV_REGEX_NS}/" + arm.task_frame_rel_path, observation_eef)
        self.assertEqual(arm.body_name, observation_eef.rsplit("/", 1)[-1])
        self.assertIsNone(arm.body_offset)
        self.assertEqual(arm.controller_cfg.target_types, ["pose_rel"])
        self.assertEqual((arm.position_scale, arm.orientation_scale, arm.stiffness_scale), (1., 1., 100.))
        self.assertTrue(arm.controller_cfg.gravity_compensation)
        velocity = cfg.terminations.hand_velocity
        self.assertEqual(velocity.func.__name__, 'joint_vel_out_of_manual_limit')
        self.assertEqual(velocity.params['max_velocity'], 3.0)
        self.assertFalse(velocity.time_out)
        self.assertEqual(velocity.params['asset_cfg'].joint_names,
                         arm.joint_names + cfg.actions.hand_action.joint_names)
        self.assertEqual(velocity.params['asset_cfg'].joint_names,
                         cfg.rewards.motion_joint_velocity.params['asset_cfg'].joint_names)
        # Reset IK targets the palm; changing the control point must preserve
        # both the hand's Blocker-front start and its existing tactile links.
        self.assertEqual(TCP_BODY_NAME, "right_plam_force_sensor")
        self.assertEqual(TACTILE_BODY_NAMES[0], TCP_BODY_NAME)
        self.assertEqual(len(set(TACTILE_BODY_NAMES)), 17)

    def test_native_eef_delta_pose_and_anisotropic_gains_rotate_into_base(self):
        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

        controller = OperationalSpaceController(OperationalSpaceControllerCfg(
            target_types=["pose_rel"], impedance_mode="variable_kp",
            motion_stiffness_limits_task=(50., 200.), nullspace_control="none",
        ), 3, "cpu")
        half = math.sqrt(.5)
        pose = torch.tensor([[.3, -.1, .4, 1., 0., 0., 0.],
                             [.5, .2, .1, half, 0., 0., half],
                             [-.2, .4, .3, half, 0., half, 0.]])
        delta = torch.tensor([[.01, -.02, .03, .2, -.12, .08]]).expand(3, -1)
        gains = torch.tensor([[50., 100., 200., 60., 120., 180.]]).expand(3, -1)
        controller.set_command(torch.cat((delta, gains), -1),
                               current_ee_pose_b=pose, current_task_frame_pose_b=pose)

        # An independent SciPy oracle checks local translation and right-multiplied
        # local rotation; the latter differs from a base-axis rotation command.
        eef_rotation = Rotation.from_quat(pose[:, [4, 5, 6, 3]].numpy())
        expected_position = pose[:, :3].numpy() + eef_rotation.apply(delta[:, :3].numpy())
        expected_rotation = eef_rotation * Rotation.from_rotvec(delta[:, 3:].numpy())
        torch.testing.assert_close(controller.desired_ee_pose_b[:, :3],
                                   torch.tensor(expected_position, dtype=pose.dtype))
        torch.testing.assert_close(rotation_matrix(controller.desired_ee_pose_b[:, 3:]),
                                   torch.tensor(expected_rotation.as_matrix(), dtype=pose.dtype))
        expected_gains = torch.tensor([[50., 100., 200., 60., 120., 180.],
                                       [100., 50., 200., 120., 60., 180.],
                                       [200., 100., 50., 180., 120., 60.]])
        torch.testing.assert_close(controller._motion_p_gains_b, torch.diag_embed(expected_gains),
                                   atol=5.e-5, rtol=1.e-6)

    def test_native_eef_target_and_axes_are_latched_until_the_next_policy_command(self):
        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

        controller = OperationalSpaceController(OperationalSpaceControllerCfg(
            target_types=["pose_rel"], impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True, motion_stiffness_limits_task=(50., 200.),
            nullspace_control="none",
        ), 1, "cpu")
        half = math.sqrt(.5)
        pose = torch.tensor([[.3, .2, .4, half, 0., 0., half]])
        command = torch.tensor([[.01, 0., 0., 0., 0., 0., 50., 100., 200., 60., 120., 180.]])
        controller.set_command(command, current_ee_pose_b=pose, current_task_frame_pose_b=pose)
        target, gains = controller.desired_ee_pose_b.clone(), controller._motion_p_gains_b.clone()
        torch.testing.assert_close(target[:, :3], torch.tensor([[.3, .21, .4]]))
        changed_pose = torch.tensor([[.31, .2, .4, half, 0., half, 0.]])
        for measured in (pose, changed_pose):
            efforts = controller.compute(jacobian_b=torch.eye(6)[None],
                current_ee_pose_b=measured, current_ee_vel_b=torch.zeros(1, 6),
                mass_matrix=torch.eye(6)[None])
            self.assertTrue(torch.isfinite(efforts).all())
            torch.testing.assert_close(controller.desired_ee_pose_b, target)
            torch.testing.assert_close(controller._motion_p_gains_b, gains)
        controller.set_command(command, current_ee_pose_b=changed_pose,
                               current_task_frame_pose_b=changed_pose)
        torch.testing.assert_close(controller.desired_ee_pose_b[:, :3], torch.tensor([[.31, .2, .39]]))
        self.assertFalse(torch.allclose(controller._motion_p_gains_b, gains))

    def test_native_hand_soft_limit_mapping_and_selected_reset(self):
        from unittest.mock import Mock
        from isaaclab.envs.mdp.actions.actions_cfg import JointPositionToLimitsActionCfg
        from isaaclab.envs.mdp.actions.joint_actions_to_limits import JointPositionToLimitsAction

        names = [f"finger_{i}" for i in range(6)]
        cfg = JointPositionToLimitsActionCfg(
            asset_name="robot", joint_names=names, preserve_order=True, rescale_to_limits=True,
        )
        limits = torch.tensor([[-.5, 1.5]]).expand(3, 6, 2).clone()
        data = SimpleNamespace(soft_joint_pos_limits=limits, joint_pos=torch.zeros(3, 6))
        asset = SimpleNamespace(data=data, num_joints=6,
            find_joints=lambda names, preserve_order: (list(range(6)), names), set_joint_position_target=Mock())
        env = SimpleNamespace(scene={'robot': asset}, num_envs=3, device='cpu')
        hand = cfg.class_type(cfg, env)
        self.assertIs(type(hand), JointPositionToLimitsAction)
        hand.process_actions(torch.tensor([[-1.]*6, [0.]*6, [1.]*6]))
        expected = torch.tensor([[-.5]*6, [.5]*6, [1.5]*6])
        torch.testing.assert_close(hand.processed_actions, expected)
        hand.apply_actions()
        torch.testing.assert_close(asset.set_joint_position_target.call_args.args[0], expected)
        # Native mapping uses the live limits and passes the full position target.
        data.soft_joint_pos_limits[1, :, 1] = .5
        hand.process_actions(torch.full((3, 6), 2.))
        torch.testing.assert_close(hand.processed_actions[1], torch.full((6,), .5))
        before = hand.processed_actions.clone()
        hand.reset(torch.tensor([1]))
        torch.testing.assert_close(hand.raw_actions[[0, 2]], torch.full((2, 6), 2.))
        torch.testing.assert_close(hand.raw_actions[1], torch.zeros(6))
        torch.testing.assert_close(hand.processed_actions, before)

    def test_native_osc_action_scaling_and_gain_clamp(self):
        from isaaclab.controllers import OperationalSpaceControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
        from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction

        names = [f"arm_{i}" for i in range(6)]
        cfg = OperationalSpaceControllerActionCfg(
            asset_name="robot", joint_names=names, body_name="tcp",
            position_scale=.015, orientation_scale=.08, stiffness_scale=100.,
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"], impedance_mode="variable_kp",
                motion_stiffness_limits_task=(5., 800.), nullspace_control="none",
            ),
        )
        root_pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]]).expand(4, -1).clone()
        body_pose = root_pose[:, None].expand(-1, 2, -1).clone()
        body_pose[:, 1, 0] = .3
        data = SimpleNamespace(root_pos_w=root_pose[:, :3], root_quat_w=root_pose[:, 3:],
            body_pos_w=body_pose[:, :, :3], body_quat_w=body_pose[:, :, 3:])
        asset = SimpleNamespace(data=data, num_joints=6, is_fixed_base=True,
            find_joints=lambda names: (list(range(6)), names),
            find_bodies=lambda name: ([1], [name]))
        env = SimpleNamespace(scene={'robot': asset}, num_envs=4, device='cpu',
            sim=SimpleNamespace(get_physics_dt=lambda: .01))
        arm = cfg.class_type(cfg, env)
        self.assertIs(type(arm), OperationalSpaceControllerAction)
        actions = torch.ones(4, 12)
        actions[:, 6:] = torch.tensor([-1., 0., 1., 10.])[:, None]
        arm.process_actions(actions)
        torch.testing.assert_close(arm.processed_actions[:, :3], torch.full((4, 3), .015))
        torch.testing.assert_close(arm.processed_actions[:, 3:6], torch.full((4, 3), .08))
        torch.testing.assert_close(arm.processed_actions[:, 6:],
                                   torch.tensor([5., 5., 100., 800.])[:, None].expand(-1, 6))
        arm.reset(torch.tensor([1]))
        torch.testing.assert_close(arm.raw_actions[1], torch.zeros(12))
        torch.testing.assert_close(arm.raw_actions[[0, 2, 3]], actions[[0, 2, 3]])

    def test_native_osc_relative_latch_and_inertia_scaled_pd(self):
        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

        cfg = OperationalSpaceControllerCfg(
            target_types=["pose_rel"], impedance_mode="variable_kp",
            inertial_dynamics_decoupling=True, partial_inertial_dynamics_decoupling=False,
            gravity_compensation=True, motion_damping_ratio_task=1.0,
            motion_stiffness_limits_task=(5., 800.), nullspace_control="none",
        )
        controller = OperationalSpaceController(cfg, 1024, "cpu")
        half = math.sqrt(.5)
        pose = torch.tensor([[.3, 0, .4, half, 0, half, 0]]).expand(1024, -1).clone()
        gains = torch.tensor([[450., 450, 450, 22.5, 22.5, 22.5]]).expand(1024, -1)
        delta = torch.tensor([[.01, -.01, .005, .03, 0, 0]]).expand(1024, -1)
        command = torch.cat((delta, gains), -1)
        controller.set_command(command, current_ee_pose_b=pose)
        target = controller.desired_ee_pose_b.clone()
        controller.set_command(command, current_ee_pose_b=pose)
        torch.testing.assert_close(controller.desired_ee_pose_b, target)
        torch.testing.assert_close(target[:, :3], pose[:, :3] + delta[:, :3])
        torch.testing.assert_close(target[0, 3:], half * torch.tensor(
            [math.cos(.015), math.sin(.015), math.cos(.015), math.sin(.015)]))
        diagonal = torch.tensor([[1., 2., 3., .01, .02, .03]]).expand(1024, -1)
        mass = torch.diag_embed(diagonal)
        velocity = torch.full((1024, 6), .02)
        gravity = torch.full_like(velocity, .1)
        efforts = controller.compute(jacobian_b=torch.eye(6).expand(1024, -1, -1),
            current_ee_pose_b=pose, current_ee_vel_b=velocity, mass_matrix=mass, gravity=gravity)
        expected = diagonal * (gains * delta - 2 * gains.sqrt() * velocity) + gravity
        torch.testing.assert_close(efforts, expected, atol=2e-5, rtol=2e-5)


def _clip_oracle(vertices, bounds):
    """Independent scalar half-plane oracle operating on SciPy's convex hull."""
    current = [np.asarray(vertex) for vertex in vertices]
    for axis, bound, keep_lower in ((0, bounds[0], False), (0, bounds[1], True),
                                     (1, bounds[2], False), (1, bounds[3], True)):
        output = []
        if not current:
            return 0.0
        for previous, following in zip(current[-1:] + current[:-1], current):
            previous_inside = previous[axis] <= bound if keep_lower else previous[axis] >= bound
            next_inside = following[axis] <= bound if keep_lower else following[axis] >= bound
            if previous_inside != next_inside:
                ratio = (bound - previous[axis]) / (following[axis] - previous[axis])
                output.append(previous + ratio * (following - previous))
            if next_inside:
                output.append(following)
        current = output
    if len(current) < 3:
        return 0.0
    polygon = np.asarray(current)
    return abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
               - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2


def fixture(n=3, *, dtype=torch.float64):
    cfg = task.MdpCfg()
    o = cfg.max_objects
    center = torch.zeros(n, o, 3, dtype=dtype)
    center[..., 0] = torch.arange(o, dtype=dtype) * 0.12
    center[..., 2] = 0.05
    signs = torch.tensor([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]], dtype=dtype)
    corners = center[..., None, :] + 0.05 * signs
    rotation = torch.zeros(n, o, 4, dtype=dtype)
    rotation[..., 0] = 1.0
    sensor_rotation = torch.zeros(n, 4, dtype=dtype)
    sensor_rotation[:, 0] = 1.0
    tcp_pose = torch.zeros(n, 7, dtype=dtype)
    tcp_pose[:, 3] = 1.0
    batch = task.TransitionBatch(
        center=center, corners=corners, rotation=rotation,
        velocity=torch.zeros(n, o, 6, dtype=dtype),
        sensor_wrench=torch.zeros(n, 6, dtype=dtype), sensor_position=torch.zeros(n, 3, dtype=dtype),
        sensor_rotation=sensor_rotation, sensor_range_exceeded=torch.zeros(n, 6, dtype=torch.bool),
        shelf_pose=torch.tensor([0., 0., 0., 1., 0., 0., 0.], dtype=dtype).expand(n, 7).clone(), state_valid=torch.ones(n, dtype=torch.bool),
        sample=torch.arange(72, dtype=dtype).expand(n, -1).clone(), tcp_pose_b=tcp_pose,
    )
    active = torch.ones(n, o, dtype=torch.bool)
    target = torch.zeros_like(active)
    target[:, 0] = True
    blocker = torch.zeros_like(active)
    blocker[:, 1] = True
    geometry = torch.zeros(n, o, 10, dtype=dtype)
    geometry[..., :3] = center
    geometry[..., 3] = 1.0
    geometry[..., 7:] = 0.1
    reset = dict(geometry_b=geometry, geometry_valid=active.clone(), active=active, target=target,
                 blocker=blocker, access_region=torch.tensor([-0.1, 0.2, -0.2, 0.2], dtype=dtype).expand(n, -1),
                 workspace_bounds=torch.tensor([-10., 10., -2., 2.], dtype=dtype).expand(n, -1),
                 shelf_top=torch.zeros(n, dtype=dtype), ceiling=torch.ones(n, dtype=dtype),
                 shelf_local_corners=(signs * signs.new_tensor([.3, .5, 1.])
                                      + signs.new_tensor([0., 0., 1.])).expand(n, -1, -1).clone())
    state = task.TaskState(n, "cpu", cfg, 0.1, dtype=dtype)
    state.reset(torch.arange(n), batch, **reset, step=0)
    return state, batch, reset


def clone_batch(batch, **updates):
    values = {field.name: getattr(batch, field.name).clone() for field in dataclasses.fields(batch)}
    values.update(updates)
    return task.TransitionBatch(**values)


def native_fixture(n=3):
    """Native tensor layout with deliberately independent camera and GT geometry."""
    class Scene(dict):
        pass

    origins = torch.zeros(n, 3)
    origins[:, 1] = torch.arange(n) * 3.0
    root_q = torch.tensor([1., 0., 0., 0.]).expand(n, 4).clone()
    body_q = root_q[:, None].expand(n, 4, 4).clone()
    body_pos = origins[:, None].expand(n, 4, 3).clone()
    body_pos[:, 1:, 2] += .5
    body_com = body_pos.clone()
    body_com[:, 3, 0] += .1
    data = types.SimpleNamespace(
        body_link_pos_w=body_pos, body_link_quat_w=body_q,
        body_com_pos_w=body_com, body_com_pos_b=body_com-body_pos, body_com_acc_w=torch.zeros(n, 4, 6),
        body_com_ang_vel_w=torch.zeros(n, 4, 3), root_pos_w=origins.clone(), root_quat_w=root_q,
        joint_pos=torch.zeros(n, 12), joint_vel=torch.zeros(n, 12), joint_acc=torch.zeros(n, 12),
        applied_torque=torch.zeros(n, 12),
    )
    inertia = torch.diag(torch.tensor([.01, .02, .03])).flatten().expand(n, 4, 9).clone()
    arm_names = [f"arm_{i}" for i in range(6)]
    hand_names = [f"hand_{i}" for i in range(6)]
    joint_names = arm_names + hand_names
    body_names = ["base", "tcp", "ft", "hand"]
    robot = types.SimpleNamespace(
        body_names=body_names, num_joints=12, data=data,
        find_joints=lambda names, preserve_order=False: ([joint_names.index(name) for name in names], names),
        find_bodies=lambda name: ([body_names.index(name)], [name]),
        root_physx_view=types.SimpleNamespace(get_masses=lambda: torch.ones(n, 4), get_inertias=lambda: inertia),
    )
    physical_pose = torch.zeros(n, 4, 7)
    physical_pose[..., :3] = origins[:, None]
    physical_pose[..., 0] += torch.tensor([-.1, .1, .3, .5])
    physical_pose[..., 2] += .05
    physical_pose[..., 3] = 1.
    objects = types.SimpleNamespace(data=types.SimpleNamespace(
        object_link_pose_w=physical_pose, object_com_pos_w=physical_pose[..., :3].clone(),
        object_com_pos_b=torch.zeros(n, 4, 3),
        object_com_vel_w=torch.zeros(n, 4, 6),
    ))
    shelf_pose = torch.cat((origins.clone(), root_q.clone()), -1)
    scene = Scene(robot=robot, object_collection=objects,
                  shelf=SimpleNamespace(data=SimpleNamespace(root_link_pose_w=shelf_pose)))
    scene.env_origins = origins
    action = types.SimpleNamespace(normalized_actions=torch.zeros(n, 18), previous_actions=torch.zeros(n, 18),
                                   applied_torque=data.applied_torque)
    object_ids = torch.full((n, 10), -1, dtype=torch.long)
    object_ids[:, :3] = torch.tensor([2, 0, 3])
    observed_ids = torch.full((n, 10), -1, dtype=torch.long)
    observed_ids[:, :3] = torch.tensor([3, 2, 0])
    geometry = torch.zeros(n, 10, 10)
    geometry[:, :3, 0] = torch.tensor([.51, .31, -.11])
    geometry[:, :3, 2] = .12
    geometry[:, :3, 3] = 1.
    geometry[:, :3, 7:] = .08
    geometry_valid = observed_ids >= 0
    source = task.SceneBatch(
        object_ids=object_ids, target_id=torch.full((n,), 2), blocker_id=torch.zeros(n, dtype=torch.long),
        geometry_b=geometry, geometry_valid=geometry_valid, observed_ids=observed_ids,
        lower=torch.full((4, 3), -.05), upper=torch.full((4, 3), .05),
        upright=torch.tensor([1., 0., 0., 0.]).expand(4, 4).clone(),
        access_region=torch.tensor([-.2, .29, -.2, .2]).expand(n, 4).clone(),
        workspace_bounds=torch.tensor([-1., 1., -1., 1.]).expand(n, 4).clone(),
        shelf_top=torch.zeros(n), ceiling=torch.ones(n), sensor_body_name="ft",
        shelf_local_corners=torch.tensor([
            [x, y, z] for x in (-.3, .3) for y in (-.5, .5) for z in (0., 2.)
        ]).expand(n, -1, -1).clone(),
    )
    external = torch.tensor([1., 2., 3., .1, .2, .3]).expand(n, 6).clone()
    self_load = torch.tensor([0., 0., 20., 0., -1., 0.])
    sensors = task.SensorBatch(
        ideal_wrench=external+self_load, wrench_valid=torch.ones(n, dtype=torch.bool),
        tactile_force=torch.zeros(n, 17, 3), tactile_valid=torch.ones(n, 17, dtype=torch.bool),
    )
    class ActionManager:
        @property
        def jacobian_w(self):
            self.jacobian_reads += 1
            return self.current_jacobian

        def get_term(self, name):
            if name != 'arm_action':
                raise KeyError(name)
            return self

        @property
        def action(self):
            return action.normalized_actions

        @property
        def prev_action(self):
            return action.previous_actions

    env = types.SimpleNamespace(
        num_envs=n, device="cpu", cfg=types.SimpleNamespace(mdp=task.MdpCfg(), actions=SimpleNamespace(
            arm_action=SimpleNamespace(joint_names=arm_names, body_name="tcp"),
            hand_action=SimpleNamespace(joint_names=hand_names))), step_dt=.1,
        common_step_counter=0, scene=scene, sim=types.SimpleNamespace(cfg=types.SimpleNamespace(gravity=(0., 0., -10.))),
        action_manager=ActionManager(),
        task_scene_data=source, sensors=sensors, extras={}, sensor_read_count=0,
    )
    env.action_manager.current_jacobian = torch.diag(torch.tensor([.25, .25, .25, 1., 1., 1.])).expand(n, 6, 6).clone()
    env.action_manager.jacobian_reads = 0
    def read_sensors():
        env.sensor_read_count += 1
        return env.sensors
    env.task_sensor_reader = read_sensors
    return env, action, external


class GeometryTests(unittest.TestCase):
    def test_random_roll_pitch_yaw_obb_hull_and_overlap_match_scipy(self):
        rng = np.random.default_rng(33)
        rotations = Rotation.random(200, random_state=rng)
        sizes = rng.uniform(0.01, 0.5, (200, 3))
        centers = rng.uniform(-0.3, 0.3, (200, 3))
        signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                          [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
        corners = np.einsum("nij,nkj->nki", rotations.as_matrix(), sizes[:, None, :] * signs) + centers[:, None, :]
        points = torch.tensor(corners[..., :2], dtype=torch.float64)
        polygons, counts = task.convex_hull_xy(points)
        areas = task.polygon_area(polygons, counts).numpy()
        bounds = np.array([-0.15, 0.25, -0.2, 0.3])
        overlaps = task.rectangle_overlap_area(polygons, counts, torch.tensor(bounds)).numpy()
        expected_areas, expected_overlaps = [], []
        for xy in corners[..., :2]:
            hull = ConvexHull(xy)
            expected_areas.append(hull.volume)
            expected_overlaps.append(_clip_oracle(xy[hull.vertices], bounds))
        np.testing.assert_allclose(areas, expected_areas, rtol=1.e-11, atol=1.e-12)
        np.testing.assert_allclose(overlaps, expected_overlaps, rtol=1.e-10, atol=1.e-12)

    def test_axis_aligned_duplicate_corners_and_disjoint_clipping(self):
        _, batch, _ = fixture(1)
        points = batch.corners[:, 0, :, :2]
        polygons, count = task.convex_hull_xy(points)
        self.assertEqual(count.item(), 4)
        torch.testing.assert_close(task.polygon_area(polygons, count), torch.tensor([0.01], dtype=points.dtype))
        overlap = task.rectangle_overlap_area(polygons, count, torch.tensor([2., 3., 2., 3.]))
        torch.testing.assert_close(overlap, torch.zeros_like(overlap))

    def test_degenerate_zero_boxes_have_zero_area_and_overlap(self):
        points = torch.zeros(10, 8, 2, dtype=torch.float64)
        polygon, count = task.convex_hull_xy(points)
        torch.testing.assert_close(task.polygon_area(polygon, count), torch.zeros(10, dtype=points.dtype))
        overlap = task.rectangle_overlap_area(polygon, count, torch.tensor([-1., 1., -1., 1.]))
        torch.testing.assert_close(overlap, torch.zeros_like(overlap))

    def test_object_geometry_removes_catalog_upright_once_with_off_center_origin(self):
        q = torch.tensor([[math.sqrt(0.5), 0., 0., math.sqrt(0.5)]], dtype=torch.float64)
        pose = torch.cat((torch.tensor([[2., 3., 0.]], dtype=q.dtype), q), -1)
        local_center = torch.tensor([[0.2, 0., 0.1]], dtype=q.dtype)
        local_corners = local_center[:, None, :].expand(-1, 8, -1).clone()
        center, corners, relative = task.object_geometry(pose, local_center, local_corners, q)
        torch.testing.assert_close(center, pose[:, :3] + local_center)
        torch.testing.assert_close(corners, (pose[:, :3] + local_center)[:, None, :].expand(-1, 8, -1))
        torch.testing.assert_close(relative, torch.tensor([[1., 0., 0., 0.]], dtype=q.dtype))


class StateTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'Support replay cache requires CUDA.')
    def test_support_graph_tracks_new_samples_counters_partial_reset_and_inference_modes(self):
        _, source, reset = fixture(4, dtype=torch.float32)
        batch = task.TransitionBatch(**{field.name: getattr(source, field.name).cuda()
                                        for field in dataclasses.fields(source)})
        reset = {name: value.cuda() for name, value in reset.items()}
        state = task.TaskState(4, 'cuda', task.MdpCfg(), .01, dtype=torch.float32)
        ids = torch.arange(4, device='cuda')
        state.reset(ids, batch, **reset, step=0)
        state.support_vertices = (batch.corners[0] - batch.center[0, :, None]).contiguous()
        state.support_vertex_counts = torch.full((10,), 8, device='cuda', dtype=torch.long)
        state.support_error = torch.zeros(10, device='cuda')
        state.bindings = SimpleNamespace(object_ids=torch.arange(10, device='cuda').expand(4, -1).clone(),
                                         local_center=torch.zeros(4, 10, 3, device='cuda'),
                                         robot=SimpleNamespace(data=SimpleNamespace(
                                             root_pos_w=torch.zeros(4, 3, device='cuda'),
                                             root_quat_w=torch.tensor([1., 0., 0., 0.], device='cuda').expand(4, -1))))
        initial = clone_batch(batch)
        for step, inference in enumerate((False, True, False, True), start=1):
            if step == 1:
                batch.center[0, 1, 0] += 11.
                batch.corners[0, 1, :, 0] += 11.
                batch.center[1, 2, 2] += .04
                batch.corners[1, 2, :, 2] += .04
                batch.velocity[2, 1, 0] = .5
            if step == 3:
                selected = torch.tensor([3, 1], device='cuda')
                for field in dataclasses.fields(batch):
                    getattr(batch, field.name)[selected] = getattr(initial, field.name)[selected]
                reset['workspace_bounds'][3, 3] = .01
                state.reset(selected, batch, **reset, step=2)
                self.assertFalse(state.support_loss_count[selected].any())
                self.assertEqual(int(state.support_loss_count[0, 1]), 2)
            with torch.inference_mode(inference):
                expected = state._support_tensors(batch)
                actual = state._support_validity(batch)
                for field in expected._fields:
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
                state.advance(batch, step)
            self.assertEqual(int(state.support_loss_count[0, 1]), min(step, 10))
            self.assertEqual(int(state.support_loss_count[1, 2]), step if step < 3 else 0)
            self.assertFalse(state.completion_valid[2])  # Updated object velocity stays visible.
            if step >= 3:
                self.assertFalse(state.completion_valid[3])  # Reset changed the support rectangle.
        self.assertEqual({key[0] for key in state._support_graphs}, {False, True})

    def test_clearance_rewards_supported_moving_blocker_then_success_waits_half_second(self):
        state, batch, _ = fixture(1)
        batch.center[:, 1, 0] += .4
        batch.corners[:, 1, :, 0] += .4
        batch.velocity[:, 1, 0] = .5
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            state.advance(batch, 1)
            self.assertTrue(state.clearance_progress_valid.all())
            self.assertTrue((rewards.clearance(env) > 0).all())
            self.assertFalse(state.success.any())
            self.assertFalse(state.stable_count.any())
            batch.velocity.zero_()
            for step in range(1, success_steps(state.cfg, state.dt) + 1):
                state.advance(batch, step + 1)
                self.assertEqual(bool(state.success[0]), step == success_steps(state.cfg, state.dt))
                self.assertEqual(float(state.terminal_reward[0]), 250. if state.success[0] else 0.)

    def test_support_geometry_follows_current_shelf_pose_instead_of_fixed_world_plane(self):
        state, batch, _ = fixture(2)
        half = math.radians(1.) / 2
        rotation = batch.rotation.new_tensor([math.cos(half), 0., math.sin(half), 0.]).expand(2, 4)
        batch.center.copy_(task.quat_apply(rotation[:, None], batch.center))
        batch.corners.copy_(task.quat_apply(rotation[:, None, None], batch.corners))
        batch.rotation.copy_(rotation[:, None].expand_as(batch.rotation))
        batch.shelf_pose[:, 3:] = rotation
        state.advance(batch, 1)
        self.assertTrue(state.clearance_progress_valid.all())
        self.assertTrue(state.completion_valid.all())
        self.assertFalse(state.failure.any())
        torch.testing.assert_close(state.tilt_angle, torch.zeros_like(state.tilt_angle), atol=1.e-7, rtol=0)

    def test_stability_buffers_follow_current_geometry_and_reset_only_selected_rows(self):
        state, batch, reset = fixture(2)
        delta = 9.98 - float(batch.center[0, 1, 0])
        batch.center[0, 1, 0] += delta
        batch.corners[0, 1, :, 0] += delta
        yaw = batch.rotation.new_tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        local = batch.corners[0, 1] - batch.center[0, 1]
        batch.corners[0, 1] = batch.center[0, 1] + task.quat_apply(yaw.expand(8, -1), local)
        batch.rotation[0, 1] = yaw
        # A shelf-supported tilted cube: its exact OBB bottom stays at zero.
        angle = math.pi / 4
        q = batch.rotation.new_tensor([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
        relative = batch.corners[1, 1] - batch.center[1, 1]
        rotated = task.quat_apply(q.expand(8, -1), relative)
        batch.center[1, 1, 2] = -rotated[:, 2].amin()
        batch.corners[1, 1] = rotated + batch.center[1, 1]
        batch.rotation[1, 1] = q
        state.advance(batch, 1)
        self.assertFalse(state.failure.any())
        torch.testing.assert_close(state.stability_distance[:, 1], batch.center.new_tensor([.03, 0.]))
        torch.testing.assert_close(state.tilt_angle[:, 1], batch.center.new_tensor([0., math.pi / 4]))
        torch.testing.assert_close(state.support_distance[:, 1], torch.zeros(2, dtype=batch.center.dtype))
        saved = (state.stability_distance[0].clone(), state.tilt_angle[0].clone())
        state.reset(torch.tensor([1]), batch, **reset, step=1)
        torch.testing.assert_close(state.stability_distance[0], saved[0])
        torch.testing.assert_close(state.tilt_angle[0], saved[1])
        self.assertFalse(state.stability_distance[1].any())
        self.assertFalse(state.tilt_angle[1].any())
        state.advance(batch, 2)
        torch.testing.assert_close(state.stability_distance[:, 1], batch.center.new_tensor([.03, 0.]))
        torch.testing.assert_close(state.tilt_angle[:, 1], batch.center.new_tensor([0., math.pi / 4]))

    def test_clearance_baseline_uses_each_initial_physical_overlap_and_stationary_reward_is_zero(self):
        state, batch, reset = fixture(3)
        # The three 10 cm blockers overlap the approach rectangle by 100%, 50%,
        # and 20%. Keep the vision snapshot unchanged to distinguish its OBB
        # estimates from the privileged physical reset geometry used by rewards.
        positions = batch.center.new_tensor([.15, .20, .23])
        delta = positions - batch.center[:, 1, 0]
        batch.center[:, 1, 0] = positions
        batch.corners[:, 1, :, 0] += delta[:, None]
        state.reset(torch.arange(3), batch, **reset, step=0)
        expected = torch.exp(batch.center.new_tensor([-1., -.5, -.2]))
        torch.testing.assert_close(state.initial_clearance_quality, expected)
        torch.testing.assert_close(state.clearance_quality, expected)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            torch.testing.assert_close(rewards.clearance(env), torch.zeros_like(expected))
            for step in (1, 2):
                state.advance(batch, step)
                torch.testing.assert_close(state.initial_clearance_quality, expected)
                torch.testing.assert_close(state.clearance_quality, expected)
                torch.testing.assert_close(rewards.clearance(env), torch.zeros_like(expected), atol=1.e-12, rtol=0)

    def test_clearance_rewards_initial_improvement_penalizes_worsening_and_returns_to_zero(self):
        state, batch, reset = fixture(3)
        delta = .2 - batch.center[:, 1, 0]
        batch.center[:, 1, 0] = .2
        batch.corners[:, 1, :, 0] += delta[:, None]
        state.reset(torch.arange(3), batch, **reset, step=0)
        initial = clone_batch(batch)
        baseline = state.initial_clearance_quality.clone()
        # Improve env 0, worsen env 1, and leave env 2 unchanged.
        shift = batch.center.new_tensor([.03, -.02, 0.])
        batch.center[:, 1, 0] += shift
        batch.corners[:, 1, :, 0] += shift[:, None]
        expected_quality = torch.exp(batch.center.new_tensor([-.2, -.7, -.5]))
        expected_payment = expected_quality - math.exp(-.5)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            for step in (1, 2):
                state.advance(batch, step)
                torch.testing.assert_close(state.clearance_quality, expected_quality)
                torch.testing.assert_close(rewards.clearance(env), expected_payment)
                torch.testing.assert_close(state.initial_clearance_quality, baseline)
            # The reference is episode-initial, not the previous step: maintaining
            # the same improved pose pays again, but returning to the start pays 0.
            self.assertGreater(float(expected_payment[0]), 0.)
            self.assertLess(float(expected_payment[1]), 0.)
            state.advance(initial, 3)
            torch.testing.assert_close(rewards.clearance(env), torch.zeros_like(baseline), atol=1.e-12, rtol=0)

    def test_clearance_denominator_stays_at_reset_area_when_blocker_tilts(self):
        state, batch, reset = fixture(2)
        shift = .2 - batch.center[:, 1, 0]
        batch.center[:, 1, 0] += shift
        batch.corners[:, 1, :, 0] += shift[:, None]
        state.reset(torch.arange(2), batch, **reset, step=0)
        initial_area = state.initial_blocker_area.clone()
        initial_quality = state.initial_clearance_quality.clone()
        # Tilt around local X: this changes the actual projected area while
        # preserving the X overlap fraction, exposing a moving-denominator bug.
        angle = math.pi / 3
        q = batch.rotation.new_tensor([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
        relative = batch.corners[0, 1] - batch.center[0, 1]
        batch.corners[0, 1] = batch.center[0, 1] + task.quat_apply(q.expand(8, -1), relative)
        batch.rotation[0, 1] = q
        current_area, overlap = task.projected_areas(batch.corners, state.blocker & state.active,
                                                    state.access_region)
        self.assertGreater(float(current_area[0].sum()), float(initial_area[0]))
        state.advance(batch, 1)
        expected = torch.exp(-state.cfg.clearance_decay *
                             (overlap.sum(-1) / initial_area).clamp(0., 1.))
        torch.testing.assert_close(state.clearance_quality, expected)
        torch.testing.assert_close(state.initial_blocker_area, initial_area)
        torch.testing.assert_close(state.initial_clearance_quality, initial_quality)
        self.assertLess(float(state.clearance_quality[0]), float(initial_quality[0]))
        torch.testing.assert_close(state.clearance_quality[1], initial_quality[1])

    def test_partial_reset_refreshes_only_selected_blocker_area_and_target_gate(self):
        state, batch, reset = fixture(4)
        batch.center[:, 0, 0] += .03
        batch.corners[:, 0, :, 0] += .03
        state.advance(batch, 1)
        self.assertFalse(state.target_protected.any())
        old_area, old_baseline = state.initial_blocker_area.clone(), state.initial_clearance_quality.clone()
        ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        scales = batch.corners.new_tensor([2., .5])
        batch.corners[ids, 1] = batch.center[ids, 1, None] + (
            batch.corners[ids, 1] - batch.center[ids, 1, None]) * scales[:, None, None]
        state.reset(ids, batch, **reset, step=1)
        torch.testing.assert_close(state.initial_blocker_area[ids], old_area[ids] * scales.square())
        torch.testing.assert_close(state.initial_blocker_area[untouched], old_area[untouched])
        torch.testing.assert_close(state.initial_clearance_quality[untouched], old_baseline[untouched])
        self.assertTrue(state.target_protected[ids].all())
        self.assertFalse(state.target_protected[untouched].any())

    def test_target_pose_gate_uses_success_tolerances_and_recovers_without_failure(self):
        state, initial, _ = fixture(5)
        batch = clone_batch(initial)
        position = state.cfg.target_translation_tolerance
        angle = state.cfg.target_rotation_tolerance
        displacements = batch.center.new_tensor([position - 1.e-6, position + 1.e-6, 0., 0., 0.])
        batch.center[:, 0, 0] += displacements
        batch.corners[:, 0, :, 0] += displacements[:, None]
        angles = batch.center.new_tensor([0., 0., angle - 1.e-6, angle + 1.e-6, 0.])
        batch.rotation[:, 0, 0] = torch.cos(angles / 2)
        batch.rotation[:, 0, 3] = torch.sin(angles / 2)
        state.advance(batch, 1)
        expected = torch.tensor([True, False, True, False, True])
        torch.testing.assert_close(state.target_protected, expected)
        self.assertFalse(state.failure.any())
        self.assertFalse(state.success.any())
        self.assertFalse(state.clearance_progress_valid[~expected].any())
        state.advance(initial, 2)
        self.assertTrue(state.target_protected.all())
        self.assertTrue(state.clearance_progress_valid.all())

    def test_unsorted_partial_reset_rebases_only_selected_clearance_rows(self):
        state, batch, reset = fixture(4)
        delta = .2 - batch.center[:, 1, 0]
        batch.center[:, 1, 0] = .2
        batch.corners[:, 1, :, 0] += delta[:, None]
        state.reset(torch.arange(4), batch, **reset, step=0)
        shift = batch.center.new_tensor([.03, .02, -.02, .04])
        batch.center[:, 1, 0] += shift
        batch.corners[:, 1, :, 0] += shift[:, None]
        state.advance(batch, 1)
        ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        before = {name: value[untouched].clone() for name, value in vars(state).items()
                  if isinstance(value, torch.Tensor) and value.shape[:1] == (4,)}
        positions = batch.center.new_tensor([.19, .25])
        delta = positions - batch.center[ids, 1, 0]
        batch.center[ids, 1, 0] = positions
        batch.corners[ids, 1, :, 0] += delta[:, None]
        selected = task.TransitionBatch(**{
            field.name: getattr(batch, field.name)[ids].clone() for field in dataclasses.fields(batch)})
        state.reset(ids, selected, **{name: value[ids] for name, value in reset.items()},
                    step=1, batch_is_selected=True)
        expected = torch.exp(batch.center.new_tensor([-.6, 0.]))
        torch.testing.assert_close(state.initial_clearance_quality[ids], expected)
        torch.testing.assert_close(state.clearance_quality[ids], expected)
        for name, saved in before.items():
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(state, name)[untouched], saved)
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            payment = rewards.clearance(env)
            torch.testing.assert_close(payment[ids], torch.zeros_like(expected))
            untouched_payment = payment[untouched].clone()
            state.advance(batch, 2)
            payment = rewards.clearance(env)
            torch.testing.assert_close(payment[ids], torch.zeros_like(expected), atol=1.e-12, rtol=0)
            torch.testing.assert_close(payment[untouched], untouched_payment)

    def test_history_current_previous_and_once_per_global_step(self):
        state, batch, _ = fixture()
        initial = state.history.clone()
        batch.sample.fill_(5.0)
        state.advance(batch, 1)
        torch.testing.assert_close(state.history[:, 1], initial[:, 0])
        torch.testing.assert_close(state.history[:, 0], batch.sample)
        snapshot = {k: v.clone() for k, v in vars(state).items() if isinstance(v, torch.Tensor)}
        batch.sample.fill_(8.0)
        state.advance(batch, 1)
        for key, value in snapshot.items():
            torch.testing.assert_close(getattr(state, key), value)
        state.advance(batch, 2)
        torch.testing.assert_close(state.history[:, 1], torch.full_like(batch.sample, 5.0))
        torch.testing.assert_close(state.history[:, 0], torch.full_like(batch.sample, 8.0))
        torch.testing.assert_close(state.elapsed_time, torch.full_like(state.elapsed_time, 0.2))

    def test_partial_reset_only_changes_selected_episode_rows(self):
        state, batch, reset = fixture(4)
        batch.corners.add_(torch.tensor([0.001, 0.0, 0.0], dtype=batch.corners.dtype))
        batch.center[..., 0].add_(0.001)
        state.advance(batch, 1)
        untouched = torch.tensor([0, 2])
        snapshot = {k: v[untouched].clone() for k, v in vars(state).items()
                    if isinstance(v, torch.Tensor) and v.shape[:1] == state.history.shape[:1]}
        ids = torch.tensor([1, 3])
        batch.sample.fill_(6.0)
        state.reset(ids, batch, **reset, step=1)
        for key, value in snapshot.items():
            torch.testing.assert_close(getattr(state, key)[untouched], value)
        torch.testing.assert_close(state.force_growth[ids], torch.zeros(2, dtype=state.force_growth.dtype))
        torch.testing.assert_close(state.elapsed_time[ids], torch.zeros(2, dtype=state.elapsed_time.dtype))
        torch.testing.assert_close(state.history[ids, 1], torch.zeros_like(state.history[ids, 1]))
        torch.testing.assert_close(state.history[ids, 0, 31:37], torch.zeros(2, 6, dtype=state.history.dtype))
        torch.testing.assert_close(state.history[ids, 0, 37:54], -torch.ones(2, 17, dtype=state.history.dtype))
        # Late settling/render physics without a completed control step cannot advance history.
        batch.sample.fill_(12.0)
        state.advance(batch, 1)
        torch.testing.assert_close(state.history[ids, 0, :31], torch.full((2, 31), 6.0, dtype=state.history.dtype))
        state.advance(batch, 2)
        torch.testing.assert_close(state.elapsed_time, torch.tensor([.2, .1, .2, .1], dtype=state.elapsed_time.dtype))

    def test_partial_reset_clears_only_selected_object_tilt_counters(self):
        state, batch, reset = fixture(4)
        batch.rotation[:, 1] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        state.advance(batch, 1)
        self.assertEqual(state.tilt_count[:, 1].tolist(), [1, 1, 1, 1])
        self.assertFalse(state.failure.any())
        selected, untouched = torch.tensor([1, 3]), torch.tensor([0, 2])
        saved = state.tilt_count[untouched].clone()
        batch.rotation[selected, 1] = batch.rotation.new_tensor([1., 0., 0., 0.])
        state.reset(selected, batch, **reset, step=1)
        self.assertFalse(state.tilt_count[selected].any())
        torch.testing.assert_close(state.tilt_count[untouched], saved)
        state.advance(batch, 2)
        self.assertTrue(state.failure[untouched].all())
        self.assertFalse(state.failure[selected].any())

    def test_partial_reset_clears_only_selected_support_timers_and_restores_progress_masks(self):
        state, batch, reset = fixture(4)
        state.cfg.support_loss_duration = .3
        batch.center[:, 1, 0] += 11.
        batch.corners[:, 1, :, 0] += 11.
        batch.center[:, 2, 2] += .04
        batch.corners[:, 2, :, 2] += .04
        state.advance(batch, 1)
        self.assertEqual(state.support_loss_count[:, 1].tolist(), [1] * 4)
        self.assertEqual(state.support_loss_count[:, 2].tolist(), [1] * 4)
        self.assertFalse(state.clearance_progress_valid.any())
        selected, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        saved = state.support_loss_count[untouched].clone()
        batch.center[selected, 1, 0] -= 11.
        batch.corners[selected, 1, :, 0] -= 11.
        batch.center[selected, 2, 2] -= .04
        batch.corners[selected, 2, :, 2] -= .04
        state.reset(selected, batch, **reset, step=1)
        self.assertFalse(state.support_loss_count[selected].any())
        torch.testing.assert_close(state.support_loss_count[untouched], saved)
        # A reset has no completed physical sample yet. Only the next step
        # may make the restarted rows eligible for positive progress again.
        self.assertFalse(state.clearance_progress_valid[selected].any())
        self.assertFalse(state.clearance_progress_valid[untouched].any())
        state.advance(batch, 2)
        self.assertTrue(state.clearance_progress_valid[selected].all())
        self.assertFalse(state.clearance_progress_valid[untouched].any())

    def test_object_cost_cache_tracks_current_twist_once_without_accumulating_pose_path(self):
        state, batch, _ = fixture(2)
        batch.velocity[0, 2, :3] = batch.velocity.new_tensor([.1, -.2, .3])
        batch.center[1, 2, 0] += .1
        batch.corners[1, 2, :, 0] += .1
        with patch.object(task, 'object_motion_cost', wraps=rewards.object_motion_cost) as evaluate:
            state.advance(batch, 1)
            state.advance(batch, 1)
            self.assertEqual(evaluate.call_count, 1)
        torch.testing.assert_close(state.object_motion_cost, batch.velocity.new_tensor([math.sqrt(.14), 0.]))
        self.assertFalse(hasattr(state, 'object_linear_velocity'))
        batch.velocity.zero_()
        state.advance(batch, 2)
        torch.testing.assert_close(state.object_motion_cost, batch.velocity.new_zeros(2))

    def test_partial_reset_refreshes_only_selected_object_costs_without_consuming_other_rows(self):
        state, batch, reset = fixture(4)
        batch.velocity[:, 2, 0] = batch.velocity.new_tensor([.1, .2, .3, .4])
        state.advance(batch, 1)
        previous = state.object_motion_cost.clone()
        ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        batch.velocity[ids, 2] = 0.
        batch.velocity[ids, 2, 5] = 2.
        state.reset(ids, batch, **reset, step=1)
        torch.testing.assert_close(state.object_motion_cost[untouched], previous[untouched])
        torch.testing.assert_close(state.object_motion_cost[ids], previous.new_full((2,), 2 * math.sqrt(.005)))

    def test_force_growth_uses_step_magnitude_increase_independently_of_control_dt(self):
        for dt in (.01, .02, .1):
            state, batch, _ = fixture(2)
            state.dt = dt
            batch.sensor_wrench[:, 0] = 2.
            batch.sensor_wrench[:, 3] = .1
            state.advance(batch, 1)
            torch.testing.assert_close(state.force_growth, batch.sensor_wrench.new_full((2,), 2.))
            torch.testing.assert_close(state.torque_growth, batch.sensor_wrench.new_full((2,), .1))
            state.advance(batch, 2)
            torch.testing.assert_close(state.force_growth, batch.sensor_wrench.new_zeros(2))
            torch.testing.assert_close(state.torque_growth, batch.sensor_wrench.new_zeros(2))

    def test_force_growth_does_not_charge_constant_load_release_or_direction_change(self):
        state, batch, _ = fixture(4)
        batch.sensor_wrench[:, 0] = 20.
        batch.sensor_wrench[:, 3] = 1.
        state.advance(batch, 1)
        batch.sensor_wrench[0, [0, 3]] = batch.sensor_wrench.new_tensor([30., 1.5])
        batch.sensor_wrench[2].zero_()
        batch.sensor_wrench[3] = batch.sensor_wrench.new_tensor([0., -20., 0., 0., -1., 0.])
        state.advance(batch, 2)
        torch.testing.assert_close(state.force_growth, batch.sensor_wrench.new_tensor([10., 0., 0., 0.]))
        torch.testing.assert_close(state.torque_growth, batch.sensor_wrench.new_tensor([.5, 0., 0., 0.]))
        env = SimpleNamespace(step_dt=state.dt)
        with patch.object(task, 'get_task_state', return_value=state):
            torch.testing.assert_close(rewards.motion_force_growth(env), batch.sensor_wrench.new_tensor([100., 0., 0., 0.]))
            torch.testing.assert_close(rewards.motion_torque_growth(env), batch.sensor_wrench.new_tensor([.25, 0., 0., 0.]))

    def test_only_designated_blocker_overlap_affects_clearance_and_success(self):
        state, batch, values = fixture(2)
        # Clear blocker, but leave target and another active object in the approach area.
        delta = torch.tensor([.4, 0., 0.], dtype=batch.center.dtype)
        batch.center[:, 1] += delta
        batch.corners[:, 1] += delta
        batch.center[:, 2, 0] = .1
        batch.corners[:, 2, :, 0] -= .14
        # Only env 1's blocker remains halfway in the access region.
        batch.center[1, 1, 0] = .2
        batch.corners[1, 1, :, 0] -= .32
        state.reset(torch.arange(2), batch, **values, step=0)
        for step in range(1, success_steps(state.cfg, state.dt) + 1):
            state.advance(batch, step)
        torch.testing.assert_close(state.blocker_overlap,
                                   torch.tensor([0., .005], dtype=batch.center.dtype))
        torch.testing.assert_close(state.clearance_quality,
                                   torch.tensor([1., math.exp(-.5)], dtype=batch.center.dtype))
        torch.testing.assert_close(state.success, torch.tensor([True, False]))

    def test_object_motion_includes_target_and_unobserved_distractor_but_excludes_blocker(self):
        state, batch, reset = fixture(3)
        reset['geometry_valid'][:, 2] = False
        state.reset(torch.arange(3), batch, **reset, step=0)
        rows = torch.arange(3)
        batch.velocity[rows, rows, 0] = .2
        state.advance(batch, 1)
        torch.testing.assert_close(state.object_motion_cost, batch.velocity.new_tensor([.2, 0., .2]))

    def test_blocker_linear_motion_remains_exempt_from_neighbor_cost(self):
        state, batch, _ = fixture(2)
        batch.velocity[:, 1] = batch.velocity.new_tensor([1., 2., 3., 4., 5., 6.])
        state.advance(batch, 1)
        torch.testing.assert_close(state.object_motion_cost, batch.velocity.new_zeros(2))

    def test_partial_reset_clears_only_selected_force_history_and_growth(self):
        state, batch, reset = fixture(4)
        batch.sensor_wrench[:, 0] = batch.sensor_wrench.new_tensor([1., 2., 3., 4.])
        batch.sensor_wrench[:, 3] = .5
        state.advance(batch, 1)
        ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        previous = state.previous_wrench[untouched].clone()
        force_growth = state.force_growth[untouched].clone()
        state.reset(ids, batch, **reset, step=1)
        torch.testing.assert_close(state.force_growth[ids], batch.sensor_wrench.new_zeros(2))
        torch.testing.assert_close(state.torque_growth[ids], batch.sensor_wrench.new_zeros(2))
        torch.testing.assert_close(state.previous_wrench[ids], batch.sensor_wrench.new_zeros(2, 6))
        torch.testing.assert_close(state.previous_wrench[untouched], previous)
        torch.testing.assert_close(state.force_growth[untouched], force_growth)
        state.advance(batch, 2)
        torch.testing.assert_close(state.force_growth[untouched], batch.sensor_wrench.new_zeros(2))
        torch.testing.assert_close(state.force_growth[ids], batch.sensor_wrench[ids, 0])

    def test_shelf_rotation_measurement_does_not_require_large_origin_corner_subtraction(self):
        state, batch, reset = fixture(1, dtype=torch.float32)
        batch.shelf_pose[:, :3] = batch.shelf_pose.new_tensor([1.e6, -1.e6, 1.e6])
        state.reset(torch.tensor([0]), batch, **reset, step=0)
        angle = math.radians(1.1)
        batch.shelf_pose[:, 3:] = batch.shelf_pose.new_tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        cost = rewards.shelf_motion_cost(batch.shelf_pose, batch.sensor_wrench.new_zeros(1, 6), state.initial_shelf_pose)
        torch.testing.assert_close(cost, batch.sensor_wrench.new_ones(1))

    def test_unsorted_partial_shelf_reset_restores_reference_only_for_selected_rows(self):
        state, batch, reset = fixture(4)
        batch.shelf_pose[:, 0] += batch.center.new_tensor([.001, .002, .003, .004])
        state.advance(batch, 1)
        ids, untouched = torch.tensor([3, 1]), torch.tensor([0, 2])
        preserved = {name: value[untouched].clone() for name, value in vars(state).items()
                     if isinstance(value, torch.Tensor) and value.shape[:1] == (4,)}
        batch.shelf_pose[ids, :3] = batch.center.new_tensor([[.7, -.2, .1], [-.4, .3, -.1]])
        selected = task.TransitionBatch(**{
            field.name: getattr(batch, field.name)[ids].clone() for field in dataclasses.fields(batch)})
        state.reset(ids, selected, **{name: value[ids] for name, value in reset.items()}, step=1, batch_is_selected=True)
        torch.testing.assert_close(state.initial_shelf_pose[ids], batch.shelf_pose[ids])
        for name, saved in preserved.items():
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(state, name)[untouched], saved)

    def test_sensor_center_growth_ignores_translation_and_rotation_without_hiding_local_moment_growth(self):
        state, batch, _ = fixture(3)
        batch.sensor_wrench[:, 1] = 10.
        batch.sensor_wrench[1, 5] = 1.
        state.advance(batch, 1)
        # Translating a sensor with constant local load must not create an r x F growth cost.
        batch.sensor_position[0, 0] = .01
        # Here a growing local moment is cancelled by r x F at the old initial origin.
        # The sensor-centre criterion must still see that actual moment growth.
        batch.sensor_position[1, 0] = -.01
        batch.sensor_wrench[1, 5] = 1.1
        # Rotate the sensor while keeping the same world force: local components change, norm does not.
        batch.sensor_rotation[2] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        batch.sensor_wrench[2, :3] = torch.tensor([10., 0., 0.])
        state.advance(batch, 2)
        torch.testing.assert_close(state.previous_wrench, batch.sensor_wrench)
        expected = torch.zeros(3, 2, dtype=batch.sensor_wrench.dtype)
        expected[1, 1] = .01  # Positive per-step 0.1 Nm increase, squared.
        torch.testing.assert_close(torch.stack((state.force_growth.square(), state.torque_growth.square()), -1), expected, atol=1.e-12, rtol=0)
        torch.testing.assert_close(state.force_growth, torch.zeros_like(state.force_growth))
        self.assertGreater(float(state.torque_growth[1]), 0.)

    def test_global_step_gaps_fail_explicitly(self):
        state, batch, _ = fixture(1)
        with self.assertRaisesRegex(RuntimeError, "once every"):
            state.advance(batch, 2)

    def test_invalid_state_failure_has_finite_learning_outputs(self):
        state, batch, _ = fixture(2)
        batch.corners[0, 1] = math.nan
        batch.sample[0, 0] = math.nan
        batch.state_valid[0] = False
        state.advance(batch, 1)
        self.assertTrue(state.failure[0])
        self.assertEqual(state.terminal_reward[0].item(), 0.)
        self.assertTrue(torch.isfinite(state.clearance_quality).all() and torch.isfinite(state.terminal_reward).all())
        self.assertTrue(torch.isfinite(state.history).all())

    def test_1024_env_advance_without_tensor_scalar_reads(self):
        state, batch, _ = fixture(1024, dtype=torch.float32)
        batch.corners[..., 0].add_(0.001)
        batch.center[..., 0].add_(0.001)
        def reject_scalar(*args, **kwargs):
            raise AssertionError("Per-step Tensor scalar read / host synchronization")
        with patch.object(torch.Tensor, "__bool__", reject_scalar), patch.object(torch.Tensor, "item", reject_scalar):
            state.advance(batch, 1)
        self.assertEqual(state.clearance_quality.shape, (1024,))
        torch.testing.assert_close(state.object_motion_cost, batch.velocity.new_zeros(1024))
        self.assertTrue(torch.isfinite(state.clearance_quality).all() and torch.isfinite(state.terminal_reward).all())


class NativeAdapterTests(unittest.TestCase):
    def test_singularity_uses_one_fresh_public_jacobian_per_completed_step_and_partial_reset_is_local(self):
        env, _, _ = native_fixture(3)
        task.reset_task(env)
        self.assertEqual(env.action_manager.jacobian_reads, 0)
        healthy = env.action_manager.current_jacobian.clone()
        env.action_manager.current_jacobian[1, 0, 0] = .25 * .02
        env.action_manager.current_jacobian[2, 5] = 0.
        env.common_step_counter = 1
        risk = terminations.failure(env, reason='singularity')
        state = env._contact_mdp
        expected_margin = rewards.singularity_margin(env.action_manager.current_jacobian)
        torch.testing.assert_close(state.singularity_margin, expected_margin)
        torch.testing.assert_close(risk, torch.tensor([False, False, True]))
        torch.testing.assert_close(state.failure, risk)
        torch.testing.assert_close(env.extras['terminated'], risk)
        expected_reward = rewards.singularity_penalty(expected_margin)
        torch.testing.assert_close(rewards.motion_singularity(env), expected_reward)
        obs.current_robot_sample(env)
        self.assertEqual(env.action_manager.jacobian_reads, 1)

        # Same native counter after auto-reset: only restarted rows lose their
        # old episode's metric. No new SDK read or other-row state change occurs.
        preserved = state.singularity_margin[:2].clone()
        env.action_manager.current_jacobian[2] = healthy[2]
        task.reset_task(env, torch.tensor([2]))
        torch.testing.assert_close(state.singularity_margin[:2], preserved)
        torch.testing.assert_close(state.singularity_margin[2],
                                   state.singularity_margin.new_tensor(state.cfg.motion_singularity_threshold))
        self.assertFalse(state.failure[2])
        self.assertEqual(env.action_manager.jacobian_reads, 1)
        torch.testing.assert_close(rewards.motion_singularity(env)[2], torch.tensor(0.))

        # Changing the public Jacobian after the physics update is reflected on
        # the next counter; the OSC's earlier private cached Jacobian is unused.
        env.action_manager.current_jacobian.copy_(healthy)
        env.common_step_counter = 2
        state = task.get_task_state(env)
        self.assertEqual(env.action_manager.jacobian_reads, 2)
        self.assertFalse(state.failure_reasons[:, 4].any())
        torch.testing.assert_close(rewards.motion_singularity(env), torch.zeros(3))

    def test_control_body_changes_robot_and_object_frames_without_changing_sensor_wrench(self):
        """Native body binding selects the wrist independently from the F/T link."""
        outputs = {}
        for selected in ("tcp", "hand"):
            env, _, _ = native_fixture(2)
            data = env.scene["robot"].data
            env.cfg.actions.arm_action.body_name = selected
            # Distinct rotated wrist and palm frames expose both reference-point
            # and reference-axis errors; root rotation also differs from world.
            data.root_quat_w[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
            data.body_link_pos_w[:, 1] += torch.tensor([.2, -.1, .1])
            data.body_link_quat_w[:, 1] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
            data.body_link_pos_w[:, 3] += torch.tensor([.3, .1, -.1])
            data.body_link_quat_w[:, 3] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
            task.reset_task(env)
            expected_body = env.scene["robot"].body_names.index(selected)
            self.assertEqual(env._contact_mdp.bindings.tcp_id, expected_body)
            self.assertEqual(env._contact_mdp.bindings.sensor_id, 2)

            # Exercise reset and the first live sensor transition with the same
            # independent SciPy frame oracle, not the production math helpers.
            for step in (0, 1):
                env.common_step_counter = step
                packed = obs.policy_observation(env)
                state = env._contact_mdp
                root_rotation = Rotation.from_quat(data.root_quat_w[:, [1, 2, 3, 0]].numpy())
                body_rotation = Rotation.from_quat(data.body_link_quat_w[:, expected_body][:, [1, 2, 3, 0]].numpy())
                expected_rotation = root_rotation.inv() * body_rotation
                expected_position = root_rotation.inv().apply(
                    (data.body_link_pos_w[:, expected_body] - data.root_pos_w).numpy())
                torch.testing.assert_close(packed[:, 124:127], torch.tensor(expected_position, dtype=packed.dtype))
                torch.testing.assert_close(rotation_matrix(packed[:, 127:131]),
                                           torch.tensor(expected_rotation.as_matrix(), dtype=packed.dtype))

                valid = state.geometry_valid
                frozen = state.geometry_b
                expected_geometry = torch.zeros_like(frozen)
                for row in range(env.num_envs):
                    rotation = expected_rotation[row].inv()
                    expected_geometry[row, valid[row], :3] = torch.tensor(rotation.apply(
                        frozen[row, valid[row], :3].numpy() - expected_position[row]), dtype=packed.dtype)
                    object_rotation = Rotation.from_quat(frozen[row, valid[row], 3:7][:, [1, 2, 3, 0]].numpy())
                    relative = torch.tensor((rotation * object_rotation).as_quat()[:, [3, 0, 1, 2]], dtype=packed.dtype)
                    relative = torch.where(relative[:, :1] < 0, -relative, relative)
                    expected_geometry[row, valid[row], 3:7] = relative
                    expected_geometry[row, valid[row], 7:] = frozen[row, valid[row], 7:]
                torch.testing.assert_close(packed[:, :100].reshape_as(frozen), expected_geometry, atol=1.e-6, rtol=1.e-5)
            outputs[selected] = packed.clone()

        self.assertFalse(torch.allclose(outputs["tcp"][:, :100], outputs["hand"][:, :100]))
        self.assertFalse(torch.allclose(outputs["tcp"][:, 124:131], outputs["hand"][:, 124:131]))
        # Identical physical sensor pose/distal loads keep SI-200-8 measured
        # sensor-frame force/torque identical despite the changed EEF frame.
        torch.testing.assert_close(outputs["tcp"][:, 131:137], outputs["hand"][:, 131:137], rtol=0, atol=0)

    @staticmethod
    def failure_fixture(termination_cfg=None):
        """Seven isolated failures, one safe row, and overlapping failure causes."""
        env, action, _ = native_fixture(9)
        if termination_cfg is not None:
            env.cfg.terminations = termination_cfg
        task.reset_task(env)
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[0, 0, 2] = -.02
        poses[1, 0, 3:] = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        env._contact_mdp.tilt_count[1, 1] = 1  # Native fixture has a .1 s control period.
        poses[2, 0, 0] = 2.
        poses[[3, 8], 0, 2] = .1
        env.sensors.ideal_wrench[[4, 8], 0] = 201.
        env.scene['shelf'].data.root_link_pose_w[[5, 8], 0] += .02
        action.applied_torque[6, 0] = math.nan
        env.common_step_counter = 1
        return env

    def test_named_failures_are_cached_independent_and_pay_zero_when_causes_overlap(self):
        env = self.failure_fixture()
        names = terminations.FAILURE_REASON_NAMES
        expected = torch.zeros(9, len(names), dtype=torch.bool)
        expected[[0, 2, 3, 8], names.index('object_support_lost')] = True
        expected[1, names.index('object_tilt')] = True
        expected[[5, 8], names.index('shelf_displacement')] = True
        expected[6, names.index('invalid_state')] = True
        with patch.object(task, '_read_transition', wraps=task._read_transition) as read, \
             patch.object(task, 'task_conditions', wraps=task.task_conditions) as conditions:
            actual = torch.stack([terminations.failure(env, reason=name) for name in names], -1)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(terminations.failure(env), expected.any(-1))
            self.assertFalse(terminations.success(env).any())
            torch.testing.assert_close(rewards.terminal(env) * env.step_dt, torch.zeros(9))
            obs.current_robot_sample(env)
            self.assertEqual(read.call_count, 1)
            self.assertEqual(conditions.call_count, 1)
        env.common_step_counter += 1
        torch.testing.assert_close(rewards.terminal(env), torch.zeros(9))
        torch.testing.assert_close(terminations.failure(env), expected.any(-1))

    def test_unknown_failure_name_rejects_configuration_before_accessing_environment(self):
        with self.assertRaisesRegex(ValueError, 'reason|failure'):
            terminations.failure(None, reason='shelf_contact_typo')

    def test_removed_hard_load_does_not_block_success_but_sensor_validity_is_retained(self):
        env, _, _ = native_fixture(2)
        task.reset_task(env)
        env.sensors.ideal_wrench[:, 0] = 201.
        env.common_step_counter = 1
        state = task.get_task_state(env)
        self.assertFalse(state.failure_reasons.any())
        self.assertFalse(state.failure.any())
        self.assertFalse(state.success.any())
        self.assertFalse(env.extras['terminated'].any())
        self.assertFalse(state.terminal_reward.any())
        with self.assertRaisesRegex(ValueError, 'reason|failure'):
            terminations.failure(env, reason='hard_load_or_overrange')
        env.scene['object_collection'].data.object_link_pose_w[:, 0, 0] = .8
        for step in range(2, success_steps(env.cfg.mdp, env.step_dt) + 2):
            env.common_step_counter = step
            state = task.get_task_state(env)
        self.assertTrue(state.success.all())
        self.assertTrue(env.extras['terminated'].all())
        self.assertFalse(state.failure.any())
        torch.testing.assert_close(state.terminal_reward, torch.full((2,), 250.))

    def test_disabled_success_term_does_not_mark_geometric_completion_as_native_termination(self):
        env, _, _ = native_fixture(1)
        env.cfg.terminations = {}
        task.reset_task(env)
        env.scene['object_collection'].data.object_link_pose_w[:, 0, 0] = .8
        for step in range(1, success_steps(env.cfg.mdp, env.step_dt) + 1):
            env.common_step_counter = step
            state = task.get_task_state(env)
        self.assertTrue(state.success.all())
        self.assertFalse(env.extras['terminated'].any())
        self.assertFalse(state.terminal_reward.any())
        self.assertFalse(state.terminal_paid.any())

    def test_native_velocity_failure_uses_resolved_manager_config_and_disabled_term_restores_success(self):
        native = native_velocity_termination()
        for enabled in (True, False):
            env, _, _ = native_fixture(3)
            names = env.cfg.actions.arm_action.joint_names + env.cfg.actions.hand_action.joint_names
            # env.cfg is deliberately unresolved; only the native manager owns usable IDs.
            unresolved = SimpleNamespace(name='robot', joint_names=names, joint_ids=dataclasses.MISSING)
            resolved = SimpleNamespace(name='robot', joint_names=names, joint_ids=list(range(12)))
            term = SimpleNamespace(func=native, time_out=False,
                                   params={'max_velocity': 3.0, 'asset_cfg': resolved})
            env.cfg.terminations = SimpleNamespace(
                success=SimpleNamespace(func=terminations.success, params={}, time_out=False),
                hand_velocity=SimpleNamespace(func=native, time_out=False,
                                              params={'max_velocity': 3.0, 'asset_cfg': unresolved})
                              if enabled else None)
            env.termination_manager = SimpleNamespace(
                active_terms=['hand_velocity'] if enabled else [], get_term_cfg=lambda name: term)
            task.reset_task(env)
            env.scene['object_collection'].data.object_link_pose_w[:, 0, 0] = .8
            env.scene['robot'].data.joint_vel[0, 0] = 3.01
            env.scene['robot'].data.joint_vel[1, 11] = -3.01
            for step in range(1, success_steps(env.cfg.mdp, env.step_dt) + 1):
                env.common_step_counter = step
                state = task.get_task_state(env)
            expected_failure = torch.tensor([enabled, enabled, False])
            torch.testing.assert_close(state.failure, expected_failure)
            torch.testing.assert_close(state.success, ~expected_failure)
            torch.testing.assert_close(state.terminal_reward,
                                       torch.where(expected_failure, 0., 250.))
            self.assertFalse(state.failure_reasons.any())
            self.assertTrue(env.extras['terminated'].all())
            # Every reader at the same counter sees the cached failure/success decision.
            with patch.object(env.termination_manager, 'get_term_cfg',
                              side_effect=AssertionError('Repeated velocity check')):
                self.assertIs(task.get_task_state(env), state)

    def test_native_termination_manager_logs_every_cause_and_retains_last_episode_after_partial_reset(self):
        # Run the checkout's real compute/reset methods and our real term config,
        # stubbing only base-manager setup that otherwise starts Isaac Sim.
        class ManagerBase:
            def __init__(self, cfg, env):
                self.cfg, self._env = cfg, env
                self._prepare_terms()

            num_envs = property(lambda self: self._env.num_envs)
            device = property(lambda self: self._env.device)

            def _resolve_common_term_cfg(self, *args, **kwargs):
                pass

        TermCfg = dataclasses.make_dataclass('TerminationTermCfg', [
            ('func', object), ('params', dict, dataclasses.field(default_factory=dict)), ('time_out', bool, False),
        ])
        root = Path(__file__).resolve().parents[3]
        manager_path = root / 'IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py'
        cfg_path = Path(task.__file__).parent.parent / 'shelf_manipulation_env_cfg.py'
        namespace = dict(torch=torch, ManagerBase=ManagerBase, ManagerTermBase=type('ManagerTermBase', (), {}),
                         TerminationTermCfg=TermCfg, configclass=lambda cls: cls, mdp=terminations,
                         joint_vel_out_of_manual_limit=native_velocity_termination(), MISSING=dataclasses.MISSING,
                         SceneEntityCfg=lambda name, **kwargs: SimpleNamespace(name=name, joint_ids=slice(None), **kwargs))
        for path, name in ((manager_path, 'TerminationManager'), (cfg_path, 'TerminationsCfg')):
            source = ast.parse(path.read_text())
            node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
            module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')],
                                                    level=0), node], type_ignores=[])
            exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        cfg = {name: value for name, value in vars(namespace['TerminationsCfg']).items()
               if isinstance(value, TermCfg)}
        expected_names = {'time_out', 'success', 'object_support_lost', 'object_tilt',
                          'shelf_displacement', 'invalid_state', 'singularity', 'hand_velocity'}
        self.assertEqual(set(cfg), expected_names)
        velocity_cfg = cfg['hand_velocity']
        self.assertIs(velocity_cfg.func, namespace['joint_vel_out_of_manual_limit'])
        self.assertEqual(velocity_cfg.params['max_velocity'], 3.0)
        self.assertFalse(velocity_cfg.time_out)
        self.assertIs(velocity_cfg.params['asset_cfg'].joint_names, dataclasses.MISSING)
        names = list(cfg)
        env = self.failure_fixture(termination_cfg=cfg)
        # Simulate native SceneEntityCfg resolution, skipped by the base-manager stub.
        velocity_cfg.params['asset_cfg'].joint_names = (
            env.cfg.actions.arm_action.joint_names + env.cfg.actions.hand_action.joint_names)
        velocity_cfg.params['asset_cfg'].joint_ids = list(range(12))
        env.scene['robot'].data.joint_vel[4, 11] = -3.01
        env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
        env.max_episode_length = 400
        manager = namespace['TerminationManager'](cfg, env)
        env.termination_manager = manager
        with patch.object(task, '_read_transition', wraps=task._read_transition) as read:
            dones = manager.compute()
        self.assertEqual(read.call_count, 1)
        torch.testing.assert_close(dones, env._contact_mdp.failure)
        self.assertFalse(manager.time_outs.any())
        expected = torch.zeros(9, len(names), dtype=torch.bool)
        for row, name in ((0, 'object_support_lost'), (1, 'object_tilt'), (2, 'object_support_lost'),
                          (3, 'object_support_lost'), (5, 'shelf_displacement'), (6, 'invalid_state'),
                          (8, 'shelf_displacement'), (8, 'object_support_lost'), (4, 'hand_velocity')):
            expected[row, names.index(name)] = True
        for column, name in enumerate(names):
            torch.testing.assert_close(manager.get_term(name), expected[:, column])
        logged = manager.reset(torch.tensor([0, 8]))
        self.assertEqual(set(logged), {'Episode_Termination/' + name for name in names})
        for column, name in enumerate(names):
            self.assertAlmostEqual(logged['Episode_Termination/' + name], expected[:, column].float().mean().item())

        # Actual task reset clears only selected current causes; native manager
        # statistics retain the completed episode's causes until its next done.
        ids, untouched = torch.tensor([0, 4, 8]), torch.tensor([1, 2, 3, 5, 6, 7])
        saved_reasons = env._contact_mdp.failure_reasons[untouched].clone()
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[ids, 0, 2] = .05
        env.sensors.ideal_wrench[ids, 0] = 1.
        env.scene['shelf'].data.root_link_pose_w[ids, 0] = 0.
        env.scene['robot'].data.joint_vel[ids] = 0.
        task.reset_task(env, ids)
        torch.testing.assert_close(env._contact_mdp.failure_reasons[untouched], saved_reasons)
        self.assertFalse(env._contact_mdp.failure_reasons[ids].any())
        self.assertEqual(env._contact_mdp.terminal_reward[ids].count_nonzero().item(), 0)
        manager.compute()
        torch.testing.assert_close(manager._last_episode_dones, expected)
        self.assertFalse(manager.dones[ids].any())
        for key, value in manager.reset(ids).items():
            self.assertEqual(value, logged[key])
        env.common_step_counter += 1
        manager.compute()
        torch.testing.assert_close(rewards.terminal(env), torch.zeros(9))

    def test_shared_geometry_cache_matches_generic_reset_and_keeps_neighbors(self):
        cached, _, _ = native_fixture(5)
        generic, _, _ = native_fixture(5)
        runtime = cached._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = cached.task_scene_data, None
        with patch.object(task, '_prepare_reset_geometry', wraps=task._prepare_reset_geometry) as prepare:
            task.reset_task(cached)
            self.assertEqual(prepare.call_args.args[1].numel(), 1)
            task.reset_task(generic)
            cached.common_step_counter = generic.common_step_counter = 1
            task.get_task_state(cached)
            task.get_task_state(generic)
            ids = torch.tensor([4, 1])
            cached.scene['robot'].data.joint_pos[ids] += .02
            generic.scene['robot'].data.joint_pos[ids] += .02
            runtime.mdp_reset_scene, runtime.mdp_reset_ids = cached.task_scene_data, ids
            with patch.object(task, '_validate_reset_data', side_effect=AssertionError('repeated shared validation')):
                task.reset_task(cached, ids)
            self.assertEqual(prepare.call_count, 2)  # One shared singleton and the generic reset.
        task.reset_task(generic, ids)
        for name, value in vars(generic._contact_mdp).items():
            if isinstance(value, torch.Tensor):
                with self.subTest(field=name):
                    torch.testing.assert_close(getattr(cached._contact_mdp, name), value)
        for name in ('object_ids', 'local_center', 'local_corners', 'upright'):
            torch.testing.assert_close(getattr(cached._contact_mdp.bindings, name),
                                       getattr(generic._contact_mdp.bindings, name))
        self.assertIsNone(runtime.mdp_reset_scene)

    def test_shared_geometry_refresh_invalidates_cache_and_external_resets_validate(self):
        env, _, _ = native_fixture()
        runtime = env._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        task.reset_task(env)
        previous = env._contact_mdp.geometry_b.clone()
        env.task_scene_data.geometry_b[:, :3, 0] += .03
        runtime.shared_template = object()
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        with patch.object(task, '_prepare_reset_geometry', wraps=task._prepare_reset_geometry) as prepare:
            task.reset_task(env)
            self.assertEqual(prepare.call_count, 1)
        self.assertFalse(torch.equal(env._contact_mdp.geometry_b, previous))
        env.task_scene_data.blocker_id[:] = env.task_scene_data.target_id
        # The producer's one-use marker was consumed. External mutations cannot
        # silently reuse the native shared-scene cache.
        with self.assertRaisesRegex(ValueError, 'distinct active target and blocker'):
            task.reset_task(env)

    def test_shared_geometry_marker_requires_the_same_reset_subset(self):
        env, _, _ = native_fixture()
        runtime = env._episode_reset = SimpleNamespace(shared_template=object())
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, None
        task.reset_task(env)
        runtime.mdp_reset_scene, runtime.mdp_reset_ids = env.task_scene_data, torch.tensor([0])
        env.task_scene_data.blocker_id[1] = env.task_scene_data.target_id[1]
        with self.assertRaisesRegex(ValueError, 'distinct active target and blocker'):
            task.reset_task(env, torch.tensor([1]))

    def test_native_reward_adapters_charge_current_object_velocity_and_step_load_growth(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        env.scene['object_collection'].data.object_com_vel_w[:, 3, 0] = .2
        env.sensors.ideal_wrench[:, 0] += 4.
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(rewards.object_collision(env), torch.full((3,), .2))
        torch.testing.assert_close(rewards.motion_force_growth(env), state.force_growth.square())
        torch.testing.assert_close(rewards.motion_torque_growth(env), state.torque_growth.square())
        torch.testing.assert_close(rewards.terminal(env) * env.step_dt, state.terminal_reward)
        env.scene['object_collection'].data.object_com_vel_w.zero_()
        env.common_step_counter = 2
        torch.testing.assert_close(rewards.object_collision(env), torch.zeros(3))
        torch.testing.assert_close(rewards.motion_force_growth(env), torch.zeros(3))
        torch.testing.assert_close(rewards.motion_torque_growth(env), torch.zeros(3))

    def test_named_observation_terms_preserve_contract_and_advance_once_in_any_order(self):
        env, action, _ = native_fixture()
        terms = (obs.initial_scene_geometry, obs.current_robot_sample, obs.previous_robot_sample)
        probed = tuple(term(env) for term in terms)
        self.assertEqual(tuple(value.shape for value in probed), ((3, 100), (3, 72), (3, 72)))
        self.assertEqual(int(torch.cat(probed, dim=-1).count_nonzero()), 0)
        self.assertEqual(env.sensor_read_count, 0)

        task.reset_task(env)
        initial_current = obs.current_robot_sample(env).clone()
        action.normalized_actions[:, 0] = .4
        env.common_step_counter = 1
        reads_before = env.sensor_read_count
        # Term evaluation order must not shift history or acquire sensors twice.
        previous = obs.previous_robot_sample(env)
        current = obs.current_robot_sample(env)
        geometry = obs.initial_scene_geometry(env)
        packed = torch.cat((geometry, current, previous), dim=-1)
        torch.testing.assert_close(packed, obs.policy_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(previous, initial_current, rtol=0, atol=0)
        torch.testing.assert_close(current[:, 54], torch.full((3,), .4))
        self.assertEqual(env.sensor_read_count, reads_before + 1)
        torch.testing.assert_close(geometry.reshape(3, 10, 10), task.geometry_in_eef_frame(
            env._contact_mdp.geometry_b, env._contact_mdp.geometry_valid, env._contact_mdp.tcp_pose_b
        ))

    def test_critic_scene_uses_true_obbs_only_for_initially_recognized_slots(self):
        env, _, _ = native_fixture()
        # Physical object 1 is active but absent from the initial vision list.
        env.task_scene_data.object_ids[:, 3] = 1
        probe = obs.critic_scene_geometry(env)
        self.assertEqual(probe.shape, (3, 100))
        self.assertEqual(int(probe.count_nonzero()), 0)
        self.assertEqual(env.sensor_read_count, 0)
        task.reset_task(env)
        actor = obs.initial_scene_geometry(env).reshape(3, 10, 10)
        critic = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        expected = torch.zeros_like(critic)
        expected[:, :3, 0] = torch.tensor([.3, -.1, .5])  # Target 2, Blocker 0, observed object 3.
        expected[:, :3, 2] = -.45
        expected[:, :3, 3] = 1.
        expected[:, :3, 7:] = .1
        torch.testing.assert_close(critic, expected, atol=1.e-6, rtol=1.e-5)
        self.assertFalse(torch.equal(actor, critic))  # Initial fitted size/center remain Actor-only.
        self.assertEqual(int(critic[:, 3:].count_nonzero()), 0)
        reads = env.sensor_read_count
        packed = torch.cat((obs.critic_scene_geometry(env), obs.current_robot_sample(env),
                            obs.previous_robot_sample(env)), dim=-1)
        self.assertEqual(packed.shape, (3, 244))
        torch.testing.assert_close(packed[:, -144:], obs.policy_observation(env)[:, -144:])
        self.assertEqual(env.sensor_read_count, reads)

    def test_native_observation_groups_keep_equal_width_and_share_only_robot_history(self):
        source = Path(__file__).resolve().parents[1] / 'contact_manipulation/shelf_manipulation_env_cfg.py'
        config = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'ObservationsCfg')
        functions = {}
        for group in (node for node in config.body if isinstance(node, ast.ClassDef)):
            functions[group.name] = [keyword.value.attr
                                     for term in group.body if isinstance(term, ast.Assign)
                                     and isinstance(term.value, ast.Call)
                                     and isinstance(term.value.func, ast.Name)
                                     and term.value.func.id == 'ObservationTermCfg'
                                     for keyword in term.value.keywords if keyword.arg == 'func']
        self.assertEqual(functions['PolicyCfg'], ['initial_scene_geometry', 'current_robot_sample', 'previous_robot_sample'])
        self.assertEqual(functions['CriticCfg'], ['critic_scene_geometry', 'current_robot_sample', 'previous_robot_sample'])
        env, _, _ = native_fixture()
        task.reset_task(env)
        policy = torch.cat([getattr(obs, name)(env) for name in functions['PolicyCfg']], dim=-1)
        critic = torch.cat([getattr(obs, name)(env) for name in functions['CriticCfg']], dim=-1)
        self.assertEqual(policy.shape, (3, 244))
        self.assertEqual(critic.shape, policy.shape)
        torch.testing.assert_close(policy, obs.policy_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(critic, obs.critic_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(critic[:, -144:], policy[:, -144:], rtol=0, atol=0)

    def test_critic_tracks_selected_motion_but_never_adds_hidden_or_newly_visible_objects(self):
        env, _, _ = native_fixture()
        env.task_scene_data.object_ids[:, 3] = 1
        task.reset_task(env)
        actor_before = obs.initial_scene_geometry(env).clone()
        before = obs.critic_scene_geometry(env).reshape(3, 10, 10).clone()
        objects = env.scene['object_collection'].data
        objects.object_link_pose_w[:, 2, 0] += .02
        objects.object_link_pose_w[:, 1, 1] += .2  # Hidden active object, never a Critic input.
        env.common_step_counter = 1
        reads = env.sensor_read_count
        after = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        expected = before.clone()
        expected[:, 0, 0] += .02
        torch.testing.assert_close(after, expected, atol=1.e-6, rtol=1.e-5)
        torch.testing.assert_close(obs.initial_scene_geometry(env), actor_before, rtol=0, atol=0)
        self.assertEqual(env.sensor_read_count, reads + 1)
        # A later visibility result must not replace the reset-time association.
        env.task_scene_data.observed_ids[:] = 1
        env.task_scene_data.geometry_valid[:] = False
        objects.object_link_pose_w[:, 3, 1] += .03
        env.common_step_counter = 2
        later = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        expected[:, 2, 1] += .03
        torch.testing.assert_close(later, expected, atol=1.e-6, rtol=1.e-5)
        self.assertEqual(int(later[:, 3:].count_nonzero()), 0)
        torch.testing.assert_close(obs.initial_scene_geometry(env), actor_before, rtol=0, atol=0)
        with patch.object(torch.Tensor, '__bool__', side_effect=AssertionError('tensor bool')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('tensor item')):
            obs.critic_scene_geometry(env)

    def test_critic_canonical_object_rotation_moves_offset_center_and_keeps_true_extent(self):
        env, _, _ = native_fixture()
        source = env.task_scene_data
        source.lower[2] = torch.tensor([0., -.05, -.03])
        source.upper[2] = torch.tensor([.1, .05, .07])
        upright = .3
        source.upright[2] = torch.tensor([math.cos(upright / 2), 0., 0., math.sin(upright / 2)])
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[:, 2, 3:] = source.upright[2]
        task.reset_task(env)
        initial = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        torch.testing.assert_close(initial[:, 0, :3], torch.tensor([.35, 0., -.43]).expand(3, -1),
                                   atol=1.e-6, rtol=1.e-5)
        torch.testing.assert_close(initial[:, 0, 3:7], torch.tensor([1., 0., 0., 0.]).expand(3, -1),
                                   atol=1.e-6, rtol=1.e-5)
        angle = .7
        poses[:, 2, 3:] = torch.tensor([math.cos((upright + angle) / 2), 0., 0.,
                                        math.sin((upright + angle) / 2)])
        env.common_step_counter = 1
        geometry = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        expected_center = torch.tensor([.3 + .05 * math.cos(angle), .05 * math.sin(angle), -.43])
        expected_rotation = torch.tensor([math.cos(angle / 2), 0., 0., math.sin(angle / 2)])
        torch.testing.assert_close(geometry[:, 0, :3], expected_center.expand(3, -1), atol=1.e-6, rtol=1.e-5)
        torch.testing.assert_close(geometry[:, 0, 3:7], expected_rotation.expand(3, -1), atol=1.e-6, rtol=1.e-5)
        torch.testing.assert_close(geometry[:, 0, 7:], torch.full((3, 3), .1), atol=1.e-6, rtol=1.e-5)

    def test_critic_geometry_uses_world_eef_pose_with_rotated_and_translated_robot_roots(self):
        env, _, _ = native_fixture()
        data = env.scene['robot'].data
        angles = torch.tensor([0., .4, -.3])
        data.root_quat_w[:, 0] = torch.cos(angles / 2)
        data.root_quat_w[:, 3] = torch.sin(angles / 2)
        data.root_pos_w += torch.tensor([[0., 0., 0.], [.06, -.02, .03], [-.05, .04, -.01]])
        data.body_link_pos_w[:, 1] = env.scene.env_origins + torch.tensor([.12, -.08, .5])
        data.body_link_quat_w[:, 1] = torch.tensor([math.cos(.1), 0., 0., -math.sin(.1)])
        task.reset_task(env)
        geometry = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        delta = np.array([[.3, 0., .05], [-.1, 0., .05], [.5, 0., .05]]) - np.array([.12, -.08, .5])
        expected_centers = torch.tensor(Rotation.from_euler('z', .2).apply(delta), dtype=geometry.dtype)
        expected_rotation = torch.tensor([math.cos(.1), 0., 0., math.sin(.1)], dtype=geometry.dtype)
        torch.testing.assert_close(geometry[:, :3, :3], expected_centers[None].expand(3, -1, -1),
                                   atol=1.e-6, rtol=1.e-5)
        torch.testing.assert_close(geometry[:, :3, 3:7], expected_rotation.expand(3, 3, -1),
                                   atol=1.e-6, rtol=1.e-5)

    def test_partial_reset_rebinds_only_selected_critic_slots_and_preserves_terminal_snapshot(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[:, 3, 0] += torch.tensor([.01, .02, .03])
        env.common_step_counter = 1
        completed_critic = obs.critic_scene_geometry(env).clone()
        terminal_critic = obs.critic_observation(env)
        terminal_critic_copy = terminal_critic.clone()
        completed_actor = obs.policy_observation(env).clone()
        terminal = env.extras['terminal_observation']
        terminal_copy = terminal.clone()
        # The new row sees only object 3; missing roles retain their reserved slots.
        source = env.task_scene_data
        source.geometry_valid[1] = False
        source.geometry_valid[1, 0] = True
        task.reset_task(env, torch.tensor([1]))
        reads = env.sensor_read_count
        restarted = obs.critic_scene_geometry(env).reshape(3, 10, 10)
        torch.testing.assert_close(restarted[[0, 2]].flatten(1), completed_critic[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(obs.policy_observation(env)[[0, 2]], completed_actor[[0, 2]], rtol=0, atol=0)
        self.assertEqual(int(restarted[1, :2].count_nonzero()), 0)
        self.assertEqual(int(restarted[1, 3:].count_nonzero()), 0)
        torch.testing.assert_close(restarted[1, 2, :3], torch.tensor([.52, 0., -.45]), atol=1.e-6, rtol=1.e-5)
        self.assertEqual(int(obs.previous_robot_sample(env)[1].count_nonzero()), 0)
        self.assertIs(env.extras['terminal_observation'], terminal)
        torch.testing.assert_close(terminal, terminal_copy, rtol=0, atol=0)
        self.assertIsNot(obs.critic_observation(env), terminal_critic)
        torch.testing.assert_close(terminal_critic, terminal_critic_copy, rtol=0, atol=0)
        self.assertEqual(env.sensor_read_count, reads)

    def test_named_observation_terms_preserve_terminal_snapshot_during_partial_reset(self):
        env, action, _ = native_fixture()
        task.reset_task(env)
        action.normalized_actions[:, 0] = .6
        env.common_step_counter = 1
        terms = (obs.initial_scene_geometry, obs.current_robot_sample, obs.previous_robot_sample)
        completed = torch.cat(tuple(term(env) for term in terms), dim=-1)
        terminal_snapshot = env.extras["terminal_observation"]
        torch.testing.assert_close(terminal_snapshot, completed, rtol=0, atol=0)

        task.reset_task(env, torch.tensor([1]))
        reads_after_reset = env.sensor_read_count
        restarted = torch.cat(tuple(term(env) for term in terms), dim=-1)
        torch.testing.assert_close(restarted, obs.policy_observation(env), rtol=0, atol=0)
        torch.testing.assert_close(restarted[[0, 2]], completed[[0, 2]], rtol=0, atol=0)
        self.assertEqual(int(restarted[1, 172:].count_nonzero()), 0)
        self.assertEqual(int(restarted[1, 154:172].count_nonzero()), 0)
        torch.testing.assert_close(env.extras["terminal_observation"], completed, rtol=0, atol=0)
        self.assertIs(env.extras["terminal_observation"], terminal_snapshot)
        self.assertEqual(env.sensor_read_count, reads_after_reset)

    def test_gaussian_actions_above_unit_range_survive_current_and_previous_observations(self):
        env, action, _ = native_fixture()
        task.reset_task(env)
        first = torch.linspace(-3., 3., 18).expand(3, -1).clone()
        action.normalized_actions.copy_(first)
        env.common_step_counter = 1
        current = obs.current_robot_sample(env)
        torch.testing.assert_close(current[:, -18:], first, rtol=0, atol=0)

        second = -2. * first
        action.normalized_actions.copy_(second)
        env.common_step_counter = 2
        packed = obs.policy_observation(env)
        self.assertEqual(packed.shape, (3, 244))
        torch.testing.assert_close(packed[:, 154:172], second, rtol=0, atol=0)
        torch.testing.assert_close(packed[:, 226:244], first, rtol=0, atol=0)
        self.assertTrue(task._read_transition(env, env._contact_mdp).state_valid.all())

    def test_native_actions_and_applied_effort_have_independent_finite_checks(self):
        env, action, _ = native_fixture(4)
        task.reset_task(env)
        action.normalized_actions[:] = torch.linspace(-2, 2, 18)
        action.normalized_actions[0, 0] = math.nan
        action.normalized_actions[0, 1] = math.inf
        action.normalized_actions[0, 2] = -math.inf
        action.applied_torque[1, 0] = math.nan
        batch = task._read_transition(env, env._contact_mdp)
        torch.testing.assert_close(batch.sample[:, -18:], torch.nan_to_num(
            action.normalized_actions, nan=0., posinf=1., neginf=-1.
        ))
        torch.testing.assert_close(batch.state_valid, torch.tensor([False, False, True, True]))

    def test_reset_diagnostics_survive_native_log_replacement_and_flush_once_at_same_step(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        state = task.get_task_state(env)
        reads = env.sensor_read_count
        diagnostics = {'Reset/robot_fallback_fraction': torch.tensor(.25),
                       'Reset/robot_retry_rounds': torch.tensor(2.)}
        env._episode_reset = SimpleNamespace(pending_reset_metrics=diagnostics)
        # Native _reset_idx replaces extras['log'] after the event has returned.
        native_log = {'Episode_Reward/clearance': torch.tensor(3.)}
        env.extras['log'] = native_log
        self.assertIs(task.get_task_state(env), state)
        self.assertIs(env.extras['log'], native_log)
        self.assertEqual(set(native_log), {*diagnostics, 'Episode_Reward/clearance'})
        for name, value in diagnostics.items():
            self.assertIs(native_log[name], value)
        self.assertIsNone(env._episode_reset.pending_reset_metrics)
        self.assertEqual(env.sensor_read_count, reads)
        # A later native log reset must not receive stale metrics from an old reset.
        env.extras['log'] = {'Episode_Termination/time_out': torch.tensor(.5)}
        task.get_task_state(env)
        self.assertEqual(set(env.extras['log']), {'Episode_Termination/time_out'})
        env._episode_reset.pending_reset_metrics = diagnostics
        del env.extras['log']
        task.get_task_state(env)
        self.assertEqual(set(env.extras['log']), set(diagnostics))
        self.assertEqual(env.sensor_read_count, reads)

    def test_completed_episode_metrics_count_only_ended_rows_and_survive_partial_reset(self):
        env, _, _ = native_fixture(4)
        task.reset_task(env)
        state = env._contact_mdp
        env._episode_reset = SimpleNamespace(pending_reset_metrics=None)
        env.termination_manager = SimpleNamespace(dones=torch.tensor([True, False, True, True]), active_terms=[])
        # Initial setup / manual reset without a completed physics transition
        # must not become an additional episode in these statistics.
        task.reset_task(env)
        self.assertFalse(state.completed_totals.any())
        env.common_step_counter = 1
        task.get_task_state(env)
        state.success[0] = True
        state.raw_goal_seen[[0, 2, 3]] = True
        state.unsupported_goal_seen[2] = True
        state.goal_failure_seen[3] = True
        env._episode_reset.pending_reset_metrics = {'Reset/existing': torch.tensor(2.)}
        task.reset_task(env, torch.tensor([3, 0, 1]))
        torch.testing.assert_close(state.completed_totals, torch.tensor([2, 1, 2, 0, 1]))
        self.assertTrue(state.raw_goal_seen[2])
        self.assertTrue(state.unsupported_goal_seen[2])
        self.assertFalse(state.raw_goal_seen[[3, 0, 1]].any())
        self.assertFalse(state.goal_failure_seen.any())
        metrics = env._episode_reset.pending_reset_metrics
        self.assertEqual(float(metrics['Reset/existing']), 2.)
        self.assertEqual(float(metrics['Task_Rate/valid_success']), .5)
        self.assertEqual(int(metrics['Task_Count/completed']), 2)
        # The previously untouched ended row contributes once when it resets.
        task.reset_task(env, torch.tensor([2]))
        torch.testing.assert_close(state.completed_totals, torch.tensor([3, 1, 3, 1, 1]))
        self.assertEqual(int(metrics['Task_Count/completed']), 2)  # Published snapshots remain immutable.
        self.assertEqual(int(env._episode_reset.pending_reset_metrics['Task_Count/completed']), 3)
        self.assertAlmostEqual(float(env._episode_reset.pending_reset_metrics['Task_Rate/valid_success']), 1 / 3)

    def test_missing_scene_and_sensor_provider_fail_before_native_binding(self):
        env, _, _ = native_fixture()
        del env.task_sensor_reader
        with self.assertRaisesRegex(RuntimeError, "SceneBatch.*task_sensor_reader"):
            task.reset_task(env)

    def test_sensor_provider_shape_and_boolean_contract_is_checked_on_reset(self):
        invalid_values = (("ideal_wrench", torch.zeros(3, 5)), ("wrench_valid", torch.ones(3, 1, dtype=torch.bool)),
                          ("wrench_valid", torch.ones(3)), ("tactile_force", torch.zeros(3, 16, 3)),
                          ("tactile_valid", torch.ones(3, 17)))
        for name, value in invalid_values:
            env, _, _ = native_fixture()
            setattr(env.sensors, name, value)
            with self.subTest(name=name, shape=value.shape), self.assertRaisesRegex(ValueError, f"SensorBatch.{name}"):
                task.reset_task(env)

    def test_valid_initial_observations_cannot_refer_to_inactive_or_duplicated_objects(self):
        for invalid_id in (-1, 1, 2):
            env, _, _ = native_fixture()
            env.task_scene_data.observed_ids[:, 0] = invalid_id
            with self.subTest(invalid_id=invalid_id), self.assertRaisesRegex(ValueError, "distinct active objects"):
                task.reset_task(env)

    def test_invalid_duplicate_association_cannot_hide_the_valid_target_slot(self):
        env, _, _ = native_fixture()
        source = env.task_scene_data
        source.geometry_valid[:, 0] = False
        source.observed_ids[:, 0] = source.target_id
        task.reset_task(env)
        state = task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b[:, 0], source.geometry_b[:, 1])
        self.assertTrue(state.geometry_valid[:, 0].all())

    def test_observation_dimension_probe_does_not_invent_geometry_and_first_step_requires_reset(self):
        env, _, _ = native_fixture()
        initial = task.get_task_state(env)
        self.assertFalse(initial.ready)
        self.assertEqual(int(initial.geometry_b.count_nonzero()), 0)
        self.assertEqual(env.sensor_read_count, 0)
        env.common_step_counter = 1
        with self.assertRaisesRegex(RuntimeError, "reset_task"):
            task.get_task_state(env)

    def test_initial_reset_keeps_role_slots_and_base_coordinates_independent_of_env_origin(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        initial_reads = env.sensor_read_count
        state = task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b[:, :3], env.task_scene_data.geometry_b[:, [1, 2, 0]])
        torch.testing.assert_close(state.tcp_pose_b[:, :3], torch.tensor([0., 0., .5]).expand(3, 3))
        torch.testing.assert_close(state.initial_center[:, :3, 0], torch.tensor([.3, -.1, .5]).expand(3, 3))
        self.assertEqual(env.sensor_read_count, initial_reads)
        self.assertEqual(int(state.history[:, 1].count_nonzero()), 0)
        torch.testing.assert_close(state.history[:, 0, 37:54], -torch.ones(3, 17))
        self.assertEqual(int(state.history[:, 0, 54:].count_nonzero()), 0)

    def test_one_native_step_keeps_gravity_and_inertia_reaction_and_updates_history_once(self):
        env, action, external = native_fixture()
        task.reset_task(env)
        initial_reads = env.sensor_read_count
        initial = env._contact_mdp.history[:, 0].clone()
        data = env.scene["robot"].data
        # The native reaction includes distal weight and inertial loads. They
        # remain measured sensor input rather than being subtracted afterwards.
        data.body_com_acc_w[:, 2:, 2] = 2.
        data.body_com_acc_w[:, 2:, 4] = 1.
        env.sensors.ideal_wrench[:, 2] += 4.
        env.sensors.ideal_wrench[:, 4] -= .16
        action.normalized_actions[:, 0] = .4
        env.sensors.tactile_force[:, 0, 2] = .011
        env.sensors.tactile_force[:, 1, 2] = .009
        env.sensors.tactile_valid[:, 2] = False
        env.common_step_counter = 1
        state = task.get_task_state(env)
        expected, _ = obs.calibrate_wrench(env.sensors.ideal_wrench)
        torch.testing.assert_close(state.history[:, 1], initial)
        torch.testing.assert_close(state.previous_wrench, expected)
        torch.testing.assert_close(state.history[:, 0, 31:37], expected / torch.tensor(task.SI_200_8_RANGES))
        self.assertFalse(torch.allclose(expected, external))
        torch.testing.assert_close(state.history[:, 0, 37:40], torch.tensor([1., 0., -1.]).expand(3, 3))
        torch.testing.assert_close(state.history[:, 0, 54], torch.full((3,), .4))
        self.assertEqual(env.sensor_read_count, initial_reads + 1)
        saved = state.history.clone()
        env.sensors.ideal_wrench[:, 0] += 10.
        self.assertIs(task.get_task_state(env), state)
        torch.testing.assert_close(state.history, saved)
        self.assertEqual(env.sensor_read_count, initial_reads + 1)

    def test_raw_measurement_clipping_and_invalid_mask_feed_both_observation_and_growth_rewards(self):
        env, _, _ = native_fixture(3)
        task.reset_task(env)
        env.sensors.ideal_wrench[:] = torch.tensor([
            [1.e20, -1.e20, 1.e20, 1.e20, -1.e20, 1.e20],
            [math.nan, 1., 2., 3., 4., 5.],
            [.14, -.16, 3., .011, -.014, .2],
        ])
        env.sensors.wrench_valid[1] = False
        env.common_step_counter = 1
        state = task.get_task_state(env)
        measured, exceeded = obs.calibrate_wrench(env.sensors.ideal_wrench)
        expected = measured.clone()
        expected[1] = 0.
        torch.testing.assert_close(state.previous_wrench, expected)
        torch.testing.assert_close(state.history[:, 0, 31:37], expected / torch.tensor(task.SI_200_8_RANGES))
        growth = expected.reshape(3, 2, 3).norm(dim=-1)
        torch.testing.assert_close(rewards.motion_force_growth(env), growth[:, 0].square())
        torch.testing.assert_close(rewards.motion_torque_growth(env), growth[:, 1].square())
        self.assertTrue(state.failure_reasons[1, state.invalid_reason_index])
        self.assertTrue(torch.isfinite(state.history[:, 0, 31:37]).all())
        self.assertTrue(exceeded[0].all())
        self.assertLessEqual(float(state.history[:, 0, 31:37].abs().max()), 1.)

    def test_partial_reset_preserves_other_environment_state_and_completed_terminal_observation(self):
        env, action, _ = native_fixture()
        task.reset_task(env)
        self.assertNotIn("terminal_env_ids", env.extras)
        action.normalized_actions[:, 0] = .6
        action.applied_torque[1, 0] = math.nan
        env.common_step_counter = 1
        state = task.get_task_state(env)
        final_observation = env.extras["terminal_observation"].clone()
        self.assertTrue(env.extras["terminated"][1])
        self.assertEqual(final_observation.shape, (3, 244))
        self.assertEqual(env.extras["terminal_env_ids"].numel(), 0)
        self.assertEqual(float(state.terminal_reward[1]), 0.)
        untouched = torch.tensor([0, 2])
        snapshot = {key: value[untouched].clone() for key, value in vars(state).items()
                    if isinstance(value, torch.Tensor) and value.shape[:1] == (env.num_envs,)}
        action.applied_torque[1, 0] = 0.
        task.reset_task(env, torch.tensor([1]))
        for key, expected in snapshot.items():
            torch.testing.assert_close(getattr(state, key)[untouched], expected)
        torch.testing.assert_close(env.extras["terminal_observation"], final_observation)
        torch.testing.assert_close(env.extras["terminal_env_ids"], torch.tensor([1]))
        self.assertEqual(int(state.history[1, 1].count_nonzero()), 0)
        self.assertEqual(int(state.history[1, 0, 54:].count_nonzero()), 0)
        self.assertFalse(state.failure[1])
        env.common_step_counter = 2
        task.get_task_state(env)
        self.assertEqual(env.extras["terminal_env_ids"].numel(), 0)
        torch.testing.assert_close(state.elapsed_time, torch.tensor([.2, .1, .2], dtype=state.elapsed_time.dtype))

    def test_current_object_ground_truth_changes_rewards_but_not_frozen_actor_geometry(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        state = task.get_task_state(env)
        initial_geometry = state.geometry_b.clone()
        object_data = env.scene["object_collection"].data
        object_data.object_link_pose_w[:, 3, 0] += .01
        object_data.object_com_pos_w[:, 3, 0] += .01
        object_data.object_com_vel_w[:, 3, 0] = .2
        # Subsequent edits to the producer are not accepted until the next reset.
        env.task_scene_data.geometry_b[:, :3, 0] += 100.
        env.common_step_counter = 1
        task.get_task_state(env)
        torch.testing.assert_close(state.geometry_b, initial_geometry)
        torch.testing.assert_close(state.object_motion_cost, torch.full((3,), .2))
        torch.testing.assert_close(rewards.object_collision(env), torch.full((3,), .2))
        actor_geometry = env.extras["terminal_observation"][:, :100].reshape(3, 10, 10)
        torch.testing.assert_close(actor_geometry, task.geometry_in_eef_frame(
            initial_geometry, state.geometry_valid, state.tcp_pose_b
        ))

    def test_overrange_is_diagnostic_while_nonfinite_effort_and_shelf_motion_fail_affected_env(self):
        env, action, _ = native_fixture(4)
        task.reset_task(env)
        env.sensors.ideal_wrench[0, 0] = 201.
        action.applied_torque[1, 0] = math.nan
        env.scene["shelf"].data.root_link_pose_w[2, 0] += .02
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([False, True, True, False]))
        torch.testing.assert_close(state.terminal_reward, torch.zeros(4))
        self.assertTrue(torch.isfinite(env.extras["terminal_observation"]).all())

    def test_shelf_reference_changes_only_on_selected_reset(self):
        env, _, _ = native_fixture(4)
        task.reset_task(env)
        state = env._contact_mdp
        initial = state.initial_shelf_pose.clone()
        env.scene['shelf'].data.root_link_pose_w[1, 0] += .02
        env.common_step_counter = 1
        task.get_task_state(env)
        self.assertTrue(state.failure[1])
        task.reset_task(env, torch.tensor([1]))
        torch.testing.assert_close(state.initial_shelf_pose[[0,2,3]], initial[[0,2,3]])
        torch.testing.assert_close(state.initial_shelf_pose[1], env.scene['shelf'].data.root_link_pose_w[1])
        self.assertFalse(state.failure[1])

    def test_nonfinite_robot_measurement_fails_and_terminal_observation_remains_finite(self):
        env, _, _ = native_fixture()
        task.reset_task(env)
        env.scene["robot"].data.joint_pos[1, 3] = math.nan
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(state.failure, torch.tensor([False, True, False]))
        self.assertTrue(torch.isfinite(env.extras["terminal_observation"]).all())
        self.assertTrue(torch.isfinite(state.clearance_quality).all() and torch.isfinite(state.terminal_reward).all())

    def test_native_adapter_hot_path_does_not_read_tensor_scalars(self):
        env, _, _ = native_fixture(32)
        task.reset_task(env)
        env.common_step_counter = 1
        def reject_scalar(*args, **kwargs):
            raise AssertionError("Per-step Tensor scalar read / host synchronization")
        with patch.object(torch.Tensor, "__bool__", reject_scalar), patch.object(torch.Tensor, "item", reject_scalar):
            state = task.get_task_state(env)
        self.assertEqual(state.history.shape, (32, 2, 72))
        self.assertEqual(env.extras["terminal_observation"].shape, (32, 244))

@wp.kernel(enable_backward=False)
def _layout_math_probe(data: wp.array2d(dtype=wp.float64), regions: wp.array2d(dtype=wp.float64),
                       gap: wp.float64, result: wp.array2d(dtype=wp.float64)):
    row = wp.tid()
    x,y,a,hx,hy = data[row,0],data[row,1],data[row,2],data[row,3],data[row,4]
    ox,oy,oa,ohx,ohy = data[row,5],data[row,6],data[row,7],data[row,8],data[row,9]
    first = task._layout_quad(x,y,a,hx,hy)
    second = task._layout_quad(ox,oy,oa,ohx,ohy)
    result[row,0] = wp.float64(task._layout_boxes_clear(x,y,wp.cos(a),wp.sin(a),hx,hy,
        ox,oy,wp.cos(oa),wp.sin(oa),ohx,ohy,gap))
    result[row,1] = task._layout_distance(first,second)
    result[row,2] = task._layout_area(first,wp.vec4d(regions[row,0],regions[row,1],regions[row,2],regions[row,3]))


class EpisodeLayoutTests(unittest.TestCase):
    @staticmethod
    def legacy_rules(**kwargs):
        """Keep rejection-packer regression tests separate from the new generator."""
        legacy = dict(generation_mode="legacy", all_asset_types=True,
                      min_shelf_occupancy=.20, min_region_occupancy=.10,
                      min_x_coverage=.55, min_y_coverage=.65,
                      min_neighbor_fraction=.75)
        legacy.update(kwargs)
        return task.SpawnRules(**legacy)

    def test_packed_path_family_preserves_motion_order_pivots_and_random_stream(self):
        import random
        geometry = task.ObjectGeometry("box", (-.08, -.04, 0.), (.08, .04, .12))
        rules = self.legacy_rules()
        for seed in range(30):
            pose = task.PlanarPose(-.7 + seed*.003, .015*math.sin(seed), seed*.57)
            access = (-.77, -.5, -.1, .1)
            first, second = random.Random(seed), random.Random(seed)
            oracle = task._clearance_candidates(geometry, pose, access, rules, first)
            packed = task._clearance_candidates(geometry, pose, access, rules, second, packed=True)
            self.assertEqual(len(packed), len(oracle))
            self.assertEqual(first.random(), second.random())
            for index, expected in enumerate(oracle):
                self.assertEqual(packed[index], expected)


    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.cache = tempfile.TemporaryDirectory(prefix='contact_layout_tests_')
        task.wp.init()
        cls.previous_cache = task.wp.config.kernel_cache_dir
        task.wp.config.kernel_cache_dir = cls.cache.name

    @classmethod
    def tearDownClass(cls):
        task.wp.config.kernel_cache_dir = cls.previous_cache
        cls.cache.cleanup()

    @staticmethod
    def catalog(copies=3, width_scale=1.1):
        sizes = ((.05, .05, .14), (.085, .06, .19), (.075, .10, .12),
                 (.065, .065, .11), (.10, .08, .17), (.055, .055, .15),
                 (.10, .10, .14), (.06, .16, .21), (.045, .095, .18), (.065, .065, .10))
        # Ten synthetic objects must cover the new 20% minimum without copies.
        return tuple(task.ObjectGeometry(f"type_{kind}_{copy}",
                                        (.013-width_scale*x/2, -.02-y/2, -.007),
                                        (.013+width_scale*x/2, -.02+y/2, z-.007), f"type_{kind}")
                     for kind, (x, y, z) in enumerate(sizes) for copy in range(copies))

    def test_fixed_ten_body_pool_meets_whole_shelf_density_without_copies(self):
        import random
        from collections import Counter
        catalog, rules = self.catalog(copies=1), self.legacy_rules()
        sample = task.sample_scene(catalog, rules, random.Random(1))
        task.validate_sample(sample, catalog, rules)
        metrics = task.scene_metrics(sample, catalog, rules)
        counts = Counter(catalog[index].asset_name for index in sample.active_ids)
        self.assertGreaterEqual(metrics.shelf_occupancy, .20)
        self.assertGreaterEqual(min(metrics.region_occupancies), .10)
        self.assertEqual(counts[catalog[sample.target_id].asset_name], 1)
        self.assertNotEqual(catalog[sample.target_id].asset_name, catalog[sample.blocker_id].asset_name)
        self.assertEqual(max(counts.values()), 1)
        self.assertEqual(len(sample.active_ids), 10)
        self.assertEqual(len(counts), 10)
        self.assertGreaterEqual(min(counts.values()), 1)
        for index in sample.active_ids:
            self.assertAlmostEqual(sample.positions[index][2] + catalog[index].lower[2], rules.shelf_top)

    def test_device_layout_feeds_physics_and_collision_boxes_without_host_pose_reads(self):
        import random
        catalog, rules = self.catalog(), self.legacy_rules()
        sample = task.sample_scene(catalog,rules,random.Random(1))
        # Catalog storage, active slots, and packing order need not be identical.
        order = sample.active_ids[::-1]
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            lower = torch.tensor([g.lower for g in catalog],device=device)
            upper = torch.tensor([g.upper for g in catalog],device=device)
            upright = torch.zeros((len(catalog),4),device=device)
            upright[:,0] = 1
            default = torch.zeros((len(catalog),13),device=device)
            default[:,0],default[:,3] = 5.,1.
            expected = task.pack_episode_states(default,[sample],lower,upper,upright)
            resident = dataclasses.replace(sample,planar_ids=order,planar_poses=torch.tensor(
                [(*sample.positions[i][:2],sample.yaws[i]) for i in order],device=device,dtype=torch.float64))
            cache = {}
            model = SimpleNamespace(base_origin=np.array([.1,.2,0.]),
                shelf_boxes=[SimpleNamespace(center=np.zeros(3),rotation=np.eye(3),half_extents=np.ones(3),path='shelf')],
                _tensor_constants=lambda *args: cache)
            env = SimpleNamespace(robot_pose_model=model,spawn_geometries=catalog)
            old_boxes = task._scene_collision_boxes(env,sample)
            with patch.object(torch.Tensor,'cpu',side_effect=AssertionError('host pose transfer')), \
                 patch.object(torch.Tensor,'numpy',side_effect=AssertionError('host pose transfer')):
                proposed = task._episode_proposals([resident],len(catalog),device,torch.float32)
                actual = task.pack_episode_states(default,[resident],lower,upper,upright,proposals=proposed)
                centers,rotations,half,valid = task._scene_boxes_tensor(env,[resident],proposed)
                # The runtime catalog supplies heights without reading per-object metadata.
                resident_without_positions = dataclasses.replace(resident, positions={})
                cached = task._episode_proposals([resident_without_positions],len(catalog),device,torch.float32,
                    support_heights=lower.new_tensor([rules.shelf_top-g.lower[2] for g in catalog]))
                for a,b in zip(cached,proposed):
                    torch.testing.assert_close(a,b,rtol=0,atol=0)
            for a,b in zip(actual,expected):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            for tensor,column in ((centers,0),(rotations,1),(half,2)):
                torch.testing.assert_close(tensor[0],torch.tensor(np.array([b[column] for b in old_boxes]),
                    device=device,dtype=tensor.dtype),rtol=2e-6,atol=2e-7)
            self.assertTrue(valid.all())

    def test_too_small_selected_species_capacity_is_rejected_before_layout(self):
        import random
        with self.assertRaisesRegex(RuntimeError, "cannot meet whole-shelf density"):
            task.sample_scene(self.catalog(copies=4, width_scale=1.), self.legacy_rules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35), random.Random(0))

    def test_layout_rejection_can_retry_composition_while_preserving_episode_roles(self):
        import random
        catalog, rules, rng = self.catalog(copies=4, width_scale=1.), self.legacy_rules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35), random.Random(0)
        target = rng.randrange(len(catalog))
        blocker = rng.choice([i for i, geometry in enumerate(catalog)
                              if geometry.asset_name != catalog[target].asset_name])
        with self.assertRaisesRegex(RuntimeError, "cannot meet whole-shelf density"):
            task.sample_scene(catalog, rules, rng, target_id=target, blocker_id=blocker)
        sample = task.sample_scene(catalog, rules, rng, target_id=target, blocker_id=blocker)
        self.assertEqual((sample.target_id, sample.blocker_id), (target, blocker))
        task.validate_sample(sample, catalog, rules)

    def test_front_start_region_respects_rotated_blocker_and_elbow_geometry(self):
        geometry = (task.ObjectGeometry("target", (-.025, -.025, 0), (.025, .025, .1)),
                    task.ObjectGeometry("blocker", (-.08, -.06, 0), (.12, .04, .1)))
        sample = task.SceneSample((0, 1), 0, 1,
                                  {0: (-.85, .1, 1.05), 1: (-.75, .1, 1.05)},
                                  {0: 0., 1: math.pi/2}, (-.8, -.5, .04, .16), (-.75, -.1))
        bounds = task.robot_start_region(sample, geometry,
                                          dataclasses.replace(self.legacy_rules(), x_range=(-.9, -.5)),
                                          task.RobotStartRules())
        np.testing.assert_allclose(bounds, (-.44, -.30, .02, .18, 1.25, 1.33))
        self.assertGreater(task.elbow_above_chord((0,0,1), (.3,0,1.4), (.6,0,1)), 0)
        self.assertLess(task.elbow_above_chord((0,0,1), (.3,0,.6), (.6,0,1)), 0)

    def test_batched_path_selection_matches_scalar_sweeps(self):
        import random
        catalog = self.catalog()
        rules = dataclasses.replace(self.legacy_rules(), x_range=(-.4,.4), y_range=(-.4,.4))
        rng = random.Random(918)
        requests, expected = [], []
        for k in range(8):
            geometry = catalog[4*k]
            initial = task.PlanarPose(.05,rng.uniform(-.15,.15),rng.uniform(-math.pi,math.pi))
            access = (-.01,.4,-.06,.06)
            target = task.rectangle_polygon((-.12,-.02,-.05,.05))
            candidates = task._clearance_candidates(geometry,initial,access,rules,rng)
            good = [task._path_supported_clear(geometry,path,(target,),rules) for path in candidates]
            requests.append(task._PathRequest(geometry,candidates,target))
            expected.append(next((p for p,v in zip(candidates,good) if v),None))
            # Check every candidate, including rejected mid-rotation collisions.
            requests.extend(task._PathRequest(geometry,[p],target) for p in candidates)
            expected.extend(p if v else None for p,v in zip(candidates,good))
        bar = task.ObjectGeometry('bar',(-.10,-.01,0.),(.10,.01,.1))
        path = (task.PlanarMotion(task.PlanarPose(0.,0.,0.),task.PlanarPose(0.,0.,math.pi/2),(0.,0.)),)
        post = task.rectangle_polygon((.052,.065,.052,.065))
        requests.append(task._PathRequest(bar,[path],post))
        expected.append(None)  # Clear endpoints, but collision halfway through the pivot.
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            self.assertEqual(task._choose_clearance_paths(requests,rules,device),expected)

    def test_pivot_sweep_detects_mid_motion_obstacle(self):
        # Endpoints clear a thin post, but the rotating long bar crosses it.
        bar = task.ObjectGeometry("bar", (-.10, -.01, 0), (.10, .01, .1))
        motion = task.PlanarMotion(task.PlanarPose(0., 0., 0.), task.PlanarPose(0., 0., math.pi/2), (0., 0.))
        obstacle = task.rectangle_polygon((.052, .065, .052, .065))
        self.assertTrue(task.polygons_separated(task.footprint_polygon(bar, (0., 0.), 0.), obstacle))
        self.assertTrue(task.polygons_separated(task.footprint_polygon(bar, (0., 0.), math.pi/2), obstacle))
        self.assertTrue(any(not task.polygons_separated(sweep, obstacle)
                            for sweep in task.motion_sweeps(bar, motion)))

    def test_device_geometry_matches_independent_scalar_oracles_and_gap_boundaries(self):
        rng = np.random.default_rng(88)
        data = rng.uniform(-1.,1.,(128,10))
        data[:,[3,4,8,9]] = rng.uniform(.01,.14,(128,4))
        data[:4] = [0.,0.,0.,.03,.01,.06,0.,0.,.03,.01]
        data[:4,5] += np.array([-.00001,0.,.003,.003-2.e-9])
        regions = np.tile([-.3,.3,-.3,.3],(len(data),1))
        pairs=[]
        for row in data:
            a=task.ObjectGeometry('a',(-row[3],-row[4],0.),(row[3],row[4],.1))
            b=task.ObjectGeometry('b',(-row[8],-row[9],0.),(row[8],row[9],.1))
            pairs.append((task.footprint_polygon(a,tuple(row[:2]),float(row[2])),
                          task.footprint_polygon(b,tuple(row[5:7]),float(row[7]))))
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for gap in (0.,.003,.017):
                output=wp.empty((len(data),3),dtype=wp.float64,device=device)
                wp.launch(_layout_math_probe,dim=len(data),inputs=[wp.array(data,dtype=wp.float64,device=device),
                    wp.array(regions,dtype=wp.float64,device=device),gap,output],device=device)
                actual=output.numpy()
                expected=np.array([[task.polygons_separated(a,b,gap),task.polygon_distance(a,b),
                    task.overlap_area(a,task.rectangle_polygon(region))] for (a,b),region in zip(pairs,regions)])
                np.testing.assert_array_equal(actual[:,0],expected[:,0])
                # Distance helper is used only after collision separation.
                separated=actual[:,0].astype(bool)
                np.testing.assert_allclose(actual[separated,1],expected[separated,1],atol=1.e-12,rtol=1.e-12)
                np.testing.assert_allclose(actual[:,2],expected[:,2],atol=1.e-12,rtol=1.e-12)

    def test_local_repair_preserves_roles_and_all_constraints(self):
        import random
        # Keep a dense multi-copy case to exercise eviction/repair, even though
        # the current ten-object experiment needs fewer placements.
        catalog = self.catalog(width_scale=1.)
        rules = self.legacy_rules(min_objects=4, max_objects=30, all_asset_types=False, min_shelf_occupancy=.35)
        stats={}
        result=task.sample_scenes(catalog,rules,[random.Random(50+i) for i in range(8)],
                                 [(0,4)]*8,[None]*8,statistics=stats)
        self.assertGreater(stats['evictions'],0)
        for sample in result:
            self.assertEqual((sample.target_id,sample.blocker_id),(0,4))
            task.validate_sample(sample,catalog,rules)

    def test_degenerate_sweep_edges_do_not_change_packing(self):
        import random
        catalog, rules = self.catalog(), self.legacy_rules()
        reference = task.sample_scene(catalog,rules,random.Random(1))
        plans=[]
        for seed in range(4):
            generator=task._scene_proposals(catalog,rules,random.Random(seed),reference.target_id,
                reference.blocker_id,reference.active_ids)
            plans.append(next(generator))
            generator.close()
        modified=[]
        for plan in plans:
            sweeps=[]
            for polygon in plan[5]:
                start, following=np.asarray(polygon[:2])
                tiny=tuple(start+(following-start)*1.e-10)
                sweeps.append((polygon[0],polygon[0],tiny,*polygon[1:]))
            modified.append((*plan[:5],tuple(sweeps),plan[-1]))
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            result=task._pack_layouts(plans+modified,catalog,rules,device)
            self.assertTrue(any(row is not None for row in result[:4]))
            for before,after in zip(result[:4],result[4:]):
                if before is None:
                    self.assertIsNone(after)
                else:
                    np.testing.assert_array_equal(before,after)

    def test_compiled_rotation_envelopes_match_scalar_hulls(self):
        box = task.ObjectGeometry('box', (-.10, -.02, 0.), (.10, .02, .1))
        for angle in (-math.pi / 2, math.pi / 4, math.pi):
            for pivot in (None, (.10, -.02)):
                motion = task.PlanarMotion(task.PlanarPose(.01, -.03, .2),
                                          task.PlanarPose(.04, .07, .2 + angle), pivot)
                previous = task._pose_polygon(box, motion.start)
                origin = pivot or (motion.start.x, motion.start.y)
                radius = max(math.hypot(x - origin[0], y - origin[1]) for x, y in previous)
                count = math.ceil(abs(angle) / math.radians(5.))
                padding = radius * (1. - math.cos(abs(angle) / (2 * count)))
                expected = []
                for step in range(1, count + 1):
                    current = task._pose_polygon(box, task._motion_pose(motion, step / count))
                    hull = task._convex_hull(previous + current)
                    expected.append(task._convex_hull(tuple((x + dx, y + dy) for x, y in hull
                        for dx, dy in ((-padding, -padding), (padding, -padding),
                                       (padding, padding), (-padding, padding)))))
                    previous = current
                self.assertEqual(task.motion_sweeps(box, motion), tuple(expected))

    def test_array_sweeps_match_scalar_pivot_translation_and_multistage_paths(self):
        box = task.ObjectGeometry('offset', (-.13, -.04, -.02), (.07, .02, .1))
        for angle in (0., 1.e-10, -math.pi/2, math.pi/4, math.pi):
            for pivot in (None, (.1, -.02)):
                motion = task.PlanarMotion(task.PlanarPose(.01, -.03, .2),
                    task.PlanarPose(.04, .07, .2+angle), pivot)
                path = (motion, task.PlanarMotion(motion.end, task.PlanarPose(.09, .12, motion.end.yaw)))
                expected = task._path_sweeps(box,path)
                actual = task._path_sweeps(box,path,packed=True)
                self.assertEqual(len(expected),len(actual))
                for polygon, vertices, count in zip(expected,actual.vertices,actual.counts):
                    np.testing.assert_allclose(vertices[:count],polygon,atol=1.e-14,rtol=1.e-14)

    def test_array_sweep_packing_matches_tuple_packing(self):
        import random
        catalog, rules = self.catalog(), self.legacy_rules()
        reference=task.sample_scene(catalog,rules,random.Random(1))
        plans=[]
        for seed in range(4):
            generator=task._scene_proposals(catalog,rules,random.Random(seed),reference.target_id,
                reference.blocker_id,reference.active_ids)
            plans.append(next(generator))
            generator.close()
        packed=[(*p[:5],task._path_sweeps(catalog[p[0][1]],p[4],packed=True),p[-1]) for p in plans]
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            result=task._pack_layouts(plans+packed,catalog,rules,device)
            self.assertTrue(any(row is not None for row in result[:4]))
            for before,after in zip(result[:4],result[4:]):
                if before is None:
                    self.assertIsNone(after)
                else:
                    np.testing.assert_array_equal(before,after)

    def test_tensor_start_regions_match_rotated_footprint_bounds(self):
        rng=np.random.default_rng(832)
        catalog=[task.ObjectGeometry(str(i),(-.02-i*.0005,-.04,0.),(.06,.03,.12)) for i in range(64)]
        spawn,rules=self.legacy_rules(),task.RobotStartRules()
        poses=np.column_stack((rng.uniform(-.65,-.52,64),rng.uniform(-.3,.3,64),
                               np.full(64,1.05),rng.uniform(-math.pi,math.pi,64)))
        expected=[]
        for i,pose in enumerate(poses):
            sample=task.SceneSample((i,),i,i,{i:tuple(pose[:3])},{i:pose[3]},(-1.,1.,-1.,1.),(0.,0.))
            expected.append(task.robot_start_region(sample,catalog,spawn,rules))
        half=(np.array([g.upper for g in catalog])-np.array([g.lower for g in catalog]))*.5
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for dtype in (torch.float64,torch.float32):
                actual=task._robot_start_regions(torch.tensor(poses,device=device,dtype=dtype),
                    torch.tensor(half,device=device,dtype=dtype),spawn,rules)
                torch.testing.assert_close(actual,torch.tensor(expected,device=device,dtype=dtype),
                    atol=1.e-7 if dtype==torch.float32 else 1.e-14,rtol=1.e-7 if dtype==torch.float32 else 1.e-14)

    def test_blocker_interval_matches_dense_independent_polygon_clipping(self):
        rng = np.random.default_rng(172)
        box = task.ObjectGeometry('blocker',(-.08,-.04,0.),(.08,.04,.1))
        access = np.array([-.02,.4,-.06,.06])
        for yaw in rng.uniform(-math.pi,math.pi,32):
            polygon = np.asarray(task.footprint_polygon(box,(.07,0.),float(yaw)))
            required = task.planar_polygon_area(tuple(map(tuple,polygon)))*.3
            low,high = task._blocker_y_interval(polygon,access,-.2,.2,required)
            self.assertLess(low,high)
            for y in np.linspace(low,high,15):
                actual = task.overlap_area(tuple(map(tuple,polygon+np.array([0.,y]))),task.rectangle_polygon(access))
                self.assertGreaterEqual(actual,required-1.e-9)
            for y in (-.2,low-1.e-5,high+1.e-5,.2):
                if -.2 <= y <= .2 and (y < low-1.e-7 or y > high+1.e-7):
                    actual = task.overlap_area(tuple(map(tuple,polygon+np.array([0.,y]))),task.rectangle_polygon(access))
                    self.assertLess(actual,required+1.e-9)
        low,high = task._blocker_y_interval(polygon,access,-.2,.2,1.)
        self.assertGreater(low,high)

    def test_resident_layouts_preserve_each_random_stream_and_subset_order(self):
        import random
        catalog, rules = self.catalog(), self.legacy_rules()
        def generate(indices, batch_size, device='cpu'):
            streams = [random.Random(50+i) for i in indices]
            result = task.sample_scenes(catalog,rules,streams,[(0,4)]*len(indices),
                                        [None]*len(indices),device=device,batch_size=batch_size)
            return result,[rng.getstate() for rng in streams]
        expected,states = generate(list(range(8)),1)
        actual,final_states = generate(list(reversed(range(8))),8)
        self.assertEqual(actual,expected[::-1])
        self.assertEqual(final_states,states[::-1])
        for sample in actual:
            task.validate_sample(sample,catalog,rules)
        if torch.cuda.is_available():
            gpu,_ = generate(list(range(8)),8,'cuda:0')
            for sample in gpu:
                task.validate_sample(sample,catalog,rules)
            # Independent device math may choose a different tied candidate;
            # within each device, ordering and batching cannot affect a seed.
            reversed_gpu,_ = generate(list(reversed(range(8))),3,'cuda:0')
            self.assertEqual(reversed_gpu,gpu[::-1])

    def test_two_object_configuration_and_fixed_composition_are_supported(self):
        import random
        catalog=self.catalog()[:1]+self.catalog()[4:5]
        rules=self.legacy_rules(min_objects=2,max_objects=2,min_asset_types=2,max_asset_types=2,all_asset_types=False,
            min_shelf_occupancy=0.,min_region_occupancy=0.,min_x_coverage=0.,min_y_coverage=0.,
            min_neighbor_fraction=0.)
        sample=task.sample_scene(catalog,rules,random.Random(44),0,1,(0,1))
        task.validate_sample(sample,catalog,rules)
        self.assertEqual(sample.active_ids,(0,1))

    def test_compiled_regional_overlap_matches_scalar_clipping_oracle(self):
        rng = np.random.default_rng(805)
        geometry = task.ObjectGeometry("box", (-.09, -.04, 0.), (.09, .04, .1))
        rules = self.legacy_rules()
        regions = task.shelf_regions(rules)
        polygons = [task.footprint_polygon(geometry, (float(x), float(y)), float(yaw))
                    for x, y, yaw in rng.uniform((-.9, -.5, -math.pi), (-.5, .5, math.pi), size=(96, 3))]
        expected = np.asarray([[task.overlap_area(polygon, task.rectangle_polygon(region)) for region in regions]
                               for polygon in polygons])
        actual = task._footprint_region_areas(np.asarray(polygons), np.asarray(regions))
        np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=2.e-16)
        self.assertEqual(task._footprint_region_areas(np.empty((0, 4, 2)), np.asarray(regions)).shape, (0, 6))

    def test_batched_placement_is_deterministic_and_independent_across_episodes(self):
        import random
        catalog, rules = self.catalog(), self.legacy_rules()
        first = task.sample_scene(catalog, rules, random.Random(1))
        repeated = task.sample_scene(catalog, rules, random.Random(1))
        self.assertEqual(first, repeated)
        other = task.sample_scene(catalog, rules, random.Random(103),
                                  target_id=first.target_id, blocker_id=first.blocker_id,
                                  active_ids=first.active_ids)
        self.assertEqual(other.active_ids, first.active_ids)
        self.assertNotEqual(other.positions, first.positions)
        task.validate_sample(other, catalog, rules)



class ConstructiveLayoutTests(unittest.TestCase):
    """Validate generated geometry with independent polygon/segment arithmetic."""

    @staticmethod
    def rules(count=10, **kwargs):
        values = dict(generation_mode="constructive", min_objects=count, max_objects=count,
                      min_asset_types=4, max_asset_types=10, all_asset_types=False,
                      x_range=(-1.6, -.3), y_range=(-1.1, 1.1),
                      min_shelf_occupancy=0., min_region_occupancy=0.,
                      min_x_coverage=0., min_y_coverage=0., min_neighbor_fraction=0.)
        values.update(kwargs)
        return task.SpawnRules(**values)

    @staticmethod
    def world_polygon(geometry, position, yaw):
        """Read collider vertices directly; do not use the production footprint helper."""
        vertices = getattr(geometry, "collision_footprint_vertices", ())
        if vertices:
            xy = np.asarray(vertices, dtype=float)[:, :2]
            xy = xy[ConvexHull(xy).vertices]
            xy -= (np.asarray(geometry.lower[:2]) + geometry.upper[:2]) * .5
        else:
            half = (np.asarray(geometry.upper[:2]) - geometry.lower[:2]) * .5
            xy = np.asarray([(-1., -1.), (1., -1.), (1., 1.), (-1., 1.)]) * half
        cosine, sine = math.cos(yaw), math.sin(yaw)
        return xy @ np.array([[cosine, sine], [-sine, cosine]]) + np.asarray(position[:2])

    @staticmethod
    def polygon_distance(first, second):
        """Independent SAT intersection test and Euclidean segment endpoint distance."""
        separated = False
        for polygon in (first, second):
            edges = np.roll(polygon, -1, axis=0) - polygon
            for edge in edges:
                axis = np.array([-edge[1], edge[0]])
                if np.dot(axis, axis) < 1.e-24:
                    continue
                a, b = first @ axis, second @ axis
                separated |= a.max() <= b.min() or b.max() <= a.min()
        if not separated:
            return 0.
        distance = math.inf
        for vertices, polygon in ((first, second), (second, first)):
            for point in vertices:
                for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
                    edge = end - start
                    parameter = np.clip(np.dot(point - start, edge) / np.dot(edge, edge), 0., 1.)
                    distance = min(distance, float(np.linalg.norm(point - start - parameter * edge)))
        return distance

    def assert_geometry(self, sample, catalog, rules):
        polygons = {index: self.world_polygon(catalog[index], sample.positions[index], sample.yaws[index])
                    for index in sample.active_ids}
        for index, polygon in polygons.items():
            np.testing.assert_array_less([rules.x_range[0] + rules.edge_margin - 1.e-8,
                                          rules.y_range[0] + rules.edge_margin - 1.e-8], polygon.min(0))
            np.testing.assert_array_less(polygon.max(0),
                                         [rules.x_range[1] - rules.edge_margin + 1.e-8,
                                          rules.y_range[1] - rules.edge_margin + 1.e-8])
            self.assertAlmostEqual(sample.positions[index][2] + catalog[index].lower[2], rules.shelf_top)
        for slot, first in enumerate(sample.active_ids):
            for second in sample.active_ids[slot + 1:]:
                self.assertGreaterEqual(self.polygon_distance(polygons[first], polygons[second]), .003 - 1.e-8,
                                        msg=f"Collider footprints {first}/{second} overlap or have too little gap")
        self.assertTrue(sample.neighbor_pairs, "Clutter must record at least one constructed neighbor relation")
        for first, second in sample.neighbor_pairs:
            self.assertIn(first, polygons)
            self.assertIn(second, polygons)
            distance = self.polygon_distance(polygons[first], polygons[second])
            self.assertGreaterEqual(distance, .003 - 1.e-8)
            self.assertLessEqual(distance, .010 + 1.e-8,
                                 msg=f"Neighbor {first}/{second} is too far apart: {distance}")

    def test_balanced_duplicates_unique_target_and_count_are_constructed(self):
        import random
        from collections import Counter
        catalog, rules = EpisodeLayoutTests.catalog(copies=8), self.rules()
        used_type_counts = set()
        for seed in range(24):
            sample = task.construct_scene(catalog, rules, random.Random(seed), target_id=0, blocker_id=8)
            self.assertEqual((sample.target_id, sample.blocker_id), (0, 8))
            self.assertEqual(len(sample.active_ids), 10)
            self.assertEqual(len(set(sample.active_ids)), 10)
            counts = Counter(catalog[index].asset_name for index in sample.active_ids)
            self.assertEqual(counts[catalog[0].asset_name], 1)
            self.assertGreaterEqual(len(counts), 4)
            self.assertLessEqual(len(counts), 10)
            others = [count for asset, count in counts.items() if asset != catalog[0].asset_name]
            self.assertLessEqual(max(others) - min(others), 1)
            used_type_counts.add(len(counts))
            self.assert_geometry(sample, catalog, rules)
        self.assertGreater(len(used_type_counts), 1, "Species count must be randomized")

    def test_future_count_is_not_hard_coded_to_ten(self):
        import random
        from collections import Counter
        catalog = EpisodeLayoutTests.catalog(copies=12)
        rules = self.rules(16, min_asset_types=4, max_asset_types=4)
        sample = task.construct_scene(catalog, rules, random.Random(41), target_id=0, blocker_id=12)
        self.assertEqual(len(sample.active_ids), 16)
        counts = Counter(catalog[index].asset_name for index in sample.active_ids)
        self.assertEqual(sorted(counts.values()), [1, 5, 5, 5])
        self.assert_geometry(sample, catalog, rules)

    def test_initial_yaw_covers_all_four_quadrants_for_each_type(self):
        import random
        catalog, rules = EpisodeLayoutTests.catalog(copies=1), self.rules(min_asset_types=10)
        quadrants = {geometry.asset_name: set() for geometry in catalog}
        for seed in range(64):
            sample = task.construct_scene(catalog, rules, random.Random(837 + seed))
            for index in sample.active_ids:
                angle = sample.yaws[index] % (2 * math.pi)
                quadrants[catalog[index].asset_name].add(int(angle / (math.pi / 2)) % 4)
        self.assertEqual(quadrants, {asset: {0, 1, 2, 3} for asset in quadrants})

    def test_typed_collision_footprints_define_clutter_gap_instead_of_render_bounds(self):
        import random
        catalog = []
        for geometry in EpisodeLayoutTests.catalog(copies=4):
            center = (np.asarray(geometry.lower[:2]) + geometry.upper[:2]) * .5
            half = (np.asarray(geometry.upper[:2]) - geometry.lower[:2]) * .5
            triangle = np.array([[-1., -1.], [1., -.65], [-.3, 1.]]) * half + center
            catalog.append(dataclasses.replace(geometry,
                           collision_footprint_vertices=tuple(map(tuple, triangle))))
        rules = self.rules()
        sample = task.construct_scene(tuple(catalog), rules, random.Random(76))
        self.assert_geometry(sample, catalog, rules)
        # The real collider is asymmetric and offset from its OBB center.
        # Comparing generated polygon distances, rather than only a stored gap,
        # catches accidental fallback to visual bounding rectangles.
        for first, second in sample.neighbor_pairs:
            distance = self.polygon_distance(self.world_polygon(catalog[first], sample.positions[first], sample.yaws[first]),
                                             self.world_polygon(catalog[second], sample.positions[second], sample.yaws[second]))
            self.assertLessEqual(distance, .010 + 1.e-8)

    def test_generation_does_not_call_legacy_rejection_or_path_search(self):
        import random
        catalog, rules = EpisodeLayoutTests.catalog(copies=4), self.rules()
        with patch.object(task, "_scene_proposals", side_effect=AssertionError("Legacy proposal rejection")), \
             patch.object(task, "_pack_layouts", side_effect=AssertionError("Legacy iterative packing")), \
             patch.object(task, "_choose_clearance_paths", side_effect=AssertionError("Legacy path search")):
            first = task.construct_scene(catalog, rules, random.Random(52))
            repeated = task.construct_scene(catalog, rules, random.Random(52))
        self.assertEqual(first, repeated)
        self.assert_geometry(first, catalog, rules)

    def test_native_batch_dispatch_keeps_streams_independent_without_rejection(self):
        import random
        catalog, rules = EpisodeLayoutTests.catalog(copies=4), self.rules()
        roles = [(0, 4)] * 4
        def generate(indices, batch_size):
            streams = [random.Random(420 + index) for index in indices]
            result = task.sample_scenes(catalog, rules, streams, [roles[index] for index in indices],
                                        [None] * len(indices), batch_size=batch_size)
            return result, [stream.getstate() for stream in streams]
        with patch.object(task, "_scene_proposals", side_effect=AssertionError("Legacy rejection path")), \
             patch.object(task, "_pack_layouts", side_effect=AssertionError("Legacy iterative packer")):
            forward, first_states = generate([0, 1, 2, 3], 1)
            reverse, reverse_states = generate([3, 2, 1, 0], 4)
        self.assertEqual(reverse, forward[::-1])
        self.assertEqual(reverse_states, first_states[::-1])
        for sample in forward:
            self.assertNotIsInstance(sample, Exception)
            self.assert_geometry(sample, catalog, rules)

    def test_reserved_blocker_motion_stays_supported_and_clear_of_other_objects(self):
        import random
        catalog, rules = EpisodeLayoutTests.catalog(copies=4), self.rules()
        sample = task.construct_scene(catalog, rules, random.Random(25), target_id=0, blocker_id=4)
        self.assertTrue(sample.clearance_path)
        blocker = catalog[sample.blocker_id]
        others = [self.world_polygon(catalog[index], sample.positions[index], sample.yaws[index])
                  for index in sample.active_ids if index != sample.blocker_id]
        for motion in sample.clearance_path:
            start = np.array([motion.start.x, motion.start.y])
            finish = np.array([motion.end.x, motion.end.y])
            for amount in np.linspace(0., 1., 25):
                angle = amount * (motion.end.yaw - motion.start.yaw)
                if motion.pivot is None:
                    position = start + amount * (finish - start)
                else:
                    pivot = np.asarray(motion.pivot)
                    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
                    position = pivot + rotation @ (start - pivot)
                polygon = self.world_polygon(blocker, position, motion.start.yaw + angle)
                self.assertGreaterEqual(polygon[:, 0].min(), rules.x_range[0] + rules.edge_margin - 1.e-8)
                self.assertLessEqual(polygon[:, 0].max(), rules.x_range[1] - rules.edge_margin + 1.e-8)
                self.assertGreaterEqual(polygon[:, 1].min(), rules.y_range[0] + rules.edge_margin - 1.e-8)
                self.assertLessEqual(polygon[:, 1].max(), rules.y_range[1] - rules.edge_margin + 1.e-8)
                for other in others:
                    self.assertGreaterEqual(self.polygon_distance(polygon, other), .003 - 1.e-8)

    def test_analytic_footprint_limit_has_exact_euclidean_gap_for_rotated_shapes(self):
        generator = np.random.default_rng(573)
        for index in range(20):
            first = np.array([[-.07, -.025], [.035, -.018], [.012, .061]])
            second = np.array([[-.061, -.037], [.055, -.037], [.055, .019], [-.061, .019]])
            for polygon in (first, second):
                angle = generator.uniform(-math.pi, math.pi)
                polygon[:] = polygon @ np.array([[math.cos(angle), math.sin(angle)],
                                                  [-math.sin(angle), math.cos(angle)]])
            second += generator.uniform(-.2, .2, 2)
            gap = (.003, .007, .010)[index % 3]
            low = second[:, 0].min() - first[:, 0].max()
            high = second[:, 0].max() - first[:, 0].min()
            for x in np.linspace(low - gap, high + gap, 7):
                limit = task.footprint_line_limit(first, second, float(x), gap)
                self.assertIsNotNone(limit)
                actual = self.polygon_distance(first + [x, limit], second)
                self.assertAlmostEqual(actual, gap, delta=3.e-12)
                self.assertGreaterEqual(self.polygon_distance(first + [x, limit + 1.e-5], second), gap - 1.e-12)
                if low + 1.e-5 < x < high - 1.e-5:
                    self.assertLess(self.polygon_distance(first + [x, limit - 1.e-5], second), gap)
            self.assertIsNone(task.footprint_line_limit(first, second, float(high + gap + .001), gap))
            self.assertIsNone(task.footprint_line_limit(first, second, float(low - gap - .001), gap))

    def test_analytic_footprint_limit_preserves_noncentered_placement_origin(self):
        first = np.array([[-.04, -.02], [.05, -.02], [.02, .05]])
        second = np.array([[-.08, -.07], [.09, -.05], [.02, .04]])
        offset = np.array([.032, -.047])
        baseline = task.footprint_line_limit(first, second, .012, .005)
        translated = task.footprint_line_limit(first + offset, second, .012 - offset[0], .005)
        self.assertAlmostEqual(translated, baseline - offset[1], places=13)
        np.testing.assert_allclose(first, [[-.04, -.02], [.05, -.02], [.02, .05]], atol=0., rtol=0.)

    def test_analytic_compaction_stays_below_initial_pose_and_records_actual_neighbors(self):
        generator = np.random.default_rng(748)
        polygons = []
        for index in range(9):
            polygon = np.array([[-.036, -.021], [.041, -.013], [.016, .035]])
            angle = generator.uniform(-math.pi, math.pi)
            polygon = polygon @ np.array([[math.cos(angle), math.sin(angle)],
                                         [-math.sin(angle), math.cos(angle)]])
            polygons.append(polygon + [.003 * index, -.002 * index])
        centers = np.array([[x, y] for y in (-.25, 0., .25) for x in (-.2, 0., .2)])
        originals = [polygon.copy() for polygon in polygons]
        initial = centers.copy()
        gap = generator.uniform(.003, .010, len(polygons))
        reserved = np.array([[-.07, -.5], [.07, -.5], [.07, -.35], [-.07, -.35]])
        compact, pairs = task._compact_constructive_positions(polygons, centers, -.5, gap, [reserved])
        np.testing.assert_array_equal(centers, initial)
        for actual, original in zip(polygons, originals):
            np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(compact[:, 0], initial[:, 0])
        self.assertTrue((compact[:, 1] <= initial[:, 1] + 1.e-14).all())
        world = [polygon + center for polygon, center in zip(polygons, compact)]
        for index, polygon in enumerate(world):
            self.assertGreaterEqual(polygon[:, 1].min(), -.5 - 1.e-14)
            self.assertGreaterEqual(self.polygon_distance(polygon, reserved), .003 - 1.e-12)
            for other in world[:index]:
                self.assertGreaterEqual(self.polygon_distance(polygon, other), .003 - 1.e-12)
        self.assertTrue(pairs)
        for parent, child in pairs:
            self.assertNotEqual(parent, child)
            self.assertGreaterEqual(self.polygon_distance(world[parent], world[child]), .003 - 1.e-12)

    def test_compaction_treats_unprocessed_rotated_cross_sections_as_obstacles(self):
        # Archived real-catalog seed 10, second directional pass. Center-Y order
        # does not order the diagonal sections: ignoring future bodies let body
        # 3 enter body 4's footprint before body 4 could be processed.
        polygons = np.array([
            [[-.07479341, -.00474399], [-.01630551, -.07314841], [.07479341, .00474399], [.01630551, .07314841]],
            [[-.04715630, .00639580], [-.00548390, -.04727102], [.04715630, -.00639580], [.00548390, .04727102]],
            [[-.07162628, .04083260], [-.06447250, -.05138990], [.07162628, -.04083260], [.06447250, .05138990]],
            [[.00103657, .05153125], [-.04120888, -.03095759], [-.00103657, -.05153125], [.04120888, .03095759]],
            [[-.05938347, -.04571830], [.02734085, -.06977849], [.05938347, .04571830], [-.02734085, .06977849]],
        ])
        centers = np.array([[-.77171751, .09750978], [-.79935462, .22792921],
                            [-.67769166, .23204809], [-.80530204, .34496923],
                            [-.71381838, .36321647]])
        minimum_gap = .006842875288086839
        for index, polygon in enumerate(polygons):
            for other in range(index):
                self.assertGreaterEqual(self.polygon_distance(polygon + centers[index],
                                                               polygons[other] + centers[other]), minimum_gap - 3.e-8)
        compact, _ = task._compact_constructive_positions(polygons, centers, .0243613,
                                                          [minimum_gap] * len(polygons))
        self.assertTrue((compact[:, 1] <= centers[:, 1] + 1.e-14).all())
        for index, polygon in enumerate(polygons):
            for other in range(index):
                actual = self.polygon_distance(polygon + compact[index], polygons[other] + compact[other])
                self.assertGreaterEqual(actual, minimum_gap - 3.e-8)


class CertifiedBackgroundTests(unittest.TestCase):
    @staticmethod
    def fixture():
        geometry = EpisodeLayoutTests.catalog(copies=1)
        rules = task.SpawnRules(x_range=(-.9, -.5), y_range=(-.455, .455))
        ids = tuple(range(8))
        capacity = task._certified_background_capacity(geometry, ids, rules)
        return geometry, ids, rules, capacity

    def test_tight_domain_constructs_actual_neighbor_gaps_on_either_module_side(self):
        import random
        geometry, ids, rules, capacity = self.fixture()
        self.assertTrue(capacity['depth_valid'])
        xmin, xmax, ymin, ymax = capacity['support_bounds']
        module_width = capacity['marginal_module_width_limit'] - .005
        for side in (-1, 1):
            module = (xmin, xmax, ymin, ymin+module_width) if side > 0 else (xmin, xmax, ymax-module_width, ymax)
            for seed in range(8):
                positions, yaws, pairs, certificate = task._sample_certified_background(
                    geometry, ids, rules, module, random.Random(seed), side=side, capacity=capacity)
                self.assertEqual(set(positions), set(ids))
                self.assertEqual(set(yaws), set(ids))
                self.assertEqual(len(pairs), len(ids)-2)
                world = {index: ConstructiveLayoutTests.world_polygon(geometry[index], positions[index], yaws[index])
                         for index in ids}
                for index, polygon in world.items():
                    self.assertGreaterEqual(polygon[:, 0].min(), xmin-1.e-10)
                    self.assertLessEqual(polygon[:, 0].max(), xmax+1.e-10)
                    self.assertGreaterEqual(polygon[:, 1].min(), ymin-1.e-10)
                    self.assertLessEqual(polygon[:, 1].max(), ymax+1.e-10)
                    self.assertAlmostEqual(positions[index][2]+geometry[index].lower[2], rules.shelf_top)
                    if side > 0:
                        self.assertGreaterEqual(polygon[:, 1].min(), module[3]+certificate['outside_gap']-1.e-10)
                    else:
                        self.assertLessEqual(polygon[:, 1].max(), module[2]-certificate['outside_gap']+1.e-10)
                    for other in ids:
                        if other < index:
                            self.assertGreaterEqual(ConstructiveLayoutTests.polygon_distance(polygon, world[other]),
                                                    rules.gap-1.e-10)
                for (first, second), requested in zip(pairs, certificate['neighbor_gaps']):
                    actual = ConstructiveLayoutTests.polygon_distance(world[first], world[second])
                    self.assertAlmostEqual(actual, requested, delta=3.e-12)
                    self.assertGreaterEqual(actual, .003-1.e-10)
                    self.assertLessEqual(actual, .010+1.e-10)
                for row in capacity['rows']:
                    self.assertTrue(set(row['ids']) & set(certificate['free_ids']))

    def test_spare_capacity_gives_all_yaws_full_freedom_without_candidate_helpers(self):
        import random
        geometry, ids, _, _ = self.fixture()
        rules = task.SpawnRules(x_range=(-.9, -.5), y_range=(-1., 1.))
        capacity = task._certified_background_capacity(geometry, ids, rules)
        xmin, xmax, ymin, _ = capacity['support_bounds']
        module = (xmin, xmax, ymin, ymin+.05)
        with patch.object(task, '_constructive_slicing_plans', side_effect=AssertionError('DP')), \
             patch.object(task, '_compact_constructive_positions', side_effect=AssertionError('Compaction')), \
             patch.object(task, '_constructive_lateral_guards', side_effect=AssertionError('Guard candidates')), \
             patch.object(task, '_scene_proposals', side_effect=AssertionError('Rejection generator')):
            first = task._sample_certified_background(geometry, ids, rules, module, random.Random(13), capacity=capacity)
            repeated = task._sample_certified_background(geometry, ids, rules, module, random.Random(13), capacity=capacity)
        self.assertEqual(first, repeated)
        self.assertEqual(set(first[3]['free_ids']), set(ids))
        self.assertEqual(set(first[3]['full_yaw_ids']), set(ids))
        self.assertEqual(set(first[3]['angle_half_widths'].values()), {math.pi})

    def test_every_background_slot_has_full_circle_marginal_support_in_tight_domain(self):
        import random
        geometry, ids, rules, capacity = self.fixture()
        xmin, xmax, ymin, _ = capacity['support_bounds']
        module = (xmin, xmax, ymin, ymin+capacity['marginal_module_width_limit']-.002)
        quadrants = {index: set() for index in ids}
        ever_constrained = False
        for seed in range(256):
            _, yaws, _, certificate = task._sample_certified_background(
                geometry, ids, rules, module, random.Random(912+seed), capacity=capacity)
            ever_constrained |= any(value < math.pi for value in certificate['angle_half_widths'].values())
            for index in certificate['free_ids']:
                quadrants[index].add(int((yaws[index] % (2*math.pi)) / (math.pi/2)) % 4)
        self.assertTrue(ever_constrained)
        self.assertEqual(quadrants, {index: {0, 1, 2, 3} for index in ids})

    def test_profiles_use_actual_triangle_and_share_cache_across_identical_copies(self):
        triangle = ((-.02, -.02), (.04, -.02), (-.02, .06))
        first = task.ObjectGeometry('first', (-.02, -.02, 0.), (.04, .06, .1), 'same',
                                    collision_footprint_vertices=triangle)
        second = dataclasses.replace(first, name='second')
        rules = task.SpawnRules(x_range=(-.5, .5), y_range=(-.5, .5))
        task._certified_footprint_profile.cache_clear()
        capacity = task._certified_background_capacity((first, second), (0, 1), rules)
        diameter, minimum, angle = capacity['profiles'][0]
        self.assertAlmostEqual(diameter, .1, places=13)
        self.assertAlmostEqual(minimum, .048, places=13)
        polygon = ConstructiveLayoutTests.world_polygon(first, (0., 0.), angle)
        self.assertAlmostEqual(float(np.ptp(polygon[:, 1])), minimum, places=13)
        info = task._certified_footprint_profile.cache_info()
        self.assertEqual((info.misses, info.hits), (1, 1))
        box = dataclasses.replace(first, collision_footprint_vertices=())
        box_capacity = task._certified_background_capacity((box,), (0,), rules)
        self.assertAlmostEqual(box_capacity['profiles'][0][1], .06, places=13)

    def test_larger_background_count_and_empty_background_are_supported(self):
        import random
        geometry = EpisodeLayoutTests.catalog(copies=2)
        ids = tuple(range(15))
        rules = task.SpawnRules(x_range=(-1.5, -.1), y_range=(-2., 2.))
        capacity = task._certified_background_capacity(geometry, ids, rules)
        positions, yaws, pairs, _ = task._sample_certified_background(
            geometry, ids, rules, (-1.4, -.2, -1.9, -1.4), random.Random(19), capacity=capacity)
        self.assertEqual((len(positions), len(yaws), len(pairs)), (15, 15, 13))
        empty = task._certified_background_capacity(geometry, (), rules)
        result = task._sample_certified_background(geometry, (), rules, (-1.4, -.2, -1.9, -1.4),
                                                    random.Random(19), capacity=empty)
        self.assertEqual(result[:3], ({}, {}, ()))

    def test_visual_outliers_remain_outside_view_strip_without_changing_collision_gaps(self):
        import random
        collision = ((-.02, -.02), (.02, -.02), (.02, .02), (-.02, .02))
        geometry = tuple(task.ObjectGeometry(f'visual_{i}', (-.07, -.05, 0.), (.07, .05, .1),
                                             collision_footprint_vertices=collision) for i in range(6))
        ids = tuple(range(6))
        rules = task.SpawnRules(x_range=(-1., 0.), y_range=(-1., 1.))
        capacity = task._certified_background_capacity(geometry, ids, rules)
        eta = math.hypot(.05, .03)
        self.assertAlmostEqual(capacity['visual_padding'][0], eta, places=13)
        self.assertEqual(len(capacity['profiles'][0]), 3)
        for row in capacity['rows']:
            expected = (sum(capacity['profiles'][i][1] for i in row['ids']) +
                        max(capacity['profiles'][i][0]-capacity['profiles'][i][1] for i in row['ids']) +
                        rules.max_neighbor_gap*(len(row['ids'])-1) + 2*eta)
            self.assertAlmostEqual(row['span_upper_by_free'][0], expected, places=13)
        xmin, xmax, ymin, ymax = capacity['support_bounds']
        for side in (-1, 1):
            module = (xmin, xmax, -.25, .25)
            for seed in range(16):
                positions, yaws, pairs, certificate = task._sample_certified_background(
                    geometry, ids, rules, module, random.Random(seed), side=side, capacity=capacity,
                    spread_gaps=bool(seed % 2))
                physical = {i: ConstructiveLayoutTests.world_polygon(geometry[i], positions[i], yaws[i]) for i in ids}
                for i in ids:
                    visual = np.asarray(task.footprint_polygon(geometry[i], positions[i], yaws[i]))
                    self.assertGreaterEqual(visual[:, 1].min(), ymin-1.e-12)
                    self.assertLessEqual(visual[:, 1].max(), ymax+1.e-12)
                    if side > 0:
                        self.assertGreaterEqual(visual[:, 1].min(), module[3]+certificate['outside_gap']-1.e-12)
                    else:
                        self.assertLessEqual(visual[:, 1].max(), module[2]-certificate['outside_gap']+1.e-12)
                for (first, second), gap in zip(pairs, certificate['neighbor_gaps']):
                    self.assertAlmostEqual(ConstructiveLayoutTests.polygon_distance(physical[first], physical[second]),
                                           gap, delta=3.e-12)

    def test_actual_visual_hull_avoids_empty_bbox_corner_padding(self):
        import random
        triangle = ((-.04, -.03), (.05, -.03), (-.04, .06))
        geometry = task.ObjectGeometry('mesh', (-.04, -.03, 0.), (.05, .06, .1),
                                       collision_footprint_vertices=triangle,
                                       visual_footprint_vertices=triangle)
        rules = task.SpawnRules(x_range=(-1., 0.), y_range=(-1., 1.))
        actual = task._certified_background_capacity((geometry,), (0,), rules)
        conservative = task._certified_background_capacity(
            (dataclasses.replace(geometry, visual_footprint_vertices=()),), (0,), rules)
        self.assertAlmostEqual(actual['visual_padding'][0], 0., places=13)
        self.assertGreater(conservative['visual_padding'][0], .06)
        self.assertLess(actual['minimum_slab_width'], conservative['minimum_slab_width'])
        copies = tuple(dataclasses.replace(geometry, name=f'mesh_{i}') for i in range(6))
        ids = tuple(range(6))
        capacity = task._certified_background_capacity(copies, ids, rules)
        xmin, xmax, ymin, ymax = capacity['support_bounds']
        for side in (-1, 1):
            for seed in range(4):
                module = (xmin, xmax, -.25, .25)
                positions, yaws, _, cert = task._sample_certified_background(
                    copies, ids, rules, module, random.Random(seed), side=side, capacity=capacity)
                for i in ids:
                    # Actual visual and collider triangles coincide here;
                    # unused visual OBB corners do not consume view space.
                    visual = ConstructiveLayoutTests.world_polygon(copies[i], positions[i], yaws[i])
                    self.assertGreaterEqual(visual[:, 1].min(), ymin-1.e-12)
                    self.assertLessEqual(visual[:, 1].max(), ymax+1.e-12)
                    if side > 0:
                        self.assertGreaterEqual(visual[:, 1].min(), module[3]+cert['outside_gap']-1.e-12)
                    else:
                        self.assertLessEqual(visual[:, 1].max(), module[2]-cert['outside_gap']+1.e-12)

    def test_collider_protrusions_are_kept_outside_view_and_motion_strip(self):
        import random
        outer = ((-.04, -.04), (.04, -.04), (.04, .04), (-.04, .04))
        inner = ((-.01, -.01), (.01, -.01), (.01, .01), (-.01, .01))
        geometry = tuple(task.ObjectGeometry(f'collision_{i}', (-.04, -.04, 0.), (.04, .04, .1),
            collision_footprint_vertices=outer, visual_footprint_vertices=inner) for i in range(4))
        ids = tuple(range(4))
        rules = task.SpawnRules(x_range=(-1., 0.), y_range=(-.455, .455))
        capacity = task._certified_background_capacity(geometry, ids, rules)
        self.assertEqual(set(capacity['visual_padding'].values()), {0.})
        xmin, xmax, ymin, ymax = capacity['support_bounds']
        width = capacity['marginal_module_width_limit']-.001
        for side in (-1, 1):
            module = (xmin, xmax, ymin, ymin+width) if side > 0 else (xmin, xmax, ymax-width, ymax)
            for seed in range(16):
                positions, yaws, _, certificate = task._sample_certified_background(
                    geometry, ids, rules, module, random.Random(seed), side=side, capacity=capacity,
                    spread_gaps=bool(seed % 2))
                for i in ids:
                    physical = ConstructiveLayoutTests.world_polygon(geometry[i], positions[i], yaws[i])
                    self.assertGreaterEqual(physical[:, 1].min(), ymin-1.e-12)
                    self.assertLessEqual(physical[:, 1].max(), ymax+1.e-12)
                    if side > 0:
                        self.assertGreaterEqual(physical[:, 1].min(), module[3]+certificate['outside_gap']-1.e-12)
                    else:
                        self.assertLessEqual(physical[:, 1].max(), module[2]-certificate['outside_gap']+1.e-12)

    def test_spread_gaps_support_dense_and_wide_distances_with_exact_budgeted_geometry(self):
        import random
        geometry, ids, rules, capacity = self.fixture()
        xmin, xmax, ymin, ymax = capacity['support_bounds']
        short, wide, short_x, wide_x = False, False, False, False
        for side in (-1, 1):
            module = (xmin, xmax, ymin, ymin+.15) if side > 0 else (xmin, xmax, ymax-.15, ymax)
            for seed in range(48):
                positions, yaws, pairs, cert = task._sample_certified_background(
                    geometry, ids, rules, module, random.Random(seed), side=side, capacity=capacity,
                    spread_gaps=True)
                physical = {i: ConstructiveLayoutTests.world_polygon(geometry[i], positions[i], yaws[i]) for i in ids}
                self.assertTrue(cert['spread_gaps'])
                short_x |= cert['row_x_gap'] <= rules.max_neighbor_gap+1.e-12
                wide_x |= cert['row_x_gap'] > rules.max_neighbor_gap+1.e-12
                self.assertGreaterEqual(cert['row_x_gap'], rules.gap-1.e-12)
                for budget in cert['row_gap_budgets']:
                    self.assertLessEqual(budget['extra_gap_total'], budget['initial_surplus']+1.e-12)
                    self.assertLessEqual(budget['width_sum']+2*budget['visual_padding']+budget['gap_sum'],
                                         cert['available_slab_width']+1.e-12)
                for i, p in physical.items():
                    self.assertGreaterEqual(p[:, 0].min(), xmin-1.e-12)
                    self.assertLessEqual(p[:, 0].max(), xmax+1.e-12)
                    self.assertGreaterEqual(p[:, 1].min(), ymin-1.e-12)
                    self.assertLessEqual(p[:, 1].max(), ymax+1.e-12)
                    if side > 0:
                        self.assertGreaterEqual(p[:, 1].min(), module[3]+cert['outside_gap']-1.e-12)
                    else:
                        self.assertLessEqual(p[:, 1].max(), module[2]-cert['outside_gap']+1.e-12)
                    for j in ids:
                        if j < i:
                            self.assertGreaterEqual(ConstructiveLayoutTests.polygon_distance(p, physical[j]),
                                                    rules.gap-1.e-12)
                for (first, second), requested in zip(pairs, cert['neighbor_gaps']):
                    self.assertGreaterEqual(requested, rules.gap-1.e-12)
                    actual = ConstructiveLayoutTests.polygon_distance(physical[first], physical[second])
                    self.assertAlmostEqual(actual, requested, delta=4.e-12)
                    short |= actual <= rules.max_neighbor_gap+1.e-12
                    wide |= actual > rules.max_neighbor_gap+1.e-12
        self.assertTrue(short and wide and short_x and wide_x)

    def test_spread_flag_defaults_to_unchanged_dense_sampling_and_uses_no_candidate_helpers(self):
        import random
        geometry, ids, rules, capacity = self.fixture()
        module = (-.885, -.515, -.44, -.25)
        default = task._sample_certified_background(geometry, ids, rules, module, random.Random(14), capacity=capacity)
        explicit = task._sample_certified_background(geometry, ids, rules, module, random.Random(14),
                                                     capacity=capacity, spread_gaps=False)
        self.assertEqual(default, explicit)
        with patch.object(task, '_constructive_slicing_plans', side_effect=AssertionError('DP')), \
             patch.object(task, '_compact_constructive_positions', side_effect=AssertionError('Compaction')), \
             patch.object(task, '_constructive_lateral_guards', side_effect=AssertionError('Guard search')), \
             patch.object(task, '_scene_proposals', side_effect=AssertionError('Rejection generator')):
            spread = task._sample_certified_background(geometry, ids, rules, module, random.Random(14),
                                                       capacity=capacity, spread_gaps=True)
        self.assertEqual(set(spread[0]), set(ids))

    def test_insufficient_measured_depth_or_module_space_fails_before_rng_draw(self):
        from unittest.mock import Mock
        geometry = tuple(task.ObjectGeometry(f'wide_{i}', (-.08, -.04, 0.), (.08, .04, .1)) for i in range(8))
        rules = task.SpawnRules(x_range=(-.89, -.51), y_range=(-.455, .455))  # usable depth 350 mm
        capacity = task._certified_background_capacity(geometry, tuple(range(8)), rules)
        self.assertFalse(capacity['depth_valid'])
        rng = Mock(side_effect=AssertionError('RNG must not be used for an invalid startup domain'))
        with self.assertRaisesRegex(ValueError, 'needs depth.*measured usable depth'):
            task._sample_certified_background(geometry, tuple(range(8)), rules, (-.8, -.6, -.4, -.3), rng, capacity=capacity)
        rng.assert_not_called()
        rng.uniform.assert_not_called()
        geometry, ids, rules, capacity = self.fixture()
        with self.assertRaisesRegex(ValueError, 'no positive continuous yaw domain'):
            task._sample_certified_background(geometry, ids, rules, (-.8, -.6, -.44, .44), rng, capacity=capacity)
        rng.uniform.assert_not_called()


class CertifiedRoleCellTests(unittest.TestCase):
    @staticmethod
    def box(geometry, center, yaw):
        half = (np.asarray(geometry.upper[:2])-geometry.lower[:2])*.5
        vertices = np.asarray(((-1., -1.), (1., -1.), (1., 1.), (-1., 1.)))*half
        c, s = math.cos(yaw), math.sin(yaw)
        return vertices @ np.asarray(((c, s), (-s, c)))+center

    @staticmethod
    def clipped_area(vertices, xmin, half_y):
        """Independent axis-coordinate clipping oracle for the access strip."""
        polygon = list(map(np.asarray, vertices))
        for axis, sign, boundary in ((0, 1., xmin), (1, 1., -half_y), (1, -1., half_y)):
            incoming, polygon = polygon, []
            if not incoming:
                return 0.
            for begin, end in zip(incoming, incoming[1:]+incoming[:1]):
                a, b = sign*(begin[axis]-boundary), sign*(end[axis]-boundary)
                if a >= 0.:
                    polygon.append(begin)
                if (a < 0.) != (b < 0.):
                    polygon.append(begin+(end-begin)*a/(a-b))
        if len(polygon) < 3:
            return 0.
        p = np.asarray(polygon)
        return abs(float(p[:, 0] @ np.roll(p[:, 1], -1)-p[:, 1] @ np.roll(p[:, 0], -1)))*.5

    @staticmethod
    def fixture():
        geometry = (task.ObjectGeometry('target', (-.02, -.035, 0.), (.04, .025, .15)),
                    task.ObjectGeometry('blocker', (-.045, -.045, 0.), (.035, .035, .13)))
        rules = task.SpawnRules(x_range=(-.9, -.5), y_range=(-.455, .455))
        return geometry, rules

    def test_continuous_cells_bound_blockage_motion_visuals_and_camera(self):
        geometry, rules = self.fixture()
        ty, by, bx, half = -.12, .08, .10, .12
        camera = (.65, .30, 1.3)
        cells = task._certify_role_translation_cells(geometry, 0, 1, ty, by, bx,
            ((-.028, .028),), rules, yaw_half_width=half, camera_eye=camera)
        self.assertGreater(len(cells), 2)
        self.assertEqual({cell['side'] for cell in cells}, {-1, 1})
        half_y = rules.hand_width*.5+rules.approach_margin
        area = .08*.08
        for cell in cells:
            self.assertEqual(cell['yaw_bounds'], (-half, half))
            self.assertGreaterEqual(cell['blockage_lower_bound'], rules.min_blocked_ratio)
            bounds = cell['module_bounds']
            visual_bounds = cell['visual_module_bounds']
            physical_bounds = cell['collision_module_bounds']
            for psi in np.linspace(-half, half, 13):
                c, s = math.cos(psi), math.sin(psi)
                matrix = np.asarray(((c, -s), (s, c)))
                target = self.box(geometry[0], (0., 0.), ty+psi)
                xmin = float(target[:, 0].max())+rules.approach_margin
                eye = matrix @ camera[:2]
                self.assertGreaterEqual(eye[0], cell['camera_x_bounds_over_yaw'][0]-1.e-12)
                self.assertLessEqual(eye[0], cell['camera_x_bounds_over_yaw'][1]+1.e-12)
                self.assertGreaterEqual(eye[1], bounds[2]-1.e-12)
                self.assertLessEqual(eye[1], bounds[3]+1.e-12)
                for translation in np.linspace(*cell['parameter_bounds'], 7):
                    blocker = self.box(geometry[1], matrix @ (bx, translation), by+psi)
                    actual = self.clipped_area(blocker, xmin, half_y)/area
                    self.assertGreaterEqual(actual, cell['blockage_lower_bound']-2.e-12)
                    for motion_y in np.linspace(translation, cell['goal_y'], 5):
                        blocker = self.box(geometry[1], matrix @ (bx, motion_y), by+psi)
                        both = np.concatenate((target, blocker))
                        for constraint in (physical_bounds, visual_bounds):
                            self.assertGreaterEqual(float(both[:, 0].min()), constraint[0]-1.e-12)
                            self.assertLessEqual(float(both[:, 0].max()), constraint[1]+1.e-12)
                            self.assertGreaterEqual(float(both[:, 1].min()), constraint[2]-1.e-12)
                            self.assertLessEqual(float(both[:, 1].max()), constraint[3]+1.e-12)
                    if cell['side'] > 0:
                        self.assertGreaterEqual(float(blocker[:, 1].min()), half_y+rules.gap-1.e-12)
                    else:
                        self.assertLessEqual(float(blocker[:, 1].max()), -half_y-rules.gap+1.e-12)

    def test_access_halfplane_certificate_retains_thin_blockers(self):
        geometry = (task.ObjectGeometry('target', (-.01, -.02, 0.), (.01, .02, .1)),
                    task.ObjectGeometry('thin', (-.002, -.04, 0.), (.002, .04, .1)))
        rules = task.SpawnRules()
        cells = task._certify_role_translation_cells(geometry, 0, 1, 0., 0., .04,
            ((-.002, .002),), rules, yaw_half_width=.10)
        self.assertEqual(len(cells), 1)
        radius = math.hypot(.042, .042)
        self.assertGreater(.002+2*radius*math.sin(.05), .002)  # OBB erosion alone is empty.
        for psi in np.linspace(-.1, .1, 31):
            c, s = math.cos(psi), math.sin(psi)
            r = np.asarray(((c, -s), (s, c)))
            t = self.box(geometry[0], (0., 0.), psi)
            for translation in np.linspace(-.002, .002, 11):
                b = self.box(geometry[1], r @ (.04, translation), psi)
                ratio = self.clipped_area(b, float(t[:, 0].max())+.01, .06)/(.004*.08)
                self.assertGreaterEqual(ratio, cells[0]['blockage_lower_bound']-2.e-12)

    def test_arc_bounds_include_interior_cardinal_extrema(self):
        points = np.asarray(((.3, .4), (-.7, .1), (-.2, -.9), (.6, -.1)))
        half = .8
        bounds = task._certified_rotated_bounds(points, half)
        values = []
        for angle in np.linspace(-half, half, 10001):
            c, s = math.cos(angle), math.sin(angle)
            values.append(points @ np.asarray(((c, s), (-s, c))))
        dense = np.concatenate(values)
        expected = (dense[:, 0].min(), dense[:, 0].max(), dense[:, 1].min(), dense[:, 1].max())
        np.testing.assert_allclose(bounds, expected, rtol=0., atol=4.e-9)

    def test_uncertifiable_domains_are_omitted_and_invalid_inputs_are_rejected(self):
        geometry, rules = self.fixture()
        result = task._certify_role_translation_cells(geometry, 0, 1, 0., 0., .02,
            ((-.01, .01),), rules, yaw_half_width=.1)
        self.assertEqual(result, [])  # Local-X support separation is not certified.
        for kwargs in (dict(yaw_half_width=math.pi/2), dict(yaw_half_width=.1, cell_width=0.)):
            with self.assertRaises(ValueError):
                task._certify_role_translation_cells(geometry, 0, 1, 0., 0., .1, ((-.01, .01),), rules, **kwargs)
        with self.assertRaises(ValueError):
            task._certify_role_translation_cells(geometry, 0, 1, 0., 0., .1, ((.01, -.01),), rules,
                                                yaw_half_width=.1)


class CertifiedNormalPivotTests(unittest.TestCase):
    @staticmethod
    def fixture():
        geometries = (task.ObjectGeometry('target', (-.03, -.03, 0.), (.03, .03, .1)),
                      task.ObjectGeometry('wide', (-.08, -.025, 0.), (.08, .025, .1)))
        rules = task.SpawnRules(x_range=(-.5, .5), y_range=(-.5, .5), min_objects=2, max_objects=2)
        by, bx, half = math.pi/2, .0615, math.radians(2.5)
        camera = (.85, .08, 1.23, .5, -.5, -.5, .5)
        intervals = ((-.02, .02),)
        cells = task._certify_role_translation_cells(geometries, 0, 1, 0., by, bx, intervals,
            rules, yaw_half_width=half, camera_eye=camera[:3])
        family = dict(target_id=0, blocker_id=1, target_yaw=0., blocker_yaw=by, blocker_x=bx,
            camera_pose=camera, role_cells=cells, visibility_domain=dict(valid_intervals=intervals))
        return geometries, rules, family

    def test_joint_pivot_arc_bounds_cover_continuous_scalar_and_common_yaw_extrema(self):
        rng = np.random.default_rng(315)
        for _ in range(80):
            vertices, pivot = rng.normal(size=(4, 2))*.2, rng.normal(size=2)*.1
            angle, half = rng.uniform(-math.pi, math.pi), rng.uniform(0., .15)
            bounds = task._certified_pivot_rotated_bounds(vertices, pivot, angle, (-.03, .07), half)
            points = []
            for theta in np.linspace(0., angle, 101):
                matrix = Rotation.from_euler('z', theta).as_matrix()[:2, :2]
                for parameter in (-.03, .07):
                    points.append((vertices-pivot)@matrix.T+pivot+(0., parameter))
            local = np.concatenate(points)
            world = np.concatenate([local@Rotation.from_euler('z', psi).as_matrix()[:2, :2].T
                                    for psi in np.linspace(-half, half, 21)])
            self.assertGreaterEqual(float(world[:, 0].min()), bounds[0]-1.e-12)
            self.assertLessEqual(float(world[:, 0].max()), bounds[1]+1.e-12)
            self.assertGreaterEqual(float(world[:, 1].min()), bounds[2]-1.e-12)
            self.assertLessEqual(float(world[:, 1].max()), bounds[3]+1.e-12)

    def test_normal_family_keeps_initial_certificates_and_a_fixed_supported_pivot_witness(self):
        import copy
        geometries, rules, source = self.fixture()
        original = copy.deepcopy(source)
        family = task._precompute_normal_pivot_family(source, geometries, rules)
        self.assertIsNotNone(family)
        self.assertEqual(source, original)
        self.assertEqual(family['kind'], 'normal_pivot')
        self.assertNotIn('fixed_objects', family)
        self.assertAlmostEqual(abs(family['pivot']['turn_angle']), math.pi/2, places=10)
        target = np.asarray(CertifiedRoleCellTests.box(geometries[0], (0., 0.), 0.))
        half_access = rules.hand_width*.5+rules.approach_margin
        for source_cell, cell in zip(source['role_cells'], family['role_cells']):
            for name in ('parameter_bounds', 'yaw_bounds', 'blockage_lower_bound', 'camera_x_bounds_over_yaw'):
                self.assertEqual(source_cell[name], cell[name])
            physical = cell['collision_module_bounds']
            half = max(abs(value) for value in cell['yaw_bounds'])
            angle = family['pivot']['turn_angle']
            for parameter in np.linspace(*cell['parameter_bounds'], 3):
                center = np.asarray((source['blocker_x'], parameter))
                pivot = center+family['pivot']['pivot_offset_from_blocker']
                initial = CertifiedRoleCellTests.box(geometries[1], center, source['blocker_yaw'])
                turned = center+family['pivot']['turn_offset_from_blocker']
                final_center = np.asarray((cell['goal_x'], cell['goal_y']))
                for theta in np.linspace(0., angle, 31):
                    transform = Rotation.from_euler('z', theta).as_matrix()[:2, :2]
                    local = (initial-pivot)@transform.T+pivot
                    self.assertGreaterEqual(float(local[:, 0].min()-target[:, 0].max()), rules.gap-2.e-12)
                    for psi in np.linspace(-half, half, 9):
                        matrix = Rotation.from_euler('z', psi).as_matrix()[:2, :2]
                        world = local@matrix.T
                        self.assertGreaterEqual(float(world[:, 0].min()), physical[0]-2.e-12)
                        self.assertLessEqual(float(world[:, 0].max()), physical[1]+2.e-12)
                        self.assertGreaterEqual(float(world[:, 1].min()), physical[2]-2.e-12)
                        self.assertLessEqual(float(world[:, 1].max()), physical[3]+2.e-12)
                for fraction in np.linspace(0., 1., 11):
                    position = turned*(1-fraction)+final_center*fraction
                    local = CertifiedRoleCellTests.box(geometries[1], position, source['blocker_yaw']+angle)
                    for psi in np.linspace(-half, half, 9):
                        matrix = Rotation.from_euler('z', psi).as_matrix()[:2, :2]
                        world = local@matrix.T
                        self.assertGreaterEqual(float(world[:, 0].min()), physical[0]-2.e-12)
                        self.assertLessEqual(float(world[:, 0].max()), physical[1]+2.e-12)
                        self.assertGreaterEqual(float(world[:, 1].min()), physical[2]-2.e-12)
                        self.assertLessEqual(float(world[:, 1].max()), physical[3]+2.e-12)
                        if fraction == 1.:
                            if cell['side'] > 0:
                                self.assertGreaterEqual(float(world[:, 1].min()), half_access+rules.gap-2.e-12)
                            else:
                                self.assertLessEqual(float(world[:, 1].max()), -half_access-rules.gap+2.e-12)

    def test_normal_family_does_not_repair_invalid_initial_domains(self):
        geometries, rules, source = self.fixture()
        self.assertIsNone(task._precompute_normal_pivot_family(dict(source, role_cells=[]), geometries, rules))
        with self.assertRaises(ValueError):
            task._precompute_normal_pivot_family(dict(source, fixed_objects={2:(0.,0.,0.)}), geometries, rules)
        with self.assertRaises(ValueError):
            task._precompute_normal_pivot_family(dict(source, visibility_domain=dict(valid_intervals=((.2,.3),))), geometries, rules)
        with self.assertRaises(ValueError):
            task._certified_pivot_rotated_bounds(np.eye(2), np.zeros(2), .2, (.1,-.1), .1)


class CertifiedGuardedFamilyTests(unittest.TestCase):
    @staticmethod
    def fixture():
        geometry = (task.ObjectGeometry('target', (-.02, -.02, 0.), (.02, .02, .1)),
                    task.ObjectGeometry('blocker', (-.08, -.03, 0.), (.08, .03, .1)),
                    task.ObjectGeometry('guard', (-.015, -.015, 0.), (.015, .015, .1)))
        rules = task.SpawnRules(x_range=(-.5, .5), y_range=(-.5, .5), min_objects=3, max_objects=3)
        by = math.pi/4
        bx = .02+.11/math.sqrt(2)+.0065
        cell = dict(parameter_bounds=(.029, .031), yaw_bounds=(-.02, .02))
        family = dict(target_id=0,blocker_id=1,target_yaw=0.,blocker_yaw=by,blocker_x=bx,
                      camera_pose=(.8,0.,1.2,.5,-.5,-.5,.5),role_cells=[cell],
                      visibility_domain=dict(valid_intervals=((.025,.035),)))
        return geometry,rules,family

    def test_compiler_builds_a_continuous_guard_family_without_source_mutation_or_search(self):
        import copy
        geometry,rules,base = self.fixture()
        original = copy.deepcopy(base)
        with patch.object(task, '_constructive_lateral_guards', side_effect=AssertionError('Old guard candidates')), \
             patch.object(task, '_scene_proposals', side_effect=AssertionError('Scene redraw')):
            family = task._precompute_guarded_role_family(base,geometry,rules,2)
        self.assertIsNotNone(family)
        self.assertEqual(base,original)
        self.assertEqual(set(family['fixed_objects']),{2})
        cell = family['role_cells'][0]
        self.assertIn('goal_x',cell)
        self.assertIn('translation_y_bounds',cell)
        cert = family['guard_certificate']
        self.assertGreaterEqual(cert['witness_gap_lower_bound'],rules.gap)
        self.assertGreater(cert['initial_visual_separation'],0.)
        self.assertGreater(cert['pure_up_configuration_margin'],0.)
        self.assertGreater(cert['pure_up_remaining_access_area'],0.)
        self.assertGreater(cert['pure_down_remaining_access_area'],0.)
        target_yaw,blocker_yaw = base['target_yaw'],base['blocker_yaw']
        guard_x,guard_y,guard_yaw = family['fixed_objects'][2]
        pivot_offset = np.asarray(family['pivot']['pivot_offset_from_blocker'])
        turn_offset = np.asarray(family['pivot']['turn_offset_from_blocker'])
        theta = family['pivot']['turn_angle']
        xmin,xmax = rules.x_range[0]+rules.edge_margin,rules.x_range[1]-rules.edge_margin
        ymin,ymax = rules.y_range[0]+rules.edge_margin,rules.y_range[1]-rules.edge_margin
        for s in np.linspace(*cell['parameter_bounds'],5):
            initial_center = np.array((base['blocker_x'],s))
            local_pivot = initial_center+pivot_offset
            turned_center = initial_center+turn_offset
            initial = CertifiedRoleCellTests.box(geometry[1],initial_center,blocker_yaw)
            for psi in np.linspace(*cell['yaw_bounds'],7):
                c,sine = math.cos(psi),math.sin(psi)
                r = np.asarray(((c,-sine),(sine,c)))
                shift = np.array((sum(cell['translation_x_bounds'])*.5,sum(cell['translation_y_bounds'])*.5))
                target = CertifiedRoleCellTests.box(geometry[0],shift,target_yaw+psi)
                guard = CertifiedRoleCellTests.box(geometry[2],r@(guard_x,guard_y)+shift,guard_yaw+psi)
                current = initial@r.T+shift
                above = current+(0.,cert['pure_up_probe'])
                self.assertEqual(ConstructiveLayoutTests.polygon_distance(above,guard),0.)
                below = current+(0.,cert['pure_down_probe'])
                self.assertLess(below[:,1].min(),rules.y_range[0])
                for fraction in np.linspace(0.,1.,33):
                    turn = theta*fraction
                    cc,ss=math.cos(turn),math.sin(turn)
                    turn_r=np.array(((cc,-ss),(ss,cc)))
                    p=((initial-local_pivot)@turn_r.T+local_pivot)@r.T+shift
                    for other in (target,guard):
                        self.assertGreaterEqual(ConstructiveLayoutTests.polygon_distance(p,other),rules.gap-1.e-10)
                    self.assertGreaterEqual(p[:,0].min(),xmin-1.e-10)
                    self.assertLessEqual(p[:,0].max(),xmax+1.e-10)
                    self.assertGreaterEqual(p[:,1].min(),ymin-1.e-10)
                    self.assertLessEqual(p[:,1].max(),ymax+1.e-10)
                    center=turned_center*(1-fraction)+np.array((cell['goal_x'],cell['goal_y']))*fraction
                    p=CertifiedRoleCellTests.box(geometry[1],r@center+shift,blocker_yaw+theta+psi)
                    for other in (target,guard):
                        self.assertGreaterEqual(ConstructiveLayoutTests.polygon_distance(p,other),rules.gap-1.e-10)
                final=p
                self.assertGreaterEqual(final[:,1].min(),shift[1]+rules.hand_width*.5+rules.approach_margin+rules.gap-1.e-10)

    def test_compiler_returns_none_for_absent_positive_domain_and_rejects_target_guard(self):
        geometry,rules,base=self.fixture()
        outside=dict(parameter_bounds=(.04,.05),yaw_bounds=(-.02,.02))
        self.assertIsNone(task._precompute_guarded_role_family(base,geometry,rules,2,role_cell=outside))
        base['role_cells']=[dict(parameter_bounds=(-.03,-.02),yaw_bounds=(-.02,.02))]
        self.assertIsNone(task._precompute_guarded_role_family(base,geometry,rules,2))
        with self.assertRaisesRegex(ValueError,'distinct non-Target'):
            task._precompute_guarded_role_family(base,geometry,rules,0)


class ConstructiveGuardTests(unittest.TestCase):
    @staticmethod
    def fixture():
        geometry = (
            task.ObjectGeometry("target", (-.02, -.02, 0.), (.02, .02, .1)),
            task.ObjectGeometry("bar", (-.08, -.03, 0.), (.08, .03, .1)),
            task.ObjectGeometry("guard_1", (-.015, -.015, 0.), (.015, .015, .1)),
            task.ObjectGeometry("guard_2", (-.015, -.015, 0.), (.015, .015, .1)),
        )
        angle = math.pi / 4
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        pivot = rotation @ [-.08, .03]
        turn_center = pivot - rotation @ pivot
        start, turn = task.PlanarPose(0., 0., angle), task.PlanarPose(*turn_center, 2 * angle)
        finish = task.PlanarPose(float(turn_center[0]), .3, 2 * angle)
        path = (task.PlanarMotion(start, turn, tuple(pivot)), task.PlanarMotion(turn, finish))
        sample = task.SceneSample((0, 1, 2, 3), 0, 1,
            {0: (-.2, 0., 1.05), 1: (0., 0., 1.05), 2: (.3, .3, 1.05), 3: (.3, -.3, 1.05)},
            {0: 0., 1: angle, 2: 0., 3: 0.}, (-.17, .5, -.06, .06),
            (finish.x, finish.y), path)
        rules = task.SpawnRules(x_range=(-.5, .5), y_range=(-.5, .5), min_objects=4, max_objects=4,
                                min_asset_types=4, max_asset_types=4)
        return sample, geometry, rules

    @staticmethod
    def forbidden_interval(moving, obstacle, x):
        """Independent exact polygon Minkowski intersection with a vertical line."""
        differences = (obstacle[:, None, :] - moving[None, :, :]).reshape(-1, 2)
        polygon = differences[ConvexHull(differences).vertices]
        crossings = []
        for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
            if min(first[0], second[0]) - 1.e-13 <= x <= max(first[0], second[0]) + 1.e-13:
                if abs(second[0] - first[0]) < 1.e-13:
                    crossings.extend((first[1], second[1]))
                else:
                    fraction = (x - first[0]) / (second[0] - first[0])
                    crossings.append(first[1] + fraction * (second[1] - first[1]))
        return (min(crossings), max(crossings)) if crossings else None

    def test_guards_block_both_pure_lateral_directions_without_collision_margin(self):
        sample, geometry, rules = self.fixture()
        original = dict(sample.positions)
        positions, guards = task._constructive_lateral_guards(sample, geometry, rules)
        self.assertEqual(set(guards), {2, 3})
        self.assertEqual(dict(sample.positions), original)
        self.assertEqual(positions[0], original[0])
        self.assertEqual(positions[1], original[1])
        moving = ConstructiveLayoutTests.world_polygon(geometry[1], (0., 0.), sample.yaws[1])
        low, high = -.5, .5
        world = {index: ConstructiveLayoutTests.world_polygon(geometry[index], positions[index], sample.yaws[index])
                 for index in sample.active_ids}
        guard_distances = []
        for guard in guards:
            distance = ConstructiveLayoutTests.polygon_distance(world[guard], world[1])
            self.assertGreaterEqual(distance, .003 - 1.e-8)
            guard_distances.append(distance)
        # The adjacent guard supplies clutter; the other can bound the reserved
        # maneuver exit and therefore need not start within 10 mm of Blocker.
        self.assertLessEqual(min(guard_distances), .010 + 1.e-8)
        for index, polygon in world.items():
            if index == 1:
                continue
            interval = self.forbidden_interval(moving, polygon, 0.)
            if interval is not None:
                if interval[1] <= 0.:
                    low = max(low, interval[1])
                elif interval[0] >= 0.:
                    high = min(high, interval[0])
                else:
                    self.fail("A constructed guard penetrates the initial Blocker")
        access = np.asarray([[-.17, -.06], [.5, -.06], [.5, .06], [-.17, .06]])
        must_leave = self.forbidden_interval(moving, access, 0.)
        self.assertGreater(low, must_leave[0])
        self.assertLess(high, must_leave[1])
        repeated, repeated_guards = task._constructive_lateral_guards(sample, geometry, rules)
        self.assertEqual(repeated, positions)
        self.assertEqual(repeated_guards, guards)

    def test_pivot_witness_remains_clear_of_guards_throughout_motion(self):
        sample, geometry, rules = self.fixture()
        positions, guards = task._constructive_lateral_guards(sample, geometry, rules)
        self.assertEqual(len(guards), 2)
        obstacles = [ConstructiveLayoutTests.world_polygon(geometry[index], positions[index], sample.yaws[index])
                     for index in sample.active_ids if index != sample.blocker_id]
        for motion in sample.clearance_path:
            for amount in np.linspace(0., 1., 61):
                angle = amount * (motion.end.yaw - motion.start.yaw)
                if motion.pivot is None:
                    center = np.array([motion.start.x, motion.start.y]) + amount * np.array(
                        [motion.end.x - motion.start.x, motion.end.y - motion.start.y])
                else:
                    pivot = np.asarray(motion.pivot)
                    rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
                    center = pivot + rotation @ (np.array([motion.start.x, motion.start.y]) - pivot)
                moving = ConstructiveLayoutTests.world_polygon(geometry[1], center, motion.start.yaw + angle)
                for obstacle in obstacles:
                    self.assertGreaterEqual(ConstructiveLayoutTests.polygon_distance(moving, obstacle), .003 - 1.e-8)

    def test_no_depth_reduction_keeps_the_original_layout(self):
        sample, geometry, rules = self.fixture()
        start = sample.clearance_path[0].start
        unchanged = dataclasses.replace(sample, clearance_path=(
            task.PlanarMotion(start, start, (0., 0.)),
            task.PlanarMotion(start, task.PlanarPose(start.x, .3, start.yaw))))
        positions, guards = task._constructive_lateral_guards(unchanged, geometry, rules)
        self.assertEqual(positions, dict(sample.positions))
        self.assertEqual(guards, ())


def episode_reset_fixture(batch=5, selected=2):
    """Native asset-shaped buffers exposing indexed writes, without a simulator."""
    class Objects:
        def __init__(self):
            self.state = torch.full((batch, 4, 13), -7.)
            self.parked_mask = torch.zeros(batch, 4, dtype=torch.bool)
            self.writes = 0
            self.activity_calls = []
            self.operations = []

        def set_active_objects(self, active, env_ids=None):
            value = torch.as_tensor(active, dtype=torch.bool)
            self.activity_calls.append((value.clone(), None if env_ids is None else env_ids.clone()))
            self.parked_mask[env_ids if env_ids is not None else slice(None)] = ~value
            self.operations.append('activity')

        def write_object_state_to_sim(self, value, env_ids):
            self.state[env_ids] = value
            self.writes += 1
            self.operations.append('state')

    class Robot:
        def __init__(self):
            default = torch.zeros(batch, 13)
            default[:, 0], default[:, 3], default[:, 7:] = .2, 1., 8.
            self.data = SimpleNamespace(default_root_state=default)
            self.root = torch.full((batch, 13), -8.)
            self.position = torch.full((batch, 12), -9.)
            self.velocity = torch.full_like(self.position, -10.)
            self.position_target = torch.full_like(self.position, -11.)
            self.velocity_target = torch.full_like(self.position, -12.)
            self.effort_target = torch.full_like(self.position, -13.)
            self.root_writes = self.joint_writes = 0

        def write_root_state_to_sim(self, value, env_ids):
            self.root[env_ids] = value
            self.root_writes += 1

        def write_joint_state_to_sim(self, position, velocity, env_ids):
            self.position[env_ids], self.velocity[env_ids] = position, velocity
            self.joint_writes += 1

        def set_joint_position_target(self, value, env_ids):
            self.position_target[env_ids] = value

        def set_joint_velocity_target(self, value, env_ids):
            self.velocity_target[env_ids] = value

        def set_joint_effort_target(self, value, env_ids):
            self.effort_target[env_ids] = value

    class Scene(dict):
        def __init__(self):
            super().__init__(object_collection=Objects(), robot=Robot(), shelf=Robot())
            self.env_origins = torch.zeros(batch, 3)
            self.env_origins[:, 0] = 2.5 * torch.arange(batch)
            self.env_origins[:, 1] = -1.5 * torch.arange(batch)
            self.writes = 0

        def write_data_to_sim(self):
            self.writes += 1

    runtime = SimpleNamespace(
        reset_counts=torch.zeros(batch, dtype=torch.long),
        robot_regions=torch.full((batch, 6), -21.), robot_tcp=torch.full((batch, 3), -22.),
        visibility=torch.full((batch, 2), -23.),
        camera_pose=torch.full((batch, 7), -24.), camera_intrinsics=torch.full((batch, 3, 3), -25.),
        episode_seeds=torch.full((batch,), -1, dtype=torch.long),
    )
    scene_data = SimpleNamespace(
        object_ids=torch.full((batch, 10), -1, dtype=torch.long),
        target_id=torch.full((batch,), -1, dtype=torch.long), blocker_id=torch.full((batch,), -1, dtype=torch.long),
        access_region=torch.full((batch, 4), -31.), geometry_b=torch.full((batch, 10, 10), -32.),
        geometry_valid=torch.zeros(batch, 10, dtype=torch.bool),
        observed_ids=torch.full((batch, 10), -1, dtype=torch.long),
    )
    from unittest.mock import Mock
    env = SimpleNamespace(num_envs=batch, device='cpu', scene=Scene(), task_scene_data=scene_data,
                          _episode_reset=runtime, task_sensor_reader=SimpleNamespace(reset=Mock()),
                          sim=SimpleNamespace(forward=Mock()))
    rows = torch.arange(selected)
    object_state = torch.zeros(selected, 4, 13)
    object_state[:, :, 0] = torch.arange(4)[None] * .1 + rows[:, None] * .013
    object_state[:, :, 2], object_state[:, :, 3] = 1.05, 1.
    slots = torch.full((selected, 10), -1, dtype=torch.long)
    # Vary roles and inactive bodies by episode, so accidental broadcasting fails.
    slots[:, :3] = (torch.tensor([2, 0, 3])[None] + rows[:, None]) % 4
    active = torch.zeros(selected, 4, dtype=torch.bool)
    active.scatter_(1, slots[:, :3], True)
    geometry = torch.zeros(selected, 10, 10)
    geometry[:, :, 0] = rows[:, None] + 100.
    geometry[:, :, 3], geometry[:, :, 7:] = 1., .1
    valid = torch.zeros(selected, 10, dtype=torch.bool)
    valid[:, :3] = True
    camera_pose = torch.zeros(selected, 7)
    camera_pose[:, 0], camera_pose[:, 3] = rows * .1, 1.
    episodes = task.EpisodeBatch(
        object_state=object_state, object_ids=slots, active=active,
        target_id=slots[:, 0].clone(), blocker_id=slots[:, 1].clone(),
        access_region=torch.tensor([-.8, -.5, -.1, .1])[None].repeat(selected, 1) + rows[:, None] * .001,
        joint_positions=torch.arange(selected * 12).reshape(selected, 12).float() / 100.,
        geometry_b=geometry, geometry_valid=valid, observed_ids=slots.clone(),
        robot_regions=torch.arange(selected * 6).reshape(selected, 6).float() / 10.,
        robot_tcp=torch.tensor([-.4, .1, 1.25])[None].repeat(selected, 1) + rows[:, None] * .01,
        visibility=torch.tensor([.45, .99])[None].repeat(selected, 1) + rows[:, None].remainder(4) * .001,
        camera_pose=camera_pose,
        camera_intrinsics=torch.eye(3)[None].repeat(selected, 1, 1) * (rows[:, None, None] + 1),
        seeds=rows + 43,
    )
    return env, episodes


class EpisodeResetTests(unittest.TestCase):
    @staticmethod
    def batch_fixture(geometries, rules, streams, roles, active_ids, **kwargs):
        return [task.sample_scene(geometries,rules,rng,target_id=t,blocker_id=b,active_ids=ids)
                for rng,(t,b),ids in zip(streams,roles,active_ids)]

    @staticmethod
    def builder_fixture():
        """Real proposal/commit buffers with rendering and collision search substituted."""
        from contextlib import nullcontext
        from unittest.mock import Mock

        env, _ = episode_reset_fixture()
        objects, robot = env.scene['object_collection'], env.scene['robot']
        default = torch.zeros(env.num_envs, 4, 13)
        default[:, :, 3] = 1.
        objects.data = SimpleNamespace(object_state_w=objects.state, default_object_state=default)
        for field, value in (
            ('root_state_w', robot.root), ('joint_pos', robot.position), ('joint_vel', robot.velocity),
            ('joint_pos_target', robot.position_target), ('joint_vel_target', robot.velocity_target),
            ('joint_effort_target', robot.effort_target),
        ):
            setattr(robot.data, field, value)
        robot.root_physx_view = SimpleNamespace(get_link_transforms=Mock(return_value=torch.zeros(5, 1, 7)))
        env.cfg = SimpleNamespace(seed=73, target_asset=None, blocker_asset=None,
                                  initial_vision=SimpleNamespace(max_layout_attempts=2),
                                  robot_start=task.RobotStartRules(), mdp=SimpleNamespace(max_objects=10))
        env.spawn_rules, env.spawn_geometries = task.SpawnRules(generation_mode="legacy"), ()
        env.task_scene_data.lower = torch.tensor([[-.05, -.05, 0.]]).repeat(4, 1)
        env.task_scene_data.upper = torch.tensor([[.05, .05, .1]]).repeat(4, 1)
        env.task_scene_data.upright = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        env._episode_reset.specs = tuple(SimpleNamespace(asset_name=f'type_{index}') for index in range(4))
        env._episode_reset.vision = SimpleNamespace(capture_session=nullcontext, capture_batch=Mock(), capture_view_domain=Mock())
        proposal_count = 0

        def sample_scene(geometries, rules, rng, *, target_id, blocker_id, active_ids):
            nonlocal proposal_count
            proposal_count += 1
            return task.SceneSample(
                (0, 1, 2, 3), target_id, blocker_id,
                {index: (-.8 + proposal_count * .01, index * .03, 1.05) for index in range(4)},
                {index: 0. for index in range(4)}, (-.8, -.5, -.1, .1), (-.7, -.2),
            )

        def sample_robot_start(env, samples, seeds, **kwargs):
            return [(np.full(12, sample.positions[0][0]), (-.4, -.2, -.1, .1, 1.2, 1.3),
                     np.array([-.3, 0., 1.25]), .08, 1) for sample in samples]

        def snapshot(env_id, object_pose=None):
            geometry = torch.zeros(10, 10)
            geometry[:4, :3] = (objects.state[env_id, :, :3] if object_pose is None else object_pose[:, :3]) - env.scene.env_origins[env_id]
            geometry[:4, 3], geometry[:4, 7:] = 1., .1
            valid, observed = torch.zeros(10, dtype=torch.bool), torch.full((10,), -1, dtype=torch.long)
            valid[:4], observed[:4] = True, torch.arange(4)
            return SimpleNamespace(
                geometry_b=geometry, valid=valid, observed_ids=observed, visible_fractions=(.5, 1.),
                camera_pose_w=torch.tensor([env_id, 0., 2., 1., 0., 0., 0.]), intrinsics=torch.eye(3),
            )

        return env, sample_scene, sample_robot_start, snapshot

    def test_packing_applies_upright_once_corrects_off_center_roots_and_clears_all_velocity(self):
        default = torch.zeros(4, 13)
        default[:, :3] = torch.tensor([7., 8., 9.])
        default[:, 3], default[:, 7:] = 1., 42.
        lower = torch.tensor([[-.01, -.04, -.02], [-.1, -.1, 0.], [-.02, -.06, -.01], [-.1, -.1, 0.]])
        upper = torch.tensor([[.05, .02, .10], [.1, .1, .1], [.08, .02, .2], [.1, .1, .1]])
        upright = torch.tensor([[1., 0., 0., 0.]]).repeat(4, 1)
        upright[2] = torch.tensor([math.sqrt(.5), -math.sqrt(.5), 0., 0.])
        sample = task.SceneSample((2, 0), 2, 0, {2: (-.7, .1, 1.06), 0: (-.8, -.1, 1.07)},
                                  {2: math.pi/2, 0: 0.}, (-.8, -.5, -.1, .1), (-.7, -.2))
        states, active, slots = task.pack_object_states(default, sample, lower, upper, upright)
        torch.testing.assert_close(states[2, :3], torch.tensor([-.72, .07, 1.06]))
        torch.testing.assert_close(states[0, :3], torch.tensor([-.82, -.09, 1.07]))
        expected_xyzw = (Rotation.from_euler('z', math.pi/2) * Rotation.from_euler('x', -math.pi/2)).as_quat()
        expected_wxyz = torch.tensor(expected_xyzw[[3, 0, 1, 2]], dtype=states.dtype)
        torch.testing.assert_close(states[2, 3:7], expected_wxyz)
        torch.testing.assert_close(states[[1, 3], :7], default[[1, 3], :7])
        self.assertEqual(int(states[:, 7:].count_nonzero()), 0)
        self.assertTrue(torch.equal(active, torch.tensor([True, False, True, False])))
        self.assertTrue(torch.equal(slots[:2], torch.tensor([2, 0])))
        self.assertTrue((slots[2:] == -1).all())
        self.assertTrue((default[:, 7:] == 42.).all())

    def test_independent_reset_writes_unsorted_selected_rows_and_clears_physical_targets(self):
        env, episodes = episode_reset_fixture()
        ids = torch.tensor([3, 1])
        untouched = torch.tensor([0, 2, 4])
        object_state_before = episodes.object_state.clone()
        task.apply_episode_batch(env, ids, episodes)
        objects, robot, data = env.scene['object_collection'], env.scene['robot'], env.task_scene_data
        expected = object_state_before.clone()
        expected[..., :3] += env.scene.env_origins[ids, None]
        torch.testing.assert_close(objects.state[ids], expected)
        torch.testing.assert_close(episodes.object_state, object_state_before)
        self.assertTrue((objects.state[untouched] == -7.).all())
        self.assertTrue((robot.root[untouched] == -8.).all())
        self.assertTrue((robot.position[untouched] == -9.).all())
        self.assertTrue((robot.velocity[untouched] == -10.).all())
        self.assertTrue((robot.position_target[untouched] == -11.).all())
        self.assertTrue((robot.velocity_target[untouched] == -12.).all())
        self.assertTrue((robot.effort_target[untouched] == -13.).all())
        torch.testing.assert_close(robot.root[ids, :3], env.scene.env_origins[ids] + torch.tensor([.2, 0., 0.]))
        torch.testing.assert_close(robot.position[ids], episodes.joint_positions)
        torch.testing.assert_close(robot.position_target[ids], episodes.joint_positions)
        for tensor in (robot.root[ids, 7:], robot.velocity[ids], robot.velocity_target[ids], robot.effort_target[ids]):
            self.assertEqual(int(tensor.count_nonzero()), 0)
        torch.testing.assert_close(objects.parked_mask[ids], ~episodes.active)
        self.assertEqual(objects.operations, ['activity', 'state'])
        self.assertEqual(len(objects.activity_calls), 1)
        active, active_ids = objects.activity_calls[0]
        torch.testing.assert_close(active, episodes.active)
        torch.testing.assert_close(active_ids, ids)
        self.assertEqual(int(objects.parked_mask[untouched].count_nonzero()), 0)
        for field in ('object_ids', 'target_id', 'blocker_id', 'access_region',
                      'geometry_b', 'geometry_valid', 'observed_ids'):
            torch.testing.assert_close(getattr(data, field)[ids], getattr(episodes, field))
        self.assertTrue((data.geometry_b[untouched] == -32.).all())
        self.assertTrue((data.object_ids[untouched] == -1).all())
        self.assertTrue((data.target_id[untouched] == -1).all())
        self.assertTrue((data.blocker_id[untouched] == -1).all())
        self.assertTrue((data.access_region[untouched] == -31.).all())
        self.assertFalse(data.geometry_valid[untouched].any())
        self.assertTrue((data.observed_ids[untouched] == -1).all())
        torch.testing.assert_close(env._episode_reset.robot_tcp[ids], episodes.robot_tcp + env.scene.env_origins[ids])
        for field in ('robot_regions', 'visibility', 'camera_pose', 'camera_intrinsics'):
            torch.testing.assert_close(getattr(env._episode_reset, field)[ids], getattr(episodes, field))
        torch.testing.assert_close(env._episode_reset.episode_seeds[ids], episodes.seeds)
        self.assertTrue((env._episode_reset.robot_tcp[untouched] == -22.).all())
        self.assertTrue((env._episode_reset.camera_pose[untouched] == -24.).all())
        self.assertTrue((env._episode_reset.episode_seeds[untouched] == -1).all())
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.tensor([0, 1, 0, 1, 0]))
        # Partial resets must not flush force buffers for untouched environments.
        self.assertEqual((objects.writes, robot.root_writes, robot.joint_writes, env.scene.writes), (1, 1, 1, 0))
        env.task_sensor_reader.reset.assert_called_once()
        torch.testing.assert_close(env.task_sensor_reader.reset.call_args.args[0], ids)
        env.sim.forward.assert_called_once()

    def test_activity_transition_failure_precedes_all_asset_and_metadata_writes(self):
        env, episodes = episode_reset_fixture()
        ids = torch.tensor([3, 1])
        objects, robot, shelf = (env.scene[name] for name in ('object_collection', 'robot', 'shelf'))
        before = [value.clone() for value in (objects.state, robot.root, robot.position, shelf.root,
                                               env.task_scene_data.geometry_b, env._episode_reset.reset_counts)]
        with patch.object(objects, 'set_active_objects', side_effect=ValueError('global mask change')):
            with self.assertRaisesRegex(ValueError, 'global mask change'):
                task.apply_episode_batch(env, ids, episodes)
        for actual, expected in zip((objects.state, robot.root, robot.position, shelf.root,
                                     env.task_scene_data.geometry_b, env._episode_reset.reset_counts), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual((objects.writes, robot.root_writes, robot.joint_writes, shelf.root_writes), (0, 0, 0, 0))
        env.task_sensor_reader.reset.assert_not_called()
        env.sim.forward.assert_not_called()

    def test_batch_application_has_no_tensor_scalar_reads_or_per_environment_writes(self):
        env, episodes = episode_reset_fixture(1024, selected=512)
        ids = torch.arange(0, 1024, 2)
        def reject(*args, **kwargs):
            raise AssertionError('Host tensor read in batched episode application')
        with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'tolist', reject), patch.object(torch.Tensor, 'cpu', reject), \
             patch.object(torch.Tensor, 'numpy', reject):
            task.apply_episode_batch(env, ids, episodes)
        self.assertEqual(env.scene['object_collection'].writes, 1)
        self.assertEqual(env.scene['robot'].joint_writes, 1)
        torch.testing.assert_close(env._episode_reset.episode_seeds[ids], episodes.seeds)
        torch.testing.assert_close(env.scene['robot'].position[ids], episodes.joint_positions)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 512)

    @staticmethod
    def shared_fixture(batch=5):
        env, template = episode_reset_fixture(batch=batch, selected=1)
        env.common_step_counter, env.max_episode_length = 0, 400
        env.episode_length_buf = torch.zeros(env.num_envs, dtype=torch.long)
        env.cfg = SimpleNamespace(seed=73, initial_vision=SimpleNamespace(max_layout_attempts=2),
                                  robot_start=task.RobotStartRules(), mdp=SimpleNamespace(max_objects=10))
        env._episode_reset.specs = (None,) * 4
        env._episode_reset.support_heights = torch.ones(4)
        sample = task.SceneSample(
            (2, 0, 3), 2, 0, {i: (-.7, i * .03, 1.05) for i in (2, 0, 3)},
            {i: 0. for i in (2, 0, 3)}, (-.8, -.5, -.1, .1), (-.7, -.2))
        return env, template, sample

    @staticmethod
    def shared_starts(env, samples, seeds, **kwargs):
        return [(torch.full((12,), (seed % 10007) / 10007.), torch.arange(6).float(),
                 torch.tensor([-.3, (seed % 97) / 1000., 1.25]), .08, 1) for seed in seeds]

    def test_shared_scene_initializes_once_and_partial_reset_only_reruns_ik(self):
        env, template, sample = self.shared_fixture()
        with patch.object(task, '_build_episode_batch', return_value=(template, sample)) as build, \
             patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts) as ik:
            task.reset_scene(env)
            before = env.scene['object_collection'].state.clone()
            joints = env.scene['robot'].position.clone()
            vision = env.task_scene_data.geometry_b.clone()
            # Physical disturbance must be restored, not cached as the next layout.
            env.scene['object_collection'].state[4, :, :3] += .1
            env.common_step_counter = 17
            task.reset_scene(env, torch.tensor([4, 1]))
        self.assertEqual(build.call_count, 1)
        self.assertEqual(build.call_args.kwargs, {'scene_generation': 0})
        torch.testing.assert_close(build.call_args.args[2], torch.tensor([0]))
        self.assertEqual(ik.call_count, 2)
        self.assertEqual(len(ik.call_args.args[1]), 2)
        torch.testing.assert_close(env.scene['object_collection'].state, before)
        torch.testing.assert_close(env.task_scene_data.geometry_b, vision)
        torch.testing.assert_close(env.scene['robot'].position[[0, 2, 3]], joints[[0, 2, 3]])
        self.assertFalse(torch.equal(env.scene['robot'].position[[4, 1]], joints[[4, 1]]))
        self.assertEqual(env.scene['robot'].position.unique(dim=0).shape[0], 5)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.tensor([1, 2, 1, 1, 2]))
        self.assertEqual(env._episode_reset.next_scene_step, 400)
        # Local camera template must not accumulate world origins on repeated resets.
        torch.testing.assert_close(env._episode_reset.camera_pose[:, :3],
                                   template.camera_pose[:, :3] + env.scene.env_origins)
        torch.testing.assert_close(env.task_scene_data.geometry_b,
                                   template.geometry_b.expand(5, -1, -1))

    def test_shared_round_boundary_truncates_recently_reset_rows_and_replaces_layout(self):
        from contact_manipulation.mdp.terminations import scene_time_out
        env, first, sample = self.shared_fixture()
        _, second = episode_reset_fixture(selected=1)
        second.object_state[:, :, 1] += .027
        second.geometry_b[:, :, 1] += .031
        with patch.object(task, '_build_episode_batch', side_effect=[(first, sample), (second, sample)]) as build, \
             patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts):
            task.reset_scene(env)
            env.common_step_counter = 399
            task.reset_scene(env, torch.tensor([4, 1]))
            self.assertFalse(scene_time_out(env).any())
            env.common_step_counter = 400
            self.assertTrue(scene_time_out(env).all())
            task.reset_scene(env, torch.tensor([4, 3, 2, 1, 0]))
        self.assertEqual(build.call_count, 2)
        self.assertEqual(build.call_args.kwargs, {'scene_generation': 1})
        self.assertEqual(env._episode_reset.next_scene_step, 800)
        self.assertFalse(scene_time_out(env).any())
        torch.testing.assert_close(env.task_scene_data.geometry_b, second.geometry_b.expand(5, -1, -1))
        local = env.scene['object_collection'].state[..., :3] - env.scene.env_origins[:, None]
        torch.testing.assert_close(local, second.object_state[..., :3].expand(5, -1, -1))

    def test_partial_reset_cannot_create_or_replace_a_shared_scene(self):
        env, _, _ = self.shared_fixture()
        with patch.object(task, '_build_episode_batch') as build:
            with self.assertRaisesRegex(RuntimeError, 'all environments reset together'):
                task.reset_scene(env, torch.tensor([4, 1]))
        build.assert_not_called()
        self.assertEqual(env.scene['object_collection'].writes, 0)

    def test_blocker_filter_binding_follows_committed_scene_and_partial_reset_is_idempotent(self):
        from unittest.mock import Mock
        env, first, sample = self.shared_fixture()
        _, second = episode_reset_fixture(selected=1)
        second.blocker_id.fill_(3)
        second.object_ids[:, 1:3] = torch.tensor([3, 0])
        second.observed_ids.copy_(second.object_ids)
        second_sample = dataclasses.replace(sample, blocker_id=3)
        calls, native_rebinds, bound = [], [], None

        def configure(blocker_id):
            nonlocal bound
            # If reset_scene binds before apply_episode_batch, initial -1 or
            # the previous scene's Blocker remains here and this fails.
            self.assertTrue((env.task_scene_data.blocker_id == blocker_id).all())
            self.assertEqual(env._episode_reset.shared_sample.blocker_id, blocker_id)
            calls.append((blocker_id, env.scene['object_collection'].writes))
            if bound != blocker_id:
                native_rebinds.append(blocker_id)
                bound = blocker_id

        env._blocker_contact_hold = SimpleNamespace(configure_blocker=Mock(side_effect=configure))
        with patch.object(task, '_build_episode_batch', side_effect=[(first, sample), (second, second_sample)]), \
             patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts):
            task.reset_scene(env)
            env.common_step_counter = 17
            task.reset_scene(env, torch.tensor([4, 1]))
            self.assertEqual(native_rebinds, [sample.blocker_id])
            env.common_step_counter = env._episode_reset.next_scene_step
            task.reset_scene(env, torch.arange(env.num_envs))
        self.assertEqual(calls, [(sample.blocker_id, 1), (sample.blocker_id, 2), (3, 3)])
        self.assertEqual(native_rebinds, [sample.blocker_id, 3])

    def test_shared_robot_retry_keeps_accepted_rows_and_reuses_perception(self):
        env, template, sample = self.shared_fixture()
        calls = []
        def starts(env, samples, seeds, **kwargs):
            calls.append(seeds)
            result = self.shared_starts(env, samples, seeds)
            if len(calls) == 1:
                result[1] = None
            return result
        with patch.object(task, '_sample_robot_starts', side_effect=starts):
            batch = task._shared_reset_batch(env, env._episode_reset, torch.tensor([4, 1]), template, sample)
        self.assertEqual([len(seeds) for seeds in calls], [2, 1])
        expected = self.shared_starts(env, [sample], calls[0][:1])[0][0]
        torch.testing.assert_close(batch.joint_positions[0], expected)
        torch.testing.assert_close(batch.geometry_b, template.geometry_b.expand(2, -1, -1))
        self.assertEqual(env.scene['object_collection'].writes, 0)

    def test_shared_proposals_are_reused_until_the_template_changes(self):
        env, template, sample = self.shared_fixture()
        with patch.object(task, '_sample_robot_starts', side_effect=self.shared_starts), \
             patch.object(task, '_episode_proposals', wraps=task._episode_proposals) as propose:
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([4, 1]), template, sample)
            cached = env._episode_reset.shared_inputs
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([2]), template, sample)
            self.assertIs(env._episode_reset.shared_inputs, cached)
            self.assertEqual(propose.call_count, 1)
            replacement = dataclasses.replace(template, geometry_b=template.geometry_b.clone())
            task._shared_reset_batch(env, env._episode_reset, torch.tensor([0]), replacement, sample)
            self.assertIsNot(env._episode_reset.shared_inputs, cached)
            self.assertEqual(propose.call_count, 2)

    def test_shared_ik_exhaustion_restores_verified_anchor_without_changing_neighboring_rows(self):
        import json
        import tempfile
        env, template, sample = self.shared_fixture()
        env._episode_reset.shared_template, env._episode_reset.shared_sample = template, sample
        env._episode_reset.next_scene_step = 400
        before = {field.name: getattr(template, field.name).clone() for field in dataclasses.fields(template)}
        calls = []
        def fail(_env, samples, seeds, **kwargs):
            calls.append(kwargs)
            return [None] * len(samples)
        with tempfile.TemporaryDirectory(prefix='contact_reset_diagnostics_') as directory:
            env.cfg.log_dir = directory
            with patch.object(task, '_sample_robot_starts', side_effect=fail):
                task.reset_scene(env, torch.tensor([4, 1]))
            records = [json.loads(line) for line in (Path(directory) / 'robot_start_fallbacks.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['env_ids'], [4, 1])
        self.assertEqual(records[0]['reset_counts'], [0, 0])
        self.assertEqual(records[0]['reason'], 'bounded_robot_start_search_exhausted_verified_template_restored')
        self.assertIsNone(env._episode_reset.last_robot_start_fallback)
        rules = env.cfg.robot_start
        self.assertEqual(len(calls), rules.reset_attempts + rules.recovery_attempts)
        self.assertEqual([bool(call.get('recovery', False)) for call in calls],
                         [False] * rules.reset_attempts + [True] * rules.recovery_attempts)
        self.assertEqual(env.scene['object_collection'].writes, 1)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.tensor([0, 1, 0, 0, 1]))
        torch.testing.assert_close(env.scene['robot'].position[[4, 1]], template.joint_positions.expand(2, -1))
        torch.testing.assert_close(env._episode_reset.robot_tcp[[4, 1]],
                                   template.robot_tcp + env.scene.env_origins[[4, 1]])
        self.assertTrue((env.scene['robot'].position[[0, 2, 3]] == -9.).all())
        self.assertTrue((env.scene['object_collection'].state[[0, 2, 3]] == -7.).all())
        self.assertIs(env._episode_reset.shared_template, template)
        self.assertIs(env._episode_reset.shared_sample, sample)
        self.assertEqual(env._episode_reset.pending_reset_metrics['Reset/robot_start_fallback_count'], 2.)
        self.assertEqual(env._episode_reset.pending_reset_metrics['Reset/robot_start_fallback_fraction'], 1.)
        for field, expected in before.items():
            torch.testing.assert_close(getattr(template, field), expected, rtol=0, atol=0)

    def test_1024_shared_reset_preserves_accepted_starts_when_one_row_uses_anchor_recovery(self):
        env, template, sample = self.shared_fixture(batch=1024)
        calls, first = [], None
        def starts(env, samples, seeds, **kwargs):
            nonlocal first
            calls.append((seeds.copy(), kwargs))
            if first is None:
                first = self.shared_starts(env, samples, seeds)
                result = first.copy()
                result[513] = None
                return result
            return [None] * len(samples)
        ids = torch.arange(1024).roll(17)
        with patch.object(task, '_sample_robot_starts', side_effect=starts):
            batch = task._shared_reset_batch(env, env._episode_reset, ids, template, sample)
        rules = env.cfg.robot_start
        self.assertEqual([len(seeds) for seeds, _ in calls],
                         [1024] + [1] * (rules.reset_attempts + rules.recovery_attempts - 1))
        mask = torch.arange(1024) != 513
        expected = torch.stack([start[0] for start in first])
        torch.testing.assert_close(batch.joint_positions[mask], expected[mask], rtol=0, atol=0)
        torch.testing.assert_close(batch.joint_positions[513], template.joint_positions[0], rtol=0, atol=0)
        torch.testing.assert_close(batch.robot_tcp[513], template.robot_tcp[0], rtol=0, atol=0)
        torch.testing.assert_close(batch.geometry_b, template.geometry_b.expand(1024, -1, -1), rtol=0, atol=0)
        torch.testing.assert_close(batch.object_state, template.object_state.expand(1024, -1, -1), rtol=0, atol=0)
        self.assertEqual(env.scene['object_collection'].writes, 0)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)
        fallback = env._episode_reset.last_robot_start_fallback
        self.assertEqual(fallback['env_ids'], [int(ids[513])])
        self.assertEqual(fallback['seeds'], [int(batch.seeds[513])])
        self.assertEqual(fallback['reset_counts'], [0])
        metrics = env._episode_reset.pending_reset_metrics
        self.assertEqual(metrics['Reset/robot_start_fallback_fraction'], 1. / 1024)
        self.assertEqual(metrics['Reset/robot_start_fallback_total'], 1.)

    def test_shared_recovery_accepts_new_pose_before_template_fallback_and_counts_separately(self):
        env, template, sample = self.shared_fixture()
        calls, accepted = [], None
        def starts(env, samples, seeds, **kwargs):
            nonlocal accepted
            calls.append(kwargs)
            if not kwargs['recovery']:
                return [None] * len(samples)
            accepted = self.shared_starts(env, samples, seeds)
            return accepted
        with patch.object(task, '_sample_robot_starts', side_effect=starts):
            batch = task._shared_reset_batch(env, env._episode_reset, torch.tensor([4, 1]), template, sample)
        self.assertEqual(len(calls), env.cfg.robot_start.reset_attempts + 1)
        torch.testing.assert_close(batch.joint_positions, torch.stack([row[0] for row in accepted]), rtol=0, atol=0)
        self.assertFalse(torch.equal(batch.joint_positions, template.joint_positions.expand(2, -1)))
        self.assertIsNone(getattr(env._episode_reset, 'last_robot_start_fallback', None))
        metrics = env._episode_reset.pending_reset_metrics
        self.assertEqual(metrics['Reset/robot_start_recovery_count'], 2.)
        self.assertEqual(metrics['Reset/robot_start_fallback_count'], 0.)

    def test_episode_seeds_are_reproducible_and_independent_of_reset_batch_order(self):
        expected = {(index, count): task._episode_seed(123, index, count)
                    for index in range(1024) for count in range(3)}
        self.assertEqual(len(set(expected.values())), 3072)
        for index in (1023, 7, 0, 83):
            for count in (2, 0, 1):
                value = task._episode_seed(123, index, count)
                self.assertEqual(value, expected[index, count])
                self.assertGreaterEqual(value, 0)
                self.assertLess(value, 2**63)
        self.assertNotEqual(task._episode_seed(123, 7, 2), task._episode_seed(124, 7, 2))

    def test_shared_domain_probe_uses_distinct_single_targets_and_preserves_anchor_inputs(self):
        env, template, sample = self.shared_fixture()
        start = (template.joint_positions[0], template.robot_regions[0], template.robot_tcp[0], .08, 1)
        proposals = task._episode_proposals([sample], 4, 'cpu', torch.float32)
        inputs_before = [value.clone() for value in proposals]
        outcomes = [start, None, start, None, start, None, start, None]
        with patch.object(task, '_sample_robot_starts', return_value=outcomes) as sampler:
            actual = task._robot_start_domain_successes(env, sample, start, proposals, 571)
            again = task._robot_start_domain_successes(env, sample, start, proposals, 571)
        self.assertEqual((actual, again), (4, 4))
        first, second = sampler.call_args_list
        expected_seeds = [task._episode_seed(571, index, 0) for index in range(8)]
        self.assertEqual(first.args[2], expected_seeds)
        self.assertEqual(second.args[2], expected_seeds)
        self.assertEqual(len(set(expected_seeds)), 8)
        self.assertTrue(all(value is sample for value in first.args[1]))
        self.assertEqual(first.kwargs['max_attempts'], 1)
        self.assertFalse(first.kwargs['return_tensors'])
        cache = first.kwargs['scene_cache']
        self.assertIs(cache.proposals, proposals)
        self.assertIs(cache.sample, sample)
        torch.testing.assert_close(cache.template.joint_positions, template.joint_positions)
        torch.testing.assert_close(cache.template.robot_tcp, template.robot_tcp)
        for value, original in zip(proposals, inputs_before):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        result = torch.zeros(8, 17)
        result[:4, -1] = 1.
        # Tensor-return counting is host-only arithmetic here; no GPU runtime is required.
        with patch.object(task.torch, 'device', return_value=SimpleNamespace(type='cuda')), \
             patch.object(task, '_sample_robot_starts', return_value=(result, template.robot_regions)) as sampler:
            self.assertEqual(task._robot_start_domain_successes(env, sample, start, proposals, 571), 4)
        self.assertTrue(sampler.call_args.kwargs['return_tensors'])

    def test_shared_builder_rejects_narrow_anchor_domain_then_accepts_threshold(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene) as propose, \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_robot_start_domain_successes', side_effect=[3, 4]) as domain, \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(1, dtype=torch.bool)) as physical:
            template, sample = task._build_episode_batch(env, env._episode_reset, torch.tensor([0]), scene_generation=9)
        self.assertEqual((propose.call_count, domain.call_count, physical.call_count), (2, 2, 1))
        self.assertEqual(env.scene['object_collection'].writes, 0)
        self.assertEqual(template.object_state.shape[0], 1)
        self.assertAlmostEqual(sample.positions[0][0], -.78)
        self.assertNotEqual(domain.call_args_list[0].args[-1], domain.call_args_list[1].args[-1])
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)

    def test_shared_builder_rejects_anchor_only_domain_and_rolls_back(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        before = env.scene['object_collection'].state.clone()
        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_robot_start_domain_successes', return_value=1) as domain, \
             patch.object(task, '_check_physical_robot_starts') as physical:
            with self.assertRaisesRegex(RuntimeError, 'only 1/8.*requires 4'):
                task._build_episode_batch(env, env._episode_reset, torch.tensor([0]), scene_generation=9)
        self.assertEqual(domain.call_count, env.cfg.initial_vision.max_layout_attempts)
        physical.assert_not_called()
        torch.testing.assert_close(env.scene['object_collection'].state, before, rtol=0, atol=0)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)

    def test_builder_retries_only_rejected_rows_and_retains_corresponding_initial_vision(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        ids = torch.tensor([4, 1])
        captures, accepted_geometry = [], {}
        objects = env.scene['object_collection']
        untouched_before = objects.state[[0, 2, 3]].clone()

        def capture(**kwargs):
            env_id = kwargs['env_id']
            captures.append(env_id)
            if env_id == 1 and captures.count(1) == 1:
                return None
            result = snapshot(env_id, kwargs.get("object_pose"))
            accepted_geometry[env_id] = result.geometry_b.clone()
            return result

        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            capture(**request, object_pose=pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene) as propose, \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_check_physical_robot_starts', side_effect=lambda env,samples,*a: torch.ones(len(samples),dtype=torch.bool)):
            episodes = task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(captures, [4, 1, 1])
        self.assertEqual(propose.call_count, 3)
        self.assertEqual(objects.writes, 0)
        self.assertEqual(env.scene['robot'].root_physx_view.get_link_transforms.call_count, 2)
        torch.testing.assert_close(episodes.geometry_b[0], accepted_geometry[4])
        torch.testing.assert_close(episodes.geometry_b[1], accepted_geometry[1])
        torch.testing.assert_close(episodes.geometry_b[:, :4, :3], episodes.object_state[:, :, :3])
        self.assertNotEqual(float(episodes.object_state[0, 0, 0]), float(episodes.object_state[1, 0, 0]))
        torch.testing.assert_close(objects.state[[0, 2, 3]], untouched_before)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.zeros(5, dtype=torch.long))
        self.assertTrue((env.task_scene_data.geometry_b == -32.).all())
        env.task_sensor_reader.reset.assert_not_called()

    def test_parameterized_builder_uses_observation_only_and_keeps_predictions_separate(self):
        from unittest.mock import Mock
        env, propose, starts, snapshot = self.builder_fixture()
        env.spawn_rules.generation_mode = 'parameterized'
        env._episode_reset.parameter_catalog = object()
        camera = (.2, .01, 1.3, 1., 0., 0., 0.)
        def generate(catalog, geometries, rules, rng, target, blocker):
            return dataclasses.replace(propose(geometries, rules, rng, target_id=target, blocker_id=blocker,
                active_ids=None), camera_pose=camera,
                parameter_certificate=dict(predicted_role_visibility=(.6, 1.), live_visibility_measured=False))
        def observe(requests, cameras, *, object_poses, root_poses):
            results = [snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
            for result in results:
                result.visible_fractions = None
            return results
        env._episode_reset.vision.observe_at_viewpoint = Mock(side_effect=observe)
        with patch.object(task, 'sample_certified_episode', side_effect=generate) as draw, \
             patch.object(task, 'sample_scenes', side_effect=AssertionError('runtime scene candidates')), \
             patch.object(task, '_sample_robot_starts', side_effect=starts), \
             patch.object(task, '_robot_start_domain_successes', side_effect=AssertionError('legacy scene gate')), \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(1, dtype=torch.bool)):
            batch, _ = task._build_episode_batch(env, env._episode_reset, torch.tensor([0]), scene_generation=1)
        self.assertEqual(draw.call_count, 1)
        self.assertEqual(env._episode_reset.vision.observe_at_viewpoint.call_count, 1)
        env._episode_reset.vision.capture_batch.assert_not_called()
        env._episode_reset.vision.capture_view_domain.assert_not_called()
        torch.testing.assert_close(batch.visibility, torch.tensor([[.6, 1.]]))

    def test_constructive_builder_captures_supplied_world_viewpoint_once(self):
        env, propose, starts, snapshot = self.builder_fixture()
        env.spawn_rules.generation_mode = "constructive"
        env.cfg.initial_vision.max_layout_attempts = 32
        local_camera = (.2, .01, 1.3, 1., 0., 0., 0.)
        def generate(*args, **kwargs):
            return dataclasses.replace(propose(*args, **kwargs), camera_pose=local_camera)
        def capture(requests, *, object_poses, root_poses, preferred_camera_poses):
            result = [snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
            for index, value in enumerate(result):
                value.camera_pose_w = preferred_camera_poses[index]
            return result
        env._episode_reset.vision.capture_view_domain.side_effect = capture
        ids = torch.tensor([4, 1])
        with patch.object(task, 'sample_scene', side_effect=generate), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture) as layouts, \
             patch.object(task, '_sample_robot_starts', side_effect=starts), \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(2, dtype=torch.bool)):
            episodes = task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(layouts.call_count, 1)
        self.assertEqual(env._episode_reset.vision.capture_view_domain.call_count, 1)
        env._episode_reset.vision.capture_batch.assert_not_called()
        expected = torch.tensor(local_camera).repeat(2, 1)
        expected[:, :3] += env.scene.env_origins[ids]
        torch.testing.assert_close(episodes.camera_pose, expected)
        torch.testing.assert_close(env._episode_reset.vision.capture_view_domain.call_args.kwargs['preferred_camera_poses'], expected)
        self.assertEqual(episodes.geometry_b.shape, (2, env.cfg.mdp.max_objects, 10))

    def test_constructive_shared_builder_accepts_verified_anchor_without_random_probe_gate(self):
        env, propose, starts, snapshot = self.builder_fixture()
        env.spawn_rules.generation_mode = 'constructive'
        def generate(*args, **kwargs):
            return dataclasses.replace(propose(*args, **kwargs), camera_pose=(.2, 0., 1.3, 1., 0., 0., 0.))
        def capture(requests, *, object_poses, root_poses, preferred_camera_poses):
            return [snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
        env._episode_reset.vision.capture_view_domain.side_effect = capture
        with patch.object(task, 'sample_scene', side_effect=generate), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture) as layouts, \
             patch.object(task, '_sample_robot_starts', side_effect=starts) as ik, \
             patch.object(task, '_robot_start_domain_successes', side_effect=AssertionError('Random probe gate')) as domain, \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(1, dtype=torch.bool)) as physical:
            template, sample = task._build_episode_batch(env, env._episode_reset, torch.tensor([0]), scene_generation=1)
        domain.assert_not_called()
        self.assertEqual((layouts.call_count, ik.call_count, physical.call_count), (1, 1, 1))
        self.assertEqual(env._episode_reset.vision.capture_view_domain.call_count, 1)
        self.assertEqual(env.scene['object_collection'].writes, 0)
        self.assertEqual(template.object_state.shape[0], 1)
        self.assertEqual(physical.call_args.args[1], [sample])
        torch.testing.assert_close(template.object_state[0, :, :3],
                                   torch.tensor([sample.positions[index] for index in sample.active_ids]))
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)

    def test_successful_preparation_restores_robot_without_writing_object_or_activity(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        ids = torch.tensor([4, 1])
        objects, robot = env.scene['object_collection'], env.scene['robot']
        physical = (objects.state, objects.parked_mask, robot.root, robot.position, robot.velocity,
                    robot.position_target, robot.velocity_target, robot.effort_target)
        before = [value.clone() for value in physical]
        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(2, dtype=torch.bool)):
            episodes = task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(episodes.object_state.shape[0], 2)
        for actual, expected in zip(physical, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(objects.operations, [])
        self.assertEqual(objects.activity_calls, [])
        self.assertEqual(objects.writes, 0)
        self.assertEqual(robot.root_physx_view.get_link_transforms.call_count, 1)

    def test_constructive_shared_builder_still_rejects_physical_invalid_anchor_and_rolls_back(self):
        env, propose, starts, snapshot = self.builder_fixture()
        env.spawn_rules.generation_mode = 'constructive'
        objects, robot = env.scene['object_collection'], env.scene['robot']
        physical_state = (objects.state, objects.parked_mask, robot.root, robot.position, robot.velocity,
                          robot.position_target, robot.velocity_target, robot.effort_target)
        original = [value.clone() for value in physical_state]
        def generate(*args, **kwargs):
            return dataclasses.replace(propose(*args, **kwargs), camera_pose=(.2, 0., 1.3, 1., 0., 0., 0.))
        def capture(requests, *, object_poses, root_poses, preferred_camera_poses):
            return [snapshot(request['env_id'], pose) for request, pose in zip(requests, object_poses)]
        env._episode_reset.vision.capture_view_domain.side_effect = capture
        with patch.object(task, 'sample_scene', side_effect=generate), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture) as layouts, \
             patch.object(task, '_sample_robot_starts', side_effect=starts) as ik, \
             patch.object(task, '_robot_start_domain_successes', side_effect=AssertionError('Random probe gate')) as domain, \
             patch.object(task, '_check_physical_robot_starts', return_value=torch.zeros(1, dtype=torch.bool)) as physical:
            with self.assertRaisesRegex(RuntimeError, 'Constructive preparation rejected invariants.*Physical articulation'):
                task._build_episode_batch(env, env._episode_reset, torch.tensor([0]), scene_generation=1)
        domain.assert_not_called()
        self.assertEqual((layouts.call_count, ik.call_count, physical.call_count), (1, 1, 1))
        self.assertEqual(env._episode_reset.vision.capture_view_domain.call_count, 1)
        for actual, expected in zip(physical_state, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)
        self.assertTrue((env._episode_reset.episode_seeds == -1).all())
        self.assertTrue((env.task_scene_data.geometry_b == -32.).all())

    def test_constructive_visibility_rejection_reports_exact_cause_without_resampling_and_rolls_back(self):
        env, propose, _, _ = self.builder_fixture()
        env.spawn_rules.generation_mode = "constructive"
        env.cfg.initial_vision.max_layout_attempts = 32
        before = env.scene['object_collection'].state.clone()
        def generate(*args, **kwargs):
            return dataclasses.replace(propose(*args, **kwargs), camera_pose=(.2, 0., 1.3, 1., 0., 0., 0.))
        env._episode_reset.vision.capture_view_domain.return_value = [None]
        env._episode_reset.vision.last_diagnostics = [dict(target_visible_fraction=.97,
            blocker_visible_fraction=1., failure_counts={'target_above_max': 1})]
        with patch.object(task, 'sample_scene', side_effect=generate), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture) as layouts, \
             patch.object(task, '_sample_robot_starts') as ik:
            with self.assertRaisesRegex(RuntimeError, 'Constructive preparation rejected invariants.*target_above_max'):
                task._build_episode_batch(env, env._episode_reset, torch.tensor([4]))
        self.assertEqual(layouts.call_count, 1)
        self.assertEqual(env._episode_reset.vision.capture_view_domain.call_count, 1)
        env._episode_reset.vision.capture_batch.assert_not_called()
        ik.assert_not_called()
        torch.testing.assert_close(env.scene['object_collection'].state, before)
        self.assertEqual(int(env._episode_reset.reset_counts.sum()), 0)

    def test_constructive_joint_variants_query_once_and_commit_the_same_vision_physics_and_ik_member(self):
        for primary_valid, base_valid, expected_index in ((True, True, 0), (False, True, 1), (False, False, None)):
            with self.subTest(primary=primary_valid, base=base_valid):
                env, propose, starts, snapshot = self.builder_fixture()
                env.spawn_rules.generation_mode = 'constructive'
                ids = torch.tensor([4])
                members = []
                def generate(*args, **kwargs):
                    base = dataclasses.replace(propose(*args, **kwargs),
                        camera_pose=(.2, 0., 1.3, 1., 0., 0., 0.), clearance_kind='base')
                    guard = next(index for index in base.active_ids if index not in (base.target_id, base.blocker_id))
                    positions = dict(base.positions)
                    x, y, z = positions[guard]
                    positions[guard] = (x, y + .025, z)
                    primary = dataclasses.replace(base, positions=positions, clearance_kind='guarded',
                                                  visibility_variants=(base,))
                    members.extend((primary, base, guard))
                    return primary
                def capture(requests, *, object_poses, root_poses, preferred_camera_poses):
                    self.assertEqual(len(requests), 2)  # Base is evaluated even if guarded passes.
                    results = []
                    env._episode_reset.vision.last_diagnostics = []
                    for row, passed in enumerate((primary_valid, base_valid)):
                        value = snapshot(requests[row]['env_id'], object_poses[row]) if passed else None
                        if value is not None:
                            value.camera_pose_w = preferred_camera_poses[row]
                        results.append(value)
                        env._episode_reset.vision.last_diagnostics.append(dict(view_count=37,
                            valid_candidate_count=2 if passed else 0,
                            target_visible_fraction=.65 if passed else 0., blocker_visible_fraction=1.,
                            failure_counts={} if passed else {'target_below_min': 37}))
                    return results
                env._episode_reset.vision.capture_view_domain.side_effect = capture
                before = env.scene['object_collection'].state.clone()
                with patch.object(task, 'sample_scene', side_effect=generate), \
                     patch.object(task, 'sample_scenes', side_effect=self.batch_fixture) as layouts, \
                     patch.object(task, '_sample_robot_starts', side_effect=starts) as ik, \
                     patch.object(task, '_check_physical_robot_starts', return_value=torch.ones(1, dtype=torch.bool)):
                    if expected_index is None:
                        with self.assertRaisesRegex(RuntimeError, "Constructive preparation rejected invariants.*variant_count.*2"):
                            task._build_episode_batch(env, env._episode_reset, ids)
                        ik.assert_not_called()
                        torch.testing.assert_close(env.scene['object_collection'].state, before)
                    else:
                        episodes = task._build_episode_batch(env, env._episode_reset, ids)
                        selected, guard = members[expected_index], members[2]
                        self.assertIs(ik.call_args.args[1][0], selected)
                        self.assertAlmostEqual(float(episodes.object_state[0, guard, 1]), selected.positions[guard][1], places=6)
                        torch.testing.assert_close(episodes.geometry_b[0, :4, :3], episodes.object_state[0, :, :3])
                        torch.testing.assert_close(ik.call_args.kwargs['proposals'][0][0, guard, :3],
                                                  torch.tensor(selected.positions[guard]))
                self.assertEqual(layouts.call_count, 1)
                self.assertEqual(env._episode_reset.vision.capture_view_domain.call_count, 1)
                detail = env._episode_reset.vision.last_diagnostics[0]
                self.assertEqual(len(env._episode_reset.vision.last_diagnostics), 1)
                self.assertEqual(detail['variant_count'], 2)
                self.assertEqual(detail['view_count'], 74)
                self.assertEqual(detail['selected_variant_index'], expected_index)
                self.assertEqual(len(detail['variant_diagnostics']), 2)

    def test_failed_builder_restores_selected_physics_and_preserves_scene_metadata(self):
        env, sample_scene, sample_robot_start, snapshot = self.builder_fixture()
        ids = torch.tensor([4, 1])
        objects, robot = env.scene['object_collection'], env.scene['robot']
        physical = (objects.state, objects.parked_mask, robot.root, robot.position, robot.velocity,
                    robot.position_target, robot.velocity_target, robot.effort_target)
        physical_before = [value.clone() for value in physical]
        metadata_before = {key: value.clone() for key, value in vars(env.task_scene_data).items()
                           if isinstance(value, torch.Tensor)}
        captures = []

        def capture(**kwargs):
            captures.append(kwargs['env_id'])
            if kwargs['env_id'] == 1:
                return None
            return snapshot(kwargs['env_id'], kwargs.get('object_pose'))

        env._episode_reset.vision.capture_batch.side_effect = lambda requests, object_poses, root_poses: [
            capture(**request, object_pose=pose) for request, pose in zip(requests, object_poses)]
        with patch.object(task, 'sample_scene', side_effect=sample_scene), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts', side_effect=sample_robot_start), \
             patch.object(task, '_check_physical_robot_starts', side_effect=lambda env,samples,*a: torch.ones(len(samples),dtype=torch.bool)):
            with self.assertRaisesRegex(RuntimeError, 'Independent episode preparation exhausted retries'):
                task._build_episode_batch(env, env._episode_reset, ids)
        self.assertEqual(captures, [4, 1, 1])
        for actual, expected in zip(physical, physical_before):
            torch.testing.assert_close(actual, expected)
        for name, expected in metadata_before.items():
            torch.testing.assert_close(getattr(env.task_scene_data, name), expected)
        torch.testing.assert_close(env._episode_reset.reset_counts, torch.zeros(5, dtype=torch.long))
        self.assertTrue((env._episode_reset.episode_seeds == -1).all())
        self.assertEqual(env.scene.writes, 0)
        env.task_sensor_reader.reset.assert_not_called()

    def test_failed_builder_does_not_touch_object_state_or_activity(self):
        env, sample_scene, _, _ = self.builder_fixture()
        objects = env.scene['object_collection']
        ids = torch.tensor([4, 1])
        objects.parked_mask[4, 2] = True
        objects.state[4, 2] = math.nan  # Preparation must not read or rewrite object state.
        expected = objects.state.clone()
        initial_mask = objects.parked_mask.clone()
        env._episode_reset.vision.capture_batch.side_effect = lambda requests, **kwargs: [None] * len(requests)
        with patch.object(task, 'sample_scene', side_effect=sample_scene), \
             patch.object(task, 'sample_scenes', side_effect=self.batch_fixture), \
             patch.object(task, '_sample_robot_starts') as ik:
            with self.assertRaisesRegex(RuntimeError, 'Independent episode preparation exhausted retries'):
                task._build_episode_batch(env, env._episode_reset, ids)
        ik.assert_not_called()
        torch.testing.assert_close(objects.state, expected, equal_nan=True)
        torch.testing.assert_close(objects.parked_mask, initial_mask)
        self.assertEqual(objects.operations, [])
        self.assertEqual(objects.activity_calls, [])
        self.assertEqual(objects.writes, 0)

    def test_shelf_reset_restores_only_selected_poses_and_zeroes_velocity(self):
        env, episodes = episode_reset_fixture()
        ids = torch.tensor([4,1])
        shelf = env.scene['shelf']
        before = shelf.root.clone()
        task.apply_episode_batch(env, ids, episodes)
        expected = shelf.data.default_root_state[ids].clone()
        expected[:, :3] += env.scene.env_origins[ids]
        expected[:, 7:] = 0
        torch.testing.assert_close(shelf.root[ids], expected)
        torch.testing.assert_close(shelf.root[[0,2,3]], before[[0,2,3]])
        self.assertEqual(shelf.root_writes, 1)

    def test_empty_reset_does_not_initialize_resources_or_write_assets(self):
        env, _ = episode_reset_fixture()
        del env._episode_reset
        with patch.object(task, '_initialize_episode_reset', side_effect=AssertionError('Unexpected initialize')):
            task.reset_scene(env, torch.empty(0, dtype=torch.long))
        self.assertEqual(env.scene['object_collection'].writes, 0)



class PartialResetBatchTests(unittest.TestCase):
    def test_reset_snapshot_matches_used_transition_fields_without_dynamics_or_sensors(self):
        env, _, _ = native_fixture(7)
        task.reset_task(env)
        state = env._contact_mdp
        data = env.scene['robot'].data
        data.joint_pos += torch.arange(7)[:, None] * .01
        data.joint_vel += .023
        data.body_link_pos_w[:, 1, 0] += .04
        ids = torch.tensor([6, 1])
        expected = task._read_transition(env, state, ids)
        class PoseOnly(SimpleNamespace):
            def __getattribute__(self, name):
                if name in ('body_com_pos_w', 'body_com_acc_w', 'body_com_ang_vel_w',
                            'object_com_pos_w', 'object_com_vel_w', 'applied_torque'):
                    raise AssertionError(f'Reset fetched unused dynamics: {name}')
                return super().__getattribute__(name)
        env.scene['robot'].data = PoseOnly(**vars(data))
        objects = env.scene['object_collection']
        objects.data = PoseOnly(**vars(objects.data))
        with patch.object(env, 'task_sensor_reader', side_effect=AssertionError('Reset read sensors')):
            actual = task._read_reset_snapshot(env, state, ids)
        for name in ('center', 'corners', 'rotation', 'shelf_pose', 'sensor_position', 'tcp_pose_b'):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        torch.testing.assert_close(actual.sample[:, :31], expected.sample[:, :31], rtol=0, atol=0)

    def test_cached_object_com_follows_rotation_without_hand_dynamics_reads(self):
        env, _, _ = native_fixture(3)
        robot = env.scene['robot'].data
        objects = env.scene['object_collection'].data
        objects.object_com_pos_b[:, 2] = torch.tensor([.03, -.04, .02])
        task.reset_task(env)
        state = env._contact_mdp
        robot.body_link_quat_w[:, 3] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
        objects.object_link_pose_w[:, 2, 3:7] = robot.body_link_quat_w[:, 3]
        objects.object_com_vel_w[:, 2, 5] = 2.
        # Native world-COM buffers are deliberately stale/nonfinite. The fixed
        # local offsets must instead rotate with current link poses on device.
        robot.body_com_pos_w.fill_(math.nan)
        objects.object_com_pos_w.fill_(math.nan)
        actual = task._read_transition(env, state, torch.arange(3))
        slot = 0  # Collection object 2 occupies the first active slot.
        pose = objects.object_link_pose_w[:, 2, :3] - env.scene.env_origins
        expected_com = pose + torch.tensor([.04, .03, .02])
        expected_velocity = torch.cross(torch.tensor([0., 0., 2.]).expand(3, -1),
                                        actual.center[:, slot] - expected_com, dim=-1)
        torch.testing.assert_close(actual.velocity[:, slot, :3], expected_velocity)
        # Native twist is at COM, whereas the cost consumes center twist.
        # Both paths must produce the same corner velocity for a rotating,
        # off-center asset; omitting or doubling the shift changes the cost.
        expected_corners = torch.cross(torch.tensor([0., 0., 2.]).expand(3, 8, -1),
                                       actual.corners[:, slot] - expected_com[:, None], dim=-1)
        expected_cost = torch.linalg.vector_norm(expected_corners, dim=-1).mean(-1)
        torch.testing.assert_close(rewards.object_motion_cost(actual.center, actual.corners, actual.velocity,
                                                              state.active, state.blocker), expected_cost)
        self.assertTrue(actual.state_valid.all())

    def test_binding_and_transition_do_not_query_hand_mass_inertia_com_or_acceleration(self):
        env, _, _ = native_fixture(3)
        class NoHandDynamics(SimpleNamespace):
            def __getattribute__(self, name):
                if name in ('body_com_pos_b', 'body_com_pos_w', 'body_com_acc_w',
                            'body_com_ang_vel_w'):
                    raise AssertionError(f'Uncompensated F/T fetched unused hand dynamics: {name}')
                return super().__getattribute__(name)
        robot = env.scene['robot']
        robot.data = NoHandDynamics(**vars(robot.data))
        reject = AssertionError('Uncompensated F/T fetched mass or inertia')
        with patch.object(robot.root_physx_view, 'get_masses', side_effect=reject), \
             patch.object(robot.root_physx_view, 'get_inertias', side_effect=reject):
            task.reset_task(env)
            env.common_step_counter = 1
            state = task.get_task_state(env)
            measured, _ = obs.calibrate_wrench(env.sensors.ideal_wrench)
            torch.testing.assert_close(state.previous_wrench, measured)

    @unittest.skipUnless(torch.cuda.is_available(), 'Transition CUDA graph requires CUDA.')
    def test_transition_graph_matches_eager_after_changed_measurements_and_subset_reset(self):
        env, action, _ = native_fixture(8)
        env.device = 'cuda'
        for holder in (env.scene, env.scene['robot'].data, env.scene['object_collection'].data, env.scene['shelf'].data,
                       env.task_scene_data, env.sensors, action):
            for name, value in vars(holder).items():
                if isinstance(value, torch.Tensor):
                    setattr(holder, name, value.cuda())
        task.reset_task(env)
        state = env._contact_mdp
        ids = torch.arange(8, device='cuda')
        with torch.inference_mode():
            previous = None
            for scale in (1., 1.3):
                env.scene['robot'].data.joint_vel.fill_(scale)
                env.sensors.ideal_wrench.add_(scale)
                expected = task._read_transition(env, state, ids)
                actual = task._read_transition(env, state)
                for name, value in vars(actual).items():
                    torch.testing.assert_close(value, getattr(expected, name), atol=0, rtol=0, equal_nan=True)
                if previous is not None:
                    for value, snapshot in previous:
                        torch.testing.assert_close(value, snapshot, atol=0, rtol=0, equal_nan=True)
                previous = [(v, v.clone()) for v in vars(actual).values()]
                task.reset_task(env, torch.tensor([1, 4], device='cuda'))

    def test_transition_subset_matches_full_batch_and_limits_geometry_work(self):
        env, action, _ = native_fixture(8)
        task.reset_task(env)
        state = env._contact_mdp
        robot = env.scene['robot'].data
        robot.joint_pos[:, 0] = torch.arange(8) * .01
        robot.joint_vel[:, 1] = torch.arange(8) * .02
        env.sensors.tactile_force[:, 2, 0] = torch.arange(8)
        env.scene["shelf"].data.root_link_pose_w[6, 0] += .02
        action.normalized_actions[:, 3] = torch.arange(8) * .01
        action.applied_torque[1, 0] = math.nan
        full = task._read_transition(env, state)
        ids = torch.tensor([6, 1])
        def reject(*args, **kwargs):
            raise AssertionError('Subset transition must not read device scalar values')
        with patch.object(task, 'object_geometry', wraps=task.object_geometry) as geometry, \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'cpu', reject):
            subset = task._read_transition(env, state, ids)
        self.assertEqual(geometry.call_args.args[0].shape[0], 2)
        for field in dataclasses.fields(full):
            with self.subTest(field=field.name):
                self.assertEqual(getattr(subset, field.name).shape[0], 2)
                torch.testing.assert_close(getattr(subset, field.name), getattr(full, field.name)[ids])

    def test_selected_state_reset_matches_full_input_api_for_unsorted_ids(self):
        state_full, batch, values = fixture(6)
        state_selected, _, _ = fixture(6)
        batch.sample[:, :12] += torch.arange(6, dtype=batch.sample.dtype)[:, None]
        batch.tcp_pose_b[:, 0] = torch.arange(6, dtype=batch.sample.dtype) * .01
        ids = torch.tensor([4, 1])
        subset = task.TransitionBatch(**{field.name: getattr(batch, field.name)[ids]
                                        for field in dataclasses.fields(batch)})
        selected_values = {name: value[ids] for name, value in values.items()}
        state_full.reset(ids, batch, **values, step=0)
        state_selected.reset(ids, subset, **selected_values, step=0, batch_is_selected=True)
        for name, value in vars(state_full).items():
            if isinstance(value, torch.Tensor):
                with self.subTest(field=name):
                    torch.testing.assert_close(getattr(state_selected, name), value)

    def test_native_partial_reset_limits_all_preparation_to_selected_rows_and_preserves_neighbors(self):
        env, _, _ = native_fixture(8)
        task.reset_task(env)
        env.common_step_counter = 1
        state = task.get_task_state(env)
        ids = torch.tensor([6, 1])
        other_ids = torch.tensor([0, 2, 3, 4, 5, 7])
        preserved = {name: value[other_ids].clone() for name, value in vars(state).items()
                     if isinstance(value, torch.Tensor) and value.shape[:1] == (8,)}
        final_observation = env.extras['terminal_observation'].clone()
        env.task_scene_data.geometry_b[ids, :3, 0] += .01
        with patch.object(task, '_read_reset_snapshot', wraps=task._read_reset_snapshot) as snapshot, \
             patch.object(task, 'prepare_initial_geometry', wraps=task.prepare_initial_geometry) as prepare, \
             patch.object(task, 'geometry_in_eef_frame', wraps=task.geometry_in_eef_frame) as transform, \
             patch.object(state, 'reset', wraps=state.reset) as commit:
            task.reset_task(env, ids)
        self.assertEqual(snapshot.call_args.args[2].numel(), 2)
        self.assertEqual(prepare.call_args.args[0].shape[0], 2)
        self.assertEqual(transform.call_args.args[0].shape[0], 2)
        self.assertEqual(commit.call_args.args[1].sample.shape, (2, 72))
        self.assertEqual(commit.call_args.kwargs['geometry_b'].shape, (2, 10, 10))
        self.assertTrue(commit.call_args.kwargs['batch_is_selected'])
        for name, expected in preserved.items():
            with self.subTest(field=name):
                torch.testing.assert_close(getattr(state, name)[other_ids], expected)
        torch.testing.assert_close(env.extras['terminal_observation'], final_observation)
        self.assertIsNot(env.extras['terminal_observation'], state.observation)


class BlockerContactManagerTests(unittest.TestCase):
    """Run the native term against one shared Blocker column without Kit."""

    @classmethod
    def setUpClass(cls):
        class ManagerTermBase:
            def __init__(self, cfg, env):
                self.cfg, self._env = cfg, env

        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'BlockerContactHold')
        namespace = dict(torch=torch, ManagerTermBase=ManagerTermBase,
                         __name__='contact_manipulation.mdp.sensors', __package__='contact_manipulation.mdp')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        cls.term_type = namespace['BlockerContactHold']
        cls.names = tuple(f'pad_{i}' for i in range(17))

    def fixture(self, weight=.01, *, configure=True):
        canonical_names = self.names

        class Sensor:
            def __init__(self):
                self.matrix = torch.zeros((3, 17, 4, 3))
                self.reads, self.enables, self._blocker_id = 0, 0, None

            def find_bodies(self, names, preserve_order=False):
                assert preserve_order and tuple(names) == canonical_names
                return list(reversed(range(17))), list(names)

            def enable_blocker_filtering(self, objects, paths, blocker_id):
                assert isinstance(blocker_id, int)
                if self._blocker_id == blocker_id:
                    return False
                self._blocker_id = blocker_id
                self.enables += 1
                return True

            @property
            def blocker_force_w(self):
                self.reads += 1
                assert self._blocker_id is not None
                return self.matrix[:, :, self._blocker_id]

            def contact(self, row, canonical_pad, object_id, force=.02):
                self.matrix[row, 16 - canonical_pad, object_id, 0] = force

        class Scene(dict):
            env_prim_paths = ['/World/envs/env_0', '/World/envs/env_1', '/World/envs/env_2']

        sensor = Sensor()
        sensor_values = SimpleNamespace(tactile_valid=torch.ones((3, 17), dtype=torch.bool))
        env = SimpleNamespace(num_envs=3, device='cpu', step_dt=.02, common_step_counter=0,
                              task_scene_data=SimpleNamespace(blocker_id=torch.full((3,), 3)),
                              task_sensor_reader=lambda: sensor_values,
                              scene=Scene(tactile_contact=sensor, object_collection=object()))
        fake_scene = types.ModuleType('contact_manipulation.config.ur5e.scene')
        fake_scene.TACTILE_BODY_NAMES = canonical_names
        with patch.dict(sys.modules, {fake_scene.__name__: fake_scene}):
            term = self.term_type(SimpleNamespace(weight=weight), env)
        if configure:
            term.configure_blocker(3)
        return env, sensor, sensor_values, term

    @staticmethod
    def seed_contacts(sensor):
        sensor.contact(0, 0, 3)
        sensor.contact(0, 4, 3)
        sensor.contact(0, 9, 1)  # A non-Blocker contact cannot earn hold reward.
        sensor.contact(1, 16, 3)
        sensor.contact(2, 8, 3)

    def test_lazy_registration_and_single_column_native_pad_order_and_step_cache(self):
        env, sensor, _, term = self.fixture(configure=False)
        self.assertIs(env._blocker_contact_hold, term)
        self.assertEqual(sensor.enables, 0)
        term.configure_blocker(3)
        self.seed_contacts(sensor)
        expected = torch.zeros((3, 17), dtype=torch.bool)
        expected[0, [0, 4]], expected[1, 16], expected[2, 8] = True, True, True
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('hot-path scalar read')):
            torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertEqual(sensor.reads, 0)
            env.common_step_counter = 1
            torch.testing.assert_close(term(env), torch.zeros(3))
            torch.testing.assert_close(term.previous_contact, expected)
            env.common_step_counter = 2
            torch.testing.assert_close(term(env), torch.tensor([2., 1., 1.]))
            sensor.matrix.zero_()
            torch.testing.assert_close(term(env), torch.tensor([2., 1., 1.]))
            torch.testing.assert_close(term.previous_contact, expected)
            self.assertEqual(sensor.reads, 2)
            env.common_step_counter = 3
            torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertFalse(term.previous_contact.any())
        self.assertEqual(sensor.enables, 1)

    def test_partial_reset_keeps_neighbor_history_and_never_rebinds_shared_blocker(self):
        env, sensor, _, term = self.fixture()
        self.seed_contacts(sensor)
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))):
            for step in (1, 2):
                env.common_step_counter = step
                term(env)
            neighbors = term.previous_contact[[0, 2]].clone()
            term.reset(torch.tensor([1]))
            term.configure_blocker(3)  # Reset events may idempotently refresh the binding.
            sensor.matrix[1].zero_()
            sensor.contact(1, 7, 3)
            torch.testing.assert_close(term.previous_contact[[0, 2]], neighbors)
            self.assertFalse(term.previous_contact[1].any())
            self.assertEqual(term.last_step, 2)
            torch.testing.assert_close(term(env), torch.tensor([2., 0., 1.]))
            self.assertEqual(sensor.reads, 2)
            env.common_step_counter = 3
            torch.testing.assert_close(term(env), torch.tensor([2., 0., 1.]))
            self.assertTrue(term.previous_contact[1, 7])
            self.assertFalse(term.previous_contact[1, 16])
            env.common_step_counter = 4
            torch.testing.assert_close(term(env), torch.tensor([2., 1., 1.]))
            self.assertEqual(sensor.enables, 1)

    def test_target_gate_masks_payment_but_continues_contact_history_and_recovers(self):
        env, sensor, _, term = self.fixture()
        self.seed_contacts(sensor)
        protected = torch.ones(3, dtype=torch.bool)
        state = SimpleNamespace(target_protected=protected)
        with patch.object(task, 'get_task_state', return_value=state), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('host scalar')):
            env.common_step_counter = 1
            torch.testing.assert_close(term(env), torch.zeros(3))
            protected[:2] = False
            env.common_step_counter = 2
            torch.testing.assert_close(term(env), torch.tensor([0., 0., 1.]))
            # Losing contact while the gate is closed must still clear history.
            sensor.matrix[1].zero_()
            env.common_step_counter = 3
            torch.testing.assert_close(term(env), torch.tensor([0., 0., 1.]))
            self.assertFalse(term.previous_contact[1].any())
            self.assertTrue(term.previous_contact[0, [0, 4]].all())
            protected.fill_(True)
            sensor.contact(1, 7, 3)
            env.common_step_counter = 4
            torch.testing.assert_close(term(env), torch.tensor([2., 0., 1.]))
            env.common_step_counter = 5
            torch.testing.assert_close(term(env), torch.tensor([2., 1., 1.]))
            self.assertEqual(sensor.reads, 5)
            # Re-evaluation in the same step changes only the return gate, not
            # sensor reads or previous/current contact bookkeeping.
            protected[0] = False
            torch.testing.assert_close(term(env), torch.tensor([0., 1., 1.]))
            self.assertEqual(sensor.reads, 5)

    def test_shared_scene_switch_clears_hold_and_uses_new_column_without_stale_role_ids(self):
        env, sensor, _, term = self.fixture()
        self.seed_contacts(sensor)
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))):
            for step in (1, 2):
                env.common_step_counter = step
                term(env)
            self.assertTrue(term.previous_contact.any())
            env.task_scene_data.blocker_id.fill_(2)
            sensor.contact(0, 0, 2)  # Same physical pad, different object.
            sensor.contact(1, 7, 2)
            term.configure_blocker(2)
            self.assertEqual(term.blocker_id, 2)
            self.assertEqual(sensor.enables, 2)
            self.assertFalse(term.previous_contact.any())
            torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertEqual(sensor.reads, 2)
            env.common_step_counter = 3
            torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertTrue(term.previous_contact[0, 0])
            self.assertFalse(term.previous_contact[0, 4])  # Still touching old Blocker 3.
            self.assertFalse(term.previous_contact[2, 8])
            env.common_step_counter = 4
            torch.testing.assert_close(term(env), torch.tensor([1., 1., 0.]))
            before = term.previous_contact.clone()
            term.configure_blocker(2)  # Same path must neither rebind nor clear continuity.
            self.assertEqual(sensor.enables, 2)
            torch.testing.assert_close(term.previous_contact, before)

    def test_canonical_sensor_validity_and_nonfinite_pair_forces_break_continuity(self):
        env, sensor, values, term = self.fixture()
        sensor.contact(0, 2, 3)
        sensor.contact(1, 14, 3, force=math.inf)
        sensor.contact(2, 8, 3, force=math.nan)
        values.tactile_valid[0, 2] = False
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))):
            env.common_step_counter = 1
            torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertFalse(term.previous_contact.any())
            values.tactile_valid[0, 2] = True
            sensor.contact(1, 14, 3)
            sensor.contact(2, 8, 3)
            env.common_step_counter = 2
            torch.testing.assert_close(term(env), torch.zeros(3))
            env.common_step_counter = 3
            torch.testing.assert_close(term(env), torch.ones(3))
            values.tactile_valid[0, 2] = False
            env.common_step_counter = 4
            torch.testing.assert_close(term(env), torch.tensor([0., 1., 1.]))
            self.assertFalse(term.previous_contact[0, 2])

    def test_stop_play_without_reset_rebinds_cached_host_role_and_restarts_contact_history(self):
        env, sensor, _, term = self.fixture()
        self.seed_contacts(sensor)
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))):
            for step in (1, 2):
                env.common_step_counter = step
                term(env)
            self.assertTrue(term.previous_contact.any())
            self.assertEqual(term.blocker_id, 3)
            # Native views were recreated by Stop/Play, with no reset event.
            sensor._blocker_id = None
            env.common_step_counter = 3
            with patch.object(torch.Tensor, 'item', side_effect=AssertionError('role copied from GPU')):
                torch.testing.assert_close(term(env), torch.zeros(3))
            self.assertEqual(sensor._blocker_id, 3)
            self.assertEqual(sensor.enables, 2)
            self.assertEqual(sensor.reads, 2)  # No old force sample is reused.
            self.assertFalse(term.previous_contact.any())
            env.common_step_counter = 4
            torch.testing.assert_close(term(env), torch.zeros(3))  # New first contact.
            env.common_step_counter = 5
            torch.testing.assert_close(term(env), torch.tensor([2., 1., 1.]))
            self.assertEqual(sensor.enables, 2)

    def test_missing_initial_host_role_fails_before_reading_or_registering_filter(self):
        env, sensor, _, term = self.fixture(configure=False)
        env.common_step_counter = 1
        with patch.object(task, 'get_task_state', return_value=SimpleNamespace(target_protected=torch.ones(3, dtype=torch.bool))), self.assertRaises(RuntimeError):
            term(env)
        self.assertEqual((sensor.enables, sensor.reads), (0, 0))

    def test_zero_weight_does_not_bind_and_buffers_survive_inference_mode_changes(self):
        with torch.inference_mode():
            _, sensor, _, term = self.fixture(weight=0.)
        self.assertEqual(sensor.enables, 0)
        self.assertFalse(torch.is_inference(term.previous_contact))
        self.assertFalse(torch.is_inference(term.held_count))
        term.reset(torch.tensor([1]))
        term.configure_blocker(2)
        self.assertEqual(sensor.enables, 0)


class BlockerContactFilterTests(unittest.TestCase):
    """Exercise the actual filter adapter with a native-shaped force backend."""

    @classmethod
    def setUpClass(cls):
        class Native:
            @property
            def contact_physx_view(self):
                return self._contact_physx_view

            def _initialize_impl(self):
                self._contact_physx_view = self.net_view

            def _update_buffers_impl(self, ids):
                rows = slice(None) if len(ids) == self._num_envs else ids
                force = self.contact_physx_view.get_net_contact_forces(self._sim_physics_dt)
                self._data.net_forces_w[rows] = force.reshape(self._num_envs, self._num_bodies, 3)[rows]

            def reset(self, ids=None):
                self._data.net_forces_w[slice(None) if ids is None else ids] = 0

        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TaskContactSensor')
        namespace = dict(ContactSensor=Native, torch=torch)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        cls.sensor_type = namespace['TaskContactSensor']

    def fixture(self):
        import weakref
        roots = [f'/World/envs/env_{i}' for i in range(3)]
        paths = [f'{root}/pad_{pad}' for root in roots for pad in range(17)]
        backend = SimpleNamespace(forces=torch.zeros(3, 17, 4, 3), shelf=torch.zeros(3, 17, 3),
                                  create_calls=[], views=[], net_reads=0, matrix_reads=0)

        class View:
            def __init__(self, filters=None):
                self.sensor_paths, self.sensor_count = list(paths), len(paths)
                self.filter_count = 0 if filters is None else 1
                self.blocker = None if filters is None else int(filters[0][0].rsplit('_', 1)[1])

            def get_net_contact_forces(self, dt):
                backend.net_reads += 1
                return (backend.forces.sum(2) + backend.shelf).reshape(-1, 3)

            def get_contact_force_matrix(self, dt):
                backend.matrix_reads += 1
                return backend.forces[:, :, self.blocker].reshape(-1, 1, 3)

        def create(sensor_paths, *, filter_patterns, max_contact_data_count):
            self.assertEqual(sensor_paths, paths)
            self.assertEqual(len(filter_patterns), 3 * 17)
            self.assertTrue(all(len(filters) == 1 for filters in filter_patterns))
            self.assertEqual(max_contact_data_count, 0)
            view = View(filter_patterns)
            backend.create_calls.append(dict(sensor_paths=list(sensor_paths), filters=filter_patterns))
            backend.views.append(weakref.ref(view))
            return view

        sensor = self.sensor_type()
        sensor.cfg = SimpleNamespace(update_period=0., track_air_time=False, track_pose=False,
                                     history_length=0, filter_prim_paths_expr=[])
        sensor._num_envs, sensor._num_bodies, sensor._device = 3, 17, 'cpu'
        sensor._is_visualizing, sensor._sim_physics_dt = False, .01
        sensor._timestamp, sensor._timestamp_last_update = torch.zeros(3), torch.zeros(3)
        sensor._is_outdated = torch.ones(3, dtype=torch.bool)
        sensor._data = SimpleNamespace(net_forces_w=torch.zeros(3, 17, 3))
        sensor.net_view = View()
        sensor._physics_sim_view = SimpleNamespace(create_rigid_contact_view=create)
        sensor._initialize_impl()
        objects = SimpleNamespace(num_instances=3, num_objects=4,
            root_physx_view=SimpleNamespace(prim_paths=[f'{root}/object_{obj}' for obj in range(4) for root in roots]))
        return sensor, backend, objects, roots

    def test_single_column_paths_rebind_only_on_id_change_and_old_views_are_not_retained(self):
        sensor, backend, objects, roots = self.fixture()
        base = sensor.contact_physx_view
        self.assertTrue(sensor.enable_blocker_filtering(objects, roots, 3))
        self.assertEqual(sensor._blocker_force_w.shape, (3, 17, 3))
        for row, filters in enumerate(backend.create_calls[0]['filters']):
            self.assertEqual(filters, [f'{roots[row // 17]}/object_3'])
        self.assertFalse(sensor.enable_blocker_filtering(objects, roots, 3))
        self.assertEqual(len(backend.create_calls), 1)
        self.assertTrue(sensor.enable_blocker_filtering(objects, roots, 1))
        self.assertEqual(len(backend.create_calls), 2)
        self.assertIsNone(backend.views[0]())  # No Python cache of old Blocker views.
        self.assertIs(sensor._blocker_filter_base_view, base)
        for row, filters in enumerate(backend.create_calls[1]['filters']):
            self.assertEqual(filters, [f'{roots[row // 17]}/object_1'])
        sensor.disable_blocker_filtering()
        self.assertIs(sensor.contact_physx_view, base)
        with self.assertRaisesRegex(RuntimeError, 'not been enabled'):
            _ = sensor.blocker_force_w

    def test_binary_net_forces_keep_all_seventeen_channels_including_other_objects_and_shelf(self):
        sensor, backend, objects, roots = self.fixture()
        backend.forces[0, 0, 1, 0] = .02  # Non-Blocker contact.
        backend.forces[0, 4, 3, 0] = .04
        backend.shelf[1, 16, 1] = .03
        expected = backend.forces.sum(2) + backend.shelf
        for blocker in (3, 1):
            sensor.enable_blocker_filtering(objects, roots, blocker)
            sensor.update(.01)
            force = sensor.blocker_force_w
            torch.testing.assert_close(force, backend.forces[:, :, blocker])
            torch.testing.assert_close(sensor._data.net_forces_w, expected)
            torch.testing.assert_close(torch.linalg.vector_norm(sensor._data.net_forces_w, dim=-1) > .01,
                                       torch.linalg.vector_norm(expected, dim=-1) > .01)
        self.assertEqual(backend.net_reads, 2)
        self.assertEqual(backend.matrix_reads, 2)

    def test_partial_reset_preserves_other_cached_forces_without_requery_or_rebind(self):
        sensor, backend, objects, roots = self.fixture()
        sensor.enable_blocker_filtering(objects, roots, 3)
        backend.forces[:, :, 3, 0] = torch.arange(3)[:, None] + 1.
        force = sensor.blocker_force_w.clone()
        net = sensor._data.net_forces_w.clone()
        sensor.reset(torch.tensor([1]))
        torch.testing.assert_close(sensor.blocker_force_w[[0, 2]], force[[0, 2]])
        torch.testing.assert_close(sensor._data.net_forces_w[[0, 2]], net[[0, 2]])
        self.assertEqual(int(sensor.blocker_force_w[1].count_nonzero()), 0)
        self.assertEqual((len(backend.create_calls), backend.matrix_reads), (1, 1))

    def test_inference_binding_and_stop_play_reinitialization_keep_safe_cache_lifetimes(self):
        sensor, backend, objects, roots = self.fixture()
        with torch.inference_mode():
            sensor.enable_blocker_filtering(objects, roots, 2)
        self.assertFalse(torch.is_inference(sensor._blocker_force_w))
        sensor.reset(torch.tensor([1]))
        sensor._initialize_impl()
        for name in ('_blocker_filter_base_view', '_blocker_force_w', '_blocker_id', '_blocker_filter_paths'):
            self.assertIsNone(getattr(sensor, name))
        self.assertTrue(sensor.enable_blocker_filtering(objects, roots, 2))
        self.assertEqual(len(backend.create_calls), 2)

    def test_invalid_config_or_environment_path_mapping_fails_before_native_view_creation(self):
        for field, value in (('filter_prim_paths_expr', ['/World/other']), ('history_length', 2),
                             ('track_pose', True), ('track_air_time', True), ('update_period', .02)):
            sensor, backend, objects, roots = self.fixture()
            setattr(sensor.cfg, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'history-free'):
                sensor.enable_blocker_filtering(objects, roots, 2)
            self.assertFalse(backend.create_calls)
        for kind in ('object_order', 'sensor_order', 'instance_count', 'blocker_range'):
            sensor, backend, objects, roots = self.fixture()
            blocker = 2
            if kind == 'object_order':
                objects.root_physx_view.prim_paths[0:2] = reversed(objects.root_physx_view.prim_paths[0:2])
            elif kind == 'sensor_order':
                sensor.contact_physx_view.sensor_paths[0:2] = [f'{roots[1]}/pad_0', f'{roots[0]}/pad_1']
            elif kind == 'instance_count':
                objects.num_instances = 4
            else:
                blocker = objects.num_objects
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                sensor.enable_blocker_filtering(objects, roots, blocker)
            self.assertFalse(backend.create_calls)


class ObjectPoolConfigurationTests(unittest.TestCase):
    def test_native_global_floor_and_default_bounds_are_restored(self):
        root = Path(__file__).resolve().parents[1] / 'contact_manipulation'
        cfg = ast.parse((root / 'shelf_manipulation_env_cfg.py').read_text())
        scene = next(node for node in cfg.body if isinstance(node, ast.ClassDef) and node.name == 'ShelfManipulationSceneCfg')
        ground = next(node.value for node in scene.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'ground' for target in node.targets))
        args = {argument.arg: argument.value for argument in ground.keywords}
        self.assertEqual(ast.literal_eval(args['prim_path']), '/World/Ground')
        self.assertEqual(ast.literal_eval(args['collision_group']), -1)
        self.assertEqual(args['spawn'].func.attr, 'GroundPlaneCfg')
        source = (root / 'config/ur5e/scene.py').read_text()
        self.assertNotIn('envIdInBoundsBitCount', source)
        self.assertNotIn('spawn_environment_floor', source)


class ContactBatchUpdateTests(unittest.TestCase):
    def test_whole_batch_sampling_has_no_nonzero_and_partial_reset_keeps_other_histories(self):
        class Native:
            def _update_buffers_impl(self, ids):
                self.reads += 1
                self.history = self.history.roll(1, 1)
                self.history[:, 0] = self.force

            def reset(self, ids=None):
                self.history[slice(None) if ids is None else ids] = 0

        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TaskContactSensor')
        namespace = dict(ContactSensor=Native)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        for history_length in (0, 4):
            sensor = namespace['TaskContactSensor']()
            sensor.cfg = SimpleNamespace(update_period=0., track_air_time=False, track_pose=False,
                                         history_length=history_length)
            sensor._num_envs, sensor._is_visualizing, sensor.reads = 3, False, 0
            sensor._timestamp = torch.zeros(3)
            sensor._timestamp_last_update = torch.zeros(3)
            sensor._is_outdated = torch.ones(3, dtype=torch.bool)
            sensor.history, sensor.force = torch.zeros(3, 4), torch.tensor([1., 2., 3.])
            with patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('host synchronization')):
                for tick in range(4):
                    sensor.force += 1
                    sensor.update(.01)
                    sensor._update_outdated_buffers()
                    sensor._update_outdated_buffers()
                self.assertEqual(sensor.reads, 4)
                before = sensor.history.clone()
                sensor.reset(torch.tensor([1]))
                sensor._update_outdated_buffers()
                torch.testing.assert_close(sensor.history[[0, 2]], before[[0, 2]])
                self.assertEqual(sensor.history[1].count_nonzero(), 0)
                self.assertEqual(sensor.reads, 4)
                sensor.update(.01)
                sensor._update_outdated_buffers()
                self.assertEqual(sensor.reads, 5)
                torch.testing.assert_close(sensor.history[:, 0], sensor.force)



class FusedGeometryTests(unittest.TestCase):
    def test_projected_area_matches_tensor_oracle_for_tilt_degeneracy_and_padding(self):
        torch.manual_seed(613)
        q = torch.randn(12, 9, 4)
        q /= q.norm(dim=-1, keepdim=True)
        cube = torch.cartesian_prod(*[torch.tensor([-.5, .5])]*3)
        corners = obs.quat_apply(q[:, :, None].expand(-1, -1, 8, -1),
                                cube * torch.rand(12, 9, 1, 3) * .3) + torch.rand(12, 9, 1, 3)
        corners[0, :2] = 0  # Degenerate projected hulls must stay finite.
        active = torch.rand(12, 9) > .25
        regions = torch.tensor([.3, .7, .3, .7]).expand(12, -1).contiguous()
        polygon, count = task.convex_hull_xy(corners[..., :2])
        expected_area = task.polygon_area(polygon, count)
        expected_overlap = task.rectangle_overlap_area(polygon, count, regions[:, None])
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            wp.init()
            c, a, r = (t.to(device) for t in (corners, active, regions))
            area, overlap = torch.empty_like(a, dtype=torch.float32), torch.empty_like(a, dtype=torch.float32)
            kw = dict(stream=wp.stream_from_torch(torch.cuda.current_stream())) if device == 'cuda' else dict(device='cpu')
            wp.launch(task._projected_areas, dim=a.shape, inputs=[wp.from_torch(t) for t in (c, a, r, area, overlap)], **kw)
            torch.testing.assert_close(area.cpu(), expected_area * active, atol=2.e-7, rtol=2.e-5)
            torch.testing.assert_close(overlap.cpu(), expected_overlap * active, atol=2.e-7, rtol=2.e-5)

    def test_device_pose_mapping_keeps_roles_and_heights_without_downloading_clutter(self):
        poses = torch.tensor([[-.7, .1, .2], [-.6, .2, .3], [-.8, -.2, .4]])
        positions = task._PoseMapping({2: (-.7, .1, 1.), 0: (-.6, .2, 1.1)}, (2, 0, 1), poses,
                                      {2: 1., 0: 1.1, 1: 1.2})
        with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('unnecessary transfer')):
            self.assertEqual(positions[2], (-.7, .1, 1.))
            self.assertEqual(positions.support_heights[1], 1.2)
        self.assertAlmostEqual(dict(positions)[1][0], -.8, places=6)


class SensorRuntimeTests(unittest.TestCase):
    """Execute the production sensor tensor path without initializing Kit."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        module = types.ModuleType('contact_manipulation.mdp._sensor_numerical_tests')
        module.__package__ = 'contact_manipulation.mdp'
        sys.modules[module.__name__] = module  # dataclasses resolves field types through this entry.
        exec(compile(path.read_text().split('# Native classes are imported only', 1)[0],
                     str(path), 'exec'), module.__dict__)
        cls.sensors = module

    @staticmethod
    def fixture(batch=4):
        tactile_names = tuple(f'touch_{index}' for index in range(17))
        ft_name = 'ToolForceTorqueLink'

        class Contact:
            def __init__(self, name):
                self.num_bodies, self.num_instances, self.body_names = 1, batch, [name]
                self.cfg = SimpleNamespace(history_length=4)
                self.data = SimpleNamespace(
                    net_forces_w=torch.zeros(batch, 1, 3),
                    force_matrix_w_history=torch.zeros(batch, 4, 1, 3, 3),
                )

            def reset(self, ids):
                self.data.net_forces_w[ids] = 0
                self.data.force_matrix_w_history[ids] = 0

            def find_bodies(self, names, preserve_order=False):
                assert preserve_order
                return [self.body_names.index(name) for name in names], list(names)

        # The force sensor is deliberately not body zero. The raw load includes
        # values above the torque sensing range; calibration belongs to the MDP.
        reaction = torch.arange(batch * 3 * 6, dtype=torch.float32).reshape(batch, 3, 6)
        scene = {'robot': SimpleNamespace(
            body_names=['base_link', ft_name, 'hand_base'],
            data=SimpleNamespace(body_incoming_joint_wrench_b=reaction),
            root_physx_view=SimpleNamespace(
                get_link_incoming_joint_force=lambda: reaction,
                link_paths=[['/World/envs/env_0/Robot/base_link',
                             '/World/envs/env_0/Robot/' + ft_name,
                             '/World/envs/env_0/Robot/ROASRightHand/base_link']],
            ),
        )}
        tactile = Contact('tactile')
        tactile.num_bodies, tactile.num_instances = 17, batch * 17
        # Deliberately use a different native body order from observation order.
        tactile.body_names = list(reversed(tactile_names))
        tactile.data.net_forces_w = torch.zeros(batch, 17, 3)
        scene['tactile_contact'] = tactile
        env = SimpleNamespace(scene=scene, num_envs=batch, device='cpu',
                              sim=SimpleNamespace(current_time=0.), cfg=SimpleNamespace(decimation=4))
        asset_module = types.ModuleType('contact_manipulation.config.ur5e.scene')
        asset_module.FORCE_TORQUE_BODY_NAME = ft_name
        asset_module.TACTILE_BODY_NAMES = tactile_names
        with patch.dict(sys.modules, {asset_module.__name__: asset_module}):
            reader = SensorRuntimeTests.sensors.SensorReader(env)
        return env, reader, reaction

    def test_initial_measurement_invalid_then_physics_valid_and_selected_reset_preserves_other_rows(self):
        env, reader, reaction = self.fixture()
        self.assertFalse(reader().wrench_valid.any())
        env.sim.current_time = .1

        def reject(*args, **kwargs):
            raise AssertionError('Sensor hot path read a tensor scalar on the host')

        with patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'item', reject), \
             patch.object(torch.Tensor, 'tolist', reject), patch.object(torch.Tensor, 'cpu', reject), \
             patch.object(torch.Tensor, 'numpy', reject):
            batch = reader()
        torch.testing.assert_close(batch.ideal_wrench, reaction[:, 1])
        self.assertTrue(batch.wrench_valid.all())
        reader.reset(torch.tensor([1, 3]))
        self.assertEqual(reader().wrench_valid.tolist(), [True, False, True, False])
        self.assertEqual(reader().ideal_wrench[1].count_nonzero(), 0)
        torch.testing.assert_close(reader().ideal_wrench[[0, 2]], reaction[[0, 2], 1])
        env.sim.current_time = .2
        self.assertTrue(reader().wrench_valid.all())

    def test_same_timestamp_read_does_not_revalidate_reset_rows_or_refresh_cached_others(self):
        env, reader, reaction = self.fixture()
        env.sim.current_time = .1
        reader.reset(torch.tensor([0]))
        before = reader().ideal_wrench[1].clone()
        reaction[1] += 3
        self.assertFalse(reader().wrench_valid[0])
        torch.testing.assert_close(reader().ideal_wrench[1], before)

    def test_nonfinite_ft_and_tactile_invalidate_only_affected_measurements(self):
        env, reader, reaction = self.fixture()
        reaction[0, 1, 2] = float('nan')
        native_index = env.scene['tactile_contact'].body_names.index('touch_3')
        env.scene['tactile_contact'].data.net_forces_w[2, native_index, 0] = float('inf')
        env.sim.current_time = .1
        batch = reader()
        self.assertEqual(batch.wrench_valid.tolist(), [False, True, True, True])
        self.assertFalse(batch.tactile_valid[2, 3])
        self.assertTrue(batch.tactile_valid[2, 2])
        self.assertTrue(batch.tactile_valid[1, 3])

    def test_grouped_tactile_preserves_every_channel_and_partial_reset(self):
        env, reader, _ = self.fixture()
        expected = torch.arange(4 * 17 * 3, dtype=torch.float32).reshape(4, 17, 3) / 1000
        expected[0, 0] = 0
        sensor = env.scene['tactile_contact']
        sensor.data.net_forces_w.copy_(expected.flip(1))
        env.sim.current_time = .1
        actual = reader()
        torch.testing.assert_close(actual.tactile_force, expected)
        torch.testing.assert_close(actual.tactile_force.norm(dim=-1) > .01, expected.norm(dim=-1) > .01)
        reader.reset(torch.tensor([1, 3]))
        torch.testing.assert_close(reader().tactile_force[[0, 2]], expected[[0, 2]])
        self.assertFalse(reader().tactile_valid[[1, 3]].any())
        torch.testing.assert_close(sensor.data.net_forces_w[[0, 2]], expected[[0, 2]].flip(1))

    def test_visible_depth_pca_is_translation_invariant_and_keeps_upright_orientation(self):
        generator = torch.Generator().manual_seed(2)
        points = torch.rand((1000, 3), generator=generator) * torch.tensor([.12, .04, .19])
        shift = torch.tensor([.1, -.2, .3])
        labels = torch.zeros((1, len(points)), dtype=torch.long)
        original = self.sensors.fit_visible_obbs(points[None], labels, 1, 16)[0][0, 0]
        translated = self.sensors.fit_visible_obbs((points + shift)[None], labels, 1, 16)[0][0, 0]
        torch.testing.assert_close(original[:3] + shift, translated[:3])
        torch.testing.assert_close(original[3:], translated[3:], atol=2.e-6, rtol=1.e-5)
        self.assertEqual(original[4:6].tolist(), [0., 0.])
        self.assertTrue((original[7:] > 0).all())

    def test_camera_config_rejects_invalid_acceptance_ranges_and_retry_counts(self):
        for kwargs in ({'min_target_visible_fraction': 1.1}, {'height_above_shelf': (.3, .2)},
                       {'max_camera_attempts': 0}, {'max_layout_attempts': 0},
                       {'roll_degrees': (float('nan'), 3)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sensors.InitialVisionCfg(**kwargs)


class BatchedRobotKinematicsTests(unittest.TestCase):
    def test_parallel_obb_axes_do_not_falsely_accept_overlapping_boxes(self):
        # A rounded rotation can produce spurious cross-axis residuals in
        # FP32. Relative-coordinate cancellation previously accepted this pair.
        rotation = np.array([[-.8018245697021484, .4289975166320801, .41597890853881836],
                             [-.13086608052253723, .5531646609306335, -.8227289915084839],
                             [-.5830535292625427, -.7141218185424805, -.3873997926712036]])
        ca = np.array([-.04387548938393593, -.18531934916973114, .15092703700065613])
        cb = np.array([-.23477087914943695, -.22714553773403168, .19318510591983795])
        ha = np.array([.08198848366737366, .031128162518143654, .12377825379371643])
        hb = np.array([.08862842619419098, .10184000432491302, .09672331064939499])
        self.assertFalse(task.oriented_box_separated(ca, rotation, ha, cb, rotation, hb, .005))
        for dtype in (torch.float32, torch.float64):
            args = [torch.tensor(value, dtype=dtype) for value in (ca, rotation, ha, cb, rotation, hb)]
            result = task.oriented_boxes_separated_batch(*args, .005)
            self.assertFalse(result)
            expanded = [value.expand(3, *value.shape) for value in args]
            self.assertFalse(task.oriented_boxes_separated_batch(*expanded, .005).any())

    @unittest.skipUnless(torch.cuda.is_available(), 'Fused clearance requires CUDA.')
    def test_fused_clearance_matches_eager_with_self_pairs_invalid_candidates_and_nan(self):
        rng = np.random.default_rng(831)
        n, colliders, obstacles = 512, 2, 5
        device = 'cuda'
        frames = torch.eye(4, device=device).expand(n, colliders, 4, 4).clone()
        frames[..., :3, :3] = torch.tensor(Rotation.random(n*colliders, random_state=rng).as_matrix(),
                                          device=device, dtype=torch.float32).reshape(n, colliders, 3, 3)
        frames[..., :3, 3] = torch.tensor(rng.normal(0., .15, (n, colliders, 3)), device=device)
        frames[..., 2, 3] += 1.1
        joints = torch.eye(4, device=device).expand(n, 4, 4, 4).clone()
        joints[:, 1:4, 2, 3] = torch.tensor([1., 1.4, 1.], device=device)
        joints[:, 2, 0, 3] = .2
        constants = dict(collider_owners=torch.arange(colliders, device=device),
                         collider_offsets=torch.eye(4, device=device).expand(colliders, 4, 4).clone(),
                         collider_half=torch.full((colliders, 3), .045, device=device),
                         self_pairs=torch.tensor([[0, 1]], device=device))
        model = SimpleNamespace(colliders=[None]*colliders, self_collision_pairs=lambda: [(0, 1)],
                                _tensor_constants=lambda *_: constants)
        env = SimpleNamespace(robot_pose_model=model, cfg=SimpleNamespace(robot_start=task.RobotStartRules()))
        centers = torch.tensor(rng.normal(0., .18, (n, obstacles, 3)), device=device, dtype=torch.float32)
        centers[..., 2] += 1.1
        rotations = torch.tensor(Rotation.random(n*obstacles, random_state=rng).as_matrix(),
                                 device=device, dtype=torch.float32).reshape(n, obstacles, 3, 3)
        half = torch.full((n, obstacles, 3), .05, device=device)
        obstacle_valid = torch.tensor(rng.random((n, obstacles)) > .2, device=device)
        candidate_valid = torch.tensor(rng.random(n) > .2, device=device)
        frames[-1, 0, 0, 0] = math.nan
        candidate_valid[-1] = True
        args = (env, frames, joints, frames[:, 0], centers, rotations, half, obstacle_valid)
        expected, expected_height = task._robot_candidate_clearance_eager(*args)
        actual, height = task._robot_candidate_clearance(*args, candidate_valid=candidate_valid)
        torch.testing.assert_close(actual, expected & candidate_valid, rtol=0, atol=0)
        torch.testing.assert_close(height, expected_height)

    @classmethod
    def setUpClass(cls):
        # Load production numerical methods without importing the simulator SDK.
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        tree = ast.parse(path.read_text())
        names = {'_RobotJoint', 'RobotColliderBox', 'RobotPoseModel'}
        nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
        nodes += [node for node in tree.body if (isinstance(node, ast.ClassDef) and node.name in names)
                  or (isinstance(node, ast.FunctionDef) and node.name in ('_arm_ik_kernel', '_ik_orientation_error'))
                  or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                      and node.targets[0].id in ('_ArmVector', '_ArmMatrix', '_ArmPoints'))]
        module = types.ModuleType('contact_test_robot_kinematics')
        sys.modules[module.__name__] = module
        module.__dict__.update(np=np, torch=torch, wp=wp, math=math, dataclass=dataclasses.dataclass,
                               FORCE_TORQUE_PROXIMAL_BODY_NAME='SensorMountProximal',
                               FORCE_TORQUE_BODY_NAME='ToolForceTorqueLink')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), module.__dict__)
        cls.numerics = module

    def model(self):
        module = self.numerics
        model = module.RobotPoseModel()
        model.joint_names = ('pan', 'lift', 'elbow', 'wrist1', 'wrist2', 'wrist3', 'finger')
        model.body_names = ('base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                            'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'palm', 'finger')
        model.arm_ids = np.arange(6)
        model.default_joint_positions = np.array([.1, -.8, 1.1, -.4, .3, -.2, .1])
        model.arm_limits = np.tile([-2*math.pi, 2*math.pi], (6, 1))
        model.root_body_name, model.tcp_body_name = 'base_link', 'palm'
        model.root_transform = np.eye(4)
        model.root_transform[:3, 3] = (.1, -.2, .8)
        joints = []
        for i in range(6):
            pre, inverse_post = np.eye(4), np.eye(4)
            pre[:3, 3] = (.06, .015, .09)
            pre[:3, :3] = Rotation.from_rotvec([.01*i, -.02*i, .03]).as_matrix()
            inverse_post[:3, 3] = (.02, -.03, .04)
            axis = np.eye(3)[[2, 1, 1, 0, 1, 2][i]]
            joints.append(module._RobotJoint(model.joint_names[i], model.body_names[i],
                                            model.body_names[i+1], pre, inverse_post, axis, i, i == 3))
        pre = np.eye(4)
        pre[:3, 3] = (.02, .03, .05)
        joints.append(module._RobotJoint('fixed_palm', 'wrist_3_link', 'palm', pre,
                                        np.eye(4), np.zeros(3), None))
        joints.append(module._RobotJoint('finger', 'palm', 'finger', pre,
                                        np.eye(4), np.array([1., 0., 0.]), 6))
        model.joints = tuple(joints)
        model.body_parents = {joint.child: joint.parent for joint in joints}
        model.colliders = [module.RobotColliderBox(name, np.array([.01, .02, .03]), np.eye(3),
                                                  np.array([.01, .015, .02]), name) for name in model.body_names]
        model.base_origin = np.zeros(3)
        return model

    @unittest.skipUnless(torch.cuda.is_available(), 'Fused IK requires CUDA.')
    def test_fused_ik_reaches_seeded_targets_and_rejects_unreachable_goals(self):
        model = self.model()
        generator = torch.Generator(device='cuda').manual_seed(192)
        truth = torch.randn(256, 6, device='cuda', generator=generator) * .3
        tcp = model.arm_forward_batch(truth)[2]
        positions, valid, _, _, result = model.solve_batch(tcp[:, :3, 3], tcp[:, :3, :3], truth + .08)
        self.assertTrue(valid.all())
        # TF32 FK accumulation can differ from full-float Warp by ~0.15 mm.
        torch.testing.assert_close(result, tcp, atol=5.e-4, rtol=0.)
        self.assertTrue(torch.isfinite(positions).all())
        _, valid, *_ = model.solve_batch(torch.full((8, 3), 9., device='cuda'),
                                         torch.eye(3, device='cuda').expand(8, -1, -1), truth[:8])
        self.assertFalse(valid.any())

    @unittest.skipUnless(torch.cuda.is_available(), 'Robot-start graph capture requires CUDA.')
    def test_robot_start_graph_captures_and_replays_updated_anchor_inputs(self):
        # Exercise the actual sampler graph, rather than only its IK kernel.
        # CPU->CUDA constant construction inside candidate_inputs must fail
        # this test at capture, even if every standalone CUDA kernel passes.
        device = 'cuda'
        default = torch.zeros(7, device=device)
        arm_ids = torch.arange(6, device=device)
        constants = dict(default=default, arm_ids=arm_ids,
                         scene_half=torch.full((2, 3), .02, device=device))

        def forward(arm):
            cosine, sine = arm[:, 0].cos(), arm[:, 0].sin()
            zero, one = torch.zeros_like(cosine), torch.ones_like(cosine)
            rotation = torch.stack((cosine, -sine, zero, sine, cosine, zero, zero, zero, one), -1)
            tcp = torch.eye(4, device=device).expand(len(arm), -1, -1).clone()
            tcp[:, :3, :3] = rotation.reshape(-1, 3, 3)
            tcp[:, 0, 3], tcp[:, 2, 3] = -.24, 1.29
            return None, None, tcp

        def solve(positions, rotations, seeds, *, active_mask):
            # Encode both graph inputs in the selected result. The final joint
            # carries a rotation component solely for this dispatch regression.
            q = torch.cat((seeds, rotations[:, :1, 0]), -1)
            tcp = torch.eye(4, device=device).expand(len(seeds), -1, -1).clone()
            tcp[:, :3, 3], tcp[:, :3, :3] = positions, rotations
            frames = positions.new_zeros((len(seeds), 1, 4, 4))
            return q, active_mask.clone(), frames, frames, tcp

        def clear(_env, frames, *_args, candidate_valid=None):
            return candidate_valid, frames.new_full((len(frames),), .1)

        model = SimpleNamespace(joint_names=tuple(range(7)), base_origin=np.zeros(3),
                                _tensor_constants=lambda *_: constants,
                                arm_forward_batch=forward, solve_batch=solve)
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=8, ik_batch_size=16, orientation_range=(0., 0., 0.))
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device=device)
        anchor = default[None].clone()
        anchor[:, 0] = .2
        cache = SimpleNamespace(sample=sample, proposals=task._episode_proposals([sample], 2, device, torch.float32),
                                robot=None, template=SimpleNamespace(joint_positions=anchor))
        boxes = (torch.zeros((1, 1, 3), device=device),
                 torch.eye(3, device=device).reshape(1, 1, 3, 3),
                 torch.ones((1, 1, 3), device=device),
                 torch.zeros((1, 1), device=device, dtype=torch.bool))
        wp.init()
        with patch.object(task, '_scene_boxes_tensor', return_value=boxes), \
             patch.object(task, '_robot_candidate_clearance', side_effect=clear), torch.inference_mode():
            first, _ = task._sample_robot_starts(env, [sample]*2, [17, 18], scene_cache=cache, return_tensors=True)
            graph = next(iter(model._robot_start_graphs.values()))[1]
            replay, _ = task._sample_robot_starts(env, [sample]*2, [17, 18], scene_cache=cache, return_tensors=True)
            torch.testing.assert_close(replay, first, rtol=0, atol=0)
            torch.testing.assert_close(first[:, :6], anchor[:, :6].expand(2, -1))
            torch.testing.assert_close(first[:, 6], anchor[:, 0].cos().expand(2))
            updated = anchor.clone()
            updated[:, :6] += .5
            cache.template = SimpleNamespace(joint_positions=updated)
            changed, _ = task._sample_robot_starts(env, [sample]*2, [17, 18], scene_cache=cache, return_tensors=True)
            self.assertEqual(len(model._robot_start_graphs), 1)
            self.assertIs(next(iter(model._robot_start_graphs.values()))[1], graph)
            torch.testing.assert_close(changed[:, :6], updated[:, :6].expand(2, -1))
            torch.testing.assert_close(changed[:, 6], updated[:, 0].cos().expand(2))
            self.assertTrue((changed[:, -1] == 1).all())

    def test_collapsed_tensor_fk_matches_all_scalar_links_including_reverse_and_default_finger(self):
        model = self.model()
        q = torch.tensor(np.tile(model.default_joint_positions, (17, 1)), dtype=torch.float64)
        q[:, :6] += torch.randn((17, 6), generator=torch.Generator().manual_seed(11), dtype=q.dtype) * .5
        actual = model.forward_batch(q)
        expected = np.stack([[model.forward(row.numpy())[name] for name in model.body_names] for row in q])
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1.e-12, rtol=1.e-12)

    def test_hand_arm_pairs_cover_forearm_and_wrist_but_exclude_fixed_mount(self):
        model = self.model()
        pairs = {frozenset((model.colliders[a].body_name, model.colliders[b].body_name))
                 for a, b in model.self_collision_pairs()}
        for hand in ('palm', 'finger'):
            for arm in ('base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
                        'wrist_1_link', 'wrist_2_link'):
                self.assertIn(frozenset((hand, arm)), pairs)
            self.assertNotIn(frozenset((hand, 'wrist_3_link')), pairs)
        self.assertNotIn(frozenset(('palm', 'finger')), pairs)

    def test_scalar_and_batch_reject_hand_arm_overlap_folded_forearm_and_low_elbow(self):
        batch = 4
        frames = torch.eye(4, dtype=torch.float64).repeat(batch, 2, 1, 1)
        frames[:, :, 2, 3] = 1.
        frames[:, 1, 0, 3] = .2
        frames[2, 1, 0, 3] = .03  # Hand intersects forearm.
        joints = torch.eye(4, dtype=torch.float64).repeat(batch, 6, 1, 1)
        joints[:, 1, :3, 3] = torch.tensor([.2, 0., 1.])
        joints[:, 2, :3, 3] = torch.tensor([0., 0., 1.3])
        joints[:, 3, :3, 3] = torch.tensor([-.2, 0., 1.])
        joints[1, 3, 0, 3] = .1  # Wrist behind elbow despite high elbow.
        joints[3, 2, 2, 3] = 1.  # No elbow-up margin.
        constants = dict(collider_owners=torch.tensor([0, 1]),
                         collider_offsets=torch.eye(4, dtype=torch.float64).repeat(2, 1, 1),
                         collider_half=torch.full((2, 3), .02, dtype=torch.float64),
                         self_pairs=torch.tensor([[0, 1]]))
        colliders = [self.numerics.RobotColliderBox(name, np.zeros(3), np.eye(3), np.full(3, .02), name)
                     for name in ('forearm_link', 'palm')]
        model = SimpleNamespace(_tensor_constants=lambda *_: constants, colliders=colliders,
                                self_collision_pairs=lambda: np.array([[0, 1]]))
        env = SimpleNamespace(robot_pose_model=model, cfg=SimpleNamespace(robot_start=task.RobotStartRules()))
        valid, height = task._robot_candidate_clearance(
            env, frames, joints, frames[:, 1], frames.new_empty(batch, 0, 3),
            frames.new_empty(batch, 0, 3, 3), frames.new_empty(batch, 0, 3), torch.empty(batch, 0, dtype=torch.bool))
        self.assertEqual(valid.tolist(), [True, False, False, False])
        for row in range(batch):
            model.arm_landmarks = lambda _, row=row: joints[row, [1, 2, 3], :3, 3].numpy()
            transforms = {box.body_name: frames[row, index].numpy() for index, box in enumerate(colliders)}
            reason = task.robot_start_collision_reason(env, None, transforms, obstacles=[])
            scalar_valid = reason is None and float(height[row]) >= env.cfg.robot_start.min_elbow_height
            self.assertEqual(scalar_valid, bool(valid[row]))
            if row == 2:
                self.assertIn('forearm_link / palm', reason)

    def test_analytic_tensor_jacobian_matches_finite_difference_of_exact_usd_chain(self):
        model = self.model()
        arm = torch.tensor(model.default_joint_positions[:6][None], dtype=torch.float64)
        _, _, tcp, jacobian = model.arm_forward_batch(arm, jacobian=True)
        for index in range(6):
            shifted = arm.clone()
            shifted[:, index] += 1.e-6
            _, _, other = model.arm_forward_batch(shifted)
            linear = (other[0, :3, 3] - tcp[0, :3, 3]) / 1.e-6
            angular = Rotation.from_matrix((other[0, :3, :3] @ tcp[0, :3, :3].T).numpy()).as_rotvec() / 1.e-6
            torch.testing.assert_close(jacobian[0, :, index], torch.cat((linear, torch.tensor(angular))),
                                       atol=3.e-7, rtol=2.e-6)

    def test_batch_ik_reaches_distinct_targets_and_obeys_limits(self):
        model = self.model()
        truth = torch.tensor(np.tile(model.default_joint_positions[:6], (6, 1)), dtype=torch.float64)
        truth += torch.randn(truth.shape, generator=torch.Generator().manual_seed(22), dtype=truth.dtype) * .15
        _, _, target = model.arm_forward_batch(truth)
        q, valid, _, _, result = model.solve_batch(target[:, :3, 3], target[:, :3, :3], truth + .08)
        self.assertTrue(valid.all())
        self.assertTrue((q[:, :6].abs() <= 2*math.pi).all())
        self.assertTrue(((result[:, :3, 3] - target[:, :3, 3]).norm(dim=-1) < 5.e-4).all())
        torch.testing.assert_close(q[:, 6], torch.full((6,), .1, dtype=q.dtype))

    def test_physical_link_check_handles_distinct_origins_and_rejects_one_corrupt_link_only(self):
        model = self.model()
        q = np.tile(model.default_joint_positions, (3, 1))
        q[:, :6] += np.array([[-.03], [0.], [.04]])
        predicted = model.forward_batch(torch.tensor(q, dtype=torch.float64)).numpy()
        origins = torch.tensor([[0., 0., 0.], [5., -3., 0.], [-2., 8., 0.]], dtype=torch.float64)
        poses = torch.zeros((3, len(model.body_names), 7), dtype=torch.float64)
        poses[..., :3] = torch.tensor(predicted[..., :3, 3]) + origins[:, None]
        poses[..., 3:] = torch.tensor(Rotation.from_matrix(predicted[..., :3, :3].reshape(-1, 3, 3)).as_quat()).reshape(3, -1, 4)
        starts = [(row, None, None, None, None) for row in q]
        env = SimpleNamespace(robot_pose_model=model)
        result = task._check_physical_robot_starts(env, [None]*3, starts, poses, origins)
        self.assertTrue(result.all())
        poses[1, -1, 0] += .001
        torch.testing.assert_close(task._check_physical_robot_starts(env, [None]*3, starts, poses, origins),
                                   torch.tensor([True, False, True]))

    def test_unreachable_ik_targets_are_reported_invalid_without_invalid_joint_values(self):
        model = self.model()
        seed = torch.tensor(model.default_joint_positions[:6][None], dtype=torch.float64)
        q, valid, *_ = model.solve_batch(torch.tensor([[9., 9., 9.]], dtype=seed.dtype),
                                        torch.eye(3, dtype=seed.dtype)[None], seed)
        self.assertFalse(valid.any())
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue((q[:, :6].abs() <= 2*math.pi).all())

    def test_initial_anchor_covers_fixed_tcp_and_orientation_strata_without_random_target_draws(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(ik_batch_size=32)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
            spawn_rules=task.SpawnRules(x_range=(-.8, -.4)), cfg=SimpleNamespace(robot_start=rules), device='cpu')
        captured = []
        def solve(positions, rotations, seeds):
            captured.append((positions.clone(), rotations.clone(), seeds.clone()))
            n = len(seeds)
            tcp = torch.eye(4).expand(n, -1, -1).clone()
            tcp[:, :3, 3] = positions
            return (torch.zeros((n, len(model.joint_names))), torch.zeros(n, dtype=torch.bool),
                    torch.zeros(n, 7, 4, 4), torch.zeros(n, 6, 4, 4), tcp)
        with patch.object(model, 'solve_batch', side_effect=solve), \
             patch.object(task, '_robot_candidate_clearance',
                          side_effect=lambda _env, frames, *_a: (torch.ones(len(frames), dtype=torch.bool),
                                                                 torch.full((len(frames),), .1))):
            result, bounds = task._sample_robot_starts(env, [sample], [571], return_tensors=True)
        positions, rotations, seeds = [torch.cat([part[k] for part in captured]) for k in range(3)]
        self.assertEqual(len(positions), 27*9*4)
        self.assertEqual(float(result[0, -1]), 0.)
        fractions = (positions-bounds[:, ::2])/(bounds[:, 1::2]-bounds[:, ::2])
        for axis in range(3):
            torch.testing.assert_close(torch.unique(fractions[:, axis]), torch.tensor((.05, .5, .95)),
                                       atol=8.e-7, rtol=0)
        nominal = torch.tensor(task.ROBOT_START_TCP_ROTATION, dtype=rotations.dtype)
        local = Rotation.from_matrix((nominal.T[None] @ rotations).numpy()).as_rotvec()
        unit = local/np.asarray(rules.orientation_range)
        unique = np.unique(np.round(unit, 5), axis=0)
        self.assertEqual(len(unique), 9)
        self.assertTrue(np.any((unique == 0.).all(-1)))
        self.assertTrue(np.isin(unique, (-.9, 0., .9)).all())
        torch.testing.assert_close(positions.reshape(-1, 4, 3), positions[::4, None].expand(-1, 4, -1),
                                   atol=0, rtol=0)
        torch.testing.assert_close(seeds[::4], seeds[:1].expand(len(seeds)//4, -1), atol=0, rtol=0)

    def test_reset_singularity_uses_native_controlled_com_jacobian_not_link_or_palm(self):
        model = self.model()
        q = torch.tensor(model.default_joint_positions[:6][None], dtype=torch.float64)
        frames, joints, _ = model.arm_forward_batch(q)
        body_index = model.body_names.index('wrist_3_link')
        com = (.011, -.009, -.0229)
        settings = ('wrist_3_link', body_index, .31, .006, com)
        with patch.object(task, 'singularity_margin', return_value=torch.ones(1)) as shared:
            self.assertTrue(task._robot_start_nonsingular(model, frames, joints, settings).all())
        jacobian = shared.call_args.args[0]
        self.assertEqual(shared.call_args.kwargs, dict(characteristic_length=.31))
        full = torch.tensor(model.default_joint_positions[None], dtype=q.dtype)
        pose = model.forward_batch(full)[0, body_index]
        point = pose[:3, 3]+pose[:3, :3] @ pose.new_tensor(com)
        columns = []
        for index in range(6):
            changed = full.clone()
            changed[0, index] += 1.e-6
            other = model.forward_batch(changed)[0, body_index]
            other_point = other[:3, 3]+other[:3, :3] @ other.new_tensor(com)
            linear = (other_point-point)/1.e-6
            angular = Rotation.from_matrix((other[:3, :3] @ pose[:3, :3].T).numpy()).as_rotvec()/1.e-6
            columns.append(torch.cat((linear, torch.tensor(angular))))
        oracle = torch.stack(columns, -1)[None]
        torch.testing.assert_close(jacobian, oracle, atol=3.e-7, rtol=3.e-6)
        palm_jacobian = model.arm_forward_batch(q, jacobian=True)[3]
        self.assertGreater(float((jacobian[:, :3]-palm_jacobian[:, :3]).abs().max()), .01)

    def test_reset_singularity_predicate_matches_enabled_failure_boundary(self):
        model = self.model()
        threshold = .005
        values = torch.tensor((float('nan'), float('inf'), 0., threshold,
                               np.nextafter(threshold, math.inf), .02), dtype=torch.float64)
        q = torch.tensor(model.default_joint_positions[:6][None], dtype=values.dtype).expand(len(values), -1)
        frames, joints, _ = model.arm_forward_batch(q)
        settings = ('wrist_3_link', model.body_names.index('wrist_3_link'), .25, threshold, (0., 0., -.0229))
        with patch.object(task, 'singularity_margin', return_value=values):
            keep = task._robot_start_nonsingular(model, frames, joints, settings)
        terminated = ~torch.isfinite(values) | (values <= threshold)
        torch.testing.assert_close(keep, ~terminated)

    def test_enabled_singularity_configuration_invalidates_cached_reset_inputs(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        term = SimpleNamespace(func=terminations.failure, params=dict(reason='singularity'), time_out=False)
        cfg = SimpleNamespace(robot_start=task.RobotStartRules(), mdp=task.MdpCfg(),
            terminations=SimpleNamespace(singularity=None),
            actions=SimpleNamespace(arm_action=SimpleNamespace(body_name='wrist_3_link')))
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
            spawn_rules=task.SpawnRules(x_range=(-.8, -.4)), cfg=cfg, device='cpu')
        coms = torch.zeros((1, len(model.body_names), 7))
        getter = unittest.mock.Mock(return_value=coms)
        env.scene = {'robot': SimpleNamespace(body_names=model.body_names,
            root_physx_view=SimpleNamespace(get_coms=getter))}
        cache = SimpleNamespace(sample=sample, proposals=task._episode_proposals([sample], 2, 'cpu', torch.float32), robot=None)
        def solve(positions, rotations, seeds):
            n = len(seeds)
            frames, joints, tcp = model.arm_forward_batch(seeds)
            return (torch.zeros((n, len(model.joint_names))), torch.zeros(n, dtype=torch.bool), frames, joints, tcp)
        keys = []
        with patch.object(model, 'solve_batch', side_effect=solve), \
             patch.object(task, '_robot_candidate_clearance',
                          side_effect=lambda _env, frames, *_a: (torch.ones(len(frames), dtype=torch.bool),
                                                                 torch.full((len(frames),), .1))):
            with patch.object(task, '_robot_start_nonsingular', side_effect=AssertionError('disabled criterion evaluated')):
                task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
            cfg.terminations.singularity = term
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
            cfg.mdp.singularity_termination_threshold *= 2
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
            cfg.mdp.singularity_characteristic_length = .3
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
            model._native_body_com_offsets['wrist_3_link'] = (0., 0., -.0229)
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
            cfg.terminations.singularity = None
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=1)
            keys.append(cache.robot[0])
        self.assertEqual(keys[0], keys[-1])
        self.assertEqual(len(set(keys)), 5)
        self.assertEqual(getter.call_count, 1)
        self.assertIsNone(task._robot_start_singularity_settings(env))
        # Both captured-dispatch and scene-input cache keys fingerprint the
        # same host settings; candidate tensors never materialize a host scalar.
        tree = ast.parse(Path(task.__file__).read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == '_sample_robot_starts')
        key_values = [node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'key' for target in node.targets)]
        self.assertEqual(len(key_values), 2)
        self.assertTrue(all(any(isinstance(child, ast.Name) and child.id == 'singularity_settings'
                                for child in ast.walk(value)) for value in key_values))

    def test_candidate_seed_stream_is_unchanged_for_small_and_large_memory_caps(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=80)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        expected = None
        for cap in (8, 1, 3, 7, 16, 32, 64, 1024):
            rules.ik_batch_size = cap
            proposals = []
            def solve(positions, rotations, seeds):
                proposals.append(torch.cat((positions, rotations.flatten(1), seeds), -1))
                n = len(seeds)
                # Force all proposals to be visited, isolating RNG/batching
                # from solver convergence and geometric acceptance.
                tcp = torch.eye(4).expand(n, -1, -1).clone()
                tcp[:, :3, 3] = positions
                return (torch.zeros((n, len(model.joint_names))), torch.zeros(n, dtype=torch.bool),
                        torch.zeros((n, 7, 4, 4)), torch.zeros((n, 6, 4, 4)), tcp)
            def clear(_env, frames, *_args):
                return torch.ones(len(frames), dtype=torch.bool), torch.full((len(frames),), .1)
            with patch.object(model, 'solve_batch', side_effect=solve), \
                 patch.object(task, '_robot_candidate_clearance', side_effect=clear):
                self.assertEqual(task._sample_robot_starts(env, [sample], [571]), [None])
            actual = torch.cat(proposals)
            if expected is None:
                expected = actual
            else:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_shared_start_perturbs_eef_orientation_and_retries_same_target_from_anchor_first(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=80)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        anchor = torch.tensor(model.default_joint_positions[None], dtype=torch.float32)
        anchor[:, :6] += torch.tensor([.05, -.1, .12, -.07, .03, .04])
        nominal = model.arm_forward_batch(anchor[:, :6])[2][0, :3, :3]
        proposals = task._episode_proposals([sample], 2, 'cpu', torch.float32)
        cache = SimpleNamespace(sample=sample, proposals=proposals, robot=None,
                                template=SimpleNamespace(joint_positions=anchor))
        captured = []

        def solve(positions, rotations, seeds):
            captured.append((positions.clone(), rotations.clone(), seeds.clone()))
            n = len(seeds)
            tcp = torch.eye(4).expand(n, -1, -1).clone()
            tcp[:, :3, 3], tcp[:, :3, :3] = positions, rotations
            return (anchor.expand(n, -1).clone(), torch.zeros(n, dtype=torch.bool),
                    torch.zeros(n, 7, 4, 4), torch.zeros(n, 6, 4, 4), tcp)

        with patch.object(model, 'solve_batch', side_effect=solve), \
             patch.object(task, '_robot_candidate_clearance',
                          side_effect=lambda _env, frames, *_a: (torch.ones(len(frames), dtype=torch.bool),
                                                                 torch.full((len(frames),), .1))):
            self.assertEqual(task._sample_robot_starts(env, [sample], [571], scene_cache=cache), [None])
        positions, rotations, seeds = (torch.cat([candidate[column] for candidate in captured]) for column in range(3))
        self.assertEqual(len(positions), rules.max_attempts)
        # A pose is proposed once, then four IK branches solve exactly that pose.
        torch.testing.assert_close(positions.reshape(-1, 4, 3), positions[::4, None].expand(-1, 4, -1), rtol=0, atol=0)
        torch.testing.assert_close(rotations.reshape(-1, 4, 3, 3),
                                   rotations[::4, None].expand(-1, 4, -1, -1), rtol=0, atol=0)
        torch.testing.assert_close(seeds[::4], anchor[:, :6].expand(len(seeds) // 4, -1), rtol=0, atol=0)
        self.assertTrue((seeds[1::4] - seeds[::4]).abs().any())
        relative = nominal.T[None] @ rotations[::4]
        perturbation = torch.tensor(Rotation.from_matrix(relative.numpy()).as_rotvec())
        limits = torch.tensor(rules.orientation_range, dtype=perturbation.dtype)
        self.assertTrue((perturbation.abs() <= limits + 2.e-6).all())
        self.assertTrue((perturbation > 0).any(0).all())
        self.assertTrue((perturbation < 0).any(0).all())
        self.assertGreater(float(perturbation.abs().max()), math.radians(1.))
        torch.testing.assert_close(rotations.transpose(-1, -2) @ rotations,
                                   torch.eye(3).expand(len(rotations), -1, -1), atol=3.e-6, rtol=0)
        geometry_cache = cache.robot
        updated_q = anchor.clone()
        updated_q[:, 0] += .2
        cache.template = SimpleNamespace(joint_positions=updated_q)
        captured.clear()
        with patch.object(model, 'solve_batch', side_effect=solve), \
             patch.object(task, '_robot_candidate_clearance',
                          side_effect=lambda _env, frames, *_a: (torch.ones(len(frames), dtype=torch.bool),
                                                                 torch.full((len(frames),), .1))):
            result, bounds = task._sample_robot_starts(
                env, [sample], [571], scene_cache=cache, max_attempts=1, return_tensors=True)
        self.assertEqual(result.shape, (1, len(model.joint_names) + 5))
        self.assertEqual(bounds.shape, (1, 6))
        self.assertEqual(float(result[0, -1]), 0.)
        self.assertEqual(len(captured), 1)
        torch.testing.assert_close(captured[0][2], updated_q[:, :6], rtol=0, atol=0)
        updated_nominal = model.arm_forward_batch(updated_q[:, :6])[2][0, :3, :3]
        torch.testing.assert_close(updated_nominal.T[None] @ captured[0][1], relative[:1], atol=2.e-6, rtol=0)
        # Refresh only the anchor: immutable scene boxes/regions remain reusable.
        self.assertIs(cache.robot, geometry_cache)
        self.assertIs(cache.start_anchor[0], cache.template)

    def test_shared_robot_geometry_cache_reuses_boxes_and_invalidates_on_rule_change(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules = task.RobotStartRules(max_attempts=2)
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry,
                              spawn_rules=task.SpawnRules(x_range=(-.8, -.4)),
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        proposals = task._episode_proposals([sample], 2, 'cpu', torch.float32)
        cache = SimpleNamespace(sample=sample, proposals=proposals, robot=None)
        before = [value.clone() for value in proposals]
        with patch.object(task, '_scene_boxes_tensor', wraps=task._scene_boxes_tensor) as boxes, \
             patch.object(task, '_robot_start_regions', wraps=task._robot_start_regions) as regions:
            first = task._sample_robot_starts(env, [sample]*3, [17, 18, 19], scene_cache=cache, max_attempts=2)
            again = task._sample_robot_starts(env, [sample]*3, [17, 18, 19], scene_cache=cache, max_attempts=2)
            self.assertEqual(boxes.call_count, 1)
            self.assertEqual(regions.call_count, 1)
            self.assertEqual(boxes.call_args.args[2][0].shape[0], 1)
            cached = cache.robot
            rules.lateral_half_width += .01
            task._sample_robot_starts(env, [sample], [17], scene_cache=cache, max_attempts=2)
            self.assertEqual(boxes.call_count, 2)
            self.assertIsNot(cache.robot, cached)
        for left, right in zip(first, again):
            if left is None:
                self.assertIsNone(right)
            else:
                for actual, expected in zip(left, right):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(proposals, before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_robot_start_recovery_restricts_position_and_rotation_around_same_scene_anchor(self):
        model = self.model()
        model.shelf_boxes = []
        geometry = tuple(task.ObjectGeometry(str(i), (-.02, -.02, 0.), (.02, .02, .08)) for i in range(2))
        sample = task.SceneSample((0, 1), 0, 1, {0: (-.7, 0., 1.05), 1: (-.55, 0., 1.05)},
                                  {0: 0., 1: 0.}, (-.6, -.4, -.05, .05), (-.55, .2))
        rules, spawn = task.RobotStartRules(), task.SpawnRules(x_range=(-.8, -.4))
        bounds = np.array(task.robot_start_region(sample, geometry, spawn, rules))
        center = (bounds[::2] + bounds[1::2]) / 2
        anchor = torch.tensor(model.default_joint_positions[None], dtype=torch.float32)
        initial_tcp = model.arm_forward_batch(anchor[:, :6])[2][0]
        model.root_transform[:3, 3] += center - initial_tcp[:3, 3].numpy()
        model._tensor_cache = {}
        nominal = model.arm_forward_batch(anchor[:, :6])[2][0, :3, :3]
        env = SimpleNamespace(robot_pose_model=model, spawn_geometries=geometry, spawn_rules=spawn,
                              cfg=SimpleNamespace(robot_start=rules), device='cpu')
        cache = SimpleNamespace(sample=sample, proposals=task._episode_proposals([sample], 2, 'cpu', torch.float32),
                                robot=None, template=SimpleNamespace(joint_positions=anchor))
        captured = []
        def solve(positions, rotations, seeds):
            captured.append((positions.clone(), rotations.clone()))
            n = len(seeds)
            tcp = torch.eye(4).expand(n, -1, -1).clone()
            tcp[:, :3, 3], tcp[:, :3, :3] = positions, rotations
            return (anchor.expand(n, -1).clone(), torch.zeros(n, dtype=torch.bool),
                    torch.zeros(n, 7, 4, 4), torch.zeros(n, 6, 4, 4), tcp)
        with patch.object(model, 'solve_batch', side_effect=solve), \
             patch.object(task, '_robot_candidate_clearance',
                          side_effect=lambda _env, frames, *_a: (torch.ones(len(frames), dtype=torch.bool),
                                                                 torch.full((len(frames),), .1))):
            _, actual_bounds = task._sample_robot_starts(
                env, [sample], [912], scene_cache=cache, recovery=True, max_attempts=9, return_tensors=True)
        position, rotation = (torch.cat([value[column] for value in captured]) for column in range(2))
        half = torch.tensor(rules.recovery_position_half_width)
        expected_low = torch.tensor(center, dtype=half.dtype) - half
        expected_high = torch.tensor(center, dtype=half.dtype) + half
        torch.testing.assert_close(actual_bounds[0, ::2], expected_low, atol=2.e-7, rtol=0)
        torch.testing.assert_close(actual_bounds[0, 1::2], expected_high, atol=2.e-7, rtol=0)
        self.assertTrue((position >= actual_bounds[0, ::2]).all())
        self.assertTrue((position <= actual_bounds[0, 1::2]).all())
        local = torch.tensor(Rotation.from_matrix((nominal.T[None] @ rotation).numpy()).as_rotvec())
        limits = torch.tensor(rules.orientation_range) * rules.recovery_rotation_scale
        self.assertTrue((local.abs() <= limits + 2.e-6).all())
        self.assertGreater(float(local.abs().max()), 0.)
        self.assertEqual(len(position), 9)
        self.assertEqual(rules.max_attempts, 80)

    def test_tensor_rotation_error_handles_exact_pi_without_silent_zero(self):
        model = self.model()
        current = torch.eye(3, dtype=torch.float64)[None]
        target = torch.tensor(Rotation.from_rotvec(np.array([1., 2., 3.]) / np.sqrt(14) * math.pi).as_matrix())[None]
        actual = model._rotation_error_batch(target, current)[0]
        self.assertAlmostEqual(float(actual.norm()), math.pi, places=10)
        np.testing.assert_allclose(Rotation.from_rotvec(actual.numpy()).as_matrix(), target[0].numpy(), atol=1.e-12)

    def test_batched_sat_matches_complete_scalar_obb_oracle_and_parallel_touches(self):
        rng = np.random.default_rng(90)
        n = 400
        first, second = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
        rotation_a = Rotation.random(n, random_state=rng).as_matrix()
        rotation_b = Rotation.random(n, random_state=rng).as_matrix()
        half_a, half_b = rng.uniform(.01, .6, size=(n, 3)), rng.uniform(.01, .6, size=(n, 3))
        for gap in (0., .015):
            expected = torch.tensor([task.oriented_box_separated(a, ra, ha, b, rb, hb, gap)
                                    for a, ra, ha, b, rb, hb in zip(first, rotation_a, half_a, second, rotation_b, half_b)])
            actual = task.oriented_boxes_separated_batch(*[torch.tensor(value) for value in
                (first, rotation_a, half_a, second, rotation_b, half_b)], gap)
            torch.testing.assert_close(actual, expected)
        center = torch.zeros((3, 3), dtype=torch.float64)
        shifted = center.clone()
        shifted[:, 0] = torch.tensor([1., 1.-1.e-6, 1.+.015], dtype=center.dtype)
        rotation = torch.eye(3, dtype=center.dtype).expand(3, -1, -1)
        half = torch.full_like(center, .5)
        torch.testing.assert_close(task.oriented_boxes_separated_batch(center, rotation, half, shifted, rotation, half),
                                   torch.tensor([True, False, True]))
        torch.testing.assert_close(task.oriented_boxes_separated_batch(center, rotation, half, shifted, rotation, half, .015),
                                   torch.tensor([False, False, True]))





class GeometricVisionTests(unittest.TestCase):
    """Use the production Warp kernel on CPU; no Kit or renderer is involved."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import warp as wp
        cls.wp = wp
        cls.cache_dir = tempfile.TemporaryDirectory(prefix='contact_ray_tests_')
        wp.init()
        cls.old_cache_dir = wp.config.kernel_cache_dir
        wp.config.kernel_cache_dir = cls.cache_dir.name
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        module = types.ModuleType('contact_geometric_test_runtime')
        module.__file__ = str(path)
        sys.modules[module.__name__] = module
        numerical = path.read_text().split('# Native classes are imported only', 1)[0]
        module.__package__ = 'contact_manipulation.mdp'
        exec(compile(numerical, str(path), 'exec'), module.__dict__)
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in ('_surface_moments', '_surface_covariance', '_surface_extents', '_fit_visible_obbs_cuda',
                                   '_initial_viewpoints', '_initial_ray_frames', '_initial_surface_rays',
                                   'InitialVisionSensor')]
        module.__dict__.update(wp=wp, np=np)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), module.__dict__)
        cls.module = module

    @classmethod
    def tearDownClass(cls):
        cls.wp.config.kernel_cache_dir = cls.old_cache_dir
        cls.cache_dir.cleanup()

    def fixture(self, batch=1, device='cpu'):
        wp = self.wp
        # Two role surfaces and one optional distractor surface.
        # Avoid tangent rays exactly on an edge: float32 rigid transforms can
        # classify that measure-zero boundary on either side of the triangle.
        points = np.array([[-.2937, -.2937, 0], [.2937, -.2937, 0],
                           [.2937, .2937, 0], [-.2937, .2937, 0]], np.float32)
        mesh = wp.Mesh(points=wp.array(points, dtype=wp.vec3, device=device),
                       indices=wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device=device))
        sensor = self.module.InitialVisionSensor.__new__(self.module.InitialVisionSensor)
        sensor.cfg = self.module.InitialVisionCfg(ray_width=80, ray_height=60,
            observation_ray_width=80, observation_ray_height=60, observation_min_depth_pixels=16)
        sensor.capacity = sensor.cfg.ray_batch_size
        sensor.object_count = 3
        sensor.meshes = [mesh]
        sensor.mesh_ids = wp.array([mesh.id] * 3, dtype=wp.uint64, device=device)
        sensor.mesh_bounds = wp.array(np.stack([np.stack((points.min(0), points.max(0)))] * 3),
                                      dtype=wp.vec3, device=device)
        rows, columns = torch.meshgrid(torch.arange(120), torch.arange(160), indexing='ij')
        sensor.directions = torch.stack(((columns + .5 - 80) / 60,
                                         (rows + .5 - 60) / 60, torch.ones_like(rows)), -1).float().reshape(-1, 3)
        sensor.crop_directions = sensor.directions.reshape(120, 160, 3)[30:90, 40:120].reshape(-1, 3).contiguous()
        sensor.intrinsics = torch.tensor([[60., 0., 40.], [0., 60., 30.], [0., 0., 1.]])
        poses = torch.zeros(batch, 3, 7)
        poses[..., 3] = 1
        poses[:, 0, 2] = 1.4
        poses[:, 1, 2] = .8
        poses[:, 2, 2] = .5
        enabled = torch.tensor([[0, 1, -1]], dtype=torch.int32).expand(batch, -1).clone()
        eyes = torch.zeros(batch, 3)
        rotation = torch.tensor([[1., 0., 0., 0.]]).expand(batch, -1).clone()
        roles = torch.tensor([[0, 1]], dtype=torch.int32).expand(batch, -1).clone()
        sensor.directions = sensor.directions.to(device)
        sensor.crop_directions = sensor.crop_directions.to(device)
        sensor.intrinsics = sensor.intrinsics.to(device)
        return (sensor, *(tensor.to(device) for tensor in (poses, enabled, eyes, rotation, roles)))

    def test_front_surface_occludes_target_but_isolated_denominator_remains(self):
        sensor, *args = self.fixture()
        depth, labels, full, visible, ratios, accepted = sensor._trace(*args)
        self.assertGreater(int(full[0, 0]), 0)
        self.assertEqual(int(visible[0, 0]), 0)
        self.assertEqual(float(ratios[0, 1]), 1.)
        self.assertFalse(accepted.any())
        torch.testing.assert_close(depth[labels == 1], torch.full_like(depth[labels == 1], .8))

    def test_low_resolution_observation_queries_only_image_rays_and_preserves_visible_hits(self):
        sensor, poses, active, eyes, rotations, roles = self.fixture()
        sensor.cfg = self.module.InitialVisionCfg()
        poses[:, 1, 0] = .3  # Partly visible Target and exposed Blocker.
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
        sensor.env = SimpleNamespace(scene=Scene(
            object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
            robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=eyes, root_quat_w=rotations))),
            device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=10)))
        request = dict(env_id=0, active_ids=(0, 1), target_id=0, blocker_id=1)
        with patch.object(self.wp, 'launch', wraps=self.wp.launch) as launch, \
             patch.object(sensor, '_fit_observation', wraps=sensor._fit_observation) as fit:
            snapshot = sensor.observe_at_viewpoint(
                [request], torch.cat((eyes, rotations), -1), retain_images=True)[0]
        image_queries = [call.kwargs['dim'] for call in launch.call_args_list
                         if call.kwargs.get('kernel') is self.module._initial_surface_rays]
        self.assertEqual(image_queries, [(1, 64 * 48)])
        self.assertEqual(fit.call_args.kwargs['min_depth_pixels'], 3)
        self.assertEqual(snapshot.depth.shape, (48, 64, 1))
        self.assertEqual(snapshot.role_masks.shape, (48, 64, 2))
        self.assertEqual(snapshot.geometry_b.shape, (10, 10))
        self.assertEqual(snapshot.geometry_valid.tolist(), [True, True] + [False] * 8)
        self.assertEqual(snapshot.observed_ids.tolist(), [0, 1] + [-1] * 8)
        self.assertIsNone(snapshot.visible_fractions)
        self.assertIsNone(snapshot.pixel_counts)
        self.assertEqual(snapshot.geometry_b[2:].count_nonzero(), 0)
        torch.testing.assert_close(snapshot.intrinsics, torch.tensor(
            [[48., 0., 32.], [0., 48., 24.], [0., 0., 1.]]))
        # Full-silhouette mode at the same pixel grid is an independent
        # first-hit oracle; observation mode omits only the denominator work.
        sensor.cfg.ray_width, sensor.cfg.ray_height = 64, 48
        sensor.directions, sensor.crop_directions, sensor.intrinsics = sensor._observation_grid('cpu')
        full = sensor._trace(poses, active, eyes, rotations, roles)
        torch.testing.assert_close(snapshot.depth[..., 0].flatten(), full[0][0], rtol=0, atol=0)
        torch.testing.assert_close(snapshot.role_masks.reshape(-1, 2),
                                   full[1][0, :, None] == roles[0], rtol=0, atol=0)

    def test_low_resolution_minimum_is_explicit_and_does_not_change_certificate_trace(self):
        sensor, poses, active, eyes, rotations, roles = self.fixture()
        reference = sensor._trace(poses, active, eyes, rotations, roles)
        sensor.cfg.observation_ray_width, sensor.cfg.observation_ray_height = 32, 24
        sensor.cfg.observation_min_depth_pixels = 3
        for actual, expected in zip(sensor._trace(poses, active, eyes, rotations, roles), reference):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sensor.env = SimpleNamespace(cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        points = torch.tensor([[[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.]]])
        labels = torch.zeros((1, 3), dtype=torch.int32)
        missing = sensor._fit_observation(points, labels, active, roles)
        observed = sensor._fit_observation(points, labels, active, roles,
                                           min_depth_pixels=sensor.cfg.observation_min_depth_pixels)
        self.assertFalse(missing[1].any())
        self.assertEqual(observed[1].tolist(), [[True, False, False]])
        self.assertEqual(sensor.cfg.min_depth_pixels, 16)

    def test_observation_grid_reuses_buffers_and_rebuilds_for_resolution_change(self):
        sensor, *_ = self.fixture()
        first = sensor._observation_grid('cpu')
        for actual, expected in zip(sensor._observation_grid(torch.device('cpu')), first):
            self.assertIs(actual, expected)
        sensor.cfg.observation_ray_width, sensor.cfg.observation_ray_height = 32, 24
        expanded, crop, intrinsics = sensor._observation_grid('cpu')
        self.assertEqual(expanded.shape, (4 * 32 * 24, 3))
        self.assertEqual(crop.shape, (32 * 24, 3))
        self.assertIsNot(crop, first[1])
        torch.testing.assert_close(intrinsics, torch.tensor(
            [[24., 0., 16.], [0., 24., 12.], [0., 0., 1.]]))

    def test_observation_grid_rejects_invalid_dimensions_and_sample_counts(self):
        for kwargs in ({'observation_ray_width': 0}, {'observation_ray_height': 13},
                       {'observation_ray_width': True}, {'observation_ray_height': 12.},
                       {'observation_min_depth_pixels': 0}, {'observation_min_depth_pixels': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.module.InitialVisionCfg(**kwargs)

    def test_inactive_body_does_not_occlude_and_active_distractor_is_distinct(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        enabled[:, 1] = -1
        first = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertEqual(float(first[4][0, 0]), 1.)
        enabled[:, 1] = 2
        second = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertEqual(int(second[3][0, 0]), 0)
        self.assertTrue((second[1] == 2).any())

    def test_batch_equals_individual_and_rigid_transform_does_not_change_visibility(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture(batch=2)
        poses[1, 1, 0] = .4
        together = sensor._trace(poses, enabled, eyes, rotation, roles)
        for row in range(2):
            alone = sensor._trace(poses[row:row+1], enabled[row:row+1], eyes[row:row+1], rotation[row:row+1], roles[row:row+1])
            for actual, expected in zip(together, alone):
                torch.testing.assert_close(actual[row:row+1], expected)
        q = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
        shift = torch.tensor([1., -.5, 2.])
        transformed = poses.clone()
        transformed[..., :3] = obs.quat_apply(q.expand(2, 3, -1), poses[..., :3]) + shift
        transformed[..., 3:] = q
        rotated = sensor._trace(transformed, enabled, eyes + shift, q.expand(2, -1), roles)
        torch.testing.assert_close(rotated[1], together[1])
        torch.testing.assert_close(rotated[2], together[2])

    def test_partly_off_frame_role_is_rejected_using_full_silhouette(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        poses[:, 1, 0] = .5
        result = sensor._trace(poses, enabled, eyes, rotation, roles)
        self.assertGreater(int(result[2][0, 1]), int(result[3][0, 1]))
        self.assertFalse(result[-1].any())

    def test_off_frame_silhouettes_follow_role_ids_not_body_order(self):
        sensor, poses, enabled, eyes, rotation, roles = self.fixture()
        poses[:, 1, 0] = .5
        expected = sensor._trace(poses, enabled, eyes, rotation, roles)
        # Move the role meshes away from slots 0/1; the last slot is a role.
        permutation = [2, 0, 1]
        active_ids = torch.where(enabled >= 0, enabled + 1, -1)
        actual = sensor._trace(poses[:, permutation], active_ids,
                               eyes, rotation, roles + 1)
        for index in (0, 2, 3, 4, 5):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        remapped = torch.where(expected[1] >= 0, expected[1] + 1, expected[1])
        torch.testing.assert_close(actual[1], remapped, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Visible-point reductions require CUDA.')
    def test_gpu_visible_fit_matches_cpu_with_background_and_hidden_objects(self):
        generator = torch.Generator().manual_seed(882)
        points = torch.rand(3, 19200, 3, generator=generator) * torch.tensor([.18, .06, .2])
        labels = torch.randint(-1, 5, (3, 19200), generator=generator, dtype=torch.int32)
        labels[:, :16000] = -1  # Large background bucket used to dominate atomics.
        points[:, :16000] = float('nan')  # Background must never contribute to an OBB.
        # The CPU oracle's background NaNs stay in its extra background bucket.
        expected = self.module.fit_visible_obbs(points, labels, 6, 16)
        actual = self.module.fit_visible_obbs(points.cuda(), labels.cuda(), 6, 16)
        torch.testing.assert_close(actual[0].cpu(), expected[0], atol=2.e-5, rtol=2.e-4)
        torch.testing.assert_close(actual[1].cpu(), expected[1])
        self.assertFalse(actual[1][:, 5].any())

    def test_projected_bounds_preserve_ray_hits_including_boxes_crossing_camera_plane(self):
        sensor, poses, active, eyes, rotation, roles = self.fixture(batch=4)
        poses[1, 1, 0] = .5
        poses[2, 1, 0] = -.5
        poses[3, 0, 2] = -.1
        actual = sensor._trace(poses, active, eyes, rotation, roles)
        # Deliberately loose bounds disable the coarse rejection. The same
        # actual triangle BVHs provide an independent uncropped query oracle.
        sensor.mesh_bounds = self.wp.array(np.array([[[-1.e4]*3, [1.e4]*3]]*3, np.float32),
                                           dtype=self.wp.vec3, device='cpu')
        expected = sensor._trace(poses, active, eyes, rotation, roles)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_visible_fit_uses_only_hits_and_leaves_hidden_slots_zero(self):
        points = torch.tensor([[[0., 0., .1], [.1, 0., .1], [0., .2, .3], [.1, .2, .3], [100., 100., 100.]]])
        labels = torch.tensor([[0, 0, 0, 0, -1]])
        geometry, valid = self.module.fit_visible_obbs(points, labels, 3, 3)
        self.assertEqual(valid.tolist(), [[True, False, False]])
        self.assertEqual(geometry[:, 1:].count_nonzero(), 0)
        self.assertTrue(torch.isfinite(geometry).all())
        torch.testing.assert_close(geometry[0, 0, :3], torch.tensor([.05, .1, .2]))
        self.assertLess(float(geometry[0, 0, 7:].max()), .3)

    def test_visible_fit_batch_and_translation_parity_without_host_scalar_reads(self):
        generator = torch.Generator().manual_seed(91)
        points = torch.rand(3, 100, 3, generator=generator)
        labels = torch.randint(-1, 4, (3, 100), generator=generator)
        def reject(*args, **kwargs):
            raise AssertionError('Visible-surface fitting synchronized with the host')
        with patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'tolist', reject), \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'cpu', reject):
            together = self.module.fit_visible_obbs(points, labels, 4, 4)
        for row in range(3):
            single = self.module.fit_visible_obbs(points[row:row+1], labels[row:row+1], 4, 4)
            torch.testing.assert_close(together[0][row:row+1], single[0])
        shifted = self.module.fit_visible_obbs(points + .5, labels, 4, 4)
        torch.testing.assert_close(shifted[0][..., :3], together[0][..., :3] + .5)
        torch.testing.assert_close(shifted[0][..., 3:], together[0][..., 3:], atol=2.e-5, rtol=1.e-4)

    def test_headless_task_has_no_camera_or_reference_resources(self):
        root = Path(__file__).resolve().parents[1]
        cfg = (root / 'contact_manipulation/config/ur5e/ur5e_env_cfg.py').read_text()
        self.assertNotIn('make_initial_camera_cfg', cfg)
        self.assertNotIn('VisionReferencesCfg', cfg)
        self.assertNotIn('sim.render(', (root / 'contact_manipulation/mdp/sensors.py').read_text())
        for kwargs in ({'ray_width': 0}, {'ray_height': 13}, {'ray_batch_size': True},
                       {'ray_candidates_per_env': 0}, {'ray_candidates_per_env': True}):
            with self.assertRaises(ValueError):
                self.module.InitialVisionCfg(**kwargs)

    def test_viewpoints_are_seeded_bounded_and_independent_of_request_order(self):
        sensor, *_ = self.fixture()
        sensor.cfg.max_camera_attempts = 7
        sensor.cfg.roll_degrees = (0., 0.)
        sensor.cfg.aim_y_jitter = sensor.cfg.aim_z_jitter = 0.
        requests = [dict(seed=seed, workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.)
                    for seed in (19, 923, (1 << 40) + 19)]
        positions = torch.tensor([[[-.8, .1, 1.1], [-.7, .05, 1.1]]]).repeat(3, 1, 1)
        eyes, rotations = sensor._viewpoints(requests, positions)
        self.assertTrue(((eyes[..., 0] >= .05) & (eyes[..., 0] <= .45)).all())
        self.assertTrue(((eyes[..., 1] >= -.4) & (eyes[..., 1] <= .4)).all())
        self.assertTrue(((eyes[..., 2] >= 1.18) & (eyes[..., 2] <= 1.48)).all())
        torch.testing.assert_close(rotations.norm(dim=-1), torch.ones(3, 7))
        target = torch.tensor([-.75, .075, 1.1])
        forward = (target - eyes) / (target - eyes).norm(dim=-1, keepdim=True)
        torch.testing.assert_close(obs.quat_apply(rotations, torch.tensor([0., 0., 1.]).expand_as(eyes)), forward)
        self.assertFalse(torch.equal(eyes[0], eyes[2]))
        for row in (2, 0, 1):
            one = sensor._viewpoints([requests[row]], positions[row:row+1])
            torch.testing.assert_close(one[0][0], eyes[row], atol=0, rtol=0)
            torch.testing.assert_close(one[1][0], rotations[row], atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA visibility parity requires a GPU.')
    def test_cuda_ray_and_viewpoint_results_match_cpu(self):
        expected, actual = self.fixture(batch=2), self.fixture(batch=2, device='cuda')
        for fixture in (expected, actual):
            fixture[1][1, 1, 0] = .4
        reference = expected[0]._trace(*expected[1:])
        result = actual[0]._trace(*actual[1:])
        for first, second in zip(reference, result):
            torch.testing.assert_close(first, second.cpu())
        requests = [dict(seed=seed, workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.) for seed in (93, 1823)]
        positions = torch.tensor([[[-.8, .1, 1.1], [-.7, .05, 1.1]]]).repeat(2, 1, 1)
        reference = expected[0]._viewpoints(requests, positions)
        def reject(*args, **kwargs):
            raise AssertionError('GPU viewpoint generation read a tensor on the host')
        with patch.object(torch.Tensor, 'item', reject), patch.object(torch.Tensor, 'tolist', reject), \
             patch.object(torch.Tensor, '__bool__', reject), patch.object(torch.Tensor, 'cpu', reject):
            result = actual[0]._viewpoints(requests, positions.cuda())
        for first, second in zip(reference, result):
            torch.testing.assert_close(first, second.cpu(), atol=1.e-6, rtol=1.e-6)

    def test_queue_selects_first_valid_view_refills_and_honors_attempt_limit(self):
        sensor, poses, *_ = self.fixture(batch=5)
        sensor.capacity = 2
        sensor.cfg.max_camera_attempts = 6  # Last group is smaller than four.
        robot = SimpleNamespace(root_pos_w=torch.zeros(5, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1))
        class Scene(dict):
            env_origins = torch.zeros(5, 3)
        scene = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                      robot=SimpleNamespace(data=robot))
        sensor.env = SimpleNamespace(scene=scene, device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        requests = [dict(env_id=row, active_ids=(0, 1), target_id=0, blocker_id=1, seed=row,
                         workspace_bounds=(-1., -.5, -.4, .4), shelf_top=1.) for row in range(5)]
        required = [0, 5, 2, 6, 4]  # Row 3 must fail: candidate 6 is beyond the search budget.
        calls = []
        def viewpoints(requests, role_positions):
            eyes = torch.zeros(len(requests), 6, 3)
            eyes[..., 0] = torch.tensor([r['env_id'] for r in requests])[:, None]
            eyes[..., 1] = torch.arange(6)
            rotations = torch.zeros(len(requests), 6, 4)
            rotations[..., 0] = 1.
            return eyes, rotations
        def trace(poses, active, eyes, rotations, roles):
            rows, attempts = eyes[:, 0].long(), eyes[:, 1].long()
            calls.append(list(zip(rows.tolist(), attempts.tolist())))
            accepted = attempts >= torch.tensor(required)[rows]
            depth = torch.ones(len(rows), 80*60)
            labels = torch.zeros_like(depth, dtype=torch.int32)
            labels[:, 80*30:] = 1
            visible = torch.full((len(rows), 2), 80*30, dtype=torch.int64)
            return depth, labels, visible, visible, torch.ones(len(rows), 2), accepted
        with patch.object(sensor, '_viewpoints', viewpoints), patch.object(sensor, '_trace', trace):
            results = sensor.capture_batch(requests, retain_images=True)
            self.assertEqual([None if r is None else r.attempts for r in results], [1, 6, 3, None, 5])
            self.assertEqual({row for row, attempt in calls[1]}, {1, 2})
            self.assertTrue(all(len(call) <= 8 for call in calls))
            reference = results
            sensor.cfg.ray_candidates_per_env = 1
            sensor.capacity = 3
            order = [4, 2, 1, 0, 3]
            results = sensor.capture_batch([requests[row] for row in order], retain_images=True)
        for result, row in zip(results, order):
            if row == 3:
                self.assertIsNone(result)
                continue
            self.assertEqual(result.attempts, reference[row].attempts)
            torch.testing.assert_close(result.geometry_b, reference[row].geometry_b)
            torch.testing.assert_close(result.camera_pose_w, reference[row].camera_pose_w)
            self.assertEqual(result.pixel_counts, ((2400, 2400), (2400, 2400)))
            self.assertEqual(result.depth.shape, (60, 80, 1))
            self.assertEqual(result.role_masks.shape, (60, 80, 2))
        self.assertEqual(sensor.capture_batch([]), [])

    def test_target_partial_visibility_upper_bound_uses_actual_mesh_silhouette(self):
        sensor, poses, active, eyes, rotations, roles = self.fixture(batch=3)
        eyes[:, 2] = -.5
        poses[:, 1, 0] = torch.tensor([.2, .3, .5])
        result = sensor._trace(poses, active, eyes, rotations, roles)
        torch.testing.assert_close(result[-1], torch.tensor([False, True, False]))
        self.assertLess(float(result[4][0, 0]), .4)
        self.assertAlmostEqual(float(result[4][1, 0]), .5)
        self.assertEqual(float(result[4][2, 0]), 1.)
        self.assertTrue((result[4][:, 1] == 1.).all())
        for kwargs in ({'max_target_visible_fraction': 0.},
                       {'min_target_visible_fraction': .9, 'max_target_visible_fraction': .4}):
            with self.assertRaises(ValueError):
                self.module.InitialVisionCfg(**kwargs)

    def test_explicit_world_viewpoints_evaluate_once_and_compact_large_pool_ids(self):
        sensor, source, _, eyes, rotations, _ = self.fixture(batch=2)
        sensor.object_count = 30
        sensor.mesh_ids = self.wp.array([sensor.meshes[0].id] * 30, dtype=self.wp.uint64, device='cpu')
        bounds = sensor.mesh_bounds.numpy()[0]
        sensor.mesh_bounds = self.wp.array(np.stack([bounds] * 30), dtype=self.wp.vec3, device='cpu')
        poses = torch.zeros(2, 30, 7)
        poses[..., 3] = 1.
        poses[:, [7, 18, 29]] = source
        poses[:, 18, 0] = .3
        eyes[:, 2] = -.5
        origins = torch.tensor([[0., 0., 0.], [10., -20., 4.]])
        poses[..., :3] += origins[:, None]
        camera = torch.cat((eyes + origins, rotations), -1)
        robot = SimpleNamespace(root_pos_w=origins, root_quat_w=rotations)
        class Scene(dict):
            pass
        scene = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                      robot=SimpleNamespace(data=robot))
        scene.env_origins = origins
        sensor.env = SimpleNamespace(scene=scene, device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        requests = [dict(env_id=row, active_ids=(18, 7), target_id=7, blocker_id=18) for row in range(2)]
        with patch.object(sensor, '_viewpoints', side_effect=AssertionError('Explicit view must not resample')), \
             patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            snapshots = sensor.capture_at_viewpoints(requests, camera, retain_images=True)
        self.assertEqual(trace.call_count, 1)
        for row, result in enumerate(snapshots):
            self.assertIsNotNone(result)
            self.assertEqual(result.attempts, 1)
            self.assertEqual(result.observed_ids.tolist(), [7, 18, -1])
            self.assertEqual(result.geometry_valid.tolist(), [True, True, False])
            self.assertEqual(result.geometry_b[2].count_nonzero(), 0)
            self.assertTrue(sensor.last_diagnostics[row]['accepted'])
            self.assertEqual(sensor.last_diagnostics[row]['failure_counts'], {})
            torch.testing.assert_close(result.camera_pose_w, camera[row])
        # PCA's axis sign can flip after float32 world-origin subtraction;
        # compare the represented OBB corner sets, rather than yaw signs.
        sign = torch.tensor(list(product((-1., 1.), repeat=3)))
        corners = []
        for snapshot in snapshots:
            geometry = snapshot.geometry_b[:2]
            corners.append(obs.quat_apply(geometry[:, None, 3:7],
                sign[None] * geometry[:, None, 7:] * .5) + geometry[:, None, :3])
        self.assertLess(float(torch.cdist(*corners).amin(-1).amax()), 2.e-6)
        # A fully exposed Target is rejected once, with the precise upper-limit cause.
        poses[:, 18, 0] = .5 + origins[:, 0]
        snapshots = sensor.capture_batch(requests, camera_poses=camera)
        self.assertEqual(snapshots, [None, None])
        for detail in sensor.last_diagnostics:
            self.assertEqual(detail['attempts'], 1)
            self.assertEqual(detail['failure_counts'], {'target_above_max': 1})
            self.assertEqual(detail['target_visible_fraction'], 1.)

    def test_compact_pool_labels_keep_object_zero_and_ignore_padding(self):
        sensor, *_ = self.fixture()
        sensor.object_count = 30
        active = torch.tensor([[0, 29, -1], [7, -1, -1]], dtype=torch.int32)
        labels = torch.tensor([[29, 0, -1, 3], [7, -1, 0, 29]], dtype=torch.int32)
        compact, native = sensor._compact_labels(labels, active, 4)
        self.assertEqual(compact.tolist(), [[1, 0, -1, -1], [0, -1, -1, -1]])
        self.assertEqual(native.tolist(), [[0, 29, -1, -1], [7, -1, -1, -1]])

    def test_fixed_view_domain_is_bounded_looks_at_roles_and_does_not_clamp_lateral(self):
        sensor, *_ = self.fixture(batch=2)
        requests = [dict(workspace_bounds=(-.89, -.51, -.455, .455), shelf_top=1.05)] * 2
        positions = torch.tensor([[[-.8, .4, 1.1], [-.6, .43, 1.1]],
                                  [[-.8, -.4, 1.1], [-.6, -.43, 1.1]]])
        eyes, rotations = sensor._domain_viewpoints(requests, positions)
        self.assertEqual(eyes.shape, (2, 36, 3))
        self.assertEqual(rotations.shape, (2, 36, 4))
        torch.testing.assert_close(eyes[..., 0].amin(1), torch.full((2,), .04))
        torch.testing.assert_close(eyes[..., 0].amax(1), torch.full((2,), .44))
        self.assertGreater(float(eyes[0, :, 1].max()), .455)
        self.assertLess(float(eyes[1, :, 1].min()), -.455)
        torch.testing.assert_close(eyes[..., 2].amin(1), torch.full((2,), 1.23))
        torch.testing.assert_close(eyes[..., 2].amax(1), torch.full((2,), 1.53))
        aim = positions.mean(1)
        aim[:, 2] = 1.15
        forward = torch.nn.functional.normalize(aim[:, None] - eyes, dim=-1)
        torch.testing.assert_close(obs.quat_apply(rotations, torch.tensor([0., 0., 1.]).expand_as(eyes)),
                                   forward, atol=2.e-6, rtol=1.e-5)
        for shape in ((2, 0, 2), (2, True, 2), (2, 9), (2., 9, 2)):
            with self.assertRaises(ValueError):
                self.module.InitialVisionCfg(view_domain_shape=shape)

    def test_inverse_translation_cache_solves_continuous_parameters_without_pose_candidates(self):
        import json
        sensor, poses, active, eyes, rotations, roles = self.fixture()
        eyes[:, 2] = -.5
        camera = torch.cat((eyes, rotations), -1)[0]
        with patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            cache = sensor.precompute_translation_domain(object_poses=poses[0], target_id=0, blocker_id=1,
                camera_pose=camera, translation=(1., 0., 0.), parameter_bounds=(-.7, .7), triangle_chunk=1)
        self.assertEqual(trace.call_count, 1)  # Isolated Target only, not Blocker pose trials.
        self.assertEqual(cache['candidate_pose_queries'], 0)
        self.assertEqual(cache['isolated_target_queries'], 1)
        self.assertGreater(cache['structural_depth_gap'], .59)
        self.assertEqual(len(cache['fingerprint']), 64)
        self.assertEqual(cache['source']['min_silhouette_pixels'], sensor.cfg.min_silhouette_pixels)
        self.assertEqual(cache['source']['min_depth_pixels'], sensor.cfg.min_depth_pixels)
        self.assertEqual(cache['source']['min_blocker_visible_fraction'], sensor.cfg.min_blocker_visible_fraction)
        self.assertEqual(cache['source']['projection_depth_bounds'], (.02, 2.5))
        self.assertEqual(cache['source']['parameter_margin'], 2.e-6)
        self.assertEqual(cache['source']['optical_assumptions']['occluders'], 'Blocker mesh only')
        np.testing.assert_allclose(cache['visual_support']['target']['world_xyz_bounds'],
                                   [[-.2937, -.2937, 1.4], [.2937, .2937, 1.4]], atol=1.e-7)
        self.assertLess(cache['visual_support']['blocker']['swept_world_xyz_bounds'][0][0], -.3)
        # Every acceptance limit participates in persistent startup cache identity.
        sensor.cfg.min_depth_pixels += 1
        changed = sensor.precompute_translation_domain(object_poses=poses[0], target_id=0, blocker_id=1,
            camera_pose=camera, translation=(1., 0., 0.), parameter_bounds=(-.7, .7), triangle_chunk=1)
        self.assertNotEqual(changed['fingerprint'], cache['fingerprint'])
        sensor.cfg.min_depth_pixels -= 1
        json.dumps(cache)  # Startup data contains no tensors or NumPy scalar objects.
        intervals = np.array(cache['valid_intervals'])
        lengths = intervals[:, 1] - intervals[:, 0]
        cumulative = lengths.cumsum()
        mass = (np.arange(32) + .37) / 32 * cumulative[-1]
        chosen = np.searchsorted(cumulative, mass)
        values = intervals[chosen, 0] + mass - np.r_[0., cumulative[:-1]][chosen]
        test_poses = poses.repeat(32, 1, 1)
        test_poses[:, 1, 0] += torch.tensor(values, dtype=test_poses.dtype)
        result = sensor._trace(test_poses, active.repeat(32, 1), eyes.repeat(32, 1),
                               rotations.repeat(32, 1), roles.repeat(32, 1))
        denominator = cache['target_pixel_count']
        events = np.array(cache['target_visibility_events'])
        occluded = np.array(cache['target_occluded_counts'])
        predicted = 1. - occluded[np.searchsorted(events, values, side='right') - 1] / denominator
        np.testing.assert_allclose(result[4][:, 0].numpy(), predicted, atol=1.e-7)
        self.assertTrue(result[-1].all())
        self.assertTrue((result[4][:, 1] == 1.).all())
        self.assertTrue((result[2][:, 1] >= cache['minimum_blocker_pixels']).all())

    def test_inverse_translation_rejects_uncertified_depth_and_too_small_blocker(self):
        sensor, poses, _, eyes, rotations, _ = self.fixture()
        eyes[:, 2] = -.5
        camera = torch.cat((eyes, rotations), -1)[0]
        behind = poses.clone()
        behind[:, 1, 2] = 1.6
        with patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            with self.assertRaisesRegex(self.module.InitialVisionRejected, 'camera-depth intervals overlap'):
                sensor.precompute_translation_domain(object_poses=behind[0], target_id=0, blocker_id=1,
                    camera_pose=camera, translation=(1., 0., 0.), parameter_bounds=(-.3, .3))
        self.assertEqual(trace.call_count, 0)
        small = np.array([[-.02, -.02, 0.], [.02, -.02, 0.], [.02, .02, 0.], [-.02, .02, 0.]], np.float32)
        mesh = self.wp.Mesh(points=self.wp.array(small, dtype=self.wp.vec3, device='cpu'),
            indices=self.wp.array([0, 1, 2, 0, 2, 3], dtype=self.wp.int32, device='cpu'))
        sensor.meshes.append(mesh)
        sensor._mesh_id_by_object = (sensor.meshes[0].id, mesh.id, sensor.meshes[0].id)
        sensor.mesh_ids = self.wp.array(sensor._mesh_id_by_object, dtype=self.wp.uint64, device='cpu')
        with self.assertRaisesRegex(self.module.InitialVisionRejected, 'Empty continuous domain'):
            sensor.precompute_translation_domain(object_poses=poses[0], target_id=0, blocker_id=1,
                camera_pose=camera, translation=(1., 0., 0.), parameter_bounds=(-.3, .3))

    def test_inverse_interval_union_counts_pixels_once_for_nested_and_touching_ranges(self):
        rng = np.random.default_rng(37)
        rows = np.repeat(np.arange(20), 30)
        lo = rng.uniform(-.8, .4, len(rows))
        hi = lo + rng.uniform(.001, .5, len(rows))
        intervals = np.stack((rows, lo, hi), -1)
        intervals = np.r_[intervals, [[20., -.7, -.2], [20., -.2, .5], [20., -.4, -.3]]]
        events, counts = self.module.InitialVisionSensor._translation_pixel_counts(intervals, 21, (-1., 1.))
        values = np.linspace(-.9997, .9993, 1000)
        expected = np.zeros(len(values), dtype=np.int64)
        for row in range(21):
            pixel = intervals[intervals[:, 0] == row, 1:]
            expected += ((values[:, None] >= pixel[None, :, 0]) & (values[:, None] < pixel[None, :, 1])).any(-1)
        np.testing.assert_array_equal(counts[np.searchsorted(events, values, side='right') - 1], expected)

    def test_inverse_persistent_cache_skips_ray_and_affine_solve_and_rejects_corruption(self):
        import json
        import tempfile
        sensor, poses, _, eyes, rotations, _ = self.fixture()
        eyes[:, 2] = -.5
        kwargs = dict(object_poses=poses[0], target_id=0, blocker_id=1, camera_pose=torch.cat((eyes, rotations), -1)[0],
            translation=(1., 0., 0.), parameter_bounds=(-.7, .7), triangle_chunk=1)
        with tempfile.TemporaryDirectory() as directory:
            initial = sensor.precompute_translation_domain(**kwargs, cache_directory=directory)
            self.assertFalse(initial['diagnostics']['persistent_cache_hit'])
            with patch.object(sensor, '_trace', side_effect=AssertionError('Cache must skip ray queries')), \
                 patch.object(sensor, '_translation_hit_intervals', side_effect=AssertionError('Cache must skip solving')):
                cached = sensor.precompute_translation_domain(**kwargs, cache_directory=directory)
            self.assertTrue(cached['diagnostics']['persistent_cache_hit'])
            self.assertEqual(cached['fingerprint'], initial['fingerprint'])
            np.testing.assert_array_equal(cached['valid_intervals'], initial['valid_intervals'])
            cache_path = Path(directory) / f"{initial['fingerprint']}.json"
            payload = json.loads(cache_path.read_text())
            payload['domain']['target_pixel_count'] += 1  # Still valid JSON, but payload digest is stale.
            cache_path.write_text(json.dumps(payload))
            with patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
                repaired = sensor.precompute_translation_domain(**kwargs, cache_directory=directory)
            self.assertEqual(trace.call_count, 1)
            self.assertFalse(repaired['diagnostics']['persistent_cache_hit'])
            self.assertEqual(repaired['target_pixel_count'], initial['target_pixel_count'])
            with patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
                changed = sensor.precompute_translation_domain(**kwargs, cache_directory=directory, parameter_margin=1.e-5)
            self.assertEqual(trace.call_count, 1)
            self.assertNotEqual(changed['fingerprint'], initial['fingerprint'])
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 3)
            self.assertEqual(len(list(Path(directory).glob('blocker_*.json'))), 1)
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_blocker_curve_reuses_exact_inputs_across_target_variants_in_memory_and_on_disk(self):
        import tempfile
        sensor, poses, _, eyes, rotations, _ = self.fixture()
        eyes[:, 2] = -.5
        kwargs = dict(target_id=0, blocker_id=1, camera_pose=torch.cat((eyes, rotations), -1)[0],
            translation=(1., 0., 0.), parameter_bounds=(-.7, .7), triangle_chunk=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sensor, '_trace', wraps=sensor._trace) as trace, \
                 patch.object(sensor, '_translation_hit_intervals', wraps=sensor._translation_hit_intervals) as affine:
                first = sensor.precompute_translation_domain(object_poses=poses[0], **kwargs, cache_directory=directory)
                shifted = poses[0].clone()
                shifted[0, 0] += .03
                second = sensor.precompute_translation_domain(object_poses=shifted, **kwargs, cache_directory=directory)
            self.assertEqual(trace.call_count, 2)  # Each Target gets its own isolated mesh query.
            self.assertEqual(affine.call_count, 3)  # Two Target solves, one shared Blocker solve.
            self.assertEqual(first['diagnostics']['blocker_subcache_hit'], 'miss')
            self.assertEqual(second['diagnostics']['blocker_subcache_hit'], 'memory')
            self.assertNotEqual(first['fingerprint'], second['fingerprint'])
            self.assertEqual(first['diagnostics']['blocker_subcache_fingerprint'], second['diagnostics']['blocker_subcache_fingerprint'])
            self.assertEqual(first['blocker_silhouette_events'], second['blocker_silhouette_events'])
            self.assertEqual(first['blocker_silhouette_counts'], second['blocker_silhouette_counts'])
            for cached in sensor._blocker_translation_cache.values():
                self.assertFalse(cached['events'].flags.writeable)
                self.assertFalse(cached['pixels'].flags.writeable)
            sensor._blocker_translation_cache.clear()
            shifted[0, 0] += .02
            with patch.object(sensor, '_trace', wraps=sensor._trace) as trace, \
                 patch.object(sensor, '_translation_hit_intervals', wraps=sensor._translation_hit_intervals) as affine:
                third = sensor.precompute_translation_domain(object_poses=shifted, **kwargs, cache_directory=directory)
            self.assertEqual(trace.call_count, 1)
            self.assertEqual(affine.call_count, 1)
            self.assertEqual(third['diagnostics']['blocker_subcache_hit'], 'disk')
            self.assertEqual(len(list(Path(directory).glob('blocker_*.json'))), 1)

    def test_visual_footprints_apply_upright_rotation_and_share_duplicate_mesh_downloads(self):
        sensor, *_ = self.fixture()
        half_angle = math.pi / 8
        upright = torch.tensor([[1., 0., 0., 0.], [math.cos(half_angle), 0., 0., math.sin(half_angle)], [1., 0., 0., 0.]])
        with patch.object(self.wp, 'to_torch', wraps=self.wp.to_torch) as copied:
            footprints = sensor.visual_footprints(upright)
        self.assertEqual(copied.call_count, 1)
        self.assertIs(footprints[0], footprints[2])
        self.assertEqual(len(footprints[0]), 4)
        self.assertAlmostEqual(float(np.abs(footprints[0]).max()), .2937, places=6)
        self.assertAlmostEqual(float(np.abs(footprints[1]).max()), math.sqrt(2) * .2937, places=6)

    def test_observation_only_capture_returns_hits_without_visibility_rejection_or_silhouettes(self):
        sensor, poses, active, eyes, rotations, roles = self.fixture()
        expected = sensor._trace(poses, active, eyes, rotations, roles)
        robot = SimpleNamespace(root_pos_w=torch.zeros(1, 3), root_quat_w=rotations)
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
        scene = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                      robot=SimpleNamespace(data=robot))
        sensor.env = SimpleNamespace(scene=scene, device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        requests = [dict(env_id=0, active_ids=(0, 1), target_id=0, blocker_id=1)]
        camera = torch.cat((eyes, rotations), -1)
        with patch.object(sensor, '_viewpoints', side_effect=AssertionError('No view candidates')), \
             patch.object(sensor, 'capture_batch', side_effect=AssertionError('No acceptance selection')), \
             patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            result = sensor.observe_at_viewpoint(requests, camera, retain_images=True)[0]
        self.assertEqual(trace.call_count, 1)
        self.assertTrue(trace.call_args.kwargs['observation_only'])
        self.assertIsNone(result.visible_fractions)
        self.assertIsNone(result.target_visible_fraction)
        self.assertIsNone(result.blocker_visible_fraction)
        self.assertIsNone(result.pixel_counts)
        self.assertEqual(result.geometry_valid.tolist(), [False, True, False])
        torch.testing.assert_close(result.depth.flatten(), expected[0][0])
        torch.testing.assert_close(result.role_masks.reshape(-1, 2), expected[1][0, :, None] == roles[0])

    def test_view_domain_evaluates_all_candidates_once_and_fits_only_selected_hits(self):
        sensor, poses, _, eyes, rotations, _ = self.fixture(batch=2)
        sensor.capacity = 4
        sensor.cfg.view_domain_shape = (1, 3, 1)
        poses[:, 1, 0] = .3
        origin = torch.tensor([[0., 0., 0.], [10., -20., 4.]])
        poses[..., :3] += origin[:, None]
        robot = SimpleNamespace(root_pos_w=origin, root_quat_w=rotations)
        class Scene(dict):
            pass
        scene = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                      robot=SimpleNamespace(data=robot))
        scene.env_origins = origin
        sensor.env = SimpleNamespace(scene=scene, device='cpu', cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=3)))
        # Fixed test domain exposes different actual mesh occlusion fractions.
        domain_eyes = eyes[:, None].repeat(1, 3, 1)
        domain_eyes[..., 2] = -.5
        domain_eyes[..., 0] = torch.tensor([-.2, 0., .2])
        domain_rotations = rotations[:, None].repeat(1, 3, 1)
        requests = [dict(env_id=row, active_ids=(0, 1), target_id=0, blocker_id=1) for row in range(2)]
        roots = torch.cat((origin, rotations), -1)
        original_fit = self.module.fit_visible_obbs
        with patch.object(sensor, '_viewpoints', side_effect=AssertionError('No random view search')), \
             patch.object(sensor, '_domain_viewpoints', return_value=(domain_eyes, domain_rotations)), \
             patch.object(sensor, '_trace', wraps=sensor._trace) as trace, \
             patch.object(self.module, 'fit_visible_obbs', wraps=original_fit) as fit:
            snapshots = sensor.capture_view_domain(requests, object_poses=poses, root_poses=roots)
        self.assertEqual(trace.call_count, 2)  # Six views, fixed memory chunks 4+2.
        self.assertEqual([call.args[0].shape[0] for call in trace.call_args_list], [4, 2])
        self.assertEqual(fit.call_count, 1)  # One final batched fit, no second trace.
        for row, snapshot in enumerate(snapshots):
            self.assertIsNotNone(snapshot)
            diagnostic = sensor.last_diagnostics[row]
            self.assertEqual(diagnostic['view_count'], 3)
            self.assertGreater(diagnostic['valid_candidate_count'], 0)
            self.assertGreaterEqual(snapshot.target_visible_fraction, .4)
            self.assertLessEqual(snapshot.target_visible_fraction, .9)
            self.assertEqual(snapshot.blocker_visible_fraction, 1.)
            self.assertEqual(snapshot.observed_ids.tolist(), [0, 1, -1])
            chosen = diagnostic['selected_candidate_index']
            torch.testing.assert_close(snapshot.camera_pose_w[:3], domain_eyes[row, chosen] + origin[row])
        # Appending the previously selected world view as the preferred first
        # member preserves its exact pixels and wins an otherwise stable tie.
        preferred = torch.stack([snapshot.camera_pose_w for snapshot in snapshots])
        with patch.object(sensor, '_domain_viewpoints', return_value=(domain_eyes, domain_rotations)), \
             patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            preferred_snapshots = sensor.capture_view_domain(requests, object_poses=poses, root_poses=roots,
                                                             preferred_camera_poses=preferred)
        self.assertEqual(trace.call_count, 2)  # Eight fixed views in chunks 4+4.
        for snapshot, diagnostic, pose in zip(preferred_snapshots, sensor.last_diagnostics, preferred):
            self.assertEqual(diagnostic['view_count'], 4)
            self.assertEqual(diagnostic['selected_candidate_index'], 0)
            torch.testing.assert_close(snapshot.camera_pose_w, pose)
        # All three views failing still executes exactly the same 4+2 chunks.
        poses[:, 1, 0] = origin[:, 0]
        with patch.object(sensor, '_viewpoints', side_effect=AssertionError('No retry after domain failure')), \
             patch.object(sensor, '_domain_viewpoints', return_value=(domain_eyes, domain_rotations)), \
             patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            snapshots = sensor.capture_view_domain(requests, object_poses=poses, root_poses=roots)
        self.assertEqual(trace.call_count, 2)
        self.assertEqual(snapshots, [None, None])
        for diagnostic in sensor.last_diagnostics:
            self.assertEqual(diagnostic['valid_candidate_count'], 0)
            self.assertEqual(diagnostic['view_count'], 3)
            self.assertIsNone(diagnostic['selected_candidate_index'])
            self.assertEqual(diagnostic['failure_counts']['target_below_min'], 3)

class NativeGeometrySharingTests(unittest.TestCase):
    """Run with Kit's USD modules available; ordinary CPU discovery skips these."""

    @classmethod
    def setUpClass(cls):
        try:
            from pxr import Gf, Usd, Sdf, UsdGeom, UsdPhysics, UsdShade
        except ImportError:
            raise unittest.SkipTest('Native USD geometry tests require an initialized Isaac Sim app.')
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        names = ('_deinstance_subtree', 'share_rigid_body_geometry', '_gf_column', '_rotation_part',
                 '_rotation_wxyz', '_object_support_vertices', '_collision_footprint_vertices',
                 '_range_bounds', '_shelf_rules')
        nodes = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = dict(Usd=Usd, Sdf=Sdf, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics,
                         Gf=Gf, np=np, math=math, product=product, Bounds=tuple, Point3=tuple,
                         SpawnRules=task.SpawnRules, replace=dataclasses.replace)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        visual_path = path.parents[2] / 'mdp/sensors.py'
        visual_node = next(node for node in ast.parse(visual_path.read_text()).body
                           if isinstance(node, ast.FunctionDef) and node.name == '_visual_triangles')
        visual_node.body = [node for node in visual_node.body if not isinstance(node, ast.ImportFrom)]
        exec(compile(ast.Module(body=[visual_node], type_ignores=[]), str(visual_path), 'exec'), namespace)
        cls.share = staticmethod(namespace['share_rigid_body_geometry'])
        cls.support_vertices = staticmethod(namespace['_object_support_vertices'])
        cls.visual_triangles = staticmethod(namespace['_visual_triangles'])
        cls.shelf_rules = staticmethod(namespace['_shelf_rules'])
        cls.collision_hull = staticmethod(namespace['_collision_footprint_vertices'])
        cls.column = staticmethod(namespace['_gf_column'])
        cls.rotation_part = staticmethod(namespace['_rotation_part'])
        cls.Gf = Gf
        cls.Usd, cls.Sdf, cls.UsdGeom, cls.UsdPhysics, cls.UsdShade = Usd, Sdf, UsdGeom, UsdPhysics, UsdShade

    def pool_fixture(self, directory):
        from unittest.mock import Mock
        stage = self.Usd.Stage.CreateInMemory()
        self.UsdGeom.Xform.Define(stage, '/World/envs/env_0')
        names = ('object_a', 'object_b', 'object_c')
        for name in names:
            body = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/' + name).GetPrim()
            self.UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(False)
            mesh = self.UsdGeom.Mesh.Define(stage, body.GetPath().AppendChild('visual'))
            mesh.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            for collider_name, enabled in (('collider', True), ('authored_disabled', False)):
                collider = self.UsdGeom.Cube.Define(stage, body.GetPath().AppendChild(collider_name)).GetPrim()
                self.UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr(enabled)
        library_path = str(Path(directory) / 'pool.usdc')
        self.assertEqual(self.share(stage, '/World/envs/env_0', library_path), 3)
        for index in (1, 2):
            self.UsdGeom.Xform.Define(stage, f'/World/envs/env_{index}').GetPrim().GetInherits().AddInherit(
                '/World/envs/env_0')
        prim_paths = [f'/World/envs/env_{index}/{name}' for name in names for index in range(3)]

        class Native:
            def __init__(self, cfg):
                self.num_instances, self.num_objects, self.device = 3, 3, 'cpu'
                self._ALL_ENV_INDICES = torch.arange(3)
                self.root_physx_view = SimpleNamespace(prim_paths=prim_paths)
                self.data = SimpleNamespace(object_state_w=torch.arange(117).reshape(3, 3, 13).float())
                self._data = SimpleNamespace(_object_com_pose_w=SimpleNamespace(timestamp=19.))
                self.state_writes = []

            def write_object_state_to_sim(self, states, env_ids=None):
                self.state_writes.append(states.clone())
                self.data.object_state_w.copy_(states)

            def write_object_link_pose_to_sim(self, pose, env_ids=None, object_ids=None):
                self.pose_write = (pose, env_ids, object_ids)

        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/config/ur5e/scene.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == 'TaskObjectPool')
        namespace = dict(RigidObjectCollection=Native, torch=torch, Usd=self.Usd,
                         Sdf=self.Sdf, UsdPhysics=self.UsdPhysics, get_current_stage=lambda: stage)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        pool = namespace['TaskObjectPool'](None)
        pool.parked_mask = torch.zeros((3, 3), dtype=torch.bool)
        interface = SimpleNamespace(flush_changes=Mock())
        native_module = types.ModuleType('omni.physx')
        native_module.get_physx_simulation_interface = lambda: interface
        return pool, stage, names, library_path, interface, native_module, node

    def test_pool_shared_kinematic_switch_preserves_all_authored_collision_geometry(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            pool, stage, names, library, interface, module, node = self.pool_fixture(directory)
            on_disk = Path(library).read_bytes()
            library_layer = self.Sdf.Layer.Find(library)
            library_contents = library_layer.ExportToString()
            active = torch.tensor([[True, False, True]]).expand(3, -1)
            with patch.dict(sys.modules, {'omni.physx': module}):
                pool.set_active_objects(active, env_ids=torch.tensor([2, 0, 1]))
                for index in range(3):
                    for slot, name in enumerate(names):
                        body = stage.GetPrimAtPath(f'/World/envs/env_{index}/{name}')
                        self.assertTrue(body.IsInstance())
                        self.assertEqual(self.UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get(), slot == 1)
                        collider = stage.GetPrimAtPath(body.GetPath().AppendChild('collider'))
                        disabled = stage.GetPrimAtPath(body.GetPath().AppendChild('authored_disabled'))
                        self.assertTrue(self.UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())
                        self.assertFalse(self.UsdPhysics.CollisionAPI(disabled).GetCollisionEnabledAttr().Get())
                torch.testing.assert_close(pool.parked_mask, ~active)
                self.assertEqual(interface.flush_changes.call_count, 1)
                self.assertEqual(pool.state_writes, [])
                # Partial resets and repeated full resets only restore poses;
                # they must not flush native USD or copy flags back to CPU.
                with patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('redundant mask copy')):
                    pool.set_active_objects(active[[2, 0]], env_ids=torch.tensor([2, 0]))
                    pool.set_active_objects(active)
                    pool.set_active_objects(torch.empty(0, 3, dtype=torch.bool), env_ids=torch.empty(0, dtype=torch.long))
                self.assertEqual(interface.flush_changes.call_count, 1)
                pool.set_active_objects(torch.ones_like(active))
                self.assertEqual(interface.flush_changes.call_count, 2)
                self.assertFalse(pool.parked_mask.any())
                for index in range(3):
                    body = stage.GetPrimAtPath(f'/World/envs/env_{index}/object_b')
                    self.assertFalse(self.UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
                    self.assertTrue(self.UsdPhysics.CollisionAPI(stage.GetPrimAtPath(
                        body.GetPath().AppendChild('collider'))).GetCollisionEnabledAttr().Get())
                    self.assertFalse(self.UsdPhysics.CollisionAPI(stage.GetPrimAtPath(
                        body.GetPath().AppendChild('authored_disabled'))).GetCollisionEnabledAttr().Get())
            self.assertEqual(Path(library).read_bytes(), on_disk)
            self.assertEqual(library_layer.ExportToString(), library_contents)
            # Runtime activation cannot remove PhysX actors or add per-step work.
            calls = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
            self.assertTrue({'set_disable_simulations', 'get_disable_simulations', 'wake_up'}.isdisjoint(calls))
            self.assertNotIn('write_data_to_sim', type(pool).__dict__)
            self.assertNotIn('update', type(pool).__dict__)
            pose = torch.zeros(2, 3, 7)
            pool.write_object_link_pose_to_sim(pose, env_ids=torch.tensor([2, 0]))
            self.assertIs(pool.pose_write[0], pose)
            self.assertEqual(pool._data._object_com_pose_w.timestamp, -1.)

    def test_pool_rejects_nonshared_or_partial_activity_changes_before_usd_commit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            pool, stage, names, library, interface, module, node = self.pool_fixture(directory)
            active = torch.tensor([[True, False, True]]).expand(3, -1)
            with patch.dict(sys.modules, {'omni.physx': module}):
                with self.assertRaisesRegex(ValueError, 'full shared-scene reset'):
                    pool.set_active_objects(active[:1], env_ids=torch.tensor([0]))
                self.assertIsNone(pool._activation_attributes)
                self.assertEqual(interface.flush_changes.call_count, 0)
                pool.set_active_objects(active)
                before = stage.GetRootLayer().ExportToString()
                invalid_masks = ((~active[:1], torch.tensor([1])),
                                 (active.clone().index_fill(0, torch.tensor([1]), False), torch.arange(3)),
                                 (~active, torch.tensor([0, 0, 2])),
                                 (torch.zeros(3, 3), torch.arange(3)),
                                 (torch.zeros(3, 2, dtype=torch.bool), torch.arange(3)))
                for mask, ids in invalid_masks:
                    with self.subTest(mask=mask, ids=ids), self.assertRaises(ValueError):
                        pool.set_active_objects(mask, env_ids=ids)
                    self.assertEqual(stage.GetRootLayer().ExportToString(), before)
                    torch.testing.assert_close(pool.parked_mask, ~active)
                    self.assertEqual(interface.flush_changes.call_count, 1)

    def test_pool_failed_native_commit_restores_flags_state_and_cached_pattern(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            pool, stage, names, library, interface, module, node = self.pool_fixture(directory)
            active = torch.tensor([[True, False, True]]).expand(3, -1)
            with patch.dict(sys.modules, {'omni.physx': module}):
                pool.set_active_objects(active)
                before = pool.data.object_state_w.clone()
                pattern, mask = pool._active_pattern.clone(), pool.parked_mask.clone()
                error = RuntimeError('native USD flush failed')
                interface.flush_changes.side_effect = [error, None]
                with self.assertRaises(RuntimeError) as raised:
                    pool.set_active_objects(~active)
                self.assertIs(raised.exception, error)
                self.assertEqual(interface.flush_changes.call_count, 3)
                torch.testing.assert_close(pool._active_pattern, pattern)
                torch.testing.assert_close(pool.parked_mask, mask)
                self.assertEqual(len(pool.state_writes), 1)
                torch.testing.assert_close(pool.state_writes[0], before)
                for slot, name in enumerate(names):
                    body = stage.GetPrimAtPath('/World/envs/env_0/' + name)
                    self.assertEqual(self.UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get(), slot == 1)
                    collider = stage.GetPrimAtPath(body.GetPath().AppendChild('collider'))
                    self.assertTrue(self.UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())

    def fixture(self):
        stage = self.Usd.Stage.CreateInMemory()
        self.UsdGeom.Xform.Define(stage, '/World/envs/env_0')
        body = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/body')
        body.AddTranslateOp().Set((1., 2., 3.))
        self.UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        self.UsdGeom.Xform.Define(stage, '/Source')
        mesh = self.UsdGeom.Mesh.Define(stage, '/Source/mesh')
        mesh.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        material = self.UsdShade.Material.Define(stage, '/Looks/Paint')
        shader = self.UsdShade.Shader.Define(stage, '/Looks/Paint/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', self.Sdf.ValueTypeNames.Color3f).Set((.1, .2, .3))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        self.UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        visual = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/body/visuals').GetPrim()
        visual.GetReferences().AddInternalReference('/Source')
        visual.SetInstanceable(True)
        collider = self.UsdGeom.Cube.Define(stage, '/World/envs/env_0/body/collider')
        collider.CreateSizeAttr(.25)
        self.UsdPhysics.CollisionAPI.Apply(collider.GetPrim()).CreateCollisionEnabledAttr(True)
        return stage, body.GetPrim()

    def test_env_reference_preserves_visual_bytes_collider_hull_and_width_angle_across_origins(self):
        stage, body = self.fixture()
        self.UsdGeom.Xform(body).AddRotateXYZOp().Set((15., -20., 30.))
        self.UsdGeom.Xform(body).AddScaleOp().Set((2., 3., 4.))
        reference = stage.GetPrimAtPath('/World/envs/env_0')
        origin = self.UsdGeom.Xformable(reference).AddTranslateOp(
            precision=self.UsdGeom.XformOp.PrecisionDouble)
        colliders = (stage.GetPrimAtPath(body.GetPath().AppendChild('collider')),)
        shear = self.Gf.Matrix4d(1.)
        shear[0, 1], shear[1, 2] = .17, -.09
        self.UsdGeom.Xformable(colliders[0]).AddTransformOp().Set(shear)
        upright = (math.sqrt(.5), -math.sqrt(.5), 0., 0.)
        def visual(frame=None):
            cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
            value = (cache.GetLocalToWorldTransform(body) if frame is None
                     else cache.ComputeRelativeTransform(body, frame)[0])
            world = self.column(value)
            rigid = np.eye(4)
            rigid[:3, :3], rigid[:3, 3] = self.rotation_part(world), world[:3, 3]
            return self.visual_triangles(body, np.linalg.inv(rigid), reference_frame=frame)
        baseline = visual()
        support, error = self.support_vertices(body, colliders, upright)
        hull = self.collision_hull(support)
        profile = task._certified_footprint_profile((0., 0., 0.), (1., 1., 1.), hull)
        for offset in ((0., 0., 0.), (46.5, -46.5, 0.), (-46.5, 46.5, 0.)):
            origin.Set(self.Gf.Vec3d(*offset))
            actual = visual(reference)
            self.assertEqual(actual[0].tobytes(), baseline[0].tobytes())
            self.assertEqual(actual[1].tobytes(), baseline[1].tobytes())
            points, actual_error = self.support_vertices(body, colliders, upright, reference_frame=reference)
            self.assertEqual(points, support)
            self.assertEqual(actual_error, error)
            actual_hull = self.collision_hull(points)
            self.assertEqual(actual_hull, hull)
            self.assertEqual(task._certified_footprint_profile((0., 0., 0.), (1., 1., 1.), actual_hull), profile)

    def test_shelf_reference_bounds_preserve_original_rules_without_world_cancellation(self):
        stage = self.Usd.Stage.CreateInMemory()
        reference = self.UsdGeom.Xform.Define(stage, '/env_0').GetPrim()
        origin = self.UsdGeom.Xformable(reference).AddTranslateOp(
            precision=self.UsdGeom.XformOp.PrecisionDouble)
        shelf = self.UsdGeom.Xform.Define(stage, '/env_0/shelf').GetPrim()
        shapes = (('support', (-.7, 0., 1.04), (.4, 1., .02)),
                  ('ceiling', (-.7, 0., 1.49), (.4, 1., .02)),
                  ('left', (-.7, -.465, .9), (.01, .02, 1.8)),
                  ('right', (-.7, .465, .9), (.01, .02, 1.8)))
        for name, center, size in shapes:
            shape = self.UsdGeom.Cube.Define(stage, shelf.GetPath().AppendChild(name))
            shape.GetSizeAttr().Set(1.)
            shape.AddTranslateOp().Set(self.Gf.Vec3d(*center))
            shape.AddScaleOp().Set(self.Gf.Vec3f(*size))
            self.UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
        def cache():
            return self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
                useExtentsHint=False, ignoreVisibility=True)
        baseline = self.shelf_rules(cache(), shelf, (0., 0., 0.), task.SpawnRules())
        for offset in ((0., 0., 0.), (46.5, -46.5, 0.), (-46.5, 46.5, 0.)):
            origin.Set(self.Gf.Vec3d(*offset))
            actual = self.shelf_rules(cache(), shelf, offset, task.SpawnRules(), reference_frame=reference)
            self.assertEqual(actual, baseline)

    def test_detached_composed_body_bounds_preserve_origin_zero_after_clone_translation(self):
        stage, body = self.fixture()
        self.UsdGeom.Xform(body).AddRotateXYZOp().Set((15., -20., 30.))
        self.UsdGeom.Xform(body).AddScaleOp().Set((2., 3., 4.))
        reference = stage.GetPrimAtPath('/World/envs/env_0')
        origin = self.UsdGeom.Xformable(reference).AddTranslateOp(
            precision=self.UsdGeom.XformOp.PrecisionDouble)
        def bounds(prim):
            cache = self.UsdGeom.BBoxCache(self.Usd.TimeCode.Default(), ['default', 'render', 'proxy'],
                useExtentsHint=False, ignoreVisibility=True)
            result = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
            return tuple(result.GetMin()), tuple(result.GetMax())
        baseline = bounds(body)
        for offset in ((0., 0., 0.), (46.5, -46.5, 0.), (-46.5, 46.5, 0.)):
            origin.Set(self.Gf.Vec3d(*offset))
            detached = self.Usd.Stage.CreateInMemory()
            isolated = detached.OverridePrim('/Object')
            isolated.GetReferences().AddReference(stage.GetRootLayer().identifier, body.GetPath())
            self.assertEqual(bounds(isolated), baseline)
            self.assertEqual(body.GetPath().pathString, '/World/envs/env_0/body')
            self.assertEqual(tuple(origin.Get()), offset)

    def test_collision_support_retains_body_scale_child_transform_upright_and_ignores_disabled_shape(self):
        stage, body = self.fixture()
        xform = self.UsdGeom.Xform(body)
        xform.AddRotateXYZOp().Set((15., -20., 30.))
        xform.AddScaleOp().Set((2., 3., 4.))
        child = self.UsdGeom.Xform.Define(stage, body.GetPath().AppendChild('offset'))
        child.AddTranslateOp().Set((.11, -.03, .07))
        child.AddRotateXYZOp().Set((0., 90., 0.))
        child.AddScaleOp().Set((.5, 1.5, .8))
        cube = self.UsdGeom.Cube.Define(stage, child.GetPrim().GetPath().AppendChild('cube'))
        cube.CreateSizeAttr(.2)
        self.UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
        disabled = self.UsdGeom.Cube.Define(stage, body.GetPath().AppendChild('disabled'))
        disabled.CreateSizeAttr(20.)
        disabled.AddTranslateOp().Set((100., 100., 100.))
        self.UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
        colliders = (stage.GetPrimAtPath(body.GetPath().AppendChild('collider')),
                     cube.GetPrim(), disabled.GetPrim())
        upright = (math.sqrt(.5), math.sqrt(.5), 0., 0.)
        actual, error = self.support_vertices(body, colliders, upright)
        self.assertEqual(error, 0.)
        cache = self.UsdGeom.XformCache(self.Usd.TimeCode.Default())
        rigid_inverse = cache.GetLocalToWorldTransform(body).RemoveScaleShear().GetInverse()
        oracle = []
        for collider, size in ((colliders[0], .25), (colliders[1], .2)):
            world = cache.GetLocalToWorldTransform(collider)
            for point in product((-.5 * size, .5 * size), repeat=3):
                oracle.append(tuple(rigid_inverse.Transform(world.Transform(self.Gf.Vec3d(*point)))))
        oracle = Rotation.from_euler('x', 90., degrees=True).apply(np.asarray(oracle))
        # Hull simplification may remove interior corners; plane support must
        # remain exactly equal to the independently transformed cube vertices.
        directions = np.random.default_rng(571).normal(size=(100, 3))
        np.testing.assert_allclose((np.asarray(actual) @ directions.T).min(0),
                                   (oracle @ directions.T).min(0), rtol=1.e-10, atol=1.e-10)
        body.GetAttribute('xformOp:translate').Set((-5., 4., 2.))
        body.GetAttribute('xformOp:rotateXYZ').Set((-12., 36., 5.))
        moved, moved_error = self.support_vertices(body, colliders, upright)
        self.assertEqual(moved_error, 0.)
        np.testing.assert_allclose((np.asarray(moved) @ directions.T).min(0),
                                   (oracle @ directions.T).min(0), rtol=1.e-10, atol=1.e-10)

    def test_nested_mesh_instances_material_and_collision_survive_and_replicas_share_geometry(self):
        import tempfile
        stage, body = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'geometry.usdc')), 1)
            self.assertTrue(body.IsInstance())
            mesh = self.UsdGeom.Mesh(stage.GetPrimAtPath('/World/envs/env_0/body/visuals/mesh'))
            self.assertEqual(list(mesh.GetFaceVertexIndicesAttr().Get()), [0, 1, 2])
            np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), [[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
            material, _ = self.UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
            self.assertTrue(material)
            surface = self.UsdShade.Shader(material.GetPrim().GetChild('Shader'))
            np.testing.assert_allclose(surface.GetInput('diffuseColor').Get(), [.1, .2, .3])
            collider = stage.GetPrimAtPath('/World/envs/env_0/body/collider')
            self.assertTrue(self.UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get())
            self.assertEqual(self.UsdGeom.Cube(collider).GetSizeAttr().Get(), .25)
            copy = self.UsdGeom.Xform.Define(stage, '/World/envs/env_1').GetPrim()
            copy.GetInherits().AddInherit('/World/envs/env_0')
            other = stage.GetPrimAtPath('/World/envs/env_1/body')
            self.assertEqual(body.GetPrototype(), other.GetPrototype())
            other.GetAttribute('xformOp:translate').Set((4., 5., 6.))
            self.assertEqual(tuple(body.GetAttribute('xformOp:translate').Get()), (1., 2., 3.))
            self.assertTrue(self.UsdPhysics.RigidBodyAPI(other))
            self.assertFalse(other.IsInstanceProxy())

    def test_joint_or_nested_body_root_is_not_made_an_instance(self):
        import tempfile
        stage, body = self.fixture()
        self.UsdPhysics.FixedJoint.Define(stage, '/World/envs/env_0/body/joint')
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'geometry.usdc')), 0)
            self.assertFalse(body.IsInstance())

    def test_hand_proxy_replaces_referenced_collision_after_sharing(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = self.Usd.Stage.CreateNew(str(Path(directory) / 'source.usda'))
            self.UsdGeom.Xform.Define(source, '/Hand')
            visual = self.UsdGeom.Mesh.Define(source, '/Hand/visual')
            visual.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
            visual.CreateFaceVertexCountsAttr([3])
            visual.CreateFaceVertexIndicesAttr([0, 1, 2])
            original = self.UsdGeom.Cube.Define(source, '/Hand/collisions/original').GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(original).CreateCollisionEnabledAttr(True)
            source.GetRootLayer().Save()
            stage = self.Usd.Stage.CreateInMemory()
            body = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/hand').GetPrim()
            body.GetReferences().AddReference(source.GetRootLayer().identifier, '/Hand')
            self.UsdPhysics.RigidBodyAPI.Apply(body)
            body.CreateAttribute('contactManipulation:handProxy', self.Sdf.ValueTypeNames.Bool).Set(True)
            stage.GetPrimAtPath(body.GetPath().AppendPath('collisions/original')).SetActive(False)
            proxy = self.UsdGeom.Cube.Define(stage, body.GetPath().AppendChild('proxy')).GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'shared.usdc')), 1)
            colliders = [p.GetName() for p in self.Usd.PrimRange(body, self.Usd.TraverseInstanceProxies())
                         if p.HasAPI(self.UsdPhysics.CollisionAPI)]
            self.assertEqual(colliders, ['proxy'])
            self.assertTrue(body.IsInstance())

    def test_reference_collision_overrides_survive_sharing_without_hand_marker(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = self.Usd.Stage.CreateNew(str(Path(directory) / 'source.usda'))
            body = self.UsdGeom.Xform.Define(source, '/Object').GetPrim()
            self.UsdPhysics.RigidBodyAPI.Apply(body)
            mesh = self.UsdGeom.Mesh.Define(source, '/Object/mesh')
            mesh.CreatePointsAttr([(-1., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
            self.UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
            mesh.GetPrim().CreateAttribute('physxCollision:contactOffset', self.Sdf.ValueTypeNames.Float).Set(.02)
            source.GetRootLayer().Save()
            stage = self.Usd.Stage.CreateInMemory()
            live = self.UsdGeom.Xform.Define(stage, '/World/envs/env_0/object').GetPrim()
            live.GetReferences().AddReference(source.GetRootLayer().identifier, '/Object')
            original = stage.GetPrimAtPath(live.GetPath().AppendChild('mesh'))
            self.UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
            original.GetAttribute('physxCollision:contactOffset').Set(.005)
            proxy = self.UsdGeom.Cube.Define(stage, live.GetPath().AppendChild('proxy')).GetPrim()
            self.UsdPhysics.CollisionAPI.Apply(proxy).CreateCollisionEnabledAttr(True)
            self.assertEqual(self.share(stage, '/World/envs/env_0', str(Path(directory) / 'shared.usdc')), 1)
            original = stage.GetPrimAtPath(live.GetPath().AppendChild('mesh'))
            self.assertFalse(self.UsdPhysics.CollisionAPI(original).GetCollisionEnabledAttr().Get())
            self.assertAlmostEqual(original.GetAttribute('physxCollision:contactOffset').Get(), .005)
            self.assertEqual(len(self.UsdGeom.Mesh(original).GetPointsAttr().Get()), 3)


class CertifiedEpisodeBindingTests(unittest.TestCase):
    """The reset binder changes pool IDs, never a certified geometric domain."""

    def fixture(self, count):
        import random
        geometry = (task.ObjectGeometry('target', (-.025, -.03, 0.), (.025, .03, .1), 'target'), *(
            task.ObjectGeometry(f'{asset}_{copy}', (-.025, -.03, 0.), (.025, .03, .1), asset)
            for asset in ('blocker', 'cup', 'can', 'bottle') for copy in range(3)))
        rules = task.SpawnRules(min_objects=count, max_objects=count, min_asset_types=4, max_asset_types=4)
        ids, target, blocker = task._constructive_composition(geometry, rules, random.Random(1), 0, 2)
        visibility = dict(valid_intervals=((- .015, .015),), target_visibility_events=(-.02, .02),
            target_occluded_counts=(40,), target_pixel_count=100, blocker_visible_fraction_structural=1.,
            fingerprint='typed-binding', source=dict(target_id=999, blocker_id=888))
        prepared = task.prepare_certified_scene_domain(geometry, rules, ids, target, blocker,
            0., 0., .065, (.83, 0., 1.23, .5, -.5, -.5, .5), visibility,
            [dict(parameter_bounds=(-.015, .015), yaw_bounds=(-.02, .02),
                  module_bounds=(-.03, .10, -.07, .15), goal_y=.11, side=1,
                  blockage_lower_bound=.7, camera_x_bounds_over_yaw=(.83*math.cos(.02), .83))])
        family = dict(target_yaw=0., blocker_yaw=0., blocker_x=.065, camera_pose=prepared.camera_pose,
            cells=prepared.cells, cumulative=prepared.cumulative, visibility_domain=visibility,
            visibility_fingerprint=visibility['fingerprint'])
        return geometry, rules, family

    def test_four_and_ten_instances_draw_balanced_copies_without_runtime_preparation(self):
        import random
        from contextlib import ExitStack
        for count in (4, 10):
            geometry, rules, family = self.fixture(count)
            catalog = {('target', 'blocker'): (family,)}
            with ExitStack() as stack:
                for name in ('prepare_certified_scene_domain', '_certify_role_translation_cells',
                             'construct_scene', '_capture_constructive_views'):
                    stack.enter_context(patch.object(task, name, side_effect=AssertionError('reset preparation')))
                samples = [task.sample_certified_episode(catalog, geometry, rules, random.Random(seed), 0, 2)
                           for seed in (2, 7, 19)]
            for sample in samples:
                assets = [geometry[index].asset_name for index in sample.active_ids]
                self.assertEqual(len(assets), count)
                self.assertEqual(len(set(sample.active_ids)), count)
                self.assertEqual(assets.count('target'), 1)
                self.assertEqual(sorted(assets.count(asset) for asset in set(assets)-{'target'}),
                                 [1 if count == 4 else 3]*3)
                self.assertEqual((sample.target_id, sample.blocker_id), (0, 2))
                self.assertEqual(sample.parameter_certificate['predicted_role_visibility'], (.6, 1.))

    def test_typed_guard_rebinds_to_first_available_nonrole_copy_and_uses_loader_once(self):
        import random
        geometry, rules, normal = self.fixture(10)
        ids, _, _ = task._constructive_composition(geometry, rules, random.Random(5), 0, 2)
        guard = next(index for index in ids if index not in (0, 2))
        guarded = dict(normal, fixed_objects={geometry[guard].asset_name: (.14, .15, .3)},
                       pivot=dict(turn_angle=-.2))
        from unittest.mock import Mock
        loader = Mock(return_value=normal['visibility_domain'])
        catalog = SimpleNamespace(families={('target', 'blocker'): (guarded,)}, load_visibility=loader)
        with patch.object(task, 'sample_certified_scene', side_effect=lambda domain, rng: domain) as sample:
            domain = task.sample_certified_episode(catalog, geometry, rules, random.Random(5), 0, 2)
        sample.assert_called_once()
        loader.assert_called_once_with(guarded)
        self.assertEqual(domain.active_ids, ids)
        self.assertEqual(domain.fixed_objects, {guard: (.14, .15, .3)})
        self.assertNotIn(guard, domain.background_ids)
        self.assertIs(domain.cells, guarded['cells'])
        self.assertIs(domain.cumulative, guarded['cumulative'])
        self.assertIs(domain.pivot, guarded['pivot'])

    def test_unavailable_guard_is_a_categorical_filter_and_never_changes_composition(self):
        import random
        geometry, rules, normal = self.fixture(4)
        ids, _, _ = task._constructive_composition(geometry, rules, random.Random(9), 0, 2)
        missing = next(asset for asset in ('cup', 'can', 'bottle')
                       if asset not in {geometry[index].asset_name for index in ids})
        guarded = dict(normal, target_yaw=999., fixed_objects={missing: (.1, .1, 0.)})
        catalog = {('target', 'blocker'): (guarded, normal)}
        with patch.object(task, 'sample_certified_scene', side_effect=lambda domain, rng: domain):
            domain = task.sample_certified_episode(catalog, geometry, rules, random.Random(9), 0, 2)
        self.assertEqual(domain.active_ids, ids)
        self.assertEqual(domain.target_yaw, normal['target_yaw'])
        with patch.object(task, 'sample_certified_scene') as sample:
            with self.assertRaisesRegex(RuntimeError, 'No redraw'):
                task.sample_certified_episode({('target', 'blocker'): (guarded,)}, geometry, rules,
                                              random.Random(9), 0, 2)
            sample.assert_not_called()

    def test_visibility_fingerprint_mismatch_fails_without_sampling_another_family(self):
        import random
        geometry, rules, family = self.fixture(4)
        family = dict(family, visibility_fingerprint='different-compiled-source')
        with patch.object(task, 'sample_certified_scene') as sample:
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                task.sample_certified_episode({('target', 'blocker'): (family,)}, geometry, rules,
                                              random.Random(1), 0, 2)
            sample.assert_not_called()


class CertifiedSceneDomainTests(unittest.TestCase):
    def test_observation_resolution_keeps_existing_certification_cache_identity(self):
        path = Path(__file__).resolve().parents[1] / 'contact_manipulation/mdp/sensors.py'
        cfg_class = next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.ClassDef) and node.name == 'InitialVisionCfg')
        namespace = dict(dataclass=dataclasses.dataclass, math=math)
        exec(compile(ast.Module(body=[cfg_class], type_ignores=[]), str(path), 'exec'), namespace)
        geometry = tuple(task.ObjectGeometry(str(i), (-.025, -.03, 0.), (.025, .03, .1), str(i))
                         for i in range(4))
        sensor = SimpleNamespace(cfg=namespace['InitialVisionCfg'](),
            _camera_mesh=lambda *args: (None, None, 'unchanged-visual-mesh'))
        env = SimpleNamespace(device='cpu', spawn_geometries=geometry, spawn_rules=task.SpawnRules(),
            cfg=SimpleNamespace(target_asset='0', blocker_asset='1'),
            task_scene_data=SimpleNamespace(upright=identity(4)))
        runtime = SimpleNamespace(vision=sensor)
        reference = task._certified_catalog_source(env, runtime)
        observation_fields = {'observation_ray_width', 'observation_ray_height', 'observation_min_depth_pixels'}
        legacy_vision = {item.name: getattr(sensor.cfg, item.name) for item in dataclasses.fields(sensor.cfg)
                         if item.name not in observation_fields}
        self.assertEqual(reference['vision'], legacy_vision)
        sensor.cfg.observation_ray_width, sensor.cfg.observation_ray_height = 32, 24
        sensor.cfg.observation_min_depth_pixels = 5
        self.assertEqual(task._certified_catalog_source(env, runtime), reference)
        sensor.cfg.ray_width *= 2
        self.assertNotEqual(task._certified_catalog_source(env, runtime), reference)

    def test_catalog_load_requires_full_yaw_all_pairs_and_a_pivot_domain(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock
        base = self.domain()
        def manifest():
            values = [dict(target_yaw=index*math.pi/2, blocker_yaw=index*math.pi/2+.2,
                cells=[dict(base.cells[0], yaw_bounds=(-math.pi/4, math.pi/4))], cumulative=[1.],
                visibility_fingerprint='test') for index in range(4)]
            values.append(dict(values[0], fixed_objects={'2': (.14, .15, .3)},
                pivot=dict(pivot_offset_from_blocker=(-.025, -.03),
                           turn_offset_from_blocker=(.005, -.055), turn_angle=-math.pi/2)))
            return dict(source=dict(assets=[dict(asset=str(i)) for i in range(4)], target_asset='0', blocker_asset='1'),
                guarded_family_count=1, pairs=[dict(target_asset='0', blocker_asset='1', families=values)])
        sensor = SimpleNamespace(_load_translation_cache=Mock(return_value=base.visibility))
        with TemporaryDirectory() as directory:
            Path(directory, 'test.json').write_text('{}')
            catalog = task.CertifiedSceneCatalog(manifest(), sensor, directory)
            self.assertEqual(set(catalog.families), {('0', '1')})
            family = catalog.families['0', '1'][0]
            self.assertIs(catalog.load_visibility(family), base.visibility)
            sensor._load_translation_cache.assert_called_once_with(Path(directory, 'test.json'), 'test')
            missing = manifest()
            missing['pairs'][0]['families'].pop(3)
            with self.assertRaisesRegex(ValueError, 'incomplete role yaw'):
                task.CertifiedSceneCatalog(missing, sensor, directory)
            missing = manifest()
            missing['pairs'] = []
            with self.assertRaisesRegex(ValueError, 'missing configured ordered'):
                task.CertifiedSceneCatalog(missing, sensor, directory)
            missing = manifest()
            missing['pairs'][0]['families'].pop()
            with self.assertRaisesRegex(ValueError, 'pivot-required'):
                task.CertifiedSceneCatalog(missing, sensor, directory)

    def test_pair_checkpoint_roundtrip_resumes_with_actual_cache_checksum_loader(self):
        import hashlib,json,os,tempfile
        from tempfile import TemporaryDirectory
        path=Path(__file__).resolve().parents[1]/'contact_manipulation/mdp/sensors.py'
        sensor_class=next(node for node in ast.parse(path.read_text()).body
                          if isinstance(node,ast.ClassDef) and node.name=='InitialVisionSensor')
        cache_class=ast.ClassDef(name='CacheOnlySensor',bases=[],keywords=[],decorator_list=[],body=[
            node for node in sensor_class.body if isinstance(node,ast.FunctionDef)
            and node.name in ('_save_translation_cache','_load_translation_cache')])
        namespace=dict(json=json,hashlib=hashlib,os=os,tempfile=tempfile)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[cache_class],type_ignores=[])),str(path),'exec'),namespace)
        CacheOnlySensor=namespace['CacheOnlySensor']
        sensor_stub=types.ModuleType('contact_manipulation.mdp.sensors')
        sensor_stub.InitialVisionRejected=type('InitialVisionRejected',(RuntimeError,),{})
        geometry=tuple(task.ObjectGeometry(str(i),(-.025,-.03,0.),(.025,.03,.1),str(i)) for i in range(4))
        rules=task.SpawnRules(min_objects=4,max_objects=4)
        source=dict(version='checkpoint-test',assets=[dict(asset=str(i),copies=1) for i in range(4)],
                    target_asset='0',blocker_asset='1')
        visibility_source=dict(kind='mock compiled role certificate')
        visibility_fingerprint=hashlib.sha256(json.dumps(visibility_source,sort_keys=True).encode()).hexdigest()
        visibility=dict(source=visibility_source,fingerprint=visibility_fingerprint,
                        valid_intervals=((- .015,.015),))
        sensor=SimpleNamespace(cfg=SimpleNamespace(front_distance_range=(.55,.95)),
            _save_translation_cache=CacheOnlySensor._save_translation_cache,
            _load_translation_cache=CacheOnlySensor._load_translation_cache)
        env=SimpleNamespace(spawn_geometries=geometry,spawn_rules=rules,
            cfg=SimpleNamespace(target_asset='0',blocker_asset='1'),
            task_scene_data=SimpleNamespace(upright=identity(4)))
        def compiled(_geometry,_upright,_sensor,target,blocker,tyaw,byaw,_rules,*,yaw_half_width,**kwargs):
            return dict(target_id=target,blocker_id=blocker,target_yaw=tyaw,blocker_yaw=byaw,blocker_x=.065,
                camera_pose=(.83,0.,1.23,.5,-.5,-.5,.5),visibility_domain=visibility,
                role_cells=[dict(parameter_bounds=(-.015,.015),yaw_bounds=(-yaw_half_width,yaw_half_width),
                    module_bounds=(-.03,.10,-.07,.15),goal_y=.11,side=1,blockage_lower_bound=.7,
                    camera_x_bounds_over_yaw=(.83*math.cos(yaw_half_width),.83))])
        def guarded(family,_geometry,_rules,guard):
            return dict(family,fixed_objects={guard:(.07,.10,0.)},guard_certificate=dict(mock=True),
                pivot=dict(pivot_offset_from_blocker=(-.025,-.03),
                    turn_offset_from_blocker=(.005,-.055),turn_angle=-math.pi/2))
        with TemporaryDirectory() as directory, patch.dict(sys.modules,{'contact_manipulation.mdp.sensors':sensor_stub}):
            directory=Path(directory)
            sensor._save_translation_cache(directory/f'{visibility_fingerprint}.json',visibility)
            with patch.object(task,'_certified_catalog_source',return_value=source), \
                 patch.object(task,'precompute_certified_role_family',side_effect=compiled) as compile_roles, \
                 patch.object(task,'_precompute_guarded_role_family',side_effect=guarded), \
                 patch('builtins.print'):
                first=task.initialize_certified_catalog(env,SimpleNamespace(vision=sensor),directory)
            self.assertGreater(compile_roles.call_count,0)
            self.assertEqual(set(first.families),{('0','1')})
            manifest_files=list(directory.glob('catalog_*.json'))
            checkpoint_files=list(directory.glob('pair_*.json'))
            self.assertEqual((len(manifest_files),len(checkpoint_files)),(1,1))
            saved=json.loads(checkpoint_files[0].read_text())['domain']
            self.assertIn('source',saved)
            self.assertIsNotNone(sensor._load_translation_cache(checkpoint_files[0],saved['fingerprint']))
            manifest_files[0].unlink()  # Interrupted whole-catalog write; retain completed ordered pair.
            with patch.object(task,'_certified_catalog_source',return_value=source), \
                 patch.object(task,'precompute_certified_role_family',side_effect=AssertionError('recompiled pair')), \
                 patch.object(task,'_precompute_guarded_role_family',side_effect=AssertionError('recompiled guard')):
                resumed=task.initialize_certified_catalog(env,SimpleNamespace(vision=sensor),directory)
            self.assertEqual(len(resumed.families['0','1']),len(first.families['0','1']))

    def test_catalog_capacity_covers_all_count_allocations_once_per_multiset(self):
        domain = self.domain()
        result = task.certified_catalog_capacity(domain.geometries, domain.rules)
        self.assertEqual(result['count_allocations'], 12)
        self.assertEqual(result['unique_background_multisets'], 6)
        self.assertTrue(result['depth_valid'])
        self.assertAlmostEqual(result['minimum_background_slab_width'], math.hypot(.05, .06))
        self.assertAlmostEqual(result['maximum_module_width'], .88-.01-math.hypot(.05, .06))

    @staticmethod
    def varied_allocations():
        """Cartesian count oracle, independent of the production combinations loop."""
        sizes=((.035,.075),(.12,.045),(.085,.085),(.15,.06),(.05,.105))
        geometry=tuple(task.ObjectGeometry(f'asset{i}_{copy}',(-w/2,-h/2,0.),(w/2,h/2,.1),f'asset{i}')
                       for i,(w,h) in enumerate(sizes) for copy in range(3))
        rules=task.SpawnRules(min_objects=10,max_objects=10,min_asset_types=4,max_asset_types=5,
                             x_range=(-1.5,0.),y_range=(-1.5,1.5))
        diameter=tuple(math.hypot(w,h) for w,h in sizes)
        minimum=tuple(min(w,h) for w,h in sizes)
        def metric(kinds):
            ordered=sorted(kinds,key=lambda i:(-diameter[i],f'asset{i}'))
            rows=[[],[]]
            for rank,i in enumerate(ordered):rows[((rank+1)//2)%2].append(i)
            span=max(sum(minimum[i] for i in row)+max((diameter[i]-minimum[i] for i in row),default=0.)+
                     rules.max_neighbor_gap*max(0,len(row)-1) for row in rows)
            depth=sum(max((diameter[i] for i in row),default=0.) for row in rows)+rules.max_neighbor_gap
            return span,depth
        allocations=[]
        for target in range(5):
            for counts in product(range(4),repeat=5):
                positive=[amount for amount in counts if amount]
                if (counts[target] or sum(counts)!=9 or len(positive) not in (3,4)
                        or max(positive)-min(positive)>1):continue
                for blocker in range(5):
                    if counts[blocker]:allocations.append((target,blocker,counts))
        return geometry,rules,allocations,metric

    def test_pair_and_guard_worst_budgets_cover_every_balanced_count_vector(self):
        geometry,rules,allocations,metric=self.varied_allocations()
        result=task.certified_catalog_capacity(geometry,rules)
        self.assertEqual(result['count_allocations'],len(allocations))
        support_width=rules.y_range[1]-rules.y_range[0]-2*rules.edge_margin
        expected={}
        for target,blocker,counts in allocations:
            kinds=[kind for kind,amount in enumerate(counts) for _ in range(amount)]
            kinds.remove(blocker)
            span,depth=metric(kinds)
            entry=expected.setdefault(f'asset{target}|asset{blocker}',dict(widths=[],depths=[],guards={}))
            entry['widths'].append(span);entry['depths'].append(depth)
            for guard in set(kinds):
                reduced=list(kinds);reduced.remove(guard)
                guard_span,guard_depth=metric(reduced)
                residual=entry['guards'].setdefault(f'asset{guard}',dict(widths=[],depths=[]))
                residual['widths'].append(guard_span);residual['depths'].append(guard_depth)
        self.assertEqual(set(result['ordered_pair_budgets']),set(expected))
        for key,wanted in expected.items():
            actual=result['ordered_pair_budgets'][key]
            self.assertAlmostEqual(actual['minimum_module_width'],support_width-rules.max_neighbor_gap-max(wanted['widths']))
            self.assertAlmostEqual(actual['maximum_module_width'],support_width-rules.max_neighbor_gap-min(wanted['widths']))
            self.assertEqual(set(actual['guard_assets']),set(wanted['guards']))
            for asset,guard in wanted['guards'].items():
                cached=actual['guard_budgets'][asset]
                self.assertAlmostEqual(cached['minimum_background_slab_width'],max(guard['widths']))
                self.assertAlmostEqual(cached['maximum_background_depth'],max(guard['depths']))
                self.assertAlmostEqual(cached['minimum_module_width'],support_width-rules.max_neighbor_gap-max(guard['widths']))

    def test_same_surrounding_composition_retains_all_role_yaw_intervals_under_pair_worst_budget(self):
        geometry,rules,allocations,metric=self.varied_allocations()
        capacity=task.certified_catalog_capacity(geometry,rules)
        budget=capacity['ordered_pair_budgets']['asset0|asset1']['minimum_module_width']
        support=capacity['support_bounds']
        pair_capacity=dict(capacity,maximum_module_width=budget,
            minimum_background_slab_width=support[3]-support[2]-rules.max_neighbor_gap-budget)
        width=budget-.002
        source=dict(parameter_bounds=(-.01,.01),yaw_bounds=(-math.pi/4,math.pi/4),
                    module_bounds=(-.03,.13,-width/2,width/2),goal_y=.11,side=1,
                    blockage_lower_bound=.7,camera_x_bounds_over_yaw=(.9,1.05))
        visibility=dict(valid_intervals=((- .01,.01),))
        families=[]
        for quadrant in range(4):
            family=dict(target_yaw=quadrant*math.pi/2,blocker_yaw=quadrant*math.pi/2+.2,role_cells=[source])
            cells,cdf=task._catalog_scene_cells(family,rules,pair_capacity)
            self.assertTrue(cells)
            self.assertTrue(all(v>0. for v in np.diff(np.r_[0.,cdf])))
            families.append(dict(family,role_cells=cells))
        checked=0
        for target,blocker,counts in allocations:
            if (target,blocker)!=(0,1):continue
            ids=(0,*[kind*3+copy for kind,amount in enumerate(counts) for copy in range(amount)])
            background=[kind for kind,amount in enumerate(counts) for _ in range(amount)]
            background.remove(blocker)
            span,_=metric(background)
            rebound=[]
            for family in families:
                # Every stored catalog translation interval remains valid for
                # this exact count vector; no other composition supplies its
                # missing angles or repairs its available shelf width.
                for cell in family['role_cells']:
                    lo,hi=cell['y_translation_bounds'];bounds=cell['module_bounds']
                    if cell['background_side']>0:
                        self.assertLessEqual(hi+bounds[3]+rules.max_neighbor_gap+span,support[3]+1.e-12)
                    else:
                        self.assertGreaterEqual(lo+bounds[2]-rules.max_neighbor_gap-span,support[2]-1.e-12)
                domain=task.prepare_certified_scene_domain(geometry,rules,ids,0,3,
                    family['target_yaw'],family['blocker_yaw'],.1,(1.,0.,1.23,.5,-.5,-.5,.5),
                    visibility,[source])
                rebound.append(dict(family,role_cells=domain.cells))
            self.assertEqual(task.certified_role_yaw_gaps(rebound),dict(target=(),blocker=()))
            checked+=1
        self.assertGreater(checked,1)

    def test_full_circle_coverage_uses_intervals_and_detects_small_holes(self):
        families = [dict(target_yaw=index*math.pi/2, blocker_yaw=index*math.pi/2+.2,
                         role_cells=[dict(yaw_bounds=(-math.pi/4, math.pi/4))]) for index in range(4)]
        self.assertEqual(task.certified_role_yaw_gaps(families), dict(target=(), blocker=()))
        broken = [dict(family, role_cells=[dict(yaw_bounds=(-math.pi/4+.001, math.pi/4))])
                  for family in families]
        gaps = task.certified_role_yaw_gaps(broken)
        self.assertAlmostEqual(sum(high-low for low, high in gaps['target']), .004)
        self.assertAlmostEqual(sum(high-low for low, high in gaps['blocker']), .004)
        self.assertEqual(task.certified_role_yaw_gaps([]), dict(target=((0., 2*math.pi),), blocker=((0., 2*math.pi),)))

    def domain(self, **changes):
        geometry = tuple(task.ObjectGeometry(str(index), (-.025, -.03, 0.), (.025, .03, .10), str(index))
                         for index in range(4))
        rules = task.SpawnRules(min_objects=4, max_objects=4)
        frame = np.array(((0., 0., -1.), (1., 0., 0.), (0., -1., 0.)))
        xyzw = Rotation.from_matrix(frame).as_quat()
        camera = (.83, 0., 1.23, xyzw[3], *xyzw[:3])
        values = dict(geometries=geometry, rules=rules, active_ids=(0, 1, 2, 3), target_id=0, blocker_id=1,
            target_yaw=0., blocker_yaw=0., blocker_x=.065, camera_pose=camera,
            visibility_domain=dict(valid_intervals=((- .015, .015),), target_visibility_events=(-.02, .02),
                target_occluded_counts=(40,), target_pixel_count=100, blocker_visible_fraction_structural=1.,
                fingerprint='synthetic-test-domain'),
            role_cells=[dict(parameter_bounds=(-.015, .015), yaw_bounds=(-.02, .02),
                module_bounds=(-.03, .10, -.07, .15), goal_y=.11, side=1, blockage_lower_bound=.7,
                camera_x_bounds_over_yaw=(.83*math.cos(.02), .83))])
        values.update(changes)
        return task.prepare_certified_scene_domain(**values)

    def test_direct_samples_preserve_roles_cones_motion_and_cached_visibility(self):
        import random
        domain = self.domain()
        old_paths = ('construct_scene', '_capture_constructive_views', 'sample_scene', 'polygon_distance')
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in old_paths:
                stack.enter_context(patch.object(task, name, side_effect=AssertionError('runtime candidate test')))
            samples = [task.sample_certified_scene(domain, random.Random(seed)) for seed in range(64)]
        for sample in samples:
            self.assertEqual(sample.active_ids, (0, 1, 2, 3))
            self.assertEqual(sample.clearance_kind, 'certified_translation')
            self.assertFalse(sample.visibility_variants)
            self.assertEqual(sample.parameter_certificate['predicted_role_visibility'], (.6, 1.))
            self.assertFalse(sample.parameter_certificate['live_visibility_measured'])
            self.assertGreaterEqual(sample.camera_pose[0]-domain.rules.x_range[1], .55-1.e-10)
            self.assertLessEqual(sample.camera_pose[0]-domain.rules.x_range[1], .95+1.e-10)
            self.assertAlmostEqual(sum(value*value for value in sample.camera_pose[3:]), 1., places=12)
            finish = sample.clearance_path[0].end
            polygon = task.footprint_polygon(domain.geometries[1], (finish.x, finish.y), finish.yaw)
            self.assertGreater(min(point[1] for point in polygon), sample.positions[0][1]+.06)
            protected = sample.reserved_regions[0]
            for index in (2, 3):
                polygon = task.footprint_polygon(domain.geometries[index], sample.positions[index][:2], sample.yaws[index])
                self.assertTrue(max(point[1] for point in polygon) < protected[2]
                                or min(point[1] for point in polygon) > protected[3])

    def test_domain_errors_fail_before_any_reset_draw(self):
        with self.assertRaisesRegex(ValueError, 'No positive continuous scene domain'):
            self.domain(role_cells=[])
        with self.assertRaisesRegex(ValueError, 'distinct active pool IDs'):
            self.domain(active_ids=(0, 1, 2, 0))
        domain = self.domain()
        cell = dict(domain.cells[0], parameter_bounds=(-.016, .015))
        with self.assertRaisesRegex(ValueError, 'inverse-visibility'):
            self.domain(role_cells=[cell])

    def test_same_seed_is_independent_of_other_draws(self):
        import random
        domain = self.domain()
        first = task.sample_certified_scene(domain, random.Random(123))
        task.sample_certified_scene(domain, random.Random(456))
        second = task.sample_certified_scene(domain, random.Random(123))
        self.assertEqual(first.positions, second.positions)
        self.assertEqual(first.yaws, second.yaws)
        self.assertEqual(first.camera_pose, second.camera_pose)

    def test_fixed_companion_and_pivot_use_the_same_continuous_assembly_transform(self):
        import random
        base = self.domain()
        geometry = (*base.geometries, task.ObjectGeometry('4', (-.025, -.03, 0.), (.025, .03, .10), '4'))
        cell = dict(base.cells[0], module_bounds=(-.03, .17, -.07, .20), goal_x=.07, goal_y=.14)
        domain = self.domain(geometries=geometry, rules=task.SpawnRules(min_objects=5, max_objects=5),
            active_ids=(0, 1, 2, 3, 4), role_cells=[cell], fixed_objects={4: (.14, .15, .3)},
            pivot=dict(pivot_offset_from_blocker=(-.025, -.03), turn_offset_from_blocker=(.005, -.055),
                       turn_angle=-math.pi/2))
        self.assertEqual(domain.background_ids, (2, 3))
        sample = task.sample_certified_scene(domain, random.Random(19))
        self.assertEqual(sample.clearance_kind, 'certified_guarded_pivot_translate')
        angle = sample.parameter_certificate['common_yaw']
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array(((c, -s), (s, c)))
        origin = np.array(sample.positions[0][:2])
        np.testing.assert_allclose(sample.positions[4][:2], origin+rotation @ (.14, .15), atol=1.e-12)
        first, second = sample.clearance_path
        relative = np.array((first.start.x, first.start.y))-first.pivot
        turn = np.array(((0., 1.), (-1., 0.))) @ relative+first.pivot
        np.testing.assert_allclose(turn, (first.end.x, first.end.y), atol=1.e-12)
        self.assertEqual(first.end, second.start)
        self.assertAlmostEqual(first.end.yaw-first.start.yaw, -math.pi/2)
        np.testing.assert_allclose((second.end.x, second.end.y), origin+rotation @ (.07, .14), atol=1.e-12)


class ObservationObbVisualizationTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        from unittest.mock import Mock
        class Env:
            pass
        class EventManager:
            pass
        class Simulation:
            def __init__(self):
                self.callbacks = {}
                self.add_count = 0
            def has_gui(self):
                return True
            def render_callback_exists(self, name):
                return name in self.callbacks
            def add_render_callback(self, name, callback):
                self.add_count += 1
                self.callbacks[name] = callback
            def remove_render_callback(self, name):
                del self.callbacks[name]
        base, _, _ = native_fixture(2)
        env = Env()
        env.__dict__.update(base.__dict__)
        env.sim, env.event_manager, env._is_closed = Simulation(), EventManager(), False
        env.cfg.role_obb_debug_vis = env.cfg.observation_obb_debug_vis = False
        env._contact_mdp = SimpleNamespace(geometry_b=env.task_scene_data.geometry_b.clone(),
            geometry_valid=env.task_scene_data.geometry_valid.clone(), bindings=object())
        # Invalid policy padding must never be drawn, including arbitrary NaN contents.
        env._contact_mdp.geometry_b[~env._contact_mdp.geometry_valid] = float('nan')
        draw, acquire, enable = Mock(), Mock(), Mock()
        acquire.return_value = draw
        modules = {name: types.ModuleType(name) for name in (
            'isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.extensions',
            'isaacsim.util', 'isaacsim.util.debug_draw')}
        modules['isaacsim.core.utils.extensions'].enable_extension = enable
        modules['isaacsim.util.debug_draw']._debug_draw = SimpleNamespace(acquire_debug_draw_interface=acquire)
        with patch.dict(sys.modules, modules):
            task.initialize_role_obb_visualization(env)
            try:
                yield env, draw, acquire, enable
            finally:
                # Native env.close() drops its event manager while the env is
                # still alive; exercise the same finalizer order in the fixture.
                del env.event_manager

    def test_observed_bounds_use_only_valid_initial_geometry_and_robot_world_transform(self):
        with self.fixture() as (env, draw, _, _):
            owner = env._role_obb_visualizer
            env._observation_obb_visualizer.set_debug_vis(True)
            state, robot = env._contact_mdp, env.scene['robot'].data
            robot.root_pos_w[:] = torch.tensor([[1., -2., .4], [-3., .5, .8]])
            robot.root_quat_w[1] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
            state.geometry_b[1, 0, 3:7] = torch.tensor([math.cos(.2), 0., 0., math.sin(.2)])
            actual = owner._observation_corners(state)
            expected = []
            bits = torch.tensor(list(product((-.5, .5), repeat=3)))
            for row, slot in state.geometry_valid.nonzero().tolist():
                geometry = state.geometry_b[row, slot]
                local = bits * geometry[7:10]
                corners_b = local @ rotation_matrix(geometry[3:7]).mT + geometry[:3]
                expected.append(corners_b @ rotation_matrix(robot.root_quat_w[row]).mT
                                + robot.root_pos_w[row])
            torch.testing.assert_close(actual, torch.stack(expected))
            self.assertEqual(actual.shape, (6, 8, 3))
            self.assertTrue(torch.isfinite(actual).all())
            # Privileged body poses can evolve while the observed snapshot stays fixed.
            env.scene['object_collection'].data.object_link_pose_w[:] = float('nan')
            torch.testing.assert_close(owner._observation_corners(state), actual, rtol=0, atol=0)
            owner.update()
            starts, ends, colors, widths = draw.draw_lines.call_args.args
            self.assertEqual(len(starts), 6 * 12)
            self.assertEqual(len(ends), 6 * 12)
            self.assertEqual(set(colors), {(0., .8, 1., 1.)})
            self.assertEqual(set(widths), {1.5})
            # A changed base transform re-expresses the same policy snapshot in world.
            robot.root_pos_w += torch.tensor([.1, -.2, .3])
            torch.testing.assert_close(owner._observation_corners(state), actual + torch.tensor([.1, -.2, .3]))

    def test_role_and_observation_toggles_share_one_callback_and_one_draw_interface(self):
        with self.fixture() as (env, draw, acquire, enable):
            owner, observed = env._role_obb_visualizer, env._observation_obb_visualizer
            self.assertIs(observed.cfg, owner.cfg)
            self.assertFalse(env.sim.callbacks)
            acquire.assert_not_called()
            enable.assert_not_called()
            observed.set_debug_vis(True)
            self.assertTrue(observed.debug_vis)
            self.assertTrue(env.cfg.observation_obb_debug_vis)
            self.assertEqual(len(env.sim.callbacks), 1)
            owner.set_debug_vis(True)
            self.assertEqual(env.sim.add_count, 1)
            self.assertEqual(len(draw.draw_lines.call_args.args[0]), (6 + 4) * 12)
            observed.set_debug_vis(False)
            self.assertTrue(owner.debug_vis)
            self.assertEqual(len(env.sim.callbacks), 1)
            self.assertEqual(len(draw.draw_lines.call_args.args[0]), 4 * 12)
            owner.set_debug_vis(False)
            self.assertFalse(env.sim.callbacks)
            observed.set_debug_vis(True)
            self.assertEqual(env.sim.add_count, 2)
            self.assertEqual(len(env.sim.callbacks), 1)
            acquire.assert_called_once()
            enable.assert_called_once_with('isaacsim.util.debug_draw')
            # Startup may be invoked again; it must preserve both existing adapters.
            task.initialize_role_obb_visualization(env)
            self.assertIs(env._role_obb_visualizer, owner)
            self.assertIs(env._observation_obb_visualizer, observed)

    def test_unobserved_snapshot_draws_no_lines_and_headless_creates_no_resources(self):
        with self.fixture() as (env, draw, _, _):
            env._contact_mdp.geometry_valid[:] = False
            env._observation_obb_visualizer.set_debug_vis(True)
            self.assertEqual(draw.draw_lines.call_args.args, ([], [], [], []))
        env = SimpleNamespace(sim=SimpleNamespace(has_gui=lambda: False))
        with patch.object(task, 'RoleObbVisualizer', side_effect=AssertionError('Headless drawing allocation')), \
             patch.object(task, 'ObservationObbVisualizer', side_effect=AssertionError('Headless adapter allocation')):
            task.initialize_role_obb_visualization(env)
        self.assertFalse(hasattr(env, '_role_obb_visualizer'))
        self.assertFalse(hasattr(env, '_observation_obb_visualizer'))


class ObservationCapacityTests(unittest.TestCase):
    """Physical occluders and fixed policy capacity are independent contracts."""

    @classmethod
    def setUpClass(cls):
        GeometricVisionTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        GeometricVisionTests.tearDownClass.__func__(cls)

    fixture = GeometricVisionTests.fixture

    def test_forty_recognized_objects_select_eight_nearest_visible_role_centers(self):
        sensor, *_ = self.fixture()
        sensor.object_count = 400
        sensor.cfg.min_depth_pixels = 4
        sensor.env = SimpleNamespace(cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=10)))
        centers = torch.stack((torch.linspace(-3., 15., 40), torch.zeros(40), torch.ones(40)), -1)
        centers[30, 0], centers[35, 0] = 0., 10.
        corners = torch.tensor(list(product((-.01, .01), repeat=3)))
        points = (centers[:, None] + corners).reshape(1, -1, 3)
        active = (torch.arange(40, dtype=torch.int32) * 7 + 3)[None]
        labels = active.repeat_interleave(8, dim=1)
        roles = active[:, [30, 35]]
        with patch.object(self.module, 'fit_visible_obbs', wraps=self.module.fit_visible_obbs) as fit:
            geometry, valid, observed = sensor._fit_observation(points, labels, active, roles)
        self.assertEqual(fit.call_args.args[2], 40)
        self.assertEqual(geometry.shape, (1, 10, 10))
        self.assertTrue(valid.all())
        expected = sorted((index for index in range(40) if index not in (30, 35)),
                          key=lambda index: (min(abs(float(centers[index, 0])),
                                                 abs(float(centers[index, 0]) - 10.)), index))[:8]
        self.assertEqual(observed[0].tolist(), active[0, [30, 35] + expected].tolist())
        torch.testing.assert_close(geometry[0, :, :3], centers[[30, 35] + expected], atol=2.e-6, rtol=1.e-5)

    def test_hidden_objects_cannot_be_selected_and_missing_role_uses_only_visible_reference(self):
        geometry = scene(batch=2, objects=40)
        geometry[:, 1, 0] = 100.
        geometry[:, 2:, 0] = torch.arange(38, dtype=geometry.dtype) + 70.
        valid = torch.ones(2, 40, dtype=torch.bool)
        valid[:, [0, 39]] = False
        # Arbitrary hidden geometry, even NaN, is not a source for ranking.
        geometry[0, 0] = float('nan')
        geometry[1, 0, :3] = -1.e9
        geometry[:, 39, 0] = 100.001
        result, result_valid = obs.prepare_initial_geometry(
            geometry, valid, torch.tensor([0, 0]), torch.tensor([1, 1]), capacity=10)
        self.assertFalse(result_valid[:, 0].any())
        self.assertEqual(int(result[:, 0].count_nonzero()), 0)
        self.assertTrue(result_valid[:, 1:].all())
        torch.testing.assert_close(result[0], result[1])
        torch.testing.assert_close(result[0, 2:, 0], torch.tensor(
            [100., 99., 101., 98., 102., 97., 103., 96.], dtype=result.dtype))

    def test_both_unobserved_roles_leave_slots_reserved_and_use_stable_observed_order(self):
        geometry = scene(batch=1, objects=40)
        valid = torch.ones(1, 40, dtype=torch.bool)
        valid[:, [0, 1, 4]] = False
        result, result_valid = obs.prepare_initial_geometry(
            geometry, valid, torch.tensor([0]), torch.tensor([1]), capacity=10)
        self.assertFalse(result_valid[:, :2].any())
        torch.testing.assert_close(result[0, 2:, 0], torch.tensor(
            [2., 3., 5., 6., 7., 8., 9., 10.], dtype=result.dtype))

    def test_twenty_and_forty_active_objects_share_batched_fit_without_selecting_padding(self):
        sensor, *_ = self.fixture()
        sensor.object_count = 40
        sensor.cfg.min_depth_pixels = 4
        sensor.env = SimpleNamespace(cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=10)))
        centers = torch.stack((torch.linspace(0., 4., 40), torch.zeros(40), torch.ones(40)), -1)
        corners = torch.tensor(list(product((-.01, .01), repeat=3)))
        points = (centers[:, None] + corners).reshape(1, -1, 3).expand(2, -1, -1).clone()
        active = torch.arange(40, dtype=torch.int32)[None].expand(2, -1).clone()
        active[0, 20:] = -1
        labels = active.repeat_interleave(8, dim=1)
        geometry, valid, observed = sensor._fit_observation(
            points, labels, active, torch.tensor([[0, 19], [0, 39]], dtype=torch.int32))
        self.assertEqual(geometry.shape, (2, 10, 10))
        self.assertTrue(valid.all())
        self.assertTrue((observed[0] < 20).all())
        self.assertEqual(observed[:, :2].tolist(), [[0, 19], [0, 39]])
        self.assertEqual(len(set(observed[0].tolist())), 10)
        self.assertEqual(len(set(observed[1].tolist())), 10)

    def test_fortieth_physical_body_still_occludes_when_observation_capacity_is_ten(self):
        sensor, original, _, eyes, rotations, roles = self.fixture()
        count = 40
        sensor.object_count = count
        sensor.mesh_ids = self.wp.array([sensor.meshes[0].id] * count, dtype=self.wp.uint64, device='cpu')
        bounds = sensor.mesh_bounds.numpy()[0]
        sensor.mesh_bounds = self.wp.array(np.stack([bounds] * count), dtype=self.wp.vec3, device='cpu')
        poses = torch.zeros(1, count, 7)
        poses[..., 0], poses[..., 2], poses[..., 3] = 100., 2., 1.
        poses[:, :2] = original[:, :2]
        poses[:, 1, 0] = .3
        poses[:, 39, 0], poses[:, 39, 2] = .08, .5
        class Scene(dict):
            env_origins = torch.zeros(1, 3)
        scene_data = Scene(object_collection=SimpleNamespace(data=SimpleNamespace(object_link_pose_w=poses)),
                           robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=eyes, root_quat_w=rotations)))
        sensor.env = SimpleNamespace(scene=scene_data, device='cpu',
                                     cfg=SimpleNamespace(mdp=SimpleNamespace(max_objects=10)))
        request = dict(env_id=0, active_ids=tuple(range(count)), target_id=0, blocker_id=1)
        with patch.object(sensor, '_trace', wraps=sensor._trace) as trace:
            snapshot = sensor.observe_at_viewpoint([request], torch.cat((eyes, rotations), -1), retain_images=True)[0]
        self.assertEqual(trace.call_args.args[1].shape, (1, 40))
        self.assertEqual(snapshot.geometry_b.shape, (10, 10))
        self.assertIn(39, snapshot.observed_ids.tolist())
        self.assertFalse(snapshot.geometry_valid[:2].any())
        self.assertEqual(int(snapshot.role_masks.count_nonzero()), 0)
        request['active_ids'] = tuple(range(count - 1))
        exposed = sensor.observe_at_viewpoint([request], torch.cat((eyes, rotations), -1))[0]
        self.assertTrue(exposed.geometry_valid[:2].all())
        self.assertNotIn(39, exposed.observed_ids.tolist())


class PhysicalStateCapacityTests(unittest.TestCase):
    """Every active physical object stays in privileged costs and failure checks."""

    @staticmethod
    def fixture():
        env, action, external = native_fixture(3)
        env.cfg.spawn_rules = SimpleNamespace(max_objects=40)
        objects = env.scene['object_collection'].data
        origins = env.scene.env_origins
        poses = torch.zeros(3, 40, 7)
        poses[..., :3] = origins[:, None]
        poses[..., 0] += torch.arange(40) * .03 - .7
        poses[..., 1] += .3
        poses[..., 2] += .05
        poses[..., 3] = 1.
        objects.object_link_pose_w = poses
        objects.object_com_pos_w = poses[..., :3].clone()
        objects.object_com_pos_b = torch.zeros(3, 40, 3)
        objects.object_com_vel_w = torch.zeros(3, 40, 6)
        source = env.task_scene_data
        source.object_ids = torch.arange(40)[None].expand(3, -1).clone()
        source.object_ids[0, 20:] = -1
        source.lower = torch.full((40, 3), -.05)
        source.upper = torch.full((40, 3), .05)
        source.upright = torch.tensor([1., 0., 0., 0.]).expand(40, -1).clone()
        # The three recognized objects are IDs 3, 2, 0. In particular, object31
        # has no observation slot; its physical state must still be evaluated.
        return env, action, external

    def test_native_bindings_hold_forty_physical_objects_and_ten_observation_slots(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        state = env._contact_mdp
        self.assertEqual(state.active_capacity, 40)
        self.assertEqual(state.active.shape, (3, 40))
        self.assertEqual(state.bindings.local_corners.shape, (3, 40, 8, 3))
        self.assertEqual(state.active.shape, (3, 40))
        self.assertEqual(state.support_loss_count.shape, (3, 40))
        self.assertEqual(state.geometry_b.shape, (3, 10, 10))
        self.assertEqual(obs.policy_observation(env).shape, (3, 244))
        self.assertEqual(state.active.sum(-1).tolist(), [20, 40, 40])
        self.assertFalse((env.task_scene_data.observed_ids == 31).any())

    def test_more_physical_objects_do_not_expand_critic_or_expose_unrecognized_geometry(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        before = obs.critic_scene_geometry(env).clone()
        self.assertEqual(before.shape, (3, 100))
        geometry = before.reshape(3, 10, 10)
        self.assertEqual((geometry[..., 7:] > 0).all(-1).sum(-1).tolist(), [3, 3, 3])
        # Object 31 is active in two rows, but was never initially recognized.
        env.scene['object_collection'].data.object_link_pose_w[1:, 31, 0] += .04
        env.common_step_counter = 1
        torch.testing.assert_close(obs.critic_scene_geometry(env), before, rtol=0, atol=0)
        self.assertEqual(obs.policy_observation(env).shape, (3, 244))

    def test_unobserved_object31_velocity_is_costed_but_padding_and_blocker_are_not(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        velocities = env.scene['object_collection'].data.object_com_vel_w
        velocities[0, 31] = float('nan')  # Padding in the 20-object row.
        velocities[1, 31, 0] = .4
        velocities[2, 0, 0] = .4  # Designated Blocker remains exempt.
        env.common_step_counter = 1
        state = task.get_task_state(env)
        torch.testing.assert_close(rewards.object_collision(env), torch.tensor([0., .4, 0.]),
                                   atol=1.e-6, rtol=1.e-5)
        self.assertFalse(state.failure_reasons[0, task.FAILURE_REASON_NAMES.index('invalid_state')])
        self.assertTrue(torch.isfinite(obs.policy_observation(env)).all())

    def test_unobserved_object31_support_loss_and_invalid_state_still_terminate(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[0, 31] = float('nan')  # Inactive objects do not fail.
        poses[1, 31, 2] -= .12
        poses[2, 31, 0] = float('nan')
        env.common_step_counter = 1
        state = task.get_task_state(env)
        loss = task.FAILURE_REASON_NAMES.index('object_support_lost')
        invalid = task.FAILURE_REASON_NAMES.index('invalid_state')
        self.assertFalse(state.failure[0])
        self.assertTrue(state.failure_reasons[1, loss])
        self.assertTrue(state.failure_reasons[2, invalid])
        self.assertEqual(state.failure.tolist(), [False, True, True])
        self.assertTrue(torch.isfinite(state.clearance_quality).all() and torch.isfinite(state.terminal_reward).all())

    def test_forty_object_transition_does_not_read_tensor_scalars(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        state = env._contact_mdp
        env.scene['object_collection'].data.object_link_pose_w[1, 31, 0] += .01
        batch = task._read_transition(env, state)
        def reject_scalar(*args, **kwargs):
            raise AssertionError('Per-step tensor scalar read or host synchronization')
        with patch.object(torch.Tensor, '__bool__', reject_scalar), patch.object(torch.Tensor, 'item', reject_scalar):
            state.advance(batch, 1)
        torch.testing.assert_close(state.object_motion_cost, torch.zeros(3))

    def test_partial_reset_restores_only_selected_physical_and_policy_rows(self):
        env, _, _ = self.fixture()
        task.reset_task(env)
        poses = env.scene['object_collection'].data.object_link_pose_w
        poses[1, 31, 0] += .04
        poses[2, 31, 2] -= .12
        env.common_step_counter = 1
        before_observation = obs.policy_observation(env).clone()
        state = env._contact_mdp
        preserved = {name: getattr(state, name)[[0, 2]].clone() for name in (
            'geometry_b', 'geometry_valid', 'active', 'initial_center', 'previous_wrench',
            'history', 'failure_reasons', 'support_loss_count', 'force_growth')}
        task.reset_task(env, torch.tensor([1]))
        self.assertEqual(obs.policy_observation(env).shape, (3, 244))
        torch.testing.assert_close(state.observation[[0, 2]], before_observation[[0, 2]], rtol=0, atol=0)
        for name, value in preserved.items():
            torch.testing.assert_close(getattr(state, name)[[0, 2]], value, rtol=0, atol=0)
        self.assertEqual(float(state.force_growth[1]), 0.)
        torch.testing.assert_close(state.initial_center[1, 31], poses[1, 31, :3] - env.scene.env_origins[1])
        self.assertFalse(state.history[1, 1].any())
        self.assertEqual(state.active.sum(-1).tolist(), [20, 40, 40])


class ClutterPositionMixTests(unittest.TestCase):
    @staticmethod
    def square(half=.025):
        return np.array([[-half, -half], [half, -half], [half, half], [-half, half]])

    def assert_safe(self, local, centers, fixed, bounds, gap):
        for index, polygon in enumerate(local):
            world = polygon + centers[index]
            self.assertGreaterEqual(world[:, 0].min(), bounds[0] - 1.e-10)
            self.assertLessEqual(world[:, 0].max(), bounds[1] + 1.e-10)
            self.assertGreaterEqual(world[:, 1].min(), bounds[2] - 1.e-10)
            self.assertLessEqual(world[:, 1].max(), bounds[3] + 1.e-10)
            for obstacle in fixed:
                self.assertGreaterEqual(task.polygon_distance(world, obstacle), gap - 1.e-9)
            for other in range(index):
                self.assertGreaterEqual(task.polygon_distance(world, local[other] + centers[other]), gap - 1.e-9)

    def test_forty_rotated_offset_footprints_preserve_all_collisions_and_guards(self):
        import random
        local = []
        for index in range(40):
            angle = index * .713
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            local.append((np.array([[-.016, -.012], [.017, -.012], [.017, .014], [-.016, .014]])
                          + (.006, -.004)) @ rotation.T)
        centers = np.array([(x, y) for x in np.linspace(-.4, .4, 5) for y in np.linspace(-.35, .35, 8)])
        original = centers.copy()
        fixed, bounds = [self.square(.02)], (-.5, .5, -.4, .4)
        for seed in (2, 17, 29):
            result, provenance = task._mix_clutter_positions(local, centers, bounds, fixed, .003, random.Random(seed))
            self.assert_safe(local, result, fixed, bounds, .003)
            self.assertEqual(provenance['coordinate_draws'], 240)
            self.assertEqual(provenance['pose_proposals'], 0)
            self.assertEqual(provenance['approval_retries'], 0)
        np.testing.assert_array_equal(centers, original)
        repeated, _ = task._mix_clutter_positions(local, centers, bounds, fixed, .003, random.Random(29))
        np.testing.assert_array_equal(repeated, result)

    def test_precomputed_line_intervals_match_independent_upper_and_lower_oracles(self):
        import random
        local = [self.square(.02), self.square(.035) + (.009, -.005)]
        fixed = [np.array([[-.04, -.02], [.03, -.03], [.04, .025], [-.02, .04]])]
        captured, original = {}, task._clutter_position_sweeps
        def capture(*args):
            captured['args'] = args
            return original(*args)
        with patch.object(task, '_clutter_position_sweeps', side_effect=capture):
            task._mix_clutter_positions(local, [[-.2, -.15], [.2, .15]], (-.5, .5, -.4, .4),
                                        fixed, .003, random.Random(3))
        _, _, _, packed, offsets, bounds, _, _, _, gap = captured['args']
        for current in range(2):
            for other in range(3):
                if current == other:
                    continue
                obstacle = local[other] if other < 2 else fixed[0]
                for coordinate in np.linspace(-.119, .117, 31):
                    lower, upper = task._clutter_line_interval(
                        packed, *offsets[current, other], bounds[current, other], 1, coordinate, gap)
                    oracle_upper = task.footprint_line_limit(local[current], obstacle, float(coordinate), gap)
                    oracle_lower = task.footprint_line_limit(local[current] * (1., -1.),
                                                           obstacle * (1., -1.), float(coordinate), gap)
                    if oracle_upper is None:
                        self.assertGreater(lower, upper)
                    else:
                        self.assertAlmostEqual(upper, oracle_upper, places=10)
                        self.assertAlmostEqual(lower, -oracle_lower, places=10)

    def test_free_union_can_move_background_to_either_side_of_role_obstacle(self):
        import random
        sides = set()
        for seed in range(40):
            result, _ = task._mix_clutter_positions([self.square(.02)], [[.2, 0.]],
                (-.4, .4, -.3, .3), [self.square(.05)], .003, random.Random(seed))
            self.assert_safe([self.square(.02)], result, [self.square(.05)], (-.4, .4, -.3, .3), .003)
            if abs(result[0, 0]) > .08:
                sides.add(int(np.sign(result[0, 0])))
        self.assertEqual(sides, {-1, 1})

    def test_exact_gap_tangency_and_isolated_supported_centers_are_valid(self):
        import random
        local = [self.square(), self.square()]
        centers = np.array([[-.0265, 0.], [.0265, 0.]])
        result, provenance = task._mix_clutter_positions(local, centers,
            (-.0515, .0515, -.025, .025), (), .003, random.Random(4))
        np.testing.assert_allclose(result, centers, atol=1.e-12, rtol=0)
        self.assert_safe(local, result, (), (-.0515, .0515, -.025, .025), .003)
        self.assertEqual(provenance['changed_coordinates'], 0)

    def test_invalid_seed_raises_without_pose_repair(self):
        import random
        for centers in ([[0., 0.], [0., 0.]], [[2., 0.], [0., 0.]]):
            with self.assertRaisesRegex(ValueError, 'Clutter mixing'):
                task._mix_clutter_positions([self.square(), self.square()], centers,
                    (-.5, .5, -.4, .4), (), .003, random.Random(2))

    def test_canonical_obb_center_bounds_are_preserved_during_every_coordinate_draw(self):
        import random
        bounds = np.asarray([[-.12, -.08, -.15, .15], [.08, .12, -.15, .15]])
        for seed in range(12):
            centers, _ = task._mix_clutter_positions([self.square(.02)]*2,
                [[-.1, 0.], [.1, 0.]], (-.5, .5, -.4, .4), (), .003,
                random.Random(seed), center_bounds=bounds)
            self.assertTrue((centers >= bounds[:, [0, 2]]-1.e-12).all())
            self.assertTrue((centers <= bounds[:, [1, 3]]+1.e-12).all())
            self.assert_safe([self.square(.02)]*2, centers, (), (-.5, .5, -.4, .4), .003)

    def test_hulls_are_prepared_once_per_input_polygon_not_per_coordinate_draw(self):
        import random
        with patch('scipy.spatial.ConvexHull', wraps=ConvexHull) as hull:
            _, provenance = task._mix_clutter_positions([self.square(), self.square()],
                [[-.2, -.15], [.2, .15]], (-.5, .5, -.4, .4), [self.square(.03)], .003,
                random.Random(9), sweeps=5)
        self.assertEqual(hull.call_count, 3)
        self.assertEqual(provenance['coordinate_draws'], 20)


class ClutterRoleRulesTests(unittest.TestCase):
    def test_type_selection_precedes_capacity_and_every_count_prefix_is_balanced(self):
        import random
        geometry = tuple(task.ObjectGeometry(f'{kind}_{copy}', (-.025, -.03, 0.), (.025, .03, .10), str(kind))
                         for kind in range(10) for copy in range(13))
        rules = task.SpawnRules(min_objects=20, max_objects=40)
        for seed in range(40):
            sequence, types = task._clutter_composition_sequence(geometry, rules, random.Random(seed), 4, 21)
            self.assertEqual(sequence[:2], (4, 21))
            self.assertEqual(len(sequence), len(set(sequence)))
            for count in range(20, 41):
                assets = [geometry[index].asset_name for index in sequence[:count]]
                self.assertEqual(assets.count('0'), 1)
                self.assertEqual(set(assets), set(types))
                amounts = [assets.count(kind) for kind in types[1:]]
                self.assertLessEqual(max(amounts)-min(amounts), 1)

    def test_role_sampling_drops_old_full_route_and_slab_bounds(self):
        import random
        base = CertifiedSceneDomainTests().domain()
        family = dict(target_yaw=base.target_yaw, blocker_yaw=base.blocker_yaw,
                      blocker_x=base.blocker_x, camera_pose=base.camera_pose, cells=base.cells)
        def sample(value):
            catalog = SimpleNamespace(families={('0', '1'): (value,)}, load_visibility=lambda _: base.visibility)
            return task._sample_clutter_roles(catalog, base.geometries, base.rules, random.Random(7), 0, 1)
        first = sample(family)
        changed = dict(family, cells=[dict(cell, module_bounds=(-100., 100., -100., 100.),
            goal_y=200., x_translation_bounds=(100., 101.), y_translation_bounds=(100., 101.))
            for cell in family['cells']])
        second = sample(changed)
        self.assertEqual(first.positions, second.positions)
        self.assertEqual(first.camera_pose, second.camera_pose)
        self.assertFalse(first.clearance_path)
        self.assertFalse(first.reserved_regions)
        self.assertFalse(first.parameter_certificate['full_route_reserved'])
        self.assertEqual(first.parameter_certificate['predicted_role_visibility'], (.6, 1.))
        with patch.object(task, '_path_sweeps', side_effect=AssertionError('full path was reserved')):
            self.assertEqual(len(task._clutter_protection_polygons(first, base.geometries)), 4)

    def test_initial_cells_keep_supported_roles_and_front_camera_over_continuous_parameters(self):
        import random
        base = CertifiedSceneDomainTests().domain()
        family = dict(target_yaw=base.target_yaw, blocker_yaw=base.blocker_yaw,
                      blocker_x=base.blocker_x, camera_pose=base.camera_pose, cells=base.cells)
        catalog = SimpleNamespace(families={('0', '1'): (family,)}, load_visibility=lambda _: base.visibility)
        for seed in range(64):
            sample = task._sample_clutter_roles(catalog, base.geometries, base.rules, random.Random(seed), 0, 1)
            for index in sample.active_ids:
                polygon = task._collision_polygon(base.geometries[index], sample.yaws[index])+sample.positions[index][:2]
                self.assertTrue(task._supported(tuple(map(tuple, polygon)), base.rules))
            self.assertGreaterEqual(sample.camera_pose[0]-base.rules.x_range[1], .55-1.e-10)
            self.assertLessEqual(sample.camera_pose[0]-base.rules.x_range[1], .95+1.e-10)
            self.assertLessEqual(abs(sample.camera_pose[1]), .35+1.e-10)
            self.assertAlmostEqual(sum(value*value for value in sample.camera_pose[3:]), 1., places=12)


class ClutterGpuSeedTests(unittest.TestCase):
    @staticmethod
    def square(x, y=None, offset=(0., 0.)):
        y = x if y is None else y
        return np.array([[-x, -y], [x, -y], [x, y], [-x, y]]) + offset

    def test_empty_input_and_invalid_contract_do_not_launch_cuda(self):
        with patch.object(torch.fft, 'rfft2', side_effect=AssertionError('Unexpected FFT')):
            centers, variants, provenance = task._clutter_gpu_seed(
                (), (), (), (-.1, .1, -.1, .1), .003, device='cuda')
            self.assertEqual(centers.shape, (0, 2))
            self.assertEqual(variants.shape, (0,))
            self.assertEqual(provenance['maximum_greedy_prefix'], 0)
            with self.assertRaises(ValueError):
                task._clutter_gpu_seed((), (), (), (-.1, .1, -.1, .1), .003, device='cpu')
            with self.assertRaises(ValueError):
                task._clutter_gpu_seed(((self.square(.02),),), (), (), (-.1, .1, -.1, .1), .003, device='cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU free-domain seed')
    def test_first_failed_body_ends_balanced_prefix_without_skipping_types(self):
        small, impossible = self.square(.015), self.square(.5)
        polygons = ((small,), (impossible,), (small,))
        centers, variants, provenance = task._clutter_gpu_seed(
            polygons, polygons, (), (-.1, .1, -.1, .1), .003, device='cuda', resolution=.0025)
        self.assertEqual(provenance['maximum_greedy_prefix'], 1)
        self.assertEqual(variants.tolist(), [0, -1, -1])
        self.assertTrue(np.isfinite(centers[0]).all())
        self.assertTrue(np.isnan(centers[1:]).all())
        self.assertEqual(provenance['placement_queries'], 2)
        self.assertEqual(provenance['approval_retries'], 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU free-domain seed')
    def test_missing_mandatory_body_keeps_other_core_rows_and_stops_optional_tail(self):
        small, impossible = self.square(.015), self.square(.5)
        polygons = ((small,), (impossible,), (small,))
        centers, variants, provenance = task._clutter_gpu_seed(
            polygons, polygons, (), (-.1, .1, -.1, .1), .003, device='cuda', resolution=.0025,
            minimum_count=2)
        self.assertEqual(provenance['processing_order'], [1, 0, 2])
        self.assertEqual(provenance['mandatory_count'], 2)
        self.assertEqual(provenance['mandatory_seeded_count'], 1)
        self.assertEqual(provenance['seeded_count'], 1)
        self.assertEqual(variants.tolist(), [0, -1, -1])
        self.assertTrue(np.isfinite(centers[0]).all())
        self.assertTrue(np.isnan(centers[1:]).all())
        self.assertEqual(provenance['placement_queries'], 2)
        self.assertTrue(provenance['row_ids_preserved'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU free-domain seed')
    def test_uninflated_body_can_fill_support_without_artificial_wall_gap(self):
        polygon = self.square(.025)
        centers, variants, provenance = task._clutter_gpu_seed(((polygon,),), ((polygon,),), (),
            (-.025, .025, -.025, .025), .003, device='cuda', resolution=.0005)
        self.assertEqual(provenance['maximum_greedy_prefix'], 1)
        np.testing.assert_allclose(centers, [[0., 0.]], atol=1.e-12, rtol=0)
        self.assertEqual(variants.tolist(), [0])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU free-domain seed')
    def test_full_obb_support_uses_separate_bounds_and_retains_noncentral_origin(self):
        collider, obb = self.square(.01), self.square(.015, offset=(.05, 0.))
        centers, variants, provenance = task._clutter_gpu_seed(((collider,),), ((obb,),), (),
            (-.1, .1, -.1, .1), .003, device='cuda', obb_support_bounds=(-.02, .02, -.03, .03),
            resolution=.001)
        self.assertEqual(provenance['maximum_greedy_prefix'], 1)
        actual, support = collider + centers[0], obb + centers[0]
        self.assertGreaterEqual(actual[:, 0].min(), -.1 - 1.e-10)
        self.assertLessEqual(actual[:, 0].max(), .1 + 1.e-10)
        self.assertGreaterEqual(support[:, 0].min(), -.02 - 1.e-10)
        self.assertLessEqual(support[:, 0].max(), .02 + 1.e-10)
        self.assertGreaterEqual(support[:, 1].min(), -.03 - 1.e-10)
        self.assertLessEqual(support[:, 1].max(), .03 + 1.e-10)
        self.assertLess(centers[0, 0], -.04)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for GPU free-domain seed')
    def test_fixed_guard_and_all_pair_gaps_pass_independent_polygon_distance(self):
        polygon = self.square(.023, .016, offset=(.004, -.003))
        rotated = polygon @ np.array([[0., 1.], [-1., 0.]])
        variants = tuple((polygon, rotated) for _ in range(8))
        guard = self.square(.04)
        centers, selected, provenance = task._clutter_gpu_seed(variants, variants, (guard,),
            (-.2, .2, -.3, .3), .003, device='cuda', resolution=.001)
        self.assertEqual(provenance['maximum_greedy_prefix'], 8)
        world = [variants[i][int(which)] + centers[i] for i, which in enumerate(selected)]
        for index, actual in enumerate(world):
            self.assertGreaterEqual(task.polygon_distance(actual, guard), .003 - 1.e-9)
            for other in world[:index]:
                self.assertGreaterEqual(task.polygon_distance(actual, other), .003 - 1.e-9)
        self.assertLess(provenance['fft_roundtrip_max_error'], .1)


class ClutterJointDomainTests(unittest.TestCase):
    @staticmethod
    def square(half):
        return np.asarray(((-half, -half), (half, -half), (half, half), (-half, half)))

    def solve(self, polygons, seed, *, seed_variants=None, walls=None, pocket=None, pocket_walls=None,
              fixed=(), physical=(), access=(.3, .5, .3, .5), minimum=1, obb_bounds=None,
              translation=None):
        variants = tuple((polygon, polygon) for polygon in polygons)
        wall_variants = variants if walls is None else tuple((polygon, polygon) for polygon in walls)
        pocket = self.square(.08) if pocket is None else pocket
        pocket_walls = pocket if pocket_walls is None else pocket_walls
        ids = np.zeros(len(polygons), dtype=np.int64) if seed_variants is None else seed_variants
        return task._solve_clutter_joint_domain(
            variants, wall_variants, seed, ids, fixed, physical, (pocket, pocket),
            (pocket_walls, pocket_walls), (-.5, .5, -.5, .5), access, .003,
            minimum_count=minimum, obb_support_bounds=obb_bounds, role_translation_bounds=translation)

    def assert_separated(self, first, second):
        self.assertGreaterEqual(task.polygon_distance(tuple(map(tuple, first)), tuple(map(tuple, second))), .003-1.e-8)

    def test_empty_pocket_reuses_view_void_while_backgrounds_preserve_it(self):
        body = self.square(.04)
        role = np.asarray(((-.49, -.05), (-.41, -.05), (-.41, .05), (-.49, .05)))
        view = np.asarray(((0., -.5), (.5, -.5), (.5, .5), (0., .5)))
        pocket_shape = self.square(.3)
        centers, _, pocket, _, stats = self.solve(
            [body, body], [[-.35, -.4], [-.35, .4]], fixed=(role, view), physical=(role,),
            pocket=pocket_shape, minimum=2)
        self.assertEqual(len(centers), 2)
        for center in centers:
            self.assert_separated(body+center, view)
            self.assert_separated(body+center, pocket_shape+pocket)
        self.assertGreater(task.overlap_area(tuple(map(tuple, pocket_shape+pocket)), tuple(map(tuple, view))), 0.)
        self.assertEqual(stats['scene_redraws'], 0)
        self.assertEqual(stats['approval_retries'], 0)

    def test_full_canonical_obb_enforces_wall_containment_separately_from_collider(self):
        physical, wall = self.square(.02), self.square(.18)
        centers, _, _, _, _ = self.solve([physical], [[.4, 0.]], walls=[wall])
        self.assertLessEqual(float(np.abs(centers[0]).max()), .32+1.e-8)
        self.assertGreater(float(np.linalg.norm(centers[0]-[.4, 0.])), .079)

    def test_final_pocket_obb_must_clear_access_even_when_its_collider_is_smaller(self):
        body, pocket, wall = self.square(.04), self.square(.02), self.square(.15)
        centers, _, center, _, _ = self.solve([body], [[-.4, -.4]], pocket=pocket,
                                             pocket_walls=wall, access=(-.1, .1, -.1, .1))
        access = np.asarray(task.rectangle_polygon((-.1, .1, -.1, .1)))
        self.assert_separated(wall+center, access)
        self.assert_separated(body+centers[0], pocket+center)

    def test_capacity_can_drop_only_balanced_tail_and_keeps_mandatory_prefix(self):
        body, pocket = self.square(.225), self.square(.24)
        seed = [[-.25, -.25], [-.25, .25], [.25, -.25], [.25, .25]]
        centers, variants, pc, _, stats = self.solve([body]*4, seed, pocket=pocket,
                                                   access=(.47, .5, .47, .5), minimum=2)
        self.assertGreaterEqual(len(centers), 2)
        self.assertLessEqual(len(centers), 3)
        self.assertEqual(variants.shape, (len(centers),))
        self.assertEqual(stats['active_background'], len(centers))
        for index, center in enumerate(centers):
            self.assert_separated(body+center, pocket+pc)
            for other in centers[index+1:]:
                self.assert_separated(body+center, body+other)

    def test_unpacked_mandatory_object_is_solved_in_same_joint_domain(self):
        body = self.square(.05)
        centers, _, pocket, _, stats = self.solve(
            [body, body], [[-.3, 0.], [np.nan, np.nan]], seed_variants=[0, -1], minimum=2)
        self.assertEqual(len(centers), 2)
        self.assertEqual(stats['seeded_background'], 1)
        self.assert_separated(body+centers[0], body+centers[1])
        self.assert_separated(body+centers[1], self.square(.08)+pocket)

    def test_infeasible_minimum_raises_without_lowering_count_or_restarting_solver(self):
        from scipy.optimize import milp
        body = self.square(.4)
        with patch('scipy.optimize.milp', wraps=milp) as solver:
            with self.assertRaisesRegex(RuntimeError, 'minimum=2'):
                self.solve([body, body], [[np.nan, np.nan]]*2, seed_variants=[-1, -1], minimum=2)
        self.assertEqual(solver.call_count, 1)

    def test_joint_role_translation_opens_background_domain_without_changing_visibility_relation(self):
        body = self.square(.15)
        view = np.asarray(((-.5, -.5), (.3, -.5), (.3, .5), (-.5, .5)))
        with self.assertRaises(RuntimeError):
            self.solve([body], [[np.nan, np.nan]], seed_variants=[-1], fixed=(view,))
        centers, _, _, _, stats = self.solve(
            [body], [[np.nan, np.nan]], seed_variants=[-1], fixed=(view,),
            translation=((- .2, -.2), (0., 0.)))
        self.assert_separated(body+centers[0], view+[-.2, 0.])
        np.testing.assert_allclose(stats['role_translation'], [-.2, 0.], atol=1.e-12)

    def test_access_front_remains_fixed_when_role_translation_moves_its_rear_plane(self):
        # Correct A0 becomes [-.3,.1] across the whole shelf width. Neither
        # side fits this 0.5m pocket. Translating xmax incorrectly to -.1
        # would create a false 0.6m right-hand parking region.
        with self.assertRaises(RuntimeError):
            self.solve([], np.empty((0, 2)), minimum=0, pocket=self.square(.25),
                       access=(-.1, .1, -.5, .5), translation=((- .2, -.2), (0., 0.)))

    def test_solver_wall_residual_does_not_invalidate_strict_coordinate_mixing(self):
        from scipy.optimize import milp
        import random
        body, wall, pocket_shape = self.square(.04), self.square(.1), self.square(.08)

        def slightly_inexact(*args, **kwargs):
            result = milp(*args, **kwargs)
            center = result.x[:2]
            lower, upper = center+.4, .4-center
            axis = int(np.minimum(lower, upper).argmin())
            direction = 1. if upper[axis] < lower[axis] else -1.
            # Measured in the original bottle/sugar reset failure: 0.364um.
            result.x[axis] += direction*3.635464610463046e-7
            return result

        with patch('scipy.optimize.milp', side_effect=slightly_inexact):
            centers, _, pc, _, stats = self.solve([body], [[.4, 0.]], walls=[wall], pocket=pocket_shape)
        self.assertGreater(float((.4-np.abs(centers[0])).min()), 5.e-6)
        mixed, _ = task._mix_clutter_positions(
            [body], centers, (-.5, .5, -.5, .5), (pocket_shape+pc,), .003,
            random.Random(4), sweeps=1, center_bounds=np.asarray(((-.4, .4, -.4, .4),)))
        self.assertTrue(np.isfinite(mixed).all())
        self.assertLessEqual(float(np.abs(mixed[0]).max()), .4+1.e-10)
        self.assertEqual(stats['numerical_wall_guard'], 1.e-5)

    def test_role_translation_inset_preserves_caller_domain_and_fixed_coordinates(self):
        bounds = np.asarray(((-.2, .2), (.01, .01)))
        original = bounds.copy()
        _, _, _, _, stats = self.solve([self.square(.04)], [[-.3, -.3]], translation=bounds)
        np.testing.assert_array_equal(bounds, original)
        self.assertEqual(stats['role_translation_inset'], (1.e-5, 0.))
        self.assertGreaterEqual(stats['role_translation'][0], -.2+5.e-6)
        self.assertLessEqual(stats['role_translation'][0], .2-5.e-6)
        self.assertEqual(stats['role_translation'][1], .01)


if __name__ == "__main__":
    unittest.main()
