"""SI-200-8 range and effective-resolution checks independent of Isaac Sim."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contact_rearrangement.src.catalog import ForceTorqueCalibration, ForceTorqueSensorCfg, SI_200_8_CALIBRATION


class ForceTorqueCalibrationTests(unittest.TestCase):
    def test_hydra_settings_produce_an_immutable_validated_calibration(self):
        settings = ForceTorqueSensorCfg()
        settings.ranges = (100., 100., 180., 4., 4., 4.)
        calibration = settings.calibration()
        settings.ranges = SI_200_8_CALIBRATION.ranges
        self.assertEqual(calibration.ranges, (100., 100., 180., 4., 4., 4.))
        self.assertEqual(SI_200_8_CALIBRATION.ranges, (200., 200., 360., 8., 8., 8.))
        with self.assertRaises(FrozenInstanceError):
            calibration.ranges = settings.ranges
        settings.resolution = (0.,) * 6
        with self.assertRaises(ValueError):
            settings.calibration()

    def setUp(self):
        self.calibration = SI_200_8_CALIBRATION

    def test_profile_matches_six_axis_si_200_8_table(self):
        self.assertEqual(self.calibration.name, "ATI Axia80 SI-200-8")
        self.assertEqual(self.calibration.ranges, (200., 200., 360., 8., 8., 8.))
        self.assertEqual(self.calibration.resolution, (.1, .1, .1, .005, .005, .005))

    def test_signed_forces_and_torques_use_their_own_resolution(self):
        raw = torch.tensor([[.14, -.16, .26, .006, -.009, .014]], dtype=torch.float64)
        measured, exceeded = self.calibration.measure(raw)
        expected = torch.tensor([[.1, -.2, .3, .005, -.010, .015]], dtype=torch.float64)
        torch.testing.assert_close(measured, expected, atol=1.e-12, rtol=0.)
        self.assertFalse(bool(exceeded.any()))

    def test_axial_force_has_360_n_capacity_while_lateral_force_has_200_n(self):
        raw = torch.tensor([[250., -250., 250., 0., 0., 0.]])
        measured, exceeded = self.calibration.measure(raw)
        torch.testing.assert_close(measured, torch.tensor([[200., -200., 250., 0., 0., 0.]]))
        self.assertEqual(exceeded.tolist(), [[True, True, False, False, False, False]])

    def test_saturation_preserves_signed_per_axis_overload_before_rounding(self):
        raw = torch.tensor([[200.04, -200.04, 360.04, 8.001, -8.001, 8.001]], dtype=torch.float64)
        measured, exceeded = self.calibration.measure(raw)
        torch.testing.assert_close(measured, torch.tensor([[200., -200., 360., 8., -8., 8.]], dtype=torch.float64))
        self.assertTrue(bool(exceeded.all()))

    def test_exact_positive_and_negative_range_boundaries_are_valid(self):
        raw = torch.tensor([[200., 200., 360., 8., 8., 8.],
                            [-200., -200., -360., -8., -8., -8.]], dtype=torch.float64)
        measured, exceeded = self.calibration.measure(raw)
        torch.testing.assert_close(measured, raw)
        self.assertFalse(bool(exceeded.any()))

    def test_measurement_does_not_rewrite_ideal_load_and_preserves_batch_dtype(self):
        raw = torch.tensor([[[250., 0., 0., 0., 0., 0.]], [[0., 0., 0., 0., 0., -9.]]],
                           dtype=torch.float64)
        before = raw.clone()
        measured, exceeded = self.calibration.measure(raw)
        torch.testing.assert_close(raw, before)
        self.assertEqual(measured.shape, (2, 1, 6))
        self.assertEqual(measured.dtype, raw.dtype)
        self.assertEqual(exceeded.dtype, torch.bool)
        self.assertEqual(exceeded[0, 0].tolist(), [True, False, False, False, False, False])
        self.assertEqual(exceeded[1, 0].tolist(), [False, False, False, False, False, True])

    def test_custom_calibration_remains_immutable_and_inside_range_after_rounding(self):
        calibration = ForceTorqueCalibration(name="test", ranges=[.26] * 6, resolution=[.1] * 6)
        self.assertIsInstance(calibration.ranges, tuple)
        with self.assertRaises(FrozenInstanceError):
            calibration.name = "changed"
        measured, exceeded = calibration.measure(torch.tensor([.26] * 6, dtype=torch.float64))
        self.assertLessEqual(float(measured.max()), .26)
        self.assertFalse(bool(exceeded.any()))

    def test_invalid_axis_specs_are_rejected(self):
        for field in ("ranges", "resolution"):
            for values in ((1.,) * 5, (1.,) * 7, (0.,) * 6, (-1.,) * 6,
                           (float("nan"),) * 6, (float("inf"),) * 6, (True,) * 6):
                with self.subTest(field=field, values=values), self.assertRaises(ValueError):
                    replace(self.calibration, **{field: values})
        with self.assertRaises(ValueError):
            replace(self.calibration, name=" ")

    def test_wrong_wrench_shape_or_integer_dtype_is_rejected(self):
        for raw in (torch.zeros(5), torch.zeros(2, 7), torch.tensor(1.), torch.zeros(6, dtype=torch.int64)):
            with self.subTest(shape=tuple(raw.shape), dtype=raw.dtype), self.assertRaises(ValueError):
                self.calibration.measure(raw)


if __name__ == "__main__":
    unittest.main()
