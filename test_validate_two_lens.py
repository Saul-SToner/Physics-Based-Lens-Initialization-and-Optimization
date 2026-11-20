from dataclasses import astuple
import unittest

from two_lens import design_two_lens
from validate_two_lens import validate_two_lens


class TwoLensValidationTests(unittest.TestCase):
    def setUp(self):
        # The initial prescription shared by Figures 2 and 3.
        self.inputs = (-2.704, 1.9055, -1.643, 105.436, -3.463)
        self.candidate = astuple(design_two_lens(*self.inputs))

    def test_paper_example(self):
        report = validate_two_lens(self.inputs, self.candidate)
        self.assertTrue(report.passed, report)

    def test_single_parameter_errors(self):
        for index, name in enumerate(("d12", "k1", "k2", "efl")):
            with self.subTest(parameter=name):
                candidate = list(self.candidate)
                candidate[index] *= 1.01
                report = validate_two_lens(self.inputs, tuple(candidate))
                self.assertFalse(report.passed, report)
                if name == "efl":
                    self.assertTrue(report.object_image_b_mm.passed)
                    self.assertTrue(report.magnification_error.passed)
                    self.assertTrue(report.pupil_b_mm.passed)
                    self.assertFalse(report.efl_error_mm.passed)
