import unittest
from gait_metrics import walking_pass


class GaitChecks(unittest.TestCase):
    def test_thresholds_are_inclusive(self):
        self.assertTrue(walking_pass(10, .05, .05, .8, 4))

    def test_do_not_round_failure_into_pass(self):
        self.assertFalse(walking_pass(30, .01, 38 / 750, .85, 30))

    def test_far_kneeling_is_not_walking(self):
        self.assertFalse(walking_pass(100, .626, .995, .002, 30))

    def test_standing_in_place_is_not_walking(self):
        self.assertFalse(walking_pass(.5, 0, 0, 1, 0))
