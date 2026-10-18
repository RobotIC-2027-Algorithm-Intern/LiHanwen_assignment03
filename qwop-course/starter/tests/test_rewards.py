import json
from pathlib import Path
import unittest
from rewards import walking_reward


class RewardDesignTests(unittest.TestCase):
    def setUp(self):
        self.c = json.loads(Path('configs/distance_only.json').read_text())
        self.standing = dict(left_knee_y=4, right_knee_y=4, nonfoot_contact=0)
        self.kneeling = dict(left_knee_y=8, right_knee_y=4, nonfoot_contact=1)

    def test_same_progress_prefers_feet_support(self):
        config = {**self.c, "posture_weight": 1.0, "target_torso_y": -1.0,
                  "knee_low_penalty": 1.0, "target_knee_y": 6.0,
                  "nonfoot_contact_penalty": 1.0}
        good, _ = walking_reward(.5, -1, self.standing, config)
        bad, _ = walking_reward(.5, 2, self.kneeling, config)
        self.assertGreater(good, bad)
        self.assertLess(bad, 0)

    def test_forward_better_than_stationary(self):
        forward, _ = walking_reward(.5, -1, self.standing, self.c)
        stationary, _ = walking_reward(0, -1, self.standing, self.c)
        self.assertGreater(forward, stationary)

    def test_backward_not_rewarded_as_forward(self):
        reward, terms = walking_reward(-.5, -1, self.standing, self.c)
        self.assertLess(terms['forward'], 0)
        self.assertLess(reward, 0)

    def test_failure_cost_applies_to_terminal_step(self):
        normal, _ = walking_reward(.5, -1, self.standing, self.c)
        failed, _ = walking_reward(.5, -1, self.standing, self.c, failed=True)
        self.assertAlmostEqual(normal - failed, self.c['fall_penalty'])

    def test_distance_only_control_has_no_hidden_pose_reward(self):
        c = json.loads(Path('configs/distance_only.json').read_text())
        reward, _ = walking_reward(.25, 3, self.kneeling, c, failed=True)
        self.assertAlmostEqual(reward, .25)

    def test_curriculum_gently_increases_pose_cost(self):
        config = {**self.c, 'posture_weight': 1.0, 'target_torso_y': -1.0}
        gentle = {**self.kneeling, 'pose_curriculum_scale': .25}
        a, _ = walking_reward(.25, 2, gentle, config)
        b, _ = walking_reward(.25, 2, self.kneeling, config)
        self.assertGreater(a, b)

    def test_reverse_reward_keeps_direction_and_does_not_pay_for_standing(self):
        c = {**self.c, 'forward_weight': -1.0, 'displacement_clip': 0.5}
        reverse, _ = walking_reward(-.1, -1, self.standing, c)
        forward, _ = walking_reward(.1, -1, self.standing, c)
        idle, _ = walking_reward(0, -1, self.standing, c)
        self.assertGreater(reverse, 0)
        self.assertLess(forward, 0)
        self.assertEqual(idle, 0)

    def test_impulsive_back_fall_cannot_cash_out_a_large_displacement(self):
        c = {**self.c, 'forward_weight': -1.0, 'displacement_clip': 0.5, 'fall_penalty': 100.0}
        reward, _ = walking_reward(-100, -1, self.standing, c)
        # Even the maximum per-step bonus over the old 2.7-second flop
        # remains smaller than the cost of falling.
        self.assertLess(54 * reward - c['fall_penalty'], 0)


if __name__ == '__main__':
    unittest.main()
