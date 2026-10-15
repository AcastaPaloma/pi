import unittest

import numpy as np

from pi_sim.tasks import TASKS, TaskEnv, TaskTeacher


def run(task, seed):
    env = TaskEnv(task, control_mode="ee_delta", object_color="red", second_color="blue", control_hz=50, episode_seconds=30)
    _, info = env.reset(seed=seed)
    teacher = TaskTeacher(env, rng=np.random.default_rng(seed))
    labels = set()
    while True:
        labels.add(teacher.subtask)
        _, _, term, trunc, info = env.step(teacher.action())
        if term or trunc:
            break
    env.close()
    return info, labels, env


class TaskLadderTest(unittest.TestCase):
    def test_every_rung_succeeds_with_the_scripted_teacher(self):
        for task in TASKS:
            with self.subTest(task=task):
                info, labels, _ = run(task, seed=3)
                self.assertTrue(info["is_success"], info)
                self.assertGreater(len(labels), 4)

    def test_handover_requires_both_arms(self):
        info, labels, env = run("handover", seed=5)
        self.assertTrue(all(info["touched"].values()), info["touched"])
        self.assertIn("arm", info["instruction"])
        self.assertTrue(any("left" in s for s in labels) and any("right" in s for s in labels))

    def test_two_bins_uses_both_arms_and_two_objects(self):
        info, labels, env = run("two_bins", seed=7)
        self.assertTrue(info["is_success"])
        self.assertIn("blue", info["instruction"])
        self.assertTrue(any("left" in s and "right" in s for s in labels))

    def test_placement_is_seeded(self):
        env = TaskEnv("bin", control_mode="joint", control_hz=50)
        a, _ = env.reset(seed=11)
        b, _ = env.reset(seed=11)
        np.testing.assert_allclose(a["cube_position"], b["cube_position"])
        np.testing.assert_allclose(a["bin_position"], b["bin_position"])
        env.close()

    def test_unknown_task_and_same_colors_rejected(self):
        with self.assertRaises(ValueError):
            TaskEnv("juggle")
        with self.assertRaises(ValueError):
            TaskEnv("bin", object_color="red", second_color="red")


if __name__ == "__main__":
    unittest.main()
