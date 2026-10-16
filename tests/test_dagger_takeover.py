"""H13 regression: a DAgger-style teacher created mid-episode must finish the task. Before the fix its arms
treated the hand position at takeover as their rest pose, which deadlocked two_bins and handover."""
import unittest

import numpy as np

from pi_model.dagger import teacher_joint_action
from pi_sim.tasks import TaskEnv, TaskTeacher


def _fresh(env, seed):
    teacher = TaskTeacher(env, rng=np.random.default_rng(seed), observe_only=True)
    teacher.observe()
    for w in teacher.workers:
        w.obj = env.object_position(w.body)
        w.ticks = 0
    return teacher


def run_with_takeover(task, seed, takeover_s):
    env = TaskEnv(task, control_mode="joint", object_color="red", second_color="blue")
    env.reset(seed=seed, options={"randomize": True})
    teacher, k = _fresh(env, seed + 1), 0
    while True:
        if k == int(takeover_s * env.control_hz):
            teacher = _fresh(env, seed + 2)
        _, _, term, trunc, info = env.step(teacher_joint_action(env, teacher))
        k += 1
        if term or trunc:
            return bool(info["is_success"])


class DaggerTakeoverTest(unittest.TestCase):
    def test_two_bins_takeover_mid_carry(self):
        self.assertTrue(run_with_takeover("two_bins", 8000, 4.0))

    def test_handover_takeover_while_lowering_and_raising(self):
        self.assertTrue(run_with_takeover("handover", 8000, 4.0))
        self.assertTrue(run_with_takeover("handover", 8001, 6.0))


if __name__ == "__main__":
    unittest.main()
