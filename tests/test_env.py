"""Integration checks with real MuJoCo physics. Run with unittest; no extra deps."""

import sys
import unittest

import gymnasium as gym
import mujoco
from gymnasium.utils.env_checker import check_env
import numpy as np

from pi_sim import RobotEnv
from pi_sim.demo import ScriptedDemo


class EnvironmentTests(unittest.TestCase):
    def test_gym_contract_and_registered_env(self):
        with RobotEnv() as env:
            check_env(env, skip_render_check=True)
        with gym.make("PiSim-Reach-v0") as env:
            obs, _ = env.reset(seed=1)
            self.assertTrue(env.observation_space.contains(obs))
        self.assertNotIn("torch", sys.modules)

    def test_seeded_trajectory_repeats(self):
        with RobotEnv(task="pick_place") as env:
            traces = []
            for _ in range(2):
                env.reset(seed=17)
                demo = ScriptedDemo(env)
                trace = []
                for _ in range(50):
                    obs, _, _, _, _ = env.step(demo.action())
                    trace.append(np.r_[obs["state"], obs["cube_position"]])
                traces.append(trace)
            np.testing.assert_array_equal(traces[0], traces[1])

    def test_tasks_succeed_with_physical_grasp(self):
        for task in ("reach", "pick_place"):
            for mode in ("ee_delta", "ee"):
                with self.subTest(task=task, mode=mode), RobotEnv(task=task, control_mode=mode) as env:
                    for seed in range(5):
                        obs, _ = env.reset(seed=seed)
                        demo = ScriptedDemo(env)
                        peak_height = 0
                        for _ in range(400):
                            obs, _, terminated, truncated, info = env.step(demo.action())
                            peak_height = max(peak_height, obs["cube_position"][2])
                            if terminated or truncated:
                                break
                        self.assertTrue(info["is_success"], (task, mode, seed, demo.stage))
                        if task == "pick_place":
                            self.assertGreater(peak_height, 0.10)
                            self.assertLess(np.linalg.norm(obs["cube_position"] - obs["goal_position"]), 0.045)
                        with self.assertRaises(RuntimeError):
                            env.step(env.hold_action())

    def test_joint_control_reaches_both_arms(self):
        with RobotEnv(control_mode="joint", task="pick_place") as env:
            env.reset(seed=0)
            before = {side: arm.joint_positions for side, arm in env.robot.items()}
            action = env.hold_action()
            action[0] += 0.15
            action[7] -= 0.15
            action[6] = action[13] = 0
            for _ in range(30):
                env.step(action)
            self.assertGreater(env.robot["left"].joint_positions[0] - before["left"][0], 0.10)
            self.assertLess(env.robot["right"].joint_positions[0] - before["right"][0], -0.10)
            self.assertLess(env.robot["left"].gripper_opening, 0.2)
            self.assertLess(env.robot["right"].gripper_opening, 0.2)
            self.assertAlmostEqual(env.data.time, 1.5, places=8)

    def test_chunk_timeout_invalid_inputs_and_observation_copies(self):
        with RobotEnv(max_episode_steps=3) as env:
            obs, _ = env.reset(seed=0)
            saved = obs["state"].copy()
            with self.assertRaises(ValueError):
                env.step(np.zeros(7))
            with self.assertRaises(ValueError):
                env.step(np.full(8, np.nan))
            self.assertEqual(env.data.time, 0)
            chunk = np.tile(env.hold_action(), (50, 1))
            bad = chunk.copy()
            bad[-1, 0] = np.inf
            with self.assertRaises(ValueError):
                env.step_chunk(bad)
            self.assertEqual(env.data.time, 0)
            transitions = env.step_chunk(chunk)
            self.assertEqual(len(transitions), 3)
            self.assertFalse(transitions[-1][2])
            self.assertTrue(transitions[-1][3])
            np.testing.assert_array_equal(obs["state"], saved)
            env.reset(seed=0)
            self.assertEqual(env.data.time, 0)
            self.assertFalse(env.step(env.hold_action())[3])

    def test_absolute_end_effector_rotation(self):
        with RobotEnv(control_mode="ee", task="pick_place") as env:
            env.reset(seed=1)
            action = env.hold_action()
            target = np.zeros(4)
            yaw = np.array([np.cos(0.15), 0, 0, np.sin(0.15)])
            mujoco.mju_mulQuat(target, yaw, env.robot["right"].quaternion)
            rotvec = np.zeros(3)
            mujoco.mju_quat2Vel(rotvec, target, 1)
            action[10:13] = rotvec
            for _ in range(40):
                env.step(action)
            error = np.zeros(3)
            mujoco.mju_subQuat(error, target, env.robot["right"].quaternion)
            self.assertLess(np.linalg.norm(error), 0.04)
            self.assertLess(np.linalg.norm(env.robot["right"].position - action[7:10]), 0.02)

    def test_cube_fall_is_failure(self):
        with RobotEnv(task="pick_place") as env:
            env.reset(seed=0)
            env.data.joint("cube_joint").qpos[:3] = [1.4, 0, -0.2]
            _, _, terminated, truncated, info = env.step(env.hold_action())
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["is_success"])
            self.assertEqual(info["failure"], "cube_fell")

    def test_cameras(self):
        with RobotEnv(images=True, image_size=448, render_mode="rgb_array") as env:
            obs, _ = env.reset(seed=0)
            self.assertTrue(env.observation_space.contains(obs))
            for frame in obs["images"].values():
                self.assertEqual(frame.shape, (448, 448, 3))
                self.assertEqual(frame.dtype, np.uint8)
                self.assertGreater(float(frame.std()), 10)
            np.testing.assert_array_equal(env.render(), obs["images"]["front"])


if __name__ == "__main__":
    unittest.main()
