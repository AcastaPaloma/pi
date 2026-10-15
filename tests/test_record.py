"""Scripted episode recording: labels, segments, randomization and layout."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pi_sim import RobotEnv
from pi_sim.record import record_episode, quality_label


class RecordTests(unittest.TestCase):
    def test_episode_layout_labels_and_randomization(self):
        with tempfile.TemporaryDirectory() as tmp, RobotEnv(task="pick_place", control_hz=50) as env:
            out = Path(tmp) / "episode_00000"
            meta = record_episode(env, out, seed=3, style="clean", image_size=64)
            steps = np.load(out / "steps.npz")
            T = meta["steps"]
            self.assertTrue(meta["success"])
            self.assertEqual(meta["control_hz"], 50)
            self.assertEqual(steps["state"].shape, (T, 12))
            self.assertEqual(steps["action"].shape, (T, 12))
            self.assertEqual(steps["segment_id"][0], 0)
            self.assertGreaterEqual(steps["segment_id"][-1], 5)
            self.assertEqual(len(meta["subtasks"]), int(steps["subtask_index"].max()) + 1)
            self.assertEqual(sorted(p.name for p in (out / "frames").iterdir()), ["left_wrist", "overhead", "right_wrist"])
            self.assertEqual(len(list((out / "frames" / "overhead").glob("*.jpg"))), T)
            self.assertIsNotNone(meta["visual_randomization"])
            self.assertIn("camera_jitter", meta["visual_randomization"])
            # Actions are joint targets in degrees: replaying them in joint mode
            # must reproduce the recorded states closely.
            self.assertLess(np.abs(steps["action"][:, :5] - steps["state"][:, :5]).mean(), 15)
            self.assertEqual(meta["quality"], 5)
            self.assertEqual(quality_label(False, "clean", 0), 1)
            self.assertEqual(quality_label(True, "sloppy", 1), 2)
            with self.assertRaises(FileExistsError):
                record_episode(env, out, seed=3)
            json.loads((out / "meta.json").read_text())

    def test_frame_stride_records_subset_and_dataset_indexes_it(self):
        from pi_model.data import EpisodeStore
        with tempfile.TemporaryDirectory() as tmp, RobotEnv(task="pick_place", control_hz=50) as env:
            out = Path(tmp) / "episode_00000"
            meta = record_episode(env, out, seed=1, style="clean", image_size=32, frame_stride=5)
            T = meta["steps"]
            self.assertEqual(meta["frame_stride"], 5)
            self.assertEqual(len(list((out / "frames" / "overhead").glob("*.jpg"))), (T + 4) // 5)
            store = EpisodeStore(Path(tmp))
            self.assertEqual(len(store.episodes), 1)
            self.assertEqual(int(store.sample_counts[0]), (T + 4) // 5)
            self.assertEqual(store.locate(3), (0, 15))
            self.assertEqual(store.episodes[0].frame_step(17), 15)
            self.assertEqual(store.frame(store.episodes[0], "overhead", 17, 32).shape, (32, 32, 3))
            with self.assertRaises(ValueError):
                record_episode(env, Path(tmp) / "bad", seed=1, frame_stride=7)

    def test_visual_randomization_restores(self):
        with RobotEnv(task="pick_place") as env:
            env.reset(seed=0)
            nominal = env.model.cam_pos.copy(), env.model.geom_rgba.copy()
            env.reset(seed=1, options={"visual_randomization": True})
            self.assertFalse(np.allclose(env.model.cam_pos, nominal[0]))
            env.reset(seed=2)
            np.testing.assert_array_equal(env.model.cam_pos, nominal[0])
            np.testing.assert_array_equal(env.model.geom_rgba, nominal[1])


if __name__ == "__main__":
    unittest.main()
