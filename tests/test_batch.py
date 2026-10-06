"""Batched observations: state/image alignment, cache isolation, split integrity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pi_sim import RobotBatch
from pi_data import FoldingDataset, FoldingBatch, prepare_observation_cache
from pi_observe.grid import observation_grid
from test_data import make_fixture


class SimBatchTest(unittest.TestCase):
    def test_no_render_for_state_no_physics_for_reads_cache_copies(self):
        with RobotBatch(4, image_size=32, seed=1) as robot:
            start_time = robot._env.data.time
            with patch.object(robot._env, "render_camera", wraps=robot._env.render_camera) as render:
                state = robot.observations(images=False)
                self.assertNotIn("images", state)
                self.assertEqual(render.call_count, 0)
                batch = robot.observations()
                self.assertEqual(batch["images"].shape, (4, 3, 32, 32, 3))
                self.assertEqual(batch["proprioception"].shape, (4, 14))
                self.assertEqual(render.call_count, 12)
                first_state = batch["proprioception"].copy()
                first_images = batch["images"].copy()
                batch["proprioception"][:] = 99
                batch["images"][:] = 0
                newer = robot.observations()
                self.assertEqual(render.call_count, 12)
                np.testing.assert_array_equal(newer["images"], first_images)
                np.testing.assert_array_equal(newer["proprioception"], first_state)
                with self.assertRaises(ValueError):
                    robot.observations(copy=False)["images"][0] = 0
                robot.resample(seed=2)
                changed = robot.observations()
                self.assertEqual(render.call_count, 24)
                self.assertFalse(np.array_equal(changed["proprioception"], first_state))
                self.assertFalse(np.array_equal(changed["images"], first_images))
                self.assertEqual(robot._env.data.time, start_time)
            self.assertEqual(robot.grid(columns=2, tile_size=32).size, (64, 104))
        with self.assertRaises(RuntimeError):
            robot.observations()

    def test_128_independent_reproducible_state_rows(self):
        with RobotBatch(128, image_size=16, seed=7) as robot:
            a = robot.observations(images=False)
            self.assertEqual(np.unique(a["proprioception"], axis=0).shape[0], 128)
            robot.resample(seed=7)
            np.testing.assert_array_equal(robot.observations(images=False)["proprioception"], a["proprioception"])


class RealBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_fixture(cls.root / "source")
        cls.ds = FoldingDataset(cls.root / "source")
        cls.cache = cls.root / "cache"
        prepare_observation_cache(cls.cache, dataset=cls.ds, image_size=32, stride=1)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_batch_alignment_to_recorded_state_and_each_camera(self):
        robot = FoldingBatch(self.cache, batch_size=32)
        batch = robot.observations()
        self.assertEqual(batch["images"].shape, (32, 3, 32, 32, 3))
        self.assertEqual(len(np.unique(batch["cache_index"])), 32)
        for i, (ep, frame) in enumerate(zip(batch["episode"], batch["frame"])):
            self.assertEqual(self.ds.split(int(ep)), "train")
            np.testing.assert_array_equal(batch["proprioception"][i], [100 * ep + frame + 0.5] * 12)
            # Each video contains a different start offset; match direct seeks.
            for c, camera in enumerate(batch["camera_names"]):
                np.testing.assert_array_equal(batch["images"][i, c], self.ds.image(int(ep), int(frame), str(camera), 32))

    def test_no_video_decode_during_batch_reads_and_mutation_isolation(self):
        robot = FoldingBatch(self.cache, batch_size=8)
        with patch("av.open", side_effect=AssertionError("Video decoder used during batch read")):
            batch = robot.observations([0, 1, 2])
            expected = batch["images"].copy()
            batch["images"][:] = 0
            np.testing.assert_array_equal(robot.observations([0, 1, 2])["images"], expected)
        self.assertEqual(observation_grid(robot.observations(), columns=4, tile_size=32).size, (128, 104))

    def test_split_bounds_and_atomic_preparation(self):
        with self.assertRaises(ValueError):
            FoldingBatch(self.cache, batch_size=128)
        robot = FoldingBatch(self.cache, batch_size=8)
        for indices in ([-1], [len(robot)], [0.5], []):
            with self.assertRaises(ValueError):
                robot.observations(indices)
        with self.assertRaises(ValueError):
            prepare_observation_cache(self.root / "mixed", dataset=self.ds, episodes=self.ds.episode_ids("validation"))
        self.assertFalse((self.root / "mixed").exists())
        with patch("pi_data.batch._decode_selected", side_effect=ValueError("bad video")), self.assertRaises(ValueError):
            prepare_observation_cache(self.root / "broken", dataset=self.ds, image_size=16)
        self.assertFalse((self.root / "broken").exists())
        self.assertFalse(list(self.root.glob(".broken.partial-*")))
        with self.assertRaises(FileExistsError):
            prepare_observation_cache(self.cache, dataset=self.ds)
        metadata = json.loads((self.cache / "metadata.json").read_text())
        self.assertEqual(metadata["count"], 36)


if __name__ == "__main__":
    unittest.main()
