"""Offline integration tests, including real video decoding across episode offsets."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

import av
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from pi_data import FoldingDataset
from pi_data.api import create_app
from pi_data.dataset import CAMERAS


def make_fixture(root):
    names = [f"joint_{i}" for i in range(12)]
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {k: {"names": names} for k in ("action", "observation.state")}
    features.update({f"observation.images.{cam}": {"shape": [32, 32, 3]} for cam in CAMERAS})
    (root / "meta/info.json").write_text(json.dumps({"fps": 2, "features": features}))
    pq.write_table(pa.Table.from_pylist([{"task_index": 0, "task": "Fold the Garment"}]), root / "meta/tasks.parquet")
    episodes, rows = [], []
    for ep in range(10):
        meta = {"episode_index": ep, "length": 4, "dataset_from_index": ep * 4,
                "dataset_to_index": (ep + 1) * 4, "data/chunk_index": 0, "data/file_index": 0}
        for i, cam in enumerate(CAMERAS):
            key = f"videos/observation.images.{cam}"
            meta.update({f"{key}/chunk_index": 0, f"{key}/file_index": 0,
                         f"{key}/from_timestamp": i + ep * 2, f"{key}/to_timestamp": i + (ep + 1) * 2})
        episodes.append(meta)
        for frame in range(4):
            rows.append({"episode_index": ep, "frame_index": frame, "timestamp": frame / 2,
                         "task_index": 0, "action": [ep * 100 + frame] * 12,
                         "observation.state": [ep * 100 + frame + 0.5] * 12})
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), root / "data/chunk-000/file-000.parquet")
    for i, cam in enumerate(CAMERAS):
        path = root / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        path.parent.mkdir(parents=True)
        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=2)
            stream.width = stream.height = 32
            stream.pix_fmt = "yuv420p"
            for frame in range(2 * i + 40):
                logical = frame - 2 * i
                color = 0 if logical < 0 else 20 + logical * 4
                video = av.VideoFrame.from_ndarray(np.full((32, 32, 3), color, np.uint8), format="rgb24")
                for packet in stream.encode(video):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


class DataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        make_fixture(cls.root)
        cls.ds = FoldingDataset(cls.root)
        cls.client = TestClient(create_app(cls.ds))

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.temp.cleanup()

    def test_chunk_never_crosses_episode(self):
        sample = self.ds.sample(0, 2, horizon=5)
        self.assertEqual(sample["actions"]["valid"], [True, True, False, False, False])
        self.assertEqual([a[0] for a in sample["actions"]["values"]], [2, 3, 3, 3, 3])
        self.assertEqual(self.ds.sample(1, 0)["actions"]["values"][0][0], 100)

    def test_history_masks_and_goals(self):
        sample = self.ds.sample(1, 1, history=3, goal_seconds=4)
        self.assertEqual(sample["history"]["frames"], [0, 0, 1])
        self.assertEqual(sample["history"]["valid"], [False, False, True])
        self.assertEqual(sample["visual_subgoal"]["frame"], 3)
        self.assertEqual(sample["visual_subgoal"]["delta_seconds"], 1)
        self.assertIsNone(sample["text"]["subtask"])
        for seed in range(30):
            goal = self.ds.sample(1, 2, seed=seed)["visual_subgoal"]
            self.assertIn(goal["frame"], (2, 3))
            self.assertEqual(goal, self.ds.sample(1, 2, seed=seed)["visual_subgoal"])
        self.assertFalse(self.ds.sample(1, 3)["visual_subgoal"]["strictly_future"])

    def test_camera_specific_video_offset(self):
        for ep, frame in ((0, 0), (0, 3), (1, 0), (1, 3), (9, 3)):
            expected = 20 + (ep * 4 + frame) * 4
            for cam in CAMERAS:
                image = self.ds.image(ep, frame, cam, 32)
                self.assertEqual(image.shape, (32, 32, 3))
                self.assertLess(abs(image.astype(float).mean() - expected), 4)

    def test_split_and_normalization_exclude_validation(self):
        train, validation = self.ds.episode_ids("train"), self.ds.episode_ids("validation")
        self.assertEqual((len(train), len(validation)), (9, 1))
        self.assertFalse(set(train) & set(validation))
        stats = self.ds.train_statistics()
        expected = np.mean([100 * ep + frame for ep in train for frame in range(4)])
        self.assertEqual(stats["features"]["action"]["count"], 36)
        np.testing.assert_allclose(stats["features"]["action"]["mean"], expected)

    def test_http_endpoints_and_numpy_payload(self):
        base = "/v1/episodes/1/frames/2"
        for path in ("/docs", "/openapi.json", "/v1/dataset", "/v1/episodes?split=train", "/v1/normalization"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        for path in ("observations", "proprioception", "text", "actions", "subgoal", "sample"):
            self.assertEqual(self.client.get(f"{base}/{path}").status_code, 200, path)
        s = self.client.get(f"{base}/sample?history=3&goal_seconds=1").json()
        for url in s["cameras"].values():
            response = self.client.get(url)
            self.assertEqual(response.headers["content-type"], "image/png")
            self.assertEqual(Image.open(BytesIO(response.content)).size, (448, 448))
        response = self.client.get(f"{base}/sample.npz?size=32&history=3&goal_seconds=1")
        self.assertEqual(response.status_code, 200)
        with np.load(BytesIO(response.content), allow_pickle=False) as a:
            self.assertEqual(a["images"].shape, (3, 32, 32, 3))
            self.assertEqual(a["history_images"].shape, (3, 3, 32, 32, 3))
            self.assertEqual(a["actions"].shape, (50, 12))
            self.assertEqual(a["action_valid"].sum(), 2)
            self.assertEqual(str(a["task"]), "Fold the Garment")
            self.assertEqual(json.loads(str(a["metadata_json"]))["episode"], 1)

    def test_errors(self):
        base = "/v1/episodes/0/frames/0"
        for query in ("history=7", "horizon=0", "goal_seconds=5", "seed=-1"):
            self.assertEqual(self.client.get(f"{base}/sample?{query}").status_code, 422)
        for path in ("/v1/episodes/100/frames/0/sample", "/v1/episodes/0/frames/4/sample",
                     "/v1/episodes/0/frames/-1/sample", f"{base}/cameras/unknown.png"):
            self.assertEqual(self.client.get(path).status_code, 404, path)
        with self.assertRaises(ValueError):
            self.ds.sample(0, 0, goal_seconds=float("nan"))


if __name__ == "__main__":
    unittest.main()
