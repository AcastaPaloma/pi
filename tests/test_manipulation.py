"""Real adapter shapes, sharded episode metadata, action boundaries and cache/API."""
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pi_data import ManipulationDataset, ObservationBatch, prepare_observation_cache
from pi_data.api import create_app
from test_data import make_fixture


class ManipulationTest(unittest.TestCase):
    def test_single_arm_multishard_replay_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            make_fixture(root)
            old = ("left_wrist", "right_wrist", "right_front")
            new = ("wrist_left", "desk_view")
            names = [f"{j}.pos" for j in ("shoulder_pan", "shoulder_lift", "elbow_flex",
                                        "wrist_flex", "wrist_roll", "gripper")]
            info = json.loads((root / "meta/info.json").read_text())
            info["total_episodes"] = 10
            info["features"] = {key: {"shape": [6], "names": names} for key in ("action", "observation.state")}
            info["features"].update({f"observation.images.{c}": {"shape": [32, 32, 3]} for c in new})
            (root / "meta/info.json").write_text(json.dumps(info))
            meta = root / "meta/episodes/chunk-000/file-000.parquet"
            rows = pq.read_table(meta).to_pylist()
            for row in rows:
                for key in list(row):
                    if any(f"observation.images.{c}/" in key for c in old):
                        value = row.pop(key)
                        for a, b in zip(old[:2], new):
                            if f"observation.images.{a}/" in key:
                                row[key.replace(a, b)] = value
            pq.write_table(pa.Table.from_pylist(rows[:5]), meta)
            pq.write_table(pa.Table.from_pylist(rows[5:]), meta.with_name("file-001.parquet"))
            path = root / "data/chunk-000/file-000.parquet"
            rows = pq.read_table(path).to_pylist()
            for row in rows:
                for key in ("action", "observation.state"):
                    row[key] = row[key][:6]
            pq.write_table(pa.Table.from_pylist(rows), path)
            for a, b in zip(old[:2], new):
                (root / "videos" / f"observation.images.{a}").rename(root / "videos" / f"observation.images.{b}")
            ds = ManipulationDataset(root)
            self.assertEqual(len(ds.episodes), 10)
            x = ds.arrays(8, 3, size=32, horizon=3)
            self.assertEqual(x["images"].shape, (2, 32, 32, 3))
            self.assertEqual(x["proprioception"].shape, (6,))
            self.assertEqual(x["action_valid"].tolist(), [True, False, False])
            np.testing.assert_array_equal(x["actions"][:, 0], [803] * 3)
            client = TestClient(create_app(ds))
            metadata = client.get("/v1/dataset").json()
            self.assertEqual(metadata["camera_order"], list(new))
            self.assertEqual(metadata["repo"], ds.repo)
            cache = Path(tmp) / "cache"
            prepare_observation_cache(cache, dataset=ds, image_size=32)
            batch = ObservationBatch(cache, batch_size=8).observations()
            self.assertEqual(batch["images"].shape, (8, 2, 32, 32, 3))
            self.assertEqual(batch["proprioception"].shape, (8, 6))
            for i, (ep, frame) in enumerate(zip(batch["episode"], batch["frame"])):
                self.assertEqual(ds.split(int(ep)), "train")
                np.testing.assert_array_equal(batch["images"][i, 1], ds.image(int(ep), int(frame), new[1], 32))


if __name__ == "__main__":
    unittest.main()
