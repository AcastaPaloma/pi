"""Pinned LeHome real data, with episode-safe action chunks and future goals.

All joint values retain their source units. This is a replay/data interface,
not a command interface to a physical robot.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import av
from huggingface_hub import hf_hub_download, snapshot_download
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

REPO = "lehome/dataset_challenge_real"
REVISION = "5a28286fb60db8bf9fab9a552a001645722752cd"
SUBSET = "four_types_merged"
CAMERAS = ("left_wrist", "right_wrist", "right_front")
UNITS = ["degree"] * 5 + ["gripper_percent_0_closed_100_open"]
UNITS = UNITS * 2
UNITS_SOURCE = "https://github.com/IliaLarchenko/lehome_solution/blob/main/src/lehome_solution/training/real_data_transforms.py"


class FoldingDataset:
    """Lazily download one immutable dataset revision into the HF cache.

    ``root`` optionally points at a *local four_types_merged directory*.
    A local root never downloads missing files. Instantiate per loader worker;
    no decoder/container is shared between threads or processes.
    """

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else None
        self.info = json.loads(self._file("meta/info.json").read_text())
        self.fps = int(self.info["fps"])
        self.episodes = {
            int(row["episode_index"]): row
            for row in pq.read_table(self._file("meta/episodes/chunk-000/file-000.parquet")).to_pylist()
        }
        self.tasks = {
            int(row["task_index"]): row["task"]
            for row in pq.read_table(self._file("meta/tasks.parquet")).to_pylist()
        }
        self.names = self.info["features"]["action"]["names"]
        if len(self.names) != 12 or self.info["features"]["observation.state"]["names"] != self.names:
            raise ValueError("Expected matching 12D bimanual state/action schemas")
        for camera in CAMERAS:
            if f"observation.images.{camera}" not in self.info["features"]:
                raise ValueError(f"Missing camera {camera}")
        for ep in self.episodes.values():
            if ep["length"] < 1 or ep["dataset_to_index"] - ep["dataset_from_index"] != ep["length"]:
                raise ValueError("Invalid episode length/index range")
            for camera in CAMERAS:
                key = f"videos/observation.images.{camera}"
                duration = ep[f"{key}/to_timestamp"] - ep[f"{key}/from_timestamp"]
                if abs(duration - ep["length"] / self.fps) > 0.5 / self.fps:
                    raise ValueError(f"Episode {ep['episode_index']}: {camera} duration mismatch")
        # Exact 90/10 episode split. Hash ranking is reproducible across machines.
        ordered = sorted(self.episodes, key=lambda ep: hashlib.sha256(f"lehome-v1:{ep}".encode()).digest())
        n_val = max(1, round(len(ordered) * 0.1))
        self.validation_ids = frozenset(ordered[:n_val])

    @lru_cache(maxsize=256)
    def _file(self, relative: str) -> Path:
        if self.root is not None:
            path = self.root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing local dataset file: {path}")
            return path
        return Path(hf_hub_download(REPO, f"{SUBSET}/{relative}", repo_type="dataset", revision=REVISION))

    def episode(self, episode: int) -> dict:
        if episode not in self.episodes:
            raise KeyError(f"Unknown episode {episode}")
        return self.episodes[episode]

    def split(self, episode: int) -> str:
        self.episode(episode)
        return "validation" if episode in self.validation_ids else "train"

    def episode_ids(self, split: str | None = None) -> list[int]:
        if split not in (None, "train", "validation"):
            raise ValueError("split must be train or validation")
        return [ep for ep in sorted(self.episodes) if split is None or self.split(ep) == split]

    @lru_cache(maxsize=2)
    def _table(self, chunk: int, file: int):
        return pq.read_table(self._file(f"data/chunk-{chunk:03d}/file-{file:03d}.parquet"))

    @lru_cache(maxsize=16)
    def _rows(self, episode: int) -> dict[str, np.ndarray]:
        ep = self.episode(episode)
        table = self._table(ep["data/chunk_index"], ep["data/file_index"])
        # Select by episode id, not global offsets into a potentially sharded file.
        import pyarrow.compute as pc
        rows = table.filter(pc.equal(table["episode_index"], episode))
        arrays = {
            key: np.asarray(rows[key].to_pylist(), dtype=np.float32 if key in ("action", "observation.state", "timestamp") else np.int64)
            for key in ("action", "observation.state", "timestamp", "frame_index", "task_index")
        }
        n = ep["length"]
        if not np.array_equal(arrays["frame_index"], np.arange(n)):
            raise ValueError(f"Episode {episode}: noncontiguous frame indices")
        if not np.allclose(arrays["timestamp"], np.arange(n) / self.fps, atol=0.5 / self.fps, rtol=0):
            raise ValueError(f"Episode {episode}: inconsistent frame timestamps")
        for key in ("action", "observation.state"):
            if arrays[key].shape != (n, 12) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Episode {episode}: invalid {key}")
        return arrays

    def _validate_frame(self, episode: int, frame: int) -> dict:
        ep = self.episode(episode)
        if not 0 <= frame < ep["length"]:
            raise IndexError(f"frame must be 0..{ep['length'] - 1} for episode {episode}")
        return ep

    def _video(self, episode: int, camera: str) -> tuple[Path, float]:
        if camera not in CAMERAS:
            raise KeyError(f"Unknown camera {camera}; expected {CAMERAS}")
        ep = self.episode(episode)
        key = f"videos/observation.images.{camera}"
        chunk, file = ep[f"{key}/chunk_index"], ep[f"{key}/file_index"]
        path = self._file(f"videos/observation.images.{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4")
        return path, float(ep[f"{key}/from_timestamp"])

    def image(self, episode: int, frame: int, camera: str, size: int = 448) -> np.ndarray:
        """RGB uint8 HWC. Direct square resize, matching the paper's input size.

        Seek using the camera's own file offset plus the recorded row timestamp.
        Refuse to silently return a distant or missing video frame.
        """
        self._validate_frame(episode, frame)
        if not 16 <= size <= 1024:
            raise ValueError("size must be between 16 and 1024")
        path, offset = self._video(episode, camera)
        target = offset + float(self._rows(episode)["timestamp"][frame])
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            container.seek(int(target / float(stream.time_base)), stream=stream, backward=True)
            best, distance = None, float("inf")
            for decoded in container.decode(stream):
                if decoded.pts is None:
                    continue
                timestamp = float(decoded.pts * stream.time_base)
                delta = abs(timestamp - target)
                if delta < distance:
                    best, distance = decoded, delta
                if timestamp >= target:
                    break
            if best is None or distance > 0.55 / self.fps:
                raise ValueError(f"No aligned {camera} frame at {target:.4f}s (error {distance:.4f}s)")
            return np.asarray(best.to_image().convert("RGB").resize((size, size), Image.Resampling.BILINEAR))

    def sample(self, episode: int, frame: int, *, horizon: int = 50,
               history: int = 1, goal_seconds: float | None = None, seed: int = 0) -> dict:
        """JSON-safe metadata and labels; image URLs are added by the HTTP API.

        History is oldest-to-current at 1 Hz, including the current frame.
        Default goal is reproducibly uniform over valid frames 0..4s ahead.
        The action at the current row is the first action label (no +1 shift).
        """
        ep = self._validate_frame(episode, frame)
        if not 1 <= horizon <= 200 or not 1 <= history <= 6:
            raise ValueError("horizon must be 1..200; history must be 1..6")
        if goal_seconds is not None and (not np.isfinite(goal_seconds) or not 0 <= goal_seconds <= 4):
            raise ValueError("goal_seconds must be 0..4 or None")
        if seed < 0:
            raise ValueError("seed must be nonnegative")
        rows = self._rows(episode)
        n = ep["length"]
        action_indices = np.arange(frame, frame + horizon)
        history_indices = np.arange(-(history - 1), 1) * self.fps + frame
        max_future = min(4 * self.fps, n - 1 - frame)
        if goal_seconds is None:
            rng = np.random.default_rng(np.random.SeedSequence([seed, episode, frame]))
            goal_delta = int(rng.integers(0, max_future + 1))
            goal_mode = "uniform_available_0_to_4_seconds"
        else:
            goal_delta = min(int(round(goal_seconds * self.fps)), n - 1 - frame)
            goal_mode = "fixed_delay_clamped_to_episode"
        goal_frame = frame + goal_delta
        task = self.tasks[int(rows["task_index"][frame])]
        return {
            "source": {"repo": REPO, "revision": REVISION, "subset": SUBSET},
            "episode": episode, "frame": frame, "split": self.split(episode),
            "timestamp_seconds": float(rows["timestamp"][frame]), "fps": self.fps,
            "camera_order": list(CAMERAS),
            "proprioception": {"position": rows["observation.state"][frame].tolist(), "names": self.names,
                               "units": "source_native", "units_per_dimension": UNITS,
                               "units_reference": UNITS_SOURCE},
            "text": {"task": task, "task_source": "dataset", "subtask": None,
                     "control_mode": "joint_position", "control_mode_source": "adapter_schema",
                     "quality": None, "mistake": None, "speed": None},
            "actions": {"values": rows["action"][np.minimum(action_indices, n - 1)].tolist(),
                        "valid": (action_indices < n).tolist(), "names": self.names,
                        "units": "source_native", "units_per_dimension": UNITS,
                        "units_reference": UNITS_SOURCE, "start_frame": frame, "step_seconds": 1 / self.fps},
            "history": {"frames": np.maximum(history_indices, 0).tolist(), "valid": (history_indices >= 0).tolist(),
                        "position": rows["observation.state"][np.maximum(history_indices, 0)].tolist()},
            "visual_subgoal": {"frame": goal_frame, "delta_seconds": goal_delta / self.fps,
                               "sampling": goal_mode, "source": "future_demonstration_frame",
                               "training_only": True, "strictly_future": goal_frame > frame},
        }

    def arrays(self, episode: int, frame: int, *, size: int = 448, **kwargs) -> dict[str, np.ndarray]:
        """Framework-free encoder inputs + supervised action targets.

        history_images: [T,3,H,W,3]; images/subgoal_images: [3,H,W,3].
        Strings use NumPy Unicode, so np.load(..., allow_pickle=False) works.
        """
        sample = self.sample(episode, frame, **kwargs)
        needed = set(sample["history"]["frames"] + [sample["visual_subgoal"]["frame"]])
        decoded = {index: np.stack([self.image(episode, index, cam, size) for cam in CAMERAS]) for index in needed}
        return {
            "images": decoded[frame],
            "camera_names": np.asarray(CAMERAS),
            "proprioception": np.asarray(sample["proprioception"]["position"], dtype=np.float32),
            "history_images": np.stack([decoded[index] for index in sample["history"]["frames"]]),
            "history_proprioception": np.asarray(sample["history"]["position"], dtype=np.float32),
            "history_valid": np.asarray(sample["history"]["valid"], dtype=bool),
            "history_frames": np.asarray(sample["history"]["frames"], dtype=np.int64),
            "subgoal_images": decoded[sample["visual_subgoal"]["frame"]],
            "subgoal_valid": np.ones(len(CAMERAS), dtype=bool),
            "subgoal_frame": np.asarray(sample["visual_subgoal"]["frame"], dtype=np.int64),
            "subgoal_delta_seconds": np.asarray(sample["visual_subgoal"]["delta_seconds"], dtype=np.float32),
            "subgoal_strictly_future": np.asarray(sample["visual_subgoal"]["strictly_future"]),
            "task": np.asarray(sample["text"]["task"]),
            "subtask": np.asarray(""),
            "subtask_valid": np.asarray(False),
            "control_mode": np.asarray(sample["text"]["control_mode"]),
            "actions": np.asarray(sample["actions"]["values"], dtype=np.float32),
            "action_valid": np.asarray(sample["actions"]["valid"], dtype=bool),
            "metadata_json": np.asarray(json.dumps(sample)),
        }

    @lru_cache(maxsize=1)
    def train_statistics(self) -> dict:
        """Fit on train episodes only, not the source's all-episode statistics."""
        result = {"split": "train", "episode_ids": self.episode_ids("train"), "features": {}}
        for key in ("observation.state", "action"):
            values = np.concatenate([self._rows(ep)[key] for ep in result["episode_ids"]]).astype(np.float64)
            std = values.std(axis=0)
            result["features"][key] = {"mean": values.mean(axis=0).tolist(), "std": std.tolist(),
                                       "scale": np.maximum(std, 1e-6).tolist(), "count": len(values)}
        return result

    def prepare(self, episode: int = 0, all_videos: bool = False) -> None:
        if all_videos and self.root is None:
            snapshot_download(REPO, repo_type="dataset", revision=REVISION, allow_patterns=f"{SUBSET}/**", max_workers=4)
        self._rows(episode)
        for camera in CAMERAS:
            self._video(episode, camera)
