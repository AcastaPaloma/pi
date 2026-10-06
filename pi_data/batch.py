"""Decode once; train from batches of memory-mapped RGB and robot state."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import shutil
import uuid

import av
import numpy as np
from PIL import Image

from .dataset import CAMERAS, REPO, REVISION, SUBSET, UNITS, FoldingDataset


def _decode_selected(path, selected, output, camera_index, size, fps):
    """One sequential decoder per video file, including camera-specific offsets."""
    selected = sorted(selected)  # (timestamp, output row)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(selected[0][0] / float(stream.time_base)), stream=stream, backward=True)
        decoded = (frame for frame in container.decode(stream) if frame.pts is not None)
        current = next(decoded, None)
        previous = None
        for target, row in selected:
            while current is not None and float(current.pts * stream.time_base) < target:
                previous, current = current, next(decoded, None)
            candidates = [frame for frame in (previous, current) if frame is not None]
            if not candidates:
                raise ValueError(f"No frame at {target}s in {path}")
            best = min(candidates, key=lambda frame: abs(float(frame.pts * stream.time_base) - target))
            if abs(float(best.pts * stream.time_base) - target) > 0.55 / fps:
                raise ValueError(f"Video timestamp mismatch at {target}s in {path}")
            output[row, camera_index] = np.asarray(best.to_image().convert("RGB").resize((size, size), Image.Resampling.BILINEAR))


def prepare_observation_cache(directory, *, dataset=None, split="train", episodes=None,
                              image_size=224, stride=1, limit=None, progress=None):
    """One-time, sequential video decoding to uncompressed .npy arrays.

    Choose one split only. Episode/frame identity is preserved. `stride` and
    `limit` are explicit small-cache controls, not silent sampling defaults.
    Downloads missing source files via FoldingDataset if needed.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Cache already exists: {directory}")
    if not isinstance(image_size, int) or not 16 <= image_size <= 1024:
        raise ValueError("image_size must be 16..1024")
    if not isinstance(stride, int) or stride < 1 or (limit is not None and (not isinstance(limit, int) or limit < 1)):
        raise ValueError("stride and optional limit must be positive integers")
    ds = dataset if dataset is not None else FoldingDataset()
    allowed = ds.episode_ids(split)
    episode_ids = allowed if episodes is None else list(episodes)
    if not episode_ids or len(set(episode_ids)) != len(episode_ids) or set(episode_ids) - set(allowed):
        raise ValueError("Choose distinct episodes belonging to the requested split")
    pairs = [(ep, frame) for ep in episode_ids for frame in range(0, ds.episode(ep)["length"], stride)]
    if limit is not None:
        pairs = pairs[:limit]
    count = len(pairs)
    required = count * len(CAMERAS) * image_size ** 2 * 3
    directory.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory.parent).free < required * 1.02 + 10_000_000:
        raise OSError(f"Cache needs approximately {required / 1e9:.2f} GB of free disk")
    staging = directory.with_name(f".{directory.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir()
    rgb = None
    try:
        rgb = np.lib.format.open_memmap(staging / "images.npy", mode="w+", dtype=np.uint8,
                                       shape=(count, len(CAMERAS), image_size, image_size, 3))
        state = np.empty((count, len(ds.names)), dtype=np.float32)
        timestamps = np.empty(count, dtype=np.float32)
        prompts = []
        jobs = defaultdict(list)
        for row, (episode, frame) in enumerate(pairs):
            values = ds._rows(episode)
            state[row] = values["observation.state"][frame]
            timestamps[row] = values["timestamp"][frame]
            prompts.append(ds.tasks[int(values["task_index"][frame])])
            for c, camera in enumerate(CAMERAS):
                path, start = ds._video(episode, camera)
                jobs[(path, c)].append((start + float(timestamps[row]), row))
        for done, ((path, c), selected) in enumerate(jobs.items(), 1):
            _decode_selected(path, selected, rgb, c, image_size, ds.fps)
            if progress is not None:
                progress(f"Decoded video {done}/{len(jobs)} ({CAMERAS[c]})")
        rgb.flush()
        del rgb
        rgb = None
        np.save(staging / "proprioception.npy", state, allow_pickle=False)
        np.save(staging / "index.npy", np.asarray(pairs, dtype=np.int64), allow_pickle=False)
        np.save(staging / "timestamp.npy", timestamps, allow_pickle=False)
        np.save(staging / "task.npy", np.asarray(prompts), allow_pickle=False)
        metadata = {"format_version": 1, "repo": REPO, "revision": REVISION, "subset": SUBSET,
                    "split": split, "count": count, "image_size": image_size, "stride": stride,
                    "episode_ids": sorted(set(ep for ep, _ in pairs)), "camera_names": list(CAMERAS),
                    "state_names": ds.names, "state_units": UNITS, "fps": ds.fps,
                    "rgb_bytes": required}
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        # Publish only a fully written cache. No partial cache is loadable.
        staging.rename(directory)
    except BaseException:
        if rgb is not None:
            del rgb
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


class FoldingBatch:
    """Fast, independent real observation samples for your encoder.

    .observations() returns a new NumPy batch. Video decode, HTTP, simulation,
    action prediction, and model code are absent from this sampling path.
    """
    def __init__(self, directory, batch_size=128, seed=0):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["format_version"] != 1:
            raise ValueError("Unsupported observation cache format")
        if not isinstance(batch_size, int) or not 1 <= batch_size <= self.metadata["count"]:
            raise ValueError(f"batch_size must be 1..{self.metadata['count']}; prepare more samples if needed")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self._arrays = {key: np.load(self.directory / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                        for key in ("images", "proprioception", "index", "timestamp", "task")}
        n, s, c, d = self.metadata["count"], self.metadata["image_size"], len(self.metadata["camera_names"]), len(self.metadata["state_names"])
        expected = {"images": ((n, c, s, s, 3), np.uint8), "proprioception": ((n, d), np.float32),
                    "index": ((n, 2), np.int64), "timestamp": ((n,), np.float32)}
        for key, (shape, dtype) in expected.items():
            if self._arrays[key].shape != shape or self._arrays[key].dtype != dtype:
                raise ValueError(f"Cache schema mismatch for {key}")
        if self._arrays["task"].shape != (n,) or self._arrays["task"].dtype.kind != "U":
            raise ValueError("Cache task strings are invalid")

    def __len__(self):
        return self.metadata["count"]

    def observations(self, indices=None):
        """Owned NumPy arrays; random samples without replacement within a batch.

        indices optionally selects explicit cache row IDs for reproducible reads.
        Across different calls samples may recur, as in ordinary random training.
        """
        if indices is None:
            indices = self.rng.choice(len(self), self.batch_size, replace=False)
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("indices must be a nonempty 1D array of valid cache row IDs")
        identity = self._arrays["index"][indices]
        return {"images": self._arrays["images"][indices],
                "proprioception": self._arrays["proprioception"][indices],
                "camera_names": np.asarray(self.metadata["camera_names"]),
                "state_names": np.asarray(self.metadata["state_names"]),
                "state_units": np.asarray(self.metadata["state_units"]),
                "task": self._arrays["task"][indices], "source": np.asarray("lehome_real_dataset"),
                "episode": identity[:, 0], "frame": identity[:, 1], "cache_index": indices.copy(),
                "timestamp_seconds": self._arrays["timestamp"][indices],
                "split": np.asarray(self.metadata["split"])}

    def grid(self, batch=None, **kwargs):
        from pi_observe.grid import observation_grid
        return observation_grid(self.observations() if batch is None else batch, **kwargs)
