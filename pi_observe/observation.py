"""Shared observation/capture contract; sensor state stays distinct from prompts."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


class TimingError(RuntimeError):
    """A sensor is stale or the host-side timing spread exceeds the limit."""


@dataclass
class SensorSnapshot:
    images: dict[str, np.ndarray]  # Native-resolution RGB uint8, owned copies.
    position: np.ndarray
    names: list[str]
    units: list[str]
    timestamps_ns: dict[str, int]  # Host monotonic timestamps, one per sensor.
    source: str
    details: dict = field(default_factory=dict)


@dataclass
class Observation:
    """One sensor capture plus the exact encoder input built from it."""
    images: dict[str, np.ndarray]
    proprioception: np.ndarray
    inputs: dict[str, np.ndarray]
    metadata: dict

    def save(self, directory: str | Path) -> Path:
        """Save native RGB PNGs, encoder arrays, and readable metadata.

        Directory must not exist. A COMPLETE marker distinguishes a finished
        capture from one interrupted while writing. No action labels are invented.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        for name, rgb in self.images.items():
            Image.fromarray(rgb).save(directory / f"{name}.png")
        np.savez_compressed(directory / "encoder.npz", **self.inputs)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        (directory / "COMPLETE").touch()
        return directory


class Observer:
    """Wrap a sensor source with prompts, reference goals and short history.

    Single-consumer interface: call read() from your control/capture thread.
    Sources implement read() -> SensorSnapshot and close(). Creating an Observer
    never starts or moves hardware. History uses host monotonic capture times.
    """
    def __init__(self, source, *, task: str, subtask: str | None = None,
                 goal_images: dict[str, np.ndarray] | None = None,
                 size: int = 448, history: int = 1,
                 max_age_ms: float = 200, max_skew_ms: float = 100):
        if not isinstance(size, int) or not 16 <= size <= 640:
            raise ValueError("size must be an integer in 16..640")
        if not isinstance(history, int) or not 1 <= history <= 6:
            raise ValueError("history must be an integer in 1..6, including current")
        if not all(np.isfinite(x) and x > 0 for x in (max_age_ms, max_skew_ms)):
            raise ValueError("Timing limits must be finite and positive")
        self.source, self.size, self.history = source, size, history
        self.max_age_ms, self.max_skew_ms = max_age_ms, max_skew_ms
        self._history = deque(maxlen=128)
        self._signature = None
        self._sequence = 0
        self._closed = False
        self.set_context(task=task, subtask=subtask, goal_images=goal_images)

    def set_context(self, *, task: str, subtask: str | None = None,
                    goal_images: dict[str, np.ndarray] | None = None):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Supply a nonempty task prompt")
        if subtask is not None and not isinstance(subtask, str):
            raise ValueError("subtask must be text or None")
        self.task, self.subtask = task, subtask
        self.goals = {name: self._rgb(image).copy() for name, image in (goal_images or {}).items()}

    @staticmethod
    def _rgb(image):
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 or 0 in image.shape:
            raise ValueError("Camera/goal images must be nonempty RGB uint8 HWC arrays")
        return image

    def _resize(self, image):
        return np.asarray(Image.fromarray(self._rgb(image)).resize((self.size, self.size), Image.Resampling.BILINEAR)).copy()

    def clear_history(self):
        """Call when resetting the simulator or starting a new robot episode."""
        self._history.clear()

    def observations(self) -> dict[str, np.ndarray]:
        """Return current encoder inputs directly, without saving a capture."""
        return self.read().inputs

    def read(self) -> Observation:
        if self._closed:
            raise RuntimeError("Observer is closed")
        raw = self.source.read()
        captured = time.monotonic_ns()
        cameras = list(raw.images)
        if not cameras or any(not name.replace("_", "").isalnum() for name in cameras):
            raise ValueError("Camera names must contain only letters, numbers, and underscores")
        if self.goals.keys() - raw.images.keys():
            raise ValueError(f"Goal images have unknown cameras: {self.goals.keys() - raw.images.keys()}")
        position = np.asarray(raw.position, dtype=np.float32).copy()
        if position.ndim != 1 or len(position) != len(raw.names) or len(position) != len(raw.units) or not np.isfinite(position).all():
            raise ValueError("Invalid proprioception or state schema")
        signature = (raw.source, tuple(cameras), tuple(raw.names), tuple(raw.units))
        if self._signature is not None and signature != self._signature:
            raise ValueError("Sensor schema changed; create a new Observer")
        if not raw.timestamps_ns or not all(f"camera/{cam}" in raw.timestamps_ns for cam in cameras):
            raise ValueError("Missing per-camera timestamps")
        if not any(key.startswith("state/") for key in raw.timestamps_ns):
            raise ValueError("Missing proprioception timestamps")
        times = np.asarray(list(raw.timestamps_ns.values()), dtype=np.int64)
        if np.any(times <= 0) or np.any(times > captured):
            raise ValueError("Sensor timestamps must be past host monotonic nanoseconds")
        ages = (captured - times) / 1e6
        skew = float((times.max() - times.min()) / 1e6)
        if ages.max() > self.max_age_ms or skew > self.max_skew_ms:
            raise TimingError(f"Observation rejected: oldest sensor {ages.max():.1f}ms, host timing spread {skew:.1f}ms; limits {self.max_age_ms}/{self.max_skew_ms}ms")
        images = {cam: self._rgb(rgb).copy() for cam, rgb in raw.images.items()}
        rgb = np.stack([self._resize(images[cam]) for cam in cameras])
        self._signature = signature
        # Retain roughly 10Hz history regardless of read frequency, while always
        # keeping the current sample. This covers 5s within the 128-item bound.
        if len(self._history) >= 2 and captured - self._history[-2][0] < 100_000_000:
            self._history.pop()
        self._history.append((captured, rgb, position))
        # Bounded memory. Nearest available sample within 250ms of each 1Hz slot;
        # missing slots are repeat-padded and marked false, never extrapolated.
        history = []
        valid = []
        for seconds_ago in range(self.history - 1, -1, -1):
            target = captured - seconds_ago * 1_000_000_000
            item = min(self._history, key=lambda item: abs(item[0] - target))
            history.append(item)
            valid.append(abs(item[0] - target) <= 250_000_000)
        while len(self._history) > 1 and self._history[0][0] < captured - (self.history - 1 + 0.25) * 1e9:
            self._history.popleft()
        goals = np.stack([self._resize(self.goals[cam]) if cam in self.goals else np.zeros_like(rgb[0]) for cam in cameras])
        control_mode = raw.details.get("control_mode", "joint_position")
        metadata = {
            "schema_version": 1, "source": raw.source, "sequence": self._sequence,
            "captured_monotonic_ns": captured, "saved_context_unix_ns": time.time_ns(),
            "camera_order": cameras, "native_image_shapes": {cam: list(image.shape) for cam, image in images.items()},
            "state_names": raw.names, "state_units": raw.units,
            "sensor_timestamps_monotonic_ns": {key: int(t) for key, t in raw.timestamps_ns.items()},
            "sensor_age_ms": dict(zip(raw.timestamps_ns, ages.tolist())), "host_timing_spread_ms": skew,
            "task": self.task, "subtask": self.subtask, "control_mode": control_mode,
            "goal_source": "user_reference_images" if self.goals else None,
            "action_labels": None, "details": raw.details,
        }
        inputs = {
            "images": rgb,
            "camera_names": np.asarray(cameras),
            "proprioception": position.copy(),
            "task": np.asarray(self.task), "subtask": np.asarray(self.subtask or ""),
            "subtask_valid": np.asarray(self.subtask is not None),
            "control_mode": np.asarray(control_mode),
            "subgoal_images": goals,
            "subgoal_valid": np.asarray([cam in self.goals for cam in cameras], dtype=bool),
            "history_images": np.stack([item[1] for item in history]),
            "history_proprioception": np.stack([item[2] for item in history]),
            "history_valid": np.asarray(valid, dtype=bool),
            "history_timestamps_ns": np.asarray([item[0] for item in history], dtype=np.int64),
            "metadata_json": np.asarray(json.dumps(metadata)),
        }
        self._sequence += 1
        # Copies in history arrays and returned tensors prevent caller mutations
        # from changing subsequent history.
        inputs["images"] = rgb.copy()
        return Observation(images, position.copy(), inputs, metadata)

    def close(self):
        if not self._closed:
            self.source.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
