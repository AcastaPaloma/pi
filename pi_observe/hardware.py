"""Read-only SO-101 positions and independently captured USB camera images.

Protocol/registers and normalization follow LeRobot's STS3215 implementation.
Only PING, READ and SYNC_READ packets are sent; no goal, torque, calibration or
configuration registers are written. Calibration must already match hardware.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time

import numpy as np

from .observation import SensorSnapshot

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def signed_magnitude(value: int, bit: int) -> int:
    return -(value & ((1 << bit) - 1)) if value & (1 << bit) else value


def load_calibration(path):
    calibration = json.loads(Path(path).expanduser().read_text())
    if set(calibration) != set(JOINTS):
        raise ValueError(f"{path}: expected the six SO-101 motor calibration entries")
    for expected_id, name in enumerate(JOINTS, 1):
        cal = calibration[name]
        if cal["id"] != expected_id or not 0 <= cal["range_min"] < cal["range_max"] <= 4095:
            raise ValueError(f"Invalid id/range for {name}")
        if cal["drive_mode"] not in (0, 1) or not -2047 <= cal["homing_offset"] <= 2047:
            raise ValueError(f"Invalid drive mode/homing offset for {name}")
    return calibration


def normalize_positions(raw, calibration):
    """LeRobot degree mode: five arm joints in degrees, gripper 0..100.

    Present_Position already includes the firmware homing offset. Do not
    subtract it a second time. Degree-mode zero is the calibrated range midpoint.
    """
    result = []
    for name in JOINTS:
        cal, value = calibration[name], raw[name]
        low, high = cal["range_min"], cal["range_max"]
        if name == "gripper":
            value = 100 * (np.clip(value, low, high) - low) / (high - low)
            if cal["drive_mode"]:
                value = 100 - value
        else:
            value = (value - (low + high) / 2) * 360 / 4095
        result.append(value)
    return np.asarray(result, dtype=np.float32)


class ReadOnlyArm:
    def __init__(self, port: str, calibration: str, baudrate: int = 1_000_000):
        self.calibration = load_calibration(calibration)
        self.port_name, self.baudrate = port, baudrate
        self.port = None

    def _checked(self, response):
        value, result, error = response
        if result != self.sdk.COMM_SUCCESS or error:
            raise OSError(f"{self.port_name}: servo communication result={result}, error={error}")
        return value

    def connect(self):
        import scservo_sdk as sdk
        if self.port is not None:
            raise RuntimeError("Arm reader already connected")
        self.sdk = sdk
        port = sdk.PortHandler(self.port_name)
        self.port = port
        self.packet = sdk.PacketHandler(0)
        try:
            # Sets the HOST serial baudrate only; no motor register write.
            if not port.setBaudRate(self.baudrate):
                raise OSError(f"Cannot open {self.port_name} at {self.baudrate} baud")
            for name, cal in self.calibration.items():
                motor_id = cal["id"]
                model = self._checked(self.packet.ping(port, motor_id))
                if model != 777:
                    raise ValueError(f"{name}: expected STS3215 model 777; got {model}")
                low = self._checked(self.packet.read2ByteTxRx(port, motor_id, 9))
                high = self._checked(self.packet.read2ByteTxRx(port, motor_id, 11))
                offset = signed_magnitude(self._checked(self.packet.read2ByteTxRx(port, motor_id, 31)), 11)
                phase = self._checked(self.packet.read1ByteTxRx(port, motor_id, 18))
                if (low, high, offset) != (cal["range_min"], cal["range_max"], cal["homing_offset"]):
                    raise ValueError(f"{name}: calibration file does not match the servo; calibrate separately in LeRobot")
                if phase & 0x10:
                    raise ValueError(f"{name}: unexpected STS3215 Phase configuration; configure separately in LeRobot")
        except BaseException:
            self.close()
            raise

    def read(self):
        if self.port is None:
            raise RuntimeError("Connect the arm reader first")
        ids = [self.calibration[name]["id"] for name in JOINTS]
        start = time.monotonic_ns()
        status = self.packet.syncReadTx(self.port, 56, 2, ids, len(ids))
        if status != self.sdk.COMM_SUCCESS:
            raise OSError(f"{self.port_name}: position request failed ({status})")
        raw = {}
        # Read status packets directly: the SDK GroupSyncRead discards servo
        # error flags, whereas the observer must reject a failed sample.
        for name, motor_id in zip(JOINTS, ids):
            data = self._checked(self.packet.readRx(self.port, motor_id, 2))
            if len(data) != 2:
                raise OSError(f"{name}: incomplete position response")
            raw[name] = signed_magnitude(data[0] | (data[1] << 8), 15)
        end = time.monotonic_ns()
        return normalize_positions(raw, self.calibration), (start + end) // 2, {
            "read_start_ns": start, "read_end_ns": end, "raw_position_ticks": raw,
        }

    def close(self):
        if self.port is not None:
            if getattr(self.port, "is_open", False):
                self.port.closePort()
            self.port = None


class LatestCamera:
    """One continuously drained camera; only the newest completed frame is kept.

    Timestamps measure host read completion, NOT hardware exposure time.
    """
    def __init__(self, config):
        if not isinstance(config["device"], (str, int)) or isinstance(config["device"], bool):
            raise ValueError("Camera device must be an index or a device path")
        self.config = config
        self.rotation = config.get("rotation", 0)
        if self.rotation not in (0, 90, 180, 270):
            raise ValueError("Camera rotation must be 0, 90, 180, or 270 degrees counterclockwise")
        if any(not np.isfinite(config.get(key, default)) or config.get(key, default) <= 0
               for key, default in (("width", 640), ("height", 480), ("fps", 30))):
            raise ValueError("Camera dimensions and FPS must be positive")
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest, self._error = None, None
        self._thread, self._cap = None, None

    def connect(self):
        import cv2
        if self._thread is not None:
            raise RuntimeError("Camera is already connected")
        self._stop.clear()
        self._latest, self._error = None, None
        self._cap = cv2.VideoCapture(self.config["device"])
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise OSError(f"Cannot open camera {self.config['device']}")
        for key, prop, default in (("width", cv2.CAP_PROP_FRAME_WIDTH, 640),
                                   ("height", cv2.CAP_PROP_FRAME_HEIGHT, 480),
                                   ("fps", cv2.CAP_PROP_FPS, 30)):
            self._cap.set(prop, self.config.get(key, default))
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; backend may ignore.
        self._thread = threading.Thread(target=self._capture, daemon=True)
        self._thread.start()

    def _capture(self):
        import cv2
        sequence = 0
        try:
            while not self._stop.is_set():
                start = time.monotonic_ns()
                ok, bgr = self._cap.read()
                end = time.monotonic_ns()
                if not ok or bgr is None:
                    raise OSError(f"Camera {self.config['device']} stopped returning frames")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if self.rotation:
                    rgb = np.rot90(rgb, self.rotation // 90).copy()
                with self._condition:
                    self._latest = (rgb, end, {"frame_sequence": sequence, "read_start_ns": start, "read_end_ns": end})
                    self._condition.notify_all()
                sequence += 1
        except Exception as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()

    def latest(self, timeout=5):
        with self._condition:
            if not self._condition.wait_for(lambda: self._latest is not None or self._error is not None, timeout=timeout):
                raise TimeoutError(f"Camera {self.config['device']} produced no frame")
            if self._error is not None:
                raise OSError(f"Camera {self.config['device']} capture failed") from self._error
            rgb, timestamp, details = self._latest
            return rgb.copy(), timestamp, details.copy()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                # Do not release a VideoCapture concurrently with native read().
                raise RuntimeError("Camera driver read is stuck; stop the capture process before reconnecting")
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SO101Source:
    """Two read-only arms plus named cameras. Explicit connect; no auto-discovery."""
    def __init__(self, config: dict):
        arms = config["arms"]
        if set(arms) != {"left", "right"} or arms["left"]["port"] == arms["right"]["port"]:
            raise ValueError("Configure distinct left and right arm ports")
        cameras = config["cameras"]
        if not cameras or len(set(str(cam["device"]) for cam in cameras.values())) != len(cameras):
            raise ValueError("Configure distinct camera devices")
        self.arms = {side: ReadOnlyArm(**arms[side]) for side in ("left", "right")}
        self.cameras = {name: LatestCamera(cam) for name, cam in cameras.items()}
        self.executor = None
        self.connected = False

    def connect(self):
        if self.connected:
            raise RuntimeError("Source already connected")
        try:
            for arm in self.arms.values():
                arm.connect()
            for camera in self.cameras.values():
                camera.connect()
            for camera in self.cameras.values():
                camera.latest()
            self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="arm-read")
            self.connected = True
        except BaseException:
            self.close()
            raise
        return self

    def read(self):
        if not self.connected:
            raise RuntimeError("Call source.connect() first")
        futures = {side: self.executor.submit(arm.read) for side, arm in self.arms.items()}
        # Consume every future even on failure so no serial operation outlives
        # read() and overlaps a later capture.
        results, errors = {}, []
        for side, future in futures.items():
            try:
                results[side] = future.result()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        images, timestamps, details = {}, {}, {"arms": {}, "cameras": {},
            "timestamp_basis": "host camera read completion; midpoint of each arm read interval; not hardware-synchronized",
            "control_mode": "joint_position"}
        for side, (position, timestamp, detail) in results.items():
            timestamps[f"state/{side}"] = timestamp
            details["arms"][side] = detail
        for name, camera in self.cameras.items():
            images[name], timestamps[f"camera/{name}"], details["cameras"][name] = camera.latest()
        names = [f"{side}_{joint}.pos" for side in self.arms for joint in JOINTS]
        units = (["degree"] * 5 + ["gripper_percent_0_closed_100_open"]) * 2
        return SensorSnapshot(images, np.concatenate([results[side][0] for side in self.arms]), names, units,
                              timestamps, "physical_bimanual_so101", details)

    def close(self):
        self.connected = False
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        errors = []
        for device in [*self.cameras.values(), *self.arms.values()]:
            try:
                device.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
