"""Capture integration and mocked physical transport tests (no robot required)."""
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pi_observe import Observer, SensorSnapshot, SimSource, TimingError
from pi_observe.hardware import JOINTS, LatestCamera, ReadOnlyArm, normalize_positions, signed_magnitude, SO101Source


class FakeSource:
    def __init__(self):
        self.age_ns = 0
        self.skew_ns = 0
        self.closed = False
        self.rgb = np.full((24, 32, 3), [20, 40, 60], dtype=np.uint8)

    def read(self):
        now = time.monotonic_ns() - self.age_ns
        return SensorSnapshot({"left_wrist": self.rgb, "right_wrist": self.rgb, "right_front": self.rgb},
                              np.arange(12), [f"joint_{i}" for i in range(12)], ["native"] * 12,
                              {"camera/left_wrist": now - self.skew_ns, "camera/right_wrist": now,
                               "camera/right_front": now, "state/left": now, "state/right": now}, "test")

    def close(self):
        self.closed = True


def calibration():
    return {name: {"id": i, "drive_mode": 0, "homing_offset": -4, "range_min": 100, "range_max": 3900}
            for i, name in enumerate(JOINTS, 1)}


class CaptureTest(unittest.TestCase):
    def test_encoder_contract_goals_and_copies(self):
        source = FakeSource()
        observer = Observer(source, task="Fold paper", history=6, size=32)
        observation = observer.read()
        x = observation.inputs
        self.assertEqual(x["images"].shape, (3, 32, 32, 3))
        self.assertEqual(x["proprioception"].shape, (12,))
        self.assertEqual(x["history_valid"].tolist(), [False] * 5 + [True])
        self.assertFalse(x["subgoal_valid"].any())
        self.assertFalse(x["subtask_valid"])
        self.assertIsNone(observation.metadata["action_labels"])
        x["images"][:] = 0
        observation.images["left_wrist"][:] = 0
        np.testing.assert_array_equal(source.rgb[0, 0], [20, 40, 60])
        observer.set_context(task="Fold paper", subtask="Align edges", goal_images={"left_wrist": source.rgb})
        newer = observer.read()
        self.assertEqual(newer.inputs["subgoal_valid"].tolist(), [True, False, False])
        self.assertEqual(str(newer.inputs["subtask"]), "Align edges")
        np.testing.assert_array_equal(newer.inputs["history_images"][0, 0, 0, 0], [20, 40, 60])
        observer.close()
        self.assertTrue(source.closed)
        with self.assertRaises(RuntimeError):
            observer.read()

    def test_timing_and_missing_camera_fail_loudly(self):
        source = FakeSource()
        observer = Observer(source, task="Fold", max_age_ms=200, max_skew_ms=50)
        source.age_ns = 300_000_000
        with self.assertRaises(TimingError):
            observer.read()
        source.age_ns, source.skew_ns = 0, 100_000_000
        with self.assertRaises(TimingError):
            observer.read()
        source.skew_ns = 0
        raw = source.read()
        del raw.timestamps_ns["camera/left_wrist"]
        with patch.object(source, "read", return_value=raw), self.assertRaises(ValueError):
            observer.read()

    def test_one_second_history_and_episode_reset(self):
        source = FakeSource()
        observer = Observer(source, task="Fold", history=3, size=16)
        # A controlled clock avoids sleeps and proves temporal history semantics.
        for seconds in (10, 11, 12):
            with patch("time.monotonic_ns", return_value=seconds * 1_000_000_000):
                observation = observer.read()
        self.assertEqual(observation.inputs["history_valid"].tolist(), [True, True, True])
        self.assertEqual(observation.inputs["history_timestamps_ns"].tolist(), [10_000_000_000, 11_000_000_000, 12_000_000_000])
        observer.clear_history()
        with patch("time.monotonic_ns", return_value=13_000_000_000):
            observation = observer.read()
        self.assertEqual(observation.inputs["history_valid"].tolist(), [False, False, True])

    def test_capture_round_trip(self):
        observation = Observer(FakeSource(), task="Fold", size=32).read()
        with tempfile.TemporaryDirectory() as temp:
            directory = observation.save(Path(temp) / "capture")
            self.assertTrue((directory / "COMPLETE").exists())
            for name in observation.images:
                np.testing.assert_array_equal(np.asarray(Image.open(directory / f"{name}.png")), observation.images[name])
            with np.load(directory / "encoder.npz", allow_pickle=False) as loaded:
                np.testing.assert_array_equal(loaded["proprioception"], observation.proprioception)
                self.assertEqual(str(loaded["task"]), "Fold")
            self.assertEqual(json.loads((directory / "metadata.json").read_text())["source"], "test")
            with self.assertRaises(FileExistsError):
                observation.save(directory)

    def test_direct_observations_dictionary(self):
        with Observer(FakeSource(), task="Fold", size=32) as robot:
            obs = robot.observations()
            self.assertEqual(obs["images"].shape, (3, 32, 32, 3))
            self.assertEqual(obs["proprioception"].shape, (12,))

    def test_live_sim_read_does_not_advance_or_reset(self):
        from pi_sim import RobotEnv
        with RobotEnv(image_size=32, control_mode="joint") as env:
            env.reset(seed=0)
            qpos, before = env.data.qpos.copy(), env.data.time
            with Observer(SimSource(env), task="Reach", size=32) as observer:
                observation = observer.read()
                self.assertEqual(observation.inputs["images"].shape, (3, 32, 32, 3))
                self.assertEqual(observation.proprioception.shape, (14,))
                self.assertEqual(observation.metadata["state_units"][0], "radian")
                self.assertEqual(observation.metadata["host_timing_spread_ms"], 0)
                np.testing.assert_array_equal(env.data.qpos, qpos)
                self.assertEqual(env.data.time, before)
            env.step(env.hold_action())  # Observer does not own/close env.

    def test_camera_worker_rgb_rotation_failure_and_cleanup(self):
        class Capture:
            fail = False
            released = False
            def isOpened(self): return True
            def set(self, *args): return True
            def read(self):
                time.sleep(0.005)
                return (False, None) if self.fail else (True, np.full((24, 32, 3), [10, 20, 30], np.uint8))
            def release(self): self.released = True
        capture = Capture()
        def convert(bgr, mode):
            self.assertEqual(mode, 4)
            return bgr[..., ::-1].copy()
        cv2 = SimpleNamespace(VideoCapture=lambda device: capture, cvtColor=convert, COLOR_BGR2RGB=4,
                              CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38)
        with patch.dict("sys.modules", {"cv2": cv2}):
            camera = LatestCamera({"device": 0, "rotation": 90})
            camera.connect()
            rgb, timestamp, details = camera.latest()
            self.assertEqual(rgb.shape, (32, 24, 3))
            np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
            self.assertLessEqual(details["read_start_ns"], timestamp)
            capture.fail = True
            with camera._condition:
                self.assertTrue(camera._condition.wait_for(lambda: camera._error is not None, timeout=1))
            with self.assertRaises(OSError):
                camera.latest()
            camera.close()
            self.assertTrue(capture.released)


class ServoTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "calibration.json"
        self.path.write_text(json.dumps(calibration()))
        self.calls = []
        self.error = 0
        self.bad_calibration = False
        self.port = SimpleNamespace(is_open=False)
        def open_port(baud):
            self.calls.append(("host_baud", baud))
            self.port.is_open = True
            return True
        def close_port():
            self.calls.append(("close",))
            self.port.is_open = False
        self.port.setBaudRate, self.port.closePort = open_port, close_port
        def ping(port, motor):
            self.calls.append(("ping", motor))
            return 777, 0, 0
        def read2(port, motor, address):
            self.calls.append(("read", motor, address))
            return {9: 99 if self.bad_calibration else 100, 11: 3900, 31: (1 << 11) | 4}[address], 0, 0
        def read1(port, motor, address):
            self.calls.append(("read", motor, address))
            return 0, 0, 0
        def sync(port, address, length, ids, count):
            self.calls.append(("sync_read", address, length, tuple(ids)))
            return 0
        def receive(port, motor, length):
            self.calls.append(("receive", motor))
            value = 2000 if motor < 6 else 3900
            return [value & 255, value >> 8], 0, self.error
        self.packet = SimpleNamespace(ping=ping, read2ByteTxRx=read2, read1ByteTxRx=read1, syncReadTx=sync, readRx=receive)
        self.sdk = SimpleNamespace(PortHandler=lambda port: self.port, PacketHandler=lambda protocol: self.packet, COMM_SUCCESS=0)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_read_operations_and_native_units(self):
        with patch.dict("sys.modules", {"scservo_sdk": self.sdk}):
            reader = ReadOnlyArm("fake", str(self.path))
            reader.connect()
            state, timestamp, info = reader.read()
            reader.close()
        np.testing.assert_allclose(state, [0, 0, 0, 0, 0, 100])
        self.assertLessEqual(info["read_start_ns"], timestamp)
        self.assertLessEqual(timestamp, info["read_end_ns"])
        self.assertEqual(set(call[0] for call in self.calls), {"host_baud", "ping", "read", "sync_read", "receive", "close"})
        self.assertFalse(self.port.is_open)

    def test_bad_calibration_and_servo_error_fail(self):
        with patch.dict("sys.modules", {"scservo_sdk": self.sdk}):
            self.bad_calibration = True
            reader = ReadOnlyArm("fake", str(self.path))
            with self.assertRaises(ValueError):
                reader.connect()
            self.assertFalse(self.port.is_open)
            self.bad_calibration = False
            reader.connect()
            self.error = 4
            with self.assertRaises(OSError):
                reader.read()
            reader.close()

    def test_normalization_sign_and_gripper_direction(self):
        self.assertEqual(signed_magnitude((1 << 15) | 10, 15), -10)
        cal = calibration()
        raw = {name: 2000 for name in JOINTS}
        raw["shoulder_pan"] = 2000 + 4095 / 4
        raw["gripper"] = 100
        value = normalize_positions(raw, cal)
        self.assertAlmostEqual(value[0], 90)
        self.assertEqual(value[-1], 0)
        cal["gripper"]["drive_mode"] = 1
        self.assertEqual(normalize_positions(raw, cal)[-1], 100)

    def test_bimanual_sensor_order_and_cleanup(self):
        config = {"arms": {side: {"port": side, "calibration": str(self.path)} for side in ("left", "right")},
                  "cameras": {name: {"device": i} for i, name in enumerate(("left_wrist", "right_wrist", "right_front"))}}
        source = SO101Source(config)
        class Arm:
            def __init__(self, n): self.n, self.closed = n, False
            def read(self): return np.full(6, self.n), time.monotonic_ns(), {}
            def close(self): self.closed = True
        class Camera:
            def latest(self): return np.zeros((16, 16, 3), np.uint8), time.monotonic_ns(), {}
            def close(self): pass
        from concurrent.futures import ThreadPoolExecutor
        source.arms = {"left": Arm(1), "right": Arm(2)}
        source.cameras = {name: Camera() for name in config["cameras"]}
        source.executor = ThreadPoolExecutor(max_workers=2)
        source.connected = True
        with Observer(source, task="Fold", size=16) as observer:
            observation = observer.read()
            np.testing.assert_array_equal(observation.proprioception, [1] * 6 + [2] * 6)
            self.assertEqual(observation.metadata["state_names"][6], "right_shoulder_pan.pos")
        self.assertTrue(all(arm.closed for arm in source.arms.values()))


if __name__ == "__main__":
    unittest.main()
