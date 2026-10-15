"""Camera transfer checks: projection, wrist attachment, native capture parity."""
import copy
import json
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from PIL import Image

from pi_sim import RobotEnv, RobotBatch
from pi_sim.cameras import CAMERAS, DEFAULT_RIG, load_camera_rig
from pi_observe import Observer, SimSource
from pi_observe.hardware import SO101Source


def rig():
    return json.loads(DEFAULT_RIG.read_text())


class CameraTests(unittest.TestCase):
    def test_native_capture_and_encoder_preprocessing_match(self):
        config = rig()
        config["cameras"]["overhead"].update(width=320, height=240, rotation=90,
                                              intrinsics={"fx": 270, "fy": 270, "cx": 160, "cy": 120})
        with RobotEnv(images=True, image_size=32, camera_rig=config) as env:
            obs, _ = env.reset(seed=0)
            before = env.data.qpos.copy()
            observation = Observer(SimSource(env), task="Inspect", size=32).read()
            self.assertEqual(tuple(observation.images), CAMERAS)
            for i, name in enumerate(CAMERAS):
                native = observation.images[name]
                self.assertEqual(native.shape, (320, 240, 3) if name == "overhead" else (480, 640, 3))
                expected = np.asarray(Image.fromarray(native).resize((32, 32), Image.Resampling.BILINEAR))
                np.testing.assert_array_equal(observation.inputs["images"][i], expected)
                # Separate renders of one state may differ by one LSB (GPU rasterization).
                np.testing.assert_allclose(obs["images"][name].astype(int), expected.astype(int), atol=1)
            np.testing.assert_array_equal(env.data.qpos, before)
            self.assertEqual(env.data.time, 0)
            metadata = observation.metadata["details"]["cameras"]["overhead"]
            self.assertEqual(metadata["rotation_ccw_degrees"], 90)
            self.assertEqual(metadata["calibration_status"], "nominal")
            self.assertIn("B030401", metadata["hardware_model"])
            self.assertIn("unmeasured", metadata["optics_note"])
            self.assertTrue(env.observation_space.contains(obs))

    def test_wrist_pose_tracks_its_arm_while_fixed_cameras_stay_fixed(self):
        with RobotEnv(control_mode="joint") as env:
            env.reset(seed=0)
            poses = {name: np.asarray(env.camera_calibration(name)["world_from_camera_opencv_before_rotation"])
                     for name in CAMERAS}
            cid = env.model.camera("left_wrist").id
            parent = env.model.body("left_camera_mount").id
            self.assertEqual(env.model.cam_bodyid[cid], parent)
            local = env.data.xmat[parent].reshape(3, 3).T @ (env.data.cam_xpos[cid] - env.data.xpos[parent])
            np.testing.assert_allclose(local, env.camera_specs["left_wrist"].pos, atol=1e-9)
            env.data.qpos[env.robot["left"].qpos_ids[0]] += 0.4
            mujoco.mj_forward(env.model, env.data)
            for name in CAMERAS:
                after = np.asarray(env.camera_calibration(name)["world_from_camera_opencv_before_rotation"])
                if name == "left_wrist":
                    self.assertGreater(np.linalg.norm(after - poses[name]), 0.1)
                else:
                    np.testing.assert_array_equal(after, poses[name])
            for name in ("overhead",):
                self.assertGreaterEqual(env.model.geom(f"{name}_camera_housing").id, 0)
                self.assertGreaterEqual(env.model.geom(f"{name}_camera_lens").id, 0)
            self.assertNotIn("front", CAMERAS)

    def test_rendered_projection_matches_opencv_calibration(self):
        config = rig()
        config["cameras"]["overhead"].update(
            width=320, height=240, intrinsics={"fx": 240, "fy": 300, "cx": 110, "cy": 90})
        with RobotEnv(camera_rig=config) as env:
            env.reset(seed=0)
            point = np.array([0.08, -0.06, 0.5])
            transform = np.asarray(env.camera_calibration("overhead")["world_from_camera_opencv_before_rotation"])
            env.data.joint("cube_joint").qpos[:3] = (transform @ np.r_[point, 1])[:3]
            geom = env.model.geom("cube_geom").id
            env.model.geom_type[geom] = mujoco.mjtGeom.mjGEOM_SPHERE
            env.model.geom_size[geom] = 0.005
            mujoco.mj_forward(env.model, env.data)
            env.render_camera("overhead", native=True)
            renderer = env._renderers[(320, 240)]
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            rows, columns = np.where((segmentation[..., 0] == geom) &
                                     (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
            self.assertGreater(len(rows), 10)
            expected = env.camera_specs["overhead"].K @ point
            # OpenGL pixel centers lie half a pixel into the pixel grid.
            np.testing.assert_allclose([columns.mean(), rows.mean()], expected[:2] / expected[2] - 0.5, atol=0.5)

    def test_distorted_feed_matches_projected_object_without_black_border(self):
        import cv2
        config = rig()
        config["cameras"]["overhead"].update(
            width=320, height=240, intrinsics={"fx": 240, "fy": 260, "cx": 150, "cy": 110},
            distortion=[-0.28, 0.07, 0.01, -0.02, 0.01])
        spec = load_camera_rig(config)["overhead"]
        with RobotEnv(camera_rig=config) as env:
            env.reset(seed=0)
            point = np.array([0.18, -0.1, 0.5])
            transform = np.asarray(env.camera_calibration("overhead")["world_from_camera_opencv_before_rotation"])
            env.data.joint("cube_joint").qpos[:3] = (transform @ np.r_[point, 1])[:3]
            geom = env.model.geom("cube_geom").id
            env.model.geom_type[geom] = mujoco.mjtGeom.mjGEOM_SPHERE
            env.model.geom_size[geom] = 0.012
            env.model.geom_rgba[geom] = [1, 0, 1, 1]
            mujoco.mj_forward(env.model, env.data)
            image = env.render_camera("overhead", native=True)
            rows, columns = np.where((image[..., 0] > 70) & (image[..., 2] > 70) & (image[..., 1] < 30))
            self.assertGreater(len(rows), 30)
            expected, _ = cv2.projectPoints(point[None], np.zeros(3), np.zeros(3), spec.K,
                                            np.asarray(spec.distortion))
            np.testing.assert_allclose([columns.mean(), rows.mean()], expected.ravel() - 0.5, atol=1)
            width, height, _, mapping = env._camera_plans["overhead"]
            self.assertGreater(width, spec.width)
            self.assertGreater(height, spec.height)
            self.assertGreaterEqual(mapping.min(), 0)
            self.assertLessEqual(mapping[..., 0].max(), width - 1)
            self.assertLessEqual(mapping[..., 1].max(), height - 1)

    def test_shared_hardware_profile_and_mismatch_rejection(self):
        config = {"arms": {side: {"port": side, "calibration": "unused.json"} for side in ("left", "right")},
                  "cameras": {name: {"device": i} for i, name in enumerate(CAMERAS)},
                  "camera_rig": rig()}
        config["camera_rig"]["cameras"]["overhead"]["rotation"] = 90
        with patch("pi_observe.hardware.ReadOnlyArm"):
            source = SO101Source(config)
        for name, camera in source.cameras.items():
            self.assertEqual(camera.config["width"], 640)
            self.assertEqual(camera.config["height"], 480)
            self.assertEqual(camera.rotation, 90 if name == "overhead" else 0)
        conflict = copy.deepcopy(config)
        conflict["cameras"]["overhead"]["width"] = 1280
        with self.assertRaisesRegex(ValueError, "conflicts with camera_rig"):
            SO101Source(conflict)
        # Prove that a USB driver's fallback mode cannot silently invalidate K.
        from concurrent.futures import ThreadPoolExecutor
        source.connected = True
        source.executor = ThreadPoolExecutor(max_workers=2)
        for arm in source.arms.values():
            arm.read.return_value = (np.zeros(6), 1, {})
        fake = unittest.mock.Mock()
        fake.latest.return_value = (np.zeros((16, 16, 3), np.uint8), 1, {})
        source.cameras = {name: fake for name in CAMERAS}
        try:
            with self.assertRaisesRegex(ValueError, "actual image shape"):
                source.read()
        finally:
            source.close()

    def test_batch_accepts_same_camera_profile(self):
        config = rig()
        config["cameras"]["overhead"].update(width=160, height=120,
            intrinsics={"fx": 120, "fy": 120, "cx": 80, "cy": 60})
        with RobotBatch(2, image_size=32, cameras=("overhead",), camera_rig=config) as batch:
            result = batch.observations()
            self.assertEqual(result["images"].shape, (2, 1, 32, 32, 3))
            self.assertEqual(batch._env.render_camera("overhead", native=True).shape, (120, 160, 3))

    def test_invalid_optics_and_mount_frames_are_rejected(self):
        for change in ({"width": 0}, {"rotation": 45}, {"distortion": [0, 0]},
                       {"intrinsics": {"fx": -1, "fy": 1, "cx": 0, "cy": 0}},
                       {"pose": {"parent_body": "world", "pos": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]}}):
            config = rig()
            config["cameras"]["left_wrist"].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_camera_rig(config)


if __name__ == "__main__":
    unittest.main()
