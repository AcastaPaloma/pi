"""Static robot scene batches for encoder development; no policy or rollouts."""
from __future__ import annotations

import mujoco
import numpy as np

from .env import RobotEnv
from .scene import JOINTS


class RobotBatch:
    """128 independent static scenes, one shared model and rendering context.

    observations() renders each changed scene once, then returns NumPy arrays.
    resample() explicitly changes the scenes. Reads never advance physics.
    Rendering is sequential in stock MuJoCo, NOT GPU-vectorized simulation.
    Call all methods on the thread that created this object.
    """

    def __init__(self, batch_size=128, image_size=224, seed=0,
                 cameras=("left_wrist", "right_wrist", "front")):
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be 1..4096")
        if not cameras or len(set(cameras)) != len(cameras) or set(cameras) - set(RobotEnv.camera_names):
            raise ValueError(f"Choose distinct cameras from {RobotEnv.camera_names}")
        self.batch_size, self.image_size, self.cameras = batch_size, image_size, tuple(cameras)
        # One environment to compile assets and settle the home grippers once.
        # Its MjData is scratch space for rendering independent stored poses.
        self._env = RobotEnv(image_size=image_size, control_mode="joint")
        self._env.reset(seed=seed)
        self._home = self._env.data.qpos.copy()
        self._home_mocap = self._env.data.mocap_pos.copy()
        self._joint_indices = np.concatenate([arm.qpos_ids for arm in self._env.robot.values()])
        self._limits = np.concatenate([self._env.model.jnt_range[arm.joint_ids] for arm in self._env.robot.values()])
        self._rng = np.random.default_rng(seed)
        self._qpos = np.tile(self._home, (batch_size, 1))
        self._mocap = np.tile(self._home_mocap, (batch_size, 1, 1))
        self._cache = None
        self._closed = False
        self._generation = 0
        self.resample(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("RobotBatch is closed")

    def resample(self, seed=None):
        """Sample new static poses and object positions; invalidate cached RGB.

        Small joint perturbations are for visual/input testing. They are not
        collision-validated demonstrations or examples of successful folding.
        """
        self._check_open()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._qpos[:] = self._home
        offsets = self._rng.uniform(-0.15, 0.15, (self.batch_size, 12))
        self._qpos[:, self._joint_indices] = np.clip(self._home[self._joint_indices] + offsets, self._limits[:, 0], self._limits[:, 1])
        cube_index = self._env.model.joint("cube_joint").qposadr[0]
        self._qpos[:, cube_index:cube_index + 2] = self._rng.uniform([-0.35, -0.22], [0.15, 0.20], (self.batch_size, 2))
        self._mocap[:] = self._home_mocap
        self._mocap[:, 0, :2] = self._rng.uniform([-0.35, -0.05], [0.1, 0.3], (self.batch_size, 2))
        self._generation += 1
        self._cache = None
        return self

    def observations(self, *, images=True, copy=True):
        """Return an in-memory batch. Default arrays are owned and writable.

        copy=False returns read-only views of the cache for inspection. Copy
        before passing those views to libraries that might write into them.
        images=False reads proprioception without rendering any cameras.
        """
        self._check_open()
        if self._cache is None:
            positions = []
            names, units = [], []
            for side, arm in self._env.robot.items():
                # Hand linkage is unchanged from the settled home configuration.
                gripper_id = self._env.model.joint(f"{side}_gripper_right_driver_joint").qposadr[0]
                opening = np.clip(1 - self._qpos[:, gripper_id] / 0.8, 0, 1)
                positions.extend([self._qpos[:, arm.qpos_ids], opening[:, None]])
                names.extend([f"{side}_{joint}.pos" for joint in JOINTS] + [f"{side}_gripper.pos"])
                units.extend(["radian"] * 6 + ["gripper_fraction_0_closed_1_open"])
            self._cache = {
                "proprioception": np.concatenate(positions, axis=1).astype(np.float32),
                "camera_names": np.asarray(self.cameras),
                "state_names": np.asarray(names), "state_units": np.asarray(units),
                "task": np.full(self.batch_size, "Observe the robot and table."),
                "source": np.asarray("mujoco_static_bimanual_ur5e"),
                "scene_id": np.arange(self.batch_size, dtype=np.int64),
                "generation": np.asarray(self._generation),
            }
        if images and "images" not in self._cache:
            rgb = np.empty((self.batch_size, len(self.cameras), self.image_size, self.image_size, 3), dtype=np.uint8)
            for i in range(self.batch_size):
                self._env.data.qpos[:] = self._qpos[i]
                self._env.data.qvel[:] = 0
                self._env.data.mocap_pos[:] = self._mocap[i]
                mujoco.mj_forward(self._env.model, self._env.data)
                for c, camera in enumerate(self.cameras):
                    rgb[i, c] = self._env.render_camera(camera)
            self._cache["images"] = rgb
        result = {}
        for key, value in self._cache.items():
            if key == "images" and not images:
                continue
            value.setflags(write=False)
            result[key] = value.copy() if copy else value.view()
        return result

    def grid(self, batch=None, **kwargs):
        """PIL contact sheet. display(robot.grid()) in a notebook."""
        from pi_observe.grid import observation_grid
        return observation_grid(self.observations(copy=False) if batch is None else batch, **kwargs)

    def close(self):
        if not self._closed:
            self._env.close()
            self._cache = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
