"""Read an existing MuJoCo environment without stepping or resetting it."""
import time
import numpy as np

from .observation import SensorSnapshot


class SimSource:
    def __init__(self, env):
        self.env = env

    def read(self):
        if not self.env._ready:
            raise RuntimeError("Reset the environment before observing it")
        started = time.monotonic_ns()
        arms = self.env.robot
        names, units, position = [], [], []
        for side in ("left", "right"):
            arm = arms[side]
            from pi_sim.scene import JOINTS
            names.extend([f"{side}_{name}.pos" for name in JOINTS] + [f"{side}_gripper.pos"])
            units.extend(["degree"] * 5 + ["gripper_percent_0_closed_100_open"])
            position.extend([*np.rad2deg(arm.joint_positions), arm.gripper_opening * 100])
        images = {cam: self.env.render_camera(cam, native=True) for cam in self.env.camera_names}
        # Rendering is sequential, but the simulation state is unchanged for all
        # images. Use a common capture timestamp, with simulation time explicit.
        captured = time.monotonic_ns()
        return SensorSnapshot(images, np.asarray(position, dtype=np.float32), names, units,
                              {**{f"camera/{cam}": captured for cam in images}, "state/robot": captured},
                              "mujoco_bimanual_so101", {"simulation_time_seconds": float(self.env.data.time),
                              "timestamp_basis": "same frozen simulation state; host render completion",
                              "control_mode": {"joint": "joint_position", "ee": "end_effector_pose", "ee_delta": "end_effector_delta"}[self.env.control_mode],
                              "render_duration_ms": (captured - started) / 1e6,
                              "cameras": {cam: self.env.camera_calibration(cam) for cam in images},
                              "joint_velocity": np.rad2deg(np.concatenate([a.joint_velocities for a in arms.values()])).tolist(),
                              "joint_velocity_units": "degree_per_second",
                              "ee_pose_xyz_wxyz": np.concatenate([np.r_[a.position, a.quaternion] for a in arms.values()]).tolist()})

    def close(self):
        # The caller owns env and its rendering context, on this same thread.
        pass
