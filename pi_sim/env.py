"""Gymnasium API, PD joint control, numerical IK, and direct MuJoCo access."""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .scene import ARMS, JOINTS, build_model


def _quat_to_rotvec(quat):
    out = np.zeros(3)
    mujoco.mju_quat2Vel(out, quat, 1.0)
    return out


def _rotvec_to_quat(rotvec):
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([1., 0., 0., 0.])
    return np.r_[np.cos(angle / 2), np.sin(angle / 2) * rotvec / angle]


@dataclass
class Arm:
    """Read robot state; use env.step() to command it. Returned arrays are copies."""

    env: "RobotEnv"
    name: str

    def __post_init__(self):
        m = self.env.model
        self.joint_ids = np.array([m.joint(f"{self.name}_{j}_joint").id for j in JOINTS])
        self.qpos_ids = m.jnt_qposadr[self.joint_ids]
        self.dof_ids = m.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([m.actuator(f"{self.name}_{j}").id for j in JOINTS])
        self.gripper_actuator = m.actuator(f"{self.name}_gripper_fingers_actuator").id
        self.site_id = m.site(f"{self.name}_gripper_pinch").id

    @property
    def joint_positions(self):
        return self.env.data.qpos[self.qpos_ids].copy()

    @property
    def joint_velocities(self):
        return self.env.data.qvel[self.dof_ids].copy()

    @property
    def gripper_opening(self):
        # Normalized linkage angle, not a calibrated distance between fingertips.
        angle = self.env.data.joint(f"{self.name}_gripper_right_driver_joint").qpos[0]
        return float(np.clip(1 - angle / 0.8, 0, 1))

    @property
    def position(self):
        return self.env.data.site_xpos[self.site_id].copy()

    @property
    def quaternion(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.env.data.site_xmat[self.site_id])
        return quat


class RobotEnv(gym.Env):
    """Bimanual UR5e, with easy left-arm reach and physical pick/place tasks.

    All positions use world coordinates in meters; rotations use radians.
    `model`, `data`, and `robot['left'/'right']` are public integration points.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}
    control_dt = 0.05
    delta_scale = 0.02
    camera_names = ("front", "left_wrist", "right_wrist")

    def __init__(self, task="reach", control_mode="ee_delta", images=False,
                 image_size=448, max_episode_steps=400, render_mode=None):
        if task not in ("reach", "pick_place"):
            raise ValueError("task must be 'reach' or 'pick_place'")
        if control_mode not in ("ee_delta", "ee", "joint"):
            raise ValueError("control_mode must be 'ee_delta', 'ee', or 'joint'")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Use render_mode='rgb_array'; interactive play: python -m pi_sim")
        if not isinstance(image_size, int) or not 16 <= image_size <= 640:
            raise ValueError("image_size must be an integer between 16 and 640")
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        self.task, self.control_mode = task, control_mode
        self.images, self.image_size = images, image_size
        self.max_episode_steps, self.render_mode = max_episode_steps, render_mode
        self.model = build_model()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.robot = {side: Arm(self, side) for side in ARMS}
        self._substeps = round(self.control_dt / self.model.opt.timestep)
        self._renderer = None
        self._ready = False
        self._done = False
        self._workspace_low = np.array([[-0.50, -0.32, 0.025], [-0.08, -0.32, 0.025]])
        self._workspace_high = np.array([[0.08, 0.32, 0.60], [0.50, 0.32, 0.60]])
        if control_mode == "ee_delta":
            low, high = np.tile([-1., -1., -1., 0.], 2), np.ones(8)
        elif control_mode == "ee":
            low = np.concatenate([np.r_[p, [-np.pi] * 3, 0.] for p in self._workspace_low])
            high = np.concatenate([np.r_[p, [np.pi] * 3, 1.] for p in self._workspace_high])
        else:
            low = np.concatenate([np.r_[self.model.actuator_ctrlrange[a.actuator_ids, 0], 0.] for a in self.robot.values()])
            high = np.concatenate([np.r_[self.model.actuator_ctrlrange[a.actuator_ids, 1], 1.] for a in self.robot.values()])
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32))
        obs = {"state": spaces.Box(-np.inf, np.inf, (14,), np.float32),
               "joint_velocity": spaces.Box(-np.inf, np.inf, (12,), np.float32),
               "ee_pose": spaces.Box(-np.inf, np.inf, (14,), np.float32),
               "cube_position": spaces.Box(-np.inf, np.inf, (3,), np.float32),
               "goal_position": spaces.Box(-np.inf, np.inf, (3,), np.float32)}
        if images:
            obs["images"] = spaces.Dict({c: spaces.Box(0, 255, (image_size, image_size, 3), np.uint8) for c in self.camera_names})
        self.observation_space = spaces.Dict(obs)

    @property
    def goal_position(self):
        pos = self.data.mocap_pos[0].copy()
        pos[2] = 0.14 if self.task == "reach" else 0.02
        return pos

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        unknown = options.keys() - {"randomize"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        mujoco.mj_resetData(self.model, self.data)
        for arm in self.robot.values():
            home = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])
            self.data.qpos[arm.qpos_ids] = home
            self.data.ctrl[arm.actuator_ids] = home
            self.data.ctrl[arm.gripper_actuator] = 0
        jitter = self.np_random.uniform(-0.025, 0.025, (2, 2)) if options.get("randomize", True) else np.zeros((2, 2))
        self.data.joint("cube_joint").qpos[:3] = np.r_[np.array([-0.22, -0.10]) + jitter[0], 0.021]
        self.data.mocap_pos[0] = np.r_[np.array([-0.18, 0.16]) + jitter[1], 0.001]
        self.model.site_rgba[self.model.site("reach_target").id, 3] = 0.55 if self.task == "reach" else 0
        mujoco.mj_forward(self.model, self.data)
        # Let contact constraints and the gripper's four-bar linkages settle.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        self._ee_targets = np.array([a.position for a in self.robot.values()])
        self._orientations = np.array([a.quaternion for a in self.robot.values()])
        self._steps, self._success_steps = 0, 0
        self._ik_errors = np.zeros(2)
        self._ready, self._done = True, False
        return self._observation(), self._info(False)

    def hold_action(self):
        """An action that holds the current command (useful as a starting point)."""
        if not self._ready:
            raise RuntimeError("Call reset() first")
        rows = []
        for i, arm in enumerate(self.robot.values()):
            opening = 1 - self.data.ctrl[arm.gripper_actuator] / 255
            if self.control_mode == "ee_delta":
                rows.append(np.r_[np.zeros(3), opening])
            elif self.control_mode == "ee":
                rows.append(np.r_[self._ee_targets[i], _quat_to_rotvec(self._orientations[i]), opening])
            else:
                rows.append(np.r_[self.data.ctrl[arm.actuator_ids], opening])
        return np.concatenate(rows).astype(np.float32)

    def action_toward(self, position, *, arm="left", opening=1.0):
        """Build one Cartesian action toward a world-space point; other arm holds.

        Call repeatedly until the measured gripper position reaches the point.
        This is a convenience controller, not a learned policy or motion planner.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("position must contain three finite coordinates")
        if arm not in ARMS or not np.isfinite(opening) or not 0 <= opening <= 1:
            raise ValueError("Choose arm='left'/'right' and opening between 0 and 1")
        if self.control_mode == "joint":
            raise ValueError("action_toward requires ee_delta or ee control")
        i = ARMS.index(arm)
        action = self.hold_action().reshape(2, -1)
        action[i, :3] = (position - self._ee_targets[i]) / self.delta_scale if self.control_mode == "ee_delta" else position
        action[i, -1] = opening
        return np.clip(action.ravel(), self.action_space.low, self.action_space.high)

    def _solve_ik(self, arm, position, quaternion):
        d = self._ik_data
        d.qpos[:] = self.data.qpos
        d.qvel[:] = 0
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        q = np.zeros(4)
        rotation_error = np.zeros(3)
        limits = self.model.jnt_range[arm.joint_ids]
        for _ in range(60):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            pos_error = position - d.site_xpos[arm.site_id]
            mujoco.mju_mat2Quat(q, d.site_xmat[arm.site_id])
            mujoco.mju_subQuat(rotation_error, quaternion, q)
            rotation_world = d.site_xmat[arm.site_id].reshape(3, 3) @ rotation_error
            error = np.r_[pos_error, rotation_world * 0.3]
            if np.linalg.norm(pos_error) < 0.0002 and np.linalg.norm(rotation_world) < 0.002:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, arm.site_id)
            jac = np.vstack([jp[:, arm.dof_ids], jr[:, arm.dof_ids] * 0.3])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            d.qpos[arm.qpos_ids] = np.clip(d.qpos[arm.qpos_ids] + np.clip(dq, -0.15, 0.15), limits[:, 0], limits[:, 1])
        mujoco.mj_kinematics(self.model, d)
        return d.qpos[arm.qpos_ids].copy(), float(np.linalg.norm(position - d.site_xpos[arm.site_id]))

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError("Call reset() before stepping a new/finished episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError(f"Expected finite action with shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        rows = action.reshape(2, -1)
        for i, (arm, row) in enumerate(zip(self.robot.values(), rows)):
            if self.control_mode == "joint":
                target = row[:6]
            else:
                if self.control_mode == "ee_delta":
                    self._ee_targets[i] = np.clip(self._ee_targets[i] + row[:3] * self.delta_scale,
                                                  self._workspace_low[i], self._workspace_high[i])
                else:
                    self._ee_targets[i] = row[:3]
                    self._orientations[i] = _rotvec_to_quat(row[3:6])
                target, self._ik_errors[i] = self._solve_ik(arm, self._ee_targets[i], self._orientations[i])
            # Rate-limit the PD target to avoid snapping between distant commands.
            current = self.data.ctrl[arm.actuator_ids]
            self.data.ctrl[arm.actuator_ids] = current + np.clip(target - current, -0.10, 0.10)
            self.data.ctrl[arm.gripper_actuator] = (1 - row[-1]) * 255
        for _ in range(self._substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo state became non-finite; reset the environment")
        cube = self.data.body("cube").xpos.copy()
        if self.task == "reach":
            distance = np.linalg.norm(self.robot["left"].position - self.goal_position)
            candidate = distance < 0.025
            reward = 1 - np.tanh(5 * distance)
        else:
            distance = np.linalg.norm(cube - self.goal_position)
            hand_distance = np.linalg.norm(self.robot["left"].position - cube)
            released = all(a.gripper_opening > 0.7 for a in self.robot.values())
            slow = np.linalg.norm(self.data.joint("cube_joint").qvel[:3]) < 0.05
            candidate = distance < 0.045 and abs(cube[2] - 0.02) < 0.01 and released and slow
            reward = 0.25 * (1 - np.tanh(8 * hand_distance)) + 0.25 * float(cube[2] > 0.06) + 0.5 * (1 - np.tanh(5 * distance))
        self._success_steps = self._success_steps + 1 if candidate else 0
        success = self._success_steps >= 5
        failed = cube[2] < -0.1
        terminated = bool(success or failed)
        truncated = bool(self._steps >= self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        return self._observation(), float(reward + success), terminated, truncated, self._info(success, failed)

    def step_chunk(self, actions):
        """Execute an (H, action_dim) chunk, returning transitions up to episode end."""
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[1:] != self.action_space.shape or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite chunk with shape (H, {self.action_space.shape[0]})")
        transitions = []
        for action in actions:
            transition = self.step(action)
            transitions.append(transition)
            if transition[2] or transition[3]:
                break
        return transitions

    def _observation(self):
        arms = list(self.robot.values())
        obs = {
            "state": np.concatenate([np.r_[a.joint_positions, a.gripper_opening] for a in arms]).astype(np.float32),
            "joint_velocity": np.concatenate([a.joint_velocities for a in arms]).astype(np.float32),
            "ee_pose": np.concatenate([np.r_[a.position, a.quaternion] for a in arms]).astype(np.float32),
            "cube_position": self.data.body("cube").xpos.copy().astype(np.float32),
            "goal_position": self.goal_position.astype(np.float32),
        }
        if self.images:
            obs["images"] = {c: self.render_camera(c) for c in self.camera_names}
        return obs

    def _info(self, success, failed=False):
        return {"is_success": bool(success), "failure": "cube_fell" if failed else None,
                "task": self.task, "control_mode": self.control_mode,
                "instruction": "Move the left gripper to the green sphere." if self.task == "reach" else "Place the red block on the green disk.",
                "ik_position_error": self._ik_errors.copy(), "elapsed_steps": self._steps}

    def render_camera(self, camera="front"):
        if camera not in self.camera_names:
            raise ValueError(f"Unknown camera {camera!r}; choose {self.camera_names}")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def render(self):
        return self.render_camera("front")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
