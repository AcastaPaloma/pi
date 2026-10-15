"""Gymnasium API, PD joint control, numerical IK, and direct MuJoCo access."""

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from PIL import Image

from .cameras import camera_metadata, load_camera_rig
from .scene import ARMS, CAMERAS, COLORS, JOINTS, build_model


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
        self.joint_ids = np.array([m.joint(f"{self.name}_{j}").id for j in JOINTS])
        self.qpos_ids = m.jnt_qposadr[self.joint_ids]
        self.dof_ids = m.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.array([m.actuator(f"{self.name}_{j}").id for j in JOINTS])
        self.gripper_actuator = m.actuator(f"{self.name}_gripper").id
        self.site_id = m.site(f"{self.name}_gripperframe").id

    @property
    def joint_positions(self):
        return self.env.data.qpos[self.qpos_ids].copy()

    @property
    def joint_velocities(self):
        return self.env.data.qvel[self.dof_ids].copy()

    @property
    def gripper_opening(self):
        angle = self.env.data.joint(f"{self.name}_gripper").qpos[0]
        low, high = self.env.model.actuator_ctrlrange[self.gripper_actuator]
        return float(np.clip((angle - low) / (high - low), 0, 1))

    @property
    def position(self):
        return self.env.data.site_xpos[self.site_id].copy()

    @property
    def quaternion(self):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.env.data.site_xmat[self.site_id])
        return quat


class RobotEnv(gym.Env):
    """Bimanual SO-101, with easy left-arm reach and physical pick/place tasks.

    Cartesian positions use world meters and rotations use radians.
    State and joint commands use degrees plus 0–100 grippers. Arm/raw MuJoCo
    angles remain radians. Model joint origins require a hardware calibration audit.
    `model`, `data`, and `robot['left'/'right']` are public integration points.

    Control runs at `control_hz` (default 50 Hz, the paper's robot rate). Speeds
    and dwell times are defined per second, so the behaviour per second does not
    depend on the rate; step counts do.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    cartesian_speed = 0.4      # m/s at a full-scale ee_delta action
    joint_rate_limit = 2.0     # rad/s applied to the PD targets
    camera_names = CAMERAS

    def __init__(self, task="reach", control_mode="ee_delta", images=False,
                 image_size=448, max_episode_steps=None, render_mode=None,
                 object_shape="cube", object_color="red", camera_rig=None,
                 control_hz=50, episode_seconds=20.0):
        if task not in ("reach", "pick_place"):
            raise ValueError("task must be 'reach' or 'pick_place'")
        if control_mode not in ("ee_delta", "ee", "joint"):
            raise ValueError("control_mode must be 'ee_delta', 'ee', or 'joint'")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Use render_mode='rgb_array'; interactive play: python -m pi_sim")
        if not isinstance(image_size, int) or not 16 <= image_size <= 640:
            raise ValueError("image_size must be an integer between 16 and 640")
        if not isinstance(control_hz, int) or not 5 <= control_hz <= 500 or 500 % control_hz:
            raise ValueError("control_hz must be an integer divisor of 500 between 5 and 500")
        if max_episode_steps is None:
            max_episode_steps = int(round(episode_seconds * control_hz))
        if not isinstance(max_episode_steps, int) or max_episode_steps < 1:
            raise ValueError("max_episode_steps must be a positive integer")
        self.task, self.control_mode = task, control_mode
        self.images, self.image_size = images, image_size
        self.max_episode_steps, self.render_mode = max_episode_steps, render_mode
        self.object_shape, self.object_color = object_shape, object_color
        self.control_hz = control_hz
        self.control_dt = 1.0 / control_hz
        self.delta_scale = self.cartesian_speed * self.control_dt
        self._rate_limit = self.joint_rate_limit * self.control_dt
        self._settle_steps = max(1, int(round(0.25 * control_hz)))
        self.camera_specs = load_camera_rig(camera_rig)
        self.model = self._build_model(object_shape, object_color, camera_rig)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self.robot = {side: Arm(self, side) for side in ARMS}
        self._substeps = round(self.control_dt / self.model.opt.timestep)
        self._renderers = {}
        self._camera_plans = {}
        self._ready = False
        self._done = False
        self._workspace_low = np.array([[-0.30, -0.08, 0.016], [0.05, -0.08, 0.016]])
        self._workspace_high = np.array([[-0.05, 0.13, 0.26], [0.30, 0.13, 0.26]])
        self._baseline = self._snapshot_visuals()
        self.visual_randomization = None
        if control_mode == "ee_delta":
            low, high = np.tile([-1., -1., -1., 0.], 2), np.ones(8)
        elif control_mode == "ee":
            low = np.concatenate([np.r_[p, [-np.pi] * 3, 0.] for p in self._workspace_low])
            high = np.concatenate([np.r_[p, [np.pi] * 3, 1.] for p in self._workspace_high])
        else:
            low = np.concatenate([np.r_[np.rad2deg(self.model.actuator_ctrlrange[a.actuator_ids, 0]), 0.] for a in self.robot.values()])
            high = np.concatenate([np.r_[np.rad2deg(self.model.actuator_ctrlrange[a.actuator_ids, 1]), 100.] for a in self.robot.values()])
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32))
        obs = {"state": spaces.Box(-np.inf, np.inf, (12,), np.float32),
               "joint_velocity": spaces.Box(-np.inf, np.inf, (10,), np.float32),
               "ee_pose": spaces.Box(-np.inf, np.inf, (14,), np.float32),
               "cube_position": spaces.Box(-np.inf, np.inf, (3,), np.float32),
               "goal_position": spaces.Box(-np.inf, np.inf, (3,), np.float32)}
        if images:
            obs["images"] = spaces.Dict({c: spaces.Box(0, 255, (image_size, image_size, 3), np.uint8) for c in self.camera_names})
        self.observation_space = spaces.Dict(obs)
        self._joint_targets = np.zeros((2, len(JOINTS)))

    def _build_model(self, object_shape, object_color, camera_rig):
        """Subclasses add task props here (see pi_sim.tasks)."""
        return build_model(object_shape, object_color, camera_rig)

    # ------------------------------------------------------------------ visuals
    def _snapshot_visuals(self):
        m = self.model
        return {"geom_rgba": m.geom_rgba.copy(), "light_pos": m.light_pos.copy(),
                "light_diffuse": m.light_diffuse.copy(), "light_ambient": m.light_ambient.copy(),
                "cam_pos": m.cam_pos.copy(), "cam_quat": m.cam_quat.copy(),
                "mat_rgba": m.mat_rgba.copy(), "headlight_diffuse": m.vis.headlight.diffuse.copy(),
                "headlight_ambient": m.vis.headlight.ambient.copy()}

    def restore_visuals(self):
        """Undo randomize_visuals(); the nominal rig and colors return."""
        m = self.model
        for key, value in self._baseline.items():
            if key.startswith("headlight"):
                getattr(m.vis.headlight, key.split("_")[1])[:] = value
            else:
                getattr(m, key)[:] = value
        self.visual_randomization = None

    def randomize_visuals(self, rng, *, camera_jitter=True, lighting=True, colors=True):
        """Per-episode domain randomization of appearance, not of physics.

        Table/floor colors, lighting, small camera pose jitter (within 1 cm and
        2 degrees) and a mild hue jitter of the named object color. The object
        keeps its named color class so the instruction stays truthful. Returns
        the sampled parameters for the episode record.
        """
        self.restore_visuals()
        m = self.model
        record = {}
        if colors:
            table = m.geom("table").id
            m.geom_rgba[table, :3] = rng.uniform([0.45, 0.4, 0.3], [0.85, 0.8, 0.7])
            floor = m.mat("floor_mat").id
            m.mat_rgba[floor, :3] = rng.uniform(0.6, 1.0, 3)
            obj = m.geom("cube_geom").id
            base = np.asarray(COLORS[self.object_color][:3])
            m.geom_rgba[obj, :3] = np.clip(base + rng.uniform(-0.08, 0.08, 3), 0, 1)
            record["table_rgb"] = m.geom_rgba[table, :3].tolist()
            record["object_rgb"] = m.geom_rgba[obj, :3].tolist()
        if lighting:
            light = m.light("key_light").id
            m.light_pos[light] = self._baseline["light_pos"][light] + rng.uniform([-0.5, -0.5, -0.3], [0.5, 0.5, 0.5])
            m.light_diffuse[light] = rng.uniform(0.5, 1.0) * np.ones(3)
            m.light_ambient[light] = rng.uniform(0.0, 0.25) * np.ones(3)
            m.vis.headlight.diffuse[:] = rng.uniform(0.35, 0.75)
            m.vis.headlight.ambient[:] = rng.uniform(0.15, 0.4)
            record["light_pos"] = m.light_pos[light].tolist()
            record["light_diffuse"] = float(m.light_diffuse[light][0])
        if camera_jitter:
            record["camera_jitter"] = {}
            for name in self.camera_names:
                cid = m.camera(name).id
                offset = rng.uniform(-0.01, 0.01, 3)
                axis = rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                angle = np.deg2rad(rng.uniform(-2, 2))
                delta = np.r_[np.cos(angle / 2), np.sin(angle / 2) * axis]
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, self._baseline["cam_quat"][cid], delta)
                m.cam_pos[cid] = self._baseline["cam_pos"][cid] + offset
                m.cam_quat[cid] = quat
                record["camera_jitter"][name] = {"pos_m": offset.tolist(), "angle_deg": float(np.rad2deg(angle))}
        self.visual_randomization = record
        return record

    # ---------------------------------------------------------------- episodes
    @property
    def goal_position(self):
        pos = self.data.mocap_pos[0].copy()
        pos[2] = 0.07 if self.task == "reach" else 0.015
        return pos

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        unknown = options.keys() - {"randomize", "visual_randomization", "placement_range"}
        if unknown:
            raise ValueError(f"Unknown reset options: {sorted(unknown)}")
        mujoco.mj_resetData(self.model, self.data)
        for arm in self.robot.values():
            home = np.array([0, -0.8, 0.8, 1.57, -0.0487])
            self.data.qpos[arm.qpos_ids] = home
            self.data.ctrl[arm.actuator_ids] = home
            self.data.qpos[self.model.joint(f"{arm.name}_gripper").qposadr[0]] = self.model.actuator_ctrlrange[arm.gripper_actuator, 1]
            self.data.ctrl[arm.gripper_actuator] = self.model.actuator_ctrlrange[arm.gripper_actuator, 1]
        # "small" keeps the historical 8 mm jitter; "wide" spreads objects and
        # goals over the left arm's reachable tabletop for training diversity.
        placement = options.get("placement_range", "small")
        if not options.get("randomize", True):
            jitter = np.zeros((2, 2))
        elif placement == "small":
            jitter = self.np_random.uniform(-0.008, 0.008, (2, 2))
        elif placement == "wide":
            for _ in range(100):
                jitter = self.np_random.uniform([[-0.05, -0.04], [-0.06, -0.05]], [[0.06, 0.05], [0.08, 0.06]])
                separation = np.linalg.norm((np.array([-0.20, -0.025]) + jitter[0]) - (np.array([-0.14, -0.02]) + jitter[1]))
                if separation >= 0.08:
                    break
            else:
                raise RuntimeError("Could not sample a cube/goal placement with 8 cm separation")
        else:
            raise ValueError("placement_range must be 'small' or 'wide'")
        self.data.joint("cube_joint").qpos[:3] = np.r_[np.array([-0.20, -0.025]) + jitter[0], 0.016]
        self.data.joint("cube_joint").qpos[3:] = [1, 0, 0, 0]
        if placement == "wide" and options.get("randomize", True):
            yaw = self.np_random.uniform(-np.pi / 4, np.pi / 4)
            self.data.joint("cube_joint").qpos[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.mocap_pos[0] = np.r_[np.array([-0.14, -0.02]) + jitter[1], 0.001]
        if options.get("visual_randomization", False):
            self.randomize_visuals(self.np_random)
        else:
            self.restore_visuals()
        self.model.site_rgba[self.model.site("reach_target").id, 3] = 0.55 if self.task == "reach" else 0
        mujoco.mj_forward(self.model, self.data)
        # Let contact constraints and the gripper's servo constraints settle.
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        self._ee_targets = np.array([a.position for a in self.robot.values()])
        self._orientations = np.array([a.quaternion for a in self.robot.values()])
        self._steps, self._success_steps = 0, 0
        self._ik_errors = np.zeros(2)
        self._joint_targets = np.array([self.data.ctrl[a.actuator_ids] for a in self.robot.values()])
        self._ready, self._done = True, False
        return self._observation(), self._info(False)

    def hold_action(self):
        """An action that holds the current command (useful as a starting point)."""
        if not self._ready:
            raise RuntimeError("Call reset() first")
        rows = []
        for i, arm in enumerate(self.robot.values()):
            low, high = self.model.actuator_ctrlrange[arm.gripper_actuator]
            opening = (self.data.ctrl[arm.gripper_actuator] - low) / (high - low)
            if self.control_mode == "ee_delta":
                rows.append(np.r_[np.zeros(3), opening])
            elif self.control_mode == "ee":
                rows.append(np.r_[self._ee_targets[i], _quat_to_rotvec(self._orientations[i]), opening])
            else:
                rows.append(np.r_[np.rad2deg(self.data.ctrl[arm.actuator_ids]), opening * 100])
        return np.concatenate(rows).astype(np.float32)

    @property
    def commanded_joint_targets(self):
        """The last joint-space command, degrees plus gripper 0–100, left then right.

        This is the target handed to the rate-limited PD servos, before rate
        limiting. It is the action label for joint-mode policies regardless of
        the control mode used to produce it.
        """
        if not self._ready:
            raise RuntimeError("Call reset() first")
        rows = []
        for i, arm in enumerate(self.robot.values()):
            low, high = self.model.actuator_ctrlrange[arm.gripper_actuator]
            opening = (self.data.ctrl[arm.gripper_actuator] - low) / (high - low)
            rows.append(np.r_[np.rad2deg(self._joint_targets[i]), opening * 100])
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
            if self.control_mode == "ee_delta":
                # Keep the finger approach axis downward; leave yaw unconstrained.
                axis = d.site_xmat[arm.site_id].reshape(3, 3)[:, 0]
                rotation_world = np.cross(axis, [0., 0., -1.])
            error = np.r_[pos_error, rotation_world * 0.08]
            if np.linalg.norm(pos_error) < 0.0002 and np.linalg.norm(rotation_world) < 0.002:
                break
            mujoco.mj_jacSite(self.model, d, jp, jr, arm.site_id)
            angular_jac = jr[:, arm.dof_ids]
            if self.control_mode == "ee_delta":
                angular_jac = (np.eye(3) - np.outer(axis, axis)) @ angular_jac
            jac = np.vstack([jp[:, arm.dof_ids], angular_jac * 0.08])
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
                target = np.deg2rad(row[:len(JOINTS)])
            else:
                if self.control_mode == "ee_delta":
                    self._ee_targets[i] = np.clip(self._ee_targets[i] + row[:3] * self.delta_scale,
                                                  self._workspace_low[i], self._workspace_high[i])
                else:
                    self._ee_targets[i] = row[:3]
                    self._orientations[i] = _rotvec_to_quat(row[3:6])
                target, self._ik_errors[i] = self._solve_ik(arm, self._ee_targets[i], self._orientations[i])
            self._joint_targets[i] = target
            # Rate-limit the PD target to avoid snapping between distant commands.
            current = self.data.ctrl[arm.actuator_ids]
            self.data.ctrl[arm.actuator_ids] = current + np.clip(target - current, -self._rate_limit, self._rate_limit)
            low, high = self.model.actuator_ctrlrange[arm.gripper_actuator]
            self.data.ctrl[arm.gripper_actuator] = low + (row[-1] / 100 if self.control_mode == "joint" else row[-1]) * (high - low)
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
            candidate = distance < 0.045 and abs(cube[2] - 0.015) < 0.01 and released and slow
            reward = 0.25 * (1 - np.tanh(8 * hand_distance)) + 0.25 * float(cube[2] > 0.06) + 0.5 * (1 - np.tanh(5 * distance))
        self._success_steps = self._success_steps + 1 if candidate else 0
        success = self._success_steps >= self._settle_steps
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
            "state": np.concatenate([np.r_[np.rad2deg(a.joint_positions), a.gripper_opening * 100] for a in arms]).astype(np.float32),
            "joint_velocity": np.rad2deg(np.concatenate([a.joint_velocities for a in arms])).astype(np.float32),
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
                "instruction": "Move the left gripper to the green sphere." if self.task == "reach" else f"Place the {self.object_color} {self.object_shape} on the green disk.",
                "ik_position_error": self._ik_errors.copy(), "elapsed_steps": self._steps}

    def camera_calibration(self, camera):
        """Native optics and current world pose (OpenCV axes, before rotation)."""
        if camera not in self.camera_names:
            raise ValueError(f"Unknown camera {camera!r}; choose {self.camera_names}")
        return camera_metadata(self.camera_specs[camera], self.model, self.data, camera)

    def render_camera(self, camera="overhead", *, native=False):
        """RGB from the configured lens; native=True matches physical frame size.

        Encoder views resize the native image using the physical Observer's
        bilinear preprocessing, rather than changing the camera's projection.
        Camera FPS describes the hardware profile; reads do not step physics.
        """
        if camera not in self.camera_names:
            raise ValueError(f"Unknown camera {camera!r}; choose {self.camera_names}")
        spec = self.camera_specs[camera]
        if camera not in self._camera_plans:
            self._camera_plans[camera] = spec.render_plan()
        width, height, K, mapping = self._camera_plans[camera]
        size = (width, height)
        if size not in self._renderers:
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self._renderers[size] = mujoco.Renderer(self.model, height=height, width=width)
        renderer = self._renderers[size]
        renderer.update_scene(self.data, camera=camera)
        # Explicit bounds also support unequal fx/fy and an overscanned lens.
        # MuJoCo's render target size is independent of camera.resolution.
        for view in renderer.scene.camera:
            near = view.frustum_near
            view.frustum_width = near * width / (2 * K[0, 0])
            view.frustum_center = near * (width / 2 - K[0, 2]) / K[0, 0]
            view.frustum_top = near * K[1, 2] / K[1, 1]
            view.frustum_bottom = -near * (height - K[1, 2]) / K[1, 1]
        rgb = renderer.render().copy()
        if mapping is not None:
            import cv2
            rgb = cv2.remap(rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
        if spec.rotation:
            rgb = np.rot90(rgb, spec.rotation // 90).copy()
        if not native:
            rgb = np.asarray(Image.fromarray(rgb).resize((self.image_size, self.image_size),
                                                       Image.Resampling.BILINEAR)).copy()
        return rgb

    def render(self):
        return self.render_camera("overhead")

    def close(self):
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
