"""A gated ladder of tabletop tasks for the bimanual SO-101 rig, plus a two-arm scripted teacher.

Rungs (TASKS order):
  pick_place_any  one object anywhere; the arm on its side places it on the green disk on that side
  bin             one object into an open container on the same side
  handover        object on one side, container on the other: the near arm sets it on the grey
                  relay pad in the middle and the far arm carries it into the container
                  (success also requires that both grippers touched the object)
  two_bins        one object per side; both arms put theirs into a shared central container at once

`TaskEnv` extends `RobotEnv` with the props and success rules; `TaskTeacher` is a privileged-state
scripted demonstrator (reads true object poses) that labels every step with a subtask sentence.
Both object and container placement are seeded from the reset seed.
"""
from __future__ import annotations

import numpy as np
import mujoco

from .env import RobotEnv
from .scene import ARMS, COLORS, build_model

TASKS = ("pick_place_any", "bin", "handover", "two_bins")
BIN_INNER = 0.045          # half-width of the container's inner floor (9 cm square)
BIN_WALL = 0.03            # wall height
PARK = {"cube2": (0.46, 0.36), "bin": (2.0, 2.0), "relay_pad": (2.0, 2.0), "goal": (2.0, 2.0)}
REGION = {  # object/prop sampling boxes per side (x range, y range), inside each arm's comfortable reach
    "left": ((-0.26, -0.10), (-0.06, 0.06)),
    "right": ((0.10, 0.26), (-0.06, 0.06)),
}
RELAY = np.array([0.0, -0.03])  # closer to the bases than the table centre, within both arms' reach
CENTER_BIN = np.array([0.0, 0.02])


def _props_xml(second_color, shape):
    size = "0.015 0.015 0.015" if shape == "cube" else "0.015 0.015"
    gtype = "box" if shape == "cube" else "cylinder"
    rgba = " ".join(map(str, COLORS[second_color]))
    t, h, w = 0.003, BIN_WALL / 2, BIN_INNER + 0.003
    walls = "".join(
        f'<geom name="bin_wall_{i}" type="box" size="{sx} {sy} {h}" pos="{px} {py} {h}" rgba="0.55 0.57 0.6 1"/>'
        for i, (sx, sy, px, py) in enumerate([(t, w, -w, 0), (t, w, w, 0), (w, t, 0, -w), (w, t, 0, w)]))
    return (f'<body name="cube2" pos="{PARK["cube2"][0]} {PARK["cube2"][1]} 0.016"><freejoint name="cube2_joint"/>'
            f'<geom name="cube2_geom" type="{gtype}" size="{size}" mass="0.015" friction="1 0.01 0.001" rgba="{rgba}"/></body>'
            f'<body name="bin" mocap="true" pos="2 2 0">'
            f'<geom name="bin_floor" type="box" size="{w} {w} 0.002" pos="0 0 0.002" rgba="0.55 0.57 0.6 1"/>{walls}</body>'
            f'<body name="relay_pad" mocap="true" pos="2 2 0.0005">'
            f'<geom name="relay_pad_geom" type="cylinder" size="0.035 0.0005" contype="0" conaffinity="0" rgba="0.45 0.45 0.48 0.8"/></body>')


class TaskEnv(RobotEnv):
    """RobotEnv with the ladder's props and success rules. Actions and observations as RobotEnv,
    plus `object2_position` and `bin_position` observations."""

    def __init__(self, task="pick_place_any", *, object_color="red", second_color="blue", object_shape="cube", **kwargs):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}")
        if second_color == object_color:
            raise ValueError("the two objects need different colors")
        self.task_name, self.second_color, self.side = task, second_color, "left"
        super().__init__(task="pick_place", object_shape=object_shape, object_color=object_color, **kwargs)
        # Let both arms reach the middle of the table (relay pad and shared container).
        self._workspace_low = np.array([[-0.30, -0.08, 0.016], [-0.06, -0.08, 0.016]])
        self._workspace_high = np.array([[0.06, 0.13, 0.26], [0.30, 0.13, 0.26]])
        m = self.model
        self._mocap = {name: m.body_mocapid[m.body(name).id] for name in ("goal", "bin", "relay_pad")}
        self._arm_geoms = {side: {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(f"{side}_")}
                           for side in ARMS}
        self._object_geoms = {"cube": m.geom("cube_geom").id, "cube2": m.geom("cube2_geom").id}

    def _build_model(self, object_shape, object_color, camera_rig):
        return build_model(object_shape, object_color, camera_rig, extra_worldbody=_props_xml(self.second_color, object_shape))

    # ------------------------------------------------------------------ helpers
    def object_position(self, name="cube"):
        return self.data.body(name).xpos.copy()

    @property
    def bin_position(self):
        return self.data.mocap_pos[self._mocap["bin"]].copy()

    def _set_object(self, name, xy, yaw):
        q = self.data.joint(f"{name}_joint").qpos
        q[:3] = [xy[0], xy[1], 0.016]
        q[3:] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def _sample(self, side, avoid=(), min_sep=0.1):
        (x0, x1), (y0, y1) = REGION[side]
        for _ in range(200):
            p = self.np_random.uniform([x0, y0], [x1, y1])
            if all(np.linalg.norm(p - a) >= min_sep for a in avoid):
                return p
        raise RuntimeError("could not sample a placement")

    def _sample_pair(self, side, min_sep):
        """Object and prop in the same region, resampling both until they are `min_sep` apart."""
        for _ in range(500):
            a, b = self._sample(side), self._sample(side)
            if np.linalg.norm(a - b) >= min_sep:
                return a, b
        raise RuntimeError("could not sample a placement pair")

    # ------------------------------------------------------------------ episodes
    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.pop("placement_range", None)
        obs, info = super().reset(seed=seed, options={**options, "placement_range": "small"})
        rng = self.np_random
        for name, (x, y) in PARK.items():
            if name in self._mocap:
                self.data.mocap_pos[self._mocap[name]] = [x, y, 0.0 if name == "bin" else 0.001]
        self._set_object("cube2", PARK["cube2"], 0.0)
        yaw = lambda: rng.uniform(-np.pi / 4, np.pi / 4)
        sides = ("left", "right")
        self.side = sides[int(rng.integers(2))]
        other = "right" if self.side == "left" else "left"
        if self.task_name == "pick_place_any":
            obj, goal = self._sample_pair(self.side, 0.08)
            self.data.mocap_pos[self._mocap["goal"]] = [goal[0], goal[1], 0.001]
        elif self.task_name == "bin":
            obj, b = self._sample_pair(self.side, 0.11)
            self.data.mocap_pos[self._mocap["bin"]] = [b[0], b[1], 0.0]
        elif self.task_name == "handover":
            obj = self._sample(self.side)
            b = self._sample(other)
            self.data.mocap_pos[self._mocap["bin"]] = [b[0], b[1], 0.0]
            self.data.mocap_pos[self._mocap["relay_pad"]] = [RELAY[0], RELAY[1], 0.0005]
        else:  # two_bins
            self.side = "both"
            obj = self._sample("left", avoid=[CENTER_BIN], min_sep=0.1)
            obj2 = self._sample("right", avoid=[CENTER_BIN], min_sep=0.1)
            self._set_object("cube2", obj2, yaw())
            self.data.mocap_pos[self._mocap["bin"]] = [CENTER_BIN[0], CENTER_BIN[1], 0.0]
        self._set_object("cube", obj, yaw())
        mujoco.mj_forward(self.model, self.data)
        for _ in range(60):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        self._ee_targets = np.array([a.position for a in self.robot.values()])
        self._touched = {side: False for side in ARMS}
        self._task_success_steps = 0
        return self._observation(), self._info(False)

    def _touches(self):
        cube = self._object_geoms["cube"]
        for c in self.data.contact[:self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if cube in pair:
                other = (pair - {cube}) or {cube}
                g = next(iter(other))
                for side in ARMS:
                    if g in self._arm_geoms[side]:
                        self._touched[side] = True

    def _in_bin(self, name):
        p, b = self.object_position(name), self.bin_position
        return bool(np.all(np.abs(p[:2] - b[:2]) < BIN_INNER) and p[2] < BIN_WALL + 0.01)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._touches()
        released = all(a.gripper_opening > 0.7 for a in self.robot.values())
        slow = np.linalg.norm(self.data.joint("cube_joint").qvel[:3]) < 0.05
        if self.task_name == "pick_place_any":
            success = bool(info["is_success"])
        else:
            if self.task_name == "two_bins":
                candidate = self._in_bin("cube") and self._in_bin("cube2") and released and slow
            else:
                candidate = self._in_bin("cube") and released and slow
            if self.task_name == "handover":
                candidate = candidate and all(self._touched.values())
            self._task_success_steps = self._task_success_steps + 1 if candidate else 0
            success = self._task_success_steps >= self._settle_steps
        failed = self.object_position("cube")[2] < -0.1 or self.object_position("cube2")[2] < -0.1
        terminated = bool(success or failed)
        truncated = bool(self._steps >= self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        return self._observation(), reward, terminated, truncated, self._info(success, failed)

    def _observation(self):
        obs = super()._observation()
        if hasattr(self, "_mocap"):
            obs["object2_position"] = self.object_position("cube2").astype(np.float32)
            obs["bin_position"] = self.bin_position.astype(np.float32)
        return obs

    def instruction(self):
        a, b = f"{self.object_color} {self.object_shape}", f"{self.second_color} {self.object_shape}"
        if self.task_name == "pick_place_any":
            return f"Place the {a} on the green disk."
        if self.task_name == "bin":
            return f"Put the {a} in the bin."
        if self.task_name == "handover":
            far = "right" if self.side == "left" else "left"
            return f"Hand the {a} over to the {far} arm and put it in the bin."
        return f"Put the {a} and the {b} in the bin, one with each arm."

    def _info(self, success, failed=False):
        info = super()._info(success, failed)
        info["task"] = getattr(self, "task_name", "pick_place")
        if hasattr(self, "_mocap"):
            info["instruction"] = self.instruction()
            info["touched"] = dict(self._touched) if hasattr(self, "_touched") else {}
        return info


# ---------------------------------------------------------------------- teacher
PHRASES = {
    "approach": "Move the {arm} gripper above the {obj}.",
    "lower": "Lower the {arm} gripper around the {obj}.",
    "grasp": "Close the {arm} gripper on the {obj}.",
    "lift": "Lift the {obj} with the {arm} arm.",
    "transfer": "Carry the {obj} to the {dest}.",
    "lower to goal": "Lower the {obj} onto the {dest}.",
    "release": "Open the {arm} gripper to release the {obj}.",
    "retreat": "Raise the {arm} gripper away.",
    "home": "Move the {arm} arm back to its rest position.",
    "wait": "Hold the {obj} up with the {arm} arm and wait.",
    "idle": "",
}


class ArmPickPlace:
    """One arm's phased pick-and-place toward a destination, re-reading the object's pose on retries."""

    PHASES = ("approach", "lower", "grasp", "lift", "transfer", "lower to goal", "release", "retreat", "home")

    def __init__(self, env: TaskEnv, arm, body, dest_xy, dest_z, dest_name, *, rng, noise=0.0, carry_z=0.075,
                 perturb_rate=0.0, bump_rate=0.0, miss_prob=0.0):
        self.env, self.arm, self.body = env, arm, body
        self.i = ARMS.index(arm)
        self.dest = np.r_[dest_xy, dest_z]
        self.dest_name, self.rng, self.noise, self.carry_z = dest_name, rng, noise, carry_z
        color = env.object_color if body == "cube" else env.second_color
        self.obj_name = f"{color} {env.object_shape}"
        self.phase, self.ticks, self.mistakes, self.mistake_active = 0, 0, 0, False
        self._last = env.robot[arm].position
        self.home = env.robot[arm].position
        self.wait_for = None  # another ArmPickPlace that must finish before this one carries
        # Recovery data: knock the target off course for ~0.5 s at `perturb_rate` events per second
        # during free-space phases; that stretch is labelled a mistake, the correction afterwards is clean.
        self.perturb_rate, self._perturb_left, self._perturb_offset = perturb_rate, 0, np.zeros(3)
        self.perturbations = 0
        # Recovery v2: shove the object while the arm approaches or grasps (the teacher re-reads its pose
        # and corrects), and deliberately grasp off-target on some attempts (labelled a mistake; the lift
        # check then triggers a clean retry).
        self.bump_rate, self.miss_prob = bump_rate, miss_prob
        self.bumps, self.misses = 0, 0
        self._miss_this_attempt = False
        self.obj = env.object_position(body)
        self._offset = rng.normal(0, noise, 2) if noise else np.zeros(2)
        hz = env.control_hz
        self._dwell_long, self._dwell_short = int(round(1.25 * hz)), int(round(0.6 * hz))

    @property
    def done(self):
        return self.phase >= len(self.PHASES)

    @property
    def waiting(self):
        return self.phase == 4 and self.wait_for is not None and not self.wait_for.done

    @property
    def stage(self):
        return "idle" if self.done else "wait" if self.waiting else self.PHASES[self.phase]

    @property
    def subtask(self):
        return PHRASES[self.stage].format(arm=self.arm, obj=self.obj_name, dest=self.dest_name)

    def sync_from_state(self):
        """Observe-only oracle: infer this arm's phase from the true state (object, hand, gripper),
        not from reaching the teacher's own waypoints, so it follows whatever path a policy takes."""
        if self.done:
            return
        before = self.phase
        self._sync_raw()
        # Hysteresis: within carry-and-place (transfer..retreat) never step back, which would make the
        # prompt flap between "lower" and "release" as the object's height jitters at a threshold.
        if 4 <= before <= 7 and 4 <= self.phase < before:
            self.phase = before

    def _sync_raw(self):
        """Labels follow the teacher's convention: a phase lasts until its target is reached."""
        env = self.env
        hand = env.robot[self.arm].position
        obj = env.object_position(self.body)
        opening = env.robot[self.arm].gripper_opening
        closed = opening < 0.7
        dest_xy = np.linalg.norm(obj[:2] - self.dest[:2]) < (BIN_INNER if self.dest_name == "bin" else 0.03)
        hand_at_dest = np.linalg.norm(hand[:2] - self.dest[:2]) < 0.025
        if not closed or np.linalg.norm(hand - obj) > 0.04:
            self._was_lifted = False
        elif obj[2] > 0.045:
            self._was_lifted = True
        # Held: closed around the object and either off the table or already carried (a placed cube is
        # back at table height while still in the gripper).
        held = closed and np.linalg.norm(hand - obj) < 0.04 and (obj[2] > 0.02 or getattr(self, "_was_lifted", False))
        if dest_xy and not held and obj[2] < self.dest[2] + 0.01 and not closed:
            # Object placed: release until fully open, retreat until clear, then this arm is done.
            if opening < 0.95 and np.linalg.norm(hand - obj) < 0.035:
                self.phase = 6
            elif np.linalg.norm(hand - obj) > 0.06:
                self.phase = len(self.PHASES)
            else:
                self.phase = 7
        elif held:
            if not hand_at_dest:
                self.phase = 3 if obj[2] < self.carry_z - 0.015 and self.phase <= 3 else 4
            elif obj[2] > self.dest[2] + 0.008:
                self.phase = 5
            else:
                self.phase = 6
        else:
            low = hand[2] < 0.027
            near = np.linalg.norm(hand[:2] - obj[:2]) < 0.02
            if near and low:
                self.phase = 2                      # closing, or holding before the lift
            elif np.linalg.norm(hand[:2] - obj[:2]) < 0.012 or (near and hand[2] < 0.07):
                self.phase = 1                      # descending around the object
            else:
                self.phase = 0

    def target(self):
        """(xyz, opening) for this step; advances phases."""
        if self.done:
            return None
        c = self.obj[:2] + self._offset
        d = self.dest
        targets = [np.r_[c, 0.075], np.r_[c, 0.022], np.r_[c, 0.022], np.r_[c, self.carry_z],
                   np.r_[d[:2], self.carry_z], d, d, np.r_[d[:2], self.carry_z], self.home]
        opening = 1.0 if self.phase in (0, 1, 6, 7, 8) else 0.0
        if self.waiting:
            return np.r_[c, self.carry_z], 0.0
        hz = self.env.control_hz
        if self.phase in (0, 1, 2):
            if self.bump_rate and self.rng.random() < self.bump_rate / hz:
                angle = self.rng.uniform(0, 2 * np.pi)
                speed = self.rng.uniform(0.12, 0.35)
                qvel = self.env.data.joint(f"{self.body}_joint").qvel
                qvel[:3] = [speed * np.cos(angle), speed * np.sin(angle), 0.0]
                self.bumps += 1
            now = self.env.object_position(self.body)
            moved = np.linalg.norm(now[:2] - self.obj[:2])
            if moved > 0.008 and now[2] < 0.03 and not self._miss_this_attempt:
                # The object moved: re-read it; back up and re-approach if the hand was already low.
                self.obj = now
                if moved > 0.015 and self.phase in (1, 2):
                    self.phase, self.ticks = 0, 0
                c = self.obj[:2] + self._offset
                targets[:4] = [np.r_[c, 0.075], np.r_[c, 0.022], np.r_[c, 0.022], np.r_[c, self.carry_z]]
                opening = 1.0 if self.phase in (0, 1, 6, 7, 8) else 0.0
            if self.phase == 0 and self.ticks == 0 and self.miss_prob and self.rng.random() < self.miss_prob:
                angle, dist = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.012, 0.025)
                self._offset = np.array([dist * np.cos(angle), dist * np.sin(angle)])
                self._miss_this_attempt = True
                self.mistake_active = True
                self.misses += 1
                c = self.obj[:2] + self._offset
                targets[:4] = [np.r_[c, 0.075], np.r_[c, 0.022], np.r_[c, 0.022], np.r_[c, self.carry_z]]
        tgt = targets[self.phase]
        if self._perturb_left > 0:
            self._perturb_left -= 1
            if self._perturb_left == 0:
                self.mistake_active = False
            return tgt + self._perturb_offset, opening
        if self.perturb_rate and self.phase in (0, 1, 4, 5) and self.rng.random() < self.perturb_rate / hz:
            angle, dist = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.015, 0.04)
            self._perturb_offset = np.r_[dist * np.cos(angle), dist * np.sin(angle), self.rng.uniform(0.0, 0.02)]
            self._perturb_left = int(round(self.rng.uniform(0.3, 0.7) * hz))
            self.mistake_active = True
            self.perturbations += 1
            return tgt + self._perturb_offset, opening
        self.ticks += 1
        hand = self.env.robot[self.arm].position
        err = np.linalg.norm(hand - tgt)
        settled = np.linalg.norm(hand - self._last) < 2e-4  # under 1 cm/s at 50 Hz
        self._last = hand
        # Exact arrival, or settled close by when the IK cannot reach the target exactly.
        reached = err < 0.012 or (settled and err < 0.03 and self.ticks > 2 * self._dwell_short)
        dwell = self._dwell_long if self.phase in (2, 6) else self._dwell_short
        if reached and self.ticks >= dwell:
            if self.phase == 3 and self.env.object_position(self.body)[2] < 0.045:
                self.mistakes += 1
                self.mistake_active = True
                self.obj = self.env.object_position(self.body)
                self._offset = self.rng.normal(0, self.noise, 2) if self.noise else np.zeros(2)
                self._miss_this_attempt = False
                self.phase, self.ticks = 0, 0
            else:
                if self.phase == 3:
                    self.mistake_active = False
                    self._miss_this_attempt = False
                self.phase += 1
                self.ticks = 0
        return tgt, opening


class TaskTeacher:
    """Scripted demonstrator for every rung; works in ee_delta control."""

    def __init__(self, env: TaskEnv, *, rng=None, speed=1.0, noise=0.0, observe_only=False, perturb_rate=0.0,
                 bump_rate=0.0, miss_prob=0.0):
        """observe_only: track the scripted phase from true state while something else (a policy)
        moves the arms; `observe()` advances it and `subtask` gives the oracle label."""
        if env.control_mode != "ee_delta" and not observe_only:
            raise ValueError("TaskTeacher needs ee_delta control")
        self.observe_only = observe_only
        self.env, self.speed = env, speed
        rng = rng if rng is not None else np.random.default_rng(0)
        t, near = env.task_name, env.side
        far = "right" if near == "left" else "left"
        if t == "pick_place_any":
            g = env.goal_position
            self.stages = [[ArmPickPlace(env, near, "cube", g[:2], 0.022, "green disk", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)]]
        elif t == "bin":
            b = env.bin_position
            self.stages = [[ArmPickPlace(env, near, "cube", b[:2], 0.06, "bin", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)]]
        elif t == "handover":
            b = env.bin_position
            self.stages = [[ArmPickPlace(env, near, "cube", RELAY, 0.022, "grey pad", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)],
                           ["handover", lambda: ArmPickPlace(env, far, "cube", b[:2], 0.06, "bin", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)]]
        else:
            b = env.bin_position
            first = ArmPickPlace(env, "left", "cube", b[:2] + [-0.022, 0], 0.055, "bin", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)
            second = ArmPickPlace(env, "right", "cube2", b[:2] + [0.022, 0], 0.055, "bin", rng=rng, noise=noise, perturb_rate=perturb_rate, bump_rate=bump_rate, miss_prob=miss_prob)
            second.wait_for = first  # both pick at once; the right arm carries once the left has left the bin
            self.stages = [[first, second]]
        self.k = 0

    @property
    def workers(self):
        stage = self.stages[min(self.k, len(self.stages) - 1)]
        return [w for w in stage if isinstance(w, ArmPickPlace)]

    @property
    def subtask(self):
        texts = [w.subtask for w in self.workers if w.subtask]
        return " ".join(texts) if texts else "Hold still."

    @property
    def stage(self):
        return "+".join(f"{w.arm}:{w.stage}" for w in self.workers)

    @property
    def mistakes(self):
        return sum(w.mistakes for s in self.stages for w in s if isinstance(w, ArmPickPlace))

    @property
    def mistake_active(self):
        return any(w.mistake_active for w in self.workers)

    def observe(self):
        """Advance phases from the current state without producing an action (observe_only mode)."""
        for w in self.workers:
            w.sync_from_state()
        self._advance()
        for w in self.workers:
            w.sync_from_state()

    def _advance(self):
        if self.k < len(self.stages) and all(w.done for w in self.workers):
            nxt = self.k + 1
            if nxt < len(self.stages) and self.stages[nxt][0] == "handover":
                self.stages[nxt] = [self.stages[nxt][1]()]
            self.k = nxt

    def action(self):
        env = self.env
        self._advance()  # the far arm reads the object's pose when the handover stage starts
        action = env.hold_action().reshape(2, -1)
        for w in self.workers:
            out = w.target()
            if out is None:
                continue
            tgt, opening = out
            row = (tgt - env._ee_targets[w.i]) / env.delta_scale
            row = np.clip(row, -1, 1) * self.speed
            action[w.i, :3] = row
            action[w.i, 3] = opening
        return np.clip(action.ravel(), env.action_space.low, env.action_space.high)
