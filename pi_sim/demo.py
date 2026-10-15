"""Transparent scripted demonstrations for checking the sim; no machine learning.

The pick/place script is phased, and each phase doubles as the π0.7 subtask
label for the recorded segment. A `style` adds target noise and a speed factor
so that recorded data spans several episode lengths and qualities, and a grasp
that fails to lift the object triggers a retry, which the recorder labels as a
mistake segment.
"""

import numpy as np

SUBTASKS = {
    "reach": "Move the left gripper to the green sphere.",
    "approach": "Move the left gripper above the object.",
    "lower": "Lower the gripper around the object.",
    "grasp": "Close the gripper on the object.",
    "lift": "Lift the object off the table.",
    "transfer": "Carry the object above the green disk.",
    "lower to goal": "Lower the object onto the disk.",
    "release": "Open the gripper to release the object.",
    "retreat": "Raise the gripper away from the object.",
}


class ScriptedDemo:
    def __init__(self, env, *, rng=None, speed=1.0, noise=0.0):
        if env.control_mode == "joint":
            raise ValueError("The scripted demo needs Cartesian control")
        if not 0.25 <= speed <= 1.0 or not 0 <= noise <= 0.02:
            raise ValueError("speed must be in [0.25, 1] and noise in [0, 0.02] meters")
        self.env = env
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.speed, self.noise = speed, noise
        self.phase = 0
        self.ticks = 0
        self.mistakes = 0
        self.mistake_active = False
        self.cube = env.data.body("cube").xpos.copy()
        self.goal = env.goal_position.copy()
        self._offsets = self.rng.normal(0, noise, (8, 2)) if noise else np.zeros((8, 2))
        hz = env.control_hz
        self._dwell_long, self._dwell_short = int(round(1.25 * hz)), int(round(0.6 * hz))

    PHASES = ("approach", "lower", "grasp", "lift", "transfer", "lower to goal", "release", "retreat")

    @property
    def stage(self):
        if self.env.task == "reach":
            return "reach"
        return self.PHASES[self.phase]

    @property
    def subtask(self):
        return SUBTASKS[self.stage]

    def _targets(self):
        c, g = self.cube[:2] + self._offsets[0], self.goal[:2] + self._offsets[4]
        return [np.r_[c, 0.075], np.r_[c, 0.022], np.r_[c, 0.022], np.r_[c, 0.075],
                np.r_[g, 0.075], np.r_[g, 0.022], np.r_[g, 0.022], np.r_[g, 0.075]]

    def action(self):
        if self.env.task == "reach":
            action = self.env.action_toward(self.goal)
            return self._slow(action)
        targets = self._targets()
        opening = 1.0 if self.phase in (0, 1, 6, 7) else 0.0
        target = targets[self.phase]
        self.ticks += 1
        reached = np.linalg.norm(self.env.robot["left"].position - target) < 0.012
        dwell = self._dwell_long if self.phase in (2, 6) else self._dwell_short
        action = self.env.action_toward(target, opening=opening)
        if reached and self.ticks >= dwell and self.phase < 7:
            if self.phase == 3 and self.env.data.body("cube").xpos[2] < 0.045:
                # The grasp did not lift the object: retry from the approach.
                self.mistakes += 1
                self.mistake_active = True
                self.cube = self.env.data.body("cube").xpos.copy()
                self._offsets[0] = self.rng.normal(0, self.noise, 2) if self.noise else 0
                self.phase, self.ticks = 0, 0
            else:
                if self.phase == 3:
                    self.mistake_active = False
                self.phase += 1
                self.ticks = 0
        return self._slow(action)

    def _slow(self, action):
        if self.speed >= 1.0 or self.env.control_mode != "ee_delta":
            return action
        action = action.copy()
        rows = action.reshape(2, -1)
        rows[:, :3] *= self.speed
        return action
