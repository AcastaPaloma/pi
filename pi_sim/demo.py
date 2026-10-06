"""Transparent scripted demonstrations for checking the sim; no machine learning."""

import numpy as np


class ScriptedDemo:
    def __init__(self, env):
        if env.control_mode == "joint":
            raise ValueError("The scripted demo needs Cartesian control")
        self.env = env
        self.phase = 0
        self.ticks = 0
        self.cube = env.data.body("cube").xpos.copy()
        self.goal = env.goal_position.copy()

    @property
    def stage(self):
        if self.env.task == "reach":
            return "reach"
        return ("approach", "lower", "grasp", "lift", "transfer", "lower to goal", "release", "retreat")[self.phase]

    def action(self):
        if self.env.task == "reach":
            return self.env.action_toward(self.goal)
        targets = [np.r_[self.cube[:2], 0.18], np.r_[self.cube[:2], 0.028],
                   np.r_[self.cube[:2], 0.028], np.r_[self.cube[:2], 0.18],
                   np.r_[self.goal[:2], 0.18], np.r_[self.goal[:2], 0.032],
                   np.r_[self.goal[:2], 0.032], np.r_[self.goal[:2], 0.20]]
        opening = 1.0 if self.phase in (0, 1, 6, 7) else 0.0
        target = targets[self.phase]
        self.ticks += 1
        reached = np.linalg.norm(self.env.robot["left"].position - target) < 0.012
        dwell = 25 if self.phase in (2, 6) else 12
        action = self.env.action_toward(target, opening=opening)
        if reached and self.ticks >= dwell and self.phase < 7:
            self.phase += 1
            self.ticks = 0
        return action
