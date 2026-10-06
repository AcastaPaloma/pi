"""MuJoCo robot playground. No learned policy or training framework required."""

from gymnasium.envs.registration import register

from .env import RobotEnv

register(id="PiSim-Reach-v0", entry_point="pi_sim:RobotEnv", kwargs={"task": "reach"})
register(id="PiSim-PickPlace-v0", entry_point="pi_sim:RobotEnv", kwargs={"task": "pick_place"})

__all__ = ["RobotEnv"]
