"""Record one state/action trajectory: python examples/collect_demo.py."""

from pathlib import Path
import json

import numpy as np

from pi_sim import RobotEnv
from pi_sim.demo import ScriptedDemo

with RobotEnv(task="pick_place") as env:
    obs, info = env.reset(seed=0)
    demo = ScriptedDemo(env)
    states, actions, rewards, next_states, terminateds, truncateds = [], [], [], [], [], []
    for _ in range(env.max_episode_steps):
        action = demo.action()
        states.append(obs["state"])
        actions.append(action.copy())
        obs, reward, terminated, truncated, info = env.step(action)
        next_states.append(obs["state"])
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)
        if terminated or truncated:
            break
    Path("outputs").mkdir(exist_ok=True)
    np.savez_compressed("outputs/demo.npz", state=states, action=actions,
                        next_state=next_states, reward=rewards,
                        terminated=terminateds, truncated=truncateds,
                        metadata=json.dumps({"task": env.task, "control_mode": env.control_mode,
                                             "control_hz": 20, "seed": 0,
                                             "instruction": info["instruction"], "success": info["is_success"]}))
    print(f"Saved {len(actions)} transitions to outputs/demo.npz; success={info['is_success']}")
