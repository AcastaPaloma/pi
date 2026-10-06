"""The smallest useful control loop: python examples/tap_in.py."""

from pi_sim import RobotEnv

with RobotEnv(task="reach") as env:
    observation, info = env.reset(seed=0)
    print(info["instruction"])
    print("Action shape:", env.action_space.shape)
    print("Left joint angles:", env.robot["left"].joint_positions)
    print("Right gripper position:", env.robot["right"].position)
    print("MuJoCo model/data:", type(env.model).__name__, type(env.data).__name__)

    for step in range(400):
        # Replace this one line with your own controller or policy later.
        action = env.action_toward(observation["goal_position"], arm="left", opening=1.0)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Finished in {step + 1} steps. Success: {info['is_success']}")
            break
