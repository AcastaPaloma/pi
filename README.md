# A small MuJoCo robot playground

**For your encoder: [128 observations directly in Python](docs/BATCH_ENCODER.md).** Use `RobotBatch(...).observations()` for static simulated scenes or `FoldingBatch(...).observations()` for fast cached real SO-101 samples. Both return batched RGB images, proprioception, and text; neither includes a model or policy. A 128-scene preview and a small real-data cache are ready in `outputs/`.

Two UR5e arms with Robotiq grippers, a table, a red block, and three cameras. Start by reaching a green target; then try picking up the block. Everything runs locally without a learned model.

**Already installed in your `mini-vla` conda environment.** Its existing PyTorch installation was left as-is. This project uses MuJoCo, NumPy and Gymnasium, with no PyTorch code.

**Read and capture robot observations:** run `python -m pi_observe sim` or use `Observer.read()` directly. The [observation guide](docs/OBSERVATIONS.md) covers RGB cameras, proprioception, timing, saved captures, optional goals/history, and the read-only physical SO-101 configuration.

**Building the low-level VLA encoder?** Start with the [folding dataset and API guide](docs/ENCODER_DATA.md). It covers real SO-101 garment-folding demonstrations, camera/state/text/goal/action endpoints, NumPy inputs, and an H100 budget. Run `python -m pi_data serve` in `mini-vla`, then open http://127.0.0.1:8000/docs.

## 1. Watch it work

From this project folder:

```bash
./play.sh --demo --task pick_place
```

The left arm picks up the block and places it on the green disk. The demo repeats. Close the window to stop.

## 2. Move the robot yourself

```bash
./play.sh
```

Click the simulation window. Move the **left gripper to the green sphere** to finish the easy reach task.

| Key | Action |
| --- | --- |
| W / S | Move along world +Y / -Y |
| A / D | Move along world -X / +X |
| Q / E | Move up / down |
| G | Close / open gripper |
| 1 / 2 | Select left / right arm |
| R | Reset the task |
| P | Pause / resume |
| B | Switch between demo and manual mode; resets the task |

Each key press moves the target **2 cm**; key repeat also works. A slow hand after a large command is normal: the joint targets are rate-limited. The default task only scores the left arm. An episode freezes after success or its 100-second simulation timeout; press R to reset. The Python environment defaults to 20-second episodes; the interactive launcher allows more practice time.

To practice grasping manually: `./play.sh --task pick_place --steps 2000`. Open the fingers, approach above the block, lower between its sides, close, lift, move above the disk, lower, and open.

## 3. Tap into it from Python

```bash
conda activate mini-vla
python examples/tap_in.py
```

The complete interaction pattern is:

```python
from pi_sim import RobotEnv

with RobotEnv(task="reach") as env:
    obs, info = env.reset(seed=0)
    for _ in range(400):
        # Replace this line with your own controller or policy.
        action = env.action_toward(obs["goal_position"], opening=1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Success:", info["is_success"])
            break
```

`reset()` starts an episode. `step(action)` advances **0.05 seconds (20 Hz)** and returns the usual Gymnasium tuple. Call `reset()` again after termination or truncation. `seed` makes the small block/goal randomization reproducible; `reset(options={"randomize": False})` gives fixed positions.

### Actions

`RobotEnv(control_mode=...)` supports three interfaces. Every action lists **left arm first, then right arm**. Gripper commands always use **0 = closed, 1 = open**. Finite out-of-range actions are clipped; wrong shapes and NaNs raise an error.

| Mode | Action per arm | Total shape |
| --- | --- | --- |
| `ee_delta` (default, easiest) | `[dx, dy, dz, opening]`; XYZ in [-1, 1], scaled to ±2 cm per step; orientation stays fixed | `(8,)` |
| `ee` | `[x, y, z, rx, ry, rz, opening]`; absolute world position in meters and absolute orientation as a rotation vector in radians | `(14,)` |
| `joint` | `[q1, q2, q3, q4, q5, q6, opening]`; absolute joint angles in radians | `(14,)` |

Joint order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3. Rotation vectors are axis × angle, **not Euler angles**. The table surface is world Z = 0. The end-effector point is the pinch center between the fingers. `ee_delta` accumulates movements in the target position, rather than teleporting the robot.

For example, move only the left arm upward:

```python
action = env.hold_action()  # Start from the current command, including grippers.
action[2] = 0.5            # Default mode: +1 cm in Z.
obs, reward, terminated, truncated, info = env.step(action)
```

`env.action_toward([x, y, z], arm="right", opening=1.0)` builds a Cartesian action for either arm. Call it each step to approach a point. It works in `ee_delta` and `ee` modes. Cartesian targets are bounded to a small tabletop workspace; IK is approximate and not a collision-aware planner. `info["ik_position_error"]` reports each arm's IK residual in meters, not its physical tracking error.

### Observations, cameras, and raw access

| Entry | Contents |
| --- | --- |
| `obs["state"]` | 14 floats: left joint angles + measured gripper opening, then right; opening is a normalized linkage angle |
| `obs["joint_velocity"]` | 12 joint velocities, radians/second |
| `obs["ee_pose"]` | 14 floats: left `[x,y,z,qw,qx,qy,qz]`, then right; quaternion is scalar-first |
| `obs["cube_position"]` | Block center, world meters |
| `obs["goal_position"]` | Reach target or desired block center, world meters |
| `info["instruction"]` | Task text for a future language-conditioned policy |
| `info["is_success"]` | Whether the task succeeded |

Observations are copies, so you can safely store them. Object/goal positions are privileged simulator state; omit them from your policy input if you want a vision-only task.

```python
with RobotEnv(images=True) as env:
    obs, info = env.reset(seed=0)
    front = obs["images"]["front"]           # uint8 RGB, (448, 448, 3)
    wrist = obs["images"]["left_wrist"]      # Also: "right_wrist"
    frame = env.render_camera("front")      # Works even with images=False.

    left = env.robot["left"]
    print(left.joint_positions, left.position, left.gripper_opening)
    model, data = env.model, env.data        # Native mujoco.MjModel / MjData
```

Images are off by default for fast physics loops. Set `image_size=224` for smaller frames. `env.render()` returns the front RGB image; use `play.sh` for the interactive viewer. Direct mutations of `model` or `data` bypass the environment's controls; prefer `step()` for ordinary use.

### Hook up a future training loop

The Gymnasium IDs are `PiSim-Reach-v0` and `PiSim-PickPlace-v0` (import `pi_sim` before `gymnasium.make`). Both provide dense reward, success termination, falling-block failure termination, and time-limit truncation. Reaching succeeds within 2.5 cm for five steps; placement requires the block at rest on the disk with grippers open for five steps. Rewards and tasks are local design choices, not paper benchmark scores.

Action chunks work without a training framework:

```python
import numpy as np

# Example only: hold the pose for up to 50 control steps (2.5 seconds).
chunk = np.tile(env.hold_action(), (50, 1))
transitions = env.step_chunk(chunk)  # Stops early at episode end.
# For a future 50-action policy, execute chunk[:15] or chunk[:25], then replan.
```

Record a working state/action demonstration:

```bash
python examples/collect_demo.py
```

This writes `outputs/demo.npz`: state, action, next_state, reward, terminated, truncated, and JSON metadata. The scripted controller uses exact simulator state; it is a debugging example, not a trained policy. It does not record camera frames.

## Installation elsewhere

To use an existing environment:

```bash
conda activate mini-vla
python -m pip install -e .
```

Or create a separate small environment with no PyTorch dependency:

```bash
conda env create -f environment.yml
conda activate pi-sim
python -m pi_sim --demo --task pick_place
```

`python -m pi_sim` automatically uses MuJoCo's `mjpython` launcher for the macOS window. Linux uses ordinary Python. `play.sh` selects `mini-vla`; override it with `PI_SIM_ENV=pi-sim ./play.sh`.

## Check the installation

```bash
conda activate mini-vla
python -m pi_sim --headless --task pick_place --episodes 10
python -m unittest discover -s tests -v
```

`--headless` runs physics without opening a window. Add `--screenshot outputs/scene.png` to render the final frame; rendering still requires a graphics context. On Linux without a display, use a supported EGL setup (`MUJOCO_GL=egl`); leave that unset on macOS. The tests include camera rendering. Gymnasium may print harmless warnings because continuous state observations use unbounded spaces.

## Relationship to π₀.₇

The supplied paper describes real robots. This is a practical simulation approximation of its bimanual UR5e platform, using 20 Hz PD control, numerical IK, and front/wrist cameras. The beginner tasks, scene dimensions, controllers and camera calibration are local choices. See [the paper mapping](docs/PAPER_MAPPING.md) for the exact matches and differences, and [asset sources and licenses](pi_sim/assets/README.md) for the pinned Menagerie models.
