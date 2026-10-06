"""Run with `python -m pi_sim`; the macOS viewer relaunches with mjpython."""

import argparse
from collections import deque
import os
from pathlib import Path
import sys
import time

from . import RobotEnv
from .demo import ScriptedDemo


def main():
    parser = argparse.ArgumentParser(description="Two UR5e arms, one easy task. No training required.")
    parser.add_argument("--task", choices=("reach", "pick_place"), default="reach")
    parser.add_argument("--demo", action="store_true", help="Run a scripted controller")
    parser.add_argument("--headless", action="store_true", help="Run the demo without a window")
    parser.add_argument("--episodes", type=int, help="Episode count; 0 repeats until the window closes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000, help="Maximum steps per episode (100 seconds)")
    parser.add_argument("--screenshot", type=Path, help="Save the final front-camera image")
    args = parser.parse_args()
    episodes = args.episodes if args.episodes is not None else (1 if args.headless else 0)
    if episodes < 0 or args.steps < 1 or (args.headless and episodes == 0):
        parser.error("Use positive --steps and --episodes (0 episodes is allowed only with a window)")
    if not args.headless and sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        launcher = Path(sys.executable).with_name("mjpython")
        os.execv(str(launcher), [str(launcher), "-m", "pi_sim", *sys.argv[1:]])
    with RobotEnv(task=args.task, max_episode_steps=args.steps) as env:
        obs, _ = env.reset(seed=args.seed)
        demo = ScriptedDemo(env)
        automatic = args.demo or args.headless
        completed = 0
        if args.headless:
            for episode in range(episodes):
                if episode:
                    obs, _ = env.reset(seed=args.seed + episode)
                    demo = ScriptedDemo(env)
                for _ in range(args.steps):
                    obs, reward, terminated, truncated, info = env.step(demo.action())
                    if terminated or truncated:
                        break
                print(f"Episode {episode + 1}: success={info['is_success']}, steps={info['elapsed_steps']}, stage={demo.stage}")
                completed += int(info["is_success"])
            print(f"Successful episodes: {completed}/{episodes}")
        else:
            import mujoco.viewer

            events = deque()
            active_arm, paused, done = 0, False, False
            finished_at = None
            print("Click the simulation window. W/S: +/-Y, A/D: -/+X, Q/E: up/down")
            print("G: open/close, 1/2: select arm, R: reset, P: pause, B: demo on/off")
            print("Each key press moves the target by 2 cm. Close the window to quit.")
            with mujoco.viewer.launch_passive(env.model, env.data, key_callback=events.append,
                                             show_left_ui=False, show_right_ui=False) as viewer:
                viewer.cam.lookat[:] = [0, 0, 0.10]
                viewer.cam.distance = 2.5
                viewer.cam.azimuth = 125
                viewer.cam.elevation = -28
                visual_flags = viewer.opt.flags.copy()
                render_flags = viewer.user_scn.flags.copy()
                geom_groups = viewer.opt.geomgroup.copy()
                while viewer.is_running():
                    start = time.monotonic()
                    action = env.hold_action().reshape(2, 4)
                    while events:
                        key = events.popleft()
                        if key in (49, 50):
                            active_arm = key - 49
                            print("Selected", ("left", "right")[active_arm], "arm")
                        elif key == ord("R"):
                            obs, _ = env.reset(seed=args.seed)
                            demo, done = ScriptedDemo(env), False
                            finished_at = None
                            action = env.hold_action().reshape(2, 4)
                        elif key == ord("P"):
                            paused = not paused
                        elif key == ord("B"):
                            automatic = not automatic
                            obs, _ = env.reset(seed=args.seed)
                            demo, done = ScriptedDemo(env), False
                            finished_at = None
                            action = env.hold_action().reshape(2, 4)
                            print("Scripted demo" if automatic else "Manual control")
                        elif key == ord("G"):
                            action[active_arm, 3] = 1 - action[active_arm, 3]
                        elif key in (ord("W"), ord("S"), ord("A"), ord("D"), ord("Q"), ord("E")):
                            axis, sign = {ord("W"): (1, 1), ord("S"): (1, -1), ord("A"): (0, -1),
                                          ord("D"): (0, 1), ord("Q"): (2, 1), ord("E"): (2, -1)}[key]
                            action[active_arm, axis] += sign
                    if not paused and not done:
                        obs, reward, terminated, truncated, info = env.step(demo.action() if automatic else action.ravel())
                        done = terminated or truncated
                        if done:
                            finished_at = time.monotonic()
                            completed += 1
                            print(f"Episode {completed}: success={info['is_success']}, steps={info['elapsed_steps']}. R resets.")
                    # MuJoCo also maps our movement keys to visual toggles. Keep
                    # the playground view stable (e.g. '2' must not hide meshes).
                    with viewer.lock():
                        viewer.opt.flags[:] = visual_flags
                        viewer.opt.geomgroup[:] = geom_groups
                        viewer.user_scn.flags[:] = render_flags
                    viewer.sync()
                    if done and automatic:
                        # Leave the result visible, while remaining responsive to window close.
                        if time.monotonic() - start < 0.05:
                            time.sleep(0.05)
                        if time.monotonic() - finished_at > 1.5:
                            if episodes and completed >= episodes:
                                break
                            obs, _ = env.reset(seed=args.seed + completed)
                            demo, done = ScriptedDemo(env), False
                            finished_at = None
                    remaining = env.control_dt - (time.monotonic() - start)
                    if remaining > 0:
                        time.sleep(remaining)
        if args.screenshot:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(env.render()).save(args.screenshot)
            print(f"Saved {args.screenshot}")
        if args.headless and completed != episodes:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
