"""Direct observation snapshots: python -m pi_observe sim|robot|devices."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from PIL import Image
import numpy as np

from . import Observer, SimSource


def main():
    parser = argparse.ArgumentParser(description="Capture images and proprioception directly into NumPy + PNG")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sim", "robot"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--output", type=Path, help="New capture directory; auto-named under outputs/observations by default")
        cmd.add_argument("--frames", type=int, default=1)
        cmd.add_argument("--fps", type=float, default=20, help="Maximum requested capture rate; PNG saving can be slower")
        cmd.add_argument("--size", type=int, default=448)
        cmd.add_argument("--history", type=int, default=1)
        if name == "robot":
            cmd.add_argument("--config", type=Path, required=True)
        else:
            cmd.add_argument("--task", choices=("reach", "pick_place"), default="reach")
    commands.add_parser("devices", help="List serial devices only; never opens cameras or changes motor settings")
    args = parser.parse_args()
    if args.command == "devices":
        from serial.tools import list_ports
        print(json.dumps([{"port": p.device, "description": p.description, "serial_number": p.serial_number}
                          for p in list_ports.comports()], indent=2))
        return
    if args.frames < 1 or not np.isfinite(args.fps) or not 0 < args.fps <= 120:
        parser.error("frames must be positive; fps must be in (0,120]")
    output = args.output or Path("outputs/observations") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    with ExitStack() as stack:
        env = None
        if args.command == "sim":
            from pi_sim import RobotEnv
            env = stack.enter_context(RobotEnv(task=args.task, image_size=args.size, control_mode="joint",
                                               max_episode_steps=max(args.frames + 1, 400)))
            _, info = env.reset(seed=0)
            observer = Observer(SimSource(env), task=info["instruction"], size=args.size, history=args.history)
        else:
            from .hardware import SO101Source
            config = json.loads(args.config.read_text())
            def path(value):
                p = Path(value).expanduser()
                return p if p.is_absolute() else args.config.parent / p
            for arm in config["arms"].values():
                arm["calibration"] = str(path(arm["calibration"]))
            goals = {cam: np.asarray(Image.open(path(p)).convert("RGB")) for cam, p in config.get("goal_images", {}).items()}
            source = SO101Source(config)
            observer = Observer(source, task=config["task"], subtask=config.get("subtask"), goal_images=goals,
                                size=args.size, history=args.history,
                                max_age_ms=config.get("max_age_ms", 200), max_skew_ms=config.get("max_skew_ms", 100))
            # Register cleanup before connect, including interrupted startup.
            stack.callback(observer.close)
            source.connect()
        if env is not None:
            stack.callback(observer.close)
        started = time.monotonic()
        for index in range(args.frames):
            tick = time.monotonic()
            if env is not None and index:
                env.step(env.hold_action())
            observation = observer.read()
            saved = observation.save(output / f"{index:06d}")
            print(f"{saved}: {len(observation.images)} cameras, state={observation.proprioception.shape}, "
                  f"timing spread={observation.metadata['host_timing_spread_ms']:.1f}ms", flush=True)
            if index + 1 < args.frames:
                time.sleep(max(0, 1 / args.fps - (time.monotonic() - tick)))
        print(f"Saved {args.frames} captures in {time.monotonic() - started:.2f}s. No action labels were generated.")


if __name__ == "__main__":
    main()
