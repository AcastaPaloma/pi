"""Closed-loop evaluation in MuJoCo: the paper's runtime prompting (Sec. VII) on the sim rig.

Metadata: speed = 15th percentile of the task's training episode lengths, quality 5,
mistake false, control mode joint. 5 denoising steps, execute 25 of 50 actions
(configurable), optional simulated inference delay handled with the RTC prefix,
optional classifier-free guidance on the metadata.

    python -m pi_model.eval_sim --checkpoint runs/pi07/step_0030000 --episodes 20 --execute 25
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from pi_sim import RobotEnv
from pi_sim.scene import COLORS
from .policy import HistoryBuffer, Pi07Policy
from .prompt import PromptContext


def run_episode(env: RobotEnv, policy: Pi07Policy, *, seed, execute, delay, cfg_beta, subtask, image_size,
                placement, visual_randomization, video_frames=None, generator=None):
    obs, info = env.reset(seed=seed, options={"placement_range": placement, "visual_randomization": visual_randomization})
    history = HistoryBuffer(policy.cameras, hz=env.control_hz)
    task = info["instruction"]
    context = PromptContext(task=task, subtask=subtask, speed_steps=policy.speed_for(task), quality=5,
                            mistake=False, control_mode="joint", speed_bin=500)

    def observe(step):
        images = {cam: np.asarray(Image.fromarray(env.render_camera(cam, native=True)).resize(
            (image_size, image_size), Image.Resampling.BILINEAR)) for cam in policy.cameras}
        history.append(step, images, obs["state"])
        if video_frames is not None:
            video_frames.append(images[policy.cameras[0]])

    # Frames are only needed at inference times and at one-second strides before
    # them, so observing every gcd(execute, stride) steps loses nothing.
    observe_every = math.gcd(execute, policy.stride)
    t = 0
    observe(t)
    chunk = policy.act(history, context, cfg_beta=cfg_beta, generator=generator)
    cursor = 0
    inference_calls = 1
    closest = float("inf")
    lifted = 0.0
    while True:
        if cursor >= execute:
            # A new chunk is requested now; the `delay` actions executed while the
            # model computes are the committed prefix the new chunk must respect.
            observe(t)
            prefix = chunk[cursor:cursor + delay] if delay else None
            if delay and len(prefix) < delay:
                prefix = np.concatenate([prefix, np.repeat(prefix[-1:], delay - len(prefix), axis=0)])
            chunk = policy.act(history, context, prefix_actions=prefix, delay=delay, cfg_beta=cfg_beta, generator=generator)
            cursor = 0
            inference_calls += 1
        action = chunk[min(cursor, len(chunk) - 1)]
        obs, reward, terminated, truncated, info = env.step(action)
        cursor += 1
        t += 1
        closest = min(closest, float(np.linalg.norm(obs["ee_pose"][:3] - obs["cube_position"])))
        lifted = max(lifted, float(obs["cube_position"][2]))
        if t % observe_every == 0 and cursor < execute:
            observe(t)  # keep the one-second-stride history populated between inferences
        if terminated or truncated:
            break
    return {"seed": seed, "success": bool(info["is_success"]), "steps": int(info["elapsed_steps"]),
            "failure": info["failure"], "inference_calls": inference_calls, "task": task,
            "object": f"{env.object_color} {env.object_shape}",
            "closest_hand_to_cube_m": round(closest, 4), "max_cube_height_m": round(lifted, 4),
            "lifted": bool(lifted > 0.045),
            "final_cube_to_goal_m": round(float(np.linalg.norm(obs["cube_position"] - obs["goal_position"])), 4)}


def write_video(frames, path, fps=10):
    try:
        import av
    except ImportError:
        for i, f in enumerate(frames[::5]):
            Image.fromarray(f).save(Path(path).with_suffix(f".{i:04d}.png"))
        return
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--execute", type=int, default=25, help="Actions executed per chunk (paper: 15 or 25)")
    parser.add_argument("--delay", type=int, default=0, help="Simulated inference delay in steps (RTC prefix)")
    parser.add_argument("--cfg-beta", type=float, help="Classifier-free guidance weight on the metadata")
    parser.add_argument("--subtask", help="Fixed subtask instruction; omit to run without one")
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--colors", default="red,green,blue,yellow")
    parser.add_argument("--shapes", default="cube")
    parser.add_argument("--placement", choices=("small", "wide"), default="wide")
    parser.add_argument("--visual-randomization", action="store_true")
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, help="Write results.json (and videos) here")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)

    policy = Pi07Policy(args.checkpoint, device=args.device, tokenizer=args.tokenizer, denoise_steps=args.denoise_steps)
    image_size = policy.cfg.vision.image_size
    colors = [c for c in args.colors.split(",") if c in COLORS]
    shapes = [s for s in args.shapes.split(",") if s in ("cube", "cylinder")]
    results, envs = [], {}
    generator = torch.Generator(device=policy.device).manual_seed(args.seed)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for i in range(args.episodes):
        color, shape = colors[i % len(colors)], shapes[(i // len(colors)) % len(shapes)]
        if (color, shape) not in envs:
            envs[(color, shape)] = RobotEnv(task="pick_place", control_mode="joint", object_color=color, object_shape=shape,
                                            control_hz=policy.hz, episode_seconds=args.episode_seconds)
        frames = [] if args.video else None
        result = run_episode(envs[(color, shape)], policy, seed=args.seed + i, execute=args.execute, delay=args.delay,
                             cfg_beta=args.cfg_beta, subtask=args.subtask, image_size=image_size,
                             placement=args.placement, visual_randomization=args.visual_randomization,
                             video_frames=frames, generator=generator)
        results.append(result)
        print(f"episode {i}: {result['object']:<14} success={result['success']} steps={result['steps']} "
              f"failure={result['failure']} calls={result['inference_calls']} closest={result['closest_hand_to_cube_m']:.3f}m "
              f"lifted={result['lifted']} final_to_goal={result['final_cube_to_goal_m']:.3f}m", flush=True)
        if args.video and args.out:
            write_video(frames, args.out / f"episode_{i:03d}.mp4")
    for env in envs.values():
        env.close()
    rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    lift_rate = float(np.mean([r["lifted"] for r in results])) if results else 0.0
    summary = {"checkpoint": str(args.checkpoint), "episodes": len(results), "success_rate": rate, "lift_rate": lift_rate,
               "execute": args.execute, "delay": args.delay, "cfg_beta": args.cfg_beta, "denoise_steps": args.denoise_steps,
               "elapsed_s": round(time.monotonic() - started, 1), "results": results}
    print(f"success rate {rate:.2%}, lift rate {lift_rate:.2%}, over {len(results)} episodes")
    if args.out:
        (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
