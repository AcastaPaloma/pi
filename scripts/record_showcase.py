"""Film one rollout of a checkpoint for the write-up's Results tab.

    MUJOCO_GL=osmesa ~/lerobot-venv/bin/python scripts/record_showcase.py --checkpoint runs/x_control_s1/step_0005000 \
        --task handover --seed 4002 --out outputs/showcase/handover_c1_5000

The episode runs exactly as in pi_model.eval_sim (oracle subtask prompts, 25 actions per chunk, no delay), but the
scene is filmed from a fixed 3D viewpoint (the live demo's default view) instead of the robot's own cameras. Writes
<out>.mp4 (H.264, web-ready), <out>.jpg (poster frame) and <out>.json (what happened, for the page's card).
Seeds 4000+i use the colour pair of episode i of the 30-seed confirmations, so the layout matches that test.
"""
import argparse
import json
import time
from pathlib import Path

import av
import mujoco
import numpy as np
import torch
from PIL import Image

from pi_model.eval_sim import run_episode
from pi_model.policy import Pi07Policy
from pi_sim.env import RobotEnv
from pi_sim.tasks import TaskEnv

VIEW = {"azimuth": 245.0, "elevation": -24.0, "distance": 0.85, "lookat": [0.0, -0.07, 0.05]}
ORDER = ["red", "green", "blue", "yellow"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--task", required=True, choices=("pick_place", "pick_place_any", "bin", "handover", "two_bins"))
    ap.add_argument("--seed", type=int, default=4000)
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--episode-seconds", type=float, default=30.0)
    ap.add_argument("--crf", type=int, default=24)
    ap.add_argument("--hold-s", type=float, default=1.2, help="Hold the last frame this long")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=("fp32", "bf16"))
    ap.add_argument("--tokenizer", type=Path, default=Path("weights/paligemma/tokenizer.model"))
    args = ap.parse_args()

    policy = Pi07Policy(args.checkpoint, device=args.device, tokenizer=args.tokenizer, denoise_steps=5, dtype=args.dtype)
    i = args.seed - 4000 if args.seed >= 4000 else args.seed
    color = ORDER[i % len(ORDER)]
    second = next(c for c in ("blue", "red", "yellow", "green") if c != color)
    if args.task == "pick_place":  # the single-arm task, built as pi_model.eval_sim builds it
        env = RobotEnv(task="pick_place", control_mode="joint", object_color=color, object_shape="cube",
                       control_hz=policy.hz, episode_seconds=args.episode_seconds)
    else:
        env = TaskEnv(args.task, control_mode="joint", object_color=color, second_color=second, object_shape="cube",
                      control_hz=policy.hz, episode_seconds=args.episode_seconds)
    env.model.vis.global_.offwidth = max(args.width, env.model.vis.global_.offwidth)
    env.model.vis.global_.offheight = max(args.height, env.model.vis.global_.offheight)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = VIEW["azimuth"], VIEW["elevation"], VIEW["distance"]
    cam.lookat[:] = VIEW["lookat"]
    every = max(1, round(policy.hz / args.fps))
    frames = []

    def film(env, t, info):
        if t % every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())

    started = time.monotonic()
    result = run_episode(env, policy, seed=args.seed, execute=25, delay=0, cfg_beta=None, subtask="oracle",
                         image_size=policy.cfg.vision.image_size, placement="wide", visual_randomization=False,
                         generator=torch.Generator(device=policy.device).manual_seed(args.seed), frame_hook=film)
    frames += [frames[-1]] * int(args.hold_s * args.fps)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(args.out.with_suffix(".mp4")), "w", options={"movflags": "faststart"}) as container:
        stream = container.add_stream("libx264", rate=args.fps)
        stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"
        stream.options = {"crf": str(args.crf), "preset": "slow", "profile": "high"}
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    Image.fromarray(frames[min(len(frames) - 1, int(0.35 * len(frames)))]).save(args.out.with_suffix(".jpg"), quality=82)
    card = {"task": args.task, "instruction": result["task"], "seed": args.seed, "object": result["object"],
            "success": result["success"], "seconds": round(result["steps"] / policy.hz, 1),
            "checkpoint": str(args.checkpoint), "video_seconds": round(len(frames) / args.fps, 1),
            "wall_s": round(time.monotonic() - started, 1)}
    args.out.with_suffix(".json").write_text(json.dumps(card, indent=2) + "\n")
    print(json.dumps(card), flush=True)


if __name__ == "__main__":
    main()
