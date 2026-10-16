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
from pi_sim.demo import SUBTASKS
from pi_sim.scene import COLORS
from .policy import HistoryBuffer, Pi07Policy
from .prompt import PromptContext


def oracle_subtask(env: RobotEnv, obs, released_after_place: bool) -> str:
    """Current scripted-demo phase from simulator state: stands in for the high-level policy.

    Uses privileged state (cube pose), so it is an evaluation aid, not a deployable component.
    """
    hand = obs["ee_pose"][:3]
    cube = obs["cube_position"]
    goal = obs["goal_position"]
    opening = env.robot["left"].gripper_opening
    lifted = cube[2] > 0.045
    at_goal_xy = np.linalg.norm(cube[:2] - goal[:2]) < 0.03
    above_cube = np.linalg.norm(hand[:2] - cube[:2]) < 0.02
    near_cube = np.linalg.norm(hand - cube) < 0.03
    if released_after_place:
        return SUBTASKS["retreat"]
    if lifted:
        if np.linalg.norm(hand[:2] - goal[:2]) > 0.03:
            return SUBTASKS["transfer"]
        return SUBTASKS["lower to goal"] if cube[2] > 0.03 else SUBTASKS["release"]
    if at_goal_xy and opening < 0.7 and near_cube:
        return SUBTASKS["release"]
    if near_cube and opening < 0.7:
        return SUBTASKS["lift"]
    if near_cube:
        return SUBTASKS["grasp"]
    if above_cube:
        return SUBTASKS["lower"]
    return SUBTASKS["approach"]


def run_episode(env: RobotEnv, policy: Pi07Policy, *, seed, execute, delay, cfg_beta, subtask, image_size,
                placement, visual_randomization, video_frames=None, generator=None, full_video_every=0):
    """full_video_every > 0: record all cameras every that many control steps with captions (real-time video)."""
    obs, info = env.reset(seed=seed, options={"placement_range": placement, "visual_randomization": visual_randomization})
    history = HistoryBuffer(policy.cameras, hz=env.control_hz)
    task = info["instruction"]
    released = False
    ladder = hasattr(env, "task_name")
    shadow = None
    if ladder and subtask == "oracle":
        from pi_sim.tasks import TaskTeacher
        shadow = TaskTeacher(env, rng=np.random.default_rng(seed), observe_only=True)

    def current_subtask():
        nonlocal released
        if shadow is not None:
            return shadow.subtask
        if subtask == "oracle":
            if not released and obs["cube_position"][2] < 0.03 and \
                    np.linalg.norm(obs["cube_position"][:2] - obs["goal_position"][:2]) < 0.03 and \
                    env.robot["left"].gripper_opening > 0.7:
                released = True
            return oracle_subtask(env, obs, released)
        return subtask

    def make_context():
        return PromptContext(task=task, subtask=current_subtask(), speed_steps=policy.speed_for(task), quality=5,
                             mistake=False, control_mode="joint", speed_bin=500)

    context = make_context()

    timing = {"policy_s": 0.0, "observe_s": 0.0}

    def observe(step):
        t0 = time.monotonic()
        images = {cam: policy.prepare_image(env.render_camera(cam, native=True)) for cam in policy.cameras}
        history.append(step, images, obs["state"])
        if video_frames is not None and not full_video_every:
            video_frames.append(images[policy.cameras[0]])
        timing["observe_s"] += time.monotonic() - t0

    def act(**kwargs):
        t0 = time.monotonic()
        chunk = policy.act(history, context, cfg_beta=cfg_beta, generator=generator, **kwargs)
        timing["policy_s"] += time.monotonic() - t0
        return chunk

    # Frames are only needed at inference times and at one-second strides before
    # them, so observing every gcd(execute, stride) steps loses nothing.
    observe_every = math.gcd(execute, policy.stride)
    t = 0
    observe(t)
    chunk = act()
    cursor = 0
    inference_calls = 1
    closest = float("inf")
    lifted = 0.0
    two_objects = ladder and env.task_name == "two_bins"
    lifted2 = 0.0
    while True:
        if cursor >= execute:
            # A new chunk is requested now; the `delay` actions executed while the
            # model computes are the committed prefix the new chunk must respect.
            observe(t)
            context = make_context()
            prefix = chunk[cursor:cursor + delay] if delay else None
            if delay and len(prefix) < delay:
                prefix = np.concatenate([prefix, np.repeat(prefix[-1:], delay - len(prefix), axis=0)])
            chunk = act(prefix_actions=prefix, delay=delay)
            cursor = 0
            inference_calls += 1
        action = chunk[min(cursor, len(chunk) - 1)]
        obs, reward, terminated, truncated, info = env.step(action)
        if shadow is not None:
            shadow.observe()
        if video_frames is not None and full_video_every and t % full_video_every == 0:
            video_frames.append(_captioned_frame(env, policy.cameras, task, context.subtask, t, info))
        cursor += 1
        t += 1
        closest = min(closest, float(min(np.linalg.norm(obs["ee_pose"][k:k + 3] - obs["cube_position"]) for k in (0, 7))))
        lifted = max(lifted, float(obs["cube_position"][2]))
        if two_objects:  # the second arm's object; failures of two_bins were invisible without it (H10)
            lifted2 = max(lifted2, float(env.object_position("cube2")[2]))
        if t % observe_every == 0 and cursor < execute:
            observe(t)  # keep the one-second-stride history populated between inferences
        if terminated or truncated:
            break
    extra = {"task_id": env.task_name, "side": env.side, "touched": info.get("touched"),
             "final_cube_to_bin_m": round(float(np.linalg.norm(obs["cube_position"][:2] - obs["bin_position"][:2])), 4)}\
        if ladder else {}
    if two_objects:
        extra.update({"cube_in_bin": env._in_bin("cube"), "cube2_in_bin": env._in_bin("cube2"),
                      "cube2_max_height_m": round(lifted2, 4), "cube2_lifted": bool(lifted2 > 0.045)})
    return {**extra, "seed": seed, "success": bool(info["is_success"]), "steps": int(info["elapsed_steps"]),
            "failure": info["failure"], "inference_calls": inference_calls, "task": task, "subtask_mode": subtask,
            "object": f"{env.object_color} {env.object_shape}",
            "closest_hand_to_cube_m": round(closest, 4), "max_cube_height_m": round(lifted, 4),
            "lifted": bool(lifted > 0.045),
            "final_cube_to_goal_m": round(float(np.linalg.norm(obs["cube_position"] - obs["goal_position"])), 4),
            "policy_s": round(timing["policy_s"], 1), "observe_s": round(timing["observe_s"], 1)}


def _captioned_frame(env, cameras, task, subtask, t, info):
    """All cameras side by side (320x240 each) with the instruction, prompted subtask and live state."""
    from PIL import ImageDraw
    tiles = [np.asarray(Image.fromarray(env.render_camera(c, native=True)).resize((320, 240))) for c in cameras]
    row = np.concatenate(tiles, 1)
    canvas = Image.new("RGB", (row.shape[1], 240 + 54), (255, 255, 255))
    canvas.paste(Image.fromarray(row), (0, 0))
    d = ImageDraw.Draw(canvas)
    cube = env.data.body("cube").xpos
    grips = " ".join(f"{side} grip {env.robot[side].gripper_opening:.2f}" for side in ("left", "right"))
    d.text((6, 244), f"{task}", fill=(20, 20, 20))
    d.text((6, 260), f"prompted subtask: {subtask or '-'}", fill=(10, 70, 160))
    touched = info.get("touched") or {}
    d.text((6, 276), f"t={t / env.control_hz:5.2f}s  cube z={cube[2] * 100:4.1f} cm  {grips}  "
                     f"touched L={touched.get('left', '-')} R={touched.get('right', '-')}"
                     + ("   SUCCESS" if info.get("is_success") else ""), fill=(150, 30, 30) if info.get("is_success") else (60, 60, 60))
    for i, c in enumerate(cameras):
        d.text((i * 320 + 6, 6), c, fill=(255, 255, 255))
    return np.asarray(canvas)


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
    parser.add_argument("--subtask", help="Subtask instruction: a fixed string, 'oracle' (scripted phase from sim state), or omit for none")
    parser.add_argument("--denoise-steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32",
                        help="Inference weights; bf16 lets the paper-size model fit a 24 GB GPU")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--colors", default="red,green,blue,yellow")
    parser.add_argument("--shapes", default="cube")
    parser.add_argument("--placement", choices=("small", "wide"), default="wide")
    parser.add_argument("--task", default="pick_place",
                        help="pick_place (original left-arm task) or a ladder rung: pick_place_any, bin, handover, two_bins")
    parser.add_argument("--visual-randomization", action="store_true")
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, help="Write results.json (and videos) here")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--full-video", type=int, default=0, metavar="N",
                        help="With --video: record all cameras with captions every N control steps (2 = real time at 25 fps)")
    args = parser.parse_args(argv)

    policy = Pi07Policy(args.checkpoint, device=args.device, tokenizer=args.tokenizer, denoise_steps=args.denoise_steps,
                        dtype=args.dtype)
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
            if args.task == "pick_place":
                envs[(color, shape)] = RobotEnv(task="pick_place", control_mode="joint", object_color=color, object_shape=shape,
                                                control_hz=policy.hz, episode_seconds=args.episode_seconds)
            else:
                from pi_sim.tasks import TaskEnv
                second = next(c for c in ("blue", "red", "yellow", "green") if c != color)
                envs[(color, shape)] = TaskEnv(args.task, control_mode="joint", object_color=color, second_color=second,
                                               object_shape=shape, control_hz=policy.hz, episode_seconds=args.episode_seconds)
        frames = [] if args.video else None
        result = run_episode(envs[(color, shape)], policy, seed=args.seed + i, execute=args.execute, delay=args.delay,
                             cfg_beta=args.cfg_beta, subtask=args.subtask, image_size=image_size,
                             placement=args.placement, visual_randomization=args.visual_randomization,
                             video_frames=frames, generator=generator, full_video_every=args.full_video)
        results.append(result)
        print(f"episode {i}: {result['object']:<14} success={result['success']} steps={result['steps']} "
              f"failure={result['failure']} calls={result['inference_calls']} closest={result['closest_hand_to_cube_m']:.3f}m "
              f"lifted={result['lifted']} final_to_goal={result['final_cube_to_goal_m']:.3f}m "
              f"policy={result['policy_s']:.0f}s observe={result['observe_s']:.0f}s", flush=True)
        if args.video and args.out:
            tag = "success" if result["success"] else "fail"
            write_video(frames, args.out / f"episode_{i:03d}_{tag}.mp4",
                        fps=int(round(policy.hz / args.full_video)) if args.full_video else 10)
    for env in envs.values():
        env.close()
    rate = float(np.mean([r["success"] for r in results])) if results else 0.0
    lift_rate = float(np.mean([r["lifted"] for r in results])) if results else 0.0
    summary = {"checkpoint": str(args.checkpoint), "episodes": len(results), "success_rate": rate, "lift_rate": lift_rate,
               "subtask": args.subtask,
               "execute": args.execute, "delay": args.delay, "cfg_beta": args.cfg_beta, "denoise_steps": args.denoise_steps, "dtype": args.dtype,
               "elapsed_s": round(time.monotonic() - started, 1), "results": results}
    print(f"success rate {rate:.2%}, lift rate {lift_rate:.2%}, over {len(results)} episodes")
    if args.out:
        (args.out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    main()
