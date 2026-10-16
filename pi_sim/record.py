"""Record scripted π0.7 training episodes from the simulator.

Each episode directory holds `meta.json`, `steps.npz` and JPEG frames per camera
at the model's 448x448 input size. Actions are the commanded joint targets in
degrees plus gripper 0–100 (left arm then right arm), i.e. joint control mode.
Subtask labels come from the script phases; a failed grasp followed by a retry
marks that segment as a mistake. Episode quality follows a fixed rule from the
outcome and the style, recorded in meta.json so it can be re-derived.

    python -m pi_sim.record --episodes 20 --out data/sim_pick_place
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .demo import ScriptedDemo
from .env import RobotEnv
from .scene import COLORS

STYLES = {
    # name: (speed factor, target noise in meters, nominal quality when successful)
    "clean": (1.0, 0.0, 5),
    "slow": (0.6, 0.0, 4),
    "sloppy": (0.85, 0.008, 3),
    "slow_sloppy": (0.5, 0.012, 2),
}


# Styles whose teacher is knocked off course (recovery data); events per second during free-space phases.
PERTURB = {"recover": 1.0}
STYLES["recover"] = (0.85, 0.004, 3)


def quality_label(success: bool, style: str, mistakes: int) -> int:
    """1–5 like the paper's overall quality score; a documented local rule."""
    if not success:
        return 1
    base = STYLES[style][2]
    return int(max(1, base - min(mistakes, 2)))


def record_episode(env: RobotEnv, out_dir: Path, *, seed: int, style: str = "clean",
                   image_size: int = 448, visual_randomization: bool = True,
                   placement_range: str = "wide", jpeg_quality: int = 90, frame_stride: int = 1,
                   teacher=ScriptedDemo) -> dict:
    """frame_stride > 1 saves camera frames only every `frame_stride` control steps.

    States and actions stay at the control rate. The history stride (1 s = 50 steps)
    must be a multiple of frame_stride so that every history frame exists.
    """
    if frame_stride < 1 or env.control_hz % frame_stride:
        raise ValueError("frame_stride must divide the control rate")
    out_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    speed, noise, _ = STYLES[style]
    obs, info = env.reset(seed=seed, options={"randomize": True, "visual_randomization": visual_randomization,
                                              "placement_range": placement_range})
    extra_kw = {"perturb_rate": PERTURB[style]} if style in PERTURB else {}
    demo = teacher(env, rng=rng, speed=speed, noise=noise, **extra_kw)
    for cam in env.camera_names:
        (out_dir / "frames" / cam).mkdir(parents=True)
    states, actions, subtasks, mistakes, phases, cube, goal, ee = [], [], [], [], [], [], [], []
    extra = {k: [] for k in ("object2_position", "bin_position") if k in obs}
    start = time.monotonic()
    t = 0
    while True:
        if t % frame_stride == 0:
            for cam in env.camera_names:
                frame = env.render_camera(cam, native=True)
                Image.fromarray(frame).resize((image_size, image_size), Image.Resampling.BILINEAR).save(
                    out_dir / "frames" / cam / f"{t:05d}.jpg", quality=jpeg_quality)
        states.append(obs["state"].copy())
        subtasks.append(demo.subtask)
        phases.append(demo.stage)
        mistakes.append(demo.mistake_active)
        cube.append(obs["cube_position"].copy())
        goal.append(obs["goal_position"].copy())
        ee.append(obs["ee_pose"].copy())
        for k in extra:
            extra[k].append(obs[k].copy())
        action = demo.action()
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(env.commanded_joint_targets.copy())
        t += 1
        if terminated or truncated:
            break
    success = bool(info["is_success"])
    subtask_names = sorted(set(subtasks), key=subtasks.index)
    subtask_index = np.asarray([subtask_names.index(s) for s in subtasks], dtype=np.int16)
    # Segment boundaries: a new segment starts whenever the subtask label changes.
    change = np.r_[True, subtask_index[1:] != subtask_index[:-1]]
    segment_id = np.cumsum(change) - 1
    np.savez_compressed(out_dir / "steps.npz",
                        state=np.asarray(states, dtype=np.float32), action=np.asarray(actions, dtype=np.float32),
                        subtask_index=subtask_index, segment_id=segment_id.astype(np.int16),
                        mistake=np.asarray(mistakes, dtype=bool), cube_position=np.asarray(cube, dtype=np.float32),
                        goal_position=np.asarray(goal, dtype=np.float32), ee_pose=np.asarray(ee, dtype=np.float32),
                        phase=np.asarray(phases), **{k: np.asarray(v, dtype=np.float32) for k, v in extra.items()})
    meta = {
        "schema_version": 1, "source": "pi_sim scripted", "task": info["instruction"],
        "task_id": getattr(env, "task_name", env.task), "object_shape": env.object_shape, "object_color": env.object_color,
        "second_color": getattr(env, "second_color", None), "active_side": getattr(env, "side", "left"),
        "arms_touched": info.get("touched"),
        "control_mode": "joint", "control_hz": env.control_hz, "steps": t, "frame_stride": frame_stride, "success": success,
        "failure": info["failure"], "style": style, "speed_factor": speed, "target_noise_m": noise,
        "mistakes": demo.mistakes, "quality": quality_label(success, style, demo.mistakes),
        "subtasks": subtask_names, "cameras": list(env.camera_names), "image_size": image_size,
        "state_names": [f"{side}_{j}.pos" for side in ("left", "right")
                        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")],
        "state_units": (["degree"] * 5 + ["gripper_percent_0_closed_100_open"]) * 2,
        "action_semantics": "commanded joint targets before rate limiting; same units as state",
        "seed": seed, "visual_randomization": env.visual_randomization,
        "placement_range": placement_range, "render_seconds": time.monotonic() - start,
        "camera_rig": {cam: env.camera_calibration(cam) for cam in env.camera_names},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, help="First episode index; default continues after existing episodes. Use disjoint ranges for parallel workers")
    parser.add_argument("--task", choices=("reach", "pick_place"), default="pick_place")
    parser.add_argument("--styles", default="clean,clean,slow,sloppy,slow_sloppy",
                        help="Comma-separated cycle of styles; see STYLES")
    parser.add_argument("--colors", default="red,green,blue,yellow")
    parser.add_argument("--shapes", default="cube,cylinder")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--frame-stride", type=int, default=5, help="Save camera frames every N control steps (states/actions stay at 50 Hz)")
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--no-visual-randomization", action="store_true")
    parser.add_argument("--placement", choices=("small", "wide"), default="wide")
    parser.add_argument("--camera-rig", type=Path)
    args = parser.parse_args(argv)
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    for style in styles:
        if style not in STYLES:
            parser.error(f"Unknown style {style}; choose from {sorted(STYLES)}")
    for color in colors:
        if color not in COLORS:
            parser.error(f"Unknown color {color}")
    args.out.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in args.out.iterdir() if p.is_dir() and p.name.startswith("episode_"))
    first = args.start_index if args.start_index is not None else len(existing)
    summary = []
    envs = {}
    for i in range(args.episodes):
        index = first + i
        seed = args.seed + index
        color, shape = colors[index % len(colors)], shapes[(index // len(colors)) % len(shapes)]
        key = (color, shape)
        target = args.out / f"episode_{index:05d}"
        if (target / "meta.json").exists():
            print(f"episode_{index:05d}: already recorded, skipping", flush=True)
            continue
        if target.exists():
            shutil.rmtree(target)  # interrupted earlier; redo it
        if key not in envs:
            envs[key] = RobotEnv(task=args.task, control_mode="ee_delta", object_shape=shape, object_color=color,
                                 camera_rig=args.camera_rig, control_hz=args.control_hz,
                                 episode_seconds=args.episode_seconds)
        meta = record_episode(envs[key], target, seed=seed,
                              style=styles[index % len(styles)], image_size=args.image_size,
                              visual_randomization=not args.no_visual_randomization,
                              placement_range=args.placement, frame_stride=args.frame_stride)
        summary.append(meta)
        print(f"episode_{index:05d}: {color} {shape} {meta['style']:<11} steps={meta['steps']:4d} "
              f"success={meta['success']} mistakes={meta['mistakes']} quality={meta['quality']} "
              f"({meta['render_seconds']:.1f}s)", flush=True)
    for env in envs.values():
        env.close()
    successes = sum(m["success"] for m in summary)
    print(f"Recorded {len(summary)} episodes into {args.out}: {successes} successful, "
          f"{sum(m['mistakes'] for m in summary)} retried grasps.")


if __name__ == "__main__":
    main()
