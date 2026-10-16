"""DAgger-style recovery data: the policy drives into its own imperfect states, the teacher finishes.

    MUJOCO_GL=egl python -m pi_model.dagger --checkpoint exports/pi05_v1/step_0008000 \
        --tasks pick_place_any,bin,handover,two_bins --episodes-per-task 200 --out data/sim_dagger_v1

Each episode: the trained policy runs closed loop (oracle-subtask prompts, as in evaluation) for a
random 1–12 s, then the privileged scripted teacher takes over from exactly that state: its phases are
synced to the state once, after which its own phase machine (with retries) runs in joint control,
converting each Cartesian step into joint targets by inverse kinematics. Only the teacher's segment is
recorded, in the same on-disk format as pi_sim.record, so training sees expert continuations from the
states the policy actually reaches (the states plain imitation never covers).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pi_sim.tasks import TaskEnv, TaskTeacher
from .policy import HistoryBuffer, Pi07Policy
from .prompt import PromptContext

COLOR_PAIRS = [("red", "blue"), ("green", "yellow"), ("blue", "red"), ("yellow", "green")]


def teacher_joint_action(env: TaskEnv, teacher: TaskTeacher, speed=0.85):
    """One teacher step in joint control: Cartesian step toward the phase target, solved by IK."""
    teacher._advance()
    action = env.commanded_joint_targets.copy().reshape(2, -1)
    step = env.delta_scale * speed
    for w in teacher.workers:
        out = w.target()
        if out is None:
            continue
        tgt, opening = out
        # Accumulate a commanded end-effector point exactly as ee_delta control does (stepping from the
        # measured hand position instead stalls: the PD arm lags under load and never closes the gap).
        if getattr(w, "_ee_cmd", None) is None:
            w._ee_cmd = env.robot[w.arm].position
        w._ee_cmd = np.clip(w._ee_cmd + np.clip(tgt - w._ee_cmd, -step, step),
                            env._workspace_low[w.i], env._workspace_high[w.i])
        q, _ = env._solve_ik(env.robot[w.arm], w._ee_cmd, env._orientations[w.i], downward=True)
        action[w.i, :5] = np.rad2deg(q)
        action[w.i, 5] = opening * 100
    return action.ravel().astype(np.float32)


def run_episode(env, policy, out_dir, *, seed, takeover_s, image_size=448, frame_stride=5, gen=None):
    obs, info = env.reset(seed=seed, options={"randomize": True, "visual_randomization": True})
    task = info["instruction"]
    history = HistoryBuffer(policy.cameras, hz=env.control_hz)
    oracle = TaskTeacher(env, rng=np.random.default_rng(seed), observe_only=True)
    takeover = int(takeover_s * env.control_hz)
    t, chunk, cursor = 0, None, 25
    # Phase 1: the policy drives (not recorded).
    while t < takeover:
        if cursor >= 25:
            images = {c: policy.prepare_image(env.render_camera(c, native=True)) for c in policy.cameras}
            history.append(t, images, obs["state"])
            ctx = PromptContext(task=task, subtask=oracle.subtask, speed_steps=policy.speed_for(task), quality=5,
                                mistake=False, control_mode="joint", speed_bin=500)
            chunk, cursor = policy.act(history, ctx, generator=gen), 0
        elif t % 50 == 0:
            history.append(t, {c: policy.prepare_image(env.render_camera(c, native=True)) for c in policy.cameras},
                           obs["state"])
        obs, _, term, trunc, info = env.step(chunk[cursor])
        oracle.observe()
        cursor += 1
        t += 1
        if term or trunc:
            return None  # finished or failed before the takeover: nothing to learn from this one
    # Phase 2: the teacher takes over from this state and its continuation is recorded.
    teacher = TaskTeacher(env, rng=np.random.default_rng(seed + 1), observe_only=True)
    teacher.observe()
    for w in teacher.workers:
        w.obj = env.object_position(w.body)
        w.ticks = 0
    out_dir.mkdir(parents=True, exist_ok=False)
    for cam in env.camera_names:
        (out_dir / "frames" / cam).mkdir(parents=True)
    rec = {k: [] for k in ("state", "action", "sub", "cube", "goal", "ee", "obj2", "bin", "phase")}
    start, k = time.monotonic(), 0
    while True:
        if k % frame_stride == 0:
            for cam in env.camera_names:
                Image.fromarray(env.render_camera(cam, native=True)).resize((image_size, image_size), Image.Resampling.BILINEAR) \
                    .save(out_dir / "frames" / cam / f"{k:05d}.jpg", quality=90)
        rec["state"].append(obs["state"].copy()); rec["sub"].append(teacher.subtask); rec["phase"].append(teacher.stage)
        rec["cube"].append(obs["cube_position"].copy()); rec["goal"].append(obs["goal_position"].copy())
        rec["ee"].append(obs["ee_pose"].copy()); rec["obj2"].append(obs["object2_position"].copy())
        rec["bin"].append(obs["bin_position"].copy())
        action = teacher_joint_action(env, teacher)
        obs, _, term, trunc, info = env.step(action)
        rec["action"].append(env.commanded_joint_targets.copy())
        k += 1
        if term or trunc:
            break
    names = sorted(set(rec["sub"]), key=rec["sub"].index)
    sub_idx = np.asarray([names.index(x) for x in rec["sub"]], dtype=np.int16)
    seg = np.cumsum(np.r_[True, sub_idx[1:] != sub_idx[:-1]]) - 1
    np.savez_compressed(out_dir / "steps.npz", state=np.asarray(rec["state"], np.float32), action=np.asarray(rec["action"], np.float32),
                        subtask_index=sub_idx, segment_id=seg.astype(np.int16), mistake=np.zeros(k, bool),
                        cube_position=np.asarray(rec["cube"], np.float32), goal_position=np.asarray(rec["goal"], np.float32),
                        ee_pose=np.asarray(rec["ee"], np.float32), phase=np.asarray(rec["phase"]),
                        object2_position=np.asarray(rec["obj2"], np.float32), bin_position=np.asarray(rec["bin"], np.float32))
    success = bool(info["is_success"])
    meta = {
        "schema_version": 1, "source": "dagger: policy rollout then scripted-teacher takeover", "task": task,
        "task_id": env.task_name, "object_shape": env.object_shape, "object_color": env.object_color,
        "second_color": env.second_color, "active_side": env.side, "arms_touched": info.get("touched"),
        "control_mode": "joint", "control_hz": env.control_hz, "steps": k, "frame_stride": frame_stride,
        "success": success, "failure": info["failure"], "style": "dagger", "speed_factor": 0.85, "target_noise_m": 0.0,
        "mistakes": 0, "quality": 5 if success else 1, "subtasks": names, "cameras": list(env.camera_names),
        "image_size": image_size, "takeover_step": takeover,
        "state_names": [f"{side}_{j}.pos" for side in ("left", "right")
                        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")],
        "state_units": (["degree"] * 5 + ["gripper_percent_0_closed_100_open"]) * 2,
        "action_semantics": "commanded joint targets before rate limiting; same units as state",
        "seed": seed, "visual_randomization": env.visual_randomization, "placement_range": "task",
        "render_seconds": time.monotonic() - start,
        "camera_rig": {cam: env.camera_calibration(cam) for cam in env.camera_names},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tasks", default="pick_place_any,bin,handover,two_bins")
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=900000)
    ap.add_argument("--takeover-min", type=float, default=1.0)
    ap.add_argument("--takeover-max", type=float, default=12.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episode-seconds", type=float, default=40.0)
    args = ap.parse_args()
    policy = Pi07Policy(args.checkpoint, device=args.device, dtype="bf16")
    gen = torch.Generator(device=policy.device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    done = {"kept": 0, "skipped": 0, "success": 0}
    for ti, task in enumerate(args.tasks.split(",")):
        for i in range(args.episodes_per_task):
            index = ti * 100000 + i
            target = args.out / f"episode_{index:06d}"
            if (target / "meta.json").exists():
                continue
            shutil.rmtree(target, ignore_errors=True)
            color, second = COLOR_PAIRS[i % len(COLOR_PAIRS)]
            env = TaskEnv(task, control_mode="joint", object_color=color, second_color=second, control_hz=policy.hz,
                          episode_seconds=args.episode_seconds)
            meta = run_episode(env, policy, target, seed=args.seed + index,
                               takeover_s=rng.uniform(args.takeover_min, args.takeover_max), gen=gen)
            env.close()
            if meta is None:
                done["skipped"] += 1
            else:
                done["kept"] += 1
                done["success"] += int(meta["success"])
            if (i + 1) % 20 == 0:
                print(f"{task} {i + 1}/{args.episodes_per_task} {done}", flush=True)
    print(f"done {done}", flush=True)


if __name__ == "__main__":
    main()
