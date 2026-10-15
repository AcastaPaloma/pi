"""Drive the simulated SO-101 rig with a public LeRobot π0.5 checkpoint, record everything.

Zero-shot test of off-the-shelf π0.5 policies in our MuJoCo scene, and a recorder for
trajectories a teacher policy produces. Runs in a LeRobot venv with pi_sim installed:

    MUJOCO_GL=egl python scripts/pi05_sim_rollout.py --model so101-ft --episodes 4 \
        --prompt "Put the red cube on the green disk" --out outputs/pi05_ft

Models:
  so101-ft  sun3689/pi05-so101-cube-fullft-30k-7799251 (step 30k): π0.5 fully fine-tuned on
            50 real SO-101 episodes (front + wrist camera, 6 joints, 30 fps). Our overhead
            camera stands in for its front camera and our left wrist camera for its wrist.
  base      lerobot/pi05_base: no SO-101 training. Our three cameras fill its base/left/right
            wrist slots; the 12 joint values are mapped to [-1, 1] with our dataset quantiles
            and padded to 32, and its first 12 outputs are mapped back the same way.

Joint angles pass through unchanged: our MJCF uses the official SO-101 joint limits, which
follow LeRobot's calibrated degree convention (an assumption the hardware audit must confirm).
Each episode saves results, a video (overhead | left wrist) and steps.npz with states,
commanded actions, raw policy chunks and frames, so a single trajectory can be inspected.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pi_sim import RobotEnv

MODELS = {
    "so101-ft": ("sun3689/pi05-so101-cube-fullft-30k-7799251",
                 "pi05_so101_cube_full_30k_7799251/checkpoints/030000/pretrained_model"),
    "base": ("lerobot/pi05_base", ""),
}
POLICY_HZ = 30


def load(model_key, device):
    from huggingface_hub import snapshot_download
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    repo, sub = MODELS[model_key]
    snap = snapshot_download(repo, allow_patterns=[f"{sub}/*"] if sub else None)
    path = str(Path(snap) / sub) if sub else snap
    policy = PI05Policy.from_pretrained(path).to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path,
                                         preprocessor_overrides={"device_processor": {"device": device}})
    return policy, pre, post


def to_chw(rgb, device):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)


class Bridge:
    """Maps our observation to the checkpoint's inputs and its output chunk to our 6 left-arm targets."""

    def __init__(self, model_key, policy, pre, post, device, stats=None):
        self.key, self.policy, self.pre, self.post, self.device = model_key, policy, pre, post, device
        if model_key == "base":
            if stats is None:
                raise ValueError("the base model needs our dataset statistics (--stats)")
            self.lo = np.asarray(stats["state_q01"][:12], np.float32)
            self.hi = np.asarray(stats["state_q99"][:12], np.float32)
            self.alo = np.asarray(stats["action_q01"][:12], np.float32)
            self.ahi = np.asarray(stats["action_q99"][:12], np.float32)

    @torch.no_grad()
    def chunk(self, images, state12, prompt):
        if self.key == "so101-ft":
            obs = {"observation.state": torch.from_numpy(state12[:6].astype(np.float32)).unsqueeze(0).to(self.device),
                   "observation.images.front": to_chw(images["overhead"], self.device),
                   "observation.images.wrist": to_chw(images["left_wrist"], self.device),
                   "task": [prompt]}
            batch = self.pre(obs)
            out = self.policy.predict_action_chunk(batch)
            out = self.post(out)
            return out[0, :, :6].float().cpu().numpy()  # [50, 6] degrees + gripper, at 30 fps
        s = 2 * (state12 - self.lo) / np.maximum(self.hi - self.lo, 1e-6) - 1
        state32 = np.zeros(32, np.float32)
        state32[:12] = np.clip(s, -1, 1)
        obs = {"observation.state": torch.from_numpy(state32).unsqueeze(0).to(self.device),
               "observation.images.base_0_rgb": to_chw(images["overhead"], self.device),
               "observation.images.left_wrist_0_rgb": to_chw(images["left_wrist"], self.device),
               "observation.images.right_wrist_0_rgb": to_chw(images["right_wrist"], self.device),
               "task": [prompt]}
        batch = self.pre(obs)
        out = self.policy.predict_action_chunk(batch)[0, :, :12].float().cpu().numpy()
        out = (np.clip(out, -1, 1) + 1) / 2 * (self.ahi - self.alo) + self.alo
        return out  # [50, 12]


def run_episode(env, bridge, *, seed, prompt, execute, placement, frame_stride=5):
    obs, info = env.reset(seed=seed, options={"placement_range": placement})
    if prompt is None:
        prompt = info["instruction"]
    hz = env.control_hz
    steps_per_call = int(round(execute * hz / POLICY_HZ))
    rec = {k: [] for k in ("state", "action", "frame_step", "overhead", "left_wrist")}
    chunks, t, closest, max_z = [], 0, np.inf, 0.0
    t0 = time.monotonic()
    while True:
        images = {c: env.render_camera(c, native=True) for c in ("overhead", "left_wrist", "right_wrist")}
        chunk = bridge.chunk(images, obs["state"].astype(np.float32), prompt)
        chunks.append(chunk)
        hold = env.commanded_joint_targets.copy()
        for k in range(steps_per_call):
            # Resample the 30 fps chunk at 50 Hz by linear interpolation.
            x = (k + 1) * POLICY_HZ / hz - 1
            i0 = int(np.clip(np.floor(x), 0, len(chunk) - 1)); i1 = min(i0 + 1, len(chunk) - 1)
            w = float(np.clip(x - i0, 0, 1))
            target = (1 - w) * chunk[i0] + w * chunk[i1]
            action = hold.copy()
            if target.shape[0] == 6:
                action[:6] = target
            else:
                action[:] = target
            if t % frame_stride == 0:
                fr = images if k == 0 else {c: env.render_camera(c, native=True) for c in ("overhead", "left_wrist")}
                rec["frame_step"].append(t); rec["overhead"].append(fr["overhead"]); rec["left_wrist"].append(fr["left_wrist"])
            rec["state"].append(obs["state"].copy()); rec["action"].append(action.copy())
            obs, _, terminated, truncated, info = env.step(action)
            t += 1
            closest = min(closest, float(np.linalg.norm(obs["ee_pose"][:3] - obs["cube_position"])))
            max_z = max(max_z, float(obs["cube_position"][2]))
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    result = {"seed": seed, "success": bool(info["is_success"]), "steps": t, "failure": info["failure"],
              "prompt": prompt, "object": f"{env.object_color} {env.object_shape}", "policy_calls": len(chunks),
              "closest_hand_to_cube_m": round(closest, 4), "lifted": bool(max_z > 0.045),
              "final_cube_to_goal_m": round(float(np.linalg.norm(obs["cube_position"] - obs["goal_position"])), 4),
              "wall_s": round(time.monotonic() - t0, 1)}
    return result, rec, np.stack(chunks)


def save(out, i, result, rec, chunks):
    from PIL import Image
    import av
    ep = out / f"episode_{i:03d}"
    ep.mkdir(parents=True, exist_ok=True)
    frames = [np.concatenate([a, b], 1) for a, b in zip(rec["overhead"], rec["left_wrist"])]
    container = av.open(str(ep / "video.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=10)
    stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
    for f in frames:
        for p in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(p)
    for p in stream.encode():
        container.mux(p)
    container.close()
    np.savez_compressed(ep / "steps.npz", state=np.asarray(rec["state"], np.float32), action=np.asarray(rec["action"], np.float32),
                        policy_chunks=chunks.astype(np.float32), frame_step=np.asarray(rec["frame_step"]),
                        overhead=np.asarray(rec["overhead"][::2]), left_wrist=np.asarray(rec["left_wrist"][::2]))
    Image.fromarray(frames[len(frames) // 2]).save(ep / "middle.png")
    (ep / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), default="so101-ft")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--prompt", default="Put the {color} {shape} on the green disk")
    ap.add_argument("--execute", type=int, default=15, help="Chunk actions (30 fps) executed per call")
    ap.add_argument("--colors", default="red,green,blue,yellow")
    ap.add_argument("--shape", default="cube")
    ap.add_argument("--placement", default="wide")
    ap.add_argument("--episode-seconds", type=float, default=20.0)
    ap.add_argument("--stats", type=Path, help="Our dataset stats.json (needed for --model base)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, pre, post = load(args.model, device)
    stats = json.loads(args.stats.read_text()) if args.stats else None
    bridge = Bridge(args.model, policy, pre, post, device, stats)
    args.out.mkdir(parents=True, exist_ok=True)
    results, envs = [], {}
    for i in range(args.episodes):
        color = args.colors.split(",")[i % len(args.colors.split(","))]
        if color not in envs:
            envs[color] = RobotEnv(task="pick_place", control_mode="joint", object_color=color, object_shape=args.shape,
                                   control_hz=50, episode_seconds=args.episode_seconds)
        prompt = args.prompt.format(color=color, shape=args.shape)
        result, rec, chunks = run_episode(envs[color], bridge, seed=args.seed + i, prompt=prompt,
                                          execute=args.execute, placement=args.placement)
        results.append(result)
        print(f"episode {i}: {result['object']:<12} success={result['success']} closest={result['closest_hand_to_cube_m']:.3f}m "
              f"lifted={result['lifted']} final_to_goal={result['final_cube_to_goal_m']:.3f}m calls={result['policy_calls']} "
              f"wall={result['wall_s']}s", flush=True)
        save(args.out, i, result, rec, chunks)
    rate = float(np.mean([r["success"] for r in results]))
    (args.out / "results.json").write_text(json.dumps({"model": args.model, "execute": args.execute, "results": results,
                                                       "success_rate": rate}, indent=2) + "\n")
    print(f"success rate {rate:.0%} over {len(results)} episodes")


if __name__ == "__main__":
    main()
