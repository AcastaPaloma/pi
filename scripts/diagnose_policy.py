"""Two checks that separate data/pipeline bugs from closed-loop failure.

    MUJOCO_GL=egl python scripts/diagnose_policy.py replay data/sim_ladder_v1 --episodes 20
    python scripts/diagnose_policy.py forced runs/full_v2/step_0002500 data/sim_ladder_v1 --episodes 6

replay: re-run each recorded episode's action labels (commanded joint targets) through the simulator
        in joint control from the recorded seed and reset options. A good label set reproduces the
        recorded outcome; if it does not, training targets cannot produce success either.
forced: teacher-forced inference. Fill the policy's history with the recorded frames and states
        (exactly what the camera saw), ask for a chunk at several points of each episode, and
        compare it with the recorded actions over the executed horizon (degrees / gripper %).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def make_env(meta, control_mode):
    if meta.get("task_id", "pick_place") in ("pick_place", "reach"):
        from pi_sim import RobotEnv
        return RobotEnv(task=meta["task_id"], control_mode=control_mode, object_color=meta["object_color"],
                        object_shape=meta["object_shape"], control_hz=meta["control_hz"], episode_seconds=30)
    from pi_sim.tasks import TaskEnv
    return TaskEnv(meta["task_id"], control_mode=control_mode, object_color=meta["object_color"],
                   second_color=meta["second_color"], object_shape=meta["object_shape"],
                   control_hz=meta["control_hz"], episode_seconds=30)


def episodes(root, n, only_success=True):
    out = []
    for d in sorted(Path(root).glob("episode_*")):
        if not (d / "meta.json").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if only_success and not meta["success"]:
            continue
        out.append((d, meta))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(out), size=min(n, len(out)), replace=False)
    return [out[i] for i in sorted(idx)]


def replay(args):
    rows = []
    for d, meta in episodes(args.root, args.episodes):
        steps = np.load(d / "steps.npz")
        env = make_env(meta, "joint")
        obs, _ = env.reset(seed=meta["seed"], options={"randomize": True, "visual_randomization": bool(meta["visual_randomization"]),
                                                        "placement_range": meta.get("placement_range", "wide")})
        start_err = float(np.abs(obs["state"] - steps["state"][0]).max())
        cube_err = float(np.linalg.norm(obs["cube_position"] - steps["cube_position"][0]))
        info, worst = {}, 0.0
        for t, a in enumerate(steps["action"]):
            obs, _, term, trunc, info = env.step(a)
            if t + 1 < len(steps["state"]):
                worst = max(worst, float(np.abs(obs["state"] - steps["state"][t + 1]).max()))
            if term or trunc:
                break
        rows.append((d.name, meta["task_id"], meta["success"], bool(info.get("is_success")), start_err, cube_err, worst))
        env.close()
        print(f"{d.name} {meta['task_id']:15s} recorded_success={meta['success']} replay_success={info.get('is_success')} "
              f"start_state_err={start_err:.3f} start_cube_err_m={cube_err:.4f} max_state_drift_deg={worst:.2f}", flush=True)
    ok = sum(r[3] for r in rows)
    print(f"replay reproduced success in {ok}/{len(rows)} recorded successes")


def forced(args):
    import torch
    from pi_model.policy import HistoryBuffer, Pi07Policy
    from pi_model.prompt import PromptContext
    from PIL import Image

    policy = Pi07Policy(args.checkpoint, device=args.device, dtype=args.dtype, denoise_steps=args.denoise_steps)
    gen = torch.Generator(device=policy.device)
    errs, base = [], []
    for d, meta in episodes(args.root, args.episodes):
        steps = np.load(d / "steps.npz")
        sub = meta["subtasks"]
        stride = meta["frame_stride"]
        T = len(steps["action"])
        queries = [int(frac * T) // stride * stride for frac in (0.1, 0.3, 0.5, 0.7)]
        live = {}
        if args.live:
            # Replay the recorded actions (they reproduce the recorded states) and render the
            # history frames live, exactly as closed-loop evaluation does.
            need = {k for t in queries for k in range(t % policy.stride, t + 1, policy.stride)}
            env = make_env(meta, "joint")
            obs, _ = env.reset(seed=meta["seed"], options={"randomize": True,
                                                           "visual_randomization": bool(meta["visual_randomization"]),
                                                           "placement_range": meta.get("placement_range", "wide")})
            for k in range(max(queries) + 1):
                if k in need:
                    live[k] = ({c: env.render_camera(c, native=True) for c in policy.cameras}, obs["state"].copy())
                obs, *_ = env.step(steps["action"][k])
            env.close()
        for t in queries:
            hist = HistoryBuffer(policy.cameras, hz=policy.hz)
            for k in range(t % policy.stride, t + 1, policy.stride):  # the 1 s history frames up to now
                if args.live:
                    native, state = live[k]
                    images = {c: policy.prepare_image(native[c]) for c in policy.cameras}
                else:
                    state = steps["state"][k]
                    images = {c: policy.prepare_image(np.asarray(Image.open(d / "frames" / c / f"{k:05d}.jpg").convert("RGB")))
                              for c in policy.cameras}
                hist.append(k, images, state)
            ctx = PromptContext(task=meta["task"], subtask=sub[int(steps["subtask_index"][t])],
                                speed_steps=policy.speed_for(meta["task"]), quality=5, mistake=False,
                                control_mode="joint", speed_bin=500)
            gen.manual_seed(0)
            chunk = policy.act(hist, ctx, generator=gen)
            H = min(args.horizon, T - t)
            truth = steps["action"][t:t + H]
            e = np.abs(chunk[:H] - truth)
            hold = np.abs(steps["action"][t:t + 1] - truth)  # "do nothing" baseline: repeat the current command
            errs.append(e.mean(0)); base.append(hold.mean(0))
            print(f"{d.name} {meta['task_id']:15s} t={t:4d} '{sub[int(steps['subtask_index'][t])][:48]}' "
                  f"mean|err| arm deg={e[:, [0,1,2,3,4,6,7,8,9,10]].mean():.2f} grip%={e[:, [5, 11]].mean():.1f} "
                  f"(hold baseline {hold[:, [0,1,2,3,4,6,7,8,9,10]].mean():.2f} / {hold[:, [5, 11]].mean():.1f})", flush=True)
    E, B = np.mean(errs, 0), np.mean(base, 0)
    print("per-dimension mean |error| (policy):", np.round(E, 2))
    print("per-dimension mean |error| (hold):  ", np.round(B, 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay"); r.add_argument("root"); r.add_argument("--episodes", type=int, default=20)
    f = sub.add_parser("forced"); f.add_argument("checkpoint"); f.add_argument("root")
    f.add_argument("--episodes", type=int, default=6); f.add_argument("--horizon", type=int, default=25)
    f.add_argument("--device", default="cuda"); f.add_argument("--dtype", default="bf16")
    f.add_argument("--denoise-steps", type=int)
    f.add_argument("--live", action="store_true", help="Render history frames live during an action replay")
    args = ap.parse_args()
    replay(args) if args.cmd == "replay" else forced(args)


if __name__ == "__main__":
    main()
