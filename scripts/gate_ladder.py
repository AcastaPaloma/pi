"""Gate a candidate teacher on the task ladder: rungs run in order, stop at the first one below threshold.

    MUJOCO_GL=egl python scripts/gate_ladder.py --teacher scripted --seeds 20 --out outputs/gate_scripted.json
    MUJOCO_GL=egl python scripts/gate_ladder.py --teacher pi05:so101-ft --seeds 10 --out outputs/gate_pi05.json

A teacher may generate data for a rung only after passing it (default 90% success on held-out
seeds 50000+). `scripted` is the privileged-state TaskTeacher (ee_delta control); `pi05:<model>`
drives the arms through scripts/pi05_sim_rollout.py's bridge in joint control (needs a LeRobot venv)
and is prompted with each episode's instruction.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_sim.tasks import TASKS, TaskEnv, TaskTeacher  # noqa: E402

COLOR_PAIRS = [("red", "blue"), ("green", "yellow"), ("blue", "red"), ("yellow", "green")]


def scripted_episode(env, seed):
    env.reset(seed=seed)
    teacher = TaskTeacher(env, rng=np.random.default_rng(seed))
    while True:
        _, _, term, trunc, info = env.step(teacher.action())
        if term or trunc:
            return {"seed": seed, "success": bool(info["is_success"]), "steps": info["elapsed_steps"],
                    "instruction": info["instruction"], "touched": info.get("touched")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="scripted", help="scripted or pi05:<model key>")
    ap.add_argument("--rungs", default=",".join(TASKS))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=50000)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--all", action="store_true", help="Run every rung even after a failure")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    bridge = None
    if args.teacher.startswith("pi05:"):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        from pi05_sim_rollout import Bridge, load, run_episode
        device = "cuda" if torch.cuda.is_available() else "cpu"
        key = args.teacher.split(":", 1)[1]
        policy, pre, post = load(key, device)
        bridge = Bridge(key, policy, pre, post, device)
    report = {"teacher": args.teacher, "threshold": args.threshold, "seeds": args.seeds, "rungs": []}
    for rung in args.rungs.split(","):
        results, start = [], time.monotonic()
        for k in range(args.seeds):
            color, second = COLOR_PAIRS[k % len(COLOR_PAIRS)]
            mode = "ee_delta" if bridge is None else "joint"
            env = TaskEnv(rung, control_mode=mode, object_color=color, second_color=second, control_hz=50, episode_seconds=30)
            seed = args.seed_base + k
            if bridge is None:
                results.append(scripted_episode(env, seed))
            else:
                r, _, _ = run_episode(env, bridge, seed=seed, prompt=None, execute=15, placement="wide")
                results.append({"seed": seed, "success": r["success"], "steps": r["steps"], "instruction": r["prompt"],
                                "closest_hand_to_cube_m": r["closest_hand_to_cube_m"], "lifted": r["lifted"]})
            env.close()
        rate = float(np.mean([r["success"] for r in results]))
        passed = rate >= args.threshold
        report["rungs"].append({"rung": rung, "success_rate": rate, "passed": passed,
                                "seconds": round(time.monotonic() - start, 1), "episodes": results})
        print(f"{rung:15s} {rate:6.1%}  {'PASS' if passed else 'FAIL'}  ({time.monotonic() - start:.0f} s)", flush=True)
        if not passed and not args.all:
            print(f"gate closed at {rung}; later rungs not attempted")
            break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
