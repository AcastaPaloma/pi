"""Record the task ladder's scripted episodes in parallel (same on-disk format as pi_sim.record).

    MUJOCO_GL=egl python -m pi_sim.record_tasks --out data/sim_ladder_v1 \
        --tasks pick_place_any,bin,handover,two_bins --episodes-per-task 1000 --workers 16

Episode i of a task uses seed `seed + i` and cycles styles, object colors (two distinct colors
per episode) and shapes, so runs are reproducible and resumable (existing episodes are skipped).
With NVIDIA EGL rendering an episode takes about 1-4 s per worker.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

COLOR_PAIRS = [("red", "blue"), ("green", "yellow"), ("blue", "red"), ("yellow", "green"),
               ("red", "green"), ("blue", "yellow")]
STYLE_CYCLE = ("clean", "clean", "slow", "sloppy", "slow_sloppy")


def _jobs(args):
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    for ti, task in enumerate(tasks):
        for i in range(args.episodes_per_task):
            index = ti * 100000 + i
            color, second = COLOR_PAIRS[i % len(COLOR_PAIRS)]
            shape = shapes[(i // len(COLOR_PAIRS)) % len(shapes)]
            yield dict(task=task, index=index, seed=args.seed + index, color=color, second=second, shape=shape,
                       style=styles[i % len(styles)])


def _worker(queue, results, args_dict):
    from .record import record_episode
    from .tasks import TaskEnv, TaskTeacher
    envs = {}
    out = Path(args_dict["out"])
    while True:
        job = queue.get()
        if job is None:
            break
        target = out / f"episode_{job['index']:06d}"
        if (target / "meta.json").exists():
            results.put((job, None, "skipped"))
            continue
        if target.exists():
            shutil.rmtree(target)
        key = (job["task"], job["color"], job["second"], job["shape"])
        if key not in envs:
            if len(envs) >= 3:  # each env holds GPU render contexts; keep a few
                envs.pop(next(iter(envs))).close()
            envs[key] = TaskEnv(job["task"], control_mode="ee_delta", object_color=job["color"], second_color=job["second"],
                                object_shape=job["shape"], control_hz=50, episode_seconds=args_dict["episode_seconds"])
        try:
            meta = record_episode(envs[key], target, seed=job["seed"], style=job["style"],
                                  image_size=args_dict["image_size"], frame_stride=args_dict["frame_stride"],
                                  visual_randomization=not args_dict["no_visual_randomization"], teacher=TaskTeacher)
            results.put((job, {k: meta[k] for k in ("success", "steps", "mistakes", "render_seconds", "task")}, "ok"))
        except Exception as exc:  # keep the worker alive; report and continue
            shutil.rmtree(target, ignore_errors=True)
            results.put((job, None, f"error: {exc!r}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", default="pick_place_any,bin,handover,two_bins")
    ap.add_argument("--episodes-per-task", type=int, default=100)
    ap.add_argument("--shapes", default="cube,cylinder")
    ap.add_argument("--styles", default=",".join(STYLE_CYCLE),
                    help="Style cycle; 'recover' knocks the teacher off course (recovery data, mistake-labelled)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--episode-seconds", type=float, default=30.0)
    ap.add_argument("--no-visual-randomization", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    # Group jobs by scene setup so each worker mostly reuses one cached environment.
    jobs = sorted(_jobs(args), key=lambda j: (j["task"], j["color"], j["second"], j["shape"], j["index"]))
    ctx = mp.get_context("spawn")
    queue, results = ctx.Queue(), ctx.Queue()
    for job in jobs:
        queue.put(job)
    for _ in range(args.workers):
        queue.put(None)
    procs = [ctx.Process(target=_worker, args=(queue, results, vars(args) | {"out": str(args.out)})) for _ in range(args.workers)]
    for p in procs:
        p.start()
    start, done, stats = time.monotonic(), 0, {}
    for _ in jobs:
        job, meta, status = results.get()
        done += 1
        s = stats.setdefault(job["task"], {"ok": 0, "success": 0, "skipped": 0, "errors": 0})
        if status == "ok":
            s["ok"] += 1
            s["success"] += int(meta["success"])
        elif status == "skipped":
            s["skipped"] += 1
        else:
            s["errors"] += 1
            print(f"episode_{job['index']:06d} {job['task']}: {status}", flush=True)
        if done % 200 == 0 or done == len(jobs):
            rate = done / max(time.monotonic() - start, 1e-6)
            print(f"{done}/{len(jobs)} episodes, {rate * 3600:.0f}/h, {stats}", flush=True)
    for p in procs:
        p.join()
    print(f"done in {time.monotonic() - start:.0f} s: {stats}")


if __name__ == "__main__":
    main()
