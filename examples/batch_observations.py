"""128 observations directly in Python, a contact sheet, and honest timings.

python examples/batch_observations.py
python examples/batch_observations.py --source real --prepare-demo
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("sim", "real"), default="sim")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--prepare-demo", action="store_true", help="Prepare only two train episodes, stride 10, for an input-pipeline smoke test")
    args = parser.parse_args()
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    started = time.perf_counter()
    if args.source == "sim":
        from pi_sim import RobotBatch
        robot = RobotBatch(batch_size=args.batch_size, image_size=args.size, seed=0)
    else:
        from pi_data import FoldingDataset, FoldingBatch, prepare_observation_cache
        cache = args.cache or output / f"folding_encoder_demo_{args.size}"
        if args.prepare_demo and not cache.exists():
            ds = FoldingDataset()
            prepare_observation_cache(cache, dataset=ds, episodes=ds.episode_ids("train")[:2],
                                      image_size=args.size, stride=10, progress=print)
        robot = FoldingBatch(cache, batch_size=args.batch_size, seed=0)
    setup_seconds = time.perf_counter() - started
    try:
        started = time.perf_counter()
        obs = robot.observations()
        first_seconds = time.perf_counter() - started
        durations = []
        for _ in range(5):
            started = time.perf_counter()
            newer = robot.observations()
            durations.append(time.perf_counter() - started)
            del newer
        grid_path = output / f"{args.source}_{args.batch_size}_robots_{args.size}.png"
        robot.grid(obs, columns=16, tile_size=128).save(grid_path)
        result = {"source": args.source, "batch_size": len(obs["images"]),
                  "images_shape": list(obs["images"].shape), "state_shape": list(obs["proprioception"].shape),
                  "images_mib": obs["images"].nbytes / 2**20, "setup_seconds": setup_seconds,
                  "first_observations_seconds": first_seconds,
                  "warm_observations_median_seconds": statistics.median(durations),
                  "timing_scope": "CPU NumPy batch only; no device transfer or model forward/backward",
                  "warm_sampling": "same cached poses, fresh array copies" if args.source == "sim" else "random cache rows, OS-warm memory-mapped images",
                  "grid": str(grid_path)}
        (output / f"{args.source}_batch_benchmark_{args.size}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        if args.source == "sim":
            robot.close()


if __name__ == "__main__":
    main()
