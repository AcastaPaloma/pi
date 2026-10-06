"""python -m pi_data prepare|serve|sample"""
import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import FoldingDataset


def main():
    parser = argparse.ArgumentParser(description="LeHome low-level encoder data (no PyTorch)")
    parser.add_argument("--root", help="Offline local four_types_merged directory; otherwise use the HF cache")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download metadata, actions, and an episode's three video files")
    prepare.add_argument("--episode", type=int, default=0)
    prepare.add_argument("--all", action="store_true", help="Download the full 4.48 GB selected subset")
    serve = commands.add_parser("serve", help="Start read-only HTTP API with interactive /docs")
    serve.add_argument("--port", type=int, default=8000)
    sample = commands.add_parser("sample", help="Export one encoder example as NumPy arrays")
    sample.add_argument("--episode", type=int, default=0)
    sample.add_argument("--frame", type=int, default=100)
    sample.add_argument("--history", type=int, default=1)
    sample.add_argument("--size", type=int, default=448)
    sample.add_argument("--goal-seconds", type=float, default=None)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--output", type=Path, default=Path("outputs/encoder_sample.npz"))
    args = parser.parse_args()
    ds = FoldingDataset(args.root)
    if args.command == "prepare":
        ds.prepare(args.episode, all_videos=args.all)
        print(json.dumps({"ready": True, "episode": args.episode, "all": args.all,
                          "train_episodes": len(ds.episode_ids("train")),
                          "validation_episodes": len(ds.episode_ids("validation"))}, indent=2))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(ds), host="127.0.0.1", port=args.port)
    else:
        arrays = ds.arrays(args.episode, args.frame, history=args.history, size=args.size,
                           goal_seconds=args.goal_seconds, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        print(args.output)
        for key, value in arrays.items():
            print(f"  {key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
