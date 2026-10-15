"""Copy a checkpoint with bf16 weights (half the bytes) for transfer to an evaluation GPU.

    python scripts/box/export_bf16.py runs/full_v1/step_0001000 exports/full_v1/step_0001000
The bf16 values load losslessly into a policy run with --dtype bf16.
"""
import shutil, sys
from pathlib import Path
import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
state = torch.load(src / "model.pt", map_location="cpu", mmap=True)
torch.save({k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in state.items()}, dst / "model.pt.tmp")
for name in ("config.json", "stats.json", "extra.json"):
    if (src / name).exists():
        shutil.copy2(src / name, dst / name)
(dst / "model.pt.tmp").rename(dst / "model.pt")
print(f"exported {src} -> {dst}")
