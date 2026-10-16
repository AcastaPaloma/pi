"""Uniformly average several checkpoints of one run (a "model soup") and save the result as bf16.

    python scripts/box/average_checkpoints.py exports/pi05_v1_soup/step_0018000 \
        runs/pi05_v1/step_0013000 runs/pi05_v1/step_0014000 ... runs/pi05_v1/step_0018000

Tests H9: if neighbouring checkpoints trade skills (one is better at pick, the next at two_bins),
their average should keep both, the same effect EMA gives during training but without retraining.
Accumulates in fp32; config/stats/extra are copied from the last checkpoint.
"""
import json, shutil, sys
from pathlib import Path
import torch

dst, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
assert len(srcs) >= 2, "need at least two checkpoints"
acc = None
for src in srcs:
    state = torch.load(src / "model.pt", map_location="cpu", mmap=True)
    if acc is None:
        acc = {k: (v.float().clone() if v.is_floating_point() else v.clone()) for k, v in state.items()}
    else:
        assert state.keys() == acc.keys(), f"{src} has different parameters"
        for k, v in state.items():
            if v.is_floating_point():
                acc[k].add_(v.float())
    print(f"added {src}", flush=True)
    del state
dst.mkdir(parents=True, exist_ok=True)
out = {k: (v.div_(len(srcs)).to(torch.bfloat16) if v.is_floating_point() else v) for k, v in acc.items()}
torch.save(out, dst / "model.pt.tmp")
for name in ("config.json", "stats.json", "extra.json"):
    if (srcs[-1] / name).exists():
        shutil.copy2(srcs[-1] / name, dst / name)
(dst / "soup.json").write_text(json.dumps({"averaged": [str(s) for s in srcs]}, indent=1))
(dst / "model.pt.tmp").rename(dst / "model.pt")
print(f"averaged {len(srcs)} checkpoints -> {dst}")
