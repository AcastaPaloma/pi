"""Does the policy's action depend on its images? Compare chunks for the same state/prompt with
real frames, black frames, and frames from a different episode.

    python scripts/vision_probe.py runs/full_v1/step_0005500 data/sim_v1
"""
import sys
from pathlib import Path

import json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_model.policy import HistoryBuffer, Pi07Policy  # noqa: E402
from pi_model.prompt import PromptContext  # noqa: E402

ckpt, root = sys.argv[1], Path(sys.argv[2])
policy = Pi07Policy(ckpt, device="cuda", dtype="fp32")
eps = sorted(p for p in root.glob("episode_*") if (p / "meta.json").exists())[:40:8]
gen = torch.Generator(device=policy.device)


def frames(d, k):
    return {c: policy.prepare_image(np.asarray(Image.open(d / "frames" / c / f"{k:05d}.jpg").convert("RGB"))) for c in policy.cameras}


def chunk(d, t, image_source):
    meta = json.loads((d / "meta.json").read_text()); steps = np.load(d / "steps.npz")
    hist = HistoryBuffer(policy.cameras, hz=policy.hz)
    for k in range(t % policy.stride, t + 1, policy.stride):
        hist.append(k, image_source(k), steps["state"][k])
    ctx = PromptContext(task=meta["task"], subtask=meta["subtasks"][int(steps["subtask_index"][t])],
                        speed_steps=policy.speed_for(meta["task"]), quality=5, mistake=False, control_mode="joint", speed_bin=500)
    gen.manual_seed(0)
    return policy.act(hist, ctx, generator=gen), steps["action"][t:t + 25]


for i, d in enumerate(eps):
    other = eps[(i + 1) % len(eps)]
    t = 20
    real, truth = chunk(d, t, lambda k: frames(d, k))
    black, _ = chunk(d, t, lambda k: {c: np.zeros_like(v) for c, v in frames(d, k).items()})
    swap, _ = chunk(d, t, lambda k: frames(other, k))
    f = lambda a: np.abs(a[:25, :5] - truth[:, :5]).mean()
    print(f"{d.name}: err vs truth (deg) real={f(real):.2f} black={f(black):.2f} other-episode={f(swap):.2f} | "
          f"|real-black|={np.abs(real - black)[:25, :5].mean():.3f} |real-other|={np.abs(real - swap)[:25, :5].mean():.3f}", flush=True)
