"""Is image content used in the training path? Loss with true vs batch-shuffled images, and how
different the vision features of different scenes are.

    python scripts/vision_flow_probe.py runs/exp_B_nostate/step_0001000 data/sim_v1
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_model.config import Pi07Config  # noqa: E402
from pi_model.data import Pi07Dataset, collate, to_device  # noqa: E402
from pi_model.model import Pi07  # noqa: E402
from pi_model.tokenizer import load_tokenizer  # noqa: E402

ckpt, root = Path(sys.argv[1]), Path(sys.argv[2])
cfg = Pi07Config.from_json((ckpt / "config.json").read_text())
extra = json.loads((ckpt / "extra.json").read_text()) if (ckpt / "extra.json").exists() else {}
model = Pi07(cfg)
model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
model = model.cuda().eval()
tok = load_tokenizer(extra.get("tokenizer"), cfg.text.vocab)
ds = Pi07Dataset(root, extra.get("meta_dir", root / "pi07_meta"), cfg, tok, train=False)
rng = np.random.default_rng(0)
idx = rng.choice(len(ds), 16, replace=False)
batch = to_device(collate([ds[int(i)] for i in idx], cfg), torch.device("cuda"))
print("image tensor:", tuple(batch["images"].shape), batch["images"].dtype,
      "range", float(batch["images"].float().min()), float(batch["images"].float().max()),
      "per-sample std of current overhead frame:", [round(float(batch["images"][b, 0, -1].float().std()), 3) for b in range(4)])
print("frame_valid[0]:", batch["frame_valid"][0].tolist(), "seconds_ago[0]:", batch["seconds_ago"][0].tolist())


def losses(b, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    B, H, A = b["actions"].shape
    noise = torch.randn(B, H, A, device="cuda", generator=g)
    tau = torch.full((B,), 0.5, device="cuda")
    with torch.no_grad():
        out = model(b, tau=tau, noise=noise)
    return float(out["flow_loss"]), float(out.get("ce_loss", 0.0))


base = losses(batch)
perm = torch.roll(torch.arange(16, device="cuda"), 1)
shuf = dict(batch); shuf["images"] = batch["images"][perm]
black = dict(batch); black["images"] = torch.full_like(batch["images"], -1.0)
print(f"flow/ce loss  true images {base[0]:.4f}/{base[1]:.4f}   shuffled {losses(shuf)[0]:.4f}/{losses(shuf)[1]:.4f}   "
      f"black {losses(black)[0]:.4f}/{losses(black)[1]:.4f}")

with torch.no_grad():
    imgs = batch["images"][:, 0]  # overhead camera, [B, K, 3, H, W]
    feats = model.vision(imgs, batch["seconds_ago"], batch["frame_valid"]).float()  # [B, T, D]
    pix = imgs[:, -1].float().flatten(1)
    f = feats.flatten(1)
    cos = torch.nn.functional.cosine_similarity(f[:, None], f[None], dim=-1)
    pcos = torch.nn.functional.cosine_similarity(pix[:, None] - pix.mean(0), pix[None] - pix.mean(0), dim=-1)
    off = ~torch.eye(16, dtype=torch.bool, device="cuda")
    print(f"vision features: mean pairwise cosine across different scenes {float(cos[off].mean()):.4f} "
          f"(centred pixel cosine {float(pcos[off].mean()):.4f}); feature norm {float(f.norm(dim=1).mean()):.1f}; "
          f"between-scene std / within-scene token std = {float(feats.std(0).mean() / feats.std(1).mean()):.4f}")
    text_norm = float(model.backbone.embed(batch["text_ids"][:1]).float().norm(dim=-1).mean())
    print(f"per-token norms: vision {float(feats.norm(dim=-1).mean()):.1f}, text {text_norm:.1f}")
