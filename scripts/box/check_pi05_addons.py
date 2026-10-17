"""Equivalence and gradient checks for the π0.7 add-ons on the π0.5 backbone, on a real checkpoint and real data.

    CUDA_VISIBLE_DEVICES=0 ~/lerobot-venv/bin/python scripts/box/check_pi05_addons.py \
        --checkpoint runs/pi05_dagger2/step_0007000 --data data/sim_mixdagclean_v1 --meta data/sim_mix_v1/pi07_meta

Each add-on switched off must reproduce LeRobot's π0.5 exactly; MEM with no valid past frame must equal the
single-frame encoder; knowledge insulation must stop every flow gradient into the VLM; an RTC sample must keep
the committed prefix. Also reports the FAST cross-entropy at the start for both FAST tokenizers.
"""
import argparse
from pathlib import Path

import torch

from pi_model.config import Pi07Config
from pi_model.data import Pi07Dataset, collate, to_device
from pi_model.pi05_backbone import Pi05Backbone
from pi_model.tokenizer import load_tokenizer


def batch_for(cfg, args, tokenizer, n=2, seed=0):
    ds = Pi07Dataset(args.data, args.meta, cfg, tokenizer, train=True, seed=seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(ds), (n,), generator=g).tolist()
    return to_device(collate([ds[i] for i in idx], cfg), "cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    args = ap.parse_args()
    base = Pi07Config.from_json((args.checkpoint / "config.json").read_text())
    tokenizer = load_tokenizer(Path("weights/paligemma/tokenizer.model"), base.text.vocab)
    model = Pi05Backbone(base, pretrained=False)
    state = torch.load(args.checkpoint / "model.pt", map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state.get("model", state), strict=False)
    print("load: missing", len(missing), "unexpected", len(unexpected))
    model = model.cuda().eval()
    pw = model.inner.paligemma_with_expert
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name, detail, flush=True)

    torch.manual_seed(0)
    batch = batch_for(base, args, tokenizer)
    B, H, A = batch["actions"].shape
    tau, noise = torch.rand(B, device="cuda"), torch.randn(B, H, A, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref = model(batch, tau=tau, noise=noise)["flow_loss"]
        model.addons = True                                 # same switches, add-on code path
        alt = model(batch, tau=tau, noise=noise)["flow_loss"]
        model.addons = False
    check("add-on path == LeRobot forward", torch.allclose(ref, alt, rtol=2e-2, atol=1e-4), f"{ref.item():.5f} vs {alt.item():.5f}")

    gen = lambda: torch.Generator(device="cuda").manual_seed(1)
    with torch.no_grad():
        a = model.sample_actions(batch, steps=5, generator=gen())
        model.addons = True
        b = model.sample_actions(batch, steps=5, generator=gen())
        model.addons = False
    check("add-on sampler == LeRobot sampler", (a - b).abs().max().item() < 2e-2, f"max diff {(a - b).abs().max().item():.2e}")

    # RTC: per-token times with delay 0 equal the scalar path; a sample keeps the committed prefix
    model.cfg = base.with_(pi05_rtc=True)
    model.addons = True
    batch0 = dict(batch, rtc_delay=torch.zeros(B, dtype=torch.long, device="cuda"))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        rtc0 = model(batch0, tau=tau, noise=noise)["flow_loss"]
    check("RTC delay 0 == plain", torch.allclose(ref, rtc0, rtol=2e-2, atol=1e-4), f"{rtc0.item():.5f}")
    prefix = torch.randn(B, 6, A, device="cuda").clamp(-1, 1)
    with torch.no_grad():
        s = model.sample_actions(batch, steps=5, prefix_actions=prefix, delay=6, generator=gen())
    check("RTC sample keeps the prefix", torch.equal(s[:, :6], prefix.float()))

    # MEM: with no valid past frame the encoder equals π0.5's
    model.cfg = base.with_(max_history=6)
    with torch.no_grad():
        frames = batch["images"][:, 0]                     # [B, 1, 3, H, W]
        hist = frames.expand(-1, 6, -1, -1, -1).contiguous()
        valid = torch.zeros(B, 6, dtype=torch.bool, device="cuda"); valid[:, -1] = True
        secs = torch.arange(5, -1, -1, device="cuda").float()[None].expand(B, -1)
        mem = model._mem_image(hist, secs, valid)
        one = pw.embed_image(frames[:, -1].float())
        valid[:] = True
        mem_all = model._mem_image(hist, secs, valid)
    check("MEM, no past frames == single frame", (mem - one).abs().max().item() < 1e-3, f"max diff {(mem - one).abs().max().item():.2e}")
    check("MEM, past frames change the tokens", (mem_all - one).abs().max().item() > 1e-3, f"max diff {(mem_all - one).abs().max().item():.2e}")

    # Knowledge insulation: flow loss gives no gradient to the VLM, CE does; CE at the start per tokenizer
    for fast in ("dataset", "universal"):
        cfg = base.with_(knowledge_insulation=True, fast_tokenizer=fast)
        model.cfg = cfg
        for layer in pw.gemma_expert.model.layers:
            layer.pi07_insulated = True
        model.addons = True
        kb = batch_for(cfg, args, tokenizer, n=2)
        model.set_gradient_checkpointing(True)
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(kb)
        out["flow_loss"].backward(retain_graph=True)
        vlm = [p for n, p in pw.paligemma.named_parameters() if p.requires_grad]
        vlm_grad = sum(float(p.grad.abs().sum()) for p in vlm if p.grad is not None)
        expert_grad = sum(float(p.grad.abs().sum()) for p in pw.gemma_expert.parameters() if p.grad is not None)
        check(f"KI ({fast}): flow loss leaves the VLM untouched", vlm_grad == 0.0 and expert_grad > 0,
              f"vlm {vlm_grad:.3g} expert {expert_grad:.3g}")
        model.zero_grad(set_to_none=True)
        out["ce_loss"].backward()
        vlm_grad = sum(float(p.grad.abs().sum()) for p in vlm if p.grad is not None)
        check(f"KI ({fast}): FAST loss trains the VLM", vlm_grad > 0, f"vlm {vlm_grad:.3g}")
        print(f"FAST {fast}: ce at start {out['ce_loss'].item():.3f}, token accuracy {out['ce_accuracy'].item():.3f}, "
              f"flow {out['flow_loss'].item():.4f}, fast tokens per chunk {kb['fast_valid'].sum(1).float().mean().item():.1f}")
        model.zero_grad(set_to_none=True)
        model.eval()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
