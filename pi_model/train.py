"""Train π0.7 (low-level policy, or the high-level subtask policy with --objective subtask).

    python -m pi_model.train --data data/sim_pick_place --config full --gemma /weights/gemma-3-4b-pt \
        --tokenizer /weights/gemma-3-4b-pt/tokenizer.model --out runs/pi07_full --steps 30000 --batch-size 16

Multi-GPU (one process per GPU, batch size is per process):

    torchrun --nproc_per_node 2 -m pi_model.train --data ... --out ... --batch-size 32

The paper-size model does not fit one 80 GB GPU with fp32 master weights and
Adam states, so use --fsdp to shard parameters, gradients and optimizer states
across the GPUs (bf16 compute, fp32 reductions and master weights):

    torchrun --nproc_per_node 2 -m pi_model.train --config full --gemma ... --fsdp --gradient-checkpointing

Knowledge insulation: the flow loss updates the expert only; cross-entropy on FAST
tokens updates the VLM. Both are summed and stepped together.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .config import Pi07Config
from .data import Pi07Dataset, collate, to_device
from .model import Pi07, make_model
from .tokenizer import load_tokenizer
from .weights import gemma_loaded_fraction, load_gemma3, save_checkpoint


def pick_device(name: str):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step, *, base, warmup, total, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def reset_action_io(model):
    """Re-initialise the layers that map joint numbers in and out of the model (state input,
    action-expert input and output) for a warm start on data with new action statistics,
    keeping the vision encoder, backbone and expert transformer."""
    model.state_proj.reset_parameters()
    model.expert.action_in.reset_parameters()
    torch.nn.init.zeros_(model.expert.action_out.weight)
    torch.nn.init.zeros_(model.expert.action_out.bias)


def build_model(cfg, *, gemma=None, device=torch.device("cpu"), checkpoint=None, reset_io=False):
    report = None
    if cfg.backbone == "pi05":
        model = make_model(cfg, pretrained=checkpoint is None)
        if checkpoint is not None:
            model.load_state_dict(torch.load(Path(checkpoint) / "model.pt", map_location="cpu"))
        return model.to(device), None
    model = Pi07(cfg)
    if checkpoint is not None:
        model.load_state_dict(torch.load(Path(checkpoint) / "model.pt", map_location="cpu"))
        if reset_io:
            reset_action_io(model)
    elif gemma is not None:
        report = load_gemma3(model, gemma)
        report["vlm_fraction"] = gemma_loaded_fraction(report, model)
    return model.to(device), report


def parse_override(cfg, item):
    """'key=value' for a top-level config field, parsed to the field's current type."""
    key, sep, value = item.partition("=")
    if not sep or not hasattr(cfg, key) or key in ("vision", "text", "expert"):
        raise SystemExit(f"--set needs KEY=VALUE with a top-level config field, got {item!r}")
    current = getattr(cfg, key)
    if isinstance(current, bool):
        if value.lower() not in ("true", "false", "1", "0"):
            raise SystemExit(f"--set {key} needs true or false, got {value!r}")
        return key, value.lower() in ("true", "1")
    return key, type(current)(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Episode root from pi_sim.record")
    parser.add_argument("--meta", type=Path, help="prepare() output; default <data>/pi07_meta")
    parser.add_argument("--config", default="full", help="full | small | tiny, or a config.json path")
    parser.add_argument("--objective", choices=("action", "subtask"), default="action")
    parser.add_argument("--gemma", type=Path, help="Gemma 3 4B safetensors directory (initialises the VLM)")
    parser.add_argument("--tokenizer", type=Path, help="Gemma tokenizer.model; byte fallback when omitted")
    parser.add_argument("--resume", type=Path, help="Checkpoint directory to warm-start from (weights only)")
    parser.add_argument("--no-state-history", action="store_true", help="Feed only the current joint state")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="Keep an exponential moving average of the weights (e.g. 0.999); checkpoints store it as model.pt")
    parser.add_argument("--no-knowledge-insulation", action="store_true",
                        help="Let the flow loss train the backbone and vision encoder (no stop-gradient)")
    parser.add_argument("--state-dropout", type=float, help="Fraction of samples trained with no joint state (1 = never)")
    parser.add_argument("--reset-action-io", action="store_true",
                        help="With --resume: re-initialise state/action input-output layers (new action statistics)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA/MPS)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--image-size", type=int, help="Override the config image size (frames are resized)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--fsdp", action="store_true", help="Shard parameters/grads/optimizer across GPUs (FSDP2)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a top-level config field, e.g. --set max_history=6 --set pi05_rtc=true")
    args = parser.parse_args(argv)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, world = 0, 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)
    cfg = Pi07Config.from_json(Path(args.config).read_text()) if args.config.endswith(".json") else Pi07Config.named(args.config)
    if args.image_size:
        cfg = cfg.with_(vision=cfg.vision.__class__(**{**cfg.vision.__dict__, "image_size": args.image_size}))
    if args.no_state_history:
        cfg = cfg.with_(state_history=False)
    if args.state_dropout is not None:
        cfg = cfg.with_(state_dropout=args.state_dropout)
    if args.no_knowledge_insulation:
        cfg = cfg.with_(knowledge_insulation=False)
    for item in args.set:
        cfg = cfg.with_(**dict([parse_override(cfg, item)]))
    device = pick_device(args.device)
    meta_dir = args.meta or args.data / "pi07_meta"
    tokenizer = load_tokenizer(args.tokenizer, cfg.text.vocab)
    dataset = Pi07Dataset(args.data, meta_dir, cfg, tokenizer, train=True, objective=args.objective, seed=args.seed)
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed, drop_last=True) if distributed else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True,
                        num_workers=args.num_workers, collate_fn=lambda s: collate(s, cfg),
                        persistent_workers=args.num_workers > 0)
    model, report = build_model(cfg, gemma=args.gemma, device=device, checkpoint=args.resume,
                                reset_io=args.reset_action_io)
    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)
    core = model
    if args.fsdp:
        if not distributed:
            raise SystemExit("--fsdp needs torchrun with more than one process")
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in [*core.vision.layers, *core.backbone.layers, *core.expert.layers]:
            fully_shard(layer, mp_policy=policy)
        fully_shard(core, mp_policy=policy)
        model = core
    elif distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    params = core.parameter_report()
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.out / "config.json").write_text(cfg.to_json())
        (args.out / "run.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "parameters": params, "device": str(device), "tokenizer": tokenizer.name,
        "gemma": None if report is None else {"loaded": len(report["loaded"]), "skipped": len(report["skipped"]),
                                               "vlm_fraction": report["vlm_fraction"]},
        "dataset": {"episodes": len(dataset.store), "samples": len(dataset), "cameras": dataset.cameras,
                    "control_hz": dataset.store.hz}, "world_size": world}, indent=2))
        print(f"device={device} world={world} params={ {k: round(v / 1e6, 1) for k, v in params.items()} }M samples={len(dataset)} "
              f"tokenizer={tokenizer.name}" + (f" gemma_vlm_fraction={report['vlm_fraction']:.3f}" if report else ""), flush=True)

    groups = {}
    for name, p in model.named_parameters():
        vlm = cfg.vlm_lr_scale != 1.0 and ".paligemma_with_expert.paligemma." in f".{name}"
        wd = 0.0 if p.ndim < 2 or "norm" in name or "embedding" in name else args.weight_decay
        groups.setdefault((wd, vlm), []).append(p)
    optimizer = torch.optim.AdamW([{"params": ps, "weight_decay": wd, "lr_scale": cfg.vlm_lr_scale if vlm else 1.0}
                                   for (wd, vlm), ps in groups.items()], lr=args.lr, betas=(0.9, 0.95))
    ema = None
    if args.ema_decay > 0:
        if args.fsdp:
            raise SystemExit("--ema-decay is not supported with --fsdp")
        ema = {name: param.detach().clone().float() for name, param in core.named_parameters() if param.requires_grad}
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=args.bf16 and device.type != "cpu" and not args.fsdp)
    log = (args.out / "log.jsonl").open("a") if rank == 0 else None
    model.train()
    step, started, epoch = 0, time.monotonic(), 0
    running = {}
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for i, batch in enumerate(loader):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            batch = to_device(batch, device)
            with autocast:
                out = model(batch)
            (out["loss"] / args.grad_accum).backward()
            for k in ("loss", "ce_loss", "flow_loss", "ce_accuracy"):
                if k in out:
                    running[k] = running.get(k, 0.0) + float(out[k])
            if (i + 1) % args.grad_accum:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, base=args.lr, warmup=args.warmup, total=args.steps) * group["lr_scale"]
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if ema is not None:
                # Exponential moving average of the weights (H9): evaluated checkpoints oscillate a lot
                # between saves; π0-style policies are usually evaluated with averaged weights.
                d = min(args.ema_decay, (1 + step) / (10 + step))
                with torch.no_grad():
                    for name, param in core.named_parameters():
                        if name in ema:
                            ema[name].mul_(d).add_(param.detach().to(ema[name].dtype), alpha=1 - d)
            if step % args.log_every == 0 or step == 1:
                n = args.log_every * args.grad_accum if step > 1 else args.grad_accum
                averaged = {k: v / n for k, v in running.items()}
                if distributed:
                    for k in averaged:
                        tensor = torch.tensor(averaged[k], device=device)
                        dist.all_reduce(tensor)
                        averaged[k] = float(tensor) / world
                record = {"step": step, **averaged, "lr": max(g["lr"] for g in optimizer.param_groups),
                          "elapsed_s": round(time.monotonic() - started, 1),
                          "samples_per_s": round(step * args.batch_size * args.grad_accum * world / (time.monotonic() - started), 1)}
                running = {}
                if rank == 0:
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                state = None
                if args.fsdp:
                    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
                    state = get_model_state_dict(core, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
                if rank == 0:
                    ckpt = args.out / f"step_{step:07d}"
                    if ema is not None and state is None:
                        # model.pt holds the averaged weights (what evaluators load); model_raw.pt the raw ones.
                        raw = core.state_dict()
                        torch.save(raw, (ckpt.mkdir(parents=True, exist_ok=True), ckpt / "model_raw.pt")[1])
                        state = {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in raw.items()}
                    save_checkpoint(ckpt, core, config=cfg, state_dict=state,
                                    extra={"step": step, "objective": args.objective,
                                           "tokenizer": str(args.tokenizer) if args.tokenizer else None,
                                           "meta_dir": str(meta_dir)})
                    shutil.copy(meta_dir / "stats.json", ckpt / "stats.json")
                    print(f"saved {ckpt}", flush=True)
                if distributed:
                    dist.barrier()
            if step >= args.steps:
                break
    if log is not None:
        log.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
