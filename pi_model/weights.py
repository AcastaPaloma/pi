"""Load Gemma 3 4B weights (HF safetensors) into the vision tower, projector and backbone.

Handles both transformers naming schemes (`language_model.model.layers...` and
`model.language_model.layers...`). The vision position embedding is interpolated
from Gemma's 64x64 grid (896 px) to this model's grid (32x32 at 448 px). The action
expert, state projection and anything else not in Gemma stay randomly initialised,
as in the paper's "initially untrained" expert.
"""
from __future__ import annotations

from pathlib import Path
import json

import torch
import torch.nn.functional as F


def _normalize(name: str) -> str:
    if name.startswith("model."):
        name = name[len("model."):]
    return name.replace("language_model.model.", "language_model.")


def map_name(hf_name: str):
    """HF parameter name -> (our name, transform) or None to skip."""
    n = _normalize(hf_name)
    if n.startswith("vision_tower.vision_model.embeddings."):
        return "vision." + n[len("vision_tower.vision_model.embeddings."):], None
    if n.startswith("vision_tower.vision_model.encoder.layers."):
        return "vision.layers." + n[len("vision_tower.vision_model.encoder.layers."):], None
    if n.startswith("vision_tower.vision_model.post_layernorm."):
        return "vision.post_layernorm." + n[len("vision_tower.vision_model.post_layernorm."):], None
    if n == "multi_modal_projector.mm_input_projection_weight":
        return "vision.projector.mm_input_projection.weight", "transpose"
    if n.startswith("multi_modal_projector.mm_soft_emb_norm."):
        return "vision.projector.mm_soft_emb_norm." + n[len("multi_modal_projector.mm_soft_emb_norm."):], None
    if n.startswith("language_model.embed_tokens."):
        return "backbone.embed_tokens." + n[len("language_model.embed_tokens."):], None
    if n.startswith("language_model.layers."):
        return "backbone.layers." + n[len("language_model.layers."):], None
    if n.startswith("language_model.norm."):
        return "backbone.norm." + n[len("language_model.norm."):], None
    if n.endswith("lm_head.weight"):
        return None  # tied to the embedding
    return None


def _resize_position_embedding(weight: torch.Tensor, target_patches: int) -> torch.Tensor:
    source = int(round(weight.shape[0] ** 0.5))
    target = int(round(target_patches ** 0.5))
    if source == target:
        return weight
    grid = weight.t().reshape(1, weight.shape[1], source, source)
    resized = F.interpolate(grid.float(), size=(target, target), mode="bicubic", align_corners=False)
    return resized.reshape(weight.shape[1], target * target).t().to(weight.dtype)


def iter_safetensors(path):
    """Yield (name, tensor) from a safetensors file or a directory of shards."""
    from safetensors import safe_open
    path = Path(path).expanduser()
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors under {path}")
    for file in files:
        with safe_open(str(file), framework="pt") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def load_gemma3(model, source, *, strict_shapes=True):
    """Copy Gemma 3 tensors into `model`; `source` is a path or an iterable of (name, tensor).

    Returns a report with loaded/skipped names and the model parameters left untouched.
    """
    tensors = iter_safetensors(source) if isinstance(source, (str, Path)) else source
    state = model.state_dict()
    loaded, skipped = [], []
    with torch.no_grad():
        for hf_name, tensor in tensors:
            mapped = map_name(hf_name)
            if mapped is None:
                skipped.append(hf_name)
                continue
            ours, transform = mapped
            if ours not in state:
                skipped.append(hf_name)
                continue
            if transform == "transpose":
                tensor = tensor.t()
            if ours == "vision.position_embedding.weight":
                tensor = _resize_position_embedding(tensor, state[ours].shape[0])
            if tensor.shape != state[ours].shape:
                message = f"{hf_name} -> {ours}: shape {tuple(tensor.shape)} vs {tuple(state[ours].shape)}"
                if strict_shapes:
                    raise ValueError(message)
                skipped.append(message)
                continue
            state[ours].copy_(tensor.to(state[ours].dtype))
            loaded.append(ours)
    untouched = sorted(set(state) - set(loaded))
    return {"loaded": loaded, "skipped": skipped, "untouched": untouched}


def gemma_loaded_fraction(report, model):
    """Share of vision+backbone parameters that came from Gemma (1.0 means the whole VLM)."""
    state = model.state_dict()
    vlm = [k for k in state if k.startswith(("vision.", "backbone."))]
    total = sum(state[k].numel() for k in vlm)
    got = sum(state[k].numel() for k in report["loaded"] if k in state)
    return got / total if total else 0.0


def save_checkpoint(path, model, *, config, extra=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    (path / "config.json").write_text(config.to_json())
    if extra:
        (path / "extra.json").write_text(json.dumps(extra, indent=2))
