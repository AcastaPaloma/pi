"""MEM video history encoder on top of Gemma 3's SigLIP vision tower.

Every 4th SigLIP layer additively extends each patch's attention with the same
patch in earlier frames (space-time separable attention). The temporal keys and
values carry a sinusoidal time encoding e(t) with e(0) = 0, the projections are
the layer's own, and no new parameters are introduced, so a single frame (K=1)
reproduces the pretrained encoder exactly. After the last temporal layer the past
frames are dropped and only the current frame continues, so the backbone sees
the same number of tokens as for one image. Gemma 3's projector then average
pools the 32x32 patch grid to 16x16 = 256 tokens and projects to the backbone width.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import VisionConfig
from .layers import GemmaRMSNorm, sinusoidal_embedding


class SiglipAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.hidden // cfg.heads
        self.q_proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.k_proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.v_proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.out_proj = nn.Linear(cfg.hidden, cfg.hidden)

    def _heads(self, x):
        B, S, _ = x.shape
        return x.view(B, S, self.heads, self.head_dim).transpose(1, 2)

    def spatial(self, x):
        """x [N, P, D] -> [N, P, D], plain bidirectional attention within each image."""
        q, k, v = (self._heads(proj(x)) for proj in (self.q_proj, self.k_proj, self.v_proj))
        out = F.scaled_dot_product_attention(q, k, v)
        B, H, S, D = out.shape
        return self.out_proj(out.transpose(1, 2).reshape(B, S, H * D))

    def space_time(self, x, time_code, frame_valid):
        """x [B, K, P, D]; time_code [B, K, D] added to past keys/values; frame_valid [B, K] bool.

        Query at (frame k, patch p) attends to every patch of frame k plus patch p
        of every valid earlier frame j < k. One softmax spans both key sets.
        """
        B, K, P, D = x.shape
        x_flat = x.reshape(B * K, P, D)
        q = self._heads(self.q_proj(x_flat)).view(B, K, self.heads, P, self.head_dim)
        k_s = self._heads(self.k_proj(x_flat)).view(B, K, self.heads, P, self.head_dim)
        v_s = self._heads(self.v_proj(x_flat)).view(B, K, self.heads, P, self.head_dim)
        coded = (x + time_code.unsqueeze(2)).reshape(B * K, P, D)
        k_t = self._heads(self.k_proj(coded)).view(B, K, self.heads, P, self.head_dim)
        v_t = self._heads(self.v_proj(coded)).view(B, K, self.heads, P, self.head_dim)
        scale = self.head_dim ** -0.5
        logits_s = torch.einsum("bkhpd,bkhqd->bkhpq", q, k_s) * scale            # [B,K,H,P,P]
        logits_t = torch.einsum("bkhpd,bjhpd->bkhpj", q, k_t) * scale            # [B,K,H,P,K]
        causal = torch.tril(torch.ones(K, K, dtype=torch.bool, device=x.device), diagonal=-1)  # j < k
        allowed = causal[None, :, None, None, :] & frame_valid[:, None, None, None, :]
        logits_t = logits_t.masked_fill(~allowed, float("-inf"))
        joint = torch.cat([logits_s, logits_t], dim=-1)
        probs = torch.softmax(joint.float(), dim=-1).type_as(q)
        probs_s, probs_t = probs[..., :P], probs[..., P:]
        out = torch.einsum("bkhpq,bkhqd->bkhpd", probs_s, v_s) + torch.einsum("bkhpj,bjhpd->bkhpd", probs_t, v_t)
        out = out.permute(0, 1, 3, 2, 4).reshape(B * K, P, self.heads * self.head_dim)
        return self.out_proj(out).view(B, K, P, D)


class SiglipLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.self_attn = SiglipAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.mlp = nn.Sequential()
        self.mlp.fc1 = nn.Linear(cfg.hidden, cfg.mlp)
        self.mlp.fc2 = nn.Linear(cfg.mlp, cfg.hidden)

    def _mlp(self, x):
        return self.mlp.fc2(F.gelu(self.mlp.fc1(x), approximate="tanh"))

    def forward(self, x, *, time_code=None, frame_valid=None):
        if time_code is None:
            x = x + self.self_attn.spatial(self.layer_norm1(x))
        else:
            x = x + self.self_attn.space_time(self.layer_norm1(x), time_code, frame_valid)
        return x + self._mlp(self.layer_norm2(x))


class MultiModalProjector(nn.Module):
    """Gemma 3: average pool the patch grid, soft-token RMSNorm, linear to the text width."""

    def __init__(self, cfg: VisionConfig, text_hidden: int, rms_eps: float):
        super().__init__()
        self.pool, self.grid = cfg.pool, cfg.grid
        self.mm_soft_emb_norm = GemmaRMSNorm(cfg.hidden, rms_eps)
        self.mm_input_projection = nn.Linear(cfg.hidden, text_hidden, bias=False)

    def forward(self, tokens):
        B, P, D = tokens.shape
        grid = tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid)
        pooled = F.avg_pool2d(grid, self.pool, self.pool).flatten(2).transpose(1, 2)
        return self.mm_input_projection(self.mm_soft_emb_norm(pooled))


class MemVisionEncoder(nn.Module):
    def __init__(self, cfg: VisionConfig, text_hidden: int, rms_eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = nn.Conv2d(3, cfg.hidden, cfg.patch, stride=cfg.patch)
        self.position_embedding = nn.Embedding(cfg.patches, cfg.hidden)
        self.layers = nn.ModuleList(SiglipLayer(cfg) for _ in range(cfg.layers))
        self.post_layernorm = nn.LayerNorm(cfg.hidden, eps=cfg.layer_norm_eps)
        self.projector = MultiModalProjector(cfg, text_hidden, rms_eps)
        self.temporal = set(cfg.temporal_layers)
        self.gradient_checkpointing = False

    def time_code(self, seconds_ago):
        """e(t) for each frame, zero at t = 0 so the current frame is untouched."""
        code = sinusoidal_embedding(seconds_ago, self.cfg.hidden, self.cfg.time_min_period, self.cfg.time_max_period)
        zero = sinusoidal_embedding(torch.zeros_like(seconds_ago), self.cfg.hidden, self.cfg.time_min_period, self.cfg.time_max_period)
        return code - zero

    def embed(self, images):
        """images [N, 3, H, W] in [-1, 1] -> [N, P, D]."""
        x = self.patch_embedding(images.to(self.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
        return x + self.position_embedding.weight[None]

    def forward(self, images, seconds_ago=None, frame_valid=None):
        """images [B, K, 3, H, W] (K history frames, oldest first, current last) -> [B, T, text_hidden].

        seconds_ago [B, K] float; frame_valid [B, K] bool. With K = 1 the temporal
        machinery is inert and the output equals the single-image encoder.
        """
        B, K = images.shape[:2]
        if seconds_ago is None:
            seconds_ago = torch.zeros(B, K, device=images.device)
        if frame_valid is None:
            frame_valid = torch.ones(B, K, dtype=torch.bool, device=images.device)
        x = self.embed(images.flatten(0, 1)).view(B, K, self.cfg.patches, self.cfg.hidden)
        code = self.time_code(seconds_ago).to(x.dtype)
        current_only = False
        run = (lambda f, *a, **kw: checkpoint(f, *a, use_reentrant=False, **kw)) if (self.gradient_checkpointing and self.training) else (lambda f, *a, **kw: f(*a, **kw))
        for i, layer in enumerate(self.layers):
            if current_only:
                x = run(layer, x)
            elif i in self.temporal and K > 1:
                x = run(layer, x, time_code=code, frame_valid=frame_valid)
            else:
                x = run(layer, x.flatten(0, 1)).view(B, K, self.cfg.patches, self.cfg.hidden)
            if not current_only and i == self.cfg.drop_past_after:
                x = x[:, -1]
                current_only = True
        if not current_only:
            x = x[:, -1]
        return self.projector(self.post_layernorm(x))


def preprocess_images(uint8_images: torch.Tensor) -> torch.Tensor:
    """uint8 [..., H, W, 3] -> float [..., 3, H, W] in [-1, 1] (SigLIP normalization)."""
    x = uint8_images.float().div_(127.5).sub_(1.0)
    return x.movedim(-1, -3)
