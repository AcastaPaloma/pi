"""Shared building blocks: Gemma RMSNorm, RoPE, GQA attention, gated MLP, adaptive norm."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GemmaRMSNorm(nn.Module):
    """Gemma's RMSNorm: float32 statistics and a (1 + weight) scale, zero-initialised."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()) * (1.0 + self.weight.float())
        return out.type_as(x)


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale and shift are modulated per token by a conditioning vector.

    Used by the action expert to inject the flow-matching timestep ("adaptive
    RMSNorm"). The modulation projection is zero-initialised, so a fresh layer
    is an ordinary Gemma RMSNorm. Per-token conditioning is what lets the
    training-time RTC prefix carry a different timestep from the postfix.
    """

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = GemmaRMSNorm(dim, eps)
        self.modulation = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, cond):
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


def sinusoidal_embedding(t: torch.Tensor, dim: int, min_period: float, max_period: float) -> torch.Tensor:
    """Sine/cosine features of a scalar per element; `t` has any shape, output appends `dim`."""
    if dim % 2:
        raise ValueError("embedding dim must be even")
    half = dim // 2
    fraction = torch.linspace(0, 1, half, device=t.device, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    angle = t.float().unsqueeze(-1) / period * (2 * math.pi)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Flow timestep tau in [0, 1] -> conditioning vector, as in π0's action expert."""

    def __init__(self, dim: int, min_period: float, max_period: float):
        super().__init__()
        self.dim, self.min_period, self.max_period = dim, min_period, max_period
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, tau):
        return self.mlp(sinusoidal_embedding(tau, self.dim, self.min_period, self.max_period))


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, scaling: float = 1.0):
    """Rotary tables for integer positions [B, S] -> cos/sin [B, 1, S, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32) / head_dim))
    inv_freq = inv_freq / scaling
    freqs = positions.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, S, D]; HF Gemma 'rotate half' convention."""
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return (x.float() * cos + rotated.float() * sin).type_as(x)


def attend(q, k, v, mask, scale):
    """Grouped-query attention. q [B,H,Sq,D], k/v [B,Hkv,Sk,D], mask [B,1,Sq,Sk] bool (True = may attend).

    Rows with no permitted key return zeros rather than NaN, so fully masked
    (dropped) tokens stay finite.
    """
    groups = q.shape[1] // k.shape[1]
    if groups > 1:
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
    any_key = mask.any(dim=-1, keepdim=True)
    safe_mask = mask | ~any_key
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=safe_mask, scale=scale)
    return out * any_key.to(out.dtype)


class GatedMLP(nn.Module):
    """Gemma GeGLU: down(gelu_tanh(gate(x)) * up(x))."""

    def __init__(self, hidden: int, mlp: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, mlp, bias=False)
        self.up_proj = nn.Linear(hidden, mlp, bias=False)
        self.down_proj = nn.Linear(mlp, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaAttention(nn.Module):
    """Gemma 3 attention projections with QK-norm. Keys/values are exposed so the
    action expert can attend to this layer's prefix keys and values."""

    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, query_pre_attn_scalar: int, rms_eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim, rms_eps)
        self.k_norm = GemmaRMSNorm(head_dim, rms_eps)
        self.scale = query_pre_attn_scalar ** -0.5

    def qkv(self, x, cos, sin):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        return q, k, v

    def output(self, q, k, v, mask):
        out = attend(q, k, v, mask, self.scale)
        B, H, S, D = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(B, S, H * D))
