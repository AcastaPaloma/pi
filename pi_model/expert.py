"""The 860M flow-matching action expert: one layer per backbone layer, adaptive RMSNorm."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import ExpertConfig, TextConfig
from .layers import AdaRMSNorm, GatedMLP, GemmaAttention, GemmaRMSNorm, TimestepEmbedding


class ExpertLayer(nn.Module):
    def __init__(self, cfg: ExpertConfig, text: TextConfig):
        super().__init__()
        self.input_layernorm = AdaRMSNorm(cfg.hidden, cfg.time_dim, text.rms_eps)
        self.self_attn = GemmaAttention(cfg.hidden, text.heads, text.kv_heads, text.head_dim,
                                        text.query_pre_attn_scalar, text.rms_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden, text.rms_eps)
        self.pre_feedforward_layernorm = AdaRMSNorm(cfg.hidden, cfg.time_dim, text.rms_eps)
        self.mlp = GatedMLP(cfg.hidden, cfg.mlp)
        self.post_feedforward_layernorm = GemmaRMSNorm(cfg.hidden, text.rms_eps)

    def forward(self, x, cond, cos, sin, prefix_k, prefix_v, mask):
        q, k, v = self.self_attn.qkv(self.input_layernorm(x, cond), cos, sin)
        keys = torch.cat([prefix_k, k], dim=2)
        values = torch.cat([prefix_v, v], dim=2)
        x = x + self.post_attention_layernorm(self.self_attn.output(q, keys, values, mask))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x, cond)))
        return x


class ActionExpert(nn.Module):
    def __init__(self, cfg: ExpertConfig, text: TextConfig, action_dim: int):
        super().__init__()
        self.cfg = cfg
        self.action_in = nn.Linear(action_dim, cfg.hidden)
        self.time = TimestepEmbedding(cfg.time_dim, cfg.time_min_period, cfg.time_max_period)
        self.layers = nn.ModuleList(ExpertLayer(cfg, text) for _ in range(text.layers))
        self.norm = AdaRMSNorm(cfg.hidden, cfg.time_dim, text.rms_eps)
        self.action_out = nn.Linear(cfg.hidden, action_dim)
        self.gradient_checkpointing = False
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_actions, tau, rope_local, rope_global, prefix_kvs, mask_local, mask_global, sliding):
        """noisy_actions [B, H, A]; tau [B, H] per-token flow timesteps; returns velocity [B, H, A]."""
        cond = self.time(tau)
        x = self.action_in(noisy_actions)
        for layer, (pk, pv), is_sliding in zip(self.layers, prefix_kvs, sliding):
            cos, sin = rope_local if is_sliding else rope_global
            mask = mask_local if is_sliding else mask_global
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, cond, cos, sin, pk, pv, mask, use_reentrant=False)
            else:
                x = layer(x, cond, cos, sin, pk, pv, mask)
        return self.action_out(self.norm(x, cond))
