"""Gemma 3 decoder: GQA with QK-norm, local/global RoPE, GeGLU, pre and post norms."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .config import TextConfig
from .layers import GatedMLP, GemmaAttention, GemmaRMSNorm, rope_cos_sin


class GemmaDecoderLayer(nn.Module):
    def __init__(self, cfg: TextConfig, index: int):
        super().__init__()
        self.sliding = cfg.is_sliding(index)
        self.input_layernorm = GemmaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.self_attn = GemmaAttention(cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim,
                                        cfg.query_pre_attn_scalar, cfg.rms_eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.mlp = GatedMLP(cfg.hidden, cfg.mlp)
        self.post_feedforward_layernorm = GemmaRMSNorm(cfg.hidden, cfg.rms_eps)

    def forward(self, x, cos, sin, mask):
        """Returns the new hidden state and this layer's keys/values (for the action expert)."""
        q, k, v = self.self_attn.qkv(self.input_layernorm(x), cos, sin)
        x = x + self.post_attention_layernorm(self.self_attn.output(q, k, v, mask))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x, k, v


class GemmaBackbone(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab, cfg.hidden)
        self.layers = nn.ModuleList(GemmaDecoderLayer(cfg, i) for i in range(cfg.layers))
        self.norm = GemmaRMSNorm(cfg.hidden, cfg.rms_eps)
        self.gradient_checkpointing = False
        nn.init.normal_(self.embed_tokens.weight, std=0.02)

    def embed(self, ids):
        return self.embed_tokens(ids) * math.sqrt(self.cfg.hidden)

    def rope(self, positions):
        """(cos, sin) tables for local and global layers."""
        local = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_local_theta)
        global_ = rope_cos_sin(positions, self.cfg.head_dim, self.cfg.rope_theta, self.cfg.rope_scaling)
        return local, global_

    def logits(self, hidden):
        """Tied output head over the final-normed hidden state."""
        return hidden @ self.embed_tokens.weight.t()

    def forward(self, embeds, positions, mask_local, mask_global):
        local, global_ = self.rope(positions)
        x = embeds
        kvs = []
        for layer in self.layers:
            cos, sin = local if layer.sliding else global_
            mask = mask_local if layer.sliding else mask_global
            if self.gradient_checkpointing and self.training:
                x, k, v = checkpoint(layer, x, cos, sin, mask, use_reentrant=False)
            else:
                x, k, v = layer(x, cos, sin, mask)
            kvs.append((k, v))
        return self.norm(x), kvs
