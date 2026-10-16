"""π0.7: prefix (images, subgoals, text, states, FAST tokens) through Gemma 3, flow expert on top.

Sequence layout and attention (paper Sec. VI-B, Fig. 19):
  observation tokens      bidirectional among themselves
  subgoal image tokens    bidirectional among themselves, may attend observations
  text prompt             causal; attends observations and subgoals
  proprioceptive states   one token per history frame, continue the causal stream
  FAST action tokens      training only; causal; never attend the flow actions
  flow action tokens      50 tokens in the expert; bidirectional among themselves,
                          attend the whole prefix except the FAST tokens

Knowledge insulation: the expert attends to the backbone's keys/values through a
stop-gradient, so the flow loss never updates the VLM; the VLM is trained with
cross-entropy on the FAST tokens (and, for the high-level policy, on subtask text).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import GemmaBackbone
from .config import Pi07Config
from .expert import ActionExpert
from .vision import MemVisionEncoder

OBS, SUBGOAL, TEXT, STATE, FAST, ACTION = range(6)
# Which key segments each query segment may attend.
RULE = torch.tensor([
    [1, 0, 0, 0, 0, 0],   # OBS
    [1, 1, 0, 0, 0, 0],   # SUBGOAL
    [1, 1, 1, 0, 0, 0],   # TEXT
    [1, 1, 1, 1, 0, 0],   # STATE
    [1, 1, 1, 1, 1, 0],   # FAST
    [1, 1, 1, 1, 0, 1],   # ACTION
], dtype=torch.bool)
BIDIRECTIONAL_KEYS = torch.tensor([True, True, False, False, False, True])


def attention_mask(seg_q, pos_q, seg_k, pos_k, valid_k, window=None):
    """Boolean [B, 1, Sq, Sk]: segment rule, causality for causal key segments, key validity, window."""
    rule = RULE.to(seg_q.device)[seg_q[:, :, None], seg_k[:, None, :]]
    causal = pos_k[:, None, :] <= pos_q[:, :, None]
    bidir = BIDIRECTIONAL_KEYS.to(seg_q.device)[seg_k][:, None, :]
    allowed = rule & valid_k[:, None, :] & (causal | bidir)
    if window is not None:
        allowed &= (pos_q[:, :, None] - pos_k[:, None, :]).abs() < window
    return allowed.unsqueeze(1)


@dataclass
class Prefix:
    embeds: torch.Tensor      # [B, N, D]
    valid: torch.Tensor       # [B, N] bool
    segment: torch.Tensor     # [B, N] int
    positions: torch.Tensor   # [B, N] int
    ids: torch.Tensor         # [B, N] token ids where applicable (else -1)


class Pi07(nn.Module):
    def __init__(self, cfg: Pi07Config):
        super().__init__()
        self.cfg = cfg
        self.vision = MemVisionEncoder(cfg.vision, cfg.text.hidden, cfg.text.rms_eps)
        self.backbone = GemmaBackbone(cfg.text)
        self.state_proj = nn.Linear(cfg.state_dim, cfg.text.hidden)
        self.expert = ActionExpert(cfg.expert, cfg.text, cfg.action_dim)
        self.register_buffer("_device_probe", torch.zeros(1), persistent=False)

    @property
    def device(self):
        return self._device_probe.device

    def set_gradient_checkpointing(self, enabled: bool = True):
        for module in (self.vision, self.backbone, self.expert):
            module.gradient_checkpointing = enabled

    # ----------------------------------------------------------------- prefix
    def encode_prefix(self, batch) -> Prefix:
        """Assemble the backbone input sequence from a batch dict (see data.py for keys)."""
        D = self.cfg.text.hidden
        images = batch["images"]                       # [B, V, K, 3, H, W] float in [-1, 1]
        B, V, K = images.shape[:3]
        frame_valid = batch["frame_valid"]             # [B, K] bool, shared across cameras
        seconds_ago = batch["seconds_ago"]             # [B, K] float
        camera_valid = batch["camera_valid"]           # [B, V] bool
        T = self.cfg.vision.tokens_per_image
        obs = self.vision(images.flatten(0, 1), seconds_ago.repeat_interleave(V, 0), frame_valid.repeat_interleave(V, 0))
        obs = obs.view(B, V * T, D)
        obs_valid = camera_valid[:, :, None].expand(B, V, T).reshape(B, V * T)
        pieces = [(obs, obs_valid, OBS, None)]

        subgoals = batch.get("subgoals")               # [B, G, 3, H, W]
        if subgoals is not None and subgoals.shape[1] > 0:
            G = subgoals.shape[1]
            goal = self.vision(subgoals.flatten(0, 1).unsqueeze(1)).view(B, G * T, D)
            goal_valid = batch["subgoal_valid"][:, :, None].expand(B, G, T).reshape(B, G * T)
            pieces.append((goal, goal_valid, SUBGOAL, None))

        text_ids = batch["text_ids"]
        pieces.append((self.backbone.embed(text_ids), batch["text_valid"], TEXT, text_ids))

        state = batch.get("state")                     # [B, K, state_dim]
        if state is not None and state.shape[1] > 0:
            pieces.append((self.state_proj(state.to(self.state_proj.weight.dtype)), batch["state_valid"], STATE, None))

        fast_ids = batch.get("fast_ids")
        if fast_ids is not None and fast_ids.shape[1] > 0:
            pieces.append((self.backbone.embed(fast_ids), batch["fast_valid"], FAST, fast_ids))

        embeds = torch.cat([p[0] for p in pieces], dim=1)
        valid = torch.cat([p[1] for p in pieces], dim=1)
        segment = torch.cat([torch.full(p[1].shape, p[2], dtype=torch.long, device=embeds.device) for p in pieces], dim=1)
        ids = torch.cat([p[3] if p[3] is not None else torch.full(p[1].shape, -1, dtype=torch.long, device=embeds.device)
                         for p in pieces], dim=1)
        positions = torch.arange(embeds.shape[1], device=embeds.device).expand(B, -1)
        embeds = embeds * valid.unsqueeze(-1).to(embeds.dtype)
        return Prefix(embeds, valid, segment, positions, ids)

    def run_prefix(self, prefix: Prefix):
        window = self.cfg.text.sliding_window
        mask_local = attention_mask(prefix.segment, prefix.positions, prefix.segment, prefix.positions, prefix.valid, window)
        mask_global = attention_mask(prefix.segment, prefix.positions, prefix.segment, prefix.positions, prefix.valid)
        hidden, kvs = self.backbone(prefix.embeds, prefix.positions, mask_local, mask_global)
        return hidden, kvs

    # ----------------------------------------------------------------- expert
    def expert_masks(self, prefix: Prefix):
        B, N = prefix.valid.shape
        H = self.cfg.action_horizon
        seg_a = torch.full((B, H), ACTION, dtype=torch.long, device=prefix.valid.device)
        pos_a = N + torch.arange(H, device=prefix.valid.device).expand(B, -1)
        seg_k = torch.cat([prefix.segment, seg_a], dim=1)
        pos_k = torch.cat([prefix.positions, pos_a], dim=1)
        valid_k = torch.cat([prefix.valid, torch.ones_like(seg_a, dtype=torch.bool)], dim=1)
        window = self.cfg.text.sliding_window
        local = attention_mask(seg_a, pos_a, seg_k, pos_k, valid_k, window)
        global_ = attention_mask(seg_a, pos_a, seg_k, pos_k, valid_k)
        return pos_a, local, global_

    def predict_velocity(self, prefix: Prefix, kvs, noisy_actions, tau):
        """Expert velocity for noisy actions [B, H, A] at per-token tau [B, H]; prefix KV are detached (KI)."""
        pos_a, mask_local, mask_global = self.expert_masks(prefix)
        rope_local, rope_global = self.backbone.rope(pos_a)
        insulated = [(k.detach(), v.detach()) for k, v in kvs] if self.cfg.knowledge_insulation else list(kvs)
        sliding = [layer.sliding for layer in self.backbone.layers]
        return self.expert(noisy_actions, tau, rope_local, rope_global, insulated, mask_local, mask_global, sliding)

    # --------------------------------------------------------------- training
    @staticmethod
    def sample_tau(B, device, generator=None):
        """Flow timestep with tau = 1 clean, tau = 0 noise; emphasises noisy steps like π0."""
        beta = torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        noise_level = beta.sample((B,)).to(device) * 0.999 + 0.001
        return 1.0 - noise_level

    def forward(self, batch, *, tau=None, noise=None):
        """Training step. Returns loss terms; `batch` keys are documented in data.py.

        Flow matching: x_tau = tau * a + (1 - tau) * eps, target velocity a - eps.
        Training-time RTC: the first `rtc_delay` tokens are the clean prefix with tau = 1
        and are excluded from the loss.
        """
        prefix = self.encode_prefix(batch)
        hidden, kvs = self.run_prefix(prefix)
        out = {}

        lm_mask = batch.get("lm_target_mask")          # [B, N] bool: positions whose next token is supervised
        if lm_mask is not None and lm_mask.any():
            targets = batch["lm_targets"]                 # [B, N] long (ignored where mask is False)
            rows = hidden[lm_mask]
            logits = self.backbone.logits(rows).float()
            out["ce_loss"] = F.cross_entropy(logits, targets[lm_mask])
            out["ce_accuracy"] = (logits.argmax(-1) == targets[lm_mask]).float().mean()
        else:
            out["ce_loss"] = hidden.new_zeros(())

        actions = batch.get("actions")                 # [B, H, A] normalized
        if actions is not None:
            B, H, A = actions.shape
            if tau is None:
                tau = self.sample_tau(B, actions.device)
            if noise is None:
                noise = torch.randn_like(actions)
            delay = batch.get("rtc_delay", torch.zeros(B, dtype=torch.long, device=actions.device))
            is_prefix = torch.arange(H, device=actions.device)[None, :] < delay[:, None]
            tau_tok = torch.where(is_prefix, torch.ones_like(tau)[:, None].expand(B, H), tau[:, None].expand(B, H))
            noisy = tau_tok[..., None] * actions + (1 - tau_tok[..., None]) * noise
            target = actions - noise
            pred = self.predict_velocity(prefix, kvs, noisy, tau_tok)
            loss_mask = (~is_prefix)[..., None] & batch["action_valid"][:, None, :].expand(B, H, A)
            if "action_step_valid" in batch:
                loss_mask &= batch["action_step_valid"][..., None]
            err = (pred - target).pow(2) * loss_mask
            out["flow_loss"] = err.sum() / loss_mask.sum().clamp(min=1)
        else:
            out["flow_loss"] = hidden.new_zeros(())
        out["loss"] = out["ce_loss"] + out["flow_loss"]
        return out

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def sample_actions(self, batch, *, prefix_actions=None, delay=0, steps=None, cfg_beta=None, uncond_batch=None,
                       generator=None):
        """Euler flow sampling of a normalized action chunk [B, H, A].

        prefix_actions [B, d, A]: already-committed actions executed during inference
        (training-time RTC); they stay fixed with tau = 1. cfg_beta applies
        classifier-free guidance against `uncond_batch`: v = v_c + beta (v_c - v_u).
        """
        steps = steps or self.cfg.denoise_steps
        prefix = self.encode_prefix(batch)
        _, kvs = self.run_prefix(prefix)
        if cfg_beta is not None:
            if uncond_batch is None:
                raise ValueError("cfg_beta needs uncond_batch")
            uprefix = self.encode_prefix(uncond_batch)
            _, ukvs = self.run_prefix(uprefix)
        B = prefix.valid.shape[0]
        H, A = self.cfg.action_horizon, self.cfg.action_dim
        x = torch.randn(B, H, A, device=self.device, generator=generator)
        is_prefix = torch.arange(H, device=self.device)[None, :] < delay
        if delay:
            if prefix_actions is None or prefix_actions.shape[1] < delay:
                raise ValueError("prefix_actions must cover the delay")
            x[:, :delay] = prefix_actions[:, :delay]
        dt = 1.0 / steps
        for s in range(steps):
            tau = torch.full((B, H), s * dt, device=self.device)
            tau = torch.where(is_prefix.expand(B, H), torch.ones_like(tau), tau)
            v = self.predict_velocity(prefix, kvs, x, tau)
            if cfg_beta is not None:
                vu = self.predict_velocity(uprefix, ukvs, x, tau)
                v = v + cfg_beta * (v - vu)
            x = torch.where(is_prefix.expand(B, H)[..., None], x, x + dt * v)
        return x

    @torch.no_grad()
    def generate_text(self, batch, *, max_new_tokens=32, eos_id=1):
        """Greedy decoding of text after the prompt (high-level policy: next subtask).

        The batch must end with the TEXT segment (no state or FAST tokens). Each
        step re-runs the prefix; fine for short outputs.
        """
        text_ids = batch["text_ids"].clone()
        text_valid = batch["text_valid"].clone()
        B = text_ids.shape[0]
        generated = [[] for _ in range(B)]
        finished = torch.zeros(B, dtype=torch.bool, device=text_ids.device)
        for _ in range(max_new_tokens):
            working = dict(batch, text_ids=text_ids, text_valid=text_valid, state=None, fast_ids=None)
            prefix = self.encode_prefix(working)
            hidden, _ = self.run_prefix(prefix)
            last = text_valid.sum(1) - 1
            offset = prefix.valid.shape[1] - text_ids.shape[1]
            rows = hidden[torch.arange(B), offset + last]
            next_ids = self.backbone.logits(rows).argmax(-1)
            for b in range(B):
                if not finished[b]:
                    generated[b].append(int(next_ids[b]))
                    if int(next_ids[b]) == eos_id:
                        finished[b] = True
            if finished.all():
                break
            text_ids = torch.cat([text_ids, next_ids[:, None]], dim=1)
            text_valid = torch.cat([text_valid, ~finished[:, None]], dim=1)
        return generated

    # ------------------------------------------------------------------ util
    def parameter_report(self):
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in (("vision", self.vision), ("backbone", self.backbone),
                                       ("state_proj", self.state_proj), ("expert", self.expert))}
        counts["total"] = sum(counts.values())
        return counts


def make_model(cfg: Pi07Config, *, pretrained: bool = False):
    """Build the model for `cfg.backbone`; pretrained loads π0.5 weights for the pi05 backbone."""
    if cfg.backbone == "pi05":
        from .pi05_backbone import Pi05Backbone
        return Pi05Backbone(cfg, pretrained=pretrained)
    return Pi07(cfg)
