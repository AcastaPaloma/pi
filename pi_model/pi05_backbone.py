"""π0.7 recipe on π0.5's pretrained VLM and action expert (LeRobot port of openpi `pi05_base`).

Why: with our small scripted datasets, a backbone trained from Gemma 3's image-text weights never
learned to read object positions from the cameras (training losses are identical with shuffled
images). π0.5's PaliGemma backbone and Gemma-300M flow expert were trained on 10k+ hours of robot
data and already ground actions in images.

What is kept from π0.7 here: our data, the prompt metadata (task, subtask, speed, quality, mistake,
control mode), the three-camera rig, and the evaluation stack. What follows π0.5 instead: PaliGemma
(SigLIP So400m/14 at 224 px, Gemma 2B), its 300M action expert with adaptive RMSNorm time
conditioning, the current frame only (no MEM history), the joint state written into the prompt as
256-bin integers, openpi's flow convention (x_t = t·noise + (1−t)·actions, t = 1 is noise), and
gradients from the flow loss into the VLM (LeRobot's fine-tuning recipe; no FAST loss, no stop-gradient).

π0.7 add-ons, each behind a config switch and off by default (with all off the model runs LeRobot's own code):
- `pi05_rtc`: training-time RTC. The first d ~ U{0..12} action tokens are the clean committed actions (flow
  time 0, no loss), which needs a flow time per action token, so the adaptive RMSNorm accepts [B, T, D].
- `max_history` > 1: MEM. Every 4th SigLIP layer lets each patch also attend the same patch in up to
  K - 1 earlier frames (1 s apart, sinusoidal time code that is zero for the current frame, the layer's own
  projections, no new parameters); past frames are dropped after the last temporal layer, so the VLM still
  sees 256 tokens per camera and K = 1 is exactly π0.5's encoder.
- `max_subgoals` > 0: goal images (future frames in training) as extra image tokens in the prefix.
- `knowledge_insulation`: FAST action tokens follow the prompt and train the VLM by cross-entropy; the action
  expert reads the VLM's keys and values through a stop-gradient and never attends the FAST tokens.

Needs `lerobot[pi]` (see scripts/box/train_pi05.sh); weights download from `lerobot/pi05_base`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import sinusoidal_embedding

PI05_REPO = "lerobot/pi05_base"


def pi05_state_text(norm_state: np.ndarray | None, max_state_dim: int = 32) -> str | None:
    """openpi's state string: normalized state in [-1, 1], padded to 32 dims, 256 bins."""
    if norm_state is None:
        return None
    s = np.zeros(max_state_dim, dtype=np.float32)
    s[: len(norm_state)] = np.clip(norm_state, -1, 1)
    bins = np.digitize(s, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    return " ".join(map(str, bins.tolist()))


def pi05_prompt(task_text: str, state_text: str | None) -> str:
    """π0.5's prompt layout with our π0.7 metadata as the task text."""
    cleaned = task_text.strip().replace("_", " ").replace("\n", " ")
    if cleaned.startswith("Task: "):  # our metadata text already opens with "Task:"
        cleaned = cleaned[len("Task: "):]
    if state_text is None:
        return f"Task: {cleaned};\nAction: "
    return f"Task: {cleaned}, State: {state_text};\nAction: "


def _pi05_config():
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(PI05_REPO)
    cfg.chunk_size = 50
    cfg.n_action_steps = 50
    cfg.max_action_dim = 32
    cfg.max_state_dim = 32
    cfg.dtype = "bfloat16"
    cfg.compile_model = False
    return cfg


def _install_lerobot_patches():
    """Per-token adaptive RMSNorm (RTC) and an insulated joint layer (knowledge insulation); idempotent."""
    from lerobot.policies import pi_gemma
    from lerobot.policies.pi05 import modeling_pi05 as mp
    if getattr(mp, "_pi07_patched", False):
        return
    base_norm = pi_gemma.PiGemmaRMSNorm.forward

    def norm_forward(self, x, cond=None):
        if cond is not None and self.dense is not None and cond.dim() == 3 and x.dim() == 3:
            scale, shift, gate = self.dense(cond).chunk(3, dim=-1)  # one modulation per token
            normed = self._norm(x) * (1 + scale.float()) + shift.float()
            return normed.to(x.dtype), gate.to(x.dtype)
        return base_norm(self, x, cond)

    base_layer = mp.compute_layer_complete

    def layer_forward(inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
        if getattr(layers[1], "pi07_insulated", False):
            return _insulated_layer(mp, inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb)
        return base_layer(inputs_embeds, attention_mask, position_ids, adarms_cond, layers=layers, rotary_emb=rotary_emb)

    pi_gemma.PiGemmaRMSNorm.forward = norm_forward
    mp.compute_layer_complete = layer_forward
    mp._pi07_patched = True


def _insulated_layer(mp, inputs_embeds, attention_mask, position_ids, adarms_cond, layers, rotary_emb):
    """LeRobot's compute_layer_complete, except that action-token queries see the prefix keys and values through a
    stop-gradient (the prefix never attends the action tokens, so its own attention is unchanged)."""
    from transformers.models.gemma import modeling_gemma
    qs, ks, vs, gates = [], [], [], []
    for i, hidden in enumerate(inputs_embeds):
        layer = layers[i]
        hidden, gate = mp.layernorm_forward(layer.input_layernorm, hidden, adarms_cond[i])
        gates.append(gate)
        shape = (*hidden.shape[:-1], -1, layer.self_attn.head_dim)
        qs.append(layer.self_attn.q_proj(hidden).view(shape).transpose(1, 2))
        ks.append(layer.self_attn.k_proj(hidden).view(shape).transpose(1, 2))
        vs.append(layer.self_attn.v_proj(hidden).view(shape).transpose(1, 2))
    q, k, v = torch.cat(qs, dim=2), torch.cat(ks, dim=2), torch.cat(vs, dim=2)
    dummy = torch.zeros(q.shape[0], q.shape[2], q.shape[-1], device=q.device, dtype=q.dtype)
    cos, sin = rotary_emb(dummy, position_ids)
    q, k = modeling_gemma.apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    P = inputs_embeds[0].shape[1]
    attn = layers[0].self_attn
    out_p, _ = modeling_gemma.eager_attention_forward(attn, q[:, :, :P], k[:, :, :P], v[:, :, :P],
                                                      attention_mask[:, :, :P, :P], attn.scaling)
    k_s = torch.cat([k[:, :, :P].detach(), k[:, :, P:]], dim=2)
    v_s = torch.cat([v[:, :, :P].detach(), v[:, :, P:]], dim=2)
    out_s, _ = modeling_gemma.eager_attention_forward(attn, q[:, :, P:], k_s, v_s, attention_mask[:, :, P:, :],
                                                      attn.scaling)
    att = torch.cat([out_p, out_s], dim=1)
    att = att.reshape(att.shape[0], att.shape[1], -1)
    outputs, start = [], 0
    for i, hidden in enumerate(inputs_embeds):
        layer = layers[i]
        end = start + hidden.shape[1]
        a = att[:, start:end]
        if a.dtype != layer.self_attn.o_proj.weight.dtype:
            a = a.to(layer.self_attn.o_proj.weight.dtype)
        out = mp._gated_residual(hidden, layer.self_attn.o_proj(a), gates[i])
        after_first = out.clone()
        out, gate = mp.layernorm_forward(layer.post_attention_layernorm, out, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out = out.to(dtype=torch.bfloat16)
        outputs.append(mp._gated_residual(after_first, layer.mlp(out), gate))
        start = end
    return outputs


def _space_time_layer(layer, x, time_code, frame_valid):
    """A SigLIP encoder layer with MEM attention. x [B, K, P, D]; the query at (frame k, patch p) attends every
    patch of frame k plus patch p of each valid earlier frame, under one softmax (as pi_model/vision.py)."""
    B, K, P, D = x.shape
    attn = layer.self_attn
    H, hd = attn.num_heads, attn.head_dim
    h = layer.layer_norm1(x)

    def heads(t):
        return t.view(B, K, P, H, hd).permute(0, 1, 3, 2, 4)        # [B, K, H, P, hd]

    q, k_s, v_s = heads(attn.q_proj(h)), heads(attn.k_proj(h)), heads(attn.v_proj(h))
    coded = h + time_code[:, :, None].to(h.dtype)
    k_t, v_t = heads(attn.k_proj(coded)), heads(attn.v_proj(coded))
    logits_s = torch.einsum("bkhpd,bkhqd->bkhpq", q, k_s) * attn.scale
    logits_t = torch.einsum("bkhpd,bjhpd->bkhpj", q, k_t) * attn.scale
    causal = torch.tril(torch.ones(K, K, dtype=torch.bool, device=x.device), diagonal=-1)
    allowed = causal[None, :, None, None, :] & frame_valid[:, None, None, None, :]
    logits_t = logits_t.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(torch.cat([logits_s, logits_t], dim=-1).float(), dim=-1).type_as(q)
    out = torch.einsum("bkhpq,bkhqd->bkhpd", probs[..., :P], v_s) + \
        torch.einsum("bkhpj,bjhpd->bkhpd", probs[..., P:], v_t)
    x = x + attn.out_proj(out.permute(0, 1, 3, 2, 4).reshape(B, K, P, H * hd))
    return x + layer.mlp(layer.layer_norm2(x))


def _plain_layer(layer, x):
    out = layer(x, None)
    return out[0] if isinstance(out, tuple) else out


class Pi05Backbone(nn.Module):
    """Same interface as `Pi07` (forward → losses, sample_actions → normalized chunk)."""

    def __init__(self, cfg, *, pretrained: bool = True):
        super().__init__()
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
        _install_lerobot_patches()
        self.cfg = cfg
        pcfg = _pi05_config()
        if pretrained:
            self.inner = PI05Policy.from_pretrained(PI05_REPO, config=pcfg).model
        else:
            self.inner = PI05Pytorch(pcfg)
        self.pcfg = pcfg
        for layer in self.inner.paligemma_with_expert.gemma_expert.model.layers:
            layer.pi07_insulated = bool(cfg.knowledge_insulation)
        # Any add-on routes through _forward_addons; with none, LeRobot's own forward and sampler run.
        self.addons = bool(cfg.pi05_rtc or cfg.max_history > 1 or cfg.max_subgoals > 0 or cfg.knowledge_insulation)

    @property
    def device(self):
        return next(self.parameters()).device

    def set_gradient_checkpointing(self, on: bool):
        (self.inner.gradient_checkpointing_enable if on else self.inner.gradient_checkpointing_disable)()

    def parameter_report(self):
        pw = self.inner.paligemma_with_expert
        counts = {"vision": sum(p.numel() for p in pw.paligemma.model.vision_tower.parameters()),
                  "backbone": sum(p.numel() for p in pw.paligemma.model.language_model.parameters()),
                  "expert": sum(p.numel() for p in pw.gemma_expert.parameters())
                  + sum(p.numel() for m in (self.inner.action_in_proj, self.inner.action_out_proj,
                                            self.inner.time_mlp_in, self.inner.time_mlp_out) for p in m.parameters())}
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts

    def _inputs(self, batch):
        images = batch["images"]                          # [B, V, K, 3, H, W] in [-1, 1]; current frame last
        V = images.shape[1]
        imgs = [images[:, v, -1].float() for v in range(V)]
        masks = [batch["camera_valid"][:, v] for v in range(V)]
        return imgs, masks, batch["text_ids"], batch["text_valid"]

    def forward(self, batch, *, tau=None, noise=None):
        if self.addons:
            return self._forward_addons(batch, tau=tau, noise=noise)
        imgs, img_masks, tokens, tmask = self._inputs(batch)
        actions = batch["actions"].float()                 # [B, H, 32], normalized to [-1, 1]
        B = actions.shape[0]
        if noise is None:
            noise = torch.randn_like(actions)
        time = self.inner.sample_time(B, actions.device) if tau is None else (1.0 - tau).float()
        per_elem = self.inner.forward(imgs, img_masks, tokens, tmask, actions, noise, time)  # [B, H, 32]
        mask = batch["action_valid"][:, None, :].expand_as(per_elem)
        if "action_step_valid" in batch:
            mask = mask & batch["action_step_valid"][..., None]
        loss = (per_elem * mask).sum() / mask.sum().clamp(min=1)
        zero = loss.new_zeros(())
        return {"loss": loss, "flow_loss": loss, "ce_loss": zero}

    # ------------------------------------------------------------------ π0.7 add-ons
    def _checkpointed(self, fn, *args):
        if self.inner.gradient_checkpointing_enabled and self.training:
            return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)
        return fn(*args)

    def _mem_image(self, frames, seconds_ago, frame_valid):
        """frames [B, K, 3, H, W] (oldest first, current last) -> [B, 256, 2048], PaliGemma's SigLIP with MEM."""
        pw = self.inner.paligemma_with_expert
        vm = pw.paligemma.model.vision_tower.vision_model
        vc = self.cfg.vision
        B, K = frames.shape[:2]
        x = vm.embeddings(frames.flatten(0, 1).float())
        P, D = x.shape[1:]
        x = x.view(B, K, P, D)
        code = sinusoidal_embedding(seconds_ago.float(), D, vc.time_min_period, vc.time_max_period) - \
            sinusoidal_embedding(torch.zeros_like(seconds_ago).float(), D, vc.time_min_period, vc.time_max_period)
        temporal, drop_after = set(vc.temporal_layers), vc.drop_past_after
        current_only = K == 1
        if current_only:
            x = x[:, -1]
        for i, layer in enumerate(vm.encoder.layers):
            if current_only:
                x = self._checkpointed(_plain_layer, layer, x)
            elif i in temporal:
                x = self._checkpointed(_space_time_layer, layer, x, code, frame_valid)
            else:
                x = self._checkpointed(_plain_layer, layer, x.flatten(0, 1)).view(B, K, P, D)
            if not current_only and i == drop_after:
                x, current_only = x[:, -1], True
        if not current_only:
            x = x[:, -1]
        return pw.paligemma.model.multi_modal_projector(vm.post_layernorm(x))

    def _embed_prefix(self, batch):
        """Camera images (MEM over the history when max_history > 1), goal images, then the prompt; one
        bidirectional block, as LeRobot's embed_prefix."""
        inner, pw, cfg = self.inner, self.inner.paligemma_with_expert, self.cfg
        images = batch["images"]
        B, V = images.shape[:2]
        embs, pads = [], []
        for v in range(V):
            if cfg.max_history > 1:
                e = self._mem_image(images[:, v], batch["seconds_ago"], batch["frame_valid"])
            else:
                e = inner._apply_checkpoint(pw.embed_image, images[:, v, -1].float())
            embs.append(e)
            pads.append(batch["camera_valid"][:, v, None].expand(B, e.shape[1]))
        if cfg.max_subgoals > 0:
            for g in range(batch["subgoals"].shape[1]):
                e = inner._apply_checkpoint(pw.embed_image, batch["subgoals"][:, g].float())
                embs.append(e)
                pads.append(batch["subgoal_valid"][:, g, None].expand(B, e.shape[1]))
        embs.append(inner._apply_checkpoint(pw.embed_language_tokens, batch["text_ids"]))
        pads.append(batch["text_valid"])
        pads = torch.cat(pads, dim=1)
        return torch.cat(embs, dim=1), pads, torch.zeros_like(pads)

    def _embed_suffix_per_token(self, x_t, t_tok):
        """LeRobot's embed_suffix with one flow time per action token (t_tok [B, H])."""
        from lerobot.policies.pi05 import modeling_pi05 as mp
        inner = self.inner
        B, H = t_tok.shape
        emb = mp.create_sinusoidal_pos_embedding(t_tok.reshape(-1), inner.action_in_proj.out_features,
                                                 min_period=inner.config.min_period, max_period=inner.config.max_period,
                                                 device=t_tok.device).view(B, H, -1).type(t_tok.dtype)
        action_emb = inner._apply_checkpoint(inner.action_in_proj, x_t)

        def time_mlp(e):
            return F.silu(inner.time_mlp_out(F.silu(inner.time_mlp_in(e))))

        cond = inner._apply_checkpoint(time_mlp, emb)
        pad = torch.ones(B, H, dtype=torch.bool, device=t_tok.device)
        att = torch.tensor([1] + [0] * (H - 1), dtype=action_emb.dtype, device=action_emb.device)[None].expand(B, H)
        return action_emb, pad, att, cond

    def _forward_addons(self, batch, *, tau=None, noise=None):
        from lerobot.policies.pi05 import modeling_pi05 as mp
        inner, pw, cfg = self.inner, self.inner.paligemma_with_expert, self.cfg
        embs, pad, att = self._embed_prefix(batch)
        n_ctx = embs.shape[1]
        ki = cfg.knowledge_insulation
        if ki:  # FAST tokens after the prompt, causal among themselves
            fast_in, fast_in_valid = batch["fast_ids"][:, :-1], batch["fast_valid"][:, :-1]
            embs = torch.cat([embs, inner._apply_checkpoint(pw.embed_language_tokens, fast_in)], dim=1)
            pad = torch.cat([pad, fast_in_valid], dim=1)
            att = torch.cat([att, torch.ones_like(fast_in_valid)], dim=1)
        actions = batch["actions"].float()
        B, H, _ = actions.shape
        if noise is None:
            noise = torch.randn_like(actions)
        time = inner.sample_time(B, actions.device) if tau is None else (1.0 - tau).float()
        if cfg.pi05_rtc:  # openpi convention: t = 1 is noise, so the committed prefix sits at t = 0
            is_prefix = torch.arange(H, device=actions.device)[None, :] < batch["rtc_delay"].to(actions.device)[:, None]
            t_tok = torch.where(is_prefix, torch.zeros_like(time)[:, None], time[:, None])
            x_t = t_tok[..., None] * noise + (1 - t_tok[..., None]) * actions
            s_embs, s_pad, s_att, cond = self._embed_suffix_per_token(x_t, t_tok)
        else:
            is_prefix = torch.zeros(B, H, dtype=torch.bool, device=actions.device)
            x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
            s_embs, s_pad, s_att, cond = inner.embed_suffix(x_t, time)
        u_t = noise - actions
        if pw.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            embs, s_embs = embs.to(torch.bfloat16), s_embs.to(torch.bfloat16)
        P = embs.shape[1]
        att_2d = mp.make_att_2d_masks(torch.cat([pad, s_pad], dim=1), torch.cat([att, s_att], dim=1))
        position_ids = torch.cumsum(torch.cat([pad, s_pad], dim=1), dim=1) - 1
        if ki:  # action tokens never read the FAST tokens and keep their inference positions
            att_2d[:, P:, n_ctx:P] = False
            position_ids[:, P:] -= fast_in_valid.sum(dim=1, keepdim=True)
        att_4d = mp.prepare_attention_masks_4d(att_2d)

        def run(embs, s_embs, att_4d, position_ids, cond):
            (p_out, s_out), _ = pw.forward(attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
                                           inputs_embeds=[embs, s_embs], use_cache=False, adarms_cond=[None, cond])
            return p_out, s_out

        p_out, s_out = inner._apply_checkpoint(run, embs, s_embs, att_4d, position_ids, cond)
        v_t = inner._apply_checkpoint(inner.action_out_proj, s_out[:, -H:].to(torch.float32))
        per_elem = F.mse_loss(u_t, v_t, reduction="none")
        mask = batch["action_valid"][:, None, :].expand_as(per_elem) & ~is_prefix[..., None]
        if "action_step_valid" in batch:
            mask = mask & batch["action_step_valid"][..., None]
        flow = (per_elem * mask).sum() / mask.sum().clamp(min=1)
        ce = flow.new_zeros(())
        out = {}
        if ki:  # the last prompt token predicts the first FAST token, FAST token j predicts token j + 1
            n_img = n_ctx - batch["text_ids"].shape[1]
            last_text = n_img + batch["text_valid"].sum(dim=1) - 1
            Lf = batch["fast_ids"].shape[1]
            rows = torch.cat([last_text[:, None], n_ctx + torch.arange(Lf - 1, device=actions.device)[None].expand(B, -1)], dim=1)
            hidden = p_out.gather(1, rows[..., None].expand(-1, -1, p_out.shape[-1]))
            valid = batch["fast_valid"]
            logits = pw.paligemma.lm_head(hidden[valid]).float()
            ce = F.cross_entropy(logits, batch["fast_ids"][valid])
            out["ce_accuracy"] = (logits.argmax(-1) == batch["fast_ids"][valid]).float().mean()
        out.update({"loss": flow + ce, "flow_loss": flow, "ce_loss": ce})
        return out

    @torch.no_grad()
    def sample_actions(self, batch, *, prefix_actions=None, delay=0, steps=None, cfg_beta=None, uncond_batch=None,
                       generator=None):
        steps = steps or self.cfg.denoise_steps
        inpaint = bool(delay) and self.cfg.pi05_rtc and prefix_actions is not None
        B = batch["text_ids"].shape[0]
        noise = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, device=batch["text_ids"].device,
                            generator=generator)
        if not self.addons:
            imgs, img_masks, tokens, tmask = self._inputs(batch)
            out = self.inner.sample_actions(imgs, img_masks, tokens, tmask, noise=noise, num_steps=steps)
            return out.float()
        from lerobot.policies.pi05 import modeling_pi05 as mp
        inner, pw = self.inner, self.inner.paligemma_with_expert
        embs, pad, att = self._embed_prefix(batch)
        pw.paligemma.model.language_model.config._attn_implementation = "eager"
        _, kv = pw.forward(attention_mask=mp.prepare_attention_masks_4d(mp.make_att_2d_masks(pad, att)),
                           position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                           inputs_embeds=[embs, None], use_cache=True)
        H = self.cfg.action_horizon
        x = noise
        is_prefix = torch.arange(H, device=x.device)[None, :] < (delay if inpaint else 0)
        if inpaint:
            x[:, :delay] = prefix_actions[:, :delay].to(x.dtype)
        dt = -1.0 / steps
        for s in range(steps):
            t = 1.0 + s * dt
            if inpaint:
                v = self._denoise_per_token(pad, kv, x, torch.where(is_prefix, 0.0, t).expand(B, H).to(x.dtype))
            else:
                v = inner.denoise_step(prefix_pad_masks=pad, past_key_values=kv, x_t=x,
                                       timestep=torch.full((B,), t, dtype=torch.float32, device=x.device))
            x = torch.where(is_prefix[..., None], x, x + dt * v)
        return x.float()

    def _denoise_per_token(self, prefix_pad, kv, x_t, t_tok):
        """LeRobot's denoise_step with per-token flow times (the committed RTC prefix at t = 0)."""
        from lerobot.policies.pi05 import modeling_pi05 as mp
        inner, pw = self.inner, self.inner.paligemma_with_expert
        s_embs, s_pad, s_att, cond = self._embed_suffix_per_token(x_t, t_tok)
        B, Ls, Lp = prefix_pad.shape[0], s_pad.shape[1], prefix_pad.shape[1]
        full = torch.cat([prefix_pad[:, None, :].expand(B, Ls, Lp), mp.make_att_2d_masks(s_pad, s_att)], dim=2)
        position_ids = prefix_pad.sum(dim=-1)[:, None] + torch.cumsum(s_pad, dim=1) - 1
        pw.gemma_expert.model.config._attn_implementation = "eager"
        out, _ = pw.forward(attention_mask=mp.prepare_attention_masks_4d(full), position_ids=position_ids,
                            past_key_values=mp.clone_past_key_values(kv), inputs_embeds=[None, s_embs],
                            use_cache=False, adarms_cond=[None, cond])
        return inner.action_out_proj(out[1][:, -self.cfg.action_horizon:].to(torch.float32))
