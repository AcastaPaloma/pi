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

Needs `lerobot[pi]` (see scripts/box/train_pi05.sh); weights download from `lerobot/pi05_base`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class Pi05Backbone(nn.Module):
    """Same interface as `Pi07` (forward → losses, sample_actions → normalized chunk)."""

    def __init__(self, cfg, *, pretrained: bool = True):
        super().__init__()
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch
        self.cfg = cfg
        pcfg = _pi05_config()
        if pretrained:
            self.inner = PI05Policy.from_pretrained(PI05_REPO, config=pcfg).model
        else:
            self.inner = PI05Pytorch(pcfg)
        self.pcfg = pcfg

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

    @torch.no_grad()
    def sample_actions(self, batch, *, prefix_actions=None, delay=0, steps=None, cfg_beta=None, uncond_batch=None,
                       generator=None):
        imgs, img_masks, tokens, tmask = self._inputs(batch)
        B = tokens.shape[0]
        noise = torch.randn(B, self.cfg.action_horizon, self.cfg.action_dim, device=tokens.device, generator=generator)
        out = self.inner.sample_actions(imgs, img_masks, tokens, tmask, noise=noise,
                                        num_steps=steps or self.cfg.denoise_steps)
        return out.float()
