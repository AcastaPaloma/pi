"""Run a trained π0.7 checkpoint as a policy: history buffer, prompt, chunk sampling, RTC prefix."""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from .config import Pi07Config
from .data import ACTION_DIMS, Normalizer, speed_prompt_value, to_device
from .model import Pi07
from .prompt import PromptContext, no_dropout, prompt_ids
from .tokenizer import load_tokenizer


class HistoryBuffer:
    """Frames and states at the control rate; samples the one-second-stride history."""

    def __init__(self, cameras, *, max_seconds=8.0, hz=50):
        self.cameras, self.hz = list(cameras), hz
        self.items = deque(maxlen=int(max_seconds * hz) + 1)

    def clear(self):
        self.items.clear()

    def append(self, step, images: dict, state):
        self.items.append((int(step), {c: np.asarray(images[c]) for c in self.cameras}, np.asarray(state, dtype=np.float32)))

    def sample(self, K, stride_steps):
        if not self.items:
            raise RuntimeError("empty history")
        now_step, now_images, now_state = self.items[-1]
        by_step = {s: (im, st) for s, im, st in self.items}
        S = now_images[self.cameras[0]].shape[0]
        frames = np.zeros((len(self.cameras), K, S, S, 3), dtype=np.uint8)
        states = np.zeros((K, now_state.shape[0]), dtype=np.float32)
        valid = np.zeros(K, dtype=bool)
        seconds = np.zeros(K, dtype=np.float32)
        for k in range(K):
            back = (K - 1 - k) * stride_steps
            target = now_step - back
            seconds[k] = back / self.hz
            if target in by_step:
                images, state = by_step[target]
                for v, cam in enumerate(self.cameras):
                    frames[v, k] = images[cam]
                states[k] = state
                valid[k] = True
        return frames, valid, states, seconds


class Pi07Policy:
    def __init__(self, checkpoint, *, device="auto", tokenizer=None, denoise_steps=None, cameras=None, dtype="fp32"):
        checkpoint = Path(checkpoint)
        self.cfg = Pi07Config.from_json((checkpoint / "config.json").read_text())
        self.stats = json.loads((checkpoint / "stats.json").read_text())
        extra = json.loads((checkpoint / "extra.json").read_text()) if (checkpoint / "extra.json").exists() else {}
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = Pi07(self.cfg)
        self.model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu"))
        if dtype not in ("fp32", "bf16"):
            raise ValueError(f"dtype must be fp32 or bf16, got {dtype}")
        if dtype == "bf16":
            self.model.to(torch.bfloat16)  # 5.2B parameters: 10.3 GB instead of 20.7 GB, fits a 24 GB GPU
        self.model.to(self.device).eval()
        self.tokenizer = load_tokenizer(tokenizer or extra.get("tokenizer"), self.cfg.text.vocab)
        self.norm_state = Normalizer(self.stats["state_q01"], self.stats["state_q99"])
        self.norm_action = Normalizer(self.stats["action_q01"], self.stats["action_q99"])
        self.cameras = list(cameras or self.stats["cameras"])
        self.hz = int(self.stats["control_hz"])
        self.stride = int(round(self.cfg.history_stride_s * self.hz))
        self.steps = denoise_steps or self.cfg.denoise_steps
        self.recorded_image_size = int(self.stats.get("recorded_image_size", 448))

    def prepare_image(self, native_rgb):
        """Native camera frame -> model input, through the same two resizes as recorded data."""
        from PIL import Image
        image = Image.fromarray(np.asarray(native_rgb))
        if image.size != (self.recorded_image_size, self.recorded_image_size):
            image = image.resize((self.recorded_image_size, self.recorded_image_size), Image.Resampling.BILINEAR)
        size = self.cfg.vision.image_size
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image)

    def speed_for(self, task, percentile=15.0):
        return speed_prompt_value(self.stats, task, percentile)

    def build_batch(self, frames, valid, states, seconds, context: PromptContext, *, subgoals=None):
        cfg = self.cfg
        K = cfg.max_history
        state = np.zeros((K, cfg.state_dim), dtype=np.float32)
        for k in range(K):
            if valid[k]:
                state[k, :ACTION_DIMS] = self.norm_state(states[k])
        text = prompt_ids(self.tokenizer, context, no_dropout())
        G = min(len(self.cameras), cfg.max_subgoals)
        S = frames.shape[2]
        goal = np.zeros((G, S, S, 3), dtype=np.uint8)
        goal_valid = np.zeros(G, dtype=bool)
        if subgoals is not None:
            for g, cam in enumerate(self.cameras[:G]):
                if cam in subgoals:
                    goal[g] = subgoals[cam]
                    goal_valid[g] = True
        batch = {
            "images": torch.from_numpy(frames[None]), "frame_valid": torch.from_numpy(valid[None]),
            "seconds_ago": torch.from_numpy(seconds[None]), "camera_valid": torch.ones(1, len(self.cameras), dtype=torch.bool),
            "subgoals": torch.from_numpy(goal[None]), "subgoal_valid": torch.from_numpy(goal_valid[None]),
            "text_ids": torch.tensor([text]), "text_valid": torch.ones(1, len(text), dtype=torch.bool),
            "state": torch.from_numpy(state[None]), "state_valid": torch.from_numpy(valid[None]),
        }
        return to_device(batch, self.device)

    @torch.no_grad()
    def act(self, history: HistoryBuffer, context: PromptContext, *, subgoals=None, prefix_actions=None,
            delay=0, cfg_beta=None, generator=None):
        """Returns an action chunk [H, 12] in dataset units (degrees, gripper 0-100)."""
        frames, valid, states, seconds = history.sample(self.cfg.max_history, self.stride)
        batch = self.build_batch(frames, valid, states, seconds, context, subgoals=subgoals)
        uncond = None
        if cfg_beta is not None:
            uncond = self.build_batch(frames, valid, states, seconds, context.unconditional(), subgoals=subgoals)
        prefix = None
        if delay:
            prefix = np.zeros((1, delay, self.cfg.action_dim), dtype=np.float32)
            prefix[0, :, :ACTION_DIMS] = self.norm_action(prefix_actions[:delay])
            prefix = torch.from_numpy(prefix).to(self.device)
        chunk = self.model.sample_actions(batch, prefix_actions=prefix, delay=delay, steps=self.steps,
                                          cfg_beta=cfg_beta, uncond_batch=uncond, generator=generator)
        return self.norm_action.inverse(chunk[0, :, :ACTION_DIMS].float().cpu().numpy())

    @torch.no_grad()
    def next_subtask(self, history: HistoryBuffer, task: str, previous: list[str], max_new_tokens=32):
        """High-level policy: decode the next subtask instruction (checkpoint trained with --objective subtask)."""
        frames, valid, states, seconds = history.sample(self.cfg.max_history, self.stride)
        prompt = f"Task: {task.rstrip('.')}. Previous subtasks: " + \
                 (" ".join(p.rstrip('.') + "." for p in previous[-4:]) if previous else "none.") + " Next subtask:"
        ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        S = frames.shape[2]
        batch = {
            "images": torch.from_numpy(frames[None]), "frame_valid": torch.from_numpy(valid[None]),
            "seconds_ago": torch.from_numpy(seconds[None]), "camera_valid": torch.ones(1, len(self.cameras), dtype=torch.bool),
            "subgoals": torch.zeros(1, 0, S, S, 3, dtype=torch.uint8), "subgoal_valid": torch.zeros(1, 0, dtype=torch.bool),
            "text_ids": torch.tensor([ids]), "text_valid": torch.ones(1, len(ids), dtype=torch.bool),
        }
        out = self.model.generate_text(to_device(batch, self.device), max_new_tokens=max_new_tokens,
                                       eos_id=self.tokenizer.eos_id)[0]
        return self.tokenizer.decode([i for i in out if i != self.tokenizer.eos_id]).strip()
