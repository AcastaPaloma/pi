"""Training batches from recorded episodes (the pi_sim.record layout).

Each sample is (episode, timestep t): up to six history frames per camera at a
one-second stride (current frame last), the proprioceptive state at those frames,
the 50-step action chunk starting at t, FAST tokens for that chunk, the prompt with
the paper's dropout, and optional subgoal images (end-of-segment frame with
probability 0.25, otherwise a frame 0–4 s ahead). States and actions are mapped
to [-1, 1] by per-dimension 1st/99th percentiles computed by `prepare`.

    python -m pi_model.data prepare --root data/sim_pick_place --out data/sim_pick_place/pi07_meta
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import Pi07Config
from .fast import FastTokenizer, fast_to_vocab
from .prompt import PromptContext, no_dropout, prompt_ids, sample_dropout
from .vision import preprocess_images

ACTION_DIMS = 12  # two SO-101 arms: 5 joints + gripper each


@dataclass
class Episode:
    path: Path
    meta: dict
    steps: dict

    @property
    def length(self) -> int:
        return int(self.meta["steps"])


class EpisodeStore:
    def __init__(self, root, *, episodes=None):
        self.root = Path(root)
        dirs = sorted(p for p in self.root.glob("episode_*") if (p / "meta.json").exists())
        if episodes is not None:
            dirs = [d for d in dirs if d.name in set(episodes)]
        if not dirs:
            raise FileNotFoundError(f"No episodes under {self.root}")
        self.episodes = []
        for d in dirs:
            meta = json.loads((d / "meta.json").read_text())
            with np.load(d / "steps.npz") as z:
                steps = {k: z[k] for k in z.files}
            self.episodes.append(Episode(d, meta, steps))
        self.cameras = list(self.episodes[0].meta["cameras"])
        self.hz = int(self.episodes[0].meta["control_hz"])
        self.lengths = np.asarray([e.length for e in self.episodes])
        self.offsets = np.r_[0, np.cumsum(self.lengths)]

    def __len__(self):
        return len(self.episodes)

    def locate(self, index: int):
        e = int(np.searchsorted(self.offsets, index, side="right") - 1)
        return e, int(index - self.offsets[e])

    def frame(self, episode: Episode, cam: str, t: int, size: int) -> np.ndarray:
        image = Image.open(episode.path / "frames" / cam / f"{t:05d}.jpg").convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image)

    def episode_lengths_by_task(self):
        out = {}
        for e in self.episodes:
            out.setdefault(e.meta["task"], []).append(e.length)
        return out


# --------------------------------------------------------------------- prepare
def prepare(root, out, cfg: Pi07Config, *, chunk_stride=5, horizon=None):
    """Fit normalization statistics and the FAST tokenizer; write them next to the data."""
    store = EpisodeStore(root)
    horizon = horizon or cfg.action_horizon
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    states = np.concatenate([e.steps["state"] for e in store.episodes])
    chunks = []
    for e in store.episodes:
        action = e.steps["action"]
        for t in range(0, e.length, chunk_stride):
            chunk = action[t:t + horizon]
            if len(chunk) < horizon:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)])
            chunks.append(chunk)
    chunks = np.stack(chunks)
    tok = FastTokenizer(horizon, ACTION_DIMS, vocab_size=cfg.fast_vocab).fit(chunks)
    tok.save(out / "fast")
    stats = {
        "schema_version": 1, "control_hz": store.hz, "cameras": store.cameras, "horizon": horizon,
        "action_dims": ACTION_DIMS,
        "state_q01": np.percentile(states, 1, axis=0).tolist(), "state_q99": np.percentile(states, 99, axis=0).tolist(),
        "action_q01": tok.low.tolist(), "action_q99": tok.high.tolist(),
        "episode_lengths_by_task": store.episode_lengths_by_task(),
        "episodes": len(store), "steps": int(store.lengths.sum()), "fast_chunks": len(chunks),
        "fast_tokens_per_chunk_mean": float(np.mean([len(tok.encode(c)) for c in chunks[:: max(1, len(chunks) // 200)]])),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


class Normalizer:
    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
        self.high = np.where(self.high - self.low < 1e-6, self.low + 1, self.high)

    def __call__(self, x):
        return np.clip(2 * (np.asarray(x, dtype=np.float32) - self.low) / (self.high - self.low) - 1, -1.5, 1.5)

    def inverse(self, y):
        return (np.asarray(y, dtype=np.float32) + 1) / 2 * (self.high - self.low) + self.low


def speed_prompt_value(stats: dict, task: str, percentile: float = 15.0) -> int:
    """Runtime speed metadata: the paper's 15th percentile of the task's episode lengths."""
    lengths = stats["episode_lengths_by_task"].get(task)
    if not lengths:
        lengths = [n for ns in stats["episode_lengths_by_task"].values() for n in ns]
    return int(np.percentile(lengths, percentile))


# ---------------------------------------------------------------------- dataset
class Pi07Dataset(Dataset):
    def __init__(self, root, meta_dir, cfg: Pi07Config, tokenizer, *, train=True, objective="action",
                 image_size=None, speed_bin=500, seed=0, episodes=None, subgoal_cameras=None):
        if objective not in ("action", "subtask"):
            raise ValueError("objective must be 'action' (low-level policy) or 'subtask' (high-level policy)")
        self.store = EpisodeStore(root, episodes=episodes)
        self.cfg, self.tokenizer, self.train, self.objective = cfg, tokenizer, train, objective
        self.image_size = image_size or cfg.vision.image_size
        self.speed_bin, self.seed = speed_bin, seed
        meta_dir = Path(meta_dir)
        self.stats = json.loads((meta_dir / "stats.json").read_text())
        self.fast = FastTokenizer.load(meta_dir / "fast") if objective == "action" else None
        self.norm_state = Normalizer(self.stats["state_q01"], self.stats["state_q99"])
        self.norm_action = Normalizer(self.stats["action_q01"], self.stats["action_q99"])
        self.cameras = self.store.cameras
        self.subgoal_cameras = list(subgoal_cameras or self.cameras[: cfg.max_subgoals])
        self.stride = int(round(cfg.history_stride_s * self.store.hz))
        if len(self.cameras) > cfg.max_cameras:
            raise ValueError(f"{len(self.cameras)} cameras exceed max_cameras={cfg.max_cameras}")

    def __len__(self):
        return int(self.store.lengths.sum())

    def _rng(self, index):
        salt = torch.initial_seed() % (2 ** 31) if self.train else 0
        return np.random.default_rng([self.seed, index, salt])

    def __getitem__(self, index):
        e, t = self.store.locate(index)
        episode = self.store.episodes[e]
        rng = self._rng(index)
        cfg = self.cfg
        plan = sample_dropout(rng) if self.train else no_dropout()
        K, H = cfg.max_history, cfg.action_horizon
        frame_steps = [t - (K - 1 - k) * self.stride for k in range(K)]
        valid = np.asarray([s >= 0 for s in frame_steps])
        if not plan.history:
            valid[:-1] = False
        S = self.image_size
        images = np.zeros((len(self.cameras), K, S, S, 3), dtype=np.uint8)
        for v, cam in enumerate(self.cameras):
            for k, s in enumerate(frame_steps):
                if valid[k]:
                    images[v, k] = self.store.frame(episode, cam, s, S)
        seconds_ago = np.asarray([(K - 1 - k) * self.stride / self.store.hz for k in range(K)], dtype=np.float32)
        state = np.zeros((K, cfg.state_dim), dtype=np.float32)
        for k, s in enumerate(frame_steps):
            if valid[k]:
                state[k, :ACTION_DIMS] = self.norm_state(episode.steps["state"][s])

        segment = episode.steps["segment_id"]
        subtask = episode.meta["subtasks"][int(episode.steps["subtask_index"][t])]
        mistake = bool(episode.steps["mistake"][t])
        context = PromptContext(task=episode.meta["task"], subtask=subtask, speed_steps=episode.length,
                                quality=int(episode.meta["quality"]), mistake=mistake,
                                control_mode="joint", speed_bin=self.speed_bin)

        sample = {"images": images, "frame_valid": valid, "seconds_ago": seconds_ago,
                  "camera_valid": np.ones(len(self.cameras), dtype=bool), "state": state, "state_valid": valid,
                  "episode": e, "t": t}

        G = len(self.subgoal_cameras)
        subgoals = np.zeros((G, S, S, 3), dtype=np.uint8)
        if plan.subgoals:
            if rng.random() < 0.25:
                end = int(np.flatnonzero(segment == segment[t]).max())
            else:
                end = min(episode.length - 1, t + int(rng.uniform(0, 4) * self.store.hz))
            for g, cam in enumerate(self.subgoal_cameras):
                subgoals[g] = self.store.frame(episode, cam, end, S)
        sample["subgoals"] = subgoals
        sample["subgoal_valid"] = np.full(G, plan.subgoals, dtype=bool)

        if self.objective == "action":
            text = prompt_ids(self.tokenizer, context, plan)
            sample["text_ids"] = np.asarray(text, dtype=np.int64)
            sample["text_targets"] = np.full(len(text), -1, dtype=np.int64)
            chunk = episode.steps["action"][t:t + H]
            step_valid = np.zeros(H, dtype=bool)
            step_valid[: len(chunk)] = True
            if len(chunk) < H:
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], H - len(chunk), axis=0)])
            actions = np.zeros((H, cfg.action_dim), dtype=np.float32)
            actions[:, :ACTION_DIMS] = self.norm_action(chunk)
            fast = fast_to_vocab(self.fast.encode(chunk), cfg.text.vocab, cfg.fast_skip) + [self.tokenizer.eos_id]
            sample.update({"actions": actions, "action_step_valid": step_valid,
                           "action_valid": np.r_[np.ones(ACTION_DIMS, bool), np.zeros(cfg.action_dim - ACTION_DIMS, bool)],
                           "fast_ids": np.asarray(fast, dtype=np.int64),
                           "rtc_delay": int(rng.integers(0, cfg.max_rtc_delay + 1)) if self.train else 0})
        else:
            # High-level policy: observation + task + past subtasks -> next subtask text.
            previous = [episode.meta["subtasks"][i] for i in sorted(set(episode.steps["subtask_index"][:t].tolist()), key=episode.steps["subtask_index"][:t].tolist().index)]
            prompt = f"Task: {episode.meta['task'].rstrip('.')}. Previous subtasks: " + \
                     (" ".join(p.rstrip('.') + "." for p in previous[-4:]) if previous else "none.") + " Next subtask:"
            prompt_tokens = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
            target_tokens = self.tokenizer.encode(" " + subtask) + [self.tokenizer.eos_id]
            sample["text_ids"] = np.asarray(prompt_tokens + target_tokens, dtype=np.int64)
            sample["text_targets"] = np.asarray([-1] * len(prompt_tokens) + target_tokens, dtype=np.int64)
            sample["state"] = np.zeros((0, cfg.state_dim), dtype=np.float32)
            sample["state_valid"] = np.zeros(0, dtype=bool)
        return sample


def collate(samples, cfg: Pi07Config):
    """Pad variable-length fields and lay out the LM targets over the full prefix."""
    B = len(samples)
    T = cfg.vision.tokens_per_image
    V = samples[0]["images"].shape[0]
    G = samples[0]["subgoals"].shape[0]
    K = samples[0]["state"].shape[0]
    Lt = max(len(s["text_ids"]) for s in samples)
    has_fast = "fast_ids" in samples[0]
    Lf = max(len(s["fast_ids"]) for s in samples) if has_fast else 0
    text_ids = np.zeros((B, Lt), dtype=np.int64)
    text_valid = np.zeros((B, Lt), dtype=bool)
    fast_ids = np.zeros((B, Lf), dtype=np.int64)
    fast_valid = np.zeros((B, Lf), dtype=bool)
    N = V * T + G * T + Lt + K + Lf
    lm_targets = np.zeros((B, N), dtype=np.int64)
    lm_mask = np.zeros((B, N), dtype=bool)
    text_start = V * T + G * T
    for b, s in enumerate(samples):
        n = len(s["text_ids"])
        text_ids[b, :n] = s["text_ids"]
        text_valid[b, :n] = True
        targets = s["text_targets"]
        for j in range(1, n):
            if targets[j] >= 0:
                lm_mask[b, text_start + j - 1] = True
                lm_targets[b, text_start + j - 1] = targets[j]
        if has_fast:
            f = len(s["fast_ids"])
            fast_ids[b, :f] = s["fast_ids"]
            fast_valid[b, :f] = True
            last_state = text_start + Lt + K - 1
            lm_mask[b, last_state] = True
            lm_targets[b, last_state] = s["fast_ids"][0]
            for i in range(f - 1):
                lm_mask[b, last_state + 1 + i] = True
                lm_targets[b, last_state + 1 + i] = s["fast_ids"][i + 1]
    batch = {
        "images": torch.from_numpy(np.stack([s["images"] for s in samples])),
        "frame_valid": torch.from_numpy(np.stack([s["frame_valid"] for s in samples])),
        "seconds_ago": torch.from_numpy(np.stack([s["seconds_ago"] for s in samples])),
        "camera_valid": torch.from_numpy(np.stack([s["camera_valid"] for s in samples])),
        "subgoals": torch.from_numpy(np.stack([s["subgoals"] for s in samples])),
        "subgoal_valid": torch.from_numpy(np.stack([s["subgoal_valid"] for s in samples])),
        "text_ids": torch.from_numpy(text_ids), "text_valid": torch.from_numpy(text_valid),
        "state": torch.from_numpy(np.stack([s["state"] for s in samples])),
        "state_valid": torch.from_numpy(np.stack([s["state_valid"] for s in samples])),
        "lm_targets": torch.from_numpy(lm_targets), "lm_target_mask": torch.from_numpy(lm_mask),
        "episode": torch.tensor([s["episode"] for s in samples]), "t": torch.tensor([s["t"] for s in samples]),
    }
    if has_fast:
        batch.update({
            "fast_ids": torch.from_numpy(fast_ids), "fast_valid": torch.from_numpy(fast_valid),
            "actions": torch.from_numpy(np.stack([s["actions"] for s in samples])),
            "action_step_valid": torch.from_numpy(np.stack([s["action_step_valid"] for s in samples])),
            "action_valid": torch.from_numpy(np.stack([s["action_valid"] for s in samples])),
            "rtc_delay": torch.tensor([s["rtc_delay"] for s in samples]),
        })
    return batch


def to_device(batch, device, image_dtype=torch.float32):
    """Move a collated batch; uint8 images become [-1, 1] float tensors in CHW order."""
    out = {}
    for k, v in batch.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        v = v.to(device, non_blocking=True)
        if k in ("images", "subgoals"):
            v = preprocess_images(v).to(image_dtype)
        out[k] = v
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Fit normalization statistics and the FAST tokenizer")
    prep.add_argument("--root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--config", default="full")
    prep.add_argument("--chunk-stride", type=int, default=5)
    args = parser.parse_args(argv)
    stats = prepare(args.root, args.out, Pi07Config.named(args.config), chunk_stride=args.chunk_stride)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("state_q01", "state_q99", "action_q01", "action_q99")}, indent=2))


if __name__ == "__main__":
    main()
