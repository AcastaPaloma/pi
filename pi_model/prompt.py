"""The π0.7 prompt: task, subtask, episode metadata, control mode; with the paper's dropout.

    Task: peel vegetables. Subtask: pick up the peeler. Speed: 8000. Quality: 5.
    Mistake: false. Control Mode: joint.

Dropout (Sec. V-E): subgoal images are attached to 25% of examples; among those
the subtask is dropped 30% of the time. Metadata is dropped entirely 15% of the
time and each component independently 5% of the time. Control mode is never
dropped. History frames are dropped entirely with probability 0.3 (Sec. VI-B).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

SPEED_BIN = 500  # episode length in steps is rounded to the nearest 500 (paper)


@dataclass
class PromptContext:
    task: str
    subtask: str | None = None
    speed_steps: int | None = None        # raw episode length; binned when rendered
    quality: int | None = None            # 1..5
    mistake: bool | None = None
    control_mode: str = "joint"           # "joint" or "ee"
    speed_bin: int = SPEED_BIN

    def rendered_speed(self):
        if self.speed_steps is None:
            return None
        return int(max(self.speed_bin, self.speed_bin * round(self.speed_steps / self.speed_bin)))

    def text(self, *, include_subtask=True, include_speed=True, include_quality=True, include_mistake=True) -> str:
        parts = [f"Task: {self.task.strip().rstrip('.')}."]
        if include_subtask and self.subtask:
            parts.append(f"Subtask: {self.subtask.strip().rstrip('.')}.")
        if include_speed and self.speed_steps is not None:
            parts.append(f"Speed: {self.rendered_speed()}.")
        if include_quality and self.quality is not None:
            parts.append(f"Quality: {int(self.quality)}.")
        if include_mistake and self.mistake is not None:
            parts.append(f"Mistake: {'true' if self.mistake else 'false'}.")
        if self.control_mode not in ("joint", "ee"):
            raise ValueError("control_mode must be 'joint' or 'ee'")
        parts.append(f"Control Mode: {self.control_mode}.")
        return " ".join(parts)

    def unconditional(self) -> "PromptContext":
        """Context for classifier-free guidance on the episode metadata."""
        return PromptContext(self.task, self.subtask, None, None, None, self.control_mode, self.speed_bin)


@dataclass
class DropoutPlan:
    subgoals: bool
    subtask: bool
    speed: bool
    quality: bool
    mistake: bool
    history: bool

    def as_dict(self):
        return asdict(self)


def sample_dropout(rng: np.random.Generator, *, p_subgoals=0.25, p_subtask_given_subgoals=0.3,
                   p_metadata_all=0.15, p_metadata_each=0.05, p_history=0.3) -> DropoutPlan:
    subgoals = rng.random() < p_subgoals
    subtask = not (subgoals and rng.random() < p_subtask_given_subgoals)
    if rng.random() < p_metadata_all:
        speed = quality = mistake = False
    else:
        speed, quality, mistake = (rng.random() >= p_metadata_each for _ in range(3))
    history = rng.random() >= p_history
    return DropoutPlan(subgoals, subtask, speed, quality, mistake, history)


def no_dropout(*, subgoals=False) -> DropoutPlan:
    return DropoutPlan(subgoals, True, True, True, True, True)


ACTION_MARKER = " Action:"  # closes the text prompt before the state tokens and FAST tokens


def prompt_ids(tokenizer, context: PromptContext, plan: DropoutPlan, *, bos=True) -> list[int]:
    text = context.text(include_subtask=plan.subtask, include_speed=plan.speed,
                        include_quality=plan.quality, include_mistake=plan.mistake) + ACTION_MARKER
    ids = tokenizer.encode(text)
    return ([tokenizer.bos_id] if bos else []) + ids
