"""Model dimensions. `full()` is the paper; `tiny()` runs on a laptop CPU."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
import json


@dataclass(frozen=True)
class VisionConfig:
    """SigLIP So400m/14 as shipped inside Gemma 3, with MEM temporal attention."""
    hidden: int = 1152
    layers: int = 27
    heads: int = 16
    mlp: int = 4304
    patch: int = 14
    image_size: int = 448          # π0.7 resizes every camera and subgoal image to 448x448
    pool: int = 2                  # 32x32 patches -> 16x16 = 256 tokens, Gemma 3's token budget per image
    temporal_every: int = 4        # MEM: every 4th layer adds attention over time (layers 4, 8, ...)
    layer_norm_eps: float = 1e-6
    time_min_period: float = 0.2   # seconds; sinusoidal temporal position encoding e(t), e(0)=0
    time_max_period: float = 100.0

    @property
    def grid(self) -> int:
        if self.image_size % self.patch:
            raise ValueError("image_size must be a multiple of the patch size")
        return self.image_size // self.patch

    @property
    def patches(self) -> int:
        return self.grid ** 2

    @property
    def tokens_per_image(self) -> int:
        if self.grid % self.pool:
            raise ValueError("patch grid must be divisible by the pooling factor")
        return (self.grid // self.pool) ** 2

    @property
    def temporal_layers(self) -> tuple[int, ...]:
        """0-based indices of the layers with temporal attention (every 4th: 3, 7, ...)."""
        return tuple(i for i in range(self.layers) if (i + 1) % self.temporal_every == 0)

    @property
    def drop_past_after(self) -> int:
        """Past-frame tokens are dropped after the last temporal layer (assumption)."""
        return self.temporal_layers[-1] if self.temporal_layers else -1


@dataclass(frozen=True)
class TextConfig:
    """Gemma 3 4B decoder (google/gemma-3-4b config.json, text_config)."""
    hidden: int = 2560
    layers: int = 34
    heads: int = 8
    kv_heads: int = 4
    head_dim: int = 256
    mlp: int = 10240
    vocab: int = 262208
    sliding_window: int = 1024
    sliding_pattern: int = 6       # every 6th layer is global, the rest are local
    rope_theta: float = 1_000_000.0
    rope_local_theta: float = 10_000.0
    rope_scaling: float = 8.0      # linear scaling on the global layers
    query_pre_attn_scalar: int = 256
    rms_eps: float = 1e-6
    pad_id: int = 0
    eos_id: int = 1
    bos_id: int = 2

    def is_sliding(self, layer: int) -> bool:
        return bool((layer + 1) % self.sliding_pattern)


@dataclass(frozen=True)
class ExpertConfig:
    """Flow-matching action expert. Width and MLP are assumptions that land near 860M.

    The expert has one layer per backbone layer so that its tokens can attend
    to the backbone's keys and values at every layer, and it uses the backbone's
    head layout so those keys and values are directly compatible.
    """
    hidden: int = 1216
    mlp: int = 4608
    time_dim: int = 256
    time_min_period: float = 4e-3
    time_max_period: float = 4.0


@dataclass(frozen=True)
class Pi07Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    action_dim: int = 32           # actions are zero-padded to this width (12 used by two SO-101s)
    state_dim: int = 32
    action_horizon: int = 50       # action chunk length, fixed in the paper
    max_history: int = 6           # five past frames plus the current one
    history_stride_s: float = 1.0
    max_cameras: int = 4           # front/overhead, two wrists, optional rear
    max_subgoals: int = 3
    fast_vocab: int = 1024
    fast_skip: int = 128           # FAST ids occupy vocab ids counted down from vocab-1-skip
    max_rtc_delay: int = 12        # training-time RTC: simulated delay 0..12 steps
    denoise_steps: int = 5
    control_hz: int = 50
    # Proprioception shortcuts: with scripted demos the past joint states reveal where the arm is
    # heading (and so where the object is), letting a policy ignore its cameras. state_history=False
    # feeds only the current state; state_dropout zeroes all states for that fraction of samples.
    state_history: bool = True
    state_dropout: float = 0.0
    name: str = "full"

    @staticmethod
    def full() -> "Pi07Config":
        return Pi07Config()

    @staticmethod
    def tiny() -> "Pi07Config":
        """Same wiring, laptop-sized: for tests and end-to-end smoke runs."""
        return Pi07Config(
            vision=VisionConfig(hidden=64, layers=4, heads=4, mlp=128, patch=14, image_size=112, pool=2),
            text=TextConfig(hidden=128, layers=6, heads=4, kv_heads=2, head_dim=32, mlp=256, vocab=4096,
                            sliding_window=1024, query_pre_attn_scalar=32),
            expert=ExpertConfig(hidden=96, mlp=192, time_dim=64),
            name="tiny")

    @staticmethod
    def small() -> "Pi07Config":
        """A few hundred million parameters: single-GPU experiments without Gemma weights."""
        return Pi07Config(
            vision=VisionConfig(hidden=384, layers=12, heads=6, mlp=1536, image_size=224, pool=2),
            text=TextConfig(hidden=768, layers=12, heads=12, kv_heads=4, head_dim=64, mlp=3072, vocab=4096,
                            query_pre_attn_scalar=64),
            expert=ExpertConfig(hidden=384, mlp=1536, time_dim=256),
            name="small")

    @staticmethod
    def named(name: str) -> "Pi07Config":
        try:
            return {"full": Pi07Config.full, "tiny": Pi07Config.tiny, "small": Pi07Config.small}[name]()
        except KeyError:
            raise ValueError(f"Unknown config {name!r}; choose full, small or tiny") from None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Pi07Config":
        raw = json.loads(text)
        return Pi07Config(vision=VisionConfig(**raw.pop("vision")), text=TextConfig(**raw.pop("text")),
                          expert=ExpertConfig(**raw.pop("expert")), **raw)

    def with_(self, **changes) -> "Pi07Config":
        return replace(self, **changes)
