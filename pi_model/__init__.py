"""π0.7 reimplementation: Gemma 3 4B backbone, MEM history encoder, 860M flow expert.

See docs/PI07_IMPLEMENTATION.md for the dimension table and every assumption.
"""
try:
    # Triton's native module segfaults when it is first loaded after an OSMesa OpenGL context
    # exists (rendering, then training or compiling in one process). Loading it up front avoids that.
    import triton  # noqa: F401
except ImportError:
    pass

from .config import Pi07Config, VisionConfig, TextConfig, ExpertConfig
from .model import Pi07

__all__ = ["Pi07", "Pi07Config", "VisionConfig", "TextConfig", "ExpertConfig"]
