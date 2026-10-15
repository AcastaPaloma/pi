"""Text tokenizers: the Gemma SentencePiece model, or a byte fallback with Gemma's special ids."""
from __future__ import annotations

from pathlib import Path


class ByteTokenizer:
    """UTF-8 bytes as ids 3..258 with Gemma's pad=0, eos=1, bos=2.

    A stand-in when the gated Gemma tokenizer is unavailable. It keeps every
    sequence layout and vocabulary id range identical, so shapes and FAST token
    placement are exact; only the text embedding content differs.
    """
    name = "byte-fallback"

    def __init__(self, vocab_size: int):
        if vocab_size < 259:
            raise ValueError("vocab must hold 256 bytes plus three specials")
        self.vocab_size = vocab_size
        self.pad_id, self.eos_id, self.bos_id = 0, 1, 2

    def encode(self, text: str) -> list[int]:
        return [3 + b for b in text.encode("utf-8")]

    def decode(self, ids) -> str:
        return bytes(int(i) - 3 for i in ids if 3 <= int(i) < 259).decode("utf-8", errors="replace")


class GemmaTokenizer:
    """google/gemma-3 tokenizer.model via sentencepiece (accept the license on the Hub first)."""
    name = "gemma-sentencepiece"

    def __init__(self, model_path, vocab_size: int):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.vocab_size = vocab_size
        self.pad_id, self.eos_id, self.bos_id = self.sp.pad_id(), self.sp.eos_id(), self.sp.bos_id()
        if (self.pad_id, self.eos_id, self.bos_id) != (0, 1, 2):
            raise ValueError("Expected Gemma special ids pad=0, eos=1, bos=2")
        if self.sp.get_piece_size() > vocab_size:
            raise ValueError("tokenizer vocabulary exceeds the model vocabulary")

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids) -> str:
        return self.sp.decode([int(i) for i in ids])


def load_tokenizer(path, vocab_size: int):
    """A tokenizer.model path gives the real Gemma tokenizer; None gives the byte fallback."""
    if path is None:
        return ByteTokenizer(vocab_size)
    path = Path(path).expanduser()
    if path.is_dir():
        path = path / "tokenizer.model"
    if not path.exists():
        raise FileNotFoundError(f"No SentencePiece model at {path}")
    return GemmaTokenizer(path, vocab_size)
