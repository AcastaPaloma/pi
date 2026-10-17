"""FAST action tokenizer (Pertsch et al. 2025): DCT, quantize, flatten low-frequency first, BPE.

Fitted per dataset: action chunks are scaled to [-1, 1] by per-dimension 1st/99th
percentiles, DCT-II transformed along time per dimension, multiplied by `scale`
and rounded. The integer grid is flattened frequency-major so low frequencies of
all dimensions come first, each integer becomes one symbol, and a byte-pair
encoder with `vocab_size` merges is trained over those symbol strings.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.fft import dct, idct
from tokenizers import Tokenizer, models, trainers, decoders

SYMBOL_BASE = 0x4E00  # map integers to a private run of CJK code points; any unique chars work


def _to_symbols(ints: np.ndarray) -> str:
    return "".join(chr(SYMBOL_BASE + int(v)) for v in ints)


def _from_symbols(text: str) -> np.ndarray:
    return np.asarray([ord(ch) - SYMBOL_BASE for ch in text], dtype=np.int64)


class FastTokenizer:
    def __init__(self, horizon: int, action_dim: int, vocab_size: int = 1024, scale: float = 10.0, max_coeff: int = 256):
        self.horizon, self.action_dim, self.vocab_size, self.scale = horizon, action_dim, vocab_size, scale
        self.max_coeff = max_coeff
        self.low = None
        self.high = None
        self.bpe: Tokenizer | None = None

    # --------------------------------------------------------------- numerics
    def normalize(self, actions):
        actions = np.asarray(actions, dtype=np.float64)
        return np.clip(2 * (actions - self.low) / (self.high - self.low) - 1, -1.5, 1.5)

    def denormalize(self, normalized):
        return (np.asarray(normalized) + 1) / 2 * (self.high - self.low) + self.low

    def quantize(self, chunk):
        """chunk [H, A] in dataset units -> [H*A] integers, frequency-major."""
        coeffs = dct(self.normalize(chunk), type=2, norm="ortho", axis=0)
        ints = np.rint(coeffs * self.scale).astype(np.int64)
        return np.clip(ints, -self.max_coeff, self.max_coeff).ravel() + self.max_coeff

    def dequantize(self, ints):
        ints = np.asarray(ints, dtype=np.int64) - self.max_coeff
        grid = np.zeros(self.horizon * self.action_dim, dtype=np.float64)
        n = min(len(ints), grid.size)
        grid[:n] = ints[:n]
        coeffs = grid.reshape(self.horizon, self.action_dim) / self.scale
        return self.denormalize(idct(coeffs, type=2, norm="ortho", axis=0))

    # ------------------------------------------------------------------- fit
    def fit(self, chunks, q=(1.0, 99.0)):
        """chunks [N, H, A] in dataset units."""
        chunks = np.asarray(chunks, dtype=np.float64)
        if chunks.ndim != 3 or chunks.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected chunks [N, {self.horizon}, {self.action_dim}]")
        flat = chunks.reshape(-1, self.action_dim)
        self.low, self.high = np.percentile(flat, q[0], axis=0), np.percentile(flat, q[1], axis=0)
        span = self.high - self.low
        self.high = np.where(span < 1e-6, self.low + 1.0, self.high)
        corpus = [_to_symbols(self.quantize(chunk)) for chunk in chunks]
        alphabet = [chr(SYMBOL_BASE + i) for i in range(2 * self.max_coeff + 1)]
        self.bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
        self.bpe.decoder = decoders.Fuse()
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]"],
                                      initial_alphabet=alphabet, limit_alphabet=len(alphabet) + 1, show_progress=False)
        self.bpe.train_from_iterator(corpus, trainer=trainer)
        return self

    # -------------------------------------------------------------- encode/decode
    def encode(self, chunk) -> list[int]:
        if self.bpe is None:
            raise RuntimeError("fit() or load() the tokenizer first")
        return self.bpe.encode(_to_symbols(self.quantize(chunk))).ids

    def decode(self, ids) -> np.ndarray:
        text = self.bpe.decode([int(i) for i in ids], skip_special_tokens=True)
        return self.dequantize(_from_symbols(text))

    # ------------------------------------------------------------------ io
    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fast.json").write_text(json.dumps({
            "horizon": self.horizon, "action_dim": self.action_dim, "vocab_size": self.vocab_size,
            "scale": self.scale, "max_coeff": self.max_coeff, "low": self.low.tolist(), "high": self.high.tolist()}))
        self.bpe.save(str(path / "fast_bpe.json"))

    @classmethod
    def load(cls, path) -> "FastTokenizer":
        path = Path(path)
        raw = json.loads((path / "fast.json").read_text())
        tok = cls(raw["horizon"], raw["action_dim"], raw["vocab_size"], raw["scale"], raw["max_coeff"])
        tok.low, tok.high = np.asarray(raw["low"]), np.asarray(raw["high"])
        tok.bpe = Tokenizer.from_file(str(path / "fast_bpe.json"))
        return tok


def fast_to_vocab(fast_ids, vocab_size: int, skip: int):
    """FAST id k -> text vocab id vocab_size - 1 - skip - k (openpi's convention)."""
    return [vocab_size - 1 - skip - int(k) for k in fast_ids]


def vocab_to_fast(vocab_ids, vocab_size: int, skip: int, fast_vocab: int):
    out = []
    for v in vocab_ids:
        k = vocab_size - 1 - skip - int(v)
        if 0 <= k < fast_vocab:
            out.append(k)
    return out


class UniversalFast:
    """π0.5's pretrained FAST tokenizer (physical-intelligence/fast, files in weights/fast_universal).

    Same recipe as FastTokenizer but with a universal BPE: the chunk must already be normalized to [-1, 1]
    (π0.5 uses 1st/99th-percentile scaling, as our Normalizer does), DCT along time, x scale, rounded,
    flattened frequency-major and offset by min_token before BPE.
    """

    def __init__(self, path):
        path = Path(path)
        raw = json.loads((path / "processor_config.json").read_text())
        self.scale, self.min_token, self.vocab_size = raw["scale"], raw["min_token"], raw["vocab_size"]
        self.bpe = Tokenizer.from_file(str(path / "tokenizer.json"))

    def encode(self, normalized_chunk) -> list[int]:
        coeffs = np.around(dct(np.asarray(normalized_chunk, dtype=np.float64), axis=0, norm="ortho") * self.scale)
        text = "".join(map(chr, np.maximum(coeffs.flatten() - self.min_token, 0).astype(int)))
        return self.bpe.encode(text).ids
