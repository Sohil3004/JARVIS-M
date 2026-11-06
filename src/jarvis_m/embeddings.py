"""Text embedders. All return L2-normalised float32 arrays.

``HashingEmbedder`` is deterministic and dependency-free (tests, CPU smoke runs).
``SentenceTransformerEmbedder`` wraps any sentence-transformers model, including
Qwen3-Embedding (which wants a query instruction prompt).
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, Optional, Protocol, Sequence

import numpy as np

from .text import word_tokens


class Embedder(Protocol):
    dim: int

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalise(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (x / n).astype(np.float32)


class HashingEmbedder:
    """Bag of hashed word unigrams + bigrams. Stable across processes (md5,
    not Python's salted ``hash``). Good enough to rank lexical overlap."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, tok: str) -> int:
        return int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little") % self.dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = word_tokens(text)
        for t in toks:
            v[self._bucket(t)] += 1.0
        for a, b in zip(toks, toks[1:]):
            v[self._bucket(a + "_" + b)] += 0.5
        return v

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._one(t) for t in texts]))

    encode_queries = encode_docs


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        batch_size: int = 64,
        query_prompt_name: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_dir or os.environ.get("JARVIS_CACHE_DIR")
        )
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.batch_size = batch_size
        prompts = getattr(self.model, "prompts", {}) or {}
        if query_prompt_name == "auto":
            query_prompt_name = "query" if "query" in prompts else None
        self.query_prompt_name = query_prompt_name

    def _enc(self, texts: Sequence[str], **kw) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        x = self.model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=len(texts) > 2000, **kw,
        )
        return x.astype(np.float32)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_prompt_name:
            return self._enc(texts, prompt_name=self.query_prompt_name)
        return self._enc(texts)


_CACHE: Dict[str, Embedder] = {}


def build_embedder(cfg: Optional[dict]) -> Embedder:
    """Embedders are expensive to load; share one instance per config."""
    cfg = dict(cfg or {"type": "hashing"})
    key = repr(sorted(cfg.items()))
    if key in _CACHE:
        return _CACHE[key]
    kind = cfg.pop("type", "hashing")
    if kind == "hashing":
        emb: Embedder = HashingEmbedder(**cfg)
    elif kind in ("st", "sentence_transformers"):
        emb = SentenceTransformerEmbedder(**cfg)
    else:
        raise ValueError(f"unknown embedder type {kind!r}")
    _CACHE[key] = emb
    return emb
