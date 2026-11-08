"""Text embedders. All return L2-normalised float32 arrays.

``HashingEmbedder`` is deterministic and dependency-free (tests, CPU smoke runs).
``SentenceTransformerEmbedder`` wraps any sentence-transformers model, including
Qwen3-Embedding (which wants a query instruction prompt).
"""

from __future__ import annotations

import hashlib
import os
import re
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
    """sentence-transformers model with settings that fit an 8 GB GPU.

    ``max_seq_length`` matters: Qwen3-Embedding defaults to 32K tokens, and long
    turns (LongMemEval assistant messages) then exhaust VRAM. Texts are truncated
    for the *dense* index only; BM25 and the reader still see the full message.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_seq_length: Optional[int] = 1024,
        dtype: str = "auto",  # auto = fp16 on CUDA, fp32 on CPU
        query_prompt_name: Optional[str] = "auto",
        cache_dir: Optional[str] = None,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer  # lazy

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            dtype = "fp16" if dev.startswith("cuda") else "fp32"
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.model = SentenceTransformer(
            model_name, device=dev, cache_folder=cache_dir or os.environ.get("JARVIS_CACHE_DIR"),
            model_kwargs={"torch_dtype": torch_dtype},
        )
        if max_seq_length:
            self.model.max_seq_length = min(max_seq_length, self.model.max_seq_length or max_seq_length)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())
        self.batch_size = batch_size
        self.name = f"{model_name}@{self.model.max_seq_length}"
        prompts = getattr(self.model, "prompts", {}) or {}
        if query_prompt_name == "auto":
            query_prompt_name = "query" if "query" in prompts else None
        self.query_prompt_name = query_prompt_name
        self._torch = torch

    def _enc(self, texts: Sequence[str], **kw) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            x = self.model.encode(
                list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=len(texts) > 2000, **kw,
            )
        except (RuntimeError, getattr(self._torch, "AcceleratorError", RuntimeError)) as e:
            # one very long batch can still OOM: retry once with a tiny batch
            if "out of memory" not in str(e).lower() or self.batch_size <= 2:
                raise
            self._torch.cuda.empty_cache()
            x = self.model.encode(list(texts), batch_size=2, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False, **kw)
        return np.asarray(x, dtype=np.float32)

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self._enc(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self.query_prompt_name:
            return self._enc(texts, prompt_name=self.query_prompt_name)
        return self._enc(texts)


class CachedEmbedder:
    """Disk cache (SQLite, stdlib) in front of any embedder, keyed by model + text.

    Embedding a benchmark is the slow part of training/eval; with the cache, re-runs,
    ablation trainings and crash restarts skip everything already embedded.
    Queries also get an in-memory cache (each question is encoded more than once).
    """

    def __init__(self, inner: Embedder, path: str | os.PathLike, model_key: str) -> None:
        import sqlite3
        from pathlib import Path

        self.inner, self.dim, self.model_key = inner, inner.dim, model_key
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        self._q: Dict[str, np.ndarray] = {}
        self.hits = self.misses = 0

    def _key(self, kind: str, text: str) -> str:
        return hashlib.sha1(f"{self.model_key}\0{kind}\0{text}".encode()).hexdigest()

    def _get(self, kind: str, texts: Sequence[str], fn) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [self._key(kind, t) for t in texts]
        found: Dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 900):  # SQLite variable limit
            chunk = keys[i: i + 900]
            q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(chunk))})"
            for k, v in self.db.execute(q, chunk):
                found[k] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
        miss = [i for i, k in enumerate(keys) if k not in found]
        self.hits += len(keys) - len(miss)
        self.misses += len(miss)
        if miss:
            vecs = fn([texts[i] for i in miss])
            rows = [(keys[i], vecs[j].astype(np.float16).tobytes()) for j, i in enumerate(miss)]
            self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
            self.db.commit()
            for j, i in enumerate(miss):
                found[keys[i]] = vecs[j]
        return _normalise(np.stack([found[k] for k in keys]))

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self._get("d", texts, self.inner.encode_docs)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if texts and all(t in self._q for t in texts):
            return np.stack([self._q[t] for t in texts])
        out = self._get("q", texts, self.inner.encode_queries)
        if len(self._q) > 50000:
            self._q.clear()
        for t, v in zip(texts, out):
            self._q[t] = v
        return out


_CACHE: Dict[str, Embedder] = {}


def build_embedder(cfg: Optional[dict]) -> Embedder:
    """Embedders are expensive to load; share one instance per config.

    sentence-transformers embedders get a disk cache by default
    (``disk_cache: false`` to disable, ``disk_cache: <path>`` to choose the file).
    """
    cfg = dict(cfg or {"type": "hashing"})
    key = repr(sorted(cfg.items()))
    if key in _CACHE:
        return _CACHE[key]
    kind = cfg.pop("type", "hashing")
    disk = cfg.pop("disk_cache", kind in ("st", "sentence_transformers"))
    if kind == "hashing":
        emb: Embedder = HashingEmbedder(**cfg)
        model_key = f"hashing-{emb.dim}"
    elif kind in ("st", "sentence_transformers"):
        emb = SentenceTransformerEmbedder(**cfg)
        model_key = emb.name
    else:
        raise ValueError(f"unknown embedder type {kind!r}")
    if disk:
        from pathlib import Path

        path = Path(disk) if isinstance(disk, str) else (
            Path(os.environ.get("JARVIS_CACHE_DIR", "cache")) / "embeddings" /
            (re.sub(r"[^A-Za-z0-9_.-]+", "_", model_key) + ".sqlite"))
        emb = CachedEmbedder(emb, path, model_key)
    _CACHE[key] = emb
    return emb
