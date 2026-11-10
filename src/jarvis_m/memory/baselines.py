"""Message-level baselines: full context, recency window, BM25, dense, hybrid.

All share the same knobs so they double as ablation arms:

* ``scope``: ``all`` | ``asker`` (intra-user only) | ``others`` (cross-user only)
* ``asker_boost``: multiply scores of the asker's own messages (soft intra/cross mix)
* ``recency_half_life_days``: event-time decay applied to retrieval scores
* ``order``: ``chronological`` (default) or ``score`` ordering inside the prompt
* ``query_with_asker``: prepend the asker id to the query (GroupMemBench protocol)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..embeddings import Embedder, build_embedder
from ..interfaces import MemorySystem, Retrieval, RetrievedItem
from ..text import ApproxTokenCounter, TokenCounter, pack, word_tokens
from ..types import Message
from .bm25 import BM25Index

_SCOPES = ("all", "asker", "others")


class MessageStoreMemory(MemorySystem):
    """Stores raw messages; subclasses provide a relevance ranking."""

    name = "store"

    def __init__(
        self,
        scope: str = "all",
        asker_boost: float = 1.0,
        recency_half_life_days: Optional[float] = None,
        order: str = "chronological",
        query_with_asker: bool = True,
        counter: Optional[TokenCounter] = None,
    ) -> None:
        if scope not in _SCOPES:
            raise ValueError(f"scope must be one of {_SCOPES}")
        self.scope = scope
        self.asker_boost = asker_boost
        self.half_life = recency_half_life_days
        self.order = order
        self.query_with_asker = query_with_asker
        self.counter = counter or ApproxTokenCounter()
        self.messages: List[Message] = []

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.messages = []

    def ingest(self, messages: Sequence[Message]) -> None:
        self.messages.extend(messages)
        self._on_ingest(messages)

    def _on_ingest(self, messages: Sequence[Message]) -> None:
        pass

    # -- ranking -----------------------------------------------------------
    def _raw_scores(self, query: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """One score per stored message. ``mask`` marks the messages that may be ranked:
        scores of the others are never used, and systems that rank (rank fusion) must rank
        the masked messages only, so hidden messages cannot shift visible ones."""
        raise NotImplementedError

    def query_text(self, query: str, asker: Optional[str]) -> str:
        return f"{asker} {query}" if (self.query_with_asker and asker) else query

    def prefetch_queries(self, pairs: Sequence[Tuple[str, Optional[str]]]) -> None:
        """Embed many (query, asker) pairs in one batch; no-op for lexical systems."""
        emb = getattr(getattr(self, "dense", self), "embedder", None)
        if emb is not None and pairs:
            emb.encode_queries([self.query_text(q, a) for q, a in pairs])

    def _candidates(self, asker: Optional[str]) -> np.ndarray:
        n = len(self.messages)
        if self.scope == "all" or asker is None:
            return np.ones(n, dtype=bool)
        is_asker = np.array([m.speaker == asker for m in self.messages], dtype=bool)
        return is_asker if self.scope == "asker" else ~is_asker

    def _adjust(self, scores: np.ndarray, asker: Optional[str], now: Optional[float]) -> np.ndarray:
        s = scores.astype(np.float64).copy()
        if asker is not None and self.asker_boost != 1.0:
            mask = np.array([m.speaker == asker for m in self.messages])
            s[mask] *= self.asker_boost
        if self.half_life and now is not None:
            lam = math.log(2) / (self.half_life * 86400.0)
            ts = np.array([m.timestamp if m.timestamp is not None else now for m in self.messages])
            s *= np.exp(-lam * np.clip(now - ts, 0, None))
        return s

    def rank(self, query: str, asker: Optional[str] = None, now: Optional[float] = None) -> List[Tuple[int, float]]:
        if not self.messages:
            return []
        cand = self._candidates(asker)
        s = self._adjust(self._raw_scores(self.query_text(query, asker), cand), asker, now)
        idx = np.nonzero(cand)[0]
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return [(int(i), float(s[i])) for i in idx]

    def retrieve(self, query, *, asker=None, budget_tokens=None, max_items=None, now=None) -> Retrieval:
        ranked = self.rank(query, asker, now)
        kept, _, _ = pack(
            ranked, lambda r: self.messages[r[0]].render(), self.counter, budget_tokens, max_items
        )
        if self.order == "chronological":
            kept = sorted(kept, key=lambda r: r[0])
        items = [self._item(i, sc) for i, sc in kept]
        ctx = "\n\n".join(it.text for it in items)
        return Retrieval(items, ctx, self.counter.count(ctx), {"candidates": len(ranked)})

    def _item(self, i: int, score: float) -> RetrievedItem:
        m = self.messages[i]
        return RetrievedItem(m.render(), score, (m.id,), "message", m.timestamp)

    def stats(self) -> Dict[str, int]:
        return {"messages": len(self.messages)}


class RecencyWindow(MessageStoreMemory):
    """Most recent messages that fit the budget (sliding-window baseline)."""

    name = "recency"

    def _raw_scores(self, query: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return np.arange(len(self.messages), dtype=np.float64)


class FullContext(RecencyWindow):
    """Everything, chronologically. Ignores the budget (for long-context readers);
    reports overflow so runs can be flagged."""

    name = "full_context"

    def retrieve(self, query, *, asker=None, budget_tokens=None, max_items=None, now=None) -> Retrieval:
        r = super().retrieve(query, asker=asker, budget_tokens=None, max_items=None, now=now)
        r.info["over_budget"] = bool(budget_tokens is not None and r.n_tokens > budget_tokens)
        return r


class BM25Memory(MessageStoreMemory):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, **kw) -> None:
        super().__init__(**kw)
        self.k1, self.b = k1, b
        self.index = BM25Index(k1, b)

    def reset(self) -> None:
        super().reset()
        self.index = BM25Index(self.k1, self.b)

    def _on_ingest(self, messages):
        self.index.add([word_tokens(m.text) for m in messages])

    def _raw_scores(self, query: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.index.scores(word_tokens(query))


class DenseMemory(MessageStoreMemory):
    name = "dense"

    def __init__(self, embedder: Optional[dict] = None, embed_header: bool = False, **kw) -> None:
        super().__init__(**kw)
        self.embedder: Embedder = build_embedder(embedder)
        self.embed_header = embed_header
        self._chunks: List[np.ndarray] = []
        self._mat: Optional[np.ndarray] = None

    def reset(self) -> None:
        super().reset()
        self._chunks, self._mat = [], None

    def _on_ingest(self, messages):
        texts = [m.render() if self.embed_header else m.text for m in messages]
        self._chunks.append(self.embedder.encode_docs(texts))
        self._mat = None

    @property
    def matrix(self) -> np.ndarray:
        if self._mat is None:
            self._mat = np.vstack(self._chunks) if self._chunks else np.zeros((0, self.embedder.dim), np.float32)
        return self._mat

    def _raw_scores(self, query: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        q = self.embedder.encode_queries([query])[0]
        if mask is None or mask.all():
            return self.matrix @ q
        # score the visible rows only: the result is then bit-identical to a store that
        # never held the hidden messages
        out = np.zeros(len(self.messages), dtype=np.float32)
        idx = np.nonzero(mask)[0]
        out[idx] = self.matrix[idx] @ q
        return out


class HybridMemory(MessageStoreMemory):
    """Reciprocal-rank fusion of BM25 and dense rankings."""

    name = "hybrid"

    def __init__(self, embedder: Optional[dict] = None, rrf_k: int = 60, bm25_weight: float = 1.0,
                 dense_weight: float = 1.0, **kw) -> None:
        super().__init__(**kw)
        inner = dict(scope="all", query_with_asker=False, counter=self.counter)
        self.bm25 = BM25Memory(**inner)
        self.dense = DenseMemory(embedder=embedder, **inner)
        self.rrf_k, self.wb, self.wd = rrf_k, bm25_weight, dense_weight

    def reset(self) -> None:
        super().reset()
        self.bm25.reset()
        self.dense.reset()

    def _on_ingest(self, messages):
        self.bm25.ingest(messages)
        self.dense.ingest(messages)

    def _raw_scores(self, query: str, mask: Optional[np.ndarray] = None) -> np.ndarray:
        n = len(self.messages)
        idx = np.arange(n) if mask is None else np.nonzero(mask)[0]
        out = np.zeros(n, dtype=np.float64)
        for w, s in ((self.wb, self.bm25._raw_scores(query, mask)), (self.wd, self.dense._raw_scores(query, mask))):
            ranks = np.empty(len(idx), dtype=np.int64)  # ranks among the visible messages only
            ranks[np.argsort(-s[idx], kind="stable")] = np.arange(len(idx))
            out[idx] += w / (self.rrf_k + ranks + 1)
        return out
