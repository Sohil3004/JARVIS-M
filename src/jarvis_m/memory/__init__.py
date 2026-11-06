"""Memory-system registry. Add new systems (e.g. the hypergraph memory) here."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Dict

from ..interfaces import MemorySystem
from ..text import ApproxTokenCounter, HFTokenCounter
from .baselines import BM25Memory, DenseMemory, FullContext, HybridMemory, RecencyWindow

REGISTRY: Dict[str, Callable[..., MemorySystem]] = {
    "full_context": FullContext,
    "recency": RecencyWindow,
    "bm25": BM25Memory,
    "dense": DenseMemory,
    "hybrid": HybridMemory,
}


def _legacy_v2(**kw):
    from .legacy_v2 import LegacyV2Memory

    return LegacyV2Memory(**kw)


REGISTRY["legacy_v2"] = _legacy_v2


def register(name: str):
    def deco(factory):
        REGISTRY[name] = factory
        return factory

    return deco


@lru_cache(maxsize=8)
def build_counter(cfg: Any = None):
    """``approx`` (default, no deps) or an HF tokenizer name for exact counts."""
    if not cfg or cfg == "approx":
        return ApproxTokenCounter()
    if isinstance(cfg, str):
        return HFTokenCounter(cfg)
    raise ValueError(f"bad tokenizer config {cfg!r}")


def build_memory(cfg: Dict[str, Any], tokenizer: Any = None) -> MemorySystem:
    cfg = dict(cfg)
    kind = cfg.pop("type")
    label = cfg.pop("label", kind)
    if kind not in REGISTRY:
        raise KeyError(f"unknown memory type {kind!r}; known: {sorted(REGISTRY)}")
    mem = REGISTRY[kind](counter=build_counter(tokenizer), **cfg)
    mem.name = label
    return mem


__all__ = ["REGISTRY", "register", "build_memory", "build_counter"]
