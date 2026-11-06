"""Tokenisation helpers: token counting for budgets, word tokens for BM25."""

from __future__ import annotations

import re
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ALNUM = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an the and or but if of to in on at by for with from as is are was were be been being "
    "it its this that these those i you he she we they me him her us them my your our their "
    "do does did done have has had not no so than then there here what which who whom when "
    "where why how all any can could should would will just about into over also".split()
)


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """Dependency-free counter (~1.3 subword tokens per word piece). Used in tests
    and as a default; switch to :class:`HFTokenCounter` for paper numbers."""

    def __init__(self, ratio: float = 1.3) -> None:
        self.ratio = ratio

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(round(len(_WORD.findall(text)) * self.ratio)))


class HFTokenCounter:
    """Exact counts with the reader model's tokenizer (lazy import)."""

    def __init__(self, name: str, cache_dir: Optional[str] = None) -> None:
        from transformers import AutoTokenizer  # noqa: WPS433

        self.tok = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))


def word_tokens(text: str, drop_stopwords: bool = True) -> List[str]:
    toks = _ALNUM.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return toks


def pack(
    items: Sequence,
    render: Callable[[object], str],
    counter: TokenCounter,
    budget_tokens: Optional[int],
    max_items: Optional[int] = None,
    separator: str = "\n\n",
) -> Tuple[List, str, int]:
    """Greedily keep items (in the given priority order) until the budget is hit.

    Items that do not fit are skipped (a later, shorter item may still fit).
    Returns (kept_items, joined_text, n_tokens).
    """
    kept, parts, used = [], [], 0
    sep_cost = counter.count(separator.strip()) if separator.strip() else 0
    for it in items:
        if max_items is not None and len(kept) >= max_items:
            break
        txt = render(it)
        cost = counter.count(txt) + (sep_cost if parts else 0)
        if budget_tokens is not None and used + cost > budget_tokens:
            continue
        kept.append(it)
        parts.append(txt)
        used += cost
    return kept, separator.join(parts), used


def join_count(texts: Iterable[str], counter: TokenCounter, separator: str = "\n\n") -> Tuple[str, int]:
    s = separator.join(texts)
    return s, counter.count(s)
