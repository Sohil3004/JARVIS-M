"""Answer metrics, failure-mode metrics and the statistics we report."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ABSTAIN = re.compile(
    r"(not (mentioned|available|specified|stated|provided|found|supported|in the (conversation|passages|context))"
    r"|no (information|mention|record|evidence)|(does not|doesn't|do not|don't) (say|mention|specify|know|have)"
    r"|cannot (be )?(determine|answer|find)|can't (determine|answer|find)|unknown|i don't know|unanswerable"
    r"|insufficient information|there is no)",
    re.I,
)


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    k = sum(common.values())
    if k == 0:
        return 0.0
    prec, rec = k / len(p), k / len(g)
    return 2 * prec * rec / (prec + rec)


def is_abstention(text: str) -> bool:
    return bool(_ABSTAIN.search(text or ""))


def mentions_any(text: str, candidates: Iterable[str]) -> bool:
    t = normalize(text)
    return any(normalize(c) and normalize(c) in t for c in candidates)


def evidence_recall(retrieved: Sequence[str], gold: Sequence[str]) -> Optional[float]:
    if not gold:
        return None
    r = set(retrieved)
    return sum(1 for g in gold if g in r) / len(gold)


# ---------------------------------------------------------------- statistics
def bootstrap_ci(x: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float, float]:
    a = np.asarray([v for v in x if v is not None], dtype=float)
    if a.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(a.mean()), float(lo), float(hi)


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 5000, seed: int = 0) -> Tuple[float, float, float, float]:
    """Mean difference a-b with 95% CI and a two-sided bootstrap p-value."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    assert x.shape == y.shape, "paired samples must align"
    d = x - y
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p))


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> Tuple[int, int, float]:
    """Exact McNemar test on paired correctness. Returns (a_only, b_only, p)."""
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return a_only, b_only, float(min(1.0, 2 * p))
