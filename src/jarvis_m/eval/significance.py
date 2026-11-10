"""Significance tests for ranking comparisons.

Questions inside one conversation share a graph, so they are not independent: the
bootstrap resamples *conversations* (clusters), never single questions. An arm trained
with several seeds passes one score vector per seed; each replicate then also resamples
the seeds of each arm, so the interval covers both question and training noise.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


def _as_2d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("scores must be one vector, or one vector per seed")
    return a


def _cluster_sums(a: np.ndarray, codes: np.ndarray, n_clusters: int) -> np.ndarray:
    out = np.zeros((a.shape[0], n_clusters))
    for s in range(a.shape[0]):
        np.add.at(out[s], codes, a[s])
    return out


def cluster_paired_bootstrap(
    a,
    b,
    clusters: Sequence[Hashable],
    n: int = 10000,
    seed: int = 0,
    strata: Optional[Sequence[Hashable]] = None,
) -> Dict[str, float]:
    """Mean difference ``a - b`` with a 95% CI and a two-sided bootstrap p-value.

    ``a`` and ``b`` are per-question scores over the same questions, in the same order:
    one vector, or one vector per seed (``(n_seeds, n_questions)``; the two arms may have
    different numbers of seeds). ``clusters`` gives the conversation of every question.

    Each replicate draws conversations with replacement and, independently per arm,
    seeds with replacement. The statistic is the mean over the drawn questions. With
    ``strata`` (one label per question, e.g. the domain), conversations are drawn inside
    each stratum and the statistic is the unweighted mean of the per-stratum means
    (the macro-average).
    """
    A, B = _as_2d(a), _as_2d(b)
    if A.shape[1] != B.shape[1] or A.shape[1] != len(clusters):
        raise ValueError("a, b and clusters must cover the same questions")
    nq = A.shape[1]
    if nq == 0:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p": 1.0,
                "n_questions": 0, "n_clusters": 0}
    strata = list(strata) if strata is not None else [0] * nq
    if len(strata) != nq:
        raise ValueError("strata must have one label per question")
    rng = np.random.default_rng(seed)
    groups: Dict[Hashable, List[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    boots = np.zeros(n)
    point = 0.0
    n_clusters_total = 0
    for s in sorted(groups, key=str):
        idx = np.asarray(groups[s])
        # a cluster id only means something inside its stratum
        uniq: Dict[Hashable, int] = {}
        codes = np.asarray([uniq.setdefault(clusters[i], len(uniq)) for i in idx])
        nc = len(uniq)
        n_clusters_total += nc
        cnt = np.bincount(codes, minlength=nc).astype(np.float64)
        Sa, Sb = _cluster_sums(A[:, idx], codes, nc), _cluster_sums(B[:, idx], codes, nc)
        point += (Sa.mean(0).sum() - Sb.mean(0).sum()) / cnt.sum()
        done = 0
        step = max(1, min(n, 4_000_000 // max(1, nc)))  # bound the (replicates x clusters) block
        while done < n:
            m = min(step, n - done)
            C = rng.multinomial(nc, np.full(nc, 1.0 / nc), size=m).astype(np.float64)  # cluster draw counts
            Wa = rng.multinomial(A.shape[0], np.full(A.shape[0], 1.0 / A.shape[0]), size=m) / A.shape[0]
            Wb = rng.multinomial(B.shape[0], np.full(B.shape[0], 1.0 / B.shape[0]), size=m) / B.shape[0]
            num = ((Wa @ Sa - Wb @ Sb) * C).sum(1)
            boots[done: done + m] += num / (C @ cnt)
            done += m
    k = len(groups)
    boots /= k
    point /= k
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # two-sided: how often the resampled difference falls on the other side of zero
    tail = min((boots <= 0).sum(), (boots >= 0).sum())
    p = min(1.0, 2.0 * (tail + 1) / (n + 1))
    return {"delta": float(point), "ci_low": float(lo), "ci_high": float(hi), "p": float(p),
            "n_questions": int(nq), "n_clusters": int(n_clusters_total)}


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=np.float64)
    m = len(p)
    out = np.zeros(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p, kind="stable")):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return [float(x) for x in out]


def non_inferior(delta_ci_low: float, margin: float) -> bool:
    """True when the lower CI bound of (arm - reference) is above ``-margin``."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    return bool(delta_ci_low > -margin)


def compare_arms(
    scores: Dict[str, Sequence[Sequence[float]]],
    clusters: Sequence[Hashable],
    reference: str,
    strata: Optional[Sequence[Hashable]] = None,
    n: int = 10000,
    seed: int = 0,
) -> List[Tuple[str, Dict[str, float]]]:
    """``[(arm, bootstrap result)]`` for every arm against ``reference`` (no correction)."""
    if reference not in scores:
        raise KeyError(f"reference arm {reference!r} not in {sorted(scores)}")
    return [(name, cluster_paired_bootstrap(vecs, scores[reference], clusters, n=n, seed=seed, strata=strata))
            for name, vecs in scores.items() if name != reference]
