"""Minimal Okapi BM25 (no external dependency, incremental adds)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tf: List[Counter] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.postings: Dict[str, List[int]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, docs_tokens: Sequence[Sequence[str]]) -> None:
        for toks in docs_tokens:
            idx = len(self.doc_len)
            tf = Counter(toks)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            for t in tf:
                self.df[t] += 1
                self.postings[t].append(idx)

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        n = len(self.doc_len)
        out = np.zeros(n, dtype=np.float64)
        if n == 0:
            return out
        avgdl = (sum(self.doc_len) / n) or 1.0
        for t in set(query_tokens):
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in self.postings[t]:
                f = self.doc_tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out
