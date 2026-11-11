"""Minimal Okapi BM25 (no external dependency, incremental adds)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

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

    def scores(self, query_tokens: Sequence[str], mask: Optional[np.ndarray] = None) -> np.ndarray:
        """BM25 score per document. With ``mask``, the collection statistics (document
        count, document frequency, average length) are computed over the masked documents
        only, and only those are scored: the result is what an index holding just the
        masked documents would return."""
        total = len(self.doc_len)
        out = np.zeros(total, dtype=np.float64)
        if total == 0:
            return out
        if mask is None:
            n, length = total, sum(self.doc_len)
        else:
            idx = np.nonzero(mask)[0]
            n, length = len(idx), sum(self.doc_len[i] for i in idx)
            if n == 0:
                return out
        avgdl = (length / n) or 1.0
        for t in dict.fromkeys(query_tokens):
            posting = self.postings.get(t, ())
            if mask is not None:
                posting = [i for i in posting if mask[i]]
            df = len(posting)
            if not df:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in posting:
                f = self.doc_tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / avgdl)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out
