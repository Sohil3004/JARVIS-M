"""Learned hypergraph scorer (HGNN). Plain PyTorch, no torch-geometric.

Per layer (UniGNN-style, edge-type aware, query-gated):
    m_e   = mean_{u in e} W_v h_u + type_emb[t_e]
    g_e   = sigmoid(<m_e, q> / sqrt(d))                  (``gating``: which hyperedges matter for *this* query)
    a_v   = mean_{e ni v} g_e * m_e
    h_v  <- LayerNorm(h_v + Dropout(ReLU(W_o [h_v ; a_v])))
score_v = MLP([h_v ; q ; h_v * q ; f_v]) + w_base * base_v  (residual on the seed score)

``layers: 0`` is the no-message-passing ablation (same features, MLP only).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover - torch is an optional extra
    raise ImportError("the HGNN scorer needs PyTorch: pip install -e .[llm]") from e

from .graph import EDGE_TYPES
from .subgraph import FEATURE_NAMES, Subgraph


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[-1], device=src.device, dtype=src.dtype)
    cnt = torch.zeros(n, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    cnt.index_add_(0, index, torch.ones(index.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / cnt.clamp(min=1.0)


class HGNN(nn.Module):
    def __init__(self, emb_dim: int, n_feats: int = len(FEATURE_NAMES), hidden: int = 128, layers: int = 2,
                 gating: bool = True, dropout: float = 0.1, n_types: int = len(EDGE_TYPES)) -> None:
        super().__init__()
        self.cfg = dict(emb_dim=emb_dim, n_feats=n_feats, hidden=hidden, layers=layers, gating=gating,
                        dropout=dropout, n_types=n_types)
        self.inp = nn.Sequential(nn.Linear(emb_dim + n_feats, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.qp = nn.Linear(emb_dim, hidden)
        self.type_emb = nn.Embedding(n_types, hidden)
        self.v2e = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.e2v = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.out = nn.Sequential(nn.Linear(3 * hidden + n_feats, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.base_w = nn.Parameter(torch.tensor(1.0))
        self.gating = gating

    def forward(self, x_emb, feats, q_emb, nl, ei, et):
        n = x_emb.shape[0]
        h = self.inp(torch.cat([x_emb, feats], -1))
        q = self.qp(q_emb).unsqueeze(0)  # (1, d)
        n_edges = int(et.shape[0])
        for v2e, e2v, norm in zip(self.v2e, self.e2v, self.norms):
            if n_edges == 0:
                break
            m = _scatter_mean(v2e(h)[nl], ei, n_edges) + self.type_emb(et)
            if self.gating:
                m = m * torch.sigmoid((m * q).sum(-1, keepdim=True) / math.sqrt(m.shape[-1]))
            a = _scatter_mean(m[ei], nl, n)
            h = norm(h + self.drop(torch.relu(e2v(torch.cat([h, a], -1)))))
        qq = q.expand(n, -1)
        return self.out(torch.cat([h, qq, h * qq, feats], -1)).squeeze(-1) + self.base_w * feats[:, 0]


def to_tensors(sg: Subgraph, x_emb: np.ndarray, q_emb: np.ndarray, device: str = "cpu"):
    nl, ei, et = sg.incidence()
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    return (T(x_emb), T(sg.feats), T(q_emb), T(nl, torch.long), T(ei, torch.long), T(et, torch.long))


def resolve_device(device: Optional[str]) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class HGNNScorer:
    """Callable used by HypergraphMemory: ``scores = scorer(subgraph, (memory, query, asker))``."""

    def __init__(self, model: HGNN, device: str = "cpu", meta: Optional[Dict[str, Any]] = None) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.meta = meta or {}

    @classmethod
    def load(cls, checkpoint: str, device: Optional[str] = None) -> HGNNScorer:
        dev = resolve_device(device)
        ck = torch.load(Path(checkpoint), map_location=dev, weights_only=False)
        if tuple(ck.get("feature_names", FEATURE_NAMES)) != tuple(FEATURE_NAMES):
            raise ValueError("checkpoint was trained with a different feature set; retrain")
        model = HGNN(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, dev, ck.get("meta"))

    @torch.no_grad()
    def __call__(self, sg: Subgraph, ctx) -> np.ndarray:
        mem, query, asker = ctx
        x = mem.node_embeddings(sg.nodes)
        if x.shape[1] != self.model.cfg["emb_dim"]:
            raise ValueError(f"embedder dim {x.shape[1]} != checkpoint emb_dim {self.model.cfg['emb_dim']}: "
                             "use the embedder the HGNN was trained with")
        q = mem.query_embedding(query, asker)
        return self.model(*to_tensors(sg, x, q, self.device)).float().cpu().numpy()


_LOADED: Dict[tuple, HGNNScorer] = {}


def load_scorer(checkpoint: str, device: Optional[str] = None) -> HGNNScorer:
    """Load once per (path, device): memories are rebuilt per conversation."""
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"HGNN checkpoint {checkpoint} not found: run `jarvis-eval train-hgnn ...` first")
    key = (str(Path(checkpoint).resolve()), resolve_device(device))
    if key not in _LOADED:
        _LOADED[key] = HGNNScorer.load(checkpoint, device)
    return _LOADED[key]
