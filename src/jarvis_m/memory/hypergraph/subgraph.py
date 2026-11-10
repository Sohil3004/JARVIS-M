"""Query-time candidate subgraph shared by the fixed propagator and the HGNN."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .graph import EDGE_TYPES, Hypergraph

MODALITY_FEATS = ("decision", "commitment", "done", "blocker", "negated", "proposal", "question", "info")
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|myself|i'm|i've|i'll)\b", re.I)
HISTORICAL = re.compile(
    r"\b(?:before|previous(?:ly)?|original(?:ly)?|initial(?:ly)?|at first|first (?:decided|chose|planned)|"
    r"used to|earlier|old|formerly|changed from|prior|was (?:it|the) (?:plan|decision)|history)\b", re.I
)
FEATURE_NAMES = (
    ["base", "base_rank", "is_seed", "is_asker", "mentions_asker", "superseded", "is_current", "in_chain",
     "recency", "log_degree", "historical_query"]
    + [f"mod_{m}" for m in MODALITY_FEATS]
    + ["first_person_query"]
)


@dataclass
class Subgraph:
    nodes: np.ndarray  # global node ids (n,)
    feats: np.ndarray  # (n, F) float32
    base: np.ndarray  # (n,) seed score in [0, 1]
    edges: List[tuple] = field(default_factory=list)  # (type, local idx array)
    historical: bool = False
    labels: Optional[np.ndarray] = None  # (n,) 1 for gold evidence (training only)
    n_participants: int = 0  # distinct non-assistant speakers in the conversation (not a feature)

    def incidence(self):
        """(node_local, edge_idx, edge_type_idx) arrays for scatter ops."""
        nl, ei, et = [], [], []
        for j, (t, idx) in enumerate(self.edges):
            nl.extend(idx.tolist())
            ei.extend([j] * len(idx))
            et.append(EDGE_TYPES.index(t))
        return np.array(nl, np.int64), np.array(ei, np.int64), np.array(et, np.int64)


def build_subgraph(
    g: Hypergraph,
    base_all: np.ndarray,
    seeds: Sequence[int],
    asker: Optional[str],
    query: str,
    now: Optional[float],
    expand_weights: Dict[str, float],
    max_edge_size: int = 60,
    max_candidates: int = 300,
    keep_edge_types: Sequence[str] = EDGE_TYPES,
) -> Subgraph:
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    cand: Dict[int, float] = {s: 2.0 for s in seeds}
    nb = g.neighbours(seeds, expand_weights, max_edge_size)
    # version pull-up: the current value of any touched slot is always a candidate
    for s in seeds:
        cur = g.current_of(s)
        if cur is not None:
            nb[cur] = max(nb.get(cur, 0.0), 1.5)
    if asker and FIRST_PERSON.search(query or ""):
        # "what am I responsible for?": the asker's own state-bearing messages are candidates
        own = [v for v in range(len(g.messages) - 1, -1, -1)
               if g.messages[v].speaker == asker and g.chain_of[v] >= 0][:20]
        for v in own:
            nb[v] = max(nb.get(v, 0.0), 1.2)
    ranked = sorted(nb.items(), key=lambda kv: (-kv[1], -base_all[kv[0]]))
    for u, w in ranked:
        if len(cand) >= max_candidates:
            break
        cand.setdefault(u, w)
    nodes = np.array(sorted(cand), dtype=np.int64)
    loc = {int(v): i for i, v in enumerate(nodes)}

    base = base_all[nodes].astype(np.float32)
    order = np.argsort(-base_all)
    rank = np.empty(len(base_all), np.int64)
    rank[order] = np.arange(len(base_all))
    hist = bool(HISTORICAL.search(query or ""))
    first_person = bool(asker and FIRST_PERSON.search(query or ""))
    ts = np.array([g.messages[v].timestamp or 0.0 for v in nodes], np.float64)
    ref = now if now is not None else (ts.max() if len(ts) else 0.0)
    age_days = np.clip((ref - ts) / 86400.0, 0, None)
    asker_tok = (asker or "").lower()
    seed_set = set(seeds)
    F = np.zeros((len(nodes), len(FEATURE_NAMES)), np.float32)
    for i, v in enumerate(nodes):
        m = g.messages[v]
        recs = g.records[v]
        ents = {e for r in recs for e in r.entities}
        F[i, 0] = base[i]
        F[i, 1] = 1.0 / (1.0 + rank[v])
        F[i, 2] = float(int(v) in seed_set)
        F[i, 3] = float(asker is not None and m.speaker == asker)
        F[i, 4] = float(bool(asker_tok) and (asker_tok in ents or asker_tok in m.text.lower()))
        F[i, 5] = float(g.superseded[v])
        F[i, 6] = float(g.chain_of[v] >= 0 and not g.superseded[v])
        F[i, 7] = float(g.chain_of[v] >= 0)
        F[i, 8] = float(np.exp(-age_days[i] / 30.0))
        F[i, 9] = float(np.log1p(len(g.node_edges[v])))
        F[i, 10] = float(hist)
        mod = g.modality[v]
        if mod in MODALITY_FEATS:
            F[i, 11 + MODALITY_FEATS.index(mod)] = 1.0
        F[i, -1] = float(first_person)

    edges = []
    seen = set()
    keep = set(keep_edge_types)
    for v in nodes:
        for e in g.node_edges[v]:
            if e in seen:
                continue
            seen.add(e)
            he = g.edges[e]
            if he.type not in keep or len(he.nodes) > max_edge_size:
                continue
            idx = np.array([loc[u] for u in he.nodes if u in loc], np.int64)
            if len(idx) >= 2:
                edges.append((he.type, idx))
    return Subgraph(nodes, F, base, edges, hist, n_participants=len(g.participants))


def fixed_propagation(
    sg: Subgraph,
    edge_weights: Dict[str, float],
    beta: float = 0.2,
    iters: int = 1,
    superseded_penalty: float = 0.35,
    version_pull: float = 0.95,
    asker_bonus: float = 0.0,
    own_state_bonus: float = 0.5,
    min_participants: int = 3,
) -> np.ndarray:
    """Fixed (unlearned) hypergraph scoring.

    Neighbour-evidence boost: for each edge type t, a node gets the best mean
    score of its *other* members over its type-t edges; the boost is the
    weighted average of those, scaled by ``beta`` and added to the seed score.
    Additive (not a convex mix) so strong direct hits are never smoothed away.
    Then version logic: current values are pulled up to their best chain-mate,
    superseded values are demoted (skipped for historical queries).

    ``asker_bonus`` is off by default: in user<->assistant chats it boosts every user
    turn over the assistant turn that holds the answer. ``own_state_bonus`` ("what am I
    responsible for?") only applies to multi-party chats (``min_participants`` distinct
    non-assistant speakers).
    """
    base = sg.base.astype(np.float64)
    s = base.copy()
    n = len(s)
    for _ in range(iters):
        best = {}
        for t, idx in sg.edges:
            w = edge_weights.get(t, 0.0)
            if w <= 0 or len(idx) < 2:
                continue
            tot = s[idx].sum()
            others = (tot - s[idx]) / (len(idx) - 1)
            b = best.setdefault(t, np.zeros(n))
            np.maximum.at(b, idx, others)
        wsum = sum(edge_weights.get(t, 0.0) for t in best) or 1.0
        boost = sum(edge_weights[t] * b for t, b in best.items()) / wsum if best else np.zeros(n)
        s = base + beta * boost
    f = sg.feats
    if not sg.historical:
        pulled = s.copy()
        for t, idx in sg.edges:
            if t != "version":
                continue
            top = s[idx].max()
            cur = idx[f[idx, 6] > 0]
            pulled[cur] = np.maximum(pulled[cur], version_pull * top)
        s = np.where(f[:, 5] > 0, pulled * superseded_penalty, pulled)
    s = s + asker_bonus * np.maximum(f[:, 3], f[:, 4]) * (s > 0)
    # first-person questions in group chats: the asker's own current commitments/decisions
    if sg.n_participants >= min_participants:
        s = s + own_state_bonus * f[:, -1] * f[:, 3] * f[:, 6]
    return s
