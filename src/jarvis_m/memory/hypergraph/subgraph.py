"""Query-time candidate subgraph shared by the fixed propagator and the HGNN.

Built from a :class:`GraphView`, so candidates, hyperedges and every feature come from
what the viewer can see and from nothing else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .graph import ASSISTANT, EDGE_TYPES, GraphView, Hyperedge, Hypergraph, state_role

MODALITY_FEATS = ("decision", "commitment", "done", "blocker", "negated", "proposal", "question", "info")
FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|myself|i'm|i've|i'll)\b", re.I)
HISTORICAL = re.compile(
    r"\b(?:before|previous(?:ly)?|original(?:ly)?|initial(?:ly)?|at first|first (?:decided|chose|planned)|"
    r"used to|earlier|old|formerly|changed from|prior|was (?:it|the) (?:plan|decision)|history)\b", re.I
)
# Size-invariant and role-aware (v2). Changing this list invalidates HGNN checkpoints.
FEATURE_NAMES = (
    ["base", "base_rank", "is_seed", "is_asker", "mentions_asker", "superseded", "is_current", "in_chain",
     "age_frac", "log_age_days", "log_degree", "historical_query"]
    + [f"mod_{m}" for m in MODALITY_FEATS]
    + ["first_person_query", "is_assistant", "query_mentions_speaker", "node_mentions_query_person",
       "log_n_participants"]
)
FI = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class Subgraph:
    nodes: np.ndarray  # global node ids (n,)
    feats: np.ndarray  # (n, F) float32
    base: np.ndarray  # (n,) seed score in [0, 1]
    edges: List[tuple] = field(default_factory=list)  # (type, local idx array)
    historical: bool = False
    labels: Optional[np.ndarray] = None  # (n,) 1 for gold evidence (training only)
    n_participants: int = 0  # distinct non-assistant speakers the viewer can see

    def incidence(self):
        """(node_local, edge_idx, edge_type_idx) arrays for scatter ops."""
        nl, ei, et = [], [], []
        for j, (t, idx) in enumerate(self.edges):
            nl.extend(idx.tolist())
            ei.extend([j] * len(idx))
            et.append(EDGE_TYPES.index(t))
        return np.array(nl, np.int64), np.array(ei, np.int64), np.array(et, np.int64)


def _version_keep(members: Sequence[int], current: Optional[int], k: int) -> List[int]:
    """Long version chains keep the current value, the first value and the last ``k`` (time order)."""
    keep = {members[0], *members[-k:]} if k > 0 else {members[0]}
    if current is not None:
        keep.add(current)
    return [v for v in members if v in keep]


def _top_by_base(members: np.ndarray, base: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(-base[members], kind="stable")[:k]
    return members[np.sort(order)]


def _expansion_members(view: GraphView, he: Hyperedge, base: np.ndarray, max_edge_size: int,
                       version_keep_k: int) -> Sequence[int]:
    """Which members of an edge a seed pulls in. Small edges: all. Large edges: the top
    ``max_edge_size`` by seed score; version edges: current + first + last K, always."""
    if len(he.nodes) <= max_edge_size:
        return he.nodes
    if he.type == "version":
        nodes = [int(v) for v in he.nodes]
        return _version_keep(nodes, view.current_of(nodes[0]), version_keep_k)
    return _top_by_base(he.nodes, base, max_edge_size)


def _persons_in_query(view: GraphView, query: str) -> Set[str]:
    """Visible speakers the query names."""
    if not query or not view.name_tokens:
        return set()
    words = set(re.findall(r"\w+", query.lower()))
    low = query.lower()
    out: Set[str] = set()
    for tok, who in view.name_tokens.items():
        if tok in words or (" " in tok and tok in low):
            out |= who
    return out


def build_subgraph(
    g,
    base_all: np.ndarray,
    seeds: Sequence[int],
    asker: Optional[str],
    query: str,
    now: Optional[float],
    expand_weights: Dict[str, float],
    max_edge_size: int = 60,
    max_candidates: int = 300,
    keep_edge_types: Sequence[str] = EDGE_TYPES,
    version_keep_k: int = 5,
) -> Subgraph:
    """``g`` is a :class:`GraphView` (or a :class:`Hypergraph`, meaning its omniscient view).
    ``base_all`` holds a seed score for every node; only visible nodes are ever read."""
    view: GraphView = g.view() if isinstance(g, Hypergraph) else g
    graph = view.g
    mask = view.mask
    seeds = list(dict.fromkeys(int(s) for s in seeds if mask[int(s)]))
    cand: Dict[int, float] = {s: 2.0 for s in seeds}
    nb: Dict[int, float] = {}
    members_of: Dict[int, Sequence[int]] = {}
    for s in seeds:
        for e in view.node_edges[s]:
            he = view.edges[e]
            w = expand_weights.get(he.type, 0.0)
            if w <= 0:
                continue
            if e not in members_of:
                members_of[e] = _expansion_members(view, he, base_all, max_edge_size, version_keep_k)
            for u in members_of[e]:
                u = int(u)
                if nb.get(u, 0.0) < w:
                    nb[u] = w
        # version pull-up: the current value of any touched slot is always a candidate
        cur = view.current_of(s)
        if cur is not None:
            nb[cur] = max(nb.get(cur, 0.0), 1.5)
    first_person = bool(asker and FIRST_PERSON.search(query or ""))
    if first_person:
        # "what am I responsible for?": the asker's own state-bearing messages are candidates
        for v in view.own_chain_nodes(asker)[-20:]:
            nb[v] = max(nb.get(v, 0.0), 1.2)
    for u, w in sorted(nb.items(), key=lambda kv: (-kv[1], -base_all[kv[0]], kv[0])):
        if len(cand) >= max_candidates:
            break
        cand.setdefault(u, w)
    nodes = np.array(sorted(cand), dtype=np.int64)
    n = len(nodes)
    loc = {int(v): i for i, v in enumerate(nodes)}

    # hyperedges, decided by their membership inside the candidates (never by global size)
    keep = set(keep_edge_types)
    inside: Dict[int, List[int]] = {}
    for v in nodes:
        for e in view.node_edges[v]:
            if view.edges[e].type in keep:
                inside.setdefault(e, []).append(int(v))
    base = base_all[nodes].astype(np.float32)
    edges = []
    degree = np.zeros(n, np.int64)
    for e, members in inside.items():
        if len(members) < 2:
            continue
        he = view.edges[e]
        if len(members) > max_edge_size:
            if he.type == "version":
                members = sorted(members, key=graph.tkey.__getitem__)
                members = _version_keep(members, view.current_of(members[0]), version_keep_k)
            else:
                members = list(_top_by_base(np.asarray(members, np.int64), base_all, max_edge_size))
            if len(members) < 2:
                continue
        idx = np.array(sorted(loc[int(u)] for u in members), np.int64)
        edges.append((he.type, idx))
        degree[idx] += 1

    hist = bool(HISTORICAL.search(query or ""))
    ts = np.array([graph.tkey[v][0] for v in nodes], np.float64)
    ref = now if now is not None else view.t_max
    age = np.clip(ref - ts, 0, None)
    span = max(ref, view.t_max) - view.t_min
    asker_tok = (asker or "").lower()
    seed_set = set(seeds)
    rank = np.empty(n, np.int64)
    rank[np.argsort(-base, kind="stable")] = np.arange(n)
    q_persons = _persons_in_query(view, query)
    q_person_pat = None
    if q_persons:
        toks = sorted({t for t, who in view.name_tokens.items() if who & q_persons}, key=len, reverse=True)
        q_person_pat = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(t) for t in toks) + r")(?!\w)")
    log_part = float(np.log1p(len(view.participants)))
    F = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    for i, v in enumerate(nodes):
        m = graph.messages[v]
        low = (m.text or "").lower()
        F[i, FI["base"]] = base[i]
        F[i, FI["base_rank"]] = 1.0 - rank[i] / (n - 1) if n > 1 else 1.0  # percentile inside the subgraph
        F[i, FI["is_seed"]] = float(int(v) in seed_set)
        F[i, FI["is_asker"]] = float(asker is not None and m.speaker == asker)
        F[i, FI["mentions_asker"]] = float(bool(asker_tok) and (
            asker_tok in low or any(asker_tok == e for r in graph.records[v] for e in r.entities)))
        F[i, FI["superseded"]] = float(view.superseded[v])
        F[i, FI["is_current"]] = float(view.current[v])
        F[i, FI["in_chain"]] = float(view.chain_of[v] >= 0)
        F[i, FI["age_frac"]] = float(min(1.0, age[i] / span)) if span > 0 else 0.0
        F[i, FI["log_age_days"]] = float(np.log1p(age[i] / 86400.0))
        F[i, FI["log_degree"]] = float(np.log1p(degree[i]))  # edges inside the subgraph only
        F[i, FI["historical_query"]] = float(hist)
        mod = graph.modality[v]
        if mod in MODALITY_FEATS:
            F[i, FI[f"mod_{mod}"]] = 1.0
        F[i, FI["first_person_query"]] = float(first_person)
        F[i, FI["is_assistant"]] = float(state_role(m) == ASSISTANT)
        F[i, FI["query_mentions_speaker"]] = float(m.speaker in q_persons)
        F[i, FI["node_mentions_query_person"]] = float(bool(q_person_pat and q_person_pat.search(low)))
        F[i, FI["log_n_participants"]] = log_part
    return Subgraph(nodes, F, base, edges, hist, n_participants=len(view.participants))


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
    cur, sup = f[:, FI["is_current"]], f[:, FI["superseded"]]
    if not sg.historical:
        pulled = s.copy()
        for t, idx in sg.edges:
            if t != "version":
                continue
            top = s[idx].max()
            c = idx[cur[idx] > 0]
            pulled[c] = np.maximum(pulled[c], version_pull * top)
        s = np.where(sup > 0, pulled * superseded_penalty, pulled)
    s = s + asker_bonus * np.maximum(f[:, FI["is_asker"]], f[:, FI["mentions_asker"]]) * (s > 0)
    # first-person questions in group chats: the asker's own current commitments/decisions
    if sg.n_participants >= min_participants:
        s = s + own_state_bonus * f[:, FI["first_person_query"]] * f[:, FI["is_asker"]] * cur
    return s
