"""HypergraphMemory: hybrid seeds -> hyperedge expansion -> fixed or learned
propagation -> version-aware ranking -> packed context (+ optional state header)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...extract import build_extractor
from ...interfaces import Retrieval, RetrievedItem
from ...types import Message
from ..baselines import HybridMemory
from .graph import EDGE_TYPES, GraphView, Hypergraph
from .subgraph import Subgraph, build_subgraph, fixed_propagation

DEFAULT_EXPAND = {"reply": 1.0, "episode": 0.8, "entity": 0.6, "version": 1.0, "speaker": 0.0, "channel": 0.0,
                  "mention": 0.0}
# Fixed-propagation weights chosen on StateBench dev: episode edges are too broad to
# propagate over without learning (they hurt every category but multi-hop); the HGNN still sees them.
DEFAULT_PROP = {"reply": 1.0, "episode": 0.0, "entity": 0.5, "version": 1.0, "speaker": 0.0, "channel": 0.0,
                "mention": 0.0}


class HypergraphMemory(HybridMemory):
    name = "hypergraph"

    def __init__(
        self,
        extractor: Optional[Dict[str, Any]] = None,
        edge_types: Sequence[str] = EDGE_TYPES,
        episode_gap_minutes: float = 60.0,
        max_episode: int = 25,
        n_seeds: int = 30,
        max_candidates: int = 300,
        max_edge_size: int = 60,
        expand_weights: Optional[Dict[str, float]] = None,
        scorer: Optional[Dict[str, Any]] = None,
        state_header: bool = False,
        header_share: float = 0.25,
        max_link_df: float = 0.02,
        link_df_window: int = 500,
        version_keep_k: int = 5,
        state_roles: Optional[Sequence[Optional[str]]] = ("user", "human", None),
        assistant_confirmation: bool = True,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.extractor_cfg = extractor or {"type": "heuristic"}
        self.extractor = build_extractor(self.extractor_cfg)
        # state_roles: who may change slot state (None = everyone); the default excludes assistant turns
        self.graph_kw = dict(episode_gap_minutes=episode_gap_minutes, max_episode=max_episode, edge_types=edge_types,
                             max_link_df=max_link_df, link_df_window=link_df_window,
                             assistant_confirmation=assistant_confirmation,
                             state_roles=None if state_roles is None else tuple(state_roles))
        self.edge_types = tuple(edge_types)
        self.graph = Hypergraph(**self.graph_kw)
        self.n_seeds, self.max_candidates, self.max_edge_size = n_seeds, max_candidates, max_edge_size
        self.version_keep_k = version_keep_k
        self.expand = {k: v for k, v in (expand_weights or DEFAULT_EXPAND).items() if k in self.edge_types}
        self.scorer_cfg = dict(scorer or {"type": "fixed"})
        self._scorer = None
        if self.scorer_cfg.get("type") == "hgnn":
            self.scorer  # noqa: B018 - load (and validate) the checkpoint now, before ingesting anything
        self.state_header, self.header_share = state_header, header_share
        self.last_subgraph: Optional[Subgraph] = None

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        super().reset()
        self.graph = Hypergraph(**self.graph_kw)

    def _on_ingest(self, messages: Sequence[Message]) -> None:
        super()._on_ingest(messages)
        by: Dict[str, list] = {}
        for r in self.extractor.extract(messages):
            by.setdefault(r.msg_id, []).append(r)
        self.graph.add(messages, by)

    # ---------------------------------------------------------------- scoring
    @property
    def scorer(self):
        if self._scorer is None:
            cfg = dict(self.scorer_cfg)
            kind = cfg.pop("type")
            if kind == "fixed":
                w = {k: v for k, v in cfg.pop("edge_weights", DEFAULT_PROP).items() if k in self.edge_types}
                self._scorer = lambda sg, q: fixed_propagation(sg, w, **cfg)
            elif kind == "seeds":  # ablation: candidates + version logic off, base scores only
                self._scorer = lambda sg, q: sg.base.astype(np.float64)
            elif kind == "hgnn":
                from .hgnn import load_scorer

                self._scorer = load_scorer(**cfg)
            else:
                raise ValueError(f"unknown scorer {kind!r}")
        return self._scorer

    def view(self, asker: Optional[str] = None) -> GraphView:
        """What this query may see. The speaker-scope ablation (``scope: asker|others``) is
        a filter on the view, so slot state is recomputed over the messages in scope."""
        return self.graph.view(self._candidates(asker))

    def subgraph(self, query: str, asker: Optional[str] = None, now: Optional[float] = None,
                 view: Optional[GraphView] = None) -> Subgraph:
        view = view or self.view(asker)
        mask, idx = view.mask, view.nodes
        base_all = np.zeros(len(self.messages), dtype=np.float64)
        if len(idx):
            # seeds are ranked and normalised over the visible messages only
            raw = self._adjust(self._raw_scores(self.query_text(query, asker), mask), asker, now)[idx]
            lo, hi = float(raw.min()), float(raw.max())
            base_all[idx] = (raw - lo) / (hi - lo) if hi > lo else 0.0
        seeds = idx[np.argsort(-base_all[idx], kind="stable")][: self.n_seeds]
        return build_subgraph(
            view, base_all, seeds, asker, query, now, self.expand,
            self.max_edge_size, self.max_candidates, self.edge_types, self.version_keep_k,
        )

    def query_embedding(self, query: str, asker: Optional[str]) -> np.ndarray:
        return self.dense.embedder.encode_queries([self.query_text(query, asker)])[0]

    def node_embeddings(self, nodes: np.ndarray) -> np.ndarray:
        return self.dense.matrix[nodes]

    # ---------------------------------------------------------------- retrieve
    def retrieve(self, query, *, asker=None, budget_tokens=None, max_items=None, now=None) -> Retrieval:
        if not self.messages:
            return Retrieval([], "", 0, {})
        view = self.view(asker)
        sg = self.subgraph(query, asker, now, view)
        self.last_subgraph = sg
        if not len(sg.nodes):
            return Retrieval([], "", 0, {"candidates": 0, "edges": 0, "historical": sg.historical})
        scores = np.asarray(self.scorer(sg, (self, query, asker)), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        ranked = [(int(sg.nodes[i]), float(scores[i])) for i in order if np.isfinite(scores[i])]

        header = ""
        if self.state_header:
            cap = int(budget_tokens * self.header_share) if budget_tokens else None
            header = self._state_header(view, [v for v, _ in ranked[: (max_items or 10)]], cap)
        hdr_tok = self.counter.count(header) if header else 0
        msg_budget = budget_tokens - hdr_tok if budget_tokens is not None else None
        kept, used = [], 0
        for v, sc in ranked:
            if max_items is not None and len(kept) >= max_items:
                break
            c = self.counter.count(self.messages[v].render())
            if msg_budget is not None and used + c > msg_budget:
                continue
            kept.append((v, sc))
            used += c
        if self.order == "chronological":
            kept.sort(key=lambda r: r[0])
        items: List[RetrievedItem] = [self._item(v, sc) for v, sc in kept]
        if header:
            items.insert(0, RetrievedItem(header, 1.0, (), "state"))
        ctx = "\n\n".join(it.text for it in items)
        info = {"candidates": int(len(sg.nodes)), "edges": len(sg.edges), "historical": sg.historical}
        return Retrieval(items, ctx, self.counter.count(ctx), info)

    def _state_header(self, view: GraphView, nodes: Sequence[int], cap: Optional[int]) -> str:
        """Current value (+ earlier values) of every slot touched by ``nodes``, most
        relevant first, within ``cap`` tokens; the first slot is always kept. Slots are the
        viewer's: the current value is the latest version the viewer can see."""
        g = self.graph
        lines, seen = ["[Current state of topics in these messages]"], set()
        for v in nodes:
            c = int(view.chain_of[v])
            if c < 0 or c in seen:
                continue
            seen.add(c)
            ch = view.slots[c]
            if not ch.records:
                continue
            cur_node, cur = ch.records[-1]
            m = g.messages[cur_node]
            when = m.meta.get("time_str", "")
            old = [r.value[:50] for _, r in ch.records[:-1]][-2:]
            prev = f" | earlier: {' -> '.join(old)}" if old else ""
            subj = f"{cur.subject}/" if cur.subject else ""
            lines.append(f"- {subj}{cur.attribute} [{cur.modality}, {cur.speaker}, {m.id} {when}]: {cur.value[:80]}{prev}")
        if len(lines) == 1:
            return ""
        out = lines[0] + "\n" + lines[1]
        for ln in lines[2:]:
            if cap and self.counter.count(out + "\n" + ln) > cap:
                break
            out += "\n" + ln
        return out

    def stats(self) -> Dict[str, Any]:
        return {**super().stats(), **self.graph.stats(), "extractor": getattr(self.extractor, "name", "?")}
