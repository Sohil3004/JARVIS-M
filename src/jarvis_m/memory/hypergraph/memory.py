"""HypergraphMemory: hybrid seeds -> hyperedge expansion -> fixed or learned
propagation -> version-aware ranking -> packed context (+ optional state header).

When access control is in force (``access:``, default ``mode: auto``), every step works on
the effective principal's view of the graph:
seeds, expansion, features, slot state, the state header and packing only ever touch
messages that principal may read, and a final check re-tests everything that is returned.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...access import HIDDEN_UPDATE_NOTE, Label, Principal
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
        if self.access is not None:
            # labels come from the policy; a channel with no membership list is readable by
            # everyone cleared for the label, so other channels may read its slot state
            self.graph_kw.update(labeler=self.access.policy.base_label, open_channel=self.access.directory.is_open,
                                 channel_size=self._directory_size)
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

    def _directory_size(self, channel: str) -> Optional[int]:
        """Members of a closed channel the directory lists (None: count who wrote there)."""
        members = self.access.directory.members(channel) if self.access.directory.known(channel) else None
        return None if members is None else len(members)

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        super().reset()
        self.graph = Hypergraph(**self.graph_kw)

    def _label(self, messages: Sequence[Message]) -> None:
        return None  # the graph labels messages (it also needs the thread structure)

    def message_labels(self) -> Sequence[Label]:
        return self.graph.labels

    def _on_ingest(self, messages: Sequence[Message]) -> None:
        super()._on_ingest(messages)
        by: Dict[str, list] = {}
        suggested: Dict[str, Label] = {}
        policy = self.access.policy if self.access is not None else None
        for r in self.extractor.extract(messages):
            by.setdefault(r.msg_id, []).append(r)
            hint = (r.meta or {}).get("sensitivity")
            if policy is not None and policy.use_extractor_suggestions and hint:
                # an extractor may suggest a label; it can only raise the message's label
                hint = hint if isinstance(hint, dict) else {"level": hint}
                try:
                    label = Label(policy.level(hint.get("level") or 0), frozenset(hint.get("compartments") or ()))
                except ValueError:
                    continue  # not a level this policy knows: ignore the suggestion
                prev = suggested.get(r.msg_id, Label())
                suggested[r.msg_id] = Label(max(prev.level, label.level), prev.compartments | label.compartments)
        self.graph.add(messages, by, suggested)

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

    def view(self, asker: Optional[str] = None, principal: Optional[Principal] = None) -> GraphView:
        """What this query may see: the messages ``principal`` may read (access control),
        narrowed by the speaker-scope ablation (``scope: asker|others``). Slot state is
        recomputed over exactly those messages."""
        return self.graph.view(self._candidates(asker, principal))

    def subgraph(self, query: str, asker: Optional[str] = None, now: Optional[float] = None,
                 view: Optional[GraphView] = None, audience: Any = None, channel: Optional[str] = None) -> Subgraph:
        view = view or self.view(asker, self.principal(asker, audience, channel))
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
    def retrieve(self, query, *, asker=None, budget_tokens=None, max_items=None, now=None, audience=None,
                 channel=None) -> Retrieval:
        audience = self.audience_for(asker, audience, channel)
        principal = self.principal(asker, audience)
        if not self.messages:
            return Retrieval([], "", 0, {}, self._audit(principal, asker, audience))
        view = self.view(asker, principal)
        sg = self.subgraph(query, asker, now, view)
        self.last_subgraph = sg
        if not len(sg.nodes):
            return Retrieval([], "", 0, {"candidates": 0, "edges": 0, "historical": sg.historical},
                             self._audit(principal, asker, audience))
        scores = np.asarray(self.scorer(sg, (self, query, asker)), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        ranked = [(int(sg.nodes[i]), float(scores[i])) for i in order if np.isfinite(scores[i])]
        enforce = principal is not None and self.access.policy.enforce
        signal = principal is not None and self.access.policy.signal_hidden_updates
        # final check, part 1: nothing unreadable is ranked (so it cannot be packed or summarised)
        bad = set(self._final_check([v for v, _ in ranked], principal))
        if bad and enforce:
            ranked = [r for r in ranked if r[0] not in bad]

        header, header_nodes = "", []
        if self.state_header:
            cap = int(budget_tokens * self.header_share) if budget_tokens else None
            header, header_nodes = self._state_header(view, [v for v, _ in ranked[: (max_items or 10)]], cap, signal)
            # final check, part 2: the header only quotes messages the principal may read
            bad_header = self._final_check(header_nodes, principal)
            if bad_header and enforce:
                header, bad = "", bad | set(bad_header)
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
            kept.sort(key=lambda r: self.graph.tkey[r[0]])  # event time, also after out-of-order ingest
        items: List[RetrievedItem] = [self._item(v, sc) for v, sc in kept]
        if header:
            items.insert(0, RetrievedItem(header, 1.0, (), "state"))
        signals = 0
        if signal:
            signals = len(self._hidden_updates(view, [v for v, _ in kept]))
            if signals and not header:
                items.append(RetrievedItem(f"[Note: {HIDDEN_UPDATE_NOTE}]", 0.0, (), "notice"))
        ctx = "\n\n".join(it.text for it in items)
        info = {"candidates": int(len(sg.nodes)), "edges": len(sg.edges), "historical": sg.historical}
        audit = self._audit(principal, asker, audience, view.mask if enforce else None, len(bad),
                            hidden_update_signals=signals)
        return Retrieval(items, ctx, self.counter.count(ctx), info, audit)

    def _hidden_updates(self, view: GraphView, nodes: Sequence[int]) -> set:
        """Viewer slots (touched by ``nodes``) that have a newer version the viewer cannot
        see. Only used with ``signal_hidden_updates: true``: this reads hidden state on
        purpose, to say "this may have been updated" and nothing more."""
        g = self.graph
        newest = {s.key: g.tkey[s.latest] for s in g.view().slots if s.latest is not None}
        out = set()
        for v in nodes:
            c = int(view.chain_of[v])
            if c < 0 or view.slots[c].latest is None:
                continue
            slot = view.slots[c]
            if newest.get(slot.key, g.tkey[slot.latest]) > g.tkey[slot.latest]:
                out.add(c)
        return out

    def _state_header(self, view: GraphView, nodes: Sequence[int], cap: Optional[int], signal: bool = False):
        """Current value (+ earlier values) of every slot touched by ``nodes``, most
        relevant first, within ``cap`` tokens; the first slot is always kept. Slots are the
        viewer's: the current value is the latest version the viewer can see. Returns the
        header and the nodes it quotes."""
        g = self.graph
        lines, seen, quoted = ["[Current state of topics in these messages]"], set(), []
        flagged = self._hidden_updates(view, nodes) if signal else set()
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
            earlier = ch.records[:-1][-2:]
            old = [r.value[:50] for _, r in earlier]
            prev = f" | earlier: {' -> '.join(old)}" if old else ""
            subj = f"{cur.subject}/" if cur.subject else ""
            note = f" ({HIDDEN_UPDATE_NOTE})" if c in flagged else ""
            lines.append(f"- {subj}{cur.attribute} [{cur.modality}, {cur.speaker}, {m.id} {when}]: "
                         f"{cur.value[:80]}{prev}{note}")
            quoted.append([cur_node, *[v for v, _ in earlier]])
        if len(lines) == 1:
            return "", []
        out, used = lines[0] + "\n" + lines[1], list(quoted[0])
        for ln, q in zip(lines[2:], quoted[1:]):
            if cap and self.counter.count(out + "\n" + ln) > cap:
                break
            out += "\n" + ln
            used += q
        return out, used

    def stats(self) -> Dict[str, Any]:
        return {**super().stats(), **self.graph.stats(), "extractor": getattr(self.extractor, "name", "?")}
