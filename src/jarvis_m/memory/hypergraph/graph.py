"""Incremental hypergraph over messages.

Nodes are messages (raw text is never discarded). Hyperedges, by type:

* ``reply``    – a reply thread (root + all descendants)
* ``episode``  – consecutive messages in one channel/session with no long gap
* ``channel``  – everyone in a channel/group (scope)
* ``speaker``  – everything one participant said (personal trajectory)
* ``entity``   – messages mentioning the same entity (tool, id, person, date)
* ``version``  – messages stating successive values of one (subject, attribute) slot

Version chains also mark nodes ``superseded`` (an older value of a slot) and
``current`` (latest value), which retrieval uses for knowledge updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...extract.records import STATEFUL, StateRecord
from ...types import Message

EDGE_TYPES = ("reply", "episode", "channel", "speaker", "entity", "version")


@dataclass
class Hyperedge:
    type: str
    key: str
    nodes: List[int] = field(default_factory=list)


@dataclass
class Chain:
    key: str
    records: List[Tuple[int, StateRecord]] = field(default_factory=list)  # (node, record), time order
    edge: int = -1

    @property
    def latest(self) -> int:
        return self.records[-1][0]

    def tokens(self) -> Set[str]:
        out: Set[str] = set()
        for _, r in self.records[-2:]:
            for e in r.entities:
                out.update(e.split())
        return out


class Hypergraph:
    def __init__(
        self,
        episode_gap_minutes: float = 60.0,
        max_episode: int = 25,
        link_on_entity: bool = True,
        edge_types: Sequence[str] = EDGE_TYPES,
    ) -> None:
        self.gap = episode_gap_minutes * 60.0
        self.max_episode = max_episode
        self.link_on_entity = link_on_entity
        self.enabled = set(edge_types)
        self.messages: List[Message] = []
        self.id2node: Dict[str, int] = {}
        self.edges: List[Hyperedge] = []
        self.node_edges: List[List[int]] = []
        self.records: List[List[StateRecord]] = []
        self._key2edge: Dict[Tuple[str, str], int] = {}
        self._thread_of: Dict[int, int] = {}  # node -> reply edge
        self._last_in_scope: Dict[str, int] = {}  # channel/session -> last node
        self._episode_of_scope: Dict[str, int] = {}
        self.chains: List[Chain] = []
        self._chains_by_key: Dict[str, List[int]] = {}
        self._chains_by_scope: Dict[str, List[int]] = {}
        self.superseded: List[bool] = []
        self.chain_of: List[int] = []  # node -> chain id or -1
        self.modality: List[str] = []
        self.rejected: Dict[int, List[int]] = {}  # chain -> negated-alternative nodes

    # ------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.messages)

    def _edge(self, etype: str, key: str) -> int:
        k = (etype, key)
        if k not in self._key2edge:
            self._key2edge[k] = len(self.edges)
            self.edges.append(Hyperedge(etype, key))
        return self._key2edge[k]

    def _attach(self, node: int, edge: int) -> None:
        e = self.edges[edge]
        if not e.nodes or e.nodes[-1] != node:
            e.nodes.append(node)
            self.node_edges[node].append(edge)

    def edge_types_of(self, node: int) -> List[str]:
        return [self.edges[e].type for e in self.node_edges[node]]

    # ------------------------------------------------------------ ingest
    def add(self, messages: Iterable[Message], records: Dict[str, List[StateRecord]]) -> None:
        for m in messages:
            v = len(self.messages)
            self.messages.append(m)
            self.id2node[m.id] = v
            self.node_edges.append([])
            recs = records.get(m.id, [])
            self.records.append(recs)
            self.superseded.append(False)
            self.chain_of.append(-1)
            self.modality.append(next((r.modality for r in recs if r.modality in STATEFUL), recs[0].modality if recs else "info"))
            self._add_structural(v, m)
            for r in recs:
                if "entity" in self.enabled:
                    for ent in r.entities:
                        self._attach(v, self._edge("entity", ent))
                if "version" in self.enabled and r.modality in STATEFUL and r.attribute != "info":
                    self._add_version(v, r)

    def _add_structural(self, v: int, m: Message) -> None:
        scope = m.session_id or m.channel
        if "channel" in self.enabled:
            self._attach(v, self._edge("channel", m.channel))
        if "speaker" in self.enabled:
            self._attach(v, self._edge("speaker", m.speaker))
        if "reply" in self.enabled and m.reply_to and m.reply_to in self.id2node:
            parent = self.id2node[m.reply_to]
            e = self._thread_of.get(parent)
            if e is None:
                e = self._edge("reply", self.messages[parent].id)
                self._attach(parent, e)
                self._thread_of[parent] = e
            self._attach(v, e)
            self._thread_of[v] = e
        if "episode" in self.enabled:
            last = self._last_in_scope.get(scope)
            ep = self._episode_of_scope.get(scope)
            new = (
                last is None or ep is None
                or len(self.edges[ep].nodes) >= self.max_episode
                or (m.timestamp is not None and self.messages[last].timestamp is not None
                    and m.timestamp - self.messages[last].timestamp > self.gap)
            )
            if new:
                ep = self._edge("episode", f"{scope}#{v}")
                self._episode_of_scope[scope] = ep
            self._attach(v, ep)
            self._last_in_scope[scope] = v

    def _scope_attr(self, r: StateRecord) -> str:
        return f"{r.subject.split('/', 1)[0].lower()}|{r.attribute.lower()}"

    def _find_chain(self, r: StateRecord) -> Optional[int]:
        """Which slot does this record update?

        1. same (subject, attribute) key -> that slot (LLM keys, or the same topic phrase)
        2. otherwise a slot in the same scope (channel / phase) with the same attribute
           that shares an entity token ("go with AWS EC2" ... "scrap EC2, Cloud Run it is")
        A replacement cue alone never links: that chains chit-chat into real slots.
        """
        same = self._chains_by_key.get(r.key, [])
        if r.attribute == "status" and r.modality == "done":
            # a completion resolves the latest blocker it shares an entity / topic with
            toks = {t for e in r.entities for t in e.split()}
            pool = same + [c for c in self._chains_by_scope.get(self._scope_attr(r), []) if c not in same]
            for c in reversed(pool):
                ch = self.chains[c]
                if self.modality[ch.latest] == "blocker" and (c in same or toks & ch.tokens()):
                    return c
        if same:
            return same[-1]
        if r.explicit_key or r.attribute == "deadline":
            return None
        toks = {t for e in r.entities for t in e.split()}
        if not toks:
            return None
        for c in reversed(self._chains_by_scope.get(self._scope_attr(r), [])):
            if self.link_on_entity and toks & self.chains[c].tokens():
                return c
        return None

    def _add_version(self, v: int, r: StateRecord) -> None:
        target = self._find_chain(r)
        if r.modality == "negated" and target is None:
            self.superseded[v] = True  # a rejection with no live slot yet: never a current value
            return
        if target is None:
            target = len(self.chains)
            self.chains.append(Chain(r.key))
            self._chains_by_key.setdefault(r.key, []).append(target)
            self._chains_by_scope.setdefault(self._scope_attr(r), []).append(target)
            self.chains[target].edge = self._edge("version", f"{r.key}#{target}")
        ch = self.chains[target]
        self._attach(v, ch.edge)
        if r.modality == "negated" and ch.records:
            # "we are NOT switching to Y": a rejected alternative. It joins the slot's
            # hyperedge (so it is found with it) but never becomes the current value.
            self.superseded[v] = True
            self.chain_of[v] = target
            self.rejected.setdefault(target, []).append(v)
            return
        if ch.records and ch.records[-1][0] != v:
            for node, _ in ch.records:
                self.superseded[node] = True
        ch.records.append((v, r))
        self.superseded[v] = False
        self.chain_of[v] = target

    # ------------------------------------------------------------ queries
    def current_of(self, node: int) -> Optional[int]:
        c = self.chain_of[node]
        return self.chains[c].latest if c >= 0 else None

    def neighbours(self, nodes: Iterable[int], weights: Dict[str, float], max_edge_size: int) -> Dict[int, float]:
        """Nodes reachable through one hyperedge, with the best edge weight seen."""
        out: Dict[int, float] = {}
        for v in nodes:
            for e in self.node_edges[v]:
                he = self.edges[e]
                w = weights.get(he.type, 0.0)
                if w <= 0 or len(he.nodes) > max_edge_size:
                    continue
                for u in he.nodes:
                    if out.get(u, 0.0) < w:
                        out[u] = w
        return out

    def stats(self) -> Dict[str, int]:
        by: Dict[str, int] = {}
        for e in self.edges:
            by[e.type] = by.get(e.type, 0) + 1
        return {
            "nodes": len(self.messages), "edges": len(self.edges), "chains": len(self.chains),
            "superseded": sum(self.superseded), **{f"edges_{k}": v for k, v in sorted(by.items())},
        }
