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

Linking evidence is guarded against natural chat, where everything shares one
channel: participant names, very frequent entities and date/time values never link
two statements into one slot, and assistant turns do not change state unless a
participant confirms them ("sounds good") right after.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ...extract.heuristic import is_time_expression
from ...extract.records import STATEFUL, StateRecord
from ...types import Message

EDGE_TYPES = ("reply", "episode", "channel", "speaker", "entity", "version")
ASSISTANT = "assistant"
# a short "go ahead" from a participant right after an assistant suggestion
_AFFIRM = re.compile(r"^\W*(?:yes|yeah|yep|ok|okay|sounds good|let'?s do that|let'?s do it|agreed|go with that|perfect)\b",
                     re.I)
CONFIRM_WINDOW = 2


def is_affirmation(text: str) -> bool:
    t = (text or "").strip()
    return bool(_AFFIRM.match(t)) and "?" not in t and len(t.split()) <= 8


def state_role(m: Message) -> Optional[str]:
    """``"assistant"`` for assistant turns (speaker or role), ``"user"``/``"human"`` when
    the speaker/role is literally that, else ``None`` (an ordinary named participant)."""
    for v in (m.speaker, m.role):
        if (v or "").strip().lower() == ASSISTANT:
            return ASSISTANT
    for v in (m.role, m.speaker):
        low = (v or "").strip().lower()
        if low in ("user", "human"):
            return low
    return None


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

    def tokens(self, link_tokens: Callable[[Sequence[str]], Set[str]]) -> Set[str]:
        out: Set[str] = set()
        for _, r in self.records[-2:]:
            out |= link_tokens(r.entities)
        return out


class Hypergraph:
    def __init__(
        self,
        episode_gap_minutes: float = 60.0,
        max_episode: int = 25,
        link_on_entity: bool = True,
        edge_types: Sequence[str] = EDGE_TYPES,
        max_link_df: float = 0.02,
        min_df_messages: int = 5,
        state_roles: Optional[Sequence[Optional[str]]] = ("user", "human", None),
        assistant_confirmation: bool = True,
    ) -> None:
        self.gap = episode_gap_minutes * 60.0
        self.max_episode = max_episode
        self.link_on_entity = link_on_entity
        self.enabled = set(edge_types)
        self.max_link_df, self.min_df_messages = max_link_df, min_df_messages
        # whose records may create / extend / supersede version chains (see state_role); None = everyone
        self.state_roles = None if state_roles is None else set(state_roles)
        self.assistant_confirmation = assistant_confirmation
        self._speaker_names: Set[str] = set()  # lowercased names + their tokens: never linking evidence
        self.participants: Set[str] = set()  # distinct non-assistant speakers (multi-party if >= 3)
        self.entity_df: Dict[str, int] = {}  # messages mentioning each entity (and each entity token)
        self.token_df: Dict[str, int] = {}
        self._pending: Dict[str, List[List]] = {}  # scope -> [[node, records, messages left]]
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
    def _note_speaker(self, name: str) -> None:
        low = (name or "").strip().lower()
        if low:
            self._speaker_names.add(low)
            self._speaker_names.update(low.split())

    def _frequent(self, df: int) -> bool:
        return df >= self.min_df_messages and df / max(1, len(self.messages)) > self.max_link_df

    def link_tokens(self, ents: Sequence[str]) -> Set[str]:
        """Entity tokens that count as evidence that two statements are about the same slot."""
        out: Set[str] = set()
        for e in ents:
            if e in self._speaker_names or self._frequent(self.entity_df.get(e, 0)) or is_time_expression(e):
                continue
            out.update(t for t in e.split()
                       if t not in self._speaker_names and not self._frequent(self.token_df.get(t, 0)))
        return out

    def _changes_state(self, m: Message) -> bool:
        return self.state_roles is None or state_role(m) in self.state_roles

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
            self._note_speaker(m.speaker)
            if state_role(m) != ASSISTANT:
                self.participants.add(m.speaker)
            self._add_structural(v, m)
            ents = {e for r in recs for e in r.entities}
            for ent in ents:
                self.entity_df[ent] = self.entity_df.get(ent, 0) + 1
            for tok in {t for e in ents for t in e.split()}:
                self.token_df[tok] = self.token_df.get(tok, 0) + 1
            if "entity" in self.enabled:
                for r in recs:
                    for ent in r.entities:
                        if ent not in self._speaker_names:
                            self._attach(v, self._edge("entity", ent))
            if "version" not in self.enabled:
                continue
            stateful = [r for r in recs if r.modality in STATEFUL and r.attribute != "info"]
            scope = m.session_id or m.channel
            if not self._changes_state(m):
                if stateful and self.assistant_confirmation:
                    self._pending.setdefault(scope, []).append([v, stateful, CONFIRM_WINDOW])
                continue
            if self._confirm(scope, m):
                continue  # "sounds good" carries no value of its own: the confirmed suggestion is the update
            for r in stateful:
                self._add_version(v, r)

    def _confirm(self, scope: str, m: Message) -> bool:
        """Apply pending assistant records if ``m`` affirms them; otherwise age the window."""
        pend = self._pending.get(scope)
        if not pend:
            return False
        if is_affirmation(m.text):
            for node, recs, _ in pend:
                for r in recs:
                    self._add_version(node, r)
            pend.clear()
            return True
        for p in pend:
            p[2] -= 1
        pend[:] = [p for p in pend if p[2] > 0]
        return False

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

    def _find_chain(self, r: StateRecord, topicless: bool = False) -> Optional[int]:
        """Which slot does this record update?

        1. same (subject, attribute) key -> that slot (LLM keys, or the same topic phrase)
        2. otherwise a slot in the same scope (channel / phase) with the same attribute
           that shares an entity token ("go with AWS EC2" ... "scrap EC2, Cloud Run it is")
        A replacement cue alone never links: that chains chit-chat into real slots.
        Entity evidence excludes participant names, entities in more than ``max_link_df``
        of messages and date/time values (see :meth:`link_tokens`). A record with no
        topic of its own (``topicless``: subject is just the channel) has no key worth
        matching, so it links on entity evidence only, never on the channel-wide key.
        """
        same = [] if topicless else self._chains_by_key.get(r.key, [])
        toks = self.link_tokens(r.entities)
        if r.attribute == "status" and r.modality == "done":
            # a completion resolves the latest blocker it shares an entity / topic with
            pool = same + [c for c in self._chains_by_scope.get(self._scope_attr(r), []) if c not in same]
            for c in reversed(pool):
                ch = self.chains[c]
                if self.modality[ch.latest] == "blocker" and (c in same or toks & ch.tokens(self.link_tokens)):
                    return c
        if same:
            return same[-1]
        if r.explicit_key or r.attribute == "deadline" or not toks or not self.link_on_entity:
            return None
        for c in reversed(self._chains_by_scope.get(self._scope_attr(r), [])):
            if toks & self.chains[c].tokens(self.link_tokens):
                return c
        return None

    def _topicless(self, v: int, r: StateRecord) -> bool:
        return not r.explicit_key and r.subject == (self.messages[v].channel or "")

    def _add_version(self, v: int, r: StateRecord) -> None:
        target = self._find_chain(r, self._topicless(v, r))
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

    def superseding_pairs(self) -> Set[Tuple[str, str]]:
        """(older message id, newer message id) for every pair of values in one version chain:
        the graph's claim that the newer statement replaces the older one."""
        out: Set[Tuple[str, str]] = set()
        for ch in self.chains:
            ids = [self.messages[v].id for v, _ in ch.records]
            for i, older in enumerate(ids):
                out.update((older, newer) for newer in ids[i + 1:] if newer != older)
        return out

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

    def stats(self) -> Dict[str, float]:
        by: Dict[str, int] = {}
        for e in self.edges:
            by[e.type] = by.get(e.type, 0) + 1
        n = len(self.messages)
        return {
            "nodes": n, "edges": len(self.edges), "chains": len(self.chains),
            "superseded": sum(self.superseded), "superseded_fraction": sum(self.superseded) / max(1, n),
            "chains_per_100_messages": 100.0 * len(self.chains) / max(1, n),
            **{f"edges_{k}": v for k, v in sorted(by.items())},
        }
