"""Incremental hypergraph over messages.

Nodes are messages (raw text is never discarded). Hyperedges, by type:

* ``reply``    – a reply thread (root + all descendants, inside one channel)
* ``episode``  – consecutive messages in one channel/session with no long gap
* ``channel``  – everyone in a channel/group (scope)
* ``speaker``  – everything one participant said (personal trajectory)
* ``entity``   – messages mentioning the same entity (tool, id, date)
* ``version``  – messages stating successive values of one (holder, subject, attribute) slot
* ``mention``  – messages that mention one participant by name

Two layers:

:class:`Hypergraph` stores messages, records and labels, and builds *version chains*.
A chain only holds records of one **visibility class** (channel, level, compartments).
When a record is processed it may *read* chains of classes that every reader of its own
class can also read, but it only ever *writes* to its own class. So the state of a class
never depends on messages its readers cannot see.

:class:`GraphView` is what one viewer sees: a mask over nodes plus everything derived from
the visible nodes only (all hyperedges, slots merged from the visible same-slot chains,
current / superseded flags, participants). ``Hypergraph.view()`` with no mask is the
omniscient view; access control (``jarvis_m.access``) supplies the mask for a principal.

Chains are a function of the messages in time order, not of ingest order: a message that
arrives out of order (or a label raised by a late thread parent) triggers a replay.

Linking evidence is guarded against natural chat, where everything shares one channel:
participant names, entities frequent in the recent window and date/time values never link
two statements into one slot, slots never chain across holders, and assistant turns do not
change state unless a participant confirms them ("sounds good") right after.
"""

from __future__ import annotations

import re
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ...access import Label, ThreadLabeler, message_label
from ...extract.heuristic import is_time_expression
from ...extract.records import STATEFUL, StateRecord, holder_key
from ...types import Message

EDGE_TYPES = ("reply", "episode", "channel", "speaker", "entity", "version", "mention")
ASSISTANT = "assistant"
# speaker labels that are roles, not names: never matched as mentions or holders
GENERIC_SPEAKERS = frozenset({"user", "assistant", "human", "system", "bot", "ai", "?"})
# a short "go ahead" from a participant right after an assistant suggestion
_AFFIRM = re.compile(r"^\W*(?:yes|yeah|yep|ok|okay|sounds good|let'?s do that|let'?s do it|agreed|go with that|perfect)\b",
                     re.I)
CONFIRM_WINDOW = 2
VisClass = Tuple[str, int, frozenset]  # (channel, level, compartments)


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


def name_tokens(name: str) -> Set[str]:
    """Lowercased forms a participant can be referred to by: the full name and its words."""
    low = (name or "").strip().lower()
    if not low or low in GENERIC_SPEAKERS:
        return set()
    return {low} | {t for t in re.findall(r"\w+", low) if len(t) >= 3 and t not in GENERIC_SPEAKERS}


@dataclass
class Hyperedge:
    type: str
    key: str
    nodes: np.ndarray  # node ids, ascending (version edges: time order)


@dataclass
class Chain:
    """Successive values of one slot inside one visibility class."""

    slot: str  # chains with the same slot are merged in a view
    cls: int
    scope_attr: str
    records: List[Tuple[int, StateRecord]] = field(default_factory=list)  # (node, record), time order
    rejected: List[int] = field(default_factory=list)  # negated alternatives: in the slot, never current

    @property
    def latest(self) -> int:
        return self.records[-1][0]


@dataclass
class Slot:
    """One slot as a viewer sees it: the visible same-slot chains merged in time order."""

    key: str
    records: List[Tuple[int, StateRecord]]
    rejected: List[int]
    nodes: List[int]  # records + rejected, time order
    edge: int = -1

    @property
    def latest(self) -> Optional[int]:
        return self.records[-1][0] if self.records else None


class Hypergraph:
    def __init__(
        self,
        episode_gap_minutes: float = 60.0,
        max_episode: int = 25,
        link_on_entity: bool = True,
        edge_types: Sequence[str] = EDGE_TYPES,
        max_link_df: float = 0.02,
        min_df_messages: int = 5,
        link_df_window: int = 500,
        state_roles: Optional[Sequence[Optional[str]]] = ("user", "human", None),
        assistant_confirmation: bool = True,
        labeler: Optional[Callable[[Message], Label]] = None,
        open_channel: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.gap = episode_gap_minutes * 60.0
        self.max_episode = max_episode
        self.link_on_entity = link_on_entity
        self.enabled = set(edge_types)
        self.max_link_df, self.min_df_messages, self.link_df_window = max_link_df, min_df_messages, link_df_window
        # whose records may create / extend / supersede version chains (see state_role); None = everyone
        self.state_roles = None if state_roles is None else set(state_roles)
        self.assistant_confirmation = assistant_confirmation
        self.labeler = labeler or message_label
        # a channel with no membership restriction: anyone cleared for the label can read it, so
        # other channels may read its state. Without a directory every channel is open.
        self.open_channel = open_channel or (lambda channel: True)

        self.messages: List[Message] = []
        self.id2node: Dict[str, int] = {}
        self.records: List[List[StateRecord]] = []
        self.modality: List[str] = []
        self.threads = ThreadLabeler(self.labeler)
        self.labels: List[Label] = self.threads.labels  # effective: own label joined with the thread parent's
        self.parent: List[int] = self.threads.parent  # reply parent in the same channel, or -1
        self.cls: List[int] = []  # node -> visibility class id
        self.classes: List[VisClass] = []
        self._cls_id: Dict[VisClass, int] = {}
        self._leq_cache: Dict[Tuple[int, int], bool] = {}
        self.tkey: List[Tuple[float, int]] = []  # (event time, ingest index): the canonical order
        self._max_tkey: Tuple[float, int] = (float("-inf"), -1)
        self._dirty = False
        self._views: OrderedDict[bytes, GraphView] = OrderedDict()
        self._mention_cache: Dict[str, List[int]] = {}
        self._mention_n = 0
        self.replays = 0
        self._reset_state()

    # ------------------------------------------------------------ chain-building state
    def _reset_state(self) -> None:
        self.class_chains: List[Chain] = []
        self._by_slot: Dict[str, List[int]] = {}
        self._by_scope: Dict[str, List[int]] = {}
        self._chain_in_class: Dict[Tuple[str, int], int] = {}
        self.lone_negated: Set[int] = set()  # a rejection with no live slot: never a current value
        self._pending: Dict[Tuple[str, int], List[List]] = {}  # (scope, class) -> [[node, records, messages left]]
        self._names: Dict[int, Dict[str, Set[str]]] = {}  # class -> name token -> speakers
        self._names_version = 0
        self._names_cache: Dict[int, Tuple[int, Dict[str, Set[str]]]] = {}
        self._chan_hist: Dict[str, List[Tuple[int, frozenset, frozenset]]] = {}
        self._win: Dict[int, Deque[Tuple[frozenset, frozenset]]] = {}
        self._ent_df: Dict[int, Counter] = {}
        self._tok_df: Dict[int, Counter] = {}
        self._chan_classes: Dict[str, List[int]] = {}

    # ------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.messages)

    def _class_of(self, channel: str, label: Label) -> int:
        key = (channel, int(label.level), frozenset(label.compartments))
        if key not in self._cls_id:
            self._cls_id[key] = len(self.classes)
            self.classes.append(key)
        return self._cls_id[key]

    def leq(self, a: int, b: int) -> bool:
        """True when everyone who can read class ``b`` can also read class ``a``."""
        if a == b:
            return True
        k = (a, b)
        if k not in self._leq_cache:
            (ca, la, pa), (cb, lb, pb) = self.classes[a], self.classes[b]
            self._leq_cache[k] = la <= lb and pa <= pb and (ca == cb or bool(self.open_channel(ca)))
        return self._leq_cache[k]

    def _names_for(self, cls: int) -> Dict[str, Set[str]]:
        """name token -> speakers, over the classes a reader of ``cls`` can read."""
        hit = self._names_cache.get(cls)
        if hit and hit[0] == self._names_version:
            return hit[1]
        out: Dict[str, Set[str]] = {}
        for w, names in self._names.items():
            if self.leq(w, cls):
                for tok, who in names.items():
                    out.setdefault(tok, set()).update(who)
        self._names_cache[cls] = (self._names_version, out)
        return out

    def _frequent(self, df: int, cls: int) -> bool:
        return df >= self.min_df_messages and df / max(1, len(self._win.get(cls, ()))) > self.max_link_df

    def link_df(self, entity: str, cls: Optional[int] = None) -> int:
        """Messages in the recent window (as a reader of ``cls`` sees it) mentioning ``entity``."""
        self._ensure()
        cls = self.cls[-1] if cls is None else cls
        return max(self._ent_df.get(cls, {}).get(entity, 0), self._tok_df.get(cls, {}).get(entity, 0))

    def link_tokens(self, ents: Sequence[str], cls: Optional[int] = None) -> Set[str]:
        """Entity tokens that count as evidence that two statements are about the same slot."""
        cls = self.cls[-1] if cls is None else cls
        names = self._names_for(cls)
        edf, tdf = self._ent_df.get(cls, {}), self._tok_df.get(cls, {})
        out: Set[str] = set()
        for e in ents:
            if e in names or self._frequent(edf.get(e, 0), cls) or is_time_expression(e):
                continue
            out.update(t for t in e.split() if t not in names and not self._frequent(tdf.get(t, 0), cls))
        return out

    def _changes_state(self, m: Message) -> bool:
        return self.state_roles is None or state_role(m) in self.state_roles

    # ------------------------------------------------------------ ingest
    def add(self, messages: Iterable[Message], records: Dict[str, List[StateRecord]],
            suggestions: Optional[Dict[str, Label]] = None) -> None:
        """``suggestions``: optional raise-only labels per message id (e.g. from an extractor)."""
        for m in messages:
            v = len(self.messages)
            self.messages.append(m)
            self.id2node[m.id] = v
            recs = records.get(m.id, [])
            self.records.append(recs)
            self.modality.append(next((r.modality for r in recs if r.modality in STATEFUL),
                                      recs[0].modality if recs else "info"))
            # no timestamp sorts as time 0: give every message a timestamp, or none of them
            # (then the order is the ingest order)
            self.tkey.append((float(m.timestamp) if m.timestamp is not None else 0.0, v))
            raised = self.threads.add(m, (suggestions or {}).get(m.id))
            self.cls.append(self._class_of(m.channel, self.labels[v]))
            for c in raised:  # a late thread parent raised these: they were processed under a lower label
                self.cls[c] = self._class_of(self.messages[c].channel, self.labels[c])
                self._dirty = True
            if self.tkey[v] < self._max_tkey:
                self._dirty = True  # out of order: chains are rebuilt in time order on next use
            else:
                self._max_tkey = self.tkey[v]
            if not self._dirty:
                self._process(v)
        self._views.clear()

    def _ensure(self) -> None:
        """Rebuild chains in time order after an out-of-order arrival or a raised label."""
        if not self._dirty:
            return
        self._reset_state()
        for v in sorted(range(len(self.messages)), key=self.tkey.__getitem__):
            self._process(v)
        self._max_tkey = max(self.tkey) if self.tkey else (float("-inf"), -1)
        self._dirty = False
        self.replays += 1
        self._views.clear()

    def _push_window(self, channel: str, cls: int, ents: frozenset, toks: frozenset) -> None:
        known = self._chan_classes.setdefault(channel, [])
        hist = self._chan_hist.setdefault(channel, [])
        if cls not in known:
            # a class seen for the first time starts from the recent messages its readers can see
            known.append(cls)
            self._win[cls], self._ent_df[cls], self._tok_df[cls] = deque(), Counter(), Counter()
            seen = [(e, t) for w, e, t in hist if self.leq(w, cls)][-self.link_df_window:]
            for e, t in seen:
                self._win[cls].append((e, t))
                self._ent_df[cls].update(e)
                self._tok_df[cls].update(t)
        hist.append((cls, ents, toks))
        for v in known:
            if not self.leq(cls, v):
                continue
            win = self._win[v]
            win.append((ents, toks))
            self._ent_df[v].update(ents)
            self._tok_df[v].update(toks)
            if len(win) > self.link_df_window:
                old_e, old_t = win.popleft()
                self._ent_df[v].subtract(old_e)
                self._tok_df[v].subtract(old_t)

    def _process(self, v: int) -> None:
        m, cls, recs = self.messages[v], self.cls[v], self.records[v]
        toks = name_tokens(m.speaker)
        if toks:
            names = self._names.setdefault(cls, {})
            for t in toks:
                if m.speaker not in names.setdefault(t, set()):
                    names[t].add(m.speaker)
                    self._names_version += 1
        ents = frozenset(e for r in recs for e in r.entities)
        self._push_window(m.channel, cls, ents, frozenset(t for e in ents for t in e.split()))
        if "version" not in self.enabled:
            return
        stateful = [r for r in recs if r.modality in STATEFUL and r.attribute != "info"]
        scope = m.session_id or m.channel
        if not self._changes_state(m):
            if stateful and self.assistant_confirmation:
                self._pending.setdefault((scope, cls), []).append([v, stateful, CONFIRM_WINDOW])
            return
        if self._confirm(scope, cls, m):
            return  # "sounds good" carries no value of its own: the confirmed suggestion is the update
        for r in stateful:
            self._add_version(v, r, cls)

    def _confirm(self, scope: str, cls: int, m: Message) -> bool:
        """Apply pending assistant records of the same class if ``m`` affirms them. A pending
        suggestion ages with every participant message its readers can see."""
        confirmed = False
        own = self._pending.get((scope, cls))
        if own and is_affirmation(m.text):
            for node, recs, _ in own:
                for r in recs:
                    self._add_version(node, r, cls)
            own.clear()
            confirmed = True
        for (sc, w), pend in self._pending.items():
            if sc != scope or not pend or not self.leq(cls, w):
                continue
            for p in pend:
                p[2] -= 1
            pend[:] = [p for p in pend if p[2] > 0]
        return confirmed

    def record_key(self, r: StateRecord, cls: int) -> str:
        """Slot key ``holder|subject|attribute``. A name the extractor found as possessor or
        subject becomes the holder when it is a participant a reader of ``cls`` knows."""
        names = self._names_for(cls)
        hk = holder_key(r.holder)
        for cand in (r.meta or {}).get("named", ()):
            who = names.get(str(cand).strip().lower())
            if who and len(who) == 1:
                hk = f"p:{next(iter(who)).strip().lower()}"
                break
        else:
            if hk.startswith("p:"):  # "priya" -> the participant "Priya Sharma"
                who = names.get(hk[2:])
                if who and len(who) == 1:
                    hk = f"p:{next(iter(who)).strip().lower()}"
        return f"{hk}|{r.subject.lower().strip()}|{r.attribute.lower().strip()}"

    @staticmethod
    def _scope_attr(key: str) -> str:
        hk, subject, attr = key.split("|", 2) if key.count("|") >= 2 else ("", key, "")
        return f"{hk}|{subject.split('/', 1)[0]}|{attr}"

    def _chain_tokens(self, ch: Chain, cls: int) -> Set[str]:
        out: Set[str] = set()
        for _, r in ch.records[-2:]:
            out |= self.link_tokens(r.entities, cls)
        return out

    def _find_chain(self, r: StateRecord, key: str, cls: int, topicless: bool) -> Optional[int]:
        """Which slot does this record update? Only chains its readers can read are candidates.

        1. same (holder, subject, attribute) key -> that slot (LLM keys, or the same topic phrase)
        2. otherwise a slot of the same holder, scope (channel / phase) and attribute that
           shares an entity token ("go with AWS EC2" ... "scrap EC2, Cloud Run it is")
        A replacement cue alone never links: that chains chit-chat into real slots.
        Entity evidence excludes participant names, entities in more than ``max_link_df``
        of the last ``link_df_window`` messages of the channel and date/time values (see
        :meth:`link_tokens`). A record with no topic of its own (``topicless``: subject is
        just the channel) has no key worth matching, so it links on entity evidence only.
        """
        chains = self.class_chains
        same = [] if topicless else [c for c in self._by_slot.get(key, []) if self.leq(chains[c].cls, cls)]
        toks = self.link_tokens(r.entities, cls)
        in_scope = [c for c in self._by_scope.get(self._scope_attr(key), []) if self.leq(chains[c].cls, cls)]
        if r.attribute == "status" and r.modality == "done":
            # a completion resolves the latest blocker it shares an entity / topic with
            pool = same + [c for c in in_scope if c not in same]
            for c in reversed(pool):
                ch = chains[c]
                if ch.records and self.modality[ch.latest] == "blocker" and (c in same or toks & self._chain_tokens(ch, cls)):
                    return c
        if same:
            return next((c for c in reversed(same) if chains[c].cls == cls), same[-1])
        if r.explicit_key or r.attribute == "deadline" or not toks or not self.link_on_entity:
            return None
        for c in reversed(in_scope):
            if toks & self._chain_tokens(chains[c], cls):
                return c
        return None

    def _new_chain(self, slot: str, cls: int, scope_attr: str) -> int:
        c = len(self.class_chains)
        self.class_chains.append(Chain(slot, cls, scope_attr))
        self._by_slot.setdefault(slot, []).append(c)
        self._by_scope.setdefault(scope_attr, []).append(c)
        self._chain_in_class[(slot, cls)] = c
        return c

    def _add_version(self, v: int, r: StateRecord, cls: int) -> None:
        m = self.messages[v]
        key = self.record_key(r, cls)
        topicless = not r.explicit_key and r.subject == (m.channel or "")
        target = self._find_chain(r, key, cls, topicless)
        if target is None:
            if r.modality == "negated":
                self.lone_negated.add(v)
                return
            # a topicless statement is its own slot: the channel-wide key would merge everything
            c = self._new_chain(f"{key}#{m.id}" if topicless else key, cls, self._scope_attr(key))
        elif self.class_chains[target].cls == cls:
            c = target
        else:
            # the slot lives in a class this record's readers can read: join the slot, but
            # write only to this record's own class (the lower class must not change)
            t = self.class_chains[target]
            c = self._chain_in_class.get((t.slot, cls))
            if c is None:
                c = self._new_chain(t.slot, cls, t.scope_attr)
        ch = self.class_chains[c]
        if r.modality == "negated":
            # "we are NOT switching to Y": a rejected alternative. It joins the slot (so it
            # is found with it) but never becomes the current value.
            if v not in ch.rejected:
                ch.rejected.append(v)
            return
        if not ch.records or ch.records[-1][0] != v:
            ch.records.append((v, r))

    # ------------------------------------------------------------ mentions
    def mention_nodes(self, tokens: Iterable[str]) -> Dict[str, List[int]]:
        """token -> nodes whose text contains it as a word (all nodes; views apply their mask)."""
        tokens = {t for t in tokens if t}
        n = len(self.messages)
        new = tokens - set(self._mention_cache)
        todo = []
        if n > self._mention_n and self._mention_cache:
            todo.append((set(self._mention_cache), range(self._mention_n, n)))
        if new:
            todo.append((new, range(0, n)))
            for t in new:
                self._mention_cache[t] = []
        for toks, rng in todo:
            pat = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in sorted(toks, key=len, reverse=True)) + r")(?!\w)")
            for v in rng:
                for hit in set(pat.findall((self.messages[v].text or "").lower())):
                    self._mention_cache[hit].append(v)
        self._mention_n = n
        return {t: self._mention_cache[t] for t in tokens}

    # ------------------------------------------------------------ views
    def view(self, mask: Optional[np.ndarray] = None) -> GraphView:
        """What a viewer who can see exactly the nodes in ``mask`` sees (None = everything)."""
        self._ensure()
        n = len(self.messages)
        mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        key = np.packbits(mask).tobytes()
        hit = self._views.get(key)
        if hit is None:
            hit = GraphView(self, mask)
            self._views[key] = hit
            while len(self._views) > 16:
                self._views.popitem(last=False)
        else:
            self._views.move_to_end(key)
        return hit

    # The omniscient view, for analysis and tests. Retrieval always goes through view(mask).
    @property
    def edges(self) -> List[Hyperedge]:
        return self.view().edges

    @property
    def chains(self) -> List[Slot]:
        return self.view().slots

    @property
    def superseded(self) -> List[bool]:
        return [bool(x) for x in self.view().superseded]

    @property
    def chain_of(self) -> List[int]:
        return [int(x) for x in self.view().chain_of]

    @property
    def participants(self) -> Set[str]:
        return self.view().participants

    def current_of(self, node: int) -> Optional[int]:
        return self.view().current_of(node)

    def superseding_pairs(self, view: Optional[GraphView] = None) -> Set[Tuple[str, str]]:
        """(older message id, newer message id) for every pair of values in one slot:
        the graph's claim that the newer statement replaces the older one."""
        out: Set[Tuple[str, str]] = set()
        for slot in (view or self.view()).slots:
            ids = [self.messages[v].id for v, _ in slot.records]
            for i, older in enumerate(ids):
                out.update((older, newer) for newer in ids[i + 1:] if newer != older)
        return out

    def stats(self) -> Dict[str, float]:
        view = self.view()
        by: Dict[str, int] = {}
        for e in view.edges:
            by[e.type] = by.get(e.type, 0) + 1
        n = len(self.messages)
        sup = int(view.superseded.sum())
        return {
            "nodes": n, "edges": len(view.edges), "chains": len(view.slots),
            "superseded": sup, "superseded_fraction": sup / max(1, n),
            "chains_per_100_messages": 100.0 * len(view.slots) / max(1, n),
            "visibility_classes": len(self.classes), "replays": self.replays,
            **{f"edges_{k}": v for k, v in sorted(by.items())},
        }


class GraphView:
    """Everything one viewer sees, derived from the visible nodes only."""

    def __init__(self, g: Hypergraph, mask: np.ndarray) -> None:
        self.g = g
        self.mask = mask
        n = len(g.messages)
        self.nodes = np.nonzero(mask)[0]
        self.edges: List[Hyperedge] = []
        self.node_edges: List[List[int]] = [[] for _ in range(n)]
        self.superseded = np.zeros(n, bool)
        self.current = np.zeros(n, bool)
        self.chain_of = np.full(n, -1, np.int64)
        self.slots: List[Slot] = []
        vis = [int(v) for v in self.nodes]
        msgs = g.messages
        self.participants: Set[str] = {msgs[v].speaker for v in vis if state_role(msgs[v]) != ASSISTANT}
        self.speakers: Set[str] = {msgs[v].speaker for v in vis}
        ts = [g.tkey[v][0] for v in vis]
        self.t_min, self.t_max = (min(ts), max(ts)) if ts else (0.0, 0.0)
        # name token -> visible speakers it can refer to
        self.name_tokens: Dict[str, Set[str]] = {}
        for s in self.speakers:
            for t in name_tokens(s):
                self.name_tokens.setdefault(t, set()).add(s)
        en = g.enabled

        def group(key_of: Callable[[int], Optional[str]]) -> Dict[str, List[int]]:
            out: Dict[str, List[int]] = {}
            for v in vis:
                k = key_of(v)
                if k is not None:
                    out.setdefault(k, []).append(v)
            return out

        if "channel" in en:
            for k, nodes in group(lambda v: msgs[v].channel).items():
                self._add("channel", k, nodes)
        if "speaker" in en:
            for k, nodes in group(lambda v: msgs[v].speaker).items():
                self._add("speaker", k, nodes)
        if "reply" in en:
            root: Dict[int, int] = {}
            for v in vis:  # ascending ids are not time order, so walk up instead of relying on order
                path, u = [], v
                while u not in root:
                    p = g.parent[u]
                    if p == -1 or not mask[p]:
                        root[u] = u
                        break
                    path.append(u)
                    u = p
                for w in path:
                    root[w] = root[u]
            threads: Dict[int, List[int]] = {}
            for v in vis:
                threads.setdefault(root[v], []).append(v)
            for r, nodes in threads.items():
                if len(nodes) > 1:
                    self._add("reply", msgs[r].id, nodes)
        if "episode" in en:
            for scope, nodes in group(lambda v: msgs[v].session_id or msgs[v].channel).items():
                nodes = sorted(nodes, key=g.tkey.__getitem__)
                cur: List[int] = []
                for v in nodes:
                    if cur and (len(cur) >= g.max_episode or (
                            msgs[v].timestamp is not None and msgs[cur[-1]].timestamp is not None
                            and msgs[v].timestamp - msgs[cur[-1]].timestamp > g.gap)):
                        self._add("episode", f"{scope}#{msgs[cur[0]].id}", sorted(cur))
                        cur = []
                    cur.append(v)
                if cur:
                    self._add("episode", f"{scope}#{msgs[cur[0]].id}", sorted(cur))
        if "entity" in en:
            ents: Dict[str, List[int]] = {}
            for v in vis:
                for e in dict.fromkeys(e for r in g.records[v] for e in r.entities):
                    if e not in self.name_tokens:  # names get mention edges instead
                        ents.setdefault(e, []).append(v)
            for k, nodes in ents.items():
                self._add("entity", k, nodes)
        if "mention" in en and self.name_tokens:
            hits = g.mention_nodes(self.name_tokens)
            by_speaker: Dict[str, Set[int]] = {}
            for tok, who in self.name_tokens.items():
                for s in who:
                    by_speaker.setdefault(s, set()).update(v for v in hits[tok] if mask[v] and msgs[v].speaker != s)
            for s in sorted(by_speaker):
                if by_speaker[s]:
                    self._add("mention", s, sorted(by_speaker[s]))
        if "version" in en:
            self._build_slots()

    def _add(self, etype: str, key: str, nodes: Sequence[int]) -> int:
        e = len(self.edges)
        self.edges.append(Hyperedge(etype, key, np.asarray(nodes, np.int64)))
        for v in nodes:
            self.node_edges[v].append(e)
        return e

    def _build_slots(self) -> None:
        g, mask = self.g, self.mask
        merged: Dict[str, Tuple[List[Tuple[int, StateRecord]], List[int]]] = {}
        for ch in g.class_chains:
            recs = [(v, r) for v, r in ch.records if mask[v]]
            rej = [v for v in ch.rejected if mask[v]]
            if recs or rej:
                slot = merged.setdefault(ch.slot, ([], []))
                slot[0].extend(recs)
                slot[1].extend(rej)
        for key, (recs, rej) in merged.items():
            recs.sort(key=lambda vr: g.tkey[vr[0]])
            rej = sorted(dict.fromkeys(rej), key=g.tkey.__getitem__)
            nodes = sorted(dict.fromkeys([v for v, _ in recs] + rej), key=g.tkey.__getitem__)
            slot = Slot(key, recs, rej, nodes)
            i = len(self.slots)
            self.slots.append(slot)
            slot.edge = self._add("version", f"{key}#{i}", nodes)
            for v in nodes:
                self.chain_of[v] = i
                self.superseded[v] = True
        for slot in self.slots:
            if slot.latest is not None:
                self.current[slot.latest] = True
        for v in g.lone_negated:
            if mask[v]:
                self.superseded[v] = True
        self.superseded &= ~self.current  # the latest value of any slot is never stale
        self._own: Optional[Dict[str, List[int]]] = None

    def current_of(self, node: int) -> Optional[int]:
        c = int(self.chain_of[node])
        return self.slots[c].latest if c >= 0 else None

    def own_chain_nodes(self, speaker: str) -> List[int]:
        """A speaker's state-bearing messages (nodes in any slot), oldest first."""
        if getattr(self, "_own", None) is None:
            own: Dict[str, List[int]] = {}
            for v in np.nonzero(self.chain_of >= 0)[0]:
                own.setdefault(self.g.messages[v].speaker, []).append(int(v))
            for nodes in own.values():
                nodes.sort(key=self.g.tkey.__getitem__)
            self._own = own
        return self._own.get(speaker, [])
