"""State records: typed facts extracted from messages.

A record never replaces its message: the hypergraph keeps raw messages as nodes
and uses records only to add structure (entity, version-chain hyperedges).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

MODALITIES = (
    "decision",    # the group settled something ("we're going with X")
    "commitment",  # someone will do something ("I'll fix it by Friday")
    "done",        # something was completed / resolved
    "blocker",     # a problem is open
    "negated",     # explicitly not / no longer
    "proposal",    # suggestion, not settled
    "question",
    "info",
)
# Modalities that set the current value of a (subject, attribute) slot.
STATEFUL = frozenset({"decision", "commitment", "done", "negated", "blocker"})


def holder_key(holder: str) -> str:
    """Normal form of a holder for slot keys. ``speaker:Priya`` and ``person:Priya`` are the
    same holder (``p:priya``); ``group:<scope>`` is kept; no holder is the empty string."""
    kind, sep, name = (holder or "").partition(":")
    name = name.strip().lower()
    if sep and kind.strip().lower() in ("speaker", "person"):
        return f"p:{name}"
    return (holder or "").strip().lower()


@dataclass(frozen=True)
class StateRecord:
    msg_id: str
    speaker: str
    modality: str
    attribute: str  # decision | deadline | owner | status | plan | info ...
    value: str
    subject: str = ""  # what the slot is about ("backend hosting", phase name, ...)
    entities: Tuple[str, ...] = ()
    timestamp: Optional[float] = None
    replaces_cue: bool = False  # message signals replacement ("instead", "scrap", "moved to")
    explicit_key: bool = False  # subject/attribute normalised by an LLM: chain on exact key
    meta: Dict[str, Any] = field(default_factory=dict)
    # Whose fact this is: a participant (``speaker:<name>``), a group (``group:<scope>``, for
    # "we / our / the team") or a named third party (``person:<name>``). Slots never chain
    # across holders. Empty = not known (records extracted before holders existed).
    holder: str = ""

    @property
    def key(self) -> str:
        """Slot key: ``holder|subject|attribute``."""
        return f"{holder_key(self.holder)}|{self.subject.lower().strip()}|{self.attribute.lower().strip()}"

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entities"] = list(self.entities)
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> StateRecord:
        d = dict(d)
        d["entities"] = tuple(d.get("entities", ()))
        return StateRecord(**d)

    def with_holder(self, holder: str) -> StateRecord:
        return replace(self, holder=holder)

    def render(self) -> str:
        who = f" ({self.speaker})" if self.speaker else ""
        subj = f"{self.subject}: " if self.subject else ""
        return f"{self.modality.upper()} {self.attribute} — {subj}{self.value}{who}"
