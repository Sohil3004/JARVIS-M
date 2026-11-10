"""Core data types shared by datasets, memory systems and the eval harness.

Design rule: a memory system only ever sees :class:`Message` objects. Benchmark
annotations that would leak the answer (gold evidence, decision labels, noise
flags, ...) live on :class:`Dataset.annotations` / :class:`Question` and are
never passed to ``MemorySystem.ingest``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Message:
    """One chat message as a memory system is allowed to see it."""

    id: str
    speaker: str
    text: str
    timestamp: Optional[float] = None  # event time, epoch seconds (not wall-clock)
    channel: str = "default"  # group / channel / DM name
    session_id: Optional[str] = None
    reply_to: Optional[str] = None
    role: Optional[str] = None  # speaker's role label, if the benchmark exposes one
    meta: Mapping[str, Any] = field(default_factory=dict)  # other *visible* fields

    def header(self) -> str:
        """Passage header in the same shape GroupMemBench shows its QA agent."""
        tags = [f"user={self.speaker}"]
        if self.role:
            tags.append(f"speaker_role={self.role}")
        if self.channel and self.channel != "default":
            tags.append(f"channel={self.channel}")
        for k in ("phase_name", "topic"):
            if self.meta.get(k):
                tags.append(f"{k}={self.meta[k]}")
        if self.meta.get("time_str"):
            tags.append(f"timestamp={self.meta['time_str']}")
        elif self.timestamp is not None:
            tags.append(f"timestamp={self.timestamp:.0f}")
        if self.reply_to:
            tags.append(f"reply_to={self.reply_to}")
        tags.append(f"msg={self.id}")
        return "[" + " / ".join(tags) + "]"

    def render(self) -> str:
        return f"{self.header()}\n{self.text}"


@dataclass
class Conversation:
    id: str
    messages: List[Message]

    @property
    def participants(self) -> List[str]:
        seen: Dict[str, None] = {}
        for m in self.messages:
            seen.setdefault(m.speaker, None)
        return list(seen)


@dataclass
class Question:
    id: str
    conversation_id: str
    question: str
    answer: str
    asker: Optional[str] = None
    category: str = "default"
    evidence_ids: Tuple[str, ...] = ()  # gold message ids, when the benchmark has them
    stale_answers: Tuple[str, ...] = ()  # superseded values; mentioning one = stale error
    # messages stating a superseded / rejected value: ranking one above the best gold message = stale-rank error
    stale_evidence_ids: Tuple[str, ...] = ()
    is_abstention: bool = False  # correct behaviour is to say the answer is not there
    question_time: Optional[float] = None  # "now" for the question, epoch seconds
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    name: str
    conversations: Dict[str, Conversation]
    questions: List[Question]
    # Hidden labels: message id -> labels (decision points, noise flags...), plus the named
    # blocks "supersession" (gold version pairs), "channels" and "principals" (access control
    # directory). Training / analysis / policy only. NEVER hand message labels to a memory system.
    annotations: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None  # team | assistant | personal | enterprise (set from the dataset config)

    def questions_for(self, conversation_id: str) -> List[Question]:
        return [q for q in self.questions if q.conversation_id == conversation_id]

    def filter(
        self,
        categories: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
        conversations: Optional[Sequence[str]] = None,
    ) -> Dataset:
        qs = self.questions
        if categories:
            cats = set(categories)
            qs = [q for q in qs if q.category in cats]
        if conversations:
            keep = set(conversations)
            qs = [q for q in qs if q.conversation_id in keep]
        if limit is not None:
            qs = qs[:limit]
        used = {q.conversation_id for q in qs}
        convs = {k: v for k, v in self.conversations.items() if k in used}
        return Dataset(self.name, convs, qs, self.annotations, self.domain)

    def summary(self) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for q in self.questions:
            cats[q.category] = cats.get(q.category, 0) + 1
        return {
            "name": self.name,
            "conversations": len(self.conversations),
            "messages": sum(len(c.messages) for c in self.conversations.values()),
            "questions": len(self.questions),
            "categories": cats,
        }
