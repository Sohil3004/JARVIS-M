"""The one interface every memory system implements (ours and baselines).

The harness owns the reader LLM and the judge, so systems are compared on the
context they hand over, nothing else. Ablations are config flags on systems.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import Message


@dataclass
class RetrievedItem:
    text: str  # what goes into the reader prompt
    score: float
    source_ids: Tuple[str, ...]  # message ids this item came from (for evidence recall)
    kind: str = "message"  # message | state | summary | node ...
    timestamp: Optional[float] = None


@dataclass
class Retrieval:
    items: List[RetrievedItem]
    context: str
    n_tokens: int
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_ids(self) -> List[str]:
        out: List[str] = []
        seen = set()
        for it in self.items:
            for s in it.source_ids:
                if s not in seen:
                    seen.add(s)
                    out.append(s)
        return out


class MemorySystem(ABC):
    """Streaming memory over one conversation.

    Lifecycle per conversation: ``reset()`` -> ``ingest(messages)`` (in event
    order; may be called several times) -> many ``retrieve(...)`` calls.
    """

    name: str = "memory"

    def reset(self) -> None:  # noqa: B027 - optional hook, default no-op
        return None

    @abstractmethod
    def ingest(self, messages: Sequence[Message]) -> None: ...

    @abstractmethod
    def retrieve(
        self,
        query: str,
        *,
        asker: Optional[str] = None,
        budget_tokens: Optional[int] = None,
        max_items: Optional[int] = None,
        now: Optional[float] = None,
    ) -> Retrieval: ...

    def stats(self) -> Dict[str, Any]:
        """Size / cost counters reported next to accuracy (nodes, LLM calls...)."""
        return {}
