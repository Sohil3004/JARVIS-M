"""Adapter: Jarvis-M v2 (regex state reconciliation) as a harness baseline.

Wraps ``legacy/v2_regex/core/reconciliation_engine.py`` unchanged, then mirrors
``MemoryConditionedContextOptimizer``: reconciled state header + the most recent
messages the engine did *not* consume, filled to the token budget. (The v2
heuristic token pruner is skipped: it needs an HF tokenizer and was not a
meaningful part of the result.)
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ..interfaces import MemorySystem, Retrieval, RetrievedItem
from ..text import ApproxTokenCounter, TokenCounter
from ..types import Message


def _legacy_core_dir() -> Path:
    env = os.environ.get("JARVIS_LEGACY_DIR")
    if env:
        return Path(env) / "v2_regex" / "core"
    return Path(__file__).resolve().parents[3] / "legacy" / "v2_regex" / "core"


def load_reconciliation_module():
    path = _legacy_core_dir() / "reconciliation_engine.py"
    spec = importlib.util.spec_from_file_location("jarvis_legacy_v2_reconciliation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load legacy v2 engine from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class LegacyV2Memory(MemorySystem):
    name = "legacy_v2"

    def __init__(self, counter: Optional[TokenCounter] = None) -> None:
        self.counter = counter or ApproxTokenCounter()
        self._mod = load_reconciliation_module()
        self.reset()

    def reset(self) -> None:
        self.engine = self._mod.TemporalStateReconciliationEngine()
        self.messages: List[Message] = []
        self.consumed: Dict[str, str] = {}  # raw text -> message id

    def ingest(self, messages: Sequence[Message]) -> None:
        for m in messages:
            nodes = self.engine.process_utterance(m.speaker, m.text, m.session_id or m.channel, timestamp=m.timestamp)
            if nodes:
                self.consumed[m.text.strip()] = m.id
            self.messages.append(m)

    def state_header(self) -> str:
        return self.engine.format_reconciled_prefix()

    def retrieve(self, query, *, asker=None, budget_tokens=None, max_items=None, now=None, audience=None,
                 channel=None) -> Retrieval:
        header = self.state_header()
        gt = self.engine.get_reconciled_ground_truth()
        src = tuple(
            self.consumed[n.raw_text] for n in gt["active_states"] + gt["active_blockers"] + gt["resolved_blockers"]
            if n.raw_text in self.consumed
        )
        items = [RetrievedItem(header, 1.0, src, "state")]
        used = self.counter.count(header)
        recent: List[RetrievedItem] = []
        for m in reversed(self.messages):
            if m.text.strip() in self.consumed:
                continue
            if max_items is not None and len(recent) >= max_items:
                break
            txt = m.render()
            c = self.counter.count(txt)
            if budget_tokens is not None and used + c > budget_tokens:
                break
            recent.append(RetrievedItem(txt, 0.0, (m.id,), "message", m.timestamp))
            used += c
        items += list(reversed(recent))
        ctx = header + ("\n\n[Recent Context]:\n" + "\n\n".join(i.text for i in items[1:]) if len(items) > 1 else "")
        return Retrieval(items, ctx, self.counter.count(ctx), {"state_nodes": len(gt["all_nodes"])})

    def stats(self):
        return {"state_nodes": len(self.engine.nodes), "messages": len(self.messages)}
