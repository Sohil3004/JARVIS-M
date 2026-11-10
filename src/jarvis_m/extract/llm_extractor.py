"""LLM state extractor (Qwen3 or any chat backend), batched with context.

Each call sees ``context`` preceding messages (for coreference: "it", "that
plan") and extracts records for ``batch`` target messages. The LLM normalises
``subject`` and ``attribute`` so version chains can link on exact keys, which is
the main thing heuristics cannot do.

Cost control: ``prefilter: heuristic`` skips messages the heuristic extractor
sees as pure chit-chat (no cue, no entity), typically 30-50% of a chat.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from ..llm import ChatModel, build_chat
from ..types import Message
from .heuristic import HeuristicExtractor, entities
from .records import MODALITIES, StateRecord

SYSTEM = """You extract state from team chat messages for a memory system.
For each TARGET message, output zero or more records. A record captures one fact that could later answer a question.
Fields:
- msg: the target message id
- subject: short canonical noun phrase the fact is about, reused verbatim across messages about the same thing (e.g. "backend hosting", "release date", "transaction database", "UX prototype approval scope")
- attribute: one of decision, deadline, owner, status, plan, requirement, preference, info
- value: the current value stated, as a short phrase (e.g. "Google Cloud Run", "Monday 10 AM", "Charlie")
- modality: one of decision (settled), commitment (someone will do it), done (completed/resolved), blocker (open problem), negated (explicitly not / no longer), proposal (suggested, not settled), question, info
- replaces: true if this message changes or reverses an earlier value of the same subject+attribute
- entities: important names, tools, ids, dates mentioned
Rules: future intent ("I will fix") is commitment, not done. "not"/"no longer" is negated. Skip greetings and chit-chat.
Output ONLY a JSON array, no prose. Output [] if nothing."""


def _fmt(m: Message) -> str:
    when = m.meta.get("time_str") or (f"{m.timestamp:.0f}" if m.timestamp is not None else "")
    rep = f" reply_to={m.reply_to}" if m.reply_to else ""
    return f"<{m.id}> [{when} #{m.channel} {m.speaker}{rep}] {m.text.strip()[:1200]}"


def parse_records(raw: str, by_id: Dict[str, Message]) -> List[StateRecord]:
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: List[StateRecord] = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        mid = str(it.get("msg", "")).strip("<> ")
        msg = by_id.get(mid)
        if msg is None:
            continue
        mod = str(it.get("modality", "info")).lower()
        if mod not in MODALITIES:
            mod = "info"
        ents = it.get("entities") or []
        ents = tuple(str(e).lower().strip() for e in ents if str(e).strip()) if isinstance(ents, list) else ()
        holder = str(it.get("holder") or "").strip()
        if holder.partition(":")[0] not in ("speaker", "group", "person") or not holder.partition(":")[2].strip():
            holder = f"group:{msg.channel or 'default'}"  # no (valid) holder from the model: a shared fact
        out.append(StateRecord(
            msg_id=mid, speaker=msg.speaker, modality=mod,
            attribute=str(it.get("attribute", "info")).lower().strip() or "info",
            value=str(it.get("value", ""))[:200], subject=str(it.get("subject", "")).lower().strip()[:80],
            entities=ents or entities(msg.text), timestamp=msg.timestamp,
            replaces_cue=bool(it.get("replaces", False)), explicit_key=True, holder=holder,
        ))
    return out


class LLMExtractor:
    def __init__(self, chat: Any = None, batch: int = 8, context: int = 6, max_tokens: int = 1024,
                 prefilter: Optional[str] = "heuristic") -> None:
        self.chat: ChatModel = chat if hasattr(chat, "chat") else build_chat(chat)
        self.batch, self.context, self.max_tokens = batch, context, max_tokens
        self.prefilter = HeuristicExtractor() if prefilter == "heuristic" else None
        self.name = f"llm:{self.chat.name}"
        self.calls = 0

    def _keep(self, m: Message) -> bool:
        if self.prefilter is None:
            return True
        return any(r.modality not in ("question", "info") or r.entities for r in self.prefilter.extract_one(m))

    def extract(self, messages: Sequence[Message], on_batch=None) -> List[StateRecord]:
        """Messages must be in event order. ``on_batch(records, done_ids)`` lets
        callers persist progress (see ExtractionStore)."""
        out: List[StateRecord] = []
        by_channel: Dict[str, List[Message]] = {}
        for m in messages:
            by_channel.setdefault(m.channel, []).append(m)
        for chan_msgs in by_channel.values():
            i = 0
            while i < len(chan_msgs):
                targets = [m for m in chan_msgs[i: i + self.batch] if self._keep(m)]
                ctx = chan_msgs[max(0, i - self.context): i]
                done_ids = [m.id for m in chan_msgs[i: i + self.batch]]
                recs: List[StateRecord] = []
                if targets:
                    user = ("CONTEXT (earlier messages, do not extract):\n" + "\n".join(_fmt(m) for m in ctx)
                            + "\n\nTARGET messages:\n" + "\n".join(_fmt(m) for m in targets))
                    raw = self.chat.chat(SYSTEM, user, self.max_tokens)
                    self.calls += 1
                    recs = parse_records(raw, {m.id: m for m in targets})
                out.extend(recs)
                if on_batch:
                    on_batch(recs, done_ids)
                i += self.batch
        out.sort(key=lambda r: (r.timestamp or 0.0, r.msg_id))
        return out
