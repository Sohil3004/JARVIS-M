"""LLM state extractor (Qwen3 or any chat backend), batched with context.

Each call sees up to ``context`` preceding messages of the same channel and session (for
coreference: "it", "that plan") and extracts records for up to ``batch`` target messages.
The LLM normalises ``subject`` and ``attribute`` so version chains can link on exact keys,
which is the main thing heuristics cannot do. The prompt is domain-neutral (team chat,
user <-> assistant chat, personal chat).

* **Slot registry.** Per (channel, holder) the most recent ``registry_size`` subjects are
  listed in the prompt ("reuse one of these verbatim if the fact is about the same thing").
  The registry is saved next to the extraction cache, so a resumed run sees the same prompts.
* **Prefilter.** ``light`` (default) skips only messages with fewer than 3 content words and
  no digit or entity; ``heuristic`` skips what the heuristic extractor sees as chit-chat (an
  ablation); ``none`` sends everything.
* **Batch cache.** Key = hash of (PROMPT_VERSION, model, context and target texts with their
  speaker and channel), so an identical session (LongMemEval-S repeats sessions across
  questions) is extracted once.
  The registry is not part of the key: a hit reuses records extracted under another
  registry state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from ..llm import ChatModel, build_chat
from ..text import word_tokens
from ..types import GROUP_MIN_PARTICIPANTS, Message
from .heuristic import HeuristicExtractor, entities
from .records import ATTRIBUTES, MODALITIES, StateRecord

PROMPT_VERSION = "2"

SYSTEM = f"""You extract state from chat messages for a memory system. The chat can be a team chat, a \
conversation between a user and an AI assistant, or a personal chat between friends or family.
For each TARGET message, output zero or more records. A record captures one fact that could later answer a question.
Fields:
- msg: the target message id
- holder: whose fact it is:
  "speaker:<name>" for a fact about the speaker (where they live, what they like or own, their job, their own plans);
  "group:<channel>" for a fact about the group, team or project;
  "person:<name>" for a fact about a named third party ("my sister moved to Leeds" -> person:sister of the speaker is fine as "person:<speaker>'s sister").
  In a GROUP chat, task ownership, status, deadlines and decisions belong to the group even when said in the first
  person ("I'll take load testing" -> holder group:<channel>, attribute owner, value the speaker's name).
  In a ONE-TO-ONE chat, and for personal statements anywhere ("my visa got approved", "I moved to Pune"),
  first-person facts belong to the speaker.
  An assistant's suggestion has the holder of the person it is for; it is a proposal until that person accepts it.
- subject: short canonical noun phrase the fact is about. If KNOWN SUBJECTS lists one about the same thing, reuse it
  verbatim (e.g. "backend hosting", "release date", "home", "diet", "car").
- attribute: one of {", ".join(ATTRIBUTES)}
- value: the value stated, as a short phrase (e.g. "Google Cloud Run", "Monday 10 AM", "Charlie", "Pune", "vegetarian")
- modality: one of decision (settled), commitment (someone will do it), done (completed/resolved), blocker (open problem),
  negated (explicitly not / ruled out), state (a current fact about the holder: lives in, prefers, owns, works as),
  proposal (suggested, not settled), question, info
- replaces: true if this message changes or reverses an earlier value of the same holder+subject+attribute
- entities: important names, tools, ids, places, dates mentioned
- sensitivity: optional, only when the text marks itself as restricted (e.g. "confidential", "HR only", "don't share");
  a level name such as "confidential", or {{"level": ..., "compartments": [...]}}
Rules: future intent ("I will fix") is commitment, not done. A rejected option ("we are not using X") is negated.
"No longer X" about a person's own state is a new state value, not negated. Skip greetings and chit-chat.
Output ONLY a JSON array, no prose. Output [] if nothing."""

# content words below which (and with no digit or entity) the light prefilter skips a message
LIGHT_MIN_WORDS = 3


def _fmt(m: Message) -> str:
    when = m.meta.get("time_str") or (f"{m.timestamp:.0f}" if m.timestamp is not None else "")
    rep = f" reply_to={m.reply_to}" if m.reply_to else ""
    return f"<{m.id}> [{when} #{m.channel} {m.speaker}{rep}] {m.text.strip()[:1200]}"


def _valid_holder(holder: str) -> bool:
    kind, sep, name = holder.partition(":")
    return bool(sep) and kind in ("speaker", "group", "person") and bool(name.strip())


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
        if not _valid_holder(holder):
            holder = f"group:{msg.channel or 'default'}"  # no (valid) holder from the model: a shared fact
        attribute = str(it.get("attribute", "info")).lower().strip() or "info"
        if attribute not in ATTRIBUTES:
            attribute = "info"
        meta: Dict[str, Any] = {}
        sens = it.get("sensitivity")
        if isinstance(sens, (str, dict)) and sens:  # raise-only label suggestion (jarvis_m.access)
            meta["sensitivity"] = sens
        out.append(StateRecord(
            msg_id=mid, speaker=msg.speaker, modality=mod, attribute=attribute,
            value=str(it.get("value", ""))[:200], subject=str(it.get("subject", "")).lower().strip()[:80],
            entities=ents or entities(msg.text), timestamp=msg.timestamp,
            replaces_cue=bool(it.get("replaces", False)), explicit_key=True, holder=holder, meta=meta,
        ))
    return out


def light_keep(m: Message) -> bool:
    """Keep a message unless it has fewer than LIGHT_MIN_WORDS content words and no digit or entity."""
    text = m.text or ""
    return len(word_tokens(text)) >= LIGHT_MIN_WORDS or bool(re.search(r"\d", text)) or bool(entities(text))


class BatchCache(Protocol):
    def get_batch(self, key: str) -> Optional[List[Dict[str, Any]]]: ...

    def put_batch(self, key: str, records: List[Dict[str, Any]]) -> None: ...


class SlotRegistry:
    """Most recent subjects per (channel, holder), oldest first."""

    def __init__(self, size: int = 40, path: Optional[str | Path] = None) -> None:
        self.size = size
        self.path = Path(path) if path else None
        self.slots: Dict[str, OrderedDict[str, None]] = {}
        if self.path is not None and self.path.exists():
            for k, subjects in json.loads(self.path.read_text(encoding="utf-8")).items():
                self.slots[k] = OrderedDict((s, None) for s in subjects)

    @staticmethod
    def key(scope: str, holder: str) -> str:
        return f"{scope}|{holder.strip().lower()}"

    def subjects(self, scope: str, holder: str) -> List[str]:
        return list(self.slots.get(self.key(scope, holder), ()))

    def add(self, scope: str, records: Sequence[StateRecord]) -> None:
        for r in records:
            if not r.subject:
                continue
            od = self.slots.setdefault(self.key(scope, r.holder), OrderedDict())
            od.pop(r.subject, None)
            od[r.subject] = None
            while len(od) > self.size:
                od.popitem(last=False)

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({k: list(v) for k, v in self.slots.items()}), encoding="utf-8")


def batch_key(model: str, context: Sequence[Message], targets: Sequence[Message]) -> str:
    h = hashlib.sha1()
    def texts(ms: Sequence[Message]) -> List[List[str]]:
        return [[m.channel, m.speaker, m.text] for m in ms]  # what the prompt shows, minus ids and times

    h.update(json.dumps([PROMPT_VERSION, model, texts(context), texts(targets)]).encode())
    return h.hexdigest()


class LLMExtractor:
    def __init__(self, chat: Any = None, batch: int = 8, context: int = 6, max_tokens: int = 1024,
                 prefilter: Optional[str] = "light", registry_size: int = 40,
                 cache: Optional[BatchCache] = None, registry_path: Optional[str] = None) -> None:
        self.chat: ChatModel = chat if hasattr(chat, "chat") else build_chat(chat)
        self.batch, self.context, self.max_tokens = batch, context, max_tokens
        if prefilter not in (None, "none", "light", "heuristic"):
            raise ValueError("prefilter must be none, light or heuristic")
        self.prefilter = prefilter or "none"
        self._heuristic = HeuristicExtractor()
        self.registry = SlotRegistry(registry_size, registry_path)
        self.cache = cache
        self.name = f"llm:{self.chat.name}"
        self.calls = 0
        self.batches = 0
        self.cache_hits = 0

    def _keep(self, m: Message) -> bool:
        if self.prefilter == "light":
            return light_keep(m)
        if self.prefilter == "heuristic":
            return any(r.modality not in ("question", "info") or r.entities for r in self._heuristic.extract_one(m))
        return True

    def _chat_line(self, channel: str, speakers: Sequence[str]) -> str:
        n = len(speakers)
        kind = "GROUP chat" if n >= GROUP_MIN_PARTICIPANTS else "ONE-TO-ONE chat"
        return f"CHAT: #{channel}, {kind} ({n} participant{'s' if n != 1 else ''}: {', '.join(speakers[:12])})"

    def _known(self, channel: str, targets: Sequence[Message]) -> str:
        holders = [f"group:{channel}"] + list(dict.fromkeys(f"speaker:{m.speaker}" for m in targets))
        lines = []
        for h in holders:
            subs = self.registry.subjects(channel, h)
            if subs:
                lines.append(f"- {h}: " + "; ".join(subs))
        return ("KNOWN SUBJECTS (reuse one verbatim if the fact is about the same thing):\n" + "\n".join(lines)
                + "\n\n") if lines else ""

    def extract(self, messages: Sequence[Message],
                on_batch: Optional[Callable[[List[StateRecord], List[str]], None]] = None) -> List[StateRecord]:
        """Messages must be in event order. ``on_batch(records, done_ids)`` lets callers
        persist progress (see ExtractionStore)."""
        out: List[StateRecord] = []
        groups: Dict[Tuple[str, str], List[Message]] = {}
        speakers: Dict[str, List[str]] = {}
        for m in messages:
            groups.setdefault((m.channel, m.session_id or ""), []).append(m)
            if (m.speaker or "").lower() != "assistant" and (m.role or "").lower() != "assistant":
                lst = speakers.setdefault(m.channel, [])
                if m.speaker not in lst:
                    lst.append(m.speaker)
        for (channel, _), msgs in groups.items():
            i = 0
            while i < len(msgs):
                chunk = msgs[i: i + self.batch]
                targets = [m for m in chunk if self._keep(m)]
                ctx = msgs[max(0, i - self.context): i]
                recs: List[StateRecord] = []
                if targets:
                    self.batches += 1
                    key = batch_key(self.chat.name, ctx, targets)
                    hit = self.cache.get_batch(key) if self.cache is not None else None
                    if hit is not None:
                        self.cache_hits += 1
                        for d in hit:
                            d = dict(d)
                            t = targets[d.pop("i")]
                            d.update(msg_id=t.id, speaker=t.speaker, timestamp=t.timestamp)
                            recs.append(StateRecord.from_json(d))
                    else:
                        user = (self._chat_line(channel, speakers.get(channel, [])) + "\n\n"
                                + self._known(channel, targets)
                                + "CONTEXT (earlier messages, do not extract):\n" + "\n".join(_fmt(m) for m in ctx)
                                + "\n\nTARGET messages:\n" + "\n".join(_fmt(m) for m in targets))
                        raw = self.chat.chat(SYSTEM, user, self.max_tokens)
                        self.calls += 1
                        recs = parse_records(raw, {m.id: m for m in targets})
                        if self.cache is not None:
                            index = {m.id: j for j, m in enumerate(targets)}
                            self.cache.put_batch(key, [{**r.to_json(), "i": index[r.msg_id]} for r in recs])
                    self.registry.add(channel, recs)
                out.extend(recs)
                if on_batch:
                    on_batch(recs, [m.id for m in chunk])
                    self.registry.save()
                i += self.batch
        out.sort(key=lambda r: (r.timestamp or 0.0, r.msg_id))
        return out

    def hit_rate(self) -> float:
        return self.cache_hits / self.batches if self.batches else 0.0
