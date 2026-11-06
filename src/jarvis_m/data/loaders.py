"""Benchmark loaders -> :class:`jarvis_m.types.Dataset`.

Supported: GroupMemBench (multi-party, primary), LoCoMo, LongMemEval, and the
native ``jarvis`` JSON format (our stress sets / WhatsApp eval set / fixtures).
Run ``jarvis-eval data fetch <name>`` first for public benchmarks.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..types import Conversation, Dataset, Message, Question


def data_root() -> Path:
    return Path(os.environ.get("JARVIS_DATA_DIR", "data/raw"))


def _ts(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ------------------------------------------------------------- GroupMemBench
GMB_DOMAINS = ("Finance", "Technology", "Healthcare", "Manufacturing")
GMB_CATEGORIES = ("multi_hop", "knowledge_update", "temporal", "user_implicit", "term_ambiguity", "abstention")
# Fields shown to the official QA agent (format_retrieved_message) -> visible.
# Everything else (is_noise, is_decision_point, decision_*, stage, tone...) -> hidden annotations.
_GMB_VISIBLE_META = ("phase_name", "topic")


def load_groupmembench(root: Optional[str] = None, domains: Sequence[str] = GMB_DOMAINS,
                       categories: Optional[Sequence[str]] = None, **_: Any) -> Dataset:
    base = Path(root) if root else data_root() / "GroupMemBench"
    convs: Dict[str, Conversation] = {}
    questions: List[Question] = []
    ann: Dict[str, Dict[str, Any]] = {}
    for dom in domains:
        path = base / "data" / "final" / dom / f"synthetic_domain_channels_rolevariants_{dom}.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        msgs: List[Message] = []
        for channel, items in raw.items():
            for m in items:
                if not isinstance(m, dict):
                    continue
                ts = _ts(datetime.fromisoformat(m["timestamp"]))
                meta = {k: m[k] for k in _GMB_VISIBLE_META if m.get(k)}
                meta["time_str"] = m["timestamp"]
                msgs.append(Message(
                    id=m["msg_node"], speaker=m.get("author") or "?", text=(m.get("content") or "").strip(),
                    timestamp=ts, channel=channel, reply_to=m.get("reply_to"), role=m.get("role"), meta=meta,
                ))
                ann[m["msg_node"]] = {k: v for k, v in m.items() if k not in ("content",) and k not in _GMB_VISIBLE_META}
        msgs.sort(key=lambda x: (x.timestamp, x.id))  # streaming order = event time across channels
        cid = f"gmb-{dom}"
        convs[cid] = Conversation(cid, msgs)
        end_time = msgs[-1].timestamp if msgs else None
        for cat in categories or GMB_CATEGORIES:
            qpath = base / "questions" / dom / f"{cat}.jsonl"
            if not qpath.exists():
                continue
            for line in qpath.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                q = json.loads(line)
                questions.append(Question(
                    id=f"{dom}/{q['id']}", conversation_id=cid, question=q["question"], answer=str(q.get("answer", "")),
                    asker=q.get("asking_user_id"), category=cat, is_abstention=(cat == "abstention"),
                    question_time=end_time, meta={"domain": dom},
                ))
    return Dataset("groupmembench", convs, questions, ann)


# ------------------------------------------------------------------- LoCoMo
LOCOMO_CATEGORIES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def _locomo_time(s: str) -> float:
    return _ts(datetime.strptime(s.strip(), "%I:%M %p on %d %B, %Y"))


def load_locomo(path: Optional[str] = None, categories: Optional[Sequence[str]] = None, **_: Any) -> Dataset:
    p = Path(path) if path else data_root() / "locomo" / "locomo10.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    convs, questions = {}, []
    for sample in raw:
        cid = sample["sample_id"]
        conv = sample["conversation"]
        msgs: List[Message] = []
        n_sess = sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))
        last_ts = None
        for s in n_sess:
            t0 = _locomo_time(conv[f"session_{s}_date_time"])
            for j, turn in enumerate(conv[f"session_{s}"]):
                text = turn.get("text", "")
                if turn.get("blip_caption"):
                    text += f" [shares a photo: {turn['blip_caption']}]"
                last_ts = t0 + j
                msgs.append(Message(id=turn["dia_id"], speaker=turn["speaker"], text=text, timestamp=last_ts,
                                    session_id=f"session_{s}", meta={"time_str": conv[f"session_{s}_date_time"]}))
        convs[cid] = Conversation(cid, msgs)
        for i, q in enumerate(sample["qa"]):
            cat = LOCOMO_CATEGORIES.get(q["category"], str(q["category"]))
            if categories and cat not in categories:
                continue
            adv = cat == "adversarial"
            questions.append(Question(
                id=f"{cid}/q{i}", conversation_id=cid, question=q["question"],
                answer="Not mentioned in the conversation" if adv else str(q.get("answer", "")),
                category=cat, evidence_ids=tuple(q.get("evidence", []) or []), is_abstention=adv,
                question_time=last_ts, meta={"distractor": q.get("adversarial_answer")} if adv else {},
            ))
    return Dataset("locomo", convs, questions)


# --------------------------------------------------------------- LongMemEval
def _lme_time(s: str) -> float:
    return _ts(datetime.strptime(re.sub(r"\s*\(\w+\)\s*", " ", s).strip(), "%Y/%m/%d %H:%M"))


def load_longmemeval(path: Optional[str] = None, variant: str = "s_cleaned",
                     categories: Optional[Sequence[str]] = None, **_: Any) -> Dataset:
    p = Path(path) if path else data_root() / "longmemeval" / f"longmemeval_{variant}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    convs, questions = {}, []
    for e in raw:
        if categories and e["question_type"] not in categories:
            continue
        cid = e["question_id"]
        msgs, evidence = [], []
        for sid, sess, date in zip(e["haystack_session_ids"], e["haystack_sessions"], e["haystack_dates"]):
            t0 = _lme_time(date)
            for j, turn in enumerate(sess):
                mid = f"{sid}#{j}"
                if turn.get("has_answer"):
                    evidence.append(mid)
                msgs.append(Message(id=mid, speaker=turn["role"], text=turn["content"], timestamp=t0 + j,
                                    session_id=sid, meta={"time_str": date}))
        msgs.sort(key=lambda m: m.timestamp)
        convs[cid] = Conversation(cid, msgs)
        questions.append(Question(
            id=cid, conversation_id=cid, question=e["question"], answer=str(e["answer"]), asker="user",
            category=e["question_type"], evidence_ids=tuple(evidence), is_abstention="_abs" in cid,
            question_time=_lme_time(e["question_date"]),
            meta={"answer_session_ids": e.get("answer_session_ids", [])},
        ))
    return Dataset(f"longmemeval_{variant}", convs, questions)


# ------------------------------------------------------------ native format
def load_jarvis(path: str, categories: Optional[Sequence[str]] = None, **_: Any) -> Dataset:
    """``{"name", "conversations": [{"id", "messages": [Message fields]}], "questions": [Question fields]}``"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    convs = {
        c["id"]: Conversation(c["id"], [Message(**m) for m in c["messages"]]) for c in raw["conversations"]
    }
    qs = []
    for q in raw["questions"]:
        q = dict(q)
        for k in ("evidence_ids", "stale_answers"):
            q[k] = tuple(q.get(k, ()))
        qs.append(Question(**q))
    ds = Dataset(raw.get("name", Path(path).stem), convs, qs, raw.get("annotations", {}))
    return ds.filter(categories=categories) if categories else ds


LOADERS: Dict[str, Callable[..., Dataset]] = {
    "groupmembench": load_groupmembench,
    "locomo": load_locomo,
    "longmemeval": load_longmemeval,
    "jarvis": load_jarvis,
}


def load_dataset(cfg: Dict[str, Any]) -> Dataset:
    cfg = dict(cfg)
    name = cfg.pop("name")
    limit = cfg.pop("limit", None)
    convs = cfg.pop("conversations", None)
    ds = LOADERS[name](**cfg)
    if limit is not None or convs:
        ds = ds.filter(limit=limit, conversations=convs)
    return ds
