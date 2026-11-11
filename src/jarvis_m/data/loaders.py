"""Benchmark loaders -> :class:`jarvis_m.types.Dataset`.

Supported: GroupMemBench (multi-party, primary), LoCoMo, LongMemEval, EverMemBench-Dynamic,
ConvoMem, SocialMemBench, and the native ``jarvis`` JSON format (our stress sets / WhatsApp
eval set / fixtures). Licences: ``docs/DATASETS.md``.
Run ``jarvis-eval data fetch <name>`` first for public benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from ..types import Conversation, Dataset, Message, Question

# Every dataset entry in eval / train configs names one of these (``domain:``).
DOMAINS = ("team", "assistant", "personal", "enterprise")
SplitSpec = Union[None, str, Dict[str, Any]]


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


def load_locomo(path: Optional[str] = None, categories: Optional[Sequence[str]] = None,
                split: SplitSpec = None, **_: Any) -> Dataset:
    """``split: {n_folds, fold, seed, part}`` holds out whole conversations (LoCoMo has no
    fixed split, so ``split: train|test`` is rejected)."""
    p = Path(path) if path else data_root() / "locomo" / "locomo10.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    kind, part, kf = parse_split(split)
    if kind == "holdout":
        raise ValueError("LoCoMo has no fixed train/test split: use split: {n_folds: 5, fold: i, seed: 0}")
    held = kfold_test_ids({"all": [s["sample_id"] for s in raw]}, **kf) if kind == "kfold" else set()
    convs, questions = {}, []
    for sample in raw:
        cid = sample["sample_id"]
        if kind == "kfold" and (cid in held) != (part == "test"):
            continue
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


def _split_hash(seed: int, qid: str) -> str:
    return hashlib.sha1(f"{seed}:{qid}".encode()).hexdigest()


def parse_split(split: SplitSpec) -> Tuple[Optional[str], str, Dict[str, int]]:
    """``None`` | ``"train"`` | ``"test"`` | ``{n_folds, fold, seed=0, part="test"}`` ->
    (kind, part, k-fold arguments). ``part: test`` is fold ``fold``; ``part: train`` is the rest."""
    if split is None:
        return None, "", {}
    if isinstance(split, str):
        if split not in ("train", "test"):
            raise ValueError(f"split must be None, 'train', 'test' or a k-fold mapping, got {split!r}")
        return "holdout", split, {}
    if not isinstance(split, dict) or "n_folds" not in split or "fold" not in split:
        raise ValueError(f"k-fold split needs n_folds and fold, got {split!r}")
    unknown = set(split) - {"n_folds", "fold", "seed", "part"}
    if unknown:
        raise ValueError(f"unknown split keys {sorted(unknown)}")
    n_folds, fold, part = int(split["n_folds"]), int(split["fold"]), str(split.get("part", "test"))
    if n_folds < 2 or not 0 <= fold < n_folds:
        raise ValueError(f"need n_folds >= 2 and 0 <= fold < n_folds, got {split!r}")
    if part not in ("train", "test"):
        raise ValueError(f"split.part must be 'train' or 'test', got {part!r}")
    return "kfold", part, {"n_folds": n_folds, "fold": fold, "seed": int(split.get("seed", 0))}


def kfold_test_ids(ids_by_stratum: Dict[str, Sequence[str]], n_folds: int, fold: int, seed: int = 0) -> Set[str]:
    """Ids of fold ``fold``. Inside each stratum, ids are ordered by a stable hash of
    (seed, id) and dealt round-robin into folds, so every fold gets every stratum and the
    assignment does not depend on file order or on any filter applied later."""
    test: Set[str] = set()
    for ids in ids_by_stratum.values():
        ordered = sorted(ids, key=lambda q: _split_hash(seed, q))
        test.update(q for i, q in enumerate(ordered) if i % n_folds == fold)
    return test


def longmemeval_test_ids(entries: Sequence[Dict[str, Any]], test_fraction: float = 0.2, split_seed: int = 0) -> set:
    """Stratified, order-independent test split: within each question_type, ids sorted by a
    stable hash of (seed, id); the first ceil(test_fraction * n_type) are test. The file is
    sorted by type, so ``limit:`` would silently drop whole types."""
    by_type: Dict[str, List[str]] = {}
    for e in entries:
        by_type.setdefault(e["question_type"], []).append(e["question_id"])
    test: set = set()
    for ids in by_type.values():
        ids = sorted(ids, key=lambda q: _split_hash(split_seed, q))
        test.update(ids[: math.ceil(test_fraction * len(ids))])
    return test


def load_longmemeval(path: Optional[str] = None, variant: str = "s_cleaned",
                     categories: Optional[Sequence[str]] = None, split: SplitSpec = None,
                     test_fraction: float = 0.2, split_seed: int = 0, **_: Any) -> Dataset:
    """``split: train|test`` is the fixed stratified hold-out; ``split: {n_folds, fold, seed,
    part}`` is k-fold, stratified by question type (each question is its own conversation)."""
    kind, part, kf = parse_split(split)
    p = Path(path) if path else data_root() / "longmemeval" / f"longmemeval_{variant}.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    # split over the whole file (before category filtering) so it is the same for every filter
    if kind == "kfold":
        by_type: Dict[str, List[str]] = {}
        for e in raw:
            by_type.setdefault(e["question_type"], []).append(e["question_id"])
        test_ids = kfold_test_ids(by_type, **kf)
    else:
        test_ids = longmemeval_test_ids(raw, test_fraction, split_seed) if kind else set()
    convs, questions = {}, []
    for e in raw:
        if categories and e["question_type"] not in categories:
            continue
        if kind and (e["question_id"] in test_ids) != (part == "test"):
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
        for k in ("evidence_ids", "stale_answers", "stale_evidence_ids"):
            q[k] = tuple(q.get(k, ()))
        qs.append(Question(**q))
    ds = Dataset(raw.get("name", Path(path).stem), convs, qs, raw.get("annotations", {}))
    return ds.filter(categories=categories) if categories else ds


# ------------------------------------------------------------ EverMemBench
def _index_ranges(spec: Any) -> List[int]:
    """``"1, 4-6, 8"`` -> [1, 4, 5, 6, 8] (EverMemBench ``message_index`` references)."""
    out: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_evermembench(root: Optional[str] = None, topics: Optional[Sequence[str]] = None,
                      categories: Optional[Sequence[str]] = None, **_: Any) -> Dataset:
    """EverMemBench-Dynamic (huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic):
    ``<root>/<topic>/dialogue.json`` (a list of days ``{topic_id, date, dialogues: {group:
    [{speaker, time, dialogue, message_index}] | null}}``) and ``<root>/<topic>/qa_*.json``
    (``{id, Q, A, R: [{date, group, message_index: "1, 4-6"}], options}``). One conversation
    per topic; groups are channels; ``R`` gives the evidence ids (a context slice around the
    answer, as the release documents). Category = the id prefix (``F_SH``, ``MA_C``, ...)."""
    base = Path(root) if root else data_root() / "EverMemBench-Dynamic"
    convs: Dict[str, Conversation] = {}
    questions: List[Question] = []
    unresolved = 0
    for tdir in sorted(p for p in base.iterdir() if p.is_dir() and (p / "dialogue.json").exists()):
        if topics and tdir.name not in topics:
            continue
        cid = f"emb-{tdir.name}"
        msgs: List[Message] = []
        for day in json.loads((tdir / "dialogue.json").read_text(encoding="utf-8")):
            for group, items in (day.get("dialogues") or {}).items():
                for m in items or []:
                    when = datetime.strptime(m["time"], "%Y-%m-%d %H:%M:%S")
                    msgs.append(Message(id=f"{day['date']}/{group}#{m['message_index']}", speaker=m["speaker"],
                                        text=m["dialogue"], timestamp=_ts(when), channel=group,
                                        session_id=f"{group}/{day['date']}", meta={"time_str": m["time"]}))
        msgs.sort(key=lambda x: (x.timestamp, x.id))
        convs[cid] = Conversation(cid, msgs)
        ids = {m.id for m in msgs}
        end = msgs[-1].timestamp if msgs else None
        for qf in sorted(tdir.glob("qa_*.json")):
            for q in json.loads(qf.read_text(encoding="utf-8")):
                cat = "_".join(q["id"].split("_")[:2])
                if categories and cat not in categories:
                    continue
                ev = [f"{r['date']}/{r['group']}#{i}" for r in q.get("R") or [] for i in _index_ranges(r["message_index"])]
                unresolved += sum(1 for e in ev if e not in ids)
                opts = q.get("options") or None
                answer = str(q["A"])
                if opts and answer in opts:
                    answer = str(opts[answer])  # multiple choice: the option text, the letter goes in meta
                questions.append(Question(
                    id=f"{cid}/{q['id']}", conversation_id=cid, question=q["Q"], answer=answer, category=cat,
                    evidence_ids=tuple(dict.fromkeys(e for e in ev if e in ids)), question_time=end,
                    meta={"options": opts, "correct_option": q["A"]} if opts else {}))
    return Dataset("evermembench", convs, questions, {"unresolved_evidence": unresolved})


# ------------------------------------------------------------ ConvoMem
CONVOMEM_TYPES = ("user_evidence", "assistant_facts_evidence", "changing_evidence", "abstention_evidence",
                  "preference_evidence", "implicit_connection_evidence")


def _norm_text(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _match_evidence(evidence: str, texts: Sequence[str]) -> List[int]:
    """Positions of the messages an evidence quote refers to: exact text, else a message that
    contains the quote (or is contained in it) after normalising case and punctuation, else
    the closest message with similarity >= 0.9. ConvoMem quotes evidence by text, and about a
    third of the quotes are excerpts or near-copies of the message."""
    import difflib

    t = (evidence or "").strip()
    exact = [i for i, x in enumerate(texts) if x.strip() == t]
    if exact:
        return exact
    n = _norm_text(t)
    norm = [_norm_text(x) for x in texts]
    inside = [i for i, x in enumerate(norm) if len(x) > 20 and len(n) > 20 and (n in x or x in n)]
    if inside:
        return inside
    best, best_i = 0.0, -1
    for i, x in enumerate(norm):
        sm = difflib.SequenceMatcher(None, n, x)
        if sm.quick_ratio() >= 0.9 and sm.ratio() > best:
            best, best_i = sm.ratio(), i
    return [best_i] if best >= 0.9 else []


def load_convomem(root: Optional[str] = None, evidence_types: Sequence[str] = ("changing_evidence",),
                  evidence_counts: Optional[Sequence[int]] = None, persons: Optional[int] = None,
                  **_: Any) -> Dataset:
    """ConvoMem (huggingface.co/datasets/Salesforce/ConvoMem), ``core_benchmark/evidence_questions/
    <type>/<n>_evidence/<person>.json`` = ``{"evidence_items": [{question, answer,
    message_evidences: [{speaker, text}], conversations: [{id, messages: [{speaker, text}]}],
    personId, category, ...}]}``. Each evidence item is one conversation (its own haystack,
    a user and an assistant). Evidence is quoted by text and matched to messages (see
    :func:`_match_evidence`); quotes that match nothing are counted in
    ``annotations["unresolved_evidence"]``. ``persons`` limits the person files per folder."""
    base = Path(root) if root else data_root() / "ConvoMem"
    qroot = base / "core_benchmark" / "evidence_questions"
    convs: Dict[str, Conversation] = {}
    questions: List[Question] = []
    unresolved = total = 0
    for etype in evidence_types:
        if etype not in CONVOMEM_TYPES:
            raise ValueError(f"unknown ConvoMem evidence type {etype!r}; known: {CONVOMEM_TYPES}")
        for ndir in sorted((qroot / etype).glob("*_evidence")):
            n_ev = int(ndir.name.split("_")[0])
            if evidence_counts and n_ev not in evidence_counts:
                continue
            for pf in sorted(ndir.glob("*.json"))[: persons or None]:
                for k, it in enumerate(json.loads(pf.read_text(encoding="utf-8"))["evidence_items"]):
                    cid = f"{etype}/{n_ev}/{pf.stem.split('_')[0][:8]}/{k}"
                    msgs: List[Message] = []
                    for ci, c in enumerate(it["conversations"]):
                        for mi, m in enumerate(c["messages"]):
                            role = "assistant" if m["speaker"].lower() == "assistant" else None
                            msgs.append(Message(id=f"{cid}#{ci}:{mi}", speaker=m["speaker"], text=m["text"],
                                                timestamp=float(len(msgs)), session_id=str(c.get("id") or ci),
                                                role=role))
                    texts = [m.text for m in msgs]
                    ev: List[str] = []
                    for e in it.get("message_evidences") or []:
                        total += 1
                        hit = _match_evidence(e["text"], texts)
                        unresolved += not hit
                        ev.extend(msgs[i].id for i in hit)
                    convs[cid] = Conversation(cid, msgs)
                    questions.append(Question(
                        id=cid, conversation_id=cid, question=it["question"], answer=str(it["answer"]), asker="User",
                        category=etype, evidence_ids=tuple(dict.fromkeys(ev)),
                        is_abstention=etype == "abstention_evidence", question_time=float(len(msgs)),
                        meta={"evidence_count": n_ev, "convomem_category": it.get("category")}))
    return Dataset("convomem", convs, questions, {"unresolved_evidence": unresolved, "evidence_quotes": total})


# ------------------------------------------------------------ SocialMemBench
SOCIALMEM_TYPES = {"Q1": "single_contact_recall", "Q2": "group_decision_recall", "Q3": "multi_contact_aggregation",
                   "Q4": "contamination_probe", "Q5": "theory_of_mind_reference", "Q6": "norm_vs_individual",
                   "Q7": "relational_edge_query", "Q8": "temporal_preference_evolution",
                   "Q9": "departed_member_recall"}


def _rows(base: Path, name: str) -> List[Dict[str, Any]]:
    """A config's rows from ``<name>.parquet`` (needs pyarrow) or ``<name>.jsonl``."""
    pq_path, jl = base / f"{name}.parquet", base / f"{name}.jsonl"
    if pq_path.exists():
        try:
            import pyarrow.parquet as pq
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("reading SocialMemBench parquet files needs pyarrow: pip install 'jarvis-m[data]'") from e
        return pq.read_table(pq_path).to_pylist()
    return [json.loads(line) for line in jl.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_field(v: Any, default: Any) -> Any:
    if v is None or v == "":
        return default
    return json.loads(v) if isinstance(v, str) else v


def _iso_lenient(ts: str) -> float:
    """ISO-8601 to epoch seconds, tolerating out-of-range minutes or seconds ("12:60:00" is
    13:00:00; the SocialMemBench release has a few)."""
    try:
        return _ts(datetime.fromisoformat(ts))
    except ValueError:
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})[T ](\d{1,2}):(\d{1,2})(?::(\d{1,2}))?.*", ts.strip())
        if not m:
            raise
        day = datetime.fromisoformat(m.group(1))
        return _ts(day) + int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4) or 0)


def load_socialmembench(root: Optional[str] = None, query_types: Optional[Sequence[str]] = None,
                        **_: Any) -> Dataset:
    """SocialMemBench (huggingface.co/datasets/anon4data/socialmembench): ``conversations``
    (one row per turn: network_id, session_id, turn_id, speaker_display_name, timestamp,
    message, reply_to_turn_id), ``qa`` (qa_id, network_id, query_type, question, answer,
    answer_format, options_json, correct_option, evidence_anchors_json) and ``networks``
    (network_id, group_name). The release stores lists and dicts as JSON strings (``*_json``
    columns), unlike its README. One conversation per network (a group chat); evidence =
    the anchors' turn ids. Multiple-choice answers become the option text."""
    base = Path(root) if root else data_root() / "SocialMemBench"
    groups = {r["network_id"]: r.get("group_name") or r["network_id"] for r in _rows(base, "networks")}
    by_net: Dict[str, List[Message]] = {}
    for r in _rows(base, "conversations"):
        ts = r.get("timestamp")
        by_net.setdefault(r["network_id"], []).append(Message(
            id=r["turn_id"], speaker=r["speaker_display_name"], text=r["message"],
            timestamp=_iso_lenient(ts) if ts else None,
            channel=groups.get(r["network_id"], r["network_id"]), session_id=r.get("session_id"),
            reply_to=r.get("reply_to_turn_id") or None,
            meta={"time_str": r.get("timestamp_label") or ts or ""}))
    convs = {}
    for net, msgs in by_net.items():
        msgs.sort(key=lambda m: (m.timestamp or 0.0, m.id))
        convs[net] = Conversation(net, msgs)
    questions: List[Question] = []
    unresolved = 0
    for q in _rows(base, "qa"):
        if query_types and q["query_type"] not in query_types:
            continue
        net = q["network_id"]
        ids = {m.id for m in convs.get(net, Conversation(net, [])).messages}
        anchors = _json_field(q.get("evidence_anchors_json", q.get("evidence_anchors")), [])
        ev = [a["turn_id"] for a in anchors]
        unresolved += sum(1 for e in ev if e not in ids)
        opts = _json_field(q.get("options_json", q.get("options")), {}) or {}
        answer = str(q["answer"])
        if q.get("answer_format") == "multiple_choice" and q.get("correct_option") in opts:
            answer = str(opts[q["correct_option"]])
        msgs = convs[net].messages if net in convs else []
        questions.append(Question(
            id=q["qa_id"] if q["qa_id"].startswith(net) else f"{net}/{q['qa_id']}", conversation_id=net,
            question=q["question"], answer=answer,
            category=SOCIALMEM_TYPES.get(q["query_type"], q["query_type"]),
            evidence_ids=tuple(dict.fromkeys(e for e in ev if e in ids)),
            question_time=msgs[-1].timestamp if msgs else None,
            meta={"answer_format": q.get("answer_format"), "difficulty": q.get("difficulty"),
                  "options": opts or None, "contamination_foil": q.get("contamination_foil") or None}))
    return Dataset("socialmembench", convs, questions, {"unresolved_evidence": unresolved})


LOADERS: Dict[str, Callable[..., Dataset]] = {
    "groupmembench": load_groupmembench,
    "locomo": load_locomo,
    "longmemeval": load_longmemeval,
    "jarvis": load_jarvis,
    "evermembench": load_evermembench,
    "convomem": load_convomem,
    "socialmembench": load_socialmembench,
}


def dataset_domain(cfg: Dict[str, Any], required: bool = False) -> Optional[str]:
    """The ``domain:`` of a dataset config entry, validated against :data:`DOMAINS`."""
    dom = cfg.get("domain")
    if dom is None:
        if required:
            raise ValueError(f"dataset entry {cfg.get('label') or cfg.get('name')!r} needs `domain:` (one of {DOMAINS})")
        return None
    if dom not in DOMAINS:
        raise ValueError(f"unknown domain {dom!r} for dataset {cfg.get('label') or cfg.get('name')!r}; known: {DOMAINS}")
    return dom


# config keys that describe how a dataset is used, not how it is loaded
_ENTRY_KEYS = ("label", "weight")


def load_dataset(cfg: Dict[str, Any]) -> Dataset:
    cfg = dict(cfg)
    name = cfg.pop("name")
    limit = cfg.pop("limit", None)
    convs = cfg.pop("conversations", None)
    domain = dataset_domain(cfg)
    cfg.pop("domain", None)
    for k in _ENTRY_KEYS:
        cfg.pop(k, None)
    ds = LOADERS[name](**cfg)
    if limit is not None or convs:
        ds = ds.filter(limit=limit, conversations=convs)
    ds.domain = domain
    return ds
