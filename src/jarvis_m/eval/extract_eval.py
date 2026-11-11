"""Extraction evaluation: are the extracted records right, field by field?

Gold slots are hand-labelled records (``data/labels/slots/*.jsonl``, see
``docs/LABELING.md``), one JSON object per line::

    {"dataset": ..., "conversation_id": ..., "msg_id": ..., "holder": "speaker:Caroline",
     "subject": "home", "attribute": "location", "modality": "state", "value": "Boston"}
    {"dataset": ..., "conversation_id": ..., "msg_id": ..., "no_record": true}

The second form marks a message that should yield no stateful record (chit-chat, a question).

Each gold record is matched to at most one predicted record of the same message (best by
attribute, holder, then value-word overlap). Reported per dataset:

* record recall      = matched gold records / gold records
* holder / attribute / modality accuracy over matched pairs (holders compared as slot-key
  holders: ``speaker:X`` and ``person:X`` are the same)
* subject purity     = predicted subject clusters (per conversation and holder) that hold one
  gold subject; 1.0 means a predicted subject never mixes two gold slots
* subject completeness = the inverse: gold subjects that got one predicted subject; this is
  what lets a version chain link on the exact key
* spurious rate      = ``no_record`` messages that got a stateful record anyway (one that can
  change slot state: a stateful modality and an attribute other than ``info``)

Work-state records the heuristic extractor marks ``group_in_group_chat`` get the group
holder when the conversation's channel has at least GROUP_MIN_PARTICIPANTS speakers, as in
the graph.

    jarvis-eval extract-eval configs/eval/extract.yaml [--labels DIR_OR_FILE ...]
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..data.loaders import load_dataset
from ..extract import build_extractor
from ..extract.records import STATEFUL, StateRecord, holder_key
from ..text import word_tokens
from ..types import GROUP_MIN_PARTICIPANTS, Conversation

DEFAULT_LABEL_DIR = "data/labels/slots"
FIELDS = ("holder", "attribute", "modality")


def load_slot_labels(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    rows = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"slot label file {f} not found")
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            need = ("dataset", "conversation_id", "msg_id") + (() if r.get("no_record") else
                                                                ("holder", "subject", "attribute", "modality"))
            missing = [k for k in need if k not in r]
            if missing:
                raise ValueError(f"{f}:{n}: missing {missing}")
            rows.append(r)
    return rows


def resolve_holders(recs: Sequence[StateRecord], conv: Conversation) -> List[StateRecord]:
    """Apply the graph's group-chat rule to records marked ``group_in_group_chat``."""
    speakers: Dict[str, set] = defaultdict(set)
    for m in conv.messages:
        if (m.speaker or "").lower() != "assistant" and (m.role or "").lower() != "assistant":
            speakers[m.channel].add(m.speaker)
    chan = {m.id: m.channel for m in conv.messages}
    out = []
    for r in recs:
        c = chan.get(r.msg_id)
        if (r.meta or {}).get("group_in_group_chat") and c is not None and len(speakers[c]) >= GROUP_MIN_PARTICIPANTS:
            r = r.with_holder(f"group:{c}")
        out.append(r)
    return out


def _overlap(a: str, b: str) -> float:
    ta, tb = set(word_tokens(a or "")), set(word_tokens(b or ""))
    return len(ta & tb) / max(1, len(ta | tb))


def match(gold: Sequence[Dict[str, Any]], pred: Sequence[StateRecord]) -> List[Tuple[Dict[str, Any], StateRecord]]:
    """Greedy one-to-one matching inside one message."""
    pairs = []
    for g in gold:
        for j, p in enumerate(pred):
            score = (p.attribute == g["attribute"], holder_key(p.holder) == holder_key(g["holder"]),
                     _overlap(p.value, g.get("value", "")))
            pairs.append((score, id(g), j, g, p))
    pairs.sort(key=lambda t: t[0], reverse=True)
    used_g, used_p, out = set(), set(), []
    for _, gid, j, g, p in pairs:
        if gid in used_g or j in used_p:
            continue
        used_g.add(gid)
        used_p.add(j)
        out.append((g, p))
    return out


def _purity(clusters: Dict[Any, Counter]) -> Optional[float]:
    total = sum(sum(c.values()) for c in clusters.values())
    return sum(max(c.values()) for c in clusters.values()) / total if total else None


def extract_eval(cfg: Dict[str, Any], label_paths: Optional[Sequence[str]] = None,
                 progress: bool = True) -> Dict[str, Any]:
    paths = list(label_paths or cfg.get("labels") or [DEFAULT_LABEL_DIR])
    rows = load_slot_labels(paths)
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["dataset"], r["conversation_id"])].append(r)
    extractor = build_extractor(cfg.get("extractor") or {"type": "heuristic"})
    out: Dict[str, Any] = {"labels": [str(p) for p in paths], "extractor": cfg.get("extractor"), "datasets": []}
    for dcfg in cfg["datasets"]:
        dcfg = dict(dcfg)
        label = dcfg.get("label") or dcfg["name"]
        ds = load_dataset(dcfg)
        n_gold = n_matched = n_none = n_spurious = n_conv = unknown = 0
        correct = Counter()
        pred_clusters: Dict[Any, Counter] = defaultdict(Counter)  # predicted subject -> gold subjects
        gold_clusters: Dict[Any, Counter] = defaultdict(Counter)  # gold subject -> predicted subjects
        for cid, conv in ds.conversations.items():
            lab = by.get((ds.name, cid))
            if not lab:
                continue
            n_conv += 1
            ids = {m.id for m in conv.messages}
            unknown += sum(1 for r in lab if r["msg_id"] not in ids)
            recs = resolve_holders(extractor.extract(conv.messages), conv)
            pred: Dict[str, List[StateRecord]] = defaultdict(list)
            for r in recs:
                pred[r.msg_id].append(r)
            gold: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in lab:
                if r.get("no_record"):
                    n_none += 1
                    n_spurious += any(p.modality in STATEFUL and p.attribute != "info"
                                      for p in pred.get(r["msg_id"], []))
                else:
                    gold[r["msg_id"]].append(r)
                    n_gold += 1
            for mid, gs in gold.items():
                for g, p in match(gs, pred.get(mid, [])):
                    n_matched += 1
                    correct["holder"] += holder_key(p.holder) == holder_key(g["holder"])
                    correct["attribute"] += p.attribute == g["attribute"]
                    correct["modality"] += p.modality == g["modality"]
                    gk = (cid, holder_key(g["holder"]), g["subject"].strip().lower())
                    pk = (cid, holder_key(p.holder), p.subject.strip().lower())
                    pred_clusters[pk][gk] += 1
                    gold_clusters[gk][pk] += 1
        res = {"label": label, "dataset": ds.name, "domain": ds.domain, "conversations": n_conv,
               "gold_records": n_gold, "matched": n_matched, "unknown_message_ids": unknown,
               "record_recall": n_matched / n_gold if n_gold else None,
               **{f"{f}_accuracy": correct[f] / n_matched if n_matched else None for f in FIELDS},
               "subject_purity": _purity(pred_clusters), "subject_completeness": _purity(gold_clusters),
               "no_record_messages": n_none, "spurious_rate": n_spurious / n_none if n_none else None}
        out["datasets"].append(res)
        if progress:
            print(f"[extract-eval] {label}: {n_conv} labelled conversations, recall={_f(res['record_recall'])} "
                  f"holder={_f(res['holder_accuracy'])} attribute={_f(res['attribute_accuracy'])}")
    return out


def _f(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:.3f}"


COLUMNS = ("record_recall", "holder_accuracy", "attribute_accuracy", "modality_accuracy", "subject_purity",
           "subject_completeness", "spurious_rate")


def to_markdown(res: Dict[str, Any]) -> str:
    head = ["Dataset", "domain", "conversations", "gold records", *[c.replace("_", " ") for c in COLUMNS]]
    lines = ["# Extraction evaluation", "", f"Extractor: `{res['extractor']}`", "",
             "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in res["datasets"]:
        lines.append("| " + " | ".join([r["label"], str(r["domain"] or "–"), str(r["conversations"]),
                                        str(r["gold_records"]), *[_f(r[c]) for c in COLUMNS]]) + " |")
    return "\n".join(lines) + "\n"


def write_report(res: Dict[str, Any], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return out
