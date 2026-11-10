"""Supersession evaluation: are the graph's version chains right?

Gold labels are pairs of messages (``data/labels/supersession/*.jsonl``, one JSON object
per line, see ``docs/LABELING.md``)::

    {"dataset": ..., "conversation_id": ..., "older_id": ..., "newer_id": ...,
     "label": "supersedes" | "same_slot_no_change" | "unrelated", "holder": "group:eng"}

The graph predicts that ``newer`` supersedes ``older`` when both are values in one version
chain. Reported per dataset:

* recall     = predicted gold ``supersedes`` pairs / gold ``supersedes`` pairs
* precision  = predicted pairs labelled ``supersedes`` / predicted pairs that have any label
  (hand labels are never complete, so unlabelled predictions are counted separately)
* false-supersession rate = labelled negatives (``same_slot_no_change``, ``unrelated``) that
  the graph chains anyway / labelled negatives

    jarvis-eval supersession-eval configs/eval/supersession.yaml [--labels DIR_OR_FILE ...]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..data.loaders import load_dataset
from ..memory import build_memory

LABELS = ("supersedes", "same_slot_no_change", "unrelated")
DEFAULT_LABEL_DIR = "data/labels/supersession"


def load_labels(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    """Read label files (``.jsonl``) or every ``*.jsonl`` in a directory; validates each row."""
    files: List[Path] = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    rows: List[Dict[str, Any]] = []
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"supersession label file {f} not found")
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in ("dataset", "conversation_id", "older_id", "newer_id", "label") if k not in r]
            if missing:
                raise ValueError(f"{f}:{n}: missing {missing}")
            if r["label"] not in LABELS:
                raise ValueError(f"{f}:{n}: label must be one of {LABELS}, got {r['label']!r}")
            rows.append(r)
    return rows


def score_pairs(predicted: Set[Tuple[str, str]], labelled: Dict[Tuple[str, str], str]) -> Dict[str, int]:
    """Counts for one conversation. ``labelled`` maps (older, newer) -> label."""
    pos = {k for k, v in labelled.items() if v == "supersedes"}
    neg = set(labelled) - pos
    return {"gold": len(pos), "negatives": len(neg), "predicted": len(predicted),
            "tp": len(predicted & pos), "fp": len(predicted & neg),
            "predicted_unlabelled": len(predicted - set(labelled))}


def _rates(c: Dict[str, int]) -> Dict[str, Any]:
    judged = c["tp"] + c["fp"]
    prec = c["tp"] / judged if judged else None
    rec = c["tp"] / c["gold"] if c["gold"] else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    return {**c, "precision": prec, "recall": rec, "f1": f1,
            "false_supersession_rate": c["fp"] / c["negatives"] if c["negatives"] else None}


def supersession_eval(cfg: Dict[str, Any], label_paths: Optional[Sequence[str]] = None,
                      progress: bool = True) -> Dict[str, Any]:
    paths = list(label_paths or cfg.get("labels") or [DEFAULT_LABEL_DIR])
    if isinstance(paths, str):
        paths = [paths]
    rows = load_labels(paths)
    by: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = defaultdict(dict)
    for r in rows:
        by[(r["dataset"], r["conversation_id"])][(r["older_id"], r["newer_id"])] = r["label"]
    mem_cfg = dict(cfg["memory"])
    mem_cfg["scorer"] = {"type": "seeds"}  # the graph is all that matters here
    out: Dict[str, Any] = {"labels": [str(p) for p in paths], "memory": cfg["memory"], "datasets": []}
    for dcfg in cfg["datasets"]:
        dcfg = dict(dcfg)
        label = dcfg.get("label") or dcfg["name"]
        ds = load_dataset(dcfg)
        tot = {"gold": 0, "negatives": 0, "predicted": 0, "tp": 0, "fp": 0, "predicted_unlabelled": 0}
        n_conv, unknown = 0, 0
        for cid, conv in ds.conversations.items():
            lab = by.get((ds.name, cid))
            if not lab:
                continue
            n_conv += 1
            ids = {m.id for m in conv.messages}
            unknown += sum(1 for a, b in lab if a not in ids or b not in ids)
            mem = build_memory(mem_cfg, cfg.get("tokenizer"))
            mem.reset()
            mem.ingest(conv.messages)
            c = score_pairs(mem.graph.superseding_pairs(), lab)
            for k in tot:
                tot[k] += c[k]
        res = {"label": label, "dataset": ds.name, "domain": ds.domain, "conversations": n_conv,
               "unknown_message_ids": unknown, **_rates(tot)}
        out["datasets"].append(res)
        if progress:
            print(f"[supersession-eval] {label}: {n_conv} labelled conversations, " + _fmt_row(res))
    return out


def _f(v: Optional[float]) -> str:
    return "–" if v is None else f"{v:.3f}"


def _fmt_row(r: Dict[str, Any]) -> str:
    return (f"P={_f(r['precision'])} R={_f(r['recall'])} F1={_f(r['f1'])} "
            f"false-supersession={_f(r['false_supersession_rate'])}")


def to_markdown(res: Dict[str, Any]) -> str:
    head = ["Dataset", "domain", "conversations", "gold pairs", "negatives", "predicted", "unlabelled predictions",
            "precision", "recall", "F1", "false-supersession rate"]
    lines = ["# Supersession evaluation", "",
             "Pairs the graph chains as (older -> newer value) against hand or generator labels.", "",
             "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in res["datasets"]:
        lines.append("| " + " | ".join([
            r["label"], str(r["domain"] or "–"), str(r["conversations"]), str(r["gold"]), str(r["negatives"]),
            str(r["predicted"]), str(r["predicted_unlabelled"]), _f(r["precision"]), _f(r["recall"]), _f(r["f1"]),
            _f(r["false_supersession_rate"])]) + " |")
    return "\n".join(lines) + "\n"


def write_report(res: Dict[str, Any], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return out
