"""Run one memory system over one dataset: ingest -> retrieve -> read -> judge.

Writes ``<output_dir>/<name>[/<arm>]/`` with ``config.yaml``, ``results.jsonl``
(one row per question, resumable) and ``summary.json``.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from ..data.loaders import load_dataset
from ..memory import build_memory
from .metrics import evidence_recall, is_abstention, mentions_any
from .qa import DEFERRED, build_judge, build_reader
from .report import summarize


def _read_done(path: Path) -> Dict[str, Dict[str, Any]]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["qid"]] = r
    return done


def run(cfg: Dict[str, Any], arm: str = "", progress: bool = True) -> Path:
    name = cfg.get("name", "run")
    out = Path(cfg.get("output_dir", "runs")) / name
    if arm:
        out = out / arm
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({"arm": arm, **cfg}, sort_keys=False), encoding="utf-8")

    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)

    ds = load_dataset(cfg["dataset"])
    reader = build_reader(cfg.get("reader"))
    judge = build_judge(cfg.get("judge"))
    rcfg = cfg.get("retrieval", {}) or {}
    budget, max_items = rcfg.get("budget_tokens"), rcfg.get("max_items")

    res_path = out / "results.jsonl"
    done = _read_done(res_path)
    rows: List[Dict[str, Any]] = list(done.values())
    todo_total = sum(1 for q in ds.questions if q.id not in done)
    if progress:
        print(f"[run] {name}{'/' + arm if arm else ''}: {ds.summary()} | {todo_total} questions to go")

    with res_path.open("a", encoding="utf-8") as f:
        for cid, conv in ds.conversations.items():
            qs = [q for q in ds.questions_for(cid) if q.id not in done]
            if not qs:
                continue
            mem = build_memory(cfg["memory"], cfg.get("tokenizer"))
            mem.reset()
            t0 = time.perf_counter()
            mem.ingest(conv.messages)
            if hasattr(mem, "prefetch_queries"):
                mem.prefetch_queries([(q.question, q.asker) for q in qs])
            ingest_s = time.perf_counter() - t0
            for i, q in enumerate(qs):
                t1 = time.perf_counter()
                r = mem.retrieve(q.question, asker=q.asker, budget_tokens=budget, max_items=max_items,
                                 now=q.question_time)
                t2 = time.perf_counter()
                pred, raw = reader.answer(q, r.context)
                t3 = time.perf_counter()
                v = judge.judge(q, pred)
                row = {
                    "qid": q.id, "conversation": cid, "category": q.category, "asker": q.asker,
                    "question": q.question, "gold": q.answer, "pred": pred, "reader_raw": raw,
                    "correct": v.correct, "judge_raw": v.raw, "memory": mem.name, "arm": arm, "seed": seed,
                    "n_ctx_tokens": r.n_tokens, "n_items": len(r.items), "retrieved_ids": r.source_ids[:200],
                    "evidence_recall": evidence_recall(r.source_ids, q.evidence_ids),
                    "is_abstention_q": q.is_abstention, "abstained": is_abstention(pred),
                    "stale_answers": list(q.stale_answers),
                    "stale_hit": (mentions_any(pred, q.stale_answers) and not v.correct) if q.stale_answers else None,
                    "retrieve_ms": round((t2 - t1) * 1000, 2), "read_ms": round((t3 - t2) * 1000, 2),
                    "ingest_s": round(ingest_s, 3), "retrieval_info": r.info, "memory_stats": mem.stats(),
                }
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                rows.append(row)
                if progress and (i + 1) % 25 == 0:
                    judged = [bool(x["correct"]) for x in rows if x["judge_raw"] != DEFERRED]
                    acc = f"running acc={np.mean(judged):.3f}" if judged else "judge deferred"
                    print(f"  {cid}: {i + 1}/{len(qs)}  {acc}")

    summary = summarize(rows)
    summary.update({"name": name, "arm": arm, "dataset": ds.name})
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if progress:
        a = summary["accuracy"]
        if summary.get("judge_pending"):
            print(f"[run] done -> {out}  ({summary['judge_pending']} answers awaiting `jarvis-eval judge`)")
        else:
            print(f"[run] done -> {out}  acc={a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
    return out
