"""Second pass: grade saved predictions with a judge loaded once for many runs.

Typical 8 GB-GPU flow::

    jarvis-eval run   configs/exp/gmb_baselines.yaml          # reader only (judge: deferred)
    jarvis-eval judge runs/gmb_baselines --judge configs/models/judge_qwen3_8b_4bit_hf.yaml
    jarvis-eval report runs/gmb_baselines --baseline bm25

Resumable: verdicts are appended to ``results.judged.jsonl`` as they are made and
swapped in for ``results.jsonl`` when a run is complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from ..types import Question
from .metrics import mentions_any
from .qa import DEFERRED
from .report import summarize


def _row_question(r: Dict[str, Any]) -> Question:
    return Question(
        id=r["qid"], conversation_id=r["conversation"], question=r["question"], answer=r["gold"],
        asker=r.get("asker"), category=r["category"], is_abstention=bool(r.get("is_abstention_q")),
    )


def needs_judging(r: Dict[str, Any], force: bool) -> bool:
    return force or r.get("judge_raw") == DEFERRED or r.get("correct") is None


def rejudge_run(run_dir: Path, judge, judge_cfg: Dict[str, Any], force: bool = False, progress: bool = True) -> Dict:
    res = run_dir / "results.jsonl"
    part = run_dir / "results.judged.jsonl"
    rows = [json.loads(x) for x in res.read_text(encoding="utf-8").splitlines() if x.strip()]
    done: Dict[str, Dict[str, Any]] = {}
    if part.exists():
        for x in part.read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                done[r["qid"]] = r
    todo = [r for r in rows if r["qid"] not in done and needs_judging(r, force)]
    if progress:
        print(f"[judge] {run_dir}: {len(todo)} to judge ({len(rows) - len(todo)} kept)")
    with part.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo):
            v = judge.judge(_row_question(r), r["pred"])
            r = {**r, "correct": v.correct, "judge_raw": v.raw, "judge": getattr(judge, "name", "?")}
            if r.get("stale_answers"):
                r["stale_hit"] = mentions_any(r["pred"], r["stale_answers"]) and not v.correct
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            done[r["qid"]] = r
            if progress and (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)}")
    merged: List[Dict[str, Any]] = [done.get(r["qid"], r) for r in rows]
    tmp = run_dir / "results.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in merged), encoding="utf-8")
    os.replace(tmp, res)
    part.unlink()
    summary_path = run_dir / "summary.json"
    meta = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary = summarize(merged)
    summary.update({k: meta[k] for k in ("name", "arm", "dataset") if k in meta})
    summary["judge"] = judge_cfg
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if progress:
        a = summary["accuracy"]
        print(f"[judge] {run_dir.name}: acc={a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]")
    return summary
