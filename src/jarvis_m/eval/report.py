"""Aggregate run rows into summaries and paired comparison tables."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .metrics import bootstrap_ci, mcnemar, paired_bootstrap


def _ci(vals: Sequence[float]) -> Dict[str, float]:
    m, lo, hi = bootstrap_ci(vals)
    return {"mean": m, "lo": lo, "hi": hi, "n": len(vals)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Accuracy is over judged rows; deferred rows are counted in ``judge_pending``.
    Unparseable judge outputs (correct=None, not deferred) count as wrong."""
    pending = [r for r in rows if r.get("judge_raw") == "deferred"]
    judged = [r for r in rows if r.get("judge_raw") != "deferred"]
    acc = [1.0 if r["correct"] else 0.0 for r in judged]
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for r, a in zip(judged, acc):
        by_cat[r["category"]].append(a)
    ev = [r["evidence_recall"] for r in rows if r.get("evidence_recall") is not None]
    stale = [1.0 if r["stale_hit"] else 0.0 for r in judged if r.get("stale_hit") is not None]
    abst_q = [r for r in rows if r.get("is_abstention_q")]
    ans_q = [r for r in rows if not r.get("is_abstention_q")]
    out: Dict[str, Any] = {
        "n": len(rows),
        "accuracy": _ci(acc),
        "per_category": {k: _ci(v) for k, v in sorted(by_cat.items())},
        "judge_pending": len(pending),
        "judge_unparsed": sum(1 for r in judged if r["correct"] is None),
        "mean_ctx_tokens": float(np.mean([r["n_ctx_tokens"] for r in rows])) if rows else 0.0,
        "p50_retrieve_ms": float(np.median([r["retrieve_ms"] for r in rows])) if rows else 0.0,
        "p95_retrieve_ms": float(np.quantile([r["retrieve_ms"] for r in rows], 0.95)) if rows else 0.0,
    }
    if ev:
        out["evidence_recall"] = _ci(ev)
    if stale:
        out["stale_answer_rate"] = _ci(stale)
    if abst_q:
        out["abstention_recall"] = float(np.mean([r["abstained"] for r in abst_q]))
    if ans_q:
        out["false_abstention_rate"] = float(np.mean([r["abstained"] for r in ans_q]))
    return out


def load_rows(run_dir: str | Path) -> List[Dict[str, Any]]:
    p = Path(run_dir) / "results.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_runs(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        p = Path(p)
        out += sorted({f.parent for f in p.rglob("results.jsonl")}) if p.is_dir() else []
    return out


def compare(run_dirs: Sequence[Path], baseline: Optional[str] = None) -> str:
    """Markdown table: accuracy + CI per run, per-category accuracy, and paired
    tests vs ``baseline`` (matched by run-dir name suffix) on shared questions."""
    runs = {str(d): {r["qid"]: r for r in load_rows(d)} for d in run_dirs}
    if not runs:
        return "(no runs found)"
    cats = sorted({r["category"] for rs in runs.values() for r in rs.values()})
    base_key = next((k for k in runs if baseline and k.rstrip("/").endswith(baseline)), None)
    head = ["run", "n", "acc [95% CI]"] + cats + ["ctx tok", "ev. recall"]
    if base_key:
        head += [f"Δ vs {baseline} [CI]", "McNemar p"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for k, rs in runs.items():
        s = summarize(list(rs.values()))
        a = s["accuracy"]
        if s["judge_pending"]:
            lines.append(f"| {Path(k).name} | {s['n']} | judge pending ({s['judge_pending']}) |" + " |" * (len(head) - 3))
            continue
        row = [Path(k).name if Path(k).parent.name in ("", ".") else f"{Path(k).parent.name}/{Path(k).name}",
               str(s["n"]), f"{a['mean']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}]"]
        row += [f"{s['per_category'][c]['mean']:.3f}" if c in s["per_category"] else "" for c in cats]
        row += [f"{s['mean_ctx_tokens']:.0f}", f"{s['evidence_recall']['mean']:.3f}" if "evidence_recall" in s else ""]
        if base_key:
            if k == base_key:
                row += ["—", "—"]
            else:
                shared = sorted(set(rs) & set(runs[base_key]))
                x = [bool(rs[q]["correct"]) for q in shared]
                y = [bool(runs[base_key][q]["correct"]) for q in shared]
                d, lo, hi, _ = paired_bootstrap([float(v) for v in x], [float(v) for v in y])
                _, _, p = mcnemar(x, y)
                row += [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]", f"{p:.3g}"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
