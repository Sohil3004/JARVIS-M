"""Reproducible retrieval-ranking tables for hypergraph scorers.

For every dataset, builds each question's candidate subgraph once (same
HypergraphMemory as training, see ``memory.hypergraph.train.build_examples``) and
ranks it with ``seeds`` (= hybrid), ``fixed`` (DEFAULT_PROP) and every HGNN
checkpoint. Reports R@k, MRR, per-category MRR, n, candidate recall and the mean
fraction of superseded nodes per conversation graph.

    jarvis-eval rank-eval configs/eval/rank_real.yaml --checkpoint models/hgnn/hgnn_v1.pt \
        [--k 5 10] [--out results/rank/rank_real.md] [--set key=value]

Config: ``memory:`` (as for training) and ``datasets:`` (dataset configs, each with an
optional ``label``); optional ``checkpoints:`` (paths) and ``k:``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..memory.hypergraph.memory import DEFAULT_PROP
from ..memory.hypergraph.subgraph import fixed_propagation
from ..memory.hypergraph.train import Example, build_examples


def _rank_of_first_hit(scores: np.ndarray, labels: np.ndarray) -> Optional[int]:
    hits = np.nonzero(labels[np.argsort(-scores, kind="stable")])[0]
    return int(hits[0]) if len(hits) else None


def score_examples(examples: Sequence[Example], score_fn: Callable[[Example], np.ndarray],
                   ks: Sequence[int]) -> Dict[str, Any]:
    """R@k (fraction of gold in the top k), MRR, per-category MRR. Every example has
    gold inside its candidates (build_examples drops the rest; see candidate recall)."""
    rec: Dict[int, List[float]] = {k: [] for k in ks}
    rr: List[float] = []
    by_cat: Dict[str, List[float]] = defaultdict(list)
    for e in examples:
        s = np.asarray(score_fn(e), dtype=np.float64)
        rel = e.labels[np.argsort(-s, kind="stable")]
        for k in ks:
            rec[k].append(float(rel[:k].sum() / max(1.0, e.labels.sum())))
        r = _rank_of_first_hit(s, e.labels)
        v = 1.0 / (r + 1) if r is not None else 0.0
        rr.append(v)
        by_cat[e.category].append(v)
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return {"n": len(examples), **{f"R@{k}": mean(rec[k]) for k in ks}, "MRR": mean(rr),
            "per_category_mrr": {c: mean(v) for c, v in sorted(by_cat.items())}}


def _checkpoint_scorer(path: str, device: str = "cpu"):
    from ..memory.hypergraph.hgnn import HGNNScorer, to_tensors

    sc = HGNNScorer.load(path, device)
    dim = sc.model.cfg["emb_dim"]

    def fn(e: Example) -> np.ndarray:
        if e.x.shape[1] != dim:
            raise ValueError(f"{path}: embedder dim {e.x.shape[1]} != checkpoint emb_dim {dim}")
        import torch

        with torch.no_grad():
            return sc.model(*to_tensors(e.sg, e.x, e.q, sc.device)).float().cpu().numpy()

    return fn


def rank_eval(cfg: Dict[str, Any], checkpoints: Sequence[str] = (), ks: Optional[Sequence[int]] = None,
              device: str = "cpu", progress: bool = True) -> Dict[str, Any]:
    ks = [int(k) for k in (ks or cfg.get("k") or (5, 10))]
    ckpts = list(dict.fromkeys([*(cfg.get("checkpoints") or []), *checkpoints]))
    mem_cfg = dict(cfg["memory"])
    mem_cfg["scorer"] = {"type": "seeds"}  # subgraphs only; scorers are applied below
    scorers: Dict[str, Callable[[Example], np.ndarray]] = {
        "seeds": lambda e: e.sg.base,
        "fixed": lambda e: fixed_propagation(e.sg, DEFAULT_PROP),
    }
    for p in ckpts:
        scorers[f"hgnn:{Path(p).stem}"] = _checkpoint_scorer(p, device)
    out: Dict[str, Any] = {"k": ks, "checkpoints": ckpts, "memory": cfg["memory"], "datasets": []}
    for dcfg in cfg["datasets"]:
        dcfg = dict(dcfg)
        label = dcfg.pop("label", None) or dcfg["name"]
        built = build_examples(mem_cfg, [dcfg], cfg.get("tokenizer"), progress)
        ex, st = built["examples"], built["stats"]
        res = {name: score_examples(ex, fn, ks) for name, fn in scorers.items()}
        out["datasets"].append({"label": label, "config": dcfg, "stats": st, "scorers": res})
        if progress:
            print(f"[rank-eval] {label}: n={len(ex)} " + " ".join(f"{n}={r['MRR']:.3f}" for n, r in res.items()))
    return out


def to_markdown(res: Dict[str, Any]) -> str:
    ks = res["k"]
    lines = ["# Ranking evaluation", "",
             "Ranking inside each question's candidate subgraph (questions with gold evidence in the candidates).", ""]
    for d in res["datasets"]:
        st = d["stats"]
        cats = sorted({c for r in d["scorers"].values() for c in r["per_category_mrr"]})
        lines += [f"## {d['label']}", "",
                  f"questions with evidence: {st['with_evidence']} | candidate recall: {st['candidate_recall']:.3f} | "
                  f"mean superseded fraction: {st.get('superseded_fraction', 0.0):.4f} | "
                  f"chains / 100 msgs: {st.get('chains_per_100_messages', 0.0):.2f}", ""]
        head = ["Scorer", "n", *[f"R@{k}" for k in ks], "MRR", *cats]
        lines += ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        for name, r in d["scorers"].items():
            row = [name, str(r["n"]), *[f"{r[f'R@{k}']:.3f}" for k in ks], f"{r['MRR']:.3f}",
                   *[f"{r['per_category_mrr'][c]:.3f}" if c in r["per_category_mrr"] else "–" for c in cats]]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def write_report(res: Dict[str, Any], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    return out
