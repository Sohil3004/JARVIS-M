"""Reproducible retrieval-ranking tables for hypergraph scorers.

For every dataset, builds each question's candidate subgraph once (same
HypergraphMemory as training, see ``memory.hypergraph.train.build_examples``) and
ranks it with ``seeds`` (= hybrid), ``fixed`` (DEFAULT_PROP) and every HGNN arm. An
arm is one checkpoint or several seeds of one model; multi-seed arms report the mean
over seeds.

    jarvis-eval rank-eval configs/eval/rank_real.yaml \
        --arm hgnn_v2=models/hgnn/hgnn_v2_s{0,1,2}.pt [--checkpoint one.pt] \
        [--k 5 10] [--reference seeds] [--n-boot 10000] [--out results/rank/rank_real.md]

Reports, per dataset, per domain and as the macro-average over domains: R@k, MRR,
nDCG@10, all@k and the stale-rank rate (questions where a stale evidence message
outranks the best gold message). For datasets with access-control metadata (and a memory
with ``access:`` set) it also reports ``leak_rate``: questions whose top-k ranking holds a
message the question's effective principal may not read. With enforcement on this is 0 by
construction; ``access.enforce: false`` measures what an unprotected system would leak. A significance table compares every arm with the
reference arm (conversation-cluster paired bootstrap over seeds x conversations,
Holm-corrected). Per-question scores go to ``<out>.questions.json``.

Config: ``memory:`` (as for training) and ``datasets:`` (dataset configs, each with
``domain:`` and an optional ``label``); optional ``arms:`` (name -> checkpoint paths),
``checkpoints:`` (paths), ``k:``, ``reference:``, ``bootstrap: {n, seed, metric}``.
"""

from __future__ import annotations

import glob as _glob
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..memory.hypergraph.memory import DEFAULT_PROP
from ..memory.hypergraph.subgraph import fixed_propagation
from ..memory.hypergraph.train import Example, build_examples
from .significance import cluster_paired_bootstrap, holm

Scorer = Callable[[Example], np.ndarray]
NDCG_K = 10


# ------------------------------------------------------------------ arm specs
def _split_commas(s: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "{") - (ch == "}")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return [x for x in out + [cur] if x]


def _braces(s: str) -> List[str]:
    m = re.search(r"\{([^{}]*)\}", s)
    if not m:
        return [s]
    return [x for alt in m.group(1).split(",") for x in _braces(s[: m.start()] + alt + s[m.end():])]


def expand_paths(spec: str) -> List[str]:
    """``a.pt,b.pt``, ``m_s{0,1,2}.pt`` and globs (``m_s*.pt``) -> a list of paths."""
    out: List[str] = []
    for part in _split_commas(spec):
        for p in _braces(part):
            hits = sorted(_glob.glob(p)) if any(c in p for c in "*?[") else [p]
            if not hits:
                raise FileNotFoundError(f"no checkpoint matches {p!r}")
            out += hits
    return list(dict.fromkeys(out))


def parse_arms(specs: Sequence[str]) -> Dict[str, List[str]]:
    """``["hgnn_v2=models/hgnn_v2_s{0,1,2}.pt", ...]`` -> {arm name: checkpoint paths}. A name
    may repeat (the shell expands braces into several ``name=path`` words)."""
    arms: Dict[str, List[str]] = {}
    for spec in specs:
        name, eq, paths = spec.partition("=")
        if not eq or not name.strip() or not paths.strip():
            raise ValueError(f"--arm expects NAME=CHECKPOINT[,CHECKPOINT...], got {spec!r}")
        cur = arms.setdefault(name.strip(), [])
        cur += [p for p in expand_paths(paths.strip()) if p not in cur]
    return arms


# ------------------------------------------------------------------ metrics
def question_metrics(scores: np.ndarray, labels: np.ndarray, ks: Sequence[int],
                     stale: Optional[np.ndarray] = None,
                     unreadable: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    """Ranking metrics for one question. ``stale_rank`` is 1 when a stale evidence node is
    ranked above the best gold node, 0 when not, None when the question has no stale evidence.
    ``leak`` is 1 when the top max(k) holds a node its principal may not read (None without
    access-control metadata)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    rel = labels[order]
    n_gold = float(labels.sum())
    hits = np.nonzero(rel)[0]
    first = int(hits[0]) if len(hits) else None
    out: Dict[str, Optional[float]] = {}
    for k in ks:
        out[f"R@{k}"] = float(rel[:k].sum() / max(1.0, n_gold))
        out[f"all@{k}"] = float(rel[:k].sum() == n_gold)
    out["MRR"] = 1.0 / (first + 1) if first is not None else 0.0
    disc = 1.0 / np.log2(np.arange(2, NDCG_K + 2))
    top = rel[:NDCG_K]
    ideal = disc[: int(min(NDCG_K, n_gold))].sum()
    out[f"nDCG@{NDCG_K}"] = float((top * disc[: len(top)]).sum() / ideal) if ideal > 0 else 0.0
    if stale is None:
        out["stale_rank"] = None
    else:
        s_hits = np.nonzero(stale[order])[0]
        out["stale_rank"] = float(len(s_hits) > 0 and (first is None or int(s_hits[0]) < first))
    out["leak"] = None if unreadable is None else float(unreadable[order][: max(ks)].sum() > 0)
    return out


def _mean(xs: Sequence[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def aggregate(rows_by_seed: Sequence[Sequence[Dict[str, Any]]], metric_names: Sequence[str]) -> Dict[str, Any]:
    """Mean over questions per seed, then mean over seeds."""
    out: Dict[str, Any] = {"n": len(rows_by_seed[0]) if rows_by_seed else 0, "n_seeds": len(rows_by_seed)}
    for name in metric_names:
        per_seed = [_mean([r[name] for r in rows]) for rows in rows_by_seed]
        out[name] = _mean(per_seed)
    mrrs = [m for m in (_mean([r["MRR"] for r in rows]) for rows in rows_by_seed) if m is not None]
    out["MRR_seed_std"] = float(np.std(mrrs)) if len(mrrs) > 1 else 0.0
    if "leak" in out:
        out["leak_rate"] = out.pop("leak")
    cats: Dict[str, List[List[float]]] = defaultdict(lambda: [[] for _ in rows_by_seed])
    for s, rows in enumerate(rows_by_seed):
        for r in rows:
            cats[r["category"]][s].append(r["MRR"])
    out["per_category_mrr"] = {c: _mean([_mean(v) for v in per]) for c, per in sorted(cats.items())}
    return out


def score_examples(examples: Sequence[Example], score_fn: Scorer, ks: Sequence[int]) -> Dict[str, Any]:
    """Aggregate metrics of one scorer (kept for callers that need a single table row)."""
    rows = [{"category": e.category, **question_metrics(score_fn(e), e.labels, ks, e.stale, e.unreadable)}
            for e in examples]
    return aggregate([rows], _metric_names(ks))


def _metric_names(ks: Sequence[int]) -> List[str]:
    return [*[f"R@{k}" for k in ks], "MRR", f"nDCG@{NDCG_K}", *[f"all@{k}" for k in ks], "stale_rank", "leak"]


def _checkpoint_scorer(path: str, device: str = "cpu") -> Scorer:
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


# ------------------------------------------------------------------ evaluation
def rank_eval(cfg: Dict[str, Any], checkpoints: Sequence[str] = (), ks: Optional[Sequence[int]] = None,
              device: str = "cpu", progress: bool = True, arms: Optional[Dict[str, Sequence[str]]] = None,
              reference: Optional[str] = None, n_boot: Optional[int] = None) -> Dict[str, Any]:
    ks = [int(k) for k in (ks or cfg.get("k") or (5, 10))]
    names = _metric_names(ks)
    ckpts = list(dict.fromkeys([*(cfg.get("checkpoints") or []), *checkpoints]))
    arm_paths: Dict[str, List[str]] = {}
    for src in (cfg.get("arms") or {}, arms or {}):
        for name, paths in src.items():
            paths = [paths] if isinstance(paths, str) else list(paths)
            cur = arm_paths.setdefault(name, [])
            cur += [q for p in paths for q in expand_paths(str(p)) if q not in cur]
    for p in ckpts:
        arm_paths.setdefault(f"hgnn:{Path(p).stem}", []).append(p)
    clash = {"seeds", "fixed"} & set(arm_paths)
    if clash:
        raise ValueError(f"arm names {sorted(clash)} are reserved")
    boot = {"n": 10000, "seed": 0, "metric": "MRR", **(cfg.get("bootstrap") or {})}
    if n_boot is not None:
        boot["n"] = int(n_boot)
    reference = reference or cfg.get("reference") or "seeds"

    mem_cfg = dict(cfg["memory"])
    mem_cfg["scorer"] = {"type": "seeds"}  # subgraphs only; scorers are applied below
    scorers: Dict[str, List[Scorer]] = {
        "seeds": [lambda e: e.sg.base],
        "fixed": [lambda e: fixed_propagation(e.sg, DEFAULT_PROP)],
    }
    for name, paths in arm_paths.items():
        scorers[name] = [_checkpoint_scorer(p, device) for p in paths]
    if reference not in scorers:
        raise KeyError(f"reference arm {reference!r} not in {list(scorers)}")

    out: Dict[str, Any] = {"k": ks, "checkpoints": ckpts, "arms": {"seeds": [], "fixed": [], **arm_paths},
                           "reference": reference, "bootstrap": boot, "memory": cfg["memory"], "datasets": []}
    # rows[arm][seed] = per-question dicts, all datasets, same question order in every arm
    rows: Dict[str, List[List[Dict[str, Any]]]] = {a: [[] for _ in fns] for a, fns in scorers.items()}
    for dcfg in cfg["datasets"]:
        dcfg = dict(dcfg)
        built = build_examples(mem_cfg, [dcfg], cfg.get("tokenizer"), progress, require_domain=True)
        label = dcfg.pop("label", None) or dcfg["name"]
        ex, st = built["examples"], built["stats"]
        res: Dict[str, Any] = {}
        for arm, fns in scorers.items():
            per_seed = []
            for s, fn in enumerate(fns):
                qrows = [{"qid": e.qid, "conversation_id": e.conversation_id, "dataset": label, "domain": e.domain,
                          "category": e.category, "arm": arm, "seed": s,
                          **question_metrics(fn(e), e.labels, ks, e.stale, e.unreadable)} for e in ex]
                rows[arm][s] += qrows
                per_seed.append(qrows)
            res[arm] = aggregate(per_seed, names)
        out["datasets"].append({"label": label, "domain": dcfg.get("domain"), "config": dcfg, "stats": st,
                                "scorers": res})
        if progress:
            print(f"[rank-eval] {label}: n={len(ex)} " + " ".join(f"{n}={r['MRR']:.3f}" for n, r in res.items()))

    base = rows[reference][0]
    domains = sorted({r["domain"] for r in base})
    out["domains"] = {
        dom: {arm: aggregate([[r for r in seed if r["domain"] == dom] for seed in per], names)
              for arm, per in rows.items()} for dom in domains}
    agg_names = ["leak_rate" if m == "leak" else m for m in names]
    out["macro"] = {arm: {"n_domains": len(domains),
                          **{m: _mean([out["domains"][d][arm][m] for d in domains]) for m in agg_names}}
                    for arm in rows}
    out["significance"] = _significance(rows, reference, domains, boot)
    out["per_question"] = [r for per in rows.values() for seed in per for r in seed]
    return out


def _significance(rows, reference: str, domains: Sequence[str], boot: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every arm vs the reference: one row per domain plus the macro-average over domains.
    Holm correction runs over the whole table (arms x scopes)."""
    metric = boot["metric"]
    base = rows[reference][0]
    if not base:
        return []
    clusters = [(r["dataset"], r["conversation_id"]) for r in base]
    strata = [r["domain"] for r in base]
    vec = {arm: np.array([[r[metric] for r in seed] for seed in per], dtype=np.float64) for arm, per in rows.items()}
    table: List[Dict[str, Any]] = []
    scopes = [(d, np.array([s == d for s in strata])) for d in domains]
    if len(domains) > 1:
        scopes.append(("macro", np.ones(len(base), bool)))
    for scope, mask in scopes:
        idx = np.nonzero(mask)[0]
        for arm in rows:
            if arm == reference:
                continue
            r = cluster_paired_bootstrap(
                vec[arm][:, idx], vec[reference][:, idx], [clusters[i] for i in idx], n=int(boot["n"]),
                seed=int(boot["seed"]), strata=[strata[i] for i in idx] if scope == "macro" else None)
            table.append({"scope": scope, "arm": arm, "reference": reference, "metric": metric, **r})
    for row, p in zip(table, holm([r["p"] for r in table])):
        row["p_holm"] = p
    return table


# ------------------------------------------------------------------ report
def _f(v: Optional[float], nd: int = 3) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def _table(head: Sequence[str], body: Sequence[Sequence[str]]) -> List[str]:
    return ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |",
            *["| " + " | ".join(r) + " |" for r in body], ""]


def to_markdown(res: Dict[str, Any]) -> str:
    ks = res["k"]
    nd = f"nDCG@{NDCG_K}"
    lines = ["# Ranking evaluation", "",
             "Ranking inside each question's candidate subgraph (questions with gold evidence in the candidates). "
             "Multi-seed arms show the mean over seeds. stale-rank = share of questions (with stale evidence) "
             "where a stale message outranks the best gold message; lower is better.", ""]
    multi = {a: len(p) for a, p in res.get("arms", {}).items() if len(p) > 1}
    if multi:
        lines += ["Seeds per arm: " + ", ".join(f"{a} x{n}" for a, n in multi.items()), ""]
    for d in res["datasets"]:
        st = d["stats"]
        cats = sorted({c for r in d["scorers"].values() for c in r["per_category_mrr"]})
        lines += [f"## {d['label']}", "",
                  f"domain: {d.get('domain') or '–'} | questions with evidence: {st['with_evidence']} | "
                  f"candidate recall: {st['candidate_recall']:.3f} | "
                  f"mean superseded fraction: {st.get('superseded_fraction', 0.0):.4f} | "
                  f"chains / 100 msgs: {st.get('chains_per_100_messages', 0.0):.2f}", ""]
        acl = any(r.get("leak_rate") is not None for r in d["scorers"].values())
        head = ["Scorer", "n", *[f"R@{k}" for k in ks], "MRR", nd, "stale-rank", *(["leak_rate"] if acl else []), *cats]
        body = [[name, str(r["n"]), *[_f(r[f"R@{k}"]) for k in ks], _f(r["MRR"]), _f(r[nd]), _f(r["stale_rank"]),
                 *([_f(r.get("leak_rate"))] if acl else []),
                 *[_f(r["per_category_mrr"].get(c)) for c in cats]] for name, r in d["scorers"].items()]
        lines += _table(head, body)
    doms = list(res.get("domains", {}))
    if doms:
        lines += ["## By domain (MRR)", "", "macro = unweighted mean of the domain values.", ""]
        arms = list(res["macro"])
        body = [[a, *[_f(res["domains"][d][a]["MRR"]) for d in doms], _f(res["macro"][a]["MRR"]),
                 _f(res["macro"][a][nd]), _f(res["macro"][a]["stale_rank"])] for a in arms]
        lines += _table(["Scorer", *[f"{d} (n={res['domains'][d][arms[0]]['n']})" for d in doms], "macro MRR",
                         f"macro {nd}", "macro stale-rank"], body)
    sig = res.get("significance") or []
    if sig:
        b = res["bootstrap"]
        lines += [f"## Significance vs `{res['reference']}` ({b['metric']})", "",
                  f"Paired bootstrap over conversations (and seeds), {b['n']} replicates; p is two-sided, "
                  "Holm-corrected over this table.", ""]
        body = [[r["scope"], r["arm"], f"{r['delta']:+.3f}", f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                 f"{r['p']:.4f}", f"{r['p_holm']:.4f}", str(r["n_questions"]), str(r["n_clusters"])] for r in sig]
        lines += _table(["Scope", "Arm", "delta", "95% CI", "p", "p (Holm)", "questions", "conversations"], body)
    return "\n".join(lines)


def write_report(res: Dict[str, Any], out: str | Path) -> Path:
    """Markdown at ``out``, the summary JSON next to it and per-question rows in ``<out>.questions.json``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_markdown(res) + "\n", encoding="utf-8")
    summary = {k: v for k, v in res.items() if k != "per_question"}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    out.with_suffix(".questions.json").write_text(json.dumps(res.get("per_question", []), default=str),
                                                  encoding="utf-8")
    return out
