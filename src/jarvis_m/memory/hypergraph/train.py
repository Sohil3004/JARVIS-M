"""Train the HGNN scorer on gold-evidence labels.

Examples are (candidate subgraph, gold evidence mask) pairs built by the same
HypergraphMemory used at eval time, so train and test see identical candidates.
Questions whose evidence never enters the candidate set are counted (the
candidate-recall ceiling) but cannot be learned from.

    jarvis-eval train-hgnn configs/train/hgnn_statebench_lme.yaml
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...data.loaders import dataset_domain, load_dataset
from .. import build_memory
from .subgraph import FEATURE_NAMES, fixed_propagation


@dataclass
class Example:
    sg: Any
    x: np.ndarray
    q: np.ndarray
    labels: np.ndarray
    qid: str
    category: str
    conversation_id: str = ""
    dataset: str = ""  # the config entry's label (or the dataset name)
    domain: Optional[str] = None
    stale: Optional[np.ndarray] = None  # (n,) 1 for stale evidence nodes inside the candidates


def build_examples(memory_cfg: Dict[str, Any], datasets: Sequence[Dict[str, Any]], tokenizer=None,
                   progress: bool = True, require_domain: bool = False) -> Dict[str, Any]:
    out: List[Example] = []
    stats = {"questions": 0, "with_evidence": 0, "evidence_in_candidates": 0}
    graph_stats: List[Dict[str, float]] = []  # per conversation graph, for supersession diagnostics
    for dcfg in datasets:
        domain = dataset_domain(dcfg, required=require_domain)
        ds = load_dataset(dcfg)
        label = dcfg.get("label") or ds.name
        t0 = time.perf_counter()
        for cid, conv in ds.conversations.items():
            qs = [q for q in ds.questions_for(cid) if q.evidence_ids and not q.is_abstention]
            stats["questions"] += len(ds.questions_for(cid))
            if not qs:
                continue
            mem = build_memory(memory_cfg, tokenizer)
            mem.reset()
            mem.ingest(conv.messages)
            g = mem.graph
            graph_stats.append(g.stats())
            if hasattr(mem, "prefetch_queries"):
                mem.prefetch_queries([(q.question, q.asker) for q in qs])
            for q in qs:
                stats["with_evidence"] += 1
                sg = mem.subgraph(q.question, q.asker, q.question_time)
                gold = {g.id2node[e] for e in q.evidence_ids if e in g.id2node}
                labels = np.array([int(v) in gold for v in sg.nodes], np.float32)
                if labels.sum() == 0:
                    continue
                stats["evidence_in_candidates"] += 1
                stale = {g.id2node[e] for e in q.stale_evidence_ids if e in g.id2node}
                out.append(Example(sg, mem.node_embeddings(sg.nodes), mem.query_embedding(q.question, q.asker),
                                   labels, q.id, q.category, cid, label, domain,
                                   np.array([int(v) in stale for v in sg.nodes], np.float32) if stale else None))
        if progress:
            print(f"[hgnn] {ds.name}: {len(ds.questions)} questions, {time.perf_counter() - t0:.1f}s")
    stats["candidate_recall"] = stats["evidence_in_candidates"] / max(1, stats["with_evidence"])
    for key in ("superseded_fraction", "chains_per_100_messages"):
        stats[key] = float(np.mean([s[key] for s in graph_stats])) if graph_stats else 0.0
    return {"examples": out, "stats": stats}


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, k: int) -> Dict[str, float]:
    order = np.argsort(-scores, kind="stable")
    rel = labels[order]
    hits = np.nonzero(rel)[0]
    return {f"recall@{k}": float(rel[:k].sum() / max(1, labels.sum())),
            "mrr": float(1.0 / (hits[0] + 1)) if len(hits) else 0.0,
            "all@k": float(rel[:k].sum() == labels.sum())}


def evaluate(examples: Sequence[Example], score_fn, k: int) -> Dict[str, float]:
    if not examples:
        return {}
    ms = [ranking_metrics(score_fn(e), e.labels, k) for e in examples]
    return {key: float(np.mean([m[key] for m in ms])) for key in ms[0]}


def train_hgnn(cfg: Dict[str, Any], progress: bool = True) -> Path:
    import torch

    from .hgnn import HGNN, resolve_device, to_tensors

    tcfg = {"epochs": 15, "lr": 1e-3, "weight_decay": 1e-4, "seed": 0, "device": "auto", "eval_k": 10,
            "patience": 4, **(cfg.get("train") or {})}
    random.seed(tcfg["seed"])
    np.random.seed(tcfg["seed"])
    torch.manual_seed(tcfg["seed"])
    dev = resolve_device(tcfg["device"])
    mem_cfg = dict(cfg["memory"])
    mem_cfg["scorer"] = {"type": "seeds"}  # building subgraphs only

    tr = build_examples(mem_cfg, cfg["train_data"], cfg.get("tokenizer"), progress)
    dv = build_examples(mem_cfg, cfg["dev_data"], cfg.get("tokenizer"), progress)
    train, dev_ex = tr["examples"], dv["examples"]
    if not train:
        raise SystemExit("no training examples (no evidence inside candidate sets)")
    k = int(tcfg["eval_k"])
    from .memory import DEFAULT_PROP

    edge_w = DEFAULT_PROP
    baselines = {
        "seeds": evaluate(dev_ex, lambda e: e.sg.base, k),
        "fixed": evaluate(dev_ex, lambda e: fixed_propagation(e.sg, edge_w), k),
    }
    if progress:
        print(f"[hgnn] train={len(train)} dev={len(dev_ex)} | candidate recall train={tr['stats']['candidate_recall']:.3f} "
              f"dev={dv['stats']['candidate_recall']:.3f} | dev seeds={baselines['seeds']} fixed={baselines['fixed']}")

    model = HGNN(emb_dim=train[0].x.shape[1], **(cfg.get("model") or {})).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    cache = {id(e): to_tensors(e.sg, e.x, e.q, dev) for e in train + dev_ex}

    def score(e):
        model.eval()
        with torch.no_grad():
            return model(*cache[id(e)]).cpu().numpy()

    best, best_state, bad, history = float("-inf"), None, 0, []  # no dev data: key is -loss (< -1 possible)
    for ep in range(int(tcfg["epochs"])):
        model.train()
        random.shuffle(train)
        tot = 0.0
        for e in train:
            s = model(*cache[id(e)])
            y = torch.as_tensor(e.labels, device=dev) > 0
            loss = torch.logsumexp(s, 0) - torch.logsumexp(s[y], 0)  # listwise, multi-positive
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        m = evaluate(dev_ex, score, k) if dev_ex else {"mrr": -tot}
        history.append({"epoch": ep + 1, "loss": tot / len(train), **m})
        if progress:
            print(f"[hgnn] epoch {ep + 1}: loss={tot / len(train):.4f} dev={m}")
        key = m.get(f"recall@{k}", 0) + m.get("mrr", 0)
        if key > best:
            best, bad = key, 0
            best_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tcfg["patience"]:
                break

    out = Path(cfg.get("output", f"models/hgnn/{cfg.get('name', 'hgnn')}.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"name": cfg.get("name"), "memory": cfg["memory"], "train_stats": tr["stats"], "dev_stats": dv["stats"],
            "dev_baselines": baselines, "history": history}
    torch.save({"state_dict": best_state, "config": model.cfg, "feature_names": list(FEATURE_NAMES), "meta": meta}, out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if progress:
        print(f"[hgnn] saved {out}")
    return out
