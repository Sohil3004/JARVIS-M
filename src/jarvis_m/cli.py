"""``jarvis-eval`` command line.

    jarvis-eval run configs/exp/smoke_toy.yaml [--set memory.type=dense ...] [--arm NAME]
    jarvis-eval judge runs/<name> --judge configs/models/judge_qwen3_8b_4bit_hf.yaml [--force]
    jarvis-eval report runs/<name> [--baseline bm25] [--out table.md]
    jarvis-eval data fetch groupmembench|locomo|longmemeval|evermembench
    jarvis-eval data make-stress --split train --n 300 [--seed 0] [--out data/stress/statebench_train.json]
    jarvis-eval extract configs/exp/<exp>.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml
    jarvis-eval train-hgnn configs/train/hgnn_statebench_lme.yaml
    jarvis-eval rank-eval configs/eval/rank_real.yaml [--checkpoint models/hgnn/hgnn_v1.pt ...] [--k 5 10]
    jarvis-eval data info configs/exp/gmb_baselines.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jarvis-eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment (all arms unless --arm)")
    r.add_argument("config")
    r.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    r.add_argument("--arm", action="append", default=None, help="only run these arm names")
    r.add_argument("--dry-run", action="store_true", help="print expanded arms and exit")
    r.add_argument("--then-judge", metavar="JUDGE_YAML", default=None,
                   help="after all arms: free the reader, load this judge, grade deferred answers")

    j = sub.add_parser("judge", help="grade saved (deferred) answers; loads the judge once for all runs")
    j.add_argument("paths", nargs="+", help="run dirs or parents of run dirs")
    j.add_argument("--judge", required=True, metavar="YAML", help="file with a top-level `judge:` block")
    j.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    j.add_argument("--force", action="store_true", help="re-grade rows that already have a verdict")

    rp = sub.add_parser("report", help="compare runs under one or more directories")
    rp.add_argument("paths", nargs="+")
    rp.add_argument("--baseline", default=None)
    rp.add_argument("--out", default=None)

    d = sub.add_parser("data")
    dsub = d.add_subparsers(dest="dcmd", required=True)
    f = dsub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--force", action="store_true")
    i = dsub.add_parser("info")
    i.add_argument("config")
    ms = dsub.add_parser("make-stress", help="generate Jarvis-StateBench conversations")
    ms.add_argument("--split", choices=["train", "dev", "test"], default="train")
    ms.add_argument("--n", type=int, default=300)
    ms.add_argument("--seed", type=int, default=0)
    ms.add_argument("--noise", type=float, default=1.0)
    ms.add_argument("--out", default=None)

    ex = sub.add_parser("extract", help="offline (LLM) state extraction for an experiment's dataset, resumable")
    ex.add_argument("config", help="experiment yaml (its `dataset:` is used)")
    ex.add_argument("--extractor", required=True, metavar="YAML", help="file with a top-level `extractor:` block")
    ex.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ex.add_argument("--out", default=None, help="cache path (default data/extract/<dataset>/<name>.jsonl)")

    th = sub.add_parser("train-hgnn", help="train the HGNN scorer on gold-evidence labels")
    th.add_argument("config")
    th.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    re_ = sub.add_parser("rank-eval", help="ranking tables (R@k, MRR) for seeds / fixed / HGNN checkpoints")
    re_.add_argument("config", help="yaml with `memory:` and `datasets:` (each with an optional `label`)")
    re_.add_argument("--checkpoint", action="append", default=[], metavar="PATH", help="HGNN checkpoint (repeatable)")
    re_.add_argument("--k", type=int, nargs="+", default=None, help="cutoffs for R@k (default 5 10)")
    re_.add_argument("--out", default=None, help="markdown path (default results/rank/<config>.md); JSON next to it")
    re_.add_argument("--device", default="cpu")
    re_.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .config import apply_overrides, expand, load_config
        from .eval.runner import run

        cfg = apply_overrides(load_config(a.config), a.overrides)
        arms = expand(cfg)
        if a.arm:
            arms = [(n, c) for n, c in arms if n in set(a.arm)]
        if a.dry_run:
            for n, c in arms:
                print(f"- {n or '(single)'}: memory={c.get('memory')} retrieval={c.get('retrieval')}")
            return 0
        dirs = [run(c, arm=n) for n, c in arms]
        if a.then_judge:
            from .llm import release_all

            release_all()  # reader weights off the GPU before the judge loads
            return _judge(dirs, a.then_judge, [], force=False)
        return 0

    if a.cmd == "judge":
        from .eval.report import find_runs

        return _judge(find_runs(a.paths), a.judge, a.overrides, a.force)

    if a.cmd == "extract":
        return _extract(a)

    if a.cmd == "train-hgnn":
        from .config import apply_overrides, load_config
        from .memory.hypergraph.train import train_hgnn

        train_hgnn(apply_overrides(load_config(a.config), a.overrides))
        return 0

    if a.cmd == "rank-eval":
        from .config import apply_overrides, load_config
        from .eval.rank_eval import rank_eval, to_markdown, write_report

        res = rank_eval(apply_overrides(load_config(a.config), a.overrides), a.checkpoint, a.k, a.device)
        out = write_report(res, a.out or Path("results/rank") / f"{Path(a.config).stem}.md")
        print(to_markdown(res))
        print(f"[rank-eval] wrote {out} and {out.with_suffix('.json')}")
        return 0

    if a.cmd == "report":
        from .eval.report import compare, find_runs

        table = compare(find_runs(a.paths), a.baseline)
        print(table)
        if a.out:
            Path(a.out).write_text(table + "\n", encoding="utf-8")
        return 0

    if a.cmd == "data":
        if a.dcmd == "fetch":
            from .data.fetch import fetch

            fetch(a.name, force=a.force)
            return 0
        if a.dcmd == "make-stress":
            from .data.stress import generate, save

            out = a.out or f"data/stress/statebench_{a.split}.json"
            ds = generate(a.n, seed=a.seed, split=a.split, noise=a.noise)
            print(f"[stress] {ds.summary()} -> {save(ds, out)}")
            return 0
        if a.dcmd == "info":
            from .config import load_config
            from .data.loaders import load_dataset

            print(load_dataset(load_config(a.config)["dataset"]).summary())
            return 0
    return 1


def _extract(a) -> int:
    import time

    from .config import apply_overrides, load_config
    from .data.loaders import load_dataset
    from .extract import ExtractionStore, build_extractor

    ecfg = apply_overrides(load_config(a.extractor), a.overrides)
    if "extractor" not in ecfg:
        raise SystemExit(f"{a.extractor} has no top-level `extractor:` block")
    ds = load_dataset(load_config(a.config)["dataset"])
    name = ecfg.get("name", Path(a.extractor).stem)
    out = Path(a.out or Path("data/extract") / ds.name / f"{name}.jsonl")
    store = ExtractionStore(out)
    done = store.done()
    extractor = build_extractor(ecfg["extractor"])
    total = sum(len(c.messages) for c in ds.conversations.values())
    print(f"[extract] {ds.name}: {total} messages, {len(done)} already in {out}")
    t0, n_done = time.perf_counter(), [0]

    def on_batch(recs, ids):
        store.append(recs, ids)
        n_done[0] += len(ids)
        if n_done[0] % 400 < len(ids):
            rate = n_done[0] / max(1e-6, time.perf_counter() - t0)
            left = (total - len(done) - n_done[0]) / max(rate, 1e-6)
            print(f"  {n_done[0]} msgs, {rate:.1f} msg/s, ~{left / 3600:.1f} h left")

    for conv in ds.conversations.values():
        todo = [m for m in conv.messages if m.id not in done]
        if not todo:
            continue
        if hasattr(extractor, "extract") and "on_batch" in extractor.extract.__code__.co_varnames:
            extractor.extract(todo, on_batch=on_batch)
        else:
            recs = extractor.extract(todo)
            on_batch(recs, [m.id for m in todo])
    print(f"[extract] done -> {out}  (use memory.extractor: {{type: cached, path: {out.as_posix()}}})")
    return 0


def _judge(run_dirs, judge_yaml, overrides, force) -> int:
    from .config import apply_overrides, load_config
    from .eval.qa import build_judge
    from .eval.rejudge import rejudge_run

    cfg = apply_overrides(load_config(judge_yaml), overrides)
    if "judge" not in cfg:
        raise SystemExit(f"{judge_yaml} has no top-level `judge:` block")
    judge = build_judge(cfg["judge"])
    if not run_dirs:
        print("[judge] no runs found")
        return 1
    for d in run_dirs:
        rejudge_run(Path(d), judge, cfg["judge"], force=force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
