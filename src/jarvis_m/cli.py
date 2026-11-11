"""``jarvis-eval`` command line.

    jarvis-eval run configs/exp/smoke_toy.yaml [--set memory.type=dense ...] [--arm NAME]
    jarvis-eval judge runs/<name> --judge configs/models/judge_qwen3_8b_4bit_hf.yaml [--force]
    jarvis-eval report runs/<name> [--baseline bm25] [--out table.md]
    jarvis-eval data fetch groupmembench|locomo|longmemeval|evermembench|convomem|socialmembench
    jarvis-eval data make-stress --split train --n 300 [--seed 0] [--out data/stress/statebench_train.json]
    jarvis-eval data make-stress --domain personal --split train --n 300 [--background locomo]   # StateBench v2
    jarvis-eval extract configs/exp/<exp>.yaml --extractor configs/models/extractor_qwen3_4b_4bit_hf.yaml
    jarvis-eval train-hgnn configs/train/hgnn_statebench_lme.yaml
    jarvis-eval rank-eval configs/eval/rank_real.yaml [--arm hgnn_v2=models/hgnn/hgnn_v2_s{0,1,2}.pt] [--k 5 10]
    jarvis-eval supersession-eval configs/eval/supersession.yaml [--labels data/labels/supersession]
    jarvis-eval extract-eval configs/eval/extract.yaml [--labels data/labels/slots]
    jarvis-eval make-lodo configs/train/<train>.yaml --out configs/train/lodo/ [--eval-config configs/eval/<rank>.yaml]
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
    ms.add_argument("--domain", default=None, choices=["team", "assistant", "personal", "enterprise", "mix"],
                    help="StateBench v2 in this domain (without --domain: v1)")
    ms.add_argument("--background", default="none", choices=["none", "locomo", "longmemeval"],
                    help="v2: splice the slot threads into real conversation turns")
    ms.add_argument("--background-path", default=None, help="background dataset file (default: the loader's)")
    ms.add_argument("--background-len", type=int, nargs=2, default=[300, 3000], metavar=("MIN", "MAX"),
                    help="v2 with background: messages per conversation")
    ms.add_argument("--paraphrase", default=None, metavar="YAML",
                    help="v2: rewrite generated messages with the chat model in this file's `chat:` block (GPU, cached)")
    ms.add_argument("--labels-dir", default=None,
                    help="where the supersession labels go (default data/labels/supersession, or "
                         "<out dir>/labels/supersession for an --out outside data/stress)")

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
    re_.add_argument("--checkpoint", action="append", default=[], metavar="PATH",
                     help="HGNN checkpoint as its own arm `hgnn:<file stem>` (repeatable)")
    re_.add_argument("--arm", action="append", nargs="+", default=[], metavar="NAME=CKPT[,CKPT...]",
                     help="named arm; several checkpoints are seeds of one model. Braces and globs work: "
                          "hgnn_v2=models/hgnn/hgnn_v2_s{0,1,2}.pt")
    re_.add_argument("--reference", default=None, help="arm every other arm is tested against (default seeds)")
    re_.add_argument("--n-boot", type=int, default=None, help="bootstrap replicates (default 10000)")
    re_.add_argument("--k", type=int, nargs="+", default=None, help="cutoffs for R@k (default 5 10)")
    re_.add_argument("--out", default=None, help="markdown path (default results/rank/<config>.md); JSON next to it")
    re_.add_argument("--device", default="cpu")
    re_.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    se = sub.add_parser("supersession-eval", help="precision / recall of the graph's version chains vs labelled pairs")
    se.add_argument("config", help="yaml with `memory:` and `datasets:`; optional `labels:` (files or directories)")
    se.add_argument("--labels", action="append", default=None, metavar="PATH",
                    help="label file or directory of *.jsonl (default data/labels/supersession)")
    se.add_argument("--out", default=None, help="markdown path (default results/supersession/<config>.md)")
    se.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    ee = sub.add_parser("extract-eval", help="per-field accuracy of extracted records vs hand-labelled slots")
    ee.add_argument("config", help="yaml with `extractor:` and `datasets:`; optional `labels:` (files or directories)")
    ee.add_argument("--labels", action="append", default=None, metavar="PATH",
                    help="label file or directory of *.jsonl (default data/labels/slots)")
    ee.add_argument("--out", default=None, help="markdown path (default results/extract/<config>.md)")
    ee.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")

    ml = sub.add_parser("make-lodo", help="write leave-one-domain-out train + rank-eval configs")
    ml.add_argument("config", help="train config whose datasets carry `domain:`")
    ml.add_argument("--out", required=True, help="directory for the generated configs")
    ml.add_argument("--eval-config", default=None, help="rank-eval config to take the held-out datasets from")

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
        from .eval.rank_eval import parse_arms, rank_eval, to_markdown, write_report

        res = rank_eval(apply_overrides(load_config(a.config), a.overrides), a.checkpoint, a.k, a.device,
                        arms=parse_arms([s for group in a.arm for s in group]), reference=a.reference,
                        n_boot=a.n_boot)
        out = write_report(res, a.out or Path("results/rank") / f"{Path(a.config).stem}.md")
        print(to_markdown(res))
        print(f"[rank-eval] wrote {out}, {out.with_suffix('.json')} and {out.with_suffix('.questions.json')}")
        return 0

    if a.cmd == "supersession-eval":
        from .config import apply_overrides, load_config
        from .eval.supersession import supersession_eval, to_markdown, write_report

        res = supersession_eval(apply_overrides(load_config(a.config), a.overrides), a.labels)
        out = write_report(res, a.out or Path("results/supersession") / f"{Path(a.config).stem}.md")
        print(to_markdown(res))
        print(f"[supersession-eval] wrote {out} and {out.with_suffix('.json')}")
        return 0

    if a.cmd == "extract-eval":
        from .config import apply_overrides, load_config
        from .eval.extract_eval import extract_eval, to_markdown, write_report

        res = extract_eval(apply_overrides(load_config(a.config), a.overrides), a.labels)
        out = write_report(res, a.out or Path("results/extract") / f"{Path(a.config).stem}.md")
        print(to_markdown(res))
        print(f"[extract-eval] wrote {out} and {out.with_suffix('.json')}")
        return 0

    if a.cmd == "make-lodo":
        from .eval.lodo import make_lodo

        for p in make_lodo(a.config, a.out, a.eval_config):
            print(f"[make-lodo] wrote {p}")
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
            from .data.stress import generate, save, save_supersession_labels

            if a.domain is None:
                if a.background != "none" or a.paraphrase:
                    raise SystemExit("--background and --paraphrase need --domain (StateBench v2)")
                out = Path(a.out or f"data/stress/statebench_{a.split}.json")
                ds = generate(a.n, seed=a.seed, split=a.split, noise=a.noise)
            else:
                from .data.stress_v2 import generate_v2, load_background, paraphrase

                bg = None if a.background == "none" else load_background(a.background, a.background_path)
                suffix = "" if a.background == "none" else f"_bg{a.background}"
                out = Path(a.out or f"data/stress/statebench_v2_{a.domain}_{a.split}{suffix}.json")
                ds = generate_v2(a.n, seed=a.seed, split=a.split, domain=a.domain, background=bg,
                                 background_len=tuple(a.background_len))
                if suffix:
                    ds.name += suffix
                    for row in ds.annotations["supersession"]:
                        row["dataset"] = ds.name
                if a.paraphrase:
                    from .config import load_config
                    from .llm import build_chat

                    pcfg = load_config(a.paraphrase)
                    if "chat" not in pcfg:
                        raise SystemExit(f"{a.paraphrase} has no top-level `chat:` block")
                    paraphrase(ds, build_chat(pcfg["chat"]), out.with_suffix(".paraphrase_cache.jsonl"))
                    ds.name += "_para"
                    out = out.with_name(out.stem + "_para.json")
                    for row in ds.annotations["supersession"]:
                        row["dataset"] = ds.name
            print(f"[stress] {ds.summary()} -> {save(ds, out)}")
            in_repo = out.parent.as_posix().endswith("data/stress")
            labels_dir = a.labels_dir or ("data/labels/supersession" if in_repo
                                          else out.parent / "labels" / "supersession")
            lp = save_supersession_labels(ds, labels_dir)
            if lp:
                print(f"[stress] {len(ds.annotations['supersession'])} supersession labels -> {lp}")
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
    xcfg = dict(ecfg["extractor"])
    if xcfg.get("type") == "llm":  # batch cache + slot registry persisted with the extraction cache
        xcfg.setdefault("cache", store)
        xcfg.setdefault("registry_path", str(store.registry_path))
    extractor = build_extractor(xcfg)
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
    if hasattr(extractor, "hit_rate"):
        print(f"[extract] {extractor.batches} batches, {extractor.cache_hits} from the batch cache "
              f"(hit rate {extractor.hit_rate():.1%}), {extractor.calls} LLM calls; "
              f"{len(done)} messages were already in the cache")
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
