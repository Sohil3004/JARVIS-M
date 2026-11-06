"""``jarvis-eval`` command line.

    jarvis-eval run configs/exp/smoke_toy.yaml [--set memory.type=dense ...] [--arm NAME]
    jarvis-eval judge runs/<name> --judge configs/models/judge_qwen3_8b_4bit_hf.yaml [--force]
    jarvis-eval report runs/<name> [--baseline bm25] [--out table.md]
    jarvis-eval data fetch groupmembench|locomo|longmemeval|evermembench
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
        if a.dcmd == "info":
            from .config import load_config
            from .data.loaders import load_dataset

            print(load_dataset(load_config(a.config)["dataset"]).summary())
            return 0
    return 1


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
