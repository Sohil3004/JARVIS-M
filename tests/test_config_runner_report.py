import json
from pathlib import Path

import yaml

from jarvis_m.cli import main
from jarvis_m.config import apply_overrides, deep_merge, expand, load_config
from jarvis_m.eval.report import compare, find_runs
from jarvis_m.eval.runner import run
from tests.conftest import ROOT, TOY


def test_merge_overrides_expand(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nb: 1\n")
    (tmp_path / "exp.yaml").write_text(
        "base: base.yaml\na: {y: 3}\nsweep: {a.x: [1, 2], b: [5]}\n"
    )
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg["a"] == {"x": 1, "y": 3}
    cfg = apply_overrides(cfg, ["a.z=[1, 2]", "c=hello"])
    assert cfg["a"]["z"] == [1, 2] and cfg["c"] == "hello"
    arms = expand(cfg)
    assert [n for n, _ in arms] == ["x=1,b=5", "x=2,b=5"]
    assert arms[1][1]["a"]["x"] == 2
    assert deep_merge({"m": {"t": 1, "k": 2}}, {"m": {"_replace": True, "t": 3}}) == {"m": {"t": 3}}


def test_shipped_configs_parse():
    for p in (ROOT / "configs" / "exp").glob("*.yaml"):
        arms = expand(load_config(p))
        assert arms and all("memory" in c and "dataset" in c for _, c in arms), p


def _smoke_cfg(tmp_path):
    cfg = load_config(ROOT / "configs" / "exp" / "smoke_toy.yaml")
    cfg["output_dir"] = str(tmp_path / "runs")
    cfg["dataset"]["path"] = str(TOY)
    return cfg


def test_smoke_run_resume_and_report(tmp_path):
    cfg = _smoke_cfg(tmp_path)
    dirs = [run(c, arm=n, progress=False) for n, c in expand(cfg)]
    for d in dirs:
        rows = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 8
        assert {"correct", "evidence_recall", "stale_hit", "n_ctx_tokens"} <= set(rows[0])
        s = json.loads((d / "summary.json").read_text())
        assert 0 <= s["accuracy"]["mean"] <= 1 and "knowledge_update" in s["per_category"]
    # resume: re-running appends nothing
    run(expand(cfg)[1][1], arm=expand(cfg)[1][0], progress=False)
    assert len((dirs[1] / "results.jsonl").read_text().splitlines()) == 8
    table = compare(find_runs([str(tmp_path / "runs")]), baseline="bm25")
    assert "McNemar" in table and "legacy_v2" in table and "recency" in table


def test_retrieval_beats_recency_on_toy(tmp_path):
    """Sanity: lexical retrieval should recover more gold evidence than recency."""
    cfg = _smoke_cfg(tmp_path)
    arms = dict(expand(cfg))
    rec = json.loads((run(arms["recency"], "recency", False) / "summary.json").read_text())
    bm = json.loads((run(arms["bm25"], "bm25", False) / "summary.json").read_text())
    assert bm["evidence_recall"]["mean"] > rec["evidence_recall"]["mean"]


def test_cli_dry_run_and_report(tmp_path, capsys):
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(_smoke_cfg(tmp_path)))
    assert main(["run", str(cfg_path), "--dry-run"]) == 0
    assert "legacy_v2" in capsys.readouterr().out
    assert main(["run", str(cfg_path), "--arm", "bm25", "--set", "retrieval.max_items=2"]) == 0
    assert main(["report", str(tmp_path / "runs"), "--out", str(tmp_path / "t.md")]) == 0
    assert Path(tmp_path / "t.md").read_text().startswith("| run")
