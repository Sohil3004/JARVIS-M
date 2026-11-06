"""Two-pass flow for small GPUs: reader pass with judge deferred, then a judge pass."""

import json

import yaml

from jarvis_m.cli import main
from jarvis_m.config import expand, load_config
from jarvis_m.eval.qa import MatchJudge
from jarvis_m.eval.rejudge import rejudge_run
from jarvis_m.eval.report import summarize
from jarvis_m.eval.runner import run
from jarvis_m.llm import _CACHE, build_chat, release_all
from tests.conftest import ROOT, TOY


def _cfg(tmp_path, judge):
    cfg = load_config(ROOT / "configs" / "exp" / "smoke_toy.yaml")
    cfg.update(output_dir=str(tmp_path / "runs"), judge=judge)
    cfg["dataset"]["path"] = str(TOY)
    return dict(expand(cfg))


def _rows(d):
    return {json.loads(x)["qid"]: json.loads(x) for x in (d / "results.jsonl").read_text().splitlines()}


def test_deferred_then_judge_matches_inline(tmp_path):
    inline = run(_cfg(tmp_path / "a", {"type": "match"})["legacy_v2"], "legacy_v2", False)
    deferred = run(_cfg(tmp_path / "b", {"type": "deferred"})["legacy_v2"], "legacy_v2", False)
    s = json.loads((deferred / "summary.json").read_text())
    assert s["judge_pending"] == 8 and s["accuracy"]["n"] == 0
    assert all(r["correct"] is None for r in _rows(deferred).values())

    rejudge_run(deferred, MatchJudge(), {"type": "match"}, progress=False)
    a, b = _rows(inline), _rows(deferred)
    assert {q: r["correct"] for q, r in a.items()} == {q: r["correct"] for q, r in b.items()}
    assert {q: r["stale_hit"] for q, r in a.items()} == {q: r["stale_hit"] for q, r in b.items()}
    assert not (deferred / "results.judged.jsonl").exists()
    assert json.loads((deferred / "summary.json").read_text())["judge_pending"] == 0


def test_rejudge_resumes_partial_progress(tmp_path):
    d = run(_cfg(tmp_path, {"type": "deferred"})["bm25"], "bm25", False)
    rows = list(_rows(d).values())
    # simulate a crash after 3 verdicts: they sit in results.judged.jsonl
    partial = [{**r, "correct": True, "judge_raw": "earlier"} for r in rows[:3]]
    (d / "results.judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in partial))
    calls = []

    class Counting(MatchJudge):
        def judge(self, q, pred):
            calls.append(q.id)
            return super().judge(q, pred)

    rejudge_run(d, Counting(), {"type": "match"}, progress=False)
    assert len(calls) == 5  # only the remaining rows were graded
    assert sum(r["judge_raw"] == "earlier" for r in _rows(d).values()) == 3


def test_cli_judge_and_then_judge(tmp_path, capsys):
    exp = tmp_path / "exp.yaml"
    cfg = load_config(ROOT / "configs" / "exp" / "smoke_toy.yaml")
    cfg.update(output_dir=str(tmp_path / "runs"), judge={"type": "deferred"})
    cfg["dataset"]["path"] = str(TOY)
    exp.write_text(yaml.safe_dump(cfg))
    jy = tmp_path / "judge.yaml"
    jy.write_text("judge: {type: match}\n")

    assert main(["run", str(exp), "--arm", "bm25"]) == 0
    assert main(["report", str(tmp_path / "runs")]) == 0
    assert "judge pending" in capsys.readouterr().out
    assert main(["judge", str(tmp_path / "runs"), "--judge", str(jy)]) == 0
    s = summarize(list(_rows(tmp_path / "runs" / "smoke_toy" / "bm25").values()))
    assert s["judge_pending"] == 0 and s["accuracy"]["n"] == 8

    assert main(["run", str(exp), "--arm", "dense", "--then-judge", str(jy)]) == 0
    assert summarize(list(_rows(tmp_path / "runs" / "smoke_toy" / "dense").values()))["judge_pending"] == 0


def test_release_all_clears_model_cache():
    build_chat({"type": "overlap"})
    assert _CACHE
    assert release_all() >= 1 and not _CACHE
