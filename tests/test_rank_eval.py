import json

import pytest
import yaml

from jarvis_m.data.stress import generate, save
from tests.conftest import TOY

MEM = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}}


def _cfg(tmp_path, sb):
    p = tmp_path / "rank.yaml"
    p.write_text(yaml.safe_dump({"memory": MEM, "datasets": [
        {"label": "toy", "name": "jarvis", "path": str(TOY)},
        {"label": "sb", "name": "jarvis", "path": str(sb)},
    ]}))
    return p


def test_rank_eval_seeds_and_fixed(tmp_path):
    from jarvis_m.cli import main

    sb = save(generate(4, split="dev"), tmp_path / "sb.json")
    out = tmp_path / "rank" / "r.md"
    assert main(["rank-eval", str(_cfg(tmp_path, sb)), "--k", "1", "5", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    md = out.read_text()
    assert [d["label"] for d in res["datasets"]] == ["toy", "sb"] and "## toy" in md and "R@1" in md
    for d in res["datasets"]:
        assert set(d["scorers"]) == {"seeds", "fixed"}
        assert 0 < d["stats"]["candidate_recall"] <= 1 and 0 <= d["stats"]["superseded_fraction"] < 1
        for r in d["scorers"].values():
            assert r["n"] > 0 and 0 < r["MRR"] <= 1 and r["R@1"] <= r["R@5"] <= 1 and r["per_category_mrr"]


def test_rank_eval_with_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from jarvis_m.cli import main
    from jarvis_m.memory.hypergraph.train import train_hgnn

    tr, sb = save(generate(8, split="train"), tmp_path / "tr.json"), save(generate(3, split="dev"), tmp_path / "sb.json")
    ck = train_hgnn({"name": "tiny", "output": str(tmp_path / "tiny.pt"), "memory": MEM,
                     "train_data": [{"name": "jarvis", "path": str(tr)}], "dev_data": [],
                     "model": {"hidden": 8, "layers": 1}, "train": {"epochs": 1, "device": "cpu", "eval_k": 5}},
                    progress=False)
    out = tmp_path / "r.md"
    assert main(["rank-eval", str(_cfg(tmp_path, sb)), "--checkpoint", str(ck), "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["k"] == [5, 10] and res["checkpoints"] == [str(ck)]
    for d in res["datasets"]:
        r = d["scorers"]["hgnn:tiny"]
        assert r["n"] == d["scorers"]["seeds"]["n"] and 0 < r["MRR"] <= 1
    assert "hgnn:tiny" in out.read_text()
