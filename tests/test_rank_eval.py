import json

import numpy as np
import pytest
import yaml

from jarvis_m.data.stress import generate, save
from jarvis_m.eval.rank_eval import expand_paths, parse_arms, question_metrics
from tests.conftest import TOY

MEM = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}}


def _cfg(tmp_path, sb, **extra):
    p = tmp_path / "rank.yaml"
    p.write_text(yaml.safe_dump({"memory": MEM, "bootstrap": {"n": 200}, "datasets": [
        {"label": "toy", "domain": "team", "name": "jarvis", "path": str(TOY)},
        {"label": "sb", "domain": "enterprise", "name": "jarvis", "path": str(sb)},
    ], **extra}))
    return p


def test_question_metrics():
    labels = np.array([0, 1, 0, 1, 0], np.float32)
    stale = np.array([1, 0, 0, 0, 0], np.float32)
    m = question_metrics(np.array([0.9, 0.8, 0.7, 0.1, 0.0]), labels, [1, 2, 5], stale)
    assert m["MRR"] == 0.5 and m["R@1"] == 0 and m["R@2"] == 0.5 and m["R@5"] == 1
    assert m["all@2"] == 0 and m["all@5"] == 1 and m["stale_rank"] == 1.0  # the stale node is ranked first
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    assert m["nDCG@10"] == pytest.approx(dcg / (1 + 1 / np.log2(3)))
    ok = question_metrics(np.array([0.1, 0.9, 0.7, 0.8, 0.0]), labels, [2], stale)
    assert ok["MRR"] == 1 and ok["nDCG@10"] == pytest.approx(1.0) and ok["stale_rank"] == 0.0
    assert question_metrics(np.array([0.1, 0.9, 0.7, 0.8, 0.0]), labels, [2])["stale_rank"] is None
    # stale evidence outside the candidates cannot outrank anything
    assert question_metrics(np.ones(5), labels, [2], np.zeros(5, np.float32))["stale_rank"] == 0.0


def test_arm_specs(tmp_path):
    for s in (0, 1, 2):
        (tmp_path / f"m_s{s}.pt").write_text("x")
    base = str(tmp_path / "m_s")
    three = [f"{base}{s}.pt" for s in (0, 1, 2)]
    assert expand_paths(base + "{0,1,2}.pt") == three and expand_paths(base + "*.pt") == three
    assert expand_paths(f"{base}0.pt,{base}2.pt") == [three[0], three[2]]
    # quoted braces, shell-expanded braces (three words) and a comma list give the same arm
    assert parse_arms([f"v2={base}{{0,1,2}}.pt"]) == {"v2": three}
    assert parse_arms([f"v2={p}" for p in three] + [f"mlp={three[0]}"]) == {"v2": three, "mlp": [three[0]]}
    with pytest.raises(ValueError):
        parse_arms(["no_equals_sign"])
    with pytest.raises(FileNotFoundError):
        expand_paths(str(tmp_path / "missing_*.pt"))


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
            assert 0 < r["nDCG@10"] <= 1 and r["all@1"] <= r["all@5"] and r["n_seeds"] == 1
    # StateBench questions carry stale evidence, the toy fixture does not
    assert res["datasets"][0]["scorers"]["seeds"]["stale_rank"] is None
    assert 0 <= res["datasets"][1]["scorers"]["seeds"]["stale_rank"] <= 1

    # per domain and macro-average over domains
    assert set(res["domains"]) == {"team", "enterprise"} and "## By domain" in md
    for arm in ("seeds", "fixed"):
        doms = [res["domains"][d][arm]["MRR"] for d in ("team", "enterprise")]
        assert res["macro"][arm]["MRR"] == pytest.approx(np.mean(doms)) and res["macro"][arm]["n_domains"] == 2
        assert res["domains"]["team"][arm]["MRR"] == pytest.approx(res["datasets"][0]["scorers"][arm]["MRR"])

    # significance vs the reference arm, Holm-corrected
    sig = res["significance"]
    assert {(r["scope"], r["arm"]) for r in sig} == {(s, "fixed") for s in ("team", "enterprise", "macro")}
    assert res["reference"] == "seeds" and "Significance vs `seeds`" in md
    for r in sig:
        assert r["ci_low"] <= r["delta"] <= r["ci_high"] and 0 < r["p"] <= r["p_holm"] <= 1
    macro = next(r for r in sig if r["scope"] == "macro")
    assert macro["delta"] == pytest.approx(res["macro"]["fixed"]["MRR"] - res["macro"]["seeds"]["MRR"])

    # per-question scores
    rows = json.loads(out.with_suffix(".questions.json").read_text())
    n = sum(d["scorers"]["seeds"]["n"] for d in res["datasets"])
    assert len(rows) == 2 * n and "per_question" not in res
    need = {"qid", "conversation_id", "dataset", "domain", "category", "arm", "seed", "R@1", "R@5", "MRR",
            "nDCG@10", "all@1", "all@5", "stale_rank"}
    assert all(need <= set(r) for r in rows)
    sb_seeds = [r for r in rows if r["dataset"] == "sb" and r["arm"] == "seeds"]
    assert np.mean([r["MRR"] for r in sb_seeds]) == pytest.approx(res["datasets"][1]["scorers"]["seeds"]["MRR"])
    assert {r["domain"] for r in sb_seeds} == {"enterprise"} and len({r["conversation_id"] for r in sb_seeds}) > 1


def test_rank_eval_needs_a_domain_and_a_known_reference(tmp_path):
    from jarvis_m.eval.rank_eval import rank_eval

    ds = {"name": "jarvis", "path": str(TOY)}
    with pytest.raises(ValueError, match="domain"):
        rank_eval({"memory": MEM, "datasets": [ds]}, progress=False)
    with pytest.raises(ValueError, match="unknown domain"):
        rank_eval({"memory": MEM, "datasets": [{**ds, "domain": "work"}]}, progress=False)
    with pytest.raises(KeyError, match="reference"):
        rank_eval({"memory": MEM, "datasets": [{**ds, "domain": "team"}]}, progress=False, reference="nope")
    one = rank_eval({"memory": MEM, "bootstrap": {"n": 50}, "datasets": [{**ds, "domain": "team"}]}, progress=False)
    assert [r["scope"] for r in one["significance"]] == ["team"]  # one domain: no macro row


def test_rank_eval_with_checkpoints_and_multi_seed_arm(tmp_path):
    pytest.importorskip("torch")
    from jarvis_m.cli import main
    from jarvis_m.memory.hypergraph.train import train_hgnn

    tr, sb = save(generate(8, split="train"), tmp_path / "tr.json"), save(generate(3, split="dev"), tmp_path / "sb.json")
    cks = [train_hgnn({"name": "tiny", "output": str(tmp_path / f"tiny_s{s}.pt"), "memory": MEM,
                       "train_data": [{"name": "jarvis", "path": str(tr)}], "dev_data": [],
                       "model": {"hidden": 8, "layers": 1},
                       "train": {"epochs": 1, "device": "cpu", "eval_k": 5, "seed": s}}, progress=False)
           for s in (0, 1)]
    out = tmp_path / "r.md"
    arm = f"tiny={tmp_path / 'tiny_s'}{{0,1}}.pt"
    assert main(["rank-eval", str(_cfg(tmp_path, sb)), "--checkpoint", str(cks[0]), "--arm", arm,
                 "--n-boot", "100", "--reference", "fixed", "--out", str(out)]) == 0
    res = json.loads(out.with_suffix(".json").read_text())
    assert res["k"] == [5, 10] and res["checkpoints"] == [str(cks[0])]
    assert res["arms"]["tiny"] == [str(c) for c in cks] and res["bootstrap"]["n"] == 100
    rows = json.loads(out.with_suffix(".questions.json").read_text())
    for d in res["datasets"]:
        single, multi = d["scorers"]["hgnn:tiny_s0"], d["scorers"]["tiny"]
        assert single["n"] == multi["n"] == d["scorers"]["seeds"]["n"] and 0 < single["MRR"] <= 1
        assert multi["n_seeds"] == 2 and multi["MRR_seed_std"] >= 0
        per_seed = [np.mean([r["MRR"] for r in rows if r["arm"] == "tiny" and r["seed"] == s
                             and r["dataset"] == d["label"]]) for s in (0, 1)]
        assert multi["MRR"] == pytest.approx(np.mean(per_seed)) and per_seed[0] == pytest.approx(single["MRR"])
    assert {r["arm"] for r in res["significance"]} == {"seeds", "hgnn:tiny_s0", "tiny"}
    md = out.read_text()
    assert "hgnn:tiny_s0" in md and "tiny x2" in md and "Significance vs `fixed`" in md
