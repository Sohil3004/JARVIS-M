import json

import pytest
import yaml

from jarvis_m.data.loaders import load_jarvis
from jarvis_m.data.stress import generate, save, save_supersession_labels
from jarvis_m.eval.supersession import LABELS, load_labels, score_pairs, supersession_eval

MEM = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}}


def test_statebench_emits_stale_evidence_and_supersession_labels(tmp_path):
    ds = generate(30, split="dev")
    when = {m.id: m.timestamp for c in ds.conversations.values() for m in c.messages}
    conv_of = {m.id: c.id for c in ds.conversations.values() for m in c.messages}
    rows = ds.annotations["supersession"]
    assert {r["label"] for r in rows} == set(LABELS)
    for r in rows:
        assert r["dataset"] == ds.name == "statebench_dev" and r["holder"].startswith("group:")
        assert conv_of[r["older_id"]] == conv_of[r["newer_id"]] == r["conversation_id"]
        assert when[r["older_id"]] < when[r["newer_id"]]
    pos = {(r["older_id"], r["newer_id"]) for r in rows if r["label"] == "supersedes"}
    n_stale = 0
    for q in ds.questions:
        assert not set(q.stale_evidence_ids) & set(q.evidence_ids) and set(q.stale_evidence_ids) <= set(when)
        if "update" in q.meta["phenomena"] and q.category == "knowledge_update":
            # the current value supersedes every stale statement of the same slot
            assert q.stale_evidence_ids and all((s, q.evidence_ids[-1]) in pos for s in q.stale_evidence_ids)
            n_stale += 1
        if "history" in q.meta["phenomena"]:  # "what was it originally?": the newer value is the stale one
            assert all((q.evidence_ids[0], s) in pos for s in q.stale_evidence_ids) and q.stale_evidence_ids
    assert n_stale > 5
    # labels do not change the conversations, and survive the native format
    back = load_jarvis(str(save(ds, tmp_path / "sb.json")))
    assert back.annotations["supersession"] == rows
    assert back.questions[0].stale_evidence_ids == ds.questions[0].stale_evidence_ids
    p = save_supersession_labels(ds, tmp_path / "labels")
    assert p.name == "statebench_dev.jsonl" and load_labels([tmp_path / "labels"]) == rows


def test_score_pairs_and_label_validation(tmp_path):
    labelled = {("a", "b"): "supersedes", ("b", "c"): "supersedes", ("a", "x"): "unrelated",
                ("b", "p"): "same_slot_no_change"}
    c = score_pairs({("a", "b"), ("a", "x"), ("y", "z")}, labelled)
    assert c == {"gold": 2, "negatives": 2, "predicted": 3, "tp": 1, "fp": 1, "predicted_unlabelled": 1}
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"dataset": "d", "conversation_id": "c", "older_id": "a", "newer_id": "b",
                               "label": "replaces"}) + "\n")
    with pytest.raises(ValueError, match="label must be one of"):
        load_labels([bad])
    bad.write_text(json.dumps({"dataset": "d", "older_id": "a", "newer_id": "b", "label": "supersedes"}) + "\n")
    with pytest.raises(ValueError, match="missing"):
        load_labels([bad])
    with pytest.raises(FileNotFoundError):
        load_labels([tmp_path / "nope.jsonl"])


def test_supersession_eval_cli(tmp_path):
    from jarvis_m.cli import main

    out = tmp_path / "data" / "sb.json"
    assert main(["data", "make-stress", "--split", "dev", "--n", "12", "--out", str(out)]) == 0
    labels = tmp_path / "data" / "labels" / "supersession" / "statebench_dev.jsonl"
    assert labels.exists()  # written automatically next to a non-default --out
    cfg = tmp_path / "sup.yaml"
    cfg.write_text(yaml.safe_dump({"memory": MEM, "labels": [str(labels.parent)], "datasets": [
        {"label": "sb", "domain": "team", "name": "jarvis", "path": str(out)}]}))
    md = tmp_path / "sup.md"
    assert main(["supersession-eval", str(cfg), "--out", str(md)]) == 0
    (r,) = json.loads(md.with_suffix(".json").read_text())["datasets"]
    assert r["conversations"] == 12 and r["gold"] > 0 and r["negatives"] > 0 and r["unknown_message_ids"] == 0
    assert 0.5 < r["precision"] <= 1 and 0.1 < r["recall"] <= 1 and 0 <= r["false_supersession_rate"] < 0.2
    assert r["tp"] <= r["predicted"] and "false-supersession rate" in md.read_text()
    # a hand-written label file for one conversation: only that conversation is scored
    hand = tmp_path / "hand.jsonl"
    first = json.loads(labels.read_text().splitlines()[0])
    hand.write_text(json.dumps(first) + "\n")
    res = supersession_eval(yaml.safe_load(cfg.read_text()), [str(hand)], progress=False)
    assert res["datasets"][0]["conversations"] == 1


def test_make_lodo(tmp_path):
    from jarvis_m.cli import main
    from jarvis_m.config import load_config

    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"memory": MEM}))
    train = tmp_path / "mix.yaml"
    train.write_text(yaml.safe_dump({
        "base": "base.yaml", "name": "mix", "output": "models/hgnn/mix.pt",
        "train_data": [{"name": "jarvis", "path": "a.json", "domain": "team", "weight": 2.0},
                       {"name": "longmemeval", "split": "train", "domain": "assistant"},
                       {"name": "locomo", "split": {"n_folds": 5, "fold": 0, "part": "train"}, "domain": "personal"}],
        "dev_data": [{"name": "jarvis", "path": "d.json", "domain": "team"},
                     {"name": "longmemeval", "split": "test", "domain": "assistant"}],
        "model": {"hidden": 8}}))
    out = tmp_path / "lodo"
    assert main(["make-lodo", str(train), "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"{pre}mix_lodo_{d}.yaml" for d in ("assistant", "personal", "team") for pre in ("", "rank_"))
    for dom in ("assistant", "personal", "team"):
        t = load_config(out / f"mix_lodo_{dom}.yaml")
        assert t["name"] == f"mix_lodo_{dom}" and t["output"] == f"models/hgnn/mix_lodo_{dom}.pt"
        assert t["memory"] == MEM and t["model"] == {"hidden": 8} and "base" not in t  # resolved, self-contained
        assert all(e["domain"] != dom for e in t["train_data"] + t["dev_data"]) and len(t["train_data"]) == 2
        r = load_config(out / f"rank_mix_lodo_{dom}.yaml")
        assert r["datasets"] and all(e["domain"] == dom and "weight" not in e for e in r["datasets"])
        assert r["arms"] == {f"mix_lodo_{dom}": [f"models/hgnn/mix_lodo_{dom}*.pt"]} and r["memory"] == MEM
    # held-out datasets taken from a rank-eval config instead
    ev = tmp_path / "rank.yaml"
    ev.write_text(yaml.safe_dump({"memory": MEM, "datasets": [
        {"label": "LoCoMo", "name": "locomo", "domain": "personal"},
        {"label": "SB test", "name": "jarvis", "path": "t.json", "domain": "team"}]}))
    assert main(["make-lodo", str(train), "--out", str(out), "--eval-config", str(ev)]) == 0
    assert [e["label"] for e in load_config(out / "rank_mix_lodo_team.yaml")["datasets"]] == ["SB test"]
    # every entry needs a domain, and one domain cannot be left out of itself
    train.write_text(yaml.safe_dump({"memory": MEM, "train_data": [{"name": "jarvis", "path": "a.json"}]}))
    with pytest.raises(ValueError, match="domain"):
        main(["make-lodo", str(train), "--out", str(out)])
    train.write_text(yaml.safe_dump({"memory": MEM, "train_data": [{"name": "jarvis", "path": "a.json", "domain": "team"}]}))
    with pytest.raises(ValueError, match="two domains"):
        main(["make-lodo", str(train), "--out", str(out)])
