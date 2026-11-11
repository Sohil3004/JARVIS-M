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


def test_extract_eval_fields_purity_and_spurious(tmp_path, write_json):
    from jarvis_m.eval.extract_eval import extract_eval, to_markdown

    conv = {"id": "c1", "messages": [
        {"id": "m0", "speaker": "Ana", "text": "I moved to Pune last month.", "channel": "dm", "timestamp": 1.0},
        {"id": "m1", "speaker": "Ben", "text": "lol nice", "channel": "dm", "timestamp": 2.0},
        {"id": "m2", "speaker": "Ana", "text": "Actually I relocated to Mumbai in May.", "channel": "dm", "timestamp": 3.0},
        {"id": "m3", "speaker": "Ana", "text": "I'll take the visa paperwork.", "channel": "dm", "timestamp": 4.0},
        {"id": "m4", "speaker": "Ben", "text": "how was the trip?", "channel": "dm", "timestamp": 5.0}]}
    p = write_json("p.json", {"name": "pers", "conversations": [conv], "questions": []})
    gold = [{"dataset": "pers", "conversation_id": "c1", "msg_id": "m0", "holder": "speaker:Ana", "subject": "home",
             "attribute": "location", "modality": "state", "value": "Pune"},
            {"dataset": "pers", "conversation_id": "c1", "msg_id": "m2", "holder": "speaker:Ana", "subject": "home",
             "attribute": "location", "modality": "state", "value": "Mumbai"},
            {"dataset": "pers", "conversation_id": "c1", "msg_id": "m3", "holder": "speaker:Ana",
             "subject": "visa paperwork", "attribute": "owner", "modality": "commitment", "value": "Ana"},
            {"dataset": "pers", "conversation_id": "c1", "msg_id": "m1", "no_record": True},
            {"dataset": "pers", "conversation_id": "c1", "msg_id": "m4", "no_record": True},
            {"dataset": "other", "conversation_id": "c1", "msg_id": "zz", "no_record": True}]
    lab = write_json("labels/pers.jsonl", gold, lines=True)
    res = extract_eval({"datasets": [{"name": "jarvis", "path": str(p), "domain": "personal"}]}, [str(lab)],
                       progress=False)
    (d,) = res["datasets"]
    assert d["gold_records"] == 3 and d["record_recall"] == 1.0 and d["unknown_message_ids"] == 0
    assert d["holder_accuracy"] == 1.0  # a DM: "I'll take ..." stays with the speaker
    assert d["attribute_accuracy"] == 1.0 and d["subject_completeness"] == 1.0 and d["spurious_rate"] == 0.0
    assert "subject purity" in to_markdown(res)
    with pytest.raises(ValueError, match="missing"):
        bad = write_json("bad.jsonl", [{"dataset": "x", "conversation_id": "c", "msg_id": "m"}], lines=True)
        extract_eval({"datasets": []}, [str(bad)], progress=False)
