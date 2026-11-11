import json

import pytest

from jarvis_m.data.loaders import load_jarvis
from jarvis_m.data.stress import SET_CHOICE, generate, save


def test_stress_is_deterministic_and_valid(tmp_path):
    a, b = generate(8, seed=3, split="train"), generate(8, seed=3, split="train")
    assert [q.question for q in a.questions] == [q.question for q in b.questions]
    ids = {m.id for c in a.conversations.values() for m in c.messages}
    for q in a.questions:
        assert set(q.evidence_ids) <= ids and q.question_time >= max(
            m.timestamp for m in a.conversations[q.conversation_id].messages)
        assert q.is_abstention == (q.category == "abstention") and q.meta["phenomena"]
    ds = load_jarvis(str(save(a, tmp_path / "a.json")))
    assert len(ds.questions) == len(a.questions) and ds.questions[0].stale_answers == a.questions[0].stale_answers


def test_test_split_uses_held_out_phrasings():
    held_out = {t.split("{")[0] for t in SET_CHOICE[2] if not t.startswith("{")}
    train_text = " ".join(m.text for c in generate(40, split="train").conversations.values() for m in c.messages)
    test_text = " ".join(m.text for c in generate(40, split="test").conversations.values() for m in c.messages)
    assert not any(h in train_text for h in held_out if h.strip())
    assert any(h in test_text for h in held_out if h.strip())


def test_phenomena_coverage():
    phen = {p for q in generate(40, split="dev").questions for p in q.meta["phenomena"]}
    assert {"update", "collision", "negation_trap", "proposal_trap", "intent_vs_done", "handoff",
            "user_implicit", "abstention", "history"} <= phen


def test_hgnn_trains_saves_and_scores(tmp_path):
    pytest.importorskip("torch")
    from jarvis_m.memory import build_memory
    from jarvis_m.memory.hypergraph.hgnn import HGNNScorer
    from jarvis_m.memory.hypergraph.train import train_hgnn

    tr, dv = save(generate(12, split="train"), tmp_path / "tr.json"), save(generate(4, split="dev"), tmp_path / "dv.json")
    mem = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}}
    ck = train_hgnn({"name": "t", "output": str(tmp_path / "h.pt"), "memory": mem,
                     "train_data": [{"name": "jarvis", "path": str(tr)}],
                     "dev_data": [{"name": "jarvis", "path": str(dv)}],
                     "model": {"hidden": 16, "layers": 1}, "train": {"epochs": 2, "device": "cpu", "eval_k": 5}},
                    progress=False)
    meta = json.loads(ck.with_suffix(".json").read_text())
    assert meta["history"] and meta["train_stats"]["candidate_recall"] > 0.9
    m = build_memory({**mem, "scorer": {"type": "hgnn", "checkpoint": str(ck), "device": "cpu"}})
    ds = load_jarvis(str(dv))
    cid = next(iter(ds.conversations))
    m.ingest(ds.conversations[cid].messages)
    q = ds.questions_for(cid)[0]
    r = m.retrieve(q.question, asker=q.asker, max_items=3)
    assert len(r.items) == 3
    bad = build_memory({"type": "hypergraph", "embedder": {"type": "hashing", "dim": 32},
                        "scorer": {"type": "hgnn", "checkpoint": str(ck), "device": "cpu"}})
    bad.ingest(ds.conversations[cid].messages)
    with pytest.raises(ValueError, match="emb_dim"):
        bad.retrieve(q.question, asker=q.asker)
    assert isinstance(HGNNScorer.load(str(ck), "cpu"), HGNNScorer)


def test_temperature_sampling_and_macro_dev():
    import random

    import numpy as np

    from jarvis_m.memory.hypergraph.train import Example, epoch_order, macro_dev, sampling_probs

    ex = [Example(None, None, None, np.array([1.0, 0.0]), f"a{i}", "c", dataset="big", domain="team") for i in range(90)]
    ex += [Example(None, None, None, np.array([0.0, 1.0]), f"b{i}", "c", dataset="small", domain="personal")
           for i in range(10)]
    p1 = sampling_probs(ex, {}, 1.0)
    assert p1["big"] == pytest.approx(0.9)
    p0 = sampling_probs(ex, {}, 0.0)
    assert p0["big"] == pytest.approx(0.5)
    ph = sampling_probs(ex, {}, 0.5)
    assert ph["big"] == pytest.approx(3 / 4)  # sqrt(90) : sqrt(10) = 3 : 1
    assert sampling_probs(ex, {"small": 9.0}, 1.0)["small"] == pytest.approx(0.5)  # weights multiply sizes
    order = epoch_order(ex, p0, random.Random(0))
    assert len(order) == 100 and 35 < sum(e.dataset == "small" for e in order) < 65
    m = macro_dev(ex, lambda e: np.array([1.0, 0.0]), 1)  # right on every "big" one, wrong on every "small" one
    assert m["mrr"] == pytest.approx(0.75) and m["domains"]["team"]["mrr"] == 1.0  # not 0.95 (pooled)
