import pytest

from jarvis_m.data.loaders import load_dataset, load_groupmembench, load_locomo, load_longmemeval


def test_toy_fixture(toy):
    s = toy.summary()
    assert s["conversations"] == 2 and s["questions"] == 8
    q = next(q for q in toy.questions if q.id == "cloud/q1")
    assert q.stale_answers and q.evidence_ids == ("cloud-m4",)
    assert toy.filter(categories=["abstention"]).questions[0].is_abstention


def test_groupmembench_hides_labels_and_sorts_by_time(tmp_path, write_json):
    conv = {
        "chan-b": [{"msg_node": "Msg_2", "content": "later &amp; decided", "author": "User_2", "role": "PM",
                    "timestamp": "2025-07-19T10:00:00", "reply_to": None, "phase_name": "P1", "topic": "T",
                    "is_noise": False, "is_decision_point": True, "decision_label": "SECRET"}],
        "chan-a": [{"msg_node": "Msg_1", "content": "first", "author": "User_1", "role": "Eng",
                    "timestamp": "2025-07-19T09:00:00", "reply_to": None, "phase_name": "P1", "topic": "T",
                    "is_noise": True, "is_decision_point": False}],
    }
    write_json("gmb/data/final/Finance/synthetic_domain_channels_rolevariants_Finance.json", conv)
    write_json("gmb/questions/Finance/abstention.jsonl",
               [{"id": "abstention_1", "question": "q?", "answer": "none", "asking_user_id": "User_1"}], lines=True)
    ds = load_groupmembench(root=str(tmp_path / "gmb"), domains=["Finance"])
    msgs = ds.conversations["gmb-Finance"].messages
    assert [m.id for m in msgs] == ["Msg_1", "Msg_2"]  # chronological across channels
    for m in msgs:
        assert "is_noise" not in m.meta and "decision_label" not in m.meta  # no leakage to systems
        assert m.meta["phase_name"] == "P1"
    assert ds.annotations["Msg_2"]["decision_label"] == "SECRET"
    q = ds.questions[0]
    assert q.is_abstention and q.asker == "User_1" and q.question_time == msgs[-1].timestamp


def test_locomo(tmp_path, write_json):
    sample = [{
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "hi"},
                          {"speaker": "B", "dia_id": "D1:2", "text": "went hiking", "blip_caption": "a trail"}],
        },
        "qa": [{"question": "What did B do?", "answer": "hiking", "evidence": ["D1:2"], "category": 4},
               {"question": "Trap?", "evidence": ["D1:1"], "category": 5, "adversarial_answer": "x"}],
    }]
    p = write_json("locomo10.json", sample)
    ds = load_locomo(str(p))
    m = ds.conversations["conv-1"].messages
    assert m[1].timestamp == m[0].timestamp + 1 and "a trail" in m[1].text
    assert ds.questions[0].category == "single_hop" and ds.questions[0].evidence_ids == ("D1:2",)
    assert ds.questions[1].is_abstention
    assert len(load_locomo(str(p), categories=["single_hop"]).questions) == 1


def test_longmemeval(tmp_path, write_json):
    e = [{
        "question_id": "q1_abs", "question_type": "knowledge-update", "question": "Where?", "answer": "Paris",
        "question_date": "2023/05/30 (Tue) 10:00", "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"], "answer_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I moved to Paris", "has_answer": True},
                               {"role": "assistant", "content": "Nice"}]],
    }]
    ds = load_longmemeval(str(write_json("lme.json", e)))
    q = ds.questions[0]
    assert q.evidence_ids == ("s1#0",) and q.is_abstention and q.asker == "user"
    assert q.question_time > ds.conversations["q1_abs"].messages[-1].timestamp


def test_longmemeval_stratified_split(write_json):
    types = ["multi-session", "knowledge-update", "single-session-assistant", "single-session-user", "temporal-reasoning"]
    sess = [[{"role": "user", "content": "I moved to Paris", "has_answer": True}, {"role": "assistant", "content": "Nice"}]]
    # sorted by type, like the real file: a `limit:` would never reach the later types
    raw = [{"question_id": f"{t}-{i}", "question_type": t, "question": "Where?", "answer": "Paris",
            "question_date": "2023/05/30 (Tue) 10:00", "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"], "haystack_sessions": sess}
           for t in types for i in range(7 + 3 * types.index(t))]
    p = str(write_json("lme.json", raw))
    tr, te = load_longmemeval(p, split="train"), load_longmemeval(p, split="test")
    tr_ids, te_ids = {q.id for q in tr.questions}, {q.id for q in te.questions}
    assert not tr_ids & te_ids and tr_ids | te_ids == {e["question_id"] for e in raw}
    for t in types:
        n = sum(e["question_type"] == t for e in raw)
        assert {q.category for q in tr.questions} >= {t} and {q.category for q in te.questions} >= {t}
        assert sum(q.category == t for q in te.questions) == -(-n // 5)  # ceil(0.2 * n)
    assert te_ids == {q.id for q in load_longmemeval(p, split="test").questions}  # deterministic
    assert te_ids != {q.id for q in load_longmemeval(p, split="test", split_seed=1).questions}
    # a category filter does not move questions across the split
    ku = load_longmemeval(p, split="test", categories=["knowledge-update"])
    assert {q.id for q in ku.questions} == {i for i in te_ids if i.startswith("knowledge-update")}
    assert set(te.conversations) == te_ids and len(load_longmemeval(p).questions) == len(raw)


def test_load_dataset_limit(toy, tmp_path):
    from tests.conftest import TOY

    ds = load_dataset({"name": "jarvis", "path": str(TOY), "limit": 3})
    assert len(ds.questions) == 3


def test_disk_embedding_cache(tmp_path):
    import numpy as np

    from jarvis_m.embeddings import CachedEmbedder, HashingEmbedder, build_embedder

    calls = []

    class Counting(HashingEmbedder):
        def encode_docs(self, texts):
            calls.append(len(texts))
            return super().encode_docs(texts)

    base = Counting(64)
    c = CachedEmbedder(base, tmp_path / "e.sqlite", "hash64")
    a = c.encode_docs(["alpha beta", "gamma"])
    b = c.encode_docs(["gamma", "alpha beta", "delta"])
    assert calls == [2, 1] and np.allclose(a[0], b[1], atol=1e-3) and c.hits == 2
    c2 = CachedEmbedder(Counting(64), tmp_path / "e.sqlite", "hash64")  # new process: reads from disk
    c2.encode_docs(["alpha beta", "gamma", "delta"])
    assert c2.misses == 0
    assert np.allclose(c.encode_queries(["q"]), c.encode_queries(["q"]))
    emb = build_embedder({"type": "hashing", "dim": 32, "disk_cache": str(tmp_path / "h.sqlite")})
    assert isinstance(emb, CachedEmbedder) and emb.encode_docs(["x y"]).shape == (1, 32)


def test_kfold_splits_hold_out_conversations(write_json):
    from jarvis_m.data.loaders import kfold_test_ids, parse_split

    def sample(i):
        return {"sample_id": f"conv-{i}", "conversation": {
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": "A", "dia_id": f"D{i}:1", "text": "hi"}]},
            "qa": [{"question": "q?", "answer": "a", "evidence": [f"D{i}:1"], "category": 4}] * 3}

    p = str(write_json("locomo10.json", [sample(i) for i in range(10)]))
    seen = []
    for fold in range(5):
        sp = {"n_folds": 5, "fold": fold, "seed": 0}
        te, tr = load_locomo(p, split=sp), load_locomo(p, split={**sp, "part": "train"})
        assert len(te.conversations) == 2 and len(tr.conversations) == 8
        assert not set(te.conversations) & set(tr.conversations)
        assert {q.conversation_id for q in te.questions} == set(te.conversations) and len(te.questions) == 6
        seen += list(te.conversations)
    assert sorted(seen) == sorted(f"conv-{i}" for i in range(10))  # every conversation is held out exactly once
    other = load_locomo(p, split={"n_folds": 5, "fold": 0, "seed": 1})
    assert set(other.conversations) != set(load_locomo(p, split={"n_folds": 5, "fold": 0}).conversations)
    with pytest.raises(ValueError, match="no fixed"):
        load_locomo(p, split="test")
    for bad in ({"n_folds": 5}, {"n_folds": 5, "fold": 5}, {"n_folds": 1, "fold": 0}, "dev",
                {"n_folds": 5, "fold": 0, "part": "dev"}, {"n_folds": 5, "fold": 0, "folds": 2}):
        with pytest.raises(ValueError):
            parse_split(bad)
    assert kfold_test_ids({"a": ["x", "y", "z"]}, 3, 0) | kfold_test_ids({"a": ["x", "y", "z"]}, 3, 1) \
        | kfold_test_ids({"a": ["x", "y", "z"]}, 3, 2) == {"x", "y", "z"}


def test_longmemeval_kfold_is_stratified_and_keeps_the_fixed_split(write_json):
    types = ["multi-session", "knowledge-update", "single-session-assistant"]
    sess = [[{"role": "user", "content": "I moved to Paris", "has_answer": True}]]
    raw = [{"question_id": f"{t}-{i}", "question_type": t, "question": "Where?", "answer": "Paris",
            "question_date": "2023/05/30 (Tue) 10:00", "haystack_session_ids": ["s1"],
            "haystack_dates": ["2023/05/20 (Sat) 02:21"], "haystack_sessions": sess}
           for t in types for i in range(10)]
    p = str(write_json("lme.json", raw))
    union = set()
    for fold in range(5):
        te = load_longmemeval(p, split={"n_folds": 5, "fold": fold})
        tr = load_longmemeval(p, split={"n_folds": 5, "fold": fold, "part": "train"})
        ids = {q.id for q in te.questions}
        assert all(sum(q.category == t for q in te.questions) == 2 for t in types)  # every type in every fold
        assert not ids & {q.id for q in tr.questions} and len(tr.questions) == 24 and not ids & union
        union |= ids
    assert union == {e["question_id"] for e in raw}
    ku = load_longmemeval(p, split={"n_folds": 5, "fold": 0}, categories=["knowledge-update"])
    full = load_longmemeval(p, split={"n_folds": 5, "fold": 0})
    assert {q.id for q in ku.questions} == {q.id for q in full.questions if q.category == "knowledge-update"}
    assert len(load_longmemeval(p, split="test").questions) == 6  # the fixed hold-out still works


def test_dataset_domain_registry(tmp_path):
    from jarvis_m.data.loaders import DOMAINS, dataset_domain
    from tests.conftest import ROOT, TOY

    ds = load_dataset({"name": "jarvis", "path": str(TOY), "domain": "team", "label": "toy", "weight": 2.0})
    assert ds.domain == "team" and ds.filter(limit=1).domain == "team"
    assert load_dataset({"name": "jarvis", "path": str(TOY)}).domain is None
    with pytest.raises(ValueError, match="unknown domain"):
        load_dataset({"name": "jarvis", "path": str(TOY), "domain": "work"})
    with pytest.raises(ValueError, match="needs `domain:`"):
        dataset_domain({"name": "jarvis"}, required=True)
    # every dataset entry in the shipped eval / train / experiment configs names a known domain
    from jarvis_m.config import load_config

    n = 0
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        cfg = load_config(path)
        entries = [*(cfg.get("datasets") or []), *(cfg.get("train_data") or []), *(cfg.get("dev_data") or [])]
        if isinstance(cfg.get("dataset"), dict):
            entries.append(cfg["dataset"])
        for e in entries:
            assert dataset_domain(e, required=True) in DOMAINS, path
            n += 1
    assert n >= 10
