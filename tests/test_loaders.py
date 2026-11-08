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
