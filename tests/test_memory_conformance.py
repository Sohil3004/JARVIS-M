"""Every registered memory system must honour the MemorySystem contract."""

import pytest

from jarvis_m.memory import build_memory
from jarvis_m.memory.legacy_v2 import _legacy_core_dir

SYSTEMS = {
    "recency": {"type": "recency"},
    "full_context": {"type": "full_context"},
    "bm25": {"type": "bm25"},
    "dense": {"type": "dense", "embedder": {"type": "hashing"}},
    "hybrid": {"type": "hybrid", "embedder": {"type": "hashing"}},
    "legacy_v2": {"type": "legacy_v2"},
    "hypergraph": {"type": "hypergraph", "embedder": {"type": "hashing"}},
    "hypergraph_header": {"type": "hypergraph", "embedder": {"type": "hashing"}, "state_header": True},
}


def _build(name):
    if name == "legacy_v2" and not (_legacy_core_dir() / "reconciliation_engine.py").exists():
        pytest.skip("legacy v2 code not present")
    return build_memory(SYSTEMS[name])


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_contract(name, toy):
    mem = _build(name)
    conv = toy.conversations["cloud"]
    mem.reset()
    mem.ingest(conv.messages[:5])
    mem.ingest(conv.messages[5:])  # incremental ingest must work
    r = mem.retrieve("What is the release deadline?", asker="Bob", budget_tokens=80, max_items=3)
    assert r.context and r.items
    ids = {m.id for m in conv.messages}
    assert set(r.source_ids) <= ids
    if name != "full_context":
        assert r.n_tokens <= 80 + 5  # header/separators may add a few tokens
    mem.reset()
    r2 = mem.retrieve("anything", asker="Bob", budget_tokens=80)
    assert not any(it.kind == "message" for it in r2.items)


@pytest.mark.parametrize("name", ["bm25", "dense", "hybrid", "hypergraph"])
def test_ranked_systems_find_keyword_evidence(name, toy):
    mem = _build(name)
    mem.ingest(toy.conversations["fintech"].messages)
    r = mem.retrieve("Which tool did the team pick for dashboards?", asker="Elena", max_items=2)
    assert "fintech-m7" in r.source_ids  # the Grafana message


@pytest.mark.parametrize("scope,expect", [("asker", {"Dana"}), ("others", {"Alice", "Bob", "Charlie"})])
def test_scope_filter(scope, expect, toy):
    mem = build_memory({"type": "bm25", "scope": scope})
    mem.ingest(toy.conversations["cloud"].messages)
    r = mem.retrieve("load testing release lunch registry", asker="Dana", max_items=20)
    speakers = {next(m for m in toy.conversations["cloud"].messages if m.id == i).speaker for i in r.source_ids}
    assert speakers and speakers <= expect


def test_chronological_order_in_context(toy):
    mem = build_memory({"type": "bm25"})
    mem.ingest(toy.conversations["cloud"].messages)
    r = mem.retrieve("EC2 Cloud Run deadline", max_items=5)
    ts = [it.timestamp for it in r.items]
    assert ts == sorted(ts)


def test_recency_half_life_prefers_newer(toy):
    msgs = toy.conversations["fintech"].messages
    now = msgs[-1].timestamp
    plain = build_memory({"type": "bm25"})
    decay = build_memory({"type": "bm25", "recency_half_life_days": 0.5, "order": "score"})
    for m in (plain, decay):
        m.ingest(msgs)
    q = "transaction records database MongoDB PostgreSQL"
    top_plain = plain.rank(q, now=now)[0][0]
    top_decay = decay.rank(q, now=now)[0][0]
    assert msgs[top_decay].timestamp >= msgs[top_plain].timestamp


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        build_memory({"type": "nope"})
