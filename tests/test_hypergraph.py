import numpy as np
import pytest

from jarvis_m.data.stress import generate, save
from jarvis_m.extract import HeuristicExtractor
from jarvis_m.memory import build_memory
from jarvis_m.memory.hypergraph import Hypergraph
from jarvis_m.memory.hypergraph.subgraph import FEATURE_NAMES
from jarvis_m.types import Message

HG = {"type": "hypergraph", "embedder": {"type": "hashing"}}


def _graph(texts, **kw):
    msgs = [Message(id=f"m{i}", speaker=s, text=t, channel="eng", timestamp=float(i * 60), **kw)
            for i, (s, t) in enumerate(texts)]
    by = {}
    for r in HeuristicExtractor().extract(msgs):
        by.setdefault(r.msg_id, []).append(r)
    g = Hypergraph()
    g.add(msgs, by)
    return g


def test_version_chain_supersedes_and_negation_does_not():
    g = _graph([("A", "Let's go with AWS EC2 for the backend hosting."),
                ("B", "Ok scrap EC2, Cloud Run it is for the backend hosting."),
                ("C", "No, the backend hosting is not moving to Azure.")])
    assert g.superseded == [True, False, True]
    assert g.current_of(0) == 1 and g.current_of(2) == 1  # the rejected alternative points at the real value


def test_unrelated_topics_in_one_channel_do_not_collide():
    g = _graph([("A", "Deadline for the security audit: April 2."),
                ("B", "Target for the release date is Tuesday 5 PM."),
                ("C", "reminder: all-hands moved to the big room")])
    assert not any(g.superseded)


def test_done_resolves_blocker_but_intent_does_not():
    g = _graph([("A", "The staging deploy is failing with Error 403."),
                ("B", "I will fix the staging deploy tomorrow.")])
    assert g.modality[g.chains[g.chain_of[0]].latest] == "blocker" and not g.superseded[0]
    g2 = _graph([("A", "The staging deploy is failing with Error 403."),
                 ("B", "Fixed the staging deploy by rotating the credentials.")])
    assert g2.superseded[0] and not g2.superseded[1]


def test_reply_and_episode_edges():
    msgs = [Message("a", "A", "Can you take over load testing?", channel="x", timestamp=0.0),
            Message("b", "B", "Sure, mine now.", channel="x", timestamp=10.0, reply_to="a"),
            Message("c", "C", "much later", channel="x", timestamp=10 * 3600.0)]
    g = Hypergraph(episode_gap_minutes=60)
    g.add(msgs, {})
    types = {(e.type, tuple(e.nodes)) for e in g.edges}
    assert ("reply", (0, 1)) in types and ("episode", (0, 1)) in types and ("episode", (2,)) in types


def test_retrieval_on_toy(toy):
    mem = build_memory({**HG, "state_header": True})
    mem.ingest(toy.conversations["cloud"].messages)
    q = next(q for q in toy.questions if q.id == "cloud/q1")  # current platform (EC2 -> Cloud Run)
    r = mem.retrieve(q.question, asker=q.asker, max_items=2, budget_tokens=200, now=q.question_time)
    assert "cloud-m4" in r.source_ids and "cloud-m0" not in r.source_ids
    assert r.items[0].kind == "state" and r.n_tokens <= 200
    assert mem.last_subgraph.feats.shape[1] == len(FEATURE_NAMES)
    q4 = next(q for q in toy.questions if q.id == "cloud/q4")  # "What am I owning" asked by Dana
    assert "cloud-m10" in mem.retrieve(q4.question, asker=q4.asker, max_items=2).source_ids


def test_historical_query_is_not_penalised(toy):
    mem = build_memory(HG)
    mem.ingest(toy.conversations["cloud"].messages)
    r = mem.retrieve("Which platform did we originally pick for the backend deployment?", max_items=2)
    assert "cloud-m0" in r.source_ids
    assert mem.last_subgraph.historical


@pytest.mark.parametrize("scorer", [{"type": "fixed"}, {"type": "seeds"}, {"type": "fixed", "edge_weights": {}}])
def test_scorer_variants_and_edge_ablation(scorer, toy):
    for types in (None, ["reply", "episode", "entity"]):
        cfg = {**HG, "scorer": scorer, **({"edge_types": types} if types else {})}
        mem = build_memory(cfg)
        mem.ingest(toy.conversations["fintech"].messages)
        r = mem.retrieve("Which database stores the transaction records?", max_items=3)
        assert r.items and np.isfinite([it.score for it in r.items]).all()
        if types:
            assert mem.graph.stats().get("edges_version", 0) == 0


def test_hypergraph_beats_hybrid_on_statebench_updates(tmp_path):
    """On knowledge-update questions the current value should rank above the stale one
    more often with version-aware hypergraph scoring than with flat hybrid retrieval."""
    ds = generate(25, split="dev")
    wins = {"hybrid": 0, "hypergraph": 0}
    for name, cfg in (("hybrid", {"type": "hybrid", "embedder": {"type": "hashing"}}), ("hypergraph", HG)):
        for cid, conv in ds.conversations.items():
            mem = build_memory(cfg)
            mem.ingest(conv.messages)
            for q in ds.questions_for(cid):
                if q.category != "knowledge_update" or "update" not in q.meta["phenomena"]:
                    continue
                r = mem.retrieve(q.question, asker=q.asker, max_items=1, now=q.question_time)
                wins[name] += int(bool(set(r.source_ids) & set(q.evidence_ids)))
    assert wins["hypergraph"] > wins["hybrid"], wins
    save(ds, tmp_path / "sb.json")  # round-trips through the native loader


def test_cli_make_stress_extract_and_hypergraph_smoke(tmp_path, monkeypatch):
    import json as _json

    import yaml

    from jarvis_m.cli import main
    from jarvis_m.config import load_config
    from tests.conftest import ROOT, TOY

    out = tmp_path / "sb.json"
    assert main(["data", "make-stress", "--split", "dev", "--n", "3", "--out", str(out)]) == 0
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"dataset": {"name": "jarvis", "path": str(out)}}))
    ex = tmp_path / "ex.yaml"
    ex.write_text("name: heur\nextractor: {type: heuristic}\n")
    cache = tmp_path / "cache.jsonl"
    assert main(["extract", str(exp), "--extractor", str(ex), "--out", str(cache)]) == 0
    n_msgs = sum(len(c["messages"]) for c in _json.loads(out.read_text())["conversations"])
    assert len(cache.read_text().splitlines()) == n_msgs
    assert main(["extract", str(exp), "--extractor", str(ex), "--out", str(cache)]) == 0  # resume: no-op
    assert len(cache.read_text().splitlines()) == n_msgs

    cfg = load_config(ROOT / "configs" / "exp" / "smoke_hypergraph.yaml")
    cfg.update(output_dir=str(tmp_path / "runs"))
    cfg["dataset"]["path"] = str(TOY)
    cfg["arms"].append({"name": "hg_cached", "set": {"memory.extractor": {"type": "cached", "path": str(cache)}}})
    p = tmp_path / "smoke.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert main(["run", str(p)]) == 0
    assert main(["report", str(tmp_path / "runs"), "--baseline", "hybrid"]) == 0


def test_missing_artifacts_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="extract"):
        build_memory({**HG, "extractor": {"type": "cached", "path": str(tmp_path / "nope.jsonl")}})
    pytest.importorskip("torch")
    with pytest.raises(FileNotFoundError, match="train-hgnn"):
        build_memory({**HG, "scorer": {"type": "hgnn", "checkpoint": str(tmp_path / "nope.pt")}})


def _graph_msgs(turns, **gkw):
    """turns: (speaker, text) or (speaker, text, session); one channel, like LoCoMo / LongMemEval."""
    msgs = [Message(id=f"m{i}", speaker=t[0], text=t[1], timestamp=float(i * 60),
                    session_id=t[2] if len(t) > 2 else "s1") for i, t in enumerate(turns)]
    by = {}
    for r in HeuristicExtractor().extract(msgs):
        by.setdefault(r.msg_id, []).append(r)
    g = Hypergraph(**gkw)
    g.add(msgs, by)
    return g


def test_speaker_names_frequent_entities_and_dates_are_not_linking_evidence():
    # two unrelated decisions whose only shared entity token is the addressee's name
    g = _graph_msgs([("Melanie", "How was your week?"),
                     ("Caroline", "Great, thanks Melanie! I decided to adopt a puppy from the shelter."),
                     ("Melanie", "So cool!"),
                     ("Caroline", "Thanks Melanie, I picked the blue paint for my art show.")])
    assert g.chain_of[1] >= 0 and g.chain_of[3] >= 0 and not any(g.superseded)
    assert not any(e.type == "entity" and e.key == "melanie" for e in g.edges)
    # a date shared by two unrelated decisions is a value, not a topic
    g = _graph_msgs([("A", "We decided to visit Grandma on May 5."),
                     ("B", "I chose the blue dress for the party on May 5.")])
    assert not any(g.superseded)
    # an entity in most messages links nothing once it is frequent
    filler = [("A", f"note {i} about Grafana and the Kubernetes dashboard") for i in range(8)]
    g = _graph_msgs(filler + [("A", "We decided to go with Grafana Loki for logs."),
                              ("B", "Agreed: picked Grafana Cloud for the billing alerts.")])
    assert g.entity_df["grafana"] >= 5 and g.token_df["grafana"] >= 5 and not any(g.superseded)
    s = g.stats()
    assert s["superseded_fraction"] == 0.0 and s["chains_per_100_messages"] > 0


def test_assistant_suggestion_does_not_supersede_user_decision():
    turns = [("user", "We decided to go with PostgreSQL for the billing database."),
             ("assistant", "Actually, let's switch the billing database to MongoDB instead."),
             ("user", "Thanks, what about backups?")]
    g = _graph_msgs(turns)
    assert not g.superseded[0] and g.chain_of[1] == -1
    assert any(e.type == "entity" and 1 in e.nodes for e in g.edges)  # still a node with entity edges
    everyone = _graph_msgs(turns, state_roles=None)  # old behaviour: anyone changes state
    assert everyone.superseded[0] and everyone.current_of(0) == 1


def test_confirmed_assistant_suggestion_becomes_current():
    turns = [("user", "We decided to go with PostgreSQL for the billing database."),
             ("assistant", "Actually, let's switch the billing database to MongoDB instead."),
             ("user", "Sounds good!")]
    g = _graph_msgs(turns)
    assert g.superseded[0] and g.current_of(0) == 1 and g.chain_of[2] == -1
    assert not _graph_msgs(turns, assistant_confirmation=False).superseded[0]
    late = turns[:2] + [("user", "hmm"), ("user", "something else entirely"), ("user", "sounds good")]
    assert not _graph_msgs(late).superseded[0]  # outside the 2-message window
    other = turns[:2] + [("user", "Sounds good!", "s2")]
    assert not _graph_msgs(other).superseded[0]  # confirmation must come in the same session
