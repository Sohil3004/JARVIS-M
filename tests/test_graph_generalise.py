"""Phase B: holders, visibility-scoped chains, order-independent ingest, windowed linking
evidence, edge-size consistency, mention edges and the size-invariant feature set."""

import random

import numpy as np
import pytest

from jarvis_m.data.stress import generate
from jarvis_m.extract import HeuristicExtractor, StateRecord
from jarvis_m.extract.heuristic import holder_of, named_holders
from jarvis_m.extract.records import holder_key
from jarvis_m.memory import build_memory
from jarvis_m.memory.hypergraph import EDGE_TYPES, Hypergraph
from jarvis_m.memory.hypergraph.subgraph import FEATURE_NAMES, FI, build_subgraph
from jarvis_m.types import Message

HG = {"type": "hypergraph", "embedder": {"type": "hashing"}}


def _records(msgs):
    by = {}
    for r in HeuristicExtractor().extract(msgs):
        by.setdefault(r.msg_id, []).append(r)
    return by


def _msgs(turns, channel="eng"):
    """turns: (speaker, text[, meta]) one minute apart."""
    return [Message(id=f"m{i}", speaker=t[0], text=t[1], channel=channel, timestamp=float(i * 60),
                    meta=t[2] if len(t) > 2 else {}) for i, t in enumerate(turns)]


def _graph(turns, order=None, **gkw):
    msgs = _msgs(turns)
    g = Hypergraph(**gkw)
    g.add([msgs[i] for i in order] if order else msgs, _records(msgs))
    return g


def _state(g):
    """Graph state by message id (node ids depend on ingest order)."""
    mid = lambda v: g.messages[v].id  # noqa: E731
    view = g.view()
    slots = sorted((s.key, tuple(mid(v) for v, _ in s.records), tuple(sorted(mid(v) for v in s.rejected)))
                   for s in view.slots)
    flags = {mid(v): (bool(view.superseded[v]), bool(view.current[v])) for v in range(len(g))}
    return slots, flags


# ---------------------------------------------------------------- B1 holders
@pytest.mark.parametrize("text,attribute,holder", [
    ("I'm vegetarian now.", "preference", ("speaker:Ana", False)),
    ("My laptop is a ThinkPad.", "possession", ("speaker:Ana", False)),
    ("We decided to use Kafka for the queue.", "decision", ("group:eng", False)),
    ("I think our release date should be Friday.", "deadline", ("group:eng", False)),
    ("The release is due Friday.", "deadline", ("group:eng", False)),
    # work state in the first person: the group's in a group chat, the speaker's in a DM (graph decides)
    ("I'll take the migration script.", "owner", ("speaker:Ana", True)),
    ("I fixed the staging deploy.", "status", ("speaker:Ana", True)),
    ("My visa got approved.", "status", ("speaker:Ana", False)),  # a personal statement
])
def test_heuristic_holder_rules(text, attribute, holder):
    assert holder_of(text, "Ana", "eng", attribute) == holder


def test_holder_keys_and_named_candidates():
    assert holder_key("speaker:Priya") == holder_key("person:priya") == "p:priya"
    assert holder_key("group:Eng") == "group:eng" and holder_key("") == ""
    r = StateRecord("m1", "A", "decision", "preference", "tea", "drink", holder="speaker:Ana")
    assert r.key == "p:ana|drink|preference" and r.with_holder("group:x").key == "group:x|drink|preference"
    assert StateRecord.from_json(r.to_json()) == r
    assert StateRecord.from_json({k: v for k, v in r.to_json().items() if k != "holder"}).holder == ""  # old caches
    assert named_holders("Priya's laptop is a ThinkPad. Kafka is not happening.") == ("Priya", "Kafka")
    assert named_holders("Priya's laptop", speaker="Priya") == ()
    (rec,) = HeuristicExtractor().extract_one(Message("m", "Ana", "We decided Priya's desk moves to floor 3.", channel="eng"))
    assert rec.holder == "group:eng" and rec.meta == {"named": ["Priya"]}


def test_slots_never_chain_across_holders():
    turns = [("Ana", "I decided to go with tea for the morning drink."),
             ("Ben", "I decided to go with juice for the morning drink."),
             ("Ana", "Actually I switched to coffee for the morning drink.")]
    g = _graph(turns)
    slots, flags = _state(g)
    assert [s[1] for s in slots] == [("m0", "m2"), ("m1",)]  # Ana's slot updates; Ben's is untouched
    assert flags["m0"] == (True, False) and flags["m1"] == (False, True) and flags["m2"] == (False, True)
    assert g.superseding_pairs() == {("m0", "m2")}
    assert {s[0].split("|")[0] for s in slots} == {"p:ana", "p:ben"}


def test_named_participant_becomes_the_holder_but_tools_do_not():
    turns = [("Priya", "morning all"),
             ("Ana", "We decided Priya's desk is on floor 2 for the office move."),
             ("Ben", "We decided Priya's desk is on floor 5 for the office move."),
             ("Ana", "Kafka is not happening for the message queue."),
             ("Ben", "We decided the message queue is Kafka after all.")]
    g = _graph(turns)
    keys = {s.key.split("|")[0] for s in g.chains}
    assert "p:priya" in keys and "p:kafka" not in keys
    assert g.superseding_pairs() == {("m1", "m2")}
    # before Priya has spoken she is not a known participant: the name is not a holder
    late = _graph(turns[1:3])
    assert {s.key.split("|")[0] for s in late.chains} == {"group:eng"}


HAND_OFF = [("Ana", "I'll take the migration script."),
            ("Ana", "Ben, can you take over the migration script from me?"),
            ("Ben", "Sure, I'll own the migration script from now on.")]


def test_owner_hand_off_supersedes_in_a_group_chat():
    g = _graph([*HAND_OFF, ("Cy", "thanks both"), ("Dee", "noted")])  # a 4-person channel
    assert g.superseded[0] and not g.superseded[2] and g.current_of(0) == 2
    assert {s.key.split("|")[0] for s in g.chains} == {"group:eng"}


def test_first_person_work_state_is_the_speakers_in_a_dm():
    g = _graph(HAND_OFF)  # two people: a DM, each owns their own task
    assert not g.superseded[0] and g.current_of(0) == 0 and g.current_of(2) == 2
    assert {s.key.split("|")[0] for s in g.chains} == {"p:ana", "p:ben"}
    # the third participant arrives later (in time or in ingest order): the chat is a group chat now
    for late in (_graph([*HAND_OFF, ("Cy", "ok")]), _graph([*HAND_OFF, ("Cy", "ok")], order=[3, 0, 1, 2])):
        assert late.current_of(late.id2node["m0"]) == late.id2node["m2"]
    # a directory decides the size when it knows the channel
    assert _graph(HAND_OFF, channel_size=lambda c: 5).current_of(0) == 2


def test_personal_statements_stay_with_the_speaker_in_a_group_chat():
    g = _graph([("Ana", "I moved to Pune last month."), ("Ben", "I live in Berlin."), ("Cy", "nice"),
                ("Ana", "Actually I relocated to Mumbai in May."), ("Ana", "My visa got approved!")])
    assert g.current_of(0) == 3 and g.current_of(1) == 1  # Ana's home changed; Ben's is his own
    keys = {s.key for s in g.chains}
    assert "p:ana|home|location" in keys and "p:ben|home|location" in keys
    assert "p:ana|eng/visa|status" in keys  # "my visa": Ana's, not the group's


# ---------------------------------------------------------------- B2 scope-aware chains
def test_chains_hold_one_visibility_class_and_views_merge_them():
    secret = {"acl_level": 2}
    turns = [("Ana", "We decided to use AWS EC2 for the backend hosting."),
             ("Ben", "Update: we switched the backend hosting to Google Cloud Run instead.", secret),
             ("Ana", "Change of plan: we switched the backend hosting to Fly.io instead."),
             ("Ben", "Update: we switched the backend hosting to Render instead.", secret)]
    g = _graph(turns)
    assert len(g.classes) == 2
    assert sorted([g.messages[v].id for v, _ in ch.records] for ch in g.class_chains) == [["m0", "m2"], ["m1", "m3"]]
    assert len({ch.slot for ch in g.class_chains}) == 1  # one slot, two chains
    everyone = g.view()
    assert [g.messages[v].id for v, _ in everyone.slots[0].records] == ["m0", "m1", "m2", "m3"]
    assert everyone.current_of(0) == 3 and list(everyone.superseded) == [True, True, True, False]
    low = g.view(np.array([g.labels[v].level == 0 for v in range(4)]))
    assert [g.messages[v].id for v, _ in low.slots[0].records] == ["m0", "m2"]
    assert low.current_of(0) == 2 and not low.superseded[2] and low.superseded[0]  # latest *visible* version
    assert not low.node_edges[1] and not low.node_edges[3] and low.chain_of[3] == -1
    assert all(set(e.nodes) <= {0, 2} for e in low.edges)
    assert g.superseding_pairs(low) == {("m0", "m2")}


def test_restricted_statement_joins_a_readable_slot_by_entity_without_changing_it():
    secret = {"acl_level": 2}
    turns = [("Ana", "Let's go with AWS EC2 for the backend hosting."),
             ("Ben", "Ok scrap EC2, Cloud Run it is.", secret)]  # no topic phrase: links on the entity
    g = _graph(turns)
    low_only = _graph(turns[:1])
    assert _state(low_only)[0] == sorted((s.key, tuple(g.messages[v].id for v, _ in s.records), ())
                                         for s in g.view(np.array([True, False])).slots)
    assert g.current_of(0) == 1 and g.superseded[0]  # a cleared viewer sees the update
    assert [len(ch.records) for ch in g.class_chains] == [1, 1]  # the public chain was not written to
    # the other way round nothing links: a public statement cannot read a restricted slot
    g2 = _graph([(turns[0][0], turns[0][1], secret), (turns[1][0], turns[1][1])])
    assert not any(g2.superseded)


def test_reply_inherits_the_thread_label_even_when_the_parent_arrives_late():
    msgs = [Message("a", "Ana", "We decided to use Kafka for the queue.", channel="eng", timestamp=0.0,
                    meta={"acl_level": 3, "acl_compartments": ["deal"]}),
            Message("b", "Ben", "Agreed, Kafka for the queue.", channel="eng", timestamp=60.0, reply_to="a"),
            Message("c", "Cy", "noted", channel="eng", timestamp=120.0, reply_to="b"),
            Message("x", "Cy", "elsewhere", channel="ops", timestamp=130.0, reply_to="a")]
    for order in ([0, 1, 2, 3], [2, 1, 3, 0], [1, 0, 3, 2]):
        g = Hypergraph()
        g.add([msgs[i] for i in order], _records(msgs))
        lab = {g.messages[v].id: g.labels[v] for v in range(4)}
        assert lab["b"] == lab["c"] == lab["a"] == (3, frozenset({"deal"})), order
        assert lab["x"] == (0, frozenset())  # threads live inside one channel
        threads = {tuple(sorted(g.messages[v].id for v in e.nodes)) for e in g.edges if e.type == "reply"}
        assert threads == {("a", "b", "c")}, order


# ---------------------------------------------------------------- B3 timestamp ordering
def test_shuffled_ingest_gives_the_same_chains():
    turns = [("Ana", "Let's go with AWS EC2 for the backend hosting."),
             ("Ben", "lunch at 1?"),
             ("Ben", "Ok scrap EC2, Cloud Run it is for the backend hosting."),
             ("Cy", "The staging deploy is failing with Error 403."),
             ("Ana", "No, the backend hosting is not moving to Azure."),
             ("assistant", "Actually, let's switch the backend hosting to Fly.io instead."),
             ("Cy", "sounds good"),
             ("Ben", "Fixed the staging deploy by rotating the credentials."),
             ("Ana", "Deadline for the security audit: April 2."),
             ("Ana", "The security audit has been rescheduled to May 20."),
             ("Ben", "Scrap Fly.io, Render it is.")]
    ordered = _graph(turns)
    want = _state(ordered)
    assert ordered.replays == 0 and len(want[0]) >= 3 and sum(s for s, _ in want[1].values()) >= 4
    rng = random.Random(0)
    for _ in range(8):
        order = list(range(len(turns)))
        rng.shuffle(order)
        g = _graph(turns, order=order)
        assert _state(g) == want, order
        assert g.replays == 1  # one replay, on first use
    # late messages after a query: replayed again, same result
    g = Hypergraph()
    msgs = _msgs(turns)
    recs = _records(msgs)
    g.add(msgs[:3] + msgs[6:], recs)
    assert g.view() is g.view() and g.stats()["nodes"] == 8
    g.add(msgs[3:6], recs)
    assert _state(g) == want and g.replays == 1


def test_shuffled_ingest_on_statebench():
    ds = generate(6, split="dev")
    for conv in ds.conversations.values():
        recs = _records(conv.messages)
        a, b = Hypergraph(), Hypergraph()
        a.add(conv.messages, recs)
        shuffled = list(conv.messages)
        random.Random(1).shuffle(shuffled)
        b.add(shuffled, recs)
        assert _state(a) == _state(b) and a.superseding_pairs() == b.superseding_pairs()


# ---------------------------------------------------------------- B4 windowed linking evidence
def test_recurring_topic_links_again_after_the_window():
    early = [("Ana", f"note {i}: the Grafana dashboard needs love") for i in range(12)]
    filler = [("Ben", f"unrelated chatter number {i}") for i in range(60)]
    late = [("Ana", "We decided to go with Grafana Loki for logs."),
            ("Ben", "Agreed: picked Grafana Cloud for logs.")]
    turns = early + filler + late
    # a global ratio (12 of 74 messages) keeps "grafana" frequent forever: the update never links
    glob = _graph(turns, link_df_window=10_000)
    assert glob.link_df("grafana") == 14 and not any(glob.superseded)
    # in a 50-message window the old burst has scrolled out: the recurring topic links again
    win = _graph(turns, link_df_window=50)
    assert win.link_df("grafana") == 2 and win.superseded[-2] and win.current_of(len(turns) - 2) == len(turns) - 1
    # still frequent inside the window -> still not linking evidence
    busy = _graph(early + late, link_df_window=50)
    assert busy.link_df("grafana") == 14 and not any(busy.superseded)


# ---------------------------------------------------------------- B5 edge-size consistency
def _sub(g, seeds, base=None, query="q", asker=None, **kw):
    base = np.linspace(1.0, 0.1, len(g)) if base is None else base
    return build_subgraph(g, base, seeds, asker, query, None,
                          {"reply": 1.0, "episode": 0.8, "entity": 0.6, "version": 1.0}, **kw)


def test_edges_are_kept_by_in_candidate_membership():
    turns = [("Ana", f"status note {i} on Terraform") for i in range(40)]
    g = _graph(turns, max_episode=100)
    assert max(len(e.nodes) for e in g.edges if e.type == "entity") == 40
    # the entity edge is larger than max_edge_size globally, but only its in-candidate part counts
    sg = _sub(g, [0, 1, 2], max_edge_size=10, max_candidates=6)
    by_type = {t: idx for t, idx in sg.edges}
    assert len(sg.nodes) == 6 and {"entity", "channel", "speaker", "episode"} <= set(by_type)
    assert all(len(idx) == 6 for idx in by_type.values())
    # more in-candidate members than max_edge_size: the top members by base score are kept
    sg = _sub(g, list(range(20, 35)), max_edge_size=10, max_candidates=30)
    assert len(sg.nodes) == 25  # 15 seeds + the 10 best members of the large edges
    ent = next(idx for t, idx in sg.edges if t == "entity")
    assert len(ent) == 10 and set(sg.nodes[ent]) == set(range(10))  # base decreases with the node id
    # large edges still expand: to their best members, never to all of them
    assert len(_sub(g, [39], max_edge_size=10, max_candidates=300).nodes) == 11  # the seed + top 10
    # log_degree counts edges inside the subgraph only
    assert sg.feats[:, FI["log_degree"]].max() == pytest.approx(np.log1p(len(sg.edges)))
    assert sg.feats[:, FI["log_degree"]].min() == 0  # a seed outside every kept edge has degree 0


def test_long_version_chains_keep_current_first_and_last_k():
    vals = ["AWS EC2", "Google Cloud Run", "Azure App Service", "Fly.io", "Render", "DigitalOcean", "Heroku",
            "Railway", "Vercel", "Netlify", "Linode", "Hetzner"]
    turns = [("Ana", f"Update: we switched the backend hosting to {v} instead.") for v in vals]
    g = _graph(turns, edge_types=["version"])
    (slot,) = g.chains
    assert len(slot.records) == 12 and g.current_of(3) == 11
    # seeded in the middle of a chain longer than max_edge_size: the chain is not skipped
    sg = _sub(g, [5], max_edge_size=4, version_keep_k=2)
    assert list(sg.nodes) == [0, 5, 10, 11]  # first + seed + last 2 (the current value is the last)
    ((t, idx),) = sg.edges
    assert t == "version" and list(sg.nodes[idx]) == [0, 5, 10, 11]
    assert sg.feats[list(sg.nodes).index(11), FI["is_current"]] == 1 and sg.feats[0, FI["superseded"]] == 1
    # a short chain is kept whole
    assert len(_sub(g, [5], max_edge_size=60).nodes) == 12


# ---------------------------------------------------------------- B6 mentions
def test_mention_edges_and_names_are_not_entities():
    assert EDGE_TYPES[-1] == "mention" and len(EDGE_TYPES) == 7
    g = _graph([("Melanie", "How was your week?"),
                ("Caroline", "Great, thanks Melanie! I decided to adopt a puppy from the shelter."),
                ("Dave", "melanie can you send the photos"),
                ("Caroline", "Thanks Melanie, I picked the blue paint for my art show."),
                ("Melanie", "Caroline that is lovely")])
    edges = {(e.type, e.key): list(e.nodes) for e in g.edges}
    assert edges[("mention", "Melanie")] == [1, 2, 3]  # case-insensitive, not her own messages
    assert edges[("mention", "Caroline")] == [4]
    assert not any(t == "entity" and k in ("melanie", "caroline") for t, k in edges)
    assert not any(g.superseded)  # a shared name is still not linking evidence
    # generic role labels are not names
    g = _graph([("user", "the assistant said hello"), ("assistant", "hello user")])
    assert not any(e.type == "mention" for e in g.edges)
    assert not any(e.type == "mention" for e in Hypergraph(edge_types=["reply", "entity"]).view().edges)


def test_hgnn_type_embedding_grows_and_old_checkpoints_are_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from jarvis_m.memory.hypergraph.hgnn import HGNN, HGNNScorer

    model = HGNN(emb_dim=8, hidden=8, layers=1)
    assert model.type_emb.num_embeddings == len(EDGE_TYPES) == 7 and model.cfg["n_feats"] == len(FEATURE_NAMES)
    old = [n for n in FEATURE_NAMES if n not in ("age_frac", "log_age_days", "is_assistant")] + ["recency"]
    path = tmp_path / "old.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.cfg, "feature_names": old}, path)
    with pytest.raises(ValueError, match="different feature set"):
        HGNNScorer.load(str(path), "cpu")


# ---------------------------------------------------------------- B7 features
def test_feature_set_is_size_invariant_and_role_aware():
    for name in ("base_rank", "age_frac", "log_age_days", "log_degree", "is_assistant", "query_mentions_speaker",
                 "node_mentions_query_person", "log_n_participants"):
        assert name in FI
    assert "recency" not in FI and len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    turns = [("Priya", "We decided to use Kafka for the queue."),
             ("assistant", "Noted. Kafka it is."),
             ("Ben", "Priya, can you also check the billing page?"),
             ("Cy", "lunch?")]
    g = _graph(turns)
    base = np.array([0.9, 0.5, 0.7, 0.1])
    sg = build_subgraph(g, base, [0, 1, 2, 3], "Ben", "What did Priya decide for the queue?", 1800.0, {})
    f = lambda name: [round(float(x), 4) for x in sg.feats[:, FI[name]]]  # noqa: E731
    assert f("base_rank") == [1.0, 0.3333, 0.6667, 0.0]  # percentile inside the subgraph
    assert f("is_assistant") == [0, 1, 0, 0]
    assert f("query_mentions_speaker") == [1, 0, 0, 0]  # node 0 was said by the person the query names
    assert f("node_mentions_query_person") == [0, 0, 1, 0]  # node 2 mentions her
    assert f("is_asker") == [0, 0, 1, 0]
    assert f("log_n_participants") == [round(float(np.log1p(3)), 4)] * 4  # the assistant is not a participant
    # age is relative to the conversation span (here: now - first message = 1800 s)
    assert f("age_frac") == [1.0, round(1740 / 1800, 4), round(1680 / 1800, 4), round(1620 / 1800, 4)]
    assert f("log_age_days") == [round(float(np.log1p(a / 86400)), 4) for a in (1800, 1740, 1680, 1620)]
    # the same conversation stretched 100x in time has the same relative ages
    msgs = [Message(m.id, m.speaker, m.text, channel=m.channel, timestamp=m.timestamp * 100) for m in _msgs(turns)]
    g2 = Hypergraph()
    g2.add(msgs, _records(msgs))
    sg2 = build_subgraph(g2, base, [0, 1, 2, 3], "Ben", "What did Priya decide for the queue?", 180000.0, {})
    assert np.allclose(sg2.feats[:, FI["age_frac"]], sg.feats[:, FI["age_frac"]])
    assert np.allclose(sg2.feats[:, FI["base_rank"]], sg.feats[:, FI["base_rank"]])


# ---------------------------------------------------------------- B8 own commitments
def test_own_commitments_come_from_the_per_speaker_index():
    turns = [("Dana", "I'll take the load testing for the release."),
             ("Ben", "We decided to use Kafka for the queue."),
             ("Dana", "lunch?"),
             ("Dana", "I will own the billing page this sprint.")]
    g = _graph(turns)
    view = g.view()
    assert view.own_chain_nodes("Dana") == [0, 3] and view.own_chain_nodes("Ben") == [1]
    assert view.own_chain_nodes("Nobody") == []
    base = np.array([0.0, 1.0, 0.0, 0.0])
    sg = build_subgraph(g, base, [1], "Dana", "What am I responsible for?", None, {})
    assert list(sg.nodes) == [0, 1, 3]  # Dana's state-bearing messages, not her chatter
    assert list(build_subgraph(g, base, [1], "Dana", "What is the queue?", None, {}).nodes) == [1]


# ---------------------------------------------------------------- scope ablation = a view filter
def test_speaker_scope_is_a_filter_on_the_view(toy):
    conv = toy.conversations["cloud"]
    speaker = {m.id: m.speaker for m in conv.messages}
    for scope, ok in (("asker", lambda s: s == "Alice"), ("others", lambda s: s != "Alice")):
        mem = build_memory({**HG, "scope": scope, "state_header": True})
        mem.ingest(conv.messages)
        r = mem.retrieve("Which platform are we deploying the backend on?", asker="Alice", max_items=20)
        assert r.source_ids and all(ok(speaker[i]) for i in r.source_ids)
        assert all(ok(conv.messages[v].speaker) for v in mem.last_subgraph.nodes)  # expansion stays in scope
        view = mem.view("Alice")
        assert all(ok(conv.messages[v].speaker) for s in view.slots for v in s.nodes)
        assert mem.view(None).mask.all()  # no asker: nothing to filter on
