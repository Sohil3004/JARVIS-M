"""Phase C: clearance-aware access control."""

import random
import re

import numpy as np
import pytest
import yaml

from jarvis_m.access import (
    HIDDEN_UPDATE_NOTE,
    AccessController,
    AccessPolicy,
    AccessViolation,
    Label,
    Principal,
    ThreadLabeler,
    build_access,
    join,
    meet,
)
from jarvis_m.memory import build_memory
from jarvis_m.memory.hypergraph.subgraph import FEATURE_NAMES
from jarvis_m.types import Message
from tests.conftest import ACCESS, TOY_ENTERPRISE

HG = {"type": "hypergraph", "embedder": {"type": "hashing"}, "access": ACCESS, "state_header": True}


def _mem(ds, cfg=HG, msgs=None, **over):
    mem = build_memory({**cfg, **over})
    mem.reset()
    mem.configure_access(ds.annotations)
    mem.ingest(msgs if msgs is not None else ds.conversations["acme"].messages)
    return mem


def _ask(mem, ds, qid, **kw):
    q = next(q for q in ds.questions if q.id == qid)
    return mem.retrieve(q.question, asker=q.asker, audience=q.meta.get("audience"), max_items=3,
                        now=q.question_time, **kw)


def _header_ids(r):
    return re.findall(r"\[\w+, [^,\]]+, (\S+) ", r.items[0].text) if r.items and r.items[0].kind == "state" else []


# ---------------------------------------------------------------- C1 types and policy
def test_policy_levels_roles_markers_and_principals():
    pol = AccessPolicy.from_config(ACCESS)
    assert pol.levels[0] == "public" and pol.level("manager") == 4 and pol.level("engineer") == 2 and pol.level(3) == 3
    for bad in ("CEO", 9, True):
        with pytest.raises(ValueError):
            pol.level(bad)
    with pytest.raises(ValueError, match="unknown access policy keys"):
        AccessPolicy.from_config({"levels": ["a"], "nope": 1})
    with pytest.raises(ValueError):
        AccessPolicy(levels=["a", "a"])
    assert pol.marker_label("FYI [confidential] numbers") == Label(3, frozenset())
    assert pol.marker_label("[HR] [EXEC ONLY] x") == Label(5, frozenset({"hr"}))
    m = Message("m", "A", "[LEGAL] the contract", channel="hr", meta={"acl_level": "L1", "acl_compartments": ["x"]})
    assert pol.base_label(m) == Label(3, frozenset({"hr", "legal", "x"}))  # channel default joined with all sources
    assert pol.base_label(Message("p", "A", "hi", channel="nowhere")) == Label()
    p = pol.principal("Ann", {"role": "manager", "compartments": ["hr"], "active": False})
    assert p == Principal("Ann", 4, frozenset({"hr"}), False)
    assert pol.principal("Ann", {"role": "manager", "level": "intern"}).level == 1  # level wins over role
    assert join(Label(1, frozenset("a")), Label(3, frozenset("b"))) == Label(3, frozenset("ab"))
    assert build_access("off") is None and build_access(False) is None
    assert build_access(None).policy.mode == "auto" and build_access(True).policy.mode == "on"
    ac = build_access({"policy": ACCESS, "final_check": "prod", "signal_hidden_updates": True})
    assert ac.policy.signal_hidden_updates and ac.policy.level("exec") == 5 and ac.policy.mode == "auto"
    assert build_access({"policy": ACCESS, "mode": "debug"}).policy.final_check == "debug"  # the old meaning of mode
    with pytest.raises(ValueError, match="access mode"):
        AccessPolicy(mode="sometimes")


def test_labels_only_go_up_and_replies_inherit():
    pol = AccessPolicy.from_config(ACCESS)
    tl = ThreadLabeler(pol.base_label)
    msgs = [Message("b", "B", "ok", channel="general", reply_to="a"),  # arrives before its parent
            Message("c", "C", "noted", channel="general", reply_to="b"),
            Message("a", "A", "[CONFIDENTIAL] numbers", channel="general"),
            Message("d", "D", "[CONFIDENTIAL] x", channel="general"),
            Message("e", "E", "public reply", channel="general", reply_to="d", meta={"acl_level": 0}),
            Message("f", "F", "other channel", channel="eng", reply_to="d")]
    for m in msgs:
        tl.add(m)
    lv = {m.id: tl.labels[i].level for i, m in enumerate(msgs)}
    assert lv == {"b": 3, "c": 3, "a": 3, "d": 3, "e": 3, "f": 1}  # e cannot lower its label; f is another channel


def test_audience_is_the_meet(toy_enterprise):
    ac = AccessController(AccessPolicy.from_config(ACCESS))
    ac.configure(toy_enterprise.annotations)
    dana, bob = ac.resolve("Dana"), ac.resolve("Bob")
    assert dana.level == 5 and dana.compartments == {"leadership", "deal"} and bob.level == 2
    assert ac.effective("Dana") == dana  # default audience: the asker alone
    duo = ac.effective("Dana", ["Bob"])
    assert (duo.level, duo.compartments, duo.members) == (2, frozenset(), ("Bob", "Dana"))
    lead = ac.effective("Dana", "leadership")
    assert (lead.level, lead.compartments, lead.members) == (4, frozenset({"leadership"}), ("Dana", "Mira"))
    eng = ac.effective("Dana", "eng")
    assert eng.level == 1 and eng.members == ("Bob", "Dana", "Ivy", "Mira")
    everyone = ac.effective("Dana", "general")  # an open channel: anyone may be reading
    assert everyone.level == 0 and not ac.directory.is_member(everyone, "eng")
    assert not ac.effective("Dana", "no-such-channel").active  # an unlisted closed channel: nobody reads
    assert not ac.effective("Dana", ["Zed"]).active  # an inactive member blocks the audience
    ac.policy.inactive_audience_members = "ignore"
    assert ac.effective("Dana", ["Zed"]).level == 5
    assert ac.resolve("stranger") == Principal("stranger") and not ac.resolve("Zed").active
    assert meet([]).active is False


def test_can_read_checks_membership_at_query_time(toy_enterprise):
    ac = AccessController(AccessPolicy.from_config(ACCESS))
    ac.configure(toy_enterprise.annotations)
    ivy, dana = ac.resolve("Ivy"), ac.resolve("Dana")
    eng1 = Label(1)
    assert ac.can_read(ivy, eng1, "eng") and not ac.can_read(ivy, Label(2), "eng")
    assert not ac.can_read(dana, Label(4, frozenset({"hr"})), "leadership")  # missing compartment
    assert ac.can_read(dana, Label(4, frozenset({"deal"})), "leadership")
    ac.directory.remove_member("eng", "Ivy")
    assert not ac.can_read(ivy, eng1, "eng")  # removed from the channel: no longer readable
    ac.directory.add_member("eng", "Ivy")
    assert ac.can_read(ivy, eng1, "eng")
    assert not ac.can_read(ivy, Label(), "unlisted-channel")  # closed by default
    assert not ac.can_read(Principal("Ivy", 5, active=False), Label(), "general")


# ---------------------------------------------------------------- C3 viewer-relative state
def test_viewer_relative_current_value(toy_enterprise):
    ds = toy_enterprise
    mem = _mem(ds)
    labels = {m.id: mem.graph.labels[i] for i, m in enumerate(mem.graph.messages)}
    assert labels["e1"].level == 3 and labels["e2"].level == 3  # marker; reply inherits
    assert labels["l0"] == Label(4, frozenset({"leadership", "deal"}))
    bob, mira = _ask(mem, ds, "acme/q0"), _ask(mem, ds, "acme/q1")
    assert "g0" in bob.source_ids and "g2" not in bob.source_ids
    assert _header_ids(bob)[0] == "g0"  # for Bob the current hosting is still AWS EC2
    assert "g2" in mira.source_ids and _header_ids(mira)[0] == "g2"  # Mira sees the update
    mira_view = mem.view("Mira", mem.principal("Mira"))
    g0 = mem.graph.id2node["g0"]
    assert mira_view.superseded[g0] and not mem.view("Bob", mem.principal("Bob")).superseded[g0]
    # Mira answering for Bob gets Bob's view
    duo = _ask(mem, ds, "acme/q2")
    assert "g2" not in duo.source_ids and _header_ids(duo)[0] == "g0"
    assert duo.audit["principal"]["members"] == ["Bob", "Mira"] and duo.audit["principal"]["level_name"] == "L1"
    # the beta launch: Bob (L1) sees March 14, Dana (exec) sees the confidential April 2
    assert _header_ids(_ask(mem, ds, "acme/q3"))[0] == "e0"
    assert _header_ids(_ask(mem, ds, "acme/q4"))[0] == "e1"
    # Dana answering into #eng: the meet of the channel members
    into_eng = _ask(mem, ds, "acme/q8")
    assert "e1" not in into_eng.source_ids and into_eng.audit["principal"]["level_name"] == "intern"
    # compartments: only Dana holds "deal"
    assert "l0" in _ask(mem, ds, "acme/q7").source_ids
    assert not {"l0", "l1"} & set(_ask(mem, ds, "acme/q6").source_ids)


def test_no_signal_of_hidden_updates_by_default(toy_enterprise):
    ds = toy_enterprise
    plain = _ask(_mem(ds), ds, "acme/q3")
    assert HIDDEN_UPDATE_NOTE not in plain.context and plain.audit["hidden_update_signals"] == 0
    sig = _mem(ds, access={"policy": ACCESS, "signal_hidden_updates": True})
    r = _ask(sig, ds, "acme/q3")  # Bob: the beta launch was moved in a message he cannot read
    line = next(ln for ln in r.items[0].text.splitlines() if " e0 " in ln)
    assert line.endswith(f"({HIDDEN_UPDATE_NOTE})") and "April" not in r.context and "e1" not in r.source_ids
    assert HIDDEN_UPDATE_NOTE not in _ask(sig, ds, "acme/q4").context  # Dana sees the newest version
    no_header = _mem(ds, access={"policy": ACCESS, "signal_hidden_updates": True}, state_header=False)
    r = _ask(no_header, ds, "acme/q3")
    assert r.items[-1].kind == "notice" and r.items[-1].source_ids == ()


# ---------------------------------------------------------------- C5 enforcement
def test_final_check_raises_in_debug_and_filters_in_prod(toy_enterprise, monkeypatch, caplog):
    ds = toy_enterprise
    for mode in ("debug", "prod"):
        monkeypatch.setenv("JARVIS_ACCESS_CHECK", mode)
        mem = _mem(ds)
        # simulate a bug upstream of the final check: the view lets everything through
        n = len(mem.messages)
        monkeypatch.setattr(mem, "_candidates", lambda asker, principal=None, n=n: np.ones(n, bool))
        if mode == "debug":
            with pytest.raises(AccessViolation):
                _ask(mem, ds, "acme/q0")
        else:
            r = _ask(mem, ds, "acme/q0")
            ok = mem.readable(mem.principal("Bob"))
            assert all(ok[mem.graph.id2node[i]] for i in r.source_ids)
            assert r.audit["filtered"] > 0 and r.audit["final_check"] == "prod" and "withheld" in caplog.text
        monkeypatch.undo()


def test_baselines_enforce_too(toy_enterprise):
    ds = toy_enterprise
    for kind in ("bm25", "dense", "hybrid", "recency", "full_context"):
        mem = _mem(ds, {"type": kind, "embedder": {"type": "hashing"}, "access": ACCESS} if kind in ("dense", "hybrid")
                   else {"type": kind, "access": ACCESS})
        for asker in ("Bob", "Ivy", "Mira", "Dana", "Zed", None):
            r = mem.retrieve("backend hosting beta launch acquire", asker=asker, max_items=20)
            ok = mem.readable(mem.principal(asker))
            assert all(ok[i] for i in (next(j for j, m in enumerate(mem.messages) if m.id == s) for s in r.source_ids))
            assert r.audit["visible"] == ok.sum()
        assert mem.retrieve("x", asker="Zed").items == []  # inactive: nothing
    off = build_memory({"type": "bm25", "access": "off"})
    off.ingest(ds.conversations["acme"].messages)
    assert off.retrieve("x").audit == {"mode": "off"} and off.principal("Bob") is None


def test_measure_only_mode_does_not_filter(toy_enterprise):
    ds = toy_enterprise
    mem = _mem(ds, access={"policy": ACCESS, "enforce": False})
    r = _ask(mem, ds, "acme/q0")
    assert "g2" in r.source_ids and not r.audit["enforced"] and r.audit["filtered"] == 0
    assert r.audit["unreadable_returned"] > 0


def test_configure_access_before_ingest(toy_enterprise):
    mem = _mem(toy_enterprise)
    with pytest.raises(RuntimeError, match="before ingest"):
        mem.configure_access(toy_enterprise.annotations)
    mem.configure_access(principals={"New": {"level": "exec"}})  # principals may change any time


# ---------------------------------------------------------------- C6 non-interference
def _random_enterprise(seed, n=70):
    rng = random.Random(seed)
    people = ["Ana", "Ben", "Cy", "Dee", "Eli"]
    chans = {"general": {"members": "*"}, "eng": {"members": ["Ana", "Ben", "Cy", "Dee"]},
             "leadership": {"members": ["Ana", "Dee"], "level": "manager", "compartments": ["leadership"]}}
    slots = ["backend hosting", "primary database", "release date", "beta launch", "message queue"]
    vals = ["AWS EC2", "Google Cloud Run", "PostgreSQL", "MongoDB", "Kafka", "NATS", "March 14", "April 2", "Fly.io"]
    temps = ["We decided to use {v} for the {s}.", "Change of plan: we switched the {s} to {v} instead.",
             "Deadline for the {s}: {v}.", "The {s} has been rescheduled to {v}.", "No, the {s} is not moving to {v}.",
             "I'll take the {s} migration.", "The {s} is failing with Error 403.", "Fixed the {s} by rotating keys.",
             "{p}, can you check the {s}?", "lunch at 1?", "Maybe we should try {v} for the {s}?"]
    msgs = []
    for i in range(n):
        ch = rng.choice(list(chans))
        meta = {}
        r = rng.random()
        if r < 0.2:
            meta["acl_level"] = rng.choice(["L1", "L2", "manager", "exec"])
        elif r < 0.3:
            meta["acl_compartments"] = [rng.choice(["deal", "hr"])]
        text = rng.choice(temps).format(s=rng.choice(slots), v=rng.choice(vals), p=rng.choice(people))
        if rng.random() < 0.1:
            text = "[CONFIDENTIAL] " + text
        prev = [m for m in msgs if m.channel == ch]
        reply = rng.choice(prev).id if prev and rng.random() < 0.25 else None
        msgs.append(Message(f"m{i}", rng.choice(people), text, timestamp=1.7e9 + 600 * i + rng.uniform(0, 900),
                            channel=ch, session_id=f"{ch}-{i // 20}", reply_to=reply, meta=meta))
    principals = {p: {"level": rng.choice(["intern", "L1", "L2", "manager", "exec"]),
                      "compartments": rng.sample(["leadership", "deal", "hr"], rng.randint(0, 3))} for p in people}
    principals["Eli"]["active"] = rng.random() < 0.7
    return msgs, {"channels": chans, "principals": principals}


def _state_by_id(mem, view):
    g = mem.graph
    mid = lambda v: g.messages[v].id  # noqa: E731
    slots = sorted((s.key, tuple(mid(v) for v, _ in s.records), tuple(mid(v) for v in s.rejected)) for s in view.slots)
    flags = {mid(v): (bool(view.superseded[v]), bool(view.current[v])) for v in view.nodes}
    edges = sorted((e.type, tuple(sorted(mid(v) for v in e.nodes))) for e in view.edges)
    return slots, flags, edges


@pytest.mark.parametrize("seed", range(6))
def test_non_interference(seed):
    msgs, ann = _random_enterprise(seed)
    cfg = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 128}, "access": ACCESS, "idf_scope": "view",
           "state_header": True}
    full = build_memory(cfg)
    full.configure_access(ann)
    full.ingest(msgs)
    queries = ["What are we using for the backend hosting?", "When is the release date?",
               "What am I responsible for?", "Is the beta launch problem fixed?", "What did Ben decide?"]
    rng = random.Random(seed)
    checked = 0
    for viewer, audience in [("Ana", None), ("Ben", None), ("Cy", None), ("Eli", None), ("Dee", ["Cy"]),
                             ("Ana", "leadership"), ("Ben", "eng"), (None, None)]:
        p = full.principal(viewer, audience)
        ok = full.readable(p)
        if ok.all() or not ok.any():
            continue
        visible = [m for m, keep in zip(msgs, ok) if keep]
        part, shuffled = build_memory(cfg), build_memory(cfg)
        for mem, order in ((part, visible), (shuffled, rng.sample(visible, len(visible)))):
            mem.configure_access(ann)
            mem.ingest(order)
            assert mem.readable(mem.principal(viewer, audience)).all()
        # graph state: identical without the hidden messages, in any ingest order
        vf = full.view(viewer, p)
        for mem in (part, shuffled):
            assert _state_by_id(full, vf) == _state_by_id(mem, mem.view(viewer, mem.principal(viewer, audience)))
        # retrieval: identical without the hidden messages (same ingest order: score ties break on it)
        for q in queries:
            a = full.retrieve(q, asker=viewer, audience=audience, max_items=8)
            b = part.retrieve(q, asker=viewer, audience=audience, max_items=8)
            assert a.source_ids == b.source_ids and a.context == b.context, (viewer, audience, q)
            sa, sb = full.last_subgraph, part.last_subgraph
            ids_a = [full.graph.messages[v].id for v in sa.nodes]
            ids_b = [part.graph.messages[v].id for v in sb.nodes]
            assert ids_a == sorted(ids_a, key=ids_b.index) and set(ids_a) == set(ids_b)
            perm = [ids_b.index(i) for i in ids_a]
            assert np.array_equal(sa.feats, sb.feats[perm]), [FEATURE_NAMES[j] for j in
                                                              np.nonzero((sa.feats != sb.feats[perm]).any(0))[0]]
        checked += 1
    assert checked >= 3


def test_global_idf_is_the_documented_residual():
    msgs, ann = _random_enterprise(0)
    base = {"type": "bm25", "access": ACCESS}
    for scope, exact in (("view", True), ("global", False)):
        full, part = build_memory({**base, "idf_scope": scope}), build_memory({**base, "idf_scope": scope})
        full.configure_access(ann)
        full.ingest(msgs)
        p = full.principal("Cy")
        ok = full.readable(p)
        part.configure_access(ann)
        part.ingest([m for m, k in zip(msgs, ok) if k])
        sf = full._raw_scores("backend hosting release date", ok)[ok]
        sp = part._raw_scores("backend hosting release date", None)
        assert np.array_equal(sf, sp) == exact  # global IDF counts hidden documents: scores shift


# ---------------------------------------------------------------- property tests
@pytest.mark.parametrize("seed", range(8))
def test_nothing_returned_is_invisible(seed):
    msgs, ann = _random_enterprise(100 + seed, n=50)
    rng = random.Random(seed)
    cfg = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}, "access": ACCESS,
           "state_header": True, "scorer": {"type": rng.choice(["fixed", "seeds"])}}
    mem = build_memory(cfg)
    mem.configure_access(ann)
    mem.ingest(msgs)
    ac = mem.access
    people = list(ann["principals"]) + ["stranger", None]
    for _ in range(15):
        asker = rng.choice(people)
        audience = rng.choice([None, None, "eng", "general", "leadership", "nowhere",
                               rng.sample(list(ann["principals"]), 2)])
        r = mem.retrieve(rng.choice(["hosting?", "release date", "what am I doing", "database", "Kafka"]),
                         asker=asker, audience=audience, max_items=rng.randint(1, 10),
                         budget_tokens=rng.choice([None, 150]))
        p = mem.principal(asker, audience)
        # independently of the memory's own masks: straight from labels and the directory
        allowed = {m.id for m, lab in zip(mem.graph.messages, mem.graph.labels) if ac.can_read(p, lab, m.channel)}
        assert set(r.source_ids) <= allowed
        assert set(_header_ids(r)) <= allowed
        assert {mem.graph.messages[v].id for v in mem.last_subgraph.nodes} <= allowed
        for m in mem.graph.messages:  # no hidden text leaks into the context
            if m.id not in allowed and len(m.text) > 25 and all(m.text not in o.text for o in mem.graph.messages
                                                               if o.id in allowed):
                assert m.text not in r.context
        assert r.audit["visible"] == len(allowed)


# ---------------------------------------------------------------- leak rate
def test_rank_eval_leak_rate(tmp_path):
    from jarvis_m.eval.rank_eval import rank_eval

    ds = [{"label": "ent", "domain": "enterprise", "name": "jarvis", "path": str(TOY_ENTERPRISE)}]
    mem = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 64}, "access": ACCESS}
    on = rank_eval({"memory": mem, "datasets": ds, "bootstrap": {"n": 50}}, progress=False)
    off = rank_eval({"memory": {**mem, "access": {"policy": ACCESS, "enforce": False}}, "datasets": ds,
                     "bootstrap": {"n": 50}}, progress=False)
    assert on["datasets"][0]["scorers"]["seeds"]["leak_rate"] == 0.0
    assert off["datasets"][0]["scorers"]["seeds"]["leak_rate"] > 0
    assert on["macro"]["seeds"]["leak_rate"] == 0.0
    plain = rank_eval({"memory": {**mem, "access": "off"}, "datasets": ds,
                       "bootstrap": {"n": 50}}, progress=False)
    assert plain["datasets"][0]["scorers"]["seeds"]["leak_rate"] is None  # access control off: not measured
    from jarvis_m.eval.rank_eval import to_markdown

    assert "leak_rate" in to_markdown(on)


def test_runner_passes_audience_and_records_audit(tmp_path):
    from jarvis_m.eval.runner import run

    out = run({"name": "ent", "output_dir": str(tmp_path), "dataset": {"name": "jarvis", "path": str(TOY_ENTERPRISE)},
               "memory": {**HG, "embedder": {"type": "hashing", "dim": 64}}, "retrieval": {"max_items": 3}},
              progress=False)
    rows = [yaml.safe_load(line) for line in (out / "results.jsonl").read_text().splitlines()]
    by = {r["qid"]: r for r in rows}
    assert by["acme/q2"]["audit"]["principal"]["members"] == ["Bob", "Mira"]
    assert by["acme/q8"]["audit"]["audience"] == "eng" and by["acme/q0"]["audit"]["final_check"] == "debug"


def test_yaml_unquoted_off_on_mode():
    import yaml

    from jarvis_m.access import build_access

    assert build_access(yaml.safe_load("mode: off")) is None
    assert build_access(yaml.safe_load("mode: on")).policy.mode == "on"
    assert build_access(yaml.safe_load("mode: auto")).policy.mode == "auto"
