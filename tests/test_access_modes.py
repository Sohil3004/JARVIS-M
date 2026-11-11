"""Phase D additions: access mode off|auto|on, the group-chat default audience and
compartment property tests."""

import random

import pytest

from jarvis_m.access import AccessConfigError, AccessPolicy, build_access
from jarvis_m.data.loaders import load_locomo
from jarvis_m.data.stress import generate
from jarvis_m.memory import build_memory
from jarvis_m.types import Message
from tests.conftest import ACCESS

HG = {"type": "hypergraph", "embedder": {"type": "hashing", "dim": 128}, "state_header": True}


def _locomo(write_json):
    turns = [("A", "I moved to Pune last month for the new job."), ("B", "Nice! I adopted a puppy called Rex."),
             ("A", "My sister visits next week."), ("B", "Rex chewed my favourite shoes lol"),
             ("A", "Actually I relocated to Mumbai in May."), ("B", "We went hiking at Lonavala on Sunday.")]
    sample = [{
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "A", "speaker_b": "B", "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [{"speaker": s, "dia_id": f"D1:{i}", "text": t} for i, (s, t) in enumerate(turns)],
        },
        "qa": [{"question": "Where does A live?", "answer": "Mumbai", "evidence": ["D1:4"], "category": 4}],
    }]
    return load_locomo(str(write_json("locomo10.json", sample)))


def _same(ds, cfg, queries):
    for conv in ds.conversations.values():
        out = []
        for access in ("off", None, "auto"):
            mem = build_memory({**cfg, "access": access})
            mem.reset()
            mem.configure_access(ds.annotations)
            mem.ingest(conv.messages)
            assert (mem.access is None) == (access == "off") and not mem.access_active()
            out.append([(r.source_ids, r.context) for r in
                        (mem.retrieve(q.question, asker=q.asker, now=q.question_time, max_items=6,
                                      channel=conv.messages[-1].channel)
                         for q in queries(conv.id))])
        assert out[0] == out[1] == out[2]


@pytest.mark.parametrize("cfg", [HG, {"type": "bm25"}, {"type": "hybrid", "embedder": {"type": "hashing"}}])
def test_auto_is_a_no_op_without_acl_metadata(cfg, write_json):
    sb = generate(3, seed=1, split="dev")
    _same(sb, cfg, sb.questions_for)
    loc = _locomo(write_json)
    _same(loc, cfg, loc.questions_for)


def test_auto_enforces_labels_even_in_a_dm():
    """The intern DM: no principals, no directory, only a labelled #leadership message."""
    msgs = [Message("l0", "Mira", "We decided the layoffs list is final: the Pune office closes.", 1.0, "leadership",
                    meta={"acl_level": "manager"}),
            Message("l1", "Dana", "Agreed, the Pune office closes in March.", 2.0, "leadership"),
            Message("d0", "Ivy", "hey, what happens to the Pune office?", 3.0, "dm-ivy-bob"),
            Message("d1", "Bob", "no idea, ask in #general", 4.0, "dm-ivy-bob")]
    for principals in (None, {"Ivy": {"level": "intern"}, "Mira": {"level": "manager"}}):
        mem = build_memory(HG)  # access not configured: mode auto
        mem.configure_access(principals=principals) if principals else None
        mem.ingest(msgs)
        assert mem.access_active()
        r = mem.retrieve("Pune office closes layoffs", asker="Ivy", channel="dm-ivy-bob", max_items=10)
        assert "l0" not in r.source_ids and "layoffs" not in r.context and r.audit["active"]
        assert r.audit["principal"]["id"] == "Ivy" and r.audit["principal"]["members"] == []  # a DM: the asker
    # off really is off
    off = build_memory({**HG, "access": "off"})
    off.ingest(msgs)
    assert "l0" in off.retrieve("Pune office closes layoffs", asker="Ivy", max_items=10).source_ids


def test_mode_on_requires_metadata():
    mem = build_memory({**HG, "access": "on"})
    mem.ingest([Message("a", "A", "We decided to use Kafka.", 1.0)])
    with pytest.raises(AccessConfigError):
        mem.retrieve("kafka", asker="A")
    mem.configure_access(principals={"A": {"level": "L1"}})
    assert mem.retrieve("kafka", asker="A").audit["mode"] == "on"
    assert build_access({"policy": ACCESS, "mode": "off"}) is None


def test_default_audience_group_chat_dm_and_explicit(toy_enterprise):
    ds = toy_enterprise
    mem = build_memory({**HG, "access": ACCESS})
    mem.configure_access(ds.annotations)
    mem.ingest(ds.conversations["acme"].messages)
    # #eng has four members: an answer posted there is for all of them (the meet: intern)
    eng = mem.retrieve("beta launch", asker="Dana", channel="eng", max_items=5)
    assert eng.audit["audience"] == "eng" and eng.audit["principal"]["level_name"] == "intern"
    assert "e1" not in eng.source_ids  # the [CONFIDENTIAL] update stays out of the group answer
    # #leadership has two members: a DM, the asker alone
    lead = mem.retrieve("beta launch", asker="Dana", channel="leadership", max_items=5)
    assert lead.audit["audience"] is None and lead.audit["principal"]["level_name"] == "exec"
    # no channel: the asker alone; an explicit audience always wins
    assert mem.retrieve("x", asker="Dana").audit["principal"]["id"] == "Dana"
    duo = mem.retrieve("x", asker="Dana", channel="eng", audience=["Mira"])
    assert duo.audit["principal"]["members"] == ["Dana", "Mira"] and duo.audit["principal"]["level_name"] == "manager"
    # an open channel: anyone may read the answer
    gen = mem.retrieve("hosting", asker="Dana", channel="general", max_items=5)
    assert gen.audit["principal"]["level_name"] == "public"
    # a channel the directory does not list: who wrote there decides
    ac = mem.access
    assert ac.default_audience("Dana", "side", ["Bob"]) is None
    assert ac.default_audience("Dana", "side", ["Bob", "Ivy"]) == ["Bob", "Dana", "Ivy"]


def test_auto_counts_policy_markers_and_directories():
    plain = build_access(None)
    assert not plain.configured()
    assert build_access(ACCESS).configured()  # channel defaults and markers
    d = build_access(None)
    d.configure(channels={"eng": {"members": ["A", "B"]}})
    assert d.configured() and not d.directory.is_open("eng") and not d.directory.is_open("other")
    assert build_access(None).directory.is_open("anything")  # an empty directory restricts nothing
    assert AccessPolicy(mode="debug").final_check == "debug"


# ---------------------------------------------------------------- compartments, property test
_COMPS = ["deal", "hr", "legal", "board"]
_CHAN = {"general": [], "hr": ["hr"], "legal": ["legal"], "deals": ["deal"]}


def _raw_compartments(m):
    """Independently of the access module: channel default + own metadata + text markers."""
    out = set(_CHAN[m.channel]) | set(m.meta.get("acl_compartments") or [])
    if "[HR]" in m.text:
        out.add("hr")
    if "[LEGAL]" in m.text:
        out.add("legal")
    return out


@pytest.mark.parametrize("seed", range(6))
def test_nothing_returned_carries_a_compartment_the_principal_lacks(seed):
    rng = random.Random(seed)
    people = ["Ana", "Ben", "Cy", "Dee", "Eli", "Fay"]
    comps = {p: set(rng.sample(_COMPS, rng.randint(0, len(_COMPS)))) for p in people}
    policy = {"policy": ACCESS,
              "channels": {"general": {"level": "public"}, "hr": {"level": "public", "compartments": ["hr"]},
                           "legal": {"level": "public", "compartments": ["legal"]},
                           "deals": {"level": "public", "compartments": ["deal"]}},
              "markers": [{"pattern": r"\[HR\]", "compartments": ["hr"]}, {"pattern": r"\[LEGAL\]", "compartments": ["legal"]}]}
    texts = ["We decided to use {v} for the deal room.", "The {v} contract is final.", "Severance for {v} is approved.",
             "I'll take the {v} review.", "Board meeting moved to {v}.", "Is {v} still blocked?"]
    vals = ["Acme", "Globex", "Initech", "March 3", "Friday", "Hooli"]
    msgs = []
    for i in range(60):
        ch = rng.choice(list(_CHAN))
        meta = {"acl_compartments": rng.sample(_COMPS, 1)} if rng.random() < 0.3 else {}
        text = rng.choice(["", "", "[HR] ", "[LEGAL] "]) + rng.choice(texts).format(v=rng.choice(vals))
        msgs.append(Message(f"m{i}", rng.choice(people), text, 1.7e9 + 60 * i, ch, meta=meta))
    principals = {p: {"level": "exec", "compartments": sorted(comps[p])} for p in people}
    for cfg in ({**HG, "scorer": {"type": rng.choice(["fixed", "seeds"])}}, {"type": "bm25"}):
        mem = build_memory({**cfg, "access": policy})
        mem.configure_access(principals=principals, channels={c: {"members": "*"} for c in _CHAN})
        mem.ingest(msgs)
        raw = {m.id: _raw_compartments(m) for m in msgs}
        for _ in range(20):
            asker = rng.choice(people)
            audience = rng.choice([None, rng.sample(people, rng.randint(1, 3))])
            holders = [asker] + (list(audience) if audience else [])
            allowed = set.intersection(*[comps[p] for p in holders])
            r = mem.retrieve(rng.choice(vals + ["contract", "severance", "deal"]), asker=asker, audience=audience,
                             max_items=rng.randint(1, 12))
            for mid in r.source_ids:
                assert raw[mid] <= allowed, (mid, raw[mid], allowed)
            if r.items and r.items[0].kind == "state":
                for mid in (m.id for m in msgs if f" {m.id} " in r.items[0].text):
                    assert raw[mid] <= allowed
