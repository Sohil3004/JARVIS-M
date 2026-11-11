"""Phase E: StateBench v2 (domains, disjoint vocabularies, ellipsis, holders, assistant
suggestions, enterprise access metadata, background splicing, paraphrase)."""

import json

import pytest

from jarvis_m.access import conversation_directory
from jarvis_m.cli import main
from jarvis_m.data.loaders import load_jarvis
from jarvis_m.data.stress import save
from jarvis_m.data.stress_v2 import (
    CHOICE_SLOTS,
    DEADLINE_SLOTS,
    PERSONAL,
    SUGGESTIONS,
    TASKS,
    generate_v2,
    paraphrase,
    pool,
)
from jarvis_m.llm import ScriptedChat
from jarvis_m.memory import build_memory
from jarvis_m.types import Message


@pytest.fixture(scope="module")
def v2():
    return {d: generate_v2(12, seed=3, split="dev", domain=d) for d in ("team", "assistant", "personal", "enterprise")}


def test_vocabularies_are_disjoint_across_splits():
    for vocab in (list(CHOICE_SLOTS), DEADLINE_SLOTS, TASKS, [p[0] for p in PERSONAL], [s for s, _ in SUGGESTIONS]):
        tr, dv, te = (set(pool(vocab, s)) for s in ("train", "dev", "test"))
        assert tr and dv and te and not (tr & dv) and not (tr & te) and not (dv & te)
    test = generate_v2(20, split="test", domain="team")
    train_only = set(pool(list(CHOICE_SLOTS), "train")) | set(pool(DEADLINE_SLOTS, "train"))
    for q in test.questions:
        assert not any(f" {s}" in q.question for s in train_only if s not in pool(list(CHOICE_SLOTS), "test")), q.question


def test_labels_ids_holders_and_determinism(v2):
    for dom, ds in v2.items():
        ids = {m.id for c in ds.conversations.values() for m in c.messages}
        assert ds.annotations["supersession"] and all(r["older_id"] in ids and r["newer_id"] in ids
                                                       for r in ds.annotations["supersession"])
        assert all(q.meta["domain"] == dom for q in ds.questions)
        for q in ds.questions:
            assert set(q.evidence_ids) <= ids and set(q.stale_evidence_ids) <= ids
        st = ds.annotations["stats"]
        assert st["updates_without_slot_name"] >= 0.5 * st["updates"], (dom, st)  # half by ellipsis / pronoun
    again = generate_v2(12, seed=3, split="dev", domain="personal")
    assert json.dumps([m.__dict__ for c in again.conversations.values() for m in c.messages], default=str) == \
        json.dumps([m.__dict__ for c in v2["personal"].conversations.values() for m in c.messages], default=str)


def test_personal_holders_and_conflicting_preferences(v2):
    ds = v2["personal"]
    holders = {r["holder"] for r in ds.annotations["supersession"]}
    assert all(h.startswith("speaker:") for h in holders if not h.startswith("group:"))
    conflicts = [q for q in ds.questions if "holder_conflict" in q.meta["phenomena"]]
    assert conflicts and all(q.stale_answers and q.stale_answers[0] != q.answer for q in conflicts)
    unrel = [r for r in ds.annotations["supersession"] if r["label"] == "unrelated"]
    assert any(r["holder"].startswith("speaker:") for r in unrel)


def test_assistant_suggestions_and_corrections(v2):
    ds = v2["assistant"]
    phen = {p for q in ds.questions for p in q.meta["phenomena"]}
    assert {"suggestion_accepted", "suggestion_implicit", "suggestion_rejected"} <= phen
    assert {"update"} <= phen and ("user_correction" in phen or "correction" in phen)
    msgs = [m for c in ds.conversations.values() for m in c.messages]
    assert {m.speaker for m in msgs} == {"user", "assistant"}
    assert all(m.role == "assistant" for m in msgs if m.speaker == "assistant" and m.id in
               {i for q in ds.questions for i in q.evidence_ids})


def test_enterprise_viewer_relative_and_audience_gold_under_access_control(v2):
    ds = v2["enterprise"]
    assert "access" in ds.annotations and not ds.annotations.get("principals")
    unsorted = 0
    for cid, conv in ds.conversations.items():
        d = conversation_directory(ds.annotations, cid)
        assert {"general", "eng", "leadership"} == set(d["channels"]) and d["principals"]
        ts = [m.timestamp for m in conv.messages]
        unsorted += ts != sorted(ts)
        mem = build_memory({"type": "bm25", "idf_scope": "view"})
        mem.configure_access(d)
        mem.ingest(conv.messages)
        assert mem.access_active()
        for q in ds.questions_for(cid):
            if q.category not in ("viewer_relative", "audience"):
                continue
            p = mem.principal(q.asker, q.meta.get("audience"), q.meta.get("channel"))
            ok = mem.readable(p)
            node = {m.id: i for i, m in enumerate(mem.messages)}
            assert all(ok[node[e]] for e in q.evidence_ids), (q.id, q.category)  # gold is readable by the viewer
            if "viewer_low" in q.meta["phenomena"] or q.category == "audience":
                hidden = [m for m in conv.messages if q.stale_answers and q.stale_answers[0] in m.text]
                assert hidden and not any(ok[node[m.id]] for m in hidden if "[CONFIDENTIAL]" in m.text
                                          or m.channel == "leadership")
    assert unsorted  # out-of-order message lists
    cats = {q.category for q in ds.questions}
    assert {"viewer_relative", "audience"} <= cats


def test_background_splice():
    bg = [[Message(f"b{c}-{i}", ["Ann", "Raj"][i % 2], f"background turn {i} about gardening", float(i))
           for i in range(400)] for c in range(2)]
    ds = generate_v2(3, split="dev", domain="personal", background=bg, background_len=(300, 500))
    for cid, conv in ds.conversations.items():
        assert 300 <= len(conv.messages) <= 500
        ts = [m.timestamp for m in conv.messages]
        assert ts == sorted(ts)
        bg_ids = {m.id for m in conv.messages if m.meta.get("background")}
        assert len(bg_ids) > 200
        for q in ds.questions_for(cid):
            assert not set(q.evidence_ids) & bg_ids  # questions only about the inserted slots
        assert {m.speaker for m in conv.messages if m.meta.get("background")} <= \
            {m.speaker for m in conv.messages if not m.meta.get("background")}  # renamed to the people of the chat


def test_paraphrase_keeps_values_and_caches(tmp_path):
    ds = generate_v2(2, split="dev", domain="team")
    protect = ds.annotations["protect"]
    calls = []

    def fake(system, user):
        calls.append(user)
        keep = json.loads(user.split("\n")[0][len("KEEP: "):])
        return ("Reworded: " + " and ".join(keep)) if len(calls) % 3 else "a rewrite that drops everything"

    stats = paraphrase(ds, ScriptedChat(fake), tmp_path / "p.jsonl", progress=False)
    assert stats["rewritten"] > 0 and stats["kept_original"] > 0
    for c in ds.conversations.values():
        for m in c.messages:
            assert all(v.lower() in m.text.lower() for v in protect.get(m.id, []))
    n = len(calls)
    ds2 = generate_v2(2, split="dev", domain="team")
    assert paraphrase(ds2, ScriptedChat(fake), tmp_path / "p.jsonl", progress=False)["cached"] == stats["messages"]
    assert len(calls) == n


def test_make_stress_v2_cli(tmp_path):
    out = tmp_path / "sb2.json"
    assert main(["data", "make-stress", "--domain", "enterprise", "--split", "dev", "--n", "3", "--out", str(out)]) == 0
    ds = load_jarvis(str(out))
    assert ds.name == "statebench_v2_enterprise_dev" and ds.annotations["access"]
    labels = tmp_path / "labels" / "supersession" / "statebench_v2_enterprise_dev.jsonl"
    assert labels.exists()
    with pytest.raises(SystemExit):
        main(["data", "make-stress", "--background", "locomo", "--out", str(tmp_path / "x.json")])
    save(ds, tmp_path / "again.json")


def test_per_person_templates_are_first_person_singular():
    import re

    for subject, _a, _v, sets, upds, _q, _qm in PERSONAL:
        for fam in sets + upds:
            for t in fam:
                assert not re.search(r"\b(?:we|we're|our|us)\b", t, re.I), (subject, t)  # holder is the speaker
