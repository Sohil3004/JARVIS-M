"""Phase E loaders, against small excerpts of the real releases (tests/fixtures/real_samples)."""

from pathlib import Path

from jarvis_m.data.loaders import (
    _index_ranges,
    _iso_lenient,
    _match_evidence,
    load_convomem,
    load_dataset,
    load_evermembench,
    load_groupmembench,
    load_socialmembench,
)

R = Path(__file__).resolve().parent / "fixtures" / "real_samples"


def test_groupmembench_release_has_no_evidence_trace():
    ds = load_groupmembench(str(R / "groupmembench"), domains=["Finance"])
    assert len(ds.conversations["gmb-Finance"].messages) == 8 and len(ds.questions) == 2
    assert all(q.evidence_ids == () for q in ds.questions)  # QA only: not trainable
    assert {q.category for q in ds.questions} == {"knowledge_update", "abstention"}
    m = ds.conversations["gmb-Finance"].messages[0]
    assert "is_noise" not in m.meta and "decision_label" not in m.meta  # hidden labels stay hidden
    assert ds.annotations[m.id]["msg_node"] == m.id


def test_evermembench_evidence_ranges_and_multiple_choice():
    assert _index_ranges("1, 4-6, 8") == [1, 4, 5, 6, 8] and _index_ranges(3) == [3]
    ds = load_evermembench(str(R / "evermembench"))
    (conv,) = ds.conversations.values()
    ids = {m.id for m in conv.messages}
    assert ds.annotations["unresolved_evidence"] == 0
    assert all(q.evidence_ids and set(q.evidence_ids) <= ids for q in ds.questions)
    assert {m.channel for m in conv.messages} <= {"Group 1", "Group 2", "Group 3"}
    mc = next(q for q in ds.questions if q.meta.get("options"))
    assert mc.meta["correct_option"] in "ABCD" and mc.answer == mc.meta["options"][mc.meta["correct_option"]]
    ts = [m.timestamp for m in conv.messages]
    assert ts == sorted(ts) and {q.category for q in ds.questions} == {"F_SH", "P_Style"}
    assert load_dataset({"name": "evermembench", "root": str(R / "evermembench"), "domain": "team"}).domain == "team"


def test_convomem_evidence_quotes_match_messages():
    texts = ["I set a call for Tuesday.", "Quick update: Northwind asked to move the call to Thursday, same time."]
    assert _match_evidence("I set a call for Tuesday.", texts) == [0]
    assert _match_evidence("northwind asked to move the call to thursday", texts) == [1]  # an excerpt
    assert _match_evidence("I set a cal for Tuesday.", texts) == [0]  # near copy
    assert _match_evidence("something else entirely", texts) == []
    ds = load_convomem(str(R / "convomem_schema"), evidence_types=["changing_evidence", "abstention_evidence"])
    assert ds.annotations == {"unresolved_evidence": 0, "evidence_quotes": 3}
    ch = next(q for q in ds.questions if q.category == "changing_evidence")
    conv = ds.conversations[ch.conversation_id]
    assert [conv.messages[[m.id for m in conv.messages].index(e)].text[:13] for e in ch.evidence_ids] == \
        ["I set a follo", "Quick update:"]
    assert {m.session_id for m in conv.messages} == {"c-1", "c-2"} and conv.messages[1].role == "assistant"
    assert next(q for q in ds.questions if q.category == "abstention_evidence").is_abstention


def test_socialmembench_json_columns_and_lenient_times():
    assert _iso_lenient("2025-10-06T12:60:00") == _iso_lenient("2025-10-06T13:00:00")
    ds = load_socialmembench(str(R / "socialmembench"))
    (conv,) = ds.conversations.values()
    assert ds.annotations["unresolved_evidence"] == 0 and len(conv.messages) == 20
    (q,) = ds.questions
    assert q.evidence_ids and set(q.evidence_ids) <= {m.id for m in conv.messages}
    assert q.category == "group_decision_recall" and q.id.startswith(conv.id)
    assert conv.messages[0].channel != conv.id  # the group's name is the channel
