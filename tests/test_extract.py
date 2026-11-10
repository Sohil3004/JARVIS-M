import json

import pytest

from jarvis_m.extract import (
    CachedExtractor,
    ExtractionStore,
    HeuristicExtractor,
    LLMExtractor,
    StateRecord,
    build_extractor,
    entities,
    parse_records,
)
from jarvis_m.extract.heuristic import topic_phrase
from jarvis_m.llm import ScriptedChat
from jarvis_m.types import Message


def _m(text, i="m1", speaker="A", channel="eng", **kw):
    return Message(id=i, speaker=speaker, text=text, channel=channel, timestamp=1.0, **kw)


@pytest.mark.parametrize("text,modality,attribute", [
    ("We decided to use PostgreSQL for the database.", "decision", "decision"),
    ("No, the backend hosting is not moving to AWS EC2.", "negated", "decision"),
    ("Ignore the rumour, Kafka is not happening for the message queue.", "negated", "decision"),
    ("I will grant the Artifact Registry Reader role tomorrow.", "commitment", "owner"),
    ("I'll get it fixed tomorrow.", "commitment", "owner"),
    ("Fixed the staging deploy by rotating the credentials.", "done", "status"),
    ("The payment webhook is failing with Error 403.", "blocker", "status"),
    ("We need the release done by Friday 5 PM.", "commitment", "deadline"),
    ("Pushing the release out, Monday 10 AM works better.", "decision", "deadline"),
    ("Maybe we should try Kafka for the queue?", "proposal", "decision"),
])
def test_heuristic_modalities(text, modality, attribute):
    (r,) = HeuristicExtractor().extract_one(_m(text))
    assert (r.modality, r.attribute) == (modality, attribute)


def test_heuristic_entities_topic_and_chitchat():
    assert {"aws ec2", "ec2"} & set(entities("Scrap AWS EC2, use EC2 spot later"))
    assert "pgbouncer" in entities("I'll put PgBouncer in front")
    assert "error 503" in entities("we get Error 503 under load")
    assert topic_phrase("Render it is for the backend hosting.") == "backend hosting"
    assert HeuristicExtractor().extract_one(_m("lol ok")) == []
    r = HeuristicExtractor().extract_one(_m("Change of plan: we switched the CI system to GitLab CI instead."))[0]
    assert r.replaces_cue and r.key == "group:eng|eng/ci system|decision" and r.holder == "group:eng"
    gmb = HeuristicExtractor().extract_one(_m("We decided to expand scope.", meta={"phase_name": "UX Approval"}))[0]
    assert gmb.subject == "UX Approval"  # visible phase metadata wins


def test_record_roundtrip():
    r = StateRecord("m1", "A", "decision", "decision", "Kafka", "eng/queue", ("kafka",), 1.0, True, True)
    assert StateRecord.from_json(json.loads(json.dumps(r.to_json()))) == r


def test_llm_extractor_batches_parses_and_persists(tmp_path):
    msgs = [_m("lunch?", "m0"), _m("We decided to use Kafka for the queue.", "m1"),
            _m("Actually switch the queue to NATS.", "m2"), _m("ok", "m3")]
    seen = []

    def fake(system, user):
        seen.append(user)
        return ('Sure! [{"msg": "m1", "subject": "message queue", "attribute": "decision", "value": "Kafka", '
                '"modality": "decision", "entities": ["Kafka"]}, {"msg": "<m2>", "subject": "message queue", '
                '"attribute": "decision", "value": "NATS", "modality": "decision", "replaces": true}, '
                '{"msg": "nope", "modality": "decision"}]')

    store = ExtractionStore(tmp_path / "x.jsonl")
    ex = LLMExtractor(ScriptedChat(fake), batch=8, context=2)
    recs = ex.extract(msgs, on_batch=store.append)
    assert [r.value for r in recs] == ["Kafka", "NATS"] and recs[1].replaces_cue and recs[0].explicit_key
    assert ex.calls == 1 and "m0" not in seen[0].split("TARGET")[1]  # chit-chat prefiltered out
    reloaded = ExtractionStore(tmp_path / "x.jsonl")
    assert reloaded.done() == {"m0", "m1", "m2", "m3"} and len(reloaded.records["m2"]) == 1
    assert parse_records("not json", {}) == [] and parse_records("[{bad", {}) == []


def test_cached_extractor_falls_back(tmp_path):
    store = ExtractionStore(tmp_path / "c.jsonl")
    store.append([StateRecord("m1", "A", "decision", "decision", "Kafka", "q", explicit_key=True)], ["m1"])
    cx = build_extractor({"type": "cached", "path": str(tmp_path / "c.jsonl")})
    assert isinstance(cx, CachedExtractor)
    recs = cx.extract([_m("x", "m1"), _m("We decided to use NATS.", "m9")])
    assert recs[0].explicit_key and recs[1].msg_id == "m9" and cx.misses == 1
    assert build_extractor({"type": "none"}).extract([_m("We decided X")]) == []
    with pytest.raises(FileNotFoundError):
        build_extractor({"type": "cached", "path": str(tmp_path / "missing.jsonl")})
    assert build_extractor({"type": "cached", "path": str(tmp_path / "missing.jsonl"), "allow_missing": True})
