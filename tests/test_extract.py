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


# ---------------------------------------------------------------- phase D: LLM extractor generality
def _session(cid, sid, texts, speakers=("user", "assistant")):
    return [Message(f"{cid}:{sid}#{j}", speakers[j % len(speakers)], t, timestamp=float(j), session_id=sid)
            for j, t in enumerate(texts)]


def test_llm_prompt_is_domain_neutral_with_holders_and_chat_type():
    from jarvis_m.extract.llm_extractor import SYSTEM
    from jarvis_m.extract.records import ATTRIBUTES

    for word in ("assistant", "personal", "holder", "sensitivity", "KNOWN SUBJECTS", "ONE-TO-ONE", "GROUP"):
        assert word in SYSTEM
    assert all(a in SYSTEM for a in ATTRIBUTES)
    seen = []
    ex = LLMExtractor(ScriptedChat(lambda s, u: seen.append(u) or "[]"), prefilter="none")
    ex.extract([_m("We decided to use Kafka.", "a", "Ana"), _m("ok fine with me", "b", "Ben"),
                _m("sounds good to me", "c", "Cy")])
    assert seen[0].startswith("CHAT: #eng, GROUP chat (3 participants")
    ex.extract([_m("I moved to Pune last month.", "d", "user", channel="dm"),
                _m("Congrats on the move!", "e", "assistant", channel="dm")])
    assert seen[1].startswith("CHAT: #dm, ONE-TO-ONE chat (1 participant: user)")


def test_llm_parses_holder_attribute_and_sensitivity():
    by = {"m1": _m("[CONFIDENTIAL] I'll take the Acme deal.", "m1", "Ana")}
    raw = ('[{"msg": "m1", "holder": "group:eng", "subject": "acme deal", "attribute": "owner", "value": "Ana", '
           '"modality": "commitment", "sensitivity": "L2"}, {"msg": "m1", "holder": "nobody", "attribute": "mood", '
           '"subject": "x", "value": "y", "modality": "state"}]')
    a, b = parse_records(raw, by)
    assert (a.holder, a.attribute, a.meta["sensitivity"]) == ("group:eng", "owner", "L2")
    assert b.holder == "group:eng" and b.attribute == "info" and b.modality == "state" and b.meta == {}


def test_light_prefilter():
    from jarvis_m.extract.llm_extractor import light_keep

    assert not light_keep(_m("lol ok")) and not light_keep(_m("thanks so much!"))
    assert light_keep(_m("ok 5pm")) and light_keep(_m("ok Kafka")) and light_keep(_m("i'm vegetarian now actually"))
    with pytest.raises(ValueError):
        LLMExtractor(ScriptedChat(), prefilter="maybe")


def test_slot_registry_is_in_the_prompt_and_persisted(tmp_path):
    seen = []

    def fake(system, user):
        seen.append(user)
        ids = [line.split(">")[0][1:] for line in user.split("TARGET messages:\n")[1].splitlines()]
        return json.dumps([{"msg": i, "holder": "group:eng", "subject": f"topic {i}", "attribute": "decision",
                            "value": "v", "modality": "decision"} for i in ids])

    store = ExtractionStore(tmp_path / "x.jsonl")
    ex = LLMExtractor(ScriptedChat(fake), batch=1, context=0, prefilter="none", registry_size=2,
                      cache=store, registry_path=str(store.registry_path))
    ex.extract([_m(f"We decided on plan number {i}.", f"m{i}") for i in range(4)], on_batch=store.append)
    assert "KNOWN SUBJECTS" not in seen[0] and "- group:eng: topic m0" in seen[1]
    assert "- group:eng: topic m1; topic m2" in seen[3]  # the most recent registry_size subjects
    resumed = LLMExtractor(ScriptedChat(fake), registry_size=2, registry_path=str(store.registry_path))
    assert resumed.registry.subjects("eng", "group:eng") == ["topic m2", "topic m3"]


def test_batch_cache_extracts_an_identical_session_once(tmp_path):
    calls = []

    def fake(system, user):
        calls.append(user)
        ids = [line.split(">")[0][1:] for line in user.split("TARGET messages:\n")[1].splitlines()]
        return json.dumps([{"msg": ids[0], "holder": "speaker:user", "subject": "home", "attribute": "location",
                            "value": "Pune", "modality": "state"}])

    texts = ["I moved to Pune last month for work.", "That sounds exciting, how is the weather there?"]
    store = ExtractionStore(tmp_path / "lme.jsonl")
    ex = LLMExtractor(ScriptedChat(fake), cache=store)
    a = ex.extract(_session("q1", "s7", texts), on_batch=store.append)
    b = ex.extract(_session("q2", "s7", texts), on_batch=store.append)  # same session in another question
    c = ex.extract(_session("q3", "s9", texts[:1] + ["Pune is lovely."]))  # different content: a miss
    assert len(calls) == 2 and ex.batches == 3 and ex.cache_hits == 1 and ex.hit_rate() == pytest.approx(1 / 3)
    assert [r.msg_id for r in b] == ["q2:s7#0"] and b[0].value == a[0].value and b[0].speaker == "user"
    assert c[0].msg_id == "q3:s9#0"
    again = LLMExtractor(ScriptedChat(fake), cache=ExtractionStore(tmp_path / "lme.jsonl"))
    again.extract(_session("q4", "s7", texts))
    assert again.calls == 0  # the cache is on disk: a resumed run hits it too
    other_model = LLMExtractor(ScriptedChat(fake, name="other"), cache=ExtractionStore(tmp_path / "lme.jsonl"))
    other_model.extract(_session("q5", "s7", texts))
    assert other_model.calls == 1  # the model is part of the key
