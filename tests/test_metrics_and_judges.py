import pytest

from jarvis_m.eval.metrics import (
    bootstrap_ci,
    evidence_recall,
    exact_match,
    is_abstention,
    mcnemar,
    mentions_any,
    normalize,
    paired_bootstrap,
    token_f1,
)
from jarvis_m.eval.qa import LLMJudge, MatchJudge, Reader, extract_final
from jarvis_m.llm import OverlapChat, ScriptedChat
from jarvis_m.types import Question


def test_text_metrics():
    assert normalize("The  Cloud-Run!") == "cloudrun"
    assert exact_match("Google Cloud Run.", "google cloud run")
    assert token_f1("Monday 10 AM", "monday 10 am") == 1.0
    assert 0 < token_f1("on Monday", "Monday 10 AM") < 1
    assert is_abstention("That is not mentioned in the conversation.")
    assert not is_abstention("PostgreSQL")
    assert mentions_any("we use AWS EC2 now", ["EC2"])
    assert evidence_recall(["a", "b"], ["b", "c"]) == 0.5 and evidence_recall(["a"], []) is None


def test_stats():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 1])
    assert lo <= m <= hi and m == pytest.approx(0.75)
    assert bootstrap_ci([1, 0, 1]) == bootstrap_ci([1, 0, 1])  # seeded
    d, lo, hi, p = paired_bootstrap([1, 1, 1, 1] * 10, [0, 0, 1, 1] * 10)
    assert d == pytest.approx(0.5) and lo > 0 and p < 0.01
    a_only, b_only, p = mcnemar([True] * 10 + [False] * 5, [False] * 10 + [False] * 5)
    assert (a_only, b_only) == (10, 0) and p == pytest.approx(2 / 1024)
    assert mcnemar([True], [True])[2] == 1.0


def _q(**kw):
    base = dict(id="q", conversation_id="c", question="Which DB?", answer="PostgreSQL")
    base.update(kw)
    return Question(**base)


def test_match_judge():
    j = MatchJudge()
    assert j.judge(_q(), "We use PostgreSQL now").correct
    assert not j.judge(_q(), "MongoDB").correct
    assert j.judge(_q(is_abstention=True, answer="n/a"), "Not mentioned anywhere").correct


def test_llm_judge_parsing():
    yes = LLMJudge(ScriptedChat(lambda s, u: "reasoning...\nFinal: Correct"))
    no = LLMJudge(ScriptedChat(lambda s, u: "Final: Incorrect"))
    junk = LLMJudge(ScriptedChat(lambda s, u: "hmm"))
    assert yes.judge(_q(), "x").correct is True
    assert no.judge(_q(), "x").correct is False
    assert junk.judge(_q(), "x").correct is None
    seen = {}

    def lme(s, u):
        seen["u"] = u
        return "Yes"

    j = LLMJudge(ScriptedChat(lme), protocol="longmemeval")
    assert j.judge(_q(category="knowledge-update"), "PostgreSQL").correct
    assert "updated answer" in seen["u"]
    j.judge(_q(is_abstention=True), "unknown")
    assert "unanswerable" in seen["u"]


def test_reader_prompt_and_final_extraction():
    chat = ScriptedChat(lambda s, u: "thinking\nFinal: PostgreSQL")
    pred, raw = Reader(chat).answer(_q(asker="User_3"), "[user=Dan]\nwe moved to PostgreSQL")
    assert pred == "PostgreSQL" and "thinking" in raw
    assert extract_final("a\nFinal answer: b") == "b" and extract_final("just text") == "just text"
    out = OverlapChat().chat("", "Question:\nwhich database for records?\n\nRetrieved passages:\n[h]\nPostgreSQL for records\n[h]\nlunch")
    assert "PostgreSQL" in out
