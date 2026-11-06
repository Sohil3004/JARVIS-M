"""Known failure modes of the v2 regex reconciler (see the audit doc).

The xfail(strict=True) cases document current behaviour: if a change makes one
of them pass, pytest reports XPASS-as-failure so we notice and update the doc.
"""

import pytest

from jarvis_m.memory.legacy_v2 import _legacy_core_dir, load_reconciliation_module

pytestmark = pytest.mark.skipif(
    not (_legacy_core_dir() / "reconciliation_engine.py").exists(), reason="legacy v2 code not present"
)


def _active(turns):
    eng = load_reconciliation_module().TemporalStateReconciliationEngine()
    for i, (spk, txt) in enumerate(turns):
        eng.process_utterance(spk, txt, f"s{i}", timestamp=1000 + i)
    gt = eng.get_reconciled_ground_truth()
    return gt, eng.format_reconciled_prefix().lower()


def test_original_phrasing_supersedes():
    gt, prefix = _active([
        ("Alice", "Team, we officially decided to use AWS EC2 for our backend deployment."),
        ("Alice", "Good catch! We officially switched to Google Cloud Run to utilize the credits."),
    ])
    assert "cloud run" in prefix and gt["superseded_count"] == 1


@pytest.mark.xfail(strict=True, reason="v2 regexes miss paraphrased decisions")
def test_paraphrased_decision():
    _, prefix = _active([("Alice", "Let's go with AWS EC2."), ("Alice", "Ok scrap EC2, Cloud Run it is.")])
    assert "cloud run" in prefix


@pytest.mark.xfail(strict=True, reason="v2 maps every decision onto one 'technology_stack' key")
def test_unrelated_decisions_both_active():
    gt, _ = _active([("Dan", "We decided to use PostgreSQL for the database."),
                     ("Dan", "We decided to use Cloud Run for hosting.")])
    assert len(gt["active_states"]) == 2


@pytest.mark.xfail(strict=True, reason="v2 reads future intent ('I will grant') as a resolution")
def test_future_intent_does_not_resolve():
    gt, _ = _active([("Bob", "Deployment failing with Error 403."),
                     ("Charlie", "I will grant Artifact Registry Reader IAM role to Bob tomorrow.")])
    assert gt["active_blockers"]


@pytest.mark.xfail(strict=True, reason="v2 ignores negation")
def test_negation():
    _, prefix = _active([("Alice", "We switched to Kafka."),
                         ("Bob", "We have not switched to RabbitMQ, ignore the earlier rumor.")])
    assert "kafka" in prefix and "rabbitmq" not in prefix
