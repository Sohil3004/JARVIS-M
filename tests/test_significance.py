import numpy as np
import pytest

from jarvis_m.eval.significance import cluster_paired_bootstrap, compare_arms, holm, non_inferior


def test_identical_arms_have_no_difference():
    x = np.linspace(0, 1, 40)
    r = cluster_paired_bootstrap(x, x, [i // 4 for i in range(40)], n=500)
    assert r["delta"] == 0 and r["ci_low"] == r["ci_high"] == 0 and r["p"] == 1.0
    assert r["n_questions"] == 40 and r["n_clusters"] == 10


def test_clear_difference_is_significant_and_seeded():
    rng = np.random.default_rng(0)
    b = rng.random(200)
    a = b + 0.2 + 0.01 * rng.standard_normal(200)
    clusters = [i // 5 for i in range(200)]
    r = cluster_paired_bootstrap(a, b, clusters, n=2000, seed=1)
    assert r["delta"] == pytest.approx(float((a - b).mean()))
    assert 0.15 < r["ci_low"] < r["delta"] < r["ci_high"] < 0.25 and r["p"] < 0.01
    assert r == cluster_paired_bootstrap(a, b, clusters, n=2000, seed=1)
    flipped = cluster_paired_bootstrap(b, a, clusters, n=2000, seed=1)
    assert flipped["delta"] == pytest.approx(-r["delta"]) and flipped["ci_high"] < 0


def test_resampling_conversations_is_wider_than_resampling_questions():
    # every question in a conversation moves together: 8 conversations, not 80 samples
    rng = np.random.default_rng(3)
    effect = np.repeat(rng.standard_normal(8), 10)
    a, b = effect, np.zeros(80)
    by_conv = cluster_paired_bootstrap(a, b, np.repeat(np.arange(8), 10), n=4000)
    by_question = cluster_paired_bootstrap(a, b, np.arange(80), n=4000)
    width = lambda r: r["ci_high"] - r["ci_low"]  # noqa: E731
    assert width(by_conv) > 2 * width(by_question)


def test_seeds_are_resampled_per_arm():
    rng = np.random.default_rng(0)
    base = rng.random(60)
    clusters = [i // 3 for i in range(60)]
    # three seeds that disagree a lot: the seed noise must show up in the interval
    noisy = np.stack([base + 0.3, base - 0.3, base + 0.0])
    steady = np.stack([base, base, base])
    r_noisy = cluster_paired_bootstrap(noisy, base, clusters, n=3000)
    r_steady = cluster_paired_bootstrap(steady, base, clusters, n=3000)
    assert r_noisy["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r_steady["ci_low"] == pytest.approx(0.0, abs=1e-12) and r_steady["ci_high"] == pytest.approx(0.0, abs=1e-12)
    assert r_noisy["ci_high"] - r_noisy["ci_low"] > 0.2 and r_noisy["p"] > 0.3
    with pytest.raises(ValueError):
        cluster_paired_bootstrap(noisy[:, :10], base, clusters)


def test_macro_average_over_strata():
    # domain A: 90 questions, +0.0; domain B: 10 questions, +1.0 -> pooled 0.1, macro 0.5
    a = np.concatenate([np.zeros(90), np.ones(10)])
    b = np.zeros(100)
    clusters = list(range(100))
    strata = ["A"] * 90 + ["B"] * 10
    assert cluster_paired_bootstrap(a, b, clusters, n=200)["delta"] == pytest.approx(0.1)
    r = cluster_paired_bootstrap(a, b, clusters, n=200, strata=strata)
    assert r["delta"] == pytest.approx(0.5) and r["ci_low"] == pytest.approx(0.5)
    out = dict(compare_arms({"ref": [b], "x": [a], "y": [b]}, clusters, "ref", strata, n=200))
    assert set(out) == {"x", "y"} and out["y"]["delta"] == 0
    with pytest.raises(KeyError):
        compare_arms({"x": [a]}, clusters, "ref")


def test_holm_and_non_inferiority():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0]) and holm([]) == []
    adj = holm([0.2, 0.001, 0.03, 0.04])
    assert adj == pytest.approx([0.2, 0.004, 0.09, 0.09]) and all(a >= p for a, p in zip(adj, [0.2, 0.001, 0.03, 0.04]))
    assert non_inferior(-0.005, 0.01) and not non_inferior(-0.02, 0.01) and not non_inferior(-0.01, 0.01)
    with pytest.raises(ValueError):
        non_inferior(0.0, -0.1)
