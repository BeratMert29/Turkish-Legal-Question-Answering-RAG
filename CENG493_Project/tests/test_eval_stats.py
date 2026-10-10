import numpy as np
import pytest

from evaluation.stats import bootstrap_ci, paired_bootstrap


def test_ci_contains_mean_and_seeded():
    vals = list(np.random.default_rng(0).uniform(0, 1, 100))
    r1, r2 = bootstrap_ci(vals), bootstrap_ci(vals)
    assert r1 == r2
    assert r1["ci_low"] <= r1["mean"] <= r1["ci_high"]
    assert r1["n_resamples"] >= 1000 and r1["n"] == 100


def test_ci_min_resamples_enforced_and_none_dropped():
    r = bootstrap_ci([0.5, None, 0.7, float("nan")], n_resamples=10)
    assert r["n"] == 2 and r["n_resamples"] == 1000


def test_ci_empty():
    assert bootstrap_ci([])["mean"] is None


def test_ci_constant_values_zero_width():
    r = bootstrap_ci([0.3] * 20)
    assert r["ci_low"] == pytest.approx(0.3) and r["ci_high"] == pytest.approx(0.3)


def test_paired_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 1, 80)
    b = a + 0.2
    r = paired_bootstrap(list(a), list(b))
    assert r["mean_diff"] == pytest.approx(0.2)
    assert r["significant"] and r["ci_low"] > 0 and r["p_value"] < 0.05


def test_paired_no_difference_not_significant():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 1, 60)
    b = a + rng.normal(0, 0.1, 60)
    r = paired_bootstrap(a, b)
    assert not r["significant"]


def test_paired_dict_alignment_and_drop_none():
    a = {"q1": 0.1, "q2": 0.2, "q3": None, "q4": 0.4}
    b = {"q2": 0.3, "q1": 0.2, "q3": 0.5, "q5": 1.0}
    r = paired_bootstrap(a, b)
    assert r["n"] == 2 and r["mean_diff"] == pytest.approx(0.1)


def test_paired_length_mismatch_raises():
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1])


def test_holm_adjust_matches_hand_calculation():
    from evaluation.stats import holm_adjust
    adj = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    # sorted: a .01*3=.03, c .03*2=.06, b .04*1=.04 -> monotone max .06
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)
    assert adj["d"] is None


def test_compare_stages_aligns_by_query_id_and_skips_missing_pairs():
    from pipeline.evaluation import compare_stages
    base = [{"query_id": f"q{i}", "f1": 0.2} for i in range(30)]
    # reversed order: alignment must use query_id, not position
    llm = [{"query_id": f"q{i}", "f1": 0.5} for i in reversed(range(30))]
    out = compare_stages({"base": base, "llm_ft": llm}, metrics=("f1", "judge_answer"))
    r = out["f1"]["base->llm_ft"]
    assert r["mean_diff"] == pytest.approx(0.3)
    assert r["n"] == 30 and r["significant_holm"]
    assert "judge_answer" not in out          # no values -> no comparison
    assert set(out["f1"]) == {"base->llm_ft"}  # pairs with an absent stage skipped


def test_cohen_kappa_known_values():
    from evaluation.stats import cohen_kappa
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # po = 0.8, pe = 0.5*0.5 + 0.5*0.5 = 0.5 -> kappa 0.6
    a = [1] * 5 + [0] * 5
    b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(0.6)
    # weighted: a near miss (0.5 vs 1) costs less than a far miss (0 vs 1)
    labels = [0, 0.5, 1]
    near = cohen_kappa([0, 0.5, 1, 1], [0, 0.5, 1, 0.5], labels, "linear")
    far = cohen_kappa([0, 0.5, 1, 1], [0, 0.5, 1, 0], labels, "linear")
    assert near > far
    assert cohen_kappa([1, None], [1, 0]) is None
