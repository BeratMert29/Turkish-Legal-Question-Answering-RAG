import pytest

from evaluation.final_score import (
    compute_all_scenario_scores, compute_scenario1_score, compute_scenario2_score,
    compute_scenario3_score,
)

R, Q = {"mrr": 0.5}, {"f1": 0.4}


def test_full_scenarios_match_rubric():
    r = compute_all_scenario_scores(R, Q, 0.8, 0.6, {"relevancy": 0.9, "coherence": 0.7})
    assert r["scenario1"] == pytest.approx(0.35 * 0.5 + 0.40 * 0.4 + 0.25 * 0.8)
    assert r["scenario2"] == pytest.approx(0.7 * 0.4 + 0.3 * 0.6)
    assert r["scenario3"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert r["components"]["scenario1"]["complete"]


def test_missing_faithfulness_renormalises_not_zero():
    r = compute_all_scenario_scores(R, Q, None, 0.6)
    exp = (0.35 * 0.5 + 0.40 * 0.4) / 0.75
    assert r["scenario1"] == pytest.approx(exp)
    c = r["components"]["scenario1"]
    assert c["missing"] == ["faithfulness"] and sum(c["weights"].values()) == pytest.approx(1)


def test_no_f1_proxy_for_missing_similarity_or_relevancy():
    r = compute_all_scenario_scores(R, Q, 0.8)
    assert r["scenario2"] == pytest.approx(0.4)  # F1 only
    assert r["components"]["scenario2"]["missing"] == ["semantic_similarity"]
    assert r["scenario3"] == pytest.approx(0.8)  # only NLI faithfulness available
    assert set(r["components"]["scenario3"]["missing"]) == {"relevancy", "coherence"}


def test_coherence_missing_not_zero():
    s = compute_scenario3_score(0.8, 0.6, None)
    assert s == pytest.approx(0.7)


def test_no_silent_nli_to_judge_switch():
    llm = {"faithfulness": 0.1, "relevancy": 0.5, "coherence": 0.5}
    nli = compute_all_scenario_scores(R, Q, 0.9, 0.5, llm, faithfulness_source="nli")
    judge = compute_all_scenario_scores(R, Q, 0.9, 0.5, llm, faithfulness_source="judge")
    assert nli["components"]["scenario1"]["used"]["faithfulness"] == 0.9
    assert judge["components"]["scenario1"]["used"]["faithfulness"] == 0.1
    # judge source requested but absent -> missing, NOT replaced by NLI
    absent = compute_all_scenario_scores(R, Q, 0.9, 0.5, {}, faithfulness_source="judge")
    assert "faithfulness" in absent["components"]["scenario1"]["missing"]


def test_all_missing_gives_none():
    assert compute_scenario3_score(None, None, None) is None
    r = compute_all_scenario_scores({}, {}, None)
    assert r["scenario1"] is None and r["scenario2"] is None and r["scenario3"] is None


def test_n_by_component_recorded_and_bad_source():
    r = compute_all_scenario_scores(R, Q, 0.8, None, None, n_by_component={"f1": 100, "faithfulness": 20})
    assert r["components"]["scenario1"]["n"]["faithfulness"] == 20
    with pytest.raises(ValueError):
        compute_all_scenario_scores(R, Q, 0.8, faithfulness_source="x")


def test_scalar_helpers():
    assert compute_scenario1_score(R, Q, None) == pytest.approx((0.175 + 0.16) / 0.75)
    assert compute_scenario2_score(Q, None) == pytest.approx(0.4)
