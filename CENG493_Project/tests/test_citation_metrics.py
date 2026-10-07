"""evaluation/citation_metrics.py: article-level citations vs chance."""

import pytest

from evaluation.citation_metrics import citation_scores, compute_citation_metrics, cited_positions

ARTS = {"c1": "TCK||86", "c2": "TCK||87", "c3": "TMK||5", "c4": "TCK||86"}
CTX = [{"chunk_id": c} for c in ("c1", "c2", "c3", "c4")]


def test_cited_positions_dedup_in_order():
    assert cited_positions("a [Kaynak 2] b [kaynak 1] c [Kaynak 2]") == [2, 1]


def test_precision_recall_and_random_baseline():
    r = citation_scores("Ceza [Kaynak 1]. Diğer [Kaynak 3].", CTX, {"TCK||86"}, ARTS)
    assert r["precision"] == 0.5          # c1 gold, c3 not
    assert r["random_precision"] == 0.5   # c1, c4 gold out of 4
    assert r["recall"] == 1.0             # the one gold article in context is cited


def test_out_of_range_citation_counts_as_invalid():
    r = citation_scores("x [Kaynak 9]", CTX, {"TCK||86"}, ARTS)
    assert r["precision"] is None and r["n_invalid"] == 1 and r["recall"] == 0.0


def test_summary_native_vs_injected():
    preds = [{"query_id": "q", "predicted": "x [Kaynak 1] [Kaynak 2] [Kaynak 3]",
              "predicted_native": "x [Kaynak 1]", "retrieved_chunks": CTX}]
    summary, per_q = compute_citation_metrics(preds, {"q": ["TCK||86"]}, ARTS)
    assert summary["native"]["precision"] == 1.0
    assert summary["injected"]["precision"] == pytest.approx(1 / 3)
    assert per_q["q"]["cite_precision_native"] == 1.0
