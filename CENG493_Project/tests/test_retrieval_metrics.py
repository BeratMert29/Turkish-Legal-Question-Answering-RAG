"""
test_retrieval_metrics.py — unit tests for evaluation/retrieval_metrics.py.

Tests exercise the pure-function logic of compute_all_metrics:
  * Recall@5 / Recall@10
  * MRR
  * nDCG@10
  * hit_at_k (custom hit-rate metric)
  * capped_recall_at_k
  * Precision@5 / Precision@10
  * total_queries / num_queries counting

ranx is the only non-stdlib dependency used by this module and is listed
in requirements-test.txt, so no stubbing is required here.
"""

from __future__ import annotations

import pytest
from evaluation.retrieval_metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single(query_id: str, retrieved: list[str], relevant: list[str]) -> list[dict]:
    return [{"query_id": query_id, "retrieved": retrieved, "relevant": relevant}]


# ---------------------------------------------------------------------------
# Edge-case / boundary tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_returns_none_metrics(self):
        """No gold-labeled query: metrics are unknown (None), never 0.0."""
        result = compute_all_metrics([])
        assert result["num_queries"] == 0
        assert result["total_queries"] == 0
        for key in ("recall_at_5", "recall_at_10", "mrr", "ndcg_at_10",
                    "hit_at_5", "precision_at_5", "precision_at_10"):
            assert result[key] is None, key

    def test_query_with_no_relevant_docs_is_excluded(self):
        results = _single("q1", ["c1", "c2"], relevant=[])
        result = compute_all_metrics(results)
        assert result["num_queries"] == 0
        assert result["total_queries"] == 1
        # No evaluable queries: unknown, not zero
        assert result["mrr"] is None

    def test_all_queries_have_no_relevant_docs(self):
        results = [
            {"query_id": "q1", "retrieved": ["c1"], "relevant": []},
            {"query_id": "q2", "retrieved": ["c2"], "relevant": []},
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] == 0
        assert result["total_queries"] == 2

    def test_empty_retrieved_list_scores_zero(self):
        results = _single("q1", retrieved=[], relevant=["c1"])
        result = compute_all_metrics(results)
        assert result["num_queries"] == 1
        assert result["recall_at_5"] == 0.0
        assert result["mrr"] == 0.0


# ---------------------------------------------------------------------------
# Correctness tests — single query
# ---------------------------------------------------------------------------

class TestSingleQueryMetrics:
    def test_perfect_retrieval_at_rank1(self):
        results = _single("q1", ["c1", "c2", "c3"], relevant=["c1"])
        result = compute_all_metrics(results)
        assert result["mrr"] == pytest.approx(1.0)
        assert result["hit_at_5"] == pytest.approx(1.0)
        assert result["hit_at_10"] == pytest.approx(1.0)
        assert result["num_queries"] == 1

    def test_relevant_at_rank2_gives_mrr_half(self):
        """Relevant doc at rank 2 → MRR = 0.5."""
        results = _single("q1", ["cx", "c1", "c2"], relevant=["c1"])
        result = compute_all_metrics(results)
        assert result["mrr"] == pytest.approx(0.5, abs=0.01)

    def test_relevant_at_rank3_gives_mrr_third(self):
        results = _single("q1", ["cx", "cy", "c1"], relevant=["c1"])
        result = compute_all_metrics(results)
        assert result["mrr"] == pytest.approx(1 / 3, abs=0.01)

    def test_no_relevant_in_retrieved(self):
        results = _single("q1", ["c1", "c2", "c3"], relevant=["c99"])
        result = compute_all_metrics(results)
        assert result["mrr"] == pytest.approx(0.0)
        assert result["hit_at_5"] == 0.0
        assert result["recall_at_5"] == pytest.approx(0.0)

    def test_precision_at_5_two_hits(self):
        results = _single(
            "q1",
            ["c1", "c2", "c3", "c4", "c5"],
            relevant=["c1", "c3"],
        )
        result = compute_all_metrics(results)
        assert result["precision_at_5"] == pytest.approx(2 / 5)

    def test_capped_recall_all_relevant_in_top5(self):
        """When all relevant docs appear in top-5, capped_recall_at_5 = 1.0."""
        results = _single(
            "q1",
            ["c1", "c2", "c3", "c4", "c5"],
            relevant=["c1", "c3"],
        )
        result = compute_all_metrics(results)
        # hits_5=2, min(5, 2)=2 → capped_recall = 2/2 = 1.0
        assert result["capped_recall_at_5"] == pytest.approx(1.0)

    def test_capped_recall_partial_hit(self):
        """Only 1 of 3 relevant docs in top-5 → capped_recall = 1/3."""
        results = _single(
            "q1",
            ["c1", "cx", "cy", "cz", "cw"],
            relevant=["c1", "c2", "c3"],
        )
        result = compute_all_metrics(results)
        # hits_5=1, min(5,3)=3 → 1/3
        assert result["capped_recall_at_5"] == pytest.approx(1 / 3, abs=0.01)

    def test_precision_at_10_with_fewer_retrieved(self):
        """Precision@10 denominator is always 10, even with <10 retrieved."""
        results = _single("q1", ["c1", "c2"], relevant=["c1"])
        result = compute_all_metrics(results)
        # hits_10=1, precision_10 = 1/10
        assert result["precision_at_10"] == pytest.approx(1 / 10)


# ---------------------------------------------------------------------------
# Multi-query aggregation tests
# ---------------------------------------------------------------------------

class TestMultiQueryAggregation:
    def test_two_queries_source_hit_average(self):
        """One hit, one miss → hit_at_5 = 0.5."""
        results = [
            {"query_id": "q1", "retrieved": ["c1"], "relevant": ["c1"]},
            {"query_id": "q2", "retrieved": ["cx"], "relevant": ["cy"]},
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] == 2
        assert result["total_queries"] == 2
        assert result["hit_at_5"] == pytest.approx(0.5)

    def test_mixed_relevant_and_no_relevant(self):
        """Queries with empty relevant sets are excluded from metrics."""
        results = [
            {"query_id": "q1", "retrieved": ["c1"], "relevant": ["c1"]},
            {"query_id": "q2", "retrieved": ["c2"], "relevant": []},
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] == 1
        assert result["total_queries"] == 2
        # q2 is excluded; metrics reflect only q1
        assert result["mrr"] == pytest.approx(1.0)

    def test_all_perfect_retrieval(self):
        results = [
            {"query_id": f"q{i}", "retrieved": [f"c{i}"], "relevant": [f"c{i}"]}
            for i in range(5)
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] == 5
        assert result["hit_at_5"] == pytest.approx(1.0)
        assert result["mrr"] == pytest.approx(1.0)

    def test_all_misses(self):
        results = [
            {"query_id": f"q{i}", "retrieved": ["cx"], "relevant": [f"c{i}"]}
            for i in range(3)
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] == 3
        assert result["hit_at_5"] == pytest.approx(0.0)
        assert result["mrr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Return-value schema tests
# ---------------------------------------------------------------------------

class TestReturnSchema:
    def test_all_expected_keys_present(self):
        results = _single("q1", ["c1"], relevant=["c1"])
        result = compute_all_metrics(results)
        expected_keys = {
            "recall_at_5",
            "recall_at_10",
            "mrr",
            "ndcg_at_10",
            "hit_at_5",
            "hit_at_10",
            "capped_recall_at_5",
            "capped_recall_at_10",
            "precision_at_5",
            "precision_at_10",
            "num_queries",
            "total_queries",
        }
        assert expected_keys <= set(result.keys()), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )

    def test_duplicate_query_id_raises(self):
        results = _single("q1", ["c1"], ["c1"]) + _single("q1", ["c2"], ["c2"])
        with pytest.raises(ValueError, match="duplicate query_id"):
            compute_all_metrics(results)

    def test_all_metric_values_are_floats_or_int(self):
        results = _single("q1", ["c1", "c2"], relevant=["c1"])
        result = compute_all_metrics(results)
        for key, val in result.items():
            assert isinstance(val, (int, float)), (
                f"Key '{key}' has unexpected type {type(val)}"
            )

    def test_metrics_in_unit_range(self):
        """All ratio metrics must be in [0.0, 1.0]."""
        results = [
            {"query_id": "q1", "retrieved": ["c1", "c2", "c3"], "relevant": ["c1"]},
            {"query_id": "q2", "retrieved": ["cx"], "relevant": ["cy"]},
        ]
        result = compute_all_metrics(results)
        ratio_keys = [
            "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10",
            "hit_at_5", "hit_at_10",
            "capped_recall_at_5", "capped_recall_at_10",
            "precision_at_5", "precision_at_10",
        ]
        for key in ratio_keys:
            assert 0.0 <= result[key] <= 1.0, (
                f"Metric '{key}' = {result[key]} is outside [0, 1]"
            )

    def test_num_queries_leq_total_queries(self):
        results = [
            {"query_id": "q1", "retrieved": ["c1"], "relevant": ["c1"]},
            {"query_id": "q2", "retrieved": ["c2"], "relevant": []},
        ]
        result = compute_all_metrics(results)
        assert result["num_queries"] <= result["total_queries"]


# ---------------------------------------------------------------------------
# Duplicate chunk ids: every metric must score the same (deduplicated) ranking
# ---------------------------------------------------------------------------

class TestDuplicateChunkIds:
    @pytest.mark.parametrize("retrieved,relevant,mrr", [
        (["x", "R", "x"], ["R"], 0.5),   # repeat must not push x below R
        (["R", "x", "R"], ["R"], 1.0),   # repeat must not push R down
    ])
    def test_mrr_uses_first_occurrence(self, retrieved, relevant, mrr):
        assert compute_all_metrics(_single("q", retrieved, relevant))["mrr"] == pytest.approx(mrr)

    def test_recall_and_hit_agree_after_dedup(self):
        # deduplicated ranking: a, b, c, R -> R at rank 4 (inside top-5)
        res = compute_all_metrics(_single("q", ["a", "a", "b", "b", "c", "R"], ["R"]))
        assert res["recall_at_5"] == pytest.approx(1.0)
        assert res["hit_at_5"] == pytest.approx(1.0)
        assert res["mrr"] == pytest.approx(0.25)

    def test_duplicate_relevant_ids_count_once(self):
        res = compute_all_metrics(_single("q", ["R", "x"], ["R", "R"]))
        assert res["recall_at_5"] == pytest.approx(1.0)
        assert res["capped_recall_at_5"] == pytest.approx(1.0)


class TestArticleLevel:
    def test_two_chunks_of_one_article_count_once(self):
        from evaluation.retrieval_metrics import compute_article_metrics
        mi = [{"query_id": "q",
               "relevant_articles": ["TCK||86"],
               # ranks: TCK 85 (two chunks) then TCK 86 -> article rank 2
               "retrieved_articles": ["TCK||85", "TCK||86", "chunk::x"]}]
        res = compute_article_metrics(mi)
        assert res["mrr"] == pytest.approx(0.5) and res["hit_at_5"] == 1.0

    def test_prepare_metric_input_maps_chunks_to_articles(self):
        from types import SimpleNamespace
        from pipeline.evaluation import prepare_metric_input
        qa = [SimpleNamespace(query_id="q", source="TCK", madde_no="86"),
              SimpleNamespace(query_id="r", source="TCK", madde_no=None)]
        chunks = [[{"chunk_id": "a1"}, {"chunk_id": "a2"}, {"chunk_id": "b1"}, {"chunk_id": "z"}],
                  [{"chunk_id": "b1"}]]
        arts = {"a1": "TCK||85", "a2": "TCK||85", "b1": "TCK||86"}
        mi, _ = prepare_metric_input(qa, chunks, {"q": ["b1"], "r": ["b1"]}, arts)
        assert mi[0]["relevant_articles"] == ["TCK||86"]
        assert mi[0]["retrieved_articles"] == ["TCK||85", "TCK||86", "chunk::z"]
        assert mi[1]["relevant_articles"] == ["TCK||86"]  # from gold chunks
