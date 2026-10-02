"""
tests/test_source_metrics.py — Tests for law-level source-hit and source-precision metrics.

All tests run offline: no network, GPU, FAISS, or Ollama.
compute_source_hit_metrics is pure Python (no ranx dependency).
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.retrieval_metrics import compute_source_hit_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(query_id, source_law, retrieved_sources):
    """Build a minimal result dict for compute_source_hit_metrics."""
    return {
        "query_id": query_id,
        "source_law": source_law,
        "retrieved_sources": retrieved_sources,
    }


# ---------------------------------------------------------------------------
# compute_source_hit_metrics
# ---------------------------------------------------------------------------

class TestComputeSourceHitMetrics:
    def test_perfect_retrieval(self):
        """All top-5 from gold law → hit@5=1.0 and prec@5=1.0."""
        results = [
            _r("q1", "LawA", ["LawA"] * 5),
            _r("q2", "LawB", ["LawB"] * 5),
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_hit_at_5_all"]       == pytest.approx(1.0)
        assert m["source_precision_at_5_all"] == pytest.approx(1.0)
        assert m["source_labeled_queries"]    == 2

    def test_zero_retrieval(self):
        """No chunks from gold law → hit@5=0.0, prec@5=0.0."""
        results = [
            _r("q1", "LawA", ["LawB", "LawC", "LawB", "LawC", "LawB"]),
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_hit_at_5_all"]       == pytest.approx(0.0)
        assert m["source_precision_at_5_all"] == pytest.approx(0.0)
        assert m["source_labeled_queries"]    == 1

    def test_partial_retrieval(self):
        """2/5 from gold law → prec@5=0.4."""
        results = [
            _r("q1", "LawA", ["LawA", "LawB", "LawA", "LawC", "LawD"]),
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_hit_at_5_all"]       == pytest.approx(1.0)   # hit because >=1
        assert m["source_precision_at_5_all"] == pytest.approx(0.4)   # 2/5

    def test_hit_at_10_deeper_results(self):
        """Gold law chunk only at rank 8 → hit@10=1, hit@5=0."""
        results = [
            _r("q1", "LawA", ["LawB"] * 7 + ["LawA"] + ["LawC"] * 2),
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_hit_at_5_all"]  == pytest.approx(0.0)
        assert m["source_hit_at_10_all"] == pytest.approx(1.0)

    def test_unknown_source_excluded(self):
        """Queries with empty source_law must be skipped (not counted)."""
        results = [
            _r("q1", "",     ["LawA"] * 5),  # unknown source → skip
            _r("q2", "LawB", ["LawB"] * 5),  # known source → include
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_labeled_queries"] == 1
        assert m["source_hit_at_5_all"]    == pytest.approx(1.0)
        assert m["total_queries"]          == 2

    def test_empty_retrieved(self):
        """No retrieved chunks → hit=0, prec=0."""
        results = [_r("q1", "LawA", [])]
        m = compute_source_hit_metrics(results)
        assert m["source_hit_at_5_all"]       == pytest.approx(0.0)
        assert m["source_precision_at_5_all"] == pytest.approx(0.0)

    def test_empty_results(self):
        """Empty list → all zeros."""
        m = compute_source_hit_metrics([])
        assert m["source_labeled_queries"]     == 0
        assert m["source_hit_at_5_all"]        == pytest.approx(0.0)
        assert m["source_hit_at_10_all"]       == pytest.approx(0.0)
        assert m["source_precision_at_5_all"]  == pytest.approx(0.0)
        assert m["source_precision_at_10_all"] == pytest.approx(0.0)

    def test_mixed_labeled_unlabeled(self):
        """Mean computed only over source-labeled queries."""
        results = [
            _r("q1", "LawA", ["LawA"] * 5),   # perfect
            _r("q2", "",      ["LawB"] * 5),   # unlabeled
            _r("q3", "LawC",  ["LawD"] * 5),   # miss
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_labeled_queries"] == 2
        assert m["source_hit_at_5_all"]    == pytest.approx(0.5)   # 1 hit / 2

    def test_keys_present(self):
        results = [_r("q1", "LawA", ["LawA"])]
        m = compute_source_hit_metrics(results)
        expected_keys = {
            "source_hit_at_5_all", "source_hit_at_10_all",
            "source_mrr_all",
            "source_precision_at_5_all", "source_precision_at_10_all",
            "source_labeled_queries", "total_queries",
        }
        assert expected_keys.issubset(m.keys())

    def test_precision_at_10(self):
        """prec@10 = 3/10 when 3 gold-law chunks in top-10."""
        srcs = ["LawA", "LawB", "LawA", "LawC", "LawA",
                "LawB", "LawB", "LawC", "LawC", "LawB"]
        results = [_r("q1", "LawA", srcs)]
        m = compute_source_hit_metrics(results)
        assert m["source_precision_at_10_all"] == pytest.approx(3 / 10)


class TestSourceMRR:
    """Tests for source-level MRR (reciprocal rank of first chunk from gold law)."""

    def test_first_rank(self):
        """Gold law chunk at rank 1 → MRR = 1.0."""
        results = [_r("q1", "LawA", ["LawA", "LawB", "LawC"])]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(1.0)

    def test_second_rank(self):
        """Gold law chunk at rank 2 → MRR = 0.5."""
        results = [_r("q1", "LawA", ["LawB", "LawA", "LawC"])]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(0.5)

    def test_third_rank(self):
        """Gold law chunk at rank 3 → MRR = 1/3."""
        results = [_r("q1", "LawA", ["LawB", "LawC", "LawA"])]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(1 / 3)

    def test_not_retrieved(self):
        """Gold law never retrieved → MRR contribution = 0."""
        results = [_r("q1", "LawA", ["LawB", "LawC", "LawD"])]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(0.0)

    def test_mean_over_queries(self):
        """MRR is mean over all source-labeled queries."""
        results = [
            _r("q1", "LawA", ["LawA", "LawB"]),         # RR = 1.0
            _r("q2", "LawB", ["LawA", "LawB"]),         # RR = 0.5
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(0.75)

    def test_empty_results(self):
        """Empty list → MRR = 0.0."""
        assert compute_source_hit_metrics([])["source_mrr_all"] == pytest.approx(0.0)

    def test_unknown_source_excluded_from_mrr(self):
        """Queries with empty source_law must not affect MRR."""
        results = [
            _r("q1", "",     ["LawA"]),   # unknown, skipped
            _r("q2", "LawA", ["LawA"]),   # RR = 1.0
        ]
        m = compute_source_hit_metrics(results)
        assert m["source_mrr_all"] == pytest.approx(1.0)
