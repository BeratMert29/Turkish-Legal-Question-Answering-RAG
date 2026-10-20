"""Tests documenting BM25 get_scores normalization behavior.

Finding: bm25_retriever.get_scores() returns min-max normalised scores over the
full corpus.  retriever._hybrid_one() applies a second min-max normalisation
over the candidate subset.  This is NOT a bug:

  1. Global normalization (get_scores) ensures the >0 filter in _hybrid_one
     correctly discards documents with zero raw BM25 relevance.
  2. Local normalization (_minmax over candidates) puts BM25 on the same
     [0, 1] scale as dense scores *within the candidate pool*, which is the
     correct behavior for linear fusion.

These tests verify the current behavior is consistent and document the
two-stage design.
"""

import numpy as np
import pytest


class TestBM25GetScoresNormalization:
    """BM25Index.get_scores returns min-max normalised values."""

    def test_returns_0_1_range(self):
        """Scores from get_scores are in [0, 1]."""
        from retrieval.bm25_retriever import BM25Index
        idx = BM25Index()
        idx.build([
            {"text": "hukuk davasi mahkeme karari", "chunk_id": "c0"},
            {"text": "ceza kanunu madde bes", "chunk_id": "c1"},
            {"text": "tamamen alakasiz konu", "chunk_id": "c2"},
        ])
        scores = idx.get_scores("hukuk davasi")
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_max_is_one_when_relevant(self):
        """The most relevant document gets score 1.0 after normalization."""
        from retrieval.bm25_retriever import BM25Index
        idx = BM25Index()
        idx.build([
            {"text": "hukuk davasi mahkeme karari", "chunk_id": "c0"},
            {"text": "ceza kanunu madde bes alti", "chunk_id": "c1"},
            {"text": "vergi usul kanunu hukumleri", "chunk_id": "c2"},
            {"text": "tamamen farkli konu basligi", "chunk_id": "c3"},
        ])
        scores = idx.get_scores("hukuk davasi")
        # At least one doc should get the top normalised score if any term matches
        if scores.max() > 0:
            assert scores.max() == pytest.approx(1.0)

    def test_all_zeros_when_no_match(self):
        """When no document matches, all scores are 0."""
        from retrieval.bm25_retriever import BM25Index
        idx = BM25Index()
        idx.build([
            {"text": "hukuk davasi", "chunk_id": "c0"},
            {"text": "ceza kanunu", "chunk_id": "c1"},
        ])
        scores = idx.get_scores("zzzzxyzzyxyz")
        np.testing.assert_array_equal(scores, np.zeros_like(scores))


class TestMinmaxLocalNormalization:
    """retriever._minmax normalises a subset to [0, 1]."""

    def test_basic(self):
        from retrieval.retriever import _minmax
        result = _minmax({0: 0.3, 1: 0.7, 2: 1.0})
        assert result[0] == pytest.approx(0.0)
        assert result[2] == pytest.approx(1.0)

    def test_constant_maps_to_one(self):
        from retrieval.retriever import _minmax
        result = _minmax({0: 0.5, 1: 0.5})
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(1.0)

    def test_empty(self):
        from retrieval.retriever import _minmax
        assert _minmax({}) == {}
