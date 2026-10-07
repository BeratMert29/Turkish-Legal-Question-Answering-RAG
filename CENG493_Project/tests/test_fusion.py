"""Hybrid (linear) and RRF fusion in retrieval/retriever.py."""

import numpy as np
import pytest

from retrieval.retriever import Retriever, _minmax, rrf_fuse


class _Index:
    """Inner-product index over fixed vectors (stands in for faiss)."""

    def __init__(self, vecs):
        self.vecs = np.asarray(vecs, dtype=np.float32)

    def search(self, q, k):
        sims = q @ self.vecs.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx

    def reconstruct(self, i):
        return self.vecs[i]


class _Emb:
    def __init__(self, q):
        self.q = np.asarray(q, dtype=np.float32)

    def encode(self, texts, is_query=False, show_progress=True):
        return np.tile(self.q, (len(texts), 1))


class _BM25:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float32)
        self.metadata = [{} for _ in scores]

    def get_scores(self, query):
        return self.scores


def _retriever(vecs, q):
    r = Retriever(_Emb(q))
    r.index = _Index(vecs)
    r.metadata = [{"chunk_id": f"c{i}", "text": "", "source": "", "doc_id": ""}
                  for i in range(len(vecs))]
    return r


def test_rrf_ties_broken_by_best_rank_then_dense():
    fused = rrf_fuse([{7: 1, 3: 2}, {5: 1, 3: 3}], rrf_k=60)
    # 3 is in both lists -> first; 7 (dense #1) and 5 (bm25 #1) tie -> dense first
    assert [d for d, _ in fused] == [3, 7, 5]


def test_minmax_constant_maps_to_one():
    assert _minmax({1: 0.3, 2: 0.3}) == {1: 1.0, 2: 1.0}


def test_hybrid_uses_bm25_score_of_dense_candidates_outside_bm25_pool():
    # doc0: best dense, decent BM25 (rank 3 of 3 BM25 scores > 0);
    # with candidate_pool=2 doc0 is outside the BM25 pool but must keep its BM25.
    vecs = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0], [0.5, 0.5]]
    r = _retriever(vecs, q=[1.0, 0.0])
    bm25 = _BM25([0.6, 0.0, 1.0, 0.9])
    out = r.batch_hybrid_retrieve(["q"], bm25, alpha=0.5, top_k=4, candidate_pool=2)[0]
    ids = [c["chunk_id"] for c in out]
    assert ids[0] == "c0"
    # doc2 was only in the BM25 pool: its dense score comes from reconstruct (0.0)
    by = {c["chunk_id"]: c["score"] for c in out}
    assert by["c2"] == pytest.approx(0.5)  # dense_norm 0, bm25_norm 1


def test_single_hybrid_matches_batch():
    vecs = [[1.0, 0.0], [0.0, 1.0]]
    r = _retriever(vecs, q=[0.8, 0.6])
    bm25 = _BM25([0.1, 0.9])
    assert r.hybrid_retrieve("q", bm25, top_k=2, candidate_pool=2) == \
        r.batch_hybrid_retrieve(["q"], bm25, top_k=2, candidate_pool=2)[0]


@pytest.mark.parametrize("name,expected", [
    ("intfloat/multilingual-e5-large", True),
    ("intfloat/e5-base-v2", True),
    ("BAAI/bge-m3", False),
    ("/home/u/run-3fe51a/bge-m3-turkish-legal", False),
    ("C:\\models\\e5x\\bge-m3", False),
])
def test_e5_prefix_detection_uses_model_folder_name(name, expected):
    from retrieval.embedder import uses_e5_prefixes
    assert uses_e5_prefixes(name) is expected
