import json
import numpy as np
import faiss
from pathlib import Path
from typing import TypedDict
import config

class RetrievedChunk(TypedDict):
    text: str
    doc_id: str
    source: str
    score: float
    chunk_id: str

def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Min-max normalise to [0, 1]; a constant list maps to 1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rrf_fuse(rankings: list[dict[int, int]], rrf_k: int) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of 1-based rankings ``{doc: rank}``.

    Returns ``[(doc, score)]`` by descending ``sum(1 / (rrf_k + rank))``.
    Ties (e.g. dense-only #1 vs BM25-only #1) are broken by the best single
    rank, then by the earlier list (dense first), then by doc index, so the
    order never depends on set iteration.
    """
    docs = list(dict.fromkeys(d for r in rankings for d in r))
    scores = {d: sum(1.0 / (rrf_k + r[d]) for r in rankings if d in r) for d in docs}

    def _key(d):
        ranks = [r.get(d, float("inf")) for r in rankings]
        return (-scores[d], min(ranks), ranks, d)

    return [(d, scores[d]) for d in sorted(docs, key=_key)]


class Retriever:
    def __init__(self, embedder, index_path=None, metadata_path=None):
        self.embedder = embedder
        self.index = None
        self.metadata: list[dict] = []
        if index_path and metadata_path:
            self.load_index(index_path, metadata_path)

    def build_index(self, texts: list[str], metadata: list[dict]) -> None:
        """Encode texts and build FAISS IndexFlatIP."""
        if len(texts) != len(metadata):
            raise ValueError(f"texts and metadata must have same length: {len(texts)} vs {len(metadata)}")
        embeddings = self.embedder.encode(texts, is_query=False)
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
        self.index.add(embeddings.astype(np.float32))
        self.metadata = metadata

    def save_index(self, index_path, metadata_path) -> None:
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(metadata_path, "w", encoding="utf-8") as f:
            for item in self.metadata:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load_index(self, index_path, metadata_path) -> None:
        from utils import read_jsonl
        self.index = faiss.read_index(str(index_path))
        self.metadata = list(read_jsonl(metadata_path))
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"Index/metadata mismatch: {self.index.ntotal} vectors vs {len(self.metadata)} metadata entries"
            )
        if self.index.d != config.EMBEDDING_DIM:
            raise ValueError(
                f"Index dimension mismatch: index has d={self.index.d}, config expects {config.EMBEDDING_DIM}. "
                f"Rebuild the index with the current embedding model."
            )

    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> list[RetrievedChunk]:
        """Retrieve top_k chunks for a single query."""
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        q_emb = self.embedder.encode([query], is_query=True, show_progress=False)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append(RetrievedChunk(
                text=meta.get("text", ""),
                doc_id=meta.get("doc_id", ""),
                source=meta.get("source", ""),
                score=float(score),
                chunk_id=meta.get("chunk_id", ""),
            ))
        return results

    def batch_retrieve(self, queries: list[str],
                       top_k: int = config.TOP_K_RETRIEVAL) -> list[list[RetrievedChunk]]:
        """Retrieve top_k chunks for all queries in one embedding call."""
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        q_embs = self.embedder.encode(queries, is_query=True)
        scores_all, indices_all = self.index.search(q_embs.astype(np.float32), top_k)
        results = []
        for scores, indices in zip(scores_all, indices_all):
            chunks = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.metadata[idx]
                chunks.append(RetrievedChunk(
                    text=meta.get("text", ""),
                    doc_id=meta.get("doc_id", ""),
                    source=meta.get("source", ""),
                    score=float(score),
                    chunk_id=meta.get("chunk_id", ""),
                ))
            results.append(chunks)
        return results

    def _check_bm25(self, bm25_index) -> None:
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if len(bm25_index.metadata) != len(self.metadata):
            raise ValueError(
                f"BM25/FAISS metadata count mismatch: {len(bm25_index.metadata)} vs {len(self.metadata)}. "
                f"Rebuild both indices from the same corpus."
            )

    def _chunk(self, i: int, score: float) -> RetrievedChunk:
        meta = self.metadata[i]
        return RetrievedChunk(
            text=meta.get("text", ""),
            doc_id=meta.get("doc_id", ""),
            source=meta.get("source", ""),
            score=float(score),
            chunk_id=meta.get("chunk_id", ""),
        )

    def _hybrid_one(self, q_emb: np.ndarray, dense_scores_row, dense_indices_row,
                    query: str, bm25_index, alpha: float, top_k: int,
                    candidate_pool: int) -> list[RetrievedChunk]:
        """Linear fusion over the union of the dense and BM25 candidate pools.

        Every candidate gets its true dense score (inner product with the
        query, reconstructed from the index when it was not in the dense pool)
        and its true BM25 score; each score list is min-max normalised over
        the candidates before blending, so both sit on the same [0, 1] scale.
        """
        dense: dict[int, float] = {
            int(i): float(sc) for sc, i in zip(dense_scores_row, dense_indices_row) if i != -1
        }
        bm25_all = np.asarray(bm25_index.get_scores(query), dtype=np.float32)
        top_bm25 = [int(i) for i in bm25_all.argsort()[::-1][:candidate_pool] if bm25_all[i] > 0]
        candidates = list(dict.fromkeys([*dense, *top_bm25]))
        for i in candidates:
            if i not in dense:
                dense[i] = float(np.dot(q_emb, self.index.reconstruct(i)))
        d_norm = _minmax({i: dense[i] for i in candidates})
        b_norm = _minmax({i: float(bm25_all[i]) for i in candidates})
        fused = {i: alpha * d_norm[i] + (1.0 - alpha) * b_norm[i] for i in candidates}
        # ties broken by candidate order (dense rank first), deterministic
        top = sorted(candidates, key=lambda i: -fused[i])[:top_k]
        return [self._chunk(i, fused[i]) for i in top]

    def hybrid_retrieve(self, query: str, bm25_index, alpha: float = 0.5,
                        top_k: int = None,
                        candidate_pool: int = None) -> list[RetrievedChunk]:
        """Hybrid dense+sparse retrieval for a single query.
        final_score = alpha * dense_norm + (1 - alpha) * bm25_norm over the
        union of the top-candidate_pool dense and BM25 candidates.
        """
        return self.batch_hybrid_retrieve([query], bm25_index, alpha=alpha, top_k=top_k,
                                          candidate_pool=candidate_pool)[0]

    def batch_hybrid_retrieve(self, queries: list[str], bm25_index,
                              alpha: float = 0.5,
                              top_k: int = None,
                              candidate_pool: int = None) -> list[list[RetrievedChunk]]:
        """Hybrid dense+sparse retrieval for a batch of queries (see _hybrid_one)."""
        self._check_bm25(bm25_index)
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        q_embs = self.embedder.encode(queries, is_query=True).astype(np.float32)
        dense_scores_all, dense_indices_all = self.index.search(q_embs, candidate_pool)
        return [
            self._hybrid_one(q_embs[q], dense_scores_all[q], dense_indices_all[q],
                             query, bm25_index, alpha, top_k, candidate_pool)
            for q, query in enumerate(queries)
        ]

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────

    def batch_rrf_retrieve(self, queries: list[str], bm25_index,
                           top_k: int = None,
                           rrf_k: int = None,
                           candidate_pool: int = None) -> list[list[RetrievedChunk]]:
        """Reciprocal Rank Fusion of dense + BM25 rankings.

        RRF is rank-based so it avoids the score-calibration issues of
        linear blending. Each document's fused score is:
            sum_over_lists( 1 / (rrf_k + rank) )
        """
        self._check_bm25(bm25_index)
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if rrf_k is None:
            rrf_k = config.RRF_K
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        q_embs = self.embedder.encode(queries, is_query=True)
        dense_scores_all, dense_indices_all = self.index.search(
            q_embs.astype(np.float32), candidate_pool,
        )

        results: list[list[RetrievedChunk]] = []
        for q_idx, query in enumerate(queries):
            dense_ranking: dict[int, int] = {}
            for rank, idx in enumerate(dense_indices_all[q_idx]):
                if idx != -1:
                    dense_ranking[int(idx)] = rank + 1  # 1-indexed for RRF

            bm25_top = bm25_index.get_top_k(query, k=candidate_pool)
            bm25_ranking = {idx: rank + 1 for rank, (idx, _score) in enumerate(bm25_top)}
            fused = rrf_fuse([dense_ranking, bm25_ranking], rrf_k)
            results.append([self._chunk(i, sc) for i, sc in fused[:top_k]])
        return results
