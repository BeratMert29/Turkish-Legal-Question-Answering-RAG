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

    # ── BGE-M3 Multi-Vector Retrieval ────────────────────────────────────────

    def multi_vector_retrieve(self, query: str, bgem3_embedder,
                              top_k: int = None,
                              dense_weight: float = 1.0,
                              sparse_weight: float = 1.0,
                              colbert_weight: float = 1.0) -> list[RetrievedChunk]:
        """
        Retrieve using BGE-M3 dense + sparse + ColBERT, fused via min-max normalized score sum.

        Strategy:
          1. FAISS dense search → top-(top_k * 5) candidate pool
          2. Re-encode all candidates with encode_multi (single forward pass)
          3. Compute sparse scores via model.compute_lexical_matching_score
          4. Compute ColBERT scores via model.colbert_score
          5. Min-max normalize each score type within the candidate set
          6. Final score = dense_norm*dense_weight + sparse_norm*sparse_weight + colbert_norm*colbert_weight
          7. Return top_k by final score

        Falls back to standard dense retrieve() if bgem3_embedder is None or if any error occurs.
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        if bgem3_embedder is None:
            return self.retrieve(query, top_k=top_k)

        try:
            candidate_k = top_k * 5

            # Step 1: Encode query with all three modes simultaneously
            q_multi = bgem3_embedder.encode_multi([query], is_query=True, show_progress=False)
            q_dense = q_multi["dense"]         # shape (1, 1024)
            q_sparse = q_multi["sparse"][0]    # dict {token_id: weight}
            q_colbert = q_multi["colbert"][0]  # shape (q_seq_len, 1024)

            # Step 2: FAISS dense search for candidate pool
            dense_scores_raw, dense_indices_raw = self.index.search(
                q_dense.astype(np.float32), candidate_k
            )
            candidate_indices = [int(idx) for idx in dense_indices_raw[0] if idx != -1]
            candidate_dense_scores = {
                int(idx): float(score)
                for idx, score in zip(dense_indices_raw[0], dense_scores_raw[0])
                if idx != -1
            }

            if not candidate_indices:
                return []

            # Step 3: Re-encode candidates with all three modes (single forward pass)
            candidate_texts = [self.metadata[i].get("text", "") for i in candidate_indices]
            doc_multi = bgem3_embedder.encode_multi(
                candidate_texts, is_query=False, show_progress=False
            )
            doc_sparse_list = doc_multi["sparse"]    # list[dict]
            doc_colbert_list = doc_multi["colbert"]  # list[np.ndarray]

            # Step 4: Compute sparse and ColBERT scores per candidate
            sparse_scores: dict[int, float] = {}
            colbert_scores: dict[int, float] = {}
            for local_i, corpus_idx in enumerate(candidate_indices):
                sparse_scores[corpus_idx] = float(
                    bgem3_embedder.model.compute_lexical_matching_score(
                        q_sparse, doc_sparse_list[local_i]
                    )
                )
                colbert_scores[corpus_idx] = float(
                    bgem3_embedder.model.colbert_score(
                        q_colbert, doc_colbert_list[local_i]
                    )
                )

            # Step 5: Min-max normalize each score type within candidate set
            def _minmax_norm(score_dict: dict) -> dict:
                vals = list(score_dict.values())
                mn, mx = min(vals), max(vals)
                rng = mx - mn
                if rng < 1e-9:
                    return {k: 1.0 for k in score_dict}
                return {k: (v - mn) / rng for k, v in score_dict.items()}

            dense_norm = _minmax_norm(candidate_dense_scores)
            sparse_norm = _minmax_norm(sparse_scores)
            colbert_norm = _minmax_norm(colbert_scores)

            # Step 6: Fuse normalized scores
            final_scores: dict[int, float] = {}
            for corpus_idx in candidate_indices:
                final_scores[corpus_idx] = (
                    dense_norm.get(corpus_idx, 0.0) * dense_weight
                    + sparse_norm.get(corpus_idx, 0.0) * sparse_weight
                    + colbert_norm.get(corpus_idx, 0.0) * colbert_weight
                )

            top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
            return [RetrievedChunk(
                text=self.metadata[i].get("text", ""),
                doc_id=self.metadata[i].get("doc_id", ""),
                source=self.metadata[i].get("source", ""),
                score=float(final_scores[i]),
                chunk_id=self.metadata[i].get("chunk_id", ""),
            ) for i in top_indices]

        except Exception as e:
            import logging
            logging.warning(
                f"multi_vector_retrieve failed ({e}), falling back to dense-only retrieve()"
            )
            return self.retrieve(query, top_k=top_k)

    def batch_multi_vector_retrieve(self, queries: list[str], bgem3_embedder,
                                    top_k: int = None,
                                    dense_weight: float = 1.0,
                                    sparse_weight: float = 1.0,
                                    colbert_weight: float = 1.0) -> list[list[RetrievedChunk]]:
        """
        Multi-vector retrieval for a batch of queries.
        Each query is processed independently via multi_vector_retrieve.
        Falls back to batch_retrieve if bgem3_embedder is None.
        """
        if bgem3_embedder is None:
            return self.batch_retrieve(queries, top_k=top_k or config.TOP_K_RETRIEVAL)
        return [
            self.multi_vector_retrieve(
                q, bgem3_embedder,
                top_k=top_k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                colbert_weight=colbert_weight,
            )
            for q in queries
        ]
