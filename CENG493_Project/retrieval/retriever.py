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
        self.index = faiss.read_index(str(index_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [json.loads(line) for line in f if line.strip()]
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

    def hybrid_retrieve(self, query: str, bm25_index, alpha: float = 0.5,
                        top_k: int = None,
                        candidate_pool: int = None) -> list[RetrievedChunk]:
        """Hybrid dense+sparse retrieval for a single query.
        final_score = alpha * dense_score + (1 - alpha) * bm25_score
        Only searches candidate_pool candidates from each source — O(candidate_pool) not O(N).
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        if len(bm25_index.metadata) != len(self.metadata):
            raise ValueError(
                f"BM25/FAISS metadata count mismatch: {len(bm25_index.metadata)} vs {len(self.metadata)}. "
                f"Rebuild both indices from the same corpus."
            )

        q_emb = self.embedder.encode([query], is_query=True, show_progress=False)
        dense_scores_raw, dense_indices_raw = self.index.search(q_emb.astype(np.float32), candidate_pool)

        # Build dense score dict: corpus_idx → score
        dense_scores: dict[int, float] = {}
        for score, idx in zip(dense_scores_raw[0], dense_indices_raw[0]):
            if idx != -1:
                dense_scores[int(idx)] = float(score)

        # Get globally-normalized BM25 scores for all docs, then take top candidates
        bm25_all_scores = bm25_index.get_scores(query)  # globally min-max normalized
        top_bm25_indices = bm25_all_scores.argsort()[::-1][:candidate_pool]
        bm25_scores: dict[int, float] = {
            int(i): float(bm25_all_scores[i]) for i in top_bm25_indices if bm25_all_scores[i] > 0
        }

        # Union and blend
        candidates = set(dense_scores) | set(bm25_scores)
        final_scores: dict[int, float] = {}
        for idx in candidates:
            d = dense_scores.get(idx, 0.0)
            b = bm25_scores.get(idx, 0.0)
            final_scores[idx] = alpha * d + (1.0 - alpha) * b

        top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        return [RetrievedChunk(
            text=self.metadata[i].get("text", ""),
            doc_id=self.metadata[i].get("doc_id", ""),
            source=self.metadata[i].get("source", ""),
            score=float(final_scores[i]),
            chunk_id=self.metadata[i].get("chunk_id", ""),
        ) for i in top_indices]

    def batch_hybrid_retrieve(self, queries: list[str], bm25_index,
                              alpha: float = 0.5,
                              top_k: int = None,
                              candidate_pool: int = None) -> list[list[RetrievedChunk]]:
        """Hybrid dense+sparse retrieval for a batch of queries.
        final_score = alpha * dense_score + (1 - alpha) * bm25_score
        Only searches candidate_pool candidates — O(candidate_pool) not O(N).
        """
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        if len(bm25_index.metadata) != len(self.metadata):
            raise ValueError(
                f"BM25/FAISS metadata count mismatch: {len(bm25_index.metadata)} vs {len(self.metadata)}. "
                f"Rebuild both indices from the same corpus."
            )

        q_embs = self.embedder.encode(queries, is_query=True)
        dense_scores_all, dense_indices_all = self.index.search(q_embs.astype(np.float32), candidate_pool)

        results = []
        for q_idx, query in enumerate(queries):
            # Dense scores for this query
            dense_scores: dict[int, float] = {}
            for score, idx in zip(dense_scores_all[q_idx], dense_indices_all[q_idx]):
                if idx != -1:
                    dense_scores[int(idx)] = float(score)

            # Get globally-normalized BM25 scores for all docs, then take top candidates
            bm25_all_scores = bm25_index.get_scores(query)  # globally min-max normalized
            top_bm25_indices = bm25_all_scores.argsort()[::-1][:candidate_pool]
            bm25_scores: dict[int, float] = {
                int(i): float(bm25_all_scores[i]) for i in top_bm25_indices if bm25_all_scores[i] > 0
            }

            # Union and blend
            candidates = set(dense_scores) | set(bm25_scores)
            final_scores: dict[int, float] = {}
            for idx in candidates:
                d = dense_scores.get(idx, 0.0)
                b = bm25_scores.get(idx, 0.0)
                final_scores[idx] = alpha * d + (1.0 - alpha) * b

            top_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
            results.append([RetrievedChunk(
                text=self.metadata[i].get("text", ""),
                doc_id=self.metadata[i].get("doc_id", ""),
                source=self.metadata[i].get("source", ""),
                score=float(final_scores[i]),
                chunk_id=self.metadata[i].get("chunk_id", ""),
            ) for i in top_indices])
        return results

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
        if self.index is None:
            raise RuntimeError("Call build_index() or load_index() before using the retriever")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if rrf_k is None:
            rrf_k = config.RRF_K
        if candidate_pool is None:
            candidate_pool = config.RERANKER_CANDIDATES

        if len(bm25_index.metadata) != len(self.metadata):
            raise ValueError(
                f"BM25/FAISS metadata count mismatch: {len(bm25_index.metadata)} vs {len(self.metadata)}. "
                f"Rebuild both indices from the same corpus."
            )

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

            candidates = set(dense_ranking) | set(bm25_ranking)
            rrf_scores: dict[int, float] = {}
            for idx in candidates:
                score = 0.0
                if idx in dense_ranking:
                    score += 1.0 / (rrf_k + dense_ranking[idx])
                if idx in bm25_ranking:
                    score += 1.0 / (rrf_k + bm25_ranking[idx])
                rrf_scores[idx] = score

            top_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
            results.append([RetrievedChunk(
                text=self.metadata[i].get("text", ""),
                doc_id=self.metadata[i].get("doc_id", ""),
                source=self.metadata[i].get("source", ""),
                score=float(rrf_scores[i]),
                chunk_id=self.metadata[i].get("chunk_id", ""),
            ) for i in top_indices])
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
