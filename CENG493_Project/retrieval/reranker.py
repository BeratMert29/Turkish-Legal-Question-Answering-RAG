"""Cross-encoder re-ranker for second-stage retrieval."""
import numpy as np
import torch
from sentence_transformers import CrossEncoder
import config


class Reranker:
    def __init__(self, model_name: str = None, batch_size: int = 64):
        self.model_name = model_name or config.RERANKER_MODEL
        self.batch_size = batch_size
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model: CrossEncoder | None = None

    def load_model(self) -> None:
        try:
            self.model = CrossEncoder(self.model_name, device=self.device)
        except torch.cuda.OutOfMemoryError:
            self.device = "cpu"
            self.model = CrossEncoder(self.model_name, device="cpu")

    def rerank(self, query: str, chunks: list[dict],
               top_k: int = None) -> list[dict]:
        """Re-rank chunks for a single query using cross-encoder scores."""
        if self.model is None:
            raise RuntimeError("Call load_model() before rerank()")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL
        if not chunks:
            return []
        pairs = [(query, c["text"]) for c in chunks]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        return [{**chunks[i], "score": float(scores[i])} for i in ranked_indices]

    def batch_rerank(self, queries: list[str],
                     all_chunks: list[list[dict]],
                     top_k: int = None) -> list[list[dict]]:
        """Re-rank chunks for a batch of queries.

        Flattens all (query, passage) pairs into one predict call for
        throughput, then splits scores back per query.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before rerank()")
        if top_k is None:
            top_k = config.TOP_K_RETRIEVAL

        all_pairs: list[tuple[str, str]] = []
        lengths: list[int] = []
        for query, chunks in zip(queries, all_chunks):
            pairs = [(query, c["text"]) for c in chunks]
            all_pairs.extend(pairs)
            lengths.append(len(pairs))

        if not all_pairs:
            return [[] for _ in queries]

        all_scores = self.model.predict(
            all_pairs, batch_size=self.batch_size, show_progress_bar=True,
        )

        results: list[list[dict]] = []
        offset = 0
        for length, chunks in zip(lengths, all_chunks):
            scores = all_scores[offset : offset + length]
            offset += length
            if len(scores) == 0:
                results.append([])
                continue
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            results.append(
                [{**chunks[i], "score": float(scores[i])} for i in ranked_indices]
            )
        return results
