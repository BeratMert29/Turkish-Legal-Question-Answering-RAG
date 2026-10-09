"""
evaluation/semantic_similarity.py — Semantic similarity between predicted and
expected answers using multilingual sentence embeddings.

Default model: config.SEMANTIC_SIM_MODEL (paraphrase-multilingual-mpnet-base-v2)
with max_seq_length raised to config.SEMANTIC_SIM_MAX_SEQ_LEN (512). Texts that
still exceed the window are split into word chunks, embedded separately and
averaged (then re-normalised), so long answers are not silently truncated
(the old MiniLM default truncated at 128 tokens).
"""

from __future__ import annotations

import numpy as np
import torch

if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"




def _chunk_words(text: str, max_words: int) -> list[str]:
    """Split text into consecutive chunks of at most max_words words."""
    words = (text or "").split()
    if len(words) <= max_words:
        return [text or ""]
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def _encode_long(model, texts: list[str], max_seq_length: int, batch_size: int = 32) -> np.ndarray:
    """Encode texts, chunk-and-average those longer than the model window.

    Turkish needs roughly 2 subword tokens per word, so a chunk holds
    max_seq_length // 2 words to stay under the window.
    """
    max_words = max(max_seq_length // 2, 1)
    flat: list[str] = []
    owner: list[int] = []
    for i, t in enumerate(texts):
        for c in _chunk_words(t, max_words):
            flat.append(c)
            owner.append(i)
    embs = np.asarray(model.encode(flat, batch_size=batch_size, show_progress_bar=False,
                                   normalize_embeddings=True))
    out = np.zeros((len(texts), embs.shape[1]), dtype=float)
    counts = np.zeros(len(texts))
    for e, o in zip(embs, owner):
        out[o] += e
        counts[o] += 1
    out /= np.maximum(counts, 1)[:, None]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


def compute_semantic_similarity(
    predictions: list[dict],
    model_name: str | None = None,
    max_seq_length: int | None = None,
) -> dict:
    """
    Compute cosine similarity between predicted and expected answer embeddings.

    Args:
        predictions: list of dicts with keys "predicted", "expected", optionally "query_id"
        model_name:  sentence-transformers model (default config.SEMANTIC_SIM_MODEL)
        max_seq_length: encoder window (default config.SEMANTIC_SIM_MAX_SEQ_LEN)

    Returns:
        {
            "mean_similarity": float,
            "per_sample": [{"query_id": ..., "similarity": float}, ...]
        }
    """
    if not predictions:
        return {"mean_similarity": 0.0, "per_sample": []}

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # Graceful degradation: return 0.0 scores rather than crashing the pipeline
        per_sample = [
            {"query_id": p.get("query_id", i), "similarity": 0.0}
            for i, p in enumerate(predictions)
        ]
        return {"mean_similarity": 0.0, "per_sample": per_sample}

    import config as _cfg
    model_name = model_name or _cfg.SEMANTIC_SIM_MODEL
    max_seq_length = int(max_seq_length or _cfg.SEMANTIC_SIM_MAX_SEQ_LEN)

    model = SentenceTransformer(model_name, device=_DEVICE)
    limit = getattr(model, "max_seq_length", None)
    if isinstance(limit, int) and limit < max_seq_length:
        try:
            model.max_seq_length = max_seq_length
        except Exception:
            max_seq_length = limit

    predicted_texts = [p.get("predicted", "") for p in predictions]
    expected_texts  = [p.get("expected",  "") for p in predictions]

    pred_embs = _encode_long(model, predicted_texts, max_seq_length)
    exp_embs = _encode_long(model, expected_texts, max_seq_length)

    per_sample = []
    similarities = []
    for i, p in enumerate(predictions):
        # With normalize_embeddings=True, dot product == cosine similarity
        sim = float(np.dot(pred_embs[i], exp_embs[i]))
        similarities.append(sim)
        per_sample.append({
            "query_id": p.get("query_id", i),
            "similarity": sim,
        })

    mean_sim = float(np.mean(similarities)) if similarities else 0.0
    return {"mean_similarity": mean_sim, "per_sample": per_sample}
