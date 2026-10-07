"""
evaluation/nli.py — multilingual NLI faithfulness of an answer against the
retrieved context.

Premise = each retrieved context chunk, hypothesis = each answer sentence.
Sentence score = max entailment probability over chunks (a sentence is
supported if any chunk entails it); answer score = mean over sentences.
The entailment label index is read from the model config (id2label).
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import softmax

_CITATION = re.compile(r"\[\s*kaynak\s+\d+\s*\]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"


def load_nli_model(model_name: str | None = None, device: str | None = None):
    """Load a CrossEncoder NLI model (falls back to CPU on CUDA OOM)."""
    import torch
    from sentence_transformers import CrossEncoder

    if model_name is None:
        try:
            import config
            model_name = config.NLI_MODEL
        except Exception:
            model_name = DEFAULT_NLI_MODEL
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    try:
        return CrossEncoder(model_name, device=device)
    except torch.cuda.OutOfMemoryError:
        return CrossEncoder(model_name, device="cpu")


def entailment_index(nli_model) -> int:
    """Entailment class index from the model's id2label. Raises if absent."""
    cfg = getattr(nli_model, "config", None) or getattr(
        getattr(nli_model, "model", None), "config", None)
    id2label = getattr(cfg, "id2label", None)
    if not id2label:
        raise ValueError("NLI model exposes no id2label; cannot locate entailment index")
    for idx, label in id2label.items():
        if str(label).lower().startswith("entail"):
            return int(idx)
    raise ValueError(f"no 'entailment' label in id2label={id2label}")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+([.!?])", r"\1", _CITATION.sub("", text or ""))
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 1]


def _entail_probs(nli_model, pairs: list[tuple[str, str]], idx: int, batch_size: int) -> np.ndarray:
    if not pairs:
        return np.zeros(0)
    logits = np.asarray(nli_model.predict(pairs, batch_size=batch_size))
    if logits.ndim == 1:
        logits = logits.reshape(1, -1)
    return softmax(logits, axis=1)[:, idx]


def nli_context_faithfulness(
    predictions: list[dict],
    nli_model,
    query_ids=None,
    max_chunks: int = 5,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> dict:
    """Score each prediction's answer against its retrieved context.

    predictions need "predicted" and "retrieved_chunks" (dicts with "text").
    ``query_ids`` restricts scoring to the shared judge sample.
    Predictions with empty answer or no context get score None (excluded
    from the mean, counted in ``n_skipped``).
    """
    idx = entailment_index(nli_model)
    if query_ids is not None:
        keep = set(query_ids)
        predictions = [p for p in predictions if p.get("query_id") in keep]

    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, int]] = []  # (prediction idx, sentence idx)
    sent_counts: list[int] = []
    for pi, p in enumerate(predictions):
        sents = split_sentences(p.get("predicted", ""))
        chunks = [c.get("text", "") for c in p.get("retrieved_chunks", [])[:max_chunks]
                  if c.get("text")]
        if not sents or not chunks:
            sent_counts.append(0)
            continue
        sent_counts.append(len(sents))
        for si, s in enumerate(sents):
            for c in chunks:
                pairs.append((c, s))
                owners.append((pi, si))

    probs = _entail_probs(nli_model, pairs, idx, batch_size)
    best: dict[tuple[int, int], float] = {}
    for (pi, si), pr in zip(owners, probs):
        best[(pi, si)] = max(best.get((pi, si), 0.0), float(pr))

    per_sample = []
    for pi, p in enumerate(predictions):
        n_s = sent_counts[pi]
        if n_s == 0:
            per_sample.append({"query_id": p.get("query_id", ""), "score": None,
                               "supported_rate": None, "n_sentences": 0})
            continue
        sc = [best[(pi, si)] for si in range(n_s)]
        per_sample.append({
            "query_id": p.get("query_id", ""),
            "score": float(np.mean(sc)),
            "supported_rate": float(np.mean([s >= threshold for s in sc])),
            "n_sentences": n_s,
        })
    valid = [s for s in per_sample if s["score"] is not None]
    return {
        "mean_score": float(np.mean([s["score"] for s in valid])) if valid else None,
        "supported_rate": float(np.mean([s["supported_rate"] for s in valid])) if valid else None,
        "n": len(valid),
        "n_skipped": len(per_sample) - len(valid),
        "entailment_idx": idx,
        "per_sample": per_sample,
    }
