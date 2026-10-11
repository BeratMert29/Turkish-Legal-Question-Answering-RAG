"""
evaluation/nli.py — multilingual NLI faithfulness of an answer against the
retrieved context.

Context faithfulness: premise = each chunk the generator saw, hypothesis =
each answer sentence.  Sentence score = max entailment probability over chunks
(a sentence is supported if any chunk entails it); answer score = mean over
sentences, ``supported_rate`` = fraction of sentences with score >= threshold.

Gold claim recall: premise = windows of the answer, hypothesis = each gold
answer sentence; the fraction of gold sentences the answer entails.  This is
the recall-side check (does the answer state what the reference states) and,
unlike entailing the whole answer from the gold, does not punish answers for
being longer than the reference.

The entailment label index is read from the model config (id2label).
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import softmax

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

# Import shared citation-strip pattern from qa_metrics to avoid duplication.
from evaluation.qa_metrics import _STRIP_CITATION_PATTERN as _CITATION

# Turkish legal abbreviations that end in a period without ending a sentence
# ("TMK m. 23", "4857 s. Kanun", "vb. haller").
_ABBREVIATIONS = frozenset({
    "m", "md", "mad", "s", "sy", "sk", "f", "fık", "b", "bkz", "vb", "vs",
    "no", "nr", "art", "c", "yy", "yön", "örn", "dr", "prof", "doç", "av",
})
# Answer windows used as NLI premise for claim recall (chars); keeps
# premise + hypothesis inside the 512-token NLI window.
CLAIM_PREMISE_CHARS = 1200


def load_nli_model(model_name: str | None = None, device: str | None = None):
    """Load a CrossEncoder NLI model (falls back to CPU on CUDA OOM)."""
    import torch
    from sentence_transformers import CrossEncoder

    if model_name is None:
        import config
        model_name = config.NLI_MODEL
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


def _continues(prev: str, nxt: str) -> bool:
    """True when the break between *prev* and *nxt* is not a sentence end."""
    if not prev.endswith("."):
        return False
    last = re.split(r"\s+", prev[:-1].strip())[-1] if prev[:-1].strip() else ""
    last = last.strip("(").casefold()
    if last.isdigit():          # "Kanun'un 25. maddesi", "2. fıkra"
        return True
    if last in _ABBREVIATIONS or (len(last) == 1 and last.isalpha()):
        return True
    return bool(nxt) and nxt[0].islower()


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences; citation markers are dropped and
    Turkish legal abbreviations / ordinal numbers do not end a sentence."""
    text = re.sub(r"\s+([.!?])", r"\1", _CITATION.sub("", text or ""))
    out: list[str] = []
    for line in re.split(r"\n+", text):
        pieces = [p.strip() for p in re.split(r"(?<=[.!?])\s+", line) if p.strip()]
        for piece in pieces:
            if out and out[-1] is not None and _continues(out[-1], piece):
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
        out.append(None)  # line break always ends a sentence
    return [s for s in out if s is not None and len(s) > 1]


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
    max_chunks: int | None = None,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> dict:
    """Score each prediction's answer against its retrieved context.

    predictions need "predicted" and "retrieved_chunks" (dicts with "text"):
    the chunks the generator actually saw.  ``max_chunks`` (None = all)
    caps how many of them are used.  ``query_ids`` restricts scoring to a
    subset.
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


def _answer_windows(text: str, max_chars: int = CLAIM_PREMISE_CHARS) -> list[str]:
    """Consecutive sentence windows of *text*, each at most ~max_chars."""
    windows: list[str] = []
    cur = ""
    for sent in split_sentences(text):
        if cur and len(cur) + 1 + len(sent) > max_chars:
            windows.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        windows.append(cur)
    return windows


def nli_claim_recall(
    predictions: list[dict],
    nli_model,
    batch_size: int = 16,
    threshold: float = 0.5,
) -> dict:
    """Fraction of gold-answer sentences entailed by the generated answer.

    Each gold sentence is a claim; its score is the max entailment
    probability over windows of the answer (premise).  ``claim_recall`` per
    query is the fraction of claims with score >= threshold.  Predictions with
    an empty answer score 0 (the answer states nothing); an empty gold answer
    gives None.
    """
    idx = entailment_index(nli_model)
    pairs: list[tuple[str, str]] = []
    owners: list[tuple[int, int]] = []
    n_claims: list[int] = []
    for pi, p in enumerate(predictions):
        claims = split_sentences(p.get("expected", ""))
        windows = _answer_windows(p.get("predicted", ""))
        n_claims.append(len(claims))
        if not claims or not windows:
            continue
        for ci, claim in enumerate(claims):
            for w in windows:
                pairs.append((w, claim))
                owners.append((pi, ci))

    probs = _entail_probs(nli_model, pairs, idx, batch_size)
    best: dict[tuple[int, int], float] = {}
    for key, pr in zip(owners, probs):
        best[key] = max(best.get(key, 0.0), float(pr))

    per_sample = []
    for pi, p in enumerate(predictions):
        n_c = n_claims[pi]
        if n_c == 0:
            per_sample.append({"query_id": p.get("query_id", ""), "claim_recall": None,
                               "claim_score": None, "n_claims": 0})
            continue
        sc = [best.get((pi, ci), 0.0) for ci in range(n_c)]
        per_sample.append({
            "query_id": p.get("query_id", ""),
            "claim_recall": float(np.mean([s >= threshold for s in sc])),
            "claim_score": float(np.mean(sc)),
            "n_claims": n_c,
        })
    valid = [s["claim_recall"] for s in per_sample if s["claim_recall"] is not None]
    return {
        "claim_recall": float(np.mean(valid)) if valid else None,
        "n": len(valid),
        "per_sample": per_sample,
    }
