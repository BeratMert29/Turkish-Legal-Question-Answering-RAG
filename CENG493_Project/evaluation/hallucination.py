import random
import torch
import config
from scipy.special import softmax as scipy_softmax

if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"


def _classify_result(result: dict) -> str:
    """
    Classify a result as hit/partial/miss based on top-1 retrieved chunk score.
    Uses result["retrieved_chunks"] — a list of RetrievedChunk dicts.
    """
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return "miss"
    top_score = chunks[0]["score"]  # dict access, not attribute
    if top_score > config.HALLUCINATION_HIT_THRESHOLD:
        return "hit"
    elif top_score >= config.HALLUCINATION_PARTIAL_THRESHOLD:
        return "partial"
    else:
        return "miss"


def stratified_sample(results: list[dict], sample_size: int = config.HALLUCINATION_SAMPLE_SIZE) -> dict:
    """Sample ~third from each retrieval-score bucket (hit/partial/miss); fills to sample_size."""
    random.seed(42)

    hit_threshold = config.HALLUCINATION_HIT_THRESHOLD      # 0.7
    partial_threshold = config.HALLUCINATION_PARTIAL_THRESHOLD  # 0.4

    hits, partial, misses = [], [], []
    for r in results:
        category = _classify_result(r)
        if category == "hit":
            hits.append(r)
        elif category == "partial":
            partial.append(r)
        else:
            misses.append(r)

    target = sample_size // 3
    h = random.sample(hits, min(target, len(hits)))
    p = random.sample(partial, min(target, len(partial)))
    m = random.sample(misses, min(target, len(misses)))

    # Compensate: if a category is short, pull from others
    total = len(h) + len(p) + len(m)
    if total < sample_size:
        sampled_ids = {r.get("query_id") for r in h + p + m if r.get("query_id") is not None}
        pool = [x for x in hits + partial + misses if x.get("query_id") not in sampled_ids]
        extra = random.sample(pool, min(sample_size - total, len(pool)))
        # distribute extra evenly across h, p, m
        for i, item in enumerate(extra):
            if i % 3 == 0:
                h.append(item)
            elif i % 3 == 1:
                p.append(item)
            else:
                m.append(item)

    return {"hits": h, "partial": p, "misses": m}


def evaluate_faithfulness(answer: str, context: str, nli_model) -> dict:
    """
    Evaluate faithfulness using NLI CrossEncoder.
    cross-encoder/nli-deberta-v3-small returns logits shape (n_pairs, 3)
    Label order: [contradiction, entailment, neutral] — index 1 is entailment.
    """
    logits = nli_model.predict([(context, answer)])  # shape (1, 3)
    logit_vec = logits[0]  # shape (3,)
    probs = scipy_softmax(logit_vec)
    # Determine entailment index dynamically if model config is available
    entailment_idx = 1  # default for cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
    if hasattr(nli_model, 'config') and hasattr(nli_model.config, 'id2label'):
        id2label = nli_model.config.id2label
        label2id = {v.lower(): k for k, v in id2label.items()}
        entailment_idx = label2id.get('entailment', 1)
    entailment_prob = float(probs[entailment_idx])
    return {"faithful": entailment_prob >= 0.5, "score": entailment_prob}


def run_hallucination_analysis(
    sample_dict: dict,
    retrieved_results: dict,
    nli_model,
) -> dict:
    """
    Run faithfulness analysis on stratified sample using batched NLI inference.

    Args:
        sample_dict: {"hits": [...], "partial": [...], "misses": [...]} from stratified_sample()
        retrieved_results: {query_id: [RetrievedChunk, ...]} — full chunk list per query
        nli_model: pre-loaded CrossEncoder instance

    Returns: {"summary": {...}, "per_sample": [...]}
    """
    import numpy as np

    # Determine entailment index from model config
    entailment_idx = 1  # default: [contradiction, entailment, neutral]
    if hasattr(nli_model, 'config') and hasattr(nli_model.config, 'id2label'):
        id2label = nli_model.config.id2label
        label2id = {v.lower(): k for k, v in id2label.items()}
        entailment_idx = int(label2id.get('entailment', 1))

    # Collect all samples with their metadata
    ordered_items = []  # (query_id, answer, context, category)
    for category, items in sample_dict.items():
        for item in items:
            query_id = item.get("query_id", "")
            answer = item.get("predicted", "")
            chunks = retrieved_results.get(query_id, [])
            context = "\n\n".join(c["text"] for c in chunks[:5]) if chunks else ""
            ordered_items.append((query_id, answer, context, category))

    # Batch NLI inference
    pairs = [(ctx, ans) for _, ans, ctx, _ in ordered_items]
    if pairs:
        all_logits = nli_model.predict(pairs, batch_size=32)  # shape (N, 3)
        if all_logits.ndim == 1:
            all_logits = all_logits.reshape(1, -1)
    else:
        all_logits = np.zeros((0, 3), dtype=np.float32)

    softmax = scipy_softmax

    per_sample = []
    faithful_count = 0
    by_category = {"hits": {"total": 0, "faithful": 0},
                   "partial": {"total": 0, "faithful": 0},
                   "misses": {"total": 0, "faithful": 0}}

    for i, (query_id, answer, context, category) in enumerate(ordered_items):
        probs = softmax(all_logits[i])
        entailment_prob = float(probs[entailment_idx])
        is_faithful = entailment_prob >= 0.5
        if is_faithful:
            faithful_count += 1
            by_category[category]["faithful"] += 1
        by_category[category]["total"] += 1
        per_sample.append({
            "query_id": query_id,
            "category": category,
            "answer": answer,
            "faithful": is_faithful,
            "score": entailment_prob,
        })

    total = sum(c["total"] for c in by_category.values())
    scores = [s["score"] for s in per_sample]
    score_stats = {
        "mean": float(np.mean(scores)) if scores else 0.0,
        "min": float(np.min(scores)) if scores else 0.0,
        "max": float(np.max(scores)) if scores else 0.0,
        "std": float(np.std(scores)) if scores else 0.0,
    }
    return {
        "summary": {
            "total": total,
            "faithful_count": faithful_count,
            "faithful_rate": faithful_count / total if total > 0 else 0.0,
            "by_category": by_category,
            "score_stats": score_stats,
        },
        "per_sample": per_sample,
    }
