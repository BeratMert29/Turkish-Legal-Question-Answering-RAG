import random
import config
from scipy.special import softmax as scipy_softmax


def _classify_result(result: dict) -> str:
    """Hit/partial/miss from top-1 retrieval score."""
    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        return "miss"
    top_score = chunks[0]["score"]
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

    total = len(h) + len(p) + len(m)
    if total < sample_size:
        sampled_ids = {r.get("query_id") for r in h + p + m if r.get("query_id") is not None}
        pool = [x for x in hits + partial + misses if x.get("query_id") not in sampled_ids]
        extra = random.sample(pool, min(sample_size - total, len(pool)))
        for i, item in enumerate(extra):
            if i % 3 == 0:
                h.append(item)
            elif i % 3 == 1:
                p.append(item)
            else:
                m.append(item)

    return {"hits": h, "partial": p, "misses": m}


def evaluate_faithfulness(answer: str, context: str, nli_model) -> dict:
    """NLI entailment prob (context → answer)."""
    logits = nli_model.predict([(context, answer)])
    logit_vec = logits[0]
    probs = scipy_softmax(logit_vec)
    entailment_idx = 1
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
    """Batch NLI: context grounding + gold-answer consistency (entailment probs)."""
    import numpy as np

    entailment_idx = 1
    if hasattr(nli_model, 'config') and hasattr(nli_model.config, 'id2label'):
        id2label = nli_model.config.id2label
        label2id = {v.lower(): k for k, v in id2label.items()}
        entailment_idx = int(label2id.get('entailment', 1))

    ordered_items = []
    for category, items in sample_dict.items():
        for item in items:
            query_id = item.get("query_id", "")
            predicted = item.get("predicted", "")
            gold_answer = item.get("expected", "")
            chunks = retrieved_results.get(query_id, [])
            context = "\n\n".join(c["text"] for c in chunks[:5]) if chunks else ""
            ordered_items.append((query_id, predicted, context, gold_answer, category))

    grounding_pairs = [(ctx, pred) for _, pred, ctx, _, _ in ordered_items]
    if grounding_pairs:
        grounding_logits = nli_model.predict(grounding_pairs, batch_size=32)
        if grounding_logits.ndim == 1:
            grounding_logits = grounding_logits.reshape(1, -1)
    else:
        grounding_logits = np.zeros((0, 3), dtype=np.float32)

    has_gold = [bool(gold) for _, _, _, gold, _ in ordered_items]
    faith_pairs = [
        (gold, pred)
        for (_, pred, _, gold, _), has in zip(ordered_items, has_gold)
        if has
    ]
    if faith_pairs:
        faith_logits = nli_model.predict(faith_pairs, batch_size=32)
        if faith_logits.ndim == 1:
            faith_logits = faith_logits.reshape(1, -1)
    else:
        faith_logits = np.zeros((0, 3), dtype=np.float32)

    softmax = scipy_softmax

    per_sample = []
    grounding_count = 0
    faith_count = 0
    faith_total = 0
    by_category = {
        "hits":    {"total": 0, "context_grounded": 0, "answer_faithful": 0},
        "partial": {"total": 0, "context_grounded": 0, "answer_faithful": 0},
        "misses":  {"total": 0, "context_grounded": 0, "answer_faithful": 0},
    }

    faith_idx = 0
    for i, (query_id, predicted, context, gold_answer, category) in enumerate(ordered_items):
        grounding_probs = softmax(grounding_logits[i])
        grounding_prob = float(grounding_probs[entailment_idx])
        is_grounded = grounding_prob >= 0.5

        answer_faith_prob = None
        is_answer_faithful = None
        if has_gold[i]:
            faith_probs = softmax(faith_logits[faith_idx])
            answer_faith_prob = float(faith_probs[entailment_idx])
            is_answer_faithful = answer_faith_prob >= 0.5
            faith_idx += 1

        if is_grounded:
            grounding_count += 1
            by_category[category]["context_grounded"] += 1
        if is_answer_faithful:
            faith_count += 1
            faith_total += 1
            by_category[category]["answer_faithful"] += 1
        elif is_answer_faithful is False:
            faith_total += 1

        by_category[category]["total"] += 1
        per_sample.append({
            "query_id": query_id,
            "category": category,
            "predicted": predicted,
            "context_grounding_score": grounding_prob,
            "context_grounded": is_grounded,
            "answer_faithfulness_score": answer_faith_prob,
            "answer_faithful": is_answer_faithful,
            "answer": predicted,
            "faithful": is_grounded,
            "score": grounding_prob,
        })

    total = sum(c["total"] for c in by_category.values())
    grounding_scores = [s["context_grounding_score"] for s in per_sample]
    faith_scores = [s["answer_faithfulness_score"] for s in per_sample if s["answer_faithfulness_score"] is not None]

    def _stats(vals):
        if not vals:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
        return {
            "mean": float(np.mean(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "std":  float(np.std(vals)),
        }

    context_grounding_rate = grounding_count / total if total > 0 else 0.0
    answer_faithfulness_rate = faith_count / faith_total if faith_total > 0 else None

    return {
        "summary": {
            "total": total,
            "context_grounding_count": grounding_count,
            "context_grounding_rate": context_grounding_rate,
            "answer_faithfulness_count": faith_count,
            "answer_faithfulness_total": faith_total,
            "answer_faithfulness_rate": answer_faithfulness_rate,
            "faithful_count": grounding_count,
            "faithful_rate":  context_grounding_rate,
            "by_category": by_category,
            "context_grounding_score_stats":  _stats(grounding_scores),
            "answer_faithfulness_score_stats": _stats(faith_scores),
            "score_stats": _stats(grounding_scores),
        },
        "per_sample": per_sample,
    }
