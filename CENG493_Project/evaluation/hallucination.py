import random
import config
from scipy.special import softmax as scipy_softmax
from evaluation.nli import entailment_index, nli_context_faithfulness


def _norm_source(s) -> str:
    from utils import normalize_turkish
    return normalize_turkish(str(s).strip()) if s else ""


def gold_rank(result: dict, k: int | None = None) -> int | None:
    """1-based rank of the first gold item in the retrieved list, or None.

    Gold is determined, in order, from: id-level ``relevant`` vs ``retrieved``;
    else ``expected_source`` vs ``retrieved_sources`` (or chunk sources).
    Returns None when the gold chunk is absent from the top-k. Raises
    LookupError when the result carries no gold label at all.
    """
    if result.get("relevant") and result.get("retrieved") is not None:
        rel = {str(x) for x in result["relevant"]}
        ranked = [str(x) for x in result["retrieved"]]
    elif result.get("expected_source"):
        rel = {_norm_source(result["expected_source"])}
        srcs = result.get("retrieved_sources")
        if srcs is None:
            srcs = [c.get("source", "") for c in result.get("retrieved_chunks", [])]
        ranked = [_norm_source(x) for x in srcs]
    else:
        raise LookupError("no gold label")
    if k is not None:
        ranked = ranked[:k]
    for i, r in enumerate(ranked, start=1):
        if r in rel:
            return i
    return None


def _classify_result(result: dict, k: int = 5) -> str | None:
    """Stratum by gold retrieval, independent of score scale per stage.

    hit     : gold at rank 1
    partial : gold at rank 2..k
    miss    : gold not in top-k
    None    : no gold label (excluded from stratification)
    """
    try:
        rank = gold_rank(result, k)
    except LookupError:
        return None
    if rank is None:
        return "miss"
    return "hit" if rank == 1 else "partial"


def stratified_sample(results: list[dict], sample_size: int = config.HALLUCINATION_SAMPLE_SIZE,
                      k: int = 5) -> dict:
    """Sample ~third from each gold-retrieval stratum (hit/partial/miss); fills to sample_size."""
    # Use a local RNG instance to avoid mutating the global random state across runs.
    rng = random.Random(42)

    hits, partial, misses = [], [], []
    for r in results:
        category = _classify_result(r, k)
        if category == "hit":
            hits.append(r)
        elif category == "partial":
            partial.append(r)
        elif category == "miss":
            misses.append(r)

    target = sample_size // 3
    h = rng.sample(hits, min(target, len(hits)))
    p = rng.sample(partial, min(target, len(partial)))
    m = rng.sample(misses, min(target, len(misses)))

    total = len(h) + len(p) + len(m)
    if total < sample_size:
        sampled_ids = {r.get("query_id") for r in h + p + m if r.get("query_id") is not None}
        pool = [x for x in hits + partial + misses if x.get("query_id") not in sampled_ids]
        extra = rng.sample(pool, min(sample_size - total, len(pool)))
        for item in extra:
            c = _classify_result(item, k)
            {"hit": h, "partial": p, "miss": m}[c].append(item)

    return {"hits": h, "partial": p, "misses": m}


def evaluate_faithfulness(answer: str, context: str, nli_model) -> dict:
    """NLI entailment prob (context → answer)."""
    logits = nli_model.predict([(context, answer)])
    logit_vec = logits[0]
    probs = scipy_softmax(logit_vec)
    entailment_idx = entailment_index(nli_model)
    entailment_prob = float(probs[entailment_idx])
    return {"faithful": entailment_prob >= 0.5, "score": entailment_prob}


def run_hallucination_analysis(
    sample_dict: dict,
    retrieved_results: dict,
    nli_model,
) -> dict:
    """Batch NLI: context grounding + gold-answer consistency (entailment probs)."""
    import numpy as np

    entailment_idx = entailment_index(nli_model)

    ordered_items = []
    for category, items in sample_dict.items():
        for item in items:
            query_id = item.get("query_id", "")
            predicted = item.get("predicted", "")
            gold_answer = item.get("expected", "")
            chunks = retrieved_results.get(query_id, [])
            context = "\n\n".join(c["text"] for c in chunks[:5]) if chunks else ""
            ordered_items.append((query_id, predicted, context, gold_answer, category))

    # Grounding: answer sentences vs each retrieved chunk (max over chunks,
    # mean over sentences); not the gold answer.
    _g = nli_context_faithfulness(
        [{"query_id": qid, "predicted": pred,
          "retrieved_chunks": retrieved_results.get(qid, [])[:5]}
         for qid, pred, _, _, _ in ordered_items],
        nli_model, max_chunks=5, batch_size=8,
    )
    grounding_scores_raw = [x["score"] for x in _g["per_sample"]]

    has_gold = [bool(gold) for _, _, _, gold, _ in ordered_items]
    faith_pairs = [
        (gold, pred)
        for (_, pred, _, gold, _), has in zip(ordered_items, has_gold)
        if has
    ]
    if faith_pairs:
        faith_logits = nli_model.predict(faith_pairs, batch_size=8)
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
        grounding_prob = grounding_scores_raw[i] if grounding_scores_raw[i] is not None else 0.0
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
            # Explicit name for the legacy metric: NLI(gold answer -> predicted answer).
            "gold_answer_entailment_rate": answer_faithfulness_rate,
            "context_grounding_count": grounding_count,
            "context_grounding_rate":  context_grounding_rate,
            "by_category": by_category,
            "context_grounding_score_stats":  _stats(grounding_scores),
            "answer_faithfulness_score_stats": _stats(faith_scores),
            "score_stats": _stats(grounding_scores),
        },
        "per_sample": per_sample,
    }
