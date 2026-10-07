import random
import config


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


def stratified_sample(results: list[dict], sample_size: int | None = config.HALLUCINATION_SAMPLE_SIZE,
                      k: int = 5) -> dict:
    """Group results by gold-retrieval stratum (hit/partial/miss/unlabeled).

    ``sample_size=None`` keeps every result (no sampling, the default): the
    stage-level rate then covers the whole eval set and every stage scores
    the same queries.  An int draws ~a third from each labeled stratum
    (filling up to sample_size) and leaves unlabeled results out.
    """
    hits, partial, misses, unlabeled = [], [], [], []
    for r in results:
        category = _classify_result(r, k)
        {"hit": hits, "partial": partial, "miss": misses, None: unlabeled}[category].append(r)

    if sample_size is None:
        return {"hits": hits, "partial": partial, "misses": misses, "unlabeled": unlabeled}

    # Use a local RNG instance to avoid mutating the global random state across runs.
    rng = random.Random(42)
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

    return {"hits": h, "partial": p, "misses": m, "unlabeled": []}


def run_hallucination_analysis(
    sample_dict: dict,
    retrieved_results: dict,
    nli_model,
    threshold: float = 0.5,
) -> dict:
    """Batch NLI over the sampled predictions.

    ``retrieved_results`` maps query_id to the chunks the generator actually
    saw (its context), which are the NLI premises.

    Reported per answer and summarised:
      * context grounding  — answer sentences vs context (``supported_rate``
        = fraction of sentences some chunk entails; ``score`` = mean
        sentence entailment);
      * gold claim recall  — fraction of gold-answer sentences the answer
        entails (needs a gold answer).
    The headline ``context_supported_sentence_rate`` is the mean of the
    per-answer supported_rate; ``context_grounding_rate`` (answers whose mean
    sentence entailment >= threshold) is kept for comparison with older runs.
    """
    import numpy as np
    from evaluation.nli import nli_claim_recall, nli_context_faithfulness

    ordered = [(category, item) for category, items in sample_dict.items() for item in items]
    preds = [
        {"query_id": item.get("query_id", ""), "predicted": item.get("predicted", ""),
         "expected": item.get("expected", ""),
         "retrieved_chunks": retrieved_results.get(item.get("query_id", ""), [])}
        for _, item in ordered
    ]
    grounding = nli_context_faithfulness(preds, nli_model, batch_size=8, threshold=threshold)
    claims = nli_claim_recall(preds, nli_model, batch_size=8, threshold=threshold)

    by_category: dict[str, dict] = {}
    per_sample = []
    for (category, _), g, c, p in zip(ordered, grounding["per_sample"],
                                      claims["per_sample"], preds):
        cat = by_category.setdefault(category, {"total": 0, "context_grounded": 0,
                                                 "supported_rate_sum": 0.0})
        score = g["score"] if g["score"] is not None else 0.0
        supported = g["supported_rate"] if g["supported_rate"] is not None else 0.0
        grounded = score >= threshold
        cat["total"] += 1
        cat["context_grounded"] += int(grounded)
        cat["supported_rate_sum"] += supported
        per_sample.append({
            "query_id": p["query_id"],
            "category": category,
            "predicted": p["predicted"],
            "context_grounding_score": score,
            "context_supported_rate": supported,
            "context_grounded": grounded,
            "n_sentences": g["n_sentences"],
            "gold_claim_recall": c["claim_recall"],
            "n_gold_claims": c["n_claims"],
        })
    for cat in by_category.values():
        cat["context_supported_sentence_rate"] = (
            cat.pop("supported_rate_sum") / cat["total"] if cat["total"] else None)

    total = len(per_sample)
    grounding_scores = [s["context_grounding_score"] for s in per_sample]
    claim_scores = [s["gold_claim_recall"] for s in per_sample
                    if s["gold_claim_recall"] is not None]

    def _stats(vals):
        if not vals:
            return {"mean": None, "min": None, "max": None, "std": None}
        return {
            "mean": float(np.mean(vals)),
            "min":  float(np.min(vals)),
            "max":  float(np.max(vals)),
            "std":  float(np.std(vals)),
        }

    n_grounded = sum(s["context_grounded"] for s in per_sample)
    return {
        "summary": {
            "total": total,
            "threshold": threshold,
            "context_supported_sentence_rate": (
                float(np.mean([s["context_supported_rate"] for s in per_sample]))
                if per_sample else None),
            "context_grounding_count": n_grounded,
            "context_grounding_rate": n_grounded / total if total else None,
            "n_empty_or_no_context": grounding["n_skipped"],
            "gold_claim_recall": float(np.mean(claim_scores)) if claim_scores else None,
            "gold_claim_recall_n": len(claim_scores),
            "by_category": by_category,
            "context_grounding_score_stats": _stats(grounding_scores),
            "gold_claim_recall_stats": _stats(claim_scores),
        },
        "per_sample": per_sample,
    }
