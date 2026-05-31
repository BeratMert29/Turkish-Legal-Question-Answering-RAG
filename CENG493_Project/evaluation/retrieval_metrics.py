from ranx import Qrels, Run, evaluate as ranx_evaluate


def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute retrieval metrics using ranx.

    Args:
        results: list of {"query_id": str, "retrieved": [chunk_id, ...], "relevant": [chunk_id, ...]}
                 Queries with empty relevant sets are excluded from metric computation.

    Returns:
        {"recall_at_5": float, "recall_at_10": float, "mrr": float, "ndcg_at_10": float, "num_queries": int}
    """
    # Build qrels: only include queries that have at least one relevant doc
    qrels_dict = {}
    run_dict = {}
    total_queries = 0
    num_queries = 0

    for r in results:
        qid = str(r["query_id"])
        relevant = r.get("relevant", [])
        retrieved = r.get("retrieved", [])

        total_queries += 1
        if not relevant:
            continue  # skip queries with no ground-truth relevant docs

        num_queries += 1
        qrels_dict[qid] = {str(doc_id): 1 for doc_id in relevant}
        # Score by inverse rank so ranx sorts correctly
        run_dict[qid] = {str(doc_id): 1.0 / (rank + 1) for rank, doc_id in enumerate(retrieved)}

    if not qrels_dict:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0, "source_hit_at_5": 0.0, "source_hit_at_10": 0.0, "capped_recall_at_5": 0.0, "capped_recall_at_10": 0.0, "precision_at_5": 0.0, "precision_at_10": 0.0, "num_queries": 0, "total_queries": total_queries}

    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    raw = ranx_evaluate(qrels, run, ["recall@5", "recall@10", "mrr", "ndcg@10"])

    # source_hit_at_k: fraction of queries where at least one retrieved
    # chunk (top-k) is in the relevant set.  More interpretable than
    # recall when relevance is defined at source (law) level.
    hit_at_5 = 0
    hit_at_10 = 0
    capped_recall_5_sum = 0.0
    capped_recall_10_sum = 0.0
    precision_5_sum = 0.0
    precision_10_sum = 0.0
    for r in results:
        qid = str(r["query_id"])
        if qid not in qrels_dict:
            continue
        relevant_set = set(str(d) for d in r.get("relevant", []))
        retrieved = [str(d) for d in r.get("retrieved", [])]
        if set(retrieved[:5]) & relevant_set:
            hit_at_5 += 1
        if set(retrieved[:10]) & relevant_set:
            hit_at_10 += 1
        hits_5 = len(set(retrieved[:5]) & relevant_set)
        hits_10 = len(set(retrieved[:10]) & relevant_set)
        capped_recall_5_sum += hits_5 / min(5, len(relevant_set))
        capped_recall_10_sum += hits_10 / min(10, len(relevant_set))
        precision_5_sum += hits_5 / 5
        precision_10_sum += hits_10 / 10
    n = len(qrels_dict)

    return {
        "recall_at_5":      float(raw["recall@5"]),
        "recall_at_10":     float(raw["recall@10"]),
        "mrr":              float(raw["mrr"]),
        "ndcg_at_10":       float(raw["ndcg@10"]),
        "source_hit_at_5":  hit_at_5 / n,
        "source_hit_at_10": hit_at_10 / n,
        "capped_recall_at_5":  capped_recall_5_sum / n,
        "capped_recall_at_10": capped_recall_10_sum / n,
        "precision_at_5":      precision_5_sum / n,
        "precision_at_10":     precision_10_sum / n,
        "num_queries":      num_queries,
        "total_queries":    total_queries,
    }
