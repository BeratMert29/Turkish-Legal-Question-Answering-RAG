def compute_source_hit_metrics(results: list[dict]) -> dict:
    """Compute law-level source-hit and source-precision metrics.

    These metrics apply to **all** queries that have a known gold source law,
    regardless of whether the query has article-level ground-truth relevance.
    They answer: "did the retriever bring back at least one chunk from the
    correct law?" — useful for HMGS queries where article-level labels are
    mostly unavailable.

    Args:
        results: list of dicts with keys:
            "query_id"          : str
            "source_law"        : str — gold law name (empty string = unknown, skipped)
            "retrieved_sources" : list[str] — source field of each retrieved chunk,
                                  in retrieval-rank order

    Returns:
        Dict with keys:
            source_hit_at_5_all      : fraction of source-known queries with a
                                       top-5 chunk from the gold law
            source_hit_at_10_all     : same for top-10
            source_precision_at_5_all: mean fraction of top-5 chunks from gold law
            source_precision_at_10_all: mean fraction of top-10 chunks
            source_labeled_queries   : number of queries with a known source law
            total_queries            : total queries passed in
    """
    hit5 = hit10 = 0
    prec5_sum = prec10_sum = 0.0
    mrr_sum = 0.0
    source_labeled = 0
    total = len(results)

    for r in results:
        law = (r.get("source_law") or "").strip()
        if not law:
            continue
        source_labeled += 1
        srcs = r.get("retrieved_sources", [])
        top5  = srcs[:5]
        top10 = srcs[:10]
        hits5  = sum(1 for s in top5  if s == law)
        hits10 = sum(1 for s in top10 if s == law)
        if hits5:
            hit5 += 1
        if hits10:
            hit10 += 1
        prec5_sum  += hits5  / max(len(top5),  1)
        prec10_sum += hits10 / max(len(top10), 1)
        # MRR: reciprocal rank of the first retrieved chunk from the gold law
        for rank, src in enumerate(srcs):
            if src == law:
                mrr_sum += 1.0 / (rank + 1)
                break

    n = source_labeled or 1  # avoid div-by-zero; metrics will be 0.0
    return {
        "source_hit_at_5_all":        hit5  / n,
        "source_hit_at_10_all":       hit10 / n,
        "source_mrr_all":             mrr_sum / n,
        "source_precision_at_5_all":  prec5_sum  / n,
        "source_precision_at_10_all": prec10_sum / n,
        "source_labeled_queries":     source_labeled,
        "total_queries":              total,
    }


def compute_all_metrics(results: list[dict]) -> dict:  # noqa: C901
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
        if not run_dict[qid]:
            # ranx crashes on an empty run entry; a single non-relevant
            # placeholder doc scores the query as 0 on every metric.
            run_dict[qid] = {"__no_retrieval__": 1.0}

    if not qrels_dict:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0, "mrr": 0.0, "ndcg_at_10": 0.0, "source_hit_at_5": 0.0, "source_hit_at_10": 0.0, "capped_recall_at_5": 0.0, "capped_recall_at_10": 0.0, "precision_at_5": 0.0, "precision_at_10": 0.0, "num_queries": 0, "total_queries": total_queries}

    from ranx import Qrels, Run, evaluate as ranx_evaluate  # lazy import; ranx optional
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
