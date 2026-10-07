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
                                  in retrieval-rank order.  Callers may deduplicate
                                  this list (e.g. keep only the first chunk per law)
                                  when per-law precision rather than per-chunk
                                  precision is desired; the function itself counts
                                  raw occurrences.

    Returns:
        Dict with keys:
            source_hit_at_5_all      : fraction of source-known queries with a
                                       top-5 chunk from the gold law
            source_hit_at_10_all     : same for top-10
            source_precision_at_5_all: mean fraction of k=5 slots from gold law
                                       (denominator is always 5, even when fewer
                                       than 5 chunks were retrieved — missing
                                       slots count as misses)
            source_precision_at_10_all: same with k=10 denominator
            source_labeled_queries   : number of queries with a known source law
            total_queries            : total queries passed in
    """
    from utils import normalize_turkish

    def _norm(s) -> str:
        return normalize_turkish(str(s or "").strip())

    hit5 = hit10 = 0
    prec5_sum = prec10_sum = 0.0
    mrr_sum = 0.0
    source_labeled = 0
    total = len(results)

    for r in results:
        law = _norm(r.get("source_law"))
        if not law:
            continue
        source_labeled += 1
        srcs = [_norm(s) for s in r.get("retrieved_sources", [])]
        top5  = srcs[:5]
        top10 = srcs[:10]
        hits5  = sum(1 for s in top5  if s == law)
        hits10 = sum(1 for s in top10 if s == law)
        if hits5:
            hit5 += 1
        if hits10:
            hit10 += 1
        prec5_sum  += hits5  / 5
        prec10_sum += hits10 / 10
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


_CHUNK_METRIC_KEYS = (
    "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10",
    "hit_at_5", "hit_at_10", "capped_recall_at_5", "capped_recall_at_10",
    "precision_at_5", "precision_at_10",
)


def _dedup(ids) -> list[str]:
    """String ids in first-occurrence order (a repeated id keeps its best rank)."""
    seen: set[str] = set()
    out: list[str] = []
    for d in ids:
        d = str(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def compute_all_metrics(results: list[dict]) -> dict:
    """
    Compute chunk-level retrieval metrics using ranx.

    Args:
        results: list of {"query_id": str, "retrieved": [chunk_id, ...], "relevant": [chunk_id, ...]}
                 Queries with empty relevant sets are excluded from metric computation.

    Every metric is computed from one ranking per query: ``retrieved`` is
    de-duplicated (a repeated chunk id keeps its first, best rank) before
    ranx and the set-based metrics see it, so they always score the same list.
    Query ids must be unique.

    ``hit_at_k`` is chunk-level (a gold chunk in the top-k); the law-level
    counterpart is :func:`compute_source_hit_metrics`.

    Returns:
        recall_at_5/10, mrr, ndcg_at_10, hit_at_5/10, capped_recall_at_5/10,
        precision_at_5/10 (all None when no query has gold labels),
        num_queries (gold-labeled) and total_queries.
    """
    qrels_dict: dict[str, dict] = {}
    run_dict: dict[str, dict] = {}
    ranked: dict[str, list[str]] = {}
    seen_qids: set[str] = set()
    total_queries = 0

    for r in results:
        qid = str(r["query_id"])
        if qid in seen_qids:
            raise ValueError(f"compute_all_metrics: duplicate query_id {qid!r}")
        seen_qids.add(qid)
        total_queries += 1

        relevant = _dedup(r.get("relevant") or [])
        if not relevant:
            continue  # skip queries with no ground-truth relevant docs

        retrieved = _dedup(r.get("retrieved") or [])
        qrels_dict[qid] = {doc_id: 1 for doc_id in relevant}
        ranked[qid] = retrieved
        # Score by inverse rank so ranx sorts correctly
        run_dict[qid] = {doc_id: 1.0 / (rank + 1) for rank, doc_id in enumerate(retrieved)}
        if not run_dict[qid]:
            # ranx crashes on an empty run entry; a single non-relevant
            # placeholder doc scores the query as 0 on every metric.
            run_dict[qid] = {"__no_retrieval__": 1.0}

    n = len(qrels_dict)
    if not n:
        # No gold labels: the metrics are unknown, not zero.
        return {**{k: None for k in _CHUNK_METRIC_KEYS},
                "num_queries": 0, "total_queries": total_queries}

    from ranx import Qrels, Run, evaluate as ranx_evaluate  # lazy import; ranx optional
    raw = ranx_evaluate(Qrels(qrels_dict), Run(run_dict),
                        ["recall@5", "recall@10", "mrr", "ndcg@10"])

    sums = dict.fromkeys(("hit_at_5", "hit_at_10", "capped_recall_at_5",
                          "capped_recall_at_10", "precision_at_5", "precision_at_10"), 0.0)
    for qid, rel in qrels_dict.items():
        relevant_set = set(rel)
        for k in (5, 10):
            hits = len(set(ranked[qid][:k]) & relevant_set)
            sums[f"hit_at_{k}"] += 1.0 if hits else 0.0
            sums[f"capped_recall_at_{k}"] += hits / min(k, len(relevant_set))
            sums[f"precision_at_{k}"] += hits / k

    return {
        "recall_at_5":      float(raw["recall@5"]),
        "recall_at_10":     float(raw["recall@10"]),
        "mrr":              float(raw["mrr"]),
        "ndcg_at_10":       float(raw["ndcg@10"]),
        **{k: v / n for k, v in sums.items()},
        "num_queries":      n,
        "total_queries":    total_queries,
    }


def article_key(source: str, madde_no) -> "str | None":
    """Chunker-independent id of a law article: ``"<source>||<madde_no>"``."""
    return f"{source}||{str(madde_no).lower()}" if source and madde_no else None


def compute_article_metrics(metric_input: list[dict]) -> dict:
    """Article-level retrieval metrics: :func:`compute_all_metrics` over the
    (law, article) ids of the ranked chunks instead of chunk ids.

    Entries need ``relevant_articles`` and ``retrieved_articles`` (see
    ``pipeline.evaluation.prepare_metric_input``); a chunk without an article
    keeps its own id so it still occupies its rank.  Two chunks of the same
    article count once, so the numbers do not move when the chunk size does.
    """
    return compute_all_metrics([
        {"query_id": m["query_id"],
         "relevant": m.get("relevant_articles") or [],
         "retrieved": m.get("retrieved_articles") or []}
        for m in metric_input
    ])
