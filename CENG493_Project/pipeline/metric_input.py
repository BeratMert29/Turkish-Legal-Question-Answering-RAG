"""Metric input preparation and per-query records for one stage.

``prepare_metric_input`` turns retrieval results into the rows the retrieval
metrics consume (pre-expansion ranking, deduplicated chunk ids, optional
article ids); ``build_per_query`` merges every per-query score into one record
per query for bootstrap CIs and paired stage comparisons.
"""

from __future__ import annotations

from typing import Optional


def chunk_article_map(corpus_chunks) -> dict[str, str]:
    """``{chunk_id: "<source>||<madde_no>"}`` for chunks with an article."""
    from evaluation.retrieval_metrics import article_key

    out = {}
    for c in corpus_chunks:
        key = article_key(c.source, getattr(c, "madde_no", None))
        if key:
            out[c.chunk_id] = key
    return out


def prepare_metric_input(
    qa_examples,
    retrieved_all: list[list[dict]],
    relevant_map: dict,
    chunk_articles: Optional[dict[str, str]] = None,
) -> tuple[list[dict], dict[str, list]]:
    """Build the metric_input list and full_retrieved map from retrieval results.

    With *chunk_articles* (see :func:`chunk_article_map`) every entry also
    carries ``relevant_articles`` / ``retrieved_articles`` for article-level
    metrics; a QA example with an explicit ``madde_no`` uses it as the gold
    article directly.

    Returns
    -------
    tuple[list[dict], dict[str, list]]
        ``(metric_input, full_retrieved)``
    """
    from evaluation.retrieval_metrics import article_key

    metric_input: list[dict] = []
    full_retrieved: dict[str, list] = {}

    for qa, chunks in zip(qa_examples, retrieved_all):
        # Retrieval metrics use the pre-expansion ranking: graph neighbours
        # (flagged ``graph_neighbor``) are spliced in only to feed generation.
        ranked = [c for c in chunks if not c.get("graph_neighbor")]
        seen: set[str] = set()
        deduped: list[str] = []
        for c in ranked:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                deduped.append(c["chunk_id"])
        entry = {
            "query_id": qa.query_id,
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved": deduped,
            "source_law": qa.source,
            "retrieved_sources": [c.get("source", "") for c in ranked],
        }
        if chunk_articles is not None:
            gold = article_key(qa.source, getattr(qa, "madde_no", None))
            rel_art = [gold] if gold else [
                chunk_articles[c] for c in entry["relevant"] if c in chunk_articles]
            entry["relevant_articles"] = list(dict.fromkeys(rel_art))
            entry["retrieved_articles"] = list(dict.fromkeys(
                chunk_articles.get(c, f"chunk::{c}") for c in deduped))
        metric_input.append(entry)
        full_retrieved[qa.query_id] = chunks

    return metric_input, full_retrieved


def _retrieval_per_query(metric_input: list[dict]) -> dict[str, dict]:
    """Per-query recall@5/10, reciprocal rank (gold-labeled queries only),
    article-level hit@5 / reciprocal rank and source-hit@5 (queries with a
    known gold law)."""
    from utils import normalize_turkish

    out: dict[str, dict] = {}
    for m in metric_input:
        rel = set(m.get("relevant") or [])
        ranked = m.get("retrieved", [])
        rec: dict = {"recall_at_5": None, "recall_at_10": None,
                     "reciprocal_rank": None, "source_hit_at_5": None,
                     "article_hit_at_5": None, "article_reciprocal_rank": None}
        if rel:
            rec["recall_at_5"] = len(rel & set(ranked[:5])) / len(rel)
            rec["recall_at_10"] = len(rel & set(ranked[:10])) / len(rel)
            first = next((i for i, c in enumerate(ranked, 1) if c in rel), None)
            rec["reciprocal_rank"] = 1.0 / first if first else 0.0
        rel_art = set(m.get("relevant_articles") or [])
        if rel_art:
            arts = m.get("retrieved_articles") or []
            rec["article_hit_at_5"] = float(bool(rel_art & set(arts[:5])))
            first = next((i for i, a in enumerate(arts, 1) if a in rel_art), None)
            rec["article_reciprocal_rank"] = 1.0 / first if first else 0.0
        gold = normalize_turkish(str(m.get("source_law") or "").strip())
        if gold:
            srcs = [normalize_turkish(str(x).strip())
                    for x in m.get("retrieved_sources", [])[:5]]
            rec["source_hit_at_5"] = float(gold in srcs)
        out[str(m["query_id"])] = rec
    return out


def build_per_query(
    all_predictions: list[dict],
    metric_input: list[dict],
    hall: dict,
    sem_per_sample: list[dict],
    judge: dict,
) -> list[dict]:
    """One record per query merging retrieval, QA, similarity, NLI and judge
    scores (None where a metric was not computed for that query)."""
    from evaluation.qa_metrics import compute_per_query_qa_metrics

    qa_rows = {r["query_id"]: r for r in compute_per_query_qa_metrics(all_predictions)}
    ret_rows = _retrieval_per_query(metric_input)
    sem = {str(r.get("query_id")): r.get("similarity") for r in sem_per_sample}
    nli = {str(r.get("query_id")): r for r in hall.get("per_sample", [])}
    judge_ps = judge.get("per_sample") or {}
    judge_by = {
        name: {str(r.get("query_id")): r.get("score") for r in rows}
        for name, rows in judge_ps.items()
    }

    records = []
    for p in all_predictions:
        qid = str(p["query_id"])
        q = qa_rows.get(p["query_id"], {})
        n = nli.get(qid, {})
        rec = {
            "query_id": qid,
            "generation_failed": not p.get("predicted"),
            **ret_rows.get(qid, {}),
            **{k: q.get(k) for k in ("em", "f1", "token_precision", "token_recall",
                                      "rouge_l", "bleu", "chrf",
                                      "answer_containment", "answer_len_words")},
            "semantic_similarity": sem.get(qid),
            "cite_precision_native": p.get("cite_precision_native"),
            "cite_recall_native": p.get("cite_recall_native"),
            "nli_context_grounding": n.get("context_grounding_score"),
            "nli_context_supported": n.get("context_supported_rate"),
            "nli_gold_claim_recall": n.get("gold_claim_recall"),
            "hallucination_category": n.get("category"),
        }
        for name, by in judge_by.items():
            rec[f"judge_{name}"] = by.get(qid)
        records.append(rec)
    return records


_CI_METRICS = (
    "recall_at_5", "recall_at_10", "reciprocal_rank", "source_hit_at_5",
    "article_hit_at_5", "article_reciprocal_rank",
    "f1", "token_precision", "token_recall", "rouge_l", "chrf",
    "answer_containment", "em", "semantic_similarity",
    "cite_precision_native", "cite_recall_native",
    "nli_context_grounding", "nli_context_supported", "nli_gold_claim_recall",
    "judge_answer", "judge_faithfulness",
    "judge_relevancy", "judge_coherence",
)


def compute_confidence_intervals(per_query: list[dict]) -> dict[str, dict]:
    """95% bootstrap CI of the mean for each per-query metric."""
    from evaluation.stats import bootstrap_ci

    return {
        m: bootstrap_ci([r.get(m) for r in per_query])
        for m in _CI_METRICS
        if any(r.get(m) is not None for r in per_query)
    }
