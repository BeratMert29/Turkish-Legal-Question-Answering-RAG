"""
evaluation/citation_metrics.py — article-level citation precision / recall.

The law-level ``citation_accuracy`` in qa_metrics asks "did a [Kaynak N]
point at a chunk of the gold law"; with ~2 distinct laws in a 5-chunk
context it is close to the retrieval hit rate.  Here a citation is checked
against the gold *article*:

* precision  — fraction of the answer's cited context chunks that belong to
               a gold article (answers with >= 1 valid citation);
* recall     — fraction of the gold articles present in the context that the
               answer cites (answers whose context holds a gold article);
* random     — the precision of citing one context chunk uniformly at random
               (fraction of gold chunks in the context), on the same answers,
               so precision can be read against chance.

Computed for the model's own citations (``predicted_native``) and, when
present, for the answer after citation injection (``predicted``).
"""

from __future__ import annotations

import re

_CITATION = re.compile(r"\[\s*kaynak\s+(\d+)\s*\]", re.IGNORECASE)


def cited_positions(text: str) -> list[int]:
    """1-based [Kaynak N] numbers in order of first appearance."""
    return list(dict.fromkeys(int(m.group(1)) for m in _CITATION.finditer(text or "")))


def _mean(values: list[float]):
    return sum(values) / len(values) if values else None


def citation_scores(answer: str, context_chunks: list[dict], gold: set[str],
                    chunk_articles: dict[str, str]) -> dict:
    """Per-answer citation precision / recall / random baseline (None when
    undefined) and the count of out-of-range citations."""
    arts = [chunk_articles.get(c.get("chunk_id", "")) for c in context_chunks]
    positions = cited_positions(answer)
    valid = [p for p in positions if 1 <= p <= len(context_chunks)]
    cited = {arts[p - 1] for p in valid if arts[p - 1]}
    gold_in_ctx = gold & {a for a in arts if a}
    rec = {"precision": None, "recall": None, "random_precision": None,
           "n_citations": len(positions), "n_invalid": len(positions) - len(valid)}
    if valid and gold:
        rec["precision"] = sum(1 for p in valid if arts[p - 1] in gold) / len(valid)
        rec["random_precision"] = sum(1 for a in arts if a in gold) / len(arts)
    if gold_in_ctx:
        rec["recall"] = len(cited & gold_in_ctx) / len(gold_in_ctx)
    return rec


def compute_citation_metrics(
    predictions: list[dict],
    gold_articles: dict[str, list[str]],
    chunk_articles: dict[str, str],
) -> tuple[dict, dict[str, dict]]:
    """``(summary, per_query)`` for native and injected citations.

    *gold_articles* maps query_id to its gold article ids; queries without
    gold articles are skipped.  ``per_query[qid]`` holds
    ``cite_precision_native`` / ``cite_recall_native``.
    """
    summary: dict = {}
    per_query: dict[str, dict] = {}
    for variant, field in (("native", "predicted_native"), ("injected", "predicted")):
        rows = []
        for p in predictions:
            gold = set(gold_articles.get(str(p.get("query_id")), []))
            if not gold or p.get(field) is None or not p.get("retrieved_chunks"):
                continue
            r = citation_scores(p[field], p["retrieved_chunks"], gold, chunk_articles)
            rows.append(r)
            if variant == "native":
                per_query[str(p["query_id"])] = {
                    "cite_precision_native": r["precision"],
                    "cite_recall_native": r["recall"],
                }
        if not rows:
            summary[variant] = None
            continue
        cited = [r for r in rows if r["precision"] is not None]
        summary[variant] = {
            "precision": _mean([r["precision"] for r in cited]),
            "random_precision": _mean([r["random_precision"] for r in cited]),
            "recall": _mean([r["recall"] for r in rows if r["recall"] is not None]),
            "presence_rate": sum(1 for r in rows if r["n_citations"]) / len(rows),
            "invalid_citation_rate": (
                sum(r["n_invalid"] for r in rows) / max(1, sum(r["n_citations"] for r in rows))),
            "n_answers": len(rows),
            "n_with_citations": len(cited),
        }
    return summary, per_query
