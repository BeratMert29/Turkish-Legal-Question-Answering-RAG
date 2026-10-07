"""Ablation reporting: headline view, ablation tables and paired stage
comparisons (printed by scripts/14 and stored in ablation_summary.json)."""

from __future__ import annotations

from typing import Optional


# One-factor ablation pairs (earlier stage, later stage): each pair differs in
# exactly one component, so a paired difference isolates that component.
ABLATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("base", "hybrid"),        # dense -> linear BM25+dense fusion
    ("base", "rrf"),           # dense -> RRF fusion
    ("rrf", "rrf_rerank"),     # + cross-encoder rerank
    ("rrf_rerank", "graph"),   # + graph neighbours in the context
    ("base", "llm_ft"),        # base LLM -> LoRA LLM (dense retrieval)
    ("rrf_rerank", "emb_ft"),  # base -> fine-tuned embedding
    ("emb_ft", "full"),        # base LLM -> LoRA LLM (best retrieval)
)

COMPARISON_METRICS: tuple[str, ...] = (
    "recall_at_5", "reciprocal_rank", "article_reciprocal_rank",
    "f1", "token_recall", "chrf", "answer_containment", "rouge_l",
    "cite_precision_native",
    "semantic_similarity", "nli_context_supported", "nli_gold_claim_recall",
    "judge_answer",
)


def compare_stages(
    per_query_by_stage: dict[str, list[dict]],
    pairs=ABLATION_PAIRS,
    metrics=COMPARISON_METRICS,
    alpha: float = 0.05,
) -> dict[str, dict[str, dict]]:
    """Paired bootstrap of later-minus-earlier stage per metric, aligned by
    query_id, with Holm correction across the pairs of each metric.

    Returns ``{metric: {"a->b": {mean_diff, ci_low, ci_high, p_value, p_holm,
    significant_holm, n, ...}}}`` for the pairs whose stages are both present.
    """
    from evaluation.stats import holm_adjust, paired_bootstrap

    out: dict[str, dict[str, dict]] = {}
    for metric in metrics:
        rows: dict[str, dict] = {}
        for a, b in pairs:
            if a not in per_query_by_stage or b not in per_query_by_stage:
                continue
            xa = {str(r["query_id"]): r.get(metric) for r in per_query_by_stage[a]}
            xb = {str(r["query_id"]): r.get(metric) for r in per_query_by_stage[b]}
            res = paired_bootstrap(xa, xb)
            if res["n"]:
                rows[f"{a}->{b}"] = res
        adjusted = holm_adjust({k: v["p_value"] for k, v in rows.items()})
        for k, res in rows.items():
            res["p_holm"] = adjusted[k]
            res["significant_holm"] = adjusted[k] is not None and adjusted[k] < alpha
        if rows:
            out[metric] = rows
    return out


def print_comparison_table(comparisons: dict[str, dict[str, dict]]) -> None:
    """Print paired stage deltas: mean diff [95% CI], Holm p, n."""
    if not comparisons:
        return
    print("=" * 100)
    print("  PAIRED STAGE COMPARISONS  (later - earlier, same queries; "
          "* = significant after Holm, alpha 0.05)")
    print("=" * 100)
    print(f"| {'Metric':<22} | {'Pair':<22} | {'Δ mean':>8} | {'95% CI':>19} | "
          f"{'p_holm':>7} | {'n':>4} |")
    print("|" + "|".join("-" * w for w in (24, 24, 10, 21, 9, 6)) + "|")
    for metric, rows in comparisons.items():
        for pair, r in rows.items():
            star = "*" if r.get("significant_holm") else " "
            print(f"| {metric:<22} | {pair:<22} | {r['mean_diff']:+8.4f} | "
                  f"[{r['ci_low']:+.4f},{r['ci_high']:+.4f}] | "
                  f"{r['p_holm']:6.4f}{star} | {r['n']:>4} |")
    print("=" * 100 + "\n")


def headline_mode(n_labeled: int, n_total: int,
                  min_fraction: Optional[float] = None) -> str:
    """Pick the headline retrieval view from the gold-labeled fraction.

    Returns ``"chunk"`` (recall/MRR/nDCG are the headline) when at least
    ``min_fraction`` of the queries have gold chunk labels, else ``"source"``
    (source-hit stays the headline).  Independent of the eval-set name.
    """
    from config import HEADLINE_CHUNK_MIN_LABELED_FRACTION
    if min_fraction is None:
        min_fraction = HEADLINE_CHUNK_MIN_LABELED_FRACTION
    if not n_total:
        return "source"
    return "chunk" if n_labeled / n_total >= min_fraction else "source"


def print_ablation_table(
    results: dict[str, dict],
    stage_order: Optional[list[str]] = None,
) -> None:
    """Print two markdown-style ablation tables to stdout.

    The PRIMARY table leads with chunk-level recall/MRR/nDCG when at least
    ``config.HEADLINE_CHUNK_MIN_LABELED_FRACTION`` of the queries have gold
    chunk labels (see :func:`headline_mode`), otherwise with source-hit.
    The SECONDARY table carries whichever retrieval view is not the headline.
    """
    if stage_order is None:
        from pipeline.stages import DEFAULT_STAGE_ORDER
        stage_order = DEFAULT_STAGE_ORDER

    def _pct(v) -> str:
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    def _f4(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

    def _src_cells(r) -> list:
        sm = r.get("source_hit_metrics", r.get("headline_metrics", {}))
        n = sm.get("source_labeled_queries", sm.get("n_source_queries", "?"))
        return [
            _f4(sm.get("source_hit_at_5_all", sm.get("source_hit_at_5"))),
            _f4(sm.get("source_hit_at_10_all", sm.get("source_hit_at_10"))),
            _f4(sm.get("source_mrr_all", sm.get("source_mrr"))),
            _f4(sm.get("source_precision_at_5_all", sm.get("source_precision_at_5"))),
            str(n),
        ]

    def _chunk_cells(r) -> list:
        ret = r.get("retrieval_metrics", {})
        return [
            _f4(ret.get("recall_at_5")), _f4(ret.get("recall_at_10")),
            _f4(ret.get("mrr")), _f4(ret.get("ndcg_at_10")),
            str(ret.get("num_queries", "?")),
        ]

    def _ci(r, metric) -> str:
        ci = (r.get("confidence_intervals") or {}).get(metric) or {}
        lo, hi = ci.get("ci_low"), ci.get("ci_high")
        if lo is None or hi is None:
            return "N/A"
        return f"[{lo * 100:.1f},{hi * 100:.1f}]"

    def _len(v) -> str:
        return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"

    def _name(r, key) -> str:
        name = r.get("hyperparameters", {}).get("stage_name", key)
        return name if r.get("status", "ok") == "ok" else f"{name} [FAILED]"

    src_hdr = ["SrcHit@5", "SrcHit@10", "SrcMRR", "SrcPrec@5", "n_src"]
    chunk_hdr = ["R@5", "R@10", "MRR", "nDCG@10", "n_gold"]

    mode = "source"
    for stage_key in stage_order:
        cov = (results.get(stage_key) or {}).get("labeling_coverage")
        if cov:
            mode = headline_mode(cov.get("labeled", 0), cov.get("total", 0))
            break
    if mode == "chunk":
        head_hdr, head_cells = chunk_hdr, _chunk_cells
        side_hdr, side_cells = src_hdr, _src_cells
        head_title = "chunk-level retrieval — gold-labeled queries"
        side_title = "source-level — all queries with known law"
    else:
        head_hdr, head_cells = src_hdr, _src_cells
        side_hdr, side_cells = chunk_hdr, _chunk_cells
        head_title = "source-level retrieval — all queries with known law"
        side_title = ("chunk-level — gold-labeled subset only; "
                      "n_gold may be small for HMGS")

    # -- Table 1: PRIMARY --------------------------------------------------
    h1 = (
        f"| {'Stage':<26} | "
        + " | ".join(f"{h:>{w}}" for h, w in zip(head_hdr, [8, 9, 7, 9, 6]))
        + f" | {'F1':>6} | {'F1 95% CI':>13} | {'Contain':>7} | {'ROUGE-L':>7} | {'chrF++':>6} | "
        f"{'CiteP-nat':>9} | {'CiteP-rnd':>9} | {'Ctx-NLI':>7} | {'ClaimR':>6} | "
        f"{'LLM-J':>6} | {'SemSim':>7} | {'AnsLen':>6} |"
    )
    sep1 = "|" + "|".join(
        ["-" * w for w in [28, 10, 11, 9, 11, 8, 8, 15, 9, 9, 8, 11, 11, 9, 8, 8, 9, 8]]
    ) + "|"

    print("\n\n" + "=" * 190)
    print(f"  PRIMARY ABLATION TABLE  ({head_title})")
    print("=" * 190)
    print(h1)
    print(sep1)

    for stage_key in stage_order:
        if stage_key not in results:
            continue
        r = results[stage_key]
        qa = r.get("qa_metrics", {})
        cite = (qa.get("citation_article_level") or {}).get("native") or {}
        stage_name = _name(r, stage_key)
        cells = " | ".join(
            f"{c:>{w}}" for c, w in zip(head_cells(r), [8, 9, 7, 9, 6])
        )
        print(
            f"| {stage_name:<26} | {cells} | "
            f"{_pct(qa.get('f1')):>6} | "
            f"{_ci(r, 'f1'):>13} | "
            f"{_pct(qa.get('answer_containment')):>7} | "
            f"{_pct(qa.get('rouge_l')):>7} | "
            f"{_pct(qa.get('chrf')):>6} | "
            f"{_pct(cite.get('precision')):>9} | "
            f"{_pct(cite.get('random_precision')):>9} | "
            f"{_pct(r.get('faithfulness_rate')):>7} | "
            f"{_pct(r.get('gold_claim_recall')):>6} | "
            f"{_f4(r.get('llm_judge_score')):>6} | "
            f"{_f4(r.get('semantic_similarity')):>7} | "
            f"{_len(qa.get('mean_answer_len_words')):>6} |"
        )
    print("=" * 190 + "\n")

    # -- Table 2: SECONDARY ------------------------------------------------
    h2 = (
        f"| {'Stage':<26} | "
        + " | ".join(f"{h:>{w}}" for h, w in zip(side_hdr, [9, 9, 7, 9, 7]))
        + f" | {'ArtHit@5':>8} | {'ArtMRR':>7} | {'Scen1':>7} | {'Scen2':>7} | {'Scen3':>7} |"
    )
    sep2 = "|" + "|".join(
        ["-" * w for w in [28, 11, 11, 9, 11, 9, 10, 9, 9, 9, 9]]
    ) + "|"

    print("=" * 100)
    print(f"  SECONDARY TABLE  ({side_title})")
    print("=" * 100)
    print(h2)
    print(sep2)

    for stage_key in stage_order:
        if stage_key not in results:
            continue
        r = results[stage_key]
        stage_name = _name(r, stage_key)
        cells = " | ".join(
            f"{c:>{w}}" for c, w in zip(side_cells(r), [9, 9, 7, 9, 7])
        )
        art = (r.get("retrieval_metrics") or {}).get("article_level") or {}
        print(
            f"| {stage_name:<26} | {cells} | "
            f"{_f4(art.get('hit_at_5')):>8} | {_f4(art.get('mrr')):>7} | "
            f"{_f4(r.get('scenario1_score')):>7} | "
            f"{_f4(r.get('scenario2_score')):>7} | "
            f"{_f4(r.get('scenario3_score')):>7} |"
        )
    print("=" * 100 + "\n")
