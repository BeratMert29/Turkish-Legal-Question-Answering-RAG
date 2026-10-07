"""Evaluation orchestration for RAG pipeline stages.

Every import of heavy modules (torch, sentence_transformers, evaluation.*,
generation.*, retrieval.*) is deferred to function bodies so the module
loads on a CPU-only / light-deps test environment.
"""

from __future__ import annotations

import gc as _gc
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.stages import StageConfig
    from retrieval.bm25_retriever import BM25Index
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from retrieval.graph_index import GraphIndex


# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------

#: Stores loaded heavy models (e.g. NLI cross-encoder) so they can be reused
#: across stages without re-loading. Call :func:`evict_model_cache` to free
#: GPU/CPU memory when switching between phases that have conflicting VRAM needs.
_model_cache: dict[str, Any] = {}


def evict_model_cache() -> None:
    """Clear cached models and release GPU memory.

    Call between pipeline stages to avoid having the 14B generation LLM,
    the 70B judge, and the NLI cross-encoder all resident simultaneously.
    """
    _model_cache.clear()
    _gc.collect()
    try:
        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Metric-input preparation
# ---------------------------------------------------------------------------

def prepare_metric_input(
    qa_examples,
    retrieved_all: list[list[dict]],
    relevant_map: dict,
) -> tuple[list[dict], dict[str, list]]:
    """Build the metric_input list and full_retrieved map from retrieval results.

    Returns
    -------
    tuple[list[dict], dict[str, list]]
        ``(metric_input, full_retrieved)``
    """
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
        metric_input.append({
            "query_id": qa.query_id,
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved": deduped,
            "source_law": qa.source,
            "retrieved_sources": [c.get("source", "") for c in ranked],
        })
        full_retrieved[qa.query_id] = chunks

    return metric_input, full_retrieved


# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def run_generation_loop(
    pipeline,
    qa_examples,
    retrieved_all: list[list[dict]],
    *,
    stage_key: str = "",
    inject_citations_fn=None,
) -> list[dict]:
    """Iterate over QA examples, assemble context, generate, optionally inject citations.

    Returns the list of prediction dicts.
    """
    from tqdm import tqdm

    predictions: list[dict] = []
    for qa, chunks in tqdm(
        zip(qa_examples, retrieved_all),
        total=len(qa_examples),
        desc=f"  [{stage_key}]" if stage_key else "Generating",
    ):
        try:
            ctx, ctx_chunks = pipeline.assemble_context(chunks)
            native_answer = pipeline.generate(qa.question, ctx)
            answer = native_answer
            if inject_citations_fn is not None:
                answer = inject_citations_fn(native_answer, ctx_chunks)
            predictions.append({
                "query_id": qa.query_id,
                "question": qa.question,
                "predicted": answer,
                # answer exactly as the LLM wrote it (before citation injection)
                "predicted_native": native_answer,
                "answer_len_words": len(native_answer.split()),
                "expected": qa.answer,
                "retrieved_sources": [c["source"] for c in ctx_chunks],
                "expected_source": qa.source,
                "retrieved_chunks": [dict(c) for c in ctx_chunks],
            })
        except Exception as exc:
            print(f"\n    ERROR on {qa.query_id}: {exc}")
            predictions.append({
                "query_id": qa.query_id,
                "question": qa.question,
                "predicted": "",
                "expected": qa.answer,
                "retrieved_sources": [],
                "expected_source": qa.source,
                "retrieved_chunks": [],
                "generation_error": True,
                "error": f"{type(exc).__name__}: {exc}",
            })

    return predictions


def _fmt_opt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"


def failure_rate_exceeded(failed: int, total: int, max_rate: float) -> bool:
    """True when failed/total is strictly greater than *max_rate*."""
    return total > 0 and (failed / total) > max_rate


# ---------------------------------------------------------------------------
# Hallucination evaluation
# ---------------------------------------------------------------------------

def run_hallucination_eval(
    predictions: list[dict],
    full_retrieved: dict[str, list],
    sample_size: Optional[int],
    *,
    nli_model=None,
    llm_model: Optional[str] = None,
    gold_info: Optional[dict[str, dict]] = None,
) -> tuple[dict, Optional[float], Any]:
    """Run hallucination analysis.

    *full_retrieved* maps query_id to the premise chunks, i.e. the context
    the generator saw for that query.  *sample_size* None scores every
    prediction.  Loads NLI model if *nli_model* is ``None``.  Frees VRAM from the generation
    LLM before loading the NLI cross-encoder.

    Returns
    -------
    tuple[dict, Optional[float], Any]
        ``(hallucination_result, faithful_rate, nli_model)``
    """
    import gc
    import torch
    from evaluation.hallucination import run_hallucination_analysis, stratified_sample
    from evaluation.nli import load_nli_model
    import config

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unload the generation LLM from Ollama VRAM
    if llm_model is not None:
        try:
            import requests as _req
            from urllib.parse import urlparse as _urlparse
            _ollama_base = "{0}://{1}".format(
                *_urlparse(config.LLM_BASE_URL)[:2]
            )
            _req.post(
                _ollama_base + "/api/generate",
                json={"model": llm_model, "keep_alive": 0},
                timeout=10,
            )
        except Exception:
            pass

    # Select NLI device
    if torch.cuda.is_available():
        _nli_device = "cuda"
    elif torch.backends.mps.is_available():
        _nli_device = "mps"
    else:
        _nli_device = "cpu"

    if nli_model is None:
        print("    Loading NLI model …")
        nli_model = load_nli_model(config.NLI_MODEL, device=_nli_device)

    # Stratify by whether the gold chunk/law was retrieved (hit@k), using the
    # pre-expansion ranking; falls back to source-level gold when unlabeled.
    strat_input = [
        {**p, **(gold_info or {}).get(str(p.get("query_id")), {})}
        for p in predictions
    ]
    sample = stratified_sample(strat_input, sample_size)
    hall = run_hallucination_analysis(
        sample, full_retrieved, nli_model, threshold=config.NLI_SUPPORT_THRESHOLD,
    )

    # Faithfulness headline = mean fraction of answer sentences entailed by the
    # context the generator saw; gold claim recall is reported separately in
    # hallucination_summary.
    faithful_rate = hall["summary"].get("context_supported_sentence_rate")

    return hall, faithful_rate, nli_model


# ---------------------------------------------------------------------------
# LLM Judge evaluation
# ---------------------------------------------------------------------------

def run_llm_judge_eval(
    predictions: list[dict],
    qa_examples,
    *,
    base_url: str,
    judge_model: str,
    sample_size: Optional[int] = None,
    query_ids: Optional[list[str]] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """Run all four LLM judge metrics.

    Parameters
    ----------
    query_ids:
        When supplied (pre-sampled by the caller via
        ``sample_judge_query_ids``), all four judge functions receive exactly
        these IDs and per-function sampling is bypassed.  When ``None`` the
        function builds a shared sample internally.

    Returns a dict with keys: ``score``, ``faithfulness``, ``relevancy``,
    ``coherence``, ``parse_failures``.
    """
    from evaluation.llm_judge import (
        llm_judge_answer,
        llm_judge_faithfulness,
        llm_judge_relevancy,
        llm_judge_coherence,
    )

    import config as _config
    import random as _random_mod

    if sample_size is None:
        sample_size = _config.LLM_JUDGE_SAMPLE_SIZE  # None => judge everything

    result: dict = {
        "score": None,
        "faithfulness": None,
        "relevancy": None,
        "coherence": None,
        "parse_failures": None,
        "per_sample": {},
        "failure_count": 0,
        "call_count": 0,
    }

    # O(Q) dict lookup instead of O(P*Q) next() scan.
    _qa_question_map: dict[str, str] = {
        qa.query_id: qa.question for qa in qa_examples
    }
    judge_preds = [
        {**p, "question": _qa_question_map.get(p["query_id"], p.get("query_id", ""))}
        for p in predictions
    ]

    # Shared sample: either use caller-supplied query_ids or build internally.
    # All four judge metrics must evaluate the same subset so cross-metric
    # comparisons are valid.
    if query_ids is not None:
        _shared_ids: set[str] = set(query_ids)
        _shared_preds = [p for p in predictions if p["query_id"] in _shared_ids]
        _shared_judge_preds = [p for p in judge_preds if p["query_id"] in _shared_ids]
    else:
        _rng = _random_mod.Random(42)
        _n = (
            len(predictions) if sample_size is None
            else min(sample_size, len(predictions))
        )
        if len(predictions) > _n:
            _built_ids: set[str] = set(
                _rng.sample([p["query_id"] for p in predictions], _n)
            )
            _shared_preds = [p for p in predictions if p["query_id"] in _built_ids]
            _shared_judge_preds = [
                p for p in judge_preds if p["query_id"] in _built_ids
            ]
        else:
            _shared_preds = predictions
            _shared_judge_preds = judge_preds
    _actual_n = len(_shared_preds)
    # When caller supplied query_ids, pass them through so the judge functions
    # skip their own per-function sampling entirely.
    _qids_arg = list(query_ids) if query_ids is not None else None

    judge_result = llm_judge_answer(
        _shared_judge_preds, base_url, judge_model,
        sample_size=_actual_n, results_dir=results_dir, query_ids=_qids_arg,
    )
    faith_result = llm_judge_faithfulness(
        _shared_preds, base_url, judge_model,
        sample_size=_actual_n, results_dir=results_dir, query_ids=_qids_arg,
    )
    relev_result = llm_judge_relevancy(
        _shared_judge_preds, base_url, judge_model,
        sample_size=_actual_n, results_dir=results_dir, query_ids=_qids_arg,
    )
    coher_result = llm_judge_coherence(
        _shared_preds, base_url, judge_model,
        sample_size=_actual_n, results_dir=results_dir, query_ids=_qids_arg,
    )

    result["score"] = judge_result["score"]
    result["faithfulness"] = faith_result["score"]
    result["relevancy"] = relev_result["score"]
    result["coherence"] = coher_result["score"]
    result["per_sample"] = {
        "answer": judge_result.get("per_sample", []),
        "faithfulness": faith_result.get("per_sample", []),
        "relevancy": relev_result.get("per_sample", []),
        "coherence": coher_result.get("per_sample", []),
    }
    result["parse_failures"] = {
        "answer": judge_result.get("parse_fail_count", 0),
        "faithfulness": faith_result.get("parse_fail_count", 0),
        "relevancy": relev_result.get("parse_fail_count", 0),
        "coherence": coher_result.get("parse_fail_count", 0),
    }

    _all = (judge_result, faith_result, relev_result, coher_result)
    result["failure_count"] = sum(r.get("parse_fail_count", 0) for r in _all)
    result["call_count"] = sum(r.get("sample_size", 0) for r in _all)

    _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
    print(
        f"    LLM Judge Answer={_fmt(result['score'])}  "
        f"Faith={_fmt(result['faithfulness'])}  "
        f"Relev={_fmt(result['relevancy'])}  "
        f"Coher={_fmt(result['coherence'])}"
    )
    for _name, _res in [
        ("answer", judge_result), ("faithfulness", faith_result),
        ("relevancy", relev_result), ("coherence", coher_result),
    ]:
        if _res.get("parse_fail_count", 0):
            print(
                f"    WARNING: {_res['parse_fail_count']}/{_res['sample_size']} "
                f"{_name} judge responses failed to parse"
            )

    return result


# ---------------------------------------------------------------------------
# Per-query arrays and confidence intervals
# ---------------------------------------------------------------------------

def _retrieval_per_query(metric_input: list[dict]) -> dict[str, dict]:
    """Per-query recall@5/10, reciprocal rank (gold-labeled queries only) and
    source-hit@5 (queries with a known gold law)."""
    from utils import normalize_turkish

    out: dict[str, dict] = {}
    for m in metric_input:
        rel = set(m.get("relevant") or [])
        ranked = m.get("retrieved", [])
        rec: dict = {"recall_at_5": None, "recall_at_10": None,
                     "reciprocal_rank": None, "source_hit_at_5": None}
        if rel:
            rec["recall_at_5"] = len(rel & set(ranked[:5])) / len(rel)
            rec["recall_at_10"] = len(rel & set(ranked[:10])) / len(rel)
            first = next((i for i, c in enumerate(ranked, 1) if c in rel), None)
            rec["reciprocal_rank"] = 1.0 / first if first else 0.0
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
            **{k: q.get(k) for k in ("em", "f1", "rouge_l", "bleu",
                                      "answer_containment", "answer_len_words")},
            "semantic_similarity": sem.get(qid),
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
    "f1", "rouge_l", "answer_containment", "em", "semantic_similarity",
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
    "recall_at_5", "reciprocal_rank", "f1", "answer_containment", "rouge_l",
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


def _gold_info(qa_examples, retrieved_all, relevant_map):
    """``(metric_input, gold_info)``; gold_info maps str(query_id) to the
    relevant/retrieved ids used to stratify hallucination by hit@k."""
    metric_input, _ = prepare_metric_input(qa_examples, retrieved_all, relevant_map)
    gold_info = {
        str(m["query_id"]): {"relevant": m["relevant"], "retrieved": m["retrieved"]}
        for m in metric_input
    }
    return metric_input, gold_info


def _persist_stage(final, all_predictions, metric_input, hall, sem_per_sample,
                   judge, results_dir) -> Path:
    """Attach bootstrap CIs, then write metrics, predictions and per-query arrays."""
    per_query = build_per_query(
        all_predictions, metric_input, hall, sem_per_sample, judge,
    )
    final["confidence_intervals"] = compute_confidence_intervals(per_query)
    return save_stage_results(final, all_predictions, results_dir, per_query=per_query)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------

def save_stage_results(
    final: dict,
    predictions: list[dict],
    results_dir: Path,
    per_query: Optional[list[dict]] = None,
) -> Path:
    """Write baseline_metrics.json, predictions.jsonl and (optionally)
    per_query.jsonl to *results_dir*.

    Returns the path to baseline_metrics.json.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "baseline_metrics.json"
    _tmp_out = out_path.with_suffix(".tmp")
    with open(_tmp_out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    os.replace(_tmp_out, out_path)

    pred_path = results_dir / "predictions.jsonl"
    _tmp_pred = pred_path.with_suffix(".tmp")
    with open(_tmp_pred, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    os.replace(_tmp_pred, pred_path)

    if per_query is not None:
        pq_path = results_dir / "per_query.jsonl"
        _tmp_pq = pq_path.with_suffix(".tmp")
        with open(_tmp_pq, "w", encoding="utf-8") as f:
            for r in per_query:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(_tmp_pq, pq_path)

    return out_path


# ---------------------------------------------------------------------------
# run_stage helpers
# ---------------------------------------------------------------------------

def _build_stage_components(
    stage: "StageConfig",
    corpus_chunks,
    embedder_cache: dict,
    retriever_cache: dict,
    bm25_cache: dict,
    reranker_cache: dict,
) -> tuple:
    """Load/retrieve embedder, FAISS index, BM25, reranker, graph index, and LLM.

    Returns
    -------
    tuple
        ``(embedder, retriever, bm25, reranker, graph_index, llm_model)``
    """
    import config

    # Embedding model
    emb_key = stage.embedding
    if emb_key not in embedder_cache:
        from retrieval.embedder import Embedder

        model_name = (
            config.FINETUNED_EMBEDDING_MODEL
            if emb_key == "finetuned"
            else config.EMBEDDING_MODEL
        )
        print(f"  Loading embedding model: {model_name}")
        emb = (
            Embedder(model_name=model_name)
            if "model_name" in Embedder.__init__.__code__.co_varnames
            else Embedder()
        )
        emb.load_model()
        embedder_cache[emb_key] = emb
    embedder = embedder_cache[emb_key]

    # FAISS index (rebuild when embedding changes)
    idx_key = emb_key
    if idx_key not in retriever_cache:
        from retrieval.retriever import Retriever as _Retriever

        print(f"  Building FAISS index ({len(corpus_chunks)} chunks) …")
        _ret = _Retriever(embedder)
        texts = [c.text for c in corpus_chunks]
        metadata = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
             "text": c.text, "source": c.source}
            for c in corpus_chunks
        ]
        t0 = time.time()
        _ret.build_index(texts, metadata)
        print(f"    Index built in {time.time() - t0:.1f}s")
        retriever_cache[idx_key] = _ret
    retriever = retriever_cache[idx_key]

    # BM25 index
    bm25 = None
    if stage.retrieval in ("hybrid", "rrf"):
        if "bm25" not in bm25_cache:
            from retrieval.bm25_retriever import BM25Index

            print("  Building BM25 index …")
            b = BM25Index()
            b.build([{"text": c.text, "chunk_id": c.chunk_id}
                      for c in corpus_chunks])
            bm25_cache["bm25"] = b
        bm25 = bm25_cache["bm25"]

    # Reranker
    reranker = None
    if stage.use_rerank:
        if "reranker" not in reranker_cache:
            from retrieval.reranker import Reranker

            print(f"  Loading reranker: {config.RERANKER_MODEL}")
            r = Reranker()
            r.load_model()
            reranker_cache["reranker"] = r
        reranker = reranker_cache["reranker"]

    # Graph index — validate JSON before loading; rebuild if corrupt
    graph_index = None
    if stage.use_graph:
        if "graph_index" not in reranker_cache:
            from retrieval.graph_index import GraphIndex

            graph_path = config.INDEX_DIR / config.GRAPH_FILE
            meta_path = config.INDEX_DIR / config.METADATA_FILE
            if not meta_path.exists():
                meta_path = (
                    config.BASE_DIR.parent / "results" / "index"
                    / config.METADATA_FILE
                )
            if graph_path.exists() and meta_path.exists():
                try:
                    with open(graph_path, encoding="utf-8") as _gf:
                        json.load(_gf)
                except (json.JSONDecodeError, OSError):
                    print(f"  WARNING: graph.json corrupt at {graph_path}; rebuilding …")
                    from pipeline.retrieval import auto_build_graph
                    auto_build_graph(graph_path)
                if graph_path.exists():
                    print(f"  Loading graph index: {graph_path}")
                    reranker_cache["graph_index"] = GraphIndex(graph_path, meta_path)
            else:
                print(f"  WARNING: graph.json not found at {graph_path}")
        graph_index = reranker_cache.get("graph_index")

    llm_model = (
        config.LLM_FINETUNED_MODEL
        if stage.llm == "finetuned"
        else config.LLM_MODEL
    )

    return embedder, retriever, bm25, reranker, graph_index, llm_model


def _run_generation_and_qa(
    stage_key: str,
    stage: "StageConfig",
    qa_examples,
    retrieved_all: list[list[dict]],
    retriever,
    llm_model: str,
    short_answer_mode: bool,
    inject_citations_fn,
    max_failure_rate: float,
) -> tuple[list[dict], int, int, int, bool, dict, list[dict]]:
    """Run generation loop, failure filtering, and QA metrics.

    Returns
    -------
    tuple
        ``(predictions, n_total, n_failed, n_errors, generation_failed,
        qa_metrics, all_predictions)``; *predictions* holds successful
        generations only, *all_predictions* includes failures (saved to disk).
    """
    import config
    from generation.rag_pipeline import RAGPipeline
    from evaluation.qa_metrics import compute_all_qa_metrics_with_citation

    # One-factor rule: both LLMs share max_tokens (and num_ctx, see config).
    pipeline = RAGPipeline(
        retriever,
        model=llm_model,
        max_tokens=config.LLM_MAX_TOKENS,
        short_answer_mode=short_answer_mode,
        graph_neighbor_budget=(
            config.GRAPH_NEIGHBOR_BUDGET
            if stage.use_graph and config.GRAPH_CONTEXT_RESERVE else 0
        ),
    )
    predictions = run_generation_loop(
        pipeline, qa_examples, retrieved_all,
        stage_key=stage_key, inject_citations_fn=inject_citations_fn,
    )

    n_total = len(predictions)
    failed = [p for p in predictions if not p.get("predicted")]
    n_errors = sum(1 for p in predictions if p.get("generation_error"))
    # Failed generations stay in the denominator and score 0 on every QA
    # metric (empty prediction); dropping them would inflate the means.  Only
    # successful predictions go on to the LLM-based metrics.
    all_predictions = predictions
    predictions = [p for p in all_predictions if p.get("predicted")]
    if failed:
        print(
            f"    {len(failed)}/{n_total} failed generation(s) scored 0 on "
            f"QA metrics (kept in the denominator)."
        )

    gen_failed = failure_rate_exceeded(len(failed), n_total, max_failure_rate)
    if gen_failed:
        print(
            f"    !!! WARNING: {len(failed)}/{n_total} generations failed "
            f"(> {max_failure_rate:.0%}); stage {stage_key} marked FAILED !!!"
        )

    qa_metrics = compute_all_qa_metrics_with_citation(all_predictions)
    qa_metrics["n_generation_failed_scored_zero"] = len(failed)
    print(
        f"    F1={qa_metrics.get('f1', 0):.4f}  "
        f"ROUGE-L={qa_metrics.get('rouge_l', 0):.4f}  "
        f"Cite(native)={_fmt_opt(qa_metrics.get('citation_accuracy_native'))}  "
        f"Cite(injected)={_fmt_opt(qa_metrics.get('citation_accuracy_injected'))}  "
        f"AnsLen={qa_metrics.get('mean_answer_len_words', 0):.1f}w"
    )
    return (
        predictions, n_total, len(failed), n_errors, gen_failed, qa_metrics,
        all_predictions,
    )


# ---------------------------------------------------------------------------
# run_stage phase helpers
# ---------------------------------------------------------------------------

def _run_retrieval_phase(
    stage: "StageConfig",
    qa_examples,
    retriever,
    bm25,
    reranker,
    graph_index,
    relevant_map: dict,
) -> tuple[list[list[dict]], dict, dict, dict[str, list]]:
    """Run retrieval and compute retrieval + source-hit metrics.

    Returns
    -------
    tuple
        ``(retrieved_all, retrieval_metrics, source_metrics, full_retrieved)``
    """
    from pipeline.retrieval import retrieve
    from evaluation.retrieval_metrics import compute_all_metrics, compute_source_hit_metrics

    print(
        f"  Retrieval ({stage.retrieval}, rerank={stage.use_rerank}, "
        f"graph={stage.use_graph}) …"
    )
    questions = [qa.question for qa in qa_examples]
    retrieved_all = retrieve(
        retriever, questions,
        retrieval_mode=stage.retrieval,
        bm25=bm25, reranker=reranker, graph_index=graph_index,
        use_rerank=stage.use_rerank, use_graph=stage.use_graph,
    )

    metric_input, full_retrieved = prepare_metric_input(
        qa_examples, retrieved_all, relevant_map,
    )
    retrieval_metrics = compute_all_metrics(metric_input)
    source_metrics = compute_source_hit_metrics(metric_input)

    _n_src = source_metrics.get("source_labeled_queries", 0)
    _n_gold = retrieval_metrics.get("num_queries", 0)
    print(
        f"    [PRIMARY] SourceHit@5={source_metrics.get('source_hit_at_5_all', 0):.4f}  "
        f"SourceHit@10={source_metrics.get('source_hit_at_10_all', 0):.4f}  "
        f"SourceMRR={source_metrics.get('source_mrr_all', 0):.4f}  "
        f"SourcePrec@5={source_metrics.get('source_precision_at_5_all', 0):.4f}  "
        f"[n={_n_src}]"
    )
    print(
        f"    [chunk-level, gold-labeled subset n={_n_gold}]  "
        f"R@5={_fmt_opt(retrieval_metrics.get('recall_at_5'))}  "
        f"R@10={_fmt_opt(retrieval_metrics.get('recall_at_10'))}  "
        f"MRR={_fmt_opt(retrieval_metrics.get('mrr'))}  "
        f"nDCG@10={_fmt_opt(retrieval_metrics.get('ndcg_at_10'))}"
    )
    return retrieved_all, retrieval_metrics, source_metrics, full_retrieved


def _run_supplemental_metrics(
    predictions: list[dict],
    llm_model: str,
    short_answer_mode: bool = False,
) -> tuple[Optional[float], dict]:
    """Compute perplexity (generator weights) and RAGAS scores (judge LLM).

    Returns
    -------
    tuple[Optional[float], dict]
        ``(perplexity_score, ragas_scores)``
    """
    import config as _config

    perplexity_score = None
    if not _config.PERPLEXITY_ENABLED:
        print("  Perplexity skipped (config.PERPLEXITY_ENABLED=False)")
    else:
        print("  Perplexity …")
        try:
            from evaluation.perplexity import compute_perplexity

            perplexity_score = compute_perplexity(
                predictions, model=llm_model,
                hf_model_id=_config.HF_PERPLEXITY_MODEL,
                sample_size=_config.PERPLEXITY_SAMPLE_SIZE,
                max_tokens=_config.PERPLEXITY_MAX_TOKENS,
                short_answer_mode=short_answer_mode,
            )
        except Exception as exc:
            print(f"    Perplexity=N/A ({exc.__class__.__name__}: {exc})")
        else:
            print(f"    Perplexity={_fmt_opt(perplexity_score)}")

    print("  RAGAS metrics …")
    from evaluation.ragas_metrics import compute_ragas_metrics

    ragas_scores = compute_ragas_metrics(predictions, llm_model=_config.LLM_JUDGE_MODEL)
    if ragas_scores:
        print(
            f"    RAGAS faithfulness={ragas_scores.get('ragas_faithfulness', 'N/A')}  "
            f"relevancy={ragas_scores.get('ragas_answer_relevancy', 'N/A')}  "
            f"ctx_precision={ragas_scores.get('ragas_context_precision', 'N/A')}  "
            f"ctx_recall={ragas_scores.get('ragas_context_recall', 'N/A')}"
        )
    else:
        print("    RAGAS=N/A (install: pip install ragas langchain-ollama)")

    return perplexity_score, ragas_scores


def _run_hallucination_phase(
    predictions: list[dict],
    llm_model: str,
    gold_info: Optional[dict[str, dict]] = None,
) -> tuple[dict, Optional[float]]:
    """Evict perplexity/RAGAS models, run hallucination analysis, update NLI cache.

    Each prediction is scored against its own ``retrieved_chunks`` -- the
    context the generator actually saw (after graph-slot selection and the
    context-window cut), not the raw retrieval list.

    The NLI cross-encoder is stored in and retrieved from the module-level
    ``_model_cache`` so it is reused across consecutive stages.

    Returns
    -------
    tuple[dict, Optional[float]]
        ``(hallucination_result, faithful_rate)``
    """
    import config as _config

    # Evict perplexity/RAGAS models before loading the NLI cross-encoder,
    # but keep the NLI model itself for reuse.
    nli_model = _model_cache.get("nli")
    evict_model_cache()

    print("  Hallucination analysis …")
    contexts = {p["query_id"]: p.get("retrieved_chunks", []) for p in predictions}
    hall, faithful_rate, nli_model = run_hallucination_eval(
        predictions, contexts,
        _config.HALLUCINATION_SAMPLE_SIZE,
        nli_model=nli_model,
        llm_model=llm_model,
        gold_info=gold_info,
    )
    _model_cache["nli"] = nli_model  # reuse across stages
    _claims = hall.get("summary", {}).get("gold_claim_recall")
    print(f"    Faithfulness (supported sentences)={_fmt_opt(faithful_rate)}  "
          f"GoldClaimRecall={_fmt_opt(_claims)}")
    return hall, faithful_rate


def _run_judge_phase(
    predictions: list[dict],
    qa_examples,
    stage: "StageConfig",
    stage_key: str,
    max_rate: float,
) -> dict:
    """Run all four LLM judge metrics with error/failure handling.

    Returns
    -------
    dict
        Keys: ``score``, ``faithfulness``, ``relevancy``, ``coherence``,
        ``parse_failures``, ``failure_count``, ``call_count``,
        ``crashed``, ``failed``.
    """
    import config as _config
    from evaluation.llm_judge import sample_judge_query_ids

    _judge_sample = _config.LLM_JUDGE_SAMPLE_SIZE
    # Sample ONCE here; all four judge functions receive the same IDs so every
    # metric evaluates the identical subset — enabling valid cross-metric comparison.
    _sampled_ids = sample_judge_query_ids(
        [p["query_id"] for p in predictions],
        n=_judge_sample,
    )
    print(f"  LLM Judge (sample={len(_sampled_ids)}) …")

    score = faithfulness = relevancy = coherence = None
    parse_failures: Optional[dict] = None
    per_sample: dict = {}
    failure_count = 0
    call_count = 0
    crashed = False

    try:
        j = run_llm_judge_eval(
            predictions, qa_examples,
            base_url=_config.LLM_BASE_URL,
            judge_model=_config.LLM_JUDGE_MODEL,
            sample_size=_judge_sample,
            query_ids=_sampled_ids,
            results_dir=stage.results_dir,
        )
        score = j["score"]
        faithfulness = j["faithfulness"]
        relevancy = j["relevancy"]
        coherence = j["coherence"]
        parse_failures = j["parse_failures"]
        per_sample = j.get("per_sample", {})
        failure_count = j["failure_count"]
        call_count = j["call_count"]
    except Exception as exc:
        crashed = True
        print(f"    WARNING: LLM Judge failed: {exc}")

    failed = crashed or failure_rate_exceeded(failure_count, call_count, max_rate)
    if failed:
        print(
            f"    !!! WARNING: LLM judge failures {failure_count}/{call_count}"
            f"{' (judge crashed)' if crashed else ''} exceed {max_rate:.0%}; "
            f"stage {stage_key} marked FAILED !!!"
        )

    return {
        "score": score,
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "coherence": coherence,
        "parse_failures": parse_failures,
        "per_sample": per_sample,
        "failure_count": failure_count,
        "call_count": call_count,
        "crashed": crashed,
        "failed": failed,
    }


def _run_semantic_sim_phase(
    predictions: list[dict],
) -> tuple[Optional[float], list[dict]]:
    """Mean semantic similarity between predicted and expected answers.

    Returns ``(mean, per_sample)``; ``(None, [])`` on failure (non-fatal;
    recorded as null in results).
    """
    print("  Semantic similarity …")
    try:
        from evaluation.semantic_similarity import compute_semantic_similarity

        sem_result = compute_semantic_similarity(predictions)
        sem_sim = sem_result["mean_similarity"]
        print(f"    SemanticSim={sem_sim:.4f}")
        return sem_sim, sem_result.get("per_sample", [])
    except Exception as exc:
        print(f"    WARNING: Semantic similarity failed (recorded as null): {exc}")
        return None, []


def _skipped_judge() -> dict:
    """Judge result for a stage where the judge phase did not run."""
    return {
        "score": None, "faithfulness": None, "relevancy": None, "coherence": None,
        "parse_failures": None, "per_sample": {}, "failure_count": 0,
        "call_count": 0, "crashed": False, "failed": False, "skipped": True,
    }


def _portable_path(value: str) -> str:
    """Model path relative to the project root when it lives inside it, so
    results carry no machine-specific absolute paths."""
    import config as _config

    try:
        return Path(value).resolve().relative_to(_config.BASE_DIR.resolve()).as_posix()
    except (ValueError, OSError):
        return value


def _assemble_final_result(
    stage_key: str,
    stage: "StageConfig",
    eval_set_name: str,
    qa_examples,
    llm_model: str,
    labeling_coverage: Optional[dict],
    retrieval_metrics: dict,
    source_metrics: dict,
    qa_metrics: dict,
    hall: dict,
    faithful_rate: Optional[float],
    judge: dict,
    sem_sim: Optional[float],
    perplexity_score: Optional[float],
    ragas_scores: dict,
    n_generated_total: int,
    n_generation_failed: int,
    n_generation_errors: int,
    generation_failed: bool,
    max_rate: float,
) -> dict:
    """Build the final results dict and print scenario scores.

    Scenario 1 uses chunk-level MRR only when the gold-labeled fraction
    reaches ``config.HEADLINE_CHUNK_MIN_LABELED_FRACTION`` (an MRR over a
    handful of labeled queries is not a stage-level number); otherwise MRR is
    recorded as a missing component.  Each scenario's used/missing components,
    effective weights and per-component n are saved under
    ``scenario_components``.

    Returns
    -------
    dict
        Full results dict matching the run_baseline JSON schema.
    """
    import config as _config
    from evaluation.final_score import compute_all_scenario_scores

    llm_scores_dict = {
        k: judge[k] for k in ("faithfulness", "relevancy", "coherence")
        if judge[k] is not None
    }
    n_gold = retrieval_metrics.get("num_queries") or 0
    mrr_ok = headline_mode(n_gold, retrieval_metrics.get("total_queries") or 0) == "chunk"
    judge_n = len((judge.get("per_sample") or {}).get("answer", []))
    scenario_scores = compute_all_scenario_scores(
        retrieval_metrics=retrieval_metrics if mrr_ok else {**retrieval_metrics, "mrr": None},
        qa_metrics=qa_metrics,
        faithfulness_score=faithful_rate,
        semantic_similarity=sem_sim,
        llm_scores=llm_scores_dict or None,
        n_by_component={
            "retrieval_mrr": n_gold,
            "f1": qa_metrics.get("num_samples"),
            "faithfulness": (hall.get("summary") or {}).get("total"),
            "semantic_similarity": (
                n_generated_total - n_generation_failed if sem_sim is not None else 0),
            "relevancy": judge_n,
            "coherence": judge_n,
        },
    )
    print(
        f"    Scenario1={_fmt_opt(scenario_scores['scenario1'])}  "
        f"Scenario2={_fmt_opt(scenario_scores['scenario2'])}  "
        f"Scenario3={_fmt_opt(scenario_scores['scenario3'])}"
    )

    return {
        "status": "failed" if (generation_failed or judge["failed"]) else "ok",
        "failure_counts": {
            "generation_total": n_generated_total,
            "generation_failed": n_generation_failed,
            "generation_errors": n_generation_errors,
            "llm_metrics_skipped": generation_failed,
            "judge_calls": judge["call_count"],
            "judge_failed": judge["failure_count"],
            "judge_crashed": judge["crashed"],
            "judge_skipped": judge.get("skipped", False),
            "semantic_similarity_failed": sem_sim is None and not generation_failed,
            "max_failure_rate": max_rate,
        },
        "hyperparameters": {
            "stage": stage_key,
            "stage_name": stage.name,
            "eval_set": eval_set_name,
            "eval_n": len(qa_examples),
            "embedding_model": (
                _portable_path(_config.FINETUNED_EMBEDDING_MODEL)
                if stage.embedding == "finetuned"
                else _config.EMBEDDING_MODEL
            ),
            "retrieval_mode": (
                stage.retrieval + ("_rerank" if stage.use_rerank else "")
            ),
            "llm_model": llm_model,
            "llm_judge_model": _config.LLM_JUDGE_MODEL,
            "llm_max_tokens": _config.LLM_MAX_TOKENS,
            "llm_num_ctx": _config.LLM_NUM_CTX,
            "llm_temperature": _config.LLM_TEMPERATURE,
            "llm_base_for_ablation": _config.LLM_BASE_FOR_ABLATION,
            "graph_context_reserve": (
                _config.GRAPH_NEIGHBOR_BUDGET
                if stage.use_graph and _config.GRAPH_CONTEXT_RESERVE else 0
            ),
            "inject_citations": stage.inject_citations,
            "article_chunking": _config.ARTICLE_CHUNKING_ENABLED,
            "chunk_size": _config.CHUNK_SIZE,
            "chunk_overlap": _config.CHUNK_OVERLAP,
            "top_k_retrieval": _config.TOP_K_RETRIEVAL,
            "top_k_for_generation": _config.TOP_K_FOR_GENERATION,
            "reranker_candidates": _config.RERANKER_CANDIDATES,
            "nli_model": _config.NLI_MODEL,
            "nli_support_threshold": _config.NLI_SUPPORT_THRESHOLD,
        },
        "headline_metrics": {
            "headline_mode": headline_mode(
                (labeling_coverage or {}).get("labeled", 0),
                (labeling_coverage or {}).get("total", 0),
            ),
            "source_hit_at_5": source_metrics.get("source_hit_at_5_all"),
            "source_hit_at_10": source_metrics.get("source_hit_at_10_all"),
            "source_mrr": source_metrics.get("source_mrr_all"),
            "source_precision_at_5": source_metrics.get("source_precision_at_5_all"),
            "n_source_queries": source_metrics.get("source_labeled_queries"),
            "chunk_recall_at_5_gold_only": retrieval_metrics.get("recall_at_5"),
            "chunk_mrr_gold_only": retrieval_metrics.get("mrr"),
            "chunk_ndcg_at_10_gold_only": retrieval_metrics.get("ndcg_at_10"),
            "n_gold_labeled": retrieval_metrics.get("num_queries"),
        },
        "retrieval_metrics": retrieval_metrics,
        "source_hit_metrics": source_metrics,
        "labeling_coverage": labeling_coverage,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hall.get("summary", {}),
        "faithfulness_rate": faithful_rate,
        "gold_claim_recall": (hall.get("summary") or {}).get("gold_claim_recall"),
        "llm_judge_score": judge["score"],
        "llm_faithfulness_score": judge["faithfulness"],
        "llm_relevancy_score": judge["relevancy"],
        "llm_coherence_score": judge["coherence"],
        "llm_judge_parse_failures": judge["parse_failures"],
        "semantic_similarity": sem_sim,
        "scenario1_score": scenario_scores["scenario1"],
        "scenario2_score": scenario_scores["scenario2"],
        "scenario3_score": scenario_scores["scenario3"],
        "scenario_components": {
            "faithfulness_source": scenario_scores["faithfulness_source"],
            "mrr_used": mrr_ok,
            **scenario_scores["components"],
        },
        "perplexity": perplexity_score,
        "ragas_scores": ragas_scores or {},
    }


def failed_results_dir(results_dir: Path) -> Path:
    """Where a failed stage is written, so it never replaces a good run."""
    return results_dir.with_name(results_dir.name + "_FAILED")


# ---------------------------------------------------------------------------
# run_stage -- full single-stage orchestration (used by 14_eval_all_stages)
# ---------------------------------------------------------------------------

def run_stage(
    stage_key: str,
    stage: "StageConfig",
    qa_examples,
    corpus_chunks,
    *,
    embedder_cache: dict,
    retriever_cache: dict,
    bm25_cache: dict,
    reranker_cache: dict,
    relevant_map: dict,
    short_answer_mode: bool,
    eval_set_name: str = "turkish_legal_rag",
    labeling_coverage: Optional[dict] = None,
    run_judge: bool = True,
    provenance: Optional[dict] = None,
) -> dict:
    """Run a single ablation stage end-to-end.

    When more than ``config.MAX_FAILURE_RATE`` of the generations fail, the
    LLM-based metrics (perplexity, RAGAS, NLI, judge, similarity) are skipped
    and recorded as None.  A failed stage is written to
    ``<results_dir>_FAILED`` instead of ``results_dir``.

    Returns the final_results dict (same JSON schema as run_baseline).
    """
    import config
    from utils import inject_citations as _inject_citations

    print(f"\n{'━' * 66}\n  {stage.name}\n{'━' * 66}")

    _embedder, retriever, bm25, reranker, graph_index, llm_model = (
        _build_stage_components(
            stage, corpus_chunks,
            embedder_cache, retriever_cache, bm25_cache, reranker_cache,
        )
    )
    retrieved_all, retrieval_metrics, source_metrics, _ = _run_retrieval_phase(
        stage, qa_examples, retriever, bm25, reranker, graph_index, relevant_map,
    )

    print(f"  Generation with {llm_model} …")
    _inject_fn = _inject_citations if stage.inject_citations else None
    _max_rate = config.MAX_FAILURE_RATE
    (
        predictions, n_generated_total, n_generation_failed, n_generation_errors,
        generation_failed, qa_metrics, all_predictions,
    ) = _run_generation_and_qa(
        stage_key, stage, qa_examples, retrieved_all,
        retriever, llm_model, short_answer_mode, _inject_fn, _max_rate,
    )
    metric_input, gold_info = _gold_info(qa_examples, retrieved_all, relevant_map)

    if generation_failed:
        print("  Skipping LLM-based metrics: generation failure rate exceeded.")
        perplexity_score, ragas_scores = None, {}
        hall, faithful_rate = {}, None
        judge, (sem_sim, sem_per_sample) = _skipped_judge(), (None, [])
    else:
        perplexity_score, ragas_scores = _run_supplemental_metrics(
            predictions, llm_model, short_answer_mode,
        )
        hall, faithful_rate = _run_hallucination_phase(
            predictions, llm_model, gold_info=gold_info,
        )
        judge = (
            _run_judge_phase(predictions, qa_examples, stage, stage_key, _max_rate)
            if run_judge else _skipped_judge()
        )
        sem_sim, sem_per_sample = _run_semantic_sim_phase(predictions)

    final = _assemble_final_result(
        stage_key, stage, eval_set_name, qa_examples, llm_model,
        labeling_coverage, retrieval_metrics, source_metrics, qa_metrics,
        hall, faithful_rate, judge, sem_sim,
        perplexity_score, ragas_scores,
        n_generated_total, n_generation_failed, n_generation_errors,
        generation_failed, _max_rate,
    )
    if provenance:
        final["provenance"] = provenance

    out_dir = stage.results_dir
    if final["status"] != "ok":
        out_dir = failed_results_dir(out_dir)
        print(f"  !!! Stage {stage_key} FAILED; results kept apart in {out_dir}")
    out_path = _persist_stage(
        final, all_predictions, metric_input, hall, sem_per_sample, judge, out_dir,
    )
    print(f"  ✓ Results → {out_path}")
    return final


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

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
        + f" | {'F1':>6} | {'F1 95% CI':>13} | {'Contain':>7} | {'ROUGE-L':>7} | "
        f"{'Cite-nat':>8} | {'Cite-inj':>8} | {'Ctx-NLI':>7} | {'ClaimR':>6} | "
        f"{'LLM-J':>6} | {'SemSim':>7} | {'AnsLen':>6} |"
    )
    sep1 = "|" + "|".join(
        ["-" * w for w in [28, 10, 11, 9, 11, 8, 8, 15, 9, 9, 10, 10, 9, 8, 8, 9, 8]]
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
            f"{_pct(qa.get('citation_accuracy_native')):>8} | "
            f"{_pct(qa.get('citation_accuracy_injected', qa.get('citation_accuracy'))):>8} | "
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
        + f" | {'Scen1':>7} | {'Scen2':>7} | {'Scen3':>7} |"
    )
    sep2 = "|" + "|".join(
        ["-" * w for w in [28, 11, 11, 9, 11, 9, 9, 9, 9]]
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
        print(
            f"| {stage_name:<26} | {cells} | "
            f"{_f4(r.get('scenario1_score')):>7} | "
            f"{_f4(r.get('scenario2_score')):>7} | "
            f"{_f4(r.get('scenario3_score')):>7} |"
        )
    print("=" * 100 + "\n")
