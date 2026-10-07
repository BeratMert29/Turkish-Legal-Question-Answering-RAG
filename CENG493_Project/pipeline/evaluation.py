"""Evaluation orchestration for RAG pipeline stages.

Metric input / per-query records live in ``pipeline.metric_input`` and the
ablation tables / stage comparisons in ``pipeline.report``; both are
re-exported here.

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


# Re-exported so existing imports from pipeline.evaluation keep working.
from pipeline.metric_input import (  # noqa: F401
    _CI_METRICS, _retrieval_per_query, build_per_query, chunk_article_map,
    compute_confidence_intervals, prepare_metric_input,
)
from pipeline.report import (  # noqa: F401
    ABLATION_PAIRS, COMPARISON_METRICS, compare_stages, headline_mode,
    print_ablation_table, print_comparison_table,
)


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
            meta = getattr(pipeline, "last_meta", None) or {}
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
                # hit max_tokens / an invented "Soru:" turn was cut off
                "truncated": meta.get("done_reason") == "length",
                "runaway_cut": bool(meta.get("runaway_cut")),
                "output_tokens": meta.get("output_tokens"),
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

    _named = {"answer": judge_result, "faithfulness": faith_result,
              "relevancy": relev_result, "coherence": coher_result}
    result["call_failures"] = {k: r.get("call_fail_count", 0) for k, r in _named.items()}
    result["score_failures_as_zero"] = {
        k: r.get("score_failures_as_zero") for k, r in _named.items()}
    _all = tuple(_named.values())
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
# Citation metrics, gold info
# ---------------------------------------------------------------------------

def _add_citation_metrics(qa_metrics: dict, predictions: list[dict],
                          metric_input: list[dict], chunk_articles: dict) -> None:
    """Article-level citation precision/recall (native and injected) into
    ``qa_metrics["citation_article_level"]``; per-query values are attached
    to the predictions as ``cite_precision_native`` / ``cite_recall_native``."""
    from evaluation.citation_metrics import compute_citation_metrics

    gold = {str(m["query_id"]): m.get("relevant_articles") or [] for m in metric_input}
    summary, per_query = compute_citation_metrics(predictions, gold, chunk_articles)
    qa_metrics["citation_article_level"] = summary
    for p in predictions:
        p.update(per_query.get(str(p.get("query_id")), {}))
    nat = summary.get("native") or {}
    print(f"    Citations (article, native): precision={_fmt_opt(nat.get('precision'))} "
          f"vs random={_fmt_opt(nat.get('random_precision'))}  "
          f"recall={_fmt_opt(nat.get('recall'))}  presence={_fmt_opt(nat.get('presence_rate'))}")


def _gold_info(qa_examples, retrieved_all, relevant_map, chunk_articles=None):
    """``(metric_input, gold_info)``; gold_info maps str(query_id) to the
    relevant/retrieved ids used to stratify hallucination by hit@k."""
    metric_input, _ = prepare_metric_input(
        qa_examples, retrieved_all, relevant_map, chunk_articles)
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

    # Graph index -- built in memory from this run's corpus (not from a saved
    # graph.json / metadata.jsonl, whose chunk ids may belong to another
    # chunking of the corpus).
    graph_index = None
    if stage.use_graph:
        if "graph_index" not in reranker_cache:
            from retrieval.graph_index import GraphIndex

            print("  Building graph index from the corpus …")
            reranker_cache["graph_index"] = GraphIndex.from_metadata([
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                 "source": c.source, "madde_no": getattr(c, "madde_no", None)}
                for c in corpus_chunks
            ])
        graph_index = reranker_cache["graph_index"]

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
        f"AnsLen={qa_metrics.get('mean_answer_len_words', 0):.1f}w  "
        f"Truncated={_fmt_opt(qa_metrics.get('truncated_rate'))}  "
        f"RunawayCut={_fmt_opt(qa_metrics.get('runaway_cut_rate'))}"
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
    chunk_articles: Optional[dict[str, str]] = None,
) -> tuple[list[list[dict]], dict, dict, dict[str, list]]:
    """Run retrieval and compute retrieval + source-hit metrics.

    With *chunk_articles* ``retrieval_metrics["article_level"]`` holds the
    article-level (chunker-independent) metrics.

    Returns
    -------
    tuple
        ``(retrieved_all, retrieval_metrics, source_metrics, full_retrieved)``
    """
    from pipeline.retrieval import retrieve
    from evaluation.retrieval_metrics import (
        compute_all_metrics, compute_article_metrics, compute_source_hit_metrics,
    )

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
        qa_examples, retrieved_all, relevant_map, chunk_articles,
    )
    retrieval_metrics = compute_all_metrics(metric_input)
    if chunk_articles is not None:
        art = compute_article_metrics(metric_input)
        retrieval_metrics["article_level"] = art
        print(
            f"    [article-level n={art['num_queries']}]  "
            f"Hit@5={_fmt_opt(art['hit_at_5'])}  Hit@10={_fmt_opt(art['hit_at_10'])}  "
            f"MRR={_fmt_opt(art['mrr'])}  nDCG@10={_fmt_opt(art['ndcg_at_10'])}"
        )
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
    extra: dict = {}

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
        extra = {"call_failures": j.get("call_failures"),
                 "score_failures_as_zero": j.get("score_failures_as_zero")}
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
        **extra,
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
        print(f"    SemanticSim={_fmt_opt(sem_sim)}")
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
            "article_hit_at_5": (retrieval_metrics.get("article_level") or {}).get("hit_at_5"),
            "article_mrr": (retrieval_metrics.get("article_level") or {}).get("mrr"),
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
        "llm_judge_call_failures": judge.get("call_failures"),
        # sensitivity: judge means with every missing score counted as 0
        "llm_judge_score_failures_as_zero": judge.get("score_failures_as_zero"),
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
    chunk_articles = chunk_article_map(corpus_chunks)
    retrieved_all, retrieval_metrics, source_metrics, _ = _run_retrieval_phase(
        stage, qa_examples, retriever, bm25, reranker, graph_index, relevant_map,
        chunk_articles,
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
    metric_input, gold_info = _gold_info(
        qa_examples, retrieved_all, relevant_map, chunk_articles)
    _add_citation_metrics(qa_metrics, all_predictions, metric_input, chunk_articles)

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
