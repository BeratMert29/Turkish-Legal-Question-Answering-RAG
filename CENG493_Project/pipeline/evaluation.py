"""Evaluation orchestration for RAG pipeline stages.

Every import of heavy modules (torch, sentence_transformers, evaluation.*,
generation.*, retrieval.*) is deferred to function bodies so the module
loads on a CPU-only / light-deps test environment.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pipeline.stages import StageConfig
    from retrieval.bm25_retriever import BM25Index
    from retrieval.embedder import Embedder
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from retrieval.graph_index import GraphIndex


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
        seen: set[str] = set()
        deduped: list[str] = []
        for c in chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                deduped.append(c["chunk_id"])
        metric_input.append({
            "query_id": qa.query_id,
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved": deduped,
            "source_law": qa.source,
            "retrieved_sources": [c.get("source", "") for c in chunks],
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
            answer = pipeline.generate(qa.question, ctx)
            if inject_citations_fn is not None:
                answer = inject_citations_fn(answer, ctx_chunks)
            predictions.append({
                "query_id": qa.query_id,
                "question": qa.question,
                "predicted": answer,
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
            })

    return predictions


# ---------------------------------------------------------------------------
# Hallucination evaluation
# ---------------------------------------------------------------------------

def run_hallucination_eval(
    predictions: list[dict],
    full_retrieved: dict[str, list],
    sample_size: int,
    *,
    nli_model=None,
    llm_model: Optional[str] = None,
) -> tuple[dict, float]:
    """Run hallucination analysis.

    Loads NLI model if *nli_model* is ``None``.  Frees VRAM from the generation
    LLM before loading the NLI cross-encoder.

    Returns
    -------
    tuple[dict, float]
        ``(hallucination_result, faithful_rate)``
    """
    import gc
    import torch
    from sentence_transformers import CrossEncoder
    from evaluation.hallucination import run_hallucination_analysis, stratified_sample
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
        try:
            nli_model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small", device=_nli_device,
            )
        except torch.cuda.OutOfMemoryError:
            nli_model = CrossEncoder(
                "cross-encoder/nli-deberta-v3-small", device="cpu",
            )

    sample = stratified_sample(predictions, sample_size)
    hall = run_hallucination_analysis(sample, full_retrieved, nli_model)

    afr = hall["summary"].get("answer_faithfulness_rate")
    faithful_rate = (
        afr if afr is not None
        else hall["summary"].get("context_grounding_rate", 0.0)
    )

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
    sample_size: int = 20,
    results_dir: Optional[Path] = None,
) -> dict:
    """Run all four LLM judge metrics.

    Returns a dict with keys: ``score``, ``faithfulness``, ``relevancy``,
    ``coherence``, ``parse_failures``.
    """
    from evaluation.llm_judge import (
        llm_judge_answer,
        llm_judge_faithfulness,
        llm_judge_relevancy,
        llm_judge_coherence,
    )

    result: dict = {
        "score": None,
        "faithfulness": None,
        "relevancy": None,
        "coherence": None,
        "parse_failures": None,
    }

    judge_preds = [
        {
            **p,
            "question": next(
                (qa.question for qa in qa_examples if qa.query_id == p["query_id"]),
                p.get("query_id", ""),
            ),
        }
        for p in predictions
    ]

    judge_result = llm_judge_answer(
        judge_preds, base_url, judge_model,
        sample_size=sample_size, results_dir=results_dir,
    )
    faith_result = llm_judge_faithfulness(
        predictions, base_url, judge_model,
        sample_size=sample_size, results_dir=results_dir,
    )
    relev_result = llm_judge_relevancy(
        judge_preds, base_url, judge_model,
        sample_size=sample_size, results_dir=results_dir,
    )
    coher_result = llm_judge_coherence(
        predictions, base_url, judge_model,
        sample_size=sample_size, results_dir=results_dir,
    )

    result["score"] = judge_result["score"]
    result["faithfulness"] = faith_result["score"]
    result["relevancy"] = relev_result["score"]
    result["coherence"] = coher_result["score"]
    result["parse_failures"] = {
        "answer": judge_result.get("parse_fail_count", 0),
        "faithfulness": faith_result.get("parse_fail_count", 0),
        "relevancy": relev_result.get("parse_fail_count", 0),
        "coherence": coher_result.get("parse_fail_count", 0),
    }

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
# Saving helpers
# ---------------------------------------------------------------------------

def save_stage_results(
    final: dict,
    predictions: list[dict],
    results_dir: Path,
) -> Path:
    """Write baseline_metrics.json and predictions.jsonl to *results_dir*.

    Returns the path to baseline_metrics.json.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    pred_path = results_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return out_path


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
    eval_set_name: str = "hmgs",
    labeling_coverage: Optional[dict] = None,
) -> dict:
    """Run a single ablation stage end-to-end.

    Returns the final_results dict (same JSON schema as run_baseline).
    """
    import config
    from pipeline.retrieval import retrieve
    from utils import inject_citations as _inject_citations

    print(f"\n{'━' * 66}")
    print(f"  {stage.name}")
    print(f"{'━' * 66}")

    # -- Embedding model ---------------------------------------------------
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

    # -- FAISS index (rebuild when embedding changes) ----------------------
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

    # -- BM25 index --------------------------------------------------------
    needs_bm25 = stage.retrieval in ("hybrid", "rrf")
    bm25 = None
    if needs_bm25:
        if "bm25" not in bm25_cache:
            from retrieval.bm25_retriever import BM25Index

            print("  Building BM25 index …")
            b = BM25Index()
            b.build([{"text": c.text, "chunk_id": c.chunk_id}
                      for c in corpus_chunks])
            bm25_cache["bm25"] = b
        bm25 = bm25_cache["bm25"]

    # -- Reranker ----------------------------------------------------------
    reranker = None
    if stage.use_rerank:
        if "reranker" not in reranker_cache:
            from retrieval.reranker import Reranker

            print(f"  Loading reranker: {config.RERANKER_MODEL}")
            r = Reranker()
            r.load_model()
            reranker_cache["reranker"] = r
        reranker = reranker_cache["reranker"]

    # -- Graph index -------------------------------------------------------
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
                print(f"  Loading graph index: {graph_path}")
                reranker_cache["graph_index"] = GraphIndex(
                    graph_path, meta_path,
                )
            else:
                print(f"  WARNING: graph.json not found at {graph_path}")
        graph_index = reranker_cache.get("graph_index")

    # -- LLM selection -----------------------------------------------------
    llm_model = (
        config.LLM_FINETUNED_MODEL
        if stage.llm == "finetuned"
        else config.LLM_MODEL
    )

    # -- Retrieval ---------------------------------------------------------
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

    # -- Retrieval metrics -------------------------------------------------
    from evaluation.retrieval_metrics import compute_all_metrics, compute_source_hit_metrics

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
        f"R@5={retrieval_metrics.get('recall_at_5', 0):.4f}  "
        f"R@10={retrieval_metrics.get('recall_at_10', 0):.4f}  "
        f"MRR={retrieval_metrics.get('mrr', 0):.4f}  "
        f"nDCG@10={retrieval_metrics.get('ndcg_at_10', 0):.4f}"
    )

    # -- Generation --------------------------------------------------------
    print(f"  Generation with {llm_model} …")
    from generation.rag_pipeline import RAGPipeline

    max_tokens = (
        config.LLM_FINETUNED_MAX_TOKENS
        if stage.llm == "finetuned"
        else config.LLM_MAX_TOKENS
    )
    pipeline = RAGPipeline(
        retriever,
        model=llm_model,
        max_tokens=max_tokens,
        short_answer_mode=short_answer_mode,
    )

    _inject_fn = _inject_citations if stage.inject_citations else None
    predictions = run_generation_loop(
        pipeline, qa_examples, retrieved_all,
        stage_key=stage_key, inject_citations_fn=_inject_fn,
    )

    failed = [p for p in predictions if not p.get("predicted")]
    predictions = [p for p in predictions if p.get("predicted")]
    if failed:
        print(f"    Filtered {len(failed)} failed generation(s) from QA metrics.")

    from evaluation.qa_metrics import compute_all_qa_metrics_with_citation

    qa_metrics = compute_all_qa_metrics_with_citation(predictions)
    print(
        f"    F1={qa_metrics.get('f1', 0):.4f}  "
        f"ROUGE-L={qa_metrics.get('rouge_l', 0):.4f}  "
        f"Citation={qa_metrics.get('citation_accuracy', 0):.4f}"
    )

    # -- Perplexity --------------------------------------------------------
    print("  Perplexity …")
    perplexity_score = None
    try:
        from evaluation.perplexity import compute_perplexity

        perplexity_score = compute_perplexity(
            predictions, model=llm_model,
            hf_model_id=config.HF_PERPLEXITY_MODEL,
        )
    except Exception as exc:
        print(f"    Perplexity=N/A ({exc.__class__.__name__}: {exc})")
    else:
        if perplexity_score is not None:
            print(f"    Perplexity={perplexity_score:.2f}")
        else:
            print("    Perplexity=N/A (logprobs not supported)")

    # -- RAGAS -------------------------------------------------------------
    print("  RAGAS metrics …")
    from evaluation.ragas_metrics import compute_ragas_metrics

    ragas_scores = compute_ragas_metrics(predictions, llm_model=llm_model)
    if ragas_scores:
        print(
            f"    RAGAS faithfulness={ragas_scores.get('ragas_faithfulness', 'N/A')}  "
            f"relevancy={ragas_scores.get('ragas_answer_relevancy', 'N/A')}  "
            f"ctx_precision={ragas_scores.get('ragas_context_precision', 'N/A')}  "
            f"ctx_recall={ragas_scores.get('ragas_context_recall', 'N/A')}"
        )
    else:
        print("    RAGAS=N/A (install: pip install ragas langchain-ollama)")

    # -- Hallucination -----------------------------------------------------
    print("  Hallucination analysis …")
    hall, faithful_rate, nli_model = run_hallucination_eval(
        predictions, full_retrieved,
        config.HALLUCINATION_SAMPLE_SIZE,
        nli_model=getattr(run_stage, "_nli_model", None),
        llm_model=llm_model,
    )
    run_stage._nli_model = nli_model  # reuse across stages
    print(f"    Faithfulness={faithful_rate:.4f}")

    # -- LLM Judge ---------------------------------------------------------
    print(f"  LLM Judge (sample={min(20, len(predictions))}) …")
    llm_judge_score = None
    llm_relevancy_score = None
    llm_coherence_score = None
    llm_faithfulness_score = None
    llm_judge_parse_failures: Optional[dict] = None

    try:
        _judge_sample = getattr(config, "LLM_JUDGE_SAMPLE_SIZE", 20)
        judge = run_llm_judge_eval(
            predictions, qa_examples,
            base_url=config.LLM_BASE_URL,
            judge_model=config.LLM_JUDGE_MODEL,
            sample_size=_judge_sample,
            results_dir=stage.results_dir,
        )
        llm_judge_score = judge["score"]
        llm_faithfulness_score = judge["faithfulness"]
        llm_relevancy_score = judge["relevancy"]
        llm_coherence_score = judge["coherence"]
        llm_judge_parse_failures = judge["parse_failures"]
    except Exception as exc:
        print(f"    WARNING: LLM Judge failed: {exc}")

    # -- Semantic Similarity -----------------------------------------------
    print("  Semantic similarity …")
    sem_sim = 0.0
    try:
        from evaluation.semantic_similarity import compute_semantic_similarity

        sem_result = compute_semantic_similarity(predictions)
        sem_sim = sem_result["mean_similarity"]
        print(f"    SemanticSim={sem_sim:.4f}")
    except Exception as exc:
        print(f"    WARNING: Semantic similarity failed: {exc}")

    # -- Final Scenario Scores ---------------------------------------------
    from evaluation.final_score import compute_all_scenario_scores

    llm_scores_dict: dict = {}
    if llm_faithfulness_score is not None:
        llm_scores_dict["faithfulness"] = llm_faithfulness_score
    if llm_relevancy_score is not None:
        llm_scores_dict["relevancy"] = llm_relevancy_score
    if llm_coherence_score is not None:
        llm_scores_dict["coherence"] = llm_coherence_score

    scenario_scores = compute_all_scenario_scores(
        retrieval_metrics=retrieval_metrics,
        qa_metrics=qa_metrics,
        faithfulness_score=faithful_rate,
        semantic_similarity=sem_sim,
        llm_scores=llm_scores_dict if llm_scores_dict else None,
    )
    print(
        f"    Scenario1={scenario_scores['scenario1']:.4f}  "
        f"Scenario2={scenario_scores['scenario2']:.4f}  "
        f"Scenario3={scenario_scores['scenario3']:.4f}"
    )

    # -- Assemble & save ---------------------------------------------------
    final = {
        "hyperparameters": {
            "stage": stage_key,
            "stage_name": stage.name,
            "eval_set": eval_set_name,
            "eval_n": len(qa_examples),
            "embedding_model": (
                config.FINETUNED_EMBEDDING_MODEL
                if stage.embedding == "finetuned"
                else config.EMBEDDING_MODEL
            ),
            "retrieval_mode": (
                stage.retrieval + ("_rerank" if stage.use_rerank else "")
            ),
            "llm_model": llm_model,
            "inject_citations": stage.inject_citations,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
        },
        "headline_metrics": {
            "source_hit_at_5": source_metrics.get("source_hit_at_5_all"),
            "source_hit_at_10": source_metrics.get("source_hit_at_10_all"),
            "source_mrr": source_metrics.get("source_mrr_all"),
            "source_precision_at_5": source_metrics.get(
                "source_precision_at_5_all",
            ),
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
        "llm_judge_score": llm_judge_score,
        "llm_faithfulness_score": llm_faithfulness_score,
        "llm_relevancy_score": llm_relevancy_score,
        "llm_coherence_score": llm_coherence_score,
        "llm_judge_parse_failures": llm_judge_parse_failures,
        "semantic_similarity": sem_sim,
        "scenario1_score": scenario_scores["scenario1"],
        "scenario2_score": scenario_scores["scenario2"],
        "scenario3_score": scenario_scores["scenario3"],
        "perplexity": perplexity_score,
        "ragas_scores": ragas_scores or {},
    }

    out_path = save_stage_results(final, predictions, stage.results_dir)
    print(f"  ✓ Results → {out_path}")
    return final


# ---------------------------------------------------------------------------
# Ablation table
# ---------------------------------------------------------------------------

def print_ablation_table(
    results: dict[str, dict],
    stage_order: Optional[list[str]] = None,
) -> None:
    """Print two markdown-style ablation tables to stdout.

    Table 1 (PRIMARY) -- source-level retrieval + QA + judge.
    Table 2 (SECONDARY) -- chunk-level recall/MRR/NDCG (gold-labeled subset).
    """
    if stage_order is None:
        from pipeline.stages import DEFAULT_STAGE_ORDER
        stage_order = DEFAULT_STAGE_ORDER

    def _pct(v) -> str:
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    def _f4(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

    # -- Table 1: PRIMARY --------------------------------------------------
    h1 = (
        f"| {'Stage':<26} | {'SrcHit@5':>8} | {'SrcHit@10':>9} | "
        f"{'SrcMRR':>7} | {'SrcPrec@5':>9} | {'n_src':>6} | "
        f"{'F1':>6} | {'Contain':>7} | {'ROUGE-L':>7} | {'Citation':>8} | "
        f"{'Faith.':>7} | {'LLM-J':>6} | {'SemSim':>7} |"
    )
    sep1 = "|" + "|".join(
        ["-" * w for w in [28, 10, 11, 9, 11, 8, 8, 9, 9, 10, 9, 8, 9]]
    ) + "|"

    print("\n\n" + "=" * 140)
    print("  PRIMARY ABLATION TABLE  "
          "(source-level retrieval — all queries with known law)")
    print("=" * 140)
    print(h1)
    print(sep1)

    for stage_key in stage_order:
        if stage_key not in results:
            continue
        r = results[stage_key]
        sm = r.get("source_hit_metrics", r.get("headline_metrics", {}))
        qa = r.get("qa_metrics", {})
        stage_name = r.get("hyperparameters", {}).get("stage_name", stage_key)
        n_src = sm.get(
            "source_labeled_queries", sm.get("n_source_queries", "?"),
        )
        print(
            f"| {stage_name:<26} | "
            f"{_f4(sm.get('source_hit_at_5_all', sm.get('source_hit_at_5'))):>8} | "
            f"{_f4(sm.get('source_hit_at_10_all', sm.get('source_hit_at_10'))):>9} | "
            f"{_f4(sm.get('source_mrr_all', sm.get('source_mrr'))):>7} | "
            f"{_f4(sm.get('source_precision_at_5_all', sm.get('source_precision_at_5'))):>9} | "
            f"{str(n_src):>6} | "
            f"{_pct(qa.get('f1')):>6} | "
            f"{_pct(qa.get('answer_containment')):>7} | "
            f"{_pct(qa.get('rouge_l')):>7} | "
            f"{_pct(qa.get('citation_accuracy')):>8} | "
            f"{_pct(r.get('faithfulness_rate')):>7} | "
            f"{_f4(r.get('llm_judge_score')):>6} | "
            f"{_f4(r.get('semantic_similarity')):>7} |"
        )
    print("=" * 140 + "\n")

    # -- Table 2: SECONDARY ------------------------------------------------
    h2 = (
        f"| {'Stage':<26} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6} | "
        f"{'nDCG@10':>7} | {'n_gold':>7} | "
        f"{'Scen1':>7} | {'Scen2':>7} | {'Scen3':>7} |"
    )
    sep2 = "|" + "|".join(
        ["-" * w for w in [28, 8, 8, 8, 9, 9, 9, 9, 9]]
    ) + "|"

    print("=" * 100)
    print("  SECONDARY TABLE  "
          "(chunk-level — gold-labeled subset only; "
          "n_gold may be small for HMGS)")
    print("=" * 100)
    print(h2)
    print(sep2)

    for stage_key in stage_order:
        if stage_key not in results:
            continue
        r = results[stage_key]
        ret = r.get("retrieval_metrics", {})
        n_gold = ret.get("num_queries", "?")
        stage_name = r.get("hyperparameters", {}).get("stage_name", stage_key)
        print(
            f"| {stage_name:<26} | "
            f"{_f4(ret.get('recall_at_5')):>6} | "
            f"{_f4(ret.get('recall_at_10')):>6} | "
            f"{_f4(ret.get('mrr')):>6} | "
            f"{_f4(ret.get('ndcg_at_10')):>7} | "
            f"{str(n_gold):>7} | "
            f"{_f4(r.get('scenario1_score')):>7} | "
            f"{_f4(r.get('scenario2_score')):>7} | "
            f"{_f4(r.get('scenario3_score')):>7} |"
        )
    print("=" * 100 + "\n")
