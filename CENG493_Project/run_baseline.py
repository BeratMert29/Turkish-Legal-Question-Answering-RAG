"""
Stage 1 Baseline Runner
Usage:
    python run_baseline.py --build-index --eval
    python run_baseline.py --retrieval-only
    python run_baseline.py --hybrid --retrieval-only
    python run_baseline.py --rerank --retrieval-only
    python run_baseline.py --eval --results-dir results/stage1
"""

import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import logging
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config
from data.data_processor import DataProcessor, CorpusChunk, QAExample
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from generation.rag_pipeline import RAGPipeline
from evaluation.retrieval_metrics import compute_all_metrics
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.hallucination import run_hallucination_analysis, stratified_sample
from utils import set_seeds, check_ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def build_index(processor: DataProcessor, embedder: Embedder, chunks: list[CorpusChunk] = None) -> tuple[Retriever, float]:
    if chunks is None:
        log.info("Building corpus chunks …")
        chunks = list(processor.build_corpus_chunks())
    log.info("  %d chunks total", len(chunks))

    texts = [c.text for c in chunks]
    metadata = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "source": c.source}
        for c in chunks
    ]

    retriever = Retriever(embedder)
    log.info("Encoding corpus (this may take a while) …")
    t0 = time.time()
    retriever.build_index(texts, metadata)
    build_time = round(time.time() - t0, 2)

    index_path = config.INDEX_DIR / config.INDEX_FILE
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    retriever.save_index(index_path, meta_path)
    log.info("Index saved → %s (%.1fs)", index_path, build_time)
    return retriever, build_time


def load_index(embedder: Embedder) -> Retriever:
    index_path = config.INDEX_DIR / config.INDEX_FILE
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    log.info("Loading index from %s …", index_path)
    retriever = Retriever(embedder, index_path=index_path, metadata_path=meta_path)
    log.info("  %d vectors loaded", retriever.index.ntotal)
    return retriever


def run_retrieval_eval(
    retriever: Retriever,
    qa_examples: list[QAExample],
    corpus_chunks: list[CorpusChunk],
    use_hybrid: bool = False,
    use_rerank: bool = False,
    bm25_index=None,
    use_rrf: bool = False,
    reranker=None,
) -> tuple[dict, list[dict]]:
    log.info("Building ground-truth relevance map …")
    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)

    questions = [qa.question for qa in qa_examples]

    # Step 1: Initial retrieval
    initial_k = config.RERANKER_CANDIDATES if use_rerank else config.TOP_K_RETRIEVAL
    if use_rrf and bm25_index is not None:
        log.info("Running RRF retrieval on %d queries …", len(qa_examples))
        t0 = time.time()
        all_retrieved = retriever.batch_rrf_retrieve(questions, bm25_index, top_k=initial_k)
    elif use_hybrid and bm25_index is not None:
        log.info("Running HYBRID retrieval on %d queries …", len(qa_examples))
        t0 = time.time()
        all_retrieved = retriever.batch_hybrid_retrieve(questions, bm25_index, top_k=initial_k)
    elif use_rerank:
        log.info("Running DENSE+RERANK retrieval on %d queries …", len(qa_examples))
        t0 = time.time()
        all_retrieved = retriever.batch_retrieve(questions, top_k=config.RERANKER_CANDIDATES)
    else:
        log.info("Running DENSE retrieval on %d queries …", len(qa_examples))
        t0 = time.time()
        all_retrieved = retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)

    retrieval_time = round(time.time() - t0, 2)
    log.info("  Retrieval done in %.1fs", retrieval_time)

    # Step 2: Rerank all candidates (covers dense+rerank, hybrid+rerank, rrf+rerank)
    if use_rerank:
        if reranker is None:
            from retrieval.reranker import Reranker
            reranker = Reranker()
            reranker.load_model()
        all_retrieved = reranker.batch_rerank(questions, all_retrieved, top_k=config.TOP_K_RETRIEVAL)

    results = []
    for qa, retrieved_chunks in zip(qa_examples, all_retrieved):
        results.append({
            "query_id": qa.query_id,
            "retrieved": [c["chunk_id"] for c in retrieved_chunks],
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved_chunks": [dict(c) for c in retrieved_chunks],
        })

    metrics = compute_all_metrics(results)
    metrics["retrieval_time_s"] = retrieval_time
    log.info("Retrieval metrics: %s", metrics)
    return metrics, results


def run_generation_eval(
    pipeline: RAGPipeline,
    qa_examples: list[QAExample],
    use_hybrid: bool = False,
    use_rerank: bool = False,
    bm25_index=None,
    use_rrf: bool = False,
    reranker=None,
) -> tuple[dict, list[dict]]:
    log.info("Batch-retrieving %d queries …", len(qa_examples))
    questions = [qa.question for qa in qa_examples]

    initial_k = config.RERANKER_CANDIDATES if use_rerank else config.TOP_K_RETRIEVAL
    if use_rrf and bm25_index is not None:
        all_retrieved = pipeline.retriever.batch_rrf_retrieve(questions, bm25_index, top_k=initial_k)
    elif use_hybrid and bm25_index is not None:
        all_retrieved = pipeline.retriever.batch_hybrid_retrieve(questions, bm25_index, top_k=initial_k)
    elif use_rerank:
        all_retrieved = pipeline.retriever.batch_retrieve(questions, top_k=config.RERANKER_CANDIDATES)
    else:
        all_retrieved = pipeline.retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)

    if use_rerank:
        if reranker is None:
            from retrieval.reranker import Reranker
            reranker = Reranker()
            reranker.load_model()
        all_retrieved = reranker.batch_rerank(questions, all_retrieved, top_k=config.TOP_K_RETRIEVAL)

    log.info("Running generation on %d examples …", len(qa_examples))
    predictions = []
    from tqdm import tqdm
    for qa, retrieved_chunks in tqdm(zip(qa_examples, all_retrieved), total=len(qa_examples), desc="Generating"):
        try:
            context_used, context_chunks = pipeline.assemble_context(retrieved_chunks)
            answer = pipeline.generate(qa.question, context_used)
            retrieved_sources = [c["source"] for c in context_chunks]
            predictions.append({
                "query_id": qa.query_id,
                "predicted": answer,
                "expected": qa.answer,
                "retrieved_sources": retrieved_sources,
                "expected_source": qa.source,
                "retrieved_chunks": [dict(c) for c in context_chunks],
            })
        except Exception as exc:
            log.warning("Generation failed for query %s: %s", qa.query_id, exc)
            predictions.append({
                "query_id": qa.query_id,
                "predicted": "",
                "expected": qa.answer,
                "retrieved_sources": [],
                "expected_source": qa.source,
                "retrieved_chunks": [],
            })
    metrics = compute_all_qa_metrics_with_citation(predictions)
    log.info("QA metrics: %s", metrics)
    return metrics, predictions


def run_hallucination_eval(predictions: list[dict]) -> dict:
    try:
        import torch
        from sentence_transformers import CrossEncoder
        log.info("Loading NLI model …")
        if torch.cuda.is_available():
            _nli_device = "cuda"
        elif torch.backends.mps.is_available():
            _nli_device = "mps"
        else:
            _nli_device = "cpu"
        nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device=_nli_device)
    except Exception as exc:
        log.warning("NLI model unavailable (%s); skipping hallucination analysis", exc)
        return {"summary": {"faithful_rate": None, "skipped": True}, "per_sample": []}

    retrieval_results_dict = {
        p["query_id"]: p.get("retrieved_chunks", [])
        for p in predictions
    }
    result_list = [
        {
            "query_id": p["query_id"],
            "predicted": p["predicted"],
            "retrieved_chunks": p.get("retrieved_chunks", []),
        }
        for p in predictions
    ]
    sample_dict = stratified_sample(result_list, sample_size=config.HALLUCINATION_SAMPLE_SIZE)
    hallucination_result = run_hallucination_analysis(sample_dict, retrieval_results_dict, nli_model)
    log.info("Hallucination analysis: %s", hallucination_result["summary"])
    return hallucination_result


def save_results(results: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info("Results saved → %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Baseline Runner")
    parser.add_argument("--build-index", action="store_true",
                        help="Build FAISS index from corpus (slow; skipped if index exists)")
    parser.add_argument("--eval", action="store_true",
                        help="Run retrieval + generation + hallucination evaluation")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Run only retrieval metrics (no LLM required)")
    parser.add_argument("--hybrid", action="store_true",
                        help="Use BM25+dense hybrid retrieval instead of dense-only")
    parser.add_argument("--rerank", action="store_true",
                        help="Apply cross-encoder reranker after dense retrieval")
    parser.add_argument("--rrf", action="store_true",
                        help="Use RRF (Reciprocal Rank Fusion) of BM25+dense instead of linear blend")
    parser.add_argument("--graph", action="store_true",
                        help="Enable graph neighbor expansion after retrieval")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                        help="Directory to write baseline_metrics.json")
    parser.add_argument(
        "--hmgs",
        action="store_true",
        help="Use HMGS exam questions instead of Kaggle eval set",
    )
    args = parser.parse_args()

    if args.hybrid and args.rrf:
        parser.error("--hybrid and --rrf are mutually exclusive (both are BM25+dense fusion strategies)")

    if args.graph:
        config.GRAPH_EXPANSION_ENABLED = True

    set_seeds(42)

    log.info("Loading data from %s …", config.RAW_DATA_PATH)
    processor = DataProcessor(config.RAW_DATA_PATH)
    summary = processor.load_and_validate()
    log.info("Dataset summary: %s", summary)

    embedder = Embedder()
    embedder.load_model()

    log.info("Building corpus chunks for reuse …")
    corpus_chunks: list[CorpusChunk] = list(processor.build_corpus_chunks())

    index_build_time: float | None = None
    if args.build_index:
        retriever, index_build_time = build_index(processor, embedder, chunks=corpus_chunks)
    else:
        retriever = load_index(embedder)

    graph_index = None
    if args.graph:
        graph_path = config.INDEX_DIR / getattr(config, "GRAPH_FILE", "graph.json")
        if graph_path.exists():
            from retrieval.graph_index import GraphIndex
            graph_index = GraphIndex(graph_path, config.INDEX_DIR / config.METADATA_FILE)
            log.info("Graph index loaded: %s", graph_path)
        else:
            log.warning("--graph set but graph.json not found at %s; run scripts/15_build_graph.py", graph_path)
    retriever.graph_index = graph_index

    if not args.eval and not args.retrieval_only:
        log.info("--eval not specified; exiting after index step.")
        return

    # BM25 index (built once, used for hybrid retrieval)
    bm25_index = None
    if args.hybrid or args.rrf:
        from retrieval.bm25_retriever import BM25Index
        log.info("Building BM25 index over %d chunks …", len(corpus_chunks))
        bm25_index = BM25Index()
        bm25_index.build([{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks])

    if args.hmgs:
        qa_examples = processor.build_gold_eval_set()
        log.info("Using HMGS eval set: %d examples", len(qa_examples))
    else:
        qa_examples = processor.build_qa_eval_set()

    # Determine retrieval mode label
    if args.rrf and args.rerank:
        retrieval_mode = "rrf_rerank"
    elif args.hybrid and args.rerank:
        retrieval_mode = "hybrid_rerank"
    elif args.rrf:
        retrieval_mode = "rrf"
    elif args.hybrid:
        retrieval_mode = "hybrid_bm25_dense"
    elif args.rerank:
        retrieval_mode = "dense_rerank"
    else:
        retrieval_mode = "dense"

    # --- Load reranker once if needed (shared by retrieval + generation eval) ---
    reranker = None
    if args.rerank:
        from retrieval.reranker import Reranker
        log.info("Loading cross-encoder reranker …")
        reranker = Reranker()
        reranker.load_model()

    # --- Retrieval metrics ---
    retrieval_metrics, retrieval_results = run_retrieval_eval(
        retriever, qa_examples, corpus_chunks,
        use_hybrid=args.hybrid, use_rerank=args.rerank, bm25_index=bm25_index,
        use_rrf=args.rrf, reranker=reranker,
    )

    qa_metrics: dict = {}
    hallucination: dict = {}

    if args.eval:
        if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
            log.warning(
                "Ollama not reachable at %s — skipping generation eval. "
                "Start Ollama and run: ollama pull %s",
                config.LLM_BASE_URL, config.LLM_MODEL,
            )
        else:
            pipeline = RAGPipeline(retriever, short_answer_mode=args.hmgs)
            qa_metrics, predictions = run_generation_eval(
                pipeline, qa_examples,
                use_hybrid=args.hybrid, use_rerank=args.rerank, bm25_index=bm25_index,
                use_rrf=args.rrf, reranker=reranker,
            )
            hallucination = run_hallucination_eval(predictions)
    else:
        log.info("Skipping generation/hallucination (--retrieval-only mode).")

    # --- Merge and save ---
    final_results = {
        "hyperparameters": {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
            "llm_model": config.LLM_MODEL,
            "llm_temperature": config.LLM_TEMPERATURE,
            "llm_max_tokens": config.LLM_MAX_TOKENS,
            "hallucination_sample_size": config.HALLUCINATION_SAMPLE_SIZE,
            "retrieval_mode": retrieval_mode,
            "device": embedder.device,
            "index_build_time_s": index_build_time,
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hallucination.get("summary", {}),
        "faithfulness_rate": hallucination.get("summary", {}).get("faithful_rate"),
    }
    save_results(final_results, args.results_dir)


if __name__ == "__main__":
    main()
