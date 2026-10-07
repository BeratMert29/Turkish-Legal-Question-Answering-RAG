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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Stage 1 Baseline Runner")
    parser.add_argument(
        "--build-index", action="store_true",
        help="Build FAISS index from corpus (slow; skipped if index exists)",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help="Run retrieval + generation + hallucination evaluation",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Run only retrieval metrics (no LLM required)",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use BM25+dense hybrid retrieval instead of dense-only",
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply cross-encoder reranker after dense retrieval",
    )
    parser.add_argument(
        "--rrf", action="store_true",
        help=(
            "Use RRF (Reciprocal Rank Fusion) of BM25+dense "
            "instead of linear blend"
        ),
    )
    parser.add_argument(
        "--graph", action="store_true",
        help="Enable graph neighbor expansion after retrieval",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=config.RESULTS_DIR,
        help="Directory to write baseline_metrics.json",
    )
    parser.add_argument(
        "--eval-set", dest="eval_set",
        choices=config.EVAL_SET_CHOICES, default=config.DEFAULT_EVAL_SET,
        help="Evaluation set (default: %(default)s)",
    )
    parser.add_argument(
        "--hmgs", action="store_true",
        help="Alias for --eval-set hmgs",
    )
    parser.add_argument(
        "--corpus", type=Path, default=None, metavar="PATH",
        help=(
            "Path to external corpus.jsonl (evaluator format). "
            "When combined with --eval-data, DataProcessor is bypassed entirely."
        ),
    )
    parser.add_argument(
        "--eval-data", type=Path, default=None, metavar="PATH",
        help=(
            "Path to external rag_eval.json or gold_benchmark.json. "
            "Auto-detects format. Mutually exclusive with --hmgs."
        ),
    )
    parser.add_argument(
        "--docs-path", type=str, default=None, dest="docs_path",
        help=(
            "Directory of .txt/.pdf documents to chunk and index. "
            "Mutually exclusive with --corpus."
        ),
    )
    return parser.parse_args(argv)


def build_index(processor, embedder, chunks=None):
    from retrieval.retriever import Retriever

    if chunks is None:
        log.info("Building corpus chunks …")
        chunks = list(processor.build_corpus_chunks())
    log.info("  %d chunks total", len(chunks))

    texts = [c.text for c in chunks]
    metadata = [
        {"chunk_id": c.chunk_id, "doc_id": c.doc_id,
         "text": c.text, "source": c.source}
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


def load_index(embedder):
    from retrieval.retriever import Retriever

    index_path = config.INDEX_DIR / config.INDEX_FILE
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    log.info("Loading index from %s …", index_path)
    retriever = Retriever(
        embedder, index_path=index_path, metadata_path=meta_path,
    )
    log.info("  %d vectors loaded", retriever.index.ntotal)
    return retriever


def save_results(results: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "baseline_metrics.json"
    _tmp = out_path.with_suffix(".tmp")
    with _tmp.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(_tmp, out_path)
    log.info("Results saved → %s", out_path)


def main() -> None:
    args = _parse_args()

    if args.hybrid and args.rrf:
        sys.exit(
            "ERROR: --hybrid and --rrf are mutually exclusive "
            "(both are BM25+dense fusion strategies)"
        )

    if args.hmgs and args.eval_data:
        sys.exit("ERROR: --hmgs and --eval-data are mutually exclusive")

    if args.corpus and args.docs_path:
        sys.exit("ERROR: --corpus and --docs-path are mutually exclusive")

    from data.data_processor import DataProcessor
    from pipeline.data_loading import load_external_corpus, load_external_qa
    from pipeline.retrieval import retrieve
    from utils import set_seeds, check_ollama

    set_seeds(config.SEED)

    # -- Load corpus and QA data -------------------------------------------
    short_answer_mode: bool = False
    processor = None

    if args.corpus:
        log.info("Loading external corpus from %s …", args.corpus)
        corpus_chunks = load_external_corpus(args.corpus)
        log.info("  %d corpus chunks loaded", len(corpus_chunks))
    elif args.docs_path:
        from data.corpus_loader import resolve_corpus
        corpus_path = resolve_corpus(None, args.docs_path)
        log.info("Loading chunked corpus from %s …", corpus_path)
        corpus_chunks = load_external_corpus(corpus_path)
        log.info("  %d corpus chunks loaded", len(corpus_chunks))
    else:
        log.info("Loading data from %s …", config.RAW_DATA_PATH)
        processor = DataProcessor(config.RAW_DATA_PATH)
        summary = processor.load_and_validate()
        log.info("Dataset summary: %s", summary)
        log.info("Building corpus chunks for reuse …")
        corpus_chunks = list(processor.build_corpus_chunks())

    from retrieval.embedder import Embedder

    embedder = Embedder()
    embedder.load_model()

    index_build_time = None
    if args.build_index:
        retriever, index_build_time = build_index(
            processor, embedder, chunks=corpus_chunks,
        )
    else:
        retriever = load_index(embedder)
        # Gold labels, BM25 and the graph are built from corpus_chunks; an
        # index built from another corpus (another chunker or corpus file)
        # silently makes gold chunks unretrievable.
        # Chunk ids are positional, so compare ids AND texts: a re-chunked
        # corpus can reuse every id for different text.
        from utils import corpus_fingerprint
        if corpus_fingerprint(retriever.metadata) != corpus_fingerprint(corpus_chunks):
            sys.exit(
                f"ERROR: the saved index ({len(retriever.metadata)} chunks, "
                f"{config.INDEX_DIR}) does not match this run's corpus "
                f"({len(corpus_chunks)} chunks).\n  Re-run with --build-index."
            )

    graph_index = None
    if args.graph:
        graph_path = config.INDEX_DIR / config.GRAPH_FILE
        if graph_path.exists():
            # Validate JSON; rebuild automatically if the file is corrupt.
            try:
                with graph_path.open(encoding="utf-8") as _gf:
                    json.load(_gf)
            except (json.JSONDecodeError, OSError):
                log.warning(
                    "graph.json corrupt at %s; attempting rebuild …",
                    graph_path,
                )
                from pipeline.retrieval import auto_build_graph
                auto_build_graph(graph_path)
            if graph_path.exists():
                from retrieval.graph_index import GraphIndex
                graph_index = GraphIndex(
                    graph_path, config.INDEX_DIR / config.METADATA_FILE,
                )
                log.info("Graph index loaded: %s", graph_path)
        else:
            log.warning(
                "--graph set but graph.json not found at %s; "
                "run scripts/15_build_graph.py",
                graph_path,
            )
    retriever.graph_index = graph_index

    if not args.eval and not args.retrieval_only:
        log.info("--eval not specified; exiting after index step.")
        return

    # -- BM25 index --------------------------------------------------------
    bm25_index = None
    if args.hybrid or args.rrf:
        from retrieval.bm25_retriever import BM25Index

        log.info("Building BM25 index over %d chunks …", len(corpus_chunks))
        bm25_index = BM25Index()
        bm25_index.build(
            [{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks],
        )

    # -- Load QA data ------------------------------------------------------
    if args.eval_data:
        log.info("Loading external QA data from %s …", args.eval_data)
        qa_examples, short_answer_mode = load_external_qa(args.eval_data)
        log.info(
            "  %d QA examples loaded (short_answer_mode=%s)",
            len(qa_examples), short_answer_mode,
        )
    elif args.hmgs or args.eval_set == "hmgs":
        if processor is None:
            processor = DataProcessor(config.RAW_DATA_PATH)
            processor.load_and_validate()
        qa_examples = processor.build_gold_eval_set()
        short_answer_mode = True
        log.info("Using HMGS eval set: %d examples", len(qa_examples))
    elif args.eval_set == "turkish_legal_rag":
        qa_examples = DataProcessor.build_turkish_legal_rag_eval_set()
        short_answer_mode = False
        log.info("Using turkish_legal_rag eval set: %d examples", len(qa_examples))
    else:
        if processor is None:
            processor = DataProcessor(config.RAW_DATA_PATH)
            processor.load_and_validate()
        qa_examples = processor.build_qa_eval_set()

    # -- Retrieval mode label ----------------------------------------------
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

    # -- Reranker ----------------------------------------------------------
    reranker = None
    if args.rerank:
        from retrieval.reranker import Reranker

        log.info("Loading cross-encoder reranker …")
        reranker = Reranker()
        reranker.load_model()

    # -- Retrieval metrics -------------------------------------------------
    log.info("Building ground-truth relevance map …")
    relevant_map = DataProcessor.build_relevant_chunk_map(
        corpus_chunks, qa_examples,
    )

    questions = [qa.question for qa in qa_examples]

    log.info("Running retrieval on %d queries …", len(qa_examples))
    t0 = time.time()

    retrieval_mode_key = (
        "rrf" if args.rrf
        else "hybrid" if args.hybrid
        else "dense"
    )
    all_retrieved = retrieve(
        retriever, questions,
        retrieval_mode=retrieval_mode_key,
        bm25=bm25_index, reranker=reranker,
        use_rerank=args.rerank,
        graph_index=graph_index, use_graph=args.graph,
    )

    retrieval_time = round(time.time() - t0, 2)

    from evaluation.retrieval_metrics import compute_all_metrics
    from pipeline.evaluation import prepare_metric_input

    # Same metric input as scripts/14: pre-expansion ranking (graph neighbours
    # excluded), deduplicated chunk ids.
    results_list, _ = prepare_metric_input(qa_examples, all_retrieved, relevant_map)
    retrieval_metrics = compute_all_metrics(results_list)
    retrieval_metrics["retrieval_time_s"] = retrieval_time
    log.info("Retrieval metrics: %s", retrieval_metrics)

    # -- Generation --------------------------------------------------------
    qa_metrics: dict = {}
    hallucination: dict = {}

    if args.eval:
        if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
            log.warning(
                "Ollama not reachable at %s -- skipping generation eval. "
                "Start Ollama and run: ollama pull %s",
                config.LLM_BASE_URL, config.LLM_MODEL,
            )
        else:
            from generation.rag_pipeline import RAGPipeline
            from pipeline.evaluation import run_generation_loop

            pipeline = RAGPipeline(
                retriever, short_answer_mode=short_answer_mode,
            )
            predictions = run_generation_loop(
                pipeline, qa_examples, all_retrieved,
                stage_key="baseline",
            )

            from evaluation.qa_metrics import compute_all_qa_metrics_with_citation

            qa_metrics = compute_all_qa_metrics_with_citation(predictions)
            log.info("QA metrics: %s", qa_metrics)

            # Hallucination
            try:
                from pipeline.evaluation import run_hallucination_eval

                retrieval_results_dict = {
                    p["query_id"]: p.get("retrieved_chunks", [])
                    for p in predictions
                }
                hallucination, _faithful_rate, _nli_model = run_hallucination_eval(
                    predictions, retrieval_results_dict,
                    config.HALLUCINATION_SAMPLE_SIZE,
                    llm_model=config.LLM_MODEL,
                )
                log.info(
                    "Hallucination analysis: %s", hallucination["summary"],
                )
            except Exception as exc:
                log.warning(
                    "NLI model unavailable (%s); "
                    "skipping hallucination analysis",
                    exc,
                )
                hallucination = {
                    "summary": {
                        "context_grounding_rate": None, "skipped": True,
                    },
                    "per_sample": [],
                }
    else:
        log.info("Skipping generation/hallucination (--retrieval-only mode).")

    # -- Merge and save ----------------------------------------------------
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
            "dataset": (
                str(args.eval_data) if args.eval_data
                else str(config.RAW_DATA_PATH)
            ),
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hallucination.get("summary", {}),
        "faithfulness_rate": hallucination.get("summary", {}).get(
            "context_grounding_rate",
        ),
    }
    save_results(final_results, args.results_dir)


if __name__ == "__main__":
    main()
