#!/usr/bin/env python3
"""Run ablation stages (Ollama) and print comparison table. See --list-stages."""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.retrieval_metrics import compute_all_metrics
from evaluation.semantic_similarity import compute_semantic_similarity
from evaluation.final_score import compute_all_scenario_scores
from evaluation.perplexity import compute_perplexity
from evaluation.ragas_metrics import compute_ragas_metrics
from generation.rag_pipeline import RAGPipeline
from retrieval.bm25_retriever import BM25Index
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.retriever import Retriever
from utils import check_ollama, inject_citations, set_seeds


@dataclass
class StageConfig:
    name: str
    embedding: str
    retrieval: str
    use_rerank: bool
    llm: str
    results_dir: Path
    inject_citations: bool = False
    requires_emb_ft: bool = False
    use_graph: bool = False


STAGE_REGISTRY: dict[str, StageConfig] = {
    "base": StageConfig(
        name="Stage 1 — Base RAG",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE,
    ),
    "hybrid": StageConfig(
        name="Stage 1b — Hybrid BM25+Dense",
        embedding="base",
        retrieval="hybrid",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE / "hybrid",
    ),
    "rrf": StageConfig(
        name="Stage 1c — RRF",
        embedding="base",
        retrieval="rrf",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE / "rrf",
    ),
    "rrf_rerank": StageConfig(
        name="Stage 3 — RRF + Rerank",
        embedding="base",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_RERANK,
    ),
    "llm_ft": StageConfig(
        name="Stage 4 — Fine-tuned LLM",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_LLM_FT,
    ),
    "emb_ft": StageConfig(
        name="Stage 2 — Fine-tuned Embedding",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_EMB_FT,
        requires_emb_ft=True,
    ),
    "full": StageConfig(
        name="Stage 5 — Full Optimized",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_FULL,
        requires_emb_ft=True,
    ),
    "graph": StageConfig(
        name="Stage 6 — RRF + Rerank + Graph",
        embedding="base",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        use_graph=True,
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE / "graph",
    ),
    "graph_full": StageConfig(
        name="Stage 7 — Full + Graph",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="finetuned",
        use_graph=True,
        inject_citations=True,
        results_dir=config.RESULTS_DIR_FULL / "graph",
        requires_emb_ft=True,
    ),
}

DEFAULT_STAGE_ORDER = ["base", "hybrid", "rrf", "rrf_rerank", "llm_ft", "emb_ft", "full", "graph", "graph_full"]


def _retrieve(
    retriever: Retriever,
    questions: list[str],
    stage: StageConfig,
    bm25: Optional[BM25Index],
    reranker: Optional[Reranker],
) -> list[list[dict]]:
    """Dispatch to the correct retrieval method and optionally rerank."""
    initial_k = config.RERANKER_CANDIDATES if stage.use_rerank else config.TOP_K_RETRIEVAL

    t0 = time.time()
    if stage.retrieval == "rrf" and bm25 is not None:
        chunks = retriever.batch_rrf_retrieve(questions, bm25, top_k=initial_k)
    elif stage.retrieval == "hybrid" and bm25 is not None:
        chunks = retriever.batch_hybrid_retrieve(questions, bm25, top_k=initial_k)
    else:
        chunks = retriever.batch_retrieve(questions, top_k=initial_k)

    if stage.use_rerank and reranker is not None:
        chunks = reranker.batch_rerank(questions, chunks, top_k=config.TOP_K_RETRIEVAL)

    print(f"    Retrieval done in {time.time()-t0:.1f}s")
    return chunks


def run_stage(
    stage_key: str,
    stage: StageConfig,
    qa_examples,
    corpus_chunks,
    *,
    embedder_cache: dict,
    retriever_cache: dict,
    bm25_cache: dict,
    reranker_cache: dict,
    relevant_map: dict,
    short_answer_mode: bool,
) -> dict:
    """Run a single stage. Returns the final_results dict (same schema as run_baseline)."""

    print(f"\n{'━'*66}")
    print(f"  {stage.name}")
    print(f"{'━'*66}")

    graph_index = None
    if stage.use_graph and getattr(config, "GRAPH_EXPANSION_ENABLED", False):
        graph_path = config.INDEX_DIR / getattr(config, "GRAPH_FILE", "graph.json")
        meta_path = config.INDEX_DIR / config.METADATA_FILE
        if graph_path.exists():
            from retrieval.graph_index import GraphIndex
            graph_index = GraphIndex(graph_path, meta_path)
            print(f"  Graph index loaded: {graph_path}")
        else:
            print(f"  WARNING: graph.json not found at {graph_path}, skipping graph expansion")

    emb_key = stage.embedding
    if emb_key not in embedder_cache:
        if emb_key == "finetuned":
            model_name = config.FINETUNED_EMBEDDING_MODEL
        else:
            model_name = config.EMBEDDING_MODEL
        print(f"  Loading embedding model: {model_name}")
        emb = Embedder(model_name=model_name) if "model_name" in Embedder.__init__.__code__.co_varnames else Embedder()
        emb.load_model()
        embedder_cache[emb_key] = emb

    embedder: Embedder = embedder_cache[emb_key]

    idx_key = emb_key
    if idx_key not in retriever_cache:
        print(f"  Building FAISS index ({len(corpus_chunks)} chunks) …")
        retriever = Retriever(embedder, graph_index=graph_index)
        texts = [c.text for c in corpus_chunks]
        metadata = [
            {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "source": c.source}
            for c in corpus_chunks
        ]
        t0 = time.time()
        retriever.build_index(texts, metadata)
        print(f"    Index built in {time.time()-t0:.1f}s")
        retriever_cache[idx_key] = retriever

    retriever: Retriever = retriever_cache[idx_key]

    needs_bm25 = stage.retrieval in ("hybrid", "rrf")
    bm25: Optional[BM25Index] = None
    if needs_bm25:
        if "bm25" not in bm25_cache:
            print(f"  Building BM25 index …")
            b = BM25Index()
            b.build([{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks])
            bm25_cache["bm25"] = b
        bm25 = bm25_cache["bm25"]

    reranker: Optional[Reranker] = None
    if stage.use_rerank:
        if "reranker" not in reranker_cache:
            print(f"  Loading reranker: {config.RERANKER_MODEL}")
            r = Reranker()
            r.load_model()
            reranker_cache["reranker"] = r
        reranker = reranker_cache["reranker"]

    llm_model = config.LLM_FINETUNED_MODEL if stage.llm == "finetuned" else config.LLM_MODEL

    print(f"  Retrieval ({stage.retrieval}, rerank={stage.use_rerank}) …")
    questions = [qa.question for qa in qa_examples]

    retrieved_all = _retrieve(retriever, questions, stage, bm25, reranker)

    metric_input = []
    full_retrieved: dict[str, list] = {}
    for qa, chunks in zip(qa_examples, retrieved_all):
        seen: set[str] = set()
        deduped = []
        for c in chunks:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                deduped.append(c["chunk_id"])
        metric_input.append({
            "query_id": qa.query_id,
            "relevant": relevant_map.get(qa.query_id, []),
            "retrieved": deduped,
        })
        full_retrieved[qa.query_id] = chunks

    retrieval_metrics = compute_all_metrics(metric_input)
    print(f"    R@5={retrieval_metrics.get('recall_at_5',0):.4f}  "
          f"R@10={retrieval_metrics.get('recall_at_10',0):.4f}  "
          f"MRR={retrieval_metrics.get('mrr',0):.4f}  "
          f"nDCG@10={retrieval_metrics.get('ndcg_at_10',0):.4f}")

    print(f"  Generation with {llm_model} …")
    max_tokens = (
        config.LLM_FINETUNED_MAX_TOKENS if stage.llm == "finetuned"
        else config.LLM_MAX_TOKENS
    )
    pipeline = RAGPipeline(
        retriever,
        model=llm_model,
        max_tokens=max_tokens,
        short_answer_mode=short_answer_mode,
    )

    predictions = []
    from tqdm import tqdm
    for qa, chunks in tqdm(zip(qa_examples, retrieved_all),
                           total=len(qa_examples), desc=f"  [{stage_key}]"):
        try:
            ctx, ctx_chunks = pipeline.assemble_context(chunks)
            answer = pipeline.generate(qa.question, ctx)
            if stage.inject_citations:
                answer = inject_citations(answer, ctx_chunks)
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

    qa_metrics = compute_all_qa_metrics_with_citation(predictions)
    print(f"    F1={qa_metrics.get('f1',0):.4f}  "
          f"ROUGE-L={qa_metrics.get('rouge_l',0):.4f}  "
          f"Citation={qa_metrics.get('citation_accuracy',0):.4f}")

    print(f"  Perplexity …")
    try:
        perplexity_score = compute_perplexity(
            predictions,
            model=llm_model,
            hf_model_id=config.HF_PERPLEXITY_MODEL,
        )
    except Exception as exc:
        perplexity_score = None
        print(f"    Perplexity=N/A ({exc.__class__.__name__}: {exc})")
    else:
        if perplexity_score is not None:
            print(f"    Perplexity={perplexity_score:.2f}")
        else:
            print(f"    Perplexity=N/A (logprobs not supported)")

    print(f"  RAGAS metrics …")
    ragas_scores = compute_ragas_metrics(predictions, llm_model=llm_model)
    if ragas_scores:
        print(f"    RAGAS faithfulness={ragas_scores.get('ragas_faithfulness','N/A')}  "
              f"correctness={ragas_scores.get('ragas_answer_correctness','N/A')}  "
              f"relevancy={ragas_scores.get('ragas_answer_relevancy','N/A')}  "
              f"ctx_precision={ragas_scores.get('ragas_context_precision','N/A')}  "
              f"ctx_recall={ragas_scores.get('ragas_context_recall','N/A')}")
    else:
        print(f"    RAGAS=N/A (install: pip install ragas langchain-ollama)")

    ragas_faithfulness = (ragas_scores or {}).get("ragas_faithfulness", 0.0) or 0.0
    faithful_rate = ragas_faithfulness

    print(f"  Semantic similarity …")
    sem_sim = 0.0
    try:
        sem_result = compute_semantic_similarity(predictions)
        sem_sim = sem_result["mean_similarity"]
        print(f"    SemanticSim={sem_sim:.4f}")
    except Exception as exc:
        print(f"    WARNING: Semantic similarity failed: {exc}")

    scenario_scores = compute_all_scenario_scores(
        retrieval_metrics=retrieval_metrics,
        qa_metrics=qa_metrics,
        faithfulness_score=faithful_rate,
        semantic_similarity=sem_sim,
        llm_scores=None,
    )
    print(f"    Scenario1={scenario_scores['scenario1']:.4f}  "
          f"Scenario2={scenario_scores['scenario2']:.4f}  "
          f"Scenario3={scenario_scores['scenario3']:.4f}")

    stage.results_dir.mkdir(parents=True, exist_ok=True)

    final = {
        "hyperparameters": {
            "stage": stage_key,
            "stage_name": stage.name,
            "embedding_model": (config.FINETUNED_EMBEDDING_MODEL
                                if stage.embedding == "finetuned"
                                else config.EMBEDDING_MODEL),
            "retrieval_mode": stage.retrieval + ("_rerank" if stage.use_rerank else ""),
            "llm_model": llm_model,
            "inject_citations": stage.inject_citations,
            "use_rerank": stage.use_rerank,
            "use_graph": stage.use_graph,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "faithfulness_rate": faithful_rate,
        "semantic_similarity": sem_sim,
        "scenario1_score": scenario_scores["scenario1"],
        "scenario2_score": scenario_scores["scenario2"],
        "scenario3_score": scenario_scores["scenario3"],
        "perplexity": perplexity_score,
        "ragas_scores": ragas_scores or {},
    }

    out_path = stage.results_dir / "baseline_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    pred_path = stage.results_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  ✓ Results → {out_path}")
    return final


def print_ablation_table(results: dict[str, dict]) -> None:
    """Print a markdown-compatible ablation table to stdout."""

    def _pct(v) -> str:
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "N/A"

    def _f4(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"

    header = (
        f"| {'Stage':<26} | {'R@5':>6} | {'R@10':>6} | {'MRR':>6} | "
        f"{'nDCG@10':>7} | {'F1':>6} | {'ROUGE-L':>7} | {'Citation':>8} | "
        f"{'Faith':>7} | {'SemSim':>7} | {'Scen1':>7} | {'Scen2':>7} | {'Scen3':>7} |"
    )
    sep = "|" + "|".join(["-"*w for w in [28, 8, 8, 8, 9, 8, 9, 10, 9, 9, 9, 9, 9]]) + "|"

    print("\n\n" + "="*120)
    print("  ABLATION TABLE")
    print("="*120)
    print(header)
    print(sep)

    for stage_key in DEFAULT_STAGE_ORDER:
        if stage_key not in results:
            continue
        r = results[stage_key]
        ret = r.get("retrieval_metrics", {})
        qa = r.get("qa_metrics", {})
        stage_name = r.get("hyperparameters", {}).get("stage_name", stage_key)
        print(
            f"| {stage_name:<26} | {_f4(ret.get('recall_at_5')):>6} | "
            f"{_f4(ret.get('recall_at_10')):>6} | {_f4(ret.get('mrr')):>6} | "
            f"{_f4(ret.get('ndcg_at_10')):>7} | {_pct(qa.get('f1')):>6} | "
            f"{_pct(qa.get('rouge_l')):>7} | {_pct(qa.get('citation_accuracy')):>8} | "
            f"{_pct(r.get('faithfulness_rate')):>7} | "
            f"{_f4(r.get('semantic_similarity')):>7} | "
            f"{_f4(r.get('scenario1_score')):>7} | "
            f"{_f4(r.get('scenario2_score')):>7} | "
            f"{_f4(r.get('scenario3_score')):>7} |"
        )
    print("="*120 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all ablation stages and print a comparison table."
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGE_ORDER),
        help=f"Comma-separated stages to run. Default: all. "
             f"Options: {', '.join(DEFAULT_STAGE_ORDER)}",
    )
    parser.add_argument(
        "--dataset", choices=["kaggle", "hmgs"], default="kaggle",
        help="Evaluation dataset (default: kaggle, 300 questions).",
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="Print available stages and exit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit QA examples per stage for quick testing (e.g. --limit 30).",
    )
    args = parser.parse_args()

    if args.list_stages:
        print("\nAvailable stages:")
        for key, cfg in STAGE_REGISTRY.items():
            print(f"  {key:<14} {cfg.name}")
        return

    set_seeds(42)

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid = []
    for key in requested:
        if key not in STAGE_REGISTRY:
            print(f"WARNING: Unknown stage '{key}' — skipping.")
            continue
        stage = STAGE_REGISTRY[key]
        if stage.requires_emb_ft:
            emb_dir = Path(config.FINETUNED_EMBEDDING_MODEL)
            if not emb_dir.exists() or not any(emb_dir.iterdir()):
                print(f"INFO: Stage '{key}' skipped — "
                      f"fine-tuned embedding model not found at {emb_dir}\n"
                      f"  Run: python scripts/12_finetune_embeddings.py first.")
                continue
        if stage.llm == "finetuned":
            import subprocess
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if config.LLM_FINETUNED_MODEL not in result.stdout:
                print(f"INFO: Stage '{key}' skipped — "
                      f"Ollama model '{config.LLM_FINETUNED_MODEL}' not found.\n"
                      f"  Run: python scripts/13_export_lora_to_ollama.py first.")
                continue
        valid.append(key)

    if not valid:
        sys.exit("ERROR: No valid stages to run.")

    print(f"\n🚀  Stages to run: {', '.join(valid)}")
    print(f"   Dataset: {args.dataset}\n")

    if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
        sys.exit(
            f"ERROR: Ollama not reachable at {config.LLM_BASE_URL}.\n"
            f"  Start with: ollama serve\n"
            f"  Pull model: ollama pull {config.LLM_MODEL}"
        )

    short_answer_mode = (args.dataset == "hmgs")
    print("Loading data …")
    processor = DataProcessor(config.RAW_DATA_PATH)
    processor.load_and_validate()
    corpus_chunks = list(processor.build_corpus_chunks())

    if args.dataset == "hmgs":
        qa_examples = DataProcessor.build_gold_eval_set()
    else:
        qa_examples = processor.build_qa_eval_set()
    if args.limit:
        qa_examples = qa_examples[:args.limit]
        print(f"  [--limit {args.limit}] Evaluating first {args.limit} examples only.")

    print(f"  Corpus: {len(corpus_chunks)} chunks  |  QA: {len(qa_examples)} examples")

    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)

    embedder_cache: dict = {}
    retriever_cache: dict = {}
    bm25_cache: dict = {}
    reranker_cache: dict = {}

    all_results: dict[str, dict] = {}

    for key in valid:
        stage = STAGE_REGISTRY[key]
        try:
            result = run_stage(
                key, stage, qa_examples, corpus_chunks,
                embedder_cache=embedder_cache,
                retriever_cache=retriever_cache,
                bm25_cache=bm25_cache,
                reranker_cache=reranker_cache,
                relevant_map=relevant_map,
                short_answer_mode=short_answer_mode,
            )
            all_results[key] = result
        except KeyboardInterrupt:
            print(f"\n  ⚠ Interrupted during stage '{key}'. Saving partial results …")
            break
        except Exception as exc:
            print(f"\n  ERROR in stage '{key}': {exc}")
            import traceback
            traceback.print_exc()
            print("  Continuing with next stage …")

    if all_results:
        print_ablation_table(all_results)

        summary_path = config.BASE_DIR / "results" / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if args.limit:
            all_results["limit_applied"] = True
            all_results["limit_value"] = args.limit
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Full results saved to: {summary_path}")
    else:
        print("No results to report.")


if __name__ == "__main__":
    main()
