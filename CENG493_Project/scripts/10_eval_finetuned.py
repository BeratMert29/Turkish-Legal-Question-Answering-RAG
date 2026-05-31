"""
Stage 1 Fine-tuned LLM Evaluation
Runs QA evaluation using the fine-tuned LoRA model (Transformers/PEFT, not Ollama)
and saves results to results/stage1_ft/baseline_metrics.json.

Usage:
    python scripts/10_eval_finetuned.py
    python scripts/10_eval_finetuned.py --corpus path/to/corpus.jsonl --eval-data path/to/rag_eval.json
"""

import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# TRL reads Jinja templates without explicit encoding; on Windows with Turkish locale
# (cp1254) this crashes. Force UTF-8 before any trl/transformers import.
os.environ.setdefault("PYTHONUTF8", "1")

import argparse
import importlib.util
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor, CorpusChunk, QAExample
from retrieval.embedder import Embedder
from run_baseline import load_index, run_generation_eval, run_hallucination_eval, run_retrieval_eval, save_results
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.llm_judge import (
    llm_judge_answer,
    llm_judge_faithfulness,
    llm_judge_relevancy,
    llm_judge_coherence,
)
from evaluation.semantic_similarity import compute_semantic_similarity
from evaluation.final_score import compute_all_scenario_scores
from utils import inject_citations
from utils import set_seeds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import FinetunedRAGPipeline from scripts/09_load_finetuned_model.py
# (filename starts with a digit so standard import is not possible)
# ---------------------------------------------------------------------------
_script_09_path = _PROJECT_ROOT / "scripts" / "09_load_finetuned_model.py"
_spec = importlib.util.spec_from_file_location("_load_finetuned_model", _script_09_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FinetunedRAGPipeline = _mod.FinetunedRAGPipeline

RESULTS_DIR = config.BASE_DIR / "results" / "stage1_ft"
RETRIEVAL_MODE = "dense_finetuned_llm"


# ─────────────────────────────────────────────────────────────────────────────
# External data loaders (same as run_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_external_corpus(path: Path) -> list[CorpusChunk]:
    """Load corpus from evaluator-format corpus.jsonl -> list[CorpusChunk]."""
    from data.corpus_loader import load_corpus_jsonl
    raw_chunks = load_corpus_jsonl(path)
    return [
        CorpusChunk(**{k: r[k] for k in ("chunk_id", "doc_id", "text", "source", "char_len")})
        for r in raw_chunks
    ]


def _load_external_qa(path: Path) -> tuple[list[QAExample], bool]:
    """Load QA examples from rag_eval.json or gold_benchmark.json.

    Auto-detects format from first item keys.
    Attaches gold_source_ids as a dynamic attribute so build_relevant_chunk_map
    can use exact chunk ID matching when available.

    Returns:
        A tuple of (qa_examples, short_answer_mode).
    """
    import json as _json
    with open(path, encoding="utf-8") as f:
        data = _json.load(f)

    if not data:
        raise ValueError(f"Empty QA file: {path}")

    if isinstance(data, dict):
        data = list(data.values())

    first = data[0]
    examples = []

    if "query_id" in first and "query" in first:
        # rag_eval.json format — open-ended answers, no short-answer mode
        short_answer_mode = False
        for item in data:
            qa = QAExample(
                query_id=item["query_id"],
                question=item["query"],
                answer=item.get("gold_answer_extract", ""),
                context="",
                source=item.get("source", ""),
                data_type="external",
            )
            qa.gold_source_ids = item.get("gold_chunk_ids", [])
            examples.append(qa)

    elif "question_id" in first and "question" in first:
        # gold_benchmark.json format — exam-style, short answers
        short_answer_mode = True
        for item in data:
            gold_sources = item.get("gold_sources", [])
            qa = QAExample(
                query_id=item["question_id"],
                question=item["question"],
                answer=item.get("verified_answer", ""),
                context="",
                source=gold_sources[0].get("source", "") if gold_sources else "",
                data_type="external",
            )
            qa.gold_source_ids = [s["source_id"] for s in gold_sources]
            examples.append(qa)

    else:
        raise ValueError(
            f"Unrecognised QA file format in {path}. "
            "Expected rag_eval.json (query_id+query) or gold_benchmark.json (question_id+question)."
        )

    return examples, short_answer_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Fine-tuned LLM Evaluation")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to external corpus.jsonl (evaluator format). "
             "When combined with --eval-data, DataProcessor is bypassed entirely.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to external rag_eval.json or gold_benchmark.json. "
             "Auto-detects format.",
    )
    args = parser.parse_args()

    set_seeds(42)

    # ------------------------------------------------------------------
    # 1. Embedder + index
    # ------------------------------------------------------------------
    log.info("Loading embedder …")
    embedder = Embedder()
    embedder.load_model()

    retriever = load_index(embedder)

    # ------------------------------------------------------------------
    # 2. Corpus chunks + QA eval set
    # ------------------------------------------------------------------
    short_answer_mode: bool = False

    if args.corpus and args.eval_data:
        log.info("Loading external corpus from %s …", args.corpus)
        corpus_chunks: list[CorpusChunk] = _load_external_corpus(args.corpus)
        log.info("Corpus: %d chunks", len(corpus_chunks))

        log.info("Loading external QA data from %s …", args.eval_data)
        qa_examples, short_answer_mode = _load_external_qa(args.eval_data)
        log.info("QA eval set: %d examples (short_answer_mode=%s)", len(qa_examples), short_answer_mode)

        dataset_label = str(args.eval_data)
    else:
        log.info("Loading data from %s …", config.RAW_DATA_PATH)
        processor = DataProcessor(config.RAW_DATA_PATH)
        summary = processor.load_and_validate()
        log.info("Dataset summary: %s", summary)

        qa_examples = processor.build_qa_eval_set()
        log.info("QA eval set: %d examples", len(qa_examples))

        corpus_chunks = list(processor.build_corpus_chunks())
        log.info("Corpus: %d chunks", len(corpus_chunks))

        dataset_label = str(config.RAW_DATA_PATH)

    # ------------------------------------------------------------------
    # 3. Fine-tuned pipeline
    # ------------------------------------------------------------------
    log.info("Instantiating FinetunedRAGPipeline …")
    pipeline = FinetunedRAGPipeline(retriever=retriever, short_answer_mode=short_answer_mode)

    log.info("Loading fine-tuned model (base Qwen2.5-7B + LoRA adapter in 4-bit) …")
    pipeline.load_model()
    log.info("Model loaded.")

    # ------------------------------------------------------------------
    # 4. Retrieval eval (dense, no rerank — same index used for generation)
    # ------------------------------------------------------------------
    log.info("Running retrieval evaluation …")
    retrieval_metrics, _ = run_retrieval_eval(retriever, qa_examples, corpus_chunks)

    # ------------------------------------------------------------------
    # 5. Generation eval
    # ------------------------------------------------------------------
    qa_metrics, predictions = run_generation_eval(pipeline, qa_examples)

    # Citation injection — append [Kaynak N] markers via token-overlap
    log.info("Injecting citations …")
    for pred in predictions:
        if pred.get("predicted"):
            pred["predicted"] = inject_citations(
                pred["predicted"], pred.get("retrieved_chunks", [])
            )
    qa_metrics = compute_all_qa_metrics_with_citation(predictions)

    # ------------------------------------------------------------------
    # 6. Hallucination eval
    # ------------------------------------------------------------------
    hallucination = run_hallucination_eval(predictions)

    # ------------------------------------------------------------------
    # 7. LLM Judge + Semantic Similarity + Final Scenario Scores
    # ------------------------------------------------------------------
    _afr = hallucination.get("summary", {}).get("answer_faithfulness_rate")
    faithful_rate = _afr if _afr is not None else hallucination.get("summary", {}).get("context_grounding_rate", 0.0)

    log.info("Running LLM Judge (sample=20) …")
    llm_judge_score = None
    llm_faithfulness_score = None
    llm_relevancy_score = None
    llm_coherence_score = None
    try:
        judge_preds = [
            {**p, "question": next(
                (qa.question for qa in qa_examples if qa.query_id == p["query_id"]),
                p.get("query_id", "")
            )}
            for p in predictions
        ]
        judge_result   = llm_judge_answer(judge_preds, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, sample_size=20)
        faith_result   = llm_judge_faithfulness(predictions, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, sample_size=20)
        relev_result   = llm_judge_relevancy(judge_preds, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, sample_size=20)
        coher_result   = llm_judge_coherence(predictions, config.LLM_BASE_URL, config.LLM_JUDGE_MODEL, sample_size=20)
        llm_judge_score        = judge_result["score"]
        llm_faithfulness_score = faith_result["score"]
        llm_relevancy_score    = relev_result["score"]
        llm_coherence_score    = coher_result["score"]
        log.info("LLM Judge: answer=%.4f faith=%.4f relev=%.4f coher=%.4f",
                 llm_judge_score, llm_faithfulness_score,
                 llm_relevancy_score, llm_coherence_score)
    except Exception as exc:
        log.warning("LLM Judge failed: %s", exc)

    log.info("Computing semantic similarity …")
    sem_sim = 0.0
    try:
        sem_result = compute_semantic_similarity(predictions)
        sem_sim = sem_result["mean_similarity"]
        log.info("Semantic similarity: %.4f", sem_sim)
    except Exception as exc:
        log.warning("Semantic similarity failed: %s", exc)

    llm_scores_dict = {}
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
    log.info("Scenario1=%.4f  Scenario2=%.4f  Scenario3=%.4f",
             scenario_scores["scenario1"], scenario_scores["scenario2"],
             scenario_scores["scenario3"])

    # ------------------------------------------------------------------
    # 8. Assemble final_results (same structure as run_baseline.py main())
    # ------------------------------------------------------------------
    final_results = {
        "hyperparameters": {
            "embedding_model": config.EMBEDDING_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k_retrieval": config.TOP_K_RETRIEVAL,
            "top_k_for_generation": config.TOP_K_FOR_GENERATION,
            "llm_model": _mod.HF_MODEL_ID,
            "llm_temperature": 0.0,
            "llm_max_tokens": config.LLM_MAX_TOKENS,
            "hallucination_sample_size": config.HALLUCINATION_SAMPLE_SIZE,
            "retrieval_mode": RETRIEVAL_MODE,
            "device": embedder.device,
            "index_build_time_s": None,
            "dataset": dataset_label,
        },
        "retrieval_metrics": retrieval_metrics,
        "qa_metrics": qa_metrics,
        "hallucination_summary": hallucination.get("summary", {}),
        "faithfulness_rate": faithful_rate,
        "llm_judge_score": llm_judge_score,
        "llm_faithfulness_score": llm_faithfulness_score,
        "llm_relevancy_score": llm_relevancy_score,
        "llm_coherence_score": llm_coherence_score,
        "semantic_similarity": sem_sim,
        "scenario1_score": scenario_scores["scenario1"],
        "scenario2_score": scenario_scores["scenario2"],
        "scenario3_score": scenario_scores["scenario3"],
    }

    # ------------------------------------------------------------------
    # 9. Save results
    # ------------------------------------------------------------------
    save_results(final_results, RESULTS_DIR)
    log.info("Done. Results written to %s/baseline_metrics.json", RESULTS_DIR)


if __name__ == "__main__":
    main()
