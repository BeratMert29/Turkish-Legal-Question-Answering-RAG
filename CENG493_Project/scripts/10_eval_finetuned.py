"""
Stage 1 Fine-tuned LLM Evaluation
Runs QA evaluation using the fine-tuned LoRA model (Transformers/PEFT, not Ollama)
and saves results to results/stage1_ft/baseline_metrics.json.

Usage:
    python scripts/10_eval_finetuned.py
"""

import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

# TRL reads Jinja templates without explicit encoding; on Windows with Turkish locale
# (cp1254) this crashes. Force UTF-8 before any trl/transformers import.
os.environ.setdefault("PYTHONUTF8", "1")

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
from data.data_processor import DataProcessor
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


def main() -> None:
    set_seeds(42)

    # ------------------------------------------------------------------
    # 1. Embedder + index
    # ------------------------------------------------------------------
    log.info("Loading embedder …")
    embedder = Embedder()
    embedder.load_model()

    retriever = load_index(embedder)

    # ------------------------------------------------------------------
    # 2. QA eval set
    # ------------------------------------------------------------------
    log.info("Loading data from %s …", config.RAW_DATA_PATH)
    processor = DataProcessor(config.RAW_DATA_PATH)
    summary = processor.load_and_validate()
    log.info("Dataset summary: %s", summary)

    qa_examples = processor.build_qa_eval_set()
    log.info("QA eval set: %d examples", len(qa_examples))

    corpus_chunks = list(processor.build_corpus_chunks())
    log.info("Corpus: %d chunks", len(corpus_chunks))

    # ------------------------------------------------------------------
    # 3. Fine-tuned pipeline
    # ------------------------------------------------------------------
    log.info("Instantiating FinetunedRAGPipeline …")
    pipeline = FinetunedRAGPipeline(retriever=retriever)

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
        judge_result   = llm_judge_answer(judge_preds, config.LLM_BASE_URL, config.LLM_MODEL, sample_size=20)
        faith_result   = llm_judge_faithfulness(predictions, config.LLM_BASE_URL, config.LLM_MODEL, sample_size=20)
        relev_result   = llm_judge_relevancy(judge_preds, config.LLM_BASE_URL, config.LLM_MODEL, sample_size=20)
        coher_result   = llm_judge_coherence(predictions, config.LLM_BASE_URL, config.LLM_MODEL, sample_size=20)
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
