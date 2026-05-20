"""Evaluate QA metrics and run hallucination analysis."""
import argparse
import json
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from data.data_processor import DataProcessor
from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
from evaluation.hallucination import stratified_sample, run_hallucination_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA metrics and hallucination")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rrf", "rerank", "hybrid_rerank", "rrf_rerank"],
        default="dense",
        help="Retrieval mode (matches 04_generate_answers output file)",
    )
    parser.add_argument(
        "--dataset",
        choices=["kaggle", "hmgs", "custom"],
        default="kaggle",
        help="Evaluation dataset to use (default: kaggle)",
    )
    parser.add_argument("--qa-file", default=None, dest="qa_file", help="Path to custom benchmark JSONL (for suffix resolution)")
    return parser.parse_args()


def main():
    args = parse_args()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset == "hmgs":
        suffix = "_hmgs"
    elif args.dataset == "custom" or args.qa_file:
        suffix = "_custom"
    else:
        suffix = ""

    # Load predictions
    predictions_path = config.RESULTS_DIR / f"qa_predictions_{args.mode}{suffix}.jsonl"
    print(f"Loading predictions from {predictions_path}")
    predictions = DataProcessor.load_jsonl(predictions_path)

    # Separate errors
    errors = [p for p in predictions if "error" in p]
    valid = [p for p in predictions if "error" not in p]
    if errors:
        print(f"WARNING: {len(errors)} predictions had errors and were excluded from metrics")
    print(f"Valid predictions: {len(valid)}")

    # QA metrics
    print("\nComputing QA metrics...")
    qa_input = [
        {
            "predicted": p["predicted"],
            "expected": p["expected"],
            "retrieved_sources": p.get("retrieved_sources", []),
            "expected_source": p.get("expected_source", ""),
            "retrieved_chunks": p.get("retrieved_chunks", []),
        }
        for p in valid
    ]
    qa_metrics = compute_all_qa_metrics_with_citation(qa_input)
    qa_metrics["error_count"] = len(errors)

    print(f"\n=== QA Metrics ===")
    for k, v in qa_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save QA results
    qa_results_path = config.RESULTS_DIR / f"qa_results_{args.mode}{suffix}.json"
    with open(qa_results_path, "w", encoding="utf-8") as f:
        json.dump(qa_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {qa_results_path}")

    # Hallucination analysis
    print("\nRunning hallucination analysis...")

    # Use retrieved_chunks from predictions — these ARE the context chunks passed to the LLM
    retrieved_results = {p["query_id"]: p.get("retrieved_chunks", []) for p in valid}

    # Load NLI model
    print("Loading NLI model: cross-encoder/nli-deberta-v3-small (~180 MB, first run downloads)")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")

    # Stratified sample + hallucination analysis
    sample = stratified_sample(valid, config.HALLUCINATION_SAMPLE_SIZE)
    print(f"Stratified sample: hits={len(sample['hits'])}, partial={len(sample['partial'])}, misses={len(sample['misses'])}")

    hall_results = run_hallucination_analysis(sample, retrieved_results, nli_model)

    summary = hall_results["summary"]
    print(f"\n=== Hallucination Analysis ===")
    print(f"  Total analyzed: {summary['total']}")
    print(f"  Faithful: {summary['faithful_count']} ({summary['faithful_rate']:.2%})")
    print(f"  By category: {summary['by_category']}")

    # Save
    hall_path = config.RESULTS_DIR / f"hallucination_results_{args.mode}{suffix}.json"
    with open(hall_path, "w", encoding="utf-8") as f:
        json.dump(hall_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {hall_path}")

    print("\n✓ Evaluation complete")

if __name__ == '__main__':
    main()
