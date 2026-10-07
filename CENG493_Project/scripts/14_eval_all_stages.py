#!/usr/bin/env python3
"""
14_eval_all_stages.py -- Full Ablation Evaluation (All Stages)

Runs every pipeline configuration through Ollama (no Transformers inference)
and prints a comparative ablation table at the end.

Prerequisites:
  - Ollama running: ollama serve
  - Base model pulled: ollama pull qwen2.5:7b
  - Fine-tuned LLM (optional): python scripts/13_export_lora_to_ollama.py
  - Fine-tuned embedding (optional): python scripts/12_finetune_embeddings.py

Usage:
    python scripts/14_eval_all_stages.py                           # all available stages (CSV format)
    python scripts/14_eval_all_stages.py --stages base,rrf_rerank,llm_ft
    python scripts/14_eval_all_stages.py --stages base --dataset hmgs
    python scripts/14_eval_all_stages.py --list-stages
    python scripts/14_eval_all_stages.py \
        --corpus /content/datasets/corpus.jsonl \
        --eval-data /content/datasets/gold_benchmark.json          # external evaluator format
    python scripts/14_eval_all_stages.py \
        --corpus /content/datasets/corpus.jsonl \
        --eval-data /content/datasets/rag_eval.json                # external rag eval

Available stages:
    base         BGE-M3 base    + dense          + qwen2.5:7b
    hybrid       BGE-M3 base    + hybrid BM25    + qwen2.5:7b
    rrf          BGE-M3 base    + RRF            + qwen2.5:7b
    rrf_rerank   BGE-M3 base    + RRF+rerank     + qwen2.5:7b   <- best retrieval
    graph        BGE-M3 base    + RRF+rerank+graph + qwen2.5:7b  <- requires graph.json
    llm_ft       BGE-M3 base    + dense          + qwen25-legal-ft (fine-tuned)
    emb_ft       BGE-M3 ft*     + RRF+rerank     + qwen2.5:7b   <- requires emb training
    full         BGE-M3 ft*     + RRF+rerank     + qwen25-legal-ft  <- best overall
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from pipeline.stages import STAGE_REGISTRY, DEFAULT_STAGE_ORDER
from pipeline.retrieval import auto_build_graph


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run all ablation stages and print a comparison table.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGE_ORDER),
        help=(
            f"Comma-separated stages to run. Default: all. "
            f"Options: {', '.join(DEFAULT_STAGE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--eval-set", "--dataset",
        dest="eval_set",
        choices=["kaggle", "hmgs"],
        default="hmgs",
        help=(
            "Evaluation dataset for ALL stages (default: hmgs, ~161 questions). "
            "Use 'kaggle' for the Kaggle-split eval set (~300 questions). "
            "All stages in a single run MUST use the same eval set so that "
            "ablation comparisons are valid.  Previously, different invocations "
            "used different datasets (base/hybrid/rrf/rrf_rerank with hmgs n=161 "
            "and llm_ft/emb_ft/full with kaggle n=300), making cross-stage "
            "comparisons invalid.  This flag enforces a single dataset per run."
        ),
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
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help=(
            "Path to external corpus.jsonl (evaluator format). "
            "Overrides DataProcessor corpus loading."
        ),
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help=(
            "Path to external rag_eval.json or gold_benchmark.json. "
            "Auto-detects format. Overrides --dataset."
        ),
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default=None,
        dest="docs_path",
        help=(
            "Directory of .txt/.pdf documents to chunk and index. "
            "Mutually exclusive with --corpus."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    if args.corpus and args.docs_path:
        sys.exit("ERROR: --corpus and --docs-path are mutually exclusive")

    if args.list_stages:
        print("\nAvailable stages:")
        for key, cfg in STAGE_REGISTRY.items():
            print(f"  {key:<14} {cfg.name}")
        return

    from data.data_processor import DataProcessor
    from pipeline.data_loading import load_external_corpus, load_external_qa
    from pipeline.evaluation import run_stage, print_ablation_table
    from utils import check_ollama, set_seeds

    set_seeds(42)

    # -- Validate requested stages -----------------------------------------
    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    valid = []
    for key in requested:
        if key not in STAGE_REGISTRY:
            print(f"WARNING: Unknown stage '{key}' -- skipping.")
            continue
        stage = STAGE_REGISTRY[key]
        if stage.requires_graph:
            graph_path = config.INDEX_DIR / config.GRAPH_FILE
            if not graph_path.exists():
                auto_build_graph(graph_path)
            if not graph_path.exists():
                print(
                    f"INFO: Stage '{key}' skipped -- "
                    f"graph.json not found at {graph_path}\n"
                    f"  Run: python scripts/15_build_graph.py first."
                )
                continue
        if stage.requires_emb_ft:
            emb_dir = Path(config.FINETUNED_EMBEDDING_MODEL)
            if not emb_dir.exists() or not any(emb_dir.iterdir()):
                print(
                    f"INFO: Stage '{key}' skipped -- "
                    f"fine-tuned embedding model not found at {emb_dir}\n"
                    f"  Run: python scripts/12_finetune_embeddings.py first."
                )
                continue
        if stage.llm == "finetuned":
            import subprocess
            try:
                _ollama_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except FileNotFoundError:
                print(
                    f"INFO: Stage '{key}' skipped -- "
                    f"'ollama' executable not found in PATH."
                )
                continue
            except subprocess.TimeoutExpired:
                print(
                    f"INFO: Stage '{key}' skipped -- "
                    f"'ollama list' timed out."
                )
                continue
            # Exact name match: compare first column, allowing ':latest' suffix.
            _target = config.LLM_FINETUNED_MODEL
            _found = any(
                (parts := line.split()) and (
                    parts[0] == _target
                    or parts[0] == _target + ":latest"
                )
                for line in _ollama_result.stdout.splitlines()
                if line.strip()
            )
            if not _found:
                print(
                    f"INFO: Stage '{key}' skipped -- "
                    f"Ollama model '{config.LLM_FINETUNED_MODEL}' not found.\n"
                    f"  Run: python scripts/13_export_lora_to_ollama.py first."
                )
                continue
        valid.append(key)

    if not valid:
        sys.exit("ERROR: No valid stages to run.")

    print(f"\n\U0001f680  Stages to run: {', '.join(valid)}")
    print(f"   Eval set: {args.eval_set}\n")

    # -- Check Ollama ------------------------------------------------------
    if not check_ollama(config.LLM_BASE_URL, config.LLM_MODEL):
        sys.exit(
            f"ERROR: Ollama not reachable at {config.LLM_BASE_URL}.\n"
            f"  Start with: ollama serve\n"
            f"  Pull model: ollama pull {config.LLM_MODEL}"
        )

    # -- Load data ---------------------------------------------------------
    print("Loading data …")

    if args.corpus:
        print(f"  Corpus source : {args.corpus} (external evaluator format)")
        corpus_chunks = load_external_corpus(Path(args.corpus))
    elif args.docs_path:
        from data.corpus_loader import resolve_corpus
        corpus_path = resolve_corpus(None, args.docs_path)
        print(f"  Chunking/loading corpus from {args.docs_path} …")
        corpus_chunks = load_external_corpus(Path(corpus_path))
    else:
        processor = DataProcessor(config.RAW_DATA_PATH)
        processor.load_and_validate()
        corpus_chunks = list(processor.build_corpus_chunks())

    if args.eval_data:
        print(f"  QA source     : {args.eval_data} (external evaluator format)")
        qa_examples, short_answer_mode = load_external_qa(Path(args.eval_data))
    else:
        short_answer_mode = args.eval_set == "hmgs"
        if args.eval_set == "hmgs":
            qa_examples = DataProcessor.build_gold_eval_set()
        else:
            if not args.corpus:
                pass  # processor already initialised above
            else:
                processor = DataProcessor(config.RAW_DATA_PATH)
                processor.load_and_validate()
            qa_examples = processor.build_qa_eval_set()

    if args.limit:
        qa_examples = qa_examples[: args.limit]
        print(
            f"  [--limit {args.limit}] Evaluating first "
            f"{args.limit} examples only."
        )

    print(
        f"  Corpus: {len(corpus_chunks)} chunks  |  "
        f"QA: {len(qa_examples)} examples"
    )

    # -- Ground-truth relevance map ----------------------------------------
    relevant_map, labeling_coverage = DataProcessor.build_relevant_chunk_map(
        corpus_chunks, qa_examples, return_coverage=True,
    )
    print(
        f"  Labeling coverage: "
        f"{labeling_coverage['labeled']}/{labeling_coverage['total']} labeled "
        f"(unlabeled={labeling_coverage['unlabeled']}) "
        f"by_strategy={labeling_coverage['by_strategy']}"
    )

    # -- Shared caches -----------------------------------------------------
    embedder_cache: dict = {}
    retriever_cache: dict = {}
    bm25_cache: dict = {}
    reranker_cache: dict = {}

    # -- Run stages --------------------------------------------------------
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
                eval_set_name=(
                    args.eval_set if not args.eval_data else "external"
                ),
                labeling_coverage=labeling_coverage,
            )
            all_results[key] = result
        except KeyboardInterrupt:
            print(
                f"\n  ⚠ Interrupted during stage '{key}'. "
                f"Saving partial results …"
            )
            break
        except Exception as exc:
            print(f"\n  ERROR in stage '{key}': {exc}")
            import traceback
            traceback.print_exc()
            import gc
            import torch as _torch
            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            print("  Continuing with next stage …")

    # -- Ablation table ----------------------------------------------------
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
