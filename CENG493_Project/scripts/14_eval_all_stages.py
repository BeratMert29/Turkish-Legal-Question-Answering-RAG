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
    python scripts/14_eval_all_stages.py --stages base --eval-set hmgs
    python scripts/14_eval_all_stages.py --list-stages
    python scripts/14_eval_all_stages.py \
        --corpus datasets/corpus.jsonl \
        --eval-data datasets/gold_benchmark.json          # external evaluator format
    python scripts/14_eval_all_stages.py \
        --corpus datasets/corpus.jsonl \
        --eval-data datasets/rag_eval.json                # external rag eval

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
import dataclasses
import json
import os
import sys
from pathlib import Path

# PYTHONUTF8 only takes effect at interpreter start-up; reconfigure the
# streams so Turkish text and symbols print on a cp125x Windows console too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
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
        nargs="+",
        default=[",".join(DEFAULT_STAGE_ORDER)],
        help=(
            f"Stages to run, comma- or space-separated. Default: all. "
            f"Options: {', '.join(DEFAULT_STAGE_ORDER)}"
        ),
    )
    parser.add_argument(
        "--eval-set", "--dataset",
        dest="eval_set",
        choices=config.EVAL_SET_CHOICES,
        default=config.DEFAULT_EVAL_SET,
        help=(
            "Evaluation dataset for ALL stages (default: turkish_legal_rag, "
            "~195 questions with explicit law+article gold labels). "
            "Use 'hmgs' for the HMGS exam set (~161 questions) or 'kaggle' "
            "for the Kaggle-split eval set (~300 questions). "
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
    parser.add_argument(
        "--no-judge", action="store_true", dest="no_judge",
        help="Skip the LLM-judge phase (judge scores and Scenario 3 become N/A).",
    )
    parser.add_argument(
        "--results-root", type=Path, default=None, dest="results_root",
        help=(
            "Root directory for this run's outputs (default: config.RESULTS_ROOT). "
            "Results go to <root>/<eval_set>[_limitN]/<stage>/."
        ),
    )
    args = parser.parse_args(argv)
    args.stages = ",".join(args.stages)
    return args


def run_dir_name(eval_set: str, limit, eval_data=None) -> str:
    """Directory name of one run: eval set (or external file stem) + limit."""
    name = f"external_{Path(eval_data).stem}" if eval_data else eval_set
    return f"{name}_limit{limit}" if limit else name


def merge_summary(summary_path: Path, new_results: dict) -> dict:
    """Stage results of this run merged over an existing summary, so a rerun
    of some stages keeps the others; bookkeeping keys are rebuilt."""
    merged: dict = {}
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                old = json.load(f)
            merged = {k: v for k, v in old.get("stages", {}).items()}
        except (json.JSONDecodeError, OSError):
            print(f"  WARNING: unreadable {summary_path}; starting a new summary")
    merged.update(new_results)
    return merged


def load_per_query(stage_dir: Path):
    path = stage_dir / "per_query.jsonl"
    if not path.exists():
        return None
    from utils import read_jsonl
    return list(read_jsonl(path))


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
    from pipeline.evaluation import (
        compare_stages, print_ablation_table, print_comparison_table, run_stage,
    )
    from utils import check_ollama, run_provenance, set_seeds

    set_seeds(config.SEED)

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
            # Rebuild if the existing file is corrupt JSON.
            if graph_path.exists():
                try:
                    with graph_path.open(encoding="utf-8") as _gf:
                        json.load(_gf)
                except json.JSONDecodeError:
                    print(f"  graph.json is corrupt — rebuilding …")
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
        if stage.llm == "finetuned" and not check_ollama(
            config.LLM_BASE_URL, config.LLM_FINETUNED_MODEL
        ):
            print(
                f"INFO: Stage '{key}' skipped -- "
                f"Ollama model '{config.LLM_FINETUNED_MODEL}' not available.\n"
                f"  Run: python scripts/13_export_lora_to_ollama.py first."
            )
            continue
        valid.append(key)

    if not valid:
        sys.exit("ERROR: No valid stages to run.")

    print(f"\n\U0001f680  Stages to run: {', '.join(valid)}")
    print(f"   Eval set: {args.eval_set}\n")

    # -- Check Ollama models up front -------------------------------------
    # A missing judge would otherwise cost ~3 retries per call per metric and
    # mark every stage failed only at the end.
    needed = []
    if any(STAGE_REGISTRY[k].llm == "base" for k in valid):
        needed.append(config.LLM_MODEL)
    if not args.no_judge:
        needed.append(config.LLM_JUDGE_MODEL)
    for model in needed:
        if not check_ollama(config.LLM_BASE_URL, model):
            sys.exit(
                f"ERROR: Ollama not reachable at {config.LLM_BASE_URL} or model "
                f"'{model}' not pulled.\n"
                f"  Start with: ollama serve\n"
                f"  Pull model: ollama pull {model}"
                + ("\n  (or pass --no-judge to skip the judge)"
                   if model == config.LLM_JUDGE_MODEL else "")
            )

    # -- Load data ---------------------------------------------------------
    print("Loading data …")

    processor = None

    def _processor():
        nonlocal processor
        if processor is None:
            processor = DataProcessor(config.RAW_DATA_PATH)
            processor.load_and_validate()
        return processor

    if args.corpus:
        print(f"  Corpus source : {args.corpus} (external evaluator format)")
        corpus_chunks = load_external_corpus(Path(args.corpus))
    elif args.docs_path:
        from data.corpus_loader import resolve_corpus
        corpus_path = resolve_corpus(None, args.docs_path)
        print(f"  Chunking/loading corpus from {args.docs_path} …")
        corpus_chunks = load_external_corpus(Path(corpus_path))
    else:
        corpus_chunks = list(
            # Hold the eval rows out of the index only for the Kaggle-split
            # eval set; the other eval sets are not drawn from the corpus.
            _processor().build_corpus_chunks(holdout=args.eval_set == "kaggle")
        )

    if args.eval_data:
        print(f"  QA source     : {args.eval_data} (external evaluator format)")
        qa_examples, short_answer_mode = load_external_qa(Path(args.eval_data))
        eval_file = Path(args.eval_data)
    else:
        short_answer_mode = args.eval_set == "hmgs"
        if args.eval_set == "hmgs":
            qa_examples = DataProcessor.build_gold_eval_set()
            eval_file = config.HMGS_DATA_PATH
        elif args.eval_set == "turkish_legal_rag":
            qa_examples = DataProcessor.build_turkish_legal_rag_eval_set()
            eval_file = config.TLR_DATA_PATH
        else:
            qa_examples = _processor().build_qa_eval_set()
            eval_file = config.RAW_DATA_PATH

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

    # An eval set that ships explicit gold labels must actually be labeled;
    # otherwise every chunk-level metric silently collapses to 0.
    if not args.eval_data and args.eval_set == "turkish_legal_rag":
        _total = labeling_coverage["total"]
        _frac = labeling_coverage["labeled"] / _total if _total else 0.0
        if _frac < config.HEADLINE_CHUNK_MIN_LABELED_FRACTION:
            sys.exit(
                f"ERROR: only {labeling_coverage['labeled']}/{_total} "
                f"({_frac:.0%}) of the turkish_legal_rag queries have gold "
                f"chunk labels (< "
                f"{config.HEADLINE_CHUNK_MIN_LABELED_FRACTION:.0%}).\n"
                f"  The corpus index does not match the eval set's gold "
                f"labels -- rebuild the index/corpus (scripts/01, 02) and "
                f"re-run scripts/16_prepare_turkish_legal_rag.py."
            )

    # The kaggle eval set is labeled from its own contexts; zero labels means
    # the holdout removed every gold passage and retrieval metrics are void.
    if (not args.eval_data and args.eval_set == "kaggle"
            and labeling_coverage["labeled"] == 0):
        sys.exit(
            "ERROR: no kaggle eval query has a gold chunk in the corpus -- the "
            "held-out split removed every gold passage from the index."
        )

    # -- Output locations & provenance -----------------------------------------
    eval_set_name = args.eval_set if not args.eval_data else "external"
    run_dir = (args.results_root or config.RESULTS_ROOT) / run_dir_name(
        args.eval_set, args.limit, args.eval_data)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run directory : {run_dir}")
    provenance = run_provenance(
        seed=config.SEED,
        base_url=config.LLM_BASE_URL,
        models=[config.LLM_MODEL, config.LLM_FINETUNED_MODEL, config.LLM_JUDGE_MODEL],
        eval_file=eval_file,
        corpus_chunks=corpus_chunks,
        extra={"limit": args.limit, "eval_n": len(qa_examples),
               "labeled_n": labeling_coverage["labeled"]},
    )

    # -- Shared caches -----------------------------------------------------
    embedder_cache: dict = {}
    retriever_cache: dict = {}
    bm25_cache: dict = {}
    reranker_cache: dict = {}

    # -- Run stages --------------------------------------------------------
    all_results: dict[str, dict] = {}

    for key in valid:
        stage = dataclasses.replace(STAGE_REGISTRY[key], results_dir=run_dir / key)
        try:
            result = run_stage(
                key, stage, qa_examples, corpus_chunks,
                embedder_cache=embedder_cache,
                retriever_cache=retriever_cache,
                bm25_cache=bm25_cache,
                reranker_cache=reranker_cache,
                relevant_map=relevant_map,
                short_answer_mode=short_answer_mode,
                eval_set_name=eval_set_name,
                labeling_coverage=labeling_coverage,
                run_judge=not args.no_judge,
                provenance=provenance,
            )
            all_results[key] = result
        except KeyboardInterrupt:
            print(
                f"\n  ⚠ Interrupted during stage '{key}'; its results are "
                f"discarded. Saving the completed stages …"
            )
            break
        except Exception as exc:
            print(f"\n  ERROR in stage '{key}': {exc}")
            import traceback
            traceback.print_exc()
            import gc
            gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
            print("  Continuing with next stage …")

    # -- Ablation table + paired comparisons -------------------------------
    if not all_results:
        print("No results to report.")
        return

    summary_path = run_dir / "ablation_summary.json"
    stages = merge_summary(summary_path, all_results)
    print_ablation_table(stages)

    per_query = {
        k: pq for k, r in stages.items()
        if r.get("status", "ok") == "ok" and (pq := load_per_query(run_dir / k))
    }
    comparisons = compare_stages(per_query)
    print_comparison_table(comparisons)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "run": {"eval_set": eval_set_name, "limit": args.limit,
                    "provenance": provenance},
            "stages": stages,
            "comparisons": comparisons,
        }, f, ensure_ascii=False, indent=2)
    print(f"Full results saved to: {summary_path}")


if __name__ == "__main__":
    main()
