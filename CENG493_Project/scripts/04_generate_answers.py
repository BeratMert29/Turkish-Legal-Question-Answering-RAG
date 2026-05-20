"""Generate answers for test set via Ollama. Supports checkpoint/resume."""
import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from data.data_processor import DataProcessor
from data.qa_loader import resolve_qa_set
from data.corpus_loader import resolve_corpus
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from generation.rag_pipeline import RAGPipeline, ChunkExpander

def check_ollama():
    """Pre-flight: verify Ollama is running."""
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print("ERROR: Ollama is not running.")
        print(f"  Start it with: ollama serve")
        print(f"  Then pull the model: ollama pull {config.LLM_MODEL}")
        print(f"  Details: {e}")
        sys.exit(1)

def count_valid_lines(path) -> int:
    """Count valid JSON lines in checkpoint file. Truncates corrupt last line."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").splitlines()
    count = 0
    for line in lines:
        try:
            json.loads(line)
            count += 1
        except json.JSONDecodeError:
            # Assumption: corruption (if any) only occurs at the final line due to per-record flush.
            # If a mid-file corrupt line is found, stop here and truncate — this is a very rare edge case.
            break
    # Truncate file to only valid lines if needed
    if count < len(lines):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines[:count]) + ("\n" if count else ""))
    return count

def parse_args():
    parser = argparse.ArgumentParser(description="Generate answers for QA eval set")
    parser.add_argument(
        "--mode",
        choices=["dense", "hybrid", "rrf", "rerank", "hybrid_rerank", "rrf_rerank"],
        default="dense",
        help="Retrieval mode (default: dense)",
    )
    parser.add_argument(
        "--dataset",
        choices=["kaggle", "hmgs", "custom"],
        default="kaggle",
        help="Evaluation dataset to use (default: kaggle)",
    )
    parser.add_argument("--qa-file", default=None, dest="qa_file", help="Path to custom benchmark JSONL")
    parser.add_argument("--corpus", default=None, help="Path to custom corpus JSONL for BM25")
    parser.add_argument("--docs-path", default=None, dest="docs_path", help="Directory of docs for BM25 corpus")
    return parser.parse_args()

def main():
    args = parse_args()
    check_ollama()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval set
    eval_set_raw, short_answer_mode, suffix = resolve_qa_set(args.dataset, args.qa_file)
    eval_set = eval_set_raw  # already list of dicts with question/answer/query_id/source
    predictions_path = config.RESULTS_DIR / f"qa_predictions_{args.mode}{suffix}.jsonl"
    print(f"Loaded {len(eval_set)} QA examples")

    # Checkpoint/resume
    already_done = count_valid_lines(predictions_path)
    if already_done > 0:
        print(f"Resuming from checkpoint: {already_done}/{len(eval_set)} already done")
    remaining = eval_set[already_done:]

    # Load retriever
    index_path = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print("Loading embedding model and index...")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    expander = ChunkExpander(metadata_path)
    pipeline = RAGPipeline(retriever, short_answer_mode=short_answer_mode, chunk_expander=expander)

    # Build BM25 if needed
    bm25_index = None
    needs_bm25 = args.mode in ("hybrid", "rrf", "hybrid_rerank", "rrf_rerank")
    if needs_bm25:
        from retrieval.bm25_retriever import BM25Index
        corpus_path = resolve_corpus(args.corpus, args.docs_path)
        bm25_chunks = DataProcessor.load_jsonl(corpus_path)
        print(f"Building BM25 index over {len(bm25_chunks)} chunks...")
        bm25_index = BM25Index()
        bm25_index.build([{"text": c["text"], "chunk_id": c["chunk_id"]} for c in bm25_chunks])

    # Load reranker if needed
    reranker = None
    needs_rerank = args.mode in ("rerank", "hybrid_rerank", "rrf_rerank")
    if needs_rerank:
        from retrieval.reranker import Reranker
        reranker = Reranker()
        reranker.load_model()

    # Batch-retrieve all remaining questions at once (single embedding + index.search call)
    print(f"Batch-retrieving {len(remaining)} questions...")
    t0 = time.time()
    remaining_questions = [e['question'] for e in remaining]
    initial_k = config.RERANKER_CANDIDATES if needs_rerank else config.TOP_K_RETRIEVAL
    if args.mode in ("rrf", "rrf_rerank"):
        all_retrieved = retriever.batch_rrf_retrieve(remaining_questions, bm25_index, top_k=initial_k)
    elif args.mode in ("hybrid", "hybrid_rerank"):
        all_retrieved = retriever.batch_hybrid_retrieve(remaining_questions, bm25_index, top_k=initial_k)
    elif args.mode == "rerank":
        all_retrieved = retriever.batch_retrieve(remaining_questions, top_k=initial_k)
    else:
        all_retrieved = retriever.batch_retrieve(remaining_questions, top_k=initial_k)

    if needs_rerank:
        all_retrieved = reranker.batch_rerank(remaining_questions, all_retrieved, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Retrieval done in {time.time()-t0:.1f}s")

    print(f"Generating {len(remaining)} answers...")

    with open(predictions_path, "a", encoding="utf-8") as f:
        for i, (example, retrieved_chunks) in enumerate(tqdm(zip(remaining, all_retrieved), total=len(remaining), desc="Generating answers")):
            global_i = already_done + i + 1
            try:
                context_used, context_chunks = pipeline.assemble_context(retrieved_chunks)
                answer = pipeline.generate(example['question'], context_used)
                record = {
                    "query_id": example['query_id'],
                    "question": example['question'],
                    "predicted": answer,
                    "expected": example['answer'],
                    "retrieved_sources": [c['source'] for c in context_chunks],
                    "expected_source": example.get('source', ''),
                    "retrieved_chunks": [dict(c) for c in context_chunks],
                }
            except Exception as e:
                record = {
                    "query_id": example['query_id'],
                    "question": example['question'],
                    "predicted": "",
                    "expected": example['answer'],
                    "retrieved_sources": [],
                    "expected_source": example.get('source', ''),
                    "retrieved_chunks": [],
                    "error": str(e),
                }
                print(f"  ERROR at query {global_i}: {e}")

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    total = already_done + len(remaining)
    print(f"\n✓ Generation complete. {total} predictions written to {predictions_path}")

if __name__ == '__main__':
    main()
