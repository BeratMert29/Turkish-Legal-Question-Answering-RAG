import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import read_jsonl, resolve_processed_file
import config
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Index
from generation.rag_pipeline import RAGPipeline
from generation.train_citations import inject_citations_strict, CitationStats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build RAG-augmented training data for LLM fine-tuning")
    p.add_argument(
        "--dense-only",
        action="store_true",
        help="Skip BM25/RRF, use plain dense retrieval (faster).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Resolve input path: no silent fallback ---
    # config.PROCESSED_DIR = CENG493_Project/data/processed (gitignored).
    # The bundled copy lives in repo_root/results/processed_data; this script
    # reads from config.PROCESSED_DIR — the caller must ensure the file is
    # there (e.g. via scripts/07 or a symlink).
    # resolve_processed_file raises FileNotFoundError when the file is absent,
    # so there is no silent fallback to qa_train.jsonl.
    dataset_path = resolve_processed_file("qa_train_merged.jsonl")
    print(f"Loading training records from {dataset_path} ...")
    records = list(read_jsonl(dataset_path))
    print(f"  {len(records)} records loaded.")
    # Never fine-tune on an eval question (07 merges an external dataset).
    from data.data_processor import DataProcessor, normalize_question
    eval_keys = DataProcessor.saved_eval_question_keys()
    kept = [r for r in records if normalize_question(r["question"]) not in eval_keys]
    print(f"  {len(records) - len(kept)} records dropped: question is in an eval set")
    records = kept

    index_path = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print(f"\nLoading FAISS index from {index_path} ...")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    print(f"  Index loaded: {retriever.index.ntotal} vectors")

    rag_pipeline = RAGPipeline(
        retriever,
        top_k_for_generation=config.TOP_K_FOR_GENERATION,
        context_window_chars=config.CONTEXT_WINDOW_CHARS,
        chunk_expander=None,
        graph_neighbor_budget=0,
    )

    bm25_index = None
    if not args.dense_only:
        print(f"\nBuilding BM25 index over {retriever.index.ntotal} metadata entries ...")
        t0 = time.time()
        bm25_index = BM25Index()
        bm25_index.build(retriever.metadata)
        print(f"  Done in {time.time()-t0:.1f}s")

    out_path = config.PROCESSED_DIR / "qa_train_rag.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = [r["question"] for r in records]
    top_k = config.TOP_K_FOR_GENERATION
    ctx_chars = config.CONTEXT_WINDOW_CHARS

    print(f"\nRetrieving context for {len(records)} records "
          f"({'dense' if args.dense_only else 'RRF'}, top_k={top_k}) ...")

    # Batch retrieve all at once for efficiency
    if args.dense_only:
        all_chunks = retriever.batch_retrieve(questions, top_k=top_k)
    else:
        all_chunks = retriever.batch_rrf_retrieve(
            questions, bm25_index, top_k=top_k,
        )

    total_ctx_len = 0
    stats = CitationStats()

    with open(out_path, "w", encoding="utf-8") as fout:
        for record, chunks in tqdm(zip(records, all_chunks), total=len(records), desc="Assembling"):
            # Use assemble_context_ex for truncation/drop metadata
            asm = rag_pipeline.assemble_context_ex(chunks)

            # Strict citation injection for training data:
            # - denominator = sentence token count (not min)
            # - stopwords stripped
            # - short sentences skipped
            # - newline structure preserved
            cited_answer, n_cited, n_sents, n_short = inject_citations_strict(
                record["answer"],
                asm.included,
                truncated_ids=asm.truncated_ids,
            )

            stats.record(
                n_sentences=n_sents,
                n_cited=n_cited,
                n_short_skips=n_short,
                has_truncation=bool(asm.truncated_ids),
                has_drop=bool(asm.dropped_ids),
            )

            total_ctx_len += len(asm.context)

            out_record = {
                "query_id": record.get("query_id", record.get("id", "")),
                "question": record["question"],
                "answer": cited_answer,
                "context_str": asm.context,
                "source": record.get("source", ""),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    n = len(records)
    avg_ctx = total_ctx_len / n if n else 0
    print("\nDone.")
    print(f"  Output       : {out_path}")
    print(f"  Examples     : {n}")
    print(f"  Avg ctx len  : {avg_ctx:.0f} chars")
    print(stats.log_summary())

    # Write metadata JSON alongside output
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "output_file": str(out_path),
        "input_file": str(dataset_path),
        "n_records": n,
        "avg_context_chars": round(avg_ctx, 1),
        **stats.summary_dict(),
    }
    with open(meta_path, "w", encoding="utf-8") as fm:
        json.dump(meta, fm, ensure_ascii=False, indent=2)
    print(f"  Metadata     : {meta_path}")


if __name__ == "__main__":
    main()
