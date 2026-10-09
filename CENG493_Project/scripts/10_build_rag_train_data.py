import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import read_jsonl
import config
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Index
from generation.rag_pipeline import RAGPipeline


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

    merged = config.PROCESSED_DIR / "qa_train_merged.jsonl"
    fallback = config.PROCESSED_DIR / "qa_train.jsonl"
    dataset_path = merged if merged.exists() else fallback
    if not dataset_path.exists():
        print(f"ERROR: training dataset not found at {dataset_path}")
        sys.exit(1)
    print(f"Loading training records from {dataset_path} ...")
    records = list(read_jsonl(dataset_path))
    print(f"  {len(records)} records loaded.")

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
    hit_limit = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for record, chunks in tqdm(zip(records, all_chunks), total=len(records), desc="Assembling"):
            ctx_str, _ = rag_pipeline.assemble_context(chunks)

            # detect if we hit the char limit (last chunk was not fully included)
            if len(chunks) > 0:
                full_len = sum(
                    len(f"[Kaynak {i+1}] ({c['source']})\n{c['text']}\n\n")
                    for i, c in enumerate(chunks[:top_k])
                )
                if full_len > ctx_chars:
                    hit_limit += 1

            total_ctx_len += len(ctx_str)

            out_record = {
                "query_id": record.get("query_id", record.get("id", "")),
                "question": record["question"],
                "answer": record["answer"],
                "context_str": ctx_str,
                "source": record.get("source", ""),
            }
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    n = len(records)
    avg_ctx = total_ctx_len / n if n else 0
    print(f"\nDone.")
    print(f"  Output       : {out_path}")
    print(f"  Examples     : {n}")
    print(f"  Avg ctx len  : {avg_ctx:.0f} chars")
    print(f"  Hit char limit: {hit_limit}/{n} ({100*hit_limit/n:.1f}%)")


if __name__ == "__main__":
    main()
